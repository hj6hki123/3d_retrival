#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two‑stage training pipeline for CAMERA_3D

    • Stage 1   — joint contrastive pre‑training of RAG Text Encoder & GateFusion‑MV
    • Stage 2   — FAISS recall + Cross‑modal reranker fine‑tuning

Optional Weights & Biases logging
---------------------------------
Pass `--wandb` *and* have the wandb package installed to enable logging.
Otherwise all wandb calls are safely skipped.
"""
import os, argparse, torch, faiss, torch.nn.functional as F
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder import RAGTextEncoder
from models.gf_mv_encoder import GFMVEncoder
from models.cross_modal_reranker import CrossModalReranker

# ---------------------------------------------------------------------------
# Optional wandb (lazy import + graceful fallback)
# ---------------------------------------------------------------------------
try:
    import wandb  # noqa: F401  (might be unavailable)
except ImportError:  # pragma: no cover –– allow running without wandb
    wandb = None

_USE_WANDB = False  # runtime flag toggled by wandb_init()


def wandb_init(args):
    """Initialise wandb only when user explicitly passes --wandb and the   
    library is available. Sets the global _USE_WANDB flag so that later   
    calls to wb_log() are no‑ops when logging is disabled.
    """
    global _USE_WANDB
    if args.wandb and wandb is not None:
        wandb.init(
            project="CAMERA3D",
            name=f"bs{args.bs}_topk{args.topk}_ep{args.ep1 + args.ep2}",
            config=vars(args),
        )
        _USE_WANDB = True
    else:
        _USE_WANDB = False


def wb_log(data: dict, step: Optional[int] = None):
    """Safely log to wandb if *_USE_WANDB* is True."""
    if _USE_WANDB:
        wandb.log(data, step=step)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def info_nce(v, t, tau: float = 0.07):
    v = F.normalize(v, 2, 1)
    t = F.normalize(t, 2, 1)
    return F.cross_entropy(v @ t.T / tau, torch.arange(v.size(0), device=v.device))


def ranknet(pos, neg):
    return F.softplus(neg - pos).mean()


# ---------------------------------------------------------------------------
# Stage 1 – contrastive pre‑training
# ---------------------------------------------------------------------------

def stage1(args, ds):
    print("stage1 begin…")
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)
    txt_enc = RAGTextEncoder(args.data_jsonl, args.topk).cuda()
    vis_enc = GFMVEncoder(args.views).cuda()

    opt = torch.optim.AdamW(
        list(txt_enc.parameters()) + list(vis_enc.parameters()), lr=args.lr1
    )

    for ep in range(args.ep1):
        pbar = tqdm(dl, desc=f"Stage‑1 Epoch {ep}", unit="batch")
        for cap, imgs, obj_id in pbar:
            imgs = imgs.cuda(non_blocking=True)
            t_vec, ret_loss, _ = txt_enc(list(cap), list(obj_id), return_loss=True)
            v_vec, _ = vis_enc(imgs, t_vec)  # Gate token active
            loss = info_nce(v_vec, t_vec, args.tau) + args.lmb * ret_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wb_log({"stage1/loss": loss.item()}, step=ep * len(dl) + pbar.n)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"txt": txt_enc.state_dict(), "vis": vis_enc.state_dict()}, f"{args.out}/enc1.pth")
    return txt_enc.eval(), vis_enc.eval()


# ---------------------------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_faiss(ds, txt_enc, vis_enc, args):
    print("build faiss…")
    dl, feats = DataLoader(ds, args.bs, num_workers=4), []
    for caps, imgs, _ in tqdm(dl, desc="Building FAISS", unit="batch"):
        q_vec, _, _ = txt_enc(list(caps))  # text query embeddings (guides gate)
        v, _ = vis_enc(imgs.cuda(non_blocking=True), q_vec.detach())
        feats.append(F.normalize(v, 2, 1).cpu())

    feats = torch.cat(feats).numpy()
    index = faiss.IndexFlatIP(512)
    index.add(feats)
    return index, feats


# ---------------------------------------------------------------------------
# Stage 2 – reranker fine‑tuning
# ---------------------------------------------------------------------------

def stage2(args, ds, txt_enc, vis_enc, index, all_v):
    print("stage2 begin…")
    rerank = CrossModalReranker().cuda()
    opt = torch.optim.AdamW(rerank.parameters(), lr=args.lr2)
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

    for ep in range(args.ep2):
        pbar = tqdm(dl, desc=f"Stage‑2 Epoch {ep}", unit="batch")
        for cap, imgs, obj_id in pbar:
            cap = list(cap)
            imgs = imgs.cuda(non_blocking=True)

            with torch.no_grad():
                t_vec, _, txt_tok = txt_enc(cap, list(obj_id))  # token_seq for reranker
                _, vis_tok = vis_enc(imgs, t_vec)               # gate ON
                sims, idx = index.search(t_vec.cpu().numpy(), args.L)

            # positive score
            s_pos = rerank(txt_tok, vis_tok)                   # (B,)

            # negative scores from retrieved neighbors
            neg_scores = []
            for j in range(1, args.L):
                neg_vec = torch.from_numpy(all_v[idx[:, j]]).to(imgs.device)
                neg_vec = neg_vec.unsqueeze(1)                # (B,1,512)
                neg_scores.append(rerank(txt_tok, neg_vec))   # (B,)
            s_neg = torch.stack(neg_scores, 1)                # (B,L‑1)

            loss = ranknet(s_pos.unsqueeze(1), s_neg)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wb_log({"stage2/rank_loss": loss.item()}, step=(args.ep1 + ep) * len(dl) + pbar.n)

    torch.save(rerank.state_dict(), f"{args.out}/rerank.pth")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_jsonl", required=True)
    p.add_argument("--out", default="ckpts")
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--lmb", type=float, default=0.1)
    p.add_argument("--lr1", type=float, default=2e-4)
    p.add_argument("--ep1", type=int, default=8)
    p.add_argument("--L", type=int, default=50, help="# candidates per query for reranker")
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--ep2", type=int, default=5)
    p.add_argument("--wandb", action="store_true", help="enable wandb logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    wandb_init(args)

    ds = UnifiedDataset(args.data_jsonl, num_views=args.views)
    txt_enc, vis_enc = stage1(args, ds)
    index, all_v = build_faiss(ds, txt_enc, vis_enc, args)
    stage2(args, ds, txt_enc, vis_enc, index, all_v)

    print("✓ Training done →", args.out)
