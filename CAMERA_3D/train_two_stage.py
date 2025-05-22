#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage training pipeline for CAMERA_3D

    • Stage  1   — joint contrastive pre-training of RAG  Text  Encoder & GateFusion-MV
    • Stage  2   — FAISS recall + Cross-modal reranker fine-tuning

Optional Weights & Biases logging
---------------------------------
Pass `--wandb` and have wandb installed to enable logging. Otherwise all wandb calls are no-ops.
"""
import os
import argparse
import torch
import faiss
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder import RAGTextEncoder
from models.gf_mv_encoder import GFMVEncoder
from models.cross_modal_reranker import CrossModalReranker
from utils.checks import (assert_valid,
                          check_gradients,
                          stable_rank_loss,
                          xavier_init)

# ---------------------------------------------------------------------------
# Optional wandb (lazy import + graceful fallback)
# ---------------------------------------------------------------------------
try:
    import wandb
except ImportError:
    wandb = None

_USE_WANDB = False

def wandb_init(args):
    global _USE_WANDB
    if args.wandb and wandb is not None:
        wandb.init(
            project="CAMERA3D",
            name=f"bs{args.bs}_topk{args.topk}_ep{args.ep1+args.ep2}",
            config=vars(args),
        )
        _USE_WANDB = True
    else:
        _USE_WANDB = False


def wb_log(data: dict, step: Optional[int] = None):
    if _USE_WANDB:
        wandb.log(data, step=step)

# ---------------------------------------------------------------------------
# Visualization: log similarity heatmap to wandb
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

def log_sim_heatmap(v_vec: torch.Tensor,
                    t_vec: torch.Tensor,
                    step: int,
                    tag: str = "stage1/sim_matrix",
                    max_show: int = 16):
    if not _USE_WANDB:
        return
    with torch.no_grad():
        v = F.normalize(v_vec, 2, 1)[:max_show]
        t = F.normalize(t_vec, 2, 1)[:max_show]
        sim = (v @ t.T).cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="viridis")
    ax.set_title(f"Sim matrix @ step {step}")
    ax.set_xlabel("text idx")
    ax.set_ylabel("vision idx")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    wandb.log({tag: wandb.Image(img)}, step=step)

def log_gradients(models: dict, top_n: int = 5, prefix: str = "grad"):
    grad_stats = {}
    for mod_name, module in models.items():
        for name, param in module.named_parameters():
            if param.grad is not None:
                key = f"{mod_name}.{name}"
                grad_stats[key] = param.grad.norm().item()
    if not grad_stats:
        print(" No gradients found.")
        return

    # 排序
    sorted_stats = sorted(grad_stats.items(), key=lambda x: x[1])
    print(f"=== {prefix} norms (lowest {top_n}) ===")
    for k, v in sorted_stats[:top_n]:
        print(f"{k:60s}: {v:.4e}")
    print(f"=== {prefix} norms (highest {top_n}) ===")
    for k, v in sorted_stats[-top_n:]:
        print(f"{k:60s}: {v:.4e}")

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def info_nce(v, t, tau: float = 0.07):
    v = F.normalize(v, 2, 1)
    t = F.normalize(t, 2, 1)
    return F.cross_entropy(v @ t.T / tau, torch.arange(v.size(0), device=v.device))

def ranknet(pos, neg):
    return F.softplus(neg - pos).mean()

# ---------------------------------------------------------------------------
# Stage 1 – contrastive pre-training
# ---------------------------------------------------------------------------
def stage1(args, ds):
    print("stage1 begin…")
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)
    txt_enc = RAGTextEncoder(args.data_jsonl, args.topk).cuda()
    # 凍結 BERT（只訓練 GateFusion-MV）
    for p in txt_enc.retriever.qenc.parameters():
        p.requires_grad = False
    vis_enc = GFMVEncoder(args.views).cuda()
    opt = torch.optim.AdamW(
        list(txt_enc.parameters()) + list(vis_enc.parameters()), lr=args.lr1
    )
    for ep in range(args.ep1):
        pbar = tqdm(dl, desc=f"Stage-1 Epoch {ep}", unit="batch")
        for step, (cap, imgs, obj_id) in enumerate(pbar):
            imgs = imgs.cuda(non_blocking=True)
            t_vec, ret_loss, _ = txt_enc(list(cap), list(obj_id), return_loss=True)
            v_vec, _ = vis_enc(imgs, t_vec)
            #v_vec, _ = vis_enc(imgs, None)
            loss = info_nce(v_vec, t_vec, args.tau) + args.lmb * ret_loss
            assert_valid(loss, "stage1_loss")
            opt.zero_grad()
            loss.backward()
            # log_gradients({
            #     "qenc": txt_enc.retriever.qenc,
            #     "kenc": txt_enc.retriever.kenc,
            #     "gf_feat": vis_enc.feat2d
            # })
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wb_log({"stage1/loss": loss.item()}, step=ep * len(dl) + step)
            if step % 1000 == 0:
                log_sim_heatmap(v_vec.detach(), t_vec.detach(), step=ep * len(dl) + step)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"txt": txt_enc.state_dict(), "vis": vis_enc.state_dict()},
               f"{args.out}/enc1.pth")
    return txt_enc.eval(), vis_enc.eval()

# ---------------------------------------------------------------------------
# Build FAISS index (global vectors)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_faiss(ds, txt_enc, vis_enc, args):
    print("build faiss…")
    dl, feats = DataLoader(ds, args.bs, num_workers=4), []
    for caps, imgs, _ in tqdm(dl, desc="Building FAISS", unit="batch"):
        q_vec, _, _ = txt_enc(list(caps))
        v, _ = vis_enc(imgs.cuda(non_blocking=True), q_vec.detach())
        feats.append(F.normalize(v, 2, 1).cpu())
    feats = torch.cat(feats).numpy()
    index = faiss.IndexFlatIP(512)
    index.add(feats)
    return index, feats

@torch.no_grad()
def build_faiss_with_tok(ds, txt_enc, vis_enc, args):
    """
    若 {args.out}/faiss_index.bin 與 vis_tok.npy 已存在 → 直接讀取
    否則重新計算並存檔（index, tokens）
    """
    idx_path = os.path.join(args.out, "faiss_index.bin")
    tok_path = os.path.join(args.out, "vis_tok.npy")

    # -------- 1. 嘗試讀取 --------
    if os.path.isfile(idx_path) and os.path.isfile(tok_path):
        print(f"✓ Load cached FAISS index & tokens from {args.out}")
        index   = faiss.read_index(idx_path, faiss.IO_FLAG_MMAP)
        all_tok = np.load(tok_path)
        assert_valid(torch.from_numpy(all_tok), "loaded_vis_tok")
        return index, all_tok

    # -------- 2. 建立 --------
    print("✗ Cache not found → building FAISS index & tokens ...")
    g_vecs, g_toks = [], []
    dl = DataLoader(ds, args.bs, num_workers=4)

    for caps, imgs, _ in tqdm(dl, desc="Building FAISS", unit="batch"):
        q_vec, _, _ = txt_enc(list(caps))
        v_vec, v_tok = vis_enc(imgs.cuda(non_blocking=True), q_vec)

        g_vecs.append(F.normalize(v_vec, 2, 1).cpu())
        v_tok = F.normalize(v_tok, 2, -1)          # token L2-norm
        g_toks.append(v_tok.cpu())

    # ---- 存 tokens ----
    all_tok = torch.cat(g_toks).numpy().astype(np.float16)
    assert_valid(torch.from_numpy(all_tok), "vis_tok_np")
    os.makedirs(args.out, exist_ok=True)
    np.save(tok_path, all_tok)

    # ---- 建 & 存 FAISS ----
    index = faiss.IndexFlatIP(512)
    index.add(torch.cat(g_vecs).numpy())
    faiss.write_index(index, idx_path)            
    print(f"✓ Saved faiss_index.bin & vis_tok.npy to {args.out}")

    return index, all_tok


# ---------------------------------------------------------------------------
# Stage 2 – reranker fine-tuning
# ---------------------------------------------------------------------------
def stage2(args, ds, txt_enc, vis_enc, index, all_tok):
    print("stage2 begin…")
    rerank = CrossModalReranker().cuda()
    rerank.apply(xavier_init)                     
    opt = torch.optim.AdamW(rerank.parameters(), lr=args.lr2)
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)
    for ep in range(args.ep2):
        pbar = tqdm(dl, desc=f"Stage-2 Epoch {ep}", unit="batch")
        for step, (cap, imgs, obj_id) in enumerate(pbar):
            cap = list(cap)
            imgs = imgs.cuda(non_blocking=True)
            with torch.no_grad():
                t_vec, _, txt_tok = txt_enc(cap, list(obj_id))
                _, vis_tok = vis_enc(imgs, t_vec)
                vis_tok = F.normalize(vis_tok, 2, -1)
                sims, idx = index.search(t_vec.cpu().numpy(), args.L)
            s_pos = rerank(txt_tok, vis_tok)
            neg_scores = []
            for j in range(1, idx.shape[1]):
                neg_tok = torch.from_numpy(all_tok[idx[:, j]]).to(imgs.device).float()
                neg_scores.append(rerank(txt_tok, neg_tok))
            s_neg = torch.stack(neg_scores, 1)
            ## loss = ranknet(s_pos.unsqueeze(1), s_neg)
            loss = stable_rank_loss(s_pos, s_neg)
            assert_valid(loss, "stage2_loss")  
            opt.zero_grad()
            loss.backward()
            check_gradients(rerank, top_n=2)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wb_log({"stage2/rank_loss": loss.item()}, step=(args.ep1 + ep) * len(dl) + step)
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
    p.add_argument("--L", type=int, default=50, help="# candidates per query for reranker")
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--ep2", type=int, default=5)
    p.add_argument("--wandb", action="store_true", help="enable wandb logging")
    p.add_argument("--resume_enc1", type=str, default=None,
               help="Path to saved Stage-1 encoder weights (skip training)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    wandb_init(args)
    ds = UnifiedDataset(args.data_jsonl, num_views=args.views)
    if args.resume_enc1: ## load stage model
        print(f"Loading Stage-1 weights from {args.resume_enc1} ...")
        state = torch.load(args.resume_enc1, map_location="cuda")
        txt_enc = RAGTextEncoder(args.data_jsonl, args.topk).cuda()
        vis_enc = GFMVEncoder(args.views).cuda()
        txt_enc.load_state_dict(state["txt"])
        vis_enc.load_state_dict(state["vis"])
        txt_enc.eval(); vis_enc.eval()
    else:
        txt_enc, vis_enc = stage1(args, ds)

    index, all_tok = build_faiss_with_tok(ds, txt_enc, vis_enc, args)
    stage2(args, ds, txt_enc, vis_enc, index, all_tok)
    print(" Training done →", args.out)
