#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, torch, faiss, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.unified_dataset   import UnifiedDataset
from models.rag_text_encoder    import RAGTextEncoder
from models.gf_mv_encoder       import GFMVEncoder
from models.cross_modal_reranker import CrossModalReranker

# ---------- Loss ----------
def info_nce(v, t, tau=0.07):
    v = F.normalize(v, 2, 1);  t = F.normalize(t, 2, 1)
    return F.cross_entropy(v @ t.T / tau,
                           torch.arange(v.size(0), device=v.device))

def ranknet(pos, neg):
    return F.softplus(neg - pos).mean()

# —— 省略 import 與其他函式 ……

# ---------- Stage-1 ---------------------------------------------------------
def stage1(args, ds):
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

    txt_enc = RAGTextEncoder(args.corpus, args.topk).cuda()
    vis_enc = GFMVEncoder(args.views).cuda()

    opt = torch.optim.AdamW(
        list(txt_enc.parameters()) + list(vis_enc.parameters()), lr=args.lr1)

    for ep in range(args.ep1):
        for cap, imgs, obj_id in dl:
            imgs = imgs.cuda()
            t_vec, ret_loss, _ = txt_enc(list(cap), list(obj_id), return_loss=True)
            v_vec, _           = vis_enc(imgs, t_vec)  # Gate token active
            loss = info_nce(v_vec, t_vec, args.tau) + args.lmb * ret_loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[Stage-1 {ep}]  total_loss={loss.item():.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"txt": txt_enc.state_dict(),
                "vis": vis_enc.state_dict()}, f"{args.out}/enc1.pth")
    return txt_enc.eval(), vis_enc.eval()

# ---------- 建索引 (Gate=零向量) -------------------------------------------
@torch.no_grad()
def build_faiss(ds, vis_enc, args):
    dl, feats = DataLoader(ds, args.bs, num_workers=4), []
    for _, imgs, _ in dl:
        zero = torch.zeros(imgs.size(0), 512).cuda()
        v, _ = vis_enc(imgs.cuda(), zero)              # Gate with zeros
        feats.append(F.normalize(v, 2, 1).cpu())
    feats = torch.cat(feats).numpy()                   # (N,512)
    index = faiss.IndexFlatIP(512); index.add(feats)
    return index, feats


# ---------- Stage-2 -------------------------------------------
def stage2(args, ds, txt_enc, vis_enc, index, all_v):
    rerank = CrossModalReranker().cuda()
    opt    = torch.optim.AdamW(rerank.parameters(), lr=args.lr2)
    dl     = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

    for ep in range(args.ep2):
        for cap, imgs, obj_id in dl:
            cap = list(cap); imgs = imgs.cuda()

            with torch.no_grad():
                t_vec, _, txt_tok = txt_enc(cap, list(obj_id))      # token_seq
                _,      vis_tok = vis_enc(imgs, t_vec)              # gate ON
                sims, idx = index.search(t_vec.cpu().numpy(), args.L)

            # positive 得分 ---------------------------------------------------
            s_pos = rerank(txt_tok, vis_tok)                       # (B,)

            # negative 得分 (逐個 rerank) -------------------------------------
            neg_scores = []
            for j in range(1, args.L):
                neg_vec  = torch.from_numpy(all_v[idx[:, j]]).to(imgs.device)
                neg_vec  = neg_vec.unsqueeze(1)                    # (B,1,512)
                neg_scores.append( rerank(txt_tok, neg_vec) )      # (B,)
            s_neg = torch.stack(neg_scores, 1)                     # (B,L-1)

            loss  = ranknet(s_pos.unsqueeze(1), s_neg)
            opt.zero_grad(); loss.backward(); opt.step()

        print(f"[Stage-2 {ep}]  rank_loss={loss.item():.4f}")

    torch.save(rerank.state_dict(), f"{args.out}/rerank.pth")


# ---------- CLI ------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_jsonl", required=True)   # unified_data.jsonl
    p.add_argument("--corpus",     required=True)   # semantic_corpus.json
    p.add_argument("--out",        default="ckpts")
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--bs",    type=int, default=8)
    p.add_argument("--tau",   type=float, default=0.07)
    p.add_argument("--topk",  type=int, default=4)
    p.add_argument("--lmb",   type=float, default=0.1)
    p.add_argument("--lr1", type=float, default=2e-4)
    p.add_argument("--ep1", type=int,   default=8)
    p.add_argument("--L",   type=int,   default=50)
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--ep2", type=int,   default=5)
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    ds  = UnifiedDataset(args.data_jsonl, num_views=args.views)
    txt_enc, vis_enc = stage1(args, ds)
    index, all_v = build_faiss(ds, vis_enc, args)
    stage2(args, ds, txt_enc, vis_enc, index, all_v)
    print("✓ Training done →", args.out)
