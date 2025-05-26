#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CAMERA_3D (baseline + rerank)
 • Recall@1/5/10, mAP@10, NDCG@10
 • QPS、平均 GPU/CPU 記憶體
 • 自動判斷 reranker/vis_tok 是否存在
用法:
  python evaluator_v2.py \
      --data datasets/unified_data.jsonl \
      --ckpt ckpts_fix \
      --bs 32 \
      --views 12 \
      --topk 4
"""

import os, time, argparse, psutil, faiss, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder  import RAGTextEncoder
from models.gf_mv_encoder     import GFMVEncoder
from models.cross_modal_reranker import CrossModalReranker
from train_two_stage          import build_faiss
from tqdm import tqdm
import numpy as np

# ----------------- ranking metrics ------------------------------------------
def dcg(rel):
    rel = np.asarray(rel)
    log = np.log2(np.arange(2, rel.size + 2))
    return (rel / log).sum()

def ndcg(rel, k):
    rel_k = rel[:k]
    idcg  = dcg(sorted(rel, reverse=True)[:k])
    return dcg(rel_k) / idcg if idcg > 0 else 0.

def apk(rel, k):
    rel = np.asarray(rel[:k])
    if rel.sum() == 0: return 0.
    prec = rel.cumsum() / (np.arange(rel.size) + 1)
    return (prec * rel).sum() / rel.sum()

# ----------------- evaluator -------------------------------------------------
def evaluate(args):
    ds   = UnifiedDataset(args.data, num_views=args.views)
    ks   = (1,5,10)
    loader = DataLoader(ds, args.bs, num_workers=4)

    # ---------- load encoders ----------------
    ck   = torch.load(os.path.join(args.ckpt, "enc1.pth"), map_location="cuda")
    txt  = RAGTextEncoder(args.data, top_k=args.topk).cuda().eval()
    txt.load_state_dict(ck["txt"])
    vis  = GFMVEncoder(args.views).cuda().eval()
    vis.load_state_dict(ck["vis"])

    # ---------- build faiss (global vector) ------------------
    L = max(ks[-1], 50)   # top-L (用於 rerank，L>=10)
    print(f"[INFO] Build FAISS index with L={L}")
    index, all_v = build_faiss(ds, txt, vis, args)

    # ---------- reranker & vis_tok（自動切換）-----------------
    rerank_path = os.path.join(args.ckpt, "rerank.pth")
    vis_tok_path = os.path.join(args.ckpt, "vis_tok.npy")
    use_rerank = os.path.exists(rerank_path) and os.path.exists(vis_tok_path)
    if use_rerank:
        print("[INFO] Found rerank.pth & vis_tok.npy, will perform rerank evaluation.")
        reranker = CrossModalReranker().cuda().eval()
        reranker.load_state_dict(torch.load(rerank_path))
        all_tok = np.load(vis_tok_path).astype(np.float32)
    else:
        print("[INFO] Only FAISS (baseline) evaluation.")

    # ---------- metrics accumulator ----------
    hit_base = {k:0 for k in ks}
    hit_rerank = {k:0 for k in ks}
    APs_base, APs_rerank = [], []
    NDCGs_base, NDCGs_rerank = [], []
    tot   = 0
    gpu_mem, cpu_mem = 0, 0
    t0    = time.time()

    with torch.no_grad():
        for q, _, obj_ids in tqdm(loader, desc="Eval"):
            gpu0 = torch.cuda.memory_allocated() / 1e6
            cpu0 = psutil.Process(os.getpid()).memory_info().rss / 1e6

            q_vec, _, txt_tok = txt(list(q), list(obj_ids))
            sims, idx = index.search(q_vec.cpu().numpy(), L)

            gpu_mem += (torch.cuda.memory_allocated()/1e6 - gpu0)
            cpu_mem += (psutil.Process(os.getpid()).memory_info().rss/1e6 - cpu0)

            for b, oid in enumerate(obj_ids):
                tot += 1
                # ========== Baseline ==========
                base_oids = [ds.items[j]["obj_id"] for j in idx[b][:ks[-1]]]
                rel_list = [1 if oid == rid else 0 for rid in base_oids]
                for k in ks:
                    hit_base[k] += rel_list[:k].count(1)
                APs_base.append(apk(rel_list, 10))
                NDCGs_base.append(ndcg(rel_list, 10))

                # ========== Rerank（如啟用） ==========
                if use_rerank:
                    txt_t = txt_tok[b:b+1].expand(L, -1, -1)
                    vis_t = torch.from_numpy(all_tok[idx[b]]).float().cuda()
                    scores = reranker(txt_t, vis_t)
                    rerank_order = torch.argsort(scores, descending=True).cpu().numpy()
                    rerank_oids = [ds.items[idx[b][j]]["obj_id"] for j in rerank_order[:ks[-1]]]
                    rel_rerank = [1 if oid == rid else 0 for rid in rerank_oids]
                    for k in ks:
                        hit_rerank[k] += rel_rerank[:k].count(1)
                    APs_rerank.append(apk(rel_rerank, 10))
                    NDCGs_rerank.append(ndcg(rel_rerank, 10))

    dt   = time.time() - t0
    print("\n=============  Evaluation  =============")
    print("== Baseline (FAISS) ==")
    for k in ks:
        print(f"Recall@{k:<2}: { hit_base[k] / tot :6.3%}")
    print(f"mAP@10  : {np.mean(APs_base):6.3%}")
    print(f"NDCG@10 : {np.mean(NDCGs_base):6.3%}")
    print(f"QPS     : { tot / dt :6.1f}  query/s")
    print(f"Avg GPU : { gpu_mem / tot :6.1f}  MB/query")
    print(f"Avg CPU : { cpu_mem / tot :6.1f}  MB/query")
    if use_rerank:
        print("\n== Rerank (Cross-Modal) ==")
        for k in ks:
            print(f"Recall@{k:<2}: { hit_rerank[k] / tot :6.3%}")
        print(f"mAP@10  : {np.mean(APs_rerank):6.3%}")
        print(f"NDCG@10 : {np.mean(NDCGs_rerank):6.3%}")
    print("========================================\n")

# ----------------- CLI -------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data",   required=True, help="unified_data.jsonl")
    pa.add_argument("--ckpt",   required=True, help="checkpoint directory")
    pa.add_argument("--bs",     type=int, default=32)
    pa.add_argument("--views",  type=int, default=12)
    pa.add_argument("--topk",   type=int, default=4)
    args = pa.parse_args()
    evaluate(args)
