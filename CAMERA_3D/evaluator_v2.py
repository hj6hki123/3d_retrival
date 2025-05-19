#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CAMERA model
  • Recall@1/5/10
  • mAP
  • NDCG@10
  • QPS   (query per second)
  • GPU / CPU RAM per-query (averaged)

Usage
  python evaluator_v2.py \
      --data   datasets/unified_data.jsonl \
      --corpus datasets/semantic_corpus.jsonl \
      --ckpt   ckpts \
      --bs     32          # eval batch
"""
import os, time, argparse, psutil, faiss, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder  import RAGTextEncoder
from models.gf_mv_encoder     import GFMVEncoder
from train_two_stage          import build_faiss
from tqdm import tqdm
import numpy as np

# ----------------- ranking metrics ------------------------------------------
def dcg(rel):
    """rel: relevance list  (1/0)"""
    rel = np.asarray(rel)
    log = np.log2(np.arange(2, rel.size + 2))
    return (rel / log).sum()

def ndcg(rel, k):
    """rel: relevance list sorted by system score"""
    rel_k = rel[:k]
    idcg  = dcg(sorted(rel, reverse=True)[:k])
    return dcg(rel_k) / idcg if idcg > 0 else 0.

def apk(rel, k):
    """average precision @k"""
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
    txt  = RAGTextEncoder(args.corpus, top_k=args.topk).cuda().eval()
    txt.load_state_dict(ck["txt"])
    vis  = GFMVEncoder(args.views).cuda().eval()
    vis.load_state_dict(ck["vis"])

    # ---------- build faiss ------------------
    index, all_v = build_faiss(ds, vis, args)   # use bs from args

    # ---------- metrics accumulator ----------
    hit   = {k:0 for k in ks}
    APs, NDCGs = [], []
    tot   = 0
    gpu_mem, cpu_mem = 0, 0
    t0    = time.time()

    with torch.no_grad():
        for q, _, obj_ids in tqdm(loader, desc="Eval"):
            # ---------------- mem before query --------------
            gpu0 = torch.cuda.memory_allocated() / 1e6
            cpu0 = psutil.Process(os.getpid()).memory_info().rss / 1e6

            q_vec,_,_ = txt(list(q), list(obj_ids))   # (B,512)
            sims, idx = index.search(q_vec.cpu().numpy(), ks[-1])

            # ---------------- mem after  query --------------
            gpu_mem += (torch.cuda.memory_allocated()/1e6 - gpu0)
            cpu_mem += (psutil.Process(os.getpid()).memory_info().rss/1e6 - cpu0)

            for b, oid in enumerate(obj_ids):
                tot += 1
                retrieved_oids = [ds.items[j]["obj_id"] for j in idx[b]]
                rel_list = [1 if oid == rid else 0 for rid in retrieved_oids]

                # Recall
                for k in ks:
                    hit[k] += rel_list[:k].count(1)

                # AP@10  &  NDCG@10
                APs.append(apk(rel_list, 10))
                NDCGs.append(ndcg(rel_list, 10))

    dt   = time.time() - t0
    print("\n=============  Evaluation  =============")
    for k in ks:
        print(f"Recall@{k:<2}: { hit[k] / tot :6.3%}")
    print(f"mAP@10  : {np.mean(APs):6.3%}")
    print(f"NDCG@10 : {np.mean(NDCGs):6.3%}")
    print(f"QPS     : { tot / dt :6.1f}  query/s")
    print(f"Avg GPU : { gpu_mem / tot :6.1f}  MB/query")
    print(f"Avg CPU : { cpu_mem / tot :6.1f}  MB/query")
    print("========================================\n")

# ----------------- CLI -------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data",   required=True)
    pa.add_argument("--corpus", required=True)
    pa.add_argument("--ckpt",   required=True)
    pa.add_argument("--bs",     type=int, default=32)
    pa.add_argument("--views",  type=int, default=12)
    pa.add_argument("--topk",   type=int, default=4)
    args = pa.parse_args()
    evaluate(args)
