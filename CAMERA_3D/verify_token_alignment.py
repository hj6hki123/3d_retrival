#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify that visual tokens (vis_tok.npy) match the global vectors in FAISS.
Also verifies that index search returns the correct token.

Usage:
  python verify_token_alignment.py \
      --data datasets/unified_data.jsonl \
      --ckpt ckpts_0523 \
      --topk 5
"""

import os, argparse, faiss, torch
import numpy as np
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder import RAGTextEncoder
from models.gf_mv_encoder import GFMVEncoder
import torch.nn.functional as F

@torch.no_grad()
def verify(args):
    # Load dataset
    ds = UnifiedDataset(args.data, num_views=12)

    # Load encoders
    txt_enc = RAGTextEncoder(args.data, top_k=args.topk).cuda().eval()
    vis_enc = GFMVEncoder(num_views=12).cuda().eval()
    ckpt = torch.load(os.path.join(args.ckpt, "enc1.pth"), map_location="cuda")
    txt_enc.load_state_dict(ckpt["txt"])
    vis_enc.load_state_dict(ckpt["vis"])

    # Load FAISS and tokens
    index = faiss.read_index(os.path.join(args.ckpt, "faiss_index.bin"))
    all_tok = np.load(os.path.join(args.ckpt, "vis_tok.npy")).astype(np.float32)

    # Select 1 query sample
    q, img, obj_id = ds[0]
    img = img.unsqueeze(0).cuda()
    q_vec, _, txt_tok = txt_enc([q], [obj_id])
    v_vec, v_tok = vis_enc(img, q_vec)

    # Normalize and compare top-1 cosine sim with FAISS
    v_vec = F.normalize(v_vec, 2, 1)
    q_vec = F.normalize(q_vec, 2, 1)

    # Check FAISS
    scores, idx = index.search(q_vec.detach().cpu().numpy(), args.topk)
    print(f"Target obj_id: {obj_id}")
    print("Top-k results:")
    for i in range(args.topk):
        found_id = ds.items[idx[0][i]]["obj_id"]
        score = scores[0][i]
        print(f"[{i}] obj_id={found_id}  score={score:.6f}")

    # Compare token vectors
    tok_live = F.normalize(v_tok.squeeze(0).cpu(), p=2, dim=-1)
    cached   = torch.from_numpy(all_tok[idx[0][0]]).float()

    token_diff = F.mse_loss(tok_live, cached)
    print(f"\nToken alignment MSE (top-1 match): {token_diff.item():.6e}")
    if token_diff.item() < 1e-5:
        print(" Token vectors match FAISS index (good).")
    else:
        print(" Token vectors do not align (check indexing logic).")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data", required=True)
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--topk", type=int, default=5)
    args = pa.parse_args()
    verify(args)
