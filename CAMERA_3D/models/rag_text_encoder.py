# -*- coding: utf-8 -*-
"""
RAGTextEncoder = DenseRetriever + Cross-Fusion
輸出 text_vec 512 維,並可回傳 retriever NLL
"""
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from models.dense_retriever import DenseRetriever, retriever_nll
from pathlib import Path
import json

class FusionBlock(nn.Module):
    def __init__(self, d=768, h=8):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d, d*4), nn.GELU(),
                                  nn.Linear(d*4, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x


class CrossFusion(nn.Module):
    def __init__(self, L=2, d=768, h=8):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(d, h) for _ in range(L)])
        self.proj = nn.Linear(d, 512)
    def forward(self, seq):
        for blk in self.blocks: seq = blk(seq)
        return self.proj(seq[:, 0])          # CLS




class RAGTextEncoder(nn.Module):
    def __init__(self, unified_jsonl, top_k=4):
        super().__init__()
        self.top_k = top_k

        # 自動從 unified_data.jsonl 抽出語意 corpus
        corpus = []
        for line in open(unified_jsonl):
            item = json.loads(line)
            oid = item["obj_id"]
            for t in item.get("corpus_texts", []):
                corpus.append({"text": t, "obj_id": oid})

        # 存成暫存 jsonl（僅第一次）
        temp_path = Path(unified_jsonl).with_name("temp_corpus.jsonl")
        if not temp_path.exists():
            with open(temp_path, "w") as f:
                for c in corpus:
                    f.write(json.dumps(c) + "\n")

        # 直接從 temp_corpus.jsonl 建 DenseRetriever
        self.retriever = DenseRetriever(str(temp_path), device="cuda", batch=24)
        self.fusion = CrossFusion()

    def forward(self, q_list, obj_ids=None, return_loss=False):
        q_vec, _, _, ctx, ret_loss = self.retriever(
            q_list, obj_ids, self.top_k)

        B = len(q_list)
        flat = [t for i in range(B) for t in ([q_list[i]] + ctx[i])]
        tok  = self.retriever.tok(flat, return_tensors="pt",
                                  padding=True, truncation=True
                                  ).to(q_vec.device)
        tok  = self.retriever.qenc(**tok).last_hidden_state[:, 0]
        tok  = tok.view(B, -1, 768)                # (B,1+k,768)

        vec  = self.fusion(tok)                    # (B,512)

        if return_loss:
            return vec, ret_loss, tok              # token_seq 給 reranker
        return vec, torch.tensor(0., device=vec.device), tok
