import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from semantic_memory import SemanticMemory

class CrossFusion(nn.Module):
    def __init__(self, dim=768, heads=8, layers=2):
        super().__init__()
        block = lambda: nn.TransformerEncoderLayer(dim, heads, dim*4, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(block(), layers)
        self.proj= nn.Linear(dim, 512)
    def forward(self, x):       # (B,1+k,768)
        fused = self.enc(x)     # (B,1+k,768)
        return self.proj(fused[:,0])  # CLS → 512

class RAGTextEncoder(nn.Module):
    def __init__(self, top_k=4):
        super().__init__()
        self.mem  = SemanticMemory(top_k=top_k)
        self.tok  = self.mem.tok                    # 共用 tokenizer
        self.base = self.mem.enc                    # 共用 BERT
        self.fuse = CrossFusion()
    @torch.no_grad()
    def _encode(self, txts):
        enc = self.tok(txts, return_tensors='pt', truncation=True, padding=True).to('cuda')
        return self.base(**enc).last_hidden_state[:,0]
    def forward(self, raw_texts: list[str]):
        batch_ctx = []
        for q in raw_texts:
            ctx,_ = self.mem.retrieve(q)
            batch_ctx.append([q]+ctx)               # 1+top_k
        flat = [s for lst in batch_ctx for s in lst]
        vecs = self._encode(flat).view(len(raw_texts), -1, 768)
        return self.fuse(vecs)                      # (B,512)