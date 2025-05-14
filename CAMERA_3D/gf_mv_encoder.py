"""GateFusion‑MV: Language‑Guided Multi‑View 3‑D Encoder
輸入 :  多視角影像 (B,V,C,H,W)  +  文字向量 (B,512)
輸出 :  global_vec (B,512)  ,  token_seq (B,V+2,512)
用途 :  global_vec → ANN  /  token_seq → 跨模態 Rerank
"""
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, n_tokens: int):
        super().__init__()
        self.register_buffer("pe", self._build(dim, n_tokens))

    def _build(self, d, n):
        pos = torch.arange(n).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe  = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)          # (1,n,d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---------- Self‑Attention Block ----------
class Block(nn.Module):
    def __init__(self, dim=512, heads=8, p=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, p, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(p), nn.Linear(dim*4, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x

# ---------- GF‑MV Encoder ----------
class GFMVEncoder(nn.Module):
    def __init__(self, num_views=12, dim=512, heads=8, layers=3, use_gate=True):
        super().__init__()
        self.dim, self.V = dim, num_views
        self.use_gate = use_gate

        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.feat2d = nn.Sequential(*list(vgg.features.children()))
        self.map2d  = nn.Linear(512*7*7, dim)

        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        if use_gate:
            self.gate_proj = nn.Linear(512, dim)
            offset = 1
        else:
            offset = 0
        self.pos = PositionalEncoding(dim, num_views + 1 + offset)  # CLS + views (+GATE)

        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.pool   = nn.AdaptiveMaxPool1d(1)
        self.proj   = nn.Linear(dim, 512)

    def forward(self, imgs: torch.Tensor, txt_vec: torch.Tensor | None=None):
        """imgs: (B,V,C,H,W)   txt_vec: (B,512)
        returns: global_vec, token_seq
        """
        B, V, C, H, W = imgs.shape
        feats = self.feat2d(imgs.view(B*V, C, H, W))
        feats = self.map2d(feats.flatten(1)).view(B, V, -1)   # (B,V,dim)

        tokens = [self.cls_token.expand(B,-1,-1)]
        if self.use_gate and txt_vec is not None:
            gate = self.gate_proj(txt_vec).unsqueeze(1)       # (B,1,dim)
            tokens.append(gate)
        tokens.append(feats)
        tokens = torch.cat(tokens, dim=1)                     # (B,V+1(+1),dim)
        tokens = self.pos(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)

        global_vec = self.pool(tokens.transpose(1,2)).squeeze(-1)  # (B,dim)
        return self.proj(global_vec), tokens