# -*- coding: utf-8 -*-
"""
GateFusion-MV:Language-Guided Multi-View Encoder
輸入 :  imgs (B,V,C,H,W) , txt_vec (B,512 or None)
輸出 :  global_vec (B,512) , token_seq (B,V(+2),512)
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class PosEnc(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        pos = torch.arange(n).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.)/d))
        pe  = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # (B,T,D)
        return x + self.pe[:, :x.size(1)]


class Block(nn.Module):
    def __init__(self, d=512, h=8, p=.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d, d*4), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(d*4, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x


class GFMVEncoder(nn.Module):
    def __init__(self, num_views=12, dim=512, heads=8, layers=3):
        super().__init__()
        self.dim, self.V = dim, num_views

        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.feat2d = nn.Sequential(*list(vgg.features.children()))
        self.map2d  = nn.Linear(512*7*7, dim)

        self.cls_tok  = nn.Parameter(torch.randn(1, 1, dim))
        self.gate_proj = nn.Linear(512, dim)

        self.pos = PosEnc(dim, num_views + 2)     # CLS+GATE+V
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.out_proj = nn.Linear(dim, 512)

    def forward(self, imgs, txt_vec=None):
        B, V, C, H, W = imgs.shape
        f = self.feat2d(imgs.view(B*V, C, H, W))
        f = self.map2d(f.flatten(1)).view(B, V, -1)           # (B,V,512)
        if txt_vec is None:
            txt_vec = torch.zeros(imgs.size(0), 512, device=imgs.device)
        gate = self.gate_proj(txt_vec).unsqueeze(1) if txt_vec is not None \
               else self.cls_tok.expand(B, -1, -1)            # 當無 txt_vec 時用 CLS
        tok = torch.cat([self.cls_tok.expand(B,-1,-1), gate, f], 1)
        tok = self.pos(tok)

        for blk in self.blocks:
            tok = blk(tok)

        global_vec = self.pool(tok.transpose(1,2)).squeeze(-1)
        return self.out_proj(global_vec), tok                  # (B,512) , (B,T,512)
