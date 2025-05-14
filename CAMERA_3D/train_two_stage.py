#!/usr/bin/env python
"""
兩階段訓練：
Stage-1  RAGTextEncoder + GFMVEncoder  →  InfoNCE
Stage-2  建 FAISS 候選  +  token MLP Reranker  →  RankNet
"""

import argparse, torch, faiss, os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.objaverse_llava_mv import ObjaverseLlavaMV
from models.gf_mv_encoder       import GFMVEncoder
from models.rag_text_encoder    import RAGTextEncoder


# --------------- 損失 ----------------------------------------
def info_nce(v,t,tau=0.07):
    v = F.normalize(v,2,1); t = F.normalize(t,2,1)
    sim = v @ t.T / tau
    return F.cross_entropy(sim, torch.arange(sim.size(0), device=sim.device))

def ranknet(pos_s,neg_s):
    return F.softplus(neg_s-pos_s).mean()

# --------------- Reranker -----------------------------------
class Reranker(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim,1))
    def forward(self, q, v):             # (B,512) , (B,512)
        return self.mlp(torch.cat([q,v],1))   # (B,1)

# --------------- Stage-1 ------------------------------------
def train_stage1(args, ds_train):
    dl = DataLoader(ds_train, args.bs, shuffle=True,
                    num_workers=4, drop_last=True)

    txt_enc = RAGTextEncoder(top_k=args.topk).cuda()
    vis_enc = GFMVEncoder(num_views=args.views).cuda()
    opt = torch.optim.AdamW(
        list(txt_enc.parameters())+list(vis_enc.parameters()), lr=args.lr1)

    for ep in range(args.ep1):
        for cap,imgs in dl:
            imgs = imgs.cuda()
            txt_feat = txt_enc(list(cap))
            vis_feat,_ = vis_enc(imgs, txt_feat)

            loss = info_nce(vis_feat, txt_feat, args.tau)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[Coarse E{ep}] loss={loss.item():.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"txt":txt_enc.state_dict(),
                "vis":vis_enc.state_dict()},
               f"{args.out}/enc_stage1.pth")
    return txt_enc, vis_enc

# --------------- 建 FAISS 候選 --------------------------------
def build_index(ds, txt_enc, vis_enc, args):
    dl = DataLoader(ds, args.bs, num_workers=4)
    all_v=[]
    with torch.no_grad():
        for cap,imgs in dl:
            imgs=imgs.cuda()
            v,_=vis_enc(imgs, None)
            all_v.append(F.normalize(v,2,1).cpu())
    all_v = torch.cat(all_v).numpy()
    index = faiss.IndexFlatIP(512)
    index.add(all_v)
    return index, all_v

# --------------- Stage-2 ------------------------------------
def train_stage2(ds_train, txt_enc, vis_enc, index, all_v, args):
    txt_enc.eval(); vis_enc.eval()
    for p in txt_enc.parameters(): p.requires_grad=False
    for p in vis_enc.parameters(): p.requires_grad=False

    rerank = Reranker().cuda()
    opt = torch.optim.AdamW(rerank.parameters(), lr=args.lr2)
    dl = DataLoader(ds_train, args.bs, shuffle=True,num_workers=4,drop_last=True)

    for ep in range(args.ep2):
        for cap,imgs in dl:
            imgs = imgs.cuda()
            with torch.no_grad():
                q = txt_enc(list(cap))                    # (B,512)
                sim, idx = index.search(q.cpu().numpy(), args.L)
                cand = torch.from_numpy(all_v[idx.reshape(-1)]
                        ).view(q.size(0), args.L, -1).to("cuda")
                cand = F.normalize(cand,2,-1)

            pos   = cand[:,0,:]                          # (B,512)
            neg   = cand[:,1:,:]                         # (B,L-1,512)
            s_pos = rerank(q, pos)                      # (B,1)

            q_rep = q.unsqueeze(1).expand(-1,args.L-1,-1).reshape(-1,512)
            s_neg = rerank.mlp(torch.cat([q_rep, neg.reshape(-1,512)],1)
                     ).view(-1, args.L-1)

            loss = ranknet(s_pos, s_neg)

            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[Fine E{ep}] loss={loss.item():.4f}")

    torch.save(rerank.state_dict(), f"{args.out}/rerank.pth")

# --------------- CLI ----------------------------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--img_root", required=True)
    p.add_argument("--out", default="ckpts")
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr1", type=float, default=2e-4)
    p.add_argument("--ep1", type=int, default=6)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--topk",type=int, default=4)
    p.add_argument("--L",   type=int, default=50)
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--ep2", type=int, default=4)
    return p.parse_args()

if __name__ == "__main__":
    args=cli()
    ds_train = ObjaverseLlavaMV(args.jsonl,args.img_root,
                                num_views=args.views,caption_strategy="random")
    # Stage-1
    txt_enc, vis_enc = train_stage1(args, ds_train)
    # 建索引
    index, all_v = build_index(ds_train, txt_enc, vis_enc, args)
    # Stage-2
    train_stage2(ds_train, txt_enc, vis_enc, index, all_v, args)
