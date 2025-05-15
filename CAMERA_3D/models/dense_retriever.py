# -*- coding: utf-8 -*-
"""
DenseRetriever：DPR 風格可微檢索器
  • memory_vec = nn.Parameter，可被梯度更新
"""
import json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class DenseRetriever(torch.nn.Module):
    def __init__(self, corpus_jsonl:str, dim:int=768):
        super().__init__()
        self.tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.qenc = AutoModel.from_pretrained("bert-base-uncased")
        self.kenc = AutoModel.from_pretrained("bert-base-uncased")
        corpus = [json.loads(l) for l in open(corpus_jsonl)]
        self.texts   = [c["text"]   for c in corpus]
        self.obj_ids = [c["obj_id"] for c in corpus]   # <‑‑ NEW
        
        with torch.no_grad():
            vecs = []
            for i in range(0, len(self.texts), 64):
                # 1) 讓 tokenizer 輸出留在 CPU
                enc = self.tok(self.texts[i:i+64],
                            return_tensors="pt",
                            padding=True,
                            truncation=True)          # <-- 移除 .to("cuda")

                # 2) self.kenc 預設在 CPU，因此 enc 與 kenc 同裝置
                v = self.kenc(**enc).last_hidden_state[:, 0]  # CLS
                vecs.append(torch.nn.functional.normalize(v, 2, -1))

        self.memory_vec = torch.nn.Parameter(torch.cat(vecs))  # (N,768)

    # 取正樣本 idx  (第一個與 query 同 obj_id 的 memory 條目)
    def pos_index_from_obj(self, obj_ids):                 # list[str] len=B
        mapping = {o: i for i,o in enumerate(self.obj_ids)}
        return torch.tensor([mapping[o] for o in obj_ids], device=self.memory_vec.device)

    def forward(self, q_list, obj_ids=None, topk=4):
        q    = self.query_encode(q_list)
        sims = q @ F.normalize(self.memory_vec,2,-1).T
        idx  = sims.topk(topk,-1).indices
        ctx  = [[self.texts[j] for j in row] for row in idx.cpu()]
        if obj_ids is not None:
            pos = self.pos_index_from_obj(obj_ids)
            loss = F.cross_entropy(sims, pos)
            return q, sims, idx, ctx, loss           # <‑‑ return retriever NLL
        return q, sims, idx, ctx, torch.tensor(0.)


def retriever_nll(logits: torch.Tensor, pos_idx: torch.Tensor):
    return torch.nn.functional.cross_entropy(logits, pos_idx)
