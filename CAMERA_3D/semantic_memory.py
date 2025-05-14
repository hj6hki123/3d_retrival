# ============================================================
#  semantic_memory.py  ----  線上可微檢索器
# ============================================================

import json, faiss, torch
from transformers import AutoTokenizer, AutoModel

class SemanticMemory:
    """ANN 檢索 + 向量可回傳梯度 (更新 self.enc)
    """
    def __init__(self, idx='sem_mem.index', meta='sem_mem.meta', model='bert-base-uncased', top_k=4):
        self.index = faiss.read_index(idx)
        self.meta  = json.load(open(meta))           # list[{id,text}]
        self.top_k = top_k
        self.tok   = AutoTokenizer.from_pretrained(model)
        self.enc   = AutoModel.from_pretrained(model).cuda()

    @torch.no_grad()
    def _vec(self, txt):
        t   = self.tok(txt, return_tensors='pt', truncation=True, padding=True).to('cuda')
        vec = self.enc(**t).last_hidden_state[:,0]
        return torch.nn.functional.normalize(vec, dim=-1).cpu().numpy()

    def retrieve(self, query: str):
        D,I = self.index.search(self._vec(query), self.top_k)
        texts = [self.meta[i]['text'] for i in I[0]]
        return texts, torch.tensor(D[0])             # 文字片段 + 內積分數