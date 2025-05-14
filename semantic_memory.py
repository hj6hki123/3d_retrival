import faiss, torch, json
from transformers import AutoTokenizer, AutoModel

class SemanticMemory:
    def __init__(self, idx='sem_mem.index', meta='sem_mem.meta',
                 model='bert-base-uncased', top_k=4):
        self.index = faiss.read_index(idx)         # 讀 FAISS 索引
        self.meta  = json.load(open(meta))         # 讀 caption 清單
        self.top_k = top_k                         # 每次取多少條
        self.tok   = AutoTokenizer.from_pretrained(model)
        self.enc   = AutoModel.from_pretrained(model).cuda().eval()
    @torch.no_grad()
    def _vec(self, txt):
        t = self.tok(txt, return_tensors='pt',
                     truncation=True, padding=True).to('cuda')
        v = self.enc(**t).last_hidden_state[:,0]
        return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
    
    def retrieve(self, query):
        D,I = self.index.search(self._vec(query), self.top_k)
        return [self.meta[i]['text'] for i in I[0]], torch.tensor(D[0])
