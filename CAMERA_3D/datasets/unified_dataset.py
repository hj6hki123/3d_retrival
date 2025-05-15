# -*- coding: utf-8 -*-
"""
讀 unified_data.jsonl：
{ "query": str, "obj_id": str, "views": [img paths], "corpus_texts": [...] }
回傳  query(str)  ,  imgs  (Tensor V,C,H,W)
"""
import json, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class UnifiedDataset(Dataset):
    def __init__(self, jsonl_path, image_size=224, num_views=12):
        super().__init__()
        self.items = [json.loads(l) for l in open(jsonl_path)]
        self.num_views = num_views
        self.tr = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item   = self.items[idx]
        query  = item["query"]
        obj_id = item["obj_id"]              # <‑‑ NEW
        imgs   = torch.stack([
            self.tr(Image.open(p).convert("RGB"))
            for p in sorted(item["views"] )[:self.num_views]
        ])
        return query, imgs.float(), obj_id    # <‑‑ return obj_id
