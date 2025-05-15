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
        # self.items = [json.loads(l) for l in open(jsonl_path)]
        self.items = [json.loads(l) for l in open(jsonl_path) if l.strip() and json.loads(l)["views"]]
        self.num_views = num_views
        self.tr = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item   = self.items[idx]
        query  = item["query"]
        obj_id = item["obj_id"]              
        # imgs   = torch.stack([
        #     self.tr(Image.open(p).convert("RGB"))
        #     for p in sorted(item["views"] )[:self.num_views]
        # ])
        paths = sorted(item["views"])[:self.num_views]
        if len(paths) == 0:                  # 再檢查一次
            raise RuntimeError(f"{obj_id} has 0 valid views")

        imgs = [self.tr(Image.open(p).convert("RGB")) for p in paths]

        # 若不足 num_views，就把最後一張複製填滿
        while len(imgs) < self.num_views:
            imgs.append(imgs[-1].clone())
        imgs = torch.stack(imgs)
        
        
        
        return query, imgs.float(), obj_id    # <‑‑ return obj_id
