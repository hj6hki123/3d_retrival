import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, num_views=12, split="train", transform=None):
        """
        初始化 ModelNet40 多視角數據集
        :param root_dir: 渲染影像數據集的根目錄
        :param num_views: 每個 3D 物件的視角數量 (12 or 20)
        :param split: "train" 或 "test"
        :param transform: 影像轉換 (augmentation, normalization)
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.split = split  # 訓練 or 測試
        self.transform = transform

        # **獲取所有類別**
        self.categories = sorted(os.listdir(root_dir))
        
        # **載入所有影像路徑與對應的標籤**
        self.data = []
        for class_idx, category in enumerate(self.categories):
            category_path = os.path.join(root_dir, category, split)  #  可選 train 或 test
            if not os.path.exists(category_path):
                continue
            for obj_name in os.listdir(category_path):
                obj_folder = os.path.join(category_path, obj_name)
                view_images = sorted(
                    [os.path.join(obj_folder, img) for img in os.listdir(obj_folder) if img.endswith(".png")]
                )
                if len(view_images) == num_views:
                    self.data.append((view_images, class_idx))
        print(f"載入 {self.split} 數據集: {len(self.data)} 筆資料，類別數: {len(self.categories)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        取得單個 3D 物件的多視角影像與標籤
        """
        view_paths, label = self.data[idx]

        # **讀取所有視角影像**
        images = [Image.open(view_path).convert("RGB") for view_path in view_paths]

        # **應用 Transform**
        if self.transform:
            images = [self.transform(img) for img in images]

        # **轉換為 Tensor**
        images = torch.stack(images)  # shape: (num_views, C, H, W)
        return images, label


def get12_views_dataloader(split = "train", batch_size = 8):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 確保影像大小符合 VGG/ResNet
    transforms.ToTensor(),  # 轉換為 PyTorch Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化
    ])
    # **載入訓練數據**
    train_dataset = ModelNet40Dataset(
        root_dir="modelnet40-princeton-3d-object-dataset/rendered_views_12",
        num_views=12,
        split=split,
        transform=transform
    )
    shuffle_flag = True if split == "train" else False
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_flag)
    return data_loader

def get20_views_dataloader(split = "train",batch_size = 8):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 影像大小符合 VGG
    transforms.ToTensor(),  # 轉換為 PyTorch Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化
    ])
    train_dataset = ModelNet40Dataset(
        root_dir="modelnet40-princeton-3d-object-dataset/rendered_views_20",
        num_views=20,
        split=split,
        transform=transform
    )
    shuffle_flag = True if split == "train" else False
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_flag)
    return data_loader
    
    
