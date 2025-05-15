import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, CLIPTokenizer
from tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer

################################################################################
# Dataset: 讀取圖像 & 文字描述
################################################################################
class ModelNet40Dataset_text(Dataset):
    def __init__(
        self,
        root_dir,
        num_views=12,
        split="train",
        transform=None,
        text_encoder_type="gru",
        tokenizer=None,      # 新增
        max_text_len=50      # 也可以增添可調參數
    ):
        self.root_dir = root_dir
        self.num_views = num_views
        self.split = split
        self.transform = transform
        self.text_encoder_type = text_encoder_type
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        self.categories = sorted(os.listdir(root_dir))
        self.data = []
        for class_idx, category in enumerate(self.categories):
            category_path = os.path.join(root_dir, category, split)
            if not os.path.exists(category_path):
                continue

            for obj_name in os.listdir(category_path):
                obj_folder = os.path.join(category_path, obj_name)
                view_images = sorted(
                    [os.path.join(obj_folder, img)
                     for img in os.listdir(obj_folder) if img.endswith(".png")]
                )
                desc_path = os.path.join(obj_folder, "description.txt")
                # 必須檢查 view 與描述同時存在
                if len(view_images) == num_views and os.path.exists(desc_path):
                    self.data.append((view_images, desc_path, class_idx))

        print(f"載入 {split} 數據集：{len(self.data)} 筆資料,類別數：{len(self.categories)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        view_paths, desc_path, label = self.data[idx]
        # 1) 讀取多視角影像
        images = [Image.open(path).convert("RGB") for path in view_paths]
        if self.transform:
            images = [self.transform(img) for img in images]
        images_tensor = torch.stack(images)  # (num_views, C, H, W)

        # 2) 讀取文字描述 (raw string)
        with open(desc_path, "r", encoding="utf-8") as f:
            description = f.read().strip()

        # 3) 分三種狀況：GRU/LSTM (自訂 tokenizer)、BERT/CLIP (內建 tokenizer)  
        if self.text_encoder_type in ["gru", "lstm"]:
            if self.tokenizer is None:
                raise ValueError("在 GRU 模式下必須提供自訂 tokenizer！")

            # 使用 subword tokenizer (WordPiece) 做分詞並轉成 ids
            encoded = self.tokenizer.encode(description)
            # 截斷長度
            ids = encoded.ids[:self.max_text_len]
            desc_tensor = torch.tensor(ids, dtype=torch.long)

            return images_tensor, desc_tensor, label
        else:
            # 如果是 BERT 或 CLIP,就原樣回傳字串給 collate_fn_transformer
            return images_tensor, description, label

################################################################################
# Collate Functions
################################################################################
def collate_fn_text(batch):
    images_list, desc_list, labels_list = [], [], []
    for images, desc_tensor, label in batch:
        images_list.append(images)
        desc_list.append(desc_tensor)
        labels_list.append(label)

    # 影像合併
    images_tensor = torch.stack(images_list, dim=0)  # (batch, num_views, C, H, W)
    # 文字合併
    desc_padded = pad_sequence(desc_list, batch_first=True, padding_value=0)  # (batch, max_seq_len)
    # label
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return images_tensor, desc_padded, labels_tensor




def build_subword_tokenizer(vocab_txt_path):
    """
    用 vocab.txt 建立一個 BertWordPieceTokenizer
    """
    tokenizer = BertWordPieceTokenizer(
        vocab_txt_path,
        lowercase=True,       # 如果你的 vocab 是小寫
        unk_token="[UNK]",
        pad_token="[PAD]"     # 記得要確定 vocab.txt 裡有定義這些token
    )
    return tokenizer   


def collate_fn_transformer(batch, tokenizer_type="clip"):

    images_list, text_list, labels_list = [], [], []
    for images, desc_str, label in batch:
        images_list.append(images)
        text_list.append(desc_str)
        labels_list.append(label)

    images_tensor = torch.stack(images_list, dim=0)

    # 根據 tokenizer_type 選擇 BERT 或 CLIP
    if tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif tokenizer_type == "clip":
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"未知的 tokenizer_type: {tokenizer_type}")

    # 做 batch tokenize
    encoded = tokenizer.batch_encode_plus(
        text_list,
        padding="longest",
        truncation=True,
        max_length=50,
        return_tensors="pt"
    )
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return images_tensor, encoded, labels_tensor

################################################################################
# 封裝: get_dataloader
################################################################################
def get_dataloader(
    root_dir,
    num_views=12,
    split="train",
    batch_size=8,
    text_encoder_type="gru",
    vocab_txt_path="tokenizer_model/vocab.txt"
):
    from tokenizers import BertWordPieceTokenizer
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 根據模式載入 tokenizer / 設定 collate_fn
    tokenizer = None
    if text_encoder_type in ["gru", "lstm"]:
        # 用 vocab.txt 建立子詞 tokenizer
        tokenizer = BertWordPieceTokenizer(
            vocab_txt_path,
            lowercase=True,
            unk_token="[UNK]",
            pad_token="[PAD]"
        )
        collate_fn = collate_fn_text
    elif text_encoder_type == "bert":
        collate_fn = lambda b: collate_fn_transformer(b, tokenizer_type="bert")
    elif text_encoder_type == "clip":
        collate_fn = lambda b: collate_fn_transformer(b, tokenizer_type="clip")
    else:
        raise ValueError(f"未知 text_encoder_type: {text_encoder_type}")

    # 建立 dataset
    dataset = ModelNet40Dataset_text(
        root_dir=root_dir,
        num_views=num_views,
        split=split,
        transform=transform,
        text_encoder_type=text_encoder_type,
        tokenizer=tokenizer     # 傳進 Dataset 裡
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn
    )
    return data_loader