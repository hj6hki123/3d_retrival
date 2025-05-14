# save_features.py  ── 依 checkpoint 動態對齊 vocab_size
import torch, torch.nn.functional as F
from modelnet40_text_datasets import get_dataloader
from MHSAN_text import (
    MHSAN, TextEncoder, LSTMTextEncoder,
    PretrainedBertTextEncoder, PretrainedCLIPTextEncoder,
    MHSAN_with_Text
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 0) 基本參數
num_views, num_layers, top_k = 12, 3, 6
text_encoder_type = "gru"                         # "gru","lstm","bert","clip"
checkpoint_path  = f"models/MHSAN_with_{text_encoder_type}_{num_views}_best.pth"

# 1) 先讀 checkpoint → 取 vocab_size
state_tmp = torch.load(checkpoint_path, map_location="cpu")
vocab_size_ckpt = state_tmp["text_encoder.embedding.weight"].shape[0]
print(f"[info] ckpt embedding size = {vocab_size_ckpt}")

# 2) 視覺模型
mhsan_model = MHSAN(num_views, 512, 8, num_layers, top_k)

# 3) 文字編碼器
if text_encoder_type == "gru":
    text_encoder = TextEncoder(vocab_size_ckpt, 300, 256, 1)
elif text_encoder_type == "lstm":
    text_encoder = LSTMTextEncoder(vocab_size_ckpt, 300, 256, 1)
elif text_encoder_type == "bert":
    text_encoder = PretrainedBertTextEncoder()
elif text_encoder_type == "clip":
    text_encoder = PretrainedCLIPTextEncoder()
else:
    raise ValueError("unknown text_encoder_type")

# 4) 多模態模型 & 載權重
model = MHSAN_with_Text(mhsan_model, text_encoder).to(device)
model.load_state_dict(state_tmp)
model.eval()

# 5) DataLoader
root_dir = f"modelnet40-princeton-3d-object-dataset/rendered_views_{num_views}"
test_loader = get_dataloader(
    root_dir, num_views=num_views,
    split="test", batch_size=16,
    text_encoder_type=text_encoder_type
)
label2category = test_loader.dataset.categories

# 6) 抽特徵
all_feats, all_labels = [], []
with torch.no_grad():
    for images, texts, labels in test_loader:
        images = images.to(device)

        # 若 texts 是 dict，需逐 key 搬到 GPU
        if isinstance(texts, dict):
            for k in texts: texts[k] = texts[k].to(device)
        else:
            texts = texts.to(device)

        _, vis_feature, _ = model(images, texts)   # (B,512)
        all_feats.append(vis_feature.cpu())
        all_labels.append(labels)

all_feats  = torch.cat(all_feats, 0)   # (N,512)
all_labels = torch.cat(all_labels, 0)  # (N,)

# 7) 儲存
torch.save({
    "gallery_feats":  all_feats,
    "gallery_labels": all_labels,
    "label2category": label2category
}, f"gallery/{text_encoder_type}.pt")

print(f"[done] feature file saved to gallery/{text_encoder_type}.pt")
