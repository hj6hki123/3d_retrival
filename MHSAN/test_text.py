# ────────────────────────────────────────────────────────────────
# test_text.py  (完整保留你的原工具函式 + 動態對齊 vocab_size)
# ────────────────────────────────────────────────────────────────
import time, torch, torch.nn.functional as F, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.manifold import TSNE
from transformers import BertTokenizer, CLIPTokenizer
from tokenizers import BertWordPieceTokenizer
from MHSAN_text import (
    MHSAN, MHSAN_with_Text,
    TextEncoder, LSTMTextEncoder,
    PretrainedBertTextEncoder, PretrainedCLIPTextEncoder
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_map(features, labels):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float, device=device)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long, device=device)

    sims = torch.matmul(features, features.T)
    N, APs = features.size(0), []
    for i in range(N):
        sim_i = sims[i].clone();  sim_i[i] = -1
        idx = torch.argsort(sim_i, descending=True)
        rel = (labels[idx] == labels[i]).float()
        nr = rel.sum().item();  precision, cnt = [], 0
        if nr == 0: continue
        for k, r in enumerate(rel, 1):
            if r == 1: cnt += 1; precision.append(cnt / k)
        APs.append(sum(precision) / nr)
    return sum(APs) / len(APs) if APs else 0.0

def compute_map_crossmodal(qf, ql, gf, gl):
    toT = lambda x: torch.tensor(x, dtype=torch.float, device=device) if not isinstance(x, torch.Tensor) else x
    qf, gf = toT(qf), toT(gf); ql, gl = ql.to(device), gl.to(device)
    sims = torch.matmul(qf, gf.T);  APs = []
    for i in range(qf.size(0)):
        idx = torch.argsort(sims[i], descending=True)
        rel = (gl[idx] == ql[i]).float();  nr = rel.sum().item()
        if nr == 0: continue
        cnt, prec = 0, []
        for k, r in enumerate(rel, 1):
            if r == 1: cnt += 1; prec.append(cnt/k)
        APs.append(sum(prec)/nr)
    return sum(APs)/len(APs) if APs else 0.0

def plot_tsne(features, labels, class_names, num_classes=40):
    ts = TSNE(2, perplexity=30, random_state=42).fit_transform(features)
    pal = np.array(sns.color_palette("husl", num_classes))
    plt.figure(figsize=(10,8))
    for i in range(num_classes):
        idx = labels == i
        if idx.sum()==0: continue
        plt.scatter(ts[idx,0], ts[idx,1], color=pal[i], label=class_names[i],
                    alpha=.7, edgecolors="k")
    plt.legend(bbox_to_anchor=(1.25,1)); plt.tight_layout(); plt.show()

def plot_tsne_multimodal(vf, tf, labels, class_names, num_classes=40):
    allf = np.concatenate([vf, tf]); md = np.array(["v"]*len(vf)+["t"]*len(tf))
    allL = np.concatenate([labels, labels])
    ts = TSNE(2, perplexity=30, random_state=42).fit_transform(allf)
    pal = np.array(sns.color_palette("husl", num_classes))
    plt.figure(figsize=(10,8))
    for i in range(num_classes):
        vis = (allL==i)&(md=="v"); txt = (allL==i)&(md=="t")
        if vis.sum(): plt.scatter(ts[vis,0], ts[vis,1], color=pal[i], marker="o", label=f"{class_names[i]}(v)", alpha=.7, edgecolors="k")
        if txt.sum(): plt.scatter(ts[txt,0], ts[txt,1], color=pal[i], marker="x", label=f"{class_names[i]}(t)", alpha=.7, edgecolors="k")
    plt.legend(bbox_to_anchor=(1.25,1)); plt.tight_layout(); plt.show()

# ───────────────────────── 主程式開始 ────────────────────────────
if __name__ == "__main__":
    # 可修改
    text_encoder_type = "gru"            # "gru","lstm","bert","clip"
    num_views, num_layers, top_k = 12, 3, 6
    checkpoint_path = f"models/MHSAN_with_{text_encoder_type}_{num_views}_best.pth"

    # ① 讀 checkpoint -> 取得 vocab_size
    state_tmp = torch.load(checkpoint_path, map_location="cpu")
    vocab_size_ckpt = state_tmp["text_encoder.embedding.weight"].shape[0]
    print(f"[info] ckpt embedding size = {vocab_size_ckpt}")

    # ② 建立視覺模型
    mhsan_model = MHSAN(num_views, 512, 8, num_layers, top_k)

    # ③ 建立文字編碼器 (依類型動態對齊)
    if text_encoder_type == "gru":
        text_encoder = TextEncoder(vocab_size_ckpt, 300, 256, 1)
    elif text_encoder_type == "lstm":
        text_encoder = LSTMTextEncoder(vocab_size_ckpt, 300, 256, 1)
    elif text_encoder_type == "bert":
        text_encoder = PretrainedBertTextEncoder()
    elif text_encoder_type == "clip":
        text_encoder = PretrainedCLIPTextEncoder()
    else:
        raise ValueError("unknown encoder type")

    # ④ 組合並載入權重
    model = MHSAN_with_Text(mhsan_model, text_encoder).to(device)
    model.load_state_dict(state_tmp)
    model.eval()

    # ⑤ 讀取 gallery 特徵
    gal = torch.load(f"gallery/{text_encoder_type}.pt", map_location=device)
    gallery_feats  = F.normalize(gal["gallery_feats"].to(device), dim=1)
    gallery_labels = gal["gallery_labels"].to(device)
    label2category = gal["label2category"]

    # ⑥ 建 tokenizer (僅 GRU/LSTM 用到)
    if text_encoder_type in ["gru","lstm"]:
        sub_tok = BertWordPieceTokenizer("tokenizer_model/vocab.txt",
                                         lowercase=True, unk_token="[UNK]", pad_token="[PAD]")

    def encode_text(sent):
        if text_encoder_type in ["gru","lstm"]:
            ids = sub_tok.encode(sent).ids[:500]
            ids = torch.tensor(ids).unsqueeze(0).to(device)
            with torch.no_grad(): feat = model.text_encoder(ids)
        elif text_encoder_type=="bert":
            tok=BertTokenizer.from_pretrained("bert-base-uncased")
            batch=tok([sent],max_length=50,padding="longest",truncation=True,return_tensors="pt").to(device)
            with torch.no_grad(): feat = model.text_encoder(batch)
        else:
            tok=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            batch=tok([sent],max_length=50,padding="longest",truncation=True,return_tensors="pt").to(device)
            with torch.no_grad(): feat = model.text_encoder(batch)
        return F.normalize(feat,dim=1)

    # ⑦ 示範檢索
    query = "The object is a staircase with a unique and intricate design. It consists of multiple steps, each with a distinct shape and structure. The staircase is made up of a combination of straight and curved lines, creating a visually appealing and functional piece of architecture. The arrangement of the steps allows for easy access and movement, while the overall design adds an artistic and aesthetically pleasing element to the space."
    qvec  = encode_text(query)
    sims  = torch.matmul(qvec, gallery_feats.T).squeeze(0)
    topk  = torch.topk(sims, 5)
    print("── Top‑5 ──")
    for idx,s in zip(topk.indices.tolist(),topk.values.tolist()):
        print(f"{idx:4d} | {label2category[gallery_labels[idx]]:>8s} | {s:.4f}")

    # ⑧ (選) 單模態 & 跨模態 mAP
    mAP_v = compute_map(gallery_feats, gallery_labels)
    print(f"[visual self mAP] {mAP_v:.4f}")

    txt_feats = []
    for lbl in gallery_labels.cpu().numpy():
        txt_feats.append( encode_text(label2category[int(lbl)]) )
    txt_feats = torch.cat(txt_feats,0)
    mAP_c = compute_map_crossmodal(txt_feats, gallery_labels, gallery_feats, gallery_labels)
    print(f"[cross‑modal  mAP] {mAP_c:.4f}")
