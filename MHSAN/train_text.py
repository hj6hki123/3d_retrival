# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from tokenizers import BertWordPieceTokenizer

from modelnet40_text_datasets import get_dataloader
from MHSAN_text import (
    MHSAN, TextEncoder,
    PretrainedBertTextEncoder,
    PretrainedCLIPTextEncoder,
    MHSAN_with_Text,
    ContrastiveLoss,
    LSTMTextEncoder
)

def train(model, train_loader, criterion, contrastive_loss_fn, optimizer,
          num_epochs=10, device="cuda", save_path="models/MHSAN_with_Text_best.pth",
          test_loader=None, lambda_contrast=0.5):
    model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0.0, 0
        total_loss_cls, total_loss_contrast = 0.0, 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, texts, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # 如果 texts 是 dict (BERT/CLIP)，要把 input_ids, attention_mask 都放到 device
            if isinstance(texts, dict):
                for k in texts:
                    texts[k] = texts[k].to(device)
            else:
                # texts: (B, seq_len) 的 token ids (GRU/LSTM)
                texts = texts.to(device)

            optimizer.zero_grad()
            outputs, vis_feature, text_feature = model(images, texts)

            # (a) 分類損失
            loss_cls = criterion(outputs, labels)
            # (b) 對比損失 (視覺-文字)
            loss_contrast = contrastive_loss_fn(vis_feature, text_feature)
            total_loss_cls += loss_cls.item()
            total_loss_contrast += loss_contrast.item()

            # 總損失 = 分類 + lambda_contrast * 對比
            loss = loss_cls + lambda_contrast * loss_contrast
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

            loop.set_postfix(
                loss=loss.item(),
                loss_cls=loss_cls.item(),
                loss_contrast=loss_contrast.item(),
                acc=correct / len(train_loader.dataset)
            )

        # 訓練完一個 epoch
        train_acc = correct / len(train_loader.dataset)

        wandb.log({
            "Train Loss": total_loss / len(train_loader),
            "loss_contrast": total_loss_contrast / len(train_loader),
            "loss_cls": total_loss_cls / len(train_loader),
            "Train Accuracy": train_acc,
            "epoch": epoch+1
        })

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

        # 如果有測試集，做簡單測試
        if test_loader is not None:
            test_acc = test(model, test_loader, device=device)
            wandb.log({"Test Accuracy (per epoch)": test_acc, "epoch": epoch+1})

            # 如果有提升，就儲存模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), save_path)
                print(f"新最佳模型已儲存 (Acc={best_acc:.4f}) ➜ {save_path}")

    # 最終結束後，再跑一次測試並log
    if test_loader is not None:
        test_acc = test(model, test_loader, device=device)
        wandb.log({"Final Test Accuracy": test_acc})


def test(model, test_loader, device="cuda"):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    loop = tqdm(test_loader, desc="Testing", leave=True)

    with torch.no_grad():
        for images, texts, labels in loop:
            images, labels = images.to(device), labels.to(device)

            if isinstance(texts, dict):
                for k in texts:
                    texts[k] = texts[k].to(device)
            else:
                texts = texts.to(device)

            outputs, vis_feature, text_feature = model(images, texts)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(acc=correct/total)

    test_acc = correct / total
    wandb.log({"Test Accuracy": test_acc})
    print(f"總數: {total}, 正確: {correct}, Test Acc: {test_acc:.4f}")
    return test_acc


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============ 在這裡切換文字編碼器類型 ==============
    # "gru", "lstm", "bert", "clip"
    text_encoder_type = "gru"

    num_views = 12
    num_layers = 3
    top_k = 6
    num_epochs = 50
    lambda_contrast = 5  # 視實驗需求調整
    lr = 1e-4
    weight_decay = 1e-4

    # 儲存檔名
    save_path = f"models/MHSAN_with_{text_encoder_type}_{num_views}_best.pth"

    # 資料路徑
    root_dir = f"modelnet40-princeton-3d-object-dataset/rendered_views_{num_views}"

    # 初始化 wandb
    wandb.init(project="MHSAN-ModelNet40", name=f"MHSAN_with_{text_encoder_type}_{num_views}_views")

    # 1. 建立資料載入器 (Dataset + DataLoader)
    train_loader = get_dataloader(
        root_dir=root_dir,
        num_views=num_views,
        split="train",
        batch_size=8,
        text_encoder_type=text_encoder_type
    )
    test_loader = get_dataloader(
        root_dir=root_dir,
        num_views=num_views,
        split="test",
        batch_size=8,
        text_encoder_type=text_encoder_type
    )

    # 2. 建立 MHSAN 視覺模型
    mhsan_model = MHSAN(
        num_views=num_views,
        embed_dim=512,
        num_heads=8,
        num_layers=num_layers,
        top_k=top_k
    )

    # 3. 依 text_encoder_type 建立文字編碼器
    if text_encoder_type in ["gru", "lstm"]:
        # (A) 先載入子詞 tokenizer (vocab.txt)
        wp_tokenizer = BertWordPieceTokenizer(
            "tokenizer_model/vocab.txt",
            lowercase=True,
            unk_token="[UNK]",
            pad_token="[PAD]"
        )
        actual_vocab_size = wp_tokenizer.get_vocab_size()

        # (B) 建立 RNN Encoder
        # 例如 GRU
        text_encoder = TextEncoder(
            vocab_size=actual_vocab_size,
            embed_dim=300,
            hidden_dim=256,
            num_layers=1
        )
        # 如果你想用 LSTM, 就換成 LSTMTextEncoder(...) 類似

    elif text_encoder_type == "bert":
        text_encoder = PretrainedBertTextEncoder()  # 內部自帶 bert-base-uncased
    elif text_encoder_type == "clip":
        text_encoder = PretrainedCLIPTextEncoder()  # 內部自帶 openai/clip-vit-base-patch32
    else:
        raise ValueError(f"不支援的 text_encoder_type: {text_encoder_type}")

    # 4. 建立多模態融合模型
    model = MHSAN_with_Text(mhsan_model, text_encoder).to(device)

    # 5. 建立損失與優化器
    criterion = nn.CrossEntropyLoss()
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 6. 開始訓練 & 測試
    train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        contrastive_loss_fn=contrastive_loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path=save_path,
        test_loader=test_loader,
        lambda_contrast=lambda_contrast
    )

    # 最後再測一次
    final_acc = test(model, test_loader, device=device)
    print(f"Final Test Accuracy: {final_acc:.4f}")

    wandb.finish()
