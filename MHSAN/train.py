from modelnet40_datasets import *
from MHSAN import *
import torch
from tqdm import tqdm
import wandb

import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, criterion, optimizer, num_epochs=10, device="cuda", save_path="MHSAN_best.pth",test_loader=None):
    model.to(device)
    best_acc = 0.0  # 記錄最高準確率
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs,final_feature = model(images)  # (batch_size, num_classes)
            # print("outputs.shape:", outputs.shape)  # 應該是 (batch_size, num_classes)
            # print("outputs:", outputs[:5].detach().cpu().numpy() )  # 看前5個
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

            # **更新 tqdm 進度條**
            loop.set_postfix(loss=loss.item(), acc=correct / len(train_loader.dataset))

        train_acc = correct / len(train_loader.dataset)
        
        # **記錄到 wandb**
        wandb.log({"Train Loss": total_loss / len(train_loader), "Train Accuracy": train_acc})

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")
        
        # 如果提供 test_loader,就做一次測試
        if test_loader is not None:
            test_acc = test(model, test_loader, device=device)
            # log to wandb
            wandb.log({"Test Accuracy (per epoch)": test_acc, "epoch": epoch + 1})


        # ** 儲存最好的模型**
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), save_path)  # 存權重
            print(f"已儲存新最佳模型 (Acc={best_acc:.4f}) ➜ {save_path}")




def test(model, test_loader, device="cuda"):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    loop = tqdm(test_loader, desc="Testing", leave=True)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs,final_feature = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(acc=correct / total)

    test_acc = correct / total
    
    # **記錄到 wandb**
    wandb.log({"Test Accuracy": test_acc})
    print(f'tatoles: {total}, correct: {correct}')
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc


def load_model(model, checkpoint_path="MHSAN20_best.pth", device="cuda"):
    """ 載入已存的最佳模型 """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f" 成功載入模型權重: {checkpoint_path}")

if __name__ == "__main__":
    
    ## **設定參數**
    num_views = 12
    num_layers = 3
    save_path = "MHSAN12_best.pth"
    
    top_k = 6 if num_views == 12 else 10 if num_views == 20 else ValueError("num_views 只能為 12 或 20")
        
    wandb.init(project="MHSAN-ModelNet40", name=f"MHSAN_{num_views}views")
    
    # **設定 DataLoader**
    train_loader = get12_views_dataloader(split="train")
    test_loader = get12_views_dataloader(split="test")

    # 先驗證：是否 train_loader 和 test_loader 真的有相同的數據？
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))



    print("Train 影像形狀:", train_images.shape)  # (batch_size, num_views, 3, 224, 224)
    print("Test 影像形狀:", test_images.shape)    # 應該要一樣


    # **初始化 MHSAN 模型**
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MHSAN(num_views=num_views, embed_dim=512, num_heads=8, num_layers=num_layers, top_k=top_k).to(device)

    # **定義損失函數與優化器**
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    
    # **訓練 MHSAN**
    train(model, train_loader, criterion, optimizer, num_epochs=10, device=device, save_path=save_path, test_loader=test_loader)

    # **測試模型**
    test(model, test_loader, device=device)
    
    wandb.finish()
