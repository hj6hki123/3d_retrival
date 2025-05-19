from modelnet40_datasets import *
from MHSAN import *
import torch
from tqdm import tqdm
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

# -----------------------------------------
# t-SNE 降維 + 繪圖
# -----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

def plot_tsne(features, labels, class_names, num_classes=40):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # **使用 seaborn 調色盤**
    palette = np.array(sns.color_palette("husl", num_classes))

    plt.figure(figsize=(10, 8))
    
    for i in range(num_classes):
        idx = labels == i  # 找出屬於類別 i 的樣本
        if np.sum(idx) == 0:  # 如果該類別沒有樣本,跳過
            continue
        
        # **畫出該類別的點**
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color=palette[i], label=class_names[i], alpha=0.7, edgecolors="black")

        # **計算該類別的中心點**
        x_mean, y_mean = np.mean(features_2d[idx, 0]), np.mean(features_2d[idx, 1])
        
        # **在中心點標上類別名稱**
        plt.text(x_mean, y_mean, class_names[i], fontsize=10, weight="bold", ha="center", va="center",
                 bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))  # 圖例放在右上角
    plt.title("t-SNE Visualization with Class Labels")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # **存檔**
    plt.savefig("tsne_result_labeled.png")
    wandb.log({"TSNE Plot": wandb.Image("tsne_result_labeled.png")})

    #plt.show()
def plot_class_accuracy(class_accuracy, class_names):
    """
    繪製每個類別的準確率,並標示類別名稱
    :param class_accuracy: 每個類別的準確率 (list 或 numpy array)
    :param class_names: ModelNet40 的類別名稱 (list)
    """
    num_classes = len(class_accuracy)
    
    plt.figure(figsize=(15, 6))
    plt.bar(class_names, class_accuracy, alpha=0.7)
    plt.xlabel("Class Name")  # 設定 X 軸標籤
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=90)  # 讓類別名稱旋轉 90 度,避免重疊
    plt.ylim(0, 1)  # 限制在 0% 到 100%

    # **存檔**
    plt.savefig("class_accuracy.png")
    wandb.log({"Class Accuracy Plot": wandb.Image("class_accuracy.png")})

    #plt.show()

def compute_class_distances(features, labels, class_names, metric="euclidean", normalize_method="zscore", top_k=5):
    """
    計算不同類別之間的平均距離,找出最容易混淆的前 K 個類別對。
    
    :param features: (N, D) numpy array,N 為樣本數,D 為特徵維度
    :param labels: (N,) numpy array,每個樣本的類別標籤
    :param class_names: 類別名稱列表
    :param metric: "euclidean"（歐幾里得距離）或 "cosine"（餘弦相似度）
    :param normalize_method: "zscore"（標準化）或 "l2"（L2 正規化）
    :param top_k: 選出最容易混淆的前 K 個類別對
    :return: 類別間距離矩陣 (C, C) 和 距離最短的前 K 個類別對
    """
    # **特徵正規化**
    if normalize_method == "zscore":
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif normalize_method == "l2":
        features = normalize(features, norm="l2")
    else:
        raise ValueError("normalize_method 必須是 'zscore' 或 'l2'")

    num_classes = len(class_names)
    
    # **計算每個類別的特徵中心點**
    class_centroids = np.zeros((num_classes, features.shape[1]))  # (C, D)
    for c in range(num_classes):
        class_centroids[c] = np.mean(features[labels == c], axis=0)

    # **計算類別間的距離**
    if metric == "euclidean":
        dist_matrix = euclidean_distances(class_centroids)  # (C, C) 距離矩陣
    elif metric == "cosine":
        dist_matrix = 1 - cosine_similarity(class_centroids)  # 餘弦距離 (1 - 相似度)
    else:
        raise ValueError("metric 必須是 'euclidean' 或 'cosine'")

    # **找出前 K 個距離最短的類別對**
    np.fill_diagonal(dist_matrix, np.inf)  # 讓自己對自己的距離設為無限大,避免干擾
    flat_indices = np.argsort(dist_matrix, axis=None)[:top_k]  # 找出最小的前 K 個索引
    row_indices, col_indices = np.unravel_index(flat_indices, dist_matrix.shape)  # 轉換為矩陣索引

    top_k_pairs = [(class_names[row_indices[i]], class_names[col_indices[i]], dist_matrix[row_indices[i], col_indices[i]])
                   for i in range(top_k)]

    print("最容易混淆的前 5 個類別對:")
    for i, (class1, class2, dist) in enumerate(top_k_pairs):
        print(f"  {i+1}. ({class1}, {class2}),距離 = {dist:.4f}")

    return dist_matrix, top_k_pairs


    

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

def compute_class_distances(features, labels, class_names, metric="euclidean", normalize_method="zscore", top_k=5):
    # **特徵正規化**
    if normalize_method == "zscore":
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif normalize_method == "L2":
        features = normalize(features, norm="l2")
    else:
        raise ValueError("normalize_method 必須是 'zscore' 或 'l2'")

    num_classes = len(class_names)
    
    # **計算每個類別的特徵中心點**
    class_centroids = np.zeros((num_classes, features.shape[1]))  # (C, D)
    for c in range(num_classes):
        class_centroids[c] = np.mean(features[labels == c], axis=0)

    # **計算類別間的距離**
    if metric == "euclidean":
        dist_matrix = euclidean_distances(class_centroids)  # (C, C) 距離矩陣
    elif metric == "cosine":
        dist_matrix = 1 - cosine_similarity(class_centroids)  # 餘弦距離 (1 - 相似度)
    else:
        raise ValueError("metric 必須是 'euclidean' 或 'cosine'")

    # **只選擇上三角矩陣,避免重複**
    triu_indices = np.triu_indices(num_classes, k=1)  # 取得上三角矩陣索引（不包含對角線）
    dist_values = dist_matrix[triu_indices]  # 取得對應的距離值

    # **找到前 K 個最短距離**
    top_k_indices = np.argsort(dist_values)[:top_k]  # 找出最小的前 K 個索引
    row_indices, col_indices = triu_indices[0][top_k_indices], triu_indices[1][top_k_indices]  # 轉換回類別索引

    top_k_pairs = [(class_names[row_indices[i]], class_names[col_indices[i]], dist_matrix[row_indices[i], col_indices[i]])
                   for i in range(top_k)]

    print("最容易混淆的前 5 個類別對（不重複）:")
    for i, (class1, class2, dist) in enumerate(top_k_pairs):
        print(f"  {i+1}. ({class1}, {class2}),距離 = {dist:.4f}")

    return dist_matrix, top_k_pairs






def test(model, test_loader, device="cuda", num_classes=40):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    # 建立空的 list 來收集所有特徵 & 標籤
    all_features = []
    all_labels = []

    # 記錄每個類別的計數
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    loop = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs, final_feature = model(images)
            

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(acc=correct / total)

            # 計算每個類別的準確度
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predictions[i] == label:
                    class_correct[label] += 1

            # 收集特徵和標籤
            all_features.append(final_feature.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    test_acc = correct / total

    # 計算每個類別的準確率
    class_accuracy = class_correct / (class_total + 1e-6)  # 避免除以 0
    class_accuracy_dict = {f"Class_{i}": class_accuracy[i] for i in range(num_classes)}

    # **記錄到 WandB**
    wandb.log({"Test Accuracy": test_acc})
    wandb.log({"Class Accuracy": class_accuracy_dict})

    print(f"Test Accuracy: {test_acc:.4f}")

    # **轉換特徵與標籤**
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return test_acc, all_features, all_labels, class_accuracy


def load_model(model, checkpoint_path="MHSAN_best.pth", device="cuda"):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f" 成功載入模型權重: {checkpoint_path}")

if __name__ == "__main__":
    ## **設定參數**
    batch_size = 8
    num_views = 12
    num_layers = 3
    checkpoint_path = "MHSAN12_best.pth"
    
    
    top_k = 6 if num_views == 12 else 10 if num_views == 20 else ValueError("num_views 只能為 12 或 20")
    wandb.init(project="MHSAN-ModelNet40", name=f"MHSAN_{num_views}views_test")

    test_loader = get12_views_dataloader(split="test" , batch_size= batch_size)
    
    # **取得類別名稱**
    dataset = test_loader.dataset  # PyTorch DataLoader 的 dataset 屬性
    class_names = dataset.categories  # 取得 ModelNet40 類別名稱列表
    
    print("測試資料類別名稱:", class_names)
    print("測試集樣本數:", len(test_loader.dataset))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MHSAN(num_views=num_views, embed_dim=512, num_heads=8, num_layers=num_layers, top_k=top_k).to(device)

    load_model(model, checkpoint_path=checkpoint_path, device=device)

    # **測試模型 + 收集特徵 + 類別準確率**
    test_acc, features, labels, class_accuracy = test(model, test_loader, device=device)

    # **tsne 繪圖**
    plot_tsne(features, labels, class_names, num_classes=40)

    # **繪製類別準確度**
    plot_class_accuracy(class_accuracy, class_names)
    
    # **計算類別間距離**
    # 計算類別間距離,找出前 5 個最容易混淆的類別
    dist_matrix, top_5_pairs = compute_class_distances(features, labels, class_names, metric="cosine",normalize_method="L2", top_k=5)


    # **視覺化類別間距離**
    plot_class_distance_matrix(dist_matrix, class_names)
    
    wandb.finish()
