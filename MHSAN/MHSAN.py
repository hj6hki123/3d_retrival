import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.models import VGG16_Weights


#  Position Embedding（sin/cos）
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_views):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views + 1  # 包含 [class] token

        position = torch.arange(self.num_views).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe = torch.zeros(self.num_views, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, num_views+1, embed_dim)

    def forward(self, x):
        return x + self.pe  # 直接加到視角特徵上


#  Multi-Head Self-Attention Layer
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # **Self-Attention with Residual**
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.multi_head_attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # 殘差連接

        # **Feed Forward with Residual**
        x_norm = self.layer_norm2(x)
        x = x + self.feed_forward(x_norm)
        return x


#  MHSAN Model
class MHSAN(nn.Module):
    def __init__(self, num_views=12, embed_dim=512, num_heads=8, num_layers=12, top_k=6):
        super(MHSAN, self).__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        self.top_k = top_k

        # **VGG16 特徵提取（視角特徵）**
        vgg16_model = models.vgg16(weights=VGG16_Weights.DEFAULT)  # 避免名稱衝突
        self.feature_extractor = nn.Sequential(*list(vgg16_model.features.children())) 
        self.fc = nn.Linear(512 * 7 * 7, embed_dim)

        # **Position Embedding**
        self.position_embedding = PositionalEncoding(embed_dim, num_views)
        
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # **第一層 Attention Network**
        self.self_attn_layers1 = nn.ModuleList([SelfAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)])

        # **第二層 Attention Network**
        self.self_attn_layers2 = nn.ModuleList([SelfAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)])

        # **全連接分類層**
        self.fc_cls = nn.Linear(embed_dim * 2, 40)  # ModelNet40 40 類別

    def forward(self, x):
        batch_size, num_views, C, H, W = x.shape
        x = x.view(batch_size * num_views, C, H, W)

        # **視角特徵提取**
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)  # 攤平成向量
        x = self.fc(x)  # 映射到 512 維特徵
        x = x.view(batch_size, num_views, -1)  # (batch, num_views, embed_dim)

        # **加入 [class] token**
        class_token = self.class_token.repeat(batch_size, 1, 1)
        #class_token = torch.zeros(batch_size, 1, self.embed_dim).to(x.device)
        
        x = torch.cat([class_token, x], dim=1)  # (batch, num_views+1, embed_dim)

        # **加入 Position Embedding**
        x = self.position_embedding(x)

        # **第一層 Self-Attention**
        for layer in self.self_attn_layers1:
            x = layer(x)

        # **計算視角分類分數，選擇最重要的視角**
        classification_scores = x[:, 1:, :].mean(dim=-1)  # (batch, num_views)
        top_k_indices = classification_scores.topk(self.top_k, dim=1).indices
        selected_views = torch.cat([x[:, :1, :], torch.gather(x[:, 1:, :], 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))], dim=1)

        # **第二層 Self-Attention**
        for layer in self.self_attn_layers2:
            selected_views = layer(selected_views)

        # **全域與辨識特徵**
        global_feature = torch.max(x, dim=1).values
        discriminative_feature = torch.max(selected_views, dim=1).values
        final_feature = torch.cat([global_feature, discriminative_feature], dim=1)  # (batch, 1024)

        # **分類**
        output = self.fc_cls(final_feature)  # (batch, 40)
        return output, final_feature
