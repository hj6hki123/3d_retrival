import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.models import VGG16_Weights

# 如果要用 BERT/CLIP,需要 transformers
from transformers import BertModel, CLIPModel

################################################################################
# MHSAN (多視角視覺部分)
################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_views):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views + 1  # 包含 [class] token
        position = torch.arange(self.num_views).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0)/embed_dim))
        pe = torch.zeros(self.num_views, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, num_views+1, embed_dim)

    def forward(self, x):
        return x + self.pe

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
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.multi_head_attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x_norm = self.layer_norm2(x)
        x = x + self.feed_forward(x_norm)
        return x

class MHSAN(nn.Module):
    def __init__(self, num_views=12, embed_dim=512, num_heads=8, num_layers=3, top_k=6):
        super(MHSAN, self).__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        self.top_k = top_k

        vgg16_model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg16_model.features.children()))
        self.fc = nn.Linear(512*7*7, embed_dim)
        self.position_embedding = PositionalEncoding(embed_dim, num_views)
        self.class_token = nn.Parameter(torch.randn(1,1,embed_dim))

        self.self_attn_layers1 = nn.ModuleList([SelfAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.self_attn_layers2 = nn.ModuleList([SelfAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.fc_cls = nn.Linear(embed_dim*2, 40)

    def forward(self, x):
        # x: (batch, num_views, C, H, W)
        batch_size, num_views, C, H, W = x.shape
        x = x.view(batch_size*num_views, C, H, W)
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)  # (batch_size*num_views, embed_dim)
        x = x.view(batch_size, num_views, -1)  # (batch, num_views, embed_dim)

        class_token = self.class_token.repeat(batch_size, 1, 1)
        x = torch.cat([class_token, x], dim=1)  # (batch, num_views+1, embed_dim)
        x = self.position_embedding(x)

        # Self-Attention: 第1階段
        for layer in self.self_attn_layers1:
            x = layer(x)

        # 選擇 top_k 視角
        classification_scores = x[:, 1:, :].mean(dim=-1)  # (batch, num_views)
        top_k_indices = classification_scores.topk(self.top_k, dim=1).indices
        selected_views = torch.cat([
            x[:, :1, :],
            torch.gather(x[:, 1:, :], 1,
                         top_k_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        ], dim=1)

        # Self-Attention: 第2階段
        for layer in self.self_attn_layers2:
            selected_views = layer(selected_views)

        global_feature = torch.max(x, dim=1).values
        discriminative_feature = torch.max(selected_views, dim=1).values
        final_feature = torch.cat([global_feature, discriminative_feature], dim=1)  # (batch, 1024)

        output = self.fc_cls(final_feature)  # (batch, 40)
        return output, final_feature

################################################################################
# 四種文字編碼器：GRU / BERT / CLIP / LSTM
################################################################################
class TextEncoder(nn.Module):
    """GRU 版文字編碼器 (自定義)"""
    def __init__(self, vocab_size=10000, embed_dim=300, hidden_dim=256, num_layers=1):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 512)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 取最後一個時間步
        out = self.fc(out)
        return out

class PretrainedBertTextEncoder(nn.Module):
    """BERT 版文字編碼器"""
    def __init__(self):
        super().__init__()
        # 載入預訓練的 bert-base-uncased
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, 512)

    def forward(self, encoded_batch):
        # encoded_batch: dict, e.g. { 'input_ids': (batch, seq_len), 'attention_mask': (batch, seq_len), ...}
        outputs = self.bert(
            input_ids=encoded_batch["input_ids"],
            attention_mask=encoded_batch["attention_mask"]
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]
        out = self.fc(cls_vec)
        return out

class PretrainedCLIPTextEncoder(nn.Module):
    """CLIP 版文字編碼器"""
    def __init__(self):
        super().__init__()
        # 載入預訓練的 CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.parameters():
            param.requires_grad = True
        # openai/clip-vit-base-patch32 的 text_features 大約是 512 維
        self.fc = nn.Identity()  # 如果需要 512 -> 512,可不加

    def forward(self, encoded_batch):
        # encoded_batch: dict, e.g. { 'input_ids':..., 'attention_mask':... }
        text_outputs = self.clip_model.get_text_features(
            input_ids=encoded_batch["input_ids"],
            attention_mask=encoded_batch["attention_mask"]
        )
        out = self.fc(text_outputs)  # 可能直接 Identity,就輸出 (batch, 512)
        return out

class LSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=300, hidden_dim=256, num_layers=1, bidirectional=False):
        super(LSTMTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=bidirectional)
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, 512)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        # 取最後一層最後時間步的 hidden state,如果是 bidirectional,就拼接兩個方向
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)  # (batch, hidden*2)
        else:
            last_hidden = hn[-1]  # (batch, hidden)
        out = self.fc(last_hidden)
        return out

    
    

################################################################################
# 多模態模型：MHSAN_with_Text
################################################################################
class MHSAN_with_Text(nn.Module):
    def __init__(self, mhsan_model, text_encoder):
        super(MHSAN_with_Text, self).__init__()
        self.mhsan_model = mhsan_model
        self.text_encoder = text_encoder
        self.vis_proj = nn.Linear(1024, 512)

    def forward(self, images, texts):
        cls_output, vis_feature = self.mhsan_model(images)  # (batch, 40), (batch, 1024)
        vis_feature_proj = self.vis_proj(vis_feature)       # (batch, 512)
        text_feature = self.text_encoder(texts)             # (batch, 512)
        return cls_output, vis_feature_proj, text_feature

################################################################################
# 對比學習損失
################################################################################
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, vis_features, text_features):
        # vis_features, text_features: (batch, d)
        vis_norm = F.normalize(vis_features, dim=1)
        text_norm = F.normalize(text_features, dim=1)
        logits = torch.matmul(vis_norm, text_norm.T) / self.temperature
        labels = torch.arange(vis_features.size(0)).to(vis_features.device)
        loss = F.cross_entropy(logits, labels)
        return loss
