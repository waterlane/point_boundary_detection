
import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNN(nn.Module):
    def __init__(self, input_dim=3, emb_dim=64):
        super(DGCNN, self).__init__()
        # 支持单点输入 (input_dim,) 或批量 (B, N, input_dim)
        self.input_dim = input_dim
        self.mlp1 = nn.Linear(input_dim, emb_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, N, input_dim) or (N, input_dim) or (input_dim,)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, input_dim)
        if x.dim() == 2:
            feat = self.relu(self.mlp1(x))  # (N, emb_dim)
            out = F.softplus(self.classifier(feat))  # (N, 1), 约束为非负距离
            return out.squeeze(-1)  # (N,)
        elif x.dim() == 3:
            feat = self.relu(self.mlp1(x))  # (B, N, emb_dim)
            out = F.softplus(self.classifier(feat))  # (B, N, 1), 约束为非负距离
            return out.squeeze(-1)  # (B, N)
        else:
            raise ValueError('输入维度不支持')
