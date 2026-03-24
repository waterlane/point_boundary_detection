
import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNN(nn.Module):
    def __init__(self, emb_dim=64):
        super(DGCNN, self).__init__()
        # 支持单点输入 (3,) 或批量 (B, N, 3)
        self.mlp1 = nn.Linear(3, emb_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, N, 3) or (N, 3) or (3,)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, 3)
        if x.dim() == 2:
            feat = self.relu(self.mlp1(x))  # (N, emb_dim)
            out = self.classifier(feat)  # (N, 1)
            return out.squeeze(-1)  # (N,)
        elif x.dim() == 3:
            feat = self.relu(self.mlp1(x))  # (B, N, emb_dim)
            out = self.classifier(feat)  # (B, N, 1)
            return out.squeeze(-1)  # (B, N)
        else:
            raise ValueError('输入维度不支持')