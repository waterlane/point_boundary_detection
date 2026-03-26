import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAbstractionMSG(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        layers = []
        last_c = in_channels
        for out_c in mlp_channels:
            layers.append(nn.Conv1d(last_c, out_c, kernel_size=1))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, patch):
        # patch: (B, K, C)
        feat = patch.transpose(1, 2).contiguous()
        feat = self.mlp(feat)
        feat = torch.max(feat, dim=2).values
        return feat


class PointNet2BoundaryNet(nn.Module):
    """
    轻量 PointNet++-style MSG 模型：
    对中心点的小邻域和大邻域分别做局部特征聚合，再融合回归中心点到边界的距离。
    """

    def __init__(self, emb_dim=64):
        super().__init__()
        self.small_branch = SetAbstractionMSG(in_channels=3, mlp_channels=[32, 64, emb_dim])
        self.large_branch = SetAbstractionMSG(in_channels=3, mlp_channels=[32, 64, emb_dim])

        self.center_embed = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        fusion_dim = emb_dim * 2 + 32
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, center_points, patch_small, patch_large):
        if center_points.dim() == 1:
            center_points = center_points.unsqueeze(0)
        if patch_small.dim() == 2:
            patch_small = patch_small.unsqueeze(0)
        if patch_large.dim() == 2:
            patch_large = patch_large.unsqueeze(0)

        small_feat = self.small_branch(patch_small)
        large_feat = self.large_branch(patch_large)
        center_feat = self.center_embed(center_points)

        fused = torch.cat([center_feat, small_feat, large_feat], dim=1)
        out = F.softplus(self.head(fused))
        return out.squeeze(-1)
