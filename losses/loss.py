import torch
import torch.nn.functional as F


def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)


def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)

    # dice
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice = 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return bce + dice


def boundary_aware_distance_loss(
    pred,
    target,
    boundary_threshold=0.005,
    boundary_weight=6.0,
    reg_loss_weight=1.0,
    cls_loss_weight=1.0,
    smooth_l1_beta=0.002,
    focal_gamma=2.0,
    focal_alpha=0.75,
    max_pos_weight=5.0,
    far_distance_threshold=0.01,
    far_penalty_weight=2.0,
    far_penalty_power=1.5,
):
    """
    针对“距离边界越近越重要”的回归损失：
    1) 对真实距离小的点分配更高权重（加权 L1）。
    2) 增加一个辅助二分类目标：是否在 boundary_threshold 内。
    """
    eps = 1e-6

    # 回归项：小距离点有更大权重
    near_factor = torch.exp(-target / (boundary_threshold + eps))
    point_weight = 1.0 + boundary_weight * near_factor
    reg_error = F.smooth_l1_loss(pred, target, beta=smooth_l1_beta, reduction='none')
    reg_loss = (point_weight * reg_error).mean()

    # 分类项：提升“是否靠近边界”的可分性
    near_target = (target <= boundary_threshold).float()
    near_logit = (boundary_threshold - pred) / (boundary_threshold + eps)

    pos_weight = (near_target.numel() - near_target.sum()) / (near_target.sum() + eps)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=max_pos_weight)

    bce = F.binary_cross_entropy_with_logits(
        near_logit,
        near_target,
        pos_weight=pos_weight,
        reduction='none',
    )
    prob = torch.sigmoid(near_logit)
    p_t = prob * near_target + (1.0 - prob) * (1.0 - near_target)
    alpha_t = focal_alpha * near_target + (1.0 - focal_alpha) * (1.0 - near_target)
    focal_weight = alpha_t * torch.pow(1.0 - p_t, focal_gamma)
    cls_loss = (focal_weight * bce).mean()

    # 额外惩罚远离边界的点被压到 near-boundary 阈值以内。
    far_mask = (target > far_distance_threshold).float()
    false_near_margin = torch.relu(boundary_threshold - pred)
    far_scale = torch.pow(
        torch.clamp(target / (far_distance_threshold + eps), min=1.0),
        far_penalty_power,
    )
    far_fp_loss = (far_mask * far_scale * false_near_margin).mean()

    return (
        reg_loss_weight * reg_loss
        + cls_loss_weight * cls_loss
        + far_penalty_weight * far_fp_loss
    )
