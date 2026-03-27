import numpy as np


def compute_metrics(pred, target, thresh=0.5):
    pred_bin = pred > thresh
    target_bin = target > 0.5

    tp = (pred_bin & target_bin).sum()
    fp = (pred_bin & ~target_bin).sum()
    fn = (~pred_bin & target_bin).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall


def compute_boundary_metrics(pred_distance, target_distance, boundary_threshold=0.005):
    """按距离阈值评估“近边界点”检出效果。"""
    pred_near = pred_distance <= boundary_threshold
    target_near = target_distance <= boundary_threshold

    tp = np.logical_and(pred_near, target_near).sum()
    fp = np.logical_and(pred_near, ~target_near).sum()
    fn = np.logical_and(~pred_near, target_near).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_boundary_quality_metrics(
    pred_distance,
    target_distance,
    boundary_threshold=0.005,
    far_distance_threshold=0.01,
):
    pred_distance = np.asarray(pred_distance)
    target_distance = np.asarray(target_distance)

    near_metrics = compute_boundary_metrics(
        pred_distance,
        target_distance,
        boundary_threshold=boundary_threshold,
    )
    pred_near = pred_distance <= boundary_threshold
    target_near = target_distance <= boundary_threshold

    fp_mask = np.logical_and(pred_near, ~target_near)
    far_fp_mask = np.logical_and(pred_near, target_distance > far_distance_threshold)
    miss_mask = np.logical_and(~pred_near, target_near)

    fp_distances = target_distance[fp_mask]
    far_fp_distances = target_distance[far_fp_mask]
    miss_distances = target_distance[miss_mask]

    return {
        **near_metrics,
        "fp_mean_distance": float(fp_distances.mean()) if fp_distances.size else 0.0,
        "fp_max_distance": float(fp_distances.max()) if fp_distances.size else 0.0,
        "far_fp_count": int(far_fp_mask.sum()),
        "far_fp_mean_distance": float(far_fp_distances.mean()) if far_fp_distances.size else 0.0,
        "miss_near_count": int(miss_mask.sum()),
        "miss_near_mean_distance": float(miss_distances.mean()) if miss_distances.size else 0.0,
        "continuity_score": float(1.0 - miss_mask.sum() / (target_near.sum() + 1e-6)),
    }
