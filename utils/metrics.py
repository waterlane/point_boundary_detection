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
