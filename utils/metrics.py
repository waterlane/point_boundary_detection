def compute_metrics(pred, target, thresh=0.5):
    pred_bin = (pred > thresh)
    target_bin = (target > 0.5)

    tp = (pred_bin & target_bin).sum()
    fp = (pred_bin & ~target_bin).sum()
    fn = (~pred_bin & target_bin).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall