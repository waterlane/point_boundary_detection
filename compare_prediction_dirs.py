import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData


def load_gt_points_and_mask(csv_path, threshold):
    data = pd.read_csv(csv_path, header=None, skiprows=1).values.astype(np.float32)
    points = data[:, :3]
    gt_mask = data[:, 3] < threshold
    return points, gt_mask


def load_ply_points(ply_path):
    if not ply_path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)


def points_to_key_set(points, decimals=6):
    rounded = np.round(points, decimals=decimals)
    return {tuple(row.tolist()) for row in rounded}


def compute_metrics_from_pred_points(all_points, gt_mask, pred_points, decimals=6):
    pred_set = points_to_key_set(pred_points, decimals=decimals)
    all_keys = np.round(all_points, decimals=decimals)
    pred_mask = np.array([tuple(row.tolist()) in pred_set for row in all_keys], dtype=bool)

    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": int(pred_mask.sum()),
        "gt_count": int(gt_mask.sum()),
    }


def compare_result_dir(result_dir, gt_csv_dir, threshold):
    result_dir = Path(result_dir)
    gt_csv_dir = Path(gt_csv_dir)
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "pred_count": 0, "gt_count": 0}

    gt_files = sorted(gt_csv_dir.glob("*.csv"))
    for gt_csv in gt_files:
        stem = gt_csv.stem
        pred_ply = result_dir / f"{stem}_predicted_below_{threshold}.ply"
        all_points, gt_mask = load_gt_points_and_mask(gt_csv, threshold)
        pred_points = load_ply_points(pred_ply)
        metrics = compute_metrics_from_pred_points(all_points, gt_mask, pred_points)
        rows.append({"name": stem, **metrics})
        for key in totals:
            totals[key] += metrics[key]

    precision = totals["tp"] / (totals["tp"] + totals["fp"] + 1e-6)
    recall = totals["tp"] / (totals["tp"] + totals["fn"] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    overall = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **totals,
    }
    return rows, overall


def main():
    parser = argparse.ArgumentParser(description="根据真实 test csv 标签，对比一个或两个预测结果目录")
    parser.add_argument("gt_csv_dir", help="真实 test csv 目录，第四列为距离标签")
    parser.add_argument("result_dir_a", help="结果目录 A")
    parser.add_argument("--result-dir-b", type=str, default="", help="可选，结果目录 B")
    parser.add_argument("--threshold", type=float, default=0.005, help="真实边界阈值")
    parser.add_argument("--report-prefix", type=str, default="compare_results", help="输出报告前缀")
    args = parser.parse_args()

    rows_a, overall_a = compare_result_dir(args.result_dir_a, args.gt_csv_dir, args.threshold)
    pd.DataFrame(rows_a).to_csv(f"{args.report_prefix}_A.csv", index=False)
    print(f"A overall: P={overall_a['precision']:.4f}, R={overall_a['recall']:.4f}, F1={overall_a['f1']:.4f}, TP={overall_a['tp']}, FP={overall_a['fp']}, FN={overall_a['fn']}")

    if args.result_dir_b:
        rows_b, overall_b = compare_result_dir(args.result_dir_b, args.gt_csv_dir, args.threshold)
        pd.DataFrame(rows_b).to_csv(f"{args.report_prefix}_B.csv", index=False)
        print(f"B overall: P={overall_b['precision']:.4f}, R={overall_b['recall']:.4f}, F1={overall_b['f1']:.4f}, TP={overall_b['tp']}, FP={overall_b['fp']}, FN={overall_b['fn']}")

        df_a = pd.DataFrame(rows_a).set_index("name")
        df_b = pd.DataFrame(rows_b).set_index("name")
        diff = df_a[["precision", "recall", "f1", "tp", "fp", "fn"]].join(
            df_b[["precision", "recall", "f1", "tp", "fp", "fn"]],
            lsuffix="_a",
            rsuffix="_b",
        )
        diff["f1_delta"] = diff["f1_b"] - diff["f1_a"]
        diff["precision_delta"] = diff["precision_b"] - diff["precision_a"]
        diff["recall_delta"] = diff["recall_b"] - diff["recall_a"]
        diff.to_csv(f"{args.report_prefix}_diff.csv")
        print(f"已保存对比差异到 {args.report_prefix}_diff.csv")


if __name__ == "__main__":
    main()
