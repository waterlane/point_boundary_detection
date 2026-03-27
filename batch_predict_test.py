import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import (
    build_model_inputs,
    compute_local_features,
    expand_boundary_predictions,
    filter_isolated_predictions,
    load_points_from_file,
    load_model_checkpoint,
    retain_largest_prediction_components,
    save_points_to_ply,
)
from utils.metrics import compute_boundary_quality_metrics


def batch_predict(
    input_dir,
    output_dir,
    model_path="model.pth",
    threshold=0.005,
    denoise_neighbor_k=64,
    denoise_min_neighbors=5,
    expand_neighbor_k=0,
    expand_min_seed_neighbors=0,
    component_neighbor_k=0,
    min_component_size=0,
    keep_top_components=0,
    far_distance_threshold=0.01,
    device="cpu",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".ply"}
    )
    if not input_files:
        raise RuntimeError(f"未在 {input_dir} 中找到 csv 或 ply 文件")

    model, model_type, input_dim, k_neighbors, small_k_neighbors, large_k_neighbors = load_model_checkpoint(
        model_path,
        device,
    )

    summary_rows = []
    for input_path in input_files:
        stem = input_path.stem
        points = load_points_from_file(str(input_path))
        gt_mask = None
        if input_path.suffix.lower() == ".csv":
            raw = pd.read_csv(input_path, header=None, skiprows=1).values.astype(np.float32)
            if raw.shape[1] >= 4:
                gt_mask = raw[:, 3] < threshold

        if model_type == "pointnet2_msg":
            import torch

            with torch.no_grad():
                centers, patch_small, patch_large = build_model_inputs(
                    points,
                    small_k=small_k_neighbors,
                    large_k=large_k_neighbors,
                )
                center_tensor = torch.from_numpy(centers).to(device)
                patch_small_tensor = torch.from_numpy(patch_small).to(device)
                patch_large_tensor = torch.from_numpy(patch_large).to(device)
                pred = model(center_tensor, patch_small_tensor, patch_large_tensor)
                pred = pred.cpu().numpy().flatten()
        else:
            import torch

            model_input = points
            if input_dim > 3:
                local_features = compute_local_features(points, k_neighbors=k_neighbors)
                model_input = np.concatenate([points, local_features], axis=1).astype(np.float32)
            with torch.no_grad():
                pts = torch.from_numpy(model_input).to(device)
                pred = model(pts).cpu().numpy().flatten()

        mask = pred < threshold
        raw_count = int(mask.sum())
        if expand_neighbor_k > 0 and expand_min_seed_neighbors > 0:
            mask = expand_boundary_predictions(
                points,
                mask,
                neighbor_k=expand_neighbor_k,
                min_seed_neighbors=expand_min_seed_neighbors,
            )
        if denoise_neighbor_k > 0 and denoise_min_neighbors > 0:
            mask = filter_isolated_predictions(
                points,
                mask,
                neighbor_k=denoise_neighbor_k,
                min_neighbor_predictions=denoise_min_neighbors,
            )
        if component_neighbor_k > 0 and (min_component_size > 0 or keep_top_components > 0):
            mask = retain_largest_prediction_components(
                points,
                mask,
                neighbor_k=component_neighbor_k,
                min_component_size=min_component_size,
                keep_top_components=keep_top_components,
            )
        filtered_count = int(mask.sum())

        tp = fp = fn = precision = recall = f1 = None
        far_fp_count = far_fp_mean_distance = continuity_score = None
        if gt_mask is not None:
            filtered_pred_distance = np.where(mask, 0.0, threshold * 2.0)
            metrics = compute_boundary_quality_metrics(
                filtered_pred_distance,
                raw[:, 3],
                boundary_threshold=threshold,
                far_distance_threshold=far_distance_threshold,
            )
            tp = metrics["tp"]
            fp = metrics["fp"]
            fn = metrics["fn"]
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1"]
            far_fp_count = metrics["far_fp_count"]
            far_fp_mean_distance = metrics["far_fp_mean_distance"]
            continuity_score = metrics["continuity_score"]

        pred_csv_path = output_dir / f"{stem}_pred.csv"
        pred_ply_path = output_dir / f"{stem}_predicted_below_{threshold}.ply"
        full_ply_path = output_dir / f"{stem}_full_cloud.ply"

        out_arr = np.concatenate([points, pred[:, None]], axis=1)
        pd.DataFrame(out_arr, columns=["x", "y", "z", "predicted_distance"]).to_csv(pred_csv_path, index=False)
        save_points_to_ply(points, full_ply_path)
        if filtered_count > 0:
            save_points_to_ply(points[mask], pred_ply_path)

        summary_rows.append(
            {
                "csv": str(input_path),
                "pred_csv": str(pred_csv_path),
                "full_ply": str(full_ply_path),
                "filtered_ply": str(pred_ply_path) if filtered_count > 0 else "",
                "input_file": str(input_path),
                "raw_selected_points": raw_count,
                "filtered_selected_points": filtered_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "far_fp_count": far_fp_count,
                "far_fp_mean_distance": far_fp_mean_distance,
                "continuity_score": continuity_score,
            }
        )
        metric_text = ""
        if f1 is not None:
            metric_text = (
                f", P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
                f"farFP={far_fp_count}, continuity={continuity_score:.4f}"
            )
        print(f"{stem}: raw={raw_count}, filtered={filtered_count}{metric_text}, pred_csv={pred_csv_path.name}")

    summary_path = output_dir / "batch_predict_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"已完成，共处理 {len(input_files)} 个文件，汇总保存到 {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量对 csv 文件夹进行预测")
    parser.add_argument("--input-dir", type=str, default="data/test_csv", help="输入 csv 文件夹")
    parser.add_argument("--output-dir", type=str, default="data/test_predict_results_0p005_k64_n5", help="输出结果文件夹")
    parser.add_argument("--model-path", type=str, default="model.pth", help="模型路径")
    parser.add_argument("--threshold", type=float, default=0.005, help="边界筛选阈值")
    parser.add_argument("--denoise-neighbor-k", type=int, default=64, help="离散点过滤使用的近邻数")
    parser.add_argument("--denoise-min-neighbors", type=int, default=5, help="离散点过滤最少近邻预测数")
    parser.add_argument("--expand-neighbor-k", type=int, default=0, help="补短缺口时使用的近邻数")
    parser.add_argument("--expand-min-seed-neighbors", type=int, default=0, help="补短缺口所需最少边界邻居数")
    parser.add_argument("--component-neighbor-k", type=int, default=0, help="连通分量过滤近邻数")
    parser.add_argument("--min-component-size", type=int, default=0, help="保留连通分量的最小大小")
    parser.add_argument("--keep-top-components", type=int, default=0, help="只保留最大的若干连通分量，0表示不限制")
    parser.add_argument("--far-distance-threshold", type=float, default=0.01, help="统计远距离假点时使用的距离阈值")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备")
    args = parser.parse_args()

    batch_predict(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        denoise_neighbor_k=args.denoise_neighbor_k,
        denoise_min_neighbors=args.denoise_min_neighbors,
        expand_neighbor_k=args.expand_neighbor_k,
        expand_min_seed_neighbors=args.expand_min_seed_neighbors,
        component_neighbor_k=args.component_neighbor_k,
        min_component_size=args.min_component_size,
        keep_top_components=args.keep_top_components,
        far_distance_threshold=args.far_distance_threshold,
        device=args.device,
    )
