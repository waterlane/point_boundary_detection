import os
import random
import shutil
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.csv_dataset import CsvPointDataset
from losses.loss import boundary_aware_distance_loss
from models.dgcnn import DGCNN
from models.pointnet2 import PointNet2BoundaryNet
from train_args import get_args
from utils.metrics import compute_boundary_metrics


args = get_args()


def save_points_to_ply(points, output_path):
    from plyfile import PlyData, PlyElement

    vertex = np.array(
        [tuple(p) for p in points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')],
    )
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(output_path)


def load_training_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_model_type = checkpoint.get("model_type", "dgcnn")
        if checkpoint_model_type != "pointnet2_msg":
            raise ValueError(
                f"续训模型类型不匹配: checkpoint={checkpoint_model_type}, 当前训练模型=pointnet2_msg"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        return {
            "epoch": int(checkpoint.get("epoch", -1)),
            "best_metric_name": checkpoint.get("best_metric_name"),
            "best_metric_value": checkpoint.get("best_metric_value"),
            "best_metric_snapshot": checkpoint.get("best_metric_snapshot"),
        }

    model.load_state_dict(checkpoint)
    return {
        "epoch": -1,
        "best_metric_name": None,
        "best_metric_value": None,
        "best_metric_snapshot": None,
    }


def save_split_manifest(file_paths, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for path in file_paths:
            f.write(f"{path}\n")


def export_fixed_split(train_files, test_files, manifest_dir, test_export_dir):
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(test_export_dir, exist_ok=True)

    train_manifest = os.path.join(manifest_dir, "train_files.txt")
    test_manifest = os.path.join(manifest_dir, "test_files.txt")
    save_split_manifest(train_files, train_manifest)
    save_split_manifest(test_files, test_manifest)

    for existing_name in os.listdir(test_export_dir):
        existing_path = os.path.join(test_export_dir, existing_name)
        if os.path.isfile(existing_path):
            os.remove(existing_path)

    for csv_path in test_files:
        shutil.copy2(csv_path, os.path.join(test_export_dir, os.path.basename(csv_path)))

    print(f"已保存固定训练集清单到 {train_manifest}")
    print(f"已保存固定测试集清单到 {test_manifest}")
    print(f"已导出固定测试集 csv 到 {test_export_dir}，共{len(test_files)}个文件")


def parse_thresholds(spec):
    thresholds = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    if not thresholds:
        raise ValueError("scan_thresholds 不能为空")
    return thresholds


def find_best_threshold(preds, labels, thresholds):
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:
        metrics = compute_boundary_metrics(preds, labels, boundary_threshold=threshold)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_dir = os.path.join(os.path.dirname(__file__), 'data', 'csv')
csv_files = sorted([os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')])
if len(csv_files) == 0:
    raise RuntimeError('未找到 data/csv/ 下的 csv 文件')

# 自动划分训练集和测试集（8:2）
random.shuffle(csv_files)
split_idx = int(len(csv_files) * 0.8)
train_files = csv_files[:split_idx]
test_files = csv_files[split_idx:]
feature_cache_dir = args.feature_cache_dir
if not os.path.isabs(feature_cache_dir):
    feature_cache_dir = os.path.join(os.path.dirname(__file__), feature_cache_dir)
split_manifest_dir = args.split_manifest_dir
if not os.path.isabs(split_manifest_dir):
    split_manifest_dir = os.path.join(os.path.dirname(__file__), split_manifest_dir)
export_test_csv_dir = args.export_test_csv_dir
if not os.path.isabs(export_test_csv_dir):
    export_test_csv_dir = os.path.join(os.path.dirname(__file__), export_test_csv_dir)
scan_thresholds = parse_thresholds(args.scan_thresholds)

export_fixed_split(train_files, test_files, split_manifest_dir, export_test_csv_dir)

train_dataset = CsvPointDataset(
    train_files,
    small_k=args.small_k_neighbors,
    large_k=args.large_k_neighbors,
    cache_dir=feature_cache_dir,
)
test_dataset = CsvPointDataset(
    test_files,
    small_k=args.small_k_neighbors,
    large_k=args.large_k_neighbors,
    cache_dir=feature_cache_dir,
)
model = PointNet2BoundaryNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print(
    "使用轻量 PointNet++-style MSG 模型训练 "
    f"(small_k={args.small_k_neighbors}, large_k={args.large_k_neighbors})"
)
print(f"Early stopping metric: {args.early_stop_metric}")

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
)


def evaluate(model, data_loader, device, max_batches=0):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        test_iter = tqdm(data_loader, desc="Testing", leave=False)
        for batch_idx, batch in enumerate(test_iter):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            pts = batch["points"].to(device)
            patch_small = batch["patch_small"].to(device)
            patch_large = batch["patch_large"].to(device)
            labels = batch["labels"].to(device)

            pred = model(pts, patch_small, patch_large)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        return None, None, None, None, None

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    near_metrics = compute_boundary_metrics(
        all_preds,
        all_labels,
        boundary_threshold=args.boundary_threshold,
    )
    best_threshold, best_threshold_metrics = find_best_threshold(
        all_preds,
        all_labels,
        scan_thresholds,
    )
    return mse, mae, near_metrics, best_threshold, best_threshold_metrics


best_metric = -float("inf") if args.early_stop_metric in {"f1", "precision", "recall"} else float("inf")
best_metric_name = args.early_stop_metric
best_metric_snapshot = None
best_state_dict = deepcopy(model.state_dict())
start_epoch = 0
epochs_without_improve = 0

if args.resume_model:
    resume_path = args.resume_model
    if not os.path.isabs(resume_path):
        resume_path = os.path.join(os.path.dirname(__file__), resume_path)
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"未找到续训模型: {resume_path}")

    resume_info = load_training_checkpoint(resume_path, model, optimizer, device)
    start_epoch = resume_info["epoch"] + 1
    best_state_dict = deepcopy(model.state_dict())
    if resume_info["best_metric_name"] == best_metric_name and resume_info["best_metric_value"] is not None:
        best_metric = resume_info["best_metric_value"]
        best_metric_snapshot = resume_info["best_metric_snapshot"]
    print(
        f"已从 {resume_path} 继续训练，"
        f"起始 epoch={start_epoch + 1}"
    )

total_epochs = start_epoch + args.epochs
last_epoch = start_epoch - 1

for epoch in range(start_epoch, total_epochs):
    last_epoch = epoch
    model.train()
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")

    for batch_idx, batch in enumerate(train_iter):
        if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
            break
        pts = batch["points"].to(device)
        patch_small = batch["patch_small"].to(device)
        patch_large = batch["patch_large"].to(device)
        labels = batch["labels"].to(device)

        pred = model(pts, patch_small, patch_large)

        loss = boundary_aware_distance_loss(
            pred,
            labels,
            boundary_threshold=args.boundary_threshold,
            boundary_weight=args.boundary_weight,
            reg_loss_weight=args.reg_loss_weight,
            cls_loss_weight=args.cls_loss_weight,
            smooth_l1_beta=args.smooth_l1_beta,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            max_pos_weight=args.max_pos_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_iter.set_postfix(loss=loss.item())

        if epoch == 0 and batch_idx < 3:
            print('points:', pts[:5])
            print('labels:', labels[:5])
            print('pred:', pred[:5])
            print('loss:', loss.item())

    print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {loss.item():.4f}")
    train_mse, train_mae, train_near_metrics, train_best_threshold, train_best_threshold_metrics = evaluate(
        model,
        train_loader,
        device,
        max_batches=args.max_train_eval_batches,
    )
    mse, mae, near_metrics, best_threshold, best_threshold_metrics = evaluate(
        model,
        test_loader,
        device,
        max_batches=args.max_test_batches,
    )
    if mse is not None:
        if train_mse is not None:
            print(
                f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, "
                f"P={train_near_metrics['precision']:.4f}, "
                f"R={train_near_metrics['recall']:.4f}, "
                f"F1={train_near_metrics['f1']:.4f}"
            )
            print(
                "Train best-threshold "
                f"(scan={scan_thresholds}): "
                f"t={train_best_threshold:.4f}, "
                f"P={train_best_threshold_metrics['precision']:.4f}, "
                f"R={train_best_threshold_metrics['recall']:.4f}, "
                f"F1={train_best_threshold_metrics['f1']:.4f}"
            )
        print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(
            "Near-boundary metrics "
            f"(threshold={args.boundary_threshold}): "
            f"P={near_metrics['precision']:.4f}, "
            f"R={near_metrics['recall']:.4f}, "
            f"F1={near_metrics['f1']:.4f}, "
            f"TP={near_metrics['tp']}, FP={near_metrics['fp']}, FN={near_metrics['fn']}"
        )
        print(
            "Test best-threshold "
            f"(scan={scan_thresholds}): "
            f"t={best_threshold:.4f}, "
            f"P={best_threshold_metrics['precision']:.4f}, "
            f"R={best_threshold_metrics['recall']:.4f}, "
            f"F1={best_threshold_metrics['f1']:.4f}, "
            f"TP={best_threshold_metrics['tp']}, "
            f"FP={best_threshold_metrics['fp']}, "
            f"FN={best_threshold_metrics['fn']}"
        )

        current_metric = mae if best_metric_name == 'mae' else near_metrics[best_metric_name]
        metric_improved = (
            current_metric > (best_metric + args.early_stop_min_delta)
            if best_metric_name in {'f1', 'precision', 'recall'}
            else current_metric < (best_metric - args.early_stop_min_delta)
        )

        if metric_improved:
            best_metric = current_metric
            best_metric_snapshot = {
                'mse': mse,
                'mae': mae,
                'precision': near_metrics['precision'],
                'recall': near_metrics['recall'],
                'f1': near_metrics['f1'],
            }
            best_state_dict = deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}, "
                f"best {best_metric_name}={best_metric:.4f}, "
                f"patience={args.early_stop_patience}"
            )
            break

# 保存模型
model.load_state_dict(best_state_dict)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_type": "pointnet2_msg",
        "small_k_neighbors": args.small_k_neighbors,
        "large_k_neighbors": args.large_k_neighbors,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "best_metric_snapshot": best_metric_snapshot,
    },
    args.output,
)
print(f"模型已保存为 {args.output}")

# 训练后自动对第一个测试csv做预测并输出test.ply
if len(test_files) > 0:
    from predict import predict_on_csv
    import pandas as pd

    print(f"\n正在对第一个测试集csv自动推理并提取预测<{args.boundary_threshold}的点...")
    pred = predict_on_csv(
        test_files[0],
        model_path=args.output,
        threshold=args.boundary_threshold,
        save_output=False,
    )

    data = pd.read_csv(test_files[0], header=None, skiprows=1).values.astype(np.float32)
    points = data[:, :3]
    labels = data[:, 3]
    csv_stem = os.path.splitext(os.path.basename(test_files[0]))[0]

    full_cloud_path = f"{csv_stem}_full_cloud.ply"
    predicted_boundary_path = f"{csv_stem}_predicted_boundary.ply"
    gt_boundary_path = f"{csv_stem}_gt_boundary.ply"

    save_points_to_ply(points, full_cloud_path)
    print(f"已保存完整测试点云到 {full_cloud_path}，共{len(points)}个点")

    pred_mask = pred < args.boundary_threshold
    pred_points = points[pred_mask]
    if len(pred_points) > 0:
        save_points_to_ply(pred_points, predicted_boundary_path)
        save_points_to_ply(pred_points, "test.ply")
        print(
            f"已保存预测<{args.boundary_threshold}的点到 {predicted_boundary_path} "
            f"(同时覆盖 test.ply)，共{len(pred_points)}个点"
        )
    else:
        print("没有预测为近边界点，未生成test.ply")

    gt_mask = labels < args.boundary_threshold
    gt_points = points[gt_mask]
    if len(gt_points) > 0:
        save_points_to_ply(gt_points, gt_boundary_path)
        print(f"已保存真实<{args.boundary_threshold}的边界点到 {gt_boundary_path}，共{len(gt_points)}个点")
    else:
        print(f"真实标签中没有<{args.boundary_threshold}的边界点，未生成 {gt_boundary_path}")
