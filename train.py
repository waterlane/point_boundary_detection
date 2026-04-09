import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.csv_dataset import CsvPointDataset
from losses.loss import boundary_aware_distance_loss
from models.dgcnn import DGCNN
from train_args import get_args
from utils.metrics import compute_boundary_metrics


args = get_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

csv_dir = os.path.join(os.path.dirname(__file__), 'data', 'csv')
csv_files = sorted([os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')])
if len(csv_files) == 0:
    raise RuntimeError('未找到 data/csv/ 下的 csv 文件')

# 自动划分训练集和测试集（8:2）
random.shuffle(csv_files)
split_idx = int(len(csv_files) * 0.8)
train_files = csv_files[:split_idx]
test_files = csv_files[split_idx:]
train_dataset = CsvPointDataset(train_files)
test_dataset = CsvPointDataset(test_files)

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
            labels = batch["labels"].to(device)

            pred = model(pts)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        return None, None, None

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    near_metrics = compute_boundary_metrics(
        all_preds,
        all_labels,
        boundary_threshold=args.boundary_threshold,
    )
    return mse, mae, near_metrics


best_mae = float("inf")
epochs_without_improve = 0

for epoch in range(args.epochs):
    model.train()
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

    for batch_idx, batch in enumerate(train_iter):
        if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
            break
        pts = batch["points"].to(device)
        labels = batch["labels"].to(device)

        pred = model(pts)

        loss = boundary_aware_distance_loss(
            pred,
            labels,
            boundary_threshold=args.boundary_threshold,
            boundary_weight=args.boundary_weight,
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

    print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}")
    mse, mae, near_metrics = evaluate(model, test_loader, device, max_batches=args.max_test_batches)
    if mse is not None:
        print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(
            "Near-boundary metrics "
            f"(threshold={args.boundary_threshold}): "
            f"P={near_metrics['precision']:.4f}, "
            f"R={near_metrics['recall']:.4f}, "
            f"F1={near_metrics['f1']:.4f}, "
            f"TP={near_metrics['tp']}, FP={near_metrics['fp']}, FN={near_metrics['fn']}"
        )

        if mae < (best_mae - args.early_stop_min_delta):
            best_mae = mae
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}, "
                f"best MAE={best_mae:.4f}, patience={args.early_stop_patience}"
            )
            break

# 保存模型
torch.save(model.state_dict(), args.output)
print(f"模型已保存为 {args.output}")

# 训练后自动对第一个测试csv做预测并输出test.ply
if len(test_files) > 0:
    from predict import predict_on_csv
    from plyfile import PlyData, PlyElement
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
    mask = pred < args.boundary_threshold
    sel_points = points[mask]

    if len(sel_points) > 0:
        vertex = np.array([tuple(p) for p in sel_points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write("test.ply")
        print(f"已保存预测<{args.boundary_threshold}的点到 test.ply，共{len(sel_points)}个点")
    else:
        print("没有预测为近边界点，未生成test.ply")
