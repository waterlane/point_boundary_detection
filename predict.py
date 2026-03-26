
import torch
import numpy as np
import pandas as pd
from datasets.csv_dataset import build_model_inputs, compute_local_features
from models.dgcnn import DGCNN
from models.pointnet2 import PointNet2BoundaryNet
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


def save_points_to_ply(points, ply_path):
    # points: (N, 3) numpy array
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(ply_path)


def load_points_from_file(input_path):
    suffix = input_path.lower().rsplit(".", 1)[-1] if "." in input_path else ""
    if suffix == "csv":
        data = pd.read_csv(input_path, header=None, skiprows=1).values.astype(np.float32)
        return data[:, :3]
    if suffix == "ply":
        ply = PlyData.read(input_path)
        vertex = ply["vertex"]
        return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    raise ValueError(f"不支持的输入格式: {input_path}")


def filter_isolated_predictions(points, pred_mask, neighbor_k=16, min_neighbor_predictions=3):
    """
    删除过于离散的预测点：
    对每个被预测为边界的点，统计其 k 个近邻中也被预测为边界的数量，
    若小于阈值则将其过滤掉。
    """
    if pred_mask.sum() == 0 or neighbor_k <= 0:
        return pred_mask

    query_k = min(neighbor_k + 1, len(points))
    nbrs = NearestNeighbors(n_neighbors=query_k, algorithm="kd_tree")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)

    filtered_mask = pred_mask.copy()
    candidate_indices = np.flatnonzero(pred_mask)
    for idx in candidate_indices:
        neighbor_idx = indices[idx, 1:]
        predicted_neighbors = pred_mask[neighbor_idx].sum()
        if predicted_neighbors < min_neighbor_predictions:
            filtered_mask[idx] = False

    return filtered_mask


def load_model_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_type = checkpoint.get("model_type", "dgcnn")
        input_dim = int(checkpoint.get("input_dim", 3))
        small_k_neighbors = int(checkpoint.get("small_k_neighbors", 16))
        large_k_neighbors = int(checkpoint.get("large_k_neighbors", 32))
        k_neighbors = int(checkpoint.get("k_neighbors", 16))
    else:
        state_dict = checkpoint
        model_type = "dgcnn"
        input_dim = 3
        small_k_neighbors = 16
        large_k_neighbors = 32
        k_neighbors = 16

    if model_type == "pointnet2_msg":
        model = PointNet2BoundaryNet()
    else:
        model = DGCNN(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_type, input_dim, k_neighbors, small_k_neighbors, large_k_neighbors

def predict_on_points_file(
    input_path,
    model_path="model.pth",
    device="cpu",
    print_num=10,
    save_output=True,
    threshold=None,
    denoise_neighbor_k=0,
    denoise_min_neighbors=0,
):
    # 加载模型
    model, model_type, input_dim, k_neighbors, small_k_neighbors, large_k_neighbors = load_model_checkpoint(model_path, device)

    points = load_points_from_file(input_path)

    with torch.no_grad():
        if model_type == "pointnet2_msg":
            centers, patch_small, patch_large = build_model_inputs(
                points,
                small_k=small_k_neighbors,
                large_k=large_k_neighbors,
            )
            center_tensor = torch.from_numpy(centers).to(device)
            patch_small_tensor = torch.from_numpy(patch_small).to(device)
            patch_large_tensor = torch.from_numpy(patch_large).to(device)
            pred = model(center_tensor, patch_small_tensor, patch_large_tensor)
        else:
            model_input = points
            if input_dim > 3:
                local_features = compute_local_features(points, k_neighbors=k_neighbors)
                model_input = np.concatenate([points, local_features], axis=1).astype(np.float32)
            pts = torch.from_numpy(model_input).to(device)
            pred = model(pts)
        pred = pred.cpu().numpy().flatten()
    print("全部点的预测结果：")
    print(pred)
    if save_output:
        # 保存为csv，和原始点云坐标拼接
        out_arr = np.concatenate([points, pred[:, None]], axis=1)
        stem = input_path.rsplit(".", 1)[0]
        out_path = f"{stem}_pred.csv"
        pd.DataFrame(out_arr, columns=["x", "y", "z", "predicted_distance"]).to_csv(out_path, index=False)
        print(f"已保存完整预测结果到 {out_path}")
    # 额外保存预测值小于阈值的点为ply
    if threshold is not None:
        mask = pred < threshold
        if denoise_neighbor_k > 0 and denoise_min_neighbors > 0:
            raw_count = int(mask.sum())
            mask = filter_isolated_predictions(
                points,
                mask,
                neighbor_k=denoise_neighbor_k,
                min_neighbor_predictions=denoise_min_neighbors,
            )
            filtered_count = int(mask.sum())
            print(
                f"已进行离散点过滤: k={denoise_neighbor_k}, "
                f"min_neighbors={denoise_min_neighbors}, "
                f"保留 {filtered_count}/{raw_count} 个预测点"
            )
        sel_points = points[mask]
        if len(sel_points) > 0:
            stem = input_path.rsplit(".", 1)[0]
            ply_path = f"{stem}_predicted_below_{threshold}.ply"
            save_points_to_ply(sel_points, ply_path)
            print(f"已保存预测<{threshold}的点到 {ply_path}，共{len(sel_points)}个点")
        else:
            print(f"没有预测<{threshold}的点，无ply文件输出")
    return pred


def predict_on_csv(
    csv_path,
    model_path="model.pth",
    device="cpu",
    print_num=10,
    save_output=True,
    threshold=None,
    denoise_neighbor_k=0,
    denoise_min_neighbors=0,
):
    return predict_on_points_file(
        csv_path,
        model_path=model_path,
        device=device,
        print_num=print_num,
        save_output=save_output,
        threshold=threshold,
        denoise_neighbor_k=denoise_neighbor_k,
        denoise_min_neighbors=denoise_min_neighbors,
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python predict.py <csv或ply文件路径> [模型路径] [阈值,可选] [denoise_neighbor_k,可选] [denoise_min_neighbors,可选]")
    else:
        input_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "model.pth"
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None
        denoise_neighbor_k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        denoise_min_neighbors = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        predict_on_points_file(
            input_path,
            model_path,
            threshold=threshold,
            denoise_neighbor_k=denoise_neighbor_k,
            denoise_min_neighbors=denoise_min_neighbors,
        )
