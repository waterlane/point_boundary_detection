import os

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


def compute_local_features(points, k_neighbors=16):
    num_points = len(points)
    if num_points == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if num_points == 1:
        return np.zeros((1, 6), dtype=np.float32)

    k_eff = min(k_neighbors, num_points - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="kd_tree")
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    neighbor_distances = distances[:, 1:].astype(np.float32)
    neighbor_points = points[indices[:, 1:]].astype(np.float32)
    neighbor_centroid = neighbor_points.mean(axis=1)
    centroid_offset = neighbor_centroid - points

    centered_neighbors = neighbor_points - neighbor_centroid[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered_neighbors, centered_neighbors)
    cov /= max(k_eff, 1)

    eigvals = np.linalg.eigvalsh(cov).astype(np.float32)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    curvature = eigvals[:, :1] / (eigvals.sum(axis=1, keepdims=True) + 1e-6)

    mean_distance = neighbor_distances.mean(axis=1, keepdims=True)
    std_distance = neighbor_distances.std(axis=1, keepdims=True)

    return np.concatenate(
        [centroid_offset, mean_distance, std_distance, curvature],
        axis=1,
    ).astype(np.float32)


def _build_knn_indices(points, k_neighbors):
    num_points = len(points)
    if num_points == 0:
        return np.zeros((0, k_neighbors), dtype=np.int64)

    query_k = min(k_neighbors, num_points)
    nbrs = NearestNeighbors(n_neighbors=query_k, algorithm="kd_tree")
    nbrs.fit(points)
    _, indices = nbrs.kneighbors(points)

    if query_k < k_neighbors:
        pad = np.repeat(indices[:, -1:], k_neighbors - query_k, axis=1)
        indices = np.concatenate([indices, pad], axis=1)

    return indices.astype(np.int64)


def load_or_compute_knn_indices(csv_path, points, small_k, large_k, cache_dir=None):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"{os.path.splitext(os.path.basename(csv_path))[0]}_knn_{small_k}_{large_k}.npz"
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return cached["small_idx"].astype(np.int64), cached["large_idx"].astype(np.int64)

    small_idx = _build_knn_indices(points, small_k)
    large_idx = _build_knn_indices(points, large_k)

    if cache_path:
        np.savez_compressed(cache_path, small_idx=small_idx, large_idx=large_idx)

    return small_idx, large_idx


def build_local_patches(points, center_idx, knn_indices):
    patch = points[knn_indices[center_idx]].astype(np.float32)
    center = points[center_idx].astype(np.float32)
    relative_patch = patch - center[None, :]
    return center, relative_patch


def build_model_inputs(points, small_k=16, large_k=32, cache_dir=None, csv_path=None):
    small_idx, large_idx = load_or_compute_knn_indices(
        csv_path or "in_memory",
        points,
        small_k=small_k,
        large_k=large_k,
        cache_dir=cache_dir,
    )

    centers = points.astype(np.float32)
    small_patches = points[small_idx].astype(np.float32) - centers[:, None, :]
    large_patches = points[large_idx].astype(np.float32) - centers[:, None, :]
    return centers, small_patches, large_patches


class CsvPointDataset(Dataset):
    """
    读取csv格式的点云+边界距离数据，前三列为xyz，第四列为label。
    每个样本返回一个中心点及其多尺度邻域 patch，供 PointNet++ 风格局部编码使用。
    """

    def __init__(self, csv_files, small_k=16, large_k=32, cache_dir=None):
        self.csv_files = csv_files
        self.small_k = small_k
        self.large_k = large_k
        self.cache_dir = cache_dir
        self.data = []
        self.points = []
        self.small_indices = []
        self.large_indices = []

        for csv_file in csv_files:
            arr = pd.read_csv(csv_file, header=None, skiprows=1).values.astype(np.float32)
            xyz = arr[:, :3]
            small_idx, large_idx = load_or_compute_knn_indices(
                csv_file,
                xyz,
                small_k=self.small_k,
                large_k=self.large_k,
                cache_dir=self.cache_dir,
            )
            self.data.append(arr)
            self.points.append(xyz)
            self.small_indices.append(small_idx)
            self.large_indices.append(large_idx)

        self.lengths = [len(arr) for arr in self.data]
        self.cum_lengths = np.cumsum([0] + self.lengths)
        self.input_dim = 3

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cum_lengths, idx, side="right") - 1
        local_idx = idx - self.cum_lengths[file_idx]

        row = self.data[file_idx][local_idx]
        points = self.points[file_idx]
        center, patch_small = build_local_patches(points, local_idx, self.small_indices[file_idx])
        _, patch_large = build_local_patches(points, local_idx, self.large_indices[file_idx])
        label = row[3]

        return {
            "points": torch.from_numpy(center),
            "patch_small": torch.from_numpy(patch_small),
            "patch_large": torch.from_numpy(patch_large),
            "labels": torch.tensor(label, dtype=torch.float32),
        }
