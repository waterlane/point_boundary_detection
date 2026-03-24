# datasets/dataset.py

import torch
from torch.utils.data import Dataset
from .label_utils import compute_distance, soft_label
import numpy as np
from plyfile import PlyData

def read_ply_xyz(filepath):
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    return xyz.astype(np.float32)

class PointDataset(Dataset):
    def __init__(self, points_files, label_points_files, sigma=0.02):
        assert len(points_files) == len(label_points_files), '点云和标签文件数需一致'
        self.points_files = points_files
        self.label_points_files = label_points_files
        self.sigma = sigma

    def __len__(self):
        return len(self.points_files)

    def __getitem__(self, idx):
        points = read_ply_xyz(self.points_files[idx])
        label_points = read_ply_xyz(self.label_points_files[idx])
        d = compute_distance(points, label_points)
        labels = soft_label(d, self.sigma).astype("float32")
        return {
            "points": torch.from_numpy(points),
            "labels": torch.from_numpy(labels)
        }