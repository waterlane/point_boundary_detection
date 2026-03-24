import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CsvPointDataset(Dataset):
    """
    读取csv格式的点云+边界距离数据，前三列为xyz，第四列为label
    """
    def __init__(self, csv_files):
        self.csv_files = csv_files
        self.data = []
        for f in csv_files:
            arr = pd.read_csv(f, header=None, skiprows=1).values.astype(np.float32)
            self.data.append(arr)
        self.lengths = [len(arr) for arr in self.data]
        self.cum_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # 支持跨多个csv文件索引
        file_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        local_idx = idx - self.cum_lengths[file_idx]
        row = self.data[file_idx][local_idx]
        xyz = row[:3]
        label = row[3]
        return {
            "points": torch.from_numpy(xyz),
            "labels": torch.tensor(label, dtype=torch.float32)
        }
