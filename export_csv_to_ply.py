import os
import sys

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement


def save_points_to_ply(points, ply_path):
    vertex = np.array(
        [tuple(p) for p in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(ply_path)


def export_csv_xyz_to_ply(csv_path, output_path=None):
    data = pd.read_csv(csv_path, header=None, skiprows=1).values.astype(np.float32)
    points = data[:, :3]

    if output_path is None:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{stem}.ply"

    save_points_to_ply(points, output_path)
    print(f"已导出 {csv_path} -> {output_path}，共{len(points)}个点")


def export_dir_xyz_to_ply(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.endswith(".csv")
    )

    if not csv_files:
        raise RuntimeError(f"目录下没有 csv 文件: {input_dir}")

    for csv_path in csv_files:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(output_dir, f"{stem}.ply")
        export_csv_xyz_to_ply(csv_path, output_path)

    print(f"已完成，共导出 {len(csv_files)} 个 ply 到 {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_input_dir = os.path.join("data", "test_csv")
        default_output_dir = os.path.join("data", "test_ply")
        export_dir_xyz_to_ply(default_input_dir, default_output_dir)
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            output_dir = f"{input_path.rstrip(os.sep)}_ply"
            export_dir_xyz_to_ply(input_path, output_dir)
        else:
            export_csv_xyz_to_ply(input_path)
    elif len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        if os.path.isdir(input_path):
            export_dir_xyz_to_ply(input_path, output_path)
        else:
            export_csv_xyz_to_ply(input_path, output_path)
    else:
        print("用法:")
        print("  python export_csv_to_ply.py")
        print("  python export_csv_to_ply.py <csv文件或目录>")
        print("  python export_csv_to_ply.py <csv文件或目录> <输出文件或目录>")
