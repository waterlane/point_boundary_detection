import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}


def load_point_cloud_points(path):
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"点云为空: {path}")
    return points


def sample_mesh_points(path, sample_points):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"mesh 为空或读取失败: {path}")
    sampled = mesh.sample_points_uniformly(number_of_points=sample_points)
    points = np.asarray(sampled.points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"mesh 采样结果为空: {path}")
    return points


def normalize_points(points):
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=1).max()
    if scale < 1e-8:
        return centered
    return centered / scale


def downsample_points(points, target_points):
    if len(points) <= target_points:
        return points
    indices = np.linspace(0, len(points) - 1, target_points, dtype=np.int64)
    return points[indices]


def chamfer_distance(points_a, points_b):
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    dist_ab, _ = tree_b.query(points_a, k=1)
    dist_ba, _ = tree_a.query(points_b, k=1)
    return float(dist_ab.mean() + dist_ba.mean())


def gather_mesh_points(mesh_dir, sample_points):
    mesh_dir = Path(mesh_dir)
    mesh_paths = sorted(
        path for path in mesh_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS
    )
    if not mesh_paths:
        raise RuntimeError(f"未在 {mesh_dir} 中找到 mesh 文件")

    mesh_points = {}
    for path in mesh_paths:
        points = sample_mesh_points(path, sample_points=sample_points)
        mesh_points[path] = normalize_points(points)
    return mesh_points


def match_single_ply(ply_path, mesh_points, sample_points):
    ply_points = load_point_cloud_points(ply_path)
    ply_points = normalize_points(downsample_points(ply_points, sample_points))

    best_mesh = None
    best_score = None
    for mesh_path, mesh_pts in mesh_points.items():
        score = chamfer_distance(ply_points, mesh_pts)
        if best_score is None or score < best_score:
            best_mesh = mesh_path
            best_score = score

    return best_mesh, best_score


def sanitize_mesh_name(mesh_name):
    if mesh_name.endswith("_normalized"):
        return mesh_name[: -len("_normalized")]
    return mesh_name


def build_unique_target_path(ply_path, mesh_name, used_names):
    base_dir = ply_path.parent
    mesh_name = sanitize_mesh_name(mesh_name)
    candidate = f"{mesh_name}{ply_path.suffix}"
    if candidate not in used_names and not (base_dir / candidate).exists():
        used_names.add(candidate)
        return base_dir / candidate

    idx = 1
    while True:
        candidate = f"{mesh_name}_{idx}{ply_path.suffix}"
        if candidate not in used_names and not (base_dir / candidate).exists():
            used_names.add(candidate)
            return base_dir / candidate
        idx += 1


def match_and_plan(ply_dir, mesh_dir, sample_points):
    ply_dir = Path(ply_dir)
    ply_paths = sorted(path for path in ply_dir.iterdir() if path.is_file() and path.suffix.lower() == ".ply")
    if not ply_paths:
        raise RuntimeError(f"未在 {ply_dir} 中找到 ply 文件")

    mesh_points = gather_mesh_points(mesh_dir, sample_points=sample_points)
    used_names = set()
    plan = []
    for ply_path in ply_paths:
        best_mesh, score = match_single_ply(ply_path, mesh_points, sample_points=sample_points)
        target_path = build_unique_target_path(ply_path, best_mesh.stem, used_names)
        plan.append(
            {
                "source": str(ply_path),
                "matched_mesh": str(best_mesh),
                "score": score,
                "target": str(target_path),
            }
        )
    return plan


def apply_plan(plan):
    for item in plan:
        source = Path(item["source"])
        target = Path(item["target"])
        if source == target:
            continue
        source.rename(target)


def main():
    parser = argparse.ArgumentParser(description="批量匹配 ply 到 mesh，并按最接近的 mesh 名称重命名")
    parser.add_argument("ply_dir", help="待匹配的 ply 文件夹")
    parser.add_argument("mesh_dir", help="候选 mesh 文件夹")
    parser.add_argument("--sample-points", type=int, default=4096, help="每个形状用于匹配的采样点数")
    parser.add_argument("--report", type=str, default="match_report.json", help="匹配结果报告输出路径")
    parser.add_argument("--apply", action="store_true", help="实际执行重命名；默认仅打印计划")
    args = parser.parse_args()

    plan = match_and_plan(args.ply_dir, args.mesh_dir, sample_points=args.sample_points)
    for item in plan:
        print(
            f"{item['source']} -> {item['target']} | "
            f"mesh={item['matched_mesh']} | score={item['score']:.6f}"
        )

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"已保存匹配报告到 {args.report}")

    if args.apply:
        apply_plan(plan)
        print("已执行重命名")
    else:
        print("当前为 dry-run，未实际重命名；确认无误后加 --apply")


if __name__ == "__main__":
    main()
