import os
import subprocess
import glob

def run_edge_detection(ply_dir, edge_out_dir, edge_script, edge_args):
    os.makedirs(edge_out_dir, exist_ok=True)
    ply_files = sorted(glob.glob(os.path.join(ply_dir, '*.ply')))
    edge_outputs = []
    for ply_path in ply_files:
        name = os.path.splitext(os.path.basename(ply_path))[0]
        out_path = os.path.join(edge_out_dir, f'edge_{name}_refined.ply')
        cmd = [
            'python', edge_script,
            '--input', ply_path,
            '--output', out_path
        ] + edge_args
        print('Running:', ' '.join(map(str, cmd)))
        subprocess.run(cmd, check=True)
        edge_outputs.append(out_path)
    return edge_outputs

def run_mesh_sampling(mesh_dir, norm_script, sample_script, norm_out_dir, sample_out_dir, sample_num):
    os.makedirs(norm_out_dir, exist_ok=True)
    os.makedirs(sample_out_dir, exist_ok=True)
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, '*')))
    sample_outputs = []
    for mesh_path in mesh_files:
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        norm_path = os.path.join(norm_out_dir, f'normalized_{name}.obj')
        # 归一化
        norm_cmd = ['python', norm_script, '--input', mesh_path, '--output', norm_path]
        print('Running:', ' '.join(map(str, norm_cmd)))
        subprocess.run(norm_cmd, check=True)
        # 采样
        sample_cmd = [
            'python', sample_script,
            '--mesh', norm_path,
            '--out_dir', sample_out_dir,
            '--samples', str(sample_num)
        ]
        print('Running:', ' '.join(map(str, sample_cmd)))
        subprocess.run(sample_cmd, check=True)
        sample_outputs.append(os.path.join(sample_out_dir, 'boundary_samples.ply'))
    return sample_outputs

def run_filter(source_files, target_files, filter_script, filter_out_dir, thresh=0.05):
    os.makedirs(filter_out_dir, exist_ok=True)
    filtered_outputs = []
    for src, tgt in zip(source_files, target_files):
        name = os.path.splitext(os.path.basename(src))[0].replace('edge_', '')
        out_path = os.path.join(filter_out_dir, f'filtered_{name}.ply')
        cmd = [
            'python', filter_script,
            '--src', src,
            '--tgt', tgt,
            '--thresh', str(thresh),
            '--output', out_path
        ]
        print('Running:', ' '.join(map(str, cmd)))
        subprocess.run(cmd, check=True)
        filtered_outputs.append(out_path)
    return filtered_outputs

def main():
    # 路径配置
    ply_dir = '/root/projects/traditional_method/output_pcd'
    edge_out_dir = '/root/projects/laplacian-edge-detection/data/output'
    edge_script = '/root/projects/laplacian-edge-detection/src/edge_by_neighborhood_distribution.py'
    edge_args = [
        '--k', '36', '--min_neighbors', '16', '--angle_gap_deg', '145', '--centroid_shift_ratio', '0.33',
        '--dominant_gap_ratio', '1.55', '--max_occupancy_ratio', '0.70', '--angle_bins', '24', '--normal_knn', '36'
    ]

    mesh_dir = '/root/projects/cloth_reconstruction/meshes'
    norm_script = '/root/projects/cloth_reconstruction/normalize_any.py'
    norm_out_dir = '/root/projects/cloth_reconstruction/normalization'
    sample_script = '/root/projects/cloth_reconstruction/sample_boundary_loops_points.py'
    sample_out_dir = '/root/projects/cloth_reconstruction/samples'
    sample_num = 2000

    filter_script = '/root/projects/mesh_fit_pointcloud/filter_by_distance.py'
    filter_out_dir = '/root/projects/mesh_fit_pointcloud/filtered'
    filter_thresh = 0.05

    # 步骤1：批量边缘检测
    edge_files = run_edge_detection(ply_dir, edge_out_dir, edge_script, edge_args)
    # 步骤2：批量归一化和采样
    sample_files = run_mesh_sampling(mesh_dir, norm_script, sample_script, norm_out_dir, sample_out_dir, sample_num)
    # 步骤3：批量过滤
    run_filter(edge_files, sample_files, filter_script, filter_out_dir, filter_thresh)

if __name__ == '__main__':
    main()
