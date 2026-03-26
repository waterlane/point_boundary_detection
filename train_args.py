import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Point Boundary Detection Training")
    parser.add_argument('--points', type=str, default=os.path.join('data', 'points.ply'),
                        help='Path to input point cloud ply file')
    parser.add_argument('--label_points', type=str, default=os.path.join('data', 'label_points.ply'),
                        help='Path to label (boundary) points ply file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='If > 0, limit train batches per epoch for faster iteration')
    parser.add_argument('--max_test_batches', type=int, default=0,
                        help='If > 0, limit test batches during evaluation')
    parser.add_argument('--max_train_eval_batches', type=int, default=32,
                        help='If > 0, limit train batches during per-epoch train-set evaluation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.02, help='Sigma for soft label')
    parser.add_argument('--output', type=str, default='model.pth', help='Path to save model')
    parser.add_argument('--resume_model', type=str, default='',
                        help='Optional checkpoint path to resume training from')
    parser.add_argument('--split_manifest_dir', type=str, default=os.path.join('data', 'splits'),
                        help='Directory used to save fixed train/test split manifests')
    parser.add_argument('--export_test_csv_dir', type=str, default=os.path.join('data', 'test_csv'),
                        help='Directory used to export the fixed test csv files')
    parser.add_argument('--boundary_threshold', type=float, default=0.005,
                        help='Boundary distance threshold for focused training and evaluation')
    parser.add_argument('--boundary_weight', type=float, default=3.0,
                        help='Extra weight for near-boundary points in the loss')
    parser.add_argument('--reg_loss_weight', type=float, default=1.0,
                        help='Weight for the distance regression term')
    parser.add_argument('--cls_loss_weight', type=float, default=1.0,
                        help='Weight for the near-boundary classification term')
    parser.add_argument('--smooth_l1_beta', type=float, default=0.002,
                        help='Beta used by Smooth L1 regression loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma for the near-boundary classification term')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Focal loss alpha for the near-boundary classification term')
    parser.add_argument('--max_pos_weight', type=float, default=5.0,
                        help='Upper bound for positive class weight to prevent extreme imbalance from dominating')
    parser.add_argument('--small_k_neighbors', type=int, default=16,
                        help='Number of nearest neighbors in the small-scale local patch')
    parser.add_argument('--large_k_neighbors', type=int, default=32,
                        help='Number of nearest neighbors in the large-scale local patch')
    parser.add_argument('--feature_cache_dir', type=str, default=os.path.join('data', 'feature_cache'),
                        help='Directory for cached neighborhood indices')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible train/test split')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Stop training if the selected validation metric does not improve for N epochs (<=0 disables)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Minimum validation metric improvement to reset early-stop patience')
    parser.add_argument('--early_stop_metric', type=str, default='f1', choices=['f1', 'mae', 'precision', 'recall'],
                        help='Validation metric used for early stopping and best-model selection')
    parser.add_argument('--scan_thresholds', type=str, default='0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.01',
                        help='Comma-separated prediction thresholds to scan for best F1 analysis')
    return parser.parse_args()
