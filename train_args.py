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
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.02, help='Sigma for soft label')
    parser.add_argument('--output', type=str, default='model.pth', help='Path to save model')
    parser.add_argument('--boundary_threshold', type=float, default=0.005,
                        help='Boundary distance threshold for focused training and evaluation')
    parser.add_argument('--boundary_weight', type=float, default=3.0,
                        help='Extra weight for near-boundary points in the loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible train/test split')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Stop training if test MAE does not improve for N epochs (<=0 disables)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Minimum MAE improvement to reset early-stop patience')
    return parser.parse_args()
