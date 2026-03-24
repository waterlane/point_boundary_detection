import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Point Boundary Detection Training")
    parser.add_argument('--points', type=str, default=os.path.join('data', 'points.ply'),
                        help='Path to input point cloud ply file')
    parser.add_argument('--label_points', type=str, default=os.path.join('data', 'label_points.ply'),
                        help='Path to label (boundary) points ply file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.02, help='Sigma for soft label')
    parser.add_argument('--output', type=str, default='model.pth', help='Path to save model')
    parser.add_argument('--boundary_threshold', type=float, default=0.005,
                        help='Boundary distance threshold for focused training and evaluation')
    parser.add_argument('--boundary_weight', type=float, default=6.0,
                        help='Extra weight for near-boundary points in the loss')
    return parser.parse_args()
