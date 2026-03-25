
import torch
import numpy as np
import pandas as pd
from models.dgcnn import DGCNN
from plyfile import PlyData, PlyElement


def save_points_to_ply(points, ply_path):
    # points: (N, 3) numpy array
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(ply_path)

def predict_on_csv(csv_path, model_path="model.pth", device="cpu", print_num=10, save_output=True, threshold=None):
    # 加载模型
    model = DGCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 读取csv，跳过表头
    data = pd.read_csv(csv_path, header=None, skiprows=1).values.astype(np.float32)
    points = data[:, :3]  # 取前三列

    with torch.no_grad():
        pts = torch.from_numpy(points).to(device)
        pred = model(pts)
        pred = pred.cpu().numpy().flatten()
    print("全部点的预测结果：")
    print(pred)
    if save_output:
        # 保存为csv，和原始点云坐标拼接
        out_arr = np.concatenate([points, pred[:, None]], axis=1)
        out_path = "output_pred.csv"
        pd.DataFrame(out_arr, columns=["x", "y", "z", "predicted_distance"]).to_csv(out_path, index=False)
        print(f"已保存完整预测结果到 {out_path}")
    # 额外保存预测值小于阈值的点为ply
    if threshold is not None:
        mask = pred < threshold
        sel_points = points[mask]
        if len(sel_points) > 0:
            ply_path = f"predicted_below_{threshold}.ply"
            save_points_to_ply(sel_points, ply_path)
            print(f"已保存预测<{threshold}的点到 {ply_path}，共{len(sel_points)}个点")
        else:
            print(f"没有预测<{threshold}的点，无ply文件输出")
    return pred

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python predict.py <csv文件路径> [模型路径] [阈值,可选]")
    else:
        csv_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "model.pth"
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None
        predict_on_csv(csv_path, model_path, threshold=threshold)
