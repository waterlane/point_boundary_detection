# datasets/label_utils.py

import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_distance(points, label_points):
    nbrs = NearestNeighbors(n_neighbors=1).fit(label_points)
    distances, _ = nbrs.kneighbors(points)
    return distances.squeeze()

def soft_label(distances, sigma):
    return np.exp(-(distances ** 2) / (sigma ** 2))

def hard_label(distances, r1, r2):
    labels = np.zeros_like(distances)
    labels[distances < r1] = 1
    labels[(distances >= r1) & (distances < r2)] = -1
    return labels