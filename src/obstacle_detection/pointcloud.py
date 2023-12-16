"""
TODO @abhaydmathur 
    1. Isolate obstacles from the pointcloud using
        - RANSAC or <something else> to ID Surface Normals
        - 
"""

import os

# print(os.popen("pwd").read())

import params.params.robot 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import plotly.express as px
from umap.umap_ import UMAP

THRESH_HEIGHT = 10  # find a value that works.

robot_params = params.params.robot

def depthimage_to_pointcloud(depth_img, K=robot_params.K, as_list=True):
    depth = depth_img / 255
    return depth_to_pointcloud(depth, K, as_list)


def depth_to_pointcloud(depth, K=robot_params.K, as_list=True):
    """
    depth : ndarray (depth image)
    K : ndarray (camera's internal calibration matrix)
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rows, cols = depth.shape
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)

    cloud = np.dstack((x, y, z))

    # cloud = cloud[~np.isnan(cloud)]

    return cloud


def visualise_pointcloud(cloud):
    """
    cloud : Nx3 ndarray
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def visualise_pointcloud_plotly(cloud, rgb=None):
    # fig = px.scatter_3d(x = after_nmf[:, 0], y = after_nmf[:, 1], z = after_nmf[:, 2], color = algo_nmf.predict(X_test))
    if rgb is None: fig = px.scatter_3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2])
    else:
        fig = px.scatter_3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2]``)
    fig.update_traces(marker_size = 2)
    fig.show()


def plane_distance(pt, ref, normal):
    assert np.linalg.norm(normal) == 1
    return np.abs(np.dot(pt-ref, normal))

class RansacSegmentation:
    def __init__(self, pointcloud, cos_thresh=0.1, ref_normal=np.array([0, 0, 1])):
        """
        pointcloud : Nx3 ndarray
        ref_normal : 3x1 ndarray
        """
        self.pointcloud = pointcloud
        self.ref_normal = ref_normal
        self.inliers = None
        self.outliers = None
        self.cos_dist_thresh = cos_thresh
        self.reference_thresh = np.cos(4*np.pi/18) # 40 deg

        self.rng = np.random.default_rng()

    def cosine_dist(self, x, normal, ref):
        x = x - ref
        x = x / max(1e-4, np.linalg.norm(x))
        return np.abs(np.dot(x, normal))

    def fit(
        self,
        max_iter=100,
        dist_threshold=0.1,
    ):
        """
        refer to nomenclature


        """
        inliers_result = set()

        for _ in range(max_iter):
            p1, p2, p3 = self.rng.choice(self.pointcloud, 3, replace=False)
            inliers = [p1, p2, p3]

            normal = np.cross(p2 - p1, p3 - p1)
            normal = normal / max(1e-4, np.linalg.norm(normal))

            if np.dot(normal, self.ref_normal) < self.reference_thresh:
                continue

            for point in self.pointcloud:
                if point in inliers:
                    continue

                d = self.cosine_dist(point, normal, p1)

                if d < self.cos_dist_thresh:
                    inliers.append(point)

            if len(inliers) > len(inliers_result):
                inliers_result.clear()
                inliers_result = inliers

        return inliers_result
