"""
TODO @abhaydmathur 
    1. Isolate obstacles from the pointcloud using
        - RANSAC or <something else> to ID Surface Normals
        - 
"""

import os

# print(os.popen("pwd").read())

import params.robot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import plotly.express as px

# from umap.umap_ import UMAP

THRESH_HEIGHT = 10  # find a value that works.

robot_params = params.robot
DEPTH_RANGE = [0.7, 7]

def depthimage_to_pointcloud(depth_img, K=robot_params.K, as_list=True):
    print(np.unique(depth_img, return_counts=True))
    depth_img[depth_img == np.inf] = np.nan
    # print(np.max(depth_img[~np.isnan(depth_img)]), np.min(depth_img[~np.isnan(depth_img)]))
    # input()
    # depth = depth_img / 255.0
    # depth_mask = np.where(depth>0, 1, -1)
    # depth = np.where(depth_mask>0, DEPTH_RANGE[0]+(depth*(DEPTH_RANGE[1]-DEPTH_RANGE[0])), np.nan)
    return depth_to_pointcloud(depth_img, K, as_list)


def depth_to_pointcloud(depth, K=robot_params.K, as_list=True):
    """
    depth : ndarray (depth image)
    K : ndarray (camera's internal calibration matrix)
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rows, cols = depth.shape
    valid = ~np.isnan(depth)
    print(f"valid ; {np.unique(valid, return_counts=True)}")
    z = np.where(valid, depth, np.nan)
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)

    cloud = np.dstack((x, y, z))
    cloud = cloud[valid]
    print(cloud.shape)
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
    if rgb is None:
        fig = px.scatter_3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2])
    else:
        fig = px.scatter_3d(x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2])
    fig.update_traces(marker_size=2)
    fig.show()


def plane_distance(pt, ref, normal):
    assert np.linalg.norm(normal) == 1
    return np.abs(np.dot(pt - ref, normal))


class RansacSegmentation:
    def __init__(
        self,
        pointcloud,
        cos_thresh=0.1,
        ref_normal=np.array([0, 0, 1]),
        stopping_limit=0.2,
    ):
        """
        pointcloud : Nx3 ndarray
        ref_normal : 3x1 ndarray
        stopping_limit : fraction of points in the pointcloud at which ransac can be terminated
        """
        self.pointcloud = pointcloud
        self.ref_normal = ref_normal
        self.inliers = None
        self.outliers = None
        self.cos_dist_thresh = cos_thresh
        self.reference_thresh = np.cos(4 * np.pi / 18)  # 40 deg

        self.stopping_limit = pointcloud.shape[0] * stopping_limit

        self.rng = np.random.default_rng()

    def distance(self, point):
        return np.abs(np.dot(point - self.ref_point, self.normal))

    def dist(self, x, normal, ref, use_cosine=False):
        x = x - ref
        if use_cosine:
            x = x / np.linalg.norm(x)
        return np.abs(np.dot(x, normal))

    def fit(
        self,
        runs=100,
        dist_threshold=0.1,
    ):
        """
        refer to nomenclature


        """
        inliers_result = set()

        for i in range(10000):
            # p1, p2, p3 = self.rng.choice(self.pointcloud, 3, replace=False)
            # inliers = [p1, p2, p3]

            p1, p2, p3 = self.rng.choice(len(self.pointcloud), 3, replace=False)
            inliers = [self.pointcloud[p1], self.pointcloud[p2], self.pointcloud[p3]]
            p1, p2, p3 = inliers

            normal = np.cross(p2 - p1, p3 - p1)
            normal = normal / np.linalg.norm(normal)

            if np.dot(normal, self.ref_normal) < self.reference_thresh:
                continue

            for point in self.pointcloud:
                point = np.array(point)
                # try:
                #     if point in inliers:
                #         continue
                # except:
                #     print(point)
                #     continue

                d = self.dist(point, normal, p1, use_cosine=False)

                if d < self.cos_dist_thresh:
                    inliers.append(point)

            if len(inliers) > len(inliers_result):
                inliers_result.clear()
                inliers_result = inliers
                self.ref_point = p1
                self.normal = normal

            if len(inliers_result) > self.stopping_limit:
                break

        return inliers_result
