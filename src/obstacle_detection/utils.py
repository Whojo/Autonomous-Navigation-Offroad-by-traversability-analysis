import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pointcloud import RansacSegmentation
from obstacle_detection.normals import normals_from_pointcloud
from obstacle_detection.pointcloud import depthimage_to_pointcloud, depth_to_pointcloud, visualise_pointcloud, visualise_pointcloud_plotly
from params.robot import K

DEPTH_RANGE = [0.7, 7]

def calc_obstacle_costs(img, depth):
    depth_mask = np.where(depth>0, 1, -1)
    depth = np.where(depth_mask>0, DEPTH_RANGE[0]+(depth*(DEPTH_RANGE[1]-DEPTH_RANGE[0])), np.nan)
    pc = depth_to_pointcloud(depth, K)
    idx = np.random.randint(pc.shape[0], size = 10000)
    pc_ = pc[idx, :]
    seg = RansacSegmentation(pc_, ref_normal=[0, -1, 0], stopping_limit = 0.7)
    plane_points = seg.fit(pc, 100)
    pts = np.array(plane_points)

    ref_normal = np.array([0, -1, 0])[:, np.newaxis]
    normals = normals_from_pointcloud(pc, mode="sphere_fit")

    inclination_cost = np.nan*np.ones_like(depth)
    inccosts = np.array([ max(x, 0) for x in (- normals @ ref_normal).squeeze()])

    inclination_cost[~np.isnan(depth)] =  inccosts # dot product of unit vecs

    elevation_cost = np.nan*np.ones_like(depth)
    elevation_cost[~np.isnan(depth)] = seg.distance(pc)

    return elevation_cost, inclination_cost
