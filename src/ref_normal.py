import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
id = np.random.randint(1, 10)
id = 6
dir = "../bagfiles/images_extracted/"
img, depth, norm = (Image.open(dir+f"{id}.png"), Image.open(dir+f"{id}d.png"), Image.open(dir+f"{id}n.png"))
img, depth, norm = np.array(img)/255, np.array(depth)/255, np.array(norm)/255

# Mask everything by valid depth
depth = np.where(depth>0, depth, np.nan)

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize = (15, 45))
# plt.axis("off")
a1.imshow(img)
a2.imshow(depth)
a3.imshow(norm)
# !pwd
import params.params.robot
K = params.params.robot.K

K
fx, fy, cx, cy = K[0,0], K[1, 1], K[0, 2], K[1, 2]

fx, fy, cx, cy
valid = (depth>=0) & (depth<1)
np.unique(valid)
z = np.where(valid, depth, np.nan)
np.unique(np.unique(z)-np.unique(depth))
from obstacle_detection.pointcloud import depthimage_to_pointcloud, depth_to_pointcloud, visualise_pointcloud, visualise_pointcloud_plotly

pc = depth_to_pointcloud(depth, K)

pc.shape
pc = pc[~np.isnan(depth)]
pc.shape
idx = np.random.randint(pc.shape[0], size = 10000)
pc_ = pc[idx, :]
visualise_pointcloud_plotly(pc_)