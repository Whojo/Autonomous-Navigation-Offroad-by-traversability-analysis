# Python librairies
import numpy as np
import cv2
import torch
import torch.nn as nn
import PIL
import matplotlib.pyplot as plt
import pandas as pd

# Importing custom made parameters
import visualparams as viz
from params import PROJECT_PATH

# Initializing some parameters for the model
transform = viz.TRANSFORM
transform_depth = viz.TRANSFORM_DEPTH
transform_normal = viz.TRANSFORM_NORMAL
device = viz.DEVICE

model = viz.MODEL
model.load_state_dict(torch.load(viz.WEIGHTS))
model.eval()

velocities = np.linspace(0.2, 1, 5)
print(velocities)
score_road = np.zeros(velocities.shape)
score_grass = np.zeros(velocities.shape)
score_branch = np.zeros(velocities.shape)

midpoints = viz.MIDPOINTS

dataset_dir = PROJECT_PATH / "datasets/dataset_road_grass_filtered"

index1 = "00309"
index2 = "00528"
index3 = "00017"

road = cv2.imread(str(dataset_dir / f"/images/{index1}.png"), cv2.IMREAD_COLOR)
road_depth = cv2.imread(str(dataset_dir / f"/images/{index1}d.png"), cv2.IMREAD_GRAYSCALE)
road_normals = cv2.imread(str(dataset_dir / f"/images/{index1}n.png"), cv2.IMREAD_COLOR)

grass = cv2.imread(str(dataset_dir / f"/images/{index2}.png"), cv2.IMREAD_COLOR)
grass_depth = cv2.imread(str(dataset_dir / f"/images/{index2}d.png"), cv2.IMREAD_GRAYSCALE)
grass_normals = cv2.imread(str(dataset_dir / f"/images/{index2}n.png"), cv2.IMREAD_COLOR)

branch = cv2.imread(str(dataset_dir / f"/images/{index3}.png"), cv2.IMREAD_COLOR)
branch_depth = cv2.imread(str(dataset_dir / f"/images/{index3}d.png"), cv2.IMREAD_GRAYSCALE)
branch_normals = cv2.imread(str(dataset_dir / f"/images/{index3}n.png"), cv2.IMREAD_COLOR)

# Make a PIL image
road = PIL.Image.fromarray(road)
road_depth = PIL.Image.fromarray(road_depth)
road_normals = PIL.Image.fromarray(road_normals)

grass = PIL.Image.fromarray(grass)
grass_depth = PIL.Image.fromarray(grass_depth)
grass_normals = PIL.Image.fromarray(grass_normals)

branch = PIL.Image.fromarray(branch)
branch_depth = PIL.Image.fromarray(branch_depth)
branch_normals = PIL.Image.fromarray(branch_normals)

# Apply transforms to the image
road = viz.TRANSFORM(road)
road_depth = viz.TRANSFORM_DEPTH(road_depth)
road_normals = viz.TRANSFORM_NORMAL(road_normals)

grass = viz.TRANSFORM(grass)
grass_depth = viz.TRANSFORM_DEPTH(grass_depth)
grass_normals = viz.TRANSFORM_NORMAL(grass_normals)

branch = viz.TRANSFORM(branch)
branch_depth = viz.TRANSFORM_DEPTH(branch_depth)
branch_normals = viz.TRANSFORM_NORMAL(branch_normals)

#Constructing the main image input to the format of the NN
multimodal_image_road = torch.cat((road, road_depth, road_normals)).float()
multimodal_image_road = torch.unsqueeze(multimodal_image_road, dim=0)
multimodal_image_road = multimodal_image_road.to(viz.DEVICE)

#Constructing the main image input to the format of the NN
multimodal_image_grass = torch.cat((grass, grass_depth, grass_normals)).float()
multimodal_image_grass = torch.unsqueeze(multimodal_image_grass, dim=0)
multimodal_image_grass = multimodal_image_grass.to(viz.DEVICE)

#Constructing the main image input to the format of the NN
multimodal_image_branch = torch.cat((branch, branch_depth, branch_normals)).float()
multimodal_image_branch = torch.unsqueeze(multimodal_image_branch, dim=0)
multimodal_image_branch = multimodal_image_branch.to(viz.DEVICE)

with torch.no_grad() :
    for i in range(velocities.shape[0]) :

        #Computing the fixated velocity
        #TODO find a way to take a variable input, or an imput of more than one velocity
        #to compute more costmaps and avoid the velocity dependance
        velocity = torch.tensor([velocities[i]]).type(torch.float32).to(viz.DEVICE)
        velocity.unsqueeze_(1)

        output_road = model(multimodal_image_road, velocity)
        output_grass = model(multimodal_image_grass, velocity)
        output_branch = model(multimodal_image_branch, velocity)

        # Case Classification
        softmax = nn.Softmax(dim=1)
        output_road = softmax(output_road)
        output_road = output_road.cpu()[0]
        probs_road = output_road.numpy()
        cost_road = np.dot(probs_road, np.transpose(midpoints))

        output_grass = softmax(output_grass)
        output_grass = output_grass.cpu()[0]
        probs_grass = output_grass.numpy()
        cost_grass = np.dot(probs_grass, np.transpose(midpoints))

        output_branch = softmax(output_branch)
        output_branch = output_branch.cpu()[0]
        probs_branch = output_branch.numpy()
        cost_branch = np.dot(probs_branch, np.transpose(midpoints))

        score_road[i] = cost_road
        score_grass[i] = cost_grass
        score_branch[i] = cost_branch

        print(velocities[i], score_road[i], score_grass[i], score_branch[i])

plt.scatter(velocities,
            score_road
            )

plt.scatter(velocities,
            score_grass
            )

plt.scatter(velocities,
            score_branch
            )

plt.xlabel("Velocity [m/s]")
plt.ylabel("Traversal cost")

df = pd.read_csv(dataset_dir / "/traversal_costs.csv")
df_plot = df.plot(x=['linear_velocity'],y=['traversal_cost'],kind="scatter")

plt.show()