# Python librairies
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import params.visualparams as viz
from src.utils.grid import get_grid_lists
from src.models_development.multimodal_velocity_regression_alt.dataset import (
    DEFAULT_MULTIMODAL_TRANSFORM,
)
from src.models_development.multimodal_velocity_regression_alt.model import (
    ResNet18Velocity_Regression_Alt,
)
from params import PROJECT_PATH


model = ResNet18Velocity_Regression_Alt()
WEIGHTS = (
    PROJECT_PATH
    / "src/models_development/multimodal_velocity_regression_alt/logs/_post_hp_tuning/network.params"
)
model.load_state_dict(torch.load(WEIGHTS))
model.eval().to(viz.DEVICE)

midpoints = viz.MIDPOINTS
VELOCITY = 1.0


def get_model_input(
    crop: np.array,
    depth_crop: np.array,
    normals_crop: np.array,
    velocity: float,
):
    crop = transforms.ToTensor()(crop)
    depth_crop = transforms.ToTensor()(depth_crop)
    normals_crop = transforms.ToTensor()(normals_crop)

    multimodal_image = torch.cat((crop, depth_crop, normals_crop)).float()
    multimodal_image = DEFAULT_MULTIMODAL_TRANSFORM(multimodal_image)
    multimodal_image = torch.unsqueeze(multimodal_image, 0)
    multimodal_image = multimodal_image.to(viz.DEVICE)

    velocity = torch.tensor([VELOCITY]).type(torch.float32).to(viz.DEVICE)
    velocity.unsqueeze_(1)

    return multimodal_image, velocity


def predict_costs(img, img_depth, img_normals, rectangle_list, model):
    """
    The main function of this programs, take a list of coordinates and the input image
    Put them in the NN and compute the cost for each crop
    Then reconstituate a costmap of costs
    Args:
        img : RGB input of the robot
        img_depth : depth image of the robot
        img_normals : RGB representation of the normals computed from the depth image
        rectangle_list : list of the rectangle coordinates indicating where to crop according to the costmap's projection on the image
        model : the NN

    Returns:
        Costmap : A numpy array of X*Y dimension with the costs
    """
    costmap = np.zeros((viz.Y, viz.X))

    with torch.no_grad():
        for x in range(viz.X):
            for y in range(viz.Y):
                patch = rectangle_list[y, x]
                if patch is None:
                    continue

                crop_patch = (
                    patch.min_x,
                    patch.min_y,
                    patch.max_x,
                    patch.max_y,
                )
                crop = img.crop(crop_patch)
                depth_crop = img_depth.crop(crop_patch)
                normals_crop = img_normals.crop(crop_patch)

                input = get_model_input(
                    crop, depth_crop, normals_crop, VELOCITY
                )

                output = model(*input)

                if viz.REGRESSION:
                    cost = output.cpu().item()
                else:
                    softmax = nn.Softmax(dim=1)
                    output = softmax(output)
                    output = output.cpu().item()
                    probs = output.numpy()
                    cost = np.dot(probs, np.transpose(midpoints))

                costmap[y, x] = cost

    return costmap


def cost_to_color(value, min, max):
    """
    A function that normalize a cost between 0 and 255
    Args :
        value : the value to normalize
        min : the minimum value of the range
        max : the maximum value of the range

    Returns :
        The normalized value
    """
    return (value - min) / (max - min) * 255


def plt_display(img: np.array, costmap: np.array, save_path: str = None):
    _, (ax1, ax2) = plt.subplots(1, 2)
    rectangle_list, grid_list = get_grid_lists()

    points_list = grid_list[rectangle_list != None]
    for points_image in points_list:
        centroid = np.mean(points_image, axis=0)
        cv2.circle(
            img,
            tuple(np.int32(centroid)),
            radius=4,
            color=(0, 0, 255),
            thickness=-1,
        )

        # Display the frontiers of the cell
        points_image_reshape = points_image.reshape((-1, 1, 2))
        cv2.polylines(
            img,
            np.int32([points_image_reshape]),
            True,
            (0, 255, 0),
            2,
        )

    ax1.imshow(img)
    ax1.axis("off")

    colormap = ax2.imshow(costmap[::-1, :], cmap="hot_r", vmin=2, vmax=6)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(colormap, ax=ax2, shrink=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def display(
    img,
    costmap,
    costmap_by_hand,
    rectangle_list,
    grid_list,
    max_cost,
    min_cost,
):
    """
    A function that displays what's currently computed
    Args :
        img : the base image
        costmap : the numerical costmap
        rectangle_list : the list of the coordinates of the cropping rectangles for the NN input
        gris_list : the list of the coordinates of the projected costmap's cells
        max_cost, min_cost : the max and min cost for the color gradient

    Returns :
        Displays a bunch of windows with funny colors on it, but nothing else.
    """
    # Buffers initialization
    imgviz = img.copy()
    costmapviz = np.zeros((viz.Y, viz.X, 3), np.uint8)
    costmapviz_hand = np.zeros((viz.Y, viz.X, 3), np.uint8)
    costmapviz_diff = np.zeros((viz.Y, viz.X, 3), np.uint8)

    costmap_diff = np.abs(costmap - costmap_by_hand)

    # For each costmap element
    for x in range(viz.X):
        for y in range(viz.Y):
            patch = rectangle_list[y, x]
            if patch is None:
                continue

            # Display the center of the cell
            points_image = grid_list[y, x]
            centroid = np.mean(points_image, axis=0)
            cv2.circle(
                imgviz,
                tuple(np.int32(centroid)),
                radius=4,
                color=(255, 0, 0),
                thickness=-1,
            )

            # Displaying the rectangle
            cv2.rectangle(
                imgviz,
                (patch.min_x, patch.min_y),
                (patch.max_x, patch.max_y),
                (255, 0, 0),
                1,
            )
            # Display the frontiers of the cell
            points_image_reshape = points_image.reshape((-1, 1, 2))
            cv2.polylines(
                imgviz,
                np.int32([points_image_reshape]),
                True,
                (0, 255, 255),
            )

    # Building cell per cell and array that will become our costmap visualization
    for x in range(viz.X):
        for y in range(viz.Y):
            costmapviz[y, x] = (0, 0, 0)
            if costmap[y, x] != 0:
                value = cost_to_color(costmap[y, x], min_cost, max_cost)
                costmapviz[y, x] = (value, value, value)

            costmapviz_hand[y, x] = (0, 0, 0)
            if costmap_by_hand[y, x] != 0:
                value = cost_to_color(
                    costmap_by_hand[y, x], min_cost, max_cost
                )
                costmapviz_hand[y, x] = (value, value, value)

            costmapviz_diff[y, x] = (0, 0, 0)
            if costmap_diff[y, x] != 0:
                value = cost_to_color(costmap_diff[y, x], min_cost, max_cost)
                costmapviz_diff[y, x] = (value, value, value)

    # Applying the color gradient
    costmapviz = cv2.applyColorMap(src=costmapviz, colormap=cv2.COLORMAP_JET)
    costmapviz_hand = cv2.applyColorMap(
        src=costmapviz_hand, colormap=cv2.COLORMAP_JET
    )
    costmapviz_diff = cv2.applyColorMap(
        src=costmapviz_diff, colormap=cv2.COLORMAP_JET
    )

    # Displaying the results
    imgviz = cv2.resize(imgviz, (viz.IMAGE_W // 2, viz.IMAGE_H // 2))
    costmapviz = cv2.resize(
        cv2.flip(costmapviz, 0), (viz.IMAGE_W // 2, viz.IMAGE_H // 2)
    )
    result = np.vstack((imgviz, costmapviz))

    costmapviz_hand = cv2.resize(
        cv2.flip(costmapviz_hand, 0), (viz.IMAGE_W // 2, viz.IMAGE_H // 2)
    )
    costmapviz_diff = cv2.resize(
        cv2.flip(costmapviz_diff, 0), (viz.IMAGE_W // 2, viz.IMAGE_H // 2)
    )

    result_bis = np.vstack((costmapviz_hand, costmapviz_diff))

    result = np.hstack((result, result_bis))
    result = cv2.resize(result, (1792, 1008))

    cv2.imshow("Result", result)
    # writer.write(result)
    cv2.waitKey(0)


if __name__ == "__main__":
    directory = PROJECT_PATH / "bagfiles/images_extracted/from_terrain_samples"
    # directory = PROJECT_PATH / "bagfiles/images_extracted"

    # Matches any file that ends with a number and .png
    # (i.e. only rgb images, not depth or normals)
    files = list(directory.glob("*[!dn].png"))

    rectangle_list, grid_list = get_grid_lists()

    # writer = cv2.VideoWriter(str(PROJECT_PATH / "Testing_Results/output.avi"), cv2.VideoWriter_fourcc(*'XVID'), 0.5, (1792,1008))
    for file in files:
        depth_name = directory / (file.stem + "_d.png")
        normal_name = directory / (file.stem + "_n.png")

        img = Image.open(str(file))
        img_depth = Image.open(str(depth_name))
        img_normal = Image.open(str(normal_name))
        # costmap_by_hand = np.load(directory / f"costmaps{int(file.stem)}.npy")

        costmap = predict_costs(
            img, img_depth, img_normal, rectangle_list, model
        )
        costmap_by_hand = np.zeros_like(costmap)

        max_cost = np.max([costmap, costmap_by_hand])
        min_cost = np.min([costmap, costmap_by_hand])

        # plt_display(np.array(img), costmap)

        img = cv2.imread(str(file))
        display(
            img,
            costmap,
            costmap_by_hand,
            rectangle_list,
            grid_list,
            max_cost,
            min_cost,
        )

    # writer.release()
