# Python librairies
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Importing custom made parameters
import utilities.frames as frames
import params.visualparams as viz

from src.data_preparation.create_dataset import get_patch_dimension
from src.models_development.multimodal_velocity_regression_alt.dataset import (
    DEFAULT_MULTIMODAL_TRANSFORM,
)
from src.models_development.multimodal_velocity_regression_alt.model import (
    ResNet18Velocity_Regression_Alt,
)
from params import PROJECT_PATH


IMAGE_W = 1920
IMAGE_H = 1080

model = ResNet18Velocity_Regression_Alt()
WEIGHTS = (
    PROJECT_PATH
    / "src/models_development/multimodal_velocity_regression_alt/logs/_post_hp_tuning/network.params"
)
model.load_state_dict(torch.load(WEIGHTS))
model.eval().to(viz.DEVICE)

midpoints = viz.MIDPOINTS
VELOCITY = 1.0


def get_corners(x, y):
    """
    Function that gives the corners of a cell in the costmap

    args :
        x , y = coordinates of the cell with the altitude set to 0
    returns :
        points = a list a 4 points (x,y)
    """

    points = np.array(
        [[x, y, 0], [x + 1, y, 0], [x + 1, y + 1, 0], [x, y + 1, 0]]
    )

    return points


def correct_points(points_image):
    """Remove the points which are outside the image
    Args:
        points_image (ndarray (n, 2)): Points coordinates in the image plan
    Returns:
        points_image but the points outside the image are now on the edge of the image
    """
    # Keep only points which are on the image
    result = np.copy(points_image)
    for i in range(len(result)):
        if result[i, 0] < 0:
            result[i, 0] = 0
        if result[i, 0] > IMAGE_W:
            result[i, 0] = IMAGE_W
        if result[i, 1] < 0:
            result[i, 1] = 0
        if result[i, 1] > IMAGE_H:
            result[i, 1] = IMAGE_H

    return result


def get_lists():
    """
    Setup the list of rectangles and lines that will be later used of inference
    Args:
        Nothing : O_o
    Returns:
        rectangle_list : a list of X*Y coordinates of the rectangle position indicating where to crop
        grid_list : a list of X*Y coordinates to indicate the visual projection of the costmap on the display
    """

    rectangle_list = np.zeros((viz.Y, viz.X, 2, 2), np.int32)
    grid_list = np.zeros((viz.Y, viz.X, 4, 2), np.int32)

    if viz.X % 2 == 1:
        offset = viz.X // 2 + 0.5
    else:
        offset = viz.X // 2

    for x in range(viz.X):
        for y in range(viz.Y):
            # Get the list of coordinates of the corners in the costmap frame
            points_costmap = get_corners(x, y)
            points_robot = points_costmap - np.array([(offset, 0, 0)])

            # Strange computation because the robot frame has the x axis toward the back of the robot
            # and the y axis to ward the left.
            points_robot = points_robot[:, [1, 0, 2]]
            points_robot = points_robot * np.array([1, -1, 1])

            # Switching from the costmap coordinates to the world coordinates using the resolution of the costmap.
            points_robot = points_robot * viz.RESOLUTION

            # Compute the points coordinates in the camera frame
            points_camera = frames.apply_rigid_motion(
                points_robot, viz.CAM_TO_ROBOT
            )

            # Compute the points coordinates in the image plan
            points_image = frames.camera_frame_to_image(points_camera, viz.K)
            grid_list[y, x] = points_image

            # Get the Area of the cell that is in the image
            intern_points = correct_points(points_image)
            intern_area = (
                (intern_points[0, 1] - intern_points[2, 1])
                * (
                    (intern_points[1, 0] - intern_points[0, 0])
                    + (intern_points[2, 0] - intern_points[3, 0])
                )
                / 2
            )

            # Get the Area of the cell in total
            area = (
                (points_image[0, 1] - points_image[2, 1])
                * (
                    (points_image[1, 0] - points_image[0, 0])
                    + (points_image[2, 0] - points_image[3, 0])
                )
                / 2
            )

            # If the area in squared pixels of the costmap cell is big enough, then relevant data can be extracted
            #
            # IMPORTANT : If there's nothing to extract because the rectangle is too small on the image,
            # KEEP THE COORDINATES TO ZERO, this is the way we're going to check later for the pertinence of the coordinates.
            if (
                intern_area / area >= viz.THRESHOLD_INTERSECT
                and area >= viz.THRESHOLD_AREA
            ):
                # We project the footprint of the robot as if it was centered on the cell
                # We then get the smallest bounding rectangle of the footprint to keep the terrain on wich
                # it would step over if it was on the costmap's cell

                # Getting the footprint coordinates
                centroid = np.mean(points_robot, axis=0)
                point_tl = centroid + [0, 0.5 * viz.L_ROBOT, 0]
                point_br = centroid - [0, 0.5 * viz.L_ROBOT, 0]

                # Projecting the footprint in the image frame
                point_tl = frames.apply_rigid_motion(
                    point_tl, viz.CAM_TO_ROBOT
                )
                point_br = frames.apply_rigid_motion(
                    point_br, viz.CAM_TO_ROBOT
                )
                point_tl = frames.camera_frame_to_image(point_tl, viz.K)
                point_br = frames.camera_frame_to_image(point_br, viz.K)

                # Extracting the parameters for the rectangle
                point_tl = point_tl[0]
                point_br = point_br[0]

                all_points = np.vstack((point_br, point_tl))
                patch = get_patch_dimension(all_points)

                crop_width = patch.max_x - patch.min_x
                crop_height = patch.max_y - patch.min_y

                tl_x = np.clip(patch.min_x, 0, IMAGE_W - crop_width)
                tl_y = np.clip(patch.min_y, 0, IMAGE_H - crop_height)

                rect_tl = np.int32([tl_x, tl_y])
                rect_br = rect_tl + [crop_width, crop_height]

                rectangle_list[y, x] = np.array([rect_tl, rect_br])

    return rectangle_list, grid_list


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
                rectangle = rectangle_list[y, x]
                if np.all(rectangle == 0):
                    continue

                crop_patch = (
                    rectangle[0, 0],
                    rectangle[0, 1],
                    rectangle[1, 0],
                    rectangle[1, 1],
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
                    output = output.cpu()[0]
                    probs = output.numpy()
                    cost = np.dot(probs, np.transpose(midpoints)).item()

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
            # Getting the rectangle coordinate
            rectangle = rectangle_list[y, x]
            # Checking if we estimated beforehand that the rectangle might have something interesting to display
            if not np.array_equal(rectangle, np.zeros(rectangle.shape)):
                # If there's something we get the coordinates of the cell and the rectangle
                rect_tl, rect_br = rectangle[0], rectangle[1]
                points_image = grid_list[y, x]

                # Display the center of the cell
                centroid = np.mean(points_image, axis=0)
                cv2.circle(
                    imgviz,
                    tuple(np.int32(centroid)),
                    radius=4,
                    color=(255, 0, 0),
                    thickness=-1,
                )

                # Displaying the rectangle
                rect_tl = tuple(rect_tl)
                rect_br = tuple(rect_br)
                cv2.rectangle(imgviz, rect_tl, rect_br, (255, 0, 0), 1)
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
    imgviz = cv2.resize(imgviz, (IMAGE_W // 2, IMAGE_H // 2))
    costmapviz = cv2.resize(
        cv2.flip(costmapviz, 0), (IMAGE_W // 2, IMAGE_H // 2)
    )
    result = np.vstack((imgviz, costmapviz))

    costmapviz_hand = cv2.resize(
        cv2.flip(costmapviz_hand, 0), (IMAGE_W // 2, IMAGE_H // 2)
    )
    costmapviz_diff = cv2.resize(
        cv2.flip(costmapviz_diff, 0), (IMAGE_W // 2, IMAGE_H // 2)
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

    rectangle_list, grid_list = get_lists()

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
