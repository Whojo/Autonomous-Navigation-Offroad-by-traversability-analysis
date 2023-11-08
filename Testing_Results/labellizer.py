import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

from params import visualparams as viz
from params import PROJECT_PATH
from utils.grid import get_grid_lists


def enter_costs(img, rectangle_list):
    """
    The main function of this programs, take a list of coordinates and the input image
    Put them in the NN and compute the cost for each crop
    Then reconstituate a costmap of costs
    Args:
        img : RGB input of the robot
        rectangle_list : list of the rectangle coordinates indicating where to crop according to the costmap's projection on the image

    Returns:
        Costmap : A numpy array of X*Y dimension with the costs
    """
    costmap = np.zeros((viz.Y, viz.X))
    cv2.imshow("Img", cv2.resize(img, (960, 540)))

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

            cv2.imshow(f"{x}, {y} :", crop)
            cv2.waitKey(1000)
            while True:
                try:
                    print("Enter the cost :")
                    cost = float(input())
                except ValueError:
                    print("Error, try again")
                    continue
                break
            cv2.destroyWindow(f"{x}, {y} :")

            # Filling the output array (the numeric costmap)
            costmap[y, x] = cost

    cv2.destroyWindow("Img")
    return costmap


def display(img, costmap, rectangle_list, grid_list, max_cost, min_cost):
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
    imgviz = img.copy()
    costmapviz = np.zeros((viz.Y, viz.X, 3), np.uint8)

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
            # If the cell is not empty because some cost has been generated
            if costmap[y, x] != 0:
                # Normalizing the content
                value = np.uint8(
                    ((costmap[y, x] - min_cost) / (max_cost - min_cost)) * 255
                )
                costmapviz[y, x] = (value, value, value)
            else:
                # If nothing we leave the image black
                costmapviz[y, x] = (0, 0, 0)
    # Applying the color gradient
    costmapviz = cv2.applyColorMap(src=costmapviz, colormap=cv2.COLORMAP_JET)

    # Displaying the results
    cv2.imshow("Result", imgviz)
    cv2.imshow(
        "Costmap",
        cv2.resize(cv2.flip(costmapviz, 0), (viz.X * 20, viz.Y * 20)),
    )
    cv2.waitKey(0)


if __name__ == "__main__":
    directory = PROJECT_PATH / "bagfiles/images_extracted/"
    print(directory.resolve())

    # Matches any file that ends with a number and .png
    # (i.e. only rgb images, not depth or normals)
    files = list(directory.glob("[!nd].png"))
    rectangle_list, grid_list = get_grid_lists()

    # Manually enters all costmaps
    # for i, file in enumerate(tqdm(files)):
    #     img = Image.open(str(file))
    #     costmap = enter_costs(img, rectangle_list)
    #     np.save(directory / f"costmaps{int(file.stem)}", costmap)

    # Displays all costmaps
    for file in files:
        img = cv2.imread(str(file))
        costmap = np.load(directory / f"costmaps{int(file.stem)}.npy")
        display(
            img,
            costmap,
            rectangle_list,
            grid_list,
            np.max(costmap),
            np.min(costmap),
        )
