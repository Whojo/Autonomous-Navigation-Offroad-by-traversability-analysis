import numpy as np

import utilities.frames as frames
import params.visualparams as viz
from data_preparation.create_dataset import get_patch_dimension


def _get_corners(x, y):
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


def _correct_points(points_image, width, height):
    """
    Move points that are outside the image to the edge of the image.

    Args:
        points_image (ndarray (n, 2)): Points coordinates in the image plan
    Returns:
        points_image but the points outside the image are now on the edge of the image
    """
    result = np.copy(points_image)
    result[:, 0] = np.clip(result[:, 0], 0, width)
    result[:, 1] = np.clip(result[:, 1], 0, height)

    return result


def get_grid_lists():
    """
    Setup the list of rectangles and lines that will be later used of inference
    Args:
        Nothing : O_o
    Returns:
        rectangle_list : a list of X*Y coordinates of the rectangle position indicating where to crop
        grid_list : a list of X*Y coordinates to indicate the visual projection of the costmap on the display
    """

    rectangle_list = np.full((viz.Y, viz.X), None)
    grid_list = np.zeros((viz.Y, viz.X, 4, 2), np.int32)

    if viz.X % 2 == 1:
        offset = viz.X // 2 + 0.5
    else:
        offset = viz.X // 2

    for x in range(viz.X):
        for y in range(viz.Y):
            # Get the list of coordinates of the corners in the costmap frame
            points_costmap = _get_corners(x, y)
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
            intern_points = _correct_points(
                points_image, viz.IMAGE_W, viz.IMAGE_H
            )
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
            # KEEP THE COORDINATES TO None, this is the way we're going to check later for the pertinence of the coordinates.
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

                if (
                    patch.min_x < 0
                    or patch.min_y < 0
                    or patch.max_x > viz.IMAGE_W
                    or patch.max_y > viz.IMAGE_H
                ):
                    continue

                rectangle_list[y, x] = patch

    return rectangle_list, grid_list
