"""
Script to build a dataset for terrain traversability estimation from images
(rgb, depth, normals) and the robot linear velocity. A dataset is a folder
with the following structure:

dataset_{name}/
├── images/
│   ├── 00000.png
│   ├── 00000d.tiff
│   ├── 00000n.tiff
│   ├── 00001.png
│   ├── 00001d.tiff
│   ├── 00001n.tiff
│   └── ...
├── images_test/
├── images_train/
├── traversal_costs.csv
├── traversal_costs_test.csv
├── traversal_costs_train.csv
└── bins_midpoints.csv

where:
- xxxxx.png, xxxxxd.tiff and xxxxxn.tiff are the rgb, depth and the normals
images respectively
- images_train/ and images_test/ are the training and testing sets of images
- traversal_costs.csv is a csv file containing the traversal costs associated
with the images, the traversability labels (obtained from the continuous
traversal cost after digitization) and the linear velocities of the robot
- traversal_costs_train.csv and traversal_costs_test.csv contain the same
information but for the training and testing sets respectively
"""


# Python libraries
import numpy as np
import os
import csv
import sys
from tqdm import tqdm
import cv2
from PIL import Image
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# ROS Python libraries
import cv_bridge
import rosbag
import rospy
import tf.transformations

from collections import namedtuple
from enum import Enum
from functools import lru_cache

# Custom modules and packages
import utilities.frames as frames
from depth.utils import Depth
import traversalcost.utils
import traversalcost.traversal_cost
import params.robot
import params.dataset
import params.traversal_cost
import params.learning

from params import PROJECT_PATH


plt.rcParams["text.usetex"] = True  # Render Matplotlib text with Tex


VelocityType = Enum("VelocityType", ["ODOMETRIC", "MANUAL"])
CostType = Enum("CostType", ["SIAMESE", "FORMULA"])
RectangleDim = namedtuple("RectangleDim", ["min_x", "max_x", "min_y", "max_y"])


def _get_recursive_bag_files(files: list) -> list:
    """
    Recursively searches through a list of files and directories to find all bag files.

    Args:
        files (list): A list of file paths to search through.

    Returns:
        list: A list of all bag files found in the given files and directories.
    """
    bag_files = []
    for file in files:
        # Check if the file is a bag file
        if os.path.isfile(file) and file.endswith(".bag"):
            bag_files.append(file)

        # If the path links to a directory, go through the files inside it
        elif os.path.isdir(file):
            bag_files.extend(
                [file + f for f in os.listdir(file) if f.endswith(".bag")]
            )

    return bag_files


def is_bag_healthy(bag: str) -> bool:
    """Check if a bag file is healthy

    Args:
        bag (str): Path to the bag file

    Returns:
        bool: True if the bag file is healthy, False otherwise
    """
    # Get the bag file duration
    duration = bag.get_end_time() - bag.get_start_time()  # [seconds]

    for topic, frequency in [
        (params.robot.IMAGE_TOPIC, params.robot.CAMERA_SAMPLE_RATE),
        (params.robot.DEPTH_TOPIC, params.robot.DEPTH_SAMPLE_RATE),
        (params.robot.ODOM_TOPIC, params.robot.ODOM_SAMPLE_RATE),
        (params.robot.IMU_TOPIC, params.robot.IMU_SAMPLE_RATE),
    ]:
        # Get the number of messages in the bag file
        nb_messages = bag.get_message_count(topic)

        # Check if the number of messages is consistent with the frequency
        if (
            np.abs(nb_messages - frequency * duration) / (frequency * duration)
            > params.dataset.NB_MESSAGES_THR
        ):
            return False

    return True


def _get_msg_depth(bag: rosbag.Bag, t_image: rospy.Time) -> np.array:
    depth_list = list(
        bag.read_messages(
            topics=[params.robot.DEPTH_TOPIC],
            start_time=t_image - rospy.Duration(params.dataset.TIME_DELTA),
            end_time=t_image + rospy.Duration(params.dataset.TIME_DELTA),
        )
    )

    min_t = params.dataset.TIME_DELTA
    msg_depth = None

    # Find the depth image whose timestamp is closest to that
    # of the rgb image
    for _, msg_depth_i, t_depth in depth_list:
        new_t = np.abs(t_depth.to_sec() - t_image.to_sec())
        if new_t < min_t:
            min_t = new_t
            msg_depth = msg_depth_i

    return msg_depth


def _get_sequence_extremum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing the start and end indexes of each sequence
    of indexes from the input DataFrame, along with their corresponding
    velocity values. A sequence is defined as a set of consecutive rows
    with the same velocity value and whose "end_index" match the next
    "start_index".

    Note that in the data collection script which produces the labels.csv file,
    the robot is alternatively moving forward and backward. Therefore, the
    velocity values are alternatively positive and negative.

    Args:
    - df: input DataFrame containing start_index, end_index, and
        linear_velocity columns

    Returns:
    - result_df: DataFrame containing start_index, end_index, and
        linear_velocity columns
    """
    start_index = df["start_index"] != df["end_index"].shift(1)
    end_index = df["end_index"] != df["start_index"].shift(-1)

    result_df = pd.DataFrame(
        df.loc[start_index, "start_index"].reset_index(drop=True),
        columns=["start_index"],
    )

    result_df["end_index"] = df.loc[end_index, "end_index"].values
    result_df["linear_velocity"] = df.loc[
        start_index, "linear_velocity"
    ].values
    result_df.loc[np.arange(1, len(result_df), 2), "linear_velocity"] *= -1

    return result_df


@lru_cache
def _get_velocity_df(file: str) -> pd.DataFrame:
    """
    Get the manually annotated velocities from params.dataset.LABELS_PATH
    for a given rosbag `file`, and computes corresponding timestamp.

    Args:
        file (str): The name of the rosbag file from "Terrain_Samples".

    Returns:
        pd.DataFrame: A pandas DataFrame containing the velocity
            manually annotated data for the given rosbag `file` and their
            start and end timestamps.
    """
    velocity_df = pd.read_csv(params.dataset.LABELS_PATH)
    velocity_df = velocity_df[velocity_df["file"] == file].reset_index()
    velocity_df = _get_sequence_extremum(velocity_df)

    df_idx = 0
    bag = rosbag.Bag(PROJECT_PATH / file)
    for imu_idx, (_, _, t) in enumerate(
        bag.read_messages(topics=[params.robot.IMU_TOPIC])
    ):
        if imu_idx == velocity_df.loc[df_idx, "start_index"]:
            velocity_df.loc[df_idx, "start_timestamp"] = t

        if imu_idx == velocity_df.loc[df_idx, "end_index"]:
            velocity_df.loc[df_idx, "end_timestamp"] = t
            df_idx += 1

        if df_idx >= len(velocity_df):
            break

    return velocity_df


def _get_velocity_from_timestamp(file: str, timestamp: rospy.Time) -> float:
    """
    Returns the velocity at a given timestamp from a velocity manually
    annotated file.

    Args:
        file (str): The path to the velocity file.
        timestamp (rospy.Time): The timestamp for which to get the velocity.

    Returns:
        float: The velocity at the given timestamp.

    Raises:
        ValueError: If there are multiple velocities for the same timestamp.
    """
    velocity_df = _get_velocity_df(file)
    velocity = velocity_df[
        (velocity_df["start_timestamp"] <= timestamp)
        & (timestamp <= velocity_df["end_timestamp"])
    ]["linear_velocity"]

    if len(velocity) == 0:
        return 0

    if len(velocity) > 1:
        raise ValueError("Multiple velocities for the same timestamp")

    return velocity.values[0]


def is_inside_image(image: np.ndarray, point: np.ndarray) -> bool:
    """Check if a point is inside an image

    Args:
        image (np.ndarray): The image
        point (np.ndarray): The point

    Returns:
        bool: True if the point is inside the image, False otherwise
    """
    x, y = point
    return (
        (x >= 0) and (x < image.shape[1]) and (y >= 0) and (y < image.shape[0])
    )


def _get_outter_rectangle(all_points: np.array) -> RectangleDim:
    """
    Given an array of points, returns the minimum bounding rectangle that contains all the points.

    Args:
    - all_points: np.array, shape (n, 2), containing the coordinates of the points in the image.

    Returns:
    - RectangleDim: a named tuple containing the dimensions of the minimum bounding rectangle.
    """
    max_y = np.max(all_points[:, 1])
    min_y = np.min(all_points[:, 1])
    min_x = np.min([all_points[:, 0]])
    max_x = np.max([all_points[:, 0]])

    return RectangleDim(min_x, max_x, min_y, max_y)


def _to_valid_patch_dimension(rec: RectangleDim) -> RectangleDim:
    """
    Converts a rectangle to a valid patch dimension by adjusting its width and height
    based on the minimum width and rectangle ratio specified in the dataset parameters.

    Args:
        rec (RectangleDim): The rectangle to be converted.

    Returns:
        RectangleDim: The converted rectangle with valid patch dimensions.
    """

    min_x, max_x, min_y, max_y = rec

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    width = max(max_x - min_x, params.dataset.MIN_WIDTH)
    height = width / params.dataset.RECTANGLE_RATIO

    min_x = int(center_x - (width / 2))
    max_x = int(center_x + (width / 2))
    min_y = int(center_y - (height / 2))
    max_y = int(center_y + (height / 2))

    return RectangleDim(min_x, max_x, min_y, max_y)


def get_patch_dimension(all_points: np.array) -> RectangleDim:
    """
    Return a new minimal rectangle patch that
    1. includes all the points
    2. with a minimal width of params.dataset.MIN_WIDTH and a width / height
    ratio of params.dataset.RECTANGLE_RATIO The center of the outputted rectangle
    is the same as the inputted one

    Args:
        all_points: np.array of shape (N, 2) of points in the image

    Returns:
        RectangleDim: namedtuple of (min_x, max_x, min_y, max_y) of the new rectangle
    """
    rec = _get_outter_rectangle(all_points)
    return _to_valid_patch_dimension(rec)


def raw_imu_to_features(
    raw_imu_line: dict,
) -> np.array:
    return traversalcost.utils.get_features(
        raw_imu_line["roll"],
        raw_imu_line["pitch"],
        raw_imu_line["vertical_acceleration"],
        params.dataset.FEATURES,
    )


def raw_imu_to_formula_traversal_cost(
    raw_imu_line: dict,
) -> float:
    """
    Calculates the traversal cost on a single IMU pathc as defined by
    Waibel et al. (2022). The normalisation between 0 and 1 from the original
    formula has been removed, as judged unnecessary and less handy to compute.
    """
    data = np.array(
        [
            raw_imu_line["roll"],
            raw_imu_line["pitch"],
            raw_imu_line["vertical_acceleration"],
        ]
    )
    data = np.abs(data)

    return np.mean(data)


class DatasetBuilder:
    """
    Class to build a terrain traversability dataset from ROS bag files
    """

    def __init__(self, name: str) -> None:
        """Constructor of the class

        Args:
            name (str): Name of the dataset
        """
        # Get the absolute path of the current directory
        directory = os.path.abspath(os.getcwd())

        # Set the name of the directory which will store the dataset
        self.dataset_directory = directory + "/datasets/dataset_" + name

        try:  # A new directory is created if it does not exist yet
            os.mkdir(self.dataset_directory)
            print(self.dataset_directory + " folder created\n")

        except OSError:  # Display a message if it already exists and quit
            print("Existing directory " + self.dataset_directory)
            print("Aborting to avoid overwriting data\n")
            sys.exit(1)  # Stop the execution of the script

        # Create a sub-directory to store images
        self.images_directory = self.dataset_directory + "/images"

        # Create a directory if it does not exist yet
        try:
            os.mkdir(self.images_directory)
            print(self.images_directory + " folder created\n")
        except OSError:
            pass

        # Create a csv file to store the traversal costs
        self.csv_name = self.dataset_directory + "/traversal_costs.csv"

    def write_images_and_compute_features(
        self,
        files: list,
        *,
        filter_static: bool = True,
        velocity_type: VelocityType = VelocityType.ODOMETRIC,
        filter_cohesion: bool = True,
        filter_coherence: bool = True,
    ) -> (np.array, np.array):
        """Write images and compute features from a list of bag files

        Args:
            files (list): List of bag files
            filter_static (bool, optional): If True, discard images when the
                robot is static. The definition of "static" depends on
                velocity_type. Default to True.
            velocity_type (VelocityType, optional): If
                VelocityType.ODOMETRIC, the velocity is computed based on
                odometry. If VelocityType.MANUAL, the velocity is extracted
                from manual annotated velocities from `labels.csv`. This mode is
                useful if the velocity data is corrupted, but we do have
                access to these (manual) annotation. Default to
                VelocityType.VELOCITY.
            filter_cohesion (bool, optional): If True, discard images that do
                not share the same sign of velocity. Default to True.
            filter_coherence (bool, optional): If True, discard images with a
                non-coherent velocity on the patch (i. e. extremum velocity are
                too far from the mean). Default to True.

        Returns:
            tuple: A tuple containing the raw IMU signals and the velocities.
        """
        velocities = []
        raw_imu = []
        source_bagfiles = []

        index_image = 0

        bag_files = _get_recursive_bag_files(files)
        for file in tqdm(bag_files):
            bag = rosbag.Bag(file)

            # Check if the bag file is healthy (i.e. if it contains all the
            # topics and if the number of messages is consistent with the
            # sampling rate)
            if not is_bag_healthy(bag):
                print("File " + file + " is incomplete. Skipping...")
                continue

            for _, msg_image, t_image in bag.read_messages(
                topics=[params.robot.IMAGE_TOPIC]
            ):
                msg_depth = _get_msg_depth(bag, t_image)
                if msg_depth is None:
                    continue

                # Get the first odometry message received after the image
                _, first_msg_odom, t_odom = next(
                    bag.read_messages(
                        topics=[params.robot.ODOM_TOPIC], start_time=t_image
                    )
                )

                # Collect images and IMU data only when the robot is moving
                # (to avoid collecting images of the same place)
                if (
                    filter_static
                    and velocity_type == VelocityType.ODOMETRIC
                    and first_msg_odom.twist.twist.linear.x
                    < params.dataset.LINEAR_VELOCITY_THR
                ):
                    continue

                if velocity_type == VelocityType.MANUAL:
                    velocity = _get_velocity_from_timestamp(file, t_odom)

                    if filter_static and velocity == 0:
                        continue

                bridge = cv_bridge.CvBridge()
                image = bridge.imgmsg_to_cv2(
                    msg_image, desired_encoding="passthrough"
                )
                depth_image = bridge.imgmsg_to_cv2(
                    msg_depth, desired_encoding="passthrough"
                )

                WORLD_TO_ROBOT = frames.pose_to_transform_matrix(
                    first_msg_odom.pose.pose
                )
                ROBOT_TO_WORLD = frames.inverse_transform_matrix(
                    WORLD_TO_ROBOT
                )

                # Define an array to store the previous front wheels
                # coordinates in the image
                points_image_old = None

                # Define a variable to store the previous timestamp
                t_odom_old = None

                # Define an array to store the previous robot position in the
                # world frame
                point_world_old = None

                x_velocity = []
                for _, msg_odom, t_odom in bag.read_messages(
                    topics=[params.robot.ODOM_TOPIC],
                    start_time=t_odom,
                    end_time=t_odom + rospy.Duration(params.dataset.T),
                ):
                    point_world = np.array(
                        [
                            [
                                msg_odom.pose.pose.position.x,
                                msg_odom.pose.pose.position.y,
                                msg_odom.pose.pose.position.z,
                            ]
                        ]
                    )
                    q = np.array(
                        [
                            msg_odom.pose.pose.orientation.x,
                            msg_odom.pose.pose.orientation.y,
                            msg_odom.pose.pose.orientation.z,
                            msg_odom.pose.pose.orientation.w,
                        ]
                    )

                    # Convert the quaternion into Euler angles
                    theta = tf.transformations.euler_from_quaternion(q)[2]

                    # Create arrays to store left and right front
                    # wheels positions
                    point_left_world = np.copy(point_world)
                    point_right_world = np.copy(point_world)

                    # Compute the distances between the wheels and
                    # the robot's origin
                    delta_X = params.robot.L * np.sin(theta) / 2
                    delta_Y = params.robot.L * np.cos(theta) / 2

                    # Compute the positions of the outer points of the two
                    # front wheels
                    point_left_world[:, 0] -= delta_X
                    point_left_world[:, 1] += delta_Y
                    point_right_world[:, 0] += delta_X
                    point_right_world[:, 1] -= delta_Y

                    # Gather front wheels outer points coordinates in
                    # a single array
                    points_world = np.concatenate(
                        [point_left_world, point_right_world]
                    )

                    # Compute the points coordinates in the robot frame
                    points_robot = frames.apply_rigid_motion(
                        points_world, ROBOT_TO_WORLD
                    )

                    # Compute the points coordinates in the camera frame
                    points_camera = frames.apply_rigid_motion(
                        points_robot, params.robot.CAM_TO_ROBOT
                    )

                    # Compute the points coordinates in the image plan
                    points_image = frames.camera_frame_to_image(
                        points_camera, params.robot.K
                    )

                    # Test if the two points are inside the image
                    if not is_inside_image(
                        image, points_image[0]
                    ) or not is_inside_image(image, points_image[1]):
                        if point_world_old is None:
                            continue
                        else:
                            # If the trajectory goes out of the image
                            break

                    # First point in the image
                    if point_world_old is None:
                        # image = dw.draw_points(image, points_image)

                        # Set the previous points to the current ones
                        point_world_old = point_world
                        points_image_old = points_image
                        t_odom_old = t_odom
                        continue

                    x_velocity.append(msg_odom.twist.twist.linear.x)

                    distance = np.linalg.norm(point_world - point_world_old)
                    if distance <= params.dataset.PATCH_DISTANCE:
                        continue

                    # image = dw.draw_points(image, points_image)

                    # Check if the velocity on the patch means something coherent : i.e. if the robot is not doing a round trip or other black magic.
                    x_velocity_array = np.array(x_velocity)
                    qup, qdown = np.percentile(x_velocity_array, [90, 10])
                    q_thresh = 0.08

                    coherence = (
                        qdown > np.mean(x_velocity) * (1.0 - q_thresh)
                    ) and (qup < np.mean(x_velocity) * (1.0 + q_thresh))
                    cohesion = (
                        np.all(x_velocity_array > 0)
                        if x_velocity_array[0] > 0
                        else np.all(x_velocity_array < 0)
                    )

                    if (filter_coherence and not coherence) or (
                        filter_cohesion and not cohesion
                    ):
                        x_velocity = []

                        # Update the old values
                        point_world_old = point_world
                        points_image_old = points_image
                        t_odom_old = t_odom

                        continue

                    # Compute the inclination of the patch in the image
                    delta_old = np.abs(
                        points_image_old[0] - points_image_old[1]
                    )
                    delta_current = np.abs(points_image[0] - points_image[1])

                    patch_angle = (
                        np.arctan(delta_old[1] / delta_old[0])
                        + np.arctan(delta_current[1] / delta_current[0])
                    ) / 2

                    # Discard rectangles that are too inclined
                    if patch_angle > params.dataset.PATCH_ANGLE_THR:
                        break

                    all_points = np.vstack((points_image_old, points_image))
                    patch = get_patch_dimension(all_points)

                    if not is_inside_image(
                        image, (patch.max_x, patch.max_y)
                    ) or not is_inside_image(
                        image, (patch.min_x, patch.max_y)
                    ):
                        break

                    if not is_inside_image(
                        image, (patch.max_x, patch.min_y)
                    ) or not is_inside_image(
                        image, (patch.min_x, patch.min_y)
                    ):
                        continue

                    # Draw a rectangle in the image to visualize the region
                    # of interest
                    # image = dw.draw_quadrilateral(
                    #     image,
                    #     np.array([[patch.min_x, patch.max_y],
                    #               [patch.min_x, patch.min_y],
                    #               [patch.max_x, patch.min_y],
                    #               [patch.max_x, patch.max_y]]),
                    #     color=(255, 0, 0))

                    image_to_save = image[
                        patch.min_y : patch.max_y, patch.min_x : patch.max_x
                    ]

                    # cv2.imshow("rgb", image_to_save)
                    # cv2.waitKey(0)

                    image_to_save = cv2.cvtColor(
                        image_to_save, cv2.COLOR_BGR2RGB
                    )

                    image_to_save = Image.fromarray(image_to_save)
                    image_name = f"{index_image:05d}.png"
                    image_to_save.save(
                        self.images_directory + "/" + image_name, "PNG"
                    )

                    # Extract the rectangular region of interest from
                    # the original depth image
                    depth_image_crop = depth_image[
                        patch.min_y : patch.max_y, patch.min_x : patch.max_x
                    ]

                    depth = Depth(
                        depth_image_crop.copy(), params.dataset.DEPTH_RANGE
                    )
                    depth.compute_normal(
                        K=params.robot.K,
                        bilateral_filter=params.dataset.BILATERAL_FILTER,
                        gradient_threshold=params.dataset.GRADIENT_THR,
                    )

                    # depth.display_depth()
                    # depth.display_normal()

                    depth_image_name = f"{index_image:05d}d.png"
                    depth_to_save = depth.get_depth(
                        fill=True,
                        default_depth=params.dataset.DEPTH_RANGE[0],
                        convert_range=True,
                    )

                    depth_to_save = Image.fromarray(depth_to_save)
                    depth_to_save.save(
                        self.images_directory + "/" + depth_image_name, "PNG"
                    )

                    normal_map_name = f"{index_image:05d}n.png"
                    normal_to_save = depth.get_normal(
                        fill=True,
                        default_normal=params.dataset.DEFAULT_NORMAL,
                        convert_range=True,
                    )

                    # Convert the image from BGR to RGB
                    normal_to_save = cv2.cvtColor(
                        normal_to_save, cv2.COLOR_BGR2RGB
                    )

                    normal_to_save = Image.fromarray(normal_to_save)
                    normal_to_save.save(
                        self.images_directory + "/" + normal_map_name, "PNG"
                    )

                    # Define lists to store IMU signals
                    roll_velocity_values = []
                    pitch_velocity_values = []
                    vertical_acceleration_values = []

                    # Read the IMU measurements within the dt second(s)
                    # interval
                    for _, msg_imu, _ in bag.read_messages(
                        topics=[params.robot.IMU_TOPIC],
                        start_time=t_odom_old,
                        end_time=t_odom,
                    ):
                        # Append angular velocities and vertical
                        # acceleration to the previously created lists
                        roll_velocity_values.append(msg_imu.angular_velocity.x)
                        pitch_velocity_values.append(
                            msg_imu.angular_velocity.y
                        )
                        vertical_acceleration_values.append(
                            msg_imu.linear_acceleration.z - 9.81
                        )

                    # Extract features from the IMU signals and fill the
                    # features array
                    raw_imu.append(
                        {
                            "roll": roll_velocity_values,
                            "pitch": pitch_velocity_values,
                            "vertical_acceleration": vertical_acceleration_values,
                        }
                    )

                    # Compute the mean velocity on the current patch
                    if velocity_type == VelocityType.ODOMETRIC:
                        velocity = np.mean(x_velocity)

                    velocities.append(velocity)
                    source_bagfiles.append(file)

                    # Increment the index of the current image
                    index_image += 1

                    # Reset the list of x velocities
                    x_velocity = []

                    # Update the old values
                    point_world_old = point_world
                    points_image_old = points_image
                    t_odom_old = t_odom

                    # cv2.imshow("Image", cv2.resize(image, (1280, 720)))
                    # cv2.waitKey()

            bag.close()

        return raw_imu, velocities, source_bagfiles

    def compute_siamese_traversal_costs(self, raw_imu: list) -> np.array:
        """
        Computes the traversal costs for the given features using a Siamese Network model.
        The costs are then discretized using K-means binning and the midpoints of the bins are saved in the dataset directory.
        If plot_velocity_distribution is True, a histogram of the traversal costs is plotted.

        Args:
            raw_imu (list): A list of dictionaries containing the raw IMU signals.
                dict_keys(['roll', 'pitch', 'vertical_acceleration'])

        Returns:
            np.array: Traversal costs.
        """
        features = np.array(list(map(raw_imu_to_features, raw_imu)))
        model = traversalcost.traversal_cost.SiameseNetwork(
            input_size=features.shape[1],
        )

        return traversalcost.traversal_cost.apply_model(
            features=features,
            model=model,
            params=params.dataset.SIAMESE_PARAMS,
            device=params.dataset.DEVICE,
        )

    def compute_formula_traversal_costs(self, raw_imu: list) -> np.array:
        """
        Calculates the traversal cost on a list as defined by Waibel et al. (2022).
        """
        costs = map(raw_imu_to_formula_traversal_cost, raw_imu)
        costs = np.array(list(costs))
        return costs.reshape(-1, 1)

    def discritize_traversal_costs(
        self,
        costs: np.array,
        *,
        plot_velocity_distribution: bool = False,
    ) -> np.array:
        """
        Discretizes the traversal costs using K-means binning.

        Args:
            costs (np.array): Array of traversal costs.
            plot_velocity_distribution (bool, optional): Whether to plot the velocity distribution. Defaults to False.

        Returns:
            np.array: Array of discretized traversal costs.
        """
        # Apply K-means binning
        discretizer = KBinsDiscretizer(
            n_bins=params.traversal_cost.NB_BINS,
            encode="ordinal",
            strategy=params.traversal_cost.BINNING_STRATEGY,
        )
        digitized_costs = np.int32(discretizer.fit_transform(costs))

        # Get the edges and midpoints of the bins
        bins_edges = discretizer.bin_edges_[0]
        bins_midpoints = (bins_edges[:-1] + bins_edges[1:]) / 2

        # Save the midpoints in the dataset directory
        np.save(self.dataset_directory + "/bins_midpoints.npy", bins_midpoints)

        if plot_velocity_distribution:
            plt.figure()
            plt.hist(costs, bins_edges, lw=1, ec="magenta", fc="blue")
            plt.title("Traversal cost binning")
            plt.xlabel("Traversal cost")
            plt.ylabel("Samples")
            plt.show()

        return digitized_costs

    def write_traversal_costs(
        self,
        raw_imu: list,
        velocities: list,
        source_bagfiles: list,
        *,
        cost_type: CostType = CostType.SIAMESE,
    ) -> None:
        """
        Write the traversal costs in a csv file.

        Args:
            raw_imu (list): A list of dictionaries containing the raw IMU signals.
                dict_keys(['roll', 'pitch', 'vertical_acceleration'])
            velocities (list): A list of velocities.
            source_bagfiles (list): A list of source bagfiles.
            cost_type (CostType): The type of traversal cost to compute.
                SIAMESE: compute the traversal cost using a Siamese Network model.
                FORMULA: compute the traversal cost using a manually defined formula
                    (mean of the absolute values of the roll, pitch angular velocities, and vertical acceleration)
        """
        file_costs = open(self.csv_name, "w")
        file_costs_writer = csv.writer(file_costs, delimiter=",")

        headers = [
            "image_id",
            "traversal_cost",
            "traversability_label",
            "linear_velocity",
            "source_bagfile",
        ]
        file_costs_writer.writerow(headers)

        if cost_type == CostType.SIAMESE:
            costs = self.compute_siamese_traversal_costs(raw_imu)
        elif cost_type == CostType.FORMULA:
            costs = self.compute_formula_traversal_costs(raw_imu)
        else:
            raise ValueError("Invalid cost type")

        labels = self.discritize_traversal_costs(costs)

        for i in range(costs.shape[0]):
            image_name = f"{i:05d}"

            cost = costs[i, 0]
            label = labels[i, 0]
            linear_velocity = velocities[i]
            source_bagfile = source_bagfiles[i]

            file_costs_writer.writerow(
                [str(image_name), cost, label, linear_velocity, source_bagfile]
            )

        file_costs.close()

    def create_train_test_splits(
        self, plot_velocity_distribution: bool = False
    ) -> None:
        """
        Splits the dataset randomly into training and testing sets, creates sub-directories to store train and test images,
        counts the number of samples per class, copies the images to the respective directories and stores the train and test
        splits in csv files.

        Args:
        - self: instance of the CreateDataset class
        - plot_velocity_distribution: boolean flag to plot the distribution of train and test sets

        Returns:
        - None
        """
        # Create a sub-directory to store train images
        train_directory = self.dataset_directory + "/images_train"
        os.mkdir(train_directory)
        print(train_directory + " folder created\n")

        # Create a sub-directory to store test images
        test_directory = self.dataset_directory + "/images_test"
        os.mkdir(test_directory)
        print(test_directory + " folder created\n")

        # Read the CSV file into a Pandas dataframe (read image_id values as
        # strings to keep leading zeros)
        dataframe = pd.read_csv(self.csv_name, converters={"image_id": str})

        # Split the dataset randomly into training and testing sets
        dataframe_train, dataframe_test = train_test_split(
            dataframe,
            train_size=params.learning.TRAIN_SIZE + params.learning.VAL_SIZE,
            stratify=dataframe["traversability_label"]
            if params.dataset.STRATIFY
            else None,
        )

        # Count the number of samples per class
        train_distribution = dataframe_train[
            "traversability_label"
        ].value_counts()
        test_distribution = dataframe_test[
            "traversability_label"
        ].value_counts()

        if plot_velocity_distribution:
            plt.bar(
                train_distribution.index,
                train_distribution.values,
                fc="blue",
                label="train",
            )
            plt.bar(
                test_distribution.index,
                test_distribution.values,
                fc="orange",
                label="test",
            )
            plt.legend()
            plt.title("Train and test sets distribution")
            plt.xlabel("Traversability label")
            plt.ylabel("Samples")
            plt.show()

        # Iterate over each row of the training set and copy the images to the
        # training directory
        for _, row in dataframe_train.iterrows():
            image_file = os.path.join(self.images_directory, row["image_id"])
            shutil.copy(image_file + ".png", train_directory)
            shutil.copy(image_file + "d.png", train_directory)
            shutil.copy(image_file + "n.png", train_directory)
            # shutil.copy(image_file + "d.tiff", train_directory)
            # shutil.copy(image_file + "n.tiff", train_directory)

        # Iterate over each row of the testing set and copy the images to the
        # testing directory
        for _, row in dataframe_test.iterrows():
            image_file = os.path.join(self.images_directory, row["image_id"])
            shutil.copy(image_file + ".png", test_directory)
            shutil.copy(image_file + "d.png", test_directory)
            shutil.copy(image_file + "n.png", test_directory)
            # shutil.copy(image_file + "d.tiff", test_directory)
            # shutil.copy(image_file + "n.tiff", test_directory)

        # Store the train and test splits in csv files
        dataframe_train.to_csv(
            self.dataset_directory + "/traversal_costs_train.csv", index=False
        )
        dataframe_test.to_csv(
            self.dataset_directory + "/traversal_costs_test.csv", index=False
        )


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    dataset = DatasetBuilder(
        name="tmp_multimodal_formula_png_no_sand_filtered_hard_higher_T_no_trajectory_limit_large_patch_no_coherence_no_cohesion_null_speed_hand_filtered"
    )

    (
        raw_imu,
        velocities,
        source_bagfiles,
    ) = dataset.write_images_and_compute_features(
        files=[
            ## Grass and roal only
            # "bagfiles/raw_bagfiles/Terrains_Samples/grass_easy.bag",
            # "bagfiles/raw_bagfiles/Terrains_Samples/grass_medium.bag",
            # "bagfiles/raw_bagfiles/Terrains_Samples/road_easy.bag",
            # "bagfiles/raw_bagfiles/Terrains_Samples/road_medium.bag",
            ## No sand
            "bagfiles/raw_bagfiles/Terrains_Samples/dust.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/forest_dirt_easy.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/forest_dirt_medium.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/forest_dirt_stones_branches.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/forest_leaves.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/forest_leaves_branches.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/grass_easy.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/grass_medium.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/gravel_easy.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/gravel_medium.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/road_easy.bag",
            "bagfiles/raw_bagfiles/Terrains_Samples/road_medium.bag",
            ## All
            # "bagfiles/raw_bagfiles/Terrains_Samples/",
            # "bagfiles/raw_bagfiles/ENSTA_Campus/",
            # "bagfiles/raw_bagfiles/Palaiseau_Forest/",
            # "bagfiles/raw_bagfiles/Troche/",
            # "bagfiles/raw_bagfiles/Terrains_Samples/troche_forest_hard_2023-05-30-13-44-49_0.bag",
            # "bagfiles/raw_bagfiles/Terrains_Samples/road1_2023-05-30-13-27-30_0.bag",
            # "bagfiles/raw_bagfiles/Terrains_Samples/road1_2023-05-30-14-05-20_0.bag"
            # "bagfiles/raw_bagfiles/Terrains_Samples/grass1_2023-05-30-13-56-09_0.bag"
        ],
        velocity_type=VelocityType.MANUAL,
        filter_coherence=False,
        filter_cohesion=False,
    )

    dataset.write_traversal_costs(
        raw_imu,
        velocities,
        source_bagfiles,
        cost_type=CostType.FORMULA,
    )

    dataset.create_train_test_splits()
