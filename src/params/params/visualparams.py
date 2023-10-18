# Python librairies
import numpy as np
import torch

from params import PROJECT_PATH

# Importing custom made parameters
from params import robot
from params import learning
from params import dataset
import utilities.frames as frames

# Ros node location, usually in catkin_ws/src
node_path = PROJECT_PATH / "ROS_NODE/visual_traversability"

# Importing parameters relative to the robot size and configuration
ALPHA = robot.alpha
ROBOT_TO_CAM = robot.ROBOT_TO_CAM
CAM_TO_ROBOT = robot.CAM_TO_ROBOT
K = robot.K
WORLD_TO_ROBOT = np.eye(4)
ROBOT_TO_WORLD = frames.inverse_transform_matrix(WORLD_TO_ROBOT)
L_ROBOT = robot.L

# Parameters relative to the costmap configuration
X = 20
Y = 20
RESOLUTION = 0.20

# Parameters relative to the rosbag input
IMAGE_H, IMAGE_W = 720, 1080
IMAGE_TOPIC = robot.IMAGE_TOPIC
IMAGE_RATE = robot.CAMERA_SAMPLE_RATE
ODOM_TOPIC = robot.ODOM_TOPIC
ODOM_RATE = robot.ODOM_SAMPLE_RATE
DEPTH_TOPIC = robot.DEPTH_TOPIC
DEPTH_RATE = robot.DEPTH_SAMPLE_RATE
NB_MESSAGES_THR = dataset.NB_MESSAGES_THR
TIME_DELTA = dataset.TIME_DELTA
INPUT_DIR = (
    PROJECT_PATH
    / "bagfiles/raw_bagfiles/ENSTA_Campus/tom_2023-05-30-13-28-39_1.bag"
)

# Parameters relative to the video recording
OUTPUT_DIR = node_path / "output"
VISUALIZE = True
RECORD = False
LIVE = False

# Parameters relative to the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REGRESSION = True
WEIGHTS = (
    node_path
    / "weights/ResNet18Velocity_Regression_V2/total_filtered_hard.params"
)

CROP_WIDTH = 210
CROP_HEIGHT = 70
NORMALIZE_PARAMS = learning.NORMALIZE_PARAMS

DATASET = (
    PROJECT_PATH / "datasets/dataset_multimodal_siamese_png_filtered_hard"
)
if not REGRESSION:
    MIDPOINTS = np.load(DATASET / "bins_midpoints.npy")
else:
    MIDPOINTS = None

VELOCITY = 0.2

# Paremeters relative to the video treatment
THRESHOLD_INTERSECT = 0.1
THRESHOLD_AREA = 25
