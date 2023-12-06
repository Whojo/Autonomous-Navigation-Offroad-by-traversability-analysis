""" Dataset creation parameters """

import torch

from params import PROJECT_PATH
import traversalcost.features


##############################################
## Vision Model Dataset creation parameters ##
##############################################

# Upper bound of the number of images to be extracted from the bag files
NB_IMAGES_MAX = 100000

# Number of messages threshold before skipping a rosbag file
NB_MESSAGES_THR = 0.08

# Linear velocity threshold from which the robot is considered to be moving
LINEAR_VELOCITY_THR = 0.05  # [m/s]

# Distance the robot travels within a patch
PATCH_DISTANCE = 0.5  # [m]

# Threshold to filter tilted patches
PATCH_ANGLE_THR = 1  # [rad]

# Ratio between the width and the height of a rectangle
RECTANGLE_RATIO = 3

# Minimum width of a rectangle patch
MIN_WIDTH = 210  # [px]

# Time during which the future trajectory is taken into account
T = 30  # [s]

# Time interval to look for the depth image corresponding to the current rgb
# image
TIME_DELTA = 0.05  # [s]

# Manual velocity labels of extracted IMU sequences
LABELS_PATH = "src/traversal_cost/datasets/dataset_200Hz_wrap_fft/labels.csv"

###################################################################
## Semantic segmentation Distilation Dataset creation parameters ##
###################################################################

# Semantic segmentation model to use ("vit_h", "vit_b")
sam_model_name = "vit_h"

# Path to the semantic segmentation model checkpoint
sam_checkpoint = (
    PROJECT_PATH / "src/semantic_segmentation/models/sam_vit_h.pth"
)

# Vision model checkpoint (from "model_development" folder,
# e.g. "multimodal_velocity_regression_alt")
weight_path = (
    PROJECT_PATH
    / "src/models_development/multimodal_velocity_regression_alt/logs/_post_hp_tuning_data_augmentation/network.params"
)

# Stability score threshold for the semantic segmentation mask with Segment
# Anything model
stability_score_thresh = 0.8

# Number of patches sampled in each image's mask to
# compute the traversal cost of the mask
nb_total_patch = 100

# Number of patches kept for the traversal cost computation
# that are the most centered on the mask from the patches sampled
# (nb_total_patch)
nb_centered_patch = 10

# Threshold to filter SAM's segmentation based on its completeness
# (i.e. the ratio of pixels included in at least one mask)
completeness_threshold = 0.92

##########################################
## Features extraction from IMU signals ##
##########################################

# Describe the features to be extracted from the IMU signals
# (if the function takes parameters, default values can be overwritten by
# specifying them in dictionaries)
# (the output of the function must be a numpy array of shape (n,) or a list
# of length n, n being the number of features)
params = {}
FEATURES = {
    "function": traversalcost.features.wrapped_signal_fft,
    "params_roll_rate": params,
    "params_pitch_rate": params,
    "params_vertical_acceleration": params,
}


##################################################
## Traversal cost computation from the features ##
##################################################

# -------------------#
#  Siamese Network  #
# -------------------#

# Path to the parameters file
SIAMESE_PARAMS = (
    PROJECT_PATH
    / "src/traversal_cost/siamese_network/logs/_2023-09-27-12-22-04/siamese.params"
)

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


###########################################################
## Depth image and surface normal computation parameters ##
###########################################################

# Set the parameters for the bilateral filter
# BILATERAL_FILTER = {"d": 5,
#                     "sigmaColor": 0.5,
#                     "sigmaSpace": 2}
BILATERAL_FILTER = None

# Threshold for the gradient magnitude
GRADIENT_THR = 8

# Set the depth range
DEPTH_RANGE = (0.7, 7)  # [m]

# Set the default normal vector to replace the invalid ones
DEFAULT_NORMAL = [0, 0, 1]


###########################
## Train and test splits ##
###########################

# Tell if the splits must be stratified or not
STRATIFY = False
