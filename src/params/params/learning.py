import torch

import params.dataset
from params import PROJECT_PATH

#########################
## Learning parameters ##
#########################

# Define the data to be used
_dataset = "multimodal_siamese_png_no_sand_filtered_hard_higher_T_no_trajectory_limit_large_patch_no_coherence_no_cohesion_null_speed_hand_filtered"
DATASET = PROJECT_PATH / f"datasets/dataset_{_dataset}"

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Set learning parameters
LEARNING = {
    "batch_size": 64,
    "nb_epochs": 200,
    "learning_rate": 0.0009,
}


####################################
## Images' transforms parameters ##
####################################

IMAGE_SHAPE = (
    params.dataset.MIN_WIDTH // params.dataset.RECTANGLE_RATIO,
    params.dataset.MIN_WIDTH,
)

JITTER_PARAMS = {
    "brightness": 0.9,
    "contrast": 0.3,
    "saturation": 1,
    "hue": 0.1,
}

NORMALIZE_PARAMS = {
    "mean": torch.tensor(
        [0.4710, 0.5030, 0.4580, 0.0855, 0.4981, 0.5832, 0.8387]
    ),
    "std": torch.tensor(
        [0.1965, 0.1859, 0.1955, 0.0684, 0.1720, 0.1991, 0.1468]
    ),
}


####################
## Network design ##
####################

# Set the parameters for the network
NET_PARAMS = {"nb_input_channels": 7, "nb_input_features": 1, "nb_classes": 10}


#######################################
## Saving the weights of the network ##
#######################################

# The name of the file in which the weights of the network will be saved
PARAMS_FILE = "network.params"


############################
## Output of the training ##
############################

# The name of the directory in which the logs will be saved
LOG_DIR = None
