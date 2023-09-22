#####################################################
## Parameters for the visualization of the results ##
#####################################################

# Set a color for each terrain class
colors = {
    "road_easy": "black",
    "road_medium": "grey",
    "forest_dirt_easy": "deeppink",
    "dust": "orange",
    "forest_leaves": "firebrick",
    "forest_dirt_medium": "navy",
    "gravel_easy": "dodgerblue",
    "grass_easy": "limegreen",
    "grass_medium": "darkgreen",
    "gravel_medium": "royalblue",
    "forest_leaves_branches": "indigo",
    "forest_dirt_stones_branches": "blueviolet",
    "sand_hard": "darkgoldenrod",
    "sand_medium": "orangered",
    }


##########################################
## Discretization of the traversal cost ##
##########################################

# Number of bins to digitized the traversal cost
NB_BINS = 10

# Set the binning strategy
BINNING_STRATEGY = "kmeans"
