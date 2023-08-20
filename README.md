# Self-supervised learning of terrain traversals costs for an Unmanned Ground Vehicle (UGV)

This repository contains the source code of the application Tom Ravaud and Gabriel Métois have been working on during their internship in the Computer Science and Systems Engineering Laboratory (U2IS), ENSTA Paris.

This project is about learning to estimate terrain traversability from vision for a mobile robot using a self-supervised approach.


# Code overview

- `bagfiles` contains the raw data as bagfiles and some helper scripts.
  - `raw_bagfiles` contains the ROSbags from which the data can be extracted to build a dataset
  - `images_extracted` contains some images extracted by hand and their respective data labelled by the user himself for result measurement purpose
  - `rosbag_records.sh` is a small bash script to record rosbags

- `datasets` contains the dataset created from bagfiles processed with the tool create_dataset.py of the src/data_preparation folder
              Each subfolder represents a dataset

- `Testing_Results` is a small folder used to store various results of measurement.
            - A subfolder named after a dataset contains a collage from show_dataset.py displaying an overview of the dataset
            - A folder named after a Model contains the videos made from results_quantifier.py analysing the network's performances over the test    frames selected in bagfiles/images_extracted
            - labellizer.py 

- `ROS_NODE` is a python node for ROS, it loads the network and reads a rosbag / listens to the ZED node, and build a costmap. It can even record    it's output in a video. Currently it cannot be interfaced with ROS' stack navigation but it's on the to-do-list.

- `script` has the install and uninstall scripts for properly install the parameters python package

- `src` has the big chunks of code
  - `data_preparation` contains the tools to prepare a dataset for the network
    - `Create_dataset` takes a list of rosbags paths and extract images from them, gives them a cost and builds a dataset in the `dataset` folder
    - `Data_preparation` & `data_transforms` are toy tools to try to prepare the data and apply transforms to it, as a standalone tool.
    - `image_extractor` is a tool for reading rosbags and extract images from the camera canal.
    - `dataset_modif` is a tool to apply some statistics-based filter to a dataset to balance it better
    - `show_dataset` creates a collage of pertinent samples from a dataset and stores it in the `results` folder
  - `depth` contains all you'll need to extract the normals and the depth image from a RGBD camera, and prepare the data for direct usage by the network
  - `model_uncertainty` contains some proto-tools to evaluate the uncertainty of the models prediction.
  - `models_development` contains everything training and architecture related
    - every subfolder represents a model iteration (New architecture, new method...)
      - `logs` contains the output of the training of main.py
      - `dataset`, `model`, `test`, `train`, `result` and `validate` contains the functions and classes a standard NN training process needs.
      - `custom_transforms` stores the custom pytorch transforms the user might want to define to build more efficient networks
      - `main.ipynb` is the jupiter notebook setting up all the components together. If you want to launch the whole process, everything will happen here.
    - Some other jupyternotebooks and files stands here as attemps to build some optuna, and other proto-methods.
  - `models_export` is the concatenation of the model class and transforms functions/class to export in a python package.
  - `params` is a python package containing a set of parameters files. It is designed to allow those variables to be used everywhere in your system and hence unify the configuration of every application of the NN
  - `ros_nodes` contains first draws of ros nodes. Special mention to `husky_speed_depency.py` that makes the robot go back and forth at different speeds to collect data
  - `traversal_cost` contains all the networks used to give the images a cost in the dataset construction. It's structure is very similar to `model_development` so I won't say more than it's there that the SSL-Space-Magic happens.
  - `utils` contains a variety of small functions mainly for drawing robot-related structures (paths, grids, points...) on a cv image.


# Code usage

Assuming you have ROS installed and some rosbags at hand for the training :

1- Go in the parameters `params` and set up all the variable following your preferences or your hardware configuration

2- setup the packages of parameters running `scripts/install.sh`

3- Go in `src/traversal_cost/siamese_network` and run `create_dataset` after setting at the bottom which rosbags you want to use

4- run `main.py` from the same folder in order to train your siamese network. When the goddamn thing has the correct mojo (you can check it out in the logs folder), proceed to the next step

5- In `src/data_preparation/` run `create_dataset.py` after specifying which rosbags you want to use. If you want to tailor a little bit this dataset (the dataset creation car sometimes be very messy) you can then use dataset_modif to balance your new dataset.

6- Go in `src/models_development` and choose the folder corresponding to the model of your choice. You can even try to build a new model by copy-pasting and existing one and editing the .py files. Once done, run the .ipynb at least until the log generation cell.

7- In the logs folder a subfolder named after the date and time of the training will appear. Along several useful informations and results, the .params are the weights of your network.

8- Copy the folder `ROS_NODE/visual_traversability` in your `catkin_ws`.

9- In the visualparams.py parameters file, don't forget to update the model and the weights you want to use.

10- Launch your visual_traversability node

11- ???

12- Profit!

# TO-DO

- Although the current state of the git indicates an extensive research and training, there's always room for improvement through more data collection, new custom transforms and new models.

- the ros node actually just gather a costmap from the network output. The next big step would be to add the output as a layer of costmap and pack it in an usable navigation plugin.
