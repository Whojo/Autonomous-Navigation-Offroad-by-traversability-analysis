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

- `semantic_segmentation` contains experiment on semantic segmentation. The semantic segmentation builds up on the `models_development` pipeline to smooth the results. A distillation method is also implemented to enable its computation on the Jetson Nano hardware.

- `src` has the big chunks of code
  - `data_preparation` contains the tools to prepare a dataset for the network
    - `create_dataset` takes a list of rosbags paths and extract images from them, gives them a cost and builds a dataset in the `dataset` folder
    - `data_preparation` & `data_transforms` are toy tools to try to prepare the data and apply transforms to it, as a standalone tool.
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
  - `params` is a python package containing a set of parameters files. It is designed to allow those variables to be used everywhere in your system and hence unify the configuration of every application of the NN
  - `ros_nodes` contains first draws of ros nodes. Special mention to `husky_speed_depency.py` that makes the robot go back and forth at different speeds to collect data
  - `traversal_cost` contains all the networks used to give the images a cost in the dataset construction. It's structure is very similar to `model_development` so I won't say more than it's there that the SSL-Space-Magic happens.
  - `utils` contains a variety of small functions mainly for drawing robot-related structures (paths, grids, points...) on a cv image.

# Installation
This project has been developed on Ubuntu 20.04 with ROS Noetic. It is highly recommended to use a virtual environment to install the dependencies.
ROS Noetic can be installed following the instructions on the [ROS website](http://wiki.ros.org/noetic/Installation/Ubuntu).

The dependencies can easily be installed with [poetry](https://python-poetry.org/) by running the following command in the root folder of the project:
```sh
poetry install
```

Do not hesitate to learn more about poetry [here](https://python-poetry.org/docs/basic-usage/).

You might need to update the `LD_LIBRARY_PATH` environment variable to make sure that the `cv_bridge` package can be found by ROS. To do so, run the following command and reboot:
```sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/noetic/lib/' >> ~/.bashrc
```

Some matplotlib's figure require a latex package. This one can easily be installed as follows:
```sh
sudo apt install texlive-full
```

## Weights for the semantic segmentation
To successfully use the semantic segmentation module, you would need to have Segment Anything's weights in "semantic_segmentation/models/" folder. These can be downloaded from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints). If you only want to use `sam_vit_h.pth` (which is the largest SAM's model and the one we used), please download it [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pt.h).

# Develop in VSCode
Change your default interpreter to the one returned by the following command: 
```sh
echo "$(poetry show -v | head -n 1)/bin/python"
```

In VSCode this can easily be done with the "Python: Select Interpreter" command from the "Command Palette" (Ctrl+Shift+P) in which you can paste the previously mentionned path (more information [here](https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters)).


Lastly, if you intend to use Notebooks directly from VSCode, first, you should select the same interpreter as well, but also correctly configure ROS Noetic for this purpose (as explained in the [ROS Noetic installation procedure](http://wiki.ros.org/noetic/Installation/Ubuntu)).
The following command does the tricks:
```sh
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
```

# Tests
You can easily run the tests by running the following command in the root folder of the project:
```sh
pytest tests
```

The architecture of the `tests/` folder is copying the one of the root folder. Each test file has the following name `./tests/dir/subdir/test_<name_of_the_file_to_test>.py`, for the corresponding file: `./dir/subdir/<name_of_the_file_to_test>.py`.

Not all files are being tested for now, but tests are being added progressively.

# TO-DO
- Although the current state of the git indicates an extensive research and training, there's always room for improvement through more data collection, new custom transforms and new models.
- the ros node actually just gather a costmap from the network output. The next big step would be to add the output as a layer of costmap and pack it in an usable navigation plugin.
