# Self-supervised learning of terrain traversals costs for an Unmanned Ground Vehicle (UGV)

This repository contains the source code of the application Tom Ravaud and Gabriel Métois have been working on during their six months internship in the Computer Science and Systems Engineering Laboratory (U2IS), ENSTA Paris.

This project is about learning to estimate terrain traversability from vision for a mobile robot using a self-supervised approach.


<!-- # Code overview

- `bagfiles/` contains the raw data as bagfiles and some helper scripts.
  - 'raw_bagfiles' contains the ROSbags from which the data can be extracted to build a dataset
  - 'images_extracted' contains some images extracted by hand and their respective data labelled by the user himself for result measurement purpose

- `datasets/` contains the dataset created from bagfiles processed with the tool create_dataset.py of the src/data_preparation folder
              Each subfolder represents a dataset

- `results' is a small folder used to store various results of measurement.
            - A subfolder named after a dataset contains a collage from show_dataset.py displaying an overview of the dataset
            - A folder named after a Model contains the videos made from results_quantifier.py analysing the network's performances over the test    frames selected in bagfiles/images_extracted
            - labellizer.py 

- `show_dataset.py` will create a collage of worst and best images from dataset

- `train_test.py` will lanch training and test of the model

- `generate_rand_params.py` generates random hyperparameters configurations

- `hyperband.py` definitions for hyperband algorithm

- `modele_simple.py` contains the description of the neural networks

- `loader.py` defines dataloaders and data augmentation

# Code usage

Start by creating the dataset from the bag files, e.g.:

`python create_dataset.py bagfiles/sample_bag.bag`

Then start training, e.g.:

`python train_test.py --batchsize 8 --learning_rate 0.001 --weight_decay 0.002 --hyp 0 --modelnetwork AlexNet`

where:

- `weight decay` is the weight of L2 regularization.

- `hyp` should be 0 to train with hyper-parameters given as parameters, or 1 to run hyperband

- `modelnetwork` can be "AlexNet" or "ResNet".

# TODOs

- Check the quality of the dataset, and the interest of the traversabilty measures
- Check the frame (axis orientations) of the IMU
- Gather more data in short sequences with uniform terrains (e.g. all road, all grass, ...)
- Make more complex image/traversability association, in particular taking robot direction into account

# Perspectives

Check interesting related work :
- https://antonilo.github.io/vision_locomotion/ -->
