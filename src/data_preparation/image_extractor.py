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
plt.rcParams['text.usetex'] = True  # Render Matplotlib text with Tex
import tifffile

# ROS Python libraries
import cv_bridge
import rosbag
import rospy
import tf.transformations

# Custom modules and packages
import utilities.drawing as dw
import utilities.frames as frames
from depth.utils import Depth
import traversalcost.utils
import traversalcost.traversal_cost
import params.robot
import params.dataset
import params.traversal_cost
import params.learning

def is_bag_healthy(bag: str) -> bool:
    """Check if a bag file is healthy

    Args:
        bag (str): Path to the bag file

    Returns:
        bool: True if the bag file is healthy, False otherwise
    """    
    # Get the bag file duration
    duration = bag.get_end_time() - bag.get_start_time()  # [seconds]

    for topic, frequency in [(params.robot.IMAGE_TOPIC,
                              params.robot.CAMERA_SAMPLE_RATE),
                             (params.robot.DEPTH_TOPIC,
                              params.robot.DEPTH_SAMPLE_RATE),
                             (params.robot.ODOM_TOPIC,
                              params.robot.ODOM_SAMPLE_RATE),
                             (params.robot.IMU_TOPIC,
                              params.robot.IMU_SAMPLE_RATE)]:

        # Get the number of messages in the bag file
        nb_messages = bag.get_message_count(topic)
        
        # Check if the number of messages is consistent with the frequency
        if np.abs(nb_messages - frequency*duration)/(frequency*duration) >\
                params.dataset.NB_MESSAGES_THR:
            return False

    return True

if __name__ == "__main__":
    
    bridge = cv_bridge.CvBridge()

    destination = "/home/gabriel/PRE/bagfiles/images_extracted/temp"

    try:  # A new directory is created if it does not exist yet
            os.mkdir(destination)
            print(destination + " folder created\n")

    except OSError:  # Display a message if it already exists and quit
        print("Existing directory " + destination)

    index = 0

    bag_file = "/home/gabriel/PRE/bagfiles/raw_bagfiles/Palaiseau_Forest/tom_2023-05-30-14-05-29_7.bag"

    bag = rosbag.Bag(bag_file)

    if not is_bag_healthy(bag):
        print("File " + bag_file + " is incomplete. Skipping...")
        sys.exit(1)

    for _, msg_image, t_image in tqdm(bag.read_messages(topics=[params.robot.IMAGE_TOPIC]), total=bag.get_message_count(params.robot.IMAGE_TOPIC)):
                
        # Define a variable to store the depth image
        msg_depth = None
        
        # Keep only the images that can be matched with a depth image
        if list(bag.read_messages(
            topics=[params.robot.DEPTH_TOPIC],
            start_time=t_image - rospy.Duration(
                params.dataset.TIME_DELTA),
            end_time=t_image + rospy.Duration(
                params.dataset.TIME_DELTA))):
            
            # Find the depth image whose timestamp is closest to that
            # of the rgb image
            min_t = params.dataset.TIME_DELTA
            
            # Go through the depth topic
            for _, msg_depth_i, t_depth in bag.read_messages(
                topics=[params.robot.DEPTH_TOPIC],
                start_time=t_image - rospy.Duration(params.dataset.TIME_DELTA),
                end_time=t_image + rospy.Duration(params.dataset.TIME_DELTA)):
                
                # Keep the depth image whose timestamp is closest to
                # that of the rgb image
                if np.abs(t_depth.to_sec()-t_image.to_sec()) < min_t:
                    min_t = np.abs(t_depth.to_sec() - t_image.to_sec())
                    msg_depth = msg_depth_i
        
        # If no depth image is found, skip the current image
        else:
            continue
            
        # Get the first odometry message received after the image
        _, first_msg_odom, t_odom = next(iter(bag.read_messages(
            topics=[params.robot.ODOM_TOPIC],
            start_time=t_image)))
        
        # Collect images and IMU data only when the robot is moving
        # (to avoid collecting images of the same place) 
        if first_msg_odom.twist.twist.linear.x < \
           params.dataset.LINEAR_VELOCITY_THR:
            continue
        # Convert the current ROS image to the OpenCV type
        image = bridge.imgmsg_to_cv2(msg_image,
                                          desired_encoding="passthrough")
        
        # Convert the current ROS depth image to the OpenCV type
        depth_image = bridge.imgmsg_to_cv2(msg_depth,
                                          desired_encoding="passthrough")
        
        resized_image = cv2.resize(image, (720,480))

        cv2.imshow("Image", resized_image)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            sys.exit(0)
        elif k == ord('s'):
            
            # Create a Depth object
            depth = Depth(depth_image.copy(),
                          params.dataset.DEPTH_RANGE)

            # Compute the surface normals
            depth.compute_normal(
                K=params.robot.K,
                bilateral_filter=params.dataset.BILATERAL_FILTER,
                gradient_threshold=params.dataset.GRADIENT_THR)

            # depth.display_depth()
            # depth.display_normal()

            # Give the depth image a name
            # depth_image_name = f"{index_image:05d}d.tiff"
            depth_image_name = f"{index:05d}d.png"
            # Save the depth image in the correct directory
            # tifffile.imwrite(self.images_directory +
            #                  "/" + depth_image_name,
            #                  depth.get_depth())
            depth_to_save = depth.get_depth(
                fill=True,
                default_depth=params.dataset.DEPTH_RANGE[0],
                convert_range=True)

            # Give the normal map a name
            # normal_map_name = f"{index_image:05d}n.tiff"
            normal_map_name = f"{index:05d}n.png"
            # Save the normal map in the correct directory
            # tifffile.imwrite(self.images_directory +
            #                  "/" + normal_map_name,
            #                  depth.get_normal())
            normal_to_save = depth.get_normal(
                fill=True,
                default_normal=params.dataset.DEFAULT_NORMAL,
                convert_range=True)

            # Convert the image from BGR to RGB
            normal_to_save = cv2.cvtColor(
            normal_to_save,
            cv2.COLOR_BGR2RGB)

            # Make a PIL image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Give the image a name
            image_name = f"{index:05d}.png"
            # Save the image in the correct directory
            image.save(destination + "/" + image_name, "PNG")

            # Make a PIL image
            normal_to_save = Image.fromarray(normal_to_save)
            # Save the image in the correct directory
            normal_to_save.save(destination + "/" + normal_map_name, "PNG")

            # Make a PIL image
            depth_to_save = Image.fromarray(depth_to_save)
            # Save the image in the correct directory
            depth_to_save.save(destination + "/" + depth_image_name, "PNG")

        index+=1