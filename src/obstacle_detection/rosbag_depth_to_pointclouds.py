import argparse
import rosbag
import numpy as np
import rospy

import cv_bridge

import params.robot
import params.dataset
import params.traversal_cost
import params.learning



def depth_from_bagfile(bag: rosbag.Bag, t_image: rospy.Time) -> np.array:
    """
    Given a rosbag and a timestamp, returns the depth image message whose timestamp is closest.

    Args:
        bag (rosbag.Bag): The rosbag containing the depth image messages.
        t_image (rospy.Time): The timestamp of the RGB image message.

    Returns:
        np.array: The depth image message whose timestamp is closest to the given timestamp.
    """

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

def rosbag_depth_to_pointclouds(rosbag_file, output_dir):
    bag = rosbag.Bag(rosbag_file, 'r')
    for topic, msg, t in bag.read_messages(topics=['/depth_topic']):
        # Convert depth message to point cloud
        point_cloud = pc2.read_points(msg)

        # Convert point cloud to numpy array
        points = np.array(list(point_cloud), dtype=np.float32)

        # Save point cloud as .bin file
        output_file = f"{output_dir}/{t.to_nsec()}.bin"
        points.tofile(output_file)

    bag.close()

def read_bag(bagfile):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert depth messages to point clouds')
    parser.add_argument('rosbag_file', type=str, help='Path to the rosbag file')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()

    rosbag_depth_to_pointclouds(args.rosbag_file, args.output_dir)