#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header

from geometry_msgs.msg import Point32

def get_normals(file_id):
    pass

def numpy_array_to_pointcloud(numpy_array):
    """
    Convert a NumPy array to a ROS PointCloud message.
    """
    pointcloud_msg = PointCloud()
    pointcloud_msg.header.stamp = rospy.Time.now()
    pointcloud_msg.header.frame_id = 'base_link'  # Set your desired frame_id

    points = []
    for point in numpy_array:
        point_msg = Point32()
        point_msg.x, point_msg.y, point_msg.z = point
        points.append(point_msg)

    pointcloud_msg.points = points

    return pointcloud_msg

def get_poses(arr):
    pass

def publish_pointcloud():
    rospy.init_node('normals_publisher', anonymous=True)
    pub = rospy.Publisher('/normals', PoseStamped, queue_size = 1)

    rate = rospy.Rate(1)  # Adjust the publishing rate as needed

    while not rospy.is_shutdown():
        # Generate a sample NumPy array (replace this with your own data)
        numpy_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Convert NumPy array to PointCloud message
        pointcloud_msg = numpy_array_to_pointcloud(numpy_array)

        # Publish the PointCloud message
        pub.publish(pointcloud_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_pointcloud()
    except rospy.ROSInterruptException:
        pass
