import numpy as np

np.float = float

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import open3d as o3d
from pointcloud import depthimage_to_pointcloud
from open3d_ros_helper import open3d_ros_helper as orh
import matplotlib.pyplot as plt
import params.robot
from scipy.spatial.transform import Rotation as R
from time import time

K = params.robot.K

def get_transform(vertical):
    vertical = vertical
    vertical = vertical / np.linalg.norm(vertical)
    z = np.array([0, 0, 1])
    z = z / np.linalg.norm(z)
    axis = np.cross(vertical, z)
    angle = np.arccos(np.dot(vertical, z))
    rotation = R.from_rotvec(axis * angle)
    return rotation.as_matrix()

def depth_to_pointcloud_(depth_image):
    start = time()
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
    depth_image = np.array(depth_image, dtype=np.float32)
    coords = depthimage_to_pointcloud(depth_image)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    plane_normal = plane_model[:3]
    scale = np.sign(np.dot(plane_normal, np.array([0, 0, 1])))

    rot = get_transform(-scale*plane_normal)
    pcd.rotate(rot)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointcloud_msg  = orh.o3dpc_to_rospc(pcd) 
    
    pointcloud_msg.header.frame_id = 'map'
    dur = time() - start
    print(pointcloud_msg.header)
    print(f"Time taken: {dur}")
    return pointcloud_msg

def depth_image_callback(depth_image):
    print("Depth image received.")
    pointcloud_msg = depth_to_pointcloud_(depth_image)
    pointcloud_pub.publish(pointcloud_msg)


if __name__ == '__main__':
    rospy.init_node('depth_to_pointcloud_node')
    depth_image_topic = rospy.get_param('~depth_image_topic', '/zed_node/depth/depth_registered')
    bridge = CvBridge()
    pointcloud_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
    rospy.Subscriber(depth_image_topic, Image, depth_image_callback)
    rospy.spin()