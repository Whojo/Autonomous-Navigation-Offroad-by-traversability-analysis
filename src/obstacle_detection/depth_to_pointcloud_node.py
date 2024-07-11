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

K = params.robot.K

def depth_to_pointcloud_(depth_image):
    
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
    plt.imshow(depth_image)
    plt.show()
    depth_image = np.array(depth_image, dtype=np.float32)
    print("Depth shape: ", depth_image.shape)
    coords = depthimage_to_pointcloud(depth_image)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    pointcloud_msg  = orh.o3dpc_to_rospc(pcd) 
    
    pointcloud_msg.header.frame_id = 'map'
    print(pointcloud_msg.header)
    # input()
    return pointcloud_msg

def depth_image_callback_(depth_image):
    print("Depth image received.")
    # input()
    pointcloud_msg = depth_to_pointcloud_(depth_image)
    pointcloud_pub.publish(pointcloud_msg)


def depth_image_callback(depth_image):
    print("Depth image received.")
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
    # depth_image = np.array(depth_image, dtype=np.uint8)
    # print("Depth shape: ", depth_image.shape)
    open3d_image = o3d.geometry.Image(depth_image)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1080, 1920, fx, fy, cx, cy)
    pointcloud = o3d.geometry.PointCloud.create_from_depth_image(open3d_image, intrinsic, depth_scale=255.0)
    pointcloud_msg = orh.o3dpc_to_rospc(pointcloud)
    pointcloud_msg.header.frame_id = 'map'
    pointcloud_pub.publish(pointcloud_msg)
    

if __name__ == '__main__':
    rospy.init_node('depth_to_pointcloud_node')
    depth_image_topic = rospy.get_param('~depth_image_topic', '/zed_node/depth/depth_registered')
    bridge = CvBridge()
    pointcloud_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)
    rospy.Subscriber(depth_image_topic, Image, depth_image_callback)
    rospy.spin()