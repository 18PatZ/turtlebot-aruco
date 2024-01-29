#!/usr/bin/env python

import rospy 
from std_msgs.msg import String 
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3

from turtlebot_aruco.msg import Aruco

# Import OpenCV libraries and tools
import cv2
from cv_bridge import CvBridge, CvBridgeError

import time

import numpy as np
from turtlebot_aruco.turtle_aruco import ArucoDetector


# Initialize the CvBridge class
bridge = CvBridge()

'''
height: 540
width: 960
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [687.794921875, 0.0, 479.5737609863281, 0.0, 687.794921875, 268.1636962890625, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [687.794921875, 0.0, 479.5737609863281, 0.0, 0.0, 687.794921875, 268.1636962890625, 0.0, 0.0, 0.0, 1.0, 0.0]

height: 621
width: 1104
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [680.811279296875, 0.0, 551.4961547851562, 0.0, 680.811279296875, 308.56463623046875, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [680.811279296875, 0.0, 551.4961547851562, 0.0, 0.0, 680.811279296875, 308.56463623046875, 0.0, 0.0, 0.0, 1.0, 0.0]
'''

class ArucoSubscribeListener(rospy.SubscribeListener):
    
    def __init__(self, tbzed):
        self.tbzed = tbzed

    def peer_subscribe(self, topic_name, topic_publish, peer_publish):
        print("New subscriber on", topic_name)
        self.tbzed.subscribe_zed()

    def peer_unsubscribe(self, topic_name, num_peers):
        print("Unsubscribe received on", topic_name, "with", num_peers, "subscribers remaining")
        self.tbzed.unsubscribe_zed()



class TurtlebotArucoZed:

    def __init__(self):
        self.pub = None
        self.pub2 = None
        self.zed_sub = None
        pass

    def image_callback(self, img_msg):
        rospy.loginfo(img_msg.header)

        # Try to convert the ROS Image message to a CV2 Image
        try:
            cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        if cv_image.shape[2] == 4:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        
        has_marker, rotation, translation, img = self.arucoDetector.process(cv_image)
        r = Vector3(rotation[0], rotation[1], rotation[2])
        t = Vector3(translation[0], translation[1], translation[2])

        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        
        aruco_msg = Aruco(has_marker, r, t, None)
        self.pub.publish(aruco_msg)
        self.pub2.publish(img_msg)


    def subscribe_zed(self):
        self.zed_sub = rospy.Subscriber("/zedm/zed_node/left/image_rect_color", Image, self.image_callback)
        print("Camera engaged.")
    
    def unsubscribe_zed(self):
        if self.zed_sub is not None:
            self.zed_sub.unregister()
            self.zed_sub = None
            print("Camera disengaged.")

            # # publish no-marker message since camera feed is disengaged
            # aruco_msg = Aruco(has_marker=False, r=Vector3(0,0,0), t=Vector3(0,0,0), debug_image=None)
            # self.pub.publish(aruco_msg)


    def run(self): 
        rospy.init_node('turtlebot_aruco_zed', anonymous=True) 

        self.sublistener = ArucoSubscribeListener(self)
        
        self.pub = rospy.Publisher('aruco', Aruco, queue_size=1, subscriber_listener=self.sublistener)
        self.pub2 = rospy.Publisher('aruco_debug_image', Image, queue_size=1)
        
        print("Waiting for camera calibration info...")
        camera_info = rospy.wait_for_message("/zedm/zed_node/left/camera_info", CameraInfo)
        print("Retrieved camera info!")

        distortion_matrix = np.array([camera_info.D])
        camera_matrix = np.array(camera_info.K).reshape(3, 3)

        print("  Distortion matrix: ", distortion_matrix)
        print("  Camera matrix: ", camera_matrix)

        self.arucoDetector = ArucoDetector(camera_matrix, distortion_matrix)

        rospy.spin()


if __name__ == '__main__': 
    try: 
        TurtlebotArucoZed().run() 
    except rospy.ROSInterruptException: 
        pass