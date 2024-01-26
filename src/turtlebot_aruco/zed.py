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



# Print "Hello ROS!" to the Terminal and to a ROS Log file located in ~/.ros/log/loghash/*.log
rospy.loginfo("Hello ROS!")

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

class TurtlebotArucoZed:

    def __init__(self):
        # self.img = None
        self.pub = None
        self.pub2 = None
        pass

    # Define a callback for the Image message
    def image_callback(self, img_msg):
        # log some info about the image topic
        rospy.loginfo(img_msg.header)

        # Try to convert the ROS Image message to a CV2 Image
        try:
            cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


        # Show the converted image
        # show_image(cv_image)
        # self.img = cv_image
        if cv_image.shape[2] == 4:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        
        has_marker, rotation, translation, img = self.arucoDetector.process(cv_image)
        r = Vector3(rotation[0], rotation[1], rotation[2])
        t = Vector3(translation[0], translation[1], translation[2])

        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        
        aruco_msg = Aruco(has_marker, r, t, None)
        self.pub.publish(aruco_msg)
        self.pub2.publish(img_msg)



    def run(self): 
        rospy.init_node('turtlebot_aruco_zed', anonymous=True) 
        
        self.pub = rospy.Publisher('aruco', Aruco, queue_size=1, latch=True)
        self.pub2 = rospy.Publisher('aruco_debug_image', Image, queue_size=1)
        
        print("Waiting for camera calibration info...")
        camera_info = rospy.wait_for_message("/zedm/zed_node/left/camera_info", CameraInfo)
        print("Retrieved camera info!")

        distortion_matrix = np.array([camera_info.D])
        camera_matrix = np.array(camera_info.K).reshape(3, 3)

        print("  Distortion matrix: ", distortion_matrix)
        print("  Camera matrix: ", camera_matrix)

        self.arucoDetector = ArucoDetector(camera_matrix, distortion_matrix)

        sub = rospy.Subscriber("/zedm/zed_node/left/image_rect_color", Image, self.image_callback)

        madeWindow = False

        # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
        # while not rospy.is_shutdown():
        #     if self.img is not None:
        #         if not madeWindow:
        #             cv2.namedWindow("Image Window", 1)
        #             madeWindow = True
                
        #         cv2.imshow("Image Window", self.img)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #         # cv2.destroyAllWindows()

        #     # rospy.spin()
        #     time.sleep(0.01)
        rospy.spin()


if __name__ == '__main__': 
    try: 
        TurtlebotArucoZed().run() 
    except rospy.ROSInterruptException: 
        pass