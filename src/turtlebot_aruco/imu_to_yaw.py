#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import tf
import math
import numpy as np

pub = None

def imu_callback(imu_msg):
    global pub

    quat = imu_msg.orientation
    q = [quat.x, quat.y, quat.z, quat.w]
    euler = np.array(tf.transformations.euler_from_quaternion(q))
    euler *= 180 / math.pi
    yaw = euler[2]

    pub.publish(yaw)




if __name__=="__main__":
    rospy.init_node('razor_imu_to_yaw')
    pub = rospy.Publisher('/razor/yaw', Float64, queue_size=1)


    sub = rospy.Subscriber("/imu", Imu, imu_callback)
    # sub = rospy.Subscriber("/razor/imu", Imu, imu_callback)

    rospy.spin()

