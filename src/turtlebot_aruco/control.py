#!/usr/bin/env python

import rospy 
from std_msgs.msg import String, Float64, Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

from turtlebot_aruco.msg import Aruco

import time

found_marker = None
pub = None
pub_pid_yaw = None
pub_pid_enabled = None
original_yaw = None

sub = None

def aruco_callback(aruco_msg):
    global found_marker
    if aruco_msg.has_marker:
        found_marker = aruco_msg


def set_velocity(control_angular_vel):
    twist = Twist()

    twist.linear.x = twist.linear.y = twist.linear.z = 0.0
    twist.angular.x = twist.angular.y = 0.0

    twist.angular.z = control_angular_vel

    pub.publish(twist)


def get_yaw():
   return rospy.wait_for_message("/razor/yaw", Float64)


def turn_to_marker():

    subscribe_aruco()

    print("Camera engaged. Beginning turn...")

    set_velocity(0.3)
    while found_marker is None:
        time.sleep(0.01)

    set_velocity(0.0)
    
    print("Found marker! Obtaining yaw...")
    current_yaw = get_yaw()
    print("We are currently at yaw", current_yaw)

    unsubscribe_aruco()

    print("Camera disengaged.")
    print("Waiting five seconds for camera CPU wind down...")
    time.sleep(5)

    print("Commanding PID to turn back to original yaw", original_yaw)
    pub_pid_yaw.publish(original_yaw)
    pub_pid_enabled.publish(True)


def subscribe_aruco():
    global sub
    sub = rospy.Subscriber("/aruco", Aruco, aruco_callback)
    print("Waiting for first processed camera image...")
    rospy.wait_for_message("/aruco", Aruco)

def unsubscribe_aruco():
    global sub

    if sub is not None:
        sub = rospy.Subscriber("/aruco", Aruco, aruco_callback)
        sub.unregister()
        sub = None
    

if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_control')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pub_pid_yaw = rospy.Publisher('/pid/target_yaw', Float64, queue_size=1, latch=True)
    pub_pid_enabled = rospy.Publisher('/pid/enabled', Bool, queue_size=1, latch=True)

    pub_pid_enabled.publish(False)
    print("Holding one second for PID to halt...")
    time.sleep(1)

    print("Obtaining current yaw...")
    original_yaw = get_yaw()

    print("Started at yaw", original_yaw,". Engaging camera...")
    turn_to_marker()
