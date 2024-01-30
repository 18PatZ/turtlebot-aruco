#!/usr/bin/env python

import rospy 
from std_msgs.msg import String, Float64, Bool, Empty
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

from turtlebot_aruco.msg import Aruco

import time

TURN_DIRECTION = -1

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
   return rospy.wait_for_message("/razor/yaw", Float64).data


def ang_diff(target, source):
    diff = target - source
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


def turn_to_marker(direction):

    subscribe_aruco()

    print("Camera engaged. Beginning turn...")

    set_velocity(0.1 * direction)
    while found_marker is None:
        time.sleep(0.01)

    set_velocity(0.0)
    
    print("Found marker! Obtaining yaw...")
    time.sleep(1)
    current_yaw = get_yaw()
    print("We are currently at yaw", current_yaw)

    unsubscribe_aruco()

    print("Camera disengaged.")

    print("Initiating gyro re-calibration.")
    pub_reset.publish(Empty()) # note there is some delay between publish and calibration starting

    print("Waiting ten seconds for camera CPU wind down and gyro calibration...")
    time.sleep(10)

    current_yaw = get_yaw() # also serves purpose of blocking until calibration finished
    print("Re-calibrated gyro states current yaw as", current_yaw)

    print("Commanding PID to turn back to original yaw", original_yaw)
    pub_pid_yaw.publish(original_yaw)
    pub_pid_enabled.publish(True)

    time.sleep(10)

    start = time.time()

    while abs(ang_diff(original_yaw, get_yaw())) > 2 and time.time()-start < 10:
        print("Angle to target:", ang_diff(original_yaw, get_yaw()))
        time.sleep(0.1)

    pub_pid_enabled.publish(False)
    set_velocity(0.0)

    print("Turn complete. Current yaw", get_yaw(), "vs original", original_yaw)
    


def subscribe_aruco():
    global sub
    sub = rospy.Subscriber("/aruco", Aruco, aruco_callback)
    print("Waiting for first processed camera image...")
    rospy.wait_for_message("/aruco", Aruco)

def unsubscribe_aruco():
    global sub

    if sub is not None:
        sub.unregister()
        sub = None
    

if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_control')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pub_pid_yaw = rospy.Publisher('/pid/target_yaw', Float64, queue_size=1, latch=True)
    pub_pid_enabled = rospy.Publisher('/pid/enabled', Bool, queue_size=1, latch=True)

    pub_reset = rospy.Publisher('/reset', Empty, queue_size=1)

    pub_pid_enabled.publish(False)
    set_velocity(0.0)
    print("Holding one second for PID to halt...")
    time.sleep(1)
    set_velocity(0.0)

    print("Obtaining current yaw...")
    original_yaw = get_yaw()

    print("Started at yaw", original_yaw,". Engaging camera...")
    turn_to_marker(-1 * TURN_DIRECTION)
