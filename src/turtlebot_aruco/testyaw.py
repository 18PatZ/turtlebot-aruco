#!/usr/bin/env python3

import rospy 
from std_msgs.msg import String, Float64, Bool, Empty, Int32

from turtlebot_aruco.msg import Aruco
from geometry_msgs.msg import Twist
import time
import shutil
import os

pub = None
pub_pid_yaw = None
pub_pid_enabled = None
original_yaw = None
sub = None

def aruco_callback(aruco_msg):
    pass

def set_velocity(control_linear_vel, control_angular_vel):
    twist = Twist()

    twist.linear.y = twist.linear.z = 0.0
    twist.angular.x = twist.angular.y = 0.0

    
    twist.linear.x = control_linear_vel
    twist.angular.z = control_angular_vel

    pub.publish(twist)

    # print("Velocity FORWARD",control_linear_vel, "ROTATE", control_angular_vel) 


def set_turn_velocity(control_angular_vel):
    set_velocity(0.0, control_angular_vel)

def get_yaw():
   return rospy.wait_for_message("/razor/yaw", Float64).data


def ang_diff(target, source):
    diff = target - source
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


def turn_with_pid(yaw, duration):
    pub_pid_yaw.publish(yaw)
    pub_pid_enabled.publish(True)

    time.sleep(duration-0.5)

    pub_pid_enabled.publish(False)
    set_turn_velocity(0.0)
    time.sleep(0.5)
    set_turn_velocity(0.0)




def subscribe_aruco():
    global sub
    sub = rospy.Subscriber("/aruco", Aruco, aruco_callback)
    print("    Waiting for first processed camera image...")
    rospy.wait_for_message("/aruco", Aruco)

def unsubscribe_aruco():
    global sub

    if sub is not None:
        sub.unregister()
        sub = None


def start():
    # if os.path.exists("footage"):
    #     shutil.rmtree("footage", ignore_errors=True)

    subscribe_aruco()
    for i in range(2):
        set_turn_velocity(0.4)
        time.sleep(4)
        set_turn_velocity(0.0)
        
        time.sleep(2)
        current_yaw = get_yaw()
        print("    We are currently at yaw", current_yaw)

        print("    Initiating gyro re-calibration.")
        pub_reset.publish(Empty()) # note there is some delay between publish and calibration starting
        time.sleep(10)

        current_yaw = get_yaw() # also serves purpose of blocking until calibration finished
        print("    Re-calibrated gyro states current yaw as", current_yaw)

        set_turn_velocity(-0.4)
        time.sleep(4)
        set_turn_velocity(0.0)

        time.sleep(1)
        print("YAW:", get_yaw())
    unsubscribe_aruco()
    




if __name__=="__main__":
    rospy.init_node('turtlebot_test')

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pub_pid_yaw = rospy.Publisher('/pid/target_yaw', Float64, queue_size=1, latch=True)
    pub_pid_enabled = rospy.Publisher('/pid/enabled', Bool, queue_size=1, latch=True)

    pub_reset = rospy.Publisher('/reset', Empty, queue_size=1)

    pub_pid_enabled.publish(False)
    set_velocity(0.0, 0.0)
    print("Holding one second for PID to halt...")
    time.sleep(1)
    set_velocity(0.0, 0.0)

    print("Obtaining current yaw...")
    original_yaw = get_yaw()
    print("Started at yaw", original_yaw,".")

    start()