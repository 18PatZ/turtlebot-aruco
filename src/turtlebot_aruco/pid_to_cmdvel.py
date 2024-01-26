#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import time

target_angular_vel = 0

BURGER_MAX_ANG_VEL = 2.84
ANG_VEL_STEP_SIZE = 0.1

def makeSimpleProfile(output, input, slop):
    if input > output:
        output = min( input, output + slop )
    elif input < output:
        output = max( input, output - slop )
    else:
        output = input

    return output


def pid_callback(f_msg):
    global target_angular_vel

    target_angular_vel = f_msg.data


if __name__=="__main__":
    rospy.init_node('pid_to_cmdvel')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    sub = rospy.Subscriber("/pid/out_angvel", Float64, pid_callback)

    print("Listening on /pid/out_angvel...")

    control_angular_vel = 0

    while not rospy.is_shutdown():
        twist = Twist()

        twist.linear.x = twist.linear.y = twist.linear.z = 0.0
        twist.angular.x = twist.angular.y = 0.0

        control_angular_vel = makeSimpleProfile(control_angular_vel, target_angular_vel, (ANG_VEL_STEP_SIZE/2.0))
        twist.angular.z = control_angular_vel

        print("Forwarding angular velocity of",control_angular_vel,"...")

        pub.publish(twist)

        time.sleep(0.01)