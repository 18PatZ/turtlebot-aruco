#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

import pygame

import time

'''
0: left joystick X (-1 left)
1: left joystick Y (-1 up)
2: left trigger (-1 released)
3: right joystick X (-1 left)
4: right joystick Y (-1 up)
5: right trigger (-1 released)
'''

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

JOY_LEFT_X = 0
JOY_LEFT_Y = 1
TRIGGER_LEFT = 2
JOY_RIGHT_X = 3
JOY_RIGHT_Y = 4
TRIGGER_RIGHT = 5

UP = -1
DOWN = 1
RIGHT = 1
LEFT = -1
PRESSED = 1
RELEASED = -1

def makeSimpleProfile(output, input, slop):
    if input > output:
        output = min( input, output + slop )
    elif input < output:
        output = max( input, output - slop )
    else:
        output = input

    return output


def getAxis(controller, axis, at_rest, deadzone):
    val = controller.get_axis(axis)
    if abs(val-at_rest) < deadzone:
        return at_rest
    return val

def norm(v, minV, maxV):
    return (v - minV) / (maxV-minV)


if __name__=="__main__":

    pygame.init()
    controller = pygame.joystick.Joystick(0)
    controller.init()

    rospy.init_node('turtlebot3_ps4')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    target_linear_vel   = 0.0
    target_angular_vel  = 0.0
    control_linear_vel  = 0.0
    control_angular_vel = 0.0

    try:
        while not rospy.is_shutdown():
            pygame.event.pump()

            forward = norm(getAxis(controller, TRIGGER_RIGHT, RELEASED, 0.05), RELEASED, PRESSED)
            back = norm(getAxis(controller, TRIGGER_LEFT, RELEASED, 0.05), RELEASED, PRESSED)
            forward -= back
            # forward = -getAxis(controller, JOY_RIGHT_Y, 0, 0.05)

            target_linear_vel = BURGER_MAX_LIN_VEL * forward

            right = getAxis(controller, JOY_RIGHT_X, 0, 0.05)
            
            target_angular_vel = BURGER_MAX_ANG_VEL * -right

            print(forward, right, target_linear_vel, target_angular_vel)

            twist = Twist()

            control_linear_vel = makeSimpleProfile(control_linear_vel, target_linear_vel, (LIN_VEL_STEP_SIZE/2.0))
            twist.linear.x = control_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0

            control_angular_vel = makeSimpleProfile(control_angular_vel, target_angular_vel, (ANG_VEL_STEP_SIZE/2.0))
            twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_angular_vel

            pub.publish(twist)

            time.sleep(0.01)
    finally:
        twist = Twist()
        twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
        pub.publish(twist)

