#!/usr/bin/env python

import rospy 
from std_msgs.msg import String, Float64, Bool, Empty
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

from turtlebot_aruco.msg import Aruco

import cv2

import time
import math
import numpy as np

TILE_SIZE = 0.2286

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


def set_velocity(control_linear_vel, control_angular_vel):
    twist = Twist()

    twist.linear.y = twist.linear.z = 0.0
    twist.angular.x = twist.angular.y = 0.0

    
    twist.linear.x = control_linear_vel
    twist.angular.z = control_angular_vel

    pub.publish(twist)

    print("Velocity FORWARD",control_linear_vel, "ROTATE", control_angular_vel) 


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


def turn_to_marker(direction):

    subscribe_aruco()

    print("Camera engaged. Beginning turn...")

    set_turn_velocity(0.1 * direction)
    while found_marker is None:
        time.sleep(0.01)

    set_turn_velocity(0.0)
    
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
    set_turn_velocity(0.0)
    time.sleep(0.5)
    set_turn_velocity(0.0)
    
    print("Turn complete. Current yaw", get_yaw(), "vs original", original_yaw)
    found_marker = None






def synchronization_turn(direction):

    period = 4
    sp = 0.2

    set_turn_velocity(sp * direction)
    time.sleep(period/2)

    set_turn_velocity(-sp * direction)
    time.sleep(period)

    set_turn_velocity(sp * direction)
    time.sleep(period)

    set_turn_velocity(-sp * direction)
    time.sleep(period/2)

    set_turn_velocity(0.0)
    
    print("Sync complete.")


def watch_for_synchronization(direction):

    subscribe_aruco()

    target_start_yaw = None
    last_measurement = None
    stage = 0


    last = time.time()

    while True:
        if found_marker is not None:
            r = found_marker.marker_rotation
            rot_vec = np.array([r.x, r.y, r.z])

            rot_mat, _ = cv2.Rodrigues(rot_vec)
            
            # rotate Z vector to align with marker. Marker's Z is normal to it
            out_of_marker = rot_mat.dot(np.array([0,0,1])) 
            out_of_marker /= np.linalg.norm(out_of_marker)

            marker_vec = np.array([out_of_marker[0], out_of_marker[2]])
            
            angle = math.atan2(marker_vec[0], marker_vec[1]) * 180/math.pi
            # print(angle)

            
            if target_start_yaw is None:
                target_start_yaw = angle
                last_measurement = 0

            diff = ang_diff(angle, target_start_yaw)
            print(diff)

            if stage == 0: # stationary
                if diff - last_measurement >= 10:
                    stage = 1
                    last_measurement = diff
                    print("RIGHT 1!")
                    last = time.time()
            elif stage == 1:  # turning to the right
                if diff - last_measurement >= 2:
                    last_measurement = diff
                    last = time.time()
                if diff - last_measurement <= -10:
                    stage = 2
                    last_measurement = diff
                    print("LEFT 2!")
            elif stage == 2:  # turning to the left
                if diff - last_measurement <= -2:
                    last_measurement = diff
                    last = time.time()
                if diff - last_measurement >= 10:
                    stage = 3
                    last_measurement = diff
                    print("RIGHT 3!")
            elif stage == 3:  # turning to the right
                if diff - last_measurement >= 2:
                    last_measurement = diff
                    last = time.time()
                if diff - last_measurement <= -10:
                    stage = 4
                    last_measurement = diff
                    last = time.time()
                    print("LEFT 4!")
            elif stage == 4:  # turning to the left
                if diff - last_measurement <= -2:
                    last_measurement = diff
                    last = time.time()
                if diff - last_measurement >= 10:
                    stage = 0
                    print("RESET")
                    last = time.time()
                    target_start_yaw = angle
                    last_measurement = 0
                elif (time.time() - last) >= 2:
                    print("COMPLETE.")
                    break

            if (time.time() - last) >= 10: # reset
                print("RESET")
                last = time.time()
                target_start_yaw = angle
                last_measurement = 0

        time.sleep(0.1)

    unsubscribe_aruco()
    


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
  
def drive_straight():
    tile_size = 0.2286
    speed = 0.2
    set_velocity(speed, 0.0)
    time.sleep(5 * tile_size / speed)
    set_velocity(0.0, 0.0)


def drive_arc(x, y, speed):

    # x forwards, y left
    # r^2 = x^2 + (r-y)^2
    # r^2 = x^2 + r^2 - 2ry + y^2
    # 2ry = x^2 + y^2
    # r = (x^2 + y^2) / (2y)


    x *= TILE_SIZE
    y *= TILE_SIZE

    if y == 0:
        distance = abs(x)
        angle = 0
        r = 0
    else:
        r = abs((x**2 + y**2) / (2 * y))

        if abs(x) >= abs(y):
            if r == y:
                angle = math.copysign(math.pi/2.0, y)
            else:
                if y >= 0:
                    angle = math.atan(x / (r-y))
                else:
                    angle = -math.atan(x / (r-(-y)))
        else:
            if y >= 0:
                angle = math.pi - math.atan(x / (y-r))
            else:
                angle = -math.pi + math.atan(x / (-y-r))

        distance = abs(angle * r)

    time_to_travel = abs(distance / speed)

    angular_velocity = angle / time_to_travel

    print(distance, angle, r, speed, time_to_travel, angular_velocity)

    set_velocity(speed, angular_velocity)
    time.sleep(time_to_travel)
    set_velocity(0.0, 0.0)

    # turn back to forward
    # turn_speed = math.copysign(0.1, -angular_velocity)
    # time_to_turn = -angle / turn_speed
    time_to_turn = 1.0
    turn_speed = -angle / time_to_turn

    set_velocity(0.0, turn_speed)
    time.sleep(time_to_turn)
    set_velocity(0.0, 0.0)


if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_control')

    agent_id = rospy.get_param('~agent_id')

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

    # print("Driving forward one...")
    # drive_straight()
    # print("Drive complete.")

    # set_velocity(0.1, 0)
    # time.sleep(TILE_SIZE)
    # set_velocity(0.2, 0)
    # time.sleep(TILE_SIZE)
    # remember wheels cant spin up to 0.2 immediately

    if agent_id == 0:
        watch_for_synchronization(-1)
    else:
        synchronization_turn(-1)

    # print("Driving arc...")
    # drive_arc(2, 1, 0.05)
    # print("Drive complete.")

    # print("Driving arc...")
    # drive_arc(1, -2, 0.1)
    # print("Drive complete.")

    # print("Driving arc...")
    # drive_arc(2, 2, 0.1)
    # print("Drive complete.")

    # time.sleep(2)

    # print("Driving arc...")
    # drive_arc(-6, -1, -0.2)
    # print("Drive complete.")

    # print("Started at yaw", original_yaw,".")
    # print("Engaging camera...")

    # if agent_id == 0:
    #     turn_direction = -1
    # elif agent_id == 1:
    #     turn_direction = 1
    # turn_to_marker(turn_direction)


    # print("Driving arc...")
    # drive_arc(-2, -1, -0.1)
    # print("Drive complete.")

    # turn_to_marker(turn_direction)
