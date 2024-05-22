#!/usr/bin/env python3

import rospy 
from std_msgs.msg import String, Float64, Bool, Empty, Int32
from geometry_msgs.msg import Twist
from turtlebot3_msgs.msg import Sound

from turtlebot_aruco.msg import Aruco

import cv2

import time
import math
import numpy as np
import json

import socket
import select

from turtlebot_aruco.common import *

found_marker = None
pub = None
pub_pid_yaw = None
pub_pid_enabled = None
pub_sound = None
original_yaw = None

sub = None
agent_id = None

yaw_offset = 0

displacement = (0,0)


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

def rotate_vec(vec, ang):
    rot_mat = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return rot_mat.dot(vec)

def turn_to_marker(direction):
    global yaw_offset
    global found_marker
    found_marker = None
    # start = time.time()

    calib_original_yaw = original_yaw + yaw_offset

    print(id()+": Turning towards believed position of companion...")
    # turn_with_pid(original_yaw + direction * 90, 5)
    angle = direction * 90 * math.pi / 180
    turn_speed = 0.5
    time_to_turn = abs(angle / turn_speed)
    print(angle,turn_speed,time_to_turn)
    set_turn_velocity(turn_speed * direction)
    time.sleep(time_to_turn)
    set_turn_velocity(0)

    print(id()+": Engaging camera...")
    subscribe_aruco()

    print(id()+": Camera engaged. Beginning sweep...")

    search_range = 90 * math.pi / 180
    search_speed = 0.1 if agent_id == 0 else 0.15

    stage = 0
    t = 0
    search_dir = -1#direction
    set_turn_velocity(search_speed * search_dir)

    while found_marker is None:
        if (stage == 0 and t > search_range/search_speed) or (stage > 0 and t > search_range*2/search_speed):
            search_dir *= -1
            set_turn_velocity(search_speed * search_dir)
            t = 0
            if stage == 0:
                stage += 1

        time.sleep(0.01)
        t += 0.01

    set_turn_velocity(0.0)
    
    print(id()+": Found marker! Obtaining yaw...")
    time.sleep(2)
    current_yaw = get_yaw()
    print(id()+": We are currently at yaw", current_yaw)

    t = found_marker.marker_translation
    target = np.array([t.x, t.y, t.z])

    print(id()+": Marker is at", target)

    vec = np.array([t.x, t.z]) # ignore y since its vertical
    heading_diff = ang_diff(current_yaw, calib_original_yaw)
    print(id()+": Robot heading is", heading_diff, "degrees off forward")

    vec = rotate_vec(vec, heading_diff * math.pi/180) # positive angle = rotate counterclockwise
    
    state = (vec[1]/100.0/TILE_SIZE, -vec[0]/100.0/TILE_SIZE) # left is positive for driving

    # world state is with respect to TB1, so we invert TB0's readings
    if agent_id == 0:
        state = (-state[0], -state[1])
    # in drive state, left is positive for 

    print("\n"+id()+": Current world state: ALONG =", state[0], "CROSS = ", state[1],"\n")

    sync_udp()
    # start = time.time()

    unsubscribe_aruco()

    print(id()+": Camera disengaged.")

    print(id()+": Initiating gyro re-calibration.")
    pub_reset.publish(Empty()) # note there is some delay between publish and calibration starting

    print(id()+": Waiting ten seconds for camera CPU wind down and gyro calibration...")
    time.sleep(10)

    current_yaw = get_yaw() # also serves purpose of blocking until calibration finished
    print(id()+": Re-calibrated gyro states current yaw as", current_yaw)

    print(id()+": Commanding PID to turn back to original yaw", calib_original_yaw)
    pub_pid_yaw.publish(calib_original_yaw)
    pub_pid_enabled.publish(True)

    turn_start = time.time()
    time.sleep(4)

    while abs(ang_diff(calib_original_yaw, get_yaw())) > 2 and time.time()-turn_start < 60:#5:
        # print(id()+": Angle to target:", ang_diff(original_yaw, get_yaw()))
        time.sleep(0.1)

    pub_pid_enabled.publish(False)
    set_turn_velocity(0.0)
    time.sleep(0.5)
    set_turn_velocity(0.0)

    # upper_bound = 8+5+0.5+0.5
    # remaining_time = upper_bound - (time.time()-start)
    # if remaining_time < 0:
    #     remaining_time = 0
    # print("Waiting remaining time:", remaining_time)
    # time.sleep(remaining_time)
    sync_udp()
    
    print(id()+": Check-in complete. Current yaw", get_yaw(), "vs calib original", calib_original_yaw, "vs original", original_yaw)
    found_marker = None

    yaw_offset += 1.5

    return state



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
    print(id()+": Waiting for first processed camera image...")
    rospy.wait_for_message("/aruco", Aruco)

def unsubscribe_aruco():
    global sub

    if sub is not None:
        sub.unregister()
        sub = None


def send_sync(sock, dest):
    sock.sendto(b"SYNC", dest)
  

def sync_udp():
    port = 6666
    ip = "turtlebot2" if agent_id == 0 else "turtlebot1"
    dest = (ip, port)
    timeout = 0.1

    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.bind(("0.0.0.0", port))

    print("\n\n"+id()+": Initializing virtual flag raising...\n\n")
    print(id()+": Sending sync messages...")

    while True:
        send_sync(sock_out, dest)

        ready = select.select([sock_in], [], [], timeout)
        if ready[0]:
            data, addr = sock_in.recvfrom(1024)
            print(id()+": Received", data.decode())

            # increase chances the other robot will get this
            for i in range(10):
                send_sync(sock_out, dest)

            break
    
    sock_out.close()
    sock_in.close()

    print(id()+": SYNC ACHIEVED")



def wait_for_start():
    msg = rospy.wait_for_message("/control/start", Int32)
    return msg.data


def checkin():
    print(id()+": Starting check-in...")

    if agent_id == 0:
        turn_direction = -1
    elif agent_id == 1:
        turn_direction = 1
    return turn_to_marker(turn_direction)


def id():
    return "TB"+str(agent_id)


def execute_calib():
    global displacement
    displacement = (0,0)

    execution_start = time.time()
    to_write = []

    try:
        with open("calib.json", 'r') as file:
            to_write = json.loads(file.read())
    except:
        pass

    print(id()+": EXECUTING CHECKIN.")

    state = checkin()#(1, 4)
    state = sepToState(state[0], state[1], CENTER_STATE)
        
    t = time.time() - execution_start
    to_write.append({
        "State": [state[0], state[1]],
        "Time": t
        })
    with open("calib.json", 'w') as file:
        file.write(json.dumps(to_write, indent=4))

    print(id()+":  COMPLETE.")


def start():
    
    print(id()+": Waiting for start command...")
    wait_for_start()
    print(id()+": Received start command.")

    execute_calib()



if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_control')

    agent_id = rospy.get_param('~agent_id')

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pub_pid_yaw = rospy.Publisher('/pid/target_yaw', Float64, queue_size=1, latch=True)
    pub_pid_enabled = rospy.Publisher('/pid/enabled', Bool, queue_size=1, latch=True)

    pub_reset = rospy.Publisher('/reset', Empty, queue_size=1)

    pub_pid_enabled.publish(False)
    set_velocity(0.0, 0.0)
    print(id()+": Holding one second for PID to halt...")
    time.sleep(1)
    set_velocity(0.0, 0.0)

    print(id()+": Obtaining current yaw...")
    original_yaw = get_yaw()
    print(id()+": Started at yaw", original_yaw,".")

    start()