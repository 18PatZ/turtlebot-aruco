#!/usr/bin/env python3

import rospy 
from std_msgs.msg import String, Float64, Bool, Empty, Int32
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

from turtlebot_aruco.msg import Aruco

import cv2

import time
import math
import numpy as np
from enum import Enum

from turtlebot_aruco.mdp.formationAnimation import *

TILE_SIZE = 0.2286
STATE_SCALE_FACTOR = 0.5
EXECUTION_SPEED = 0.1

LIN_VEL_STEP_SIZE = 0.01
STEP_TIME = 0.01
     
found_marker = None
pub = None
pub_pid_yaw = None
pub_pid_enabled = None
original_yaw = None

sub = None
agent_id = None

displacement = (0,0)

# class ACTION(Enum):
#     FORWARD = (1,0)
#     DOUBLE = (2,0)
#     LEFT = (1, 1)
#     RIGHT = (1, -1)
#     DOUBLE_FW_LEFT = (2, 1)
#     DOUBLE_FW_RIGHT = (2, -1)

ACTIONS = {
    "DOUBLE": (1, 0), 
    "FORWARD": (0.5, 0),
    "LEFT": (0.5, 1),
    # "SLIGHT_LEFT": (1, 0.5), 
    "RIGHT": (0.5, -1)
    # "SLIGHT_RIGHT": (1, -0.5)
}



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
    global found_marker
    start = time.time()

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

    print(id()+": Camera engaged. Beginning turn...")

    stage = 0
    t = 0
    search_dir = -1#direction
    set_turn_velocity(0.1 * search_dir)

    search_range = 90 * math.pi / 180
    search_speed = 0.1

    while found_marker is None:
        if (stage == 0 and t > search_range/search_speed) or (stage > 0 and t > search_range*2/search_speed):
            search_dir *= -1
            set_turn_velocity(0.1 * search_dir)
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
    heading_diff = ang_diff(current_yaw, original_yaw)
    print(id()+": Robot heading is", heading_diff, "degrees off forward")

    vec = rotate_vec(vec, heading_diff * math.pi/180) # positive angle = rotate counterclockwise
    
    state = (vec[1]/100.0/TILE_SIZE, -vec[0]/100.0/TILE_SIZE) # left is positive for driving

    # world state is with respect to TB1, so we invert TB0's readings
    if agent_id == 0:
        state = (-state[0], -state[1])
    # in drive state, left is positive for 

    print("\n"+id()+": Current world state: ALONG =", state[0], "CROSS = ", state[1],"\n")


    unsubscribe_aruco()

    print(id()+": Camera disengaged.")

    print(id()+": Initiating gyro re-calibration.")
    pub_reset.publish(Empty()) # note there is some delay between publish and calibration starting

    print(id()+": Waiting eight seconds for camera CPU wind down and gyro calibration...")
    time.sleep(8)

    current_yaw = get_yaw() # also serves purpose of blocking until calibration finished
    print(id()+": Re-calibrated gyro states current yaw as", current_yaw)

    print(id()+": Commanding PID to turn back to original yaw", original_yaw)
    pub_pid_yaw.publish(original_yaw)
    pub_pid_enabled.publish(True)

    turn_start = time.time()
    time.sleep(3)

    while abs(ang_diff(original_yaw, get_yaw())) > 2 and time.time()-turn_start < 5:
        # print(id()+": Angle to target:", ang_diff(original_yaw, get_yaw()))
        time.sleep(0.1)

    pub_pid_enabled.publish(False)
    set_turn_velocity(0.0)
    time.sleep(0.5)
    set_turn_velocity(0.0)

    remaining_time = 25 - (time.time()-start)
    if remaining_time < 0:
        remaining_time = 0
    print("Waiting remaining time:", remaining_time)
    time.sleep(remaining_time)
    
    print(id()+": Check-in complete. Current yaw", get_yaw(), "vs original", original_yaw)
    found_marker = None

    return state



def turn_with_pid(yaw, duration):
    pub_pid_yaw.publish(yaw)
    pub_pid_enabled.publish(True)

    time.sleep(duration-0.5)

    pub_pid_enabled.publish(False)
    set_turn_velocity(0.0)
    time.sleep(0.5)
    set_turn_velocity(0.0)



def synchronization_turn(direction):
    print("INITIATING OWN SYNC GESTURE...")

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
    
    print("SYNC GESTURE COMPLETE.")


def watch_for_synchronization(direction, bail_on_reset=False):

    subscribe_aruco()

    target_start_yaw = None
    last_measurement = None
    stage_measurement = None
    stage = 0

    last = time.time()

    print("WATCHING FOR SYNC GESTURE...")

    time_to_stage_1 = 0

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
            # print(diff)

            if stage == 0: # stationary
                if diff - last_measurement >= 10:
                    stage = 1
                    stage_measurement = last_measurement = diff
                    print("  SYNC STAGE 1: RIGHT")
                    time_to_stage_1 = time.time() - last
                    last = time.time()
            elif stage == 1:  # turning to the right
                if diff - last_measurement >= 2:
                    last_measurement = diff
                    last = time.time()
                if diff - stage_measurement <= -10:
                    stage = 2
                    stage_measurement = last_measurement = diff
                    print("  SYNC STAGE 2: LEFT")
            elif stage == 2:  # turning to the left
                if diff - last_measurement <= -2:
                    last_measurement = diff
                    last = time.time()
                if diff - stage_measurement >= 10:
                    stage = 3
                    stage_measurement = last_measurement = diff
                    print("  SYNC STAGE 3: RIGHT")
            elif stage == 3:  # turning to the right
                if diff - last_measurement >= 2:
                    last_measurement = diff
                    last = time.time()
                if diff - stage_measurement <= -10:
                    stage = 4
                    stage_measurement = last_measurement = diff
                    last = time.time()
                    print("  SYNC STAGE 4: LEFT")
            elif stage == 4:  # turning to the left
                if diff - last_measurement <= -2:
                    last_measurement = diff
                    last = time.time()
                if diff - stage_measurement >= 10:
                    stage = 0
                    print("  SYNC LOST - RESET")
                    last = time.time()
                    target_start_yaw = angle
                    last_measurement = 0
                elif (time.time() - last) >= 2:
                    print("  SYNC OBTAINED.")
                    break

            if (time.time() - last) >= 10: # reset
                print("  SYNC RESET")
                last = time.time()
                target_start_yaw = angle
                last_measurement = 0
                if bail_on_reset:
                    unsubscribe_aruco()
                    return False, None

        time.sleep(0.1)

    unsubscribe_aruco()
    return True, time_to_stage_1


def sync(agent_id):

    # agent 0: <-LAG0-><-- RX_SYNC --><-2s-><-- TX_SYNC --><-       10s       -><SYNCED>
    # agent 1: <-- TX_SYNC -->              <-LAG1-><-- RX_SYNC --><-2s-><-ADJ-><SYNCED>
    #                         <-LAG0-><-2s->
    #                         <-      DELAY       ->
    # DELAY = LAG0 + LAG1 + 2
    # assume LAG0 = LAG1
    # LAG = (DELAY-2)/2

    if agent_id == 0:
        turn_with_pid(original_yaw-90, 5) # for pid, right turn is negative

        watch_for_synchronization(-1, bail_on_reset=False)
        synchronization_turn(-1)
        time.sleep(10)

        turn_with_pid(original_yaw, 5)

    else:
        turn_with_pid(original_yaw+90, 5)

        while True:
            synchronization_turn(-1)
            sync_end = time.time()

            synced, delay = watch_for_synchronization(-1, bail_on_reset=True)
            if synced:
                # other agent picked up gesture and did doing own gesture
                lag = (delay-2)/2.0
                adjust = max(10 - (lag+2), 0)
                time.sleep(adjust)
                break
            
            # sync not picked up, repeat
            time.sleep(10)

        turn_with_pid(original_yaw, 5)
    


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
  
def drive_straight():
    tile_size = 0.2286
    speed = 0.2
    set_velocity(speed, 0.0)
    time.sleep(5 * tile_size / speed)
    set_velocity(0.0, 0.0)

def compute_drive_params(x, y, speed):
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
            if r == abs(y):
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

    # time_to_travel = abs(distance / speed)

    # motor cannot jump to certain speeds immediately so needs a ramp up period
    num_steps = int(math.ceil(abs(speed) / LIN_VEL_STEP_SIZE))
    ramp_up_time = num_steps * STEP_TIME

    # 0.01*0.01 + (1+2+3+...+t)
    distance_ramp_up = LIN_VEL_STEP_SIZE * STEP_TIME * num_steps*(num_steps+1) / 2.0

    if distance_ramp_up > distance:
        # (n^2+n) * step/2 = d
        # n^2 + n - 2d/step = 0
        # n = -1 + sqrt(1 + 4*2d/step) / 2
        num_steps = int(-1 + math.sqrt(1 + 4 * 2 * distance / LIN_VEL_STEP_SIZE) / 2)
        print(id()+": Not enough distance - cannot ramp up to", speed)
        speed = num_steps * LIN_VEL_STEP_SIZE
        print(id()+": Ramping up to", speed, "instead")
        distance_ramp_up = LIN_VEL_STEP_SIZE * num_steps*(num_steps+1) / 2.0

    time_to_travel_remaining = abs((distance-distance_ramp_up) / speed)
    time_to_travel = ramp_up_time + time_to_travel_remaining

    angular_velocity = angle / time_to_travel

    turn_speed = 0.5
    time_to_turn = min(1.0, abs(angle / turn_speed))

    # print(distance, num_steps, angle, r, speed, time_to_travel, angular_velocity)
    # print(num_steps, ramp_up_time, time_to_travel_remaining, time_to_travel)
    # print(distance, distance_ramp_up)

    return num_steps, time_to_travel_remaining, time_to_travel, angle, angular_velocity, time_to_turn

def drive_arc(x, y, speed):

    num_steps, time_to_travel_remaining, time_to_travel, angle, angular_velocity, time_to_turn = compute_drive_params(x, y, speed)    

    # ramp up
    sp = 0
    for i in range(num_steps):
        sp += math.copysign(LIN_VEL_STEP_SIZE, speed)
        set_velocity(sp, angular_velocity)
        time.sleep(STEP_TIME)

    set_velocity(speed, angular_velocity)
    time.sleep(time_to_travel_remaining)
    set_velocity(0.0, 0.0)

    # turn back to forward
    
    if time_to_turn > 0:
        turn_speed = -angle / time_to_turn
        
        set_velocity(0.0, turn_speed)
        time.sleep(time_to_turn)
        set_velocity(0.0, 0.0)

    return time_to_travel + time_to_turn


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


def execute_action(action):
    global displacement
    # print(id()+": Executing action", action.name)
    # drive_arc(action.value[0], action.value[1], 0.1)
    print(id()+": Executing action", action)
    print(id()+":",ACTIONS[action][0], ACTIONS[action][1])

    x = ACTIONS[action][0]
    y = ACTIONS[action][1]

    displacement = (displacement[0] + x, displacement[1] + y)

    return drive_arc(x, y, EXECUTION_SPEED)


def policyActionQuery(time, state, policy, checkin_period):
    macro_action = policy[state]
    time_within_period = time % checkin_period
    return macro_action[time_within_period], macro_action

def getPolicy(grid, mdp, start_state, discount, discount_checkin, checkin_period):
    _, policy, _, compMDP = run(grid, mdp, discount, start_state, 
                checkin_period=checkin_period, doBranchAndBound=False, 
                drawPolicy=False, drawIterations=False, 
                outputPrefix="", doLinearProg=True, 
                bnbGreedy=-1, doSimilarityCluster=False, 
                simClusterParams=None, outputDir="/home/patty/output")
    
    # print(policy)
    # policy = {(0, 0): ('DOUBLE-LEFT',), (1, 0): ('DOUBLE-LEFT',), (2, 0): ('DOUBLE-LEFT',), (3, 0): ('DOUBLE-LEFT',), (4, 0): ('RIGHT-LEFT',), (5, 0): ('RIGHT-DOUBLE',), (6, 0): ('RIGHT-DOUBLE',), (7, 0): ('RIGHT-DOUBLE',), (8, 0): ('RIGHT-DOUBLE',), (0, 1): ('DOUBLE-LEFT',), (1, 1): ('DOUBLE-LEFT',), (2, 1): ('DOUBLE-LEFT',), (3, 1): ('DOUBLE-LEFT',), (4, 1): ('FORWARD-LEFT',), (5, 1): ('RIGHT-DOUBLE',), (6, 1): ('RIGHT-DOUBLE',), (7, 1): ('RIGHT-DOUBLE',), (8, 1): ('RIGHT-DOUBLE',), (0, 2): ('DOUBLE-FORWARD',), (1, 2): ('DOUBLE-FORWARD',), (2, 2): ('DOUBLE-FORWARD',), (3, 2): ('DOUBLE-FORWARD',), (4, 2): ('DOUBLE-DOUBLE',), (5, 2): ('FORWARD-DOUBLE',), (6, 2): ('FORWARD-DOUBLE',), (7, 2): ('FORWARD-DOUBLE',), (8, 2): ('FORWARD-DOUBLE',), (0, 3): ('DOUBLE-RIGHT',), (1, 3): ('DOUBLE-RIGHT',), (2, 3): ('DOUBLE-RIGHT',), (3, 3): ('DOUBLE-RIGHT',), (4, 3): ('FORWARD-RIGHT',), (5, 3): ('LEFT-DOUBLE',), (6, 3): ('LEFT-DOUBLE',), (7, 3): ('LEFT-DOUBLE',), (8, 3): ('LEFT-DOUBLE',), (0, 4): ('DOUBLE-LEFT',), (1, 4): ('DOUBLE-LEFT',), (2, 4): ('DOUBLE-LEFT',), (3, 4): ('LEFT-RIGHT',), (4, 4): ('LEFT-RIGHT',), (5, 4): ('LEFT-RIGHT',), (6, 4): ('LEFT-DOUBLE',), (7, 4): ('LEFT-DOUBLE',), (8, 4): ('LEFT-DOUBLE',), (0, 5): ('DOUBLE-LEFT',), (1, 5): ('DOUBLE-LEFT',), (2, 5): ('DOUBLE-LEFT',), (3, 5): ('DOUBLE-LEFT',), (4, 5): ('FORWARD-LEFT',), (5, 5): ('RIGHT-DOUBLE',), (6, 5): ('RIGHT-DOUBLE',), (7, 5): ('RIGHT-DOUBLE',), (8, 5): ('RIGHT-DOUBLE',), (0, 6): ('DOUBLE-FORWARD',), (1, 6): ('DOUBLE-FORWARD',), (2, 6): ('DOUBLE-FORWARD',), (3, 6): ('DOUBLE-FORWARD',), (4, 6): ('DOUBLE-DOUBLE',), (5, 6): ('FORWARD-DOUBLE',), (6, 6): ('FORWARD-DOUBLE',), (7, 6): ('FORWARD-DOUBLE',), (8, 6): ('FORWARD-DOUBLE',), (0, 7): ('DOUBLE-RIGHT',), (1, 7): ('DOUBLE-RIGHT',), (2, 7): ('DOUBLE-RIGHT',), (3, 7): ('DOUBLE-RIGHT',), (4, 7): ('FORWARD-RIGHT',), (5, 7): ('LEFT-DOUBLE',), (6, 7): ('LEFT-DOUBLE',), (7, 7): ('LEFT-DOUBLE',), (8, 7): ('LEFT-DOUBLE',), (0, 8): ('DOUBLE-RIGHT',), (1, 8): ('DOUBLE-RIGHT',), (2, 8): ('DOUBLE-RIGHT',), (3, 8): ('DOUBLE-RIGHT',), (4, 8): ('LEFT-RIGHT',), (5, 8): ('LEFT-DOUBLE',), (6, 8): ('LEFT-DOUBLE',), (7, 8): ('LEFT-DOUBLE',), (8, 8): ('LEFT-DOUBLE',)}
    # policy = {(0, 0): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-FORWARD'), (1, 0): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-RIGHT'), (2, 0): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-DOUBLE'), (3, 0): ('DOUBLE-LEFT', 'FORWARD-LEFT', 'DOUBLE-DOUBLE'), (4, 0): ('RIGHT-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 0): ('RIGHT-DOUBLE', 'FORWARD-LEFT', 'DOUBLE-DOUBLE'), (6, 0): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'DOUBLE-DOUBLE'), (7, 0): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'LEFT-DOUBLE'), (8, 0): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'FORWARD-DOUBLE'), (0, 1): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 1): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-RIGHT'), (2, 1): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 1): ('DOUBLE-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 1): ('FORWARD-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 1): ('RIGHT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 1): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 1): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'LEFT-DOUBLE'), (8, 1): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 2): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 2): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-RIGHT'), (2, 2): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 2): ('DOUBLE-FORWARD', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 2): ('DOUBLE-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 2): ('FORWARD-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 2): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 2): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'LEFT-DOUBLE'), (8, 2): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 3): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 3): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-RIGHT'), (2, 3): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 3): ('DOUBLE-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 3): ('FORWARD-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 3): ('LEFT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 3): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 3): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'LEFT-DOUBLE'), (8, 3): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 4): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-FORWARD'), (1, 4): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-LEFT'), (2, 4): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-DOUBLE'), (3, 4): ('RIGHT-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (4, 4): ('RIGHT-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 4): ('RIGHT-LEFT', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (6, 4): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'DOUBLE-DOUBLE'), (7, 4): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'RIGHT-DOUBLE'), (8, 4): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'FORWARD-DOUBLE'), (0, 5): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 5): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-LEFT'), (2, 5): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 5): ('DOUBLE-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 5): ('FORWARD-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 5): ('RIGHT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 5): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 5): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'RIGHT-DOUBLE'), (8, 5): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 6): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 6): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-LEFT'), (2, 6): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 6): ('DOUBLE-FORWARD', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 6): ('DOUBLE-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 6): ('FORWARD-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 6): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 6): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'RIGHT-DOUBLE'), (8, 6): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 7): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 7): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-LEFT'), (2, 7): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 7): ('DOUBLE-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 7): ('FORWARD-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 7): ('LEFT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 7): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 7): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'RIGHT-DOUBLE'), (8, 7): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 8): ('DOUBLE-RIGHT', 'DOUBLE-RIGHT', 'DOUBLE-FORWARD'), (1, 8): ('DOUBLE-RIGHT', 'DOUBLE-RIGHT', 'DOUBLE-LEFT'), (2, 8): ('DOUBLE-RIGHT', 'DOUBLE-RIGHT', 'DOUBLE-DOUBLE'), (3, 8): ('DOUBLE-RIGHT', 'FORWARD-RIGHT', 'DOUBLE-DOUBLE'), (4, 8): ('LEFT-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 8): ('LEFT-DOUBLE', 'FORWARD-RIGHT', 'DOUBLE-DOUBLE'), (6, 8): ('LEFT-DOUBLE', 'LEFT-DOUBLE', 'DOUBLE-DOUBLE'), (7, 8): ('LEFT-DOUBLE', 'LEFT-DOUBLE', 'RIGHT-DOUBLE'), (8, 8): ('LEFT-DOUBLE', 'LEFT-DOUBLE', 'FORWARD-DOUBLE')}

    # return lambda time, state: policyActionQuery(time, state, policy, checkin_period)
    return policy


def sepToState(sepX, sepY, center_state):
    sepX = int(round(sepX/STATE_SCALE_FACTOR))
    sepY = -int(round(sepY/STATE_SCALE_FACTOR))
    return (center_state[0] + sepX, center_state[1] + sepY)

def convert_action(action):
    a = ACTIONS[action]
    return (int(a[0]/STATE_SCALE_FACTOR), int(-a[1]/STATE_SCALE_FACTOR))

def execute_policy(policy, center_state):
    global displacement
    displacement = (0,0)

    print(id()+": EXECUTING POLICY.")

    for i in range(2):

        state = checkin()#(1, 4)
        print("\n"+id()+": CHECKIN",i+1," | STATE: ALONG =", state[0], "CROSS = ", state[1])
        
        state = sepToState(state[0], state[1], center_state)

        joint_macro_action = policy[state]
        print(id()+": CHECKIN",i+1," | JOINT MACRO ACTION", joint_macro_action)

        my_time = 0
        companion_time = 0

        for j in range(len(joint_macro_action)):
            joint_action = joint_macro_action[j]
            print(id()+":   JOINT ACTION", joint_action)

            spl = joint_action.split("-")
            action = spl[agent_id]
            print(id()+":   INDIVIDUAL ACTION", action)

            my_time += execute_action(action)

            companion_action = spl[0 if agent_id == 1 else 1]
            cx = ACTIONS[companion_action][0]
            cy = ACTIONS[companion_action][1]
            _, _, time_to_travel, _, _, time_to_turn = compute_drive_params(cx, cy, EXECUTION_SPEED)
            companion_time += time_to_travel + time_to_turn
        

        time_to_wait = companion_time - my_time
        if time_to_wait > 0: # companion's actions take longer, wait for them
            print(id()+":   Waiting", time_to_wait, "for companion to finish execution")
            time.sleep(time_to_wait)

    final_state = checkin()

    # check reward
    print(id()+":  RETURNING TO START")
    drive_arc(-displacement[0], -displacement[1], -0.2)
    print(id()+":  COMPLETE.")


def run_mdp():
    maxSeparation = 6#4
    desiredSeparation = 2
    moveProb = 0.95#0.9

    actions_convert = {k:convert_action(k) for k in ACTIONS}

    grid, mdp, start_state = formationMDP(
        formation1_actions=actions_convert,
        maxSeparation = maxSeparation/STATE_SCALE_FACTOR, 
        desiredSeparation = desiredSeparation/STATE_SCALE_FACTOR, 
        moveProb = moveProb, 
        wallPenalty = -10, 
        movePenalty = 0, 
        collidePenalty = -100, 
        desiredStateReward=5)

    discount = math.sqrt(0.99)
    discount_checkin = discount

    centerInd = int(len(grid) / 2)
    center_state = (centerInd, centerInd)

    checkin_period = 2
    policy = getPolicy(grid, mdp, start_state, discount, discount_checkin, checkin_period)

    # initial_sep = np.array([0, -1])

    print(id()+": Waiting for start command...")
    wait_for_start()
    print(id()+": Received start command.")

    execute_policy(policy, center_state)




if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_control')

    agent_id = rospy.get_param('~agent_id')

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pub_pid_yaw = rospy.Publisher('/pid/target_yaw', Float64, queue_size=1, latch=True)
    pub_pid_enabled = rospy.Publisher('/pid/enabled', Bool, queue_size=1, latch=True)

    pub_reset = rospy.Publisher('/reset', Empty, queue_size=1)

    # runFormationAnimation()
    # run_mdp()

    # if True:
    #     exit(0)

    pub_pid_enabled.publish(False)
    set_velocity(0.0, 0.0)
    print(id()+": Holding one second for PID to halt...")
    time.sleep(1)
    set_velocity(0.0, 0.0)

    print(id()+": Obtaining current yaw...")
    original_yaw = get_yaw()
    print(id()+": Started at yaw", original_yaw,".")

    run_mdp()