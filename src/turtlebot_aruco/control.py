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

found_marker = None
pub = None
pub_pid_yaw = None
pub_pid_enabled = None
original_yaw = None

sub = None
agent_id = None

class ACTION(Enum):
    FORWARD = (1,0)
    DOUBLE = (2,0)
    LEFT = (1, 1)
    RIGHT = (1, -1)
    DOUBLE_FW_LEFT = (2, 1)
    DOUBLE_FW_RIGHT = (2, -1)



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
    turn_with_pid(original_yaw + direction * 90, 5)

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
    time.sleep(3)
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

    print(id()+": Waiting ten seconds for camera CPU wind down and gyro calibration...")
    time.sleep(10)

    current_yaw = get_yaw() # also serves purpose of blocking until calibration finished
    print(id()+": Re-calibrated gyro states current yaw as", current_yaw)

    print(id()+": Commanding PID to turn back to original yaw", original_yaw)
    pub_pid_yaw.publish(original_yaw)
    pub_pid_enabled.publish(True)

    turn_start = time.time()
    time.sleep(5)

    while abs(ang_diff(original_yaw, get_yaw())) > 2 and time.time()-turn_start < 10:
        # print(id()+": Angle to target:", ang_diff(original_yaw, get_yaw()))
        time.sleep(0.1)

    pub_pid_enabled.publish(False)
    set_turn_velocity(0.0)
    time.sleep(0.5)
    set_turn_velocity(0.0)

    remaining_time = 35 - (time.time()-start)
    if remaining_time < 0:
        remaining_time = 0
    print("Waiting remaining time:", remaining_time)
    time.sleep(remaining_time)
    
    print(id()+": Check-in complete. Current yaw", get_yaw(), "vs original", original_yaw)
    found_marker = None



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
    LIN_VEL_STEP_SIZE = 0.01
    step_time = 0.01
    num_steps = int(math.ceil(abs(speed) / LIN_VEL_STEP_SIZE))
    ramp_up_time = num_steps * step_time

    # 0.01*0.01 + (1+2+3+...+t)
    distance_ramp_up = LIN_VEL_STEP_SIZE * step_time * num_steps*(num_steps+1) / 2.0

    if distance_ramp_up > distance:
        # (n^2+n) * step/2 = d
        # n^2 + n - 2d/step = 0
        # n = -1 + sqrt(1 + 4*2d/step) / 2
        num_steps = int(-1 + math.sqrt(1 + 4 * 2 * distance / LIN_VEL_STEP_SIZE) / 2)
        print("Not enough distance - cannot ramp up to", speed)
        speed = num_steps * LIN_VEL_STEP_SIZE
        print("Ramping up to", speed, "instead")
        distance_ramp_up = LIN_VEL_STEP_SIZE * num_steps*(num_steps+1) / 2.0

    time_to_travel_remaining = abs((distance-distance_ramp_up) / speed)
    time_to_travel = ramp_up_time + time_to_travel_remaining

    angular_velocity = angle / time_to_travel

    # print(distance, num_steps, angle, r, speed, time_to_travel, angular_velocity)
    # print(num_steps, ramp_up_time, time_to_travel_remaining, time_to_travel)
    # print(distance, distance_ramp_up)

    # ramp up
    sp = 0
    for i in range(num_steps):
        sp += math.copysign(LIN_VEL_STEP_SIZE, speed)
        set_velocity(sp, angular_velocity)
        time.sleep(step_time)

    set_velocity(speed, angular_velocity)
    time.sleep(time_to_travel_remaining)
    set_velocity(0.0, 0.0)

    # turn back to forward
    # turn_speed = math.copysign(0.1, -angular_velocity)
    # time_to_turn = -angle / turn_speed

    turn_speed = 0.5
    time_to_turn = min(1.0, abs(angle / turn_speed))

    if time_to_turn > 0:
        turn_speed = -angle / time_to_turn
        
        set_velocity(0.0, turn_speed)
        time.sleep(time_to_turn)
        set_velocity(0.0, 0.0)


def wait_for_start():
    msg = rospy.wait_for_message("/control/start", Int32)
    return msg.data


def checkin():
    print(id()+": Starting check-in...")

    if agent_id == 0:
        turn_direction = -1
    elif agent_id == 1:
        turn_direction = 1
    turn_to_marker(turn_direction)

def id():
    return "TB"+str(agent_id)

def execute_action(action):
    print("Executing action", action.name)
    drive_arc(action.value[0], action.value[1], 0.1)

def run():
    # checkin()

    execute_action(ACTION.DOUBLE)
    # execute_action(ACTION.RIGHT)
    
    checkin()

    drive_arc(-2, 0, -0.1)

    checkin()

    # execute_action(ACTION.DOUBLE_FW_LEFT)

    # checkin()


    # # print("Driving back...")
    # drive_arc(-5, 0, -0.1)
    # # print("Drive complete.")

    # checkin()


    # print("Driving arc...")
    # drive_arc(2, 2, 0.1)
    # print("Drive complete.")


if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_control')

    agent_id = rospy.get_param('~agent_id')

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    pub_pid_yaw = rospy.Publisher('/pid/target_yaw', Float64, queue_size=1, latch=True)
    pub_pid_enabled = rospy.Publisher('/pid/enabled', Bool, queue_size=1, latch=True)

    pub_reset = rospy.Publisher('/reset', Empty, queue_size=1)

    print("jo")
    runFormationAnimation()

    if True:
        exit(0)

    pub_pid_enabled.publish(False)
    set_velocity(0.0, 0.0)
    print(id()+": Holding one second for PID to halt...")
    time.sleep(1)
    set_velocity(0.0, 0.0)

    print(id()+": Obtaining current yaw...")
    original_yaw = get_yaw()
    print(id()+": Started at yaw", original_yaw,".")

    print(id()+": Waiting for start command...")
    wait_for_start()
    print(id()+": Received start command.")

    run()

    # print("Driving forward one...")
    # drive_straight()
    # print("Drive complete.")

    # set_velocity(0.1, 0)
    # time.sleep(TILE_SIZE)
    # set_velocity(0.2, 0)
    # time.sleep(TILE_SIZE)
    # remember wheels cant spin up to 0.2 immediately

    # sync(agent_id)

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

    


    # print("Driving arc...")
    # drive_arc(-2, -1, -0.1)
    # print("Drive complete.")

    # turn_to_marker(turn_direction)
