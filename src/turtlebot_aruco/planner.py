#!/usr/bin/env python3

import rospy 
from std_msgs.msg import String
import math
import json

import socket
import os

from turtlebot_aruco.mdp.formationAnimation import *
from turtlebot_aruco.common import *


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
    
    # return lambda time, state: policyActionQuery(time, state, policy, checkin_period)
    
    return policy


def convert_action(action):
    a = ACTIONS[action]
    return (int(a[0]/STATE_SCALE_FACTOR), int(-a[1]/STATE_SCALE_FACTOR))


def run_mdp():
    actions_convert = {k:convert_action(k) for k in ACTIONS}

    grid, mdp, start_state = formationMDP(
        formation1_actions=actions_convert,
        maxSeparation = MAX_SEPARATION, 
        desiredSeparation = DESIRED_SEPARATION, 
        moveProb = MOVE_FORWARD_PROBABILITY, 
        wallPenalty = -10, 
        movePenalty = 0, 
        collidePenalty = -100, 
        desiredStateReward=5)

    discount = math.sqrt(0.99)
    discount_checkin = discount

    checkin_period = 2
    policy = getPolicy(grid, mdp, start_state, discount, discount_checkin, checkin_period)

    print("Planning complete, publishing policy.")

    # print(policy)
    s = json.dumps(policyToJsonFriendly([policy]), indent=4)
    print(s)

    with open("policy.txt", 'w') as file:
        file.write(s)

    return s


def load_policy():

    if not os.path.isfile("policy.txt"):
        print("Policy not generated yet.")
        return None

    with open("policy.txt", 'r') as file:
        s = file.read()
        return s

    return None


def sendMessage(sock, message):
    sock.send(len(message).to_bytes(2, 'big', signed=False))
    sock.send(message.encode())


def send_plan(plan, host, port):
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    print("Connecting to forwarder at " + host + ":" + str(port) + "...")
    sock.connect((host, port))
    print("Connected!")

    print("Sending policy to forwarder...")
    
    sendMessage(sock, plan)

    print("Policy sent!")

    sock.close()
    

def forward(plan, port1, port2):
    send_plan(plan, "127.0.0.1", port1)
    send_plan(plan, "127.0.0.1", port2)


if __name__=="__main__":
    rospy.init_node('turtlebot_mdp_planner')

    mode = rospy.get_param('~mode')
    
    port1 = rospy.get_param('~port1')
    port2 = rospy.get_param('~port2')

    if mode == 'generate':
        forward(run_mdp(), port1, port2)
    elif mode == 'load':
        forward(load_policy(), port1, port2)
    else:
        print("Unknown mode.")