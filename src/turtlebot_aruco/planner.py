#!/usr/bin/env python3

import rospy 
# from std_msgs.msg import String
import numpy as np
import math
import json

import socket
import os

# from turtlebot_aruco.mdp.formationAnimation import *
from turtlebot_aruco.common import *
# from turtlebot_aruco.mdp_schedule.formation_schedule import formationPolicy


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

    with open("policy.json", 'w') as file:
        file.write(s)

    return s

def run_mdp_schedule():
    policy, policy_raw = formationPolicy(gridSize=GRID_SIZE, 
            actionScale=STATE_SCALE_FACTOR, 
            checkin_reward=-1.5, transition_alpha=0.75, draw=True)

    print("Planning complete, publishing policy.")

    # print(policy)
    s = json.dumps(policyToJsonFriendly([policy]), indent=4)
    print(s)

    s2 = json.dumps(policyToJsonFriendly2([policy_raw]), indent=4)

    with open("policy.json", 'w') as file:
        file.write(s)
    with open("policy-raw.json", 'w') as file:
        file.write(s2)

    return s

def runWithParams(c, t, horizon):
    hS = "_H4" if horizon == 3 else ""

    policy, policy_raw, state_values, indifference = formationPolicy(gridSize=GRID_SIZE, 
        actionScale=STATE_SCALE_FACTOR, 
        checkin_reward=-c, transition_alpha=t, draw=False, max_obs_time_horizon=horizon)
    
    s2 = json.dumps(policyToJsonFriendly2([policy_raw]), indent=4)
    with open(f"output/C-{c}_T{t}{hS}_policy-raw.json", 'w') as file:
        file.write(s2)

    if state_values is not None:
        s3 = json.dumps(valuesToJsonFriendly2(state_values), indent=4)
        with open(f"output/C-{c}_T{t}{hS}_state-values.json", 'w') as file:
            file.write(s3)

    if indifference is not None:
        s3 = json.dumps(valuesToJsonFriendly2(indifference), indent=4)
        with open(f"output/C-{c}_T{t}{hS}_indifference.json", 'w') as file:
            file.write(s3)

def run_test():
    # c = 1.5
    # t = 0.75

    horizon = 2

    # runWithParams(c, t, horizon)    
    for c in np.linspace(0, 2, num=5):
        for t in np.linspace(0, 1, num=5):
            print(f"CHECKIN {c} TRANSITION {t}")
            runWithParams(c, t, horizon)

    # for c in np.linspace(0, 2, num=5):
    #     for t in np.linspace(0, 1, num=5):
    #         print(f"CHECKIN {c} TRANSITION {t}")
    #         policy, policy_raw = formationPolicy(gridSize=GRID_SIZE, 
    #                 actionScale=STATE_SCALE_FACTOR, 
    #                 checkin_reward=-c, transition_alpha=t, draw=False)
            
    #         s2 = json.dumps(policyToJsonFriendly2([policy_raw]), indent=4)
    #         with open(f"output/C-{c}_T{t}_policy-raw.json", 'w') as file:
    #             file.write(s2)

    # for c in np.linspace(0, 2, num=5):
    #     formationPolicy(gridSize=GRID_SIZE, 
    #                     actionScale=STATE_SCALE_FACTOR, 
    #                     checkin_reward=-c, transition_alpha=0.0, draw=True)
    # formationPolicy(gridSize=GRID_SIZE, 
    #     actionScale=STATE_SCALE_FACTOR, 
    #     checkin_reward=-2.0, transition_alpha=0.5, draw=True)

def load_raw_policy():

    if not os.path.isfile("policy-raw.json"):
        print("Policy not generated yet.")
        return None

    with open("policy-raw.json", 'r') as file:
        s = file.read()
        policy = jsonFriendlyToPolicy2(json.loads(s))[0]
        conv_policy = convertPolicy(STATE_SCALE_FACTOR, policy)
    
        s = json.dumps(policyToJsonFriendly([conv_policy]), indent=4)
        with open("policy.json", 'w') as file2:
            file2.write(s)
        return s

    return None

def load_policy():

    if not os.path.isfile("policy.json"):
        print("Policy not generated yet.")
        return None

    with open("policy.json", 'r') as file:
        s = file.read()
        return s

    return None


def sendMessage(sock, message):
    print("Message of length",len(message))
    sock.send(len(message).to_bytes(8, 'big', signed=False))
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
    if port1 > 0:
        send_plan(plan, "127.0.0.1", port1)
    if port2 > 0:
        send_plan(plan, "127.0.0.1", port2)


if __name__=="__main__":
    # run_test()
    # load_raw_policy()
    
    rospy.init_node('turtlebot_mdp_planner')
    
    mode = rospy.get_param('~mode')
    
    port1 = rospy.get_param('~port1')
    port2 = rospy.get_param('~port2')

    # mode = 'generate'

    # port1 = 0
    # port2 = 0

    if mode == 'generate':
        forward(run_mdp_schedule(), port1, port2)
    elif mode == 'load':
        forward(load_policy(), port1, port2)
    else:
        print("Unknown mode.")