#!/usr/bin/env python3

# import rospy 
# from std_msgs.msg import String
import numpy as np
import math
import json

import socket
import os

# from turtlebot_aruco.mdp.formationAnimation import *
from turtlebot_aruco.common import *
from turtlebot_aruco.mdp_schedule.formation_schedule import formationPolicy


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


def oneStepDelay(discount_factor, multi_step_rendezvous):
    ####
    base_discount_factor = discount_factor#0.95
    variable_discount_factor = {}
    for state in multi_step_rendezvous.transitions.keys():
        variable_discount_factor[state] = {}
        for action in multi_step_rendezvous.transitions[state].keys():
            variable_discount_factor[state][action] =  0.9 * base_discount_factor**len(action) + 0.1 * base_discount_factor**(len(action)+1) # Expected check-in time is one step _after_ we would like

    return variable_discount_factor
    ####


def delayMap(delay_map, discount_factor, multi_step_rendezvous):
    ####
    base_discount_factor = discount_factor#0.95
    variable_discount_factor = {}
    for state in multi_step_rendezvous.transitions.keys():
        variable_discount_factor[state] = {}
        for action in multi_step_rendezvous.transitions[state].keys():
            delay = delay_map[state]
            variable_discount_factor[state][action] = base_discount_factor**(len(action)+delay) # Expected check-in time is one step _after_ we would like

    return variable_discount_factor
    ####

def delayMapFunc(delay_map):
    return lambda discount_factor, multi_step_rendezvous: delayMap(delay_map, discount_factor, multi_step_rendezvous)

def pregenDelayMap():
    delay_map = {(0, 0): 3.1975548028945924, (1, 0): 3.590383017063141, (2, 0): 3.9832112312316896, (3, 0): 3.512261176109314, (4, 0): 3.0413111209869386, (5, 0): 1.5851098775863648, (6, 0): 0.12890863418579102, (7, 0): 0.12787959575653077, (8, 0): 0.12711397409439087, (9, 0): 0.12634835243225098, (10, 0): 0.668155574798584, (11, 0): 0.668155574798584, (12, 0): 0.668155574798584, (0, 1): 3.1975548028945924, (1, 1): 3.590383017063141, (2, 1): 3.9832112312316896, (3, 1): 3.512261176109314, (4, 1): 3.0413111209869386, (5, 1): 1.5851098775863648, (6, 1): 0.12890863418579102, (7, 1): 0.12787959575653077, (8, 1): 0.12711397409439087, (9, 1): 0.12634835243225098, (10, 1): 0.668155574798584, (11, 1): 0.668155574798584, (12, 1): 0.668155574798584, (0, 2): 3.1975548028945924, (1, 2): 3.590383017063141, (2, 2): 3.9832112312316896, (3, 2): 3.512261176109314, (4, 2): 3.0413111209869386, (5, 2): 1.5851098775863648, (6, 2): 0.12890863418579102, (7, 2): 0.12787959575653077, (8, 2): 0.12711397409439087, (9, 2): 0.12634835243225098, (10, 2): 0.668155574798584, (11, 2): 0.668155574798584, (12, 2): 0.668155574798584, (0, 3): 3.1975548028945924, (1, 3): 3.590383017063141, (2, 3): 3.9832112312316896, (3, 3): 3.512261176109314, (4, 3): 3.0413111209869386, (5, 3): 1.5851098775863648, (6, 3): 0.12890863418579102, (7, 3): 0.12787959575653077, (8, 3): 0.12711397409439087, (9, 3): 0.12634835243225098, (10, 3): 0.668155574798584, (11, 3): 0.668155574798584, (12, 3): 0.668155574798584, (0, 4): 3.1975548028945924, (1, 4): 3.590383017063141, (2, 4): 3.9832112312316896, (3, 4): 3.512261176109314, (4, 4): 3.0413111209869386, (5, 4): 1.5851098775863648, (6, 4): 0.12890863418579102, (7, 4): 0.12787959575653077, (8, 4): 0.12711397409439087, (9, 4): 0.12634835243225098, (10, 4): 0.668155574798584, (11, 4): 0.668155574798584, (12, 4): 0.668155574798584, (0, 5): 3.1975548028945924, (1, 5): 3.590383017063141, (2, 5): 3.9832112312316896, (3, 5): 3.4372767388820646, (4, 5): 2.89134224653244, (5, 5): 1.5101254403591156, (6, 5): 0.12890863418579102, (7, 5): 0.12787959575653077, (8, 5): 0.12711397409439087, (9, 5): 0.12634835243225098, (10, 5): 0.7201713164647419, (11, 5): 0.7201713164647419, (12, 5): 0.7201713164647419, (0, 6): 3.1975548028945924, (1, 6): 3.590383017063141, (2, 6): 3.9832112312316896, (3, 6): 3.362292301654816, (4, 6): 2.741373372077942, (5, 6): 1.4397080183029174, (6, 6): 0.13804266452789307, (7, 6): 0.2929579973220825, (8, 6): 0.20965317487716675, (9, 6): 0.12634835243225098, (10, 6): 0.7721870581309002, (11, 6): 0.7721870581309002, (12, 6): 0.7721870581309002, (0, 7): 3.1975548028945924, (1, 7): 3.014090859889984, (2, 7): 2.830626916885376, (3, 7): 2.786000144481659, (4, 7): 2.741373372077942, (5, 7): 1.4442750334739685, (6, 7): 0.14717669486999513, (7, 7): 0.45803639888763426, (8, 7): 0.2921923756599426, (9, 7): 0.12634835243225098, (10, 7): 0.8242027997970581, (11, 7): 0.8242027997970581, (12, 7): 0.8242027997970581, (0, 8): 3.1975548028945924, (1, 8): 3.014090859889984, (2, 8): 2.830626916885376, (3, 8): 2.786000144481659, (4, 8): 2.741373372077942, (5, 8): 1.4442750334739685, (6, 8): 0.14717669486999513, (7, 8): 0.45803639888763426, (8, 8): 0.2921923756599426, (9, 8): 0.12634835243225098, (10, 8): 0.8242027997970581, (11, 8): 0.8242027997970581, (12, 8): 0.8242027997970581, (0, 9): 3.1975548028945924, (1, 9): 3.014090859889984, (2, 9): 2.830626916885376, (3, 9): 2.786000144481659, (4, 9): 2.741373372077942, (5, 9): 1.4442750334739685, (6, 9): 0.14717669486999513, (7, 9): 0.45803639888763426, (8, 9): 0.2921923756599426, (9, 9): 0.12634835243225098, (10, 9): 0.8242027997970581, (11, 9): 0.8242027997970581, (12, 9): 0.8242027997970581, (0, 10): 3.1975548028945924, (1, 10): 3.014090859889984, (2, 10): 2.830626916885376, (3, 10): 2.786000144481659, (4, 10): 2.741373372077942, (5, 10): 1.4442750334739685, (6, 10): 0.14717669486999513, (7, 10): 0.45803639888763426, (8, 10): 0.2921923756599426, (9, 10): 0.12634835243225098, (10, 10): 0.8242027997970581, (11, 10): 0.8242027997970581, (12, 10): 0.8242027997970581, (0, 11): 3.1975548028945924, (1, 11): 3.014090859889984, (2, 11): 2.830626916885376, (3, 11): 2.786000144481659, (4, 11): 2.741373372077942, (5, 11): 1.4442750334739685, (6, 11): 0.14717669486999513, (7, 11): 0.45803639888763426, (8, 11): 0.2921923756599426, (9, 11): 0.12634835243225098, (10, 11): 0.8242027997970581, (11, 11): 0.8242027997970581, (12, 11): 0.8242027997970581, (0, 12): 3.1975548028945924, (1, 12): 3.014090859889984, (2, 12): 2.830626916885376, (3, 12): 2.786000144481659, (4, 12): 2.741373372077942, (5, 12): 1.4442750334739685, (6, 12): 0.14717669486999513, (7, 12): 0.45803639888763426, (8, 12): 0.2921923756599426, (9, 12): 0.12634835243225098, (10, 12): 0.8242027997970581, (11, 12): 0.8242027997970581, (12, 12): 0.8242027997970581}
    return delayMapFunc(delay_map)

def runWithParams(c, t, horizon):
    hS = "_H4" if horizon == 3 else ""

    variable_discount_factor_func = pregenDelayMap()#oneStepDelay

    policy, policy_raw, state_values, indifference = formationPolicy(gridSize=GRID_SIZE, 
        actionScale=STATE_SCALE_FACTOR, 
        checkin_reward=-c, transition_alpha=t, draw=False, max_obs_time_horizon=horizon, variable_discount_factor_func=variable_discount_factor_func)
    
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
    c = 1.5
    t = 0.75

    horizon = 2

    runWithParams(c, t, horizon)    
    # for c in np.linspace(0, 2, num=5):
    #     for t in np.linspace(0, 1, num=5):
    #         print(f"CHECKIN {c} TRANSITION {t}")
    #         runWithParams(c, t, horizon)

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
    run_test()
    # load_raw_policy()
    
    # rospy.init_node('turtlebot_mdp_planner')
    
    # mode = rospy.get_param('~mode')
    
    # port1 = rospy.get_param('~port1')
    # port2 = rospy.get_param('~port2')

    # # mode = 'generate'

    # # port1 = 0
    # # port2 = 0

    # if mode == 'generate':
    #     forward(run_mdp_schedule(), port1, port2)
    # elif mode == 'load':
    #     forward(load_policy(), port1, port2)
    # else:
    #     print("Unknown mode.")