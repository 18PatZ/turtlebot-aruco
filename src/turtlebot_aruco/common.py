#!/usr/bin/env python3

TILE_SIZE = 0.2286
STATE_SCALE_FACTOR = 0.5
EXECUTION_SPEED = 0.1

LIN_VEL_STEP_SIZE = 0.01
STEP_TIME = 0.01

# class ACTION(Enum):
#     FORWARD = (1,0)
#     DOUBLE = (2,0)
#     LEFT = (1, 1)
#     RIGHT = (1, -1)
#     DOUBLE_FW_LEFT = (2, 1)
#     DOUBLE_FW_RIGHT = (2, -1)

ACTIONS = {
    "DOUBLE": (4, 0), 
    "FORWARD": (2, 0),
    "LEFT": (2, 1),
    # "SLIGHT_LEFT": (1, 0.5), 
    "RIGHT": (2, -1)
    # "SLIGHT_RIGHT": (1, -0.5)
}


MAX_SEPARATION = 6 / STATE_SCALE_FACTOR
DESIRED_SEPARATION = 2 / STATE_SCALE_FACTOR
MOVE_FORWARD_PROBABILITY = 0.95#0.9

GRID_SIZE = int(MAX_SEPARATION * 2 + 1)
CENTER_INDEX = int(GRID_SIZE / 2)
CENTER_STATE = (CENTER_INDEX, CENTER_INDEX)

def sepToState(sepX, sepY, center_state):
    sepX = int(round(sepX/STATE_SCALE_FACTOR))
    sepY = -int(round(sepY/STATE_SCALE_FACTOR))
    return (center_state[0] + sepX, center_state[1] + sepY)



def stateTupleToStr(tup):
    return str(tup[0]) + "-" + str(tup[1])

def strToStateTuple(stateStr):
    spl = stateStr.split("-")
    return (int(spl[0]), int(spl[1]))

def policyToJsonFriendly(policies):
    return [{stateTupleToStr(state): list(policy[state]) for state in policy} for policy in policies]

def jsonFriendlyToPolicy(policies):
    # return {strToStateTuple(state): tuple(policy[state]) for state in policy}
    return [{strToStateTuple(state): tuple(policy[state]) for state in policy} for policy in policies]
    # return [{strToStateTuple(state): policy[state] for state in policy} for policy in policies]
