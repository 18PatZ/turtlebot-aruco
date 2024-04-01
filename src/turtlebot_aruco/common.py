#!/usr/bin/env python3

TILE_SIZE = 0.2286
STATE_SCALE_FACTOR = 1#0.5
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
    "DOUBLE": (2, 0), 
    "DOUBLELEFT": (2, 0.5),
    "DOUBLERIGHT": (2, -0.5),
    "FORWARD": (1, 0), 
    "LEFT": (1, 0.5),
    # "SLIGHT_LEFT": (1, 0.5), 
    "RIGHT": (1, -0.5)
    # "SLIGHT_RIGHT": (1, -0.5)
}


MAX_SEPARATION = 6 / STATE_SCALE_FACTOR
DESIRED_SEPARATION = 3.5 / STATE_SCALE_FACTOR
MOVE_FORWARD_PROBABILITY = 0.95#0.9

GRID_SIZE = int(MAX_SEPARATION * 2 + 1)
CENTER_INDEX = int(GRID_SIZE / 2)
CENTER_STATE = (CENTER_INDEX, CENTER_INDEX)

def sepToStateSep(sepX, sepY):
    sepX = int(round(sepX/STATE_SCALE_FACTOR))

    # for original, robot1 being on left (positive sep) means negative Y
    # sepY = -int(round(sepY/STATE_SCALE_FACTOR))
    # return (center_state[0] + sepX, center_state[1] + sepY)

    # for new, robot1 being on left (positive sep) means positive Y
    # also, state is error from desired separation not the actual separation
    sepY = int(round(sepY/STATE_SCALE_FACTOR - DESIRED_SEPARATION))
    return sepX, sepY


def ensureBounds(state):
    x = state[0]
    y = state[1]

    if x < 0:
        x = 0
    elif x >= GRID_SIZE:
        x = GRID_SIZE-1
    
    if y < 0:
        y = 0
    elif y >= GRID_SIZE:
        y = GRID_SIZE-1

    return (x, y)


def sepToState(sepX, sepY, center_state):
    sepX, sepY = sepToStateSep(sepX, sepY)
    return ensureBounds((center_state[0] + sepX, center_state[1] + sepY))


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



def actionSeqToJsonFriendly(action_seq):
    return [list(a) for a in action_seq]
def jsonFriendlyToActionSeq(action_seq):
    return tuple([tuple(a) for a in action_seq])

def policyToJsonFriendly2(policies):
    return [{stateTupleToStr(state): actionSeqToJsonFriendly(policy[state]) for state in policy} for policy in policies]
def jsonFriendlyToPolicy2(policies):
    return [{strToStateTuple(state): jsonFriendlyToActionSeq(policy[state]) for state in policy} for policy in policies]

def valuesToJsonFriendly2(values):
    return {stateTupleToStr(state): values[state] for state in values}
def jsonFriendlyToValues2(values):
    return {strToStateTuple(state): values[state] for state in values}