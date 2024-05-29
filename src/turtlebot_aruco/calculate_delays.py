import json
import os
from turtlebot_aruco.common import *
from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_multi_step_action, plot_next_state_distribution, plot_grid_world_mdp, plot_grid_world_blind_drive, plot_grid_world_policy

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



c = 1.5
t = 0.75
max_obs_time_horizon = 2
hS = "_H4" if max_obs_time_horizon == 3 else ""

with open(f"output/C-{c}_T{t}{hS}_policy-raw.json", 'r') as file:
    s = file.read()
    policy = jsonFriendlyToPolicy2(json.loads(s))[0]

# policy_for_plotting = {k: (v, (-1,)) for k, v in policy.items()}
policy_for_plotting = {k: None for k, v in policy.items()}

delays = {}

with open(f"calib.json", 'r') as file:
    s = file.read()
    calibs = json.loads(s)

    for calib in calibs:
        s = calib["State"]
        state = (s[0], s[1])
        time = calib["Time"]
        
        delays[state] = time

x_min = float("inf")
x_max = -float("inf")
y_min = float("inf")
y_max = -float("inf")
patches = []
    
for state, _policy in policy.items():
    x_min=min(x_min, state[0])
    y_min=min(y_min, state[1])
    x_max=max(x_max, state[0])
    y_max=max(y_max, state[1])


def smooth(smoothed, delays, x, minV, maxV, stateFunc):

    left = []
    right = []
    
    for xp in range(int(minV), int(maxV)+1):
        if stateFunc(xp) in delays:
            if xp < x: 
                left.append(xp)
            elif xp > x:
                right.append(xp)

    if len(left) > 0 and len(right) > 0:
        x1 = left[-1]
        x2 = right[0]
    # elif len(left) > 1:
    #     x1 = left[-2]
    #     x2 = left[-1]
    elif len(left) > 0:
        x1 = x2 = left[-1]
    # elif len(right) > 1:
    #     x1 = right[0]
    #     x2 = right[1]
    elif len(right) > 0:
        x1 = x2 = right[0]
    else:
        return
    
    v1 = delays[stateFunc(x1)]
    v2 = delays[stateFunc(x2)]

    if x1 == x2:
        v = v1
    else:
        v = ((v2 - v1) / (x2 - x1)) * (x - x1) + v1

    smoothed[state] = v

smoothed = {}
for y in range(int(y_min), int(y_max)+1):
    for x in range(int(x_min), int(x_max)+1):
        state = (x, y)
        if state in delays:
            smoothed[state] = delays[state]
            continue
        smooth(smoothed, delays, y, y_min, y_max, lambda yp: (x, yp))

delays = smoothed
smoothed = {}
for y in range(int(y_min), int(y_max)+1):
    for x in range(int(x_min), int(x_max)+1):
        state = (x, y)
        if state in delays:
            smoothed[state] = delays[state]
            continue
        smooth(smoothed, delays, x, x_min, x_max, lambda xp: (xp, y))

converted = {
    state: max((delay - 20)/10, 0) for state, delay in smoothed.items()
}

print(converted)

vmin = float("inf")
vmax = -float("inf")

for state, delay in converted.items():
    vmin=int(min(vmin, delay))
    vmax=int(max(vmax, delay))

policy_length = {
    state: (converted[state] if state in converted else 0) for state, action in policy.items()
}


# fig = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, vmin=vmin, vmax=vmax, cmap=cmap)
# fig.savefig(f"output/C-{c}_T{t}{hS}_LEN.pdf", bbox_inches='tight')

fig = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, vmin=vmin, vmax=vmax)
fig.savefig(f"output/C-CALIB.pdf", bbox_inches='tight')
                
# fig = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, vmin=vmin, vmax=vmax, cmap=cmap, state_highlight=states[2], legend=False)
# fig.savefig(f"output/C-{c}_T{t}{hS}_LEN_CHK2.pdf", bbox_inches='tight')
