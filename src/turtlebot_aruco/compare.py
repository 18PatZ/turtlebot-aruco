import json
import os
from turtlebot_aruco.common import *
from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_multi_step_action, plot_next_state_distribution, plot_grid_world_mdp, plot_grid_world_blind_drive, plot_grid_world_policy


c = 1.5
t = 0.75
max_obs_time_horizon = 2#2
hS = "_H4" if max_obs_time_horizon == 3 else ""

with open(f"output/C-{c}_T{t}{hS}_policy-raw_unfixed.json", 'r') as file:
    s = file.read()
    policy1 = jsonFriendlyToPolicy2(json.loads(s))[0]

with open(f"output/C-{c}_T{t}{hS}_policy-raw.json", 'r') as file:
    s = file.read()
    policy2 = jsonFriendlyToPolicy2(json.loads(s))[0]

print("POLICY OVERALL")
for k in policy1:
    if policy1[k] != policy2[k]:
        print(k, policy1[k], "vs", policy2[k])

states = []
with open("states.json", 'r') as file:
    s = file.read()
    checkins = json.loads(s)

    i = 0
    for checkin in checkins:
        for t0 in checkin:
            # print(t)
            # print(i)
            state = tuple(checkin[t0]['State'])
            # print(state, checkin[t0]['Action'])

            if state not in states:
                states.append(state)

print("POLICY EXEC")
for k in states:
    print(k)
    if policy1[k] != policy2[k]:
        print(k, policy1[k], "vs", policy2[k])

