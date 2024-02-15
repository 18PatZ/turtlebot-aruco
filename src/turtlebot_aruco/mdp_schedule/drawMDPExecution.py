import json
import os
from turtlebot_aruco.common import *
from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_multi_step_action, plot_next_state_distribution, plot_grid_world_mdp, plot_grid_world_blind_drive, plot_grid_world_policy


policy = None

with open("policy-raw.json", 'r') as file:
    s = file.read()
    policy = jsonFriendlyToPolicy2(json.loads(s))[0]

print(policy)


policy_for_plotting = {k: (v, (-1,)) for k, v in policy.items()}

policy_length = {
    state: len(action) for state, action in policy.items()
}



if not os.path.exists("draws"):
    os.makedirs("draws")


with open("states.json", 'r') as file:
    s = file.read()
    checkins = json.loads(s)

    i = 0
    for c in checkins:
        for t in c:
            # print(t)
            print(i)
            state = tuple(c[t]['State'])
            print(state, c[t]['Action'])

            
            fig = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, state_highlight=state)
            fig.savefig(f"draws/EXEC_{i}.png", bbox_inches='tight')

            i += 1
            break


# to_write.append({t: {
#     "State": [state[0], state[1]],
#     "Action": list(joint_macro_action)
# }})