import json
import os
from turtlebot_aruco.common import *
from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_multi_step_action, plot_next_state_distribution, plot_grid_world_mdp, plot_grid_world_blind_drive, plot_grid_world_policy

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

c = 1.5
t = 0.75

with open(f"output/C-{c}_T{t}_H4_policy-raw.json", 'r') as file:
    s = file.read()
    policy = jsonFriendlyToPolicy2(json.loads(s))[0]

policy_for_plotting = {k: (v, (-1,)) for k, v in policy.items()}

policy_length = {
    state: len(action) for state, action in policy.items()
}


max_obs_time_horizon = 3#2
vmin=1
vmax=max_obs_time_horizon+1

num_colors = vmax-vmin + 1
cmap = plt.get_cmap('cividis', num_colors - 1)
colors = list(cmap.colors)
print(colors)
new_color = [0.850556, 0.566949, 0.057568, 1.      ]
colors.append(new_color)
cmap = ListedColormap(colors, name='cividis_extended', N=num_colors)

# print(colors)
# fig = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, vmin=vmin, vmax=vmax, cmap=cmap)
# fig.savefig(f"output/C-{c}_T{t}_H4_LEN.pdf", bbox_inches='tight')



# with open(f"output/C-{c}_T{t}_state-values.json", 'r') as file:
#     s = file.read()
#     state_values = jsonFriendlyToValues2(json.loads(s))

# fig2 = plot_grid_world_mdp(policy_for_plotting, state_values, policy_linewidth=.5, colormap='RdYlBu', plot_colorbar=True, policy_color="#FFFFFF")
# # fig.set_size_inches(10.5, 10.5)
# fig2.savefig(f"output/C-{c}_T{t}_VALUES.pdf", bbox_inches='tight')



states = []
with open("states.json", 'r') as file:
    s = file.read()
    checkins = json.loads(s)

    i = 0
    for checkin in checkins:
        for t0 in checkin:
            # print(t)
            print(i)
            state = tuple(checkin[t0]['State'])
            print(state, checkin[t0]['Action'])

            if state not in states:
                states.append(state)

            i += 1
            break

fig = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, vmin=vmin, vmax=vmax, cmap=cmap, state_highlight=states)
fig.savefig(f"output/C-{c}_T{t}_H4_LEN_EXEC.pdf", bbox_inches='tight')