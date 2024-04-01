import json
import os
from turtlebot_aruco.common import *
from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_multi_step_action, plot_next_state_distribution, plot_grid_world_mdp, plot_grid_world_blind_drive, plot_grid_world_policy


c = 1.5
t = 0.75

with open(f"output/C-{c}_T{t}_policy-raw.json", 'r') as file:
    s = file.read()
    policy = jsonFriendlyToPolicy2(json.loads(s))[0]

policy_for_plotting = {k: (v, (-1,)) for k, v in policy.items()}

policy_length = {
    state: len(action) for state, action in policy.items()
}


max_obs_time_horizon = 3#2
vmin=1
vmax=max_obs_time_horizon+1

fig = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, vmin=vmin, vmax=vmax)
fig.savefig(f"output/C-{c}_T{t}_LEN.pdf", bbox_inches='tight')



with open(f"output/C-{c}_T{t}_state-values.json", 'r') as file:
    s = file.read()
    state_values = jsonFriendlyToValues2(json.loads(s))

fig2 = plot_grid_world_mdp(policy_for_plotting, state_values, policy_linewidth=.5, colormap='RdYlBu', plot_colorbar=True, policy_color="#FFFFFF")
# fig.set_size_inches(10.5, 10.5)
fig2.savefig(f"output/C-{c}_T{t}_VALUES.pdf", bbox_inches='tight')

