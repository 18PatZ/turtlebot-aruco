#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import json
from turtlebot_aruco.common import *
from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_multi_step_action, plot_next_state_distribution, plot_grid_world_mdp, plot_grid_world_blind_drive, plot_grid_world_policy



def combine():
    rows = 5
    columns = 5
    plt.rcParams['axes.titlesize'] = 30

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(50,50))
    # fig.tight_layout()
    
    i = 1

    for c in np.linspace(0, 2, num=5):
        for t in np.linspace(0, 1, num=5):
            ax = fig.add_subplot(rows, columns, i)

            i += 1

            with open(f"output/C-{c}_T{t}_policy-raw.json", 'r') as file:
                s = file.read()
                policy = jsonFriendlyToPolicy2(json.loads(s))[0]

            policy_for_plotting = {k: (v, (-1,)) for k, v in policy.items()}

            policy_length = {
                state: len(action) for state, action in policy.items()
            }

            max_obs_time_horizon = 2
            vmin=1
            vmax=max_obs_time_horizon+1

            _f = plot_grid_world_blind_drive(None, policy_for_plotting, policy_length, vmin=vmin, vmax=vmax, ax=ax, legend=False)
            # _f.savefig(f"output/C-{c}_T{t}_LEN.pdf", bbox_inches='tight')

            ax.title.set_text((f"CHECK-IN COST: {c} \n TRANSITION ERROR: {t}"))

            ax.axis('off')

    plt.setp(axes, xticks=[], yticks=[])
    [ax.axis('off') for ax in fig.axes]
    fig.savefig(f"output/_COMBINED_LEN.pdf", bbox_inches='tight')
            
if __name__=="__main__":
    combine()