# from mdp import MDP
from turtlebot_aruco.mdp.lp import linearProgrammingSolve
from turtlebot_aruco.mdp_schedule.mdp import MDP

import matplotlib.pyplot as plt
# from turtlebot_aruco.mdp_schedule.periodic_observations_mdp import grid_world_mdp, add_terminal_state
# from turtlebot_aruco.mdp_schedule.periodic_observations_mdp_creation import create_recursive_actions_and_transitions

from turtlebot_aruco.mdp_schedule.so_mdp_creation import grow_transition_probabilities_and_rewards, make_existing_actions_composite

from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_multi_step_action, plot_next_state_distribution, plot_grid_world_mdp, plot_grid_world_blind_drive, plot_grid_world_policy

import collections

# from mdp import MDP, value_iteration, state_action_value_iteration


import time
import math
import os


def one_step_two_rover_mdp(
    x_steps,
    y_steps,
    actions_and_displacements,
    state_rewards,
    action_rewards,
    out_of_bounds_reward,
    terminal_states = {}
    ):
    """
    A helper function to build the two-rover MDP
    """
    states = [(x, y) for x in range(x_steps) for y in range(y_steps) if (x,y) not in terminal_states.keys()]
#     actions = [(along_track, cross_track) for along_track in (-1, 0, 1) for cross_track in (-1, 0, 1)]
    actions = actions_and_displacements.keys()
    transitions = {state: {action: collections.defaultdict(lambda : 0) for action in actions} for state in states}
    rewards = {state: {action: 0. for action in actions} for state in states}
    
    for state in states:
        for action, next_delta_state_dict in actions_and_displacements.items():
            _reward = state_rewards[state]
            for next_delta_state, next_delta_state_probability in next_delta_state_dict.items():
                _reward += action_rewards[action]*next_delta_state_probability
                next_state = state[0]+next_delta_state[0], state[1]+next_delta_state[1]
                if next_state not in states and next_state not in terminal_states.keys(): # Out of bounds!
                    _reward += out_of_bounds_reward*next_delta_state_probability
                    next_state = state
                transitions[state][action][next_state] += next_delta_state_probability
            rewards[state][action] += _reward
    
    return MDP(states=states, actions=actions, transitions=transitions, rewards=rewards,terminal_states=terminal_states)


def addDict(d, key, value):
    if key in d:
        d[key] += value
    else:
        d[key] = value

def formationMDP(
        gridSize,
        correction_inaccuracy,
        baseline_inaccuracy,
        actionScale,
        _action_reward,
        _motion_reward,
        terminal_states={}):
        
    # actions = [(along_track, cross_track) for along_track in (-1, 0, 1) for cross_track in (-2, -1, 0, 1, 2)]
    actions = [(int(along_track/actionScale), int(cross_track/actionScale)) for along_track in (-1, 0, 1) for cross_track in (-1, 0, 1)]
    a = int(1.0/actionScale)

    # correction_inaccuracy = 0.025 / 2
    # baseline_inaccuracy = 0.025 / 2

    # _action_reward = -1
    # _motion_reward = -.5

    transitions = {}
    action_rewards = {}
    for action in actions:
        outcomes = {}
        probability_mass = 1.

        c = correction_inaccuracy
        if action[0] != 0 and action[1] != 0:
            c /= 2
        
        action_rewards[action] = _action_reward
        if action[0] != 0:
            # addDict(outcomes, (action[0]+a, action[1]), correction_inaccuracy)
            # addDict(outcomes, (action[0]-a, action[1]), correction_inaccuracy)
            # addDict(outcomes, (action[0], a), correction_inaccuracy)
            # addDict(outcomes, (action[0], -a), correction_inaccuracy)
            # addDict(outcomes, (action[0]*2, a), correction_inaccuracy)
            # addDict(outcomes, (action[0]*2, -a), correction_inaccuracy)
            # probability_mass -= 4*correction_inaccuracy
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    addDict(outcomes, (action[0]+dx, dy), c)
                    probability_mass -= c
            action_rewards[action] += _motion_reward
            
        if action[1] != 0:
            # addDict(outcomes, (action[0], action[1]+a), correction_inaccuracy)
            # addDict(outcomes, (action[0], action[1]-a), correction_inaccuracy)
            # addDict(outcomes, (a, action[1]), correction_inaccuracy)
            # addDict(outcomes, (-a, action[1]), correction_inaccuracy)
            # addDict(outcomes, (a, action[1]*2), correction_inaccuracy)
            # addDict(outcomes, (-a, action[1]*2), correction_inaccuracy)
            # probability_mass -= 4*correction_inaccuracy
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    addDict(outcomes, (dx, action[1]+dy), c)
                    probability_mass -= c
            action_rewards[action] += _motion_reward
            
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx !=0 or dy !=0:
                    addDict(outcomes, (action[0]+dx, action[1]+dy), baseline_inaccuracy)
                    probability_mass -= baseline_inaccuracy
        outcomes[action] = probability_mass

        transitions[action] = outcomes

    # If driving along, smear along forward dict
    # If driving across, smear across that dict
    # IF driving across and along, smear in a rectangle?
        
    x_steps = gridSize
    y_steps = gridSize

    states = [(x, y) for x in range(x_steps) for y in range(y_steps)]
    median_state = ((x_steps-1)/2, (y_steps-1)/2)

    state_rewards = {}
    for state in states:
        r = -(abs(state[0]-median_state[0])**2 + 2*abs(state[1]-median_state[1])**2)
        # if state[1]-median_state[1] == -1 or state[1]-median_state[1] == -3:
        #     r += -80
        # if state[1]-median_state[1] == -2:
        #     r += -160
        state_rewards[state] = r

    mdp = one_step_two_rover_mdp(
        x_steps=x_steps,
        y_steps=y_steps,
        actions_and_displacements=transitions,
        action_rewards=action_rewards,
        state_rewards=state_rewards,
        out_of_bounds_reward=-2,#-2000#-1,
        terminal_states=terminal_states,
    )
    # mdp.terminals = terminal_states

    return mdp


def convertAction(actionScale, action):
    bot_left = ""
    bot_right = ""

    action = (int(action[0] * actionScale), int(action[1] * actionScale))

    # left is positive, therefore +1 means increase distance
    if action[1] == 1:
        bot_left = "LEFT"
        bot_right = "RIGHT"
    elif action[1] == -1: # close distance
        bot_left = "RIGHT"
        bot_right = "LEFT"
    elif action[1] == 0:
        if action[0] == 1:
            bot_right = "FORWARD"
        elif action[0] == -1:
            bot_left = "FORWARD"


    # forward is positive, therefore +1 means left bot should go forward more
    if action[0] == 0 or action[0] == 1: # both are side by side or left go more
        bot_left = "DOUBLE" + bot_left
    if action[0] == 0 or action[0] == -1: # both are side by side or left go less
        bot_right = "DOUBLE" + bot_right

    return bot_left + "-" + bot_right


def convertPolicy(actionScale, policy):
    new_policy = {k: tuple([convertAction(actionScale, a) for a in v]) for k, v in policy.items()}
    return new_policy


def getIndifference(values):
    policyBest = {}
    policySecond = {}
    indifference = {}

    for state in values:
        best_action = None
        second_best_action = None
        
        for action in values[state]:
            expected_value = values[state][action]
            if best_action is None or expected_value > values[state][best_action]:
                if best_action is not None and len(action) != len(best_action):
                    second_best_action = best_action
                best_action = action
            elif len(action) != len(best_action): # different length action
                if second_best_action is None or expected_value > values[state][second_best_action]:
                    second_best_action = action
        
        policyBest[state] = best_action
        policySecond[state] = second_best_action

        indifference[state] = values[state][best_action] - values[state][second_best_action]

    return policyBest, policySecond, indifference


def policyLen(policy):
    policy_length = {
        state: len(action) for state, action in policy.items()
    }
    return policy_length


def formationPolicy(gridSize = 13, actionScale = 1, 
                    checkin_reward = -0.2, transition_alpha = 0.5, draw = False):
    
    correction_inaccuracy = 0.025 * transition_alpha# / 2
    baseline_inaccuracy = 0.025 * transition_alpha# / 2
    
    mdp = formationMDP(
        gridSize = gridSize,
        correction_inaccuracy = correction_inaccuracy,
        baseline_inaccuracy = baseline_inaccuracy,
        actionScale = actionScale,
        _action_reward = -1,
        _motion_reward = -.5
    )

    # Multi step

    max_obs_time_horizon = 2
    discount_factor = .95
    illegal_action_reward = -1

    start = time.time()
    print("Starting composite")

    current_transitions = make_existing_actions_composite(mdp.transitions)
    current_rewards = make_existing_actions_composite(mdp.rewards)

    cumulative_transitions = {0: current_transitions}
    cumulative_rewards = {0: current_rewards}

    for t in range(1,max_obs_time_horizon+1):
        current_transitions, current_rewards = grow_transition_probabilities_and_rewards(
            current_transitions,
            current_rewards,
            mdp.transitions,
            mdp.rewards,
            discount_factor=discount_factor,
            illegal_action_rewarder=lambda s,a: illegal_action_reward,
        )
        cumulative_transitions[t] = current_transitions
        cumulative_rewards[t] = current_rewards


    flattened_transitions = {
        state:
        {
            action: cumulative_transitions[_t][state][action]
                for _t in cumulative_transitions.keys()
                for action in cumulative_transitions[_t][state]
        }
        for state in mdp.transitions.keys()
    }

    # checkin_reward = -0.2#.25

    print("\n\nRunning checkin_reward", checkin_reward,"\n\n")

    flattened_rewards = {
        state:
        {
            action: cumulative_rewards[_t][state][action]+checkin_reward
                for _t in cumulative_transitions.keys()
                for action in cumulative_transitions[_t][state]
        }
        for state in mdp.transitions.keys()
    }

    multi_step_rendezvous = MDP(
        states=mdp.states,
        actions=flattened_transitions[mdp.states[0]].keys(),
        transitions=flattened_transitions,
        rewards=flattened_rewards,
        terminal_states={}
    )

    print(time.time() - start, "to build MDP")

    print(len(multi_step_rendezvous.states), "states and", len(multi_step_rendezvous.actions), "actions")

    start = time.time()

    # 50s to build, 93s for value iteration, 73s for LP

    policy, state_values, values = linearProgrammingSolve(
        multi_step_rendezvous, 
        discount = discount_factor,#0.85, 
        is_negative=True,
        variable_discount_factor=True,
        restricted_action_set = None)

    print(time.time() - start)
    
    print(policy)

    conv_policy = convertPolicy(actionScale, policy)
    print("converted policy:")
    print(conv_policy)

    if draw:

        print("Drawing...")

        if not os.path.exists("output"):
            os.makedirs("output")

        vmin=1
        vmax=max_obs_time_horizon+1


        policyBest, policySecond, indifference = getIndifference(values)

        policy_for_plotting = {k: (v, (-1,)) for k, v in policy.items()}
        fig = plot_grid_world_mdp(policy_for_plotting, state_values, policy_linewidth=.5)
        fig.set_size_inches(10.5, 10.5)
        fig.savefig(f"output/C{checkin_reward}_T{transition_alpha}_POL.png")

        _f = plot_grid_world_blind_drive(mdp, policy_for_plotting, policyLen(policy), vmin=vmin, vmax=vmax)
        _f.savefig(f"output/C{checkin_reward}_T{transition_alpha}_LEN.png")



        # p1 = {k: (v, (-1,)) for k, v in policyBest.items()}
        # fig = plot_grid_world_mdp(p1, {k: v[policyBest[k]] for k, v in values.items()}, policy_linewidth=.5, plot_colorbar=True)
        # fig.set_size_inches(10.5, 10.5)
        # fig.savefig(f"output/C{checkin_reward}_T{transition_alpha}_POL_A.png")

        # _f = plot_grid_world_blind_drive(mdp, p1, policyLen(policyBest), vmin=vmin, vmax=vmax)
        # _f.savefig(f"output/C{checkin_reward}_T{transition_alpha}_LEN_A.png")


        # p2 = {k: (v, (-1,)) for k, v in policySecond.items()}
        # fig = plot_grid_world_mdp(p2, {k: v[policySecond[k]] for k, v in values.items()}, policy_linewidth=.5, plot_colorbar=True)
        # fig.set_size_inches(10.5, 10.5)
        # fig.savefig(f"output/C{checkin_reward}_T{transition_alpha}_POL_B.png")

        # _f = plot_grid_world_blind_drive(mdp, p2, policyLen(policySecond), vmin=vmin, vmax=vmax)
        # _f.savefig(f"output/C{checkin_reward}_T{transition_alpha}_LEN_B.png")

        
        indiff_vmax = None
        if max(indifference.values()) <= 1.0:
            indiff_vmax = 1.0
        
        fig = plot_grid_world_mdp(policy_for_plotting, indifference, policy_linewidth=.5, plot_colorbar=True, vmin=0, vmax=indiff_vmax)
        fig.set_size_inches(10.5, 10.5)
        fig.savefig(f"output/C{checkin_reward}_T{transition_alpha}_INDIFF.png")

        print("Drawing done.")

    return conv_policy, policy