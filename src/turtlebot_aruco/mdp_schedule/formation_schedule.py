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


def one_step_two_rover_mdp(
    x_steps,
    y_steps,
    actions_and_displacements,
    state_rewards,
    action_rewards,
    out_of_bounds_reward,
    ):
    """
    A helper function to build the two-rover MDP
    """
    states = [(x, y) for x in range(x_steps) for y in range(y_steps)]
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
                if next_state not in states: # Out of bounds!
                    _reward += out_of_bounds_reward*next_delta_state_probability
                    next_state = state
                transitions[state][action][next_state] += next_delta_state_probability
            rewards[state][action] += _reward
        
    terminal_states = {}
    
    return MDP(states=states, actions=actions, transitions=transitions, rewards=rewards,terminal_states=terminal_states)


def formationMDP(gridSize, correction_inaccuracy, baseline_inaccuracy, 
                 actionScale,
                 _action_reward, _motion_reward):
    # actions = [(along_track, cross_track) for along_track in (-1, 0, 1) for cross_track in (-2, -1, 0, 1, 2)]
    actions = [(int(along_track/actionScale), int(cross_track/actionScale)) for along_track in (-1, 0, 1) for cross_track in (-1, 0, 1)]


    transitions = {}
    action_rewards = {}
    for action in actions:
        outcomes = {}
        probability_mass = 1.
        
        action_rewards[action] = _action_reward
        if action[0] != 0:
            dir = math.copysign(1, action[0])
            outcomes[(action[0]+dir, action[1])] = correction_inaccuracy
            outcomes[(action[0]-dir, action[1])] = correction_inaccuracy
            probability_mass -= 2*correction_inaccuracy
            action_rewards[action] += _motion_reward
            
        if action[1] != 0:
            dir = math.copysign(1, action[1])
            outcomes[(action[0], action[1]+dir)] = correction_inaccuracy
            outcomes[(action[0], action[1]-dir)] = correction_inaccuracy
            probability_mass -= 2*correction_inaccuracy
            action_rewards[action] += _motion_reward
            
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx !=0 or dy !=0:
                    outcomes[(action[0]+dx, action[1]+dy)] = baseline_inaccuracy
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
        state_rewards[state] = -(abs(state[0]-median_state[0]) + abs(state[1]-median_state[1]))

    terminal_states = {}

    mdp = one_step_two_rover_mdp(
        x_steps=x_steps,
        y_steps=y_steps,
        actions_and_displacements=transitions,
        action_rewards=action_rewards,
        state_rewards=state_rewards,
        out_of_bounds_reward=-1,

    )
    mdp.terminals = terminal_states

    return mdp


def convertAction(action):
    bot_left = ""
    bot_right = ""

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
        else:
            bot_left = "FORWARD"


    # forward is positive, therefore +1 means left bot should go forward more
    if action[0] == 0 or action[0] == 1: # both are side by side or left go more
        bot_left = "DOUBLE" + bot_left
    if action[0] == 0 or action[0] == -1: # both are side by side or left go less
        bot_right = "DOUBLE" + bot_right

    return bot_left + "-" + bot_right


def convertPolicy(policy):
    new_policy = {k: tuple([convertAction(a) for a in v]) for k, v in policy.items()}
    return new_policy



def formationPolicy(gridSize = 13, actionScale = 1, 
                    checkin_reward = -0.2, draw = False):
    
    mdp = formationMDP(
        gridSize = gridSize,
        correction_inaccuracy = 0.02,
        baseline_inaccuracy = 0.025,
        actionScale = 1,
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

    conv_policy = convertPolicy(policy)
    print("converted policy:")
    print(conv_policy)

    if draw:

        policy_for_plotting = {k: (v, (-1,)) for k, v in policy.items()}
        fig = plot_grid_world_mdp(policy_for_plotting, state_values, policy_linewidth=.5)
        fig.set_size_inches(10.5, 10.5)
        fig.savefig(f"output/pol{checkin_reward}.png")

        policy_length = {
            state: len(action) for state, action in policy.items()
        }

        _f = plot_grid_world_blind_drive(mdp, policy_for_plotting, policy_length)
        _f.savefig(f"output/len{checkin_reward}.png")

    return conv_policy