"""
 Copyright 2020 by California Institute of Technology.  ALL RIGHTS RESERVED.
 United  States  Government  sponsorship  acknowledged.   Any commercial use
 must   be  negotiated  with  the  Office  of  Technology  Transfer  at  the
 California Institute of Technology.
 
 This software may be subject to  U.S. export control laws  and regulations.
 By accepting this document,  the user agrees to comply  with all applicable
 U.S. export laws and regulations.  User  has the responsibility  to  obtain
 export  licenses,  or  other  export  authority  as may be required  before
 exporting  such  information  to  foreign  countries or providing access to
 foreign persons.
 
 This  software  is a copy  and  may not be current.  The latest  version is
 maintained by and may be obtained from the Mobility  and  Robotics  Sytstem
 Section (347) at the Jet  Propulsion  Laboratory.   Suggestions and patches
 are welcome and should be sent to the software's maintainer.
 
"""

from turtlebot_aruco.mdp_schedule.mdp import MDP, value_iteration, state_action_value_iteration
import numpy as np
import random
import multiprocessing as mp
from turtlebot_aruco.mdp_schedule.periodic_observations_mdp_creation import multi_step_grid_world_mdp, append_action_function_grid_world, expand_multi_step_grid_world_mdp_v2, extend_mdp_sa_values

# state_actions = prune_dominated_actions(state_action_values_ub, values_lb,verbose=True)

# More bookkeeping, less efficient

def test_mdp_transitions(mdp):
    """ Check that MDP tansitions sum to one """
    for state in mdp.transitions.keys():
        for action in mdp.transitions[state].keys():
            cum_prob = 0.
            for next_state, next_prob in mdp.transitions[state][action].items():
                cum_prob += next_prob
            if np.abs(cum_prob-1)>1e-8:
                print("ERROR: state {}, action {}, sum of transition probabilities is {} (should be 1)".format(
                state,action,cum_prob))

def how_far_can_I_drive_blindly(mdp, policy, max_depth=10):
    how_far = {state: 0 for state in policy.keys() if state not in mdp.terminal_states.keys()}
    for state in how_far.keys():
        frontier = [state]
        for depth in range(max_depth):
            new_frontier = []
            for fr in frontier:
                new_frontier += [new_state for new_state in mdp.transitions[fr][policy[fr]].keys() if (mdp.transitions[fr][policy[fr]][new_state]>0 and new_state not in mdp.terminal_states.keys())]
            frontier_policies = set(policy[fr] for fr in new_frontier)
            if len(frontier_policies) > 1:
                break
            frontier = new_frontier
        how_far[state] = depth+1
    return how_far

def mle_rollout(mdp, policy, initial_state, max_steps=int(1e4)):
    """
    Performs a pseudo-rollout on mdp following policy from initial_state.
    Only goes to the maximum-likelihood next state for each state-action.
    Returns a list of nodes traversed.
    """
    states_traversed = [initial_state,]
    for step in range(max_steps):
        current_state = states_traversed[-1]
        if type(policy) is dict:
            current_action = policy[current_state]
        else: # Assume it is a function
            current_action = policy(current_state)
        next_state_distribution = mdp.transitions[current_state][current_action]
        _prob = -np.infty
        next_state = None
        for next_state_candidate, next_state_prob in next_state_distribution.items():
            if next_state_prob >= _prob:
                next_state = next_state_candidate
                _prob = next_state_prob
        states_traversed.append(next_state)
        if next_state in mdp.terminal_states:
            break
    return states_traversed

def stochastic_rollout(mdp, policy, initial_state, max_steps=int(1e4)):
    """
    Performs a pseudo-rollout on mdp following policy from initial_state.
    Only goes to the maximum-likelihood next state for each state-action.
    Returns a list of nodes traversed.
    """
    states_traversed = [initial_state,]
    for step in range(max_steps):
        current_state = states_traversed[-1]
        if type(policy) is dict:
            current_action = policy[current_state]
        else: # Assume it is a function
            current_action = policy(current_state)
        next_state_distribution = mdp.transitions[current_state][current_action]
        next_state = None
        _cum_prob = 0.
        _rand = random.random()
        for next_state_candidate, next_state_prob in next_state_distribution.items():
            _cum_prob += next_state_prob
            if _cum_prob>= _rand:
                next_state = next_state_candidate
                break
        states_traversed.append(next_state)
        if next_state in mdp.terminal_states:
            break
    return states_traversed

def does_rollout_cross_these_states(mdp, policy, initial_states, crossed_states, number_of_rollouts=100, max_steps=int(1e4), mle=False):
    likelihood_of_crossing = {initial_state: None for initial_state in initial_states}
    all_states_traversed = {initial_state: [] for initial_state in initial_states}
    for initial_state in initial_states:
        number_of_crossings = 0
        for rollout in range(number_of_rollouts):
            if mle:
                _states_traversed = mle_rollout(mdp, policy, initial_state)
            else:
                _states_traversed = stochastic_rollout(mdp, policy, initial_state)
            all_states_traversed[initial_state].append(_states_traversed)
            for _state_traversed in _states_traversed:
                if _state_traversed in crossed_states:
                    number_of_crossings+=1
                    break
        likelihood_of_crossing[initial_state] = float(number_of_crossings)/float(number_of_rollouts)
    return likelihood_of_crossing, all_states_traversed