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
import copy
import time
# from periodic_observations.periodic_observations_mdp_utils import *
import multiprocessing as mp

def grid_world_mdp(x_steps=10, y_steps=10, action_precision=.7, action_reward = -1., drift=True):
    """
    A helper function to build a grid-world MDP
    """
    states = [(x, y) for x in range(x_steps) for y in range(y_steps)]
    actions = [(1,0), (0,1), (-1,0), (0,-1)]
    transitions = {state: {action: {} for action in actions} for state in states}
    rewards = {state: {action: 0. for action in actions} for state in states}
    for state in states:
        for action in actions:
            _reward = action_reward
            next_state = (state[0]+action[0], state[1]+action[1])
            if next_state not in states:
                _reward -= action_precision
                next_state = state
            if next_state in transitions[state][action].keys():
                transitions[state][action][next_state] += action_precision
            else:
                transitions[state][action][next_state] = action_precision
            for other_action in actions:
                if other_action == action:
                    continue
                else:
                    next_state = (state[0]+other_action[0], state[1]+other_action[1])
                    if next_state not in states:
                        _reward -= (1.-action_precision)/(len(actions)-1)
                        next_state = state
                    if next_state in transitions[state][action].keys():
                        transitions[state][action][next_state] += (1.-action_precision)/3.
                    else:
                        transitions[state][action][next_state] = (1.-action_precision)/3.
            rewards[state][action] = _reward
        if drift is False:
            driftless_action = (0,0)
            transitions[state][driftless_action] = {state: 1.}
            rewards[state][driftless_action] = _reward

#     terminal_states = {(x_steps-1, y_steps-1): 100}
    if drift is False:
        actions.append(driftless_action)
        
    terminal_states = {}
    
    return MDP(states=states, actions=actions, transitions=transitions, rewards=rewards,terminal_states=terminal_states)

def make_existing_composite_actions_and_transitions(one_step_actions_and_transitions):
    existing_composite_actions_and_transitions = {
        ( (key,) , key, 1): val for key, val in one_step_actions_and_transitions.items()
    }
    return existing_composite_actions_and_transitions
        
def create_recursive_actions_and_transitions(
    one_step_actions_and_transitions,
    remaining_steps,
    existing_composite_actions_and_transitions=None,
    cumulative_composite_actions_and_transitions=None,
    action_composer = None,
    ):
    """
    existing_composite_actions_and_transitions is a dictionary
    Keys are (sequential_action, composite_action) pairs.
    For each action, the value is a dictionary with
     key: action actually being applied
     value: probability of applying the key
    
    return: a dict identical to existing_composite_actions_and_transitions
    a dict with actions as keys
     for every action, a dict with
     key: the subactions applied (think of it as "the places I may end up in at intermediate steps")
          the probabilities of ending up there, summed over time (so sum(probabilities)=remaining_steps)
    """

    # If this is the first step, pose the problem in the format we like
    if existing_composite_actions_and_transitions is None:
        existing_composite_actions_and_transitions = make_existing_composite_actions_and_transitions(one_step_actions_and_transitions)
        cumulative_composite_actions_and_transitions = copy.deepcopy(existing_composite_actions_and_transitions)
    if action_composer is None:
        action_composer = lambda a1,a2 : (a1[0]+a2[0], a1[1]+a2[1]) 
#     print("Remaining steps: {}".format(remaining_steps))
    # Then recurse

    # If we are done...
    if remaining_steps == 0:
        return existing_composite_actions_and_transitions, cumulative_composite_actions_and_transitions
    # If we aren't done
    else:
        new_composite_actions_and_transitions = {}
        new_cumulative_composite_actions_and_transitions = {}
        for _key, composite_action_distribution in existing_composite_actions_and_transitions.items():
            overall_action = _key[0]
            composite_action = _key[1]
            action_length = _key[2]
            # Expand every action
#             print("Expanding action {}".format(overall_action))
            for new_action, new_action_distribution in one_step_actions_and_transitions.items():
                # This is what I intend to do
                new_overall_action = overall_action + (new_action,)
                new_composite_action= action_composer(composite_action, new_action) #(composite_action[0]+new_action[0],composite_action[1]+new_action[1])
                new_action_length = action_length + 1
                new_key = (new_overall_action, new_composite_action, new_action_length)
                if new_key in new_composite_actions_and_transitions.keys():
                    print("This should not happen")
                    import pdb
                    pdb.set_trace()
                new_composite_actions_and_transitions[new_key] = {}
                # New action
                new_cumulative_composite_actions_and_transitions[new_key] = copy.deepcopy(cumulative_composite_actions_and_transitions[_key])
                # This is what actually ends up happening
                for old_action, old_probability in composite_action_distribution.items():
                    for new_action, new_probability in new_action_distribution.items():
                        prob = old_probability*new_probability
                        act = action_composer(old_action, new_action) #(old_action[0]+new_action[0], old_action[1]+new_action[1])
                        if act not in new_composite_actions_and_transitions[new_key].keys():
                            new_composite_actions_and_transitions[new_key][act] = 0.
                        new_composite_actions_and_transitions[new_key][act] += prob
                        if act not in new_cumulative_composite_actions_and_transitions[new_key].keys():
                            new_cumulative_composite_actions_and_transitions[new_key][act] = 0.
                        new_cumulative_composite_actions_and_transitions[new_key][act] += prob
        # Recurse        
        return create_recursive_actions_and_transitions(one_step_actions_and_transitions,remaining_steps-1, new_composite_actions_and_transitions, new_cumulative_composite_actions_and_transitions)

def multi_step_grid_world_mdp_v0(
    x_steps,
    y_steps,
    one_step_actions_and_transitions,
    actions_between_checkins,
    terminal_states,
    composite_action_cost=None,
    wall_penalty=5.,
    verbose=False,
):
    """
    A helper function to build a multi-step grid-world MDP
    """
    if verbose:
        print("Building multi-step MDP")
    states = [(x, y) for x in range(x_steps) for y in range(y_steps)]
    terminal_states = terminal_states
    wall_penalty = wall_penalty
    if composite_action_cost is None:
        composite_action_cost = actions_between_checkins

    if verbose:
        print("Computing joint actions")
    composite_actions_and_transitions, cumulative_actions_and_transitions = create_recursive_actions_and_transitions(one_step_actions_and_transitions, actions_between_checkins-1)
    
    composite_actions = list(composite_actions_and_transitions.keys())
    
    
    transitions = {state: {action: {} for action in composite_actions} for state in states}
    rewards = {state: {action: 0. for action in composite_actions} for state in states}

    if verbose:
        print("Computing transitions")
    for state in states:
        for rolled_out_action in composite_actions:
#             _reward = -actions_between_checkins
            actually_applied_action_distribution = composite_actions_and_transitions[rolled_out_action]
            # Where do we end up?
            for actually_applied_action, actually_applied_probability in actually_applied_action_distribution.items():
                next_state = (state[0]+actually_applied_action[0], state[1]+actually_applied_action[1])
                if next_state not in states:
#                     _reward -= wall_penalty*actually_applied_probability
                    next_state = state
                if next_state in transitions[state][rolled_out_action].keys():
                    transitions[state][rolled_out_action][next_state] += actually_applied_probability
                else:
                    transitions[state][rolled_out_action][next_state] = actually_applied_probability
    if verbose:
        print("Computing rewards")
    for state in states:
        for rolled_out_action in composite_actions:
            _reward = - composite_action_cost
            actually_applied_cumulative_action_distribution = cumulative_actions_and_transitions[rolled_out_action]
            # To compute reward, we look at all the places we may have traversed.
            # If we ever strayed out of bounds, we get a ding. If we hit a final state, good!
            # This is actually not accurate. THe thing is, if we hit a final state, we should _stop there_, not continue propagating.
            # If we go out of bounds and then come back, we should not only get a penalty, but also an adjustment to our distribution.

            for cumulative_actual_action, cumulative_actual_probability in actually_applied_cumulative_action_distribution.items():
                _next_state = (state[0]+cumulative_actual_action[0], state[1]+cumulative_actual_action[1])
                if _next_state not in states:
                    _reward -= wall_penalty*cumulative_actual_probability
                if _next_state in terminal_states.keys():
                    _reward += cumulative_actual_probability*terminal_states[_next_state]
            
            rewards[state][rolled_out_action] = _reward
#     terminal_states = {(x_steps-1, y_steps-1): 100}
#     terminal_states = {}
    
    return MDP(states=states, actions=composite_actions, transitions=transitions, rewards=rewards,terminal_states=terminal_states)

def add_terminal_state(x0, y0, dx, dy, cost=-1000):
    """
    A helper function to add terminal states to a MDP
    """
    terminal_states = {
        (x,y): cost for x in range(x0, x0+dx) for y in range(y0, y0+dy)
    }
    return terminal_states

def expand_multi_step_grid_world_mdp(
    mdp,
    non_dominated_actions,
    one_step_actions_and_transitions,
    sa_values=None,
    wall_penalty=5.,
    step_penalty = 1.
):
    # Input:
    # - States
    # - Actions to expand for every state
    # - Transitions for those actions
    # - Rewards for those actions
    # - One-step actions
    
    # Output:
    # - States
    # - Actions
    # - Transitions
    # - Rewards
    # - Terminal states (passed through)
    
    # Naive implementations:
    # - Expand actions by calling create_recursive_actions_and_transitions.
    # - Recompute rewards and transitions with the new actions
    
    # Better implementation:
    # - Reuse previous state-specific rewards and transitions in the expansion
    # - That is, take in a multi-step problem and a one-step problem.
    #   Iterate over state-action pairs in the multi-step problem.
    #   See what next_state you end up in.
    #   Now, look at the actions available in next_state in the one-step problem
    #   And append those to the transitions, with the relevant reward.
    # HOWEVER. There could be next_states where an action is not available.
    # If that is the case, taking the action in that state leaves us where we are,
    # with a penalty (as if we had bounced against the wall).
    # This is true for terminal states, of course, but it could also be true elsewhere.
    # So we _do_ want a list of actions to expand, in addition to the one-step problem.
    # For every action we want to expand, we try to fish it out of the one-step problem.
    # If the action is not there, then new_state=next_state, and the action reward is
    # lowered accordingly.
    
    # For every state-action pair, create a new action, and see where it ends up
    new_transitions = {state: {} for state in mdp.states }
    new_rewards = {state: {} for state in mdp.states}
    if sa_values is not None:
        new_sa_values = {state: {} for state in mdp.states}
    
    if type(step_penalty) is not dict:
        step_penalty_dict = {action: step_penalty for action in one_step_actions_and_transitions.keys()}
    else:
        step_penalty_dict = step_penalty
    
    if non_dominated_actions is None:
        non_dominated_actions = {state: mdp.transitions[state].keys() for state in mdp.states}
    
    for state in mdp.states:
        for action in non_dominated_actions[state]:
            if action not in mdp.rewards[state].keys():
                print("Who dis?")
                import pdb; pdb.set_trace()
            # We unpack the action, which, as a reminder, is (a string describing it, the actual dxdy)
            overall_action = action[0]
            composite_action = action[1]
            action_length = action[2]
            for next_action, next_transitions in one_step_actions_and_transitions.items():
                new_overall_action = overall_action + (next_action,)
                new_composite_action= (composite_action[0]+next_action[0],composite_action[1]+next_action[1])
                next_action_length = action_length+1
                # And this will be the new action key
                new_action_key = (new_overall_action, new_composite_action, next_action_length)
                # We will expand the probability from the current possible locations to where the agent may
                # end up at the next step
                new_transitions[state][new_action_key] = {}
                # On the other hand, rewards are cumulative. So we just add the new reward for the new states
                # on _top_ of the previous reward, which was collected by crossing other states.
                new_rewards[state][new_action_key] = mdp.rewards[state][action]
                # Now, where could we end up? We look at 1. where we were earlier and
                # 2. the effect of next_action
                # _pre is where we were _before_ next_action
                for next_state_pre, next_prob_pre in mdp.transitions[state][action].items():
                    if next_state_pre in mdp.terminal_states:
                        # In a terminal state, you do NOT take actions, just sit there.
                        # You also do not receive a reward, the reward will come from the
                        # fixed value at the terminal state
                        next_prob_post = next_prob_pre
                        next_state_post = next_state_pre
                        if next_state_post in new_transitions[state][new_action_key].keys():
                            new_transitions[state][new_action_key][next_state_post] += next_prob_post
                        else:
                            new_transitions[state][new_action_key][next_state_post] = next_prob_post
                    else:
                        # Apply the next_action
                        for next_transition, next_transition_probability in next_transitions.items():
                            next_prob_post = next_prob_pre*next_transition_probability
                            next_state_post = (next_state_pre[0] + next_transition[0], next_state_pre[1] + next_transition[1])
                            # If we bounce against a wall
                            if next_state_post not in mdp.states:
    #                             print(
    #                                 "Bounce off the wall at state {} with action {}-{} ".format(
    #                                     state, new_overall_action, new_composite_action
    #                                 )+"resulting in transition {} with total prob {} (with transition prob {}) ".format(
    #                                     next_transition, next_prob_post,next_transition_probability
    #                                 )+"(next_state_post would be {}, instead will be {})".format(
    #                                     next_state_post, next_state_pre
    #                                 )
    #                             )
                                next_state_post = next_state_pre
                                # Wall penalty
                                new_rewards[state][new_action_key] -= next_prob_post*wall_penalty
                            if next_state_post in new_transitions[state][new_action_key].keys():
                                new_transitions[state][new_action_key][next_state_post] += next_prob_post
                            else:
                                new_transitions[state][new_action_key][next_state_post] = next_prob_post
    #                         if next_state_post in mdp.terminal_states:
    #                             new_rewards[state][new_action_key] += next_prob_post*mdp.terminal_states[next_state_post]
                new_rewards[state][new_action_key] -= step_penalty_dict[next_action]
                if sa_values is not None:
                    new_sa_values[state][new_action_key] = sa_values.get(state, 0.).get(action,0.)
                    
    all_actions = set([_a for _s in new_transitions.keys() for _a in new_transitions[_s].keys() ])
    new_mdp = MDP(
        states=mdp.states,
        actions=all_actions,
        transitions=new_transitions,
        rewards=new_rewards,
        terminal_states=mdp.terminal_states
    )
    
    if sa_values is None:
        return new_mdp
    else:
        return new_mdp, new_sa_values

def expand_multi_step_grid_world_mdp_v2(
    base_mdp,
    suffix_mdp,
    one_step_actions,
    append_action_function,
    non_dominated_actions=None,
    sa_values=None,
    terminal_state_step_penalty=1.,
    action_unavailable_penalty=5.,
    non_dominated_actions_dict=None,
):
    # Input:
    # - A base multi-step MDP
    # - A MDP containing the actions and transitions to append
    # - The actions we actually want to append to the end of every action in the MDP
    # - A function to extend an action, returning a new one
    # - what actions to discard in the base MDP because they are dominated
    # - One-step actions
    
    # Output:
    # A new MDP 
    
    # Better implementation:
    # - Reuse previous state-specific rewards and transitions in the expansion
    # - That is, take in a multi-step problem and a one-step problem.
    #   Iterate over state-action pairs in the multi-step problem.
    #   See what next_state you end up in.
    #   Now, look at the actions available in next_state in the one-step problem
    #   And append those to the transitions, with the relevant reward.
    # HOWEVER. There could be next_states where an action is not available.
    # If that is the case, taking the action in that state leaves us where we are,
    # with a penalty (as if we had bounced against the wall).
    # This is true for terminal states, of course, but it could also be true elsewhere.
    # So we _do_ want a list of actions to expand, in addition to the one-step problem.
    # For every action we want to expand, we try to fish it out of the one-step problem.
    # If the action is not there, then new_state=next_state, and the action reward is
    # lowered accordingly.
    
    # For every state-action pair, create a new action, and see where it ends up
    new_transitions = {state: {} for state in base_mdp.states }
    new_rewards = {state: {} for state in base_mdp.states}
    if sa_values is not None:
        new_sa_values = {state: {} for state in base_mdp.states}
    
    if type(action_unavailable_penalty) is not dict:
        action_unavailable_penalty_dict = {action: action_unavailable_penalty for action in one_step_actions}
    else:
        action_unavailable_penalty_dict = action_unavailable_penalty

    if type(terminal_state_step_penalty) is not dict:
        terminal_state_step_penalty_dict = {action: terminal_state_step_penalty for action in one_step_actions}
    else:
        terminal_state_step_penalty_dict = terminal_state_step_penalty

    if non_dominated_actions is None:
        print("Filling in non-dominated actions")
        non_dominated_actions = {state: base_mdp.transitions[state].keys() for state in base_mdp.states}
    
    for state in base_mdp.states:
        for action in non_dominated_actions[state]:
            if action not in base_mdp.rewards[state].keys():
                print("Who dis? I want to apply action {}, but available actions are {}...".format(action, list(base_mdp.rewards[state].keys())[:10]))
                print("To double-check, defined actions are {}...".format(list(base_mdp.actions)[:10]))
                print("whereas non-dominated actions for this state are {}...".format(list(non_dominated_actions[state])[:10]))
                print("And non_dominated_actions_dict keys are: {}".format(non_dominated_actions_dict.keys()))
                max_key = sorted(list(non_dominated_actions_dict.keys()))[-1]
                print("And non_dominated actions for key {} are {}...".format(max_key, non_dominated_actions_dict[max_key][state][:10]))
                if max_key>1:
                    print("FYI non_dominated actions for key {} are {}...".format(max_key-1, non_dominated_actions_dict[max_key-1][state][:10]))
                import pdb; pdb.set_trace()
            for next_action in one_step_actions:
                # What if some of the actions in suffix_mdp are NOT in one_step_actions?
                cumulative_next_action_probability=0.
                new_action_key = append_action_function(action, next_action)
                # We will expand the probability from the current possible locations to where the agent may
                # end up at the next step
                new_transitions[state][new_action_key] = {}
                # On the other hand, rewards are cumulative. So we just add the new reward for the new states
                # on _top_ of the previous reward, which was collected by crossing other states.
                new_rewards[state][new_action_key] = base_mdp.rewards[state][action]
                # Now, where could we end up? We look at 1. where we were earlier and
                # 2. the effect of next_action
                # _pre is where we were _before_ next_action
                for next_state_pre, next_prob_pre in base_mdp.transitions[state][action].items():
                    if next_state_pre in base_mdp.terminal_states:
                        # In a terminal state, you do NOT take actions, just sit there.
                        # You also do not receive a reward, the reward will come from the
                        # fixed value at the terminal state. But you do spend one extra step here.
                        next_prob_post = next_prob_pre
                        next_state_post = next_state_pre
                        cumulative_next_action_probability += next_prob_post
                        if next_state_post in new_transitions[state][new_action_key].keys():
                            new_transitions[state][new_action_key][next_state_post] += next_prob_post
                        else:
                            new_transitions[state][new_action_key][next_state_post] = next_prob_post
#                         print("State-action {}-{} ends up in terminal state {} wp {}, action reward {}".format(
#                             state,
#                             new_action_key,
#                             next_state_pre,
#                             next_prob_pre,
#                             new_rewards[state][new_action_key],
#                         ))
                        reward_increase = -terminal_state_step_penalty_dict[next_action]*next_prob_post
                        # new_rewards[state][new_action_key] is untouched
                    else:
                        if next_action in suffix_mdp.transitions[next_state_pre].keys():
                            # If the action is in the suffix MDP, look at those transitions and append them
                            reward_increase = 0.
                            for next_state_post, next_prob_suffix in suffix_mdp.transitions[next_state_pre][next_action].items():
                                next_prob_post = next_prob_pre*next_prob_suffix
                                cumulative_next_action_probability += next_prob_post
                                reward_increase += suffix_mdp.rewards[next_state_pre][next_action]*next_prob_post
#                                 print("In state {} action suffix {} IS available. Overall action is {}".format(
#                                     state,
#                                     next_action,
#                                     new_action_key,
#                                 ))
                                if next_state_post in new_transitions[state][new_action_key].keys():
                                    new_transitions[state][new_action_key][next_state_post] += next_prob_post
                                else:
                                    new_transitions[state][new_action_key][next_state_post] = next_prob_post
                        else:
                            # This action is not available in the appended MDP. So we stay where we are and get penalized
                            next_state_post = next_state_pre
                            next_prob_post = next_prob_pre
                            cumulative_next_action_probability += next_prob_post
                            reward_increase = -action_unavailable_penalty_dict[next_action]*next_prob_post
                            print("In state {} action suffix {} is NOT available. Probability is {} and penalty is {}".format(
                                state,
                                next_action,
                                next_prob_post,
                                reward_increase,
                            ))

                            if next_state_post in new_transitions[state][new_action_key].keys():
                                new_transitions[state][new_action_key][next_state_post] += next_prob_post
                            else:
                                new_transitions[state][new_action_key][next_state_post] = next_prob_post

#                     print("Reward for state {}, action {} is {} (reward for prefix {} was {})".format(
#                     state, new_action_key, new_rewards[state][new_action_key] + reward_increase,
#                         action, new_rewards[state][new_action_key]))
                    new_rewards[state][new_action_key] += reward_increase

#                 new_rewards[state][new_action_key] -= step_penalty_dict[next_action]
                if np.abs(cumulative_next_action_probability-1)>1e-8:
                    print("WARNING: cumulative action probability for state-action {}-{} is problematic (is {}, should be {})".format(
                    state, new_action_key,cumulative_next_action_probability, 1))
                if sa_values is not None:
                    new_sa_values[state][new_action_key] = sa_values.get(state, 0.).get(action,0.)
                    
    all_actions = set([_a for _s in new_transitions.keys() for _a in new_transitions[_s].keys() ])
    new_mdp = MDP(
        states=base_mdp.states,
        actions=all_actions,
        transitions=new_transitions,
        rewards=new_rewards,
        terminal_states=base_mdp.terminal_states
    )
    
    if sa_values is None:
        return new_mdp
    else:
        return new_mdp, new_sa_values

def extend_mdp_sa_values(
    base_mdp,
    one_step_actions,
    append_action_function,
    non_dominated_actions=None,
    sa_values={},
):
    # Input:
    # - A base multi-step MDP
    # - The actions we actually want to append to the end of every action in the MDP
    # - A function to extend an action, returning a new one
    # - what actions to discard in the base MDP because they are dominated
    # - The previous MDP's state-action bounds
    
    # Output:
    # new_sa_values, the extended MDP's state-action bounds
    new_sa_values = {state: {} for state in base_mdp.states}
    for state in base_mdp.states:
        for action in non_dominated_actions[state]:
            if action not in base_mdp.rewards[state].keys():
                print("Who dis? I want to apply action {}, but available actions are {}...".format(action, list(base_mdp.rewards[state].keys())[:10]))
                import pdb; pdb.set_trace()
            for next_action in one_step_actions:
                new_action_key = append_action_function(action, next_action)
                new_sa_values[state][new_action_key] = sa_values.get(state, 0.).get(action,0.)
    return new_sa_values

def append_action_function_grid_world(action, next_action):
    overall_action = action[0]
    composite_action = action[1]
    action_length = action[2]

    next_action_onestep = next_action[1]
    new_overall_action = overall_action + (next_action_onestep,)
    new_composite_action= (composite_action[0]+next_action_onestep[0],composite_action[1]+next_action_onestep[1])
    next_action_length = action_length+1
    # And this will be the new action key
    new_action_key = (new_overall_action, new_composite_action, next_action_length)
    return new_action_key



def multi_step_grid_world_mdp(
    x_steps,
    y_steps,
    one_step_actions_and_transitions, # A dict with action as key, and a dict as val. In this dict, the actual dx dy is the key and the probability of taking that step is the val
    actions_between_checkins,
    terminal_states,
    action_cost=None,
    wall_penalty=5.,
    verbose=False,
):
    """
    A helper function to build a multi-step grid-world MDP
    """
    
    if verbose:
        print("Building multi-step MDP")
    
    states = [(x, y) for x in range(x_steps) for y in range(y_steps)]
    terminal_states = terminal_states
    wall_penalty = wall_penalty
    
    # If action_cost is not given, cost of 1 per action
    if action_cost is None:
        action_cost = {action: 1. for action in one_step_actions_and_transitions.keys()}
    # If action_cost is not a dict, give the same cost to all actions
    if type(action_cost) is not dict:
        action_cost = {action: action_cost for action in one_step_actions_and_transitions.keys()}

#     if verbose:
#         print("Computing one-step actions")
#     composite_actions_and_transitions = make_existing_composite_actions_and_transitions(one_step_actions_and_transitions)

    
#     # Convert action cost to composit
#     composite_action_cost = {action: action_cost[action[1]] for action in composite_actions_and_transitions.keys()}
    
#     transitions = {state: {action: {} for action in composite_actions} for state in states}
#     rewards = {state: {action: 0. for action in composite_actions} for state in states}

    transitions = {state: {} for state in states}
    rewards = {state: {} for state in states}
    # Make the zero-step MDP
    # Add an initial dummy action that does nothjing and counts as action 0
    if verbose:
        print("Making dummy zero-step MDP")
    dummy_action = ((),(0,0),0)
    for state in states:
        if state not in terminal_states:
            transitions[state] = {dummy_action: {state: 1.}}
            rewards[state] = {dummy_action: 0.}
    
    mdp_sa = MDP(states=states, actions=[dummy_action,], transitions=transitions, rewards=rewards,terminal_states=terminal_states)

    # Recurse in depth by using expand_multi_step_grid_world_mdp
    step_penalty = action_cost

    one_step_mdp = expand_multi_step_grid_world_mdp(
        mdp_sa,
        non_dominated_actions=None,
        one_step_actions_and_transitions=one_step_actions_and_transitions,
        wall_penalty=wall_penalty,
        step_penalty = step_penalty
    )
    mdp = copy.deepcopy(one_step_mdp)

    # Convert action cost to composit
    composite_step_penalty = {action: action_cost[action[1]] for action in one_step_mdp.actions}

    if verbose:
        print("Growing the MDP")
    for epoch in range(1,actions_between_checkins):
        if verbose:
            print("Action length {}/{}".format(epoch+1, actions_between_checkins))
        # mdp = expand_multi_step_grid_world_mdp(
        #     mdp,
        #     non_dominated_actions=None,
        #     one_step_actions_and_transitions=one_step_actions_and_transitions,
        #     wall_penalty=wall_penalty,
        #     step_penalty = step_penalty
        # )
        mdp = expand_multi_step_grid_world_mdp_v2(
            base_mdp=mdp,
            suffix_mdp=one_step_mdp,
            one_step_actions=one_step_mdp.actions,
            append_action_function=append_action_function_grid_world,
            non_dominated_actions=None,
            sa_values=None,
            terminal_state_step_penalty=composite_step_penalty,
            action_unavailable_penalty=wall_penalty,
        )

    return mdp

def prune_dominated_actions(_state_action_values_ub, _values_lb,_mdp_ub=None, _mdp_lb=None, verbose=False):
    state_actions = {state: [] for state in _state_action_values_ub.keys()}
    action_count = 0
    dominated_action_count = 0
    for state, action_list in _state_action_values_ub.items():
        if state in _values_lb.keys():
            state_action_count = 0
            dominated_state_action_count = 0
            for action in action_list:
                action_count += 1
                state_action_count += 1
                if _state_action_values_ub[state][action] < _values_lb[state]:
                    dominated_action_count += 1
                    dominated_state_action_count +=1
                    if verbose==2:
                        print("In state {}, action {} is dominated!".format(state, action))
                else:
                    state_actions[state].append(action)
        if state_action_count == dominated_state_action_count:
            print("Strange: we pruned ALL actions for state {}".format(state))
            import pdb; pdb.set_trace()
            raise RuntimeError
    if verbose:
        print("Pruned {}/{} actions".format(dominated_action_count, action_count))
    return state_actions