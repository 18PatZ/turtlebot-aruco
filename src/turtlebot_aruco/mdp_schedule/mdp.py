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

# A few helper functions to solve MDPs
# There are tons of good packages for solving MDPs, and they are all way faster than this. But this is a proof of concept.

import numpy as np
import copy

class MDP(object):
    """
    An object representing a Markov decision process
    Inputs:
    - states, a list of states
    - actions, a list of actions
    - transitions, a nested dictionary.
      transitions[state][action][new_state] is the probability of ending up in new_state when taking action from state
    - rewards, a nested dictionary.
      rewards[state][action] is the reward for taking action from state
    - terminal_states: a dictionary.
      terminal states are terminal_states.keys().
      terminal_states[terminal_state] is the reward for ending up in terminal_state
    """
    
    def __init__(self, states: list=[], actions: list=[], transitions: dict={}, rewards: dict={}, terminal_states: dict={}):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.terminal_states = terminal_states

    def __str__(self):
        return "MDP\nStates: {}\nActions: {}".format(self.states, self.actions)
        
    def __repr__(self):
        return "MDP\nStates: {}\nActions: {}\nTransitions: {}\nRewards: {}\nTerminal states: {}".format(self.states, self.actions, self.transitions, self.rewards, self.terminal_states)

def value_iteration(mdp, max_iterations=int(1e4), discount_factor = 1., relative_value_threshold = 0., print_progress=False, default_value = np.nan, initial_values_guess = None):

    if initial_values_guess is None:
        values = {state: 0. for state in mdp.states}
    else:
        values = initial_values_guess
    values.update(mdp.terminal_states)
#     policies = {state: None for state in mdp.states}
    # Value iteration
    for iteration in range(max_iterations):
        old_values = np.array(list(values.values()))
        values.update({state: np.max(
            [mdp.rewards[state][action] + 
             discount_factor*np.sum([
            probability * values.get(new_state, default_value)
             for new_state, probability in mdp.transitions[state][action].items()])
             for action in mdp.actions]
        ) for state in mdp.states if state not in mdp.terminal_states.keys()})
        new_values = np.array(list(values.values()))
        relative_value_difference = ((np.linalg.norm(new_values-old_values)
                              )/np.linalg.norm(new_values))
        # if np.max(list(values.values()))> np.max(old_values):
        #     print("Weird")
        #     import pdb
        #     pdb.set_trace()
        if print_progress and not iteration%10:
            print("{}/{}, value difference {}% (was {}, is {})(threshold {}%)".format(
                iteration,
                max_iterations,
                100.*relative_value_difference,
                np.linalg.norm(old_values),
                np.linalg.norm(new_values),
                100*relative_value_threshold,
            )
        )
        if relative_value_difference<=relative_value_threshold:
            break
            
    if iteration >= max_iterations-1:
        print("Maximum number of iterations reached!")
    if print_progress:
        print("Converged after {} iterations. Relative value error: {}%.".format(
            iteration,
            relative_value_difference*100)
        )
    
    # Extract policy
    try:
        actions_list = list(mdp.actions)
        policy = {state: actions_list[np.argmax(
                [mdp.rewards[state][action] + 
                 discount_factor*np.sum([
                probability * values.get(new_state, default_value)
                 for new_state, probability in mdp.transitions[state][action].items()])
                 for action in actions_list]
            )] for state in mdp.states if state not in list(mdp.terminal_states.keys())}
    except Exception as e:
        import pdb
        pdb.set_trace()
        raise(e)
    return policy, values

def state_action_value_iteration(
        mdp,
        max_iterations=int(1e4),
        discount_factor = 1.,
        relative_value_threshold = 0.,
        print_progress=False,
        default_value = np.nan,
        initial_values_guess = None,
        variable_discount_factor = False,
):

    def get_state_values(state_action_values):
        _state_values = {state: max(state_action_values[state][action] for action in state_action_values[state].keys()) for state in state_action_values.keys()}
        return _state_values

    if initial_values_guess is None:
        values = {state: {action: 0. for action in mdp.transitions.get(state, [])} for state in mdp.states}
    else:
        values = copy.deepcopy(initial_values_guess)
    for ts, ts_value in mdp.terminal_states.items():
        values[ts] = {"end": ts_value}
    
    _state_values = get_state_values(values)

    # Value iteration
    for iteration in range(max_iterations):
#         print(values)
        old_values = np.array(list([_state_values[state] for state in mdp.states]))
#         print(old_values)
        for state in mdp.states:
            if state not in mdp.terminal_states.keys():
                for action in mdp.transitions.get(state, []):
    #                 print((state, action))
                    # values[state][action] = (mdp.rewards[state][action] + 
                    #     discount_factor * np.sum([probability *
                    #     np.max([values[new_state][new_action] for new_action in values[new_state].keys()])
                    #     for new_state, probability in mdp.transitions[state][action].items()])
                    #     )
                    values[state][action] = mdp.rewards[state][action] 
                    for new_state, probability in mdp.transitions[state][action].items():
                        if variable_discount_factor:
                            _df = discount_factor**len(action)
                        else:
                            _df = discount_factor
                        values[state][action] += _df * probability * _state_values[new_state]
                        # np.max([values[new_state][new_action] for new_action in values[new_state].keys()])

        _state_values = get_state_values(values)
        new_values = np.array(list([_state_values[state] for state in mdp.states]))

        relative_value_difference = ((np.linalg.norm(new_values-old_values)
                              )/np.linalg.norm(new_values))

        if print_progress and not iteration%10:
            print("{}/{}, value difference {}% (was {}, is {})(threshold {}%)".format(
                iteration,
                max_iterations,
                100.*relative_value_difference,
                np.linalg.norm(old_values),
                np.linalg.norm(new_values),
                100*relative_value_threshold,
            )
        )
        if relative_value_difference<=relative_value_threshold:
            break
            
    if iteration >= max_iterations-1 and print_progress:
        print("Maximum number of iterations reached!")
    if print_progress:
        print("Converged after {} iterations. Relative value error: {}%.".format(
            iteration,
            relative_value_difference*100)
        )
    
    # Extract policy
    try:
        policy = {state: max(values[state], key=lambda a: values[state][a])
         for state in mdp.states if state not in mdp.terminal_states.keys() }
        state_values = {state: values[state][policy[state]]
         for state in mdp.states if state not in mdp.terminal_states.keys() }
    except Exception as e:
        import pdb
        pdb.set_trace()
        raise(e)
    return policy, state_values, values