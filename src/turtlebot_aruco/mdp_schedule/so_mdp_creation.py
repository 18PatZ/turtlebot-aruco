"""
 Copyright 2024 by California Institute of Technology.  ALL RIGHTS RESERVED.
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

import numpy as np
import itertools
import collections

from turtlebot_aruco.mdp_schedule.mdp import MDP

import random

class so_mdp(object):
    """
    A container class for stochastically observed MDPs
    """
    def __init__(
        self,
        states: list,               # A list of states
        one_step_transitions: map,  # A map from state+action to map(next_state: p_next_state)
        one_step_rewards: map,      # A map from (state, action) to reward, a float   
        terminal_states: map,       # Absorbing states and their reward
        observation_mc_states: list,     # A list of states for the observation MC
        observation_mc_transitions: map, # Map (state: next state: probability)
        observation_mc_obs_states: list,  # A list of states where we do receive an observation
        observation_mc_loc_states: list  # A list of states where we do receive information about where we are in the MC
    ):
        self.states = states
        self.one_step_transitions = one_step_transitions
        self.one_step_rewards = one_step_rewards
        self.terminal_states = terminal_states
        self.observation_mc_states = observation_mc_states
        self.observation_mc_transitions = observation_mc_transitions
        self.observation_mc_obs_states = observation_mc_obs_states
        self.observation_mc_loc_states = observation_mc_loc_states


# Let's start with a MDP approach. For that, we need to know more about the observation Markov chain. Recall that we assume we _don't_ know where we are in the MC.

# Recall that the observation MC has three type of states.
# - Transparent states give no information.
# - "Blue" states provide an observation to the MDP process (and knowledge that we are in one blue state, although not necessarily which one!).
# - "Yellow" states provide knowledge of where we are in the observation MC ("a blue state will come up in two steps").

# Let's consider a MC where all states are either transparent or (blue+yellow). That is, when we get an observation, we know exactly where we are.

# The question is: can we convert the Markov Chain into a form where we only transition between "blue" observation states (possibly in multiple time steps)? Can we rewrite this as "if I am in blue state A, I will transition to blue state B in t steps with probability $p_{AB}^t$, and to state A in u steps with probability $P_{AA}^u$"? And does this distribution have a nice finite support?
        
def find_observation_mc_sparse_form(problem: so_mdp, max_time_steps: int):
    # Write the Markov chain as a transition matrix.
    # For each observation state ("blue") A
    # Mix up the matrix for T time steps
    # If probability mass falls into another blue state B, record it  as $p_{AB}$^t and remove from the mix.
    # If there is no more probability left, return
    # If there is still probability left after T, then the MC is not reducible.
    
    # This is the transition matrix
    mc_transition_matrix = np.zeros([len(problem.observation_mc_states), len(problem.observation_mc_states)])
    # This is where we will store the probabilities
    reduced_probabilities = {s: {} for s in problem.observation_mc_obs_states}
    
    # Let's first build the transition probability
    for state_ix, state in enumerate(problem.observation_mc_states):
        for next_state in problem.observation_mc_transitions[state]:
            next_state_ix = np.nonzero(np.array(problem.observation_mc_states)==next_state)
            assert len(next_state_ix)==1, "Error: state {} found {} times in MC states but appears in the transition matrix".format(next_state, len(next_state_ix))
            next_state_ix = next_state_ix[0][0]
            mc_transition_matrix[next_state_ix][state_ix] += problem.observation_mc_transitions[state][next_state]
    
    # Now let's go through every "blue" absorbing state
    max_time_horizon = -1
    for absorbing_state in problem.observation_mc_obs_states:
        # Find the location of the state
        _distribution = np.zeros([len(problem.observation_mc_states),1])
        absorbing_state_ix = np.nonzero(np.array(problem.observation_mc_states)==absorbing_state)
        assert len(absorbing_state_ix)==1
        absorbing_state_ix = absorbing_state_ix[0][0]
        _distribution[absorbing_state_ix] = 1
        
        for t in range(max_time_steps):
            # Propagate the distribution
            _distribution = mc_transition_matrix.dot(_distribution)
            # Check if we ended up somewhere interesting
            for next_absorbing_state in problem.observation_mc_obs_states:
                next_absorbing_state_ix = np.nonzero(np.array(problem.observation_mc_states)==next_absorbing_state)
                assert len(next_absorbing_state_ix)==1
                next_absorbing_state_ix = next_absorbing_state_ix[0][0]
                if _distribution[next_absorbing_state_ix]>0:
                    if next_absorbing_state not in reduced_probabilities[absorbing_state].keys():
                        reduced_probabilities[absorbing_state][next_absorbing_state]={t: 0.}
                    if t not in reduced_probabilities[absorbing_state][next_absorbing_state].keys():
                        reduced_probabilities[absorbing_state][next_absorbing_state][t]=0.
                    reduced_probabilities[absorbing_state][next_absorbing_state][t]+=_distribution[next_absorbing_state_ix][0]
                    _distribution[next_absorbing_state_ix]=0
            if np.sum(_distribution)<=0:
                break
        max_time_horizon = max(max_time_horizon, t)
        
    is_mc_reducible = False
    if np.sum(_distribution)<=0:
        is_mc_reducible = True
    return (reduced_probabilities, is_mc_reducible, max_time_horizon)

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

def grow_transition_probabilities_and_rewards(
    current_transitions,
    current_rewards,
    one_step_actions_and_transitions,
    one_step_rewards,
    discount_factor=1.,
    compose_actions=lambda oa, na: (oa + (na,)),
    illegal_action_propagator=lambda s,a: {s: 1.},
    illegal_action_rewarder=lambda s,a: -1,
):
    """
    Recursively grow a MDP with transitions and rewards current_{transition,rewards} by 
    appending one_step{transitions,rewards}.
    """
    new_transitions = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : 0)))
    # State-Action-Next state-probability
    new_rewards = collections.defaultdict(lambda : collections.defaultdict(lambda : 0))
    # State-action-reward
    for state in current_transitions.keys():
        for composite_action in current_transitions[state].keys():
            available_actions = set()
            # First let's see what actions we may want to take in this probability distribution.
            
            for next_state in current_transitions[state][composite_action].keys():
                # Let's add this action!
                available_actions = available_actions.union(one_step_actions_and_transitions[next_state].keys())
            # Now let's go over these actions
            for next_action in available_actions:
                
                new_composite_action = compose_actions(composite_action, next_action) # TODO what does this mean? Typically append
                # Note that here we assume that action sequences are unique, which is true if actions are unique.
                new_rewards[state][new_composite_action] = current_rewards[state][composite_action]
                # We are in next_state, and we want to take one extra action.
                for next_state in current_transitions[state][composite_action].keys():
                    # If this is indeed a legal action here (i.e., we won't collide)
                    if next_action in one_step_actions_and_transitions[next_state]:
                        # Let's see where we go!
                        for next_next_state in one_step_actions_and_transitions[next_state][next_action].keys():
                            # Chain probabilities
                            new_transitions[state][new_composite_action][next_next_state]+=(
                                one_step_actions_and_transitions[next_state][next_action][next_next_state]*
                                current_transitions[state][composite_action][next_state]
                            )
                        # Reward does not depend on where you end up
                        new_rewards[state][new_composite_action]+=(
                            (discount_factor**len(composite_action))*
                            one_step_rewards[next_state][next_action]*
                            current_transitions[state][composite_action][next_state]
                        )
                    else:
                        # We took an illegal action. With what probability do we end up somewhere? This returns a dict with state as key and prob as value
                        next_states_and_probabilities = illegal_action_propagator(next_state, next_action)
                        for next_next_state in next_states_and_probabilities.keys():
                            new_transitions[state][new_composite_action][next_next_state]+=(
                                next_states_and_probabilities[next_next_state]*
                                current_transitions[state][composite_action][next_state]
                            )
                        new_rewards[state][new_composite_action]+=(
                            (discount_factor**len(composite_action))
                            *illegal_action_rewarder(next_state, next_action)*
                            current_transitions[state][composite_action][next_state]
                        )
    return ddict2dict(new_transitions), ddict2dict(new_rewards)

def make_existing_actions_composite(one_step_actions_and_transitions):
    """
    A helper function to bootstrap the composite MDP creator
    """
    composite_actions_and_transitions = {
        _s: {
            (_a,): one_step_actions_and_transitions[_s][_a]
            for _a in one_step_actions_and_transitions[_s].keys()
        }
        for _s in one_step_actions_and_transitions.keys() 
    } 
    return composite_actions_and_transitions

def validate_transition_probabilities(actions_and_transitions, epsilon=0.001):
    """
    Do the transition probabilities sum to one?
    """
    for s in actions_and_transitions.keys():
        for a in actions_and_transitions[s].keys():
            probs = [actions_and_transitions[s][a][ns] for ns in actions_and_transitions[s][a].keys()]
            assert (abs(sum(probs)-1)<epsilon), "ERROR: probabilities for (s,a): {},{} sum up to {}".format(s,a,sum(probs))

    return True

def cast_so_mdp_as_mdp(so_problem: so_mdp, discount_factor: float=0.99, illegal_action_reward: float=-1):
    """
    Cast a SO-MDP as a MDP where the state is (agent_stateXobservation_state, and the rest follows)
    """


    #     1. We unpack the agent state and MC state
    somdp_states = list(itertools.product(so_problem.states, so_problem.observation_mc_obs_states))
    
    reduced_probabilities, is_mc_reducible, max_obs_time_horizon = find_observation_mc_sparse_form(so_problem, 100)
    assert is_mc_reducible, "ERROR: observation Markov chain is not reducible. Have you tried POMDPs?"
    
    #     2. We roll out the agent transitions by T time steps. We also keep around the transitions after 1, 2, ..., T time steps (this is new).
    
    current_transitions = make_existing_actions_composite(so_problem.one_step_transitions)
    current_rewards = make_existing_actions_composite(so_problem.one_step_rewards)
    
    cumulative_transitions = {0: current_transitions}
    cumulative_rewards = {0: current_rewards}
    
    for t in range(1,max_obs_time_horizon+1):
        current_transitions, current_rewards = grow_transition_probabilities_and_rewards(
            current_transitions,
            current_rewards,
            so_problem.one_step_transitions,
            so_problem.one_step_rewards,
            discount_factor=discount_factor,
            illegal_action_rewarder=lambda s,a: illegal_action_reward,
        )
        cumulative_transitions[t] = current_transitions
        cumulative_rewards[t] = current_rewards

        
    somdp_transitions = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : 0)))
    somdp_rewards = collections.defaultdict(lambda : collections.defaultdict(lambda :  0))
    
    #     3. We enumerate when the next observation may happen (t<=T), and where we may end up (s_mc).
    for somdp_state in somdp_states:
        agent_state = somdp_state[0]
        mc_state    = somdp_state[1]
        max_time_until_next_checkin = max([t for _ns in reduced_probabilities[mc_state] for t in reduced_probabilities[mc_state][_ns].keys()  if reduced_probabilities[mc_state][_ns][t]>0])
        
        full_length_agent_state_transitions = cumulative_transitions[max_time_until_next_checkin]
        for next_full_length_action in full_length_agent_state_transitions[agent_state].keys():
            for next_mc_state, next_mc_time_steps_and_probabilities in reduced_probabilities[mc_state].items():
                #     4. This determines the next MC state, s_mc.
                for next_mc_time_steps, next_mc_probability in next_mc_time_steps_and_probabilities.items():
                    #     5. The next agent state is given by the transition after t time steps, which we computed in 3.
                    #        We multiply these probabilities by the probability the next observation happens after t time steps.
                    _agent_state_transitions = cumulative_transitions[next_mc_time_steps]
                    _agent_state_rewards     = cumulative_rewards[next_mc_time_steps]

    #                 Here you are picking an action of length next_mc_time_steps and just adding it.
    #                 But you don't know how long an action sequence you'll have to commit to!
    #                 We're saying "maybe we'll get a checkin in n_m_t_s steps", and we're computing the reward
    #                 and transition correctly. BUT this reward and transition should be attributed to the actions of 
    #                 length max_time_until_next_checkin with the right prefix.
    #                 What this means is that:
    #                     - For each action *with that prefix*, there is some chance we stop early.
    #                     - That action has a chance of getting the corresponding reward. 
    #                 This means iterating over all "long" actions and picking the prefix.

    #                 for next_action_prefix in _agent_state_transitions[agent_state].keys():
                    next_action_prefix = next_full_length_action[:next_mc_time_steps+1]
#                     print(len(next_action_prefix))
#                     print(list(_agent_state_transitions[agent_state].keys())[0])
                    
                    for next_agent_state in _agent_state_transitions[agent_state][next_action_prefix].keys():
                        next_somdp_state = (next_agent_state, next_mc_state)
                        somdp_transitions[somdp_state][next_full_length_action][next_somdp_state] += (
                            _agent_state_transitions[agent_state][next_action_prefix][next_agent_state] *
                            next_mc_probability
                        )
                    somdp_rewards[somdp_state][next_full_length_action] += (
                        _agent_state_rewards[agent_state][next_action_prefix] *
                        next_mc_probability
                    )
    
    somdp_actions = set()
    for somdp_state in somdp_states:
        somdp_actions = somdp_actions.union(somdp_transitions[somdp_state])
    
    somdp_terminal_states = {
        (mdp_ts, obs_state): so_problem.terminal_states[mdp_ts]
        for mdp_ts in so_problem.terminal_states.keys() for obs_state in so_problem.observation_mc_obs_states
    }
    
    somdp_as_mdp = MDP(
        states=somdp_states,
        actions=somdp_actions,
        transitions=somdp_transitions,
        rewards=somdp_rewards,
        terminal_states=somdp_terminal_states
    )
    return somdp_as_mdp

# Rollout:
# -  Pick a start state in both agent and MC state
# - Get the corresponding action in the policy
# - Sample the next agent state (one step!)
# - Sample the next MC state (one step)
# - Execute the action
# - Iterate until a check-in occurs or we run out of actions (which should never happen)
# - If a check-in occurs, requery the policy

def so_mdp_rollout(
    so_problem: so_mdp,
    policy: dict,
    starting_agent_state: tuple,
    starting_observation_state,
    terminal_states_reveal_themselves: bool=False,
    MAX_ROLLOUT_ITERATIONS: int=1000,
):
    """ 
    Roll out the SO-MDP
    """
    
    action_sequence = []
    
    reward = 0.

    
    state = (starting_agent_state, starting_observation_state)
    
    states_traversed = [state]
    actions_taken = []
    upcoming_action_sequence = []
    
    for iteration in range(MAX_ROLLOUT_ITERATIONS):
        agent_state = state[0]
        mc_state = state[1]
        
        # If we arrived, return
        if agent_state in so_problem.terminal_states.keys():
            if (mc_state in so_problem.observation_mc_obs_states) or terminal_states_reveal_themselves:
                reward += so_problem.terminal_states[agent_state]
                return states_traversed, actions_taken, reward, upcoming_action_sequence
        
        #         - If a check-in occurs, requery the policy
        if mc_state in so_problem.observation_mc_obs_states:
            action_sequence = policy[state]
        #         Execute the action
        assert len(action_sequence)>0, "We ran out of actions! That shouldn't happen"
        
        next_action = action_sequence[0]
        # Pop the action from the sequence
        
        _next_agent_states = list(so_problem.one_step_transitions[agent_state][next_action].keys())
        next_agent_state = random.choices(
            _next_agent_states,
            weights=[so_problem.one_step_transitions[agent_state][next_action][ns] for ns in _next_agent_states]
        )[0]
        reward+= so_problem.one_step_rewards[agent_state][next_action]
        
        _next_mc_states = list(so_problem.observation_mc_transitions[mc_state])
        next_mc_state = random.choices(
            _next_mc_states,
            weights=[so_problem.observation_mc_transitions[mc_state][nms] for nms in _next_mc_states]
        )[0]
        
        state = (next_agent_state, next_mc_state)
        
        actions_taken.append(next_action)
        states_traversed.append(state)
        upcoming_action_sequence.append(action_sequence)
        
        action_sequence = action_sequence[1:]
        
        
    return states_traversed, actions_taken, reward, upcoming_action_sequence
    
    # Pick the starting state
    # Choose the optimal action sequence and queue it up
    # Execute one step of the optimal action sequence in both the MC and the state
    #  by sampling from the so_mdp
    # Are we in a check-in state? If so, get the location 