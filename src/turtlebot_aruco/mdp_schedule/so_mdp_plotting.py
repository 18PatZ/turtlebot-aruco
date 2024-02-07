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

import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from turtlebot_aruco.mdp_schedule.periodic_observations_plotting import plot_grid_world_mdp, plot_grid_world_values, plot_multi_step_action, plot_grid_world_terminal_states, plot_grid_world_rollout

import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation

def plot_observation_mc(
    observation_mc_states: list,     # A list of states for the observation MC
    observation_mc_transitions: map, # Map (state: next state: probability)
    observation_mc_obs_states: list,  # A list of states where we do receive an observation
    observation_mc_loc_states: list,  # A list of states where we do receive information about where we are in the MC
    highlighted_nodes=[],
    highlight_color='tab:orange',
    ax=None
):
    """
    Plot the observation Markov chain, and optionally highlight selected nodes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    mc = nx.DiGraph()
    
    mc.add_nodes_from(observation_mc_states, localization=False, observation=False)
    for obsnode in observation_mc_obs_states:
        mc.nodes[obsnode]['observation'] = True
    for locnode in observation_mc_loc_states:
        mc.nodes[locnode]['localization'] = True
    for s1 in observation_mc_transitions.keys():
        for s2, p in observation_mc_transitions[s1].items():
            mc.add_edge(s1,s2, weight=p)

    node_positions = pos=nx.kamada_kawai_layout(mc)
    highlighted_nodes_pc = nx.draw_networkx_nodes(mc, pos=node_positions,
                           nodelist=highlighted_nodes,
                           node_size=600,
                           node_color=highlight_color,
                           ax=ax,
                          )

    all_nodes_pc = nx.draw_networkx(mc, with_labels=True,
            font_weight='regular',
            width=[2*mc.edges[e]['weight'] for e in mc.edges],
            edge_color=[mc.edges[e]['weight'] for e in mc.edges],
            edge_cmap=mpl.colormaps['Blues'],
            edge_vmin=0,edge_vmax=1,
            pos=node_positions,
            node_size=300,
            node_color=[
                 ( 'xkcd:green' if mc.nodes[o]['observation'] and mc.nodes[o]['localization']
                     else ('xkcd:sky blue' if mc.nodes[o]['observation']
                     else ('xkcd:yellow' if mc.nodes[o]['localization']
                     else (.8,.8,.8))))
                for o in mc.nodes
            ],
            ax=ax,       
           )
    ax.set_axis_off()
    return mc, highlighted_nodes_pc, all_nodes_pc



# So this is tricky. We can certainly visualize arrows of different length, or maybe different colors for different step sizes.

# Perhaps different maps depending on which observation state we're in?

# So one map per observation state.

# In that map, we have one loong action. That's it!

# But how do we communicate that we may stop early? Perhaps narrow the arrow as we traverse?

# - Show the MC with networkx
# - Show the grid with PSOMDP code
# - Show actions with some "narrowing" or color-coding.


def plot_so_mdp(so_problem, policy, state_values, *args, **kwargs):
    
    so_mc_states = so_problem.observation_mc_obs_states
    
    fig, axes = plt.subplots(2,len(so_mc_states),figsize=[10,10])
    
    # Unpack the states into agent states and MC states
    policy_by_mc_state = {mcs: {} for mcs in so_mc_states}
    state_values_by_mc_state = {mcs: {} for mcs in so_mc_states}
    for mcs in so_mc_states:
        for so_state in policy.keys():
            agent_state = so_state[0]
            mc_state = so_state[1]
            
            policy_by_mc_state[mc_state][agent_state] = (policy[so_state],(None,))
            state_values_by_mc_state[mc_state][agent_state] = state_values[so_state]

#     return policy_by_mc_state, state_values_by_mc_state
            
    for mc_state_ix, mc_state in enumerate(so_mc_states):
        plot_grid_world_mdp(
            policy=policy_by_mc_state[mc_state],
            values=state_values_by_mc_state[mc_state],
            ax=axes[0][mc_state_ix],
            mdp=MDP(terminal_states=so_problem.terminal_states),
            *args, **kwargs)
    
        _mc, _, _ = plot_observation_mc(
            observation_mc_states=so_problem.observation_mc_states,
            observation_mc_transitions=so_problem.observation_mc_transitions,
            observation_mc_obs_states=so_problem.observation_mc_obs_states,
            observation_mc_loc_states=so_problem.observation_mc_loc_states,
            highlighted_nodes=[mc_state],
            ax=axes[1][mc_state_ix],
        )

def plot_somdp_rollout(so_problem, states_traversed, actions_taken, reward, state_values):
    # Plot the background with the values
    # Plot the states traversed
    # Plot the actions taken for every state
    # Plot different colors for where we were in the MC chain
    # Highlight where the checkins occurred
    # Next week....
    
    fig, axes = plt.subplots(nrows=2,figsize=[10,10])
    
    # We need one MC state to plot the corresponding state values
    for representative_mc_state in so_problem.observation_mc_obs_states: 
#     representative_mc_state = so_problem.observation_mc_states[0]
    
        mc_state_values = {k[0]: state_values[k] for k in state_values.keys() if k[1] == representative_mc_state}
        
        plot_grid_world_values(
            mc_state_values,
            fig=fig,
            ax=axes[0],
            colormap='RdYlGn',
            alpha = .2/len(so_problem.observation_mc_obs_states),
            plot_colorbar=False,
            vmin=None,
            vmax=None,
        )
        
    mdp = MDP()
    mdp.terminal_states = so_problem.terminal_states
    ts_patches = plot_grid_world_terminal_states(
        mdp,
        x_offset=-.5,
        y_offset=-.5,
        linewidth=3,
        colormap='RdYlGn',
        edgecolor='w',
        alpha=None,
        plot_colorbar=False
    )
    for patch in ts_patches:
        axes[0].add_patch(patch)
        
    ro_patches = plot_grid_world_rollout(rollout_states=[s[0] for s in states_traversed], linewidth=.5,color='g')
    for patch in ro_patches:
        axes[0].add_patch(patch)
        
        
    mc_state_colormap = mpl.colormaps['Paired']
    mc_state_color = {
        mc_state: mc_state_colormap(mc_state_ix/len(so_problem.observation_mc_obs_states))
        for mc_state_ix, mc_state in enumerate(so_problem.observation_mc_obs_states)
    }
        

    # Plot the check-ins
    check_in_patches = []
    for state in states_traversed:
        mc_state = state[1]
        a_state = state[0]
        if mc_state in so_problem.observation_mc_obs_states:
            check_in_patches.append(mpatches.Circle(a_state,.2,facecolor=mc_state_color[mc_state]))
    
    for patch in check_in_patches:
        axes[0].add_patch(patch)
    
    for mc_state in so_problem.observation_mc_obs_states:
        _, _, _ = plot_observation_mc(
            observation_mc_states=so_problem.observation_mc_states,
            observation_mc_transitions=so_problem.observation_mc_transitions,
            observation_mc_obs_states=so_problem.observation_mc_obs_states,
            observation_mc_loc_states=so_problem.observation_mc_loc_states,
            highlighted_nodes=[mc_state],
            highlight_color=mc_state_color[mc_state],
            ax=axes[1]
        )
    
    axes[0].set_axis_off()

def animate_somdp_rollout(
    so_problem,
    states_traversed,
    actions_taken,
    reward,
    state_values,
    upcoming_action_sequence,
    interval_ms=500,
):
    # Plot the background with the values
    # Plot the states traversed
    # Plot the actions taken for every state
    # Plot different colors for where we were in the MC chain
    # Highlight where the checkins occurred

    def draw_frame(frame_number: int):
        # Show trajectory
        
        for rollout_patch in ro_plotted_patches[:frame_number]:
            if rollout_patch is not None:
                rollout_patch.set_visible(True)
        for rollout_patch in ro_plotted_patches[frame_number:]:
            if rollout_patch is not None:
                rollout_patch.set_visible(False)
                
        # Show check-ins
        for cip in check_in_patches_plotted[:frame_number+1]:
            if cip is not None:
                cip.set_visible(True)
        for cip in check_in_patches_plotted[frame_number+1:]:
            if cip is not None:
                cip.set_visible(False)
                
        for aps in action_patches_plotted:
            if aps is not None:
                for _ap in aps:
                    _ap.set_visible(False)
                    
        if action_patches_plotted[frame_number] is not None:
            for _ap in action_patches_plotted[frame_number]:
                _ap.set_visible(True)
                
        current_pos_patch[0].set_data(states_traversed[frame_number][0][0],states_traversed[frame_number][0][1])

        # Plot the observation MC
        axes[1].clear()
        for mc_state in so_problem.observation_mc_obs_states:
            plot_observation_mc(
                observation_mc_states=so_problem.observation_mc_states,
                observation_mc_transitions=so_problem.observation_mc_transitions,
                observation_mc_obs_states=so_problem.observation_mc_obs_states,
                observation_mc_loc_states=so_problem.observation_mc_loc_states,
                highlighted_nodes=[mc_state],
                highlight_color=mc_state_color[mc_state],
                ax=axes[1]
            )
        plot_observation_mc(
            observation_mc_states=so_problem.observation_mc_states,
            observation_mc_transitions=so_problem.observation_mc_transitions,
            observation_mc_obs_states=so_problem.observation_mc_obs_states,
            observation_mc_loc_states=so_problem.observation_mc_loc_states,
            highlighted_nodes=[states_traversed[frame_number][1],],
            highlight_color='xkcd:orange',
            ax=axes[1]
        )
        
        axes[1].set_title("Step {}".format(frame_number))
        
        return

        
    fig, axes = plt.subplots(nrows=2,figsize=[10,10])
    
    # We need one MC state to plot the corresponding state values
    for representative_mc_state in so_problem.observation_mc_obs_states: 
#     representative_mc_state = so_problem.observation_mc_states[0]
    
        mc_state_values = {k[0]: state_values[k] for k in state_values.keys() if k[1] == representative_mc_state}
        
        plot_grid_world_values(
            mc_state_values,
            fig=fig,
            ax=axes[0],
            colormap='RdYlGn',
            alpha = .2/len(so_problem.observation_mc_obs_states),
            plot_colorbar=False,
            vmin=None,
            vmax=None,
        )
        
    mdp = MDP()
    # Terminal states
    mdp.terminal_states = so_problem.terminal_states
    ts_patches = plot_grid_world_terminal_states(
        mdp,
        x_offset=-.5,
        y_offset=-.5,
        linewidth=3,
        colormap='RdYlGn',
        edgecolor='w',
        alpha=None,
        plot_colorbar=False
    )
    for patch in ts_patches:
        axes[0].add_patch(patch)
        
    # States traversed
    ro_patches = plot_grid_world_rollout(rollout_states=[s[0] for s in states_traversed], linewidth=.5,color='g')
    ro_plotted_patches = [None]*len(states_traversed)
    for patch_ix, patch in enumerate(ro_patches):
        ro_plotted_patches[patch_ix] = axes[0].add_patch(patch)
    
    # Check-ins
    mc_state_colormap = mpl.colormaps['Paired']
    mc_state_color = {
        mc_state: mc_state_colormap(mc_state_ix/len(so_problem.observation_mc_obs_states))
        for mc_state_ix, mc_state in enumerate(so_problem.observation_mc_obs_states)
    }
        

    # Plot the check-ins
    check_in_patches = []
    check_in_patches_plotted = [None]*len(states_traversed)
    for state_ix, state in enumerate(states_traversed):
        mc_state = state[1]
        a_state = state[0]
        if mc_state in so_problem.observation_mc_obs_states:
            _patch = mpatches.Circle(a_state,.2,facecolor=mc_state_color[mc_state])
            check_in_patches.append(_patch)
            check_in_patches_plotted[state_ix] = axes[0].add_patch(_patch)
    
    
    # Plot the actions
    action_patches = []
    action_patches_plotted = [None]*len(states_traversed)
    for state_ix, state in enumerate(states_traversed[:-1]):
        a_state = state[0]
        _action_patch = plot_multi_step_action(a_state, upcoming_action_sequence[state_ix],color='b')
        action_patches.append(_action_patch)
        action_patches_plotted[state_ix] = [axes[0].add_patch(_p) for _p in _action_patch]

    # Plot the current position
    current_pos_patch = axes[0].plot(states_traversed[0][0][0],states_traversed[0][0][1],'*m',markersize=10)
        
    # Plot the observation MC
#     for mc_state in so_problem.observation_mc_obs_states:
#         plot_observation_mc(
#             observation_mc_states=so_problem.observation_mc_states,
#             observation_mc_transitions=so_problem.observation_mc_transitions,
#             observation_mc_obs_states=so_problem.observation_mc_obs_states,
#             observation_mc_loc_states=so_problem.observation_mc_loc_states,
#             highlighted_nodes=[mc_state],
#             highlight_color=mc_state_color[mc_state],
#             ax=axes[1]
#         )
        
    axes[0].set_axis_off()
    
    anim = animation.FuncAnimation(fig, draw_frame, frames=len(states_traversed), interval=interval_ms)
    
    return anim
    # Test the animation