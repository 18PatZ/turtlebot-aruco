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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.cm as cm
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from turtlebot_aruco.mdp_schedule.periodic_observations_mdp_bounds import create_recursive_actions_and_transitions

def plot_multi_step_action(start_state, full_intended_action, linewidth=1, color='k'):
    patches = []
    pos = start_state
    vertices = [[pos[0], pos[1]]]
    codes = [mpath.Path.MOVETO]
    action_length=0

    for step_ix, step in enumerate(full_intended_action):
        if step[0] != 0 or step[1] != 0:
            action_length += 1

    if action_length<2:
        line_style = mpath.Path.LINETO
    elif action_length<3:
        line_style = mpath.Path.CURVE3
    elif action_length<4:
        line_style = mpath.Path.CURVE4
    else:
        line_style = mpath.Path.LINETO
    for step_ix, step in enumerate(full_intended_action):
        if step[0] != 0 or step[1] != 0:
            vertices.append([pos[0]+step[0], pos[1]+step[1]])
            codes.append(line_style)
            pos = (pos[0] + step[0], pos[1] + step[1])
#     patches.append(mpatches.PathPatch(mpath.Path(vertices,codes=codes),fill=False,linewidth=linewidth,color='k'))
    if len(vertices)>1:
        patches.append(mpatches.FancyArrowPatch(path=mpath.Path(vertices,codes=codes),arrowstyle="->",mutation_scale=10.,fill=False,linewidth=linewidth,color=color))
    else:
        patches.append(mpatches.Circle(pos,.2,facecolor=color))
#     patches.append(mpatches.Arrow(
#         start_state[0],
#         start_state[1],
#         pos[0]-start_state[0],
#         pos[1]-start_state[1],
#         color='r',
#         linestyle="-.",
#         linewidth=.1,
#         width=.5,
#     ))
#     step_ix = len(full_intended_action)
#     for step in reversed(full_intended_action):
#         pos = (pos[0] - step[0], pos[1] - step[1])
#         if step[0] != 0 or step[1] != 0:
#             patches.append(mpatches.Arrow(pos[0], pos[1], step[0], step[1],color=color))
#             break
#         step_ix -=1
#     if step_ix == 0:
#         print("Weird action of all zeroes: {}".format(full_intended_action))
#         patches.append(mpatches.Circle(pos,radius=.9))

#     patches.append(mpatches.FancyArrowPatch(path=mpath.Path(vertices,codes=codes),fill=False,linewidth=linewidth,color='k'))
    return patches

def plot_grid_world_policy(policy, x_offset=.5, y_offset=.5, x_len=.9, y_len=.9, linewidth=1, linecolor="b"):
    patches = []
    for state, action in policy.items():
        if type(action[1]) is tuple and type(action[0]) is tuple: #Multi-step MDP
            full_intended_action = action[0]  # This is a list of tuples
            pos = (state[0]+x_offset,state[1]+y_offset)
            patches += plot_multi_step_action(start_state=pos, full_intended_action=full_intended_action, linewidth=linewidth, color=linecolor)
        else: # Single-step MDP 
            patches.append(mpatches.Arrow(state[0]+x_offset, state[1]+y_offset, action[0]*x_len, action[1]*y_len, width=linewidth))
    return patches

def plot_grid_world_mdp_legacy(policy, values, policy_linewidth=1, save=False, save_name="mdp.png"):
    """ A class to plot the values of a grid world MDP """
    
    fig, ax = plt.subplots()
    
    x_min = np.float("inf")
    x_max = -np.float("inf")
    y_min = np.float("inf")
    y_max = -np.float("inf")

    cmap = cm.get_cmap('RdYlGn')
    
    patches = []
    def get_normalized_value(value):
        return ((value-np.min(np.array(list(values.values()))))/
                (np.max(np.array(list(values.values())))-np.min(np.array(list(values.values())))))

    for state, value in values.items():
        x_min=min(x_min, state[0])
        y_min=min(y_min, state[1])
        x_max=max(x_max, state[0])
        y_max=max(y_max, state[1])
        patches.append(mpatches.Rectangle(state, 1, 1, color=cmap(get_normalized_value(values[state]))))

    policy_plt = plot_grid_world_policy(policy, linewidth=policy_linewidth)
    patches += policy_plt
        
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max+1)
    ax.axis('equal')
    plt.axis('off')
    if save:
        plt.savefig(save_name)
        
    return fig


def plot_grid_world_mdp(
    policy,
    values,
    policy_linewidth=1,
    policy_color= "#1f77b4",
    mdp=None,
    colormap='RdYlGn',
    values_alpha=1.,
    obstacle_colormap='RdYlGn',
    obstacle_edgecolor='w',
    plot_colorbar=False,
    save=False,
    save_name="mdp.png",
    vmin=None,
    vmax=None,
    fig=None,
    ax=None,
    ): 

    if ax is None:
        fig, ax = plt.subplots()
    
    # x_min = np.float("inf")
    # x_max = -np.float("inf")
    # y_min = np.float("inf")
    # y_max = -np.float("inf")

    # cmap = cm.get_cmap(colormap)
    
    # for state, value in values.items():
    #     x_min=min(x_min, state[0])
    #     y_min=min(y_min, state[1])
    #     x_max=max(x_max, state[0])
    #     y_max=max(y_max, state[1])
        
    # data = np.ones([y_max-y_min+1, x_max-x_min+1])*np.NaN

    # for state, value in values.items():
    #     data[state[1],state[0]] = values[state]

    # im = ax.imshow(data,cmap=cmap)

    fig, ax = plot_grid_world_values(values,fig=fig,ax=ax,colormap=colormap,plot_colorbar=plot_colorbar,alpha=values_alpha,vmin=vmin,vmax=vmax)
    
    policy_plt = plot_grid_world_policy(policy, x_offset=0., y_offset=0., linewidth=policy_linewidth,linecolor=policy_color)        
    patches = policy_plt

    if mdp is not None:
        tspatches = plot_grid_world_terminal_states(mdp, colormap=obstacle_colormap,edgecolor=obstacle_edgecolor)
        patches += tspatches
        
    for patch in patches:
        ax.add_patch(patch)
    # ax.set_xlim(x_min, x_max+1)
    # ax.set_ylim(y_min, y_max+1)
    # if plot_colorbar:
    #     ax.figure.colorbar(im)
    ax.axis('equal')
    ax.set_axis_off()
    # plt.axis('off')
    if save:
        plt.savefig(save_name)
        
    return fig

def plot_grid_world_mdp_policy(policy):
    """
    A class to plot the policies of a grid world MDP
    Regions with the same policy have the same color
    """

    
    fig, ax = plt.subplots()
    
    x_min = np.float("inf")
    x_max = -np.float("inf")
    y_min = np.float("inf")
    y_max = -np.float("inf")
    patches = []
    
    cmap = cm.get_cmap('viridis')
    
    actions = list(set([action for action in policy.values()]))
    
    def get_policy_value(_policy):
        return actions.index(_policy)/len(actions)
        
    for state, _policy in policy.items():
        x_min=min(x_min, state[0])
        y_min=min(y_min, state[1])
        x_max=max(x_max, state[0])
        y_max=max(y_max, state[1])
        patches.append(mpatches.Rectangle(state, 1, 1, color=cmap(get_policy_value(_policy))))

    policy_plt = plot_grid_world_policy(policy)
    patches += policy_plt
        
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max+1)
    ax.axis('equal')
    plt.axis('off')
        
    return fig


def plot_grid_world_mdp_policy_length(policy):
    """
    A class to plot the policies of a grid world MDP
    Color denotes the number of steps in each state's policy
    """

    
    fig, ax = plt.subplots()
    
    x_min = np.float("inf")
    x_max = -np.float("inf")
    y_min = np.float("inf")
    y_max = -np.float("inf")
    patches = []
    
    cmap = cm.get_cmap('viridis')
    
    actions = list(set([action for action in policy.values()]))
    
    
    def get_policy_len(_policy):
        if type(_policy[0]) is str and type(_policy[1]) is tuple:
            _len = 1
            for step in eval(_policy[0]):
                if step[0] != 0 or step[1] != 0:
                    _len+=1
            return _len
        else: #Single-step
            return 1
        
    max_policy_len = 0
    for state, _policy in policy.items():
        x_min=min(x_min, state[0])
        y_min=min(y_min, state[1])
        x_max=max(x_max, state[0])
        y_max=max(y_max, state[1])
        max_policy_len = max(max_policy_len,get_policy_len(_policy))
    for state, _policy in policy.items():
        patches.append(mpatches.Rectangle(state, 1, 1, color=cmap(get_policy_len(_policy)/max_policy_len)))

    policy_plt = plot_grid_world_policy(policy)
    patches += policy_plt
        
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max+1)
    ax.axis('equal')
    plt.axis('off')
        
    return fig

def plot_grid_world_mdp_policy_length_v2(policy, policy_linewidth=1):
    """
    A class to plot the policies of a grid world MDP
    Color denotes the number of steps in each state's policy
    """

    
    fig, ax = plt.subplots()
    
    x_min = np.float("inf")
    x_max = -np.float("inf")
    y_min = np.float("inf")
    y_max = -np.float("inf")
    patches = []
    
    cmap = cm.get_cmap('viridis')
    
    actions = list(set([action for action in policy.values()]))
    
    
    def get_policy_len(_policy):
        if type(_policy[0]) is tuple and type(_policy[1]) is tuple:
            _len = 0
            for step in _policy[0]:
                if step[0] != 0 or step[1] != 0:
                    _len+=1
            return _len
        else: #Single-step
            return 1
        
    for state, _policy in policy.items():
        x_min=min(x_min, state[0])
        y_min=min(y_min, state[1])
        x_max=max(x_max, state[0])
        y_max=max(y_max, state[1])
        

    data = np.ones([y_max-y_min+1, x_max-x_min+1])*np.NaN
    for state, _policy in policy.items():
        data[state[1],state[0]] = get_policy_len(_policy)
        
    im = ax.imshow(data,cmap='cividis')
    
    policy_plt = plot_grid_world_policy(policy, x_offset=0., y_offset=0., linewidth=policy_linewidth)        
    patches += policy_plt
        
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max+1)
    ax.figure.colorbar(im)
    ax.axis('equal')
    plt.axis('off')
        
    return fig

def plot_grid_world_values(
    values,
    fig=None,
    ax=None,
    colormap='RdYlGn',
    alpha = 1.,
    plot_colorbar=False,
    vmin=None,
    vmax=None,
    ): 

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.subplots(111)
    if vmin is None:
        vmin = min(values.values())
    if vmax is None:
        vmax = max(values.values())
    # fig, ax = plt.subplots()
    
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    cmap = cm.get_cmap(colormap)
    
    for state, value in values.items():
        x_min=min(x_min, state[0])
        y_min=min(y_min, state[1])
        x_max=max(x_max, state[0])
        y_max=max(y_max, state[1])
        
    data = np.ones([y_max-y_min+1, x_max-x_min+1])*np.NaN

    for state, value in values.items():
        data[state[1],state[0]] = values[state]

    im = ax.imshow(data,cmap=cmap, alpha=alpha,vmin=vmin,vmax=vmax)
    
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max+1)
    if plot_colorbar:
            colorbar = ax.figure.colorbar(im)
    ax.axis('equal')
    plt.axis('off')
        
    return fig, ax

def plot_grid_world_terminal_states(mdp, x_offset=-.5, y_offset=-.5, linewidth=3, colormap='RdYlGn', edgecolor=None, alpha=None, plot_colorbar=False):
    patches = []
    def get_terminal_color(value):
        vals = list(mdp.terminal_states.values())
        if max(vals)-min(vals) == 0:
            return 0.5
        else:
            return (value-min(vals))/(max(vals)-min(vals))
    
    cmap = cm.get_cmap(colormap)

    for ts, val in mdp.terminal_states.items():
        _facecolor = cmap(get_terminal_color(val))
        if edgecolor is None:
            _edgecolor = _facecolor
        else:
            _edgecolor = edgecolor
        patches.append(mpatches.Rectangle((ts[0]+x_offset, ts[1]+y_offset), 1, 1, facecolor=_facecolor, edgecolor=_edgecolor, linewidth=linewidth, alpha=alpha))
    return patches


def plot_next_state_distribution(one_step_actions_and_transitions, actions_between_checkins, figsize=[12,12]):
    # Create the multi-step distribution
    composite_actions_and_transitions, _ = create_recursive_actions_and_transitions(
        one_step_actions_and_transitions,
        actions_between_checkins-1
    )
    cmap = cm.get_cmap('YlGnBu')
    num_actions = len(composite_actions_and_transitions)
    actions_per_side = int(np.ceil(np.sqrt(num_actions)))
    fig, ax = plt.subplots(nrows=actions_per_side, ncols=actions_per_side,sharex=True,sharey=True, figsize=figsize)
    # For every possible action
    action_ix = 0
    
    for composite_action, next_state_distribution in composite_actions_and_transitions.items():
        pos = (0.5,0.5)  # Center of the 0,0 cell
        patches = []
        action_row = int(action_ix%actions_per_side)
        action_col = int((action_ix-action_row)/actions_per_side)

        # Plot the distribution of future states
        for next_state, next_probability in next_state_distribution.items():
            patches.append(mpatches.Rectangle(next_state, 1, 1, color=cmap(next_probability*10)))
        # Plot the policy
        full_intended_action = composite_action[0]  # This is a list of tuples
        
        patches.append(mpatches.Circle(pos,.2,facecolor='r'))
        
        action_patches = plot_multi_step_action(
            start_state=pos,
            full_intended_action=full_intended_action,
            linewidth=2,
            color='r'
        )
        
        patches += action_patches
        # Plot
        for patch in patches:
            ax[action_row][action_col].add_patch(patch)
            
        action_ix+=1
        ax[action_row][action_col].set_title(str(composite_action[1]))
    
    plt.setp(ax, xlim=[-2*actions_between_checkins, 2*actions_between_checkins+1], ylim=[-2*actions_between_checkins, 2*actions_between_checkins+1])
    return fig

def plot_grid_world_blind_drive(mdp, policy, how_far, vmin=None, vmax=None, state_highlight=None, ax=None, legend=True, cmap=None):
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    
    x_min = float("inf")
    x_max = -float("inf")
    y_min = float("inf")
    y_max = -float("inf")
    patches = []
       
    actions = list(set([action for action in policy.values()]))
    
    def get_normalized_distance(distance):
        return (distance-np.min(np.array(list(how_far.values()))))/(np.max(np.array(list(how_far.values())))-np.min(np.array(list(how_far.values()))))
        
    for state, _policy in policy.items():
        x_min=min(x_min, state[0])
        y_min=min(y_min, state[1])
        x_max=max(x_max, state[0])
        y_max=max(y_max, state[1])
#         patches.append(mpatches.Rectangle(state, 1, 1, color=cmap(get_normalized_distance(how_far.get(state, 0)))))

    data = np.ones([y_max-y_min+1, x_max-x_min+1])*np.NaN
    for state, dist in how_far.items():
        data[state[1],state[0]] = dist
        
    # im = ax.imshow(data,cmap='cividis', vmin=vmin, vmax=vmax)
    if cmap is None:
        cmap = plt.get_cmap('cividis', vmax-vmin + 1)
    im = ax.imshow(data,cmap=cmap, vmin=vmin, vmax=vmax)
        
#     for state, action in policy.items():
#         print(action)
#         patches.append(mpatches.Arrow(
#             state[0],
#             state[1],
#             action[0]*.9,
#             action[1]*.9,))
    
    policy_plt = plot_grid_world_policy(policy, x_offset=0., y_offset=0.)
    patches += policy_plt

    if state_highlight is not None:
        if not isinstance(state_highlight, list):
            state_highlight = [state_highlight]

        for state in state_highlight:
            # pos = (state[0]-0.5/2,state[1]-0.5/2)
            # patches.append(mpatches.Rectangle(pos, width=0.5, height=0.5, color="r", fill=True, alpha=0.5))
            pos = (state[0],state[1])
            patches.append(mpatches.Circle(pos, radius=0.25, color="r", fill=True, alpha=1.0))

            if len(state_highlight) == 1:
                state = state_highlight[0]
                action = policy[state]
                x_offset = 0#.5
                y_offset = 0#.5
                linewidth=2
                linecolor = "r"
                if type(action[1]) is tuple and type(action[0]) is tuple: #Multi-step MDP
                    full_intended_action = action[0]  # This is a list of tuples
                    
                    pos = (state[0]+x_offset,state[1]+y_offset)
                    patches += plot_multi_step_action(start_state=pos, full_intended_action=full_intended_action, linewidth=linewidth, color=linecolor)
        
    
        
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max+1)
    # ax.figure.colorbar(im)
    mat = plt.matshow(data, cmap=cmap, vmin=vmin - 0.5, 
                      vmax=vmax + 0.5)
    
    ax.tick_params(left=False, bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set(yticklabels=[])
    ax.set(xticklabels=[])

    if legend:
        ax.figure.colorbar(mat, ticks=np.arange(vmin, vmax + 1))
    ax.axis('equal')
    ax.axis('off')
        
    return fig

def plot_grid_world_rollout(rollout_states, x_offset=0., y_offset=0., x_len=.9, y_len=.9, linewidth=1,alpha=1, color="#1f77b4"):
    patches = []
    for state_ix, state in enumerate(rollout_states[:-1]):
        next_state = rollout_states[state_ix+1]
        patches.append(mpatches.Arrow(state[0]+x_offset, state[1]+y_offset, (next_state[0]-state[0])*x_len, (next_state[1]-state[1])*y_len, width=linewidth, alpha=alpha, color=color))
    return patches

def plot_grid_world_mdp_values_legacy(values, save=False, save_name="values.png", fig=None, ax=None):
    """ A class to plot the values of a grid world MDP """
    
    if fig is None:
        fig = plt.figure(figsize=(6,6))
    if ax is None:
        ax = fig.subplots(1,1)
    
    x_min = np.float("inf")
    x_max = -np.float("inf")
    y_min = np.float("inf")
    y_max = -np.float("inf")

    cmap = cm.get_cmap('RdYlGn')
    
    patches = []
    def get_normalized_value(value):
        return ((value-np.min(np.array(list(values.values()))))/
                (np.max(np.array(list(values.values())))-np.min(np.array(list(values.values())))))

    for state, value in values.items():
        x_min=min(x_min, state[0])
        y_min=min(y_min, state[1])
        x_max=max(x_max, state[0])
        y_max=max(y_max, state[1])
        patches.append(mpatches.Rectangle(state, 1, 1, color=cmap(get_normalized_value(values[state]))))
        
    for patch in patches:
        ax.add_patch(patch)
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max+1)
    ax.axis('equal')
    plt.axis('off')
    if save:
        plt.savefig(save_name)
        
    return fig, ax