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
from turtlebot_aruco.mdp_schedule.periodic_observations_mdp_creation import *
from turtlebot_aruco.mdp_schedule.periodic_observations_mdp_utils import *
import multiprocessing as mp

def iterative_action_growth_value_iteration(
    x_steps,
    y_steps,
    one_step_actions_and_transitions,
    one_step_action_costs,
    actions_between_checkins,
    terminal_states,
    wall_penalty,
    discount_factor = .9,
    relative_value_threshold = 0.00025,
    verbose=True,
    parallel=False,
    timer_function=time.perf_counter, # or process_time
):
    
    if discount_factor==1 and one_step_action_costs:
        raise ValueError("If discount_factor is set to 1, one_step_action_costs should be unset to allow adding costs for waiting")
    if discount_factor!=1 and (one_step_action_costs is None):
        raise ValueError("If discount_factor is set to <1, one_step_action_costs should be set")
    
    # Do we solve MDPs in parallel? Expand them?
    parallel_solve = parallel
    parallel_expand = parallel
    
    # Bookkeeping, because parallelism is hard.
    mdp_ubs = {}
    mdp_lbs = {}
    values_lb_dict = {}
    values_ub_dict = {}
    state_action_values_lb_dict = {}
    state_action_values_ub_dict = {}
    non_dominated_actions_dict = {}
    
    def vprint(args):
        if verbose:
            print(args)
    
    
    # Start counting time
    overall_tic = timer_function()
    
    # Time bookkeeping
    overall_compute_time = 0.
    overall_expand_time = 0.
    
    if parallel:
        pool = mp.Pool(processes=2)

    # Create the one-step UB and LB    
    vprint("Creating upper bound")
    
    # TODO this needs to change to align with the paper
    if one_step_action_costs:
        mdp_ub_one_step_action_costs = one_step_action_costs
    else:
        mdp_ub_one_step_action_costs = 1
    mdp_ub = multi_step_grid_world_mdp(
        x_steps=x_steps,
        y_steps=y_steps,
        one_step_actions_and_transitions=one_step_actions_and_transitions,
        actions_between_checkins=1,
        action_cost=mdp_ub_one_step_action_costs, #1,
        terminal_states=terminal_states,
        wall_penalty=wall_penalty,
        verbose=verbose,
    )
    mdp_ubs[1] = mdp_ub
    
    # When expanding the upper-bound MDP, we will use the transitions and rewards from the problem above.
    mdp_ub_one_step = copy.deepcopy(mdp_ub)
    
    
    # We only update the upper bound if the epoch is a divisor of the stride.
    # The variable ub_updated helps us keep track of when to re-solve the upper bound.
    ub_updated=True
    # And the variable retrieve_ub_parallel helps with bookkeeping if we solve the MDP in parallel
    retrieve_ub_parallel = False #Bookkeeping 
    
    vprint("Creating lower bound")
    
    if one_step_action_costs:
        mdp_lb_one_step_action_costs = one_step_action_costs
        mdp_lb = copy.deepcopy(mdp_ub) # If we do have gamma != 1 and action costs, mdp_lb and mdp_ub are identical!
    else:
        mdp_lb_one_step_action_costs = actions_between_checkins
        mdp_lb = multi_step_grid_world_mdp(
            x_steps=x_steps,
            y_steps=y_steps,
            one_step_actions_and_transitions=one_step_actions_and_transitions,
            actions_between_checkins=1,
            action_cost=mdp_lb_one_step_action_costs, #actions_between_checkins,  # The cost of one action + actions_between_checkins-1 noops
            terminal_states=terminal_states,
            wall_penalty=wall_penalty,
            verbose=verbose,
        )
    
    mdp_lbs[1] = mdp_lb
    
    # For the lower-bound MDP, we will use a one-step problem with zero step cost, since all the cost
    # is contabilized at step zero
    if one_step_action_costs:
        mdp_lb_one_step_action_costs_base = one_step_action_costs
        mdp_lb_one_step = copy.deepcopy(mdp_ub) # If we do have gamma != 1 and action costs, mdp_lb and mdp_ub are identical!
    else:
        mdp_lb_one_step_action_costs_base = 0.
        mdp_lb_one_step = multi_step_grid_world_mdp(
            x_steps=x_steps,
            y_steps=y_steps,
            one_step_actions_and_transitions=one_step_actions_and_transitions,
            actions_between_checkins=1,
            action_cost=mdp_lb_one_step_action_costs_base, # 0.,  # The cost of one action + actions_between_checkins-1 noops
            terminal_states=terminal_states,
            wall_penalty=wall_penalty,
            verbose=verbose,
        )
    
    guess_values_ubs = {1: None}
    guess_values_lbs = {1: None}
    
    # For every epoch, starting at 1
    for current_action_length_epoch in range(1,actions_between_checkins):
        epoch_tic = timer_function()
        vprint("\nEpoch (action length) {}/{}".format(current_action_length_epoch, actions_between_checkins))
        parallel_solve = (parallel and len(mdp_ubs[current_action_length_epoch].actions)>1e3)
#         if (parallel_solve != parallel) and verbose:
#             print("Solving MDP serially, parallelism will probably not help (there are only {} actions)".format(len(mdp_ub.actions)))
        
        # Solve the UB and the LB
        solve_tic = timer_function() 
        if ub_updated is True:
            vprint("Solving upper bound")
               
            if parallel_solve:
                retrieve_ub_parallel = True # This is true if we computed the UB in the parallel loop. If we don't check, we may end up retrieving a past result!
                vprint("Parallel UB value iteration")
                res_ub = pool.apply_async(
                    state_action_value_iteration,
                    (mdp_ubs[current_action_length_epoch],
                     int(1e4),
                     discount_factor**current_action_length_epoch,
                     relative_value_threshold,
                     verbose,
                     np.nan, guess_values_ubs[current_action_length_epoch]))
            else:
                policy_ub, values_lb_dict[current_action_length_epoch], state_action_values_ub_dict[current_action_length_epoch] = state_action_value_iteration(
                    mdp_ubs[current_action_length_epoch],
                    discount_factor = discount_factor**current_action_length_epoch,
                    relative_value_threshold = relative_value_threshold,
                    print_progress=verbose,
                    initial_values_guess = guess_values_ubs[current_action_length_epoch],        
                )
            ub_updated=False
        else:
            # We reuse the state-action guesses fro
            retrieve_ub_parallel = False
            state_action_values_ub_dict[current_action_length_epoch] = guess_values_ubs[current_action_length_epoch]
            vprint("Skipping solving upper bound, epoch {} is not a divisor of stride {}".format(current_action_length_epoch, actions_between_checkins))
        
        vprint("Solving lower bound")
        if parallel_solve:
            vprint("Parallel LB value iteration")
            res_lb = pool.apply_async(
                state_action_value_iteration,
                (mdp_lbs[current_action_length_epoch],
                 int(1e4),
                 discount_factor**actions_between_checkins,
                 relative_value_threshold,
                 verbose,
                 np.nan,
                 guess_values_lbs[current_action_length_epoch]))
        else:
            policy_lb, values_lb_dict[current_action_length_epoch], state_action_values_lb_dict[current_action_length_epoch] = state_action_value_iteration(
                mdp_lbs[current_action_length_epoch],
                discount_factor = discount_factor**actions_between_checkins,
                relative_value_threshold = relative_value_threshold,
                print_progress=verbose,
                initial_values_guess = guess_values_lbs[current_action_length_epoch],
            )

        if parallel_solve:
            (
                policy_lb,
                values_lb_dict[current_action_length_epoch],
                state_action_values_lb_dict[current_action_length_epoch]
            ) = res_lb.get()
            if retrieve_ub_parallel is True:
                (
                    policy_ub,
                     values_ub_dict[current_action_length_epoch],
                    state_action_values_ub_dict[current_action_length_epoch]
                ) = res_ub.get()
        overall_compute_time = timer_function() - solve_tic
        vprint("Time to solve: {} s".format(overall_compute_time))
        
        # Prune the actions
        vprint("Pruning actions")
        prune_tic = timer_function()
        non_dominated_actions_dict[current_action_length_epoch] = prune_dominated_actions(
            state_action_values_ub_dict[current_action_length_epoch],
            values_lb_dict[current_action_length_epoch],
            _mdp_ub=mdp_ubs[current_action_length_epoch],
            _mdp_lb=mdp_lbs[current_action_length_epoch],
            verbose=verbose
        )
        prune_toc = timer_function() - prune_tic
        vprint("Time to prune: {} s".format(prune_toc))
        
        # Expand the actions for the next UB and LB.
        # Here we will have actions of length (current_action_length_epoch+1)
        # Add one step penalty to UB. In the previous step, we considered all actions of length i
        # followed by a check-in. Now we consider a tighter bound, i.e. all actions of length i+1
        # followed by a check-in (which is still an UB on all action of length k followed by a check-in)
        vprint("Expanding upper bound")
            
        expand_tic = timer_function()
#         print("Non dominated actions: {}".format(non_dominated_actions))
        if one_step_action_costs:
            ub_terminal_state_step_penalty=0.
        else:
            ub_terminal_state_step_penalty=1. # Penalty for spending an extra time step in terminal state
        if parallel_expand:
            exp_ub = pool.apply_async(
                expand_multi_step_grid_world_mdp_v2,
                (
                    mdp_ubs[current_action_length_epoch],
                    mdp_ub_one_step,
                    mdp_ub_one_step.actions,
                    append_action_function_grid_world,
                    non_dominated_actions_dict[current_action_length_epoch],
                    state_action_values_ub_dict[current_action_length_epoch],
                    ub_terminal_state_step_penalty,
                    wall_penalty,
                )
            )
                
        else:
            mdp_ubs[current_action_length_epoch+1], guess_values_ubs[current_action_length_epoch+1] = expand_multi_step_grid_world_mdp_v2(
                base_mdp=mdp_ubs[current_action_length_epoch],
                suffix_mdp=mdp_ub_one_step,
                one_step_actions = mdp_ub_one_step.actions,
                append_action_function=append_action_function_grid_world,
                non_dominated_actions=non_dominated_actions_dict[current_action_length_epoch],
                sa_values = state_action_values_ub_dict[current_action_length_epoch],
                terminal_state_step_penalty=ub_terminal_state_step_penalty,
                action_unavailable_penalty=wall_penalty,
            )
            
        
        if (actions_between_checkins % (current_action_length_epoch+1)) == 0:
            vprint("UB will be solved at next iteration: {} is a divisor of {}".format(current_action_length_epoch+1, actions_between_checkins))
            ub_updated = True
        else:
            vprint("UB will *not* be re-solved at next iteration: {} is not a divisor of {}".format(current_action_length_epoch+1, actions_between_checkins))
            ub_updated = False

        # Zero penalty in the LB MDP: we already paid a cost of "stride" for each action at step 1,
        # now we add on multiple actions for free (think of it as replacing noops with actions)
        
        # Only update the LB if we will use it at the next time step
        if current_action_length_epoch+1<actions_between_checkins:
            vprint("Expanding lower bound")
            if one_step_action_costs:
                lb_terminal_state_step_penalty=0.
                # This is identical to the UB!
                if parallel_expand:
                    pass # We will copy from the UB later, see ~30 lines below
                else:
                    mdp_lbs[current_action_length_epoch+1] = copy.deepcopy(mdp_ubs[current_action_length_epoch+1])
                # Expand the LB 
                guess_values_lbs[current_action_length_epoch+1] = extend_mdp_sa_values(
                    base_mdp=mdp_lbs[current_action_length_epoch],
                    one_step_actions=mdp_lb_one_step.actions,
                    append_action_function=append_action_function_grid_world,
                    non_dominated_actions=non_dominated_actions_dict[current_action_length_epoch],
                    sa_values=state_action_values_lb_dict[current_action_length_epoch],
                )
            else:
                # Expand separately
                lb_terminal_state_step_penalty=0. # Penalty for spending an extra time step in terminal state already charged            
                if parallel_expand:
                    exp_lb = pool.apply_async(
                        expand_multi_step_grid_world_mdp_v2,
                        (
                            mdp_lbs[current_action_length_epoch],
                            mdp_lb_one_step,
                            mdp_lb_one_step.actions,
                            append_action_function_grid_world,
                            non_dominated_actions_dict[current_action_length_epoch],
                            state_action_values_lb_dict[current_action_length_epoch],
                            lb_terminal_state_step_penalty,
                            wall_penalty,
                        )
                    )
                else:
                    mdp_lbs[current_action_length_epoch+1], guess_values_lbs[current_action_length_epoch+1] = expand_multi_step_grid_world_mdp_v2(
                        base_mdp=mdp_lbs[current_action_length_epoch],
                        suffix_mdp=mdp_lb_one_step,
                        one_step_actions = mdp_lb_one_step.actions,
                        append_action_function=append_action_function_grid_world,
                        non_dominated_actions=non_dominated_actions_dict[current_action_length_epoch],
                        sa_values = state_action_values_lb_dict[current_action_length_epoch],
                        terminal_state_step_penalty=lb_terminal_state_step_penalty,
                        action_unavailable_penalty=wall_penalty,
                    )
        else:
            vprint("Last step, no need to expand the LB")
            mdp_lbs[current_action_length_epoch+1] = None
#             guess_values_lbs[current_action_length_epoch+1] = TODO

        if parallel_expand:
            mdp_ubs[current_action_length_epoch+1], guess_values_ubs[current_action_length_epoch+1] = exp_ub.get()
            if current_action_length_epoch+1<actions_between_checkins:# If it's not the final step
                if (one_step_action_costs is None):  # and we didn't just copy over the solution from the UB
                    mdp_lbs[current_action_length_epoch+1], guess_values_lbs[current_action_length_epoch+1] = exp_lb.get()
                else: # Just copy over the solution from the UB
                    mdp_lbs[current_action_length_epoch+1] = copy.deepcopy(mdp_ubs[current_action_length_epoch+1])
        overall_expand_time = timer_function() - expand_tic
        vprint("Time to expand UB and LB: {} s".format(overall_expand_time))
        
        epoch_toc = timer_function() - epoch_tic
        vprint("Time for epoch {}/{}: {} s".format(current_action_length_epoch, actions_between_checkins-1, epoch_toc))
    vprint("Solving full, pruned problem")
    # When epoch=action_between_checkins, ub and lb collapse, just solve one
    solve_full_tic = timer_function()
    policy, values, state_action_values = state_action_value_iteration(
        mdp_ubs[actions_between_checkins],
        discount_factor = discount_factor**actions_between_checkins,
        relative_value_threshold = relative_value_threshold,
        print_progress=verbose,
        initial_values_guess = guess_values_ubs[actions_between_checkins],
    )
    solve_full_toc = timer_function() - solve_full_tic
    overall_compute_time += solve_full_toc
    vprint("Time to solve final pruned problem: {} s".format(solve_full_toc))
    overall_time = timer_function()-overall_tic
    
    vprint("Total time to solve full problem: {} s".format(overall_time))
    compute_times = {"overall": overall_time, "solve": overall_compute_time, "expand": overall_expand_time}
    return policy, values, state_action_values, mdp_ubs[actions_between_checkins], compute_times

def naive_action_growth_value_iteration(
    x_steps,
    y_steps,
    one_step_actions_and_transitions,
    one_step_action_costs=None,
    actions_between_checkins=3,
    terminal_states={},
    wall_penalty=50,
    discount_factor = .9,
    relative_value_threshold = 0.00025,
    verbose=True,
    timer_function=time.perf_counter, # or time.process_time,
):
    def vprint(args):
        if verbose:
            print(args)
    overall_tic = timer_function()
    # Create the one-step UB and LB    
    vprint("Creating problem bound")
    
    create_problem_tic = timer_function()
    
    if one_step_action_costs is None:
        one_step_action_costs=1
    mdp = multi_step_grid_world_mdp(
        x_steps=x_steps,
        y_steps=y_steps,
        one_step_actions_and_transitions=one_step_actions_and_transitions,
        actions_between_checkins=actions_between_checkins,
        action_cost=one_step_action_costs, #1,
        terminal_states=terminal_states,
        wall_penalty=wall_penalty,
        verbose=verbose,
    )
    create_problem_toc = timer_function() - create_problem_tic
    vprint("Time to create full problem: {} s".format(create_problem_toc))
    
    vprint("Solving full, unpruned problem")
    # When epoch=action_between_checkins, ub and lb collapse, just solve one
    solve_full_tic = timer_function()
    policy, values, state_action_values = state_action_value_iteration(
        mdp,
        discount_factor = discount_factor**actions_between_checkins,
        relative_value_threshold = relative_value_threshold,
        print_progress=verbose,
    )
    solve_full_toc = timer_function() - solve_full_tic
    vprint("Time to solve naive problem: {} s".format(solve_full_toc))
    overall_time = timer_function()-overall_tic
    vprint("Total time to expand and solve naive problem: {} s".format(overall_time))
    compute_times = {"overall": overall_time, "solve": solve_full_toc, "expand": create_problem_toc}

    return policy, values, state_action_values, mdp, compute_times