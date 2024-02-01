from docplex.mp.model import Model
import time

# Efficient DOCPLEX code:
# https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/efficient.ipynb
# Test case paper example 2A with 3 walls and checkin 5: 
#   OG: 17.5s
#   scal_prod instead of sum: 13.7s
#   batch constraints: 8.8s
#   argument checker off: 7.3
# -> 58% faster

# vS: start layer values
# vE: end layer values. For single layer (normal value iteration), vS = vE
def makeConstraint(mdp, discount, lp, vS, vE, state, action, is_negative):
    if not is_negative:
        a1 = [vE[end_state] for end_state in mdp.transitions[state][action].keys()]
    else:
        a1 = [-vE[end_state] for end_state in mdp.transitions[state][action].keys()]
    a2 = [mdp.transitions[state][action][end_state] for end_state in mdp.transitions[state][action].keys()]
    leSum = lp.scal_prod(a1, a2)

    if not is_negative:
        return vS[state] >= (mdp.rewards[state][action] + discount * leSum)
    else:
        return vS[state] <= -(mdp.rewards[state][action] + discount * leSum)

def makeConstraintsList(mdp, discount, lp, vS, vE, restricted_action_set, is_negative):
    return [makeConstraint(mdp, discount, lp, vS, vE, state, action, is_negative) for state in mdp.states for action in (mdp.actions if restricted_action_set is None else restricted_action_set[state]) if action in mdp.transitions[state]]

def linearProgrammingSolve(mdp, discount, restricted_action_set = None, is_negative = False):

    time_start = time.time()

    lp = Model(ignore_names=True, checker='off')
    v = lp.continuous_var_dict(keys = mdp.states, name = "v")

    lp.v_sum = lp.sum_vars(v)

    if not is_negative:
        objective = lp.minimize(lp.v_sum)
    else:
        objective = lp.maximize(lp.v_sum)

    lp.add_constraints(makeConstraintsList(mdp, discount, lp, v, v, restricted_action_set, is_negative))
    
    time_elapsed = (time.time() - time_start)
         
    print("time to create the model: "+str(time_elapsed))
        
    # print("ILP was made") 
        
    # print("------Solve the model--------------")
        
    # print("Solve the ILP\n")

    time_start = time.time()
    lp.solve()

    time_elapsed = (time.time() - time_start)
         
    print("time to solve the model: "+str(time_elapsed))

    # print(lp.solution) 
    # print(lp.solution.get_value_dict(v))

    values = lp.solution.get_value_dict(v)
    if is_negative:
        values = {state: -values[state] for state in values}

    policy = {}
    for state in mdp.states:
        best_action = None
        max_expected = None
        
        action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
        for action in action_set:
            if action in mdp.transitions[state]:
                expected_value = mdp.rewards[state][action]
                for end_state in mdp.transitions[state][action].keys():
                    prob = mdp.transitions[state][action][end_state]
                    expected_value += discount * prob * values[end_state]

                if max_expected is None or expected_value > max_expected:
                    best_action = action
                    max_expected = expected_value

        if max_expected is None:
            max_expected = 0
        
        policy[state] = best_action

    return policy, values

def linearProgrammingSolveMultiLayer(mdps, discounts, restricted_action_set = None, is_negative = False):

    time_start = time.time()

    lp = Model(ignore_names=True, checker='off')
    v_layers = [lp.continuous_var_dict(keys = mdps[i].states, name = f"v{i}") for i in range(len(mdps))]

    lp.v_sum = lp.sum_vars(v_layers[0])

    if not is_negative:
        objective = lp.minimize(lp.v_sum)
    else:
        objective = lp.maximize(lp.v_sum)

    for i in range(len(mdps)):
        mdp = mdps[i]
        discount = discounts[i]
        vS = v_layers[i]
        vE = v_layers[(i+1) % len(v_layers)] # last layer wraps around to first layer
        lp.add_constraints(makeConstraintsList(mdp, discount, lp, vS, vE, restricted_action_set, is_negative))
    
    time_elapsed = (time.time() - time_start)
         
    # print("time to create the model: "+str(time_elapsed))
        
    # print("ILP was made") 
        
    # print("------Solve the model--------------")
        
    # print("Solve the ILP\n")

    time_start = time.time()
    lp.solve()

    time_elapsed = (time.time() - time_start)
         
    # print("time to solve the model: "+str(time_elapsed))

    # print(lp.solution) 
    # print(lp.solution.get_value_dict(v))

    value_layers = []
    policy_layers = []

    for i in range(len(mdps)):
        mdp = mdps[i]
        v = v_layers[i]
        discount = discounts[i]

        values = lp.solution.get_value_dict(v)
        if is_negative:
            values = {state: -values[state] for state in values}
        
        policy = {}
        for state in mdp.states:
            best_action = None
            max_expected = None
            
            action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
            for action in action_set:
                if action in mdp.transitions[state]:
                    expected_value = mdp.rewards[state][action]
                    for end_state in mdp.transitions[state][action].keys():
                        prob = mdp.transitions[state][action][end_state]
                        expected_value += discount * prob * values[end_state]

                    if max_expected is None or expected_value > max_expected:
                        best_action = action
                        max_expected = expected_value

            if max_expected is None:
                max_expected = 0
            
            policy[state] = best_action

        value_layers.append(values)
        policy_layers.append(policy)

    return policy_layers, value_layers