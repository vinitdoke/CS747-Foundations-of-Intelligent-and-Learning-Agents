import numpy as np
import argparse
import pulp

# import time

tol = 1e-12


def parse():
    """Returns a dictionary of MDP Params"""

    parser = argparse.ArgumentParser(description='Instructions')

    parser.add_argument('--mdp', nargs='?', default='blank', type=argparse.FileType('r'))
    parser.add_argument('--algorithm', nargs='?', default='vi', type=str)  # Default vi

    args = parser.parse_args()

    mdp = {}
    transitions = []

    for line in args.mdp:
        contents = line.split()
        if contents[0] in ["numStates", "numActions", "end", "discount", "mdptype"]:
            if contents[0] == "mdptype":
                mdp[contents[0]] = contents[1]
            elif contents[0] == "discount":
                mdp[contents[0]] = float(contents[1])
            elif contents[0] == "end":
                mdp[contents[0]] = [int(i) for i in contents[1:]]
            else:
                mdp[contents[0]] = int(contents[1])

        else:
            holder = []
            for i in range(3):
                holder.append(int(contents[i + 1]))
            for i in range(2):
                holder.append(float(contents[i + 4]))
            transitions.append(holder)

    mdp["transitions"] = transitions

    mdp["algorithm"] = args.algorithm

    return mdp


def get_pi(mdp, sparseDict, V):
    """ Get Optimal Actions from Optimal Value Function New"""
    numStates = mdp["numStates"]
    numActions = mdp["numActions"]
    discount = mdp["discount"]
    states = sparseDict.keys()

    pi = np.zeros(numStates)

    for state in states:
        Q_vector = np.zeros(numActions)
        actions = sparseDict[state].keys()
        for action in actions:
            final_states = sparseDict[state][action].keys()
            for fstate in final_states:
                reward, prob = sparseDict[state][action][fstate]
                Q_vector[action] += prob * (reward + discount * V[fstate])

        max_act = 0
        for action in actions:
            if Q_vector[action] >= Q_vector[max_act]:
                max_act = action

        pi[state] = max_act
        # pi[state] = np.argmax(Q_vector)

    return pi


def vi(mdp):
    """ Value Iteration New """

    numStates = mdp["numStates"]
    discount = mdp["discount"]
    sparseDict = getSparseDict(mdp)
    states = sparseDict.keys()

    V0 = np.ones(numStates)
    V1 = np.zeros(numStates)

    while np.linalg.norm(V1 - V0) > tol:
        V0 = np.copy(V1)

        for state in states:
            collate = []
            actions = sparseDict[state].keys()
            for action in actions:
                final_states = sparseDict[state][action].keys()
                action_sum = 0
                for fstate in final_states:
                    reward, prob = sparseDict[state][action][fstate]
                    action_sum += prob * (reward + discount * V0[fstate])
                collate.append(action_sum)
            V1[state] = np.max(collate)

    pi_star = get_pi(mdp, sparseDict, V0)

    return V1, pi_star


def value_eval(mdp, policy):
    numStates = mdp["numStates"]
    discount = mdp["discount"]
    SparseDict = mdp['SparseDict']
    states = SparseDict.keys()

    V0 = np.ones(numStates)
    V1 = np.zeros(numStates)

    while not np.array_equal(V1, V0):
        V0 = np.copy(V1)
        for state in states:
            action = policy[state]
            action_sum = 0
            if action in SparseDict[state]:
                final_states = SparseDict[state][action].keys()
                # print(final_states)
                for fstate in final_states:
                    reward, prob = SparseDict[state][action][fstate]
                    action_sum += prob * (reward + discount * V0[fstate])

            V1[state] = action_sum

    return V1


def action_value_eval(mdp, state, action, value):
    SparseDict = mdp['SparseDict']
    discount = mdp["discount"]

    Q = 0
    fstates = SparseDict[state][action].keys()

    for fstate in fstates:
        reward, prob = SparseDict[state][action][fstate]
        Q += prob * (reward + discount * value[fstate])

    return Q


def hpi(mdp):
    """ Howard's Policy Iteration"""
    # mdp['TRMatrix'] = getTRM(mdp)  # [s1, s2, action, (probability, reward)]
    mdp['SparseDict'] = getSparseDict(mdp)
    SparseDict = mdp['SparseDict']
    numStates = mdp["numStates"]
    states = SparseDict.keys()

    # print(numStates)

    policy = np.zeros(numStates, dtype=int)

    # initialize with valid actions
    for state in states:
        actions = SparseDict[state].keys()
        for action in actions:
            policy[state] = int(action)

    value = value_eval(mdp, policy)

    done = False

    while not done:
        done = True

        for i in states:
            IA = None

            actions = SparseDict[i].keys()
            for j in actions:
                action_val = action_value_eval(mdp, i, j, value)

                if action_val > value[i]:
                    IA = j
                    done = False

            if IA is not None:
                policy[i] = IA

        value = value_eval(mdp, policy)

    return value, policy


def lp(mdp):
    """ Linear Programming """
    # TRMatrix = getTRM(mdp)  # [s1, s2, action, (probability, reward)]
    numStates = mdp["numStates"]
    numActions = mdp["numActions"]
    discount = mdp["discount"]
    SparseDict = getSparseDict(mdp)

    model = pulp.LpProblem("model", pulp.LpMaximize)

    # Variables:
    decision_variables = [pulp.LpVariable(f'V{i}') for i in range(numStates)]

    # Objective Function:
    model += pulp.LpAffineExpression([(decision_variables[i], -1) for i in range(numStates)])

    # Constraints
    for state in range(numStates):
        for action in range(numActions):
            temp = 0
            tuplelist = []
            if state in SparseDict.keys():
                if action in SparseDict[state].keys():
                    identity = False
                    for fstate in SparseDict[state][action].keys():
                        if fstate != state:
                            reward, prob = SparseDict[state][action][fstate]
                            tuplelist.append((decision_variables[fstate], discount * prob))
                            temp += - prob * reward
                        else:
                            identity = True
                            reward, prob = SparseDict[state][action][fstate]
                            tuplelist.append((decision_variables[fstate], discount * prob - 1))
                            temp += - prob * reward

                    if not identity:
                        tuplelist.append((decision_variables[state], - 1))

                else:
                    tuplelist.append((decision_variables[state], -1))

            else:
                temp = 0
                tuplelist.append((decision_variables[state], -1))

            model += pulp.LpAffineExpression(tuplelist) <= temp

    # Solve Model
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Fetch Optimal Values
    V_optimal = [decision_variables[i].varValue for i in range(numStates)]

    # Generate Policy from Optimal Values
    policy = get_pi(mdp, SparseDict, V_optimal)

    return V_optimal, policy


# def getTRM(mdp):
#     """ returns 4D Matrix : [S_initial, S_final, Action, [Probability, Reward]] """
#
#     transitions = mdp["transitions"]
#     numStates = mdp["numStates"]
#     numActions = mdp["numActions"]
#
#     matrix = np.zeros((numStates, numStates, numActions, 2))
#     # print(len(transitions[0]))
#
#     for vector in transitions:
#         matrix[vector[0], vector[2], vector[1], 0] = vector[4]
#         matrix[vector[0], vector[2], vector[1], 1] = vector[3]
#
#     return matrix


def getSparseDict(mdp):
    transitions = mdp["transitions"]
    sparse_dict = {}
    for vector in transitions:
        if vector[0] not in sparse_dict:
            sparse_dict[vector[0]] = {}
        if vector[1] not in sparse_dict[vector[0]]:
            sparse_dict[vector[0]][vector[1]] = {}
        sparse_dict[vector[0]][vector[1]][vector[2]] = (vector[3], vector[4])

    return sparse_dict


def plan(mdp_dict):
    function_map = {"vi": vi, "hpi": hpi, "lp": lp}

    if mdp_dict["algorithm"] in function_map:

        # start = time.time()
        optimal_value, optimal_policy = function_map[mdp_dict["algorithm"]](mdp_dict)
        # end = time.time()
        # print(end - start)
        for i in range(mdp_dict["numStates"]):
            print(f"{optimal_value[i]}\t{int(optimal_policy[i])}")
        # print(getSparseDict(mdp_dict))

    else:
        print("Algo Not Found")

    return


if __name__ == "__main__":
    plan(parse())
