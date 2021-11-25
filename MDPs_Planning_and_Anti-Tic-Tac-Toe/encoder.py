import numpy as np
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Intake Player i policy and Player j state file')

    parser.add_argument('--policy', nargs='?', default='blank', type=argparse.FileType('r'))
    parser.add_argument('--states', nargs='?', default='blank', type=argparse.FileType('r'))

    args = parser.parse_args()

    opponent = None
    opponent_policy = {}
    agent_states = []

    for line in args.policy:
        contents = line.split()
        if opponent is None:
            opponent = contents[0]  # String
        else:
            opponent_policy[contents[0]] = [float(contents[i]) for i in range(1, 10)]

    if opponent == '1':
        agent = '2'
    else:
        agent = '1'

    for line in args.states:
        contents = line.split()
        agent_states.append(contents[0])

    problem_dict = {'agent': agent, 'opponent': opponent, 'opponent_policy': opponent_policy,
                    'agent_states': agent_states}

    return problem_dict


def check_win(state, agent):
    BLANK = 0
    grid = [[int(state[3 * i + j]) for j in range(3)] for i in range(3)]
    grid = np.array(grid)

    for i in range(3):
        if grid[i][0] == grid[i][1] and grid[i][1] == grid[i][2] and grid[i][0] != BLANK:
            if grid[i][0] == int(agent):
                return 0
            else:
                return 1

    for j in range(3):
        if grid[0][j] == grid[1][j] and grid[1][j] == grid[2][j] and grid[0][j] != BLANK:
            if grid[0][j] == int(agent):
                return 0
            else:
                return 1
    if grid[0][0] == grid[1][1] and grid[1][1] == grid[2][2] and grid[1][1] != BLANK:
        if grid[0][0] == int(agent):
            return 0
        else:
            return 1
    if grid[2][0] == grid[1][1] and grid[1][1] == grid[0][2] and grid[1][1] != BLANK:
        if grid[2][0] == int(agent):
            return 0
        else:
            return 1
    if (grid == BLANK).sum() == 0:
        return 0
    return 0


def getTransitions(problem_dict, mdp_dict):
    agent = problem_dict['agent']
    opponent = problem_dict['opponent']
    agent_states = problem_dict['agent_states']
    opponent_policy = problem_dict['opponent_policy']
    numStates = mdp_dict['numStates']
    # numActions = mdp_dict['numActions']

    # TRMatrix = np.zeros((numStates, numStates, numActions, 2))
    transitions = []

    for i, base_state in enumerate(agent_states):
        valid_actions = []
        for j, pos in enumerate(base_state):
            if pos == "0":
                valid_actions.append(j)

        for action in valid_actions:
            intermediate = base_state[:action] + agent + base_state[(action + 1):]
            if intermediate.count("0") > 0:
                if check_win(intermediate, opponent):
                    transitions.append([i, action, numStates - 1, 0, 1])
                else:
                    probabilities = opponent_policy[intermediate]
                    for ind, prob in enumerate(probabilities):
                        if prob != 0:
                            fstate = intermediate[:ind] + opponent + intermediate[(ind + 1):]
                            if fstate.count("0") > 0:
                                if check_win(fstate, agent):
                                    transitions.append([i, action, numStates - 2, 1, 1])
                                else:
                                    end_index = agent_states.index(fstate)
                                    transitions.append([i, action, end_index, 0, prob])

                            else:
                                if check_win(fstate, agent):
                                    transitions.append([i, action, numStates - 2, 1, prob])
                                else:
                                    transitions.append([i, action, numStates - 1, 0, prob])

            else:
                if check_win(intermediate, agent):
                    transitions.append([i, action, numStates - 2, 1, 1])
                else:
                    transitions.append([i, action, numStates - 1, 0, 1])

    return transitions


def encode(problem_dict):
    mdp_dict = {'numStates': (len(problem_dict['agent_states']) + 2), 'numActions': 9,
                'end': [len(problem_dict['agent_states']), len(problem_dict['agent_states']) + 1],
                'mdptype': 'episodic', 'discount': 1}

    mdp_dict['transition'] = getTransitions(problem_dict, mdp_dict)

    # print mdp_dict
    to_print = ['numStates', 'numActions', 'end', 'transition', 'mdptype', 'discount']
    for i, param in enumerate(to_print):
        if i in [0, 1, 2, 4, 5]:
            if param == "end":
                print(f"{param} {mdp_dict[param][0]} {mdp_dict[param][1]}")
                pass
            else:
                print(f"{param} {mdp_dict[param]}")

        else:
            for line in mdp_dict['transition']:
                print(f"{param} {line[0]} {line[1]} {line[2]} {line[3]} {line[4]}")

    return


if __name__ == "__main__":
    encode(parse())
