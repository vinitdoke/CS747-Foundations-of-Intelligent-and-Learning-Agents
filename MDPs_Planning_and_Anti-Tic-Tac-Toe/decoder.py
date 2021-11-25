import argparse


def parse():
    parser = argparse.ArgumentParser(description='Intake Policy, States, Player ID')

    parser.add_argument('--value-policy', nargs='?', default='blank', type=argparse.FileType('r'))
    parser.add_argument('--states', nargs='?', default='blank', type=argparse.FileType('r'))
    parser.add_argument('--player-id', nargs='?', default='blank', type=str)

    args = parser.parse_args()

    actions = []
    agent = args.player_id
    agent_states = []

    for line in args.value_policy:
        contents = line.split()
        if len(contents) > 0:
            actions.append(int(contents[1]))

    for line in args.states:
        contents = line.split()
        if len(contents) > 0:
            agent_states.append(contents[0])

    my_dict = {'agent': agent, 'agent_states': agent_states, 'actions': actions}

    return my_dict


def print_policy(my_dict):
    print(my_dict['agent'])

    for i, state in enumerate(my_dict['agent_states']):
        appender = ""
        for j in range(9):

            if my_dict['actions'][i] == j:
                appender += "1 "
            else:
                appender += "0 "
        print(state + " " + appender[:-1])

    return


if __name__ == "__main__":
    print_policy(parse())
