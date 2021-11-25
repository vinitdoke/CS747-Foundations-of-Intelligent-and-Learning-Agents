import subprocess
import os

statesfiles = [None, "data/attt/states/states_file_p1.txt", "data/attt/states/states_file_p2.txt"]
policiesfolder = [None, "task3/p1/policies", "task3/p2/policies"]
valuesfolder = [None, "task3/p1/values", 'task3/p2/values']
basepolicy = "data/attt/policies/p1_policy2.txt"


os.makedirs(policiesfolder[1], exist_ok=True)
os.makedirs(valuesfolder[1], exist_ok=True)
os.makedirs(policiesfolder[2], exist_ok=True)
os.makedirs(valuesfolder[2], exist_ok=True)

initialPlayer = 1
secondPlayer = 2
# if initialPlayer:
#     secondPlayer = 2
# else:
#     secondPlayer = 1


def run_iteration(opponent, player_id, iteration):

    if player_id == secondPlayer:
        policy_filepath = f"{policiesfolder[opponent]}/{iteration - 1}.txt"
        states_filepath = statesfiles[player_id]
        new_policypath = f"{policiesfolder[player_id]}/{iteration}.txt"
        new_valuepath = f"{valuesfolder[player_id]}/{iteration}.txt"
    else:
        policy_filepath = f"{policiesfolder[opponent]}/{iteration}.txt"
        states_filepath = statesfiles[player_id]
        new_policypath = f"{policiesfolder[player_id]}/{iteration}.txt"
        new_valuepath = f"{valuesfolder[player_id]}/{iteration}.txt"

    # print(iteration, policy_filepath, states_filepath)

    with open("encoded_mdp.txt", 'w') as f:
        subprocess.run(f'python encoder.py --policy {policy_filepath} --states {states_filepath}', text=True,
                       shell=True, stdout=f)
    # print("encoded", iteration)

    with open(new_valuepath, 'w') as f:
        subprocess.run(f'python planner.py --mdp encoded_mdp.txt', text=True, shell=True, stdout=f)

    with open(new_policypath, 'w') as f:
        subprocess.run(
            f'python decoder.py --value-policy {new_valuepath} --states {states_filepath} --player-id {player_id}',
            text=True, shell=True, stdout=f)

    os.remove("encoded_mdp.txt")

    return


def main():
    for i in range(1, 11):
        run_iteration(initialPlayer, secondPlayer, i)
        run_iteration(secondPlayer, initialPlayer, i)

    return


def initializePolicy():
    # container = []
    # with open(statesfiles[initialPlayer], 'r') as f:
    #     all_lines = f.readlines()
    #     for line in all_lines:
    #         container.append(line.strip())
    #
    # value_and_actions = []
    # for state in container:
    #     ind = state.index('0')
    #     value_and_actions.append((0, ind))
    #
    # with open(f"{valuesfolder[initialPlayer]}/0.txt", 'w+') as f:
    #     for line in value_and_actions:
    #         f.writelines(f"{line[0]} {line[1]}\n")
    #
    # with open(f"{policiesfolder[initialPlayer]}/0.txt", 'w') as f:
    #     subprocess.run(
    #         f'python decoder.py --value-policy {valuesfolder[initialPlayer]}/0.txt --states {statesfiles[initialPlayer]} --player-id {initialPlayer}',
    #         text=True, shell=True, stdout=f)
    with open(basepolicy, 'r') as firstfile, open(f"{policiesfolder[initialPlayer]}/0.txt", 'a') as secondfile:
        for line in firstfile:
            secondfile.write(line)

    return


if __name__ == "__main__":
    initializePolicy()
    main()
