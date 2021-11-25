import argparse
import numpy as np


def parse():
    """Returns a dictionary of parameters"""

    parser = argparse.ArgumentParser(description='Instructions')

    parser.add_argument('--instance', nargs='?', default='blank', type=argparse.FileType('r'))
    parser.add_argument('--algorithm', nargs='?', default='blank', type=str)
    parser.add_argument('--randomSeed', nargs='?', default=42, type=int)
    parser.add_argument('--epsilon', nargs='?', default=0.02, type=float)
    parser.add_argument('--scale', nargs='?', default=2, type=float)
    parser.add_argument('--threshold', nargs='?', default=0, type=float)
    parser.add_argument('--horizon', nargs='?', default=0, type=int)

    args = parser.parse_args()

    # dictionary to store all parameters
    parameters = {}

    # get mean values for instance
    mean_rewards = []
    for line in args.instance:
        mean_rewards.append([float(i) for i in line.split()])

    if args.algorithm[-2:] == "t1" or args.algorithm[-2:] == "t2":
        mean_rewards = np.ravel(mean_rewards)
    else:
        mean_rewards = np.array(mean_rewards)

    # get all other arguments and store in dictionary
    parameters['means'] = mean_rewards
    parameters['algorithm'] = args.algorithm
    parameters['randomSeed'] = args.randomSeed
    parameters['epsilon'] = args.epsilon
    parameters['scale'] = args.scale
    parameters['threshold'] = args.threshold
    parameters['horizon'] = args.horizon
    parameters['instance'] = str(args.instance.name)

    return parameters


def bernoulliBandit(mean):
    return np.random.binomial(1, mean, 1)


def initializeInstance(means):
    """returns a multi-armed instance of bernoulli bandits"""
    def banditInstance(arm_index):
        return bernoulliBandit(means[arm_index])

    return banditInstance


def epsilon_greedy_t1(params):
    # Generate Bandit Instance
    means = params['means']
    bandit = initializeInstance(means)
    n_arms = len(means)

    # Algo Specific Variables
    cumulative_rewards = np.zeros(n_arms, dtype=np.uint64)
    times_pulled = np.zeros(n_arms, dtype=np.uint64)
    epsilon = params['epsilon']

    # Initialize Empirical Means by pulling each arm once
    for i in range(n_arms):
        cumulative_rewards[i] += bandit(i)
        times_pulled[i] += 1
    empirical_means = cumulative_rewards / times_pulled

    # Proceed according to epsilon greedy algorithm
    horizon = params['horizon']
    for t in range(n_arms, horizon):
        random = np.random.uniform(0, 1, 1)
        if random <= epsilon:
            index = np.random.randint(0, n_arms, 1)
        else:
            index = np.argmax(empirical_means)

        cumulative_rewards[index] += int(bandit(index))
        times_pulled[index] += 1
        empirical_means[index] = cumulative_rewards[index] / times_pulled[index]

    # Calculate Regret
    max_expected_reward = np.max(means) * horizon
    regret = max_expected_reward - np.sum(cumulative_rewards)

    return regret


def ucb_t1(params):
    # Generate Bandit Instance
    means = params['means']
    bandit = initializeInstance(means)
    n_arms = len(means)

    # Algo Specific Variables
    cumulative_rewards = np.zeros(n_arms, dtype=np.uint64)
    times_pulled = np.zeros(n_arms, dtype=np.uint64)

    # Initialize Empirical Means by pulling each arm once
    for i in range(n_arms):
        cumulative_rewards[i] += bandit(i)
        times_pulled[i] += 1

    empirical_means = cumulative_rewards / times_pulled

    # Proceed according to UCB Algorithm
    horizon = params['horizon']
    for t in range(n_arms, horizon):
        ucb = empirical_means + np.sqrt(2 * np.log(t) / times_pulled)
        index = np.argmax(ucb)

        # Pull and Update
        cumulative_rewards[index] += bandit(index)
        times_pulled[index] += 1
        empirical_means[index] = cumulative_rewards[index] / times_pulled[index]

    # Calculate Regret
    max_expected_reward = np.max(means) * horizon
    regret = max_expected_reward - np.sum(cumulative_rewards)

    return regret


def kl_divergence(a, b):
    """returns kl-divergence"""
    if a == 0:
        return (1 - a) * np.log((1 - a) / (1 - b))
    elif a == 1:
        return a * np.log(a / b)
    else:
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def get_KL_UCB(emp_means, times_pulled, time_step, c=3, precision=1e-3):
    """returns an array of kl-ucb calculated for each arm"""
    t = time_step

    kl_ucb = np.zeros(len(emp_means), dtype=np.float64)
    for i in range(len(emp_means)):
        emp_mean = emp_means[i]
        low = emp_mean
        high = 1
        mid = (low + high) / 2
        rhs = (np.log(t) + c * np.log(np.log(t))) / times_pulled[i]

        while abs(high - low) > precision:
            if kl_divergence(emp_mean, mid) > rhs:
                high = mid
            else:
                low = mid
            mid = (low + high) / 2

        kl_ucb[i] = mid

    return kl_ucb


def kl_ucb_t1(params):
    # Generate Bandit Instance
    means = params['means']
    bandit = initializeInstance(means)
    n_arms = len(means)

    # Algo Specific Variables
    cumulative_rewards = np.zeros(n_arms, dtype=np.uint64)  # check for higher values
    times_pulled = np.zeros(n_arms, dtype=np.uint64)

    # Initialize Empirical Means by pulling each arm once
    for i in range(n_arms):
        cumulative_rewards[i] += bandit(i)
        times_pulled[i] += 1

    empirical_means = cumulative_rewards / times_pulled

    # Proceed according to KL-UCB Algorithm
    horizon = params['horizon']
    for t in range(n_arms, horizon):
        kl_ucb = get_KL_UCB(empirical_means, times_pulled, t)
        index = np.argmax(kl_ucb)

        # Update Params
        cumulative_rewards[index] += bandit(index)
        times_pulled[index] += 1
        empirical_means[index] = cumulative_rewards[index] / times_pulled[index]

    # Calculate Regret
    max_expected_reward = np.max(means) * horizon
    regret = max_expected_reward - np.sum(cumulative_rewards)

    return regret


def thompson_sampling_t1(params):
    # Generate Bandit Instance
    means = params['means']
    bandit = initializeInstance(means)
    n_arms = len(means)

    # Algo Specific Variables
    cumulative_rewards = np.zeros(n_arms, dtype=np.uint64)
    failures = np.zeros(n_arms, dtype=np.uint64)
    times_pulled = np.zeros(n_arms, dtype=np.uint64)

    # Proceed according to Thompson Sampling
    horizon = params['horizon']
    for t in range(0, horizon):
        beta_samples = np.random.beta(cumulative_rewards + 1, failures + 1)
        index = np.argmax(beta_samples)

        # Update Params
        if bandit(index):
            cumulative_rewards[index] += 1
        else:
            failures[index] += 1
        times_pulled[index] += 1

    # Calculate Regret
    max_expected_reward = np.max(means) * horizon
    regret = max_expected_reward - np.sum(cumulative_rewards)

    return regret


def ucb_t2(params):
    # Generate Bandit Instance
    means = params['means']
    bandit = initializeInstance(means)
    n_arms = len(means)

    # Algo Specific Variables
    cumulative_rewards = np.zeros(n_arms, dtype=np.uint64)  # check for higher values
    times_pulled = np.zeros(n_arms, dtype=np.uint64)
    constant = params['scale']

    # Initialize Empirical Means by pulling each arm once
    for i in range(n_arms):
        cumulative_rewards[i] += bandit(i)
        times_pulled[i] += 1

    empirical_means = cumulative_rewards / times_pulled

    # Proceed according to UCB Algorithm
    horizon = params['horizon']
    for t in range(n_arms, horizon):
        ucb = empirical_means + np.sqrt(constant * np.log(t) / times_pulled)
        index = np.argmax(ucb)

        # Pull and Update
        cumulative_rewards[index] += bandit(index)
        times_pulled[index] += 1
        empirical_means[index] = cumulative_rewards[index] / times_pulled[index]

    # Calculate Regret
    max_expected_reward = np.max(means) * horizon
    regret = max_expected_reward - np.sum(cumulative_rewards)

    return regret


def getGeneralBandit(means_data):
    rewards = means_data[0, :]
    probabilities = means_data[1:, :]

    def bandit(arm_to_pull):
        return np.random.choice(rewards, 1, p=probabilities[arm_to_pull])

    return bandit


def alg_t3(params):
    bandit = getGeneralBandit(params['means'])
    rewards = params['means'][0, :]
    means = params['means'][1:, :]
    n_arms = means.shape[0]
    n_rewards = len(rewards)

    # Algo Specific Variables
    cumulative_rewards = np.zeros(n_arms, dtype=np.float64)  # check for higher values
    alphas = np.ones((n_arms, n_rewards), dtype=np.uint64)

    # Proceed according to KL-UCB
    horizon = params['horizon']
    for t in range(0, horizon):
        dir_samples = np.zeros((n_arms, n_rewards))
        for i in range(n_arms):
            dir_samples[i] = np.random.dirichlet(alphas[i], 1)

        final = np.matmul(dir_samples, rewards)
        index = np.argmax(final)

        # Update Params
        reward = bandit(index)
        alphas[index][np.where(rewards == reward)] += 1
        cumulative_rewards[index] += reward

    # Calculate Regret
    expected_rewards = np.matmul(means, rewards)
    max_expected_reward = np.max(expected_rewards)*horizon
    regret = max_expected_reward - np.sum(cumulative_rewards)

    return regret


def alg_t4(params):
    # Generate Bandit Instance
    bandit = getGeneralBandit(params['means'])
    rewards = params['means'][0, :]
    means = params['means'][1:, :]
    n_arms = means.shape[0]
    threshold = params['threshold']

    # Algo Specific Variables
    successes = np.zeros(n_arms, dtype=np.uint64)
    failures = np.zeros(n_arms, dtype=np.uint64)

    # Proceed according to Thompson Sampling
    horizon = params['horizon']
    for t in range(0, horizon):
        beta_samples = np.random.beta(successes + 1, failures + 1)
        index = np.argmax(beta_samples)

        # Update Params
        if bandit(index) > threshold:
            successes[index] += 1
        else:
            failures[index] += 1

    # Calculate highs_regret
    binary = np.where(rewards > threshold, 1, 0)
    probabilities = np.matmul(means, binary)
    max_chance = np.max(probabilities)
    max_HIGHS = max_chance*horizon
    HIGHS = np.sum(successes)
    highs_regret = max_HIGHS - HIGHS

    return highs_regret, HIGHS


def main(parameters):

    # Seeding
    np.random.seed(parameters['randomSeed'])

    # choose algo
    algo = parameters['algorithm']

    algorithms = {'epsilon-greedy-t1': epsilon_greedy_t1, 'ucb-t1': ucb_t1, 'kl-ucb-t1': kl_ucb_t1,
                  'thompson-sampling-t1': thompson_sampling_t1, 'ucb-t2': ucb_t2, 'alg-t3': alg_t3,
                  'alg-t4': alg_t4}

    # Execute Algo
    if algo in algorithms:
        if algo[-2:] in ["t1", "t2", "t3"]:
            parameters["REG"] = algorithms[algo](parameters)  # Executed Here
            parameters["HIGHS"] = 0

            print(
                f"{parameters['instance']}, {parameters['algorithm']}, {parameters['randomSeed']}, {parameters['epsilon']}, {parameters['scale']}, {parameters['threshold']}, {parameters['horizon']}, {np.round(parameters['REG'], 3)}, {parameters['HIGHS']}")
        else:
            parameters["REG"], parameters["HIGHS"] = algorithms[algo](parameters)  # Executed Here
            print(
                f"{parameters['instance']}, {parameters['algorithm']}, {parameters['randomSeed']}, {parameters['epsilon']}, {parameters['scale']}, {parameters['threshold']}, {parameters['horizon']}, {np.round(parameters['REG'], 3)}, {parameters['HIGHS']}")
    else:
        print("Algorithm Not Found")

    return


if __name__ == "__main__":
    parsed_inputs = parse()
    main(parsed_inputs)

    # print("Execution Successful")
