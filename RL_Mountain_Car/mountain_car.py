"""
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

"""

import gym
import numpy as np
# import math
# import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent:
    """
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.

    """

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        self.x_res = 25
        self.v_res = 25
        self.x_space = np.linspace(self.lower_bounds[0], self.upper_bounds[0], self.x_res + 1)
        self.v_space = np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.v_res + 1)
        self.epsilon_T1 = 0.001
        self.epsilon_T2 = 0.001
        self.learning_rate_T1 = 0.1
        self.learning_rate_T2 = 0.1
        self.weights_T1 = np.random.rand(self.x_res * self.v_res, self.env.action_space.n)
        self.tilings = 12
        self.weights_T2 = np.random.rand(self.x_res * self.v_res, self.tilings, self.env.action_space.n)
        self.discount = 0.9
        self.train_num_episodes = 10000
        self.test_num_episodes = 100

        self.x_unit = self.x_space[1] - self.x_space[0]
        self.v_unit = self.v_space[1] - self.v_space[0]

        self.x_offset = self.x_unit / (self.tilings - 1)
        self.v_offset = self.v_unit / (self.tilings - 1)

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):

        x, v = obs
        xd = np.digitize(x, self.x_space)
        vd = np.digitize(v, self.v_space)

        index = xd + (vd - 1) * self.v_res

        return index

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''

    def get_better_features(self, obs):

        x, v = obs

        x_initial = np.linspace(self.lower_bounds[0], self.upper_bounds[0] + self.x_unit, self.x_res + 1)
        v_initial = np.linspace(self.lower_bounds[1], self.upper_bounds[1] + self.v_unit, self.v_res + 1)

        states = []
        for i in range(self.tilings):
            xd = np.digitize(x, x_initial)
            vd = np.digitize(v, v_initial)

            index = xd + (vd - 1) * self.v_res
            states.append((index, i))

            x_initial -= self.x_offset
            v_initial -= self.v_offset

        return states

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):

        sample = np.random.uniform()
        if sample > epsilon:
            if len(weights.shape) == 2:
                best_action = np.argmax(weights[state])
                return best_action
            else:
                action_values = np.zeros(self.env.action_space.n)
                for element in state:
                    for i in range(self.env.action_space.n):
                        action_values[i] += weights[element[0], element[1], i]

                return np.argmax(action_values)
        else:
            best_action = np.random.choice(list(range(self.env.action_space.n)))
            return best_action

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):

        if len(weights.shape) == 2:
            weights[state, action] = weights[state, action] + learning_rate * (
                    reward + self.discount * (weights[new_state, new_action]) - weights[state, action])
        else:
            q = 0
            q_new = 0
            for element in new_state:
                q_new += weights[element[0], element[1], new_action]
            for element in state:
                q += weights[element[0], element[1], action]
            delta = learning_rate * (reward + self.discount * q_new - q)

            for element in state:
                weights[element[0], element[1], action] += delta

        return weights

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if task == 'T1':
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                self.weights_T1 = weights
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1
        self.save_data(task)
        reward_list = [np.mean(reward_list[i - 100:i]) for i in range(100, len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if task == 'T1':
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if task == 'T1':
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
                    help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task = args['task']
    train = int(args['train'])
    np.random.seed(0)
    agent = sarsaAgent()
    agent.env.seed(0)

    agent.env.action_space.seed(0)
    if train:
        agent.train(task)
    else:
        print(agent.test(task))
