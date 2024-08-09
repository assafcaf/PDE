import numpy as np

from tqdm import tqdm



class QLearninigAgent:
    # initial an agent with certain amount of memory
    def __init__(self, env, learning_rate=0.2, gamma=0.9, epsilon=1, epochs=5000, agent_memory=2):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.total_rewards = 0
        self.immediate_reward = 0
        self.agent_memory = agent_memory
        # self.epsilon_decay = self.epsilon/(epochs*0.35)
        self.epsilon_decay = 0.9995

        self.delta_q = 1

        # set q_table size where q(state,action) and state = (my_state, opponent_state)
        rows = (2**self.agent_memory)**2
        self.q_table = np.zeros((rows, env.action_space.n))
    
    def state_to_row(self, state):
        """
        :param state: tuple of the form (my_hist, op_hist)
        :return: row number as a function of this tuple
        """
        my_hist, op_hist = state[0], state[1]
        total_state = list(my_hist) + list(op_hist)
        total_state = list(reversed(total_state))

        row_number = 0
        for i, value in enumerate(total_state):
            row_number += (2 ** i) * int(total_state[i])
        return row_number

    # get observation and return action
    def predict(self, env, obs, deterministic=False):
        row = self.state_to_row(obs)
        if 0 in self.q_table[row]:
            action = env.action_space.sample()
        else:
            action_value_list = self.q_table[row]
            action = self.eps_greedy(action_value_list, self.epsilon)
        return action

    def eps_greedy(self, q_values, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)
    # store
    def update(self, obs, action, reward, next_obs):
        # print('state', state[0], type(state[0]))
        # sum total rewards:
        self.total_rewards += reward
        self.immediate_reward = reward

        # get q_table old value
        row = self.state_to_row(obs)
        old_value = self.q_table[row, action]


        # update q_table
        next_row = self.state_to_row(next_obs)
        next_max = np.nanmax(self.q_table[next_row])  # check nanmax
        new_value = (1-self.learning_rate) * old_value + self.learning_rate*(reward + self.gamma*next_max)
        self.q_table[row, action] = new_value

        self.delta_q = new_value-old_value  # returns Delta_Q_table
    
    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        