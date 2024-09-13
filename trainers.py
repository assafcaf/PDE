import matplotlib.pyplot as plt
import numpy as np
from utils import smooth, smooth_2d

class TwoAgentsTrainer():
    def __init__(self, agents, env, max_steps=1000, verbose=False):
        self.agents = agents
        self.env = env
        self.max_steps = max_steps
        self.verbose = verbose
        self.metadata = {"cooperation": [],
                         "reward": [],
                         'action': []
                         }
    
    def store_metadata(self, cooperation, reward, action):
        self.metadata['cooperation'].append(float(cooperation[1]))
        self.metadata['reward'].append(reward)
        self.metadata['action'].append(action)
    
    def plot_metadata(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        
        ax[0,0].plot(smooth(self.metadata['cooperation']))
        ax[0,0].set_title("Cooperation")
        
        ax[0, 1].plot(smooth_2d(self.metadata['reward']))
        ax[0, 1].set_title("reward")
        
        ax[1, 0].plot(smooth_2d(np.array(self.metadata['action'])[:, 0, :]))
        ax[1, 0].set_title("actions agent 1")
        ax[1, 0].legend(['cooperate', 'defect'])
        
        ax[1, 1].plot(smooth_2d(np.array(self.metadata['action'])[:, 1, :]))
        ax[1, 1].set_title("actions agent 2")
        plt.show()
    
    def train(self, n_episodes):
            # training loop
        for ep in range(n_episodes):
            obs = self.env.reset()
            done = False
            r = np.zeros(2)
            a = np.zeros((2, 2))
            while not done:
                action1 = self.agents[0].predict(self.env, obs)
                action2 = self.agents[1].predict(self.env, obs)
                next_obs, rewards, done, info = self.env.step((action1, action2))
                
                self.agents[0].update(obs, action1, rewards[0], next_obs)
                self.agents[1].update(obs, action2, rewards[1], next_obs)
                obs = next_obs
                r += rewards
                a[0, action1] += 1
                a[1, action2] += 1
                
            self.agents[0].update_epsilon()
            self.agents[1].update_epsilon()
            self.store_metadata(info['cooperation'], r, a)
            if ep % 100 == 0:
                print(f"ep: {ep:>4d}, eps: {self.agents[0].epsilon:.3f} -> {info}")
                
class SingleAgentTrainer():
    def __init__(self, agent, env, strategy, max_steps=1000, verbose=False):
        self.agent = agent
        self.env = env
        self.max_steps = max_steps
        self.strategy = strategy
        self.verbose = verbose
        self.metadata = {"cooperation": [],
                         "reward": [],
                         'action': []
                         }
    
    def store_metadata(self, cooperation, reward, action):
        self.metadata['cooperation'].append(float(cooperation[1]))
        self.metadata['reward'].append(reward)
        self.metadata['action'].append(action)
    
    def plot_metadata(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        
        ax[0,0].plot(smooth(self.metadata['cooperation']))
        ax[0,0].set_title("Cooperation")
        
        ax[0, 1].plot(smooth(self.metadata['reward']))
        ax[0, 1].set_title("reward")
        
        ax[1, 0].plot(smooth_2d(self.metadata['action']))
        ax[1, 0].set_title("action")
        ax[1, 0].legend(['cooperate', 'defect'])
        plt.show()
    
    def train(self, n_episodes):
        # training loop
        for ep in range(n_episodes):
            obs = self.env.reset()
            done = False
            r = 0
            a = np.zeros(2)
            while not done:
                action = self.agent.predict(self.env, obs)
                next_obs, reward, done, info = self.env.step(action)
                self.agent.update(obs, action, reward, next_obs)
                obs = next_obs
                r += reward
                a[action] += 1
            self.agent.update_epsilon()
            self.store_metadata(info['cooperation'], r, a)
            if ep % 100 == 0:
                print(f"ep: {ep:>4d}, eps: {self.agent.epsilon:.3f} -> {info}")