import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

from tables import prisoner
from collections import deque

from typing import Any, SupportsFloat

class PdeEnv(gym.Env):
    def __init__(self, window_size=2, table=prisoner, history_len=100, op_strategy=None, max_len=100) -> None:
        super().__init__()
        self.env_name = 'PDE'
        self.window_size = window_size
        self.table = table
        self.table = np.resize(self.table, (2, 2, 2))
        self.history_len = history_len
        
        self.history_deque = deque(maxlen=history_len)
        self.total_history = []
        
        self.op_strategy = op_strategy
        self.t = 0
        self.max_len = max_len
        
    @property
    def action_space(self):
        # action_space can be rather 0 - Defect or 1 - Cooperate
        return Discrete(2)

    @property
    def observation_space(self):
        # observation_space can be 0/1, shape - memory_sizeX1, type - int
        return Box(low=0, high=1, shape=(self.window_size, 2), dtype=int)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        obs = self.observation_space.sample()
        
        self.total_history = [*obs]
        self.history_deque = deque(self.total_history, maxlen=self.history_len)
        self.t = 0
        
        return obs
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.t += 1
        
        # apply opponent strategy
        agent_last_move = self.history_deque[-1][1]
        op = self.op_strategy.act(agent_last_move)
        
        # apply action 
        transition  = np.array([op, action])
        r = self.table[tuple(transition)][1]
        
           # save game history
        self.total_history.append(transition)
        self.history_deque.append(transition)
        
        obs = np.array(self.total_history[-self.window_size:])
        done = True if self.t >= self.max_len else False
        info = {}
        if done:
            info = {
                'cooperation': self.compute_cooperation()
            }
        return obs, r, done, info

    def compute_cooperation(self):
        c1 =  np.where(self.history_deque == np.array([0, 0]))[1].sum()/self.history_len
        c2 = np.where(self.total_history[self.window_size:] == np.array([0, 0]))[1].sum()/self.max_len
        return c1, c2
        