from pde_env import PdeEnv
from strategies import TitForTat
from tables import prisoner

env = PdeEnv(window_size=2, table=prisoner, history_len=100, op_strategy=TitForTat(), max_len=100)
_ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
    print()
print (info)
