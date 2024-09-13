from pde_env import PdeEnv
from strategies import TitForTat
from tables import prisoner
from agents import QLearningAgent

window_size = 2
n_episodes = 1000
env = PdeEnv(window_size=window_size, table=prisoner, history_len=100, op_strategy=TitForTat(), max_len=100)


agent = QLearningAgent(agent_memory=window_size, env=env, learning_rate=0.001, epochs=n_episodes)

# traing loop
for ep in range(n_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = agent.predict(env, obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs)
        obs = next_obs
    agent.update_epsilon()
    print(f"ep: {ep:>4d}, eps: {agent.epsilon:.3f} -> {info}")
