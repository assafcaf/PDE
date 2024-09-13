from pde_env import SinglePdeEnv, TwoPlayersPdeEnv
from strategies import TitForTat
from tables import prisoner
from agents import QLearningAgent
from trainers import SingleAgentTrainer, TwoAgentsTrainer

window_size = 4
n_episodes = 5000
env = TwoPlayersPdeEnv(window_size=window_size, table=prisoner, history_len=20, max_len=100)
agent1 = QLearningAgent(agent_memory=window_size, env=env, learning_rate=0.00001, epochs=n_episodes)
agent2 = QLearningAgent(agent_memory=window_size, env=env, learning_rate=0.00001, epochs=n_episodes)

trainer = TwoAgentsTrainer(agents=[agent1, agent2],  
                             env=env,
                             max_steps=1000,
                             verbose=True)

trainer.train(n_episodes)
trainer.plot_metadata()

