from poker_env.agents.random_agent import RandomAgent
#from poker_env.agents.mccfr_agent import ToyPokerMCCFRAgent as MCCFRAgent
from poker_env.agents.mccfr_agent2 import MCCFRAgent
from poker_env.ToyPoker.env import ToyPokerEnv
from poker_env.ToyPoker.utils import set_global_seed


set_global_seed(0)

# Make environment
env = ToyPokerEnv(num_players=2, allow_step_back=True)
eval_env = ToyPokerEnv(num_players=2)

# Set the iteration numbers
episode_num = 10000
update_interval = 50
discount_interval = 100
# Set the evaluation numbers
evaluate_every = 10
evaluate_num = 1000

# Initialize MCCFR agent
agent = MCCFRAgent(env=env, update_interval=update_interval, discount_interval=discount_interval)

# Evaluate MCCFR against Random agent
eval_env.set_agents([agent, RandomAgent()])

for episode in range(episode_num):
    agent.train()
    print('\rIteration {}'.format(episode), end='')
    if episode % evaluate_every == 0:
        reward = 0
        for eval_episode in range(evaluate_num):
            _, payoffs = eval_env.run(is_training=False)
            reward += payoffs[0]
        print('\n########## Evaluation ##########')
        print('Iteration: {} Average reward is {}'.format(episode, float(reward) / evaluate_num))

# agent.save_agent(20200225)
