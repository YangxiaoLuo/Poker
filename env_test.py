from poker_env.agents.random_agent import RandomAgent
from poker_env.agents.lbr_agent import ToyPokerLBRAgent
# from poker_env.agents.mccfr_agent import ToyPokerMCCFRAgent
from poker_env.agents.mccfr_agent_tree import ToyPokerMCCFRAgent
from poker_env.ToyPoker.utils import set_global_seed
from poker_env.ToyPoker.env import ToyPokerEnv
import pandas as pd
import time


if __name__ == '__main__':
    set_global_seed(0)

    # Make environment
    env = ToyPokerEnv(num_players=2, allow_step_back=True)
    eval_env = ToyPokerEnv(allow_step_back=False)
    # Set the evaluation numbers
    multi = 1
    episode_num = 100000
    update_interval = 50
    discount_interval = 100
    # Set the evaluation numbers
    evaluate_every = 100
    evaluate_num = 1000

    agent = ToyPokerMCCFRAgent(env=env,
                               update_interval=update_interval,
                               discount_interval=discount_interval)
    lbr_agent = ToyPokerLBRAgent(mode='simple')

    eval_env.set_agents([agent, RandomAgent()])
    # eval_env.set_agents([agent, lbr_agent])

    rewardlist = []

    for episode in range(episode_num):
        agent.train()
        print('\rIteration {}'.format(episode), end='')
        if episode % evaluate_every == 0:
            start = time.process_time()
            print('\n########## Evaluation ##########')
            reward = 0
            for eval_episode in range(evaluate_num):
                _, payoffs = eval_env.run(is_training=False)
                reward += payoffs[0]
            end = time.process_time()
            print('time', end - start)
            print('Iteration: {} Average reward is {}'.format(evaluate_num, float(reward) / evaluate_num))
            rewardlist.append(float(reward) / evaluate_num)
    data = pd.DataFrame({"index": range(int(episode_num / evaluate_every)),
                         "average_reward": rewardlist})
    data.to_csv("mccfr_vs_LBR.csv")
