# 德州扑克博弈环境poker_env

poker_env是一个可用于多人无限注德州扑克博弈训练的环境

- 支持任意人数对战
- 支持设定不同初始筹码
- 支持底池、边池划分
- 支持转译状态空间
- 支持任意整数额的加注
- 推荐python版本3.6+

## 如何使用poker_env

- 无限注德扑环境：NoLimitTexasHoldemEnv

```python
from poker_env.env import NoLimitTexasHoldemEnv
from poker_env.agents.random_agent import RandomAgent
from poker_env.agents.human_agent import TexasHumanAgent as HumanAgent


# 创建environment对象
env = NoLimitTexasHoldemEnv(num_players=6)
# 设置agent(以随机agent为例)
env.set_agents([HumanAgent()] + [RandomAgent() for _ in range(5)])
episode_num = 1
for episode in range(episode_num):
    # 获取一局游戏中的轨迹和结果
    trajectories, payoffs = env.run(is_training=False)
    print(payoffs)
    # 获取整局游戏的历史（行动序列）
    for node in env.get_game_tree():
        print(node)
```

- 简单扑克环境：ToyPokerEnv

```python
from poker_env.ToyPoker.env import ToyPokerEnv
from poker_env.agents.human_agent import ToyPokerHumanAgent as HumanAgent
env = ToyPokerEnv()
env.set_agents([HumanAgent() for _ in range(2)])
_, payoffs = env.run(is_training=False)
print(payoffs)
```

- 训练DeepCFR agent

```python
import torch
from poker_env.agents.random_agent import RandomAgent
from poker_env.agents.deep_cfr.deepcfr_agent import ToyPokerDeepCFRAgent
from poker_env.agents.deep_cfr.DeepCFRModel import DeepCFRModel
from poker_env.agents.deep_cfr.memory import memory
from poker_env.ToyPoker.env import ToyPokerEnv


env = ToyPokerEnv(allow_step_back=True)
agent = ToyPokerDeepCFRAgent(env, DeepCFRModel(ncardtypes = 2, nbets = 8, nactions = 3, dim = 8))
# create memory
memory_p1, memory_p2, memory_strategy = memory(), memory(), memory()
#create DeepCFRModel and loss function, optimizer.
player_1 = DeepCFRModel(ncardtypes = 2, nbets = 8, nactions = 3, dim = 8)
player_2 = DeepCFRModel(ncardtypes = 2, nbets = 8, nactions = 3, dim = 8)
strategy = DeepCFRModel(ncardtypes = 2, nbets = 8, nactions = 3, dim = 8)
optimizer_1 = torch.optim.Adam(player_1.parameters(), lr = 0.001)
optimizer_2 = torch.optim.Adam(player_2.parameters(), lr = 0.001)
optimizer_p = torch.optim.Adam(strategy.parameters(), lr = 0.001)
# give path that used to save and load model in each cfr iteration time.
path_1 = '/Users/xsw/Desktop/DeepAgent/player_1.pth'
path_2 = '/Users/xsw/Desktop/DeepAgent/player_2.pth'
path_p = '/Users/xsw/Desktop/DeepAgent/strategy.pth'
state1 = {'net':player_1.state_dict(), 'optimizer':optimizer_1.state_dict()}
state2 = {'net':player_2.state_dict(), 'optimizer':optimizer_2.state_dict()}
statep = {'net':strategy.state_dict(), 'optimizer':optimizer_p.state_dict()}
torch.save(state1, path_1)
torch.save(state2, path_2)
torch.save(statep, path_p)
load_path = [path_1, path_2, path_p]
# if save_model == True, then define path
save_path =  '/Users/xsw/Desktop/DeepAgent/test1.pth'

#training
agent.train(10, 10, [player_1, player_2], [memory_p1, memory_p2], 
            memory_strategy, load_path, 4000)
```