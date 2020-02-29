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
