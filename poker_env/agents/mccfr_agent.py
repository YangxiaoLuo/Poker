import numpy as np
import pandas as pd
import collections
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.env import ToyPokerEnv as Env
from poker_env.ToyPoker.data.eval_potential import calc_final_potential


class ToyPokerMCCFRAgent(Agent):
    '''
    A Implementation of External-Sampling MCCFR with Negative-Regret Pruning

    Attributes:
        iterations (int): the number of iterations in training
    '''
    def __init__(self, env, update_interval=100, discount_interval=1000):
        '''
        Initialize the random agent

        Args:
            env (Env): Env instance for training agent
        '''
        super().__init__(agent_type='MCCFRAgent')
        if isinstance(env, Env):
            self.env = env
        else:
            raise TypeError("Env must be a instance of NoLimitTexasHoldemEnv!")
        self.policy = collections.defaultdict(np.array)
        self.regrets = collections.defaultdict(np.array)
        self.action_space = env.get_action_space()
        self.iterations = 0
        self.update_interval = update_interval
        self.discount_interval = discount_interval

        # state abstraction result
        self.first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_ehs_vector.csv', index_col=None)
        self.final_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', index_col=None)

    @property
    def action_num(self):
        return len(self.action_space)

    def train(self):
        '''
        Conduct External-Sampling Monte Carlo CFR with Pruning.
        '''
        self.iterations += 1
        if self.iterations % self.update_interval == 0:
            self.update_policy()
        for player_id in range(self.env.player_num):
            self.env.init_game()
            self.traverse_mccfr(player_id)
        # Discount
        if self.iterations % self.discount_interval == 0:
            discount = self.iterations / (self.iterations + self.discount_interval)
            for info_set in self.regrets.keys():
                self.regrets[info_set] *= discount
        # TODO: Threshold and Pruning

    def calculate_strategy(self, info_set, legal_actions):
        '''
        Calculates the strategy based on regrets. Set zero if this infoset hasn't been initialized in memory,

        Args:
            info_set (str): key in policy dictionary which represent the information of state
            legal_actions (list): indices of legal actions

        Returns:
            (np.ndarray): the action probabilities
        '''
        sum_regret = 0
        # calculate sum positive-regrets of current infoset
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(self.action_num)
        else:
            regrets = self.regrets[info_set]
            for action in legal_actions:
                sum_regret += max(regrets[action], 0)

        # calculate strategy based on regrets
        action_probs = np.zeros(self.action_num)
        for action in legal_actions:
            if sum_regret > 0:
                action_probs[action] = max(regrets[action], 0) / sum_regret
            else:
                action_probs[action] = 1 / len(legal_actions)

        return action_probs

    def update_strategy(self):
        '''
        Update the average strategy of specific infoset for current player.

        Args:
            info_set (str): key in policy dictionary which represent the information of state
        '''
        for info_set, regret in self.regrets.items():
            positive_regret_sum = np.sum(np.maximum(regret, 0))
            if positive_regret_sum > 0:
                action_probs = np.maximum(regret, 0) / positive_regret_sum
            else:
                action_probs = np.ones(regret.shape) / len(regret)
            self.policy[info_set] = action_probs

    def traverse_mccfr(self, player_id):
        '''
        Traverse the game tree, update the regrets.

        Args:
            player_id (int): The traverser id of this traverse

        Returns:
            state_utilities (float): the expected value/payoff of current player
        '''
        # (1) For terminal node, return the game payoff.
        if self.env.is_over():
            return self.env.get_payoffs()[player_id]
        current_player = self.env.get_player_id()
        state = self.env.get_state()
        info_set = self.encode_state(state)
        legal_actions = self.encode_action(state)
        # (2) If current player is the traverser, traverse each action.
        if current_player == player_id:
            # determine the strategy at this infoset
            action_probs = self.calculate_strategy(info_set, legal_actions)
            # calculate the value_expectation of current state/history
            value_expectation = 0  # v_h
            # traverse each action
            action_utilities = np.zeros(self.action_num)
            for action in legal_actions:
                self.env.step(self.decode_action(action))
                action_utilities[action] = self.traverse_mccfr(player_id)
                value_expectation += action_probs[action] * action_utilities[action]
                self.env.step_back()
            # update the regret of each action
            for action in legal_actions:
                self.regrets[info_set][action] += action_utilities[action] - value_expectation
            return value_expectation
        # (3) For the opponent node, sample an action from the probability distribution.
        else:
            action_probs = self.calculate_strategy(info_set, legal_actions)
            action = np.random.choice(self.action_num, p=action_probs)
            self.env.step(self.decode_action(action))
            value_expectation = self.traverse_mccfr(player_id)
            self.env.step_back()
            return value_expectation

    def update_policy(self):
        '''
        Update policy based on the current regrets
        '''
        for info_set, regret in self.regrets.items():
            positive_regret_sum = np.sum(np.maximum(regret, 0))
            if positive_regret_sum > 0:
                action_probs = np.maximum(regret, 0) / positive_regret_sum
            else:
                action_probs = np.ones(regret.shape) / len(regret)
            self.policy[info_set] = action_probs

    def get_action_probs(self, state):
        '''
        Get the action probabilities of the current state.

        Args:
            state (PokerState): the state of the game

        Returns:
            (numpy.ndarray): the action probabilities
        '''
        action_length = self.action_num
        info_set = self.encode_state(state)
        legal_actions = self.encode_action(state)
        if info_set not in self.policy:
            action_probs = np.array([1.0 / action_length for _ in range(action_length)])
            self.policy[info_set] = action_probs
        else:
            action_probs = self.policy[info_set]
        # Remove illegal actions
        legal_probs = np.zeros(action_length)
        legal_probs[legal_actions] = action_probs[legal_actions]
        # Normalization
        if np.sum(legal_probs) == 0:
            legal_probs[legal_actions] = 1 / len(legal_actions)
        else:
            legal_probs /= np.sum(legal_probs)
        return legal_probs

    def encode_state(self, state):
        '''
        Get infoset of state

        Args:
            state (PokerState): the state of the game

        Returns:
            (string): infoset keys.
        '''
        lossless_state = state.get_infoset()
        if len(state.public_cards) == 3:
            cluster_label = self.first_round_table[self.first_round_table['cards_str'] == lossless_state]['label'].values[0]
            lossy_state = 'first_{}'.format(cluster_label)
        else:
            # Calculate directly(√) / Read from table
            # cluster_label = self.final_round_table[self.final_round_table['cards_str'] == lossless_state]['label'].values[0]
            cluster_label = int(calc_final_potential(state.hand_cards, state.public_cards) * 50)
            lossy_state = 'final_{}'.format(cluster_label)
        print(lossy_state)
        return lossy_state

    def encode_action(self, state):
        '''
        Get legal actions.

        Args:
            state (PokerState): the state of the game

        Returns:
            (list): Indices of legal actions(Abstract or not).
        '''
        encode_actions = []
        for action in state.legal_actions:
            encode_actions.append(self.action_space.index(action))
        return encode_actions

    def decode_action(self, action):
        '''
        Decode the action according to current state

        Args:
            action (int): index of action in action space

        Returns:
            (str): legal action string for the state
        '''
        return self.action_space[action]

    def step(self, state):
        '''
        Predict the action given the current state.

        Args:
            state (PokerState): the current state

        Returns:
            action (str): random action based on the action probability vector
        '''
        action_probs = self.calculate_strategy(
            info_set=state.get_info_set(),
            legal_actions=self.encode_action(state)
        )
        action = np.random.choice(len(action_probs), p=action_probs)
        return self.decode_action(action)

    def eval_step(self, state):
        '''
        Given a state, predict the best action based on average policy

        Args:
            state (PokerState): the current state

        Returns:
            action (str): best action based on policy
        '''
        action_probs = self.get_action_probs(state)
        action = np.argmax(action_probs)
        return self.decode_action(action)

    def save_agent(self, index):
        '''
        Args:
            index (int): number of th policy/regrets
        '''
        # save policy
        file = open('policy_{}'.format(index), 'w')
        policy_dict = {}
        for key, value in self.policy.items():
            policy_dict[key] = value.tolist()
        file.write(str(policy_dict))
        file.close()
        # save regret
        file = open('regret_{}'.format(index), 'w')
        regret_dict = {}
        for key, value in self.regrets.items():
            regret_dict[key] = value.tolist()
        file.write(str(regret_dict))
        file.close()

    def load_agent(self, index):
        '''
        Args:
            index (int): number of th policy/regrets
        '''
        # load policy
        file = open('policy_{}'.format(index))
        policy_dict = eval(file.read())
        for key, value in policy_dict.items():
            self.policy[key] = np.array(value)
        file.close()
        # load regret
        file = open('regret_{}'.format(index))
        regret_dict = eval(file.read())
        for key, value in regret_dict.items():
            self.regrets[key] = np.array(value)
        file.close()
