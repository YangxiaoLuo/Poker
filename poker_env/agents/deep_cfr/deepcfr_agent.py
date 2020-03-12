import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import collections
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.env import ToyPokerEnv as Env
from poker_env.agents.deep_cfr.memory import memory
from poker_env.agents.deep_cfr.DeepCFRModel import DeepCFRModel
from poker_env.agents.deep_cfr.DeepCFRModel import LossFunc


class ToyPokerDeepCFRAgent(Agent):
    '''
    A Implementation of External-Sampling DeepCFR

    '''

    def __init__(self, env, policy):
        '''
        Initialize the random agent

        Args:
            env (Env): Env instance for training agent
            policy: DeepCFRModel
        '''
        super().__init__(agent_type='DeepCFRAgent')
        if isinstance(env, Env):
            self.env = env
        else:
            raise TypeError("Env must be a instance of NoLimitTexasHoldemEnv!")
        self.action_space = env.get_action_space()
        self.policy = policy

    @property
    def action_num(self):
        return len(self.action_space)

    def train(self, sampling_times, cfr_times, player, memory_v, memory_p, load_path, minibatch,
              save_model=False, save_path=None):
        '''
        Conduct External-Sampling Deep CFR.

        Args:
            sampling_times (int): external samping iteration times.
            cfr_times (int): CFR iteration times.
            player (list): a list of neural network.
            memory_v (list): a list of memory that store transition of each player at traverser node.
            memory_p (memory): memory that store transition at opponent node.
            load_path (str): a list of path for loading network.
            minibatch (int): the amount of sample chosen from memory.
            save_model (bool): choose whether to save DeepCFRModel while training
            save_path (list): a list of path that used to save model while save_model equals to True


        Return:
            (DeepCFRModel) neural network that approximate average strategy
        '''
        for t in range(cfr_times):
            print('CFR Iteration times: {}'.format(t + 1))
            loss_function, optimizer = [LossFunc(), LossFunc()], [0, 0]
            for p in range(2):
                for k in range(sampling_times):
                    self.env.init_game()
                    for i in range(2):
                        check = torch.load(load_path[i])
                        player[i].load_state_dict(check['net'])
                    self.traverse(p, memory_v[p], memory_p, t, player)
                handcard, boardcard, bet_history, time, regret = memory_v[p].get_sample(minibatch)
                # learning advantage
                check = torch.load(load_path[p])
                player[p].load_state_dict(check['net'])
                optimizer[p] = torch.optim.Adam(player[p].parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
                optimizer[p].load_state_dict(check['optimizer'])
                optimizer[p].zero_grad()
                output = player[p]([handcard, boardcard], bet_history)
                loss = loss_function[p](regret.float(), output, time)
                loss.backward()
                nn.utils.clip_grad_norm_(player[p].parameters(), max_norm=1)
                optimizer[p].step()
                state = {'net': player[p].state_dict(), 'optimizer': optimizer[p].state_dict()}
                torch.save(state, load_path[p])
            # learning strategy
            handcard, boardcard, bet_history, time, strategy = memory_p.get_sample(minibatch)
            check = torch.load(load_path[2])
            self.policy.load_state_dict(check['net'])
            optimizer_p = torch.optim.Adam(self.policy.parameters(), lr=0.001)  # 传入 net 的所有参数, 学习率
            optimizer_p.load_state_dict(check['optimizer'])
            optimizer_p.zero_grad()
            output = self.policy([handcard, boardcard], bet_history)
            loss_p = LossFunc()
            loss = loss_p(strategy.float(), output, time)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
            optimizer_p.step()
            state = {'net': self.policy.state_dict(), 'optimizer': optimizer_p.state_dict()}
            torch.save(state, load_path[2])
            # save_model
            if save_model == True:
                if save_path == None:
                    raise TypeError("Save path are not given!")
                else:
                    step = cfr_times // len(save_path)
                if (t + 1) % step == 0:
                    torch.save(self.policy.state_dict(), save_path[t // step])
        return self.policy

    def calculate_strategy(self, policy, handcard, boardcard, bet_history):
        '''
        Calculates the strategy based on the ouput of neural network.

        Args:
            player (DeepCFRModel): neural network that average strategy

        Returns:
            (nparray): the action probability of current state
        '''
        logit = policy([handcard, boardcard], bet_history)
        shift = logit - torch.max(logit)
        exps = torch.exp(shift)
        action_probs = exps / torch.sum(exps)
        return action_probs.detach().numpy()

    def get_bet_history(self, state):
        '''
        get bet history

        Args:
            state (PokerState): the current state

        Returns:
            (torch): the bet history
        '''
        round_1, round_2 = [], []
        game_history = state.history
        for i in range(len(game_history)):
            action = game_history[i][1]
            opponent_bet = game_history[i][2]
            round_counter = game_history[i][3]
            if round_counter == 1:
                if action == 'raise':
                    round_1.append(opponent_bet + 2)
                elif action == 'call' or action == 'check':
                    round_1.append(opponent_bet)
            elif round_counter == 2:
                if action == 'raise':
                    round_2.append(opponent_bet + 2)
                elif action == 'call' or action == 'check':
                    round_2.append(opponent_bet)
        bet_history = torch.zeros(1, 8, dtype=torch.float)
        for i in range(len(round_1)):
            bet_history[0][i] = round_1[i]
        for i in range(len(round_2)):
            bet_history[0][4 + i] = round_2[i]
        return bet_history

    def get_card(self, card):
        '''
        translate str into list

        Args:
            card (str): card

        Returns:
            (list): a list that stored number, each number represent a card
        '''
        new_card = []
        for i in range(len(card)):
            rank = (int(card[i][0]) - 2) * 4
            suit = card[i][1]
            if suit == 's':
                suit = 0
            elif suit == 'c':
                suit = 1
            elif suit == 'd':
                suit = 2
            else:
                suit = 3
            new_card.append(rank + suit)
        return new_card

    def regret_matching(self, regret):
        '''
        calculate new action probability by regret matching

        Args:
            regret (ndarray): regret value

        Returns:
            (ndarray): action probability
        '''
        temp = 0
        for i in regret:
            if i > 0:
                temp += i
        if temp == 0:
            regret[np.where(regret == np.max(regret))] = 1
            regret[np.where(regret != 1)] = 0
        else:
            regret = np.maximum(regret, 0)
        return regret / np.sum(regret)

    def traverse(self, traverser, memory_v, memory_p, t, agent):
        '''
        Traverse the game tree, store the transitions.

        Args:
            traverser (int）: The player who traverse the game tree
            memory_v (list): a list that store traverser's transitions
            memory_p (memory: a memory store average strategy's transitions
            t (int_: traverse times
            agent (DeepCFRModel): neural network that approximate traverser's regret at each node

        Returns:
            state_utilities (float): the expected value/payoff of current player
        '''
        # (1) For terminal node, return the game payoff.
        if self.env.is_over():
            return self.env.get_payoffs()[traverser]
        current_player = self.env.get_player_id()
        state = self.env.get_state()
        legal_actions = self.encode_action(state)
        hole = self.get_card(state.hand_cards)
        board = self.get_card(state.public_cards)
        holecard = torch.zeros(1, 24, dtype=torch.long)
        boardcard = torch.zeros(1, 24, dtype=torch.long)
        bet_history = self.get_bet_history(state)
        for i in hole:
            holecard[0][i] = 1
        for i in board:
            boardcard[0][i] = 1
        # (2) If current player is the traverser, traverse each action.
        if current_player == traverser:
            # determine the strategy at this infoset by regret matching
            regret = agent[traverser]([holecard, boardcard], bet_history).detach().numpy()[0]
            action_probs = self.regret_matching(regret)
            # calculate the value_expectation of current state/history
            value_expectation = 0  # v_h
            # traverse each action
            action_utilities = np.zeros(3)
            # fold call raise
            actions = [1, 3, 4]
            for action in actions:
                self.env.step(self.decode_action(action, legal_actions))
                action_utilities[actions.index(action)] = self.traverse(traverser, memory_v, memory_p, t, agent)
                value_expectation += action_probs[actions.index(action)] * action_utilities[actions.index(action)]
                self.env.step_back()
            # update the regret of each action
            regrets = torch.zeros(1, 3, dtype=float)
            for action in actions:
                regrets[0][actions.index(action)] += action_utilities[actions.index(action)] - value_expectation
            # store transition
            transition = [holecard, boardcard, bet_history, t + 1, regrets]
            memory_v.update(transition)
            return value_expectation
        # (3) For the opponent node, sample an action from the probability distribution and store transition.
        else:
            # determine the strategy at this infoset by regret matching
            regret = agent[1 - traverser]([holecard, boardcard], bet_history).detach().numpy()[0]
            action_probs = self.regret_matching(regret).reshape(1, 3)
            transition = [holecard, boardcard, bet_history, t + 1, torch.from_numpy(action_probs)]
            memory_p.update(transition)
            action = np.random.choice(3, p=action_probs[0])
            self.env.step(self.decode_action(action, legal_actions))
            value_expectation = self.traverse(traverser, memory_v, memory_p, t, agent)
            self.env.step_back()
            return value_expectation

    def step(self, state):
        '''
        Given a state, predict the best action based on average policy

        Args:
            state (PokerState): the current state

        Returns:
            action (str): random action based on policy
        '''
        legal_actions = self.encode_action(state)
        bet_history = self.get_bet_history(state)
        hole = self.get_card(state.hand_cards)
        board = self.get_card(state.public_cards)
        holecard = torch.zeros(1, 24, dtype=torch.long)
        boardcard = torch.zeros(1, 24, dtype=torch.long)
        for i in hole:
            holecard[0][i] = 1
        for i in board:
            boardcard[0][i] = 1
        action_probs = self.calculate_strategy(
            self.policy, holecard, boardcard, bet_history
        )
        action = np.random.choice(3, p=action_probs[0])
        return self.decode_action(action, legal_actions)

    def eval_step(self, state):
        '''
        Given a state, predict the best action based on average policy

        Args:
            state (PokerState): the current state

        Returns:
            action (str): best action based on policy
        '''
        legal_actions = self.encode_action(state)
        bet_history = self.get_bet_history(state)
        hole = self.get_card(state.hand_cards)
        board = self.get_card(state.public_cards)
        holecard = torch.zeros(1, 24, dtype=torch.long)
        boardcard = torch.zeros(1, 24, dtype=torch.long)
        for i in hole:
            holecard[0][i] = 1
        for i in board:
            boardcard[0][i] = 1
        action_probs = self.calculate_strategy(
            self.policy, holecard, boardcard, bet_history
        )
        action = np.argmax(action_probs[0])
        return self.decode_action(action, legal_actions)

    def encode_action(self, state):
        '''
        Get legal actions.

        Args:
            state (PokerState): the state of the game

        Returns:
            (list): Indices of legal actions.
        '''
        encode_actions = []
        for action in state.legal_actions:
            encode_actions.append(self.action_space.index(action))
        return encode_actions

    def decode_action(self, action, legal_actions):
        '''
        Decode the action according to current state
        treat call and check as same action
        if not allow to raise, then call

        Args:
            action (int): index of action in action space

        Returns:
            (str): legal action string for the state
        '''
        new_action = action
        if action not in legal_actions:
            if action == 1:
                new_action = 2
            elif action == 2:
                new_action = 1
            else:
                new_action = 3
        return self.action_space[new_action]

    def save_agent(self, path, network):
        '''

        Args:
            path (str): the path for saving neural network
        '''
        # save nerual network
        torch.save(network.state_dict(), path)
        return

    def load_agent(self, path):
        '''

        Args:
            path (str): the path for loading neural network
        '''
        # load policy
        check = torch.load(path)
        self.policy.load_state_dict(check)