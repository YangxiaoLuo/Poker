import numpy as np
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.action import Action as ToyPokerAction


class ToyPokerLBRAgent(Agent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, mode='simple'):
        '''
        Initialize the agent

        Args:
            mode(str): mode of LBR:
                'simple' means simply set LBR win probability and opponent's fold probability to be 0.5
        '''
        super().__init__(agent_type='LBRAgent')
        self.mode = mode
        self.action_space = []
        for action in ToyPokerAction:
            self.action_space.append(action.value)

    @property
    def action_num(self):
        return len(self.action_space)

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

    def eval_step(self, state):
        '''
        Given a state, predict the best action according to LBR formula

        Args:
            state (PokerState): the current state

        Returns:
            action (str): local best response action
        '''
        max_action_value, action_index = self.get_value(state)
        if max_action_value <= 0:
            action = ToyPokerAction.FOLD.value
        else:
            action = self.decode_action(action_index)
        return action

    def get_value(self, state):
        '''
        Go through all the legal actions of LBR agent at any decision point to
        get the action_utilities

        Args:
            state (PokerState): the current state

        Returns:
            (float): maximal action_utilities
            (int): the action with greatest value
        '''
        wp = self.WpRollout()
        fp = self.get_fp()
        pot_lbr = self.get_pot(state)
        asked = sum(state.pot) - 2 * pot_lbr
        pot = asked + 2 * pot_lbr
        # Go through all the leagal actions of current state
        legal_actions = self.encode_action(state)
        action_utilities = np.zeros(self.action_num)
        for a in legal_actions:
            str_action = self.decode_action(a)
            if (str_action == ToyPokerAction.CALL.value) or (str_action == ToyPokerAction.CHECK.value):
                action_utilities[a] = wp * pot - (1 - wp) * asked
            elif str_action == ToyPokerAction.FOLD.value:
                continue
            else:
                raise_size = state.raise_money
                action_utilities[a] = fp * pot + (1 - fp) * (wp * (pot + raise_size) - (1 - wp) * (asked + raise_size))
        max_value_action = np.argmax(action_utilities)
        return action_utilities[max_value_action], max_value_action

    def WpRollout(self):
        '''
        Get LBR wining probability
        '''
        if self.mode == 'simple':
            return 0.5
        # TODO: use MC to get opponent's cards to get the win Rate

    def get_fp(self):
        '''
        Get opponents' fold probability
        '''
        if self.mode == 'simple':
            return 0.5
        # TODO: use MC to estimate opponent cards and according to 
        # opponent's strategy to get opponent's fold probability

    def get_pot(self, state):
        '''
        Get current pot size of LBR agent

        Args:
            state (PokerState): the current state
        '''
        return state.pot[self.player_id]

    # Deprecated: 当前版本不需要
    # def Deal_cards(self, br_id):
    #     '''
    #     Get all possible opponent cards
    #     Return:
    #         all possible opponent cards and rest cards
    #     '''
    #
    #     # get rest cards in pot
    #     rest_pot = self.env.get_dealer().get_deck()
    #     # give back the information that br unknown
    #     br_cards = deepcopy(self.env.get_hand_cards(br_id))
    #     rest_pot += br_cards
    #     self.env.get_dealer().set_deck(rest_pot)
    #
    #     # get all combinations
    #     hands = []
    #     for com in combinations(deepcopy(rest_pot), 2):
    #         hands.append(list(com))
    #
    #     return hands
    #
