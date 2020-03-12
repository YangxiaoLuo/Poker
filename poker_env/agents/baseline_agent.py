import numpy as np
import pandas as pd
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.data.eval_potential import calc_final_potential
from poker_env.ToyPoker.action import Action


class ToypokerTPAgent(Agent):
    '''
    A TightPassive agent in Toypoker game, which moves based on EHS.

    Attributes:
        agent_type (str): the type name of the agent
        player_id (str): the corresponding player id of the agent in a game
    '''
    def __init__(self, threshold_fold=0.3, threshold_call=0.7):
        '''
        Initialize the TightPassive agent

        Args:
            threshold_fold (float): a value between 0 ~ 1 to decide whether to fold.
            threshold_call (float): a value between 0 ~ 1 to decide whether to call, must larger than threshold_fold.
        '''
        super().__init__(agent_type='TPAgent')
        if threshold_call <= threshold_fold:
            raise ValueError("threshold_call must larger than threshold_fold!")
        self.threshold_fold = threshold_fold
        self.threshold_call = threshold_call
        self.first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_EHS_vector.csv', index_col=None)

    def step(self, state):
        '''
        Predict the action given the current state in gerenerating training data.

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (choose according to EHS) by the TightPassive agent
        '''
        lossless_state = state.get_infoset()
        # Search in precomputed table for the first round
        if len(state.public_cards) == 3:
            found_row = self.first_round_table[self.first_round_table['cards_str'] == lossless_state]
            ehs_value = np.mean(found_row.iloc[:, 2:].values)
        # Calculate ehs for the final round
        elif len(state.public_cards) == 5:
            ehs_value = calc_final_potential(state.hand_cards, state.public_cards)

        # take action from action space according to EHS
        # Choose raise if ehs is high
        if ehs_value > self.threshold_call and Action.RAISE.value in state.legal_actions:
            return Action.RAISE.value
        # Choose call/check if ehs is medium
        elif ehs_value > self.threshold_fold:
            if Action.CALL.value in state.legal_actions:
                return Action.CALL.value
            elif Action.CHECK.value in state.legal_actions:
                return Action.CHECK.value
        # Choose fold if ehs is low
        else:
            return Action.FOLD.value

    def eval_step(self, state):
        '''
        Predict the action given the current state for evaluation.
        Since the TightPassive agents are not trained. This function is equivalent to step function

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state)


class ToypokerLPAgent(Agent):
    '''
    A LoosePassive agent in Toypoker game, which usually call even it is time to fold or raise

    Attributes:
        agent_type (str): the type name of the agent
        player_id (str): the corresponding player id of the agent in a game
    '''
    def __init__(self, call_prob=0.6, judge_value=0.4):
        '''
        Initialize the TightPassive agent

        Args:
            call_prob(float): probability of select action 'call'.
            judge_value (float): a value between 0 ~ 1 to judge it is a good hand or a bad one.
        '''
        super().__init__(agent_type='LPAgent')
        self.call_prob = call_prob
        self.judge_value = judge_value
        self.first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_EHS_vector.csv', index_col=None)

    def step(self, state):
        '''
        Predict the action given the current state in gerenerating training data.

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (choose according to effective hand strength) by the TightPassive agent
        ''' 
        lossless_state = state.get_infoset()
        # Search in precomputed table for the first round
        if len(state.public_cards) == 3:
            found_row = self.first_round_table[self.first_round_table['cards_str'] == lossless_state]
            ehs_value = np.mean(found_row.iloc[:, 2:].values)
        # Calculate ehs for the final round
        elif len(state.public_cards) == 5:
            ehs_value = calc_final_potential(state.hand_cards, state.public_cards)

        # select call with the probability of call_prob
        if np.random.random() < self.call_prob:
            if Action.CALL.value in state.legal_actions:
                return Action.CALL.value
            elif Action.CHECK.value in state.legal_actions:
                return Action.CHECK.value
            else:
                return Action.FOLD.value
        else:
            if ehs_value > self.judge_value and Action.RAISE.value in state.legal_actions:
                return Action.RAISE.value
            else:
                return np.random.choice(state.legal_actions)

    def eval_step(self, state):
        '''
        Predict the action given the current state for evaluation.
        Since the TightPassive agents are not trained. This function is equivalent to step function

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state)


class ToypokerLAAgent(Agent):
    '''
    A LooseAggressive agent in Toypoker game, which often raise to put pressure on opponent.

    Attributes:
        agent_type (str): the type name of the agent
        player_id (str): the corresponding player id of the agent in a game
    '''
    def __init__(self, raise_prob=0.8):
        '''
        Initialize the TightPassive agent

        Args:
            raise_prob(float): probability of select action 'raise'.
        '''
        super().__init__(agent_type='LAAgent')
        self.raise_prob = raise_prob

    def step(self, state):
        '''
        Predict the action given the current state in gerenerating training data.

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (choose according to effective hand strength) by the TightPassive agent
        '''
        if np.random.random() < self.raise_prob and Action.RAISE.value in state.legal_actions:
            return Action.RAISE.value
        else:
            if Action.CALL.value in state.legal_actions:
                return Action.CALL.value
            elif Action.CHECK.value in state.legal_actions:
                return Action.CHECK.value
            else:
                return Action.FOLD.value

    def eval_step(self, state):
        '''
        Predict the action given the current state for evaluation.
        Since the TightPassive agents are not trained. This function is equivalent to step function

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state)
