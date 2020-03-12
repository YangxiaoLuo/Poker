import numpy as np
from poker_env.agents.base_agent import Agent


class RandomAgent(Agent):
    '''
    A random agent.

    Attributes:
        agent_type (str): the type name of the agent
        player_id (str): the corresponding player id of the agent in a game
    '''
    def __init__(self):
        '''
        Initialize the random agent
        '''
        super().__init__(agent_type='RandomAgent')

    def step(self, state):
        '''
        Predict the action given the current state in gerenerating training data.

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return np.random.choice(state.legal_actions)

    def eval_step(self, state):
        '''
        Predict the action given the current state for evaluation.
        Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state)


class CallAgent(Agent):
    '''
    A call agent who will always choose call/check action if legal.

    Attributes:
        agent_type (str): the type name of the agent
        player_id (str): the corresponding player id of the agent in a game
    '''
    def __init__(self):
        '''
        Initialize the call agent
        '''
        super().__init__(agent_type='CallAgent')

    def step(self, state):
        '''
        Predict the action given the current state in gerenerating training data.

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the call/check action
        '''
        if 'call' in state.legal_actions:
            return 'call'
        elif 'check' in state.legal_actions:
            return 'check'
        else:
            raise ValueError("CallAgent can't call or check!")

    def eval_step(self, state):
        '''
        Predict the action given the current state for evaluation.
        Since the call agents are not trained. This function is equivalent to step function

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the call action
        '''
        return self.step(state)
