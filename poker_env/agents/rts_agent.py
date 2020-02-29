import numpy as np
from poker_env.agents.base_agent import Agent


class RTSAgent(Agent):
    '''
    A RealtimeSearch agent.

    Attributes:
        agent_type (str): the type name of the agent
        player_id (str): the corresponing player id of the agent in a game
    '''

    def __init__(self):
        '''
        Initilize the random agent
        '''
        super().__init__(agent_type='RTSAgent')

    def step(self, state):
        '''
        Predict the action given the curent state in gerenerating training data.

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return np.random.choice(state.legal_actions)

    def eval_step(self, state):
        '''
        Predict the action given the curent state for evaluation.
        Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (PokerState): the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state)