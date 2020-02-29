import numpy as np
import copy
import math

def sample_action(actionspace, distribution):
    return np.random.choice(actionspace, size=1, p=distribution)[0]

def card_abstraction(state):
    # print(state.get_infoset())
    # trivial card abstraction
    return get_visible_card(state)

def action_abstraction(state):
    # trivial action abstraction
    return get_actionspace(state)

def get_player(state):
    player = state.player_id
    return player

def get_visible_card(state):
    all_cards = state.hand_cards + state.public_cards
    # print(all_cards)
    all_cards.sort()
    return all_cards

def get_actionspace(state):
    actionspace = state.legal_actions
    return actionspace
