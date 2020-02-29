import random
import numpy as np
from poker_env.ToyPoker.card import Card, LookUpStr
from poker_env.ToyPoker.hand import compare_2_hands


def init_24_deck():
    '''
    Initialize a toy deck of 24 cards.
    Returns:
        (list): A list of Card object
    '''
    return [Card(suit, rank) for suit in LookUpStr.SUIT.value for rank in LookUpStr.RANK.value]


def compare_all_hands(hands):
    '''
    Compare all palyer's all seven cards.

    Args:
        hands(list) : seven cards of all players
    Returns:
        [0, ... , i, ... , 0]: player i wins
    '''
    winner = []
    winner_hand = None
    for i, hand in enumerate(hands):
        result = compare_2_hands(winner_hand, hand)
        if result == [0, 1]:
            winner = [i]
            winner_hand = hand
        elif result == [1, 1]:
            winner.append(i)
    result = [1 if (i in winner) else 0 for i in range(len(hands))]
    return result


def set_global_seed(seed):
    '''
    Set global seed for random, numpy, etc.
    '''
    random.seed(seed)
    np.random.seed(seed)


def card2index(data):
    '''
    Translate card-str array to card-index array.

    Args:
        data (numpy.ndarray): array of shape (n_samples, n_cards)

    Returns:
        (numpy.ndarray): array of shape (n_samples, n_cards)
    '''
    if isinstance(data, str):
        return LookUpStr.RANK.index(data[0]) + LookUpStr.SUIT.index(data[1]) * len(LookUpStr.RANK.value)
    elif isinstance(data, np.ndarray):
        n_samples = data.shape[0]
        n_allcards = data.shape[1]
        result_array = np.zeros((n_samples, n_allcards), dtype=np.float)
        for i in range(n_samples):
            for j in range(n_allcards):
                result_array[i][j] = card2index(data[i][j])
        return result_array
