# Implement of Potential evaluation
# For River Round:
#   Assume random distribution of opponent's hand, return the equity.(Expected Hand Strength, EHS)
# For Turn Round:
#   Assume random distribution of the river card and return the vector of EHS at river round.
# For Flop Round:
#   Assume random distribution of the turn card and return 2 vectors:
#       (vector1): vector for mean of EHS vector at turn round.(potential at turn round)
#       (vector2): vector for EHS value at river round.(potential at river round)

import os
import sys
import random
import numpy as np
import pandas as pd
from itertools import combinations
from multiprocessing import Pool
from poker_env.Texas.hand import compare_2_hands
from poker_env.Texas.card import LookUpStr
from poker_env.Texas.state import PokerState


def init_deck():
    '''
    Initialize a deck.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    deck = [rank + suit for suit in suit_list for rank in rank_list]
    return deck


def calc_river_potential(hand_cards, public_cards):
    '''
    Calculate Expected Hand Strength at River Round(Final Round).
    Args:
        hand_cards (list): cards that the player holds in hand.
        public_cards (list): cards that is accessable to all players.
    Returns:
        (float): value of EHS
    '''
    deck = init_deck()
    for card in hand_cards + public_cards:
        deck.remove(card)
    total_opponent_hands = 0
    win_num = 0
    for opp_hands in combinations(deck, 2):
        total_opponent_hands += 1
        result = compare_2_hands(hand0=hand_cards + public_cards,
                                 hand1=list(opp_hands) + public_cards)
        if result[0] == result[1]:
            win_num += 0.5
        else:
            win_num += result[0]
    return win_num / total_opponent_hands


def calc_turn_potential(hand_cards, public_cards):
    '''
    Calculate the distribution of EHS at Turn Round(Third Round)
    Args:
        hand_cards (list): cards that the player holds in hand.
        public_cards (list): cards that is accessable to all players.
    Returns:
        (numpy.ndarray): vector of EHS distribution
    '''
    deck = init_deck()
    for card in hand_cards + public_cards:
        deck.remove(card)
    vector_EHS = []
    for river_card in combinations(deck, 1):
        public_5_cards = public_cards + list(river_card)
        vector_EHS.append(calc_river_potential(hand_cards, public_5_cards))
    return np.array(sorted(vector_EHS))


def calc_flop_potential(hand_cards, public_cards):
    '''
    Calculate the distribution of EHS at Flop Round(Second Round)
    Args:
        hand_cards (list): cards that the player holds in hand.
        public_cards (list): cards that is accessable to all players.
    Returns:
        (tuple):
            (numpy.ndarray): vector of EHS distribution in turn round
            (numpy.ndarray): vector of EHS distribution in river round
    '''
    deck = init_deck()
    for card in hand_cards + public_cards:
        deck.remove(card)
    vector_EHS_at_turn = []
    vector_EHS_at_river = []
    for i in range(len(deck)):
        public_4_cards = public_cards + [deck[i]]
        vector_EHS_at_turn.append(np.mean(calc_turn_potential(hand_cards, public_4_cards)))
        for j in range(len(deck)):
            if j != i:
                public_5_cards = public_4_cards + [deck[j]]
                vector_EHS_at_river.append(calc_river_potential(hand_cards, public_5_cards))
    return np.array(vector_EHS_at_turn), np.array(vector_EHS_at_river)


def init_all_flop_label(filename):
    '''
    Create a table with all possible flop round cards.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    possible_hands = []
    for i in range(len(rank_list)):
        possible_hands.append([rank_list[i] + suit_list[0], rank_list[i] + suit_list[1]])
        for j in range(i + 1, len(rank_list)):
            possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[1]])
            possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[0]])

    columns = ('cards_str', 'label')
    df = pd.DataFrame(columns=columns)
    df.to_csv(filename, sep=',', index=False, mode='w')
    total_num = 0
    for hand_cards in possible_hands:
        deck = init_deck()
        for card in hand_cards:
            deck.remove(card)
        for public_cards in combinations(deck, 3):
            total_num += 1
            cards = PokerState.get_suit_normalization(hand_cards, public_cards)
            data_dict = {}
            data_dict[columns[0]] = [''.join(cards)]
            data_dict[columns[1]] = [-1]
            df = df.append(pd.DataFrame(data_dict), ignore_index=True, sort=False)
            # Quick Save
            if df.shape[0] == 10000:
                df.to_csv(filename, sep=',', index=False, mode='a', header=False)
                df = pd.DataFrame(columns=columns)
            print("\rFinish:{}".format(total_num), end='')
    # Data deduplicate
    df = pd.read_csv(filename, index_col=None)
    df = df.drop_duplicates([columns[0]])
    df.to_csv(filename, sep=',', index=False, mode='w')


def init_all_turn_label(filename):
    '''
    Create a table with all possible turn round cards.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    # Length of turn round EHS vector is (52 - 6) = 46
    columns = ('cards_str', 'label')
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=None)
    else:
        df = pd.DataFrame(columns=columns)    # num_of_type for possible hands is 13 * 13 = 169
    possible_hands = []
    for i in range(len(rank_list)):
        possible_hands.append([rank_list[i] + suit_list[0], rank_list[i] + suit_list[1]])
        for j in range(i + 1, len(rank_list)):
            possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[1]])
            possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[0]])
    total_num = 0
    for hand_cards in possible_hands:
        deck = init_deck()
        for card in hand_cards:
            deck.remove(card)
        for public_cards in combinations(deck, 4):
            total_num += 1
            # Skip existing data
            if total_num <= df.shape[0]:
                continue
            cards = PokerState.get_suit_normalization(hand_cards, public_cards)
            data_dict = {}
            for k in range(len(columns)):
                data_dict[columns[0]] = [''.join(cards)]
                data_dict[columns[1]] = [-1]
            df = df.append(pd.DataFrame(data_dict), ignore_index=True, sort=False)
            print("\rFinish:{}".format(total_num), end='')
    # Data deduplicate
    df = pd.read_csv(filename, index_col=None)
    df = df.drop_duplicates([columns[0]])
    df.to_csv(filename, sep=',', index=False, mode='w')


def init_all_river_data(filename, is_empty=True, hand_cards=None):
    '''
    Create a table with all possible turn round cards.
    '''
    # Deprecated! must set hand_cards
    # suit_list = list(LookUpStr.SUIT.value)
    # rank_list = list(LookUpStr.RANK.value)
    # possible_hands = []
    # if hand_cards is None:
    #     # num_of_type for possible hands is 13 * 13 = 169
    #     for i in range(len(rank_list)):
    #         possible_hands.append([rank_list[i] + suit_list[0], rank_list[i] + suit_list[1]])
    #         for j in range(i + 1, len(rank_list)):
    #             possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[1]])
    #             possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[0]])
    # else:
    #     possible_hands.append(hand_cards)
    if hand_cards is None:
        raise ValueError("must set hand_cards!")

    columns = ('hand_cards', 'flop_cards', 'turn_card', 'river_card', 'label', 'ehs')

    # Load existing file
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=None)
    else:
        # Create file
        df = pd.DataFrame(columns=columns)
        df.to_csv(filename, sep=',', index=False, mode='w')
    exist_num = df.shape[0]

    # Initialize left deck cards
    deck = init_deck()
    for card in hand_cards:
        deck.remove(card)
    df = pd.DataFrame(columns=columns)
    # Traverse all possible combinations
    for index, public_cards in enumerate(combinations(deck, 5)):
        # Skip existing data
        if (index + 1) <= exist_num:
            print("\rSkip:{}".format(index + 1), end='')
            continue
        cards = PokerState.get_suit_normalization(hand_cards, public_cards)
        data_dict = {}
        # hand_cards
        data_dict[columns[0]] = [''.join(cards[:2])]
        # flop_cards
        data_dict[columns[1]] = [''.join(cards[2:5])]
        # turn_card
        data_dict[columns[2]] = [cards[5]]
        # river_card
        data_dict[columns[3]] = [cards[6]]
        data_dict[columns[4]] = [-1]
        data_dict[columns[5]] = [0.0 if is_empty else calc_river_potential(cards[:2], cards[2:])]

        df = df.append(pd.DataFrame(data_dict), ignore_index=True, sort=False)
        # return
        # Quick Save
        if df.shape[0] == 10000:
            df.to_csv(filename, sep=',', index=False, mode='a', header=False)
            df = pd.DataFrame(columns=columns)
            print("{} Finish:{}".format(''.join(hand_cards), index + 1))
    # Save for the tail
    df.to_csv(filename, sep=',', index=False, mode='a', header=False)
    print("{} Finish:{}".format(''.join(hand_cards), index + 1))


def get_histogram_vector(array, bins=50, range=(0, 1)):
    '''
    Get the probability histogram vector of a numpy array.
    '''
    histogram_vec, _ = np.histogram(array, bins=bins, range=range, density=True)
    prob_vec = histogram_vec / np.sum(histogram_vec)
    return prob_vec


def sample_river_data(sample_num, subID=0):
    '''
    Sample poker data in river round.
    '''
    if subID != 0:
        print('Subprogress({}) start...'.format(subID))
    deck = init_deck()
    filename = 'poker_river_hand_potential.csv'
    column = ('hand_card0', 'hand_card1', 'flop_card0', 'flop_card1', 'flop_card2', 'turn_card', 'river_card', 'ehs')
    df = pd.DataFrame(columns=column)
    # Create csv
    if not os.path.exists(filename):
        df.to_csv(filename, sep=',', index=False, mode='w')
    # Sample
    for k in range(sample_num):
        cards = random.sample(deck, k=7)
        cards = PokerState.get_suit_normalization(cards[:2], cards[2:])
        ehs = calc_river_potential(cards[:2], cards[2:])
        data_dict = {}
        for i in range(len(cards)):
            data_dict[column[i]] = [cards[i]]
        data_dict[column[-1]] = [ehs]
        new_row = pd.DataFrame(data_dict)
        df = df.append(new_row, ignore_index=True)
        # Quick save
        if k % (sample_num / 10) == 0:
            df.to_csv(filename, sep=',', index=False, mode='a', header=False)
            df = pd.DataFrame(columns=column)

    df.to_csv(filename, sep=',', index=False, mode='a', header=False)

    if i != 0:
        print('Subprogress({}) finish.'.format(subID))


def sample_turn_data(sample_num, subID=0):
    '''
    Sample poker data in turn round.
    '''
    if subID != 0:
        print('Subprogress({}) start...'.format(subID))
    deck = init_deck()
    filename = 'poker_turn_hand_potential.csv'
    column = ['hand_card0', 'hand_card1', 'flop_card0', 'flop_card1', 'flop_card2', 'turn_card']
    column = tuple(column + ['{}~{}'.format(i, i + 2) for i in range(0, 100, 2)])
    df = pd.DataFrame(columns=column)
    # Create csv
    if not os.path.exists(filename):
        df.to_csv(filename, sep=',', index=False, mode='w')
    # Sample
    for k in range(sample_num):
        cards = random.sample(deck, k=6)
        cards = PokerState.get_suit_normalization(cards[:2], cards[2:])
        ehs_vector = calc_turn_potential(cards[:2], cards[2:])
        ehs_histogram = get_histogram_vector(ehs_vector)
        data_dict = {}
        for i in range(len(cards)):
            data_dict[column[i]] = [cards[i]]
        for i in range(0, 100, 2):
            data_dict['{}~{}'.format(i, i + 2)] = [ehs_histogram[int(i / 2)]]
        new_row = pd.DataFrame(data_dict)
        df = df.append(new_row, ignore_index=True, sort=False)
        # Quick save
        if k % (sample_num / 10) == 0:
            df.to_csv(filename, sep=',', index=False, mode='a', header=False)
            df = pd.DataFrame(columns=column)

    df.to_csv(filename, sep=',', index=False, mode='a', header=False)

    if subID != 0:
        print('Subprogress({}) finish.'.format(subID))


def sample_flop_data(sample_num, subID=0):
    '''
    Sample poker data in flop round.
    '''
    if subID != 0:
        print('Subprogress({}) start...'.format(subID))
    deck = init_deck()
    filename = 'poker_flop_hand_potential.csv'
    column = ['hand_card0', 'hand_card1', 'flop_card0', 'flop_card1', 'flop_card2']
    column += ['next:{}~{}'.format(i, i + 2) for i in range(0, 100, 2)]
    column += ['final:{}~{}'.format(i, i + 2) for i in range(0, 100, 2)]
    column = tuple(column)
    df = pd.DataFrame(columns=column)
    # Create csv
    if not os.path.exists(filename):
        df.to_csv(filename, sep=',', index=False, mode='w')
    # Sample
    for k in range(sample_num):
        cards = random.sample(deck, k=5)
        cards = PokerState.get_suit_normalization(cards[:2], cards[2:])
        potential_in_turn, potential_in_river = calc_flop_potential(cards[:2], cards[2:])
        potential_in_turn = get_histogram_vector(potential_in_turn)
        potential_in_river = get_histogram_vector(potential_in_river)
        data_dict = {}
        for i in range(len(cards)):
            data_dict[column[i]] = [cards[i]]
        for i in range(0, 100, 2):
            data_dict['next:{}~{}'.format(i, i + 2)] = [potential_in_turn[int(i / 2)]]
            data_dict['final:{}~{}'.format(i, i + 2)] = [potential_in_river[int(i / 2)]]
        new_row = pd.DataFrame(data_dict)
        df = df.append(new_row, ignore_index=True, sort=False)
        # Quick save
        if k % (sample_num / 10) == 0:
            df.to_csv(filename, sep=',', index=False, mode='a', header=False)
            df = pd.DataFrame(columns=column)

    df.to_csv(filename, sep=',', index=False, mode='a', header=False)

    if subID != 0:
        print('Subprogress({}) finish.'.format(subID))


def sample_data(func, sample_num, thread_num=1):
    '''
    Get sampled data of card potential. Save as csv files.

    Args:
        func (str): 'flop', 'turn', 'river'
        sample_num (int): number of sample data for each thread
        thread_num (int): multiprocessing(default=1)
    '''
    run_function = func
    if run_function == 'river':
        func_name = sample_river_data
    elif run_function == 'turn':
        func_name = sample_turn_data
    elif run_function == 'flop':
        func_name = sample_flop_data
    else:
        raise ValueError("func must be 'flop', 'turn' or 'river'!")

    p = Pool(thread_num)
    for i in range(thread_num):
        p.apply_async(func_name, args=(int(sample_num), i + 1))
    p.close()
    p.join()
    # data deduplicate
    filename = 'poker_{}_hand_potential.csv'.format(run_function)
    df = pd.read_csv(filename, sep=',')
    df = df.drop_duplicates()
    df.to_csv(filename, sep=',', index=False, mode='w')
    print('Finish {} data sampling.'.format(run_function))


if __name__ == "__main__":

    # sample_flop_data(1)
    sample_num = sys.argv[1]
    subprocess_num = sys.argv[2]
    run_function = sys.argv[3]
    if not (sample_num.isdecimal() and subprocess_num.isdecimal()):
        raise TypeError('argv must be integers.')
    sample_data(run_function, int(sample_num), int(subprocess_num))
