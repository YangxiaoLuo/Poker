import os
import sys
import random
import numpy as np
import pandas as pd
from itertools import combinations
from multiprocessing import Pool
from poker_env.ToyPoker.hand import compare_2_hands
from poker_env.ToyPoker.card import LookUpStr
from poker_env.ToyPoker.state import PokerState


def init_deck():
    '''
    Initialize a deck.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    deck = [rank + suit for suit in suit_list for rank in rank_list]
    return deck


def calc_final_potential(hand_cards, public_cards):
    '''
    Calculate Expected Hand Strength at Final Round.
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


def calc_first_potential(hand_cards, public_cards):
    '''
    Calculate the distribution of EHS at First Round.
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
    for other_public_cards in combinations(deck, 2):
        public_5_cards = public_cards + list(other_public_cards)
        vector_EHS.append(calc_final_potential(hand_cards, public_5_cards))
    return np.array(sorted(vector_EHS))


def get_histogram_vector(array, bins=50, range=(0, 1)):
    '''
    Get the probability histogram vector of a numpy array.
    '''
    histogram_vec, _ = np.histogram(array, bins=bins, range=range, density=True)
    prob_vec = histogram_vec / np.sum(histogram_vec)
    return prob_vec


def sample_final_data(sample_num, subID=0):
    '''
    Sample toypoker data in final round.
    '''
    if subID != 0:
        print('Subprogress({}) start...'.format(subID))
    deck = init_deck()
    filename = 'toypoker_final_hand_potential.csv'
    column = ('hand_card0', 'hand_card1', 'first_card0', 'first_card1', 'first_card2', 'final_card0', 'final_card1', 'ehs')
    df = pd.DataFrame(columns=column)
    # Create csv
    if not os.path.exists(filename):
        df.to_csv(filename, sep=',', index=False, mode='w')
    # Sample (Randomly)
    for k in range(sample_num):
        cards = random.sample(deck, k=7)
        cards = PokerState.get_suit_normalization(cards[:2], cards[2:])
        ehs = calc_final_potential(cards[:2], cards[2:])
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


def sample_first_data(sample_num, subID=0):
    '''
    Sample toypoker data in first round.
    '''
    if subID != 0:
        print('Subprogress({}) start...'.format(subID))
    deck = init_deck()
    filename = 'toypoker_first_hand_potential.csv'
    column = ['hand_card0', 'hand_card1', 'first_card0', 'first_card1', 'first_card2']
    vector_length = int((len(deck) - 5) * (len(deck) - 6) / 2)
    column = tuple(column + [str(i) for i in range(0, vector_length)])
    df = pd.DataFrame(columns=column)
    # Create csv
    if not os.path.exists(filename):
        df.to_csv(filename, sep=',', index=False, mode='w')
    # Sample (Randomly)
    for k in range(sample_num):
        cards = random.sample(deck, k=5)
        cards = PokerState.get_suit_normalization(cards[:2], cards[2:])
        ehs_vector = calc_first_potential(cards[:2], cards[2:])
        data_dict = {}
        for i in range(len(column)):
            if i < len(cards):
                data_dict[column[i]] = [cards[i]]
            else:
                data_dict[column[i]] = [ehs_vector[i - len(cards)]]
        new_row = pd.DataFrame(data_dict)
        df = df.append(new_row, ignore_index=True, sort=False)
        # Quick save
        if k % (sample_num / 10) == 0:
            df.to_csv(filename, sep=',', index=False, mode='a', header=False)
            df = pd.DataFrame(columns=column)

    df.to_csv(filename, sep=',', index=False, mode='a', header=False)

    if subID != 0:
        print('Subprogress({}) finish.'.format(subID))


def init_all_first_data(filename):
    '''
    Create a empty table with all possible first round cards.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    # Length of first round EHS vector is (24 - 5) * (24 - 6) / 2 = 171
    vector_length = int((len(suit_list) * len(rank_list) - 5) * (len(suit_list) * len(rank_list) - 6) / 2)
    columns = tuple(['cards_str', 'label'] + [str(i) for i in range(0, vector_length)])
    df = pd.DataFrame(columns=columns)
    # num_of_type for possible hands is 6 * 6 = 36
    possible_hands = []
    for i in range(len(rank_list)):
        possible_hands.append([rank_list[i] + suit_list[0], rank_list[i] + suit_list[1]])
        for j in range(i + 1, len(rank_list)):
            possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[1]])
            possible_hands.append([rank_list[j] + suit_list[0], rank_list[i] + suit_list[0]])

    for hand_cards in possible_hands:
        deck = init_deck()
        for card in hand_cards:
            deck.remove(card)
        for public_cards in combinations(deck, 3):
            cards = PokerState.get_suit_normalization(hand_cards, public_cards)
            data_dict = {}
            for k in range(len(columns)):
                if k == 0:
                    data_dict[columns[k]] = [''.join(cards)]
                elif k == 1:
                    data_dict[columns[k]] = [-1]
                else:
                    data_dict[columns[k]] = [0.0]
            df = df.append(pd.DataFrame(data_dict), ignore_index=True, sort=False)
    # df = df.drop_duplicates()
    df = df.drop_duplicates([columns[0]])
    df.to_csv(filename, sep=',', index=False, mode='w')


def init_all_final_data(filename):
    '''
    Create a table with all possible final round cards and calculate corresponding ehs value.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    columns = ('cards_str', 'label', 'ehs')
    df = pd.DataFrame(columns=columns)
    # num_of_type for possible hands is 6 * 6 = 36
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
        for public_cards in combinations(deck, 5):
            total_num += 1
            cards = PokerState.get_suit_normalization(hand_cards, public_cards)
            data_dict = {}
            for k in range(len(columns)):
                if k == 0:
                    data_dict[columns[k]] = [''.join(cards)]
                elif k == 1:
                    data_dict[columns[k]] = [-1]
                else:
                    data_dict[columns[k]] = [calc_final_potential(cards[:2], cards[2:])]
            df = df.append(pd.DataFrame(data_dict), ignore_index=True, sort=False)
            # Quick Save
            if total_num % 1000 == 0:
                if total_num == 1000:
                    df.to_csv(filename, sep=',', index=False, mode='w')
                else:
                    df.to_csv(filename, sep=',', index=False, mode='a', header=False)
                print("Finish:{}".format(total_num))
                df = pd.DataFrame(columns=columns)
    # Data deduplicate
    df = pd.read_csv(filename, index_col=None)
    df = df.drop_duplicates([columns[0]])
    df.to_csv(filename, sep=',', index=False, mode='w')


def sample_data(func, sample_num, thread_num=1):
    '''
    Get sampled data of card potential. Save as csv files.

    Args:
        func (str): 'flop', 'turn', 'river'
        sample_num (int): number of sample data for each thread
        thread_num (int): multiprocessing(default=1)
    '''
    run_function = func
    if run_function == 'final':
        func_name = sample_final_data
    elif run_function == 'first':
        func_name = sample_first_data
    else:
        raise ValueError("func must be 'first' or 'final'!")

    p = Pool(thread_num)
    for i in range(thread_num):
        p.apply_async(func_name, args=(int(sample_num), i + 1))
    p.close()
    p.join()
    # data deduplicate
    filename = 'toypoker_{}_hand_potential.csv'.format(run_function)
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