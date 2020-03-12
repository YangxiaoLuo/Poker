import pickle
import pandas as pd
from itertools import combinations
from poker_env.ToyPoker.hand import compare_2_hands
from poker_env.ToyPoker.card import LookUpStr
from poker_env.ToyPoker.state import PokerState
from poker_env.ToyPoker.data import abs_tree


def init_deck():
    '''
    Initialize a deck.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    deck = [rank + suit for rank in rank_list for suit in suit_list]
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


def generate_final_data(final_abs_csv_path):

    deck = init_deck()
    column = ('cards_str', 'label', 'ehs')
    df = pd.DataFrame(columns=column)
    df.to_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', sep=',', index=False)
    complete_list = []
    for hand_cards in combinations(deck, 2):
        deck_ = [card for card in deck if card not in hand_cards]
        for public_cards in combinations(deck_, 5):
            final_lossless_state = ''.join(PokerState.get_suit_normalization(hand_cards, public_cards))
            if final_lossless_state in complete_list:
                continue
            else:
                complete_list.append(final_lossless_state)
            data_dict = {}
            data_dict['cards_str'] = final_lossless_state
            data_dict['label'] = None
            data_dict['ehs'] = calc_final_potential(hand_cards, public_cards)
            new_row = pd.DataFrame(data_dict)
            new_row.to_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', sep=',', index=False, mode='a', header=False)


def generate_first_data():
    deck = init_deck()
    file = open('poker_env/ToyPoker/data/abs_tree', 'rb')
    c_root = pickle.load(file)
    column = ('cards_str', 'potential')
    df = pd.DataFrame(columns=column)
    df.to_csv('poker_env/ToyPoker/data/toypoker_first_potential.csv', sep=',', index=False)
    complete_list = []
    for hand_cards in combinations(deck, 2):
        deck_ = [card for card in deck if card not in hand_cards]
        for public_cards in combinations(deck_, 3):
            first_lossless_state = ''.join(PokerState.get_suit_normalization(hand_cards, public_cards))
            if first_lossless_state in complete_list:
                continue
            else:
                complete_list.append(first_lossless_state)
            data_dict = {}
            data_dict['cards_str'] = first_lossless_state
            potential = []
            deck__ = [card for card in deck_ if card not in public_cards]

            for final_cards in combinations(deck__, 2):
                lossless_state = ''.join(PokerState.get_suit_normalization(hand_cards, public_cards + final_cards))
                cluster = c_root.get_label(lossless_state)
                if cluster not in potential:
                    potential.append(cluster)
            data_dict['potential'] = potential
            new_row = pd.DataFrame(data_dict)
            new_row.to_csv('poker_env/ToyPoker/data/toypoker_first_potential.csv', sep=',', index=False, mode='a', header=False)
