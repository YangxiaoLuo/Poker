import random
import sys
import os
from poker_env.Texas.hand import compare_2_hands
from poker_env.Texas.card import LookUpStr


def init_all_hand_type():
    '''
    AKs~32s, AKo~32o, AA~22
    Return:
        (dict):
    '''
    all_hands = {}
    rank_string = LookUpStr.RANK.value
    for i in range(0, len(rank_string)):
        all_hands[rank_string[i] + rank_string[i]] = [0, 0]
        for j in range(i + 1, len(rank_string)):
            all_hands[rank_string[j] + rank_string[i] + 's'] = [0, 0]
            all_hands[rank_string[j] + rank_string[i] + 'o'] = [0, 0]
    return all_hands


def init_deck():
    '''
    Initialize a deck.
    '''
    suit_list = list(LookUpStr.SUIT.value)
    rank_list = list(LookUpStr.RANK.value)
    deck = [rank + suit for suit in suit_list for rank in rank_list]
    return deck


def get_hand_type(hand_cards):
    '''
    Judge which type the hand cards is.
    Return:
        (string): hand type
    '''
    if hand_cards[0][0] == hand_cards[1][0]:
        return hand_cards[0][0] + hand_cards[1][0]
    elif LookUpStr.RANK.index(hand_cards[0][0]) > LookUpStr.RANK.index(hand_cards[1][0]):
        hand_type = hand_cards[0][0] + hand_cards[1][0]
    else:
        hand_type = hand_cards[1][0] + hand_cards[0][0]

    if hand_cards[0][1] == hand_cards[1][1]:
        return hand_type + 's'
    else:
        return hand_type + 'o'


def game_simulation(hand_dict, num):
    deck = init_deck()
    for _ in range(num):
        all_cards = random.sample(deck, k=9)
        hand_cards0 = all_cards[0:2]
        hand_cards1 = all_cards[2:4]
        pub_cards = all_cards[4:]
        result = compare_2_hands(hand_cards0 + pub_cards, hand_cards1 + pub_cards)
        hand_dict[get_hand_type(hand_cards0)][0] += result[0]
        hand_dict[get_hand_type(hand_cards0)][1] += 1
        hand_dict[get_hand_type(hand_cards1)][0] += result[1]
        hand_dict[get_hand_type(hand_cards1)][1] += 1


def save_dict(data):
    f = open('hand_evaluation.txt', 'w')
    f.write(str(data))
    f.close()


def load_dict():
    if not os.path.exists('hand_evaluation.txt'):
        save_dict(init_all_hand_type())
    f = open('hand_evaluation.txt')
    data = eval(f.read())
    f.close()
    return data


if __name__ == "__main__":
    # hand_dict = init_all_hand_type()
    # save_dict(hand_dict)
    sample_num = sys.argv[1]
    hand_dict = load_dict()
    game_simulation(hand_dict, int(sample_num))
    save_dict(hand_dict)
