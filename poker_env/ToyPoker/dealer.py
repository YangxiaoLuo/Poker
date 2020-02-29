import random
from poker_env.ToyPoker.utils import init_24_deck


class ToyPokerDealer:
    '''
    The Dealer class for Toy Poker
    '''
    def __init__(self):
        '''
        Initialize a dealer class.
        '''
        self.__deck = init_24_deck()
        self.shuffle()

    def shuffle(self):
        '''
        Shuffle the deck.
        '''
        random.shuffle(self.__deck)

    def deal_card(self):
        '''
        Deal one card from the deck.
        Returns:
            (Card): The drawn card from the deck
        '''
        return self.__deck.pop()
