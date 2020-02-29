from poker_env.ToyPoker.utils import compare_all_hands


class ToyPokerJudger:
    '''
    The Judger class for Toy Poker.
    '''
    def __init__(self):
        '''
        Initialize a judger class.
        '''
    def judge_game(self, players, hands):
        '''
        Judge the winner of the game.

        Args:
            players (list): The list of the players who play the game.
            hand (list): The list of hands that from the players.
        Returns:
            (list): Each entry of the list corresponds to one entry of the plays.
        '''
        # Convert the hands into card indexes
        for i, hand in enumerate(hands):
            if hands[i] is not None:
                h = [card.get_index() for card in hand]
                hands[i] = h

        payoffs = [float(-p.in_chips) for p in players]

        # Compute the total chips
        main_pot = [p.in_chips for p in players]

        winners = compare_all_hands(hands)
        each_win = float(sum(main_pot)) / sum(winners)
        for i, _ in enumerate(players):
            if winners[i] == 1:
                payoffs[i] += each_win

        return payoffs


# from player import NoLimitTexasHoldemPlayer as Player
# from card import Card

# if __name__ == "__main__":

#     p0 = Player(0, 200)
#     p0.in_chips = 200
#     p0.status = 'all-in'
#     p1 = Player(0, 400)
#     p1.in_chips = 400
#     p1.status = 'all-in'
#     p2 = Player(0, 700)
#     p2.in_chips = 700
#     p2.status = 'all-in'
#     p3 = Player(0, 1600)
#     p3.in_chips = 1500
#     p3.status = 'alive'
#     p4 = Player(0, 2000)
#     p4.in_chips = 1500
#     p4.status = 'alive'
#     p5 = Player(0, 2000)
#     p5.in_chips = 1100
#     p5.status = 'folded'
#     players = [p0, p1, p2, p3, p4, p5]

#     # 6s6d+5c5h+8h Two_Pair Rank4
#     p0.hand = [Card('s', '6'), Card('d', '6')]
#     # 5s5c5h+4h4c Full_House Rank1
#     p2.hand = [Card('s', '2'), Card('s', '5')]
#     # 5d5c5h+4h4c Full_House Rank1
#     p1.hand = [Card('s', '3'), Card('d', '5')]
#     # 4h5h6h7s8h Straight Rank2
#     p3.hand = [Card('h', '6'), Card('s', '7')]
#     # 8s8h+5c5h+9s Two_Pair Rank3
#     p4.hand = [Card('s', '8'), Card('s', '9')]
#     # 8d8h+5c5h+4h Two_Pair Rank5
#     p5.hand = [Card('d', '8'), Card('h', '2')]
#     public_cards = [Card('h', '4'), Card('c', '4'), Card('c', '5'), Card('h', '5'), Card('h', '8')]
#     hands = [p.hand + public_cards if p.status != 'folded' else None for p in players]

#     judger = NoLimitTexasHoldemJudger()
#     payoffs = judger.judge_game(players, hands)
#     # expected payoffs = [-200, 700, 1600, 500, -1500, -1100]
#     print(payoffs)
