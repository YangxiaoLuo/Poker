import numpy as np
from copy import deepcopy
from poker_env.Texas.dealer import NoLimitTexasHoldemDealer as Dealer
from poker_env.Texas.judger import NoLimitTexasHoldemJudger as Judger
from poker_env.LimitTexas.player import LimitTexasHoldemPlayer as Player
from poker_env.LimitTexas.round import LimitTexasHoldemRound as Round
from poker_env.LimitTexas.state import PokerState


class LimitTexasHoldemGame:
    '''
    The Game class for Heads-Up Limit Texas Hold'em Poker. Only support 2 players in this version.
    Suggest that each player has infinite chips in each game due to the 'Limit'.

    Rules reference:
        Deep Counterfactual Regret Minimization. Noam Brown, etc. ICML 2019:793-802

    Attributes:
        allow_step_back (boolean): allow game step back or not(Default=False).
        small_blind (int): the number of Small Blind(SB) chips of the game(Default=1).
        big_blind (int): the number of Big Blind(BB) chips of the game(Default=2).
    '''

    def __init__(self, allow_step_back=False, num_players=2, small_blind=1, big_blind=2):
        '''
        Initialize the settings of No Limit Texas Hold'em Game Class
        '''
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind

    def init_game(self, button=None):
        '''
        Initialize the objects of Heads-Up Limit Texas Hold'em Game Class.

        Deal 2 cards to each players and start the first round. (pre-flop)

        Args:
            button (int): the position of the button(Default=None, which means random).
        Returns:
            Returns:
            (tuple): Tuple containing:
                (PokerState): The first state of the game
                (int): Current player's id
        '''
        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger()
        # Initialize a dealer that can deal cards
        self.dealer = Dealer()
        # Initialize a round class which will proceed the game
        self.round = Round(self.num_players, self.big_blind)
        # Count the round. There are 4 rounds in each game.
        self.round_counter = 0
        # Initialize several players to play the game
        self.players = [Player(i) for i in range(self.num_players)]
        self.game_pointer = 0
        # Initialize public cards
        self.public_cards = []
        # Initialize the game tree
        self.game_tree = PokerState(self.num_players)
        # Save the history for stepping back to the last state.
        self.history = []

        # Set position: Button=small blind
        if button is not None and button >= self.num_players:
            raise ValueError("button is out of ranges")
        # Button
        self.button = np.random.randint(0, self.num_players) if button is None else button

        # Pre-flop round
        # The small blind acts first on the first round of betting before the flop (pre-flop).
        self.game_pointer = self.button
        self.players[self.button].in_chips = self.small_blind
        self.players[(self.button + 1) % self.num_players].in_chips = self.big_blind
        for i in range(2 * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal_card())
        self.round.start_new_round(game_pointer=self.game_pointer,
                                   round_index=self.round_counter,
                                   raised=[p.in_chips for p in self.players])
        state = self.get_state()
        return state, self.game_pointer

    def step(self, action):
        '''
        Get the next state

        Args:
            action (str): a specific action. (call, fold, check or raise)

        Ruturns:
            (tuple): Tuple containing:
                (PokerState): next player's state
                (int): next plater's id
        '''
        # If allowed, snapshot the current state
        if self.allow_step_back:
            r = deepcopy(self.round)
            b = deepcopy(self.game_pointer)
            r_c = deepcopy(self.round_counter)
            d = deepcopy(self.dealer)
            p = deepcopy(self.public_cards)
            ps = deepcopy(self.players)
            tr = deepcopy(self.game_tree)
            self.history.append((r, b, r_c, d, p, ps, tr))
        # Save the action and move game tree
        self.game_tree.move(self.game_pointer, action, self.game_tree.pot, self.round_counter)
        # Then proceed the action and get to the next state
        self.game_pointer = self.round.proceed_round(self.players, action)

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # For the flop round, we deal 3 cards
            if self.round_counter == 0:
                for _ in range(3):
                    self.public_cards.append(self.dealer.deal_card())
            # For the following rounds, we deal only 1 card
            elif self.round_counter <= 2:
                self.public_cards.append(self.dealer.deal_card())

            self.round_counter += 1
            # The person who was big blind pre-flop is first to act on the flop, turn, and river round.
            self.game_pointer = (self.button + 1) % self.num_players
            self.round.start_new_round(game_pointer=self.game_pointer,
                                       round_index=self.round_counter,
                                       raised=self.round.raised)
        state = self.get_state()
        return state, self.game_pointer

    def is_over(self):
        '''
        Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        # only one player left
        alive_players = [1 if p.status != 'folded' else 0 for p in self.players]
        if sum(alive_players) == 1:
            return True
        # 4 rounds finished
        if self.round_counter >= 4:
            return True
        return False

    def step_back(self):
        '''
        Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if len(self.history) > 0:
            self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, self.players, self.game_tree = self.history.pop()
            return True
        return False

    def get_player_num(self):
        '''
        Return the number of players in the game

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    def get_player_id(self):
        '''
        Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.game_pointer

    def get_state(self):
        '''
        Return the current player's state

        Returns:
            (PokerState): The observable state of the current player
        '''
        self.game_tree.set_state(player_id=self.round.game_pointer,
                                 hand_cards=[card.get_index() for card in self.players[self.round.game_pointer].hand],
                                 public_cards=[card.get_index() for card in self.public_cards],
                                 legal_actions=self.get_legal_actions(),
                                 pot=[p.in_chips for p in self.players])
        return self.game_tree

    def get_legal_actions(self):
        '''
        Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        return self.round.get_legal_actions(self.players)

    def get_payoffs(self):
        '''
        Return the payoffs of the game.

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        hands = [p.hand + self.public_cards if p.status != 'folded' else None for p in self.players]
        payoffs = self.judger.judge_game(self.players, hands)
        # payoffs = np.array(payoffs) / self.big_blind
        return payoffs
