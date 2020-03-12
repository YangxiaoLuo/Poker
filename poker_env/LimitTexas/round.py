from poker_env.LimitTexas.action import Action


class LimitTexasHoldemRound:
    '''
    The Round class for Heads-Up Limit Texas Hold'em Poker.
    Round can call other Classes' functions to keep the game running.
    '''
    def __init__(self, num_players=2, big_blind=2):
        '''
        Initialize a round class.

        Args:
            num_players (int): The number of players
            big_blind (int): The big blind of the current game.
        '''
        self.game_pointer = None
        self.round_index = 0
        self.num_players = num_players
        self.big_blind = big_blind

        # If a player calls while both players have acted, a round ends.
        self.have_acted = False
        self.__is_over = False
        self.raise_times = 0

        # Raised amount for each player during the whole game.
        self.raised = [0 for _ in range(self.num_players)]

    @property
    def allow_raise_again(self):
        # In the 1st/2nd(0,1) betting round: no more than 3 raises
        # In the 3rd/4th(2,3) betting round: no more than 4 raises
        if self.round_index <= 1:
            return self.raise_times < 3
        else:
            return self.raise_times < 4

    @property
    def once_raise_amount(self):
        # In the 1st/2nd(0,1) betting round: equals to big blind
        # In the 3rd/4th(2,3) betting round: equals to 2 big blind
        if self.round_index <= 1:
            return self.big_blind
        else:
            return 2 * self.big_blind

    def start_new_round(self, game_pointer, round_index, raised=None):
        '''
        Start a new betting round

        Args:
            game_pointer (int): The index of the current player.
            round_index (int): (0~3 in each game).
            raised (list): Initialize the bet chips for each player

        Note: For the first round of the game, we need to setup the big/small blind
        '''
        self.game_pointer = game_pointer
        self.round_index = round_index
        self.raise_times = 0
        self.have_acted = False
        self.__is_over = False
        if raised:
            self.raised = raised
        else:
            self.raised = [0 for _ in range(self.num_players)]

    def proceed_round(self, players, action):
        '''
        Call other Classes' functions to keep one round running

        Args:
            players (list): The list of players that play the game
            action (str): An legal action taken by the player

        Returns:
            (int): The game_pointer that indicates the next player
        '''
        if action == Action.RAISE.value:
            call_amount = max(self.raised) - self.raised[self.game_pointer]
            rebet_amount = call_amount + self.once_raise_amount
            self.raised[self.game_pointer] += rebet_amount
            players[self.game_pointer].in_chips += rebet_amount
            self.have_acted = True
            self.raise_times += 1

        elif action == Action.CALL.value:
            call_amount = max(self.raised) - self.raised[self.game_pointer]
            self.raised[self.game_pointer] += call_amount
            players[self.game_pointer].in_chips += call_amount
            if self.have_acted:
                self.__is_over = True
            else:
                self.have_acted = True

        elif action == Action.FOLD.value:
            players[self.game_pointer].status = 'folded'
            self.__is_over = True

        self.game_pointer = (self.game_pointer + 1) % self.num_players
        return self.game_pointer

    def get_legal_actions(self, players):
        '''
        Obtain the legal actions for the current player

        Args:
            players (list): The players in the game

        Returns:
           (list):  A list of legal actions
        '''
        full_actions = [Action.FOLD.value, Action.CALL.value]

        if self.allow_raise_again:
            full_actions.append(Action.RAISE.value)

        return full_actions

    def is_over(self):
        '''
        Check whether the round is over.

        Returns:
            (boolean): True if the current round is over
        '''
        return self.__is_over


# # test
# from player import LimitTexasHoldemPlayer as Player
# from action import Action


# if __name__ == "__main__":
#     num_players = 2
#     big_blind = 2
#     small_blind = 1

#     players = [Player(i) for i in range(num_players)]

#     game_pointer = 1
#     r = LimitTexasHoldemRound(num_players=num_players, big_blind=big_blind)
#     r.start_new_round(game_pointer=game_pointer, round_index=2, raised=[p.in_chips for p in players])

#     r.proceed_round(players, Action.CALL.value)
#     r.proceed_round(players, Action.RAISE.value)
#     print(r.get_legal_actions(players))
#     print(r.is_over())
#     print([p.in_chips for p in players])
