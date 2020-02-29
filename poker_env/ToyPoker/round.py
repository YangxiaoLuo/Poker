from poker_env.ToyPoker.action import Action
# from action import Action


class ToyPokerRound:
    '''
    The Round class for Toy Poker.
    Round can call other Classes' functions to keep the game running.
    '''
    def __init__(self, num_players=2, init_raise_amount=2):
        '''
        Initialize a round class.

        Args:
            num_players (int): The number of players
            init_raise_amount (int): The min raise amount when every round starts
        '''
        self.game_pointer = None
        self.num_players = num_players
        self.init_raise_amount = init_raise_amount

        # If each player agree to not raise, the round is over.
        self.not_raise_num = 0
        self.alive_player_num = self.num_players

        # If having raised for 2 times, the round is over
        self.raise_times = 0

        # Raised amount for each player
        self.raised = [0 for _ in range(self.num_players)]

    def start_new_round(self, game_pointer, raised=None):
        '''
        Start a new bidding round

        Args:
            game_pointer (int): The index of the current player.
            raised (list): Initialize the bet chips for each player

        Note: For the first round of the game, we need to setup the big/small blind
        '''
        self.game_pointer = game_pointer
        self.not_raise_num = 0
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
            rebet_amount = call_amount + self.init_raise_amount
            self.raised[self.game_pointer] += rebet_amount
            players[self.game_pointer].in_chips += rebet_amount
            self.not_raise_num = 1
            self.raise_times += 1

        elif action == Action.CALL.value:
            call_amount = max(self.raised) - self.raised[self.game_pointer]
            self.raised[self.game_pointer] += call_amount
            players[self.game_pointer].in_chips += call_amount
            self.not_raise_num += 1

        elif action == Action.FOLD.value:
            players[self.game_pointer].status = 'folded'
            self.alive_player_num -= 1

        elif action == Action.CHECK.value:
            self.not_raise_num += 1

        self.game_pointer = (self.game_pointer + 1) % self.num_players
        # Skip the folded players and the all_in players
        while players[self.game_pointer].status != 'alive':
            self.game_pointer = (self.game_pointer + 1) % self.num_players
        return self.game_pointer

    # Deprecated!
    # def step_back(self, players, player_id, action):
    #     '''
    #     ()
    #     Restore the round before the specified action happens

    #     Args:
    #         players (list): The list of players that play the game
    #         player_id (int): The id of the player whose action need to be restored
    #         action (str or int): An legal action has been taken by the player
    #     '''
    #     pass

    def get_legal_actions(self, players):
        '''
        Obtain the legal actions for the current player

        Args:
            players (list): The players in the game

        Returns:
           (list):  A list of legal actions
        '''
        full_actions = [Action.FOLD.value]
        call_amount = max(self.raised) - self.raised[self.game_pointer]
        remained_chips = players[self.game_pointer].get_remained_chips()

        if self.raise_times < 2:
            full_actions.append(Action.RAISE.value)

        # If the current player has put in the chips that are more than others, he can check.
        if call_amount == 0:
            full_actions.append(Action.CHECK.value)
        # If the current player cannot provide call amount, he has to fold.
        elif call_amount >= remained_chips:
            return [Action.FOLD.value]
        # If the current chips are less than that of the highest one in the round, he can call.
        elif call_amount > 0:
            full_actions.append(Action.CALL.value)

        return full_actions

    def is_over(self):
        '''
        Check whether the round is over.

        Returns:
            (boolean): True if the current round is over
        '''
        if self.alive_player_num == self.num_players - 1:
            return True
        if self.not_raise_num == self.alive_player_num:
            return True
        else:
            return False

    def get_action_player_num(self):
        '''
        Return the number of players who can action

        Returns:
            (int): the result number
        '''
        return self.alive_player_num - self.alive_player_num


# # test
# import numpy as np
# import random
# from player import ToyPokerPlayer as Player

# if __name__ == "__main__":
#     num_players = 2
#     big_blind = 2
#     small_blind = 1

#     players = [Player(i, 100) for i in range(num_players)]
#     s = 1
#     b = 0
#     players[s].in_chips = small_blind
#     players[b].in_chips = big_blind

#     game_pointer = 1
#     r = ToyPokerRound(num_players=num_players, init_raise_amount=big_blind)
#     r.start_new_round(game_pointer=game_pointer, raised=[p.in_chips for p in players])

#     r.proceed_round(players, Action.CALL.value)
#     r.proceed_round(players, Action.CHECK.value)
#     r.proceed_round(players, Action.FOLD.value)
#     r.proceed_round(players, Action.FOLD.value)
#     r.proceed_round(players, Action.FOLD.value)
#     r.proceed_round(players, Action.FOLD.value)

#     while not r.is_over():
#         legal_actions = r.get_legal_actions(players)
#         action = random.choice(legal_actions)
#         if isinstance(action, int):
#             print(game_pointer, 'raise{}'.format(action))
#         else:
#             print(game_pointer, action)
#         game_pointer = r.proceed_round(players, action)
#         print(r.raised, '{}/{}'.format(r.not_raise_num+r.all_in_player_num, r.alive_player_num))
