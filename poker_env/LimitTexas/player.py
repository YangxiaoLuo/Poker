class LimitTexasHoldemPlayer:
    '''
    The Player class for Heads-Up Limit Texas Hold'em Poker.
    Suggest that each player has infinite chips in each game due to the 'Limit'.
    '''
    def __init__(self, player_id):
        '''
        Initialize a player class.

        Args:
            player_id (int): The id of the player
        '''
        self.player_id = player_id
        # The chips that this player has put in until now
        self.in_chips = 0
        self.hand = []
        # Status can be alive, folded
        self.status = 'alive'

    def get_player_id(self):
        '''
        Return the id of the player
        '''
        return self.player_id
