from poker_env.Texas.card import LookUpStr


class PokerState:
    '''
    A Class of the Heads-Up Limit Texas Hold'em Poker game tree.

    Attributes:
        player_id (int): the id of the current player
        legal_actions (list): a list of Strings, each element corresponds to a legal action
        history (list): action sequence of observable history
        hand_cards (list): a list of str
        public_cards (list): a list of str
    '''
    def __init__(self, num_player=2):
        '''
        Initialize a node.
        '''
        self.__num_player = num_player
        self.__player_id = -1
        self.__legal_actions = []
        self.__pot = []
        self.__history = []
        self.__hand_cards = []
        self.__public_cards = []

    @property
    def num_player(self):
        return self.__num_player

    @property
    def player_id(self):
        # get the current player's id to select action
        return self.__player_id

    @property
    def legal_actions(self):
        return self.__legal_actions

    @property
    def history(self):
        # return action sequence
        return self.__history

    @property
    def previous_own_actions(self):
        action_sequence = ''
        for player_id, action in self.__history:
            if player_id == self.player_id:
                action_sequence += action[0]
        return action_sequence

    @property
    def previous_all_actions(self):
        action_sequence = ''
        for player_id, action in self.__history:
            action_sequence += action[0]
        return action_sequence

    @property
    def hand_cards(self):
        return self.__hand_cards

    @property
    def public_cards(self):
        return self.__public_cards

    @property
    def pot(self):
        return self.__pot

    @property
    def lossless_cards(self):
        return PokerState.get_suit_normalization(self.hand_cards, self.public_cards)

    def move(self, player_id, action, pot, round_counter):
        self.__history.append((player_id, action, sum(pot) - pot[player_id], round_counter))

    def set_state(self, player_id, hand_cards, public_cards, legal_actions, pot):
        self.__player_id = player_id
        self.__hand_cards = hand_cards
        self.__public_cards = public_cards
        self.__legal_actions = legal_actions
        self.__pot = pot

    def get_infoset(self):
        '''
        Conduct lossless abstraction for cards information and return the infoset this history belongs to.

        Returns:
            (str): a long string representing the infoset
        '''
        normalized_cards = PokerState.get_suit_normalization(self.hand_cards, self.public_cards)
        infoset_str = ''.join(normalized_cards)
        return infoset_str

    @classmethod
    def get_suit_normalization(cls, hand_cards, public_cards):
        '''
        Remove the influence of suit order.

        Returns:
            (list): strings of cards' ranks and suits. First 2 are hand cards.
        '''
        # card[0] -> rank, card[1] -> suit
        # (1) Conduct suit normalization
        # Sort cards through their ranks
        sorted_cards = sorted(hand_cards, key=lambda card: LookUpStr.RANK.index(card[0]), reverse=True) \
            + sorted(public_cards, key=lambda card: LookUpStr.RANK.index(card[0]), reverse=True)
        sorted_string = ''.join(sorted_cards)

        # Special handling for Pair hand cards + public cards
        if sorted_cards[0][0] == sorted_cards[1][0] and len(sorted_cards) > 2:
            # Sort hand cards through flush counts
            suit_num = [sorted_string.count(card[1]) for card in sorted_cards[:2]]
            if suit_num[0] < suit_num[1]:
                sorted_cards = sorted_cards[1::-1] + sorted_cards[2:]
            elif suit_num[0] == suit_num[1] and suit_num[0] > 1:
                # Sort hand cards through flush rank
                max_rank = [sorted_string[sorted_string[4:].index(card[1]) + 3] for card in sorted_cards[:2]]
                if max_rank[0] < max_rank[1]:
                    sorted_cards = sorted_cards[1::-1] + sorted_cards[2:]
            sorted_string = ''.join(sorted_cards)
        # suit replacement
        origin_suit_order = ''
        for suit in sorted_string[1::2]:
            if suit not in origin_suit_order:
                origin_suit_order += suit
        replaced_suit_order = LookUpStr.SUIT.value[0:len(origin_suit_order)]
        normal_string = sorted_string.translate(str.maketrans(origin_suit_order, replaced_suit_order))
        normal_cards = [normal_string[i:i + 2] for i in range(0, len(normal_string), 2)]

        # (2) Conduct cards sorted
        sorted_normal_cards = sorted(normal_cards[:2], key=lambda card: (-LookUpStr.RANK.index(card[0]), LookUpStr.SUIT.index(card[1]))) \
            + sorted(normal_cards[2:], key=lambda card: (-LookUpStr.RANK.index(card[0]), LookUpStr.SUIT.index(card[1])))

        return sorted_normal_cards
