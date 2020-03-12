import os
import pickle
import pandas as pd


class Node:
    '''
    Node in abs_tree, only leaves carry labels.
    '''
    def __init__(self, card, label, level):
        '''
        Initiate a node in abs_tree.
        
        Args:
            card (str): string of a card, which is rank + suit
            label (int): label of the encoded lossless state
            level (int): level of this Node in abs_tree, which is also the number of cards
        '''
        self.card = card
        self.label = label
        self.level = level
        self.next = []

    def add(self, cards, label):
        '''
        Add a node to the tree.

        Args:
            cards(str): lossless state corresponding to this added node
            label(str): label of the lossless state
        '''
        if len(cards) == 2 * self.level:
            self.label = label
        else:
            card = cards[2 * self.level] + cards[2 * self.level + 1]
            for next_node in self.next:
                if next_node.card == card:
                    next_node.add(cards, label)
                    return
            new_node = Node(card, None, self.level + 1)
            self.next.append(new_node)
            new_node.add(cards, label)

    def get_label(self, cards):
        '''
        Get the label of lossless state.

        Args:
            cards(str): a lossless state

        Return:
            (str): label of lossless state
        '''
        if len(cards) == 2 * self.level:
            return self.label
        else:
            card = cards[2 * self.level] + cards[2 * self.level + 1]
            for next_node in self.next:
                if next_node.card == card:
                    return next_node.get_label(cards)


def generate_abs_tree(stage):
    '''
    Generate abs_tree from csv files.

    Args:
        stage(str): 'first' or 'final'
    '''
    if os.path.isfile('poker_env/ToyPoker/data/abs_tree') is False:
        root = Node(None, None, 0)
    else:
        file = open('poker_env/ToyPoker/data/abs_tree', 'rb')
        root = pickle.load(file)
        file.close()

    if stage == 'first':
        table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_EHS_vector.csv', index_col=None, low_memory=False)
    elif stage == 'final':
        table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_EHS.csv', index_col=None, low_memory=False)
    else:
        raise ValueError("Stage must be 'first' or 'final'.")

    n = len(table.index)
    for i in range(n):
        cards = table['cards_str'][i]
        label = table['label'][i]
        root.add(cards, label)
    file = open('poker_env/ToyPoker/data/abs_tree', 'wb')
    pickle.dump(root, file)
    file.close()
