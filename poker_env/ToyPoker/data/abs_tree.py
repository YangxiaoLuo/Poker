import pickle
import pandas as pd

class Node:
    
    def __init__(self, card, label, level):
        self.card = card
        self.label = label
        self.level = level
        self.next = []
    
    def add(self, cards, label):
        if len(cards) == 2 * self.level:
            self.label = label
            return
        else:
            card = cards[2 * self.level] + cards[2 * self.level + 1]
            for next_node in self.next:
                if next_node.card == card:
                    next_node.add(cards, label)
                    return
            new_node = Node(card, None, self.level + 1)
            self.next.append(new_node)
            new_node.add(cards, label)
            return
    
    def get_label(self, cards):
        if len(cards) == 2 * self.level:
            return self.label
        else:
            card = cards[2 * self.level] + cards[2 * self.level + 1]
            for next_node in self.next:
                if next_node.card == card:
                    return next_node.get_label(cards)
         
def generate(path):
    
    root = Node(None, None, 0)
    first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_ehs_vector.csv', index_col=None, low_memory = False)
    n1 = len(first_round_table.index)
    for i in range(n1):
        cards = first_round_table['cards_str'][i]
        label = "first_{}".format(first_round_table['label'][i])
        root.add(cards, label)
    
    #final_round_table = pd.read_csv('toypoker_final_ehs.csv', index_col=None, low_memory = False)


    final_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', index_col=None, low_memory = False)
    n2 = len(final_round_table.index)    
    for i in range(n2):
        cards = final_round_table['cards_str'][i]
        if cards != 'cards_str':
            label = 'final_{}'.format(final_round_table['label'][i])
            root.add(cards, label)
            
    file = open(path, 'wb')
    pickle.dump(root, file)
    file.close()
    print("The generation is done.")

def test():
    file = open('poker_env/ToyPoker/data/abs_tree', 'rb')
    root = pickle.load(file)
    first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_ehs_vector.csv', index_col=None, low_memory = False)
    n1 = len(first_round_table.index)
    for i in range(n1):
        cards = first_round_table['cards_str'][i]
        true_label = 'first_{}'.format(first_round_table['label'][i])
        label = root.get_label(cards)
        if label != true_label:
            print("The test of first table failed.")
    final_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', index_col=None, low_memory = False)
    n2 = len(final_round_table.index)
    for i in range(n2):
        cards = final_round_table['cards_str'][i]
        true_label = 'final_{}'.format(final_round_table['label'][i])
        label = root.get_label(cards)
        if label != true_label:
            print("The test of final table failed.")
    file.close()
    print("The test is done.")
