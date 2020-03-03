import pandas as pd

def cards_sort(cards):
    n_cards = int(len(cards)/2)
    cards_list = [cards[2*i]+cards[2*i+1]  for i in range(n_cards)]
    cards_list.sort()
    sorted_cards = "".join(cards_list)
    return sorted_cards

if __name__ == "__main__":
    first_round_table = pd.read_csv('toypoker_first_ehs_vector.csv', index_col=None, low_memory = False)
    final_round_table = pd.read_csv('toypoker_final_ehs.csv', index_col=None, low_memory = False)
    
    #final_round_table = final_round_table.drop([63])
    
    n1 = len(first_round_table.index)
    for i in range(n1):
        sorted_cards = cards_sort(first_round_table['cards_str'][i])
        first_round_table.at[i, 'cards_str'] = sorted_cards
        
    n2 = len(final_round_table.index)
    for i in range(n2):
        sorted_cards = cards_sort(final_round_table['cards_str'][i])
        final_round_table.at[i, 'cards_str'] = sorted_cards   
    
    first_round_table = first_round_table.sort_values('cards_str')
    final_round_table = final_round_table.sort_values('cards_str')
    
    first_round_table.to_csv('toypoker_first_ehs_vector_sorted.csv', sep=',', index=False)
    final_round_table.to_csv('toypoker_final_ehs_sorted.csv', sep=',', index=False)
