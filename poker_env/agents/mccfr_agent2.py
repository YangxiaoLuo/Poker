import os
import copy
import random
#import math
import pickle

import numpy as np
import pandas as pd

from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.data.eval_potential import calc_final_potential
import poker_env.ToyPoker.data.abs_tree as abs_tree

# index of infoset
ind_i = 1
# index of publicstate
ind_p = 1

neg_p_threshold = -5
neg_p_prob = 0.95

T = 100

#first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_ehs_vector.csv', index_col='cards_str', low_memory = False)
#final_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', index_col='cards_str', low_memory = False)

file = open('poker_env/ToyPoker/data/abs_tree', 'rb')
c_root = pickle.load(file)

class PublicState:

    def __init__(self, last_action, actionspace, done):
    
        global ind_p
        self.name = "PublicState "+str(ind_p)
        ind_p += 1
        self.infoset = []
        self.next = []
        self.last_action = last_action
        self.actionspace = actionspace
        self.done = done
        
    def disp(self):
        print(self.name, "*")
        print("last_action", self.last_action)
        print("infoset:", end = " ")
        for i in self.infoset:
            print(i.name, end = " ")
        print()
        print("next:")
        for p in self.next:
            print(p.name, p.last_action)
        print("actionspace: ", self.actionspace)
        print("done: ", self.done)
        print()
        
    def print_tree(self):
        self.disp()
        for ps in self.next:
            ps.print_tree()

    def search_infoset(self, cards, action, actionspace, done):
        '''
        First, search the next publicstate ps in self.next.
        Secondly, search infoset in ps.infoset.
        
        Args:
            state (PokerState)
            action (str): can be replaced by the last action in state.history
            env (Env): just need env.is_over(), which is not found in state
        Returns:
            (Infoset): the infoset corresponding to state
        '''
        ps = self._search_publicstate(action, actionspace, done)
        return ps._search_infoset(cards)
        
    def _search_publicstate(self, action, actionspace, done):
        '''
        The first step in search_infoset
        '''
        last_action = action
        for ps in self.next:
            if ps.last_action == last_action:
                return ps
        ps = PublicState(last_action, actionspace, done)
        self.next.append(ps)
        return ps
        
    def _search_infoset(self, cards):
        '''
        The second step in search_infoset
        '''
        for ins in self.infoset:
            if ins.cards == cards:
                return ins
        ins = Infoset(cards, self)
        self.infoset.append(ins)
        return ins
        
class Infoset:

    def __init__(self, cards, publicstate):
    
        global ind_i
        self.name = "Infoset "+str(ind_i)
        ind_i += 1
        self.publicstate = publicstate
        if self.publicstate.done == False:
            actionspace_size = len(self.publicstate.actionspace)
            self.param = [{'action': action, 'regret': 0, 'strategy': 1/actionspace_size} for action in self.publicstate.actionspace]
        self.cards = cards
        
    def disp(self):
        print(self.name)
        print("publicstate: ", self.publicstate.name)
        print("cards: ", self.cards)
        print("param: ", self.param)
        print()
    
    def update_strategy(self, player, env):
        '''
        Update strategy, then sample an action at player's node.
        Traverse all actions otherwise.
        
        Args:
            player(int): id of the player updating strategy.
            env(Env)
        '''
        if self.publicstate.done == True:
            env.step_back()
            # the previous node
            return
        else:
            state = env.get_state()
            cur_player = state.player_id
            
            if cur_player == player:
                positive_regret = [a['regret'] for a in self.param]
                for i,r in enumerate(positive_regret):
                    if r < 0:
                        positive_regret[i] = 0
                sum_regret = sum(positive_regret)
                if sum_regret != 0:
                    for i,a in enumerate(self.param):
                        a['strategy'] = positive_regret[i]/sum_regret
                else:
                    actionspace_size = len(self.publicstate.actionspace)
                    for a in self.param:
                        a['strategy'] = 1/actionspace_size
                       
                action = sample_action(self.publicstate.actionspace, [a['strategy'] for a in self.param])
                next_state, _ = env.step(action)
                # the next node
                cards = encode_state_tree(next_state)
                done = env.is_over()
                actionspace = next_state.legal_actions
                next_infoset = self.publicstate.search_infoset(cards, action, actionspace, done)
                next_infoset.update_strategy(player, env)
                # the current node
            else:
                for action in self.publicstate.actionspace:
                    next_state, _ = env.step(action)
                    # the next node
                    cards = encode_state_tree(next_state)
                    done = env.is_over()
                    actionspace = next_state.legal_actions
                    next_infoset = self.publicstate.search_infoset(cards, action, actionspace, done)
                    next_infoset.update_strategy(player, env)
                    # the current node
            env.step_back()
            # the last node
            return
                    
    def update_regret(self, player, reward, neg_p, discount, env):
        '''
        Traverse every action at player's node, and then update regret.
        Sample an action otherwise.
        temp_param is used to store a temporary strategy.
        
        Args:
            player(int)
            reward(float)
            neg_p(Bool): whether use negative pruning
            discount(float)
            env(Env)
            
        Return:
            (float)
        '''
        
        if self.publicstate.done == True:
            env.step_back()
            return reward
            
        else:
            global neg_p_threshold
            state = env.get_state()
            cur_player = state.player_id
            # update temp strategy
            temp_param = copy.deepcopy(self.param)
            sum_regret = sum([a['regret'] for a in self.param]) # positive regret?
            if sum_regret != 0:
                for a in temp_param:
                    a['strategy'] = a['regret']/sum_regret
            else:
                actionspace_size = len(self.publicstate.actionspace)
                for a in temp_param:
                    a['strategy'] = 1/actionspace_size
                    
            v_I = 0
            if cur_player == player:
                for i,a in enumerate(temp_param):
                    # negative pruning
                    if self.param[i]['regret'] > neg_p_threshold or neg_p == False:
                        next_state, _ = env.step(a['action'])
                        # the next node
                        if env.is_over() == True:
                            reward = env.get_payoffs()[player]
                        else:
                            reward = 0
                            
                        cards = encode_state_tree(next_state)
                        done = env.is_over()
                        actionspace = next_state.legal_actions
                        next_infoset = self.publicstate.search_infoset(cards, self.publicstate.actionspace[i], actionspace, done)
                        a['value'] = next_infoset.update_regret(player, reward, neg_p, discount, env)
                        
                        # the current node
                        a['explored'] = True
                        v_I += a['value'] * a['strategy']
                    else:
                        a['explored'] = False
                for i,a in enumerate(temp_param):
                    if a['explored'] == True:
                        self.param[i]['regret'] += a['value'] - v_I
                        self.param[i]['regret'] *= discount
            else:
                action = sample_action(self.publicstate.actionspace, [a['strategy'] for a in self.param])
                next_state, _ = env.step(action)
                # the next node
                if env.is_over() == True:
                    reward = env.get_payoffs()[player]
                else:
                    reward = 0
                cards = encode_state_tree(next_state)
                done = env.is_over()
                actionspace = next_state.legal_actions
                next_infoset = self.publicstate.search_infoset(cards, action, actionspace, done)
                v_I = next_infoset.update_regret(player, reward, neg_p, discount, env)
                # the current node
                
            env.step_back()
            # the last node
            return v_I

class MCCFRAgent(Agent):

    def __init__(self, env, update_interval=100, discount_interval=1000):
        
        super().__init__(agent_type='MCCFRAgent')
        self.env = env
        self.update_interval = update_interval
        self.discount_interval = discount_interval
        
    def train(self):
        
        self.load_agent(20200302)
        
        if os.path.isfile('poker_env/ToyPoker/data/abs_tree') == False:
            abs_tree.generate('poker_env/ToyPoker/data/abs_tree')
            print('generate done')
            
        
        
        #abs_tree.test()
        
        #print('test done')
        
        for t in range(T):
            for p in range(self.env.player_num):
                initial_state, _ = self.env.init_game()
                self.root.actionspace = initial_state.legal_actions # relunctant
                cards = encode_state_tree(initial_state)
                initial_infoset = self.root._search_infoset(cards)
                
                a = random.random()
                if a < neg_p_prob:
                    neg_p = True
                else:
                    neg_p = False
                if t % self.discount_interval == 0:
                    discount = (t/self.discount_interval)/((t/self.discount_interval)+1)
                else:
                    discount = 1
                _ = initial_infoset.update_regret(p, 0, neg_p, discount, self.env)
                if t % self.update_interval == 0:
                    initial_infoset.update_strategy(p, self.env)

        self.save_agent(20200302)
        
    def save_agent(self, index):
        
        global ind_i
        global ind_p
        file = open('policy_{}'.format(index), 'wb')
        pickle.dump(self.root, file)
        pickle.dump(ind_i, file)
        pickle.dump(ind_p, file)
        file.close()
        return
    
    def load_agent(self, index):
    
        if os.path.isfile('policy_{}'.format(index)) == False:
            self.root = PublicState(0, [], False)
            return
        global ind_i
        global ind_p
        file = open('policy_{}'.format(index), 'rb')
        self.root = pickle.load(file)
        ind_i = pickle.load(file)
        ind_p = pickle.load(file)
        file.close()        
        return
    
    def step(self, state):
        return
    
    def eval_step(self, state):
    
        self.load_agent(20200302)
        history = state.history
        cards = encode_state_tree(state)
        actionspace = state.legal_actions
        actionspace_size = len(actionspace)
        ps = self.root
        strategy = []
        ps_flag = False
        iset_flag = False
        for a in history:
            for n_ps in ps.next:
                if n_ps.last_action == a[1]:
                    ps = n_ps
                    ps_flag = True
                    break
            if ps_flag == False:
                print("No matching publicstate.")
                strategy = [1/actionspace_size]*actionspace_size
                break
        if history == []:
            ps_flag = True
        if ps_flag == True:   
            for iset in ps.infoset:
                if iset.cards == cards:
                    strategy = [a['strategy'] for a in iset.param] # nontrivial strategy
                    iset_flag = True
                    break
            if iset_flag == False:
                print("No matching infoset.")
                strategy = [1/actionspace_size]*actionspace_size
        if actionspace != ps.actionspace:
            print("The actionspace is not matched.")
            strategy = [1/actionspace_size]*actionspace_size
        action = sample_action(actionspace, strategy)
        return action
        
def encode_state(state):
    '''
    Get infoset of state
    
    Args:
        state (PokerState): the state of the game
        
    Returns:
        (string): infoset keys.
    '''
    global first_round_table
    global final_round_table
    
    lossless_state = state.get_infoset()
    if len(state.public_cards) == 3:
        cluster_label = first_round_table.loc[lossless_state]['label']
        lossy_state = 'first_{}'.format(cluster_label)
        print(lossy_state)
    else:
        # Calculate directly / Read from table(√)
        # cluster_label = int(calc_final_potential(state.hand_cards, state.public_cards) * 50)
        cluster_label = final_round_table.loc[lossless_state]['label']
        lossy_state = 'final_{}'.format(cluster_label)
    return lossy_state
            
def encode_state_tree(state):

    global c_root
    lossless_state = state.get_infoset()
    lossy_state = c_root.get_label(lossless_state)
    return lossy_state
    
def sample_action(actionspace, distribution):
    return np.random.choice(actionspace, size=1, p=distribution)[0]
