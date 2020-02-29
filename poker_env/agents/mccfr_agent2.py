import numpy as np
import os
import pandas as pd
import collections
import copy
import random
import math
import pickle

import poker_env.agents.mccfr_agent2_utils as utils

from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.data.eval_potential import calc_final_potential

# index of infoset
ind_i = 1
# index of publicstate
ind_p = 1

neg_p_threshold = -5
neg_p_prob = 0.95

T = 100

first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_ehs_vector.csv', index_col=None)
final_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', index_col=None)

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

    def search_infoset(self, state, action, env):
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
        ps = self._search_publicstate(state, action, env)
        return ps._search_infoset(state)
        
    def _search_publicstate(self, state, action, env):
        '''
        The first step in search_infoset
        '''
        last_action = action
        for ps in self.next:
            if ps.last_action == last_action:
                return ps
        actionspace = utils.action_abstraction(state)
        done = env.is_over()
        ps = PublicState(last_action, actionspace, done)
        self.next.append(ps)
        return ps
        
    def _search_infoset(self, state):
        '''
        The second step in search_infoset
        '''
        #card = utils.card_abstraction(state)
        card = encode_state(state)
        for ins in self.infoset:
            if ins.card == card:
                return ins
        ins = Infoset(card, self)
        self.infoset.append(ins)
        return ins
        
class Infoset:

    def __init__(self, card, publicstate):
    
        global ind_i
        self.name = "Infoset "+str(ind_i)
        ind_i += 1
        self.publicstate = publicstate
        if self.publicstate.done == False:
            # self.param = copy.deepcopy(self.publicstate.actionspace)
            self.param = [{'action': action} for action in self.publicstate.actionspace]
            for a in self.param:
                a['regret'] = 0
                a['strategy'] = 1/len(self.publicstate.actionspace)
        self.card = card
        
    def disp(self):
        print(self.name)
        print("publicstate: ", self.publicstate.name)
        print("card: ", self.card)
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
            cur_player = utils.get_player(state)
            if cur_player == player:
                positive_regret = [a['regret'] for a in self.param]
                for i,a in enumerate(self.param):
                    if a['regret'] < 0:
                        positive_regret[i] = 0
                sum_regret = sum(positive_regret)
                if sum_regret != 0:
                    for i,a in enumerate(self.param):
                        a['strategy'] = positive_regret[i]/sum_regret
                else:
                    for a in self.param:
                        a['strategy'] = 1/len(self.publicstate.actionspace)
                #if [a['regret'] for a in self.param] != [0]*len(self.param):
                    #self.disp()
                action = utils.sample_action(self.publicstate.actionspace, [a['strategy'] for a in self.param])
                next_state, _ = env.step(action)
                # the next node
                next_infoset = self.publicstate.search_infoset(next_state, action, env)
                next_infoset.update_strategy(player, env)
                # the current node
            else:
                for action in self.publicstate.actionspace:
                    next_state, _ = env.step(action)
                    # the next node
                    next_infoset = self.publicstate.search_infoset(next_state, action, env)
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
            neg_p(Bool)
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
            cur_player = utils.get_player(state)
            # temp strategy
            temp_param = copy.deepcopy(self.param)
            sum_regret = sum([a['regret'] for a in self.param])
            if sum_regret != 0:
                for a in temp_param:
                    a['strategy'] = a['regret']/sum_regret
            else:
                for a in temp_param:
                    a['strategy'] = 1/len(self.publicstate.actionspace)
                    
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
                        next_infoset = self.publicstate.search_infoset(next_state, self.publicstate.actionspace[i], env)
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
                action = utils.sample_action(self.publicstate.actionspace, [a['strategy'] for a in self.param])
                next_state, _ = env.step(action)
                # the next node
                if env.is_over() == True:
                    reward = env.get_payoffs()[player]
                else:
                    reward = 0
                next_infoset = self.publicstate.search_infoset(next_state, action, env)
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
        
        self.load_agent(20200224)
        #self.root.recur()
        #return
        for t in range(T):
            for p in range(self.env.player_num):
                initial_state, _ = self.env.init_game()
                self.root.actionspace =utils.get_actionspace(initial_state) # relunctant
                initial_infoset = self.root._search_infoset(initial_state)
                
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
        #self.root.print_tree()
        self.save_agent(20200224)
        #print("The agent is saved.")
        
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
        #print(ind_i)
        #print(ind_p)
        file.close()        
        return
    
    def step(self, state):
        return
    
    def eval_step(self, state):
        self.load_agent(20200224)
        history = state.history
        #card = utils.card_abstraction(state)
        card = encode_state(state)
        actionspace = utils.action_abstraction(state)
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
                strategy = [1/len(actionspace)]*len(actionspace)
                break
        if history == []:
            ps_flag = True
        if ps_flag == True:   
            for iset in ps.infoset:
                if iset.card == card:
                    strategy = [a['strategy'] for a in iset.param] # nontrivial strategy
                    iset_flag = True
                    break
            if iset_flag == False:
                print("No matching infoset.")
                strategy = [1/len(actionspace)]*len(actionspace)
        if actionspace != ps.actionspace:
            print("Actionspaces are not matched.")
            strategy = [1/len(actionspace)]*len(actionspace)
        action = utils.sample_action(actionspace, strategy)
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
        cluster_label = first_round_table[first_round_table['cards_str'] == lossless_state]['label'].values[0]
        lossy_state = 'first_{}'.format(cluster_label)
    else:
        # Calculate directly(√) / Read from table
        cluster_label = final_round_table[final_round_table['cards_str'] == lossless_state]['label'].values[0]
        # cluster_label = int(calc_final_potential(state.hand_cards, state.public_cards) * 50)
        lossy_state = 'final_{}'.format(cluster_label)
    return lossy_state
