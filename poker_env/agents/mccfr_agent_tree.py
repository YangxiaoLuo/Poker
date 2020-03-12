import os
import copy
import random
import pickle
import numpy as np
import poker_env.ToyPoker.data.abs_tree
from poker_env.agents.base_agent import Agent


class PublicState:
    '''
    PublicState represents a sequence of actions.
    '''
    def __init__(self, last_action, actionspace, done):
        '''
        Initialize an publicstate

        Args:
            last_action (str): the last action in the action sequence represented by this publicstate
            actionspace (list): the actionspace of this publicstate
            done (bool): whether this publicstate is a termination
        '''
        self.infoset = []
        self.next = []
        self.last_action = last_action
        self.actionspace = actionspace
        self.done = done

    def search_infoset(self, cards, action, actionspace, done):
        '''
        Search the next publicstate ps in self.next, then search infoset in ps.infoset.

        Args:
            cards (int): label of cards corresponding to the searched infoset
            action (str): the next action
            actionspace (list): the actionspace of next publicstate, needed when constructing a new publicstate
            done(bool): whether the next publicstate is a termination, needed when constructing a new publicstate

        Returns:
            (Infoset): the searched infoset
        '''
        ps = self._search_publicstate(action, actionspace, done)
        return ps._search_infoset(cards)

    def _search_publicstate(self, action, actionspace, done):
        '''
        Search the next publicstate corresponding to action.

        Args:
            action (str): the next action
            actionspace (list): the actionspace of next publicstate
            done (bool): whether the next publicstate is a termination

        Returns:
            (PublicState): the searched publicstate
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
        Search the infoset attaching to self, corresponding to cards.
        Notice that this is also used when searching the initial state.

        Args:
            cards (int): label of cards corresponding to the searched infoset

        Returns:
            (Infoset): the searched infoset
        '''
        for ins in self.infoset:
            if ins.cards == cards:
                return ins
        ins = Infoset(cards, self)
        self.infoset.append(ins)
        return ins

    def update_all_strategy(self):
        '''
        Update all strategy. This is only possible in Toy Poker.
        '''
        for ins in self.infoset:
            positive_regret = np.maximum([a['regret'] for a in ins.param], 0)
            sum_regret = np.sum(positive_regret)
            if sum_regret != 0:
                for i, a in enumerate(ins.param):
                    a['strategy'] = positive_regret[i]/sum_regret
            else:
                actionspace_size = len(self.actionspace)
                for a in ins.param:
                    a['strategy'] = 1/actionspace_size
        for ps in self.next:
            ps.update_all_strategy()

    def regret_discount(self, discount):
        '''
        Do regret discount on every infoset. This is only possible in Toy Poker.

        Args:
            discount (float): the amount of discount
        '''
        for ins in self.infoset:
            for a in ins.param:
                a['regret'] *= discount
        for ps in self.next:
            ps.regret_discount(discount)


class Infoset:
    '''
    Infoset represents n cards attaching to a specific Publicstate, it also stores regret and strategy on it.
    '''
    def __init__(self, cards, publicstate):
        '''
        Initialize an Infoset

        Args:
            cards (int): the label of cards this infoset representing.
            publicstate (PublicState): the publicstate this infoset attaching to.
        '''
        self.publicstate = publicstate
        actionspace_size = len(self.publicstate.actionspace)
        self.param = [{'action': action, 'regret': 0, 'strategy': 1/actionspace_size} for action in self.publicstate.actionspace]
        self.cards = cards

    def update_strategy(self, player, env):
        '''
        Update strategy, then sample an action at player's node. Traverse all actions otherwise.
        This behaves horribly on Toy Poker, but is useful on Texas.

        Args:
            player (int): id of the player updating strategy
            env (Env): Env instance for training agent
        '''
        if self.publicstate.done is True:
            env.step_back()
        else:
            state = env.get_state()
            cur_player = state.player_id
            if cur_player == player:
                positive_regret = [a['regret'] for a in self.param]
                for i, r in enumerate(positive_regret):
                    if r < 0:
                        positive_regret[i] = 0
                sum_regret = sum(positive_regret)
                if sum_regret != 0:
                    for i, a in enumerate(self.param):
                        a['strategy'] = positive_regret[i]/sum_regret
                else:
                    actionspace_size = len(self.publicstate.actionspace)
                    for a in self.param:
                        a['strategy'] = 1/actionspace_size
                action = np.random.choice(self.publicstate.actionspace, size=1, p=[a['strategy'] for a in self.param])[0]
                next_state, _ = env.step(action)
                cards = ToyPokerMCCFRAgent.encode_state(next_state)
                done = env.is_over()
                actionspace = next_state.legal_actions
                next_infoset = self.publicstate.search_infoset(cards, action, actionspace, done)
                next_infoset.update_strategy(player, env)
            else:
                for action in self.publicstate.actionspace:
                    next_state, _ = env.step(action)
                    cards = ToyPokerMCCFRAgent.encode_state(next_state)
                    done = env.is_over()
                    actionspace = next_state.legal_actions
                    next_infoset = self.publicstate.search_infoset(cards, action, actionspace, done)
                    next_infoset.update_strategy(player, env)
            env.step_back()

    def update_regret(self, player, reward, neg_p, neg_p_threshold, discount, env):
        '''
        Traverse every action at player's node, and then update regret. Sample an action otherwise.

        Args:
            player (int): id of the player updating regret
            reward (float): reward that player get
            neg_p (Bool): whether use negative pruning
            discount (float): the amount of discount
            env (Env): Env instance for training agent

        Return:
            (float): utility of the action branch
        '''
        if self.publicstate.done is True:
            env.step_back()
            return reward
        else:
            state = env.get_state()
            cur_player = state.player_id
            temp_param = copy.deepcopy(self.param)
            sum_regret = np.sum(np.maximum([a['regret'] for a in self.param], 0))
            if sum_regret != 0:
                for a in temp_param:
                    a['strategy'] = a['regret']/sum_regret
            else:
                actionspace_size = len(self.publicstate.actionspace)
                for a in temp_param:
                    a['strategy'] = 1/actionspace_size
            v_I = 0
            if cur_player == player:
                for i, a in enumerate(temp_param):
                    if self.param[i]['regret'] > neg_p_threshold or neg_p is False:
                        next_state, _ = env.step(a['action'])

                        if env.is_over() is True:
                            reward = env.get_payoffs()[player]
                        else:
                            reward = None

                        cards = ToyPokerMCCFRAgent.encode_state(next_state)
                        done = env.is_over()
                        actionspace = next_state.legal_actions
                        next_infoset = self.publicstate.search_infoset(cards, self.publicstate.actionspace[i], actionspace, done)
                        a['value'] = next_infoset.update_regret(player, reward, neg_p, neg_p_threshold, discount, env)
                        a['explored'] = True
                        v_I += a['value'] * a['strategy']
                    else:
                        a['explored'] = False
                for i, a in enumerate(temp_param):
                    if a['explored'] is True:
                        self.param[i]['regret'] += a['value'] - v_I
                        self.param[i]['regret'] *= discount
            else:
                action = np.random.choice(self.publicstate.actionspace, size=1, p=[a['strategy'] for a in self.param])[0]
                next_state, _ = env.step(action)
                if env.is_over() is True:
                    reward = env.get_payoffs()[player]
                else:
                    reward = 0
                cards = ToyPokerMCCFRAgent.encode_state(next_state)
                done = env.is_over()
                actionspace = next_state.legal_actions
                next_infoset = self.publicstate.search_infoset(cards, action, actionspace, done)
                v_I = next_infoset.update_regret(player, reward, neg_p, neg_p_threshold, discount, env)
            env.step_back()
            return v_I


class ToyPokerMCCFRAgent(Agent):
    '''
    An agent with external sampling MCCFR + negative pruning + discount factor + Tree structure
    '''
    abs_tree = pickle.load(open('poker_env/ToyPoker/data/abs_tree', 'rb'))

    def __init__(self, env, update_interval=100, discount_interval=1000):
        '''
        Initialize an ToyPokerMCCFRAgent

        Args:
            env (Env): Env instance for training agent
            update_interval (int): intervals of policy updating
            discount_interval (int): intervals of regret discount
        '''
        super().__init__(agent_type='MCCFRAgent')
        self.env = env
        self.update_interval = update_interval
        self.discount_interval = discount_interval
        self.neg_p_threshold = -5
        self.neg_p_prob = 0.95
        self.iterations = 0
        self.root = PublicState(0, [], False)

    def train(self):
        '''
        Update strategy for every self.update_interval times.
        Update regrets every time.
        Do regret discount for every self.discount_interval times.
        Do negative pruning by probablity of neg_p_prob.
        '''
        # self.load_agent(20200308)

        self.iterations += 1
        if self.iterations % self.update_interval == 0:
            self.root.update_all_strategy()
        for p in range(self.env.player_num):
            initial_state, _ = self.env.init_game()
            self.root.actionspace = initial_state.legal_actions
            cards = ToyPokerMCCFRAgent.encode_state(initial_state)
            initial_infoset = self.root._search_infoset(cards)
            a = random.random()
            if a < self.neg_p_prob:
                neg_p = True
            else:
                neg_p = False
            if self.iterations % self.discount_interval == 0:
                discount = (self.iterations/self.discount_interval)/((self.iterations/self.discount_interval)+1)
            else:
                discount = 1
            initial_infoset.update_regret(p, 0, neg_p, self.neg_p_threshold, discount, self.env)

            # if self.iterations % self.update_interval == 0:
            #     initial_infoset.update_strategy(p, self.env)

        # if self.iterations % self.discount_interval == 0:
        #    discount = (self.iterations/self.discount_interval)/((self.iterations/self.discount_interval)+1)
        #    self.root.regret_discount(discount)

        # self.save_agent(20200308)

    def save_agent(self, index):
        '''
        Save self.root with pickle. Actually, the whole tree including all infosets is saved.

        Args:
            index (int): index of the saved agent
        '''
        file = open('policy_{}'.format(index), 'wb')
        pickle.dump(self.root, file)
        file.close()
        return

    def load_agent(self, index):
        '''
        Load self.root from a pickle file. Create an empty root if no policy is available.

        Args:
            index (int): index of the loaded agent
        '''
        if os.path.isfile('policy_{}'.format(index)) is False:
            return
        file = open('policy_{}'.format(index), 'rb')
        self.root = pickle.load(file)
        file.close()

    def step(self, state):
        return

    def eval_step(self, state):
        '''
        Choose an action according to current policy.

        Args:
            state (PokerState): the state agent is at when acting

        Returns:
            (str): the chosen action
        '''
        # self.load_agent(20200308)
        strategy = self.search_strategy(state)
        actionspace = state.legal_actions
        # action = np.random.choice(actionspace, size=1, p=strategy)[0]
        action = actionspace[np.argmax(strategy)]
        if strategy == [1/len(actionspace)] * len(actionspace):
            action = 'call'
        return action

    def search_strategy(self, state):
        '''
        Search the strategy corresponding to the state in publicstate tree

        Args:
            state (PokerState): the state agent is at when acting

        Returns:
            (list): the strategy corresponding to the state
        '''
        history = state.history
        cards = ToyPokerMCCFRAgent.encode_state(state)
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
            if ps_flag is False:
                strategy = [1/actionspace_size]*actionspace_size
                break
        if history == []:
            ps_flag = True
        if ps_flag is True:
            for iset in ps.infoset:
                if iset.cards == cards:
                    strategy = [a['strategy'] for a in iset.param]
                    iset_flag = True
                    break
            if iset_flag is False:
                strategy = [1/actionspace_size]*actionspace_size
        if actionspace != ps.actionspace:
            strategy = [1/actionspace_size]*actionspace_size
        return strategy

    @classmethod
    def encode_state(cls, state):
        '''
        Get the label of a state using abs_tree.

        Args:
            state(PokerState): the state that need to be encoded as a label

        Returns:
            (int): label of the state
        '''
        lossless_state = state.get_infoset()
        lossy_state = cls.abs_tree.get_label(lossless_state)
        return lossy_state
