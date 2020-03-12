import numpy as np
import pandas as pd
import random
import torch

class memory(object):
    def __init__(self, memo=[], count=0, capacity=10000):
        self.memo = memo
        self.count = count
        self.capacity = capacity

    # reservoir sampling 更新 memory
    def update(self, a):
        N = self.count
        if N < self.capacity:
            self.memo.append(a)
            self.count += 1
        else:
            random1 = random.randint(0, N - 1)
            if random1 < self.capacity:
                self.memo[random1] = a
            self.count += 1

    def get_sample(self, minibatch):
        if len(self.memo) <= minibatch:
            training_sample = self.memo
        else:
            training_sample = random.sample(self.memo, minibatch)

        training_sample = pd.DataFrame(training_sample)

        training_handcard = training_sample[:][0]
        training_handcard = np.array(training_handcard)
        handcard_sample = training_handcard[0]
        for i in range(1, len(training_handcard)):
            handcard_sample = torch.cat((handcard_sample, training_handcard[i]), dim=0)

        training_boardcard = training_sample[:][1]
        training_boardcard = np.array(training_boardcard)
        boardcard_sample = training_boardcard[0]
        for i in range(1, len(training_boardcard)):
            boardcard_sample = torch.cat((boardcard_sample, training_boardcard[i]), dim=0)

        training_bet_history = training_sample[:][2]
        training_bet_history = np.array(training_bet_history)
        bet_history_sample = training_bet_history[0]
        for i in range(1, len(training_bet_history)):
            bet_history_sample = torch.cat((bet_history_sample, training_bet_history[i]), dim=0)

        training_t = training_sample[:][3]
        training_t = np.array(training_t)
        t_sample = torch.from_numpy(training_t)

        training_y = training_sample[:][4]
        training_y = np.array(training_y)
        y_sample = training_y[0].float()
        for i in range(1, len(training_y)):
            y_sample = torch.cat((y_sample, training_y[i].float()), dim=0)

        return handcard_sample, boardcard_sample, bet_history_sample, t_sample, y_sample