import torch
import torch.nn as nn
import torch.nn.functional as F

class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self, input):
        B, num_cards = input.shape
        x = input.view(-1)

        valid = x.ge(0.5).float()
        x = x.clamp(min=0)

        embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid.unsqueeze(1)  # zero out ’no card ’ embeddings

        # sum across the cards in the hole / board
        return embs.view(B, num_cards, -1).sum(1)


class DeepCFRModel(nn.Module):
    def __init__(self, ncardtypes, nbets, nactions, dim=256):
        super(DeepCFRModel, self).__init__()

        self.card_embeddings = nn.ModuleList([CardEmbedding(dim) for _ in range(ncardtypes)])

        self.card1 = nn.Linear(dim * ncardtypes, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        self.bet1 = nn.Linear(nbets * 2, dim)
        self.bet2 = nn.Linear(dim, dim)

        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)

        self.action_head = nn.Linear(dim, nactions)

    def forward(self, cards, bets):
        card_embs = []
        for embedding, card_group in zip(self.card_embeddings, cards):
            card_embs.append(embedding(card_group))
        card_embs = torch.cat(card_embs, dim=1)

        x = F.relu(self.card1(card_embs))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        # 1. bet branch
        bet_size = bets.clamp(0, 1e6)
        bet_occurred = bets.ge(0.5)
        bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1)
        y = F.relu(self.bet1(bet_feats))
        y = F.relu(self.bet2(y) + y)

        # 3. combined trunk
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)

        z = (z - torch.mean(z)) / torch.std(z)  # (z - mean) / std
        return self.action_head(z)


class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        return

    def forward(self, regret, predict, t):
        y = torch.zeros(predict.shape[0], 1)
        y = torch.pow((regret - predict), 2)
        y = torch.sum(y, dim=1)
        y = t * y
        loss = torch.mean(y)
        return loss