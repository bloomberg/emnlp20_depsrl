import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    def __init__(self, p=0.5):
        super(SharedDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        info = f"p={self.p}"
        return info

    def forward(self, x):
        if self.training:
            mask = self.get_mask(x[:, 0], self.p)
            x *= mask.unsqueeze(1)

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask


class WordDropout(nn.Module):
    def __init__(self, p=0.0):
        super(WordDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        info = f"p={self.p}"
        return info

    def forward(self, x):
        if self.training:
            mask = self.get_mask(x, self.p)
            x *= mask.unsqueeze(-1)
        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape[:-1], 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask
