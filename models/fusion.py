import torch
from torch import nn


class AddFusion(nn.Module):
    def forward(self, vectors):
        return torch.stack(vectors, dim=1).sum(dim=1)


class ConcatFusion(nn.Module):
    def __init__(self, stages):
        super().__init__()
        self.reduce = nn.Linear(64 * stages, 64)

    def forward(self, vectors):
        return self.reduce(torch.cat(vectors, dim=1))


class WeightedFusion(nn.Module):
    def __init__(self, stages, uniform=False):
        super().__init__()
        if uniform:
            self.register_buffer("scores", torch.zeros(stages))
        else:
            self.scores = nn.Parameter(torch.zeros(stages))

    def weights(self):
        return self.scores.softmax(dim=0)

    def forward(self, vectors):
        return (torch.stack(vectors, dim=1) * self.weights()[None, :, None]).sum(dim=1)


class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared bottleneck applied to each stage. Normalize across stages per channel.
        self.gate = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64))

    def weights(self, stacked):
        return self.gate(stacked).softmax(dim=1)

    def forward(self, vectors):
        stacked = torch.stack(vectors, dim=1)
        return (stacked * self.weights(stacked)).sum(dim=1)
