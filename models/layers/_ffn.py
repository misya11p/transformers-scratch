import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, activation):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x):
        out = self.fc2(self.activation(self.fc1(x)))
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_model, d_ff)
        self.fc3 = nn.Linear(d_ff, d_model)
        self.swish = nn.SiLU()

    def forward(self, x):
        out = self.fc3(self.swish(self.fc2(x)) * self.fc1(x))
        return out
