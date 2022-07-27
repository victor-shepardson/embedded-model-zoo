import torch
from torch import nn

from ..util import register

@register
class single_mm(nn.Module):

    def __init__(self, 
            input_size=1024):
        super().__init__()

        self.input_shape = (1, input_size,)
        self.output_shape = (1, input_size,)

        self.net = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        return self.net(x)

@register
class basic_mlp(nn.Module):

    def __init__(self, 
            input_size=1024):
        super().__init__()

        self.input_shape = (1, input_size,)
        self.output_shape = (1, input_size,)

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size),
        )

    def forward(self, x):
        return self.net(x)

@register
class siren_mlp(nn.Module):

    def __init__(self, 
            input_size=1024, hidden_size=256):
        super().__init__()

        self.input_shape = (1, input_size,)
        self.output_shape = (input_size, 1,)

        self.root = nn.Linear(1, hidden_size)
        self.stem = nn.ModuleList((
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1))
        )

    def forward(self, x):
        x = x.reshape(*self.input_shape[::-1])
        x = self.root(x)
        for layer in self.stem:
            x = layer(x.sin())

        return x