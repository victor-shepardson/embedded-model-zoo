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