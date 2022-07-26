import torch
from torch import nn

from ..util import register

@register
class transformer_block(nn.Module):

    def __init__(self, 
            input_size=1024, hidden_size=512, input_chan=8, heads=4):
        super().__init__()

        self.input_shape = (1,input_size) # batch, (timexchannel)
        self.output_shape = (1,hidden_size)

        self.reshape_to = (1, input_size//input_chan, input_chan)

        self.root = nn.Linear(input_chan, hidden_size)

        self.stem = nn.TransformerEncoderLayer(hidden_size, heads, hidden_size)

    def forward(self, x):
        x = x.reshape(*self.reshape_to)
        x = self.root(x)
        x = self.stem(x)
        return x.mean(-2)
