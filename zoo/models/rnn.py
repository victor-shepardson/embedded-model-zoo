import torch
from torch import nn

from ..util import register

@register
class simple_rnn(nn.Module):

    def __init__(self, 
            input_size=1024, hidden_size=256, depth=1):
        super().__init__()

        self.input_shape = (1,input_size) # batch, time
        self.output_shape = (depth,1,hidden_size)

        self.register_buffer('h', torch.randn(self.output_shape))

        self.net = nn.RNN(1, hidden_size, depth)

    def forward(self, x):
        _, x = self.net(x.t()[...,None], self.h)
        return x

# tried this version with no transpose to get TOSA to work, but it still
# chokes on an unsqueeze op for some reason
# @register
# class simple_rnn(nn.Module):

#     def __init__(self, 
#             input_size=1024, hidden_size=128, depth=1):
#         super().__init__()

#         self.input_shape = (input_size,1,1) # batch, time
#         self.output_shape = (depth,1,hidden_size)

#         self.register_buffer('h', torch.randn(self.output_shape))

#         self.net = nn.RNN(1, hidden_size, depth)

#     def forward(self, x):
#         _, h = self.net(x, self.h)
#         return h