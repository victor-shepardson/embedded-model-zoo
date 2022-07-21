import torch
from torch import nn

from ..util import export

@export
class simple_conv_1d(nn.Module):

    def __init__(self, 
            input_size=1024, hidden_size=256, depth=1, kernel=9, classes=8):
        super().__init__()

        self.input_shape = (1,1,input_size) # batch, channel, time
        self.output_shape = (1,classes)

        net = [
            nn.Conv1d(1, hidden_size, kernel),
            nn.AvgPool1d(2),
            nn.ReLU()
        ]
        for _ in range(depth):
            net.extend([
                nn.Conv1d(hidden_size, hidden_size, kernel),
                nn.AvgPool1d(2),
                nn.ReLU()
            ])
        net.append(nn.Conv1d(hidden_size, 8, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        y = self.net(x)
        return y.mean(-1).softmax(-1)

