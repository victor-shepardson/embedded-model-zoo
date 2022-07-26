import torch
from torch import nn

from ..util import register

@register
class simple_conv_1d(nn.Module):

    def __init__(self, 
            input_size=1024, hidden_size=256, depth=1, kernel=9, classes=8):
        super().__init__()

        self.input_shape = (1,input_size) # batch, time
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
        net.append(nn.Conv1d(hidden_size, classes, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        y = self.net(x[:,None])
        return y.mean(-1)#.softmax(-1)

@register
class resnet_1d(nn.Module):

    def __init__(self, 
            input_size=1024, hidden_size=256, depth=3, kernel=3, classes=8):
        super().__init__()

        self.input_shape = (1,input_size) # batch, time
        self.output_shape = (1,classes)

        self.root = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel),
        )

        self.stem = nn.ModuleList()
        for _ in range(depth):
            self.stem.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Dropout1d(0.1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=kernel//2),
                nn.LeakyReLU(0.2),
                nn.Dropout1d(0.1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=kernel//2)
            ))

        self.flower = nn.Conv1d(hidden_size, classes, 1)

    def forward(self, x):
        x = self.root(x[:,None])
        for block in self.stem:
            x = x + block(x)
        return self.flower(x).mean(-1).softmax(-1)
