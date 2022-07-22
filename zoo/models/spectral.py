import torch
from torch import nn

from ..util import register

@register
class log_spectrum(nn.Module):

    def __init__(self, input_size=1024):
        super().__init__()

        self.input_shape = (1,input_size) # batch, channel, time
        self.output_shape = (1,input_size//2+1)

        # self.window = self.register_buffer('window', torch.hann_window(input_size))

    def forward(self, x):
        size = self.input_shape[-1]
        X = torch.fft.rfft(x * torch.hann_window(size), size, -1, norm='ortho')
        return (X*X + 1e-7).log()

