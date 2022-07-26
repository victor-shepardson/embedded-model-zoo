import torch
from torch import nn

from ..util import register

@register
class log_spectrum(nn.Module):

    def __init__(self, input_size=1024):
        super().__init__()

        self.input_shape = (1,input_size) # batch, channel, time
        self.output_shape = (1,input_size//2+1)
        self.fft_size = input_size
        # self.window = self.register_buffer('window', torch.hann_window(input_size))

    def forward(self, x):
        size = self.input_shape[-1]
        w = torch.ones(size) # torch.hann_window(size) 
        X = torch.fft.rfft(x * w, self.fft_size, -1, norm='ortho')
        return (X*X + 1e-7).log()

