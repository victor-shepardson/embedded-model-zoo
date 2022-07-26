import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions

from ..util import register

@register
class variational_encoder(nn.Module):

    def __init__(self, input_size=1024, latent_dims=128):
        super().__init__()

        self.input_shape = (1, input_size) # batch
        self.output_shape = (1, latent_dims)

        self.linear1 = nn.Linear(input_size, latent_dims)
        self.linear2 = nn.Linear(input_size, latent_dims)
        self.kl_divergence = 0

    def forward(self, x):
        mu = F.relu(self.linear1(x))
        sigma = torch.exp(self.linear2(x))
        z = mu + sigma*torch.randn(mu.shape)
        self.kl_divergence = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

# torch.distributions does not appear to be supported by torchscript just yet.
# @register
# class variational_encoder_distributions(nn.Module):
#     """
#     Vartiational Encoder using torch.distributions sampling and KL divergence
#     """
#
#     def __init__(self, input_size=1024, latent_dims=128):
#         super().__init__()
#
#         self.input_shape = (1, input_size) # batch
#         self.output_shape = (1, latent_dims)
#
#         self.linear1 = nn.Linear(input_size, latent_dims)
#         self.linear2 = nn.Linear(input_size, latent_dims)
#         self.kl_divergence = 0
#
#     def forward(self, x):
#         mu = F.relu(self.linear1(x))
#         sigma = torch.exp(self.linear2(x))
#         N = torch.distributions.Normal(0, 1)
#         z = mu + sigma*N.sample(mu.shape)
#         # TODO experiment with torch.distributions.kl.kl_divergence or torch.nn.KLDivLoss
#         self.kl_divergence = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
#         return z