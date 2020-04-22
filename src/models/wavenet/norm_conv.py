import torch
from torch import nn
from ..vae.variational import variational

class NormConv(nn.Module):

    def __init__(self, in_channels, out_channels, bayesian, activation = None,*args, **kwargs):
        super().__init__()

        self.layer = nn.Conv1d(in_channels, out_channels, *args, **kwargs)
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.ones(out_channels) * 0.1)
        self.activation = activation

        if bayesian:
            variational(self.layer) # consider calling on weights and biases separately

        else:
            # Initialize weights
            nn.init.kaiming_normal_(self.layer.weight, nonlinearity="relu")
            if self.layer.bias is not None:
                nn.init.constant_(self.layer.bias, 0)
                nn.utils.weight_norm(self.layer, dim = None)

    def forward(self, *args, **kwargs):
        x = self.layer(*args, **kwargs)
        x = x.permute(0, 2, 1)
        x = self.gamma * x + self.beta

        if self.activation is not None:
            x = self.activation(x)

        x = x.permute(0, 2, 1)
        return x
