import torch
from torch import nn
from ..vae.variational import variational

class NormConv(nn.Conv1d):

    def __init__(self, in_channels, out_channels, bayesian, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        if bayesian:
            variational(self) # consider calling on weights and biases separately

        else:
            # Initialize weights
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
                nn.utils.weight_norm(self, dim = None)

    # def forward(self, *args, **kwargs):
    #     x = super().forward(*args, **kwargs)
    #     return x
