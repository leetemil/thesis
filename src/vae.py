# Inspired by:
#   https://github.com/pytorch/examples/blob/master/vae/main.py
#   https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/

import torch
import torch.nn as nn
from torch.nn import functional as F

from protein_data import NUM_TOKENS

class VAE(nn.Module):
    """Variational Auto-Encoder for protein sequences"""

    def __init__(self, layer_sizes):
        super().__init__()

        assert len(layer_sizes) >= 2

        self.layer_sizes = layer_sizes
        self.layer_sizes[0] *= NUM_TOKENS

        # Construct encode layers except last ones
        self.encode_layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 2):
            lz1 = self.layer_sizes[i]
            lz2 = self.layer_sizes[i + 1]
            layer = nn.Linear(lz1, lz2)
            self.encode_layers.append(layer)

        # Last two layers to get to bottleneck size
        self.encode_mu = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])
        self.encode_logvar = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])

        # Construct decode layers
        self.decode_layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            lz1 = self.layer_sizes[-i - 1]
            lz2 = self.layer_sizes[-i - 2]
            layer = nn.Linear(lz1, lz2)
            self.decode_layers.append(layer)

    def encode(self, x):
        x = F.one_hot(x, NUM_TOKENS).to(torch.float).flatten(1)

        # Encode x by sending it through all encode layers
        for layer in self.encode_layers:
            x = F.relu(layer(x))

        mu = self.encode_mu(x)
        logvar = self.encode_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """THE REPARAMETERIZATION IDEA:

        For each training sample

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [BATCH_SIZE, ZDIMS] mean matrix
        logvar : [BATCH_SIZE, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.

        """

        # Multiply log variance with 0.5, then exponent yielding the standard deviation
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # Sample by multiplying the unit Gaussian with standard deviation and adding the mean
        return mu + eps * std

    def decode(self, z):
        # Send z through all decode layers
        for layer in self.decode_layers[:-1]:
            z = F.relu(layer(z))

        z = torch.sigmoid(self.decode_layers[-1](z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (f"Variational Auto-Encoder summary:\n"
                f"  Layer sizes: {self.layer_sizes}\n"
                f"  Parameters:  {num_params:,}\n")

    @staticmethod
    def CE_loss(recon_x, x):
        # How well do input x and output recon_x agree?
        CE = F.cross_entropy(recon_x.view(-1, NUM_TOKENS), x.flatten(), reduction = "sum")
        return CE

    @staticmethod
    def KLD_loss(mu, logvar):
        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # See Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    @staticmethod
    def vae_loss(recon_x, x, mu, logvar):
        return VAE.CE_loss(recon_x, x) + VAE.KLD_loss(mu, logvar)