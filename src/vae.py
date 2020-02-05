# Inspired by:
#   https://github.com/pytorch/examples/blob/master/vae/main.py
#   https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/

import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):
    """Variational Auto-Encoder for protein sequences"""

    def __init__(self, layer_sizes, num_tokens):
        super().__init__()

        assert len(layer_sizes) >= 2

        self.layer_sizes = layer_sizes
        self.num_tokens = num_tokens

        bottleneck_idx = layer_sizes.index(min(layer_sizes))

        # Construct encode layers except last ones
        encode_layers = []
        layer_sizes_doubles = [(s1, s2) for s1, s2 in zip(layer_sizes[:bottleneck_idx], layer_sizes[1:])]
        for s1, s2 in layer_sizes_doubles[:-1]:
            encode_layers.append(nn.Linear(s1, s2))
            encode_layers.append(nn.ReLU())
        self.encode_layers = nn.Sequential(*encode_layers)

        # Last two layers to get to bottleneck size
        s1, s2 = layer_sizes_doubles[-1]
        self.encode_mu = nn.Linear(s1, s2)
        self.encode_logvar = nn.Linear(s1, s2)

        # Construct decode layers
        decode_layers = []
        layer_sizes_doubles = [(s1, s2) for s1, s2 in zip(layer_sizes[bottleneck_idx:], layer_sizes[bottleneck_idx + 1:])]
        for s1, s2 in layer_sizes_doubles[:-1]:
            decode_layers.append(nn.Linear(s1, s2))
            decode_layers.append(nn.ReLU())

        # Last decode layer has no activation
        s1, s2 = layer_sizes_doubles[-1]
        decode_layers.append(nn.Linear(s1, s2))

        self.decode_layers = nn.Sequential(*decode_layers)

    def encode(self, x):
        x = F.one_hot(x, self.num_tokens).to(torch.float).flatten(1)

        # Encode x by sending it through all encode layers
        x = self.encode_layers(x)

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
        z = self.decode_layers(z)
        z = z.view(z.size(0), -1, self.num_tokens)
        z = torch.log_softmax(z, dim = -1)
        return z

    def sample(self, z):
        z = self.decode(z)
        sample = z.exp().argmax(dim = -1)
        return sample

    def sample_random(self, batch_size = 1):
        z = torch.randn(batch_size, self.layer_sizes[-1])
        return self.sample(z)

    def forward(self, x):
        # Forward pass + loss + metrics
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        loss = self.vae_loss(recon_x, x, mu, logvar)

        # Metrics
        metrics_dict = {}

        # Accuracy
        with torch.no_grad():
            acc = (self.decode(mu).exp().argmax(dim = -1) == x).to(torch.float).mean().item()
            metrics_dict["train_accuracy"] = acc

        return loss, metrics_dict

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (f"Variational Auto-Encoder summary:\n"
                f"  Layer sizes: {self.layer_sizes}\n"
                f"  Parameters:  {num_params:,}\n")

    def protein_log_probability(self, x):
        mu, _ = self.encode(x)
        recon_x = self.decode(mu).permute(0, 2, 1)
        log_probability = -1 * F.nll_loss(recon_x, x, reduction = "none")

        # amino acid probabilities are independent conditioned on z
        return log_probability.sum(1)

    def NLL_loss(self, recon_x, x):
        # How well do input x and output recon_x agree?
        # CE = F.cross_entropy(recon_x.view(-1, self.num_tokens), x.flatten(), reduction = "sum")
        nll = F.nll_loss(recon_x.view(-1, self.num_tokens), x.flatten(), reduction = "sum")
        return nll

    def KLD_loss(self, mu, logvar):
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

    def vae_loss(self, recon_x, x, mu, logvar):
        nll_loss = self.NLL_loss(recon_x, x)
        kld_loss = self.KLD_loss(mu, logvar)
        return nll_loss + kld_loss
