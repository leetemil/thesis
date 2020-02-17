# Inspired by:
#   https://github.com/pytorch/examples/blob/master/vae/main.py
#   https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/

import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):
    """Variational Auto-Encoder for protein sequences"""

    def __init__(self, layer_sizes, num_tokens, dropout = 0.5):
        super().__init__()

        assert len(layer_sizes) >= 2

        self.layer_sizes = layer_sizes
        self.num_tokens = num_tokens
        self.dropout = dropout

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
        self.encode_mean = nn.Linear(s1, s2)
        self.encode_logvar = nn.Linear(s1, s2)

        # Construct decode layers
        decode_layers = []
        layer_sizes_doubles = [(s1, s2) for s1, s2 in zip(layer_sizes[bottleneck_idx:], layer_sizes[bottleneck_idx + 1:])]
        for s1, s2 in layer_sizes_doubles[:-2]:
            decode_layers.append(nn.Linear(s1, s2))
            decode_layers.append(nn.ReLU())
            decode_layers.append(nn.Dropout(self.dropout))

        # Second-to-last decode layer has sigmoid activation
        s1, s2 = layer_sizes_doubles[-2]
        decode_layers.append(nn.Linear(s1, s2))
        decode_layers.append(nn.Sigmoid())
        decode_layers.append(nn.Dropout(self.dropout))

        # Last decode layer has no activation
        s1, s2 = layer_sizes_doubles[-1]
        decode_layers.append(nn.Linear(s1, s2))

        self.decode_layers = nn.Sequential(*decode_layers)

        #! CRAZY MODE

        E_dim = 40

        # W_tilde_3: E_dim (40) x size of h_2 (2000)
        self.weight = nn.Parameter(torch.randn(E_dim, self.layer_sizes[-2]))

        # C: q (30) x E (40)
        self.dictionary = nn.Parameter(torch.randn(self.num_tokens, E_dim))

        # Lambda tilde : scalar
        self.inverse_temp_param = nn.Parameter(torch.randn(1))

        # S_tilde: h_2/k (500) x L (263)
        self.scale_param = nn.Parameter(torch.randn(self.layer_sizes[-2] // 4, self.layer_sizes[0] // self.num_tokens) * 4 - 12.36)

    def encode(self, x):
        x = F.one_hot(x, self.num_tokens).to(torch.float).flatten(1)

        # Encode x by sending it through all encode layers
        x = self.encode_layers(x)

        mean = self.encode_mean(x)
        logvar = self.encode_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """THE REPARAMETERIZATION IDEA:

        For each training sample

        - take the current learned mean, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mean will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mean : [BATCH_SIZE, ZDIMS] mean matrix
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
        return mean + eps * std

    def insane_decode(self, z):
        breakpoint()

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

    def reconstruct(self, x):
        mean, _ = self.encode(x)
        return self.sample(mean)

    def forward(self, x, weights):
        # Forward pass + loss + metrics
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.insane_decode(z)
        recon_x = self.decode(z)
        loss = self.vae_loss(recon_x, x, mean, logvar)
        weighted_loss = torch.sum(loss * weights)

        # Metrics
        metrics_dict = {}

        # Accuracy
        with torch.no_grad():
            acc = (self.decode(mean).exp().argmax(dim = -1) == x).to(torch.float).mean().item()
            metrics_dict["accuracy"] = acc

        return weighted_loss, metrics_dict

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (f"Variational Auto-Encoder summary:\n"
                f"  Layer sizes: {self.layer_sizes}\n"
                f"  Parameters:  {num_params:,}\n")

    def protein_logp(self, x):
        mean, logvar = self.encode(x)
        kld = 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(1)

        recon_x = self.decode(mean).permute(0, 2, 1)
        logp = F.nll_loss(recon_x, x, reduction = "none").mul(-1).sum(1)
        elbo = logp + kld

        # amino acid probabilities are independent conditioned on z
        return elbo, logp, kld

    def nll_loss(self, recon_x, x):
        # How well do input x and output recon_x agree?
        # CE = F.cross_entropy(recon_x.view(-1, self.num_tokens), x.flatten(), reduction = "sum")
        nll = F.nll_loss(recon_x.permute(0, 2, 1), x, reduction = "none").sum(1)

        # amino acid probabilities are independent conditioned on z
        return nll

    def kld_loss(self, mean, logvar):
        # kld is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # See Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        # Note the negative D_{KL} in appendix B of the paper

        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim = 1)
        return kld

    def vae_loss(self, recon_x, x, mean, logvar):
        nll_loss = self.nll_loss(recon_x, x)
        kld_loss = self.kld_loss(mean, logvar)
        return nll_loss + kld_loss

    #! CRAZY MODE
    def scale_matrix():
        s1 = 1 / (1 + self.scale_param.exp())
        s2 = torch.stack([s1, s1, s1, s1])
        return s2.transpose(0, 1).reshape(-1, self.layer_sizes[0] / self.num_tokens)

    def inverse_temp():
        return F.softplus(self.inverse_temp_param)
