import torch
from torch import nn
from torch.nn import functional as F

from .norm_conv import NormConv
from ..vae.variational import variational, Variational
from ..utils import layer_kld
from data import IUPAC_SEQ2IDX

class WaveNet(nn.Module):

    def __init__(self, input_channels, residual_channels, gate_channels, skip_out_channels, out_channels, stacks, layers_per_stack, total_samples, bias = True, dropout = 0.5, bayesian = True, backwards = False):
        super().__init__()

        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.skip_out_channels = skip_out_channels
        self.out_channels = out_channels
        self.stacks = stacks
        self.layers_per_stack = layers_per_stack
        self.bias = bias
        self.dropout = dropout
        self.bayesian = bayesian
        self.total_samples = total_samples
        self.backwards = backwards

        self.first_conv = NormConv(self.input_channels, self.residual_channels, self.bayesian, kernel_size = 1, bias = self.bias)

        self.dilations = []

        for stack in range(self.stacks):
            for layer in range(self.layers_per_stack):
                self.dilations.append(2**layer)

        self.dilated_conv_stack = WaveNetStack(
            dilations = self.dilations,
            residual_channels = self.residual_channels,
            gate_channels = self.gate_channels,
            dropout = self.dropout,
            skip_out_channels = self.skip_out_channels,
            kernel_size = 2,
            bayesian = self.bayesian # bayesian version
        )

        self.last_conv_layers = nn.Sequential(
            nn.ReLU(inplace = True),
            NormConv(self.skip_out_channels, self.skip_out_channels, self.bayesian, kernel_size = 1, bias = self.bias),
            nn.ReLU(inplace = True),
            NormConv(self.skip_out_channels, self.skip_out_channels, self.bayesian, kernel_size = 1, bias = self.bias),
        )

    def get_predictions(self, xb):
        """
        Returns log-softmax distributions of amino acids over the input sequences.

        Returns:
        Tensor: shape (batch size, num tokens, seq length)
        """
        # one-hot encode and permute to (batch size x channels x length)
        xb_encoded = F.one_hot(xb, self.input_channels).to(torch.float).permute(0, 2, 1)

        pred = self.first_conv(xb_encoded)
        skip_sum = self.dilated_conv_stack(pred)
        pred = self.last_conv_layers(skip_sum)

        return F.log_softmax(pred, dim = 1)

    def protein_logp(self, xb):
        loss, _ = self(xb, loss_reduction = "none")
        log_probabilities = -1 * loss.sum(dim = 1)
        return log_probabilities

    def parameter_kld(self):
        kld = layer_kld(self.first_conv) if isinstance(self.first_conv, NormConv) else 0

        # get loss from last convolution layers
        for layer in self.last_conv_layers:
            if isinstance(layer, NormConv):
                kld += layer_kld(layer)

        # get loss from stack layers
        for layer in self.dilated_conv_stack.layers:
            if isinstance(layer, NormConv):
                kld += layer_kld(layer)

        return kld

    def forward(self, xb, loss_reduction = "mean"):
        if self.backwards:
            lengths = (xb != 0).sum(dim = 1)
            for seq, length in zip(xb, lengths):
                seq[1:length - 1] = reversed(seq[1:length - 1])

        pred = self.get_predictions(xb)

        # Calculate loss
        mask = xb >= IUPAC_SEQ2IDX["A"]
        true = (xb * mask)[:, 1:-1]
        pred = pred[:, :, :-2]

        # Compare each timestep in cross entropy loss
        nll_loss = F.nll_loss(pred, true, ignore_index = 0, reduction = loss_reduction)

        # Metrics
        metrics_dict = {}

        if loss_reduction == "mean":
            metrics_dict["nll_loss"] = nll_loss.item()

        # If we use bayesian parameters and we're not doing predictions, calculate kld loss
        if self.bayesian and loss_reduction == "mean":
            kld_loss = self.parameter_kld() * (1 / self.total_samples) # distribute global loss onto the batch
            metrics_dict["kld_loss"] = kld_loss.item()
            total_loss = nll_loss + kld_loss
        else:
            total_loss = nll_loss

        return total_loss, metrics_dict

    def sample_new_weights(self):
        # rsample first layer
        for hook in self.first_conv._forward_pre_hooks.values():
            if isinstance(hook, Variational):
                hook.rsample_new(self.first_conv)

        # rsample layers in stack
        self.dilated_conv_stack.sample_new_weights()

        # rsample last conv layers
        for layer in self.last_conv_layers:
            for hook in layer._forward_pre_hooks.values():
                if isinstance(hook, Variational):
                    hook.rsample_new(layer)

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())

        return (f"WaveNet summary:\n"
                f"  Input channels: {self.input_channels}\n"
                f"  Residual channels: {self.residual_channels}\n"
                f"  Gate channels: {self.gate_channels}\n"
                f"  Skip output channels: {self.skip_out_channels}\n"
                f"  Output channels: {self.out_channels}\n"
                f"  Stacks: {self.stacks}\n"
                f"  Layers: {self.layers_per_stack} (max. {2**(self.layers_per_stack - 1)} dilation)\n"
                f"  Parameters:  {num_params:,}\n"
                f"  Bayesian: {self.bayesian}\n")

    def save(self, f):
        args_dict = {
            "input_channels": self.input_channels,
            "residual_channels": self.residual_channels,
            "gate_channels": self.gate_channels,
            "skip_out_channels": self.skip_out_channels,
            "out_channels": self.out_channels,
            "stacks": self.stacks,
            "layers_per_stack": self.layers_per_stack,
            "bias": self.bias,
            "dropout": self.dropout,
            "bayesian": self.bayesian,
            "total_samples": self.total_samples,
            "backwards": self.backwards,
        }

        torch.save({
            "name": "WaveNet",
            "state_dict": self.state_dict(),
            "args_dict": args_dict,
        }, f)

class WaveNetLayer(nn.Module):
    def __init__(self, residual_channels, gate_channels, dropout, kernel_size, dilation, skip_out_channels = None, causal = True, bias = True, bayesian = True):
        super().__init__()

        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.bias = bias
        self.bayesian = bayesian

        if skip_out_channels is not None:
            self.skip_out_channels = skip_out_channels
        else:
            self.skip_out_channels = self.residual_channels

        if self.causal:
            self.padding = (self.kernel_size - 1) * self.dilation
        else:
            self.padding = (self.kernel_size - 1) // 2 * self.dilation

        # Conv layer that the input is put through before the non-linear activations
        self.dilated_conv = NormConv(
            in_channels = self.residual_channels,
            out_channels = self.gate_channels * 2,
            bayesian = self.bayesian, # bayesian version
            kernel_size = self.kernel_size,
            dilation = self.dilation,
            bias = self.bias
        )
        # Conv layer for the output, which goes to the next WaveNetLayer
        self.residual_conv = NormConv(
            in_channels = self.gate_channels,
            out_channels = self.residual_channels,
            bayesian = self.bayesian,
            kernel_size = 1,
            bias = self.bias
        )

        # Conv layer for the skip connction which goes directly to the output
        self.skip_conv = NormConv(
            in_channels = self.gate_channels,
            out_channels = self.skip_out_channels,
            bayesian = self.bayesian,
            kernel_size = 1,
            bias = self.bias
        )

    def forward(self, x):
        residual = x

        x = F.pad(x, (self.padding, 0))
        x = self.dilated_conv(x)

        tanh_filters, sigmoid_filters = x.split(x.size(1) // 2, dim=1)
        x = torch.tanh(tanh_filters) * torch.sigmoid(sigmoid_filters)

        output = self.residual_conv(x) + residual
        skip = self.skip_conv(x)

        return output, skip

class WaveNetStack(nn.Module):
    def __init__(self, dilations, residual_channels, gate_channels, dropout, kernel_size, skip_out_channels = None, causal = True, bias = True, bayesian = True):
        super().__init__()

        self.dilations = dilations
        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.causal = causal
        self.bias = bias
        self.bayesian = bayesian

        if skip_out_channels is not None:
            self.skip_out_channels = skip_out_channels
        else:
            self.skip_out_channels = self.residual_channels

        self.layers = nn.ModuleList()
        for d in dilations:
            layer = WaveNetLayer(
                residual_channels = self.residual_channels,
                gate_channels = self.gate_channels,
                dropout = self.dropout,
                kernel_size = self.kernel_size,
                skip_out_channels = self.skip_out_channels,
                dilation = d,
                causal = self.causal,
                bias = self.bias,
                bayesian = self.bayesian # bayesian version
            )
            self.layers.append(layer)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        skip_sum = 0
        for layer in self.layers:
            x, skip = layer(x)
            x = self.dropout(x)
            skip_sum += skip
        return skip_sum

    def sample_new_weights(self):
        for layer in self.layers:
            for hook in layer._forward_pre_hooks.values():
                if isinstance(hook, Variational):
                    hook.rsample_new(layer)
