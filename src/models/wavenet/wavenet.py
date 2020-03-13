import torch
from torch import nn

from .norm_conv import NormConv

class WaveNet(nn.Module):

	def __init__(self, input_channels, residual_channels, gate_channels, skip_out_channels, out_channels, stacks, layers_per_stack, bias = True):
		super().__init__()

		self.input_channels = input_channels
		self.residual_channels = residual_channels
		self.gate_channels = gate_channels
		self.skip_out_channels = skip_out_channels
		self.out_channels = out_channels
		self.stacks = stacks
		self.layers_per_stack = layers_per_stack
		self.bias = bias

		self.first_conv = NormConv(input_channels, self.residual_channels, kernel_size = 1, bias = self.bias)

		self.dilations = []
		for stack in range(self.stacks):
			for layer in range(self.layers_per_stack):
				self.dilations.append(2**layer)

		self.dilated_conv_stack = WaveNetStack(
			dilations = self.dilations,
			residual_channels = self.residual_channels,
			gate_channels = self.gate_channels,
			skip_out_channels = self.skip_out_channels,
			kernel_size = 2
		)

		self.last_conv_layers = nn.Sequential(
			nn.ReLU(),
			NormConv(self.skip_out_channels, self.skip_out_channels, kernel_size = 1, bias = self.bias),
			nn.ReLU(),
			NormConv(self.skip_out_channels, self.out_channels, kernel_size = 1, bias = self.bias)
		)

	def forward(self, x):
		x = self.first_conv(x)
		skip_sum = self.dilated_conv_stack(x)
		x = self.last_conv_layers(skip_sum)
		return x

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

class WaveNetLayer(nn.Module):
	def __init__(self, residual_channels, gate_channels, kernel_size, dilation, skip_out_channels = None, causal = True, bias = True):
		super().__init__()

		self.residual_channels = residual_channels
		self.gate_channels = gate_channels
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.causal = causal
		self.bias = bias

		if skip_out_channels is not None:
			self.skip_out_channels = skip_out_channels
		else:
			self.skip_out_channels = self.residual_channels

		if self.causal:
			self.padding = (kernel_size - 1) * dilation
		else:
			self.padding = (kernel_size - 1) // 2 * dilation

		# Conv layer that the input is put through before the non-linear activations
		self.dilated_conv = NormConv(
			in_channels = self.residual_channels,
			out_channels = self.gate_channels * 2,
			kernel_size = self.kernel_size,
			dilation = self.dilation,
			padding = self.padding,
			bias = self.bias
		)

		# Conv layer for the output, which goes to the next WaveNetLayer
		self.residual_conv = NormConv(
			in_channels = self.gate_channels,
			out_channels = self.residual_channels,
			kernel_size = 1,
			bias = self.bias
		)

		# Conv layer for the skip connction which goes directly to the output
		self.skip_conv = NormConv(
			in_channels = self.gate_channels,
			out_channels = self.skip_out_channels,
			kernel_size = 1,
			bias = self.bias
		)

	def forward(self, x):
		residual = x

		x = self.dilated_conv(x)
		if self.causal:
			x = x[:, :, :residual.size(-1)]

		tanh_filters, sigmoid_filters = x.split(x.size(1) // 2, dim=1)
		x = torch.tanh(tanh_filters) * torch.sigmoid(sigmoid_filters)

		output = self.residual_conv(x) + residual
		skip = self.skip_conv(x)

		return output, skip

class WaveNetStack(nn.Module):
	def __init__(self, dilations, residual_channels, gate_channels, kernel_size, skip_out_channels = None, causal = True, bias = True):
		super().__init__()

		self.dilations = dilations
		self.residual_channels = residual_channels
		self.gate_channels = gate_channels
		self.kernel_size = kernel_size
		self.causal = causal
		self.bias = bias

		if skip_out_channels is not None:
			self.skip_out_channels = skip_out_channels
		else:
			self.skip_out_channels = self.residual_channels

		self.layers = nn.ModuleList()
		for d in dilations:
			layer = WaveNetLayer(
				residual_channels = self.residual_channels,
				gate_channels = self.gate_channels,
				kernel_size = self.kernel_size,
				skip_out_channels = self.skip_out_channels,
				dilation = d,
				causal = self.causal,
				bias = self.bias
			)
			self.layers.append(layer)

	def forward(self, x):
		skip_sum = 0
		for layer in self.layers:
			x, skip = layer(x)
			skip_sum += skip
		return skip_sum
