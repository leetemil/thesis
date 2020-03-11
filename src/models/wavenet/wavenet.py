import torch
from torch import nn

class WaveNet(nn.Module):

	def __init__(self, residual_channels, gate_channels, stacks, layers_per_stack, bias = True):
		super().__init__()

		self.residual_channels = residual_channels
		self.gate_channels = gate_channels
		self.stacks = stacks
		self.layers_per_stack = layers_per_stack
		self.bias = bias

		self.first_conv = nn.Conv1d(1, self.residual_channels, kernel_size = 1, bias = self.bias)

		self.dilations = []
		for stack in range(self.stacks):
			for layer in range(self.layers_per_stack):
				self.dilations.append(2**layer)

		self.dilated_conv_stack = WaveNetStack(
			dilations = self.dilations,
			residual_channels = self.residual_channels,
			gate_channels = self.gate_channels,
			kernel_size = 2
		)

		self.last_conv_layers = nn.Sequential(
			nn.ReLU(),
			nn.Conv1d(self.residual_channels, self.residual_channels, kernel_size = 1, bias = self.bias),
			nn.ReLU(),
			nn.Conv1d(self.residual_channels, self.residual_channels, kernel_size = 1, bias = self.bias)
		)

	def forward(self, x):
		x = self.first_conv(x)
		skip_outputs = self.dilated_conv_stack(x)
		x = sum(skip_outputs)
		x = self.last_conv_layers(x)
		return x

class WaveNetLayer(nn.Module):
	def __init__(self, residual_channels, gate_channels, kernel_size, dilation, out_channels = None, causal = True, bias = True):
		super().__init__()

		self.residual_channels = residual_channels
		self.gate_channels = gate_channels
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.causal = causal
		self.bias = bias

		if out_channels is not None:
			self.out_channels = out_channels
		else:
			self.out_channels = self.residual_channels

		if self.causal:
			self.padding = (kernel_size - 1) * dilation
		else:
			self.padding = (kernel_size - 1) // 2 * dilation

		# Conv layer that the input is put through before the non-linear activations
		self.dilated_conv = nn.Conv1d(
			in_channels = self.residual_channels,
			out_channels = self.gate_channels,
			kernel_size = self.kernel_size,
			dilation = self.dilation,
			padding = self.padding,
			bias = self.bias
		)

		# Conv layer for the output, which goes to the next WaveNetLayer
		self.residual_conv = nn.Conv1d(
			in_channels = self.gate_channels // 2,
			out_channels = self.residual_channels,
			kernel_size = 1,
			bias = self.bias
		)

		# Conv layer for the skip connction which goes directly to the output
		self.skip_conv = nn.Conv1d(
			in_channels = self.gate_channels // 2,
			out_channels = self.out_channels,
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
	def __init__(self, dilations, residual_channels, gate_channels, kernel_size, out_channels = None, causal = True, bias = True):
		super().__init__()

		self.dilations = dilations
		self.residual_channels = residual_channels
		self.gate_channels = gate_channels
		self.kernel_size = kernel_size
		self.causal = causal
		self.bias = bias

		if out_channels is not None:
			self.out_channels = out_channels
		else:
			self.out_channels = self.residual_channels

		self.layers = nn.ModuleList()
		for d in dilations:
			layer = WaveNetLayer(
				residual_channels = self.residual_channels,
				gate_channels = self.gate_channels,
				kernel_size = self.kernel_size,
				dilation = d,
				causal = self.causal,
				bias = self.bias
			)
			self.layers.append(layer)

	def forward(self, x):
		skip_outputs = []
		for layer in self.layers:
			x, skip = layer(x)
			skip_outputs.append(skip)

		return skip_outputs
