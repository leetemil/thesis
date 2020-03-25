import torch
from torch import nn

class NormConv(nn.Conv1d):

	def __init__(self, in_channels, out_channels, *args, **kwargs):
		super().__init__(in_channels, out_channels, *args, **kwargs)

		# Initialize weights
		nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
		if self.bias is not None:
			nn.init.constant_(self.bias, 0)

		nn.utils.weight_norm(self)
		self.layer_norm = nn.LayerNorm(out_channels)

	def forward(self, *args, **kwargs):
		x = super().forward(*args, **kwargs)
		x = x.permute(0, 2, 1)
		x = self.layer_norm(x)
		x = x.permute(0, 2, 1)
		return x
