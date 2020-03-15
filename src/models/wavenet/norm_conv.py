import torch
from torch import nn

class NormConv(nn.Conv1d):

	def __init__(self, in_channels, out_channels, dropout, *args, **kwargs):
		super().__init__(in_channels, out_channels, *args, **kwargs)
		nn.utils.weight_norm(self)

		self.layer_norm = nn.LayerNorm(out_channels)
		self.dropout = nn.Dropout(dropout)

	def forward(self, *args, **kwargs):
		x = super().forward(*args, **kwargs)
		x = x.permute(0, 2, 1)
		x = self.layer_norm(x)
		x = x.permute(0, 2, 1)
		x = self.dropout(x)
		return x
