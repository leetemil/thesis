import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Transformer

from data import IUPAC_SEQ2IDX

class LossTransformer(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.transformer = Transformer(*args, **kwargs)

	def protein_logp(self, xb, *args, **kwargs):
		tgt = torch.zeros_like(xb)
		tgt[:, :-1] = xb[:, 1:]

		src = F.one_hot(xb, self.transformer.d_model).to(torch.float).permute(1, 0, 2)
		tgt = F.one_hot(tgt, self.transformer.d_model).to(torch.float).permute(1, 0, 2)

		pred = self.transformer.forward(src, tgt, *args, **kwargs)

		mask = xb >= IUPAC_SEQ2IDX["A"]
		true = (xb * mask)[:, 1:-1]
		pred = pred.permute(1, 2, 0)
		pred = pred[:, :, :-2]

		# Compare each timestep in cross entropy loss
		loss = F.cross_entropy(pred, true, ignore_index = 0, reduction = "none")
		log_probabilities = -1 * loss.sum(dim = 1)

		return log_probabilities

	def forward(self, xb, *args, **kwargs):
		tgt = torch.zeros_like(xb)
		tgt[:, :-1] = xb[:, 1:]

		src = F.one_hot(xb, self.transformer.d_model).to(torch.float).permute(1, 0, 2)
		tgt = F.one_hot(tgt, self.transformer.d_model).to(torch.float).permute(1, 0, 2)

		pred = self.transformer.forward(src, tgt, *args, **kwargs)

		mask = xb >= IUPAC_SEQ2IDX["A"]
		true = (xb * mask)[:, 1:-1]
		pred = pred.permute(1, 2, 0)
		pred = pred[:, :, :-2]

		# Compare each timestep in cross entropy loss
		loss = F.cross_entropy(pred, true, ignore_index = 0, reduction = "mean")

		with torch.no_grad():
			percent = F.softmax(pred, 1)
			mean_max = percent.max(dim = 1)[0].mean().item()

		metrics_dict = {"mean_max": mean_max}

		return loss, metrics_dict

	def summary(self):
		num_params = sum(p.numel() for p in self.parameters())

		return (f"Transformer summary:\n"
				f"  Heads: {self.transformer.nhead}\n"
				f"  Encoder layers: {len(self.transformer.encoder.layers)}\n"
				f"  Decoder layers: {len(self.transformer.decoder.layers)}\n"
				f"  Feedforward size: {self.transformer.encoder.layers[0].linear1.out_features}\n"
				f"  Dropout: {self.transformer.encoder.layers[0].dropout.p}\n"
				f"  Parameters:  {num_params:,}\n")
