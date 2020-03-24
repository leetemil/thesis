import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Transformer

from data import IUPAC_SEQ2IDX

class LossTransformer(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.transformer = Transformer(*args, **kwargs)

	def protein_logp(self, xb):
		xb_src = xb[:, 1:]

		tgt = torch.zeros_like(xb_src)
		tgt[:, 0] = IUPAC_SEQ2IDX["<cls>"]
		tgt[:, 1:] = xb_src[:, :-1]
		tgt[tgt == IUPAC_SEQ2IDX["<sep>"]] = 0

		src = F.one_hot(xb_src, self.transformer.d_model).to(torch.float).permute(1, 0, 2)
		tgt = F.one_hot(tgt, self.transformer.d_model).to(torch.float).permute(1, 0, 2)

		src_mask = self.generate_subsequent_mask(src.size(0), device = src.device)
		tgt_mask = self.generate_subsequent_mask(tgt.size(0), device = src.device)
		memory_mask = self.generate_subsequent_mask(tgt.size(0), src.size(0), device = src.device)

		pred = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)
		pred = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)
		pred = pred.permute(1, 2, 0)

		# Compare each timestep in cross entropy loss
		loss = F.cross_entropy(pred, xb_src, ignore_index = 0, reduction = "none")
		log_probabilities = -1 * loss.sum(dim = 1)

		return log_probabilities

	def generate_subsequent_mask(self, *sizes, device = None):
		sizes = list(sizes)
		if len(sizes) == 1:
			sizes = sizes * 2

		sizes.reverse()
		mask = (torch.triu(torch.ones(*sizes, device = device)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def forward(self, xb):
		xb_src = xb[:, 1:]

		tgt = torch.zeros_like(xb_src)
		tgt[:, 0] = IUPAC_SEQ2IDX["<cls>"]
		tgt[:, 1:] = xb_src[:, :-1]
		tgt[tgt == IUPAC_SEQ2IDX["<sep>"]] = 0

		src = F.one_hot(xb_src, self.transformer.d_model).to(torch.float).permute(1, 0, 2)
		tgt = F.one_hot(tgt, self.transformer.d_model).to(torch.float).permute(1, 0, 2)

		src_mask = self.generate_subsequent_mask(src.size(0), device = src.device)
		tgt_mask = self.generate_subsequent_mask(tgt.size(0), device = src.device)
		memory_mask = self.generate_subsequent_mask(tgt.size(0), src.size(0), device = src.device)

		pred = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)
		pred = pred.permute(1, 2, 0)

		# Compare each timestep in cross entropy loss
		loss = F.cross_entropy(pred, xb_src, ignore_index = 0, reduction = "mean")

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
