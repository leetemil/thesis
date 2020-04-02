import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

from data import IUPAC_SEQ2IDX

class LossTransformer(nn.Module):
	def __init__(self, dropout = 0.1, *args, **kwargs):
		super().__init__()
		self.transformer = Transformer(dropout = dropout, *args, **kwargs)
		self.pos_encoder = PositionalEncoding(30, dropout, 300)

	def prediction(self, xb_src):
		tgt = torch.zeros_like(xb_src)
		tgt[:, 0] = IUPAC_SEQ2IDX["<cls>"]
		tgt[:, 1:] = xb_src[:, :-1]
		tgt[tgt == IUPAC_SEQ2IDX["<sep>"]] = 0

		src = F.one_hot(xb_src, self.transformer.d_model).to(torch.float).permute(1, 0, 2)
		tgt = F.one_hot(tgt, self.transformer.d_model).to(torch.float).permute(1, 0, 2)

		src = self.pos_encoder(src)
		tgt = self.pos_encoder(tgt)

		# src_mask = self.generate_subsequent_mask(src.size(0), device = src.device)
		tgt_mask = self.generate_subsequent_mask(tgt.size(0), device = src.device)

		pred = self.transformer(src, tgt, tgt_mask = tgt_mask)
		pred = pred.permute(1, 2, 0)
		return pred

	def protein_logp(self, xb):
		xb_src = xb[:, 1:]
		pred = self.prediction(xb_src)

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
		pred = self.prediction(xb_src)

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

class TransformerModel(nn.Module):

	def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
		super().__init__()

		self.model_type = 'Transformer'
		self.ninp = ninp
		self.src_mask = None

		self.pos_encoder = PositionalEncoding(ninp, dropout)

		encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

		self.encoder = nn.Embedding(ntoken, ninp)
		self.decoder = nn.Linear(ninp, ntoken)

		self.init_weights()

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, src):
		if self.src_mask is None or self.src_mask.size(0) != len(src):
			device = src.device
			mask = self._generate_square_subsequent_mask(len(src)).to(device)
			self.src_mask = mask

		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, self.src_mask)
		output = self.decoder(output)
		return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
