from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from vae import VAE
from protein_data import ProteinDataset

def plot_data(filepath, model, dataset, batch_size = 64):
	dataloader = DataLoader(dataset, batch_size = batch_size)

	with torch.no_grad():
		for xb in dataloader:
			mean, _ = model.encode(xb)
			mean = mean.cpu()
			plt.scatter(mean[:, 0], mean[:, 1], s = 1, c = "blue")

	if filepath is not None:
		plt.savefig(filepath.with_suffix(".pdf"))
	else:
		plt.show()

if __name__ == "__main__":
	device = torch.device("cuda")

	model = VAE([2594, 128, 2]).to(device)
	model.load_state_dict(torch.load("model.torch"))

	data = ProteinDataset("data/PF00144_full.txt", device = device)

	plot_data(None, model, data)
