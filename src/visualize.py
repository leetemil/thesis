from collections import defaultdict
from pathlib import Path


from Bio import SeqIO
import torch
import matplotlib.pyplot as plt

from vae import VAE
from protein_data import ProteinDataset, get_protein_dataloader

def get_label_dict():
    file = Path("data/PF00144_full_length_sequences_labeled.fasta")
    seqs = SeqIO.parse(file, "fasta")

    def getKeyLabel(seq):
        s = seq.description.split(" ")
        return s[0], s[2].replace("[", "").replace("]", "")

    return {protein_id: label for protein_id, label in map(getKeyLabel, seqs)}

LABEL_DICT = get_label_dict()

def plot_data(filepath, model, dataset, batch_size = 64, only_good_names = True):
	plt.figure()
	good_names = set([
		"Acidobacteria",
		"Actinobacteria",
		"Bacteroidetes",
		"Chloroflexi",
		"Cyanobacteria",
		"Deinococcus-Thermus",
		"Other",
		"Firmicutes",
		"Fusobacteria",
		"Proteobacteria"
	])

	dataloader = get_protein_dataloader(dataset, batch_size = batch_size, get_names = True)

	error_count = 0
	scatter_dict = defaultdict(lambda: [])
	with torch.no_grad():
		for i, (xb, names) in enumerate(dataloader):
			mean, _ = model.encode(xb)
			mean = mean.cpu()
			for name, point in zip(names, mean):
				try:
					label = LABEL_DICT[name]
					if only_good_names:
						if label in good_names:
							scatter_dict[label].append(point)
					else:
						scatter_dict[label].append(point)
				except KeyError:
					error_count += 1

	for name, points in scatter_dict.items():
		points = torch.stack(points)
		plt.scatter(points[:, 0], points[:, 1], s = 1, label = name)

	if filepath is not None:
		plt.savefig(filepath.with_suffix(".png"))
	else:
		plt.show()
	plt.close()

if __name__ == "__main__":
	device = torch.device("cuda")

	model = VAE([2594, 128, 2]).to(device)
	model.load_state_dict(torch.load("model.torch"))

	plot_data(None, model, "data/PF00144_full.txt", device)
