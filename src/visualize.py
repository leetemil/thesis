from collections import defaultdict
from pathlib import Path
import itertools

from Bio import SeqIO
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_data(filepath, model, dataset, batch_size = 64, only_subset_labels = True, show = False, pca_dim = 2):
	fig = plt.figure()
	subset_labels = set([
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
					if only_subset_labels:
						if label in subset_labels:
							scatter_dict[label].append(point)
					else:
						scatter_dict[label].append(point)
				except KeyError:
					error_count += 1

	plt.title(f"Encoded points")

	all_points = torch.stack(list(itertools.chain(*scatter_dict.values())))
	if all_points.size(1) >= 3:
		axis = Axes3D(fig)
		pca = PCA(pca_dim)
		pca.fit(all_points)
		explained_variance = pca.explained_variance_ratio_.sum()
		plt.title(f"PCA of encoded points ({explained_variance:.3f} explained variance)")

	for name, points in scatter_dict.items():
		points = torch.stack(points)
		if points.size(1) == 2:
			plt.scatter(points[:, 0], points[:, 1], s = 1, label = name)
		elif points.size(1) > 2:
			pca_points = pca.transform(points)
			axis.scatter(pca_points[:, 0], pca_points[:, 1], s = 1, label = name)

	if show:
		plt.show()

	if filepath is not None:
		if all_points.size(1) >= 3:
			axis.view_init()
		fig.savefig(filepath)

	plt.close(fig)

if __name__ == "__main__":
	device = torch.device("cuda")

	model = VAE([2594, 128, 2]).to(device)
	model.load_state_dict(torch.load("model.torch"))

	plot_data(None, model, "data/PF00144_full.txt", device)
