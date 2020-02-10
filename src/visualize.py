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

def get_pfam_label_dict():
    file = Path("data/PF00144_full_length_sequences_labeled.fasta")
    seqs = SeqIO.parse(file, "fasta")

    def getKeyLabel(seq):
        s = seq.description.split(" ")
        return s[0], s[2].replace("[", "").replace("]", "")

    return {protein_id: label for protein_id, label in map(getKeyLabel, seqs)}

PFAM_LABEL_DICT = get_pfam_label_dict()

def get_BLAT_label_dict():
    file = Path("data/alignments/BLAT_ECOLX_1_b0.5_LABELS.a2m")

    with open(file, "r") as f:
        lines = f.readlines()

    return dict([line[:-1].split(": ") for line in lines])

BLAT_LABEL_DICT = get_BLAT_label_dict()

def plot_data(name, figure_type, model, dataset, batch_size = 64, only_subset_labels = True, show = False, pca_dim = 2):
    pca_fig = plt.figure()
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

    dataloader = get_protein_dataloader(dataset, batch_size = batch_size, get_ids = True)

    error_count = 0
    scatter_dict = defaultdict(lambda: [])
    with torch.no_grad():
        for xb, ids in dataloader:
            mean, _ = model.encode(xb)
            mean = mean.cpu()
            for point, ID in zip(mean, ids):
                try:
                    label = BLAT_LABEL_DICT[ID]
                    if only_subset_labels:
                        if label in subset_labels:
                            scatter_dict[label].append(point)
                    else:
                        scatter_dict[label].append(point)
                except KeyError:
                    error_count += 1

    plt.title(f"Encoded points")

    all_points = torch.stack(list(itertools.chain(*scatter_dict.values())))
    if all_points.size(1) > 2:
        if pca_dim == 3:
            axis = Axes3D(pca_fig)
        pca = PCA(pca_dim)
        pca.fit(all_points)
        explained_variance = pca.explained_variance_ratio_.sum()
        plt.title(f"PCA of encoded points ({explained_variance:.3f} explained variance)")

        # Make explained variance figure
        variance_fig = plt.figure()
        plt.title("Explained variance of principal components")
        plt.xlabel("Principal components")
        plt.ylabel("Ratio of variance")
        plt.ylim((0, 1))

        pca_highdim = PCA(all_points.size(1))
        pca_highdim.fit(all_points)
        explained_variances = pca_highdim.explained_variance_ratio_
        plt.plot(range(len(explained_variances)), explained_variances, label = "Explained variance ratio")
        plt.legend()

        if name is not None:
            variance_fig.savefig(name.with_name("variance_" + name.name).with_suffix(figure_type))
        plt.close(variance_fig)
        plt.figure(pca_fig.number)

    for label, points in scatter_dict.items():
        points = torch.stack(points)
        if points.size(1) == 2:
            plt.scatter(points[:, 0], points[:, 1], s = 1, label = label)
        elif points.size(1) > 2:
            pca_points = pca.transform(points)
            if pca_dim == 2:
                plt.scatter(pca_points[:, 0], pca_points[:, 1], s = 1, label = label)
            elif pca_dim == 3:
                axis.scatter(pca_points[:, 0], pca_points[:, 1], pca_points[:, 2], s = 1, label = label)

    breakpoint()

    if name is not None:
        pca_fig.savefig(name.with_suffix(figure_type))

    if show:
        plt.show()

    plt.close(pca_fig)

if __name__ == "__main__":
    device = torch.device("cuda")

    model = VAE([2594, 128, 2]).to(device)
    model.load_state_dict(torch.load("model.torch"))

    plot_data(None, model, "data/PF00144_full.txt", device)
