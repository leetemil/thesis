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

def get_BLAT_label_dict(file):
    with open(file, "r") as f:
        lines = f.readlines()

    return dict([line[:-1].split(": ") for line in lines])

BLAT_LABEL_DICT = get_BLAT_label_dict(Path("data/alignments/BLAT_ECOLX_1_b0.5_LABELS.a2m"))

BLAT_HMMERBIT_LABEL_DICT = get_BLAT_label_dict(Path("data/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_LABELS.a2m"))

def plot_data(name, figure_type, model, dataset, rho, batch_size = 64, only_subset_labels = True, show = False, pca_dim = 2):
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

    dataloader = get_protein_dataloader(dataset, batch_size = batch_size, get_seqs = True)

    scatter_dict = defaultdict(lambda: [])
    with torch.no_grad():
        for xb, weights, neff, seqs in dataloader:
            ids = [s.id for s in seqs]
            dist = model.encode(xb)
            mean = dist.mean.cpu()
            for point, ID in zip(mean, ids):
                try:
                    label = BLAT_HMMERBIT_LABEL_DICT[ID]
                    if only_subset_labels:
                        if label in subset_labels:
                            scatter_dict[label].append(point)
                    else:
                        scatter_dict[label].append(point)
                except KeyError:
                    if not only_subset_labels:
                        scatter_dict["Others"].append(point)

    plt.title(f"Encoded points")

    all_points_list = list(itertools.chain(*scatter_dict.values()))
    all_points = torch.stack(all_points_list) if len(all_points_list) > 0 else torch.zeros(0, 0)
    if all_points.size(1) > 2:
        if pca_dim == 3:
            axis = Axes3D(pca_fig)
        pca = PCA(pca_dim)
        pca.fit(all_points)
        explained_variance = pca.explained_variance_ratio_.sum()
        plt.title(f"PCA of encoded points ({explained_variance:.3f} explained variance). Spearman's $\\rho$: {rho:.3f}")

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
            variance_fig.savefig(name.with_name("explained_variance").with_suffix(figure_type))
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

    if name is not None:
        pca_fig.savefig(name.with_suffix(figure_type))

    if show:
        plt.show()

    plt.close(pca_fig)

def plot_spearman(name, epochs, rhos):
    fig = plt.figure()
    plt.title('Spearman\'s $\\rho$')
    plt.plot(epochs, rhos, '+--')
    plt.savefig(name)
    plt.close(fig)

def plot_loss(epochs, train_recon_loss, train_kld_loss, train_param_loss, train_total_loss, val_recon_loss, val_kld_loss, val_param_loss, val_total_loss, name, figure_type = 'png', show = False):
    fig, axs = plt.subplots(2, 2, figsize = (12, 7))
    axs[0, 0].plot(epochs, train_recon_loss, label = "Train")
    axs[0, 0].set_title('Reconstruction Loss')
    axs[0, 0].set(ylabel='Loss')
    axs[0, 1].plot(epochs, train_param_loss, label = "Train")
    axs[0, 1].set_title('$\\theta$ loss')
    axs[0, 1].set(ylabel='Loss')
    axs[1, 0].plot(epochs, train_kld_loss, label = "Train")
    axs[1, 0].set_title('KLD loss')
    axs[1, 0].set(xlabel='Epoch', ylabel='Loss')
    axs[1, 1].plot(epochs, train_total_loss, label = "Train")
    axs[1, 1].set_title('Total loss')
    axs[1, 1].set(xlabel='Epoch', ylabel='Loss')

    if len(val_recon_loss) > 0:
        axs[0, 0].plot(epochs, val_recon_loss, label = "Validation")
        axs[0, 1].plot(epochs, val_param_loss, label = "Validation")
        axs[1, 0].plot(epochs, val_kld_loss, label = "Validation")
        axs[1, 1].plot(epochs, val_total_loss, label = "Validation")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[0, 1].yaxis.tick_right()
    axs[1, 1].yaxis.tick_right()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    if name is not None:
        plt.savefig(name.with_suffix(figure_type))

    if show:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":
    device = torch.device("cuda")

    model = VAE([2594, 128, 2]).to(device)
    model.load_state_dict(torch.load("model.torch"))

    plot_data(None, model, "data/PF00144_full.txt", device)
