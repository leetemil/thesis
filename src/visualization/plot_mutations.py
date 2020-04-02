import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models import VAE
from data import get_datasets, NUM_TOKENS, IUPAC_SEQ2IDX

parser = argparse.ArgumentParser(description = "Mutation representations", formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

# Required arguments
parser.add_argument("protein_family", type = str, help = "Protein family alignment data")
parser.add_argument("data_sheet", type = str, help = "Protein family data sheet in mutation_data.pickle.")
parser.add_argument("metric", type = str, help = "Metric column of sheet used for Spearman's Rho calculation")

args = parser.parse_args()

print("Arguments given:")
for arg, value in args.__dict__.items():
	print(f"  {arg}: {value}")
print("")

ALIGNPATH = Path('data/alignments')
PICKLE_FILE = Path('data/mutation_data.pickle')

SHEET = args.data_sheet#'PABP_YEAST_Fields2013-singles'
PROTEIN_FAMILY = ALIGNPATH / Path(args.protein_family)
METRIC_COLUMN = args.metric

# only tested on cpu device ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

protein_dataset, *_ = get_datasets(PROTEIN_FAMILY, device, None)
print('Data loaded')

wt, _, wt_seq = protein_dataset[0]
wt_id = wt_seq.id
size = len(wt) * NUM_TOKENS
wt = wt.unsqueeze(0)

model = VAE([size, 1500, 1500, 30, 100, 2000, size], NUM_TOKENS).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device)["state_dict"])

def protein_accuracy(trials = 100, model = model, data = protein_dataset):
    model.eval()
    print(f'{wt_id}: Prediction accuracies for {trials} proteins.')
    data = iter(data)
    for _ in range(trials):
        p, _,  p_seq = next(data)
        p_recon = model.reconstruct(p.unsqueeze(0)).squeeze(0).numpy()
        p = p.numpy()
        loss = 1 - (p == p_recon).mean()
        print(f'{p_seq.id:<60s}{100 * loss:>4.1f}%')

def plot_mutations(model = model, data = protein_dataset):
    model.eval()

    with open(PICKLE_FILE, 'rb') as f:
        proteins = pickle.load(f)
        p = proteins[SHEET]

    wt, _, wt_seq = data[0]

    offset = int(wt_seq.id.split("/")[1].split("-")[0])
    def h(s, offset = offset):
        wildtype = IUPAC_SEQ2IDX[s[0]]
        mutant = IUPAC_SEQ2IDX[s[-1]]
        location = int(s[1:-1]) - offset
        return wildtype, mutant, location

    df = pd.DataFrame([h(s) for s in p.mutant], columns = ['wt', 'mt', 'loc'])

    df = pd.concat([p.loc[:, [METRIC_COLUMN]], df], axis = 1)
    data_size = len(df)

    mutants = torch.stack([wt.squeeze(0)] * data_size)
    idx = range(data_size), df['loc'][:data_size]
    mutants[idx] = torch.tensor(df['mt'], device = device)

    z_mutants = model.encode(mutants)[0].squeeze(0)
    z_wt = model.encode(wt.unsqueeze(0).unsqueeze(0))[0]

    if z_wt.size(1) > 2:
        print(f'Latent space has {z_wt.size(1)} dimensions. Using PCA to project to 2d.')
        pca = PCA(2)
        z_mutants = pca.fit_transform(z_mutants)
        z_wt = pca.transform(z_wt)

    plt.scatter(z_mutants[:, 0], z_mutants[:, 1])
    plt.scatter(z_wt[:, 0], z_wt[:, 1])
    plt.show()

    return None

with torch.no_grad():
    plot_mutations()
