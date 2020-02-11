import pandas as pd
import pickle
import torch
import numpy as np
from vae import VAE
from pathlib import Path
from protein_data import ProteinDataset, get_protein_dataloader, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq, IUPAC_AMINO_IDX_PAIRS
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ALIGNPATH = Path('data/alignments')
BLAT_ECOL = 'BLAT_ECOLX_Palzkill2012'
BLAT_SEQ_FILE = ALIGNPATH / Path('BLAT_ECOLX_1_b0.5.a2m')
BLAT_SEQ_FILE = ALIGNPATH / Path('BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m')
PICKLE_FILE = Path('data/mutation_data.pickle')

device = torch.device("cpu")
protein_dataset = ProteinDataset(BLAT_SEQ_FILE, device)

model = VAE([7890, 1500, 1500, 30, 100, 2000, 7890], NUM_TOKENS).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))

# Softmax for the first 4 proteins
fig, axes = plt.subplots(4)
for i, ax in enumerate(axes):
    sample, _, _ = protein_dataset[i]
    mu, _ = model.encode(sample.unsqueeze(0))
    # z = model.reparameterize(mu, logvar)
    ds = model.decode(mu).squeeze(0).exp().detach().numpy()

    ax.imshow(ds.T, cmap=plt.get_cmap("Blues"))

    acids, _ = zip(*IUPAC_AMINO_IDX_PAIRS)
    ax.set_yticks(np.arange(len(IUPAC_AMINO_IDX_PAIRS)))
    ax.set_yticklabels(list(acids))

plt.show()
