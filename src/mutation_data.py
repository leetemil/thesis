import pandas as pd
import pickle
import torch
import numpy as np
from vae import VAE
from pathlib import Path
from protein_data import ProteinDataset, get_protein_dataloader, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ALIGNPATH = Path('data/alignments')
BLAT_ECOL = 'BLAT_ECOLX_Palzkill2012'
BLAT_SEQ_FILE = ALIGNPATH / Path('BLAT_ECOLX_1_b0.5.a2m')
BLAT_SEQ_FILE = ALIGNPATH / Path('BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m')
PICKLE_FILE = Path('data/mutation_data.pickle')

# only tested on cpu device ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

protein_dataset = ProteinDataset(BLAT_SEQ_FILE, device)

wt, wt_id = protein_dataset[0]
wt = wt.unsqueeze(0)

model = VAE([7890, 1500, 1500, 30, 100, 2000, 7890], NUM_TOKENS).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))

def protein_accuracy(trials = 100, model = model, data = protein_dataset):
    model.eval()
    fname = data.file.stem
    print(f'{fname}: Prediction accuracies for {trials} proteins.')
    data = iter(data)
    for _ in range(trials):
        protein, name = next(data)
        wtr = model.reconstruct(protein.unsqueeze(0)).squeeze(0).numpy()
        wt = protein.numpy()
        loss = 1 - (wtr == wt).mean()
        print(f'{name:<60s}{100 * loss:>4.1f}%')

def mutation_effect_prediction(model = model, data = protein_dataset):
    model.eval()

    with open(PICKLE_FILE, 'rb') as f:
        proteins = pickle.load(f)

    p = proteins[BLAT_ECOL]
    offset = protein_dataset.offsets[wt_id]

    def h(s, offset = 0):
        wildtype = IUPAC_SEQ2IDX[s[0]]
        mutant = IUPAC_SEQ2IDX[s[-1]]
        location = int(s[1:-1]) - offset
        return wildtype, mutant, location

    df = pd.DataFrame([h(s, offset) for s in p.mutant], columns = ['wildtype', 'mutant', 'location'])

    df = pd.concat([p.loc[:, ['ddG_stat']], df], axis = 1)
    data_size = len(df)

    mutants = torch.stack([wt.squeeze(0)] * data_size)
    idx = range(data_size), df.location[:data_size]
    mutants[idx] = torch.tensor(df.mutant, device = device)

    m_elbo, m_logp, m_kld = model.protein_logp(mutants)
    wt_elbo, wt_logp, wt_kld = model.protein_logp(wt)

    predictions = m_elbo - wt_elbo
    scores = df.ddG_stat

    # plt.scatter(predictions, scores)
    # plt.show()

    cor, pval = spearmanr(scores, predictions.cpu())

    return cor, pval

with torch.no_grad():
    cor, pval = mutation_effect_prediction()
    # protein_accuracy()

    print(f'Spearman\'s Rho: {cor:5.3f}. Pval: {pval}')
