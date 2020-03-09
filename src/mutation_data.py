import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description = "Mutation prediction and analysis", formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
# Required arguments
parser.add_argument("protein_family", type = Path, help = "Protein family alignment data")
parser.add_argument("data_sheet", type = str, help = "Protein family data sheet in mutation_data.pickle.")
parser.add_argument("metric_column", type = str, help = "Metric column of sheet used for Spearman's Rho calculation")
parser.add_argument("model_path", type = Path, help = "The path of the model")
parser.add_argument("--ensemble_count", type = int, default = 500, help = "How many samples of the model to use for evaluation as an ensemble.")

from datetime import datetime
import pickle

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from Bio import SeqIO

from vae import VAE
# from unirep import UniRep
from protein_data import get_datasets, get_protein_dataloader, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq

PICKLE_FILE = Path('data/mutation_data.pickle')

# def protein_accuracy(trials = 100, model = model, data = protein_dataset):
#     model.eval()
#     print(f'{wt_id}: Prediction accuracies for {trials} proteins.')
#     data = iter(data)
#     for _ in range(trials):
#         p, _,  p_seq = next(data)
#         p_recon = model.reconstruct(p.unsqueeze(0)).squeeze(0).numpy()
#         p = p.numpy()
#         loss = 1 - (p == p_recon).mean()
#         print(f'{p_seq.id:<60s}{100 * loss:>4.1f}%')

def mutation_effect_prediction(model, data_path, sheet, metric_column, device, ensemble_count = 500, results_dir = Path("."), savefig = True):
    model.eval()

    # load mutation and experimental pickle
    with open(PICKLE_FILE, 'rb') as f:
        proteins = pickle.load(f)
        p = proteins[sheet].dropna(subset=['mutation_effect_prediction_vae_ensemble']).reset_index(drop=True)

    # load dataset
    wt_seq = next(SeqIO.parse(data_path, "fasta"))
    wt_indices = np.array([i for i, c in enumerate(str(wt_seq.seq)) if c == c.upper() and c != "."])
    wt = seq2idx(wt_seq, device)

    offset = int(wt_seq.id.split("/")[1].split("-")[0])
    positions = wt_indices + offset
    positions_dict = {pos: i for i, pos in enumerate(positions)}

    def h(s, offset = offset):
        wildtype = IUPAC_SEQ2IDX[s[0]]
        mutant = IUPAC_SEQ2IDX[s[-1]]
        location = positions_dict[int(s[1:-1])]
        return wildtype, mutant, location

    df = pd.DataFrame([h(s) for s in p.mutant if s != 'WT'], columns = ['wt', 'mt', 'loc'])

    df = pd.concat([p.loc[:, [metric_column]], df], axis = 1)
    data_size = len(df)
    mutants = torch.stack([wt.squeeze(0)] * data_size)
    idx = range(data_size), df['loc'].to_list()

    mutants[idx] = torch.tensor(df['mt'].astype('int64').to_list(), device = device)

    if isinstance(model, VAE):
        acc_m_elbo = 0
        acc_wt_elbo = 0

        for i in range(ensemble_count):
            # print(f"Doing model {i}...", end = "\r")
            model.sample_new_decoder()
            m_elbo, m_logp, m_kld = model.protein_logp(mutants)
            wt_elbo, wt_logp, wt_kld = model.protein_logp(wt.unsqueeze(0))

            acc_m_elbo += m_elbo
            acc_wt_elbo += wt_elbo
        # print("Done!" + " " * 50)

        ensemble_m_elbo = acc_m_elbo / ensemble_count
        ensemble_wt_elbo = acc_wt_elbo / ensemble_count
        predictions = ensemble_m_elbo - ensemble_wt_elbo

    else:
        wt_logp = model.protein_logp(wt.unsqueeze(0))

        batch_size = 128
        batches = len(mutants) // batch_size + 1
        log_probs = []
        for i in range(batches):
            batch_mutants = mutants[batch_size * i: batch_size * (i + 1)]
            m_logp = model.protein_logp(batch_mutants)
            log_probs.append(m_logp)

        predictions = torch.cat(log_probs) - wt_logp

    scores = df[metric_column]

    if savefig:
        plt.scatter(predictions.cpu(), scores)
        plt.title("Correlation")
        plt.xlabel("$\\Delta$-elbo")
        plt.ylabel("Experimental value")
        plt.savefig(results_dir / Path("Correlation_scatter.png"))

    cor, _ = spearmanr(scores, predictions.cpu())
    return cor

# if __name__ in ["__main__", "__console__"]:
#     with torch.no_grad():
#         args = parser.parse_args()

#         print("Arguments given:")
#         for arg, value in args.__dict__.items():
#             print(f"  {arg}: {value}")
#         print("")

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         protein_dataset, *_ = get_datasets(args.protein_family, device, 0.8)
#         print('Data loaded')

#         wt, *_ = protein_dataset[0]
#         size = len(wt) * NUM_TOKENS

#         # load model
#         model = VAE([size, 1500, 1500, 30, 100, 2000, size], NUM_TOKENS).to(device)
#         model.load_state_dict(torch.load(args.model_path, map_location=device))

#         cor = mutation_effect_prediction(model, args.protein_family, args.data_sheet, device, args.metric_column, args.ensemble_count)
#         # protein_accuracy()

#         print(f'Spearman\'s Rho: {cor:5.3f}')
