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

from vae import VAE
from protein_data import get_datasets, get_protein_dataloader, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq

PICKLE_FILE = Path('data/mutation_data.pickle')

"""
BRCA1_HUMAN_BRCT
B3VI55_LIPSTSTABLE
BLAT_ECOLX_Ranganathan2015
F7YBW7_MESOW_vae
RL401_YEAST_Fraser2016
CALM1_HUMAN_Roth2017
parEparD_Laub2015_all
UBC9_HUMAN_Roth2017
PABP_YEAST_Fields2013-doubles
SUMO1_HUMAN_Roth2017
RASH_HUMAN_Kuriyan
BG_STRSQ_hmmerbit
RL401_YEAST_Bolon2014
MK01_HUMAN_Johannessen
HSP82_YEAST_Bolon2016
YAP1_HUMAN_Fields2012-singles
BF520_env_Bloom2018
UBE4B_MOUSE_Klevit2013-singles
tRNA_mutation_effect
HG_FLU_Bloom2016
B3VI55_LIPST_Whitehead2015
TPK1_HUMAN_Roth2017
BLAT_ECOLX_Palzkill2012
GAL4_YEAST_Shendure2015
TIM_SULSO_b0
POLG_HCVJF_Sun2014
HIS7_YEAST_Kondrashov2017
TPMT_HUMAN_Fowler2018
DLG4_RAT_Ranganathan2012
MTH3_HAEAESTABILIZED_Tawfik2015
AMIE_PSEAE_Whitehead
BLAT_ECOLX_Tenaillon2013
BLAT_ECOLX_Ostermeier2014
BRCA1_HUMAN_RING
P84126_THETH_b0
PTEN_HUMAN_Fowler2018
PABP_YEAST_Fields2013-singles
IF1_ECOLI_Kishony
PA_FLU_Sun2015
RL401_YEAST_Bolon2013
KKA2_KLEPN_Mikkelsen2014
POL_HV1N5-CA_Ndungu2014
TIM_THEMA_b0
BG505_env_Bloom2018
"""

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

def mutation_effect_prediction(model, data_path, sheet, metric_column, ensemble_count = 500, show_scatter = False):
    model.eval()

    # load mutation and experimental pickle
    with open(PICKLE_FILE, 'rb') as f:
        proteins = pickle.load(f)
        p = proteins[sheet].dropna(subset=['mutation_effect_prediction_vae_ensemble']).reset_index(drop=True)

    # load dataset
    data, *_ = get_datasets(data_path, device, 0.8)

    wt, _, wt_seq = data[0]
    offset = int(wt_seq.id.split("/")[1].split("-")[0])
    def h(s, offset = offset):
        wildtype = IUPAC_SEQ2IDX[s[0]]
        mutant = IUPAC_SEQ2IDX[s[-1]]

        # should not happen
        if s[1:-1] == '':
            breakpoint()

        location = int(s[1:-1]) - offset
        return wildtype, mutant, location

    df = pd.DataFrame([h(s) for s in p.mutant if s != 'WT'], columns = ['wt', 'mt', 'loc'])

    df = pd.concat([p.loc[:, [metric_column]], df], axis = 1)
    data_size = len(df)
    mutants = torch.stack([wt.squeeze(0)] * data_size)
    idx = range(data_size), df['loc'].to_list()

    mutants[idx] = torch.tensor(df['mt'].astype('int64').to_list(), device = device)

    acc_m_elbo = 0
    acc_wt_elbo = 0

    for i in range(ensemble_count):
        print(f"Doing model {i}...", end = "\r")
        model.sample_new_decoder()
        m_elbo, m_logp, m_kld = model.protein_logp(mutants)
        wt_elbo, wt_logp, wt_kld = model.protein_logp(wt.unsqueeze(0))

        acc_m_elbo += m_elbo
        acc_wt_elbo += wt_elbo
    print("Done!" + " " * 50)

    ensemble_m_elbo = acc_m_elbo / args.ensemble_count
    ensemble_wt_elbo = acc_wt_elbo / args.ensemble_count

    predictions = ensemble_m_elbo - ensemble_wt_elbo
    scores = df[metric_column]

    if show_scatter:
        plt.scatter(predictions, scores)
        plt.show()

    cor, pval = spearmanr(scores, predictions.cpu())

    return cor, pval

if __name__ in ["__main__", "__console__"]:
    with torch.no_grad():
        args = parser.parse_args()

        print("Arguments given:")
        for arg, value in args.__dict__.items():
            print(f"  {arg}: {value}")
        print("")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        protein_dataset, *_ = get_datasets(args.protein_family, device, 0.8)
        print('Data loaded')

        wt, *_ = protein_dataset[0]
        size = len(wt) * NUM_TOKENS

        # load model
        model = VAE([size, 1500, 1500, 2, 100, 2000, size], NUM_TOKENS).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        cor, pval = mutation_effect_prediction(model, args.protein_family, args.data_sheet, args.metric_column, args.ensemble_count)
        # protein_accuracy()

        print(f'Spearman\'s Rho: {cor:5.3f}. Pval: {pval}')
