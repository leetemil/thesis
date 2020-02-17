import pandas as pd
import pickle
import torch
import numpy as np
from vae import VAE
from pathlib import Path
from protein_data import get_datasets, get_protein_dataloader, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description = "Mutation prediction and analysis", formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

# Required arguments
parser.add_argument("protein_family", type = str, help = "Protein family alignment data")
parser.add_argument("data_sheet", type = str, help = "Protein family data sheet in mutation_data.pickle.")
parser.add_argument("metric", type = str, help = "Metric column of sheet used for Spearman's Rho calculation")
parser.add_argument("model_path", type = Path, help = "The path of the model")

args = parser.parse_args()

print("Arguments given:")
for arg, value in args.__dict__.items():
	print(f"  {arg}: {value}")
print("")

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

ALIGNPATH = Path('data/alignments')
PICKLE_FILE = Path('data/mutation_data.pickle')

SHEET = args.data_sheet#'PABP_YEAST_Fields2013-singles'
PROTEIN_FAMILY = ALIGNPATH / Path(args.protein_family)
METRIC_COLUMN = args.metric
MODEL = args.model_path

# only tested on cpu device ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

protein_dataset, *_ = get_datasets(PROTEIN_FAMILY, device, 0.8)
print('Data loaded')

wt, _, wt_seq = protein_dataset[0]
wt_id = wt_seq.id
size = len(wt) * NUM_TOKENS
wt = wt.unsqueeze(0)

model = VAE([size, 1500, 1500, 30, 100, 2000, size], NUM_TOKENS).to(device)
model.load_state_dict(torch.load(MODEL, map_location=device))

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

def mutation_effect_prediction(model = model, data = protein_dataset):
    model.eval()

    with open(PICKLE_FILE, 'rb') as f:
        proteins = pickle.load(f)
        p = proteins[SHEET].dropna(subset=['mutation_effect_prediction_vae_ensemble']).reset_index(drop=True)

    wt, _, wt_seq = data[0]
    offset = int(wt_seq.id.split("/")[1].split("-")[0])
    def h(s, offset = offset):
        wildtype = IUPAC_SEQ2IDX[s[0]]
        mutant = IUPAC_SEQ2IDX[s[-1]]
        if s[1:-1] == '':
            breakpoint()
        location = int(s[1:-1]) - offset
        return wildtype, mutant, location

    df = pd.DataFrame([h(s) for s in p.mutant if s != 'WT'], columns = ['wt', 'mt', 'loc'])

    df = pd.concat([p.loc[:, [METRIC_COLUMN]], df], axis = 1)#.dropna()
    # breakpoint()
    data_size = len(df)
    mutants = torch.stack([wt.squeeze(0)] * data_size)
    idx = range(data_size), df['loc'].to_list()

    mutants[idx] = torch.tensor(df['mt'].astype('int64').to_list(), device = device)
    m_elbo, m_logp, m_kld = model.protein_logp(mutants)
    wt_elbo, wt_logp, wt_kld = model.protein_logp(wt.unsqueeze(0))

    predictions = m_elbo - wt_elbo
    scores = df[METRIC_COLUMN]

    # plt.scatter(predictions, scores)
    # plt.show()

    cor, pval = spearmanr(scores, predictions.cpu())

    return cor, pval

with torch.no_grad():
    cor, pval = mutation_effect_prediction()
    # protein_accuracy()

    print(f'Spearman\'s Rho: {cor:5.3f}. Pval: {pval}')
