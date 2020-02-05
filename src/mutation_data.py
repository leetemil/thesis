import pandas as pd
import pickle
import torch
import numpy as np
from vae import VAE
from pathlib import Path
from protein_data import ProteinDataset, get_protein_dataloader, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


BLAT_ECOL = 'BLAT_ECOLX_Palzkill2012'
BLAT_SEQ_FILE = Path('data/alignments/BLAT_ECOLX_1_b0.5.a2m')
PICKLE_FILE = Path('data/mutation_data.pickle')

sequence = ('hpetlvKVKDAEDQLGARVGYIELDLNSGKILeSFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQL'
            'GRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGD''HVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRS''ALPAGWFIADKSGAGErGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIkhw')

device = torch.device("cpu")
protein_dataset = ProteinDataset(BLAT_SEQ_FILE, device)

wild_type, wild_type_id = next(iter(protein_dataset))
wild_type = wild_type.unsqueeze(0)

model = VAE([7890, 1500, 1500, 30, 100, 2000, 7890], NUM_TOKENS).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))

with torch.no_grad():

    wild_type_logp = model.protein_log_probability(wild_type)

    with open(PICKLE_FILE, 'rb') as f:
        proteins = pickle.load(f)

    p = proteins[BLAT_ECOL]
    offset = protein_dataset.offsets[wild_type_id]

    def h(s, offset = 0):
        wildtype = IUPAC_SEQ2IDX[s[0]]
        mutant = IUPAC_SEQ2IDX[s[-1]]
        location = int(s[1:-1]) - offset
        return wildtype, mutant, location

    df = pd.DataFrame([h(s, offset) for s in p.mutant], columns = ['wildtype', 'mutant', 'location'])

    df = pd.concat([p.loc[:, ['ddG_stat']], df], axis = 1)

    predictions = np.empty(len(df))
    scores = np.empty(len(df))

    for batch in np.array_split(df, len(df)/len(df)):
        wt = torch.stack([wild_type.squeeze(0)] * len(batch))
        bs = len(batch)
        idx = range(bs), batch.location
        wt[idx] = torch.tensor(batch.mutant)

        mutant_logp = model.protein_log_probability(wt)

        scores = batch.ddG_stat
        predictions = mutant_logp #- wild_type_logp

        # plt.scatter(scores, predictions)
        cor, pval = spearmanr(scores, predictions)
        print(cor)
