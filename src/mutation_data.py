import pandas as pd
import pickle
import torch
import numpy as np
from vae import VAE
from pathlib import Path
from protein_data import ProteinDataset, get_protein_dataloader, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ


BLAT_ECOL = 'BLAT_ECOLX_Palzkill2012'
BLAT_SEQ_FILE = Path('data/alignments/BLAT_ECOLX_1_b0.5.a2m')
PICKLE_FILE = Path('data/mutation_data.pickle')

sequence = ('hpetlvKVKDAEDQLGARVGYIELDLNSGKILeSFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQL'
            'GRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGD''HVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRS''ALPAGWFIADKSGAGErGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIkhw')

device = torch.device("cpu")
protein_dataset = ProteinDataset(BLAT_SEQ_FILE, device)

wild_type, _ = next(iter(protein_dataset))
wild_type = wild_type.unsqueeze(0)

model = VAE([7890, 1500, 1500, 30, 100, 2000, 7890], NUM_TOKENS).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device))

wild_type_logp = model.protein_log_probability(wild_type)

with open(PICKLE_FILE, 'rb') as f:
    proteins = pickle.load(f)

p = proteins[BLAT_ECOL]

def h(s):
    wildtype = IUPAC_SEQ2IDX[s[0]]
    mutant = IUPAC_SEQ2IDX[s[-1]]
    location = int(s[1:-1])
    return wildtype, mutant, location

df = pd.DataFrame([h(s) for s in p.mutant], columns = ['wildtype', 'mutant', 'location'])

df = pd.concat([p.loc[:, ['ddG_stat']], df], axis = 1)

batch_size = 64
for batch in np.array_split(df, len(df)/batch_size):
    wt = torch.stack([wild_type.squeeze(0)] * len(batch))


    breakpoint()
