from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO

IUPAC_AMINO_IDX_PAIRS = [
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)
]
IUPAC_IDX_AMINO_PAIRS = [(i, a) for (a, i) in IUPAC_AMINO_IDX_PAIRS]

NUM_TOKENS = len(IUPAC_AMINO_IDX_PAIRS)

IUPAC_SEQ2IDX = OrderedDict(IUPAC_AMINO_IDX_PAIRS)
IUPAC_IDX2SEQ = OrderedDict(IUPAC_IDX_AMINO_PAIRS)

# Add gap tokens as the same as mask
IUPAC_SEQ2IDX["-"] = IUPAC_SEQ2IDX["<mask>"]
IUPAC_SEQ2IDX["."] = IUPAC_SEQ2IDX["<mask>"]

def seq2idx(seq, device = None):
    return torch.tensor([IUPAC_SEQ2IDX[s] for s in seq], device = device)

def idx2seq(idxs):
    return "".join([IUPAC_IDX2SEQ[i] for i in idxs])

class ProteinDataset(Dataset):
    def __init__(self, file, device = None):
        super().__init__()
        self.device = device

        seqs = SeqIO.parse(file, "fasta")
        list_seqs = map(lambda s: list(s.upper()), seqs)
        encodedSeqs = map(lambda s: seq2idx(s, self.device), list_seqs)
        self.seqs = list(encodedSeqs)

        seqs = SeqIO.parse(file, "fasta")
        self.names = list(map(lambda s: s.id.split("/")[0], seqs))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.names[i]

def discard_names_collate(tensors):
    tensors, names = zip(*tensors)
    return torch.stack(tensors)

def get_protein_dataloader(dataset, batch_size = 32, shuffle = False, get_names = False):
    return DataLoader(dataset, shuffle = shuffle, batch_size = batch_size, collate_fn = None if get_names else discard_names_collate)
