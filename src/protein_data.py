from collections import OrderedDict

import torch
from torch.utils.data import Dataset
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

def seq2idx(seq, device = None):
    return torch.tensor([IUPAC_SEQ2IDX[s] for s in seq], device = device)

def idx2seq(idxs):
    return "".join([IUPAC_IDX2SEQ[i] for i in idxs])

class ProteinDataset(Dataset):
    def __init__(self, file, device = None):
        super().__init__()
        self.device = device
        seqs = SeqIO.parse(file, "fasta")
        list_seqs = map(lambda s: ["<cls>"] + list(s.upper()) + ["<sep>"], seqs)
        mask_seqs = map(lambda s: [a if a != "-" else "<mask>" for a in s], list_seqs)
        encodedSeqs = map(lambda s: seq2idx(s, self.device), mask_seqs)
        self.seqs = list(encodedSeqs)

        # with open(file) as f:
        #     seqs = f.readlines()

        # list_seqs = map(lambda x: [CLS] + list(x[:-1]) + [SEP], seqs)
        # encodedSeqs = map(lambda x: seq2idx(x, self.device), list_seqs)
        # self.seqs = list(encodedSeqs)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]
