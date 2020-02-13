from pathlib import Path
from collections import OrderedDict
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from bioservices import UniProt

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

def process_seq(seq, device = None):
    seq_id = seq.id
    seq_offset = int(seq.id.split("/")[1].split('-')[0])
    seq_encoded = seq2idx(seq.upper(), device)
    return seq_id, seq_offset, seq_encoded

class ProteinDataset(Dataset):
    def __init__(self, seqs, device = None):
        super().__init__()
        self.device = device

        self.seqs = seqs if isinstance(seqs, list) else list(SeqIO.parse(seqs, "fasta"))
        self.encoded_seqs = torch.stack([seq2idx(seq.upper(), device) for seq in self.seqs])
        self.weights = torch.stack([1.0 / (t != self.encoded_seqs).to(torch.float).mean(1).lt(0.2).to(torch.float).sum() for t in self.encoded_seqs])

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        return self.encoded_seqs[i], self.weights[i], self.seqs[i]

def get_datasets(file, device, train_ratio):
    saved_datasets = file.with_suffix(".saved_datasets")
    if saved_datasets.exists():
        print(f"Loading data from preprocessed {saved_datasets}...")
        return get_datasets_from_saved_data(saved_datasets, device)
    else:
        print(f"Loading raw data from {file}...")
        data = get_datasets_from_raw_data(file, device, train_ratio)
        print(f"Saving data to {saved_datasets}...")
        torch.save(data, saved_datasets)
        return data

def get_datasets_from_saved_data(saved_datasets, device):
    data = torch.load(saved_datasets, map_location = device)
    return data

def get_datasets_from_raw_data(file, device, train_ratio):
    seqs = list(SeqIO.parse(file, "fasta"))
    data_len = len(seqs)
    seq_len = len(seqs[0])

    # Split into train/validation
    train_length = int(train_ratio * data_len)
    val_length = data_len - train_length

    indices = list(range(data_len))
    random.shuffle(indices)
    train_indices = indices[:train_length]
    val_indices = indices[train_length:]

    train_seqs = [seqs[i] for i in train_indices]
    val_seqs = [seqs[i] for i in val_indices]

    all_data = ProteinDataset(seqs, device)
    train_data = ProteinDataset(train_seqs, device)
    val_data = ProteinDataset(val_seqs, device)
    return all_data, train_data, val_data

def get_seqs_collate(tensors):
    encoded_seq, weights, seq = zip(*tensors)
    return torch.stack(encoded_seq), torch.stack(weights), list(seq)

def discard_seqs_collate(tensors):
    encoded_seq, weights, seq = zip(*tensors)
    return torch.stack(encoded_seq), torch.stack(weights)

def get_protein_dataloader(dataset, batch_size = 32, shuffle = False, get_seqs = False):
    return DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn = get_seqs_collate if get_seqs else discard_seqs_collate)

def retrieve_labels(infile, outfile):
    seqs = SeqIO.parse(infile, "fasta")
    uniprot = UniProt()

    if outfile.exists():
        with open(outfile, "r") as out:
            lines = out.readlines()
    else:
        lines = []

    with open(outfile, "a") as out:
        for i, seq in enumerate(seqs):
            if i < len(lines):
                continue

            if "|" in seq.id:
                ID = seq.id.split("|")[1]
            else:
                ID = seq.id.split("/")[0].replace("UniRef100_", "")
            print(f"Doing ID {i:6}, {ID + ':':15} ", end = "")
            try:
                # Try get_df
                df = uniprot.get_df(ID)
                label = df["Taxonomic lineage (PHYLUM)"][0]

                if type(label) == np.float64 and np.isnan(label):
                    columns, values = uniprot.search(ID, database = "uniparc", limit = 1)[:-1].split("\n")
                    name_idx = columns.split("\t").index("Organisms")
                    name = values.split("\t")[name_idx].split("; ")[0]
                    columns, values = uniprot.search(name, database = "taxonomy", limit = 1)[:-1].split("\n")
                    lineage_idx = columns.split("\t").index("Lineage")
                    label = values.split("\t")[lineage_idx].split("; ")[:2][-1]
            except:
                try:
                    columns, values = uniprot.search(ID, database = "uniparc", limit = 1)[:-1].split("\n")
                    name_idx = columns.split("\t").index("Organisms")
                    name = values.split("\t")[name_idx].split("; ")[0]
                    columns, values = uniprot.search(name, database = "taxonomy", limit = 1)[:-1].split("\n")
                    lineage_idx = columns.split("\t").index("Lineage")
                    label = values.split("\t")[lineage_idx].split("; ")[:2][-1]
                except:
                    try:
                        columns, values = uniprot.search(ID, database = "uniparc", limit = 1)[:-1].split("\n")
                        name_idx = columns.split("\t").index("Organisms")
                        name = values.split("\t")[name_idx].split("; ")[0].split(" ")[0]
                        columns, values = uniprot.search(name, database = "taxonomy", limit = 1)[:-1].split("\n")
                        lineage_idx = columns.split("\t").index("Lineage")
                        label = values.split("\t")[lineage_idx].split("; ")[:2][-1]
                    except:
                        print("Couldn't handle it!")
                        breakpoint()

            print(f"{label}")
            out.write(f"{seq.id}: {label}\n")

if __name__ == "__main__":
    retrieve_labels(Path("data/alignments/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m"), Path("data/alignments/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48_LABELS.a2m"))
