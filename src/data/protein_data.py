from pathlib import Path
from collections import OrderedDict
import random
import multiprocessing
import threading

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, WeightedRandomSampler
from Bio import SeqIO
from bioservices import UniProt

IUPAC_IDX_AMINO_PAIRS = list(enumerate([
    "<pad>",
    "<mask>",
    "<cls>",
    "<sep>",
    "<unk>",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]))
IUPAC_AMINO_IDX_PAIRS = [(a, i) for (i, a) in IUPAC_IDX_AMINO_PAIRS]

NUM_TOKENS = len(IUPAC_AMINO_IDX_PAIRS)

IUPAC_SEQ2IDX = OrderedDict(IUPAC_AMINO_IDX_PAIRS)
IUPAC_IDX2SEQ = OrderedDict(IUPAC_IDX_AMINO_PAIRS)

# Add gap tokens as the same as mask
IUPAC_SEQ2IDX["-"] = IUPAC_SEQ2IDX["<mask>"]
# IUPAC_SEQ2IDX["B"] = IUPAC_SEQ2IDX["<mask>"]
# IUPAC_SEQ2IDX["O"] = IUPAC_SEQ2IDX["<mask>"]
# IUPAC_SEQ2IDX["U"] = IUPAC_SEQ2IDX["<mask>"]
# IUPAC_SEQ2IDX["X"] = IUPAC_SEQ2IDX["<mask>"]
# IUPAC_SEQ2IDX["Z"] = IUPAC_SEQ2IDX["<mask>"]
# IUPAC_SEQ2IDX["."] = IUPAC_SEQ2IDX["<mask>"]

# Add small letters as the same as mask
# for amino, idx in IUPAC_AMINO_IDX_PAIRS:
#     if len(amino) == 1:
#         IUPAC_SEQ2IDX[amino.lower()] = IUPAC_SEQ2IDX["<mask>"]

def seq2idx(seq, device = None):
    return torch.tensor([IUPAC_SEQ2IDX[s] for s in seq if len(s) > 1 or (s == s.upper() and s != ".")], device = device)

def idx2seq(idxs):
    return "".join([IUPAC_IDX2SEQ[i] for i in idxs])

class ProteinDataset(Dataset):
    def __init__(self, seqs, device = None, weight_batch_size = 1000):
        super().__init__()
        self.device = device

        self.seqs = seqs if isinstance(seqs, list) else list(SeqIO.parse(seqs, "fasta"))

        if len(self.seqs) == 0:
            self.encoded_seqs = torch.Tensor()
            self.weights = torch.Tensor()
            self.neff = 0
            return

        self.encoded_seqs = torch.stack([seq2idx(seq, device) for seq in self.seqs])

        # Calculate weights
        weights = []
        flat_one_hot = F.one_hot(self.encoded_seqs).float().flatten(1)
        for i in range(self.encoded_seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (self.encoded_seqs[i * weight_batch_size : (i + 1) * weight_batch_size] != 1).sum(1).unsqueeze(-1)
            w = 1.0 / (similarities / lengths).gt(0.8).sum(1).float()
            weights.append(w)
        self.weights = torch.cat(weights)
        self.neff = self.weights.sum()

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        return self.encoded_seqs[i], self.weights[i], self.neff, self.seqs[i]

class VariableLengthProteinDataset(Dataset):
    def __init__(self, seqs, device = None):
        super().__init__()
        self.device = device

        seqs = seqs if isinstance(seqs, list) else list(SeqIO.parse(seqs, "fasta"))
        CLS = "<cls>"
        SEP = "<sep>"
        unpadded_seqs = [[CLS] + list(filter(lambda c: c not in ".-" and c == c.upper(), str(s.seq))) + [SEP] for s in seqs]
        self.encoded_seqs = [seq2idx(seq, device) for seq in unpadded_seqs]

    def __len__(self):
        return len(self.encoded_seqs)

    def __getitem__(self, i):
        return self.encoded_seqs[i]

class IterProteinDataset(IterableDataset):
    def __init__(self, file, device = None):
        super().__init__()
        self.file = file
        self.device = device

        with open(self.file) as f:
            self.length = int(f.readline().replace("#", ""))

    def __iter__(self):
        CLS = "<cls>"
        SEP = "<sep>"

        raw_seqs = SeqIO.parse(self.file, "fasta")
        list_seqs = map(lambda x: [CLS] + list(str(x.seq)) + [SEP], raw_seqs)
        encodedSeqs = map(lambda x: seq2idx(x, self.device), list_seqs)
        return encodedSeqs

    def __len__(self):
        return self.length

def get_datasets(file, device, train_ratio, use_saved = False):
    saved_datasets = file.with_suffix(".saved_datasets")
    if saved_datasets.exists() and use_saved:
        print(f"Loading data from preprocessed {saved_datasets}")
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
    encoded_seq, weights, neffs, seq = zip(*tensors)
    return torch.stack(encoded_seq), torch.stack(weights), neffs[0], list(seq)

def discard_seqs_collate(tensors):
    encoded_seq, weights, neffs, seq = zip(*tensors)
    return torch.stack(encoded_seq), torch.stack(weights), neffs[0]

def get_protein_dataloader(dataset, batch_size = 128, shuffle = False, get_seqs = False):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = get_seqs_collate if get_seqs else discard_seqs_collate)

def variable_length_sequence_collate(sequences):
    return torch.nn.utils.rnn.pad_sequence(sequences, padding_value = IUPAC_SEQ2IDX["<pad>"], batch_first = True)

def get_variable_length_protein_dataLoader(dataset, batch_size = 128, shuffle = False):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = variable_length_sequence_collate)

def retrieve_label_from_uniprot_df(ID):
    uniprot = UniProt()
    df = uniprot.get_df(ID)
    label = df["Taxonomic lineage (PHYLUM)"][0]

    if type(label) == np.float64 and np.isnan(label):
        raise ValueError("Label was NaN")
    return label

def retrieve_label_from_uniparc(ID):
    uniprot = UniProt()
    columns, values = uniprot.search(ID, database = "uniparc", limit = 1)[:-1].split("\n")
    name_idx = columns.split("\t").index("Organisms")
    name = values.split("\t")[name_idx].split("; ")[0]
    columns, values = uniprot.search(name, database = "taxonomy", limit = 1)[:-1].split("\n")
    lineage_idx = columns.split("\t").index("Lineage")
    label = values.split("\t")[lineage_idx].split("; ")[:2][-1]
    return label

lock = threading.Lock()
def retrieve_label(ID):
    try:
        label = retrieve_label_from_uniprot_df(ID)
    except:
        label = retrieve_label_from_uniparc(ID)
    return label

def retrieve_many_labels(seqs, outlist):
    thread_list = []
    for seq in seqs:
        ID = seq.id.split("/")[0].split("|")[:2][-1]
        label = retrieve_label(ID)
        thread_list.append((ID, label))
        print(".", end = "")

    with lock:
        outlist += thread_list
        print(f"List: {thread_list}")

def parallel_retrieve_labels(infile, outfile):
    print("Creating threads...")
    seqs = list(SeqIO.parse(infile, "fasta"))
    threads = []
    results = []
    chunk_size = 10
    for i in range(len(seqs) // chunk_size):
        args = [seqs[i * chunk_size:(i + 1) * chunk_size], results]
        threads.append(threading.Thread(target = retrieve_many_labels, args = args))

    print(f"Created {len(threads)} threads.")
    print("Starting threads...")
    for thread in threads:
        thread.start()

    print("Joining on threads...")
    joined = 0
    for thread in threads:
        thread.join()
        joined += 1
        print(f"Joined on {joined} threads.")

    breakpoint()
    with open(outfile, "w") as out:
        pass

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
    infile = Path("data/alignments/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m")
    outfile = Path("data/alignments/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48_LABELS.a2m")
    parallel_retrieve_labels(infile, outfile)