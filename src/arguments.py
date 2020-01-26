import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description = "Variational Auto-Encoder on aligned protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data", type = Path, help = "Fasta input file of sequences")
parser.add_argument("layer_sizes", type = int, nargs = "+", help = "Sizes of the hidden layers of the VAE. First size is the input size, last is the latent space size.")
parser.add_argument("--embed_size", type = int, default = 10, help = "Size of the embedded vectors of the amino acids.")
parser.add_argument("--epochs", type = int, default = 10, help = "Maximum number of epochs to train (patience may cause fewer epochs to be run).")
parser.add_argument("--batch_size", type = int, default = 128, help = "Input batch size for training.")
parser.add_argument("--val_ratio", type = float, default = 0.2, help = "What fraction of data to use for validation.")
parser.add_argument("--device", type = str, default = "cuda", choices = ["cpu", "cuda"], help = "Which device to use (CPU or CUDA for GPU).")
parser.add_argument("--patience", type = int, default = 50, help = "Training will stop if the model does not improve on the validation set for this many epochs.")
parser.add_argument("--log_interval", type = float, default = 1, help = "How many seconds to wait between training status logging.")
parser.add_argument("--save_path", type = Path, default = Path("model.torch"), help = "Path to save the model to every time it improves on validation loss.")
parser.add_argument("--seed", type = int, help = "Seed for random number generation. If not set, a random seed will be used.")
args = parser.parse_args()

# Argument postprocessing
if len(args.layer_sizes) < 2:
	raise ValueError("At least 2 layers sizes (input size and latent space size) must be given.")

args.train_ratio = 1 - args.val_ratio
