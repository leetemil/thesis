import argparse
from pathlib import Path
from datetime import datetime
from typing import Union

def basic_args(parser):
	parser.add_argument("--epochs", type = int, default = 10, help = "Maximum number of epochs to train (patience may cause fewer epochs to be run).")
	parser.add_argument("--batch_size", type = int, default = 128, help = "Input batch size for training.")
	parser.add_argument("--device", type = str, default = "cuda", choices = ["cpu", "cuda"], help = "Which device to use (CPU or CUDA for GPU).")
	parser.add_argument("--patience", type = int, default = 50, help = "Training will stop if the model does not improve on the validation set for this many epochs.")
	parser.add_argument("--log_interval", type = lambda x: x if x == "batch" else float(x), default = 1, help = "How many seconds to wait between training status logging. 0 to disable loading bar progress. \"batch\" for log at every batch.")
	parser.add_argument("--results_dir", type = Path, default = Path(f"results_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}"), help = "Directory to save results to.")
	parser.add_argument("--seed", type = int, help = "Seed for random number generation. If not set, a random seed will be used.")

def get_vae_args():
	parser = argparse.ArgumentParser(description = "Variational Auto-Encoder on aligned protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	basic_args(parser)

	parser.add_argument("data", type = Path, help = "Fasta input file of sequences")
	parser.add_argument("layer_sizes", type = int, nargs = "+", help = "Sizes of the hidden layers of the VAE, except the first and last, which will be inferred from the data argument. The smallest size is understood as the bottleneck size and will be the size of the output of the encoder, and the size of the input of the decoder.")
	parser.add_argument("--val_ratio", type = float, default = 0.2, help = "What fraction of data to use for validation.")
	parser.add_argument("--visualize_interval", type = str, default = "improvement", choices = ["always", "improvement", "never"], help = "Visualize the output at every epoch (always), only at validation loss improvement or never.")
	parser.add_argument("--visualize_style", type = str, default = "save", choices = ["save", "show", "both"], help = "Save or show the visualization, or both.")
	parser.add_argument("--figure_type", type = str, default = ".png", help = "Filetype of the visualization figures.")

	args = parser.parse_args()

	# Argument postprocessing
	args.train_ratio = 1 - args.val_ratio
	args.results_dir.mkdir(exist_ok = True)
	print_args(args)
	return args

def get_unirep_args():
	parser = argparse.ArgumentParser(description = "UniRep model on protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	basic_args(parser)

	# Data
	parser.add_argument("train_data", type = Path, help = "Fasta input file for training.")
	parser.add_argument("validation_data", type = Path, help = "Fasta input file for validation.")

	parser.add_argument("--embed_size", type = int, default = 10, help = "Size of the amino acid embedding.")
	parser.add_argument("--hidden_size", type = int, default = 512, help = "Size of the hidden state of the LSTM.")
	parser.add_argument("--num_layers", type = int, default = 1, help = "Number of layers of the LSTM.")

	args = parser.parse_args()

	args.results_dir.mkdir(exist_ok = True)
	print_args(args)
	return args

def print_args(args):
	print("Arguments given:")
	for arg, value in args.__dict__.items():
		print(f"  {arg}: {value}")
	print("")
