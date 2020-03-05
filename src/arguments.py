import argparse
from pathlib import Path
from datetime import datetime
from typing import Union

def basic_args(parser):
    parser.add_argument("--epochs", type = int, default = 10, help = "Maximum number of epochs to train (patience may cause fewer epochs to be run).")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Input batch size for training.")
    parser.add_argument("--clip_grad_norm", type = lambda x: None if x is None else float(x), default = None, help = "Gradient will be clipped to this norm. Disabled if None.")
    parser.add_argument("--clip_grad_value", type = lambda x: None if x is None else float(x), default = None, help = "Gradient values will be clipped to this value. Disabled if None.")
    parser.add_argument("--device", type = str, default = "cuda", choices = ["cpu", "cuda"], help = "Which device to use (CPU or CUDA for GPU).")
    parser.add_argument("--patience", type = int, default = 50, help = "Training will stop if the model does not improve on the validation set for this many epochs.")
    parser.add_argument("--log_interval", type = lambda x: x if x == "batch" else float(x), default = 1, help = "How many seconds to wait between training status logging. 0 to disable loading bar progress. \"batch\" for log at every batch.")
    parser.add_argument("--results_dir", type = Path, default = Path(f"results_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}"), help = "Directory to save results to.")
    parser.add_argument("--seed", type = int, help = "Seed for random number generation. If not set, a random seed will be used.")

def get_vae_args():
    parser = argparse.ArgumentParser(description = "Variational Auto-Encoder on aligned protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    basic_args(parser)

    parser.add_argument("--layer_sizes", type = int, default = [1500, 1500, 30, 100, 2000], nargs = "+", help = "Sizes of the hidden layers of the VAE, except the first and last, which will be inferred from the data argument. The smallest size is understood as the bottleneck size and will be the size of the output of the encoder, and the size of the input of the decoder.")
    parser.add_argument("--data", type = Path, default = Path("data/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file of sequences.")
    parser.add_argument("--val_ratio", type = float, default = 0.2, help = "What fraction of data to use for validation. Set to 0 to disable validation (patience will then work on training loss).")
    parser.add_argument("--dropout", type = float, default = 0.0, help = "Rate of dropout to apply to the encoder and decoder layers.")
    parser.add_argument("--layer_mod", type = str, default = "variational", choices = ["none", "variational"], help = "Layer modification on the decoder's linear layers.")
    parser.add_argument("--z_samples", type = int, default = 1, help = "How many latent variables to sample per batch point.")
    parser.add_argument("--data_sheet", type = str, default = "BLAT_ECOLX_Ranganathan2015", help = "Protein family data sheet in mutation_data.pickle.")
    parser.add_argument("--metric_column", type = str, default = "2500", help = "Metric column of sheet used for Spearman's Rho calculation.")
    parser.add_argument("--ensemble_count", type = int, default = 500, help = "How many samples of the model to use for evaluation as an ensemble.")
    parser.add_argument("--dictionary", action = "store_true", dest = "dictionary", default = False, help = "Enables the dictionary of the VAE.")
    parser.add_argument("--no_dictionary", action = "store_false", dest = "dictionary", default = True, help = "Disables the dictionary of the VAE.")
    parser.add_argument("--param_loss", action = "store_true", dest = "param_loss", default = True, help = "Enables the param_loss.")
    parser.add_argument("--no_param_loss", action = "store_false", dest = "param_loss", default = False, help = "Disables the param loss")
    parser.add_argument("--warm_up", type = int, default = 0, help = "Number of warm-up batches. Will affect the scale of global param loss during warm-up.")
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

def get_unirep_finetune_args():
    parser = argparse.ArgumentParser(description = "UniRep model on protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    basic_args(parser)

    # Data
    parser.add_argument("--data", type = Path, default = Path("data/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file of sequences.")
    parser.add_argument("--load_model", type = Path, default = Path("."), help = "The model to load before training. Can be omitted.")
    parser.add_argument("--save_model", type = Path, default = Path("results_unirep_finetuned/model.torch"), help = "The path to save the trained model to.")
    parser.add_argument("--val_ratio", type = float, default = 0.2, help = "What fraction of data to use for validation.")
    parser.add_argument("--data_sheet", type = str, default = "BLAT_ECOLX_Ranganathan2015", help = "Protein family data sheet in mutation_data.pickle.")
    parser.add_argument("--metric_column", type = str, default = "2500", help = "Metric column of sheet used for Spearman's Rho calculation.")
    parser.add_argument("--ensemble_count", type = int, default = 2000, help = "How many samples of the model to use for evaluation as an ensemble.")

    parser.add_argument("--embed_size", type = int, default = 10, help = "Size of the amino acid embedding.")
    parser.add_argument("--hidden_size", type = int, default = 512, help = "Size of the hidden state of the LSTM.")
    parser.add_argument("--num_layers", type = int, default = 1, help = "Number of layers of the LSTM.")

    args = parser.parse_args()

    args.train_ratio = 1 - args.val_ratio
    args.results_dir.mkdir(exist_ok = True)
    print_args(args)
    return args

def print_args(args):
    print("Arguments given:")
    for arg, value in args.__dict__.items():
        print(f"  {arg}: {value}")
    print("")
