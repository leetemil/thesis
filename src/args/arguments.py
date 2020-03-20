import argparse
from pathlib import Path
from datetime import datetime
from typing import Union

def basic_args(parser):
    parser.add_argument("-e", "--epochs", type = int, default = 10000, help = "Maximum number of epochs to train (patience may cause fewer epochs to be run).")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 0.001, help = "Learning rate for Adam optimizer.")
    parser.add_argument("-bs", "--batch_size", type = int, default = 128, help = "Input batch size for training.")
    parser.add_argument("-cn", "--clip_grad_norm", type = lambda x: None if x is None else float(x), default = None, help = "Gradient will be clipped to this norm. Disabled if None.")
    parser.add_argument("-cv", "--clip_grad_value", type = lambda x: None if x is None else float(x), default = None, help = "Gradient values will be clipped to this value. Disabled if None.")
    parser.add_argument("-l2", "--L2", type = float, default = 0, help = "L2 regularization added as weight decay to Adam optimizer.")
    parser.add_argument("-d", "--device", type = str, default = "cuda", choices = ["cpu", "cuda"], help = "Which device to use (CPU or CUDA for GPU).")
    parser.add_argument("-p", "--patience", type = int, default = 500, help = "Training will stop if the model does not improve on the validation set for this many epochs.")
    parser.add_argument("-li", "--log_interval", type = lambda x: x if x == "batch" else float(x), default = 0, help = "How many seconds to wait between training status logging. 0 to disable loading bar progress. \"batch\" for log at every batch.")
    parser.add_argument("-r", "--results_dir", type = Path, default = Path(f"{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}"), help = "Directory name to save results under. Will be saved under results/results_dir.")
    parser.add_argument("-s", "--seed", type = int, help = "Seed for random number generation. If not set, a random seed will be used.")

def mutation_effect_prediction_args(parser):
    parser.add_argument("-qp", "--query_protein", type = Path, default = Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file containing the query protein sequence for mutation effect prediction.")
    parser.add_argument("-ds", "--data_sheet", type = str, default = "BLAT_ECOLX_Ranganathan2015", help = "Protein family data sheet in mutation_data.pickle.")
    parser.add_argument("-mc", "--metric_column", type = str, default = "2500", help = "Metric column of sheet used for Spearman's Rho calculation.")

def get_vae_args():
    parser = argparse.ArgumentParser(description = "Variational Auto-Encoder on aligned protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = "@")
    basic_args(parser)

    parser.add_argument("-ls", "--layer_sizes", type = int, default = [1500, 1500, 30, 100, 2000], nargs = "+", help = "Sizes of the hidden layers of the VAE, except the first and last, which will be inferred from the data argument. The smallest size is understood as the bottleneck size and will be the size of the output of the encoder, and the size of the input of the decoder.")
    parser.add_argument("--data", type = Path, default = Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file of sequences.")
    parser.add_argument("-vr", "--val_ratio", type = float, default = 0.0, help = "What fraction of data to use for validation. Set to 0 to disable validation (patience will then work on training loss).")
    parser.add_argument("-do", "--dropout", type = float, default = 0.0, help = "Rate of dropout to apply to the encoder and decoder layers.")
    parser.add_argument("-lm", "--layer_mod", type = str, default = "variational", choices = ["none", "variational"], help = "Layer modification on the decoder's linear layers.")
    parser.add_argument("-zs", "--z_samples", type = int, default = 1, help = "How many latent variables to sample per batch point.")
    mutation_effect_prediction_args(parser)
    parser.add_argument("-ec", "--ensemble_count", type = int, default = 500, help = "How many samples of the model to use for evaluation as an ensemble.")
    parser.add_argument("-dict", "--dictionary", action = "store_true", dest = "dictionary", default = False, help = "Enables the dictionary of the VAE.")
    parser.add_argument("-no_dict", "--no_dictionary", action = "store_false", dest = "dictionary", default = True, help = "Disables the dictionary of the VAE.")
    parser.add_argument("-pl", "--param_loss", action = "store_true", dest = "param_loss", default = True, help = "Enables the param_loss.")
    parser.add_argument("-no_pl", "--no_param_loss", action = "store_false", dest = "param_loss", default = False, help = "Disables the param loss")
    parser.add_argument("-wu", "--warm_up", type = int, default = 0, help = "Number of warm-up batches. Will affect the scale of global param loss during warm-up.")
    parser.add_argument("-vi", "--visualize_interval", type = str, default = "improvement", choices = ["always", "improvement", "never"], help = "Visualize the output at every epoch (always), only at validation loss improvement or never.")
    parser.add_argument("-vs", "--visualize_style", type = str, default = "save", choices = ["save", "show", "both"], help = "Save or show the visualization, or both.")
    parser.add_argument("-ft", "--figure_type", type = str, default = ".png", help = "Filetype of the visualization figures.")

    args = parser.parse_args()

    # Argument postprocessing
    args.train_ratio = 1 - args.val_ratio
    args.results_dir = Path("results") / args.results_dir
    args.results_dir.mkdir(exist_ok = True)
    print_args(args)
    return args

def get_unirep_args():
    parser = argparse.ArgumentParser(description = "UniRep model on protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    basic_args(parser)

    # Data
    parser.add_argument("-td", "train_data", type = Path, help = "Fasta input file for training.")
    parser.add_argument("-vd", "validation_data", type = Path, help = "Fasta input file for validation.")

    parser.add_argument("-es", "--embed_size", type = int, default = 10, help = "Size of the amino acid embedding.")
    parser.add_argument("-hs", "--hidden_size", type = int, default = 512, help = "Size of the hidden state of the LSTM.")
    parser.add_argument("-nl", "--num_layers", type = int, default = 1, help = "Number of layers of the LSTM.")

    args = parser.parse_args()

    args.results_dir = Path("results") / args.results_dir
    args.results_dir.mkdir(exist_ok = True)
    print_args(args)
    return args

def get_unirep_finetune_args():
    parser = argparse.ArgumentParser(description = "UniRep model on protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    basic_args(parser)

    # Data
    parser.add_argument("--data", type = Path, default = Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file of sequences.")
    parser.add_argument("-lm", "--load_model", type = Path, default = Path("."), help = "The model to load before training. Can be omitted.")
    parser.add_argument("-vr", "--val_ratio", type = float, default = 0.2, help = "What fraction of data to use for validation.")
    mutation_effect_prediction_args(parser)
    parser.add_argument("-es", "--embed_size", type = int, default = 10, help = "Size of the amino acid embedding.")
    parser.add_argument("-hs", "--hidden_size", type = int, default = 512, help = "Size of the hidden state of the LSTM.")
    parser.add_argument("-nl", "--num_layers", type = int, default = 1, help = "Number of layers of the LSTM.")

    args = parser.parse_args()

    args.train_ratio = 1 - args.val_ratio
    args.results_dir = Path("results") / args.results_dir
    args.results_dir.mkdir(exist_ok = True)
    print_args(args)
    return args

def get_wavenet_args():
    parser = argparse.ArgumentParser(description = "WaveNet model on protein sequences", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    basic_args(parser)

    # Data
    parser.add_argument("--data", type = Path, default = Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file of sequences.")
    parser.add_argument("-vr", "--val_ratio", type = float, default = 0.2, help = "What fraction of data to use for validation.")
    mutation_effect_prediction_args(parser)
    parser.add_argument("-rc", "--residual_channels", type = int, default = 48, help = "Number of channels in the residual layers.")
    parser.add_argument("-gc", "--gate_channels", type = int, default = 48, help = "Number of channels given to each non-linear gate of the residual layers.")
    parser.add_argument("-sc", "--skip_out_channels", type = int, default = 48, help = "Number of output channels of the skip connections.")
    parser.add_argument("-ss", "--stacks", type = int, default = 6, help = "Number of stacks of dilated convolutions.")
    parser.add_argument("-ls", "--layers", type = int, default = 9, help = "Number of layers for each stack.")
    parser.add_argument("-b", "--bias", action = "store_true", dest = "bias", default = True, help = "Enables bias.")
    parser.add_argument("-no_b", "--no_bias", action = "store_false", dest = "bias", default = False, help = "Disables bias.")
    parser.add_argument("-do", "--dropout", type = float, default = 0.5, help = "Rate of dropout to apply between layers.")

    args = parser.parse_args()

    args.train_ratio = 1 - args.val_ratio
    args.results_dir = Path("results") / args.results_dir
    args.results_dir.mkdir(exist_ok = True)
    print_args(args)
    return args

def print_args(args):
    print("Arguments given:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("")
