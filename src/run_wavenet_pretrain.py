from args import get_wavenet_args
args = get_wavenet_args(pretrain = True)

import time
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models import WaveNet
from data import IterProteinDataset, get_variable_length_protein_dataLoader, NUM_TOKENS
from training import train_batch, validate, readable_time, get_memory_usage, mutation_effect_prediction
from visualization import plot_spearman


if __name__ == "__main__" or __name__ == "__console__":
    # Argument postprocessing
    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device specified, but CUDA is not available. Use --device cpu.")
    device = torch.device(args.device)
    try:
        device_name = torch.cuda.get_device_name()
    except:
        device_name = "CPU"

    print(f"Using device: {device_name}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Load data
    train_data = IterProteinDataset(args.train_data, device = device)
    validation_data = IterProteinDataset(args.validation_data, device = device)
    val_len = len(validation_data)
    train_seqs_per_epoch = val_len * 9

    train_loader = get_variable_length_protein_dataLoader(train_data, batch_size = args.batch_size)
    val_loader = get_variable_length_protein_dataLoader(validation_data, batch_size = args.batch_size)
    print("Data loaded!")

    total_samples = 39_069_211 # magic number

    model = WaveNet(
        input_channels = NUM_TOKENS,
        residual_channels = args.residual_channels,
        out_channels = NUM_TOKENS,
        stacks = args.stacks,
        layers_per_stack = args.layers,
        total_samples = total_samples,
        l2_lambda = args.L2,
		bias = args.bias,
        dropout = args.dropout,
        use_bayesian = args.bayesian,
        backwards = args.backwards
	).to(device)

    print(model.summary())
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    if args.anneal_learning_rates:
        T_0 = 1
        T_mult = 2
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    else:
        scheduler = None

    model_save_name = args.results_dir / Path("model.torch")
    if model_save_name.exists():
        print(f"Loading saved model from {model_save_name}...")
        model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])
        print(f"Model loaded.")

    best_val_loss = float("inf")
    ensemble_count = args.ensemble_count if args.bayesian else 0
    patience = args.patience
    seqs_processed = 0
    acc_train_loss = 0
    train_loss_count = 0
    start_time = time.time()
    epoch = 0

    improved_epochs = []
    spearman_rhos = []
    spearman_name = args.results_dir / Path("spearman_rhos.png")
    total_batches = total_samples // args.batch_size # used for annealing

    print_every_samples = 250_000
    print_seqs_count = 0

    try:
        stop = False
        while not stop:
            for batch_idx, xb in enumerate(train_loader):
                batch_size, batch_train_loss, batch_metrics_dict = train_batch(model, optimizer, xb, args.clip_grad_norm, args.clip_grad_value, scheduler=scheduler, epoch=epoch, batch = batch_idx, num_batches=total_batches)

                seqs_processed += batch_size
                acc_train_loss += batch_train_loss
                train_loss_count += 1

                print_seqs_count += batch_size
                if print_seqs_count >= print_every_samples:
                    print(f'Processed {seqs_processed} out of {total_samples}.')
                    print_seqs_count = 0

                if seqs_processed >= train_seqs_per_epoch:
                    epoch += 1
                    train_loss = acc_train_loss / train_loss_count
                    seqs_processed = 0
                    acc_train_loss = 0
                    train_loss_count = 0
                    val_loss, _ = validate(epoch, model, val_loader)

                    improved = val_loss < best_val_loss

                    if improved:
                        # If model improved, save the model
                        model.save(model_save_name)
                        print(f"Validation loss improved from {best_val_loss:.5f} to {val_loss:.5f}. Saved model to: {model_save_name}")
                        best_val_loss = val_loss
                        patience = args.patience

                        with torch.no_grad():
                            rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, 100, args.results_dir, savefig = False)

                        spearman_rhos.append(rho)
                        improved_epochs.append(epoch)
                        plot_spearman(args.data, spearman_name, improved_epochs, spearman_rhos)

                    elif args.patience:
                        # If save path and patience was specified, and model has not improved, decrease patience and possibly stop
                        patience -= 1
                        if patience == 0:
                            print(f"Model has not improved for {args.patience} epochs. Stopping training. Best validation loss achieved was: {best_val_loss:.5f}.")
                            stop = True
                            break

                    if epoch >= args.epochs:
                        stop = True
                        break

                    print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} Validation loss: {val_loss:.5f} Time: {readable_time(time.time() - start_time)} Memory: {get_memory_usage(device):.2f}GiB", end = "\n" if improved else "\n\n")

                    if improved:
                        print(f"Spearman\'s Rho: {rho:.3f}", end = "\n\n")

                    start_time = time.time()

        print('Computing mutation effect prediction correlation...')

        with torch.no_grad():
            if model_save_name.exists():
                model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])

            rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, ensemble_count, args.results_dir)

        print(f'Spearman\'s Rho: {rho:.3f}')

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_val_loss:.5f}.\n")
        breakpoint()
