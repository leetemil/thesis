from args import get_transformer_args
args = get_transformer_args()

import time
from pathlib import Path

import torch
from torch import optim

from models import LossTransformer
from data import VariableLengthProteinDataset, get_variable_length_protein_dataLoader, NUM_TOKENS, IUPAC_SEQ2IDX
from training import train_epoch, validate, readable_time, get_memory_usage, mutation_effect_prediction
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

    # Load data
    all_data = VariableLengthProteinDataset(args.data, device = device, remove_gaps = args.remove_gaps, max_len = args.max_len)
    train_length = int(len(all_data) * args.train_ratio)
    val_length = len(all_data) - train_length

    train_data, val_data = torch.utils.data.random_split(all_data, [train_length, val_length])

    train_loader = get_variable_length_protein_dataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    val_loader = get_variable_length_protein_dataLoader(val_data, batch_size = args.batch_size)
    print("Data loaded!")

    model = LossTransformer(
		num_tokens = NUM_TOKENS,
		d_model = args.d_model,
		nhead = args.nhead,
		num_encoder_layers = args.num_encoder_layers,
		num_decoder_layers = args.num_decoder_layers,
		dim_feedforward = args.dim_feedforward,
        dropout = args.dropout,
        embed = args.embed,
        max_len = all_data[0].size(0),
	).to(device)

    print(model.summary())
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.L2)

    model_save_name = args.results_dir / Path("model.torch")
    if model_save_name.exists():
        print(f"Loading saved model from {model_save_name}...")
        model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])
        print(f"Model loaded.")

    best_loss = float("inf")
    patience = args.patience
    improved_epochs = []
    spearman_rhos = []
    spearman_name = args.results_dir / Path("spearman_rhos.png")
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss, train_metrics = train_epoch(epoch, model, optimizer, train_loader, args.log_interval, args.clip_grad_norm, args.clip_grad_value)

            if args.val_ratio > 0:
                val_loss, val_metrics = validate(epoch, model, val_loader)
                loss_str = "Validation"
                loss_value_str = f"{val_loss:.5f}"
                val_str = f"{loss_str} loss: {loss_value_str} "
                improved = val_loss < best_loss

            else:
                loss_str = "Training"
                loss_value_str = f"{train_loss:.5f}"
                val_str = ""
                improved = train_loss < best_loss

            rho_str = ""
            if improved:
                # If model improved, save the model
                model.save(model_save_name)
                print(f"{loss_str} loss improved from {best_loss:.5f} to {loss_value_str}. Saved model to: {model_save_name}")
                best_loss = val_loss if args.val_ratio > 0 else train_loss
                patience = args.patience

                with torch.no_grad():
                    rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, 0, args.results_dir, savefig = False)
                spearman_rhos.append(rho)
                improved_epochs.append(epoch)
                plot_spearman(spearman_name, improved_epochs, spearman_rhos)
                rho_str = f"Spearman's rho: {rho:.3f} "
            elif args.patience:
                # If save path and patience was specified, and model has not improved, decrease patience and possibly stop
                patience -= 1
                if patience == 0:
                    print(f"Model has not improved for {args.patience} epochs. Stopping training. Best {loss_str.lower()} loss achieved was: {best_loss:.5f}.")
                    break

            print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} {val_str}{rho_str}Time: {readable_time(time.time() - start_time)} Memory: {get_memory_usage(device):.2f}GiB", end = "\n\n")

        print("Computing mutation effect prediction correlation...")
        with torch.no_grad():
            if model_save_name.exists():
                model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])
            rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, 0, args.results_dir)
        print(f"Spearman's Rho: {rho}")

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_loss:.5f}.\n")
        breakpoint()
