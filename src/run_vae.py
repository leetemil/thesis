# First, command-line arguments
from arguments import get_vae_args
args = get_vae_args()

import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import random_split

from vae import VAE
from protein_data import ProteinDataset, get_protein_dataloader, NUM_TOKENS, get_datasets
from training import train_epoch, validate
from utils import readable_time, get_memory_usage
from visualize import plot_data, plot_loss, plot_spearman
from mutation_data import mutation_effect_prediction

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
    all_data, train_data, val_data = get_datasets(args.data, device, args.train_ratio)

    # Construct dataloaders for batches
    train_loader = get_protein_dataloader(train_data, batch_size = args.batch_size, shuffle = True)
    val_loader = get_protein_dataloader(val_data, batch_size = args.batch_size)
    print("Data loaded!")

    # Define model and optimizer
    data_size = all_data[0][0].size(-1) * NUM_TOKENS
    model = VAE(
        [data_size] + args.layer_sizes + [data_size],
        NUM_TOKENS,
        z_samples = args.z_samples,
        dropout = args.dropout,
        layer_mod = args.layer_mod,
        use_param_loss = args.param_loss,
        use_dictionary = args.dictionary,
        warm_up = args.warm_up
    ).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters())

    model_save_name = args.results_dir / Path("model.torch")
    if model_save_name.exists():
        print(f"Loading saved model from {model_save_name}...")
        model.load_state_dict(torch.load(model_save_name, map_location = device))
        print(f"Model loaded.")

    # Train, validate, save
    show = False
    save = False
    if args.visualize_style == "show" or args.visualize_style == "both":
        show = True
    if args.visualize_style == "save" or args.visualize_style == "both":
        save = True

    best_val_loss = float("inf")
    patience = args.patience
    try:
        epochs = []
        train_nll_losses = []
        train_kld_losses = []
        train_param_klds = []
        train_total_losses = []
        val_nll_losses = []
        val_kld_losses = []
        val_param_klds = []
        val_total_losses = []
        spearman_rhos = []
        if args.visualize_interval != "never":
            plot_data(args.results_dir / Path(f"epoch_0_val_loss_inf.png") if save else None, args.figure_type, model, all_data, args.batch_size, show = show)
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss, train_metrics = train_epoch(epoch, model, optimizer, train_loader, args.log_interval, args.clip_grad_norm, args.clip_grad_value)
            val_loss, val_metrics = validate(epoch, model, val_loader)

            print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} Validation loss: {val_loss:.5f} Time: {readable_time(time.time() - start_time)} Memory: {get_memory_usage(device):.2f}GiB")

            epochs.append(epoch)
            train_nll_losses.append(train_metrics["nll_loss"])
            train_kld_losses.append(train_metrics["kld_loss"])
            train_param_klds.append(train_metrics["param_kld"])
            train_total_losses.append(train_loss)
            val_nll_losses.append(val_metrics["nll_loss"])
            val_kld_losses.append(val_metrics["kld_loss"])
            val_param_klds.append(val_metrics["param_kld"])
            val_total_losses.append(val_loss)
            train_name = args.results_dir / Path("Train_losses.png")
            plot_loss(epochs, train_nll_losses, train_kld_losses, train_param_klds, train_total_losses, val_nll_losses, val_kld_losses, val_param_klds, val_total_losses, train_name, figure_type = args.figure_type, show = show)

            improved = val_loss < best_val_loss
            if args.visualize_interval == "always" or (args.visualize_interval == "improvement" and improved):
                with torch.no_grad():
                    rho = mutation_effect_prediction(model, args.data, args.data_sheet, args.metric_column, device, 500, args.results_dir, savefig = False)

                    spearman_rhos.append(rho)

                name = args.results_dir / Path(f"epoch_{epoch}_val_loss_{val_loss:.5f}.png") if save else None
                plot_data(name, args.figure_type, model, all_data, rho, args.batch_size, show = show)

            if improved:
                # If model improved, save the model
                torch.save(model.state_dict(), model_save_name)
                print(f"Validation loss improved from {best_val_loss:.5f} to {val_loss:.5f}. Saved model to: {model_save_name}")
                best_val_loss = val_loss
                patience = args.patience
            elif args.patience:
                # If save path and patience was specified, and model has not improved, decrease patience and possibly stop
                patience -= 1
                if patience == 0:
                    print(f"Model has not improved for {args.patience} epochs. Stopping training. Best validation loss achieved was: {best_val_loss:.5f}.")
                    break

            print("")

        print('Computing mutation effect prediction correlation...')
        with torch.no_grad():
            if model_save_name.exists():
                model.load_state_dict(torch.load(model_save_name, map_location = device))
            cor = mutation_effect_prediction(model, args.data, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir)
        print(f'Spearman\'s Rho: {cor}')

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_val_loss:.5f}.\n")
        breakpoint()
    finally:
        spearman_name = args.results_dir / Path("spearman_rhos.png")
        plot_spearman(spearman_name, spearman_rhos)

        train_name = args.results_dir / Path("Train_losses.png")
        plot_loss(epochs, train_nll_losses, train_kld_losses, train_param_klds, train_total_losses, val_nll_losses, val_kld_losses, val_param_klds, val_total_losses, train_name, figure_type = args.figure_type, show = show)
