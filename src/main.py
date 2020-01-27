# First, command-line arguments
from arguments import args

import time
from pathlib import Path
# import logging
# logging.basicConfig(level = args.log_level, format = "[{levelname}] {asctime} {module}:{funcName}:{lineno}: {message}", style = "{")

import torch
from torch import optim
from torch.utils.data import random_split

from vae import VAE
from protein_data import ProteinDataset, get_protein_dataloader
from training import train, validate
from utils import readable_time
from visualize import plot_data

if __name__ == "__main__":
    # Argument postprocessing
    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Device
    if args.device == "cuda" and not torch.cuda.is_available:
        raise ValueError("CUDA device specified, but CUDA is not available. Use --device cpu.")
    device = torch.device(args.device)
    print(f"Using device: {device.type.upper()}")

    # Load data
    print(f"Loading data from {args.data}...")
    protein_dataset = ProteinDataset(args.data, device)
    data_len = protein_dataset[0][0].size(0)

    # Split into train/validation
    train_length = int(args.train_ratio * len(protein_dataset))
    val_length = len(protein_dataset) - train_length
    train_data, val_data = random_split(protein_dataset, [train_length, val_length])

    # Construct dataloaders for batches
    train_loader = get_protein_dataloader(train_data, batch_size = args.batch_size, shuffle = True)
    val_loader = get_protein_dataloader(val_data, batch_size = args.batch_size)
    print("Data loaded!")

    # Define model and optimizer
    model = VAE([data_len] + args.layer_sizes).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters())

    # Train, validate, save
    best_val_loss = float("inf")
    patience = args.patience
    try:
        plot_data(args.results_dir / Path(f"epoch_0_val_loss_inf.pdf"), model, protein_dataset, args.batch_size),
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss = train(epoch, model, optimizer, VAE.vae_loss, train_loader, args.log_interval)
            val_loss = validate(epoch, model, VAE.vae_loss, val_loader)

            print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} Validation loss: {val_loss:.5f} Time: {readable_time(time.time() - start_time)} Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GiB")
            torch.cuda.reset_max_memory_allocated()

            improved = val_loss < best_val_loss

            if args.visualize == "always" or (args.visualize == "improvement", improved):
                plot_data(args.results_dir / Path(f"epoch_{epoch}_val_loss_{best_val_loss:.5f}.pdf"), model, protein_dataset, args.batch_size),

            if improved:
                # If model improved, save the model
                model_save_name = args.results_dir / Path("model.torch")
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

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_val_loss:.5f}.\n")
