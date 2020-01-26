# First, command-line arguments
from arguments import args

import time
# import logging
# logging.basicConfig(level = args.log_level, format = "[{levelname}] {asctime} {module}:{funcName}:{lineno}: {message}", style = "{")

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from vae import VAE
from protein_data import ProteinDataset
from training import train, validate
from utils import readable_time

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

    # Split into train/validation
    train_length = int(args.train_ratio * len(protein_dataset))
    val_length = len(protein_dataset) - train_length
    train_data, val_data = random_split(protein_dataset, [train_length, val_length])

    # Construct dataloaders for batches
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = args.batch_size)
    print("Data loaded!")

    # Define model and optimizer
    model = VAE(args.layer_sizes).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters())

    # Train, validate, save
    best_val_loss = float("inf")
    patience = args.patience
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train(epoch, model, optimizer, VAE.vae_loss, train_loader, args.log_interval)
        val_loss = validate(epoch, model, VAE.vae_loss, val_loader)

        print(f"Summary epoch: {epoch} Train loss: {train_loss:.4f} Validation loss: {val_loss:.4f} Time: {readable_time(time.time() - start_time)}")

        # If save path was specified, and model improved, save the model
        if args.save_path and val_loss <= best_val_loss:
            torch.save(model.state_dict(), args.save_path)
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saved model to: {args.save_path}")
            best_val_loss = val_loss
            patience = args.patience
        # If save path and patience was specified, and model has not improved, decrease patience and possibly stop
        elif args.save_path and args.patience:
            patience -= 1
            if patience == 0:
                print(f"Model has not improved for {args.patience} epochs. Stopping training.")
                break
