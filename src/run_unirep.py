from arguments import get_unirep_args
args = get_unirep_args()

import time
from pathlib import Path

import torch
from torch import optim

from unirep import UniRep
from protein_data import IterProteinDataset, get_iter_protein_DataLoader, NUM_TOKENS, IUPAC_SEQ2IDX
from utils import readable_time, get_memory_usage
from training import train_epoch, validate
from visualize import plot_data

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
    print(f"Using device: {device.type.upper()}")

    # Load data
    train_data = IterProteinDataset(args.train_data, device = device)
    validation_data = IterProteinDataset(args.validation_data, device = device)

    train_loader = get_iter_protein_DataLoader(train_data, batch_size = args.batch_size)
    val_loader = get_iter_protein_DataLoader(validation_data, batch_size = args.batch_size)
    print("Data loaded!")

    model = UniRep(NUM_TOKENS, IUPAC_SEQ2IDX["<pad>"], args.embed_size, args.hidden_size, args.num_layers).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters())

    model_save_name = args.results_dir / Path("model.torch")
    if model_save_name.exists():
        print(f"Loading saved model from {model_save_name}...")
        model.load_state_dict(torch.load(model_save_name, map_location = device))
        print(f"Model loaded.")

    best_val_loss = float("inf")
    patience = args.patience
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss = train_epoch(epoch, model, optimizer, train_loader, args.log_interval)
            val_loss = validate(epoch, model, val_loader)

            print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} Validation loss: {val_loss:.5f} Time: {readable_time(time.time() - start_time)} Memory: {get_memory_usage(device):.2f}GiB")

            improved = val_loss < best_val_loss

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

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_val_loss:.5f}.\n")
        breakpoint()
