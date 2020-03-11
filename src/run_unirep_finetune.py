from arguments import get_unirep_finetune_args
args = get_unirep_finetune_args()

import time
from pathlib import Path

import torch
from torch import optim

from models import UniRep
from data import VariableLengthProteinDataset, get_variable_length_protein_dataLoader, NUM_TOKENS, IUPAC_SEQ2IDX
from training import train_epoch, validate, readable_time, get_memory_usage, mutation_effect_prediction

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
    all_data = VariableLengthProteinDataset(args.data, device = device)
    train_length = int(len(all_data) * args.train_ratio)
    val_length = len(all_data) - train_length

    train_data, val_data = torch.utils.data.random_split(all_data, [train_length, val_length])

    train_loader = get_variable_length_protein_dataLoader(train_data, batch_size = args.batch_size)
    val_loader = get_variable_length_protein_dataLoader(val_data, batch_size = args.batch_size)
    print("Data loaded!")

    model = UniRep(NUM_TOKENS, IUPAC_SEQ2IDX["<pad>"], args.embed_size, args.hidden_size, args.num_layers).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters())

    if args.load_model.exists() and args.load_model.is_file():
        print(f"Loading saved model from {args.load_model}...")
        model.load_state_dict(torch.load(args.load_model, map_location = device))
        print(f"Model loaded.")

    best_val_loss = float("inf")
    patience = args.patience
    epoch = 0
    model_save_name = args.results_dir / Path("model.torch")
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss, train_metrics = train_epoch(epoch, model, optimizer, train_loader, args.log_interval, args.clip_grad_norm, args.clip_grad_value)
            val_loss, val_metrics = validate(epoch, model, val_loader)

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

        print('Computing mutation effect prediction correlation...')
        with torch.no_grad():
            if model_save_name.exists():
                model.load_state_dict(torch.load(model_save_name, map_location = device))
            cor = mutation_effect_prediction(model, args.data, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir)
        print(f'Spearman\'s Rho: {cor}')

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_val_loss:.5f}.\n")
        breakpoint()
