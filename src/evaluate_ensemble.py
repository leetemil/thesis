from args import get_evaluate_ensemble_args
args = get_evaluate_ensemble_args()

from pathlib import Path
import glob

import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from models import VAE, UniRep, WaveNet, LossTransformer
from training import mutation_effect_prediction

# Device
if args.device == "cuda" and not torch.cuda.is_available():
    raise ValueError("CUDA device specified, but CUDA is not available. Use --device cpu.")
device = torch.device(args.device)
try:
    device_name = torch.cuda.get_device_name()
except:
    device_name = "CPU"

print(f"Using device: {device_name}")

# Construct models
models = []
for directory in args.model_directories:
    model_list = glob.glob(str(directory / Path("*.torch")))
    for model_path in model_list:
        load_dict = torch.load(model_path, map_location = device)

        name = load_dict["name"]
        state_dict = load_dict["state_dict"]
        args_dict = load_dict["args_dict"]

        if name == "VAE":
            model_type = VAE
        elif name == "UniRep":
            model_type = UniRep
        elif name == "WaveNet":
            model_type = WaveNet
        elif name == "Transformer":
            model_type = LossTransformer
        else:
            raise ValueError("Unrecognized model name.")

        args_dict["use_bayesian"] = args_dict.pop("bayesian") # use this if you get bayesian keyword error for wavenet
        model = model_type(**args_dict).to(device)
        model.load_state_dict(state_dict)

        models.append(model)

with torch.no_grad():
    # Evaluate on mutation effect
    scores = mutation_effect_prediction(models[0], args.data, args.query_protein, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir, return_scores = True)

    acc_m_logp = 0
    acc_wt_logp = 0
    for model in models:
        m_logp, wt_logp = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir, return_logps = True)

        acc_m_logp += m_logp
        acc_wt_logp += wt_logp

    ensemble_m_logp = acc_m_logp / len(models)
    ensemble_wt_logp = acc_wt_logp / len(models)

    predictions = ensemble_m_logp - ensemble_wt_logp

    plt.scatter(predictions.cpu(), scores)
    plt.title("Correlation")
    plt.xlabel("$\\Delta$-elbo")
    plt.ylabel("Experimental value")
    plt.savefig(args.results_dir / Path("Correlation_scatter.png"))

    cor, _ = spearmanr(scores, predictions.cpu())
    print(f'Ensemble Spearman\'s Rho over {len(models)} models: {cor:5.3f}')
