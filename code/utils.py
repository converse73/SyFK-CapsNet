import os
import json
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List
from config import Config

def set_seed(seed: int = Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def squash(input_tensor: torch.Tensor, dim: int = -1, epsilon: float = 1e-7) -> torch.Tensor:
    """Squash activation function for Capsule Networks."""
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale_factor = squared_norm / (1. + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale_factor * unit_vector

def save_params_to_json(params_dict: Dict, filepath: str):
    """Saves hyperparameters to a JSON file."""
    serializable_params = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params_dict.items()}
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    print(f"Hyperparameters saved to: {filepath}")

def export_results_to_csv(location_infos: List[str], actual_yields: np.ndarray, estimated_yields: np.ndarray,
                          r2_score: float, save_folder: str, filename_prefix: str = "test_predictions"):
    """
    Exports the actual and estimated yields of the test set to a CSV file.
    """
    state_ansis = []
    counties = []
    county_ansis = []

    for loc_str in location_infos:
        parts = loc_str.split('|')
        state_ansis.append(parts[0])
        counties.append(parts[1])
        county_ansis.append(parts[2])

    results_df = pd.DataFrame({
        'State ANSI': state_ansis,
        'County': counties,
        'County ANSI': county_ansis,
        'Actual Yield': actual_yields,
        'Estimate Yield': estimated_yields
    })

    filename = os.path.join(save_folder, f"{filename_prefix}_R2_{r2_score:.4f}.csv")
    results_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Test set estimation results saved to: {filename}")