import os
import json
import torch
import torch.nn as nn
from config import Config
from dataset import YieldDataset, setup_training
from models import SyKCABModel
from train import evaluate_metrics
from utils import set_seed, export_results_to_csv


def load_hyperparameters(json_path: str) -> dict:
    """Loads saved hyperparameters from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    set_seed()
    print("Starting independent evaluation pipeline...")

    # ---------------------------------------------------------
    # IMPORTANT: Update these filenames to match your best saved
    # model after you run train.py for the first time.
    # ---------------------------------------------------------
    saved_model_name = r"D:\pytorch_test\SyFK-CapsNet\Pre-training weight\models_SyFK-CapsNet_2023(USA_Statistical_Data)\SyKCABModel_R2_0.8146.pth"
    saved_params_name = r"D:\pytorch_test\SyFK-CapsNet\Pre-training weight\models_SyFK-CapsNet_2023(USA_Statistical_Data)\SyKCABModel_R2_0.8146_params.json"

    model_weights_path = os.path.join(Config.SAVE_FOLDER, saved_model_name)
    hyperparams_path = os.path.join(Config.SAVE_FOLDER, saved_params_name)

    if not os.path.exists(model_weights_path) or not os.path.exists(hyperparams_path):
        print(f"Error: Model files not found in {Config.SAVE_FOLDER}.")
        print("Please ensure you have run main.py to train the model first, and update the filenames in evaluate.py.")
        return

    # Load hyperparameters
    hyperparams = load_hyperparameters(hyperparams_path)
    print(f"Loaded hyperparameters from: {saved_params_name}")

    # Initialize Dataset
    dataset = YieldDataset(
        data_folder=Config.DATA_FOLDER, start_year=Config.START_YEAR, end_year=Config.END_YEAR,
        months=Config.MONTHS, feature_cols=Config.FEATURE_COLS, rs_feature_cols=Config.RS_FEATURE_COLS,
        m_feature_cols=Config.M_FEATURE_COLS, label_col=Config.LABEL_COL, normalize=Config.NORMALIZE,
    )

    # Setup DataLoader (We only need the val_loader for evaluation)
    _, val_loader, _, _ = setup_training(dataset, Config.TEST_YEAR, hyperparams["BATCH_SIZE"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Reconstruct KAN parameters
    kan_params_for_model = {
        'num_grids': hyperparams["FASTKAN_NUM_GRIDS"],
        'grid_min': -3.0,
        'grid_max': 3.0,
        'use_base_update': hyperparams["USE_KAN_IN_PRIMARY_CAPS"],
        'spline_weight_init_scale': hyperparams["FASTKAN_INIT_SCALE"]
    }

    # Initialize the model architecture with loaded parameters
    model = SyKCABModel(
        rs_input_feature_dim=len(Config.RS_FEATURE_COLS),
        m_input_feature_dim=len(Config.M_FEATURE_COLS),
        time_steps=len(Config.MONTHS),
        transformer_d_model=hyperparams["TRANSFORMER_D_MODEL"],
        transformer_nhead=hyperparams["TRANSFORMER_NHEAD"],
        transformer_num_encoder_layers=hyperparams["TRANSFORMER_NUM_ENCODER_LAYERS"],
        transformer_dim_feedforward=hyperparams["TRANSFORMER_DIM_FEEDFORWARD"],
        transformer_dropout=hyperparams["TRANSFORMER_DROPOUT"],
        lstm_hidden_size=hyperparams["LSTM_HIDDEN_SIZE"],
        lstm_num_layers=hyperparams["LSTM_NUM_LAYERS"],
        lstm_bidirectional=hyperparams["LSTM_BIDIRECTIONAL"],
        num_rs_primary_caps=hyperparams["NUM_RS_PRIMARY_CAPS"],
        num_m_primary_caps=hyperparams["NUM_M_PRIMARY_CAPS"],
        num_fused_primary_caps=hyperparams["NUM_FUSED_PRIMARY_CAPS"],
        primary_cap_dim=hyperparams["PRIMARY_CAP_DIM"],
        num_digit_caps=hyperparams["NUM_DIGIT_CAPS"],
        digit_cap_dim=hyperparams["DIGIT_CAP_DIM"],
        routing_iterations=hyperparams["ROUTING_ITERATIONS"],
        kan_params=kan_params_for_model,
        use_kan_in_primary_caps=hyperparams["USE_KAN_IN_PRIMARY_CAPS"],
    ).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    criterion = nn.HuberLoss(delta=hyperparams["HUBER_DELTA"]).to(device)

    print("\nRunning evaluation on the test set...")
    _, final_r2, final_rmse, final_mae, final_preds, final_trues, final_location_infos = evaluate_metrics(
        model, val_loader, device, criterion
    )

    print("-" * 50)
    print("Standalone Evaluation Results:")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"RMSE:     {final_rmse:.4f}")
    print(f"MAE:      {final_mae:.4f}")
    print("-" * 50)

    # Export the results to a CSV file
    export_results_to_csv(
        final_location_infos,
        final_trues,
        final_preds,
        final_r2,
        Config.SAVE_FOLDER,
        filename_prefix="standalone_evaluation"
    )


if __name__ == "__main__":
    main()