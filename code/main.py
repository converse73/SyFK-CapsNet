import os
import torch
import torch.nn as nn
import traceback
from config import Config
from utils import set_seed, export_results_to_csv
from dataset import YieldDataset, setup_training
from models import SyKCABModel
from train import train_and_validate_final, evaluate_metrics


def main():
    set_seed()
    print("Program started (using SyK-CapsNet model with KAN enhanced collaborative fusion)...")
    if not os.path.exists(Config.SAVE_FOLDER):
        os.makedirs(Config.SAVE_FOLDER)
    print(f"Model save directory: {Config.SAVE_FOLDER}")

    current_batch_size = #Provided in the paper
    current_learning_rate = #Provided in the paper
    current_lstm_hidden_size = #Provided in the paper
    current_fastkan_num_grids = #Provided in the paper
    current_fastkan_grid_min = #Provided in the paper
    current_fastkan_grid_max = #Provided in the paper
    current_fastkan_init_scale = #Provided in the paper
    current_huber_delta = #Provided in the paper

    current_num_rs_primary_caps = #Provided in the paper
    current_num_m_primary_caps = #Provided in the paper
    current_num_fused_primary_caps = #Provided in the paper

    current_primary_cap_dim = #Provided in the paper
    current_num_digit_caps = #Provided in the paper
    current_digit_cap_dim = #Provided in the paper
    current_routing_iterations = #Provided in the paper

    current_transformer_d_model = #Provided in the paper
    current_transformer_nhead = #Provided in the paper
    current_transformer_num_encoder_layers = #Provided in the paper
    current_transformer_dim_feedforward = #Provided in the paper
    current_transformer_dropout = #Provided in the paper
    current_lstm_num_layers = #Provided in the paper
    current_lstm_bidirectional = #Provided in the paper

    full_hyperparams_to_save = {
        "BATCH_SIZE": current_batch_size, "LEARNING_RATE": current_learning_rate,
        "LSTM_HIDDEN_SIZE": current_lstm_hidden_size,
        "NUM_RS_PRIMARY_CAPS": current_num_rs_primary_caps,
        "NUM_M_PRIMARY_CAPS": current_num_m_primary_caps,
        "NUM_FUSED_PRIMARY_CAPS": current_num_fused_primary_caps,
        "PRIMARY_CAP_DIM": current_primary_cap_dim,
        "NUM_DIGIT_CAPS": current_num_digit_caps,
        "DIGIT_CAP_DIM": current_digit_cap_dim,
        "ROUTING_ITERATIONS": current_routing_iterations,
        "FASTKAN_NUM_GRIDS": current_fastkan_num_grids, "FASTKAN_INIT_SCALE": current_fastkan_init_scale,
        "USE_KAN_IN_PRIMARY_CAPS": Config.USE_KAN_IN_PRIMARY_CAPS, "HUBER_DELTA": current_huber_delta,
        "TRANSFORMER_D_MODEL": current_transformer_d_model, "TRANSFORMER_NHEAD": current_transformer_nhead,
        "TRANSFORMER_NUM_ENCODER_LAYERS": current_transformer_num_encoder_layers,
        "TRANSFORMER_DIM_FEEDFORWARD": current_transformer_dim_feedforward,
        "TRANSFORMER_DROPOUT": current_transformer_dropout,
        "LSTM_NUM_LAYERS": current_lstm_num_layers, "LSTM_BIDIRECTIONAL": current_lstm_bidirectional,
        "START_YEAR": Config.START_YEAR, "END_YEAR": Config.END_YEAR,
        "TEST_YEAR": Config.TEST_YEAR, "SEED": Config.SEED, "NORMALIZE": Config.NORMALIZE,
    }
    kan_params_for_model = {
        'num_grids': current_fastkan_num_grids, 'grid_min': current_fastkan_grid_min,
        'grid_max': current_fastkan_grid_max, 'use_base_update': Config.FASTKAN_USE_BASE_UPDATE,
        'spline_weight_init_scale': current_fastkan_init_scale
    }

    try:
        dataset = YieldDataset(
            data_folder=Config.DATA_FOLDER, start_year=Config.START_YEAR, end_year=Config.END_YEAR,
            months=Config.MONTHS, feature_cols=Config.FEATURE_COLS, rs_feature_cols=Config.RS_FEATURE_COLS,
            m_feature_cols=Config.M_FEATURE_COLS, label_col=Config.LABEL_COL, normalize=Config.NORMALIZE,
        )
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
        if len(dataset) == 0: return
    except Exception as e:
        print(f"Error loading dataset: {e}");
        traceback.print_exc();
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    try:
        train_loader, val_loader, _, _ = setup_training(
            dataset, Config.TEST_YEAR, current_batch_size
        )
    except Exception as e:
        print(f"Error setting up DataLoader: {e}");
        traceback.print_exc();
        return

    try:
        model = SyKCABModel(
            rs_input_feature_dim=len(Config.RS_FEATURE_COLS),
            m_input_feature_dim=len(Config.M_FEATURE_COLS),
            time_steps=len(Config.MONTHS),
            transformer_d_model=current_transformer_d_model,
            transformer_nhead=current_transformer_nhead,
            transformer_num_encoder_layers=current_transformer_num_encoder_layers,
            transformer_dim_feedforward=current_transformer_dim_feedforward,
            transformer_dropout=current_transformer_dropout,
            lstm_hidden_size=current_lstm_hidden_size,
            lstm_num_layers=current_lstm_num_layers,
            lstm_bidirectional=current_lstm_bidirectional,
            num_rs_primary_caps=current_num_rs_primary_caps,
            num_m_primary_caps=current_num_m_primary_caps,
            num_fused_primary_caps=current_num_fused_primary_caps,
            primary_cap_dim=current_primary_cap_dim,
            num_digit_caps=current_num_digit_caps,
            digit_cap_dim=current_digit_cap_dim,
            routing_iterations=current_routing_iterations,
            kan_params=kan_params_for_model,
            use_kan_in_primary_caps=Config.USE_KAN_IN_PRIMARY_CAPS,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nSyK-CapsNet Model Parameters: Total {total_params:,}, Trainable {trainable_params:,}")
        print(f"Model size: {total_params * 4 / (1024 ** 2):.2f} MB\n")

    except Exception as e:
        print(f"Error initializing model: {e}");
        traceback.print_exc();
        return

    try:
        print("Starting training of SyK-CapsNet model...")
        FULL_EPOCHS = 200
        FULL_EARLY_STOP_PATIENCE = 40
        best_r2, best_r2_epoch, best_model_path = train_and_validate_final(
            model, train_loader, val_loader, device,
            epochs=FULL_EPOCHS, lr=current_learning_rate, patience=FULL_EARLY_STOP_PATIENCE,
            huber_delta=current_huber_delta, save_folder=Config.SAVE_FOLDER,
            current_hyperparams=full_hyperparams_to_save
        )
        print(f"\nTraining completed! Best validation R2: {best_r2:.4f} (Epoch {best_r2_epoch}).")

        if best_model_path and os.path.exists(best_model_path):
            print(f"\nLoading best model ({os.path.basename(best_model_path)}) for final evaluation...")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            final_criterion = nn.HuberLoss(delta=current_huber_delta).to(device)

            _, final_r2, final_rmse, final_mae, final_preds, final_trues, final_location_infos = evaluate_metrics(model,
                                                                                                                  val_loader,
                                                                                                                  device,
                                                                                                                  final_criterion)
            print("Best model final performance on validation set:")
            print(f"  R2: {final_r2:.4f}, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")

            export_results_to_csv(final_location_infos, final_trues, final_preds, final_r2, Config.SAVE_FOLDER)
    except Exception as e:
        print(f"Error during training/evaluation: {e}");
        traceback.print_exc()


if __name__ == "__main__":
    main()