import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
from utils import save_params_to_json


def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    all_preds_inv, all_trues_inv, all_location_infos, total_loss = [], [], [], 0.0
    original_dataset = loader.dataset.dataset
    with torch.no_grad():
        for (rs_batch_x, m_batch_x), batch_y_true, batch_location_infos in loader:
            inputs = (x.to(device) for x in (rs_batch_x, m_batch_x))
            batch_y_true = batch_y_true.to(device)
            y_pred = model(inputs)
            loss = criterion(y_pred, batch_y_true)
            total_loss += loss.item() * rs_batch_x.size(0)
            all_preds_inv.extend(original_dataset.inverse_transform_labels(y_pred))
            all_trues_inv.extend(original_dataset.inverse_transform_labels(batch_y_true))
            all_location_infos.extend(batch_location_infos)

    avg_loss = total_loss / len(loader.dataset)
    preds, trues = np.array(all_preds_inv), np.array(all_trues_inv)

    r2 = r2_score(trues, preds) if len(np.unique(trues)) > 1 else 0.0
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)

    return avg_loss, r2, rmse, mae, preds, trues, all_location_infos


def train_and_validate_final(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                             epochs: int, lr: float, huber_delta: float,
                             save_folder: str, current_hyperparams: Dict):
    # 注意：这里的入参我去掉了 patience
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=huber_delta).to(device)

    # 【修改 1】：换用余弦退火学习率调度器。它只根据 epoch 数量自动平滑衰减，完全不看测试集的脸色。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    final_model_path = os.path.join(save_folder, "SyKCABModel_Final.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss_epoch = 0.0
        print(f"\n--- Epoch {epoch}/{epochs} Training Started ---")
        for (rs_batch_x, m_batch_x), batch_y_true, _ in train_loader:
            inputs = (x.to(device) for x in (rs_batch_x, m_batch_x))
            batch_y_true = batch_y_true.to(device)

            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, batch_y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss_epoch += loss.item() * rs_batch_x.size(0)

        avg_train_loss = total_train_loss_epoch / len(train_loader.dataset)

        _, train_r2, train_rmse, _, _, _, _ = evaluate_metrics(model, train_loader, device, criterion)

        # 这里的 val_loader 其实就是测试集。现在它只是一个“无情的旁观者”，它的成绩仅用于打印查看，不决定任何事情。
        avg_val_loss, val_r2, val_rmse, val_mae, _, _, _ = evaluate_metrics(model, val_loader, device, criterion)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.1e} | Train Loss={avg_train_loss:.4f} | Train R2={train_r2:.4f} | "
            f"Test(Obs) Loss={avg_val_loss:.4f} | Test(Obs) R2={val_r2:.4f} | Test RMSE={val_rmse:.4f} | Test MAE={val_mae:.4f}")

        # 【修改 2】：调度器前进一步，不再传入 val_r2
        scheduler.step()

    # 【修改 3】：去掉早停和挑最好模型的逻辑，无条件直接保存最后一轮的模型。
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed! Final model saved to {final_model_path}")

    params_filename = "SyKCABModel_Final_params.json"
    save_params_to_json(current_hyperparams, os.path.join(save_folder, params_filename))

    return val_r2, epochs, final_model_path