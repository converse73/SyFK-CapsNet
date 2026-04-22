import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from config import Config


class YieldDataset(Dataset):
    def __init__(self, data_folder: str, start_year: int, end_year: int, months: List[int],
                 feature_cols: List[str], rs_feature_cols: List[str], m_feature_cols: List[str],
                 label_col: str, normalize: bool = True):
        self.feature_cols = feature_cols
        self.rs_feature_cols = rs_feature_cols
        self.m_feature_cols = m_feature_cols
        self.label_col = label_col
        self.normalize = normalize
        self.time_steps = len(months)
        self.data = []
        all_dfs_by_year_month = {}

        for year in range(start_year, end_year + 1):
            all_dfs_by_year_month[year] = {}
            for month in months:
                file_pattern = os.path.join(data_folder, f"Corn_USA_{year}_{month:02d}.csv")
                found_files = glob.glob(file_pattern)
                if not found_files:
                    file_pattern = os.path.join(data_folder, f"Corn_USA_{year}_{month}.csv")
                    found_files = glob.glob(file_pattern)
                if not found_files:
                    continue
                file_path = found_files[0]
                try:
                    df = pd.read_csv(file_path)
                    required_cols_from_csv = feature_cols + [label_col, 'State ANSI', 'County', 'County ANSI']
                    if not all(col in df.columns for col in required_cols_from_csv):
                        missing_cols = [col for col in required_cols_from_csv if col not in df.columns]
                        raise ValueError(f"File {file_path} is missing columns: {missing_cols}")

                    all_dfs_by_year_month[year][month] = df.set_index(['State ANSI', 'County', 'County ANSI'])
                except Exception as e:
                    print(f"Error loading or processing file {file_path}: {e}")
                    continue

        for year in range(start_year, end_year + 1):
            available_months_this_year = [m for m in months if m in all_dfs_by_year_month[year]]
            if not available_months_this_year: continue

            common_indices = None
            for month in available_months_this_year:
                current_month_df_indices = all_dfs_by_year_month[year][month].index
                if common_indices is None:
                    common_indices = current_month_df_indices
                else:
                    common_indices = common_indices.intersection(current_month_df_indices)

            if common_indices is None or len(common_indices) == 0:
                continue

            aligned_dfs_this_year = {m: all_dfs_by_year_month[year][m].loc[common_indices] for m in
                                     available_months_this_year}

            for (state_ansi, county, county_ansi) in common_indices:
                features_for_county_year = []
                for month_idx, month in enumerate(months):
                    if month in aligned_dfs_this_year:
                        monthly_features = aligned_dfs_this_year[month].loc[(state_ansi, county, county_ansi)][
                            feature_cols].values.tolist()
                        features_for_county_year.append(monthly_features)
                    else:
                        features_for_county_year.append(np.full(len(feature_cols), np.nan).tolist())

                features_stacked = np.stack(features_for_county_year, axis=0).astype(np.float32)

                label_for_county_year = \
                    aligned_dfs_this_year[available_months_this_year[0]].loc[(state_ansi, county, county_ansi)][
                        label_col].astype(np.float32)

                location_info_str = f"{state_ansi}|{county}|{county_ansi}"
                self.data.append((features_stacked, label_for_county_year, year, location_info_str))

        if len(self.data) == 0:
            raise ValueError("No valid samples loaded. Please check data path, file naming, and column names.")

        self.all_features = np.array([item[0] for item in self.data])
        self.all_labels = np.array([item[1] for item in self.data])

        if np.isnan(self.all_features).any():
            print("Warning: NaN values exist in data, filling with 0.0.")
            self.all_features = np.nan_to_num(self.all_features, nan=0.0)

        self.feature_scaler = None
        self.label_scaler = None
        self.samples = []

    def apply_normalization(self, train_indices: List[int]):
        if self.normalize:
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()

            train_features = self.all_features[train_indices]
            train_labels = self.all_labels[train_indices]

            N_train, T, F_total = train_features.shape
            self.feature_scaler.fit(train_features.reshape(N_train * T, F_total))
            self.label_scaler.fit(train_labels.reshape(-1, 1))

            N_all = self.all_features.shape[0]
            self.features_normalized = self.feature_scaler.transform(
                self.all_features.reshape(N_all * T, F_total)).reshape(N_all, T, F_total)
            self.labels_normalized = self.label_scaler.transform(self.all_labels.reshape(-1, 1)).ravel()
        else:
            self.features_normalized = self.all_features
            self.labels_normalized = self.all_labels

        self.samples = []
        for i in range(len(self.data)):
            self.samples.append(
                (self.features_normalized[i], self.labels_normalized[i], self.data[i][2], self.data[i][3])
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label, year, location_info_str = self.samples[idx]

        rs_features_indices = [self.feature_cols.index(col) for col in Config.RS_FEATURE_COLS]
        m_features_indices = [self.feature_cols.index(col) for col in Config.M_FEATURE_COLS]

        rs_features = features[:, rs_features_indices]
        m_features = features[:, m_features_indices]

        return (torch.FloatTensor(rs_features), torch.FloatTensor(m_features)), torch.FloatTensor(
            [label]), location_info_str

    def inverse_transform_labels(self, labels_tensor: torch.Tensor) -> np.ndarray:
        labels_np = labels_tensor.cpu().detach().numpy()
        if labels_np.ndim == 1: labels_np = labels_np.reshape(-1, 1)
        return self.label_scaler.inverse_transform(
            labels_np).ravel() if self.normalize and self.label_scaler else labels_np.ravel()


def setup_training(dataset: YieldDataset, test_year: int, batch_size: int):
    train_indices = [i for i, item in enumerate(dataset.data) if item[2] != test_year]
    val_indices = [i for i, item in enumerate(dataset.data) if item[2] == test_year]

    if len(train_indices) == 0:
        raise ValueError(f"Training set size is 0. Please check data and test year setup (TEST_YEAR={test_year}).")
    if len(val_indices) == 0:
        raise ValueError(f"Validation set size is 0. Please check data and test year setup (TEST_YEAR={test_year}).")

    dataset.apply_normalization(train_indices)

    train_ds, val_ds = Subset(dataset, train_indices), Subset(dataset, val_indices)

    kwargs = {'num_workers': 2 if torch.cuda.is_available() else 0, 'pin_memory': torch.cuda.is_available()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, val_ds, train_ds