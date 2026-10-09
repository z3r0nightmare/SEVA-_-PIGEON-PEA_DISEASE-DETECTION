"""
PyTorch and Tabular Dataset Definitions for Crop Phenology and Harvest Date Prediction.
Provides causal temporal sequence grouping, padding, categorical indexing, and sample weighting.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure config can be imported from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    MAX_SEQ_LEN, CATEGORICAL_FEATURES, EXCLUDE_COLS, RAW_BAND_COLS,
    PROCESSED_DIR
)


def get_feature_lists(df: pd.DataFrame):
    """
    Extracts numerical and categorical feature column names from the DataFrame.
    Filters out metadata, raw uncalibrated bands, and target columns.
    """
    drop_cols = [c for c in EXCLUDE_COLS + RAW_BAND_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    return num_cols, cat_cols


class AgriSequenceDataset(Dataset):
    """
    Groups tabular 5-day time-step observations into causal temporal sequences per (point_id, year).
    
    Properties:
    - Reconstructs up to MAX_SEQ_LEN (36 steps = 180 days) temporal trajectory.
    - Strictly causal padding: sequences shorter than MAX_SEQ_LEN are padded with constant values.
    - Masking: Target values for padded steps are set to -1.0 so loss functions ignore them.
    - Sample weights: High-confidence pseudo-labels receive 1.0, medium 0.5, low 0.1.
    """
    def __init__(self, df: pd.DataFrame, cat_cols: list, num_cols: list, max_seq_len: int = MAX_SEQ_LEN):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_seq_len = max_seq_len
        df = df.copy()

        # Compute sample weights based on label agreement confidence
        weights = np.ones(len(df), dtype=np.float32)
        if "label_confidence" in df.columns:
            conf = df["label_confidence"].values
            weights = np.where(conf == "high", 1.0,
                      np.where(conf == "medium", 0.5,
                      np.where(conf == "low", 0.1, 1.0)))
        df["weight"] = weights

        # Handle numeric missing values and infinite numbers
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = np.nan_to_num(df[col].values, nan=0.0, posinf=0.0, neginf=0.0)

        # Categorical code mapping
        for col in cat_cols:
            df[col] = df[col].astype("category").cat.codes.astype(np.int64)
            df[col] = np.maximum(df[col], 0)

        # Group by farm point trajectory across each season
        self.sequences = []
        for _, group in df.groupby(["point_id", "year"]):
            group = group.sort_values("date")
            
            x_num = group[num_cols].values.astype(np.float32)
            x_cat = group[cat_cols].values.astype(np.int64)
            y = group["days_to_harvest"].values.astype(np.float32)
            w = group["weight"].values.astype(np.float32)
            
            seq_len = len(x_num)
            if seq_len == 0:
                continue
                
            # Truncate to maximum sequence length if longer (take recent timesteps)
            if seq_len > self.max_seq_len:
                x_num = x_num[-self.max_seq_len:]
                x_cat = x_cat[-self.max_seq_len:]
                y = y[-self.max_seq_len:]
                w = w[-self.max_seq_len:]
                seq_len = self.max_seq_len
                
            # Apply causal sequence padding
            pad_len = self.max_seq_len - seq_len
            if pad_len > 0:
                x_num = np.pad(x_num, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                x_cat = np.pad(x_cat, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
                y = np.pad(y, (0, pad_len), mode="constant", constant_values=-1.0)  # -1.0 is masked
                w = np.pad(w, (0, pad_len), mode="constant", constant_values=0.0)
            
            self.sequences.append((
                torch.tensor(x_num, dtype=torch.float32),
                torch.tensor(x_cat, dtype=torch.long),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(w, dtype=torch.float32),
                torch.tensor(seq_len, dtype=torch.long)
            ))
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]


def create_dataloaders(csv_path: Path = None, batch_size: int = 128):
    """
    Loads dataset from CSV and returns train, validation, and test PyTorch DataLoaders.
    """
    if csv_path is None:
        csv_path = PROCESSED_DIR / "training_data.csv"
        
    df = pd.read_csv(csv_path)
    df = df[df["days_to_harvest"].notna() & (df["days_to_harvest"] > 0)].copy()
    
    num_cols, cat_cols = get_feature_lists(df)
    cat_dims = [int(df[c].nunique() + 1) for c in cat_cols]
    
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()
    
    train_ds = AgriSequenceDataset(train_df, cat_cols, num_cols)
    val_ds   = AgriSequenceDataset(val_df, cat_cols, num_cols)
    test_ds  = AgriSequenceDataset(test_df, cat_cols, num_cols)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    meta = {
        "num_dim": len(num_cols),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_dims": cat_dims,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds)
    }
    
    return train_loader, val_loader, test_loader, meta
