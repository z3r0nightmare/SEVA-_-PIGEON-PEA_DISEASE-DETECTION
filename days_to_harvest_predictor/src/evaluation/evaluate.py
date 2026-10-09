"""
Comprehensive Evaluation Engine for Days to Harvest Prediction Models.
Computes MAE, RMSE, MBE, R2, accuracy within ±N day tolerance windows, lead-time error curves, and district breakdowns.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    PROCESSED_DIR, WEIGHTS_DIR, EVAL_WINDOWS, CATEGORICAL_FEATURES,
    EXCLUDE_COLS, RAW_BAND_COLS
)
from src.dataset import create_dataloaders, get_feature_lists
from src.models.lstm_model import LSTMRegressor
from src.models.wtte_model import WTTERNN
from src.models.baselines import NaiveBaselineModel, CatBoostHarvestModel, XGBoostAFTHarvestModel


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray = None) -> dict:
    """
    Computes standard regression & phenology metrics.
    """
    valid_mask = (y_true > 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    w   = weights[valid_mask] if weights is not None else np.ones_like(y_t)
    w   = w / w.sum()

    diff = y_p - y_t
    abs_diff = np.abs(diff)

    mae = float(np.sum(abs_diff * w) if weights is not None else np.mean(abs_diff))
    rmse = float(np.sqrt(np.sum((diff ** 2) * w) if weights is not None else np.mean(diff ** 2)))
    mbe = float(np.sum(diff * w) if weights is not None else np.mean(diff))
    med_ae = float(np.median(abs_diff))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    tolerances = {}
    for win in EVAL_WINDOWS:
        pct = float(np.mean(abs_diff <= win) * 100.0)
        tolerances[f"within_±{win}d"] = round(pct, 2)

    return {
        "samples": int(len(y_t)),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mbe_bias": round(mbe, 2),
        "median_ae": round(med_ae, 2),
        "r2_score": round(r2, 4),
        **tolerances
    }


def evaluate_lstm_model(model_path: Path = None, split: str = "test", device: str = None) -> tuple:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if model_path is None:
        model_path = WEIGHTS_DIR / "lstm_harvest.pth"

    train_loader, val_loader, test_loader, meta = create_dataloaders(batch_size=128)
    loader = test_loader if split == "test" else val_loader

    model = LSTMRegressor(
        num_dim=meta["num_dim"],
        cat_dims=meta["cat_dims"],
        hidden_dim=64,
        num_layers=2
    ).to(device)

    if not model_path.exists():
        print(f"[ERROR] Model weights not found at {model_path}.")
        return {}, None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_weights = []

    with torch.no_grad():
        for x_num, x_cat, y, w, lengths in loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            preds = model(x_num, x_cat).cpu().numpy()
            y_np = y.numpy()
            w_np = w.numpy()

            for i in range(len(lengths)):
                L = lengths[i].item()
                # Padded mask: first L elements
                # Recall in AgriSequenceDataset, real sequence is padded up to MAX_SEQ_LEN
                # Targets are -1 for padded elements
                valid_idx = (y_np[i] != -1.0)
                all_preds.extend(preds[i][valid_idx])
                all_targets.extend(y_np[i][valid_idx])
                all_weights.extend(w_np[i][valid_idx])

    y_p = np.array(all_preds, dtype=np.float32)
    y_t = np.array(all_targets, dtype=np.float32)
    w_t = np.array(all_weights, dtype=np.float32)

    metrics = calculate_metrics(y_t, y_p, weights=None)
    return metrics, y_t, y_p


def run():
    print("=" * 70)
    print("EVALUATING PRIMARY LSTM MODEL ON HELD-OUT TEST SET (2025)")
    print("=" * 70)

    val_metrics, _, _ = evaluate_lstm_model(split="val")
    print(f"\n--- Validation Set (2024) ---")
    for k, v in val_metrics.items():
        print(f"  {k:15s}: {v}")

    test_metrics, y_true, y_pred = evaluate_lstm_model(split="test")
    print(f"\n--- Held-Out Test Set (2025) ---")
    for k, v in test_metrics.items():
        print(f"  {k:15s}: {v}")

    # Save evaluation summary
    out_path = WEIGHTS_DIR / "lstm_test_evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    print(f"\nSaved evaluation summary -> {out_path}")


if __name__ == "__main__":
    run()
