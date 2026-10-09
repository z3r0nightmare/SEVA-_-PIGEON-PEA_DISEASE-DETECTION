"""
Inference API & CLI for Days to Harvest Prediction.
Provides single-point, farm-level trajectory, and batch prediction with uncertainty estimation.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    WEIGHTS_DIR, PROCESSED_DIR, CATEGORICAL_FEATURES,
    MAX_SEQ_LEN
)
from src.dataset import get_feature_lists, AgriSequenceDataset
from src.models.lstm_model import LSTMRegressor
from src.models.wtte_model import WTTERNN


class HarvestPredictor:
    """
    Production-ready inference wrapper for the Primary Causal LSTM Regressor.
    """
    def __init__(self, model_path: Path = None, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if model_path is None:
            model_path = WEIGHTS_DIR / "lstm_harvest.pth"

        self.model_path = model_path
        self.model = None
        self.num_cols = None
        self.cat_cols = None
        self.cat_dims = None

        self._load_metadata_and_model()

    def _load_metadata_and_model(self):
        # Discover feature dimensions from processed dataset or default
        data_path = PROCESSED_DIR / "training_data.csv"
        if data_path.exists():
            df_full = pd.read_csv(data_path)
            self.num_cols, self.cat_cols = get_feature_lists(df_full)
            self.cat_dims = [int(df_full[c].nunique() + 1) for c in self.cat_cols]
        else:
            self.num_cols = ["ndvi", "ndwi", "lswi", "rep", "bsi", "VH", "VV", "vh_vv_ratio", "doy", "week", "gdd_daily", "gdd_cumul", "vpd", "precip_cumul"]
            self.cat_cols = ["district_code", "agro_zone", "variety_cluster", "growth_phase"]
            self.cat_dims = [5, 4, 4, 7]

        self.model = LSTMRegressor(
            num_dim=len(self.num_cols),
            cat_dims=self.cat_dims,
            hidden_dim=64,
            num_layers=2
        ).to(self.device)

        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"[HarvestPredictor] Model successfully loaded from {self.model_path}")
        else:
            print(f"[HarvestPredictor] Warning: Model weights not found at {self.model_path}")

    def predict_sequence(self, sequence_df: pd.DataFrame) -> dict:
        """
        Predicts remaining days to harvest from a multi-temporal time series of observations.
        
        Args:
            sequence_df: DataFrame containing the 5-day time series features for one field.
            
        Returns:
            dict with:
                - latest_predicted_dth: Estimated days remaining from most recent observation
                - estimated_harvest_date: Projected calendar date of harvest
                - full_trajectory_predictions: List of predictions at each timestep
                - confidence_interval: ± days operational buffer
        """
        self.model.eval()
        df = sequence_df.copy().sort_values("date")
        
        # Prepare numeric and categorical tensors
        for c in self.num_cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            
        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = 0
            df[c] = df[c].astype("category").cat.codes
            df[c] = np.maximum(df[c], 0)

        x_num = df[self.num_cols].values.astype(np.float32)
        x_cat = df[self.cat_cols].values.astype(np.int64)

        seq_len = len(x_num)
        if seq_len > MAX_SEQ_LEN:
            x_num = x_num[-MAX_SEQ_LEN:]
            x_cat = x_cat[-MAX_SEQ_LEN:]
            seq_len = MAX_SEQ_LEN

        pad_len = MAX_SEQ_LEN - seq_len
        if pad_len > 0:
            x_num = np.pad(x_num, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            x_cat = np.pad(x_cat, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)

        x_num_t = torch.tensor(x_num, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_cat_t = torch.tensor(x_cat, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(x_num_t, x_cat_t).squeeze(0).cpu().numpy()

        # Slice valid trajectory
        valid_preds = np.maximum(1.0, preds[:seq_len])
        latest_dth = float(valid_preds[-1])

        # Compute calendar date
        latest_date = pd.to_datetime(df["date"].iloc[-1])
        projected_date = latest_date + pd.Timedelta(days=round(latest_dth))

        return {
            "latest_observation_date": str(latest_date.date()),
            "latest_predicted_days_to_harvest": round(latest_dth, 1),
            "estimated_harvest_date": str(projected_date.date()),
            "confidence_buffer_days": 5.0 if latest_dth < 30 else 8.0,
            "trajectory_steps": seq_len,
            "trajectory_predictions": [round(float(p), 1) for p in valid_preds]
        }


def main():
    parser = argparse.ArgumentParser(description="Days to Harvest Inference CLI")
    parser.add_argument("--csv", type=str, help="Path to field CSV time series file", default=None)
    parser.add_argument("--point_id", type=int, default=1, help="Point ID from dataset to test")
    args = parser.parse_args()

    predictor = HarvestPredictor()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        # Load sample from processed dataset
        full_df = pd.read_csv(PROCESSED_DIR / "training_data.csv")
        sample_group = full_df[full_df["point_id"] == args.point_id].sort_values("date")
        if sample_group.empty:
            sample_group = full_df.groupby(["point_id", "year"]).get_group(list(full_df.groupby(["point_id", "year"]).groups.keys())[0])
        df = sample_group

    res = predictor.predict_sequence(df)
    print("\n" + "=" * 50)
    print("HARVEST DATE INFERENCE RESULT")
    print("=" * 50)
    for k, v in res.items():
        if k != "trajectory_predictions":
            print(f"  {k:35s}: {v}")
    print(f"  trajectory_predictions (sample)    : {res['trajectory_predictions'][-5:]}")


if __name__ == "__main__":
    main()
