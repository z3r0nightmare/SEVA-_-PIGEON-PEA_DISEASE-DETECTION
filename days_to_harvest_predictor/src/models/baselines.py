"""
Baseline Models for Benchmark Comparison:
- Naive Historical Median Baseline (Tier 0)
- CatBoost Gradient Boosted Regressor (Tier 1a)
- XGBoost-AFT Accelerated Failure Time Survival Model (Tier 1b)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


class NaiveBaselineModel:
    """
    Tier 0 Baseline: Historical Median Harvest Day of Year (DOY).
    Calculates the training split median harvest DOY globally and per district.
    """
    def __init__(self):
        self.global_median_doy = 301.0
        self.district_medians = {
            "haryana": 306.0,
            "punjab_north": 296.0,
            "punjab_south": 301.0,
            "up_west": 291.0
        }
        
    def fit(self, train_df: pd.DataFrame):
        labeled = train_df.dropna(subset=["harvest_doy"])
        if len(labeled) > 0:
            self.global_median_doy = float(labeled["harvest_doy"].median())
            for d in labeled["district_code"].unique():
                sub = labeled[labeled["district_code"] == d]
                if len(sub) > 0:
                    self.district_medians[str(d)] = float(sub["harvest_doy"].median())
                    
    def predict(self, df: pd.DataFrame, use_district: bool = True) -> np.ndarray:
        # Predictions are: (Harvest DOY - Current DOY)
        preds = []
        for _, row in df.iterrows():
            current_doy = row.get("doy", 200)
            dist = str(row.get("district_code", "unknown"))
            
            if use_district and dist in self.district_medians:
                target_doy = self.district_medians[dist]
            else:
                target_doy = self.global_median_doy
                
            days_remaining = max(1.0, target_doy - current_doy)
            preds.append(days_remaining)
            
        return np.array(preds, dtype=np.float32)


class CatBoostHarvestModel:
    """
    Tier 1a: CatBoost Gradient Boosted Trees Regressor wrapper.
    """
    def __init__(self, model_path: Path = None):
        self.model = None
        self.model_path = model_path
        if model_path and model_path.exists():
            self.load(model_path)
            
    def load(self, model_path: Path):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
        return np.maximum(1.0, self.model.predict(X))


class XGBoostAFTHarvestModel:
    """
    Tier 1b: XGBoost Accelerated Failure Time (AFT) Survival Model wrapper.
    """
    def __init__(self, model_path: Path = None):
        self.booster = None
        self.model_path = model_path
        if model_path and model_path.exists():
            self.load(model_path)
            
    def load(self, model_path: Path):
        import xgboost as xgb
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))
        
    def predict(self, dmatrix) -> np.ndarray:
        if self.booster is None:
            raise ValueError("Booster is not loaded or trained.")
        preds = self.booster.predict(dmatrix)
        return np.maximum(1.0, preds)
