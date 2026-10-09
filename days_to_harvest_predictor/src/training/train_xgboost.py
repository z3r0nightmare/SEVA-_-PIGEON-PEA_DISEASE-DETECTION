"""
Training Routine for Accelerated Failure Time Survival Baseline: XGBoost-AFT (Tier 1b).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    PROCESSED_DIR, WEIGHTS_DIR, XGBOOST_AFT_PARAMS,
    CATEGORICAL_FEATURES, EXCLUDE_COLS, RAW_BAND_COLS
)


def train_xgboost():
    print("=" * 70)
    print("TRAINING BASELINE MODEL: XGBoost-AFT Survival (Tier 1b)")
    print("=" * 70)

    df = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    df = df[df["days_to_harvest"].notna() & (df["days_to_harvest"] > 0)].copy()

    drop_cols = [c for c in EXCLUDE_COLS + RAW_BAND_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["days_to_harvest"])
    dval   = xgb.DMatrix(val_df[feature_cols],   label=val_df["days_to_harvest"])

    # Survival AFT target bounds (uncensored historical events: lower_bound = upper_bound = y)
    y_train = train_df["days_to_harvest"].values
    y_val   = val_df["days_to_harvest"].values
    dtrain.set_float_info("label_lower_bound", y_train)
    dtrain.set_float_info("label_upper_bound", y_train)
    dval.set_float_info("label_lower_bound", y_val)
    dval.set_float_info("label_upper_bound", y_val)

    params = XGBOOST_AFT_PARAMS.copy()
    num_boost_round = params.pop("n_estimators", 2000)
    early_stopping  = params.pop("early_stopping_rounds", 100)

    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping,
        verbose_eval=100
    )

    save_path = WEIGHTS_DIR / "xgboost_aft_harvest.json"
    booster.save_model(str(save_path))
    print(f"Saved XGBoost-AFT model -> {save_path}")

    return booster


if __name__ == "__main__":
    train_xgboost()
