"""
Training Routine for Gradient Boosted Trees Baseline: CatBoost Regressor (Tier 1a).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    PROCESSED_DIR, WEIGHTS_DIR, CATBOOST_PARAMS,
    CATEGORICAL_FEATURES, EXCLUDE_COLS, RAW_BAND_COLS
)


def train_catboost():
    print("=" * 70)
    print("TRAINING BASELINE MODEL: CatBoost Regressor (Tier 1a)")
    print("=" * 70)
    
    df = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    df = df[df["days_to_harvest"].notna() & (df["days_to_harvest"] > 0)].copy()

    drop_cols = [c for c in EXCLUDE_COLS + RAW_BAND_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("missing")

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    print(f"Features ({len(feature_cols)}): {feature_cols[:5]} ... Categoricals: {cat_cols}")
    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    X_train, y_train = train_df[feature_cols], train_df["days_to_harvest"]
    X_val,   y_val   = val_df[feature_cols],   val_df["days_to_harvest"]
    X_test,  y_test  = test_df[feature_cols],  test_df["days_to_harvest"]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    save_path = WEIGHTS_DIR / "catboost_harvest.cbm"
    model.save_model(str(save_path))
    print(f"Saved CatBoost model -> {save_path}")

    # Feature importances
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    fi.to_csv(WEIGHTS_DIR / "catboost_feature_importance.csv", index=False)

    return model


if __name__ == "__main__":
    train_catboost()
