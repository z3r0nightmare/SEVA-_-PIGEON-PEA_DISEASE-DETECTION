"""
Step 8: Assemble Final Training Dataset & Rigorous Causality Audit.
Input:  data/processed/features_all.csv
        data/processed/pseudo_labels.csv
        data/processed/variety_clusters.csv
        data/processed/static_features.csv
Output: data/processed/training_data.csv
        data/processed/dataset_metadata.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR, TRAIN_YEARS, VAL_YEARS, TEST_YEARS


def assign_district_approx(df: pd.DataFrame) -> pd.DataFrame:
    if "district_code" in df.columns and df["district_code"].notna().mean() > 0.5:
        return df

    def lat_to_district(lat, lon):
        if lat > 30.5:
            return "punjab_north"
        elif lat > 29.5:
            return "punjab_south" if lon < 76.0 else "haryana"
        elif lat > 28.5:
            return "haryana" if lon < 77.0 else "up_west"
        else:
            return "up_west"

    lats = df["lat"].values if "lat" in df.columns else np.full(len(df), 30.0)
    lons = df["lon"].values if "lon" in df.columns else np.full(len(df), 75.5)
    df["district_code"] = [lat_to_district(la, lo) for la, lo in zip(lats, lons)]
    return df


def audit_causality(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 50)
    print("STRICT CAUSALITY AUDIT")
    print("=" * 50)
    leakage_cols = ["peak_ndvi", "days_since_peak", "amplitude", "cumul_ndvi", "sos_doy", "eos_doy"]
    clean = True
    for col in leakage_cols:
        if col in df.columns:
            print(f"  [FAIL] Leakage column found: {col}")
            clean = False
        else:
            print(f"  [OK] No '{col}' column (correctly absent)")

    causal_cols = [
        "rolling_max_ndvi_30d", "days_since_rolling_max_30d",
        "cumul_ndvi_to_t", "rolling_amplitude_30d",
        "ndvi_rate", "ndvi_accel"
    ]
    for col in causal_cols:
        if col in df.columns:
            print(f"  [OK] Causal feature '{col}' present")
        else:
            print(f"  [WARN] Expected causal feature '{col}' missing")

    return clean


def run():
    print("=" * 70)
    print("STEP 8: Assemble Final Training DataFrame")
    print("=" * 70)
    f_path = PROCESSED_DIR / "features_all.csv"
    if not f_path.exists():
        print("features_all.csv missing.")
        return

    feat_df = pd.read_csv(f_path)
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    # Labels
    l_path = PROCESSED_DIR / "pseudo_labels.csv"
    if l_path.exists():
        lab_df = pd.read_csv(l_path)
        feat_df = feat_df.merge(
            lab_df[["point_id", "year", "harvest_doy", "label_confidence", "ndvi_harvest_doy", "sar_harvest_doy"]],
            on=["point_id", "year"],
            how="left"
        )

    # Variety clusters
    c_path = PROCESSED_DIR / "variety_clusters.csv"
    if c_path.exists():
        clu_df = pd.read_csv(c_path)
        feat_df = feat_df.merge(clu_df[["point_id", "year", "variety_cluster"]], on=["point_id", "year"], how="left")
        feat_df["variety_cluster"] = feat_df["variety_cluster"].fillna(0).astype(int)

    # Static features
    s_path = PROCESSED_DIR / "static_features.csv"
    if s_path.exists():
        stat_df = pd.read_csv(s_path)
        feat_df = feat_df.merge(stat_df, on="point_id", how="left")

    feat_df = assign_district_approx(feat_df)

    # Target: days_to_harvest = harvest_doy - doy
    if "harvest_doy" in feat_df.columns:
        feat_df["days_to_harvest"] = feat_df["harvest_doy"] - feat_df["doy"]
    else:
        feat_df["days_to_harvest"] = np.nan

    # Split assignment
    def get_split(yr):
        if yr in TRAIN_YEARS:
            return "train"
        elif yr in VAL_YEARS:
            return "val"
        else:
            return "test"

    feat_df["split"] = feat_df["year"].apply(get_split)

    # Filter to valid labeled points
    if "harvest_doy" in feat_df.columns:
        final_df = feat_df[feat_df["harvest_doy"].notna()].copy()
    else:
        final_df = feat_df.copy()

    audit_causality(final_df)

    out_path = PROCESSED_DIR / "training_data.csv"
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path.name} ({len(final_df)} rows, {len(final_df.columns)} columns)")

    # Save metadata
    meta = {
        "total_rows": len(final_df),
        "columns": list(final_df.columns),
        "train_rows": int((final_df["split"] == "train").sum()),
        "val_rows": int((final_df["split"] == "val").sum()),
        "test_rows": int((final_df["split"] == "test").sum()),
    }
    meta_path = PROCESSED_DIR / "dataset_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    run()
