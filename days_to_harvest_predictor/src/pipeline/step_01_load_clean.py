"""
Step 1: Load raw satellite CSVs, validate spectral bands, merge S1 + S2, and clean.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import RAW_DIR, PROCESSED_DIR, YEARS, S2_BANDS, S1_BANDS


def load_sentinel2(year: int) -> pd.DataFrame:
    """Load and clean a single year of Sentinel-2 surface reflectance data."""
    path = RAW_DIR / f"sentinel2_raw_{year}.csv"
    if not path.exists():
        print(f"  Warning: S2 file {path.name} not found.")
        return pd.DataFrame()
        
    print(f"  Loading S2 {year} from {path.name}...")
    df = pd.read_csv(path)

    missing = [b for b in S2_BANDS if b not in df.columns]
    if missing:
        raise ValueError(f"S2 {year}: Missing bands {missing}. Columns: {list(df.columns)}")

    keep_cols = ["point_id", "date"] + S2_BANDS + ["lat", "lon"]
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = year

    # Scale DN to Surface Reflectance [0, 1]
    for band in S2_BANDS:
        df[band] = df[band].astype(np.float32) / 10000.0
        outlier_mask = (df[band] < 0) | (df[band] > 1.5)
        df.loc[outlier_mask, band] = np.nan

    return df


def load_sentinel1(year: int) -> pd.DataFrame:
    """Load and clean a single year of Sentinel-1 GRD SAR backscatter data."""
    path = RAW_DIR / f"sentinel1_raw_{year}.csv"
    if not path.exists():
        print(f"  Warning: S1 file {path.name} not found.")
        return pd.DataFrame()
        
    print(f"  Loading S1 {year} from {path.name}...")
    df = pd.read_csv(path)

    missing = [b for b in S1_BANDS if b not in df.columns]
    if missing:
        raise ValueError(f"S1 {year}: Missing bands {missing}. Columns: {list(df.columns)}")

    keep_cols = ["point_id", "date"] + S1_BANDS
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = year

    # Outlier filtering for SAR dB backscatter
    for band in S1_BANDS:
        df[band] = df[band].astype(np.float32)
        outlier_mask = (df[band] < -35.0) | (df[band] > 5.0)
        df.loc[outlier_mask, band] = np.nan

    return df


def merge_year(year: int) -> pd.DataFrame:
    """Merge S1 and S2 for a given year on point_id + date."""
    s2 = load_sentinel2(year)
    s1 = load_sentinel1(year)
    
    if s2.empty and s1.empty:
        return pd.DataFrame()

    if s2.empty:
        merged = s1
    elif s1.empty:
        merged = s2
    else:
        merged = pd.merge(s2, s1, on=["point_id", "date", "year"], how="outer")

    merged = merged.sort_values(["point_id", "date"]).reset_index(drop=True)
    out_path = PROCESSED_DIR / f"merged_raw_{year}.csv"
    merged.to_csv(out_path, index=False)
    return merged


def run():
    print("=" * 70)
    print("STEP 1: Load, Clean & Merge Raw Satellite Data")
    print("=" * 70)
    all_dfs = []
    for y in YEARS:
        df_y = merge_year(y)
        if not df_y.empty:
            all_dfs.append(df_y)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_path = PROCESSED_DIR / "merged_raw_all_years.csv"
        full_df.to_csv(out_path, index=False)
        print(f"Saved merged all years -> {out_path.name} ({len(full_df)} rows)")


if __name__ == "__main__":
    run()
