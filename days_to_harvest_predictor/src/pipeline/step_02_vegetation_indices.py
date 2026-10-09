"""
Step 2: Compute Biophysical Vegetation Indices from Sentinel-2 & Sentinel-1.
Computes NDVI, NDWI, LSWI, REP, BSI, and SAR VH/VV ratio.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR, S2_BANDS, VI_NAMES


def compute_ndvi(df: pd.DataFrame) -> pd.Series:
    """NDVI = (B8 - B4) / (B8 + B4)"""
    nir, red = df["B8"], df["B4"]
    denom = nir + red
    return np.where(denom != 0, (nir - red) / denom, np.nan).astype(np.float32)


def compute_ndwi(df: pd.DataFrame) -> pd.Series:
    """NDWI = (B8 - B11) / (B8 + B11) (Canopy moisture)"""
    nir, swir1 = df["B8"], df["B11"]
    denom = nir + swir1
    return np.where(denom != 0, (nir - swir1) / denom, np.nan).astype(np.float32)


def compute_lswi(df: pd.DataFrame) -> pd.Series:
    """LSWI = (B8 - B12) / (B8 + B12) (Surface water / flooding)"""
    nir, swir2 = df["B8"], df["B12"]
    denom = nir + swir2
    return np.where(denom != 0, (nir - swir2) / denom, np.nan).astype(np.float32)


def compute_rep(df: pd.DataFrame) -> pd.Series:
    """Red Edge Position = 705 + 35 * ((B4+B7)/2 - B5) / (B6 - B5)"""
    b4, b5, b6, b7 = df["B4"], df["B5"], df["B6"], df["B7"]
    denom = b6 - b5
    result = np.where(
        np.abs(denom) > 1e-6,
        705.0 + 35.0 * ((b4 + b7) / 2.0 - b5) / denom,
        np.nan
    )
    return np.clip(result, 680.0, 760.0).astype(np.float32)


def compute_bsi(df: pd.DataFrame) -> pd.Series:
    """Bare Soil Index = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))"""
    b11, b4, b8, b2 = df["B11"], df["B4"], df["B8"], df["B2"]
    num = (b11 + b4) - (b8 + b2)
    denom = (b11 + b4) + (b8 + b2)
    return np.where(denom != 0, num / denom, np.nan).astype(np.float32)


def compute_vh_vv_ratio(df: pd.DataFrame) -> pd.Series:
    """VH/VV ratio in dB = VH - VV"""
    return (df["VH"] - df["VV"]).astype(np.float32)


def run():
    print("=" * 70)
    print("STEP 2: Compute Vegetation Indices")
    print("=" * 70)
    in_path = PROCESSED_DIR / "merged_raw_all_years.csv"
    if not in_path.exists():
        print(f"Error: {in_path.name} not found.")
        return

    df = pd.read_csv(in_path)
    df["ndvi"] = compute_ndvi(df)
    df["ndwi"] = compute_ndwi(df)
    df["lswi"] = compute_lswi(df)
    df["rep"]  = compute_rep(df)
    df["bsi"]  = compute_bsi(df)
    if "VH" in df.columns and "VV" in df.columns:
        df["vh_vv_ratio"] = compute_vh_vv_ratio(df)

    out_path = PROCESSED_DIR / "with_indices_all_years.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.name} ({len(df)} rows)")


if __name__ == "__main__":
    run()
