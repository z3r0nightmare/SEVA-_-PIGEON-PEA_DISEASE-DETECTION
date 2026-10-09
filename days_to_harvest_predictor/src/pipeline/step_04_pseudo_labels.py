"""
Step 4: Derive Pseudo-Labels (Harvest Date Detection via Optical + SAR Consensus).
Identifies harvest day-of-year (DOY) per point-year based on rapid NDVI decline and SAR VH drop.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    PROCESSED_DIR,
    NDVI_PEAK_THRESHOLD, NDVI_DROP_THRESHOLD, NDVI_DROP_RATE,
    VH_DROP_THRESHOLD_DB, VH_CONFIRMATION_WINDOW_DAYS,
    LABEL_AGREE_HIGH_DAYS, LABEL_AGREE_MED_DAYS,
)


def detect_ndvi_harvest(df_season: pd.DataFrame) -> tuple:
    df_clean = df_season.dropna(subset=["ndvi"]).sort_values("doy")
    if len(df_clean) < 6:
        return None, None, None

    doys = df_clean["doy"].values
    ndvis = df_clean["ndvi"].values

    peak_mask = (doys >= 230) & (doys <= 290)
    if peak_mask.sum() == 0:
        return None, None, None

    peak_idx_local = np.argmax(ndvis[peak_mask])
    peak_idx = np.where(peak_mask)[0][peak_idx_local]
    peak_ndvi = ndvis[peak_idx]
    peak_doy = doys[peak_idx]

    if peak_ndvi < NDVI_PEAK_THRESHOLD:
        return None, None, None

    post_peak_mask = np.arange(len(doys)) > peak_idx
    if post_peak_mask.sum() == 0:
        return None, None, None

    post_doys = doys[post_peak_mask]
    post_ndvis = ndvis[post_peak_mask]

    drop_rates = np.diff(post_ndvis)
    rapid_drop_idx = np.where(drop_rates <= NDVI_DROP_RATE)[0]

    if len(rapid_drop_idx) > 0:
        harvest_idx_local = rapid_drop_idx[0] + 1
        return post_doys[harvest_idx_local], peak_doy, peak_ndvi

    below_thresh_idx = np.where(post_ndvis <= NDVI_DROP_THRESHOLD)[0]
    if len(below_thresh_idx) > 0:
        return post_doys[below_thresh_idx[0]], peak_doy, peak_ndvi

    halfway_val = peak_ndvi - 0.5 * (peak_ndvi - post_ndvis.min())
    half_idx = np.where(post_ndvis <= halfway_val)[0]
    if len(half_idx) > 0:
        return post_doys[half_idx[0]], peak_doy, peak_ndvi

    return None, peak_doy, peak_ndvi


def detect_sar_harvest(df_season: pd.DataFrame) -> tuple:
    df_clean = df_season.dropna(subset=["VH"]).sort_values("doy")
    if len(df_clean) < 6:
        return None, None, None

    doys = df_clean["doy"].values
    vhs = df_clean["VH"].values

    peak_mask = (doys >= 220) & (doys <= 290)
    if peak_mask.sum() == 0:
        return None, None, None

    peak_idx_local = np.argmax(vhs[peak_mask])
    peak_idx = np.where(peak_mask)[0][peak_idx_local]
    peak_vh = vhs[peak_idx]
    peak_doy = doys[peak_idx]

    post_peak_mask = np.arange(len(doys)) > peak_idx
    if post_peak_mask.sum() == 0:
        return None, None, None

    post_doys = doys[post_peak_mask]
    post_vhs = vhs[post_peak_mask]

    drops = peak_vh - post_vhs
    drop_idx = np.where(drops >= VH_DROP_THRESHOLD_DB)[0]
    if len(drop_idx) > 0:
        return post_doys[drop_idx[0]], peak_doy, peak_vh

    return None, peak_doy, peak_vh


def derive_pseudo_label(df_season: pd.DataFrame) -> dict:
    ndvi_harvest_doy, ndvi_peak_doy, ndvi_peak_val = detect_ndvi_harvest(df_season)
    sar_harvest_doy, sar_peak_doy, sar_peak_val = detect_sar_harvest(df_season)

    if ndvi_harvest_doy is None and sar_harvest_doy is None:
        return None

    if ndvi_harvest_doy is not None and sar_harvest_doy is not None:
        delta = abs(ndvi_harvest_doy - sar_harvest_doy)
        if delta <= VH_CONFIRMATION_WINDOW_DAYS:
            consensus_doy = int(np.round((ndvi_harvest_doy + sar_harvest_doy) / 2.0))
            conf = "high" if delta <= LABEL_AGREE_HIGH_DAYS else "medium" if delta <= LABEL_AGREE_MED_DAYS else "low"
        else:
            consensus_doy = ndvi_harvest_doy
            conf = "low"
    elif ndvi_harvest_doy is not None:
        consensus_doy = ndvi_harvest_doy
        conf = "medium"
    else:
        consensus_doy = sar_harvest_doy
        conf = "low"

    return {
        "harvest_doy": consensus_doy,
        "ndvi_harvest_doy": ndvi_harvest_doy,
        "sar_harvest_doy": sar_harvest_doy,
        "label_confidence": conf,
        "ndvi_peak_doy": ndvi_peak_doy,
        "ndvi_peak_val": ndvi_peak_val,
        "sar_peak_doy": sar_peak_doy,
        "sar_peak_val": sar_peak_val,
    }


def run():
    print("=" * 70)
    print("STEP 4: Derive Pseudo-Labels (Harvest Date Detection)")
    print("=" * 70)
    in_path = PROCESSED_DIR / "interpolated_5day.csv"
    if not in_path.exists():
        print(f"Error: {in_path.name} not found.")
        return

    df = pd.read_csv(in_path)
    records = []
    groups = df.groupby(["point_id", "year"])
    for (pid, yr), group in groups:
        res = derive_pseudo_label(group)
        if res is not None:
            lat = group["lat"].iloc[0]
            lon = group["lon"].iloc[0]
            rec = {"point_id": pid, "year": yr, "lat": lat, "lon": lon}
            rec.update(res)
            records.append(rec)

    labels_df = pd.DataFrame(records)
    out_path = PROCESSED_DIR / "pseudo_labels.csv"
    labels_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.name} ({len(labels_df)} point-years labeled)")


if __name__ == "__main__":
    run()
