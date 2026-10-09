"""
Step 3: Strictly Causal Temporal Interpolation to a regular 5-day grid.
Input:  data/processed/with_indices_all_years.csv
Output: data/processed/interpolated_5day.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    PROCESSED_DIR, YEARS,
    TEMPORAL_RESOLUTION_DAYS, SEASON_START_MMDD, SEASON_END_MMDD,
    EMA_ALPHA, MAX_GAP_INTERP_DAYS, CARRY_FORWARD_MAX_DAYS,
    VI_NAMES, S1_BANDS,
)

OPTICAL_COLS = VI_NAMES
SAR_COLS = ["VH", "VV", "vh_vv_ratio"]
ALL_SIGNAL_COLS = OPTICAL_COLS + SAR_COLS


def create_regular_grid(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-{SEASON_START_MMDD}")
    end   = pd.Timestamp(f"{year}-{SEASON_END_MMDD}")
    return pd.date_range(start, end, freq=f"{TEMPORAL_RESOLUTION_DAYS}D")


def causal_interpolate_series(dates: np.ndarray, values: np.ndarray, grid_dates: np.ndarray) -> tuple:
    n_grid = len(grid_dates)
    result = np.full(n_grid, np.nan, dtype=np.float32)
    days_since = np.full(n_grid, 999.0, dtype=np.float32)

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() == 0:
        return result, days_since

    v_dates = dates[valid_mask].astype("datetime64[ns]")
    v_vals  = values[valid_mask].astype(np.float32)

    for i, gdate in enumerate(grid_dates):
        gdate_dt = np.datetime64(gdate, "ns")
        past_mask = v_dates <= gdate_dt
        if not np.any(past_mask):
            continue

        past_dates = v_dates[past_mask]
        past_vals  = v_vals[past_mask]

        last_date = past_dates[-1]
        last_val  = past_vals[-1]

        delta_days = (gdate_dt - last_date).astype("timedelta64[D]").astype(float)
        days_since[i] = delta_days

        if delta_days == 0:
            result[i] = last_val
        elif delta_days <= MAX_GAP_INTERP_DAYS and len(past_dates) >= 2:
            prev_date = past_dates[-2]
            prev_val  = past_vals[-2]
            gap = (last_date - prev_date).astype("timedelta64[D]").astype(float)
            if gap > 0 and delta_days < gap:
                frac = delta_days / gap
                result[i] = last_val + frac * (last_val - prev_val)
            else:
                result[i] = last_val
        elif delta_days <= CARRY_FORWARD_MAX_DAYS:
            result[i] = last_val

    # Causal EMA smoothing
    ema_vals = np.full(n_grid, np.nan, dtype=np.float32)
    running_ema = None
    for i in range(n_grid):
        val = result[i]
        if not np.isnan(val):
            if running_ema is None:
                running_ema = val
            else:
                running_ema = EMA_ALPHA * val + (1.0 - EMA_ALPHA) * running_ema
            ema_vals[i] = running_ema
        else:
            ema_vals[i] = running_ema if running_ema is not None else np.nan

    return ema_vals, days_since


def process_point_year(df_point: pd.DataFrame, point_id: int, year: int, lat: float, lon: float) -> pd.DataFrame:
    grid = create_regular_grid(year)
    n_steps = len(grid)

    df_out = pd.DataFrame({
        "point_id": point_id,
        "year": year,
        "date": grid,
        "lat": lat,
        "lon": lon,
        "doy": grid.dayofyear,
        "step": range(n_steps),
    })

    dates = pd.to_datetime(df_point["date"]).values

    for col in ALL_SIGNAL_COLS:
        if col in df_point.columns:
            vals = df_point[col].values.astype(float)
            interp_vals, days_since = causal_interpolate_series(dates, vals, grid.values)
            df_out[col] = interp_vals
            if col == "ndvi":
                df_out["days_since_valid_optical"] = days_since
            elif col == "VH":
                df_out["days_since_valid_sar"] = days_since
        else:
            df_out[col] = np.nan

    return df_out


def run():
    print("=" * 70)
    print("STEP 3: Causal Temporal Interpolation (5-Day Grid)")
    print("=" * 70)
    in_path = PROCESSED_DIR / "with_indices_all_years.csv"
    if not in_path.exists():
        print(f"Error: {in_path.name} not found.")
        return

    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])

    out_dfs = []
    groups = df.groupby(["point_id", "year"])
    for (pid, yr), group in groups:
        lat = group["lat"].dropna().iloc[0] if "lat" in group.columns and group["lat"].notna().any() else np.nan
        lon = group["lon"].dropna().iloc[0] if "lon" in group.columns and group["lon"].notna().any() else np.nan
        out_dfs.append(process_point_year(group, pid, yr, lat, lon))

    full_interp = pd.concat(out_dfs, ignore_index=True)
    out_path = PROCESSED_DIR / "interpolated_5day.csv"
    full_interp.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.name} ({len(full_interp)} rows)")


if __name__ == "__main__":
    run()
