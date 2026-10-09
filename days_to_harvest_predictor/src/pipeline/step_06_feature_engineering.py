"""
Step 6: Feature Engineering (Strictly Causal Phenological & Agrometeorological Transformations).
Input:  data/processed/interpolated_5day.csv
        data/processed/pseudo_labels.csv
        data/processed/weather_daily.csv
Output: data/processed/features_all.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    PROCESSED_DIR,
    TEMPORAL_RESOLUTION_DAYS, ROLLING_WINDOW_DAYS, LAG_STEPS,
    GDD_T_BASE, GDD_T_UPPER,
    VPD_A, VPD_B, VPD_C,
    SOS_NDVI_THRESHOLD,
)

ROLLING_STEPS = ROLLING_WINDOW_DAYS // TEMPORAL_RESOLUTION_DAYS


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["doy"] = df["date"].dt.dayofyear.astype(np.int16)
    df["week"] = df["date"].dt.isocalendar().week.astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    return df


def compute_sos_features(group: pd.DataFrame) -> pd.DataFrame:
    ndvi = group["ndvi"].values
    dates = group["date"].values

    sos_idx = -1
    for i in range(len(ndvi)):
        if not np.isnan(ndvi[i]) and ndvi[i] > SOS_NDVI_THRESHOLD:
            sos_idx = i
            break

    if sos_idx >= 0:
        sos_date = dates[sos_idx]
        group["days_since_sos"] = (
            (group["date"].values - sos_date).astype("timedelta64[D]").astype(np.float32)
        )
        group["days_since_sos"] = group["days_since_sos"].clip(lower=0)
    else:
        group["days_since_sos"] = np.nan

    return group


def compute_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    group["rolling_max_ndvi_30d"] = (
        group["ndvi"].rolling(ROLLING_STEPS, min_periods=1).max().astype(np.float32)
    )
    rolling_min = group["ndvi"].rolling(ROLLING_STEPS, min_periods=1).min()
    group["rolling_amplitude_30d"] = (
        (group["rolling_max_ndvi_30d"] - rolling_min).astype(np.float32)
    )

    max_vals = group["rolling_max_ndvi_30d"].values
    ndvi_vals = group["ndvi"].values
    days_since_max = np.zeros(len(group), dtype=np.float32)
    for i in range(len(group)):
        if np.isnan(max_vals[i]) or np.isnan(ndvi_vals[i]):
            days_since_max[i] = np.nan
            continue
        start = max(0, i - ROLLING_STEPS + 1)
        window = ndvi_vals[start:i+1]
        if len(window) > 0 and not np.all(np.isnan(window)):
            max_pos = np.nanargmax(window)
            days_since_max[i] = (len(window) - 1 - max_pos) * TEMPORAL_RESOLUTION_DAYS
    group["days_since_rolling_max_30d"] = days_since_max
    return group


def compute_cumulative_features(group: pd.DataFrame) -> pd.DataFrame:
    ndvi = group["ndvi"].values
    days_since_sos = group["days_since_sos"].values

    cumul = np.zeros(len(group), dtype=np.float32)
    for i in range(1, len(group)):
        if np.isnan(ndvi[i]) or np.isnan(days_since_sos[i]) or days_since_sos[i] <= 0:
            cumul[i] = cumul[i-1] if i > 0 else 0
        else:
            prev_val = ndvi[i-1] if not np.isnan(ndvi[i-1]) else 0
            cumul[i] = cumul[i-1] + (prev_val + ndvi[i]) / 2.0 * TEMPORAL_RESOLUTION_DAYS
    group["cumul_ndvi_to_t"] = cumul
    return group


def compute_derivative_features(group: pd.DataFrame) -> pd.DataFrame:
    dt = TEMPORAL_RESOLUTION_DAYS
    group["ndvi_rate"] = group["ndvi"].diff() / dt
    group["ndvi_accel"] = group["ndvi_rate"].diff() / dt
    if "VH" in group.columns:
        group["vh_rate"] = group["VH"].diff() / dt
        group["vh_accel"] = group["vh_rate"].diff() / dt
    else:
        group["vh_rate"] = np.nan
        group["vh_accel"] = np.nan

    for col in ["ndvi_rate", "ndvi_accel", "vh_rate", "vh_accel"]:
        group[col] = group[col].astype(np.float32)
    return group


def compute_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    for lag in LAG_STEPS:
        for col in ["ndvi", "ndwi", "VH", "VV"]:
            if col in group.columns:
                group[f"{col}_lag{lag}"] = group[col].shift(lag).astype(np.float32)
    return group


def compute_growth_phase(group: pd.DataFrame) -> pd.DataFrame:
    ndvi = group["ndvi"].values
    ndvi_rate = group["ndvi_rate"].values if "ndvi_rate" in group.columns else np.zeros(len(group))
    lswi = group["lswi"].values if "lswi" in group.columns else np.zeros(len(group))
    phase = np.full(len(group), 0, dtype=np.int8)

    for i in range(len(group)):
        if np.isnan(ndvi[i]):
            phase[i] = 0
        elif ndvi[i] < 0.2:
            phase[i] = 0 if i < len(group) // 2 else 5
        elif not np.isnan(lswi[i]) and lswi[i] > 0.2 and ndvi[i] < 0.4:
            phase[i] = 1
        elif not np.isnan(ndvi_rate[i]) and ndvi_rate[i] > 0.005:
            phase[i] = 2
        elif ndvi[i] > 0.5 and (np.isnan(ndvi_rate[i]) or abs(ndvi_rate[i]) < 0.005):
            phase[i] = 3
        elif not np.isnan(ndvi_rate[i]) and ndvi_rate[i] < -0.005:
            phase[i] = 4
        else:
            phase[i] = 0

    group["growth_phase"] = phase
    return group


def merge_weather_features(satellite_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    satellite_df["date"] = pd.to_datetime(satellite_df["date"])

    mean_cols = ["t_max", "t_min", "et0_mm", "solar_rad", "wind_speed"]
    sum_cols = ["precip_mm"]
    mean_cols = [c for c in mean_cols if c in weather_df.columns]
    sum_cols = [c for c in sum_cols if c in weather_df.columns]

    weather_df = weather_df.sort_values(["point_id", "date"]).set_index("date")
    rolled_means = weather_df.groupby("point_id")[mean_cols].rolling("5D").mean().reset_index()
    if sum_cols:
        rolled_sums = weather_df.groupby("point_id")[sum_cols].rolling("5D").sum().reset_index()
        agg_wx = rolled_means.merge(rolled_sums, on=["point_id", "date"])
    else:
        agg_wx = rolled_means

    merged = satellite_df.merge(agg_wx, on=["point_id", "date"], how="left")
    return merged


def compute_gdd_vpd(df: pd.DataFrame) -> pd.DataFrame:
    t_mean = (df["t_max"] + df["t_min"]) / 2.0
    df["gdd_daily"] = np.maximum(0, np.minimum(t_mean, GDD_T_UPPER) - GDD_T_BASE).astype(np.float32)

    e_s_tmax = VPD_A * np.exp(VPD_B * df["t_max"] / (df["t_max"] + VPD_C))
    e_s_tmin = VPD_A * np.exp(VPD_B * df["t_min"] / (df["t_min"] + VPD_C))
    e_s = (e_s_tmax + e_s_tmin) / 2.0
    e_a = e_s_tmin
    df["vpd"] = (e_s - e_a).clip(lower=0).astype(np.float32)
    return df


def compute_cumulative_weather(group: pd.DataFrame) -> pd.DataFrame:
    sos_mask = group["days_since_sos"].notna() & (group["days_since_sos"] >= 0)
    gdd = group["gdd_daily"].values.copy()
    gdd[~sos_mask.values] = 0
    group["gdd_cumul"] = np.nancumsum(gdd).astype(np.float32)

    precip = group["precip_mm"].values.copy()
    precip[~sos_mask.values] = 0
    group["precip_cumul"] = np.nancumsum(np.nan_to_num(precip)).astype(np.float32)

    et0 = group["et0_mm"].values.copy()
    et0[~sos_mask.values] = 0
    group["et0_cumul"] = np.nancumsum(np.nan_to_num(et0)).astype(np.float32)

    precip_series = group["precip_mm"].fillna(0)
    group["precip_7d"] = precip_series.rolling(2, min_periods=1).sum().astype(np.float32)

    dry = (group["precip_mm"].fillna(0) < 1.0).astype(int)
    consec = np.zeros(len(group), dtype=np.int16)
    for i in range(len(group)):
        if dry.iloc[i]:
            consec[i] = (consec[i-1] + 1) if i > 0 else 1
        else:
            consec[i] = 0
    group["dry_days_consec"] = consec
    return group


def compute_photoperiod(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lat = df["lat"].values if "lat" in df.columns else 30.0
    phi = np.radians(lat)
    J = df["date"].dt.dayofyear.values
    delta = 0.409 * np.sin((2 * np.pi * J / 365) - 1.39)
    tan_product = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    omega_s = np.arccos(tan_product)
    day_length = (24 / np.pi) * omega_s
    df["photoperiod"] = day_length.astype(np.float32)
    return df


def run():
    print("=" * 70)
    print("STEP 6: Causal Feature Engineering")
    print("=" * 70)
    in_path = PROCESSED_DIR / "interpolated_5day.csv"
    if not in_path.exists():
        print("interpolated_5day.csv missing.")
        return

    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_calendar_features(df)
    df = compute_photoperiod(df)

    # Weather
    w_path = PROCESSED_DIR / "weather_daily.csv"
    if w_path.exists():
        w_df = pd.read_csv(w_path)
        df = merge_weather_features(df, w_df)
        df = compute_gdd_vpd(df)

    # Trajectory features
    processed_groups = []
    groups = df.groupby(["point_id", "year"])
    for _, group in groups:
        g = group.sort_values("date").copy()
        g = compute_sos_features(g)
        g = compute_rolling_features(g)
        g = compute_cumulative_features(g)
        g = compute_derivative_features(g)
        g = compute_lag_features(g)
        g = compute_growth_phase(g)
        if "gdd_daily" in g.columns:
            g = compute_cumulative_weather(g)
        processed_groups.append(g)

    feat_df = pd.concat(processed_groups, ignore_index=True)
    out_path = PROCESSED_DIR / "features_all.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.name} ({len(feat_df)} rows, {len(feat_df.columns)} cols)")


if __name__ == "__main__":
    run()
