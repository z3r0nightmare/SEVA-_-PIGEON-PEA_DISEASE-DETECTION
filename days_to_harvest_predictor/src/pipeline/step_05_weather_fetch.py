"""
Step 5: Agrometeorological Data Fetch & Cache (Open-Meteo Historical Archive API).
Fetches daily temperature (max/min), precipitation, evapotranspiration (ET0), solar radiation, and wind speed.
If `data/processed/weather_daily.csv` exists, loads from cache.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR, DATA_DIR, SEASON_START_MMDD, SEASON_END_MMDD, YEARS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SAFE_WEATHER_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
]
RENAME_MAP = {
    "temperature_2m_max": "t_max",
    "temperature_2m_min": "t_min",
    "precipitation_sum": "precip_mm",
    "et0_fao_evapotranspiration": "et0_mm",
    "shortwave_radiation_sum": "solar_rad",
    "wind_speed_10m_max": "wind_speed",
}


def fetch_weather_for_point(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(SAFE_WEATHER_VARIABLES),
        "timezone": "auto",
    }
    for attempt in range(3):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                if "time" in daily:
                    df = pd.DataFrame(daily)
                    df = df.rename(columns={"time": "date"})
                    df = df.rename(columns=RENAME_MAP)
                    return df
            time.sleep(1.0)
        except Exception:
            time.sleep(2.0)
    return pd.DataFrame()


def run():
    print("=" * 70)
    print("STEP 5: Fetch / Load Weather Data")
    print("=" * 70)
    out_path = PROCESSED_DIR / "weather_daily.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        print(f"Weather dataset already exists at {out_path.name} ({len(df)} rows). Using cached data.")
        return df
        
    print("Fetching weather data from Open-Meteo...")
    # Read points
    interp_path = PROCESSED_DIR / "interpolated_5day.csv"
    if not interp_path.exists():
        print("Interpolated dataset missing.")
        return
        
    pts = pd.read_csv(interp_path)[["point_id", "lat", "lon"]].drop_duplicates()
    all_weather = []
    for yr in YEARS:
        start_d = f"{yr}-{SEASON_START_MMDD}"
        end_d = f"{yr}-{SEASON_END_MMDD}"
        for _, pt in pts.iterrows():
            w_df = fetch_weather_for_point(pt["lat"], pt["lon"], start_d, end_d)
            if not w_df.empty:
                w_df["point_id"] = pt["point_id"]
                w_df["year"] = yr
                all_weather.append(w_df)

    if all_weather:
        full_w = pd.concat(all_weather, ignore_index=True)
        full_w.to_csv(out_path, index=False)
        print(f"Saved -> {out_path.name} ({len(full_w)} rows)")


if __name__ == "__main__":
    run()
