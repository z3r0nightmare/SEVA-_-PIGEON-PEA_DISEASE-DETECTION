"""
Centralized Configuration for Days to Harvest Prediction Engine.
All paths, band lists, biophysical constants, model hyperparameters, and evaluation windows.
Designed to be 100% self-contained and portable.
"""
import os
from pathlib import Path

# --- Root & Directory Paths (Dynamically resolved relative to this file) -------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
WEIGHTS_DIR  = MODELS_DIR / "saved_weights"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FIGURES_DIR  = PROJECT_ROOT / "figures"
RESEARCH_DIR = PROJECT_ROOT / "research"

# Ensure all necessary runtime folders exist
for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, WEIGHTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, RESEARCH_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- Historical & Operational Timeline ---------------------------------------
YEARS = list(range(2017, 2026))
TRAIN_YEARS = list(range(2017, 2024))  # 7 years (2017–2023)
VAL_YEARS   = [2024]                   # 1 year (2024)
TEST_YEARS  = [2025]                   # 1 year (2025 held-out test)

# --- Remote Sensing Spectral Bands (Sentinel-2 L2A & Sentinel-1 SAR GRD) -----
S2_BANDS = ["B2", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
S1_BANDS = ["VH", "VV"]

# --- Vegetation Indices (Derived Biophysical Features) -----------------------
# Computed on-the-fly from surface reflectance:
# - NDVI: Normalized Difference Vegetation Index (Chlorophyll & biomass density)
# - NDWI: Normalized Difference Water Index (Canopy water content)
# - LSWI: Land Surface Water Index (Soil moisture / flooding signal)
# - REP:  Red Edge Position (Chlorophyll absorption transition)
# - BSI:  Bare Soil Index (Post-harvest soil exposure)
VI_NAMES = ["ndvi", "ndwi", "lswi", "rep", "bsi"]

# --- Temporal Regularization Grid --------------------------------------------
TEMPORAL_RESOLUTION_DAYS = 5       # 5-day regularized grid
SEASON_START_MMDD = "05-01"        # May 1 (Pre-monsoon / early nursery)
SEASON_END_MMDD   = "12-15"        # December 15 (Post-harvest burning window)
MAX_SEQ_LEN       = 36             # 36 timesteps × 5 days = 180 days phenology curve

# --- Causal Interpolation & Smoothing ----------------------------------------
EMA_ALPHA              = 0.3       # Exponential moving average smoothing factor
MAX_GAP_INTERP_DAYS    = 15        # Max causal interpolation gap for cloud cover
CARRY_FORWARD_MAX_DAYS = 45        # Max persistence window for cloud-obscured observations

# --- Pseudo-Labeling Consensus Parameters (Optical + SAR Phenology) ---------
NDVI_PEAK_THRESHOLD         = 0.5   # Minimum peak NDVI for healthy paddy canopy
NDVI_DROP_THRESHOLD         = 0.25  # Post-harvest NDVI floor
NDVI_DROP_RATE              = -0.03 # Minimum rate of NDVI drop per 5-day step
VH_DROP_THRESHOLD_DB        = 3.0   # SAR VH backscatter drop (depolarization loss)
VH_CONFIRMATION_WINDOW_DAYS = 15    # Allowed delta between optical & SAR harvest drop

LABEL_AGREE_HIGH_DAYS       = 3     # ±3 days delta -> High confidence weight (1.0)
LABEL_AGREE_MED_DAYS        = 7     # ±7 days delta -> Medium confidence weight (0.5)
# > 7 days -> Low confidence weight (0.1)

# --- Agrometeorological Variables (Open-Meteo / ERA5 Reanalysis) --------------
WEATHER_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
]

# Growing Degree Days (GDD) Piecewise Thermal Constants for Paddy Rice
GDD_T_BASE  = 10.0   # °C (Base temperature below which growth ceases)
GDD_T_UPPER = 35.0   # °C (Upper thermal threshold / heat stress cap)

# Vapor Pressure Deficit (VPD) Constants (Tetens Formula)
VPD_A = 0.6108
VPD_B = 17.27
VPD_C = 237.3

# --- Rolling & Lag Features --------------------------------------------------
ROLLING_WINDOW_DAYS = 30
LAG_STEPS           = [1, 2, 3, 4]   # 5, 10, 15, 20 days lookback
SOS_NDVI_THRESHOLD  = 0.3            # Start of Season (transplanting / emergence)

# --- Paddy Duration / Variety Clustering -------------------------------------
VARIETY_CLUSTER_K  = 3               # k=3 (Short, Medium, Long duration varieties)
CLUSTER_START_MMDD = "07-01"
CLUSTER_END_MMDD   = "08-31"

# --- Primary Model (LSTM & WTTE-RNN) Hyperparameters -------------------------
LSTM_PARAMS = {
    "hidden_dim": 64,
    "num_layers": 2,
    "embed_dim": 8,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 300,
    "early_stopping_patience": 25,
    "weight_decay": 1e-4,
}

# --- Baseline Model Hyperparameters ------------------------------------------
CATBOOST_PARAMS = {
    "iterations": 10000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "early_stopping_rounds": 500,
    "random_seed": 42,
    "verbose": 200,
    "task_type": "GPU",
}

XGBOOST_AFT_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.0,
    "learning_rate": 0.05,
    "max_depth": 8,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
}

# --- Evaluation Metrics & Tolerances -----------------------------------------
EVAL_WINDOWS = [3, 5, 7, 10, 14]     # Tolerances for harvest operational logistics
LABEL_NOISE_FLOOR_DAYS = 5

# --- Feature Columns ---------------------------------------------------------
CATEGORICAL_FEATURES = ["district_code", "agro_zone", "variety_cluster", "growth_phase"]
EXCLUDE_COLS = [
    "point_id", "year", "date", "lat", "lon",
    "harvest_doy", "harvest_date", "days_to_harvest",
    "label_confidence", "ndvi_harvest_doy", "sar_harvest_doy",
    "split", ".geo", "system:index", "weight", "photoperiod"
]
RAW_BAND_COLS = ["B2", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

if __name__ == "__main__":
    print(f"Loaded Days to Harvest Config:")
    print(f"  Project Root : {PROJECT_ROOT}")
    print(f"  Processed Dir: {PROCESSED_DIR}")
    print(f"  Weights Dir  : {WEIGHTS_DIR}")
    print(f"  Figures Dir  : {FIGURES_DIR}")
