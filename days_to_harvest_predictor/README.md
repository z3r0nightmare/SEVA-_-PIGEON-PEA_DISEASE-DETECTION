# 🌾 Days to Harvest (DTH) Predictor — Multi-Sensor Phenology & Causal Sequence Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Remote Sensing](<https://img.shields.io/badge/Sensors-Sentinel--1%20SAR%20%2B%20Sentinel--2%20Optical-brightgreen>)](https://sentinels.copernicus.eu/)
[![Model](<https://img.shields.io/badge/Primary%20Model-Causal%202--Layer%20LSTM-orange.svg>)]()
[![Hackathon](<https://img.shields.io/badge/Hackathon-Seva%20Bharat%20Ready-success.svg>)]()

> **A Standalone, Self-Contained Phenology & Harvest Prediction Package for Agricultural Waste Valorization and Stubble Burning Mitigation.**

---

## Executive Summary & Problem Context (*Seva Bharat*)

Every autumn across the Indo-Gangetic Plains (Punjab, Haryana, Western Uttar Pradesh), farmers face a narrow **15-to-20 day window** between paddy (rice) harvest and wheat sowing. Due to severe labor shortages and lack of synchronized baling equipment, farmers resort to burning over **35 million tons of crop residue (stubble)**, causing hazardous smog, greenhouse gas emissions, and severe respiratory crises across North India.

### The Technological Solution

The **Days to Harvest (DTH) Prediction Engine** forecasts field-level crop harvest dates up to **90 days in advance** using multi-sensor satellite remote sensing (Sentinel-1 SAR + Sentinel-2 Optical) and high-resolution agrometeorological reanalysis (Open-Meteo / ERA5).

By providing highly accurate, forward-looking harvest dates:

1. **Biomass Aggregators & Transporters** can pre-route balers and tractor fleets days before fields are cleared.
2. **Bio-Energy & Pelletization Plants** receive predictable feedstock delivery schedules.
3. **Government & Extension Agencies** can monitor harvest progression district-by-district in real time to enforce zero-burn zones.

---

## Why Causal LSTM is the Primary Architecture (Zero Lookahead Leakage)

### The PatchTST Future-Leakage Post-Mortem

During initial architectural benchmarking, a **Patch Time Series Transformer (PatchTST)** model initially posted an apparent MAE of `1.91 days`. However, exhaustive causal auditing revealed **severe bidirectional temporal leakage**:

- In PatchTST, unmasked self-attention across 30-day patches allowed early-season timesteps ($t=15\text{d}$) to look ahead at the late-season ($t=140\text{d}$) catastrophic NDVI drop.
- While mathematically impressive in an offline matrix multiplication, this model **fails in production** because future satellite passes do not exist during live real-time inference.

### The Causal LSTM Guarantee

Our primary production model is a **2-Layer Unidirectional Causal LSTM Regressor**:

- **Strictly Causal (`bidirectional=False`)**: State $h_t$ is purely a recurrent function of observations up to $t$ ($x_1, \dots, x_t$).
- **Multi-Sensor Latent Fusion**: Combines 5 optical vegetation indices, 3 SAR polarimetric variables, and 8 meteorological features with learnable entity embeddings for spatial categoricals (districts, agro-zones, variety clusters).
- **Weibull Time-To-Event (WTTE-RNN)**: Provides complementary probabilistic survival analysis with exact 10th-to-90th percentile confidence envelopes.

```mermaid
graph LR
    subgraph Data Input
        S2["Sentinel-2 Optical<br>(NDVI, NDWI, LSWI, REP, BSI)"]
        S1["Sentinel-1 SAR<br>(VH, VV, VH/VV)"]
        WX["Agrometeorology<br>(GDD, VPD, Precip, ET0)"]
    end

    subgraph Causal Sequence Engine
        EMB["Joint Latent Embedding<br>(64-Dim Projection)"]
        LSTM["2-Layer Unidirectional LSTM<br>(Zero Future Lookahead)"]
        HEAD["Causal Linear Regressor"]
    end

    subgraph Operational Output
        DTH["Days to Harvest<br>(e.g., 14.2 Days Remaining)"]
        CAL["Estimated Calendar Date<br>(e.g., 2025-10-28)"]
        CONF["Confidence Window<br>(±4.2 Days)"]
    end

    S2 --> EMB
    S1 --> EMB
    WX --> EMB
    EMB --> LSTM
    LSTM --> HEAD
    HEAD --> DTH
    HEAD --> CAL
    HEAD --> CONF
```

---

## Comprehensive Benchmark Results (Held-Out 2025 Test Set)

Evaluated strictly on the **unseen 2025 harvest season** (100% locked away during training and hyperparameter tuning):

| Model Architecture                          | MAE (Full Season) | MAE (< 30d to Harvest) |        RMSE        | Accuracy (±5 Days) | Accuracy (±14 Days) |       Causal Validity       |
| :------------------------------------------ | :----------------: | :--------------------: | :-----------------: | :-----------------: | :------------------: | :-------------------------: |
| **Tier 0: Naive Historical Median**   |     16.7 days     |       15.1 days       |      25.8 days      |        25.1%        |        68.3%        |          ✅ Causal          |
| **Tier 1a: CatBoost Regressor**       |     13.8 days     |        8.9 days        |      19.7 days      |        28.1%        |        64.3%        |          ✅ Causal          |
| **Tier 1b: XGBoost-AFT (Survival)**   |     14.7 days     |        9.4 days        |      21.2 days      |        27.0%        |        61.6%        |          ✅ Causal          |
| **Primary Model: Causal LSTM (SOTA)** | **9.4 days** |   **4.2 days**   | **14.1 days** |   **48.6%**   |   **82.4%**   | ✅**Strictly Causal** |

> [!NOTE]
> **Operational Lead-Time Dynamic**: As harvest approaches, error drops sharply from ~9.4 days at 90 days out to **< 4.2 days** in the critical 30-day pre-harvest harvest window, providing high reliability for machine dispatch.

---

## Repository & Directory Layout

```
days_to_harvest_predictor/
│
├── README.md                      # Complete documentation & project guide
├── requirements.txt                # Self-contained Python dependencies
├── .gitignore                      # Git tracking configuration
│
├── config/
│   ├── __init__.py
│   └── config.py                  # Centralized configuration with relative path discovery
│
├── data/
│   ├── raw/                       # Raw Sentinel-1 & Sentinel-2 spectral data (2017–2025)
│   └── processed/                 # Feature-engineered training data, pseudo-labels, clusters
│       ├── training_data.csv      # Unified 135k-row training dataset (splits: train, val, test)
│       ├── weather_daily.csv      # Daily Open-Meteo agrometeorological dataset
│       └── dataset_metadata.json  # Data schema and partition statistics
│
├── models/
│   ├── saved_weights/             # Production model checkpoints & benchmarks
│   │   ├── lstm_harvest.pth       # Primary SOTA Causal LSTM weights
│   │   ├── wtte_harvest.pth       # Probabilistic WTTE-RNN weights
│   │   ├── catboost_harvest.cbm   # CatBoost baseline weights
│   │   ├── xgboost_aft_harvest.json # XGBoost-AFT survival baseline weights
│   │   └── benchmark_comparison_table.csv # Full evaluation table
│   └── checkpoints/               # Training runtime checkpoints
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # AgriSequenceDataset & causal padding/masking loaders
│   ├── models/                    # Model architecture definitions
│   │   ├── __init__.py
│   │   ├── lstm_model.py          # Primary Causal LSTM Phenology Regressor
│   │   ├── wtte_model.py          # Weibull Time-To-Event RNN (Probabilistic)
│   │   ├── baselines.py           # Naive, CatBoost, XGBoost-AFT baselines
│   │   └── losses.py              # Masked MSE, Masked Huber, Weibull NLL loss
│   ├── pipeline/                  # Modular data preprocessing steps (1–8)
│   │   ├── step_01_load_clean.py
│   │   ├── step_02_vegetation_indices.py
│   │   ├── step_03_temporal_interpolation.py
│   │   ├── step_04_pseudo_labels.py
│   │   ├── step_05_weather_fetch.py
│   │   ├── step_06_feature_engineering.py
│   │   ├── step_07_variety_clustering.py
│   │   └── step_08_assemble_dataset.py
│   ├── training/                  # Model training engines
│   │   ├── train_lstm.py          # Primary LSTM training with early stopping & scheduler
│   │   ├── train_wtte.py          # WTTE-RNN training
│   │   ├── train_catboost.py      # CatBoost baseline training
│   │   └── train_xgboost.py       # XGBoost-AFT baseline training
│   ├── evaluation/                # Test evaluation & benchmarking
│   │   ├── evaluate.py            # Comprehensive test set evaluation
│   │   └── benchmark.py           # Cross-model comparative suite
│   ├── visualization/             # 300+ DPI publication graph generation
│   │   └── generate_research_plots.py
│   └── inference/                 # Real-time field prediction API & CLI
│       └── predict.py             # Single-point & batch harvest forecasting
│
├── notebooks/                     # Interactive Jupyter Notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_lstm_harvest_prediction.ipynb
│   └── 03_model_evaluation_and_research.ipynb
│
├── research/                      # Scientific papers & technical reports
│   ├── methodology_report.md      # Full mathematical & biophysical formulation
│   ├── patchtst_leakage_analysis.md # Detailed post-mortem on Transformer data leakage
│   └── literature_review.md       # Academic literature & theoretical foundations
│
├── figures/                       # Publication-Grade High-DPI Visualizations (300 DPI)
│   ├── 01_loss_curves.png
│   ├── 02_prediction_vs_actual_scatter.png
│   ├── 03_lead_time_error_analysis.png
│   ├── 04_residual_distribution_qq.png
│   ├── 05_phenology_trajectory_sample.png
│   ├── 06_feature_correlation_heatmap.png
│   └── 07_model_benchmark_comparison.png
│
└── scripts/                       # One-click CLI entry points
    ├── run_full_pipeline.py       # Execute full data preprocessing (Steps 1–8)
    ├── run_training.py            # Train all models
    ├── run_evaluation.py          # Run benchmark evaluation
    ├── generate_all_figures.py    # Generate all 300+ DPI research graphics
    └── run_inference_demo.py      # Run sample farm inference demo
```

---

## Research Visualizations 

All high-DPI figures are automatically generated into `figures/`:

| Figure                                                              | Description                                                                                                                                            |
| :------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![Loss Curves](figures/01_loss_curves.png)                          | **Convergence Dynamics**: Training MSE Loss vs Validation MAE across 150 epochs with optimal checkpoint discovery.                               |
| ![Prediction Scatter](figures/02_prediction_vs_actual_scatter.png)  | **Ground Truth vs Prediction Scatter**: 2025 held-out test set with 1:1 identity line, $\pm 5\text{d}$ / $\pm 14\text{d}$ operational bands. |
| ![Lead Time Error](figures/03_lead_time_error_analysis.png)         | **Lead-Time Accuracy Horizon**: Shows error dropping from 9.4d down to < 4.2d as harvest date approaches.                                        |
| ![Residuals](figures/04_residual_distribution_qq.png)               | **Error Residual Diagnostics**: Gaussian KDE and Normal Q-Q plot proving zero-mean unbiased prediction.                                          |
| ![Phenology Trajectory](figures/05_phenology_trajectory_sample.png) | **Multi-Sensor Trajectory**: Dual-axis plot of Optical NDVI and SAR VH progression with real-time LSTM countdown.                                |
| ![Feature Correlation](figures/06_feature_correlation_heatmap.png)  | **Correlation Matrix**: Spearman rank correlation heatmap across spectral, thermal, and phenological variables.                                  |
| ![Benchmark](figures/07_model_benchmark_comparison.png)             | **Model Comparison Chart**: Comparative bar chart of Naive vs CatBoost vs XGBoost-AFT vs Primary LSTM.                                           |

## Quickstart Guide

### 1. Installation

Clone or copy this standalone folder and install dependencies:

```bash
cd days_to_harvest_predictor
pip install -r requirements.txt
```

### 2. Run Single Farm Inference Demo

Test real-time sequential prediction on a sample farm from the test set:

```bash
python scripts/run_inference_demo.py --point_id 10
```

**Sample Output:**

```
===========================================================
HARVEST PREDICTION OUTPUT FOR FIELD LOGISTICS:
===========================================================
  Latest Observation Date : 2025-10-15
  Predicted Days Remaining: 12.4 Days
  Estimated Harvest Date  : 2025-10-27
  Confidence Envelope     : ±4.2 Days
  Sequential Step Trend   : [22.1, 18.5, 15.3, 13.1, 12.4]
===========================================================
```

### 3. Run Benchmark Evaluation

Evaluate all models on the held-out 2025 test dataset:

```bash
python scripts/run_evaluation.py
```

### 4. Generate All Research Figures (300+ DPI)

Generate all publication-ready performance plots in `figures/`:

```bash
python scripts/generate_all_figures.py
```

### 5. Train Models from Scratch (Optional)

```bash
# Train primary Causal LSTM Regressor
python scripts/run_training.py --model lstm --epochs 150

# Train all models (LSTM, WTTE, CatBoost, XGBoost-AFT)
python scripts/run_training.py --model all
```

### 6. Run Full Data Pipeline (Optional)

To re-run data extraction, vegetation index computation, and causal feature engineering from raw satellite CSVs:

```bash
python scripts/run_full_pipeline.py
```

---

## Python API Usage

```python
import pandas as pd
from src.inference.predict import HarvestPredictor

# 1. Initialize predictor with pre-trained weights
predictor = HarvestPredictor()

# 2. Load farm time-series observations
farm_df = pd.read_csv("path/to/farm_satellite_timeseries.csv")

# 3. Generate harvest prediction
result = predictor.predict_sequence(farm_df)

print(f"Predicted Days to Harvest: {result['latest_predicted_days_to_harvest']} days")
print(f"Estimated Harvest Date   : {result['estimated_harvest_date']}")
print(f"Confidence Window        : ±{result['confidence_buffer_days']} days")
```

---

## Mathematical Formulations

### 1. Growing Degree Days (Piecewise Thermal Time)

$$
\text{GDD}_{\text{daily}} = \max\left(0, \min\left(\frac{T_{\text{max}} + T_{\text{min}}}{2}, T_{\text{upper}}\right) - T_{\text{base}}\right)
$$

*where $T_{\text{base}} = 10.0^\circ\text{C}$ and $T_{\text{upper}} = 35.0^\circ\text{C}$ for paddy rice.*

### 2. Vapor Pressure Deficit (Tetens Equation)

$$
e_s(T) = 0.6108 \cdot \exp\left(\frac{17.27 \cdot T}{T + 237.3}\right)
$$

$$
\text{VPD} = e_s\left(\frac{T_{\text{max}} + T_{\text{min}}}{2}\right) - e_s(T_{\text{min}})
$$

### 3. Weibull Time-To-Event Hazard Distribution

$$
f(t; \alpha, \beta) = \frac{\beta}{\alpha} \left(\frac{t}{\alpha}\right)^{\beta - 1} \exp\left(-\left(\frac{t}{\alpha}\right)^\beta\right)
$$

$$
\mathbb{E}[T] = \alpha \cdot \Gamma\left(1 + \frac{1}{\beta}\right)
$$

---

## Credits

> Built with **Sentinel-1/2 Earth Observation Data (ESA Copernicus)**, **Open-Meteo Historical Agrometeorology**, and **PyTorch Deep Sequence Modeling**.
