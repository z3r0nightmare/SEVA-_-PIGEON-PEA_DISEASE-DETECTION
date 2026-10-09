# Academic Literature Review: Remote Sensing Crop Phenology & Deep Sequence Modeling

## 1. Multi-Sensor Optical & SAR Fusion for Phenological Monitoring

- **Gao et al. (2017) — Optical/SAR Synergy in Crop Monitoring**:
  Demonstrates that combining Sentinel-1 SAR with Sentinel-2 optical data mitigates cloud contamination during tropical and monsoonal growing seasons. SAR VH backscatter correlates strongly with fresh vegetative biomass ($R^2 > 0.82$), while optical Red Edge Position (REP) tracks chlorophyll dynamics past canopy closure.
- **Bousbih et al. (2018) — Soil Moisture and Inundation Detection via LSWI**:
  Proves that SWIR-based indices (LSWI) reliably identify paddy rice transplanting dates through specular reflection signatures during initial field flooding.
- **You et al. (2017) — Deep Learning for Crop Yield and Phenology Forecasting**:
  Establishes that deep sequential models (LSTMs) outperform standard tree-based regressors (Random Forest, Gradient Boosting) when modeling time-series satellite signals due to temporal memory and recurrent parameter sharing.

## 2. Thermal Time & Agrometeorological Modeling

- **Ritchie et al. (1998) — CERES-Rice Phenological Modeling**:
  Formulates the piecewise Growing Degree Day (GDD) equation for Oryza sativa, establishing $T_{\text{base}} = 10^\circ\text{C}$ and optimal development bounds at $30–35^\circ\text{C}$.
- **Allen et al. (1998) — FAO-56 Evapotranspiration & VPD**:
  Standardizes the calculation of reference evapotranspiration ($\text{ET}_0$) and Vapor Pressure Deficit (VPD) for tracking crop water stress and accelerated senescence in arid and semi-arid agro-ecological zones.

## 3. Survival Analysis & Event Time Forecasting

- **Martinsson (2016) — WTTE-RNN: Weibull Time-To-Event Recurrent Neural Network**:
  Introduces parametric time-to-event modeling in deep learning using the Weibull distribution. Enables both point estimation and exact confidence interval quantification for censored and uncensored industrial and biological event forecasting.
- **Nie et al. (2023) — Time Series Transformers and the Patching Mechanism**:
  Explores patch tokenization for long-term multivariate forecasting. Analyzes the failure modes of unmasked self-attention in causal sequential regression tasks.
