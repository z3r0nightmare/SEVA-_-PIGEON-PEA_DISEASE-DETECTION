"""
Step 7: Variety Duration Clustering.
Unsupervised k-means clustering on July-August vegetative NDVI dynamics (Short, Medium, Long duration).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR, VARIETY_CLUSTER_K, CLUSTER_START_MMDD, CLUSTER_END_MMDD


def extract_cluster_features(group: pd.DataFrame) -> dict:
    point_id = group["point_id"].iloc[0]
    year = group["year"].iloc[0]

    start = pd.Timestamp(f"{year}-{CLUSTER_START_MMDD}")
    end   = pd.Timestamp(f"{year}-{CLUSTER_END_MMDD}")
    veg_phase = group[(group["date"] >= start) & (group["date"] <= end)]

    result = {"point_id": point_id, "year": year}
    ndvi = veg_phase["ndvi"].dropna()

    if len(ndvi) < 3:
        result["max_ndvi_jul_aug"] = np.nan
        result["ndvi_rise_rate"] = np.nan
        result["doy_first_ndvi_05"] = np.nan
        return result

    result["max_ndvi_jul_aug"] = ndvi.max()
    x = np.arange(len(ndvi))
    if len(x) >= 2:
        coeffs = np.polyfit(x, ndvi.values, 1)
        result["ndvi_rise_rate"] = coeffs[0]
    else:
        result["ndvi_rise_rate"] = 0.0

    above_05 = veg_phase[veg_phase["ndvi"] > 0.5]
    if len(above_05) > 0:
        first_date = pd.Timestamp(above_05["date"].iloc[0])
        result["doy_first_ndvi_05"] = first_date.dayofyear
    else:
        result["doy_first_ndvi_05"] = 999

    return result


def run():
    print("=" * 70)
    print("STEP 7: Variety Duration Clustering")
    print("=" * 70)
    in_path = PROCESSED_DIR / "interpolated_5day.csv"
    if not in_path.exists():
        print(f"Error: {in_path.name} not found.")
        return

    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])

    features = []
    for _, group in df.groupby(["point_id", "year"]):
        feat = extract_cluster_features(group.sort_values("date"))
        features.append(feat)

    feat_df = pd.DataFrame(features)
    feat_cols = ["max_ndvi_jul_aug", "ndvi_rise_rate", "doy_first_ndvi_05"]
    valid = feat_df.dropna(subset=feat_cols)

    if len(valid) < VARIETY_CLUSTER_K:
        feat_df["variety_cluster"] = 0
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(valid[feat_cols].values)
        kmeans = KMeans(n_clusters=VARIETY_CLUSTER_K, random_state=42, n_init=10)
        valid_clusters = kmeans.fit_predict(X)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        order = np.argsort(centroids[:, 0])
        cluster_map = {old: new for new, old in enumerate(order)}
        valid_clusters = np.array([cluster_map[c] for c in valid_clusters])
        feat_df.loc[valid.index, "variety_cluster"] = valid_clusters
        most_common = int(pd.Series(valid_clusters).mode()[0])
        feat_df["variety_cluster"] = feat_df["variety_cluster"].fillna(most_common)

    feat_df["variety_cluster"] = feat_df["variety_cluster"].astype(np.int8)
    out_df = feat_df[["point_id", "year", "variety_cluster"]].copy()
    out_path = PROCESSED_DIR / "variety_clusters.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.name}")


if __name__ == "__main__":
    run()
