"""
Interactive Inference & Field Demo Script.

Usage:
    python scripts/run_inference_demo.py
    python scripts/run_inference_demo.py --point_id 42
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import HarvestPredictor
from config.config import PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Days to Harvest Inference Demo")
    parser.add_argument("--point_id", type=int, default=10, help="Point ID from the dataset")
    args = parser.parse_args()

    print("=" * 75)
    print("DAYS TO HARVEST PREDICTOR — FIELD INFERENCE DEMO")
    print("=" * 75)

    data_path = PROCESSED_DIR / "training_data.csv"
    if not data_path.exists():
        data_path = PROCESSED_DIR / "demo_sample.csv"
    if not data_path.exists():
        print("Dataset not found for sample extraction.")
        return

    df = pd.read_csv(data_path)
    
    # Select test year (2025) or fallback
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        test_df = df

    available_points = test_df["point_id"].unique()
    selected_pid = args.point_id if args.point_id in available_points else available_points[0]

    farm_data = test_df[test_df["point_id"] == selected_pid].sort_values("date")
    lat = farm_data["lat"].iloc[0] if "lat" in farm_data.columns else "N/A"
    lon = farm_data["lon"].iloc[0] if "lon" in farm_data.columns else "N/A"
    dist = farm_data["district_code"].iloc[0] if "district_code" in farm_data.columns else "N/A"
    ground_truth_doy = farm_data["harvest_doy"].iloc[0] if "harvest_doy" in farm_data.columns else "N/A"

    print(f"\nTarget Farm Point #{selected_pid}:")
    print(f"  Coordinates    : Lat {lat:.4f}, Lon {lon:.4f}")
    print(f"  District / Zone: {dist}")
    print(f"  Ground Truth   : Harvest DOY {ground_truth_doy}")
    print(f"  Observations   : {len(farm_data)} multi-temporal steps ({farm_data['date'].min()} to {farm_data['date'].max()})")

    predictor = HarvestPredictor()
    result = predictor.predict_sequence(farm_data)

    print("\n" + "-" * 55)
    print("PREDICTION OUTPUT FOR FIELD LOGISTICS & SATELLITE APP:")
    print("-" * 55)
    print(f"  Latest Observation Date : {result['latest_observation_date']}")
    print(f"  Predicted Days Remaining: {result['latest_predicted_days_to_harvest']} Days")
    print(f"  Estimated Harvest Date  : {result['estimated_harvest_date']}")
    print(f"  Uncertainty Envelope    : ±{result['confidence_buffer_days']} Days")
    print(f"  Sequential Step Trend   : {result['trajectory_predictions'][-5:]}")
    print("-" * 55)


if __name__ == "__main__":
    main()
