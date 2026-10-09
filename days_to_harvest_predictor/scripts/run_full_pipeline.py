"""
Master Pipeline Runner — Executes Data Preprocessing & Feature Engineering Steps 1 to 8.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --from 3
    python scripts/run_full_pipeline.py --skip-weather
"""
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import (
    step_01_load_clean,
    step_02_vegetation_indices,
    step_03_temporal_interpolation,
    step_04_pseudo_labels,
    step_05_weather_fetch,
    step_06_feature_engineering,
    step_07_variety_clustering,
    step_08_assemble_dataset,
)

STEPS = {
    1: ("Load & Clean Raw Satellite Data", step_01_load_clean.run),
    2: ("Compute Vegetation Indices", step_02_vegetation_indices.run),
    3: ("Causal Temporal Interpolation (5-Day Grid)", step_03_temporal_interpolation.run),
    4: ("Derive Pseudo-Labels (Harvest Detection)", step_04_pseudo_labels.run),
    5: ("Weather Fetch & Agrometeorology", step_05_weather_fetch.run),
    6: ("Causal Feature Engineering", step_06_feature_engineering.run),
    7: ("Variety Duration Clustering", step_07_variety_clustering.run),
    8: ("Assemble Final Training Dataset & Audit", step_08_assemble_dataset.run),
}


def main():
    args = sys.argv[1:]
    skip_weather = "--skip-weather" in args
    args = [a for a in args if not a.startswith("--")]

    start_step = 1
    if "--from" in sys.argv:
        from_idx = sys.argv.index("--from")
        if from_idx + 1 < len(sys.argv) and sys.argv[from_idx + 1].isdigit():
            start_step = int(sys.argv[from_idx + 1])

    steps_to_run = [s for s in STEPS if s >= start_step]
    if skip_weather and 5 in steps_to_run:
        steps_to_run.remove(5)

    print("=" * 75)
    print("  DAYS TO HARVEST PREDICTOR — END-TO-END DATA & FEATURE PIPELINE")
    print("=" * 75)
    print(f"Steps scheduled: {steps_to_run}\n")

    total_start = time.time()
    results = {}

    for step_num in steps_to_run:
        name, func = STEPS[step_num]
        print(f"\n{'#' * 70}")
        print(f"#  STEP {step_num}: {name}")
        print(f"{'#' * 70}")

        t0 = time.time()
        try:
            func()
            elapsed = time.time() - t0
            print(f"[OK] Step {step_num} completed in {elapsed:.1f}s")
            results[step_num] = True
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[FAIL] Step {step_num} FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results[step_num] = False
            break

    total_time = time.time() - total_start
    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    for s, ok in results.items():
        status = "[SUCCESS]" if ok else "[FAILED] "
        print(f"  {status} Step {s}: {STEPS[s][0]}")
    print(f"\nTotal Pipeline Run Time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
