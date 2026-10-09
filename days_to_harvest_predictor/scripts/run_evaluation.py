"""
Model Evaluation & Benchmark Runner.

Usage:
    python scripts/run_evaluation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import run_benchmark


def main():
    print("Running comprehensive test evaluation across all models...")
    bench_df = run_benchmark()
    print("\nEvaluation completed successfully. Benchmark table saved in models/saved_weights/")


if __name__ == "__main__":
    main()
