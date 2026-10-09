"""
Model Training Runner — Trains Primary Causal LSTM, WTTE-RNN, and Baselines.

Usage:
    python scripts/run_training.py              # Train all models
    python scripts/run_training.py --model lstm  # Train primary LSTM only
    python scripts/run_training.py --model wtte  # Train WTTE only
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_lstm import train_lstm
from src.training.train_wtte import train_wtte
from src.training.train_catboost import train_catboost
from src.training.train_xgboost import train_xgboost


def main():
    parser = argparse.ArgumentParser(description="Model Training Suite")
    parser.add_argument("--model", type=str, default="all", choices=["all", "lstm", "wtte", "catboost", "xgboost"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=150, help="Epochs for neural network training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    print("=" * 75)
    print("DAYS TO HARVEST PREDICTOR — MODEL TRAINING SUITE")
    print("=" * 75)

    if args.model in ["all", "lstm"]:
        print("\n>>> [1/4] Training Primary SOTA Model: Causal LSTM Regressor...")
        train_lstm(epochs=args.epochs, batch_size=args.batch_size)

    if args.model in ["all", "wtte"]:
        print("\n>>> [2/4] Training Probabilistic Model: WTTE-RNN...")
        train_wtte(epochs=args.epochs, batch_size=args.batch_size)

    if args.model in ["all", "catboost"]:
        print("\n>>> [3/4] Training Baseline: CatBoost Regressor...")
        try:
            train_catboost()
        except Exception as e:
            print(f"CatBoost training notice: {e}")

    if args.model in ["all", "xgboost"]:
        print("\n>>> [4/4] Training Baseline: XGBoost-AFT Survival Model...")
        try:
            train_xgboost()
        except Exception as e:
            print(f"XGBoost training notice: {e}")

    print("\nTraining suite execution complete! Weights saved in models/saved_weights/")


if __name__ == "__main__":
    main()
