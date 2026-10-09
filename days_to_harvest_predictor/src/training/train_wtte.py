"""
Training Routine for Probabilistic Model: Weibull Time-To-Event RNN (WTTE-RNN).
Optimizes Weibull Negative Log-Likelihood (NLL) with confidence weight masking.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR, WEIGHTS_DIR, CHECKPOINTS_DIR, LSTM_PARAMS
from src.dataset import create_dataloaders
from src.models.wtte_model import WTTERNN
from src.models.losses import wtte_nll_loss


def train_wtte(
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 25,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("=" * 70)
    print(f"TRAINING PROBABILISTIC MODEL: WTTE-RNN on {device}")
    print("=" * 70)

    train_loader, val_loader, test_loader, meta = create_dataloaders(batch_size=batch_size)
    num_dim = meta["num_dim"]
    cat_dims = meta["cat_dims"]

    model = WTTERNN(
        num_dim=num_dim,
        cat_dims=cat_dims,
        hidden_dim=LSTM_PARAMS.get("hidden_dim", 64),
        num_layers=LSTM_PARAMS.get("num_layers", 2),
        embed_dim=LSTM_PARAMS.get("embed_dim", 8),
        dropout=LSTM_PARAMS.get("dropout", 0.2)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, verbose=True)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "epoch": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        n_train_batches = 0

        for x_num, x_cat, y, w, lengths in train_loader:
            x_num, x_cat, y, w = x_num.to(device), x_cat.to(device), y.to(device), w.to(device)
            mask = (y != -1.0).float()
            optimizer.zero_grad()

            alpha, beta = model(x_num, x_cat)
            loss = wtte_nll_loss(y, alpha, beta, mask=mask, weights=w)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_acc += loss.item()
            n_train_batches += 1

        train_loss = train_loss_acc / max(1, n_train_batches)

        # Validation pass
        model.eval()
        val_loss_acc = 0.0
        val_mae_acc = 0.0
        val_mask_sum = 0
        n_val_batches = 0

        with torch.no_grad():
            for x_num, x_cat, y, w, lengths in val_loader:
                x_num, x_cat, y, w = x_num.to(device), x_cat.to(device), y.to(device), w.to(device)
                mask = (y != -1.0).float()

                alpha, beta = model(x_num, x_cat)
                v_loss = wtte_nll_loss(y, alpha, beta, mask=mask, weights=w)
                val_loss_acc += v_loss.item()
                n_val_batches += 1

                # Median point prediction from Weibull distribution
                preds = alpha * (torch.log(torch.tensor(2.0, device=device)) ** (1.0 / beta))
                abs_err = torch.abs(preds - y) * mask
                val_mae_acc += abs_err.sum().item()
                val_mask_sum += mask.sum().item()

        val_loss = val_loss_acc / max(1, n_val_batches)
        val_mae = val_mae_acc / (val_mask_sum + 1e-5)

        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_mae"].append(float(val_mae))

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train NLL: {train_loss:8.4f} | Val NLL: {val_loss:8.4f} | Val Approx MAE: {val_mae:6.2f} days")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save(best_state, CHECKPOINTS_DIR / "wtte_best_checkpoint.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}. Best Val NLL: {best_val_loss:.4f}.")
            break

    elapsed = time.time() - start_time
    print(f"\nWTTE-RNN training finished in {elapsed:.1f}s.")

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = WEIGHTS_DIR / "wtte_harvest.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved WTTE model -> {save_path}")

    log_path = WEIGHTS_DIR / "wtte_training_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    train_wtte(epochs=100, batch_size=128, lr=1e-3, patience=25)
