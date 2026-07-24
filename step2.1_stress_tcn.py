"""
step2.1_stress_tcn.py — Stress Zone 2-layer TCN (Dual-Mode Q2)
=================================================================
Only trains on FILT_NTU >= THETA_MODEL (stress zone, ~800-960 samples).

Model design:
  Input window: 8 steps (16h), shift-1: uses t-8..t-1 to predict FILT(t)
                Shift-1 ensures physical causality: input at t-1 is the
                most recent that could physically have reached the filter.

  Target:       Delta_FILT(t) = FILT(t) - FILT(t-1)
                Avoids auto-correlation false signal.

  Architecture: 2-layer TCN, dil=[1,2], kernel=3, RF=7 steps=14h.
                CausalConv1d prevents future information leakage.

  Loss:         Huber(delta=1.0) + lambda1 * L1(Delta_FILT_hat)
                Sparsity: most Delta_FILT ~ 0 even in stress zone.

Output: stress_tcn_model.pt, q2_stress_metrics.csv,
        q2_lag_weights.json (distributed lag from kernel analysis)
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stress zone threshold
THETA_MODEL = 0.15

# TCN hyperparameters
WINDOW = 8       # input window (steps), 8*2h=16h
N_VARS = 3       # RW_FLOW, ALUM, RW_NTU
HIDDEN = 32      # hidden channels
N_LAYERS = 2     # TCN layers
KERNEL = 3
DILATIONS = [1, 2]
BATCH_SIZE = 64
LR = 3e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 25
N_SPLITS = 5
LAMBDA_L1 = 0.01  # sparsity penalty


class CausalConv1d(nn.Module):
    """1D causal convolution: padding on the left only, trim right."""
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :out.shape[2] - self.pad]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.1):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class StressTCN(nn.Module):
    """2-layer TCN for stress zone DELTA_FILT prediction."""
    def __init__(self, n_vars=N_VARS, hidden=HIDDEN, kernel=KERNEL,
                 dilations=DILATIONS, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i, dil in enumerate(dilations):
            in_ch = n_vars if i == 0 else hidden
            self.blocks.append(TCNBlock(in_ch, hidden, kernel, dil, dropout))
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, V) -> (B, V, T) for Conv1d
        h = x.permute(0, 2, 1)
        for block in self.blocks:
            h = block(h)
        # Global average over last 4 positions -> (B, HIDDEN)
        h_last = h[:, :, -4:].mean(dim=2)
        return self.fc(h_last).squeeze(-1)


def load_and_build_sequences(clean_csv, theta=THETA_MODEL):
    df = pd.read_csv(clean_csv)
    required = ["RW_FLOW", "ALUM", "RW_NTU", "FILT_NTU"]
    df = df.dropna(subset=required)
    n = len(df)

    rw_flow = df["RW_FLOW"].values.astype(np.float32)
    alum    = df["ALUM"].values.astype(np.float32)
    rw_ntu  = df["RW_NTU"].values.astype(np.float32)
    filt    = df["FILT_NTU"].values.astype(np.float32)

    # Shift-1 input: use t-WINDOW..t-1 to predict FILT(t)
    # Pad with first valid values at the front
    X_data = np.column_stack([rw_flow, alum, rw_ntu])  # (n, 3)
    # Build sequences: for each t, take X[t-WINDOW : t] as input window
    # This naturally provides shift-1: window is [t-WINDOW, t-1], not including t

    seqs_x, seqs_y, seqs_idx = [], [], []
    for t in range(WINDOW, n):
        x_window = X_data[t - WINDOW:t]          # (WINDOW, 3)
        y_target = filt[t] - filt[t - 1]          # Delta_FILT
        seqs_x.append(x_window)
        seqs_y.append(y_target)
        seqs_idx.append(t)

    X = np.stack(seqs_x).astype(np.float32)       # (n-WINDOW, WINDOW, 3)
    Y = np.array(seqs_y, dtype=np.float32)
    idx = np.array(seqs_idx, dtype=np.int32)
    filt_all = filt[idx]

    # Stress zone mask: FILT(t) >= theta
    stress_mask = filt_all >= theta
    X_s = X[stress_mask]
    Y_s = Y[stress_mask]

    print(f"  Total sequences: {len(X)}, stress zone: {len(X_s)} ({100*len(X_s)/max(1,len(X)):.1f}%)")
    return X_s, Y_s, stress_mask.sum()


def train_epoch(model, loader, optimizer):
    model.train()
    model.to(DEVICE)
    total_loss = 0.0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        pred = model(bx)
        # Huber loss
        diff = pred - by
        abs_diff = torch.abs(diff)
        mask = abs_diff <= HUBER_DELTA
        huber = torch.where(mask, 0.5 * diff ** 2,
                            HUBER_DELTA * (abs_diff - 0.5 * HUBER_DELTA))
        l1 = LAMBDA_L1 * torch.mean(torch.abs(pred))
        loss = huber.mean() + l1
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(bx)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    model.to(DEVICE)
    preds, trues = [], []
    for bx, by in loader:
        bx = bx.to(DEVICE)
        pred = model(bx)
        preds.append(pred.cpu().numpy())
        trues.append(by.numpy())
    yp = np.concatenate(preds)
    yt = np.concatenate(trues)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    return rmse, r2, mae, yp, yt


def extract_lag_weights(model, X_sample):
    model.eval()
    model.to(DEVICE)
    X_tensor = torch.FloatTensor(X_sample[:1]).to(DEVICE)
    base = model(X_tensor).item()
    weights = np.zeros((WINDOW, N_VARS))
    eps = 0.02

    for t in range(WINDOW):
        for v in range(N_VARS):
            X_pert = X_tensor.clone()
            X_pert[0, t, v] += eps
            delta = (model(X_pert).item() - base) / max(eps, 1e-8)
            weights[t, v] = abs(delta)

    for v in range(N_VARS):
        s = weights[:, v].sum()
        if s > 0:
            weights[:, v] /= s
    return weights


def main():
    print("=" * 60)
    print("  step2.1 — Stress Zone TCN (Dual-Mode Q2)")
    print("=" * 60)

    clean_csv = os.path.join(OUTPUT_DIR, "clean_data.csv")
    X_s, Y_s, n_stress = load_and_build_sequences(clean_csv, THETA_MODEL)

    if n_stress < 50:
        print(f"  [ERROR] Only {n_stress} stress samples, insufficient for training.")
        return

    n = len(X_s)
    print(f"  Stress samples: {n}, Delta_FILT mean={Y_s.mean():.4f} std={Y_s.std():.4f}")

    # 5-fold CV on stress subset
    print(f"\n[CV] {N_SPLITS}-fold TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []
    best_r2, best_state = -999, None

    for fold, (tr, vl) in enumerate(tscv.split(np.arange(n).reshape(-1, 1))):
        X_tr, Y_tr = X_s[tr], Y_s[tr]
        X_vl, Y_vl = X_s[vl], Y_s[vl]

        tr_ldr = DataLoader(TensorDataset(torch.FloatTensor(X_tr),
                                           torch.FloatTensor(Y_tr)),
                            batch_size=BATCH_SIZE, shuffle=False)
        vl_ldr = DataLoader(TensorDataset(torch.FloatTensor(X_vl),
                                           torch.FloatTensor(Y_vl)),
                            batch_size=BATCH_SIZE, shuffle=False)

        model = StressTCN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

        best_rmse, patience_cnt = float("inf"), 0
        best_fold_state = None

        for epoch in range(MAX_EPOCHS):
            train_epoch(model, tr_ldr, optimizer)
            rmse_v, _, _, _, _ = evaluate(model, vl_ldr)
            scheduler.step(rmse_v)
            if rmse_v < best_rmse:
                best_rmse = rmse_v
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

        model.load_state_dict(best_fold_state)
        rmse_v, r2_v, mae_v, yp, yt = evaluate(model, vl_ldr)
        fold_metrics.append({"fold": fold, "rmse": rmse_v, "r2": r2_v, "mae": mae_v})
        if r2_v > best_r2:
            best_r2 = r2_v
            best_state = {k: v.cpu().clone() for k, v in best_fold_state.items()}

        print(f"  Fold{fold}: RMSE={rmse_v:.4f} R2={r2_v:.4f} MAE={mae_v:.4f} "
              f"n_train={len(tr)} n_val={len(vl)}")

    # Extract lag weights using best fold's data
    best_model = StressTCN()
    if best_state:
        best_model.load_state_dict(best_state)
    best_model.to(DEVICE)
    lag_weights = extract_lag_weights(best_model, X_s)
    torch.save(best_model.state_dict(), os.path.join(OUTPUT_DIR, "stress_tcn_model.pt"))
    lag_summary = {
        "RW_FLOW": {"weights": lag_weights[:, 0].tolist(),
                     "peak_lag": int(np.argmax(lag_weights[:, 0])),
                     "peak_h": int(np.argmax(lag_weights[:, 0]) * 2)},
        "ALUM": {"weights": lag_weights[:, 1].tolist(),
                  "peak_lag": int(np.argmax(lag_weights[:, 1])),
                  "peak_h": int(np.argmax(lag_weights[:, 1]) * 2)},
        "RW_NTU": {"weights": lag_weights[:, 2].tolist(),
                    "peak_lag": int(np.argmax(lag_weights[:, 2])),
                    "peak_h": int(np.argmax(lag_weights[:, 2]) * 2)},
    }
    with open(os.path.join(OUTPUT_DIR, "q2_lag_weights.json"), "w", encoding="utf-8") as f:
        json.dump(lag_summary, f, indent=2, ensure_ascii=False)

    # Average metrics
    avg_rmse = np.mean([f["rmse"] for f in fold_metrics])
    avg_r2   = np.mean([f["r2"] for f in fold_metrics])
    avg_mae  = np.mean([f["mae"] for f in fold_metrics])

    print(f"\n  5-fold mean: RMSE={avg_rmse:.4f} R2={avg_r2:.4f} MAE={avg_mae:.4f}")
    for var, info in lag_summary.items():
        print(f"  {var}: peak_lag={info['peak_lag']} ({info['peak_h']}h) "
              f"wmax={max(info['weights']):.4f}")

    # Save metrics
    pd.DataFrame(fold_metrics).to_csv(
        os.path.join(OUTPUT_DIR, "q2_stress_metrics.csv"), index=False, encoding="utf-8-sig")

    # Figure: lag weights
    fig, ax = plt.subplots(figsize=(10, 4))
    lag_h = np.arange(WINDOW) * 2
    colors = {"RW_FLOW": "steelblue", "ALUM": "darkorange", "RW_NTU": "seagreen"}
    for var, color in colors.items():
        ax.plot(lag_h, lag_weights[:, list(colors.keys()).index(var)],
                "-o", color=color, lw=2, ms=5, label=var)
    ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Normalized weight")
    ax.set_title(f"Distributed Lag Weights (Stress Zone n={n})")
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_lag_weights.png"), dpi=300)
    plt.close()

    print(f"\n  [DONE] stress_tcn_model.pt, q2_stress_metrics.csv, q2_lag_weights.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
