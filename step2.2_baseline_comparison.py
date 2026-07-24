"""
step2.2_baseline_comparison.py — AR Baseline on Stress Zone
=============================================================
Re-run AR(6) and ARMAX(6,4) on stress zone subset (FILT >= 0.15)
for fair comparison with Stress TCN.
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THETA = 0.15
AR_LAGS = 6


def load_stress_data(clean_csv):
    df = pd.read_csv(clean_csv)
    df = df.dropna(subset=["FILT_NTU", "RW_FLOW", "ALUM", "RW_NTU"])
    filt = df["FILT_NTU"].values.astype(np.float64)
    rw_flow = df["RW_FLOW"].values.astype(np.float64)
    alum = df["ALUM"].values.astype(np.float64)
    rw_ntu = df["RW_NTU"].values.astype(np.float64)

    n = len(filt)
    # Build lag features + Delta_FILT target (same as TCN)
    X_all, y_all, mask = [], [], []
    for t in range(AR_LAGS, n):
        feats = []
        feats.append(rw_flow[t - 1])    # shift-1
        feats.append(alum[t - 1])
        feats.append(rw_ntu[t - 1])
        for lag in range(1, AR_LAGS + 1):
            feats.append(filt[t - lag])
        X_all.append(feats)
        y_all.append(filt[t] - filt[t - 1])
        mask.append(filt[t] >= THETA)

    X_all = np.array(X_all, dtype=np.float64)
    y_all = np.array(y_all, dtype=np.float64)
    mask = np.array(mask, dtype=bool)

    X_s, y_s = X_all[mask], y_all[mask]
    return X_s, y_s


def evaluate(X, y, model_cls, fit_kwargs=None):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rmses, r2s, maes = [], [], []
    for tr, vl in tscv.split(X):
        model = model_cls()
        kw = fit_kwargs or {}
        model.fit(X[tr], y[tr], **kw)
        pred = model.predict(X[vl])
        rmse = np.sqrt(np.mean((pred - y[vl]) ** 2))
        r2 = 1 - np.sum((pred - y[vl])**2) / (np.sum((y[vl] - y[vl].mean())**2) + EPS)
        maes.append(np.mean(np.abs(pred - y[vl])))
        rmses.append(rmse); r2s.append(r2)
    return np.mean(rmses), np.mean(r2s), np.mean(maes)


class AR6:
    def fit(self, X, y, **kw):
        ar_feats = X[:, 3:3+AR_LAGS]  # FILT lags
        X_ar = np.column_stack([np.ones(len(y)), ar_feats])
        th = np.linalg.lstsq(X_ar, y, rcond=None)[0]
        self.intercept = th[0]; self.coef = th[1:]

    def predict(self, X):
        return self.intercept + X[:, 3:3+AR_LAGS] @ self.coef


class ARMAX:
    def __init__(self):
        self.lr = LinearRegression()
    def fit(self, X, y, **kw):
        self.lr.fit(X, y)
    def predict(self, X):
        return self.lr.predict(X)


def main():
    clean_csv = os.path.join(OUTPUT_DIR, "clean_data.csv")
    X, y = load_stress_data(clean_csv)
    print(f"Stress zone samples: {len(y)}")

    # Also compute "naive" baseline: always predict 0 (Delta_FILT mean)
    naive_rmse = np.sqrt(np.mean(y ** 2))
    naive_r2 = 0.0
    print(f"Naive (Delta=0): RMSE={naive_rmse:.4f}")

    # AR(6)
    rmse1, r21, mae1 = evaluate(X, y, AR6)
    print(f"AR(6):            RMSE={rmse1:.4f} R2={r21:.4f}")

    # ARMAX
    rmse2, r22, mae2 = evaluate(X, y, ARMAX)
    print(f"ARMAX:            RMSE={rmse2:.4f} R2={r22:.4f}")

    # Save
    results = [
        {"model": "Naive(0)", "RMSE": naive_rmse, "R2": naive_r2, "MAE": 0},
        {"model": "AR(6)", "RMSE": round(rmse1,4), "R2": round(r21,4), "MAE": round(mae1,4)},
        {"model": "ARMAX", "RMSE": round(rmse2,4), "R2": round(r22,4), "MAE": round(mae2,4)},
    ]
    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "q2_stress_baseline.csv"), index=False, encoding="utf-8-sig")
    print("[DONE] q2_stress_baseline.csv")


if __name__ == "__main__":
    main()
