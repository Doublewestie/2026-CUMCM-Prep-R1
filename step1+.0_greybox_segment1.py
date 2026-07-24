"""
step1.0_greybox_segment1.py — 段1: 混凝+沉淀+过滤 → FILT.NTU
================================================================
灰箱递推方程:
  FILT(t) = β₁·FILT(t-1) + (1-β₁)·RW_NTU(t)·[1 - η(t)]

  其中:
    η(t) = ALUM(t) / (ALUM(t) + K_m(t) + α·CLR(t))
    K_m(t) = K_m₀ + K_m₁·day_sin(t) + K_m₂·day_cos(t)

参数 (6个): β₁, K_m₀, K_m₁, K_m₂, α, FILT(0)

输入: clean_data.csv
输出: segment1_params.json, segment1_metrics.csv, FILT_pred.npy
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

PARAM_NAMES = ["beta1", "Km0", "Km1", "Km2", "alpha", "FILT0"]
PARAM_INIT = np.array([
    GREYBOX_PARAM_INIT["beta1"],
    GREYBOX_PARAM_INIT["Km0"],
    GREYBOX_PARAM_INIT["Km1"],
    GREYBOX_PARAM_INIT["Km2"],
    GREYBOX_PARAM_INIT["alpha"],
    GREYBOX_PARAM_INIT["FILT0"],
])
PARAM_BOUNDS = [
    GREYBOX_PARAM_BOUNDS["beta1"],
    GREYBOX_PARAM_BOUNDS["Km0"],
    GREYBOX_PARAM_BOUNDS["Km1"],
    GREYBOX_PARAM_BOUNDS["Km2"],
    GREYBOX_PARAM_BOUNDS["alpha"],
    GREYBOX_PARAM_BOUNDS["FILT0"],
]

LAM = GREYBOX_LAMBDA


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    df = df.dropna(subset=["RW_NTU", "FILT_NTU", "ALUM", "CLR", "DATE"])
    n = len(df)

    rw_ntu  = df["RW_NTU"].values.astype(np.float64)
    filt    = df["FILT_NTU"].values.astype(np.float64)
    alum    = df["ALUM"].values.astype(np.float64)
    clr     = df["CLR"].values.astype(np.float64)
    date    = pd.to_datetime(df["DATE"])
    day_of_year = date.dt.dayofyear.values

    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)

    return rw_ntu, filt, alum, clr, day_sin, day_cos


def run_segment1(params, rw_ntu, filt_obs, alum, clr, day_sin, day_cos):
    """
    正向递推段1，返回预测序列 + 损失。

    params: [β₁, K_m₀, K_m₁, K_m₂, α, FILT₀]
    """
    beta1, Km0, Km1, Km2, alpha, FILT0 = params
    n = len(rw_ntu)

    Km_t = Km0 + Km1 * day_sin + Km2 * day_cos   # (n,)
    eta  = alum / (alum + Km_t + alpha * clr + EPS)  # Michaelis-Menten
    eta  = np.clip(eta, 0.0, 1.0)

    FILT_pred = np.zeros(n)
    FILT_pred[0] = FILT0

    for t in range(1, n):
        FILT_pred[t] = (beta1 * FILT_pred[t - 1]
                        + (1.0 - beta1) * rw_ntu[t] * (1.0 - eta[t]))

    # Loss
    mse = np.mean((FILT_pred - filt_obs) ** 2)
    viol_upper = np.mean(np.maximum(0.0, FILT_pred - rw_ntu))
    viol_nonneg = np.mean(np.maximum(0.0, -FILT_pred))
    viol_km = np.mean(np.maximum(0.0, 0.001 - Km_t))

    loss = mse + LAM["filter_upper"] * viol_upper \
               + LAM["nonneg"] * viol_nonneg \
               + LAM["km_pos"] * viol_km

    return FILT_pred, loss, mse


def objective(params, rw_ntu, filt_obs, alum, clr, day_sin, day_cos):
    _, loss, _ = run_segment1(params, rw_ntu, filt_obs, alum, clr, day_sin, day_cos)
    return loss


def calibrate(rw_ntu, filt_obs, alum, clr, day_sin, day_cos):
    """多起点L-BFGS-B优化"""
    n_restarts = GREYBOX_N_RESTARTS
    best_loss, best_params = float("inf"), None

    for i in range(n_restarts):
        if i == 0:
            x0 = PARAM_INIT.copy()
        else:
            x0 = np.array([
                np.random.uniform(*GREYBOX_PARAM_BOUNDS["beta1"]),
                np.random.uniform(*GREYBOX_PARAM_BOUNDS["Km0"]),
                np.random.uniform(*GREYBOX_PARAM_BOUNDS["Km1"]),
                np.random.uniform(*GREYBOX_PARAM_BOUNDS["Km2"]),
                np.random.uniform(*GREYBOX_PARAM_BOUNDS["alpha"]),
                np.random.uniform(*GREYBOX_PARAM_BOUNDS["FILT0"]),
            ])

        res = minimize(objective, x0,
                       args=(rw_ntu, filt_obs, alum, clr, day_sin, day_cos),
                       method="L-BFGS-B", bounds=PARAM_BOUNDS,
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})

        if res.fun < best_loss:
            best_loss, best_params = res.fun, res.x
            print(f"  restart {i+1}: loss={res.fun:.6f} ★")

    return best_params, best_loss


def evaluate_folds(rw_ntu, filt_obs, alum, clr, day_sin, day_cos):
    """TimeSeriesSplit交叉验证"""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []
    n = len(rw_ntu)

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(np.arange(n).reshape(-1, 1))):
        tr_rw = rw_ntu[tr_idx]; tr_filt = filt_obs[tr_idx]
        tr_alum = alum[tr_idx]; tr_clr = clr[tr_idx]
        tr_ds = day_sin[tr_idx]; tr_dc = day_cos[tr_idx]

        vl_rw = rw_ntu[vl_idx]; vl_filt = filt_obs[vl_idx]
        vl_alum = alum[vl_idx]; vl_clr = clr[vl_idx]
        vl_ds = day_sin[vl_idx]; vl_dc = day_cos[vl_idx]

        params, _ = calibrate(tr_rw, tr_filt, tr_alum, tr_clr, tr_ds, tr_dc)

        # 在验证集上递推
        beta1, Km0, Km1, Km2, alpha, FILT0 = params
        n_vl = len(vl_rw)
        Km_t_vl = Km0 + Km1 * vl_ds + Km2 * vl_dc
        eta_vl = vl_alum / (vl_alum + Km_t_vl + alpha * vl_clr + EPS)
        eta_vl = np.clip(eta_vl, 0.0, 1.0)

        pred = np.zeros(n_vl)
        pred[0] = vl_filt[0]  # 用验证集第一个真实值初始化，更公平
        for t in range(1, n_vl):
            pred[t] = (beta1 * pred[t - 1]
                       + (1.0 - beta1) * vl_rw[t] * (1.0 - eta_vl[t]))

        rmse = np.sqrt(mean_squared_error(vl_filt, pred))
        r2 = r2_score(vl_filt, pred)
        mae = np.mean(np.abs(vl_filt - pred))
        viol = np.mean(pred > vl_rw)

        fold_metrics.append({"fold": fold, "rmse": rmse, "r2": r2,
                             "mae": mae, "violation_rate": viol,
                             "params": {n: float(v) for n, v in zip(PARAM_NAMES, params)}})
        print(f"  Fold{fold}: RMSE={rmse:.4f} R2={r2:.4f} MAE={mae:.4f} viol={viol:.4f}")

    return fold_metrics


def main():
    print("=" * 60)
    print("  step1.0 — Segment1 Greybox: Coag+Sedi+Filt -> FILT.NTU")
    print("=" * 60)

    rw_ntu, filt_obs, alum, clr, day_sin, day_cos = load_data()
    n = len(rw_ntu)
    print(f"  Valid samples: {n}")

    # 全量数据校准
    print("\n[Calibrate] Multi-start L-BFGS-B...")
    params_full, loss_full = calibrate(rw_ntu, filt_obs, alum, clr, day_sin, day_cos)
    FILT_pred, _, mse_full = run_segment1(params_full, rw_ntu, filt_obs, alum, clr, day_sin, day_cos)

    rmse_full = np.sqrt(mean_squared_error(filt_obs, FILT_pred))
    r2_full = r2_score(filt_obs, FILT_pred)

    param_dict = {n: float(v) for n, v in zip(PARAM_NAMES, params_full)}
    print(f"\n  Full calibration:")
    for k, v in param_dict.items():
        print(f"    {k} = {v:.6f}")
    print(f"    RMSE = {rmse_full:.4f}  R2 = {r2_full:.4f}")

    # 5折CV
    print(f"\n[CV] {N_SPLITS}-fold TimeSeriesSplit...")
    fold_metrics = evaluate_folds(rw_ntu, filt_obs, alum, clr, day_sin, day_cos)

    avg_rmse = np.mean([f["rmse"] for f in fold_metrics])
    avg_r2   = np.mean([f["r2"] for f in fold_metrics])
    print(f"\n  5-fold mean: RMSE={avg_rmse:.4f}  R2={avg_r2:.4f}")

    # 保存
    with open(os.path.join(OUTPUT_DIR, "segment1_params.json"), "w", encoding="utf-8") as f:
        json.dump({
            "params_full": param_dict,
            "rmse_full": rmse_full,
            "r2_full": r2_full,
            "fold_metrics": fold_metrics,
            "avg_rmse": avg_rmse,
            "avg_r2": avg_r2,
        }, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{"fold": f["fold"], "rmse": f["rmse"], "r2": f["r2"],
                    "mae": f["mae"], "violation_rate": f["violation_rate"]}
                   for f in fold_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "segment1_metrics.csv"), index=False, encoding="utf-8-sig")

    np.save(os.path.join(OUTPUT_DIR, "segment1_filt_pred.npy"), FILT_pred)

    print(f"\n  [DONE] segment1_params.json, segment1_metrics.csv saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
