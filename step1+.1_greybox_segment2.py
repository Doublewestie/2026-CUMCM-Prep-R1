"""
step1.1_greybox_segment2.py — Segment 2: CSTR ClearWell -> NTU
================================================================
Greybox recursion:
  NTU(t) = beta2(t) * NTU(t-1) + (1 - beta2(t)) * FILT_NTU(t)

  beta2(t) = exp(-DELTA_T / theta(t))
  theta(t) = A * CW_WELL_LEVEL(t-1) / (TW_FLOW(t-1) + EPS)

Input:  clean_data.csv
Output: segment2_params.json, segment2_metrics.csv, q1_greybox_ntu_pred.npy
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    required = ["FILT_NTU", "NTU", "CW_WELL_LEVEL", "TW_FLOW"]
    df = df.dropna(subset=required)
    n = len(df)

    filt      = df["FILT_NTU"].values.astype(np.float64)
    ntu_obs   = df["NTU"].values.astype(np.float64)
    cw_level  = df["CW_WELL_LEVEL"].values.astype(np.float64)
    tw_flow   = df["TW_FLOW"].values.astype(np.float64)

    return filt, ntu_obs, cw_level, tw_flow


def theta_t(A, cw_level, tw_flow):
    """CSTR residence time: theta = V/Q = A*h / Q"""
    return A * cw_level / (tw_flow + EPS)


def beta2_t(A, cw_level, tw_flow):
    """CSTR decay factor: exp(-dt/theta)"""
    theta = theta_t(A, cw_level, tw_flow)
    return np.exp(-DELTA_T / (theta + EPS))


def run_segment2(A, filt_obs, ntu_obs, cw_level, tw_flow, NTU0):
    """
    Forward recursion for Segment 2.

    Uses shifted-1 CW_WELL/TW_FLOW to avoid data leakage.
    beta2(t) = exp(-DELTA_T / theta(t-1))
    """
    n = len(ntu_obs)
    NTU_pred = np.zeros(n)
    NTU_pred[0] = NTU0

    # shift-1 for leak-free: use previous timestep's CSTR state
    beta2_vals = np.zeros(n)
    for t in range(1, n):
        beta2_vals[t] = beta2_t(A, cw_level[t - 1], tw_flow[t - 1])
        NTU_pred[t] = (beta2_vals[t] * NTU_pred[t - 1]
                       + (1.0 - beta2_vals[t]) * filt_obs[t])

    # Loss
    mse = np.mean((NTU_pred - ntu_obs) ** 2)
    viol_upper = np.mean(np.maximum(0.0, NTU_pred - filt_obs))
    viol_nonneg = np.mean(np.maximum(0.0, -NTU_pred))

    LAM = GREYBOX_LAMBDA
    loss = mse + LAM["cstr_upper"] * viol_upper + LAM["nonneg"] * viol_nonneg

    return NTU_pred, loss, mse, beta2_vals


def objective(params, filt_obs, ntu_obs, cw_level, tw_flow):
    A, NTU0 = params
    _, loss, _, _ = run_segment2(A, filt_obs, ntu_obs, cw_level, tw_flow, NTU0)
    return loss


def calibrate(filt_obs, ntu_obs, cw_level, tw_flow):
    """Multi-start L-BFGS-B"""
    bounds = [(1.0, 1000.0), (0.01, 5.0)]  # A, NTU0
    n_restarts = GREYBOX_N_RESTARTS
    best_loss, best_params = float("inf"), None

    for i in range(n_restarts):
        if i == 0:
            x0 = np.array([GREYBOX_PARAM_INIT["A"], GREYBOX_PARAM_INIT["NTU0"]])
        else:
            x0 = np.array([np.random.uniform(10, 500),
                           np.random.uniform(0.05, 2.0)])

        res = minimize(objective, x0,
                       args=(filt_obs, ntu_obs, cw_level, tw_flow),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})

        if res.fun < best_loss:
            best_loss, best_params = res.fun, res.x
            print(f"  restart {i+1}: loss={res.fun:.6f} *")

    return best_params, best_loss


def evaluate_folds(filt_obs, ntu_obs, cw_level, tw_flow):
    """TimeSeriesSplit cross validation"""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []
    n = len(ntu_obs)

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(np.arange(n).reshape(-1, 1))):
        tr_filt = filt_obs[tr_idx]; tr_ntu = ntu_obs[tr_idx]
        tr_cw = cw_level[tr_idx]; tr_tw = tw_flow[tr_idx]

        vl_filt = filt_obs[vl_idx]; vl_ntu = ntu_obs[vl_idx]
        vl_cw = cw_level[vl_idx]; vl_tw = tw_flow[vl_idx]

        params, _ = calibrate(tr_filt, tr_ntu, tr_cw, tr_tw)
        A_best, NTU0_best = params

        n_vl = len(vl_ntu)
        pred = np.zeros(n_vl)
        # use first observed value of validation set as initial (fair)
        pred[0] = vl_ntu[0]
        for t in range(1, n_vl):
            b2 = beta2_t(A_best, vl_cw[t - 1], vl_tw[t - 1])
            pred[t] = b2 * pred[t - 1] + (1.0 - b2) * vl_filt[t]

        rmse = np.sqrt(mean_squared_error(vl_ntu, pred))
        r2 = r2_score(vl_ntu, pred)
        mae = np.mean(np.abs(vl_ntu - pred))
        viol = np.mean(pred > vl_filt)

        fold_metrics.append({
            "fold": fold, "rmse": rmse, "r2": r2, "mae": mae,
            "violation_rate": viol,
            "A": float(A_best), "NTU0": float(NTU0_best),
        })
        print(f"  Fold{fold}: RMSE={rmse:.4f} R2={r2:.4f} MAE={mae:.4f} "
              f"A={A_best:.1f} viol={viol:.4f}")

    return fold_metrics


def main():
    print("=" * 60)
    print("  step1.1 — Segment2 Greybox: CSTR -> NTU")
    print("=" * 60)

    filt, ntu_obs, cw_level, tw_flow = load_data()
    n = len(ntu_obs)
    print(f"  Valid samples: {n}")

    # Full calibration
    print("\n[Calibrate] Multi-start L-BFGS-B...")
    params_full, loss_full = calibrate(filt, ntu_obs, cw_level, tw_flow)
    A_full, NTU0_full = params_full

    NTU_pred, _, mse_full, beta2_vals = run_segment2(
        A_full, filt, ntu_obs, cw_level, tw_flow, NTU0_full)

    rmse_full = np.sqrt(mean_squared_error(ntu_obs, NTU_pred))
    r2_full = r2_score(ntu_obs, NTU_pred)

    print(f"\n  Full calibration:")
    print(f"    A     = {A_full:.2f} m2")
    print(f"    NTU0  = {NTU0_full:.4f} NTU")
    print(f"    RMSE  = {rmse_full:.4f}")
    print(f"    R2    = {r2_full:.4f}")

    # Show beta2 range
    b2_valid = beta2_vals[1:]
    if len(b2_valid) > 0:
        print(f"    beta2 range: [{np.min(b2_valid):.4f}, {np.max(b2_valid):.4f}]")
        print(f"    beta2 mean:  {np.mean(b2_valid):.4f}")

    # 5-fold CV
    print(f"\n[CV] {N_SPLITS}-fold TimeSeriesSplit...")
    fold_metrics = evaluate_folds(filt, ntu_obs, cw_level, tw_flow)

    avg_rmse = np.mean([f["rmse"] for f in fold_metrics])
    avg_r2   = np.mean([f["r2"] for f in fold_metrics])
    print(f"\n  5-fold mean: RMSE={avg_rmse:.4f}  R2={avg_r2:.4f}")

    # Save
    import json as _json
    with open(os.path.join(OUTPUT_DIR, "segment2_params.json"), "w", encoding="utf-8") as f:
        _json.dump({
            "params_full": {"A": float(A_full), "NTU0": float(NTU0_full)},
            "rmse_full": rmse_full,
            "r2_full": r2_full,
            "fold_metrics": fold_metrics,
            "avg_rmse": avg_rmse,
            "avg_r2": avg_r2,
        }, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{"fold": f["fold"], "rmse": f["rmse"], "r2": f["r2"],
                    "mae": f["mae"], "violation_rate": f["violation_rate"]}
                   for f in fold_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "segment2_metrics.csv"), index=False, encoding="utf-8-sig")

    np.save(os.path.join(OUTPUT_DIR, "segment2_ntu_pred.npy"), NTU_pred)
    np.save(os.path.join(OUTPUT_DIR, "segment2_beta2.npy"), beta2_vals)

    print(f"\n  [DONE] segment2_params.json, segment2_metrics.csv saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
