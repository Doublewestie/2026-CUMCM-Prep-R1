"""
step1.2_greybox_joint.py — Two-Segment Greybox: Multi-Variant Comparison
==========================================================================
Segment 1 diagnostic: beta1 -> 1, external inputs have negligible short-term
  effect on FILT_NTU.

Segment 2 variants:
  V1 (AR1_physical): NTU(t) = beta2(t)*NTU(t-1) + (1-beta2(t))*FILT(t)
      beta2(t) = exp(-2h/theta), theta = A*CW/TW
      params: A, NTU0  (2 params)

  V2 (AR2_free):      NTU(t) = w1*NTU(t-1) + w2*NTU(t-2) + (1-w1-w2)*FILT(t)
      params: w1, w2, NTU0  (3 params, w1+w2<1)

  V3 (AR2_splitA):     V2 with separate A for normal (NTU_prev<0.5) vs high turbidity
      params: w1, w2, A_lo, A_hi, NTU0  (5 params)

Input:  clean_data.csv
Output: q1_greybox_params.json, q1_greybox_metrics.csv, comparison table
"""

import os, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LAM = GREYBOX_LAMBDA
THETA_COMFORT = 0.15


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    required = ["RW_NTU", "FILT_NTU", "NTU", "ALUM", "CLR", "CW_WELL_LEVEL", "TW_FLOW"]
    df = df.dropna(subset=required)
    rw_ntu   = df["RW_NTU"].values.astype(np.float64)
    filt_obs = df["FILT_NTU"].values.astype(np.float64)
    ntu_obs  = df["NTU"].values.astype(np.float64)
    alum     = df["ALUM"].values.astype(np.float64)
    clr      = df["CLR"].values.astype(np.float64)
    cw_level = df["CW_WELL_LEVEL"].values.astype(np.float64)
    tw_flow  = df["TW_FLOW"].values.astype(np.float64)
    date     = pd.to_datetime(df["DATE"])
    doy      = date.dt.dayofyear.values
    day_sin  = np.sin(2 * np.pi * doy / 365)
    day_cos  = np.cos(2 * np.pi * doy / 365)
    return rw_ntu, filt_obs, ntu_obs, alum, clr, cw_level, tw_flow, day_sin, day_cos


def beta2_from_A(A, cw, tw):
    theta = A * cw / (tw + EPS)
    return np.exp(-DELTA_T / (theta + EPS))


# ==================== V1: AR(1) Physical ====================

def run_ar1_physical(A, filt, ntu_obs, cw, tw, NTU0):
    n = len(ntu_obs)
    pred = np.zeros(n)
    pred[0] = NTU0
    b2 = np.zeros(n)
    for t in range(1, n):
        b2[t] = beta2_from_A(A, cw[t - 1], tw[t - 1])
        pred[t] = b2[t] * pred[t - 1] + (1 - b2[t]) * filt[t]
    return pred, b2


def loss_ar1(params, filt, ntu_obs, cw, tw):
    A, NTU0 = params
    pred, _ = run_ar1_physical(A, filt, ntu_obs, cw, tw, NTU0)
    mse  = np.mean((pred - ntu_obs) ** 2)
    viol = np.mean(np.maximum(0, pred - filt))
    nn   = np.mean(np.maximum(0, -pred))
    return mse + LAM["cstr_upper"] * viol + LAM["nonneg"] * nn


def calibrate_ar1(filt, ntu, cw, tw):
    bnds = [(1, 1000), (0.01, 5)]
    best_loss, best_x = float("inf"), None
    for i in range(GREYBOX_N_RESTARTS):
        x0 = ([GREYBOX_PARAM_INIT["A"], GREYBOX_PARAM_INIT["NTU0"]] if i == 0
              else [np.random.uniform(10, 500), np.random.uniform(0.05, 2)])
        res = minimize(loss_ar1, x0, args=(filt, ntu, cw, tw),
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})
        if res.fun < best_loss:
            best_loss, best_x = res.fun, res.x
    return best_x


# ==================== V2: AR(2) Free ====================

def run_ar2_free(w1, w2, filt, ntu_obs, NTU0):
    n = len(ntu_obs)
    pred = np.zeros(n)
    pred[0] = NTU0
    if n >= 2:
        pred[1] = w1 * pred[0] + (1 - w1) * filt[1]
    w_filt = 1 - w1 - w2
    for t in range(2, n):
        pred[t] = w1 * pred[t - 1] + w2 * pred[t - 2] + w_filt * filt[t]
    return pred


def loss_ar2(params, filt, ntu_obs, cw, tw):
    w1, w2, NTU0 = params
    if w1 + w2 >= 1:
        return 1e10
    pred = run_ar2_free(w1, w2, filt, ntu_obs, NTU0)
    mse  = np.mean((pred - ntu_obs) ** 2)
    viol = np.mean(np.maximum(0, pred - filt))
    nn   = np.mean(np.maximum(0, -pred))
    return mse + LAM["cstr_upper"] * viol + LAM["nonneg"] * nn


def calibrate_ar2(filt, ntu, cw, tw):
    bnds = [(0.01, 0.95), (0.01, 0.95), (0.01, 5)]
    best_loss, best_x = float("inf"), None
    for i in range(GREYBOX_N_RESTARTS):
        if i == 0:
            x0 = [0.4, 0.2, GREYBOX_PARAM_INIT["NTU0"]]
        else:
            x0 = [np.random.uniform(0.1, 0.8), np.random.uniform(0.05, 0.3),
                  np.random.uniform(0.05, 2)]
        res = minimize(loss_ar2, x0, args=(filt, ntu, cw, tw),
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})
        if res.fun < best_loss:
            best_loss, best_x = res.fun, res.x
    return best_x


# ==================== V3: AR(2) Split A ====================

def run_ar2_split(w1, w2, A_lo, A_hi, filt, ntu_obs, cw, tw, NTU0):
    n = len(ntu_obs)
    pred = np.zeros(n)
    pred[0] = NTU0
    if n >= 2:
        b2_1 = beta2_from_A(A_lo, cw[0], tw[0])
        pred[1] = b2_1 * pred[0] + (1 - b2_1) * filt[1]

    w_filt = 1 - w1 - w2
    for t in range(2, n):
        A_use = A_hi if pred[t - 1] > 0.5 else A_lo
        b2 = beta2_from_A(A_use, cw[t - 1], tw[t - 1])
        pred[t] = w1 * pred[t - 1] + w2 * pred[t - 2] + w_filt * filt[t]
    return pred


def loss_ar2_split(params, filt, ntu_obs, cw, tw):
    w1, w2, A_lo, A_hi, NTU0 = params
    if w1 + w2 >= 1 or A_lo <= 0 or A_hi <= 0:
        return 1e10
    pred = run_ar2_split(w1, w2, A_lo, A_hi, filt, ntu_obs, cw, tw, NTU0)
    mse  = np.mean((pred - ntu_obs) ** 2)
    viol = np.mean(np.maximum(0, pred - filt))
    nn   = np.mean(np.maximum(0, -pred))
    return mse + LAM["cstr_upper"] * viol + LAM["nonneg"] * nn


def calibrate_ar2_split(filt, ntu, cw, tw):
    bnds = [(0.01, 0.95), (0.01, 0.95), (1, 500), (1, 500), (0.01, 5)]
    best_loss, best_x = float("inf"), None
    for i in range(GREYBOX_N_RESTARTS):
        if i == 0:
            x0 = [0.35, 0.15, GREYBOX_PARAM_INIT["A"], GREYBOX_PARAM_INIT["A"] * 0.5,
                  GREYBOX_PARAM_INIT["NTU0"]]
        else:
            x0 = [np.random.uniform(0.1, 0.7), np.random.uniform(0.05, 0.25),
                  np.random.uniform(10, 300), np.random.uniform(1, 100),
                  np.random.uniform(0.05, 2)]
        res = minimize(loss_ar2_split, x0, args=(filt, ntu, cw, tw),
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})
        if res.fun < best_loss:
            best_loss, best_x = res.fun, res.x
    return best_x


# ==================== V4: Dual-Mode CSTR ====================

def run_ar1_dual_mode(A, filt, ntu_obs, cw, tw, NTU0, theta=THETA_COMFORT):
    """
    Dual-mode CSTR:
      Comfort (FILT < theta): NTU_hat = rolling mean of NTU[last 12]
      Stress  (FILT >= theta): NTU_hat = beta2*NTU_hat(t-1) + (1-beta2)*FILT(t)
    """
    n = len(ntu_obs)
    pred = np.zeros(n)
    pred[0] = ntu_obs[0] if ntu_obs[0] < 5 else NTU0
    comfort = np.zeros(n, dtype=bool)
    stress = np.zeros(n, dtype=bool)
    win = 12

    for t in range(1, n):
        if filt[t] < theta:
            comfort[t] = True
            lo = max(0, t - win)
            pred[t] = np.mean(ntu_obs[lo:t])
        else:
            stress[t] = True
            b2 = beta2_from_A(A, cw[t - 1], tw[t - 1])
            pred[t] = b2 * pred[t - 1] + (1 - b2) * filt[t]

    return pred, comfort, stress


def loss_ar1_dual(params, filt, ntu_obs, cw, tw):
    A, NTU0 = params
    pred, comfort, stress = run_ar1_dual_mode(A, filt, ntu_obs, cw, tw, NTU0)
    if stress.sum() < 20:
        return 1e10
    mse  = np.mean((pred[stress] - ntu_obs[stress]) ** 2)
    viol = np.mean(np.maximum(0, pred[stress] - filt[stress]))
    nn   = np.mean(np.maximum(0, -pred[stress]))
    return mse + LAM["cstr_upper"] * viol + LAM["nonneg"] * nn


def calibrate_ar1_dual(filt, ntu, cw, tw):
    bnds = [(1, 1000), (0.01, 5)]
    best_loss, best_x = float("inf"), None
    for i in range(GREYBOX_N_RESTARTS):
        x0 = ([GREYBOX_PARAM_INIT["A"], GREYBOX_PARAM_INIT["NTU0"]] if i == 0
              else [np.random.uniform(10, 500), np.random.uniform(0.05, 2)])
        res = minimize(loss_ar1_dual, x0, args=(filt, ntu, cw, tw),
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})
        if res.fun < best_loss:
            best_loss, best_x = res.fun, res.x
    return best_x


# ==================== Common evaluation ====================

def metrics(pred, true, filt):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2   = r2_score(true, pred)
    mae  = mean_absolute_error(true, pred)
    viol = np.mean(pred > filt)
    return rmse, r2, mae, viol


def segment1_diagnostic(rw_ntu, filt_obs, alum, clr, day_sin, day_cos):
    def s1_loss(params):
        beta1, Km0, Km1, Km2, alpha, f0 = params
        n = len(rw_ntu)
        Km = Km0 + Km1 * day_sin + Km2 * day_cos
        eta = alum / (alum + Km + alpha * clr + EPS)
        eta = np.clip(eta, 0, 1)
        pred = np.zeros(n); pred[0] = f0
        for t in range(1, n):
            pred[t] = beta1 * pred[t - 1] + (1 - beta1) * rw_ntu[t] * (1 - eta[t])
        return np.mean((pred - filt_obs) ** 2)

    bnds = [(0.01, 0.999), (0.001, 0.5), (-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1), (0.001, 10.0)]
    x0 = [0.85, 0.05, 0.01, 0.01, 0.005, 0.2]
    res = minimize(s1_loss, x0, method="L-BFGS-B", bounds=bnds,
                   options={"maxiter": 2000, "ftol": 1e-12})
    return {k: float(v) for k, v in zip(["beta1","Km0","Km1","Km2","alpha","FILT0"], res.x)}


# ==================== Main ====================

def main():
    print("=" * 60)
    print("  step1.2 — Greybox Variant Comparison")
    print("=" * 60)

    rw_ntu, filt, ntu, alum, clr, cw, tw, ds, dc = load_data()
    n = len(ntu)
    print(f"  Valid samples: {n}")

    # Segment1 diagnostic
    s1 = segment1_diagnostic(rw_ntu, filt, alum, clr, ds, dc)
    print(f"\n  [Seg1] beta1={s1['beta1']:.4f} Km0={s1['Km0']:.4f} alpha={s1['alpha']:.4f}")

    # ==================== Full calibration: all 3 variants ====================
    print(f"\n[Full calibration]")
    variants = []

    # V1
    p1 = calibrate_ar1(filt, ntu, cw, tw)
    pred1, b2 = run_ar1_physical(p1[0], filt, ntu, cw, tw, p1[1])
    rmse1, r21, mae1, v1 = metrics(pred1, ntu, filt)
    b2m = np.mean(b2[b2 > 0])
    variants.append({"name": "V1_AR1_phys", "params": {"A": float(p1[0]), "NTU0": float(p1[1])},
                     "rmse": rmse1, "r2": r21, "mae": mae1, "viol": v1,
                     "beta2_mean": float(b2m), "n_params": 2})

    print(f"  V1_AR1_phys     A={p1[0]:.1f}  RMSE={rmse1:.4f}  R2={r21:.4f}  "
          f"beta2_mean={b2m:.4f}  n_param=2")

    # V2
    p2 = calibrate_ar2(filt, ntu, cw, tw)
    pred2 = run_ar2_free(p2[0], p2[1], filt, ntu, p2[2])
    rmse2, r22, mae2, v2 = metrics(pred2, ntu, filt)
    variants.append({"name": "V2_AR2_free", "params": {"w1": float(p2[0]), "w2": float(p2[1]),
                     "NTU0": float(p2[2]), "w_filt": float(1-p2[0]-p2[1])},
                     "rmse": rmse2, "r2": r22, "mae": mae2, "viol": v2, "n_params": 3})

    print(f"  V2_AR2_free      w1={p2[0]:.4f} w2={p2[1]:.4f} w_filt={1-p2[0]-p2[1]:.4f}  "
          f"RMSE={rmse2:.4f}  R2={r22:.4f}  n_param=3")

    # V3
    p3 = calibrate_ar2_split(filt, ntu, cw, tw)
    pred3 = run_ar2_split(p3[0], p3[1], p3[2], p3[3], filt, ntu, cw, tw, p3[4])
    rmse3, r23, mae3, v3 = metrics(pred3, ntu, filt)
    variants.append({"name": "V3_AR2_splitA", "params": {"w1": float(p3[0]), "w2": float(p3[1]),
                     "A_lo": float(p3[2]), "A_hi": float(p3[3]), "NTU0": float(p3[4]),
                     "w_filt": float(1-p3[0]-p3[1])},
                     "rmse": rmse3, "r2": r23, "mae": mae3, "viol": v3, "n_params": 5})

    print(f"  V3_AR2_splitA    w1={p3[0]:.4f} w2={p3[1]:.4f}  "
          f"A_lo={p3[2]:.1f} A_hi={p3[3]:.1f}  "
          f"RMSE={rmse3:.4f}  R2={r23:.4f}  n_param=5")

    # V4: Dual-mode
    p_dual = calibrate_ar1_dual(filt, ntu, cw, tw)
    pred_dual, cmask, smask = run_ar1_dual_mode(p_dual[0], filt, ntu, cw, tw, p_dual[1])
    rmse_d, r2_d, mae_d, v_d = metrics(pred_dual, ntu, filt)
    if cmask.sum() > 0:
        rmse_c, r2_c, _, _ = metrics(pred_dual[cmask], ntu[cmask], filt[cmask])
    else:
        rmse_c, r2_c = float("nan"), float("nan")
    if smask.sum() > 0:
        rmse_s, r2_s, _, _ = metrics(pred_dual[smask], ntu[smask], filt[smask])
    else:
        rmse_s, r2_s = float("nan"), float("nan")
    variants.append({
        "name": "V4_DualMode", "params": {"A": float(p_dual[0]), "NTU0": float(p_dual[1])},
        "rmse": rmse_d, "r2": r2_d, "mae": mae_d, "viol": v_d,
        "comfort_r2": r2_c, "stress_r2": r2_s,
        "comfort_n": int(cmask.sum()), "stress_n": int(smask.sum()), "n_params": 2,
    })
    print(f"  V4_DualMode      A={p_dual[0]:.1f}  RMSE={rmse_d:.4f}  R2={r2_d:.4f}  "
          f"Comfort_R2={r2_c:.4f}(n={int(cmask.sum())})  "
          f"Stress_R2={r2_s:.4f}(n={int(smask.sum())})")

    # ==================== 5-fold CV for all ====================
    print(f"\n[5-fold CV]")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    variant_list = [
        ("V1_AR1_phys", calibrate_ar1, run_ar1_physical, 2),
        ("V2_AR2_free", calibrate_ar2, run_ar2_free, 3),
        ("V3_AR2_splitA", calibrate_ar2_split, run_ar2_split, 5),
        ("V4_DualMode", calibrate_ar1_dual, run_ar1_dual_mode, 2),
    ]

    for vname, cal_fn, run_fn, n_params in variant_list:
        fold_rmses, fold_r2s, fold_maes = [], [], []
        for fold, (tr, vl) in enumerate(tscv.split(np.arange(n).reshape(-1, 1))):
            tr_f, tr_n, tr_c, tr_t = filt[tr], ntu[tr], cw[tr], tw[tr]
            vl_f, vl_n, vl_c, vl_t = filt[vl], ntu[vl], cw[vl], tw[vl]

            if vname == "V1_AR1_phys":
                p = cal_fn(tr_f, tr_n, tr_c, tr_t)
                pred, _ = run_fn(p[0], vl_f, vl_n, vl_c, vl_t, vl_n[0])
            elif vname == "V2_AR2_free":
                p = cal_fn(tr_f, tr_n, tr_c, tr_t)
                pred = run_fn(p[0], p[1], vl_f, vl_n, vl_n[0])
            elif vname == "V3_AR2_splitA":
                p = cal_fn(tr_f, tr_n, tr_c, tr_t)
                pred = run_fn(p[0], p[1], p[2], p[3], vl_f, vl_n, vl_c, vl_t, vl_n[0])
            else:  # V4_DualMode
                p = cal_fn(tr_f, tr_n, tr_c, tr_t)
                pred, cm, sm = run_fn(p[0], vl_f, vl_n, vl_c, vl_t, vl_n[0])

            rmse, r2, mae, _ = metrics(pred, vl_n, vl_f)
            fold_rmses.append(rmse); fold_r2s.append(r2); fold_maes.append(mae)

        cv_rmse = np.mean(fold_rmses)
        cv_r2   = np.mean(fold_r2s)
        cv_mae  = np.mean(fold_maes)
        r2_std  = np.std(fold_r2s)

        # Find the variant in the list and update
        for v in variants:
            if v["name"] == vname:
                v["cv_rmse"] = cv_rmse
                v["cv_r2"] = cv_r2
                v["cv_r2_std"] = r2_std
                v["cv_folds"] = [f"R2={r:.3f}" for r in fold_r2s]

        print(f"  {vname:<18s} CV: RMSE={cv_rmse:.4f}  R2={cv_r2:.4f}+/-{r2_std:.3f}  "
              f"folds={[f'{r:.3f}' for r in fold_r2s]}")

    # ==================== Comparison table ====================
    print(f"\n{'='*70}")
    print(f"  Model Comparison")
    print(f"{'='*70}")
    print(f"  {'Model':<18s} {'n_param':>7s} {'Full R2':>8s} {'CV R2':>8s} {'CV R2_std':>10s} {'Full RMSE':>10s}")
    print(f"  {'-'*60}")

    best_cv_r2 = max(v.get("cv_r2", -999) for v in variants)
    for v in variants:
        star = " *" if v.get("cv_r2", -999) == best_cv_r2 else ""
        print(f"  {v['name']:<18s} {v['n_params']:>7d} {v['r2']:>8.4f} "
              f"{v.get('cv_r2', 0):>8.4f} {v.get('cv_r2_std', 0):>10.4f} {v['rmse']:>10.4f}{star}")

    print(f"\n  * best CV R2")
    print(f"{'='*70}")

    # ==================== Stratified for best variant ====================
    best_name = max(variants, key=lambda v: v.get("cv_r2", -999))["name"]
    print(f"\n[Best variant: {best_name}] Stratified evaluation:")

    if best_name == "V1_AR1_phys":
        best_pred, _ = run_ar1_physical(p1[0], filt, ntu, cw, tw, p1[1])
    elif best_name == "V2_AR2_free":
        best_pred = run_ar2_free(p2[0], p2[1], filt, ntu, p2[2])
    else:
        best_pred = run_ar2_split(p3[0], p3[1], p3[2], p3[3], filt, ntu, cw, tw, p3[4])

    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 100)]:
        mask = (ntu >= lo) & (ntu < hi)
        if mask.sum() < 5:
            continue
        s_rmse, s_r2, _, _ = metrics(best_pred[mask], ntu[mask], filt[mask])
        print(f"  NTU[{lo}-{hi}): n={mask.sum():4d}  RMSE={s_rmse:.4f}  R2={s_r2:.4f}")

    # ==================== Save ====================
    grey_params = {
        "segment1_diagnostic": s1,
        "best_model": best_name,
        "variants": variants,
    }
    with open(os.path.join(OUTPUT_DIR, "q1_greybox_params.json"), "w", encoding="utf-8") as f:
        json.dump(grey_params, f, indent=2, ensure_ascii=False, default=str)

    df_comp = pd.DataFrame([{"Model": v["name"], "n_params": v["n_params"],
                              "Full_RMSE": v["rmse"], "Full_R2": v["r2"],
                              "CV_RMSE": v.get("cv_rmse", 0), "CV_R2": v.get("cv_r2", 0),
                              "CV_R2_std": v.get("cv_r2_std", 0)}
                             for v in variants])
    df_comp.to_csv(os.path.join(OUTPUT_DIR, "q1_greybox_metrics.csv"), index=False, encoding="utf-8-sig")

    np.save(os.path.join(OUTPUT_DIR, "q1_greybox_ntu_pred.npy"), best_pred)

    # ==================== Figure ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # All three: predicted vs actual
    for ax, pred_i, name_i in zip(axes, [pred1, pred2, pred3],
                                   ["V1: AR(1) Physical", "V2: AR(2) Free", "V3: AR(2) Split-A"]):
        ax.scatter(ntu, pred_i, alpha=0.3, s=3, c="steelblue")
        lim = max(ntu.max(), pred_i.max())
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        r2_i = r2_score(ntu, pred_i)
        ax.set_xlabel("Observed NTU"); ax.set_ylabel("Predicted NTU")
        ax.set_title(f"{name_i}\nR2={r2_i:.3f}")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q1_greybox_variant_comparison.png"), dpi=300)
    plt.close()

    print(f"\n  [DONE] figures/q1_greybox_variant_comparison.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
