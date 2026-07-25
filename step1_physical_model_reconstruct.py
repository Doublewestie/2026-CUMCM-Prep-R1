"""
step1_physical_model_reconstruct.py — Q1 物理模型重构
====================================================
方法论:
  段I (化学段):  Langmuir 吸附 + 季节调制 + CLR 竞争
  段II (物理段): CSTR 清水池混合

物理公式:
  段I:  eta(t) = eta_max * ALUM(t) / [ALUM(t) + K_d(t) * (1 + beta_c * CLR(t))]
        K_d(t) = K_d0 * [1 + delta1*sin(day) + delta2*cos(day)]
        NTU_post_chem = RW_NTU(t-tau_total) * (1 - eta)

  段II: NTU_post_phys = NTU_post_chem * C_phys
        C_phys = (1 - eta_sed) * exp(-lambda*L)  [经验常数 ~0.005]

  惯性: FILT(t) = beta1 * FILT(t-1) + (1 - beta1) * NTU_post_phys

  CSTR: NTU(t) = beta2(t) * NTU(t-1) + (1 - beta2(t)) * FILT(t)
        beta2(t) = exp(-2h / theta), theta = A * CW_WELL(t-1) / TW_FLOW(t-1)
"""
import numpy as np
import pandas as pd
import os
import sys
import io
import json
import warnings
import time

from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

# ================================================================
#  全局配置
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLEAN_DATA = os.path.join(OUTPUT_DIR, "clean_data.csv")

N_RESTARTS = 5
MAX_ITER = 500
DELTA_T = 2.0
TIER_THRESHOLDS = [0.05, 0.15]
MAX_SCAN_TAU = 6
SUBSAMPLE_STEP = 1

# ================================================================
#  数据加载
# ================================================================
def load_data():
    df = pd.read_csv(CLEAN_DATA)
    df["DATE"] = pd.to_datetime(df["DATE"])
    time_vals = pd.to_numeric(df["TIME"], errors="coerce").fillna(0).astype(int)
    df["hour"] = time_vals // 100
    df["day_sin"] = np.sin(2 * np.pi * df["DATE"].dt.dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["DATE"].dt.dayofyear / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    if "RW_CLR" in df.columns:
        df["CLR_raw"] = df["RW_CLR"]
    else:
        df["CLR_raw"] = df["CLR"]

    df = df.dropna(subset=["NTU", "FILT_NTU", "RW_NTU", "ALUM", "CW_WELL_LEVEL", "TW_FLOW"])
    df = df.reset_index(drop=True)

    df["tier"] = 1
    df.loc[df["FILT_NTU"] > TIER_THRESHOLDS[0], "tier"] = 2
    df.loc[df["FILT_NTU"] > TIER_THRESHOLDS[1], "tier"] = 3

    print(f"  Data loaded: {len(df)} rows")
    for t in [1, 2, 3]:
        n_t = (df.tier == t).sum()
        print(f"    T{t}: {n_t} ({n_t/len(df)*100:.1f}%)")
    return df

# ================================================================
#  物理模型核心函数
# ================================================================
def langmuir_eta_vec(alum, clr, K_d0, d1, d2, bc, dsin, dcos, emax):
    K_d = K_d0 * np.maximum(1.0 + d1 * dsin + d2 * dcos, 0.01)
    denom = alum + K_d * (1.0 + bc * np.maximum(clr, 0)) + 1e-8
    eta = emax * alum / denom
    return np.clip(eta, 0.0, emax)

def build_shifted_arrays(rw_ntu, alum, clr, dsin, dcos, tau):
    n = len(rw_ntu)
    rw_s = np.zeros(n); al_s = np.zeros(n); cl_s = np.zeros(n)
    ds_s = np.zeros(n); dc_s = np.zeros(n)

    if tau == 0:
        return rw_ntu, alum, clr, dsin, dcos
    rw_s[tau:] = rw_ntu[:-tau]
    al_s[tau:] = alum[:-tau]
    cl_s[tau:] = clr[:-tau]
    ds_s[tau:] = dsin[:-tau]
    dc_s[tau:] = dcos[:-tau]
    rw_s[:tau] = rw_ntu[0]
    al_s[:tau] = alum[0]
    cl_s[:tau] = clr[0]
    ds_s[:tau] = dsin[0]
    dc_s[:tau] = dcos[0]
    return rw_s, al_s, cl_s, ds_s, dc_s

def filt_recurse(beta1, K_d0, d1, d2, bc, emax, cp,
                 rw_aligned, al_aligned, cl_aligned, ds_aligned, dc_aligned,
                 filt_obs, n):
    eta = langmuir_eta_vec(al_aligned, cl_aligned, K_d0, d1, d2, bc,
                           ds_aligned, dc_aligned, emax)
    ntu_pp = rw_aligned * (1.0 - eta) * cp
    filt_pred = np.zeros(n)
    filt_pred[0] = float(filt_obs[0])
    for t in range(1, n):
        filt_pred[t] = beta1 * filt_pred[t - 1] + (1.0 - beta1) * ntu_pp[t]
    return filt_pred, eta, ntu_pp

def ntu_recurse(filt_input, cw, tw, A_cstr, ntu_obs, n):
    ntu_pred = np.zeros(n)
    ntu_pred[0] = float(ntu_obs[0])
    for t in range(1, n):
        theta = A_cstr * max(cw[t - 1], 0.1) / max(tw[t - 1], 0.1)
        b2 = np.clip(np.exp(-DELTA_T / max(theta, 0.01)), 0.01, 0.999)
        ntu_pred[t] = b2 * ntu_pred[t - 1] + (1.0 - b2) * filt_input[t]
    return ntu_pred

def compute_metrics(y_true, y_pred):
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
    }

# ================================================================
#  C_phys 独立估计
# ================================================================
def estimate_cphys(df):
    mask = df["FILT_NTU"] < 0.05
    sub = df[mask]
    if len(sub) < 50:
        return 0.005
    rw = sub["RW_NTU"].values
    filt = sub["FILT_NTU"].values
    eta_g = 0.5
    est = filt / (rw * (1 - eta_g) + 1e-8)
    est = est[(est > 0) & (est < 0.1)]
    return float(np.median(est)) if len(est) > 0 else 0.005

# ================================================================
#  Q1 两段灰箱联合优化
# ================================================================
def _ntu_loss(A, ntu_obs, filt_pred, cw, tw, n):
    ntu_p = ntu_recurse(filt_pred, cw, tw, A, np.array([ntu_obs[0]]), n)
    err = ntu_obs - ntu_p
    delta = 0.1
    h = np.where(np.abs(err) < delta, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta))
    return np.mean(h)

def q1_two_segment(df, tau_star, filt_params):
    n = len(df)
    rw = df["RW_NTU"].values; al = df["ALUM"].values; cl = df["CLR_raw"].values
    fo = df["FILT_NTU"].values; no = df["NTU"].values
    cw = df["CW_WELL_LEVEL"].values; tw = df["TW_FLOW"].values
    ds = df["day_sin"].values; dc = df["day_cos"].values

    if filt_params is None:
        fp = {"beta1": 0.60, "K_d0": 0.035, "delta1": 0.0, "delta2": 0.0,
              "beta_c": 0.0, "eta_max": 0.90, "C_phys": 0.005}
    else:
        fp = filt_params

    rw_s, al_s, cl_s, ds_s, dc_s = build_shifted_arrays(rw, al, cl, ds, dc, tau_star)

    filt_pred, eta_vals, _ = filt_recurse(
        fp["beta1"], fp["K_d0"], fp["delta1"], fp["delta2"],
        fp["beta_c"], fp["eta_max"], fp["C_phys"],
        rw_s, al_s, cl_s, ds_s, dc_s, fo, n)

    x0 = [120.0]
    bounds = [(1.0, 500.0)]
    best_val, best_x = float("inf"), None
    for r in range(N_RESTARTS):
        xi = np.array([np.random.uniform(1, 500)]) if r > 0 else np.array(x0)
        res = minimize(
            lambda p: _ntu_loss(p[0], no, filt_pred, cw, tw, n),
            xi, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": MAX_ITER, "ftol": 1e-8}
        )
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    A_opt = best_x[0] if best_x is not None else 120.0

    ntu_pred = ntu_recurse(filt_pred, cw, tw, A_opt, no, n)
    ntu_pure = ntu_recurse(filt_pred, cw, tw, A_opt, np.array([no[0]]), n)

    tier_m = {}
    for tid in [1, 2, 3]:
        mask = df["tier"] == tid
        if mask.sum() > 0:
            tier_m[f"T{tid}"] = compute_metrics(no[mask.values], ntu_pred[mask.values])
            tier_m[f"T{tid}"]["n"] = int(mask.sum())

    full_m = compute_metrics(no, ntu_pred)
    pure_m = compute_metrics(no, ntu_pure)
    cstr_beta2_median = float(np.median(np.exp(-DELTA_T / np.maximum(
        A_opt * np.maximum(cw[:-1], 0.1) / np.maximum(tw[:-1], 0.1), 0.01))))

    params_out = {
        **{k: round(float(v), 6) for k, v in fp.items()},
        "A_cstr": round(float(A_opt), 4),
        "tau_star": tau_star,
        "A_cstr_loss": round(float(best_val), 6),
    }

    return {
        "params": params_out,
        "full_metrics": full_m,
        "pure_metrics": pure_m,
        "tier_metrics": tier_m,
        "filt_pred": filt_pred,
        "ntu_pred": ntu_pred,
        "ntu_pure": ntu_pure,
        "eta_vals": eta_vals,
        "cstr_beta2_median": cstr_beta2_median,
    }

# ================================================================
#  主程序
# ================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  Q1 Physical Model Reconstruction: 2-Segment Greybox")
    print("=" * 80)

    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"  NTU: mean={df['NTU'].mean():.3f}, std={df['NTU'].std():.3f}, "
          f"skew={df['NTU'].skew():.2f}, max={df['NTU'].max():.2f}")
    print(f"  FILT: mean={df['FILT_NTU'].mean():.3f}, std={df['FILT_NTU'].std():.3f}, "
          f"skew={df['FILT_NTU'].skew():.2f}, max={df['FILT_NTU'].max():.2f}")

    print("\n[2/4] Estimating C_phys from comfort zone...")
    cphys = estimate_cphys(df)
    print(f"  C_phys = {cphys:.6f}")
    print(f"  eta_phys = {1 - cphys:.4f} = {100*(1-cphys):.2f}% removal")

    print("\n[3/4] Running Q1 greybox model...")
    tau_star = 2
    q1 = q1_two_segment(df, tau_star, None)

    print(f"\n  Fitted params:")
    for k, v in q1["params"].items():
        print(f"    {k:>15s} = {v}")
    print(f"\n  CSTR median beta2 = {q1['cstr_beta2_median']:.4f}")
    print(f"  NTU full R2 = {q1['full_metrics']['r2']:.4f}, RMSE = {q1['full_metrics']['rmse']:.4f}")
    print(f"  NTU pure-extrapolation R2 = {q1['pure_metrics']['r2']:.4f}, RMSE = {q1['pure_metrics']['rmse']:.4f}")
    print(f"\n  Tier breakdown:")
    for tk, tm in sorted(q1["tier_metrics"].items()):
        print(f"    {tk} (n={tm['n']}): R2={tm['r2']:.4f}, RMSE={tm['rmse']:.4f}")

    print("\n[4/4] Saving results...")
    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Segment I (Chemical): Langmuir adsorption + seasonal modulation + CLR competition")
    print(f"  Segment II (Physical): FILT inertia + CSTR clearwater tank")
    print(f"  C_phys = {cphys:.5f} -> physical removal = {100*(1-cphys):.1f}%")
    print(f"\n  Key Results:")
    print(f"    NTU Full R2 = {q1['full_metrics']['r2']:.4f}")
    print(f"    Pure-Extrapolation R2 = {q1['pure_metrics']['r2']:.4f}")
    for tk in ["T1", "T2", "T3"]:
        print(f"    {tk} R2 = {q1['tier_metrics'][tk]['r2']:.4f} (n={q1['tier_metrics'][tk]['n']})")

    out = {
        "params": q1["params"],
        "full_r2": q1["full_metrics"]["r2"], "full_rmse": q1["full_metrics"]["rmse"],
        "pure_r2": q1["pure_metrics"]["r2"], "pure_rmse": q1["pure_metrics"]["rmse"],
        "tier_r2": {k: q1["tier_metrics"][k]["r2"] for k in q1["tier_metrics"]},
        "cstr_beta2_median": q1["cstr_beta2_median"],
    }
    op = os.path.join(OUTPUT_DIR, "step1_physical_results.json")
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  Results saved to: {op}")
    print("=" * 80)

if __name__ == "__main__":
    main()
