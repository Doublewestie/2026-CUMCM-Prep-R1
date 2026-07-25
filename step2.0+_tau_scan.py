"""
step2.0+_tau_scan.py — Q2 结构化时滞辨识
====================================================
方法论:
  Q2 伪数据验证:  已知 tau_true 的合成数据 → 验证扫描方法
  Q2 真实时滞辨识: 对真实数据扫描 tau_total

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
#  伪数据生成器
# ================================================================
def generate_pseudo_data(tau_true=2, n_samples=None, seed=42):
    rng = np.random.RandomState(seed)
    if n_samples is None:
        n_samples = 4000
    total_len = n_samples + tau_true + 10

    true_p = {
        "beta1": 0.65, "K_d0": 0.035, "delta1": 0.40, "delta2": 0.25,
        "beta_c": 0.0008, "eta_max": 0.92, "C_phys": 0.0055, "A_cstr": 135.0
    }

    start = pd.Timestamp("2025-01-01 00:00:00")
    times = [start + pd.Timedelta(hours=i * DELTA_T) for i in range(total_len)]
    doy = np.array([t.dayofyear for t in times])
    dsin = np.sin(2 * np.pi * doy / 365)
    dcos = np.cos(2 * np.pi * doy / 365)

    rw = rng.lognormal(mean=3.0, sigma=0.85, size=total_len)
    rw = np.clip(rw, 2, 500)

    al_base = 0.04 + 0.03 * np.log1p(rw) / np.log1p(500)
    al = al_base + rng.normal(0, 0.003, total_len)
    al = np.clip(al, 0.04, 0.08)

    cl = 100 + 2.5 * rw + rng.normal(0, 40, total_len)
    cl = np.clip(cl, 50, 1200)

    cw = 3.5 + 0.02 * dsin + rng.normal(0, 0.15, total_len)
    cw = np.clip(cw, 3.0, 4.5)
    tw = 45 + 3 * dsin + rng.normal(0, 1.5, total_len)
    tw = np.clip(tw, 40, 55)

    rw_s, al_s, cl_s, ds_s, dc_s = build_shifted_arrays(rw, al, cl, dsin, dcos, tau_true)

    filt_clean = np.zeros(total_len)
    filt_clean[0] = 0.05
    for t in range(1, total_len):
        eta_t = langmuir_eta_vec(
            np.array([al_s[t]]), np.array([cl_s[t]]),
            true_p["K_d0"], true_p["delta1"], true_p["delta2"],
            true_p["beta_c"], np.array([ds_s[t]]), np.array([dc_s[t]]),
            true_p["eta_max"])[0]
        ntu_pp = rw_s[t] * (1.0 - eta_t) * true_p["C_phys"]
        filt_clean[t] = true_p["beta1"] * filt_clean[t - 1] + (1.0 - true_p["beta1"]) * ntu_pp

    ntu_clean = ntu_recurse(filt_clean, cw, tw, true_p["A_cstr"], np.array([filt_clean[0]]), total_len)

    si = tau_true + 5
    ei = si + n_samples
    obs_noise = rng.normal(0, 0.025, n_samples)
    ntu_noise = rng.normal(0, 0.015, n_samples)

    return {
        "RW_NTU": rw[si:ei], "ALUM": al[si:ei], "CLR": cl[si:ei],
        "FILT_NTU": filt_clean[si:ei] + obs_noise,
        "NTU": ntu_clean[si:ei] + ntu_noise,
        "CW_WELL": cw[si:ei], "TW_FLOW": tw[si:ei],
        "day_sin": dsin[si:ei], "day_cos": dcos[si:ei],
        "FILT_clean": filt_clean[si:ei],
        "NTU_clean": ntu_clean[si:ei],
        "true_params": true_p, "tau_true": tau_true,
    }

# ================================================================
#  参数优化
# ================================================================
def _filt_loss_vec(p, filt_obs, rw_aligned, al_aligned, cl_aligned,
                    ds_aligned, dc_aligned):
    beta1, K_d0, d1, d2, bc, emax, cp = p
    n = len(filt_obs)
    filt_pred, _, _ = filt_recurse(beta1, K_d0, d1, d2, bc, emax, cp,
                                    rw_aligned, al_aligned, cl_aligned,
                                    ds_aligned, dc_aligned, filt_obs, n)
    err = filt_obs - filt_pred
    delta = 0.05
    h = np.where(np.abs(err) < delta, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta))
    smooth = 0.05 * np.mean(np.abs(np.diff(filt_pred)))
    return np.mean(h) + smooth

def optimize_filt_model(filt_obs, rw_ntu, alum, clr, dsin, dcos, tau):
    rw_s, al_s, cl_s, ds_s, dc_s = build_shifted_arrays(rw_ntu, alum, clr, dsin, dcos, tau)

    x0 = [0.60, 0.035, 0.40, 0.25, 0.0008, 0.90, 0.005]
    bounds = [
        (0.01, 0.99), (0.001, 0.2), (-0.8, 0.8), (-0.8, 0.8),
        (0.0, 0.01), (0.50, 1.00), (0.0005, 0.05),
    ]
    best_val, best_x = float("inf"), None
    for r in range(N_RESTARTS):
        xi = np.array(x0) if r == 0 else np.array(
            [np.random.uniform(l, u) for l, u in bounds])
        res = minimize(
            lambda p: _filt_loss_vec(p, filt_obs, rw_s, al_s, cl_s, ds_s, dc_s),
            xi, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": MAX_ITER, "ftol": 1e-8}
        )
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    if best_x is None:
        return None, float("inf")

    beta1, K_d0, d1, d2, bc, emax, cp = best_x
    filt_pred, eta, _ = filt_recurse(beta1, K_d0, d1, d2, bc, emax, cp,
                                      rw_s, al_s, cl_s, ds_s, dc_s, filt_obs, len(filt_obs))
    return {
        "beta1": beta1, "K_d0": K_d0, "delta1": d1, "delta2": d2,
        "beta_c": bc, "eta_max": emax, "C_phys": cp,
        "filt_pred": filt_pred, "eta": eta,
        "loss": float(best_val),
        "rmse": float(np.sqrt(mean_squared_error(filt_obs, filt_pred))),
        "r2": float(r2_score(filt_obs, filt_pred)),
        "mae": float(mean_absolute_error(filt_obs, filt_pred)),
    }, best_val

def scan_tau(filt_obs, rw_ntu, alum, clr, dsin, dcos, max_tau=MAX_SCAN_TAU):
    results = []
    for tau in range(max_tau + 1):
        opt, _ = optimize_filt_model(filt_obs, rw_ntu, alum, clr, dsin, dcos, tau)
        if opt is None:
            results.append({"tau": tau, "tau_hours": tau * DELTA_T,
                            "rmse": float("inf"), "r2": float("-inf"), "mae": float("inf")})
            continue
        results.append({
            "tau": tau, "tau_hours": tau * DELTA_T, "rmse": opt["rmse"],
            "r2": opt["r2"], "mae": opt["mae"], "loss": opt["loss"],
            "params": {k: round(float(opt[k]), 6) for k in
                       ["beta1", "K_d0", "delta1", "delta2", "beta_c", "eta_max", "C_phys"]},
        })
    return results

# ================================================================
#  Q2 分区模型
# ================================================================
def two_zone_filt_model(df, tau_star, stress_params):
    n = len(df)
    rw = df["RW_NTU"].values; al = df["ALUM"].values; cl = df["CLR_raw"].values
    fo = df["FILT_NTU"].values; ds = df["day_sin"].values; dc = df["day_cos"].values

    mask_stress = df["FILT_NTU"] >= TIER_THRESHOLDS[1]
    mask_comfort = ~mask_stress

    if stress_params is None:
        stress_params = {"beta1": 0.95, "K_d0": 0.035, "delta1": 0.0, "delta2": 0.0,
                         "beta_c": 0.0, "eta_max": 0.90, "C_phys": 0.005}

    rw_s, al_s, cl_s, ds_s, dc_s = build_shifted_arrays(rw, al, cl, ds, dc, tau_star)

    filt_pred = np.zeros(n)
    filt_pred[0] = fo[0]
    eta_all = np.zeros(n)

    for t in range(1, n):
        if mask_stress.iloc[t]:
            eta_t = langmuir_eta_vec(
                np.array([al_s[t]]), np.array([cl_s[t]]),
                stress_params["K_d0"], stress_params["delta1"],
                stress_params["delta2"], stress_params["beta_c"],
                np.array([ds_s[t]]), np.array([dc_s[t]]),
                stress_params["eta_max"])[0]
            ntu_pp = rw_s[t] * (1.0 - eta_t) * stress_params["C_phys"]
            filt_pred[t] = stress_params["beta1"] * filt_pred[t - 1] + (1.0 - stress_params["beta1"]) * ntu_pp
            eta_all[t] = eta_t
        else:
            filt_pred[t] = 0.98 * filt_pred[t - 1] + 0.02 * fo[0]

    sm = mask_stress.values
    cm = mask_comfort.values
    full_m = compute_metrics(fo, filt_pred)
    stress_m = compute_metrics(fo[sm], filt_pred[sm]) if sm.sum() > 0 else {}
    comfort_m = compute_metrics(fo[cm], filt_pred[cm]) if cm.sum() > 0 else {}

    return {
        "filt_pred": filt_pred, "eta": eta_all,
        "full_rmse": full_m["rmse"], "full_r2": full_m["r2"],
        "stress_rmse": stress_m.get("rmse", 0), "stress_r2": stress_m.get("r2", 0),
        "comfort_rmse": comfort_m.get("rmse", 0),
        "n_stress": int(sm.sum()), "n_comfort": int(cm.sum()),
        "ar_beta_dominant": 0.98,
    }

# ================================================================
#  主程序
# ================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  Q2 Structured Tau Identification: Pseudo Verification + Real Scan")
    print("=" * 80)

    # [1] Data
    print("\n[1/8] Loading data...")
    df = load_data()

    # [2] Pseudo-data
    print("\n[2/8] Generating pseudo-data and verifying methodology...")
    tau_true = 2
    pseudo = generate_pseudo_data(tau_true=tau_true, n_samples=len(df), seed=42)
    print(f"  Ground truth: tau_total = {tau_true} steps ({tau_true * DELTA_T}h)")
    print(f"  Pseudo data: {len(pseudo['FILT_NTU'])} samples")
    print(f"  Scanning tau...")

    pseudo_scan = scan_tau(pseudo["FILT_NTU"], pseudo["RW_NTU"], pseudo["ALUM"],
                           pseudo["CLR"], pseudo["day_sin"], pseudo["day_cos"])
    best_pseudo = max(pseudo_scan, key=lambda x: x["r2"])
    correct = (best_pseudo["tau"] == tau_true)
    print(f"  tau={best_pseudo['tau']} ({best_pseudo['tau']*DELTA_T}h), R2={best_pseudo['r2']:.4f}, RMSE={best_pseudo['rmse']:.5f}")
    print(f"  Correct identification: {'YES' if correct else 'NO'} (ground truth={tau_true})")

    # [3] Robustness
    print("\n[3/8] Noise robustness test...")
    for sigma in [0.01, 0.05, 0.10]:
        pn = generate_pseudo_data(tau_true=tau_true, n_samples=min(len(df), 2000), seed=42)
        noise = np.random.RandomState(100 + int(sigma * 100)).normal(0, sigma, len(pn["FILT_NTU"]))
        pn["FILT_NTU"] = pn["FILT_clean"] + noise
        ps = scan_tau(pn["FILT_NTU"], pn["RW_NTU"], pn["ALUM"], pn["CLR"], pn["day_sin"], pn["day_cos"])
        bp = max(ps, key=lambda x: x["r2"])
        print(f"  sigma={sigma:.2f}: tau_found={bp['tau']} ({'OK' if bp['tau']==tau_true else 'FAIL'}), R2={bp['r2']:.4f}")

    # [4] Real data tau scan
    print("\n[4/8] Real data tau identification...")
    df_stress = df[df["tier"] == 3].reset_index(drop=True)
    print(f"  Stress zone: {len(df_stress)} samples (FILT >= 0.15)")
    print(f"  FILT->NTU r = {df_stress['FILT_NTU'].corr(df_stress['NTU']):.4f}")

    print("  Full scan...")
    full_scan = scan_tau(df["FILT_NTU"].values, df["RW_NTU"].values, df["ALUM"].values,
                         df["CLR_raw"].values, df["day_sin"].values, df["day_cos"].values)
    print("  Stress zone scan...")
    stress_scan = scan_tau(df_stress["FILT_NTU"].values, df_stress["RW_NTU"].values,
                           df_stress["ALUM"].values, df_stress["CLR_raw"].values,
                           df_stress["day_sin"].values, df_stress["day_cos"].values)

    best_full = max(full_scan, key=lambda x: x["r2"])
    best_stress = max(stress_scan, key=lambda x: x["r2"])
    tau_star = best_stress["tau"]

    print(f"\n  tau  Full_R2    Full_RMSE   Stress_R2   Stress_RMSE")
    for i in range(len(full_scan)):
        fr = full_scan[i]; sr = stress_scan[i]
        mk = " *" if fr["tau"] == tau_star else ""
        print(f"  {fr['tau']:3d}  {fr['r2']:8.4f}  {fr['rmse']:10.5f}  "
              f"{sr['r2']:8.4f}  {sr['rmse']:10.5f}{mk}")

    print(f"\n  tau* = {tau_star} steps ({tau_star * DELTA_T}h) [from stress zone]")
    print(f"  Full optimum at tau={best_full['tau']}")
    print(f"  Pseudo verification: {'PASS' if correct else 'CHECK'}")
    print(f"  Physics prior: 4h (2 steps) = {'CONSISTENT' if tau_star == 2 else 'DEVIATES by ' + str(abs(tau_star-2)) + ' steps'}")

    # [5] Q2 zone model
    print("\n[5/8] Q2: Zone-based FILT dynamic model...")
    zr = two_zone_filt_model(df, tau_star, best_stress.get("params"))
    print(f"  Stress zone (n={zr['n_stress']}): R2={zr['stress_r2']:.4f}, RMSE={zr['stress_rmse']:.4f}")
    print(f"  Comfort zone (n={zr['n_comfort']}): AR(1) dominant")
    print(f"  Full: R2={zr['full_r2']:.4f}, RMSE={zr['full_rmse']:.4f}")

    # [6] Save results
    print("\n[6/8] Saving results...")
    elapsed = time.time() - t0

    out = {
        "tau_scan_full": [{k: r[k] for k in ["tau", "tau_hours", "r2", "rmse"]} for r in full_scan],
        "tau_scan_stress": [{k: r[k] for k in ["tau", "tau_hours", "r2", "rmse"]} for r in stress_scan],
        "tau_star": tau_star,
        "tau_star_hours": tau_star * DELTA_T,
        "pseudo_verify": {"tau_true": tau_true, "tau_found": best_pseudo["tau"], "correct": correct},
        "stress_zone_r2": zr["stress_r2"], "stress_zone_rmse": zr["stress_rmse"],
        "full_zone_r2": zr["full_r2"], "full_zone_rmse": zr["full_rmse"],
    }
    op = os.path.join(OUTPUT_DIR, "step2_tau_results.json")
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  Results saved to: {op}")
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Pseudo-data verified: tau_true={tau_true} -> tau_found={best_pseudo['tau']} ({'OK' if correct else 'FAIL'})")
    print(f"  Real data tau*_total = {tau_star * DELTA_T}h")
    print(f"  Stress zone FILT R2 = {zr['stress_r2']:.4f}")
    print(f"  Comfort zone: AR(1) dominant (exogenous signal masked by control loop)")
    print("=" * 80)

if __name__ == "__main__":
    main()
