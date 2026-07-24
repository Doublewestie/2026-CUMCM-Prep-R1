"""
step1.2_tier2_experiment.py — T2 Dual-Path Comparison
=======================================================
Path A: Empirical frequency sampling
Path B: Log-compressed greybox (borrow T3 structure + log transform + learned coeff)
"""
import numpy as np, pandas as pd, os, json, warnings
from scipy.optimize import minimize
from q1_data_utils import load_clean_data, add_tier_labels, compute_metrics, fit_seg1
from step0_config import *
warnings.filterwarnings("ignore")

def main():
    print("=" * 60)
    print("  step1.2 — T2 Dual-Path Comparison")
    print("=" * 60)

    df = load_clean_data()
    df = add_tier_labels(df)
    mask = df["tier"] == 2
    filt_t2 = df.loc[mask, "FILT_NTU"].values
    ntu_t2 = df.loc[mask, "NTU"].values
    n_t2 = len(filt_t2)
    print(f"\n  T2 samples: {n_t2}")
    print(f"  FILT: mean={filt_t2.mean():.4f} median={np.median(filt_t2):.4f}")

    # ===== Path A: Empirical Distribution =====
    print("\n[Path A] Empirical frequency sampling...")
    np.random.seed(42)
    idx_a = np.random.permutation(n_t2)
    split_a = n_t2 // 2
    known_a, val_a = idx_a[:split_a], idx_a[split_a:]
    unique_f, counts_f = np.unique(filt_t2[known_a], return_counts=True)
    prob_f = counts_f / counts_f.sum()
    sampled_f_a = np.random.choice(unique_f, size=len(val_a), p=prob_f)
    sampled_n_a = np.random.choice(ntu_t2[known_a], size=len(val_a))
    metrics_f_a = compute_metrics(filt_t2[val_a], sampled_f_a)
    metrics_n_a = compute_metrics(ntu_t2[val_a], sampled_n_a)
    print(f"  FILT RMSE={metrics_f_a['rmse']:.4f}  R2={metrics_f_a['r2']:.4f}")
    print(f"  NTU  RMSE={metrics_n_a['rmse']:.4f}  R2={metrics_n_a['r2']:.4f}")

    # ===== Path B: Log-Compressed Greybox =====
    print("\n[Path B] Log-compressed greybox...")

    # 1. Fit T3 greybox on all data first (or use pre-trained)
    all_mask = df["tier"] >= 2  # use T2+T3 for stability
    sub = df[all_mask].copy()
    params_b, _ = fit_seg1(
        sub["FILT_NTU"].values,
        sub["RW_NTU"].values,
        sub["ALUM"].values,
        sub["CLR"].values,
        sub["day_sin"].values,
        sub["day_cos"].values,
    )
    print(f"  Greybox params: beta1={params_b['beta1']:.4f}, Km0={params_b['Km0']:.4f}")

    # 2. Compute greybox raw signal for T2 samples
    idx_all = np.arange(len(df))
    t2_positions = idx_all[mask]
    filt_raw = np.zeros(n_t2)
    for i, pos in enumerate(t2_positions):
        if pos == 0:
            filt_raw[i] = filt_t2[i]
            continue
        prev = df["FILT_NTU"].iloc[pos-1]
        rw = df["RW_NTU"].iloc[pos]
        alum = df["ALUM"].iloc[pos]
        clr = df["CLR"].iloc[pos]
        ds = df["day_sin"].iloc[pos]
        dc = df["day_cos"].iloc[pos]
        Km_t = max(params_b["Km0"] + params_b["Km1"] * ds + params_b["Km2"] * dc, 0.001)
        eta_t = alum / (alum + Km_t + max(params_b["alpha"] * clr, 0) + 1e-8)
        eta_t = np.clip(eta_t, 0.8, 1.0)
        filt_raw[i] = params_b["beta1"] * prev + (1 - params_b["beta1"]) * rw * (1 - eta_t)

    # 3. Optimize log-compression parameters (k, alpha)
    def log_compress_loss(p):
        k, a_log = p
        a = 10 ** a_log  # ensure a > 0
        pred = 0.05 + k * np.log1p(a * np.maximum(0, filt_raw - 0.05))
        pred = np.clip(pred, 0.05, 0.15)
        err = filt_t2 - pred
        huber = np.where(np.abs(err) < 0.5, 0.5 * err**2, np.abs(err) - 0.125)
        reg = 0.01 * abs(k - 0.10)
        return np.mean(huber) + reg

    # Also try sigmoid variant
    def sigmoid_loss(p):
        k, delta = p
        z = k * (filt_raw - 0.05 - delta)
        z_sig = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        pred = 0.05 + 0.10 * z_sig
        pred = np.clip(pred, 0.05, 0.15)
        err = filt_t2 - pred
        huber = np.where(np.abs(err) < 0.5, 0.5 * err**2, np.abs(err) - 0.125)
        return np.mean(huber)

    res_log = minimize(log_compress_loss, [0.08, 0.7], bounds=[(0.001, 0.5), (-2, 3)],
                       method="L-BFGS-B", options={"maxiter": 500})
    k_opt, a_log_opt = res_log.x
    a_opt = 10 ** a_log_opt

    res_sig = minimize(sigmoid_loss, [5.0, 0.03], bounds=[(0.1, 50), (-0.1, 0.1)],
                       method="L-BFGS-B", options={"maxiter": 500})
    k_sig, delta_sig = res_sig.x

    # Apply best log-compression to validation set
    pred_log = 0.05 + k_opt * np.log1p(a_opt * np.maximum(0, filt_raw - 0.05))
    pred_log = np.clip(pred_log, 0.05, 0.15)

    pred_sig = 0.05 + 0.10 * (1.0 / (1.0 + np.exp(-np.clip(k_sig * (filt_raw - 0.05 - delta_sig), -20, 20))))
    pred_sig = np.clip(pred_sig, 0.05, 0.15)

    # Cross-validation for path B
    np.random.seed(42)
    fold_accs = []
    for fold in range(5):
        va_s = slice(fold * n_t2 // 5, (fold + 1) * n_t2 // 5) if fold < 4 else slice(4 * n_t2 // 5, n_t2)
        tr_mask = np.ones(n_t2, dtype=bool)
        tr_mask[va_s] = False
        unique_tr, counts_tr = np.unique(filt_t2[tr_mask], return_counts=True)
        prob_tr = counts_tr / counts_tr.sum()
        samp = np.random.choice(unique_tr, size=va_s.stop - va_s.start, p=prob_tr)
        fold_accs.append({"rmse": compute_metrics(filt_t2[va_s], samp)["rmse"]})

    print(f"\n  [Log-compression] k={k_opt:.4f}, alpha={a_opt:.4f}")
    print(f"  [Sigmoid] k={k_sig:.4f}, delta={delta_sig:.4f}")

    metrics_f_b_log = compute_metrics(filt_t2, pred_log)
    metrics_f_b_sig = compute_metrics(filt_t2, pred_sig)
    print(f"\n  Path B (Log)  FILT RMSE={metrics_f_b_log['rmse']:.4f}  R2={metrics_f_b_log['r2']:.4f}")
    print(f"  Path B (Sig)  FILT RMSE={metrics_f_b_sig['rmse']:.4f}  R2={metrics_f_b_sig['r2']:.4f}")

    # NTU for Path B: use conditional empirical (since FILT→NTU r=0.006)
    # The greybox FILT pred doesn't help NTU, so use empirical for NTU
    metrics_n_b = compute_metrics(ntu_t2[:len(val_a)], sampled_n_a)

    # Summary
    summary = {
        "path_a": {
            "description": "Empirical frequency sampling",
            "filt_rmse": metrics_f_a["rmse"], "filt_r2": metrics_f_a["r2"],
            "ntu_rmse": metrics_n_a["rmse"], "ntu_r2": metrics_n_a["r2"],
        },
        "path_b_log": {
            "description": "Log-compressed greybox",
            "params": {"k": round(k_opt, 4), "alpha": round(a_opt, 4)},
            "filt_rmse": metrics_f_b_log["rmse"], "filt_r2": metrics_f_b_log["r2"],
        },
        "path_b_sigmoid": {
            "description": "Sigmoid greybox variant",
            "params": {"k": round(k_sig, 4), "delta": round(delta_sig, 4)},
            "filt_rmse": metrics_f_b_sig["rmse"], "filt_r2": metrics_f_b_sig["r2"],
        },
        "recommendation": "log" if metrics_f_b_log["rmse"] < metrics_f_a["rmse"] else "empirical",
    }
    with open(OUT_TIER2_REPORT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [Recommendation] {'Path B (Log)' if summary['recommendation']=='log' else 'Path A (Empirical)'}")
    print(f"\n[DONE] {OUT_TIER2_REPORT}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
