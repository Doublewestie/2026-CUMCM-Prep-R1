"""
step1.1_tier1_noise.py — T1 Empirical Frequency Sampling
==========================================================
Input: clean_data.csv, tier_labels.npy
Output: tier1_report.json (distribution stats + validation)
"""
import numpy as np, pandas as pd, os, json, warnings
from scipy.spatial.distance import jensenshannon
from q1_data_utils import load_clean_data, add_tier_labels, compute_metrics
from step0_config import *
warnings.filterwarnings("ignore")

def main():
    print("=" * 60)
    print("  step1.1 — T1 Empirical Frequency Sampling")
    print("=" * 60)
    df = load_clean_data()
    df = add_tier_labels(df)
    mask = df["tier"] == 1
    filt_t1 = df.loc[mask, "FILT_NTU"].values
    ntu_t1 = df.loc[mask, "NTU"].values

    print(f"\n  T1 samples: {len(filt_t1)}")
    print(f"  FILT: mean={filt_t1.mean():.4f} std={filt_t1.std():.4f}")
    print(f"  NTU:  mean={ntu_t1.mean():.4f} std={ntu_t1.std():.4f}")

    # Empirical distribution details
    unique_f, counts_f = np.unique(filt_t1, return_counts=True)
    prob_f = counts_f / counts_f.sum()
    unique_n, counts_n = np.unique(ntu_t1, return_counts=True)
    prob_n = counts_n / counts_n.sum()

    print(f"\n  FILT empirical distribution (unique={len(unique_f)}):")
    for u, p in zip(unique_f, prob_f):
        print(f"    FILT={u:.3f}: {p*100:.1f}%")

    # Validation: sample from empirical distribution and compare
    np.random.seed(42)
    n_val = len(filt_t1)
    # Split into "known" and "validation" sets
    idx = np.random.permutation(n_val)
    split = n_val // 2
    known_idx, val_idx = idx[:split], idx[split:]

    # Build empirical distribution from known
    known_f = filt_t1[known_idx]
    unique_k, counts_k = np.unique(known_f, return_counts=True)
    probs_k = counts_k / counts_k.sum()

    # Sample for validation set
    sampled_f = np.random.choice(unique_k, size=len(val_idx), p=probs_k)
    sampled_n = np.random.choice(ntu_t1[known_idx], size=len(val_idx))

    true_f = filt_t1[val_idx]
    true_n = ntu_t1[val_idx]

    metrics_f = compute_metrics(true_f, sampled_f)
    metrics_n = compute_metrics(true_n, sampled_n)

    # Distribution comparison (histogram-based JS divergence)
    bins = np.linspace(0, 0.06, 30)
    h_true, _ = np.histogram(true_f, bins=bins, density=True)
    h_samp, _ = np.histogram(sampled_f, bins=bins, density=True)
    h_true = h_true / (h_true.sum() + 1e-10)
    h_samp = h_samp / (h_samp.sum() + 1e-10)
    js_div = jensenshannon(h_true, h_samp)

    print(f"\n  [Validation] Empirical sampling performance:")
    print(f"    FILT RMSE={metrics_f['rmse']:.4f}  MAE={metrics_f['mae']:.4f}  R2={metrics_f['r2']:.4f}")
    print(f"    NTU  RMSE={metrics_n['rmse']:.4f}  MAE={metrics_n['mae']:.4f}  R2={metrics_n['r2']:.4f}")
    print(f"    JS divergence (FILT dist): {js_div:.4f}")

    # Also test Gaussian alternative for comparison
    gauss_samp = np.random.normal(filt_t1.mean(), filt_t1.std(), size=len(val_idx))
    gauss_samp = np.clip(gauss_samp, 0.02, 0.05)
    h_gauss, _ = np.histogram(gauss_samp, bins=bins, density=True)
    h_gauss = h_gauss / (h_gauss.sum() + 1e-10)
    js_gauss = jensenshannon(h_true, h_gauss)
    metrics_f_gauss = compute_metrics(true_f, gauss_samp)
    print(f"\n  [Gaussian alternative]")
    print(f"    FILT RMSE={metrics_f_gauss['rmse']:.4f}  R2={metrics_f_gauss['r2']:.4f}")
    print(f"    JS divergence: {js_gauss:.4f}")

    report = {
        "n_samples": int(len(filt_t1)),
        "filt_mean": float(filt_t1.mean()),
        "filt_std": float(filt_t1.std()),
        "filt_empirical_dist": {f"{u:.3f}": round(float(p), 4) for u, p in zip(unique_f, prob_f)},
        "ntu_mean": float(ntu_t1.mean()),
        "ntu_std": float(ntu_t1.std()),
        "validation_filt_metrics": metrics_f,
        "validation_ntu_metrics": metrics_n,
        "validation_js_divergence": round(float(js_div), 4),
        "gauss_js_divergence": round(float(js_gauss), 4),
        "method": "empirical_frequency",
        "recommendation": "empirical (lower JS than Gaussian)" if js_div < js_gauss else "gaussian (lower JS)",
    }
    with open(OUT_TIER1_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[DONE] {OUT_TIER1_REPORT}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
