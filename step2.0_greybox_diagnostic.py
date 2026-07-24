"""
step2.0_greybox_diagnostic.py — Dual-Mode Threshold Detection
===============================================================
Determine FILT_NTU threshold theta via 3 independent methods:
  1. Jenks natural breaks (minimize within-class variance, k=2)
  2. Correlation change point (maximize |r_high - r_low| for RW_FLOW->FILT)
  3. GMM 2-component lognormal mixture (crossover probability point)

Output: theta.json, figures/theta_diagnostic.png
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    required = ["FILT_NTU", "RW_FLOW", "RW_NTU", "ALUM", "NTU", "CW_WELL_LEVEL", "TW_FLOW"]
    df = df.dropna(subset=required)
    return (df["FILT_NTU"].values.astype(np.float64),
            df["RW_FLOW"].values.astype(np.float64),
            df["RW_NTU"].values.astype(np.float64),
            df["ALUM"].values.astype(np.float64),
            df["NTU"].values.astype(np.float64))


def jenks_break(filt, k=2):
    """Jenks natural breaks: minimize within-class variance for k classes."""
    sorted_filt = np.sort(filt)
    n = len(sorted_filt)
    # Scan candidate splits
    best_sse, best_brk = float("inf"), 0
    for i in range(int(n * 0.05), int(n * 0.95)):  # exclude extremes
        left  = sorted_filt[:i]
        right = sorted_filt[i:]
        sse = (np.var(left) * len(left) + np.var(right) * len(right))
        if sse < best_sse:
            best_sse = sse
            best_brk = sorted_filt[i]
    return float(best_brk)


def correlation_break(filt, rw_flow, n_bins=50):
    """
    Sliding threshold scan: find the threshold that maximizes
    |corr(FILT>t) - corr(FILT<=t)| for RW_FLOW -> FILT.
    """
    candidates = np.percentile(filt, np.linspace(5, 95, n_bins))
    best_diff, best_thr = 0, 0
    results = []
    for thr in candidates:
        mask_hi = filt >= thr
        if mask_hi.sum() < 20:
            continue
        r_hi = np.corrcoef(filt[mask_hi], rw_flow[mask_hi])[0, 1]
        r_lo = np.corrcoef(filt[~mask_hi], rw_flow[~mask_hi])[0, 1]
        diff = abs(r_hi - r_lo)
        results.append({"thr": float(thr), "r_lo": float(r_lo), "r_hi": float(r_hi), "diff": float(diff)})
        if diff > best_diff:
            best_diff = diff
            best_thr = thr
    return float(best_thr), results


def gmm_break(filt):
    """2-component lognormal GMM: crossover probability point."""
    # Work in log space (FILT is lognormal)
    log_filt = np.log1p(filt[filt > 0])
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(log_filt.reshape(-1, 1))

    mu = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_.flatten())

    # Find crossover: solve p1*N(x|mu1,sigma1) = p2*N(x|mu2,sigma2)
    w = gmm.weights_
    x_range = np.linspace(min(log_filt) - 1, max(log_filt) + 1, 1000)

    def log_pdf(x, m, s):
        return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((x - m) / s) ** 2

    diff = (np.log(w[0]) + log_pdf(x_range, mu[0], sigma[0]) -
            np.log(w[1]) - log_pdf(x_range, mu[1], sigma[1]))

    crossover_idx = np.argmin(np.abs(diff))
    crossover_log = x_range[crossover_idx]
    crossover_ntu = np.expm1(crossover_log)

    gmm_info = {
        "mu": [float(mu[0]), float(mu[1])],
        "sigma": [float(sigma[0]), float(sigma[1])],
        "weights": [float(w[0]), float(w[1])],
        "crossover_log": float(crossover_log),
        "crossover_ntu": float(crossover_ntu),
    }
    return float(crossover_ntu), gmm_info


def compute_stress_correlations(filt, rw_flow, rw_ntu, alum, ntu, thr):
    """Compute correlations in stress zone (FILT >= thr)."""
    mask = filt >= thr
    n = mask.sum()
    if n < 10:
        return {"n": n, "error": "too few samples"}
    return {
        "n": int(n),
        "pct": round(100.0 * n / len(filt), 1),
        "FILT_mean": float(filt[mask].mean()),
        "FILT_std": float(filt[mask].std()),
        "r_RW_FLOW_FILT": round(float(np.corrcoef(filt[mask], rw_flow[mask])[0, 1]), 4),
        "r_ALUM_FILT": round(float(np.corrcoef(filt[mask], alum[mask])[0, 1]), 4),
        "r_RW_NTU_FILT": round(float(np.corrcoef(filt[mask], rw_ntu[mask])[0, 1]), 4),
        "r_FILT_NTU": round(float(np.corrcoef(filt[mask], ntu[mask])[0, 1]), 4),
    }


def make_figure(filt, corr_results, thr_jenks, thr_corr, thr_gmm, thr_final):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: FILT distribution + thresholds
    ax = axes[0]
    ax.hist(filt, bins=100, alpha=0.5, color="steelblue", density=True, log=True)
    for thr, label, ls in [(thr_jenks, "Jenks", "--"),
                            (thr_corr, "CorrBreak", "-."),
                            (thr_gmm, "GMM", ":"),
                            (thr_final, "Final", "-")]:
        ax.axvline(x=thr, color="red" if label == "Final" else "gray",
                   ls=ls, lw=2 if label == "Final" else 1, alpha=0.7, label=f"{label}={thr:.3f}")
    ax.set_xlabel("FILT_NTU"); ax.set_ylabel("log density")
    ax.set_title("FILT_NTU Distribution with Thresholds"); ax.legend(fontsize=7)

    # Middle: correlation change point scan
    ax = axes[1]
    if corr_results:
        thrs = [r["thr"] for r in corr_results]
        diffs = [r["diff"] for r in corr_results]
        r_lo = [r["r_lo"] for r in corr_results]
        r_hi = [r["r_hi"] for r in corr_results]
        ax.plot(thrs, r_lo, "b-", lw=1.5, alpha=0.7, label="r(FILT<=t)")
        ax.plot(thrs, r_hi, "r-", lw=1.5, alpha=0.7, label="r(FILT>t)")
        ax.axvline(x=thr_corr, color="gray", ls="--", label=f"max diff={thr_corr:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Correlation(RW_FLOW, FILT)")
    ax.set_title("Correlation Change Point"); ax.legend(fontsize=7)

    # Right: stress zone sample size vs threshold
    ax = axes[2]
    thr_range = np.percentile(filt, np.linspace(50, 99, 50))
    n_hi = [(filt >= t).sum() for t in thr_range]
    ax.plot(thr_range, n_hi, "g-", lw=2)
    ax.axhline(y=(filt >= thr_final).sum(), color="red", ls="--",
               label=f"n={int((filt>=thr_final).sum())}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("N(FILT >= threshold)")
    ax.set_title("Stress Zone Sample Count"); ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "theta_diagnostic.png"), dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print("  step2.0 — Dual-Mode Threshold Detection")
    print("=" * 60)

    filt, rw_flow, rw_ntu, alum, ntu = load_data()
    n = len(filt)
    print(f"  Valid samples: {n}")
    print(f"  FILT: mean={filt.mean():.3f} median={np.median(filt):.3f} "
          f"P90={np.percentile(filt,90):.3f}")

    # Method 1: Jenks
    thr_jenks = jenks_break(filt)
    print(f"\n  [Jenks]       theta_crit = {thr_jenks:.4f}  "
          f"n_hi={(filt>=thr_jenks).sum()} ({(filt>=thr_jenks).sum()/n*100:.1f}%)")

    # Method 2: Correlation change point
    thr_corr, corr_results = correlation_break(filt, rw_flow)
    print(f"  [CorrBreak]   theta_crit = {thr_corr:.4f}  "
          f"n_hi={(filt>=thr_corr).sum()} ({(filt>=thr_corr).sum()/n*100:.1f}%)")

    # Method 3: Correlation divergence onset (where r_hi exceeds r_lo by >= 0.10)
    thr_corr_onset = 0.20  # fallback
    if corr_results:
        for r in sorted(corr_results, key=lambda x: x["thr"]):
            if r["diff"] > 0.10:
                thr_corr_onset = r["thr"]
                break
    print(f"  [CorrOnset]   theta_mdl = {thr_corr_onset:.4f}  "
          f"n_hi={(filt>=thr_corr_onset).sum()} ({(filt>=thr_corr_onset).sum()/n*100:.1f}%)")

    # Method 4: GMM
    thr_gmm, gmm_info = gmm_break(filt)
    print(f"  [GMM]         theta = {thr_gmm:.4f} (may fail on near-zero dominant data)")

    # Critical: Jenks hard break (objective bimodal partition)
    theta_critical = float(np.median([thr_jenks, thr_corr]))
    # Modeling: softer threshold for sufficient training samples
    theta_model = thr_corr_onset

    print(f"\n  [CRITICAL]    theta_c = {theta_critical:.4f}  "
          f"n_hi={(filt>=theta_critical).sum()} ({(filt>=theta_critical).sum()/n*100:.1f}%)")
    print(f"  [MODELING]    theta_m = {theta_model:.4f}  "
          f"n_hi={(filt>=theta_model).sum()} ({(filt>=theta_model).sum()/n*100:.1f}%)")

    # Stress zone statistics at modeling threshold
    stress = compute_stress_correlations(filt, rw_flow, rw_ntu, alum, ntu, theta_model)
    stress_crit = compute_stress_correlations(filt, rw_flow, rw_ntu, alum, ntu, theta_critical)

    print(f"\n  [Stress (model)] (FILT >= {theta_model:.4f}, n={stress.get('n','?')})")
    for k in ["r_RW_FLOW_FILT", "r_ALUM_FILT", "r_RW_NTU_FILT", "r_FILT_NTU"]:
        if k in stress:
            print(f"    {k} = {stress[k]:.4f}")
    print(f"\n  [Critical zone] (FILT >= {theta_critical:.4f}, n={stress_crit.get('n','?')})")
    for k in ["r_RW_FLOW_FILT", "r_ALUM_FILT", "r_RW_NTU_FILT", "r_FILT_NTU"]:
        if k in stress_crit:
            print(f"    {k} = {stress_crit[k]:.4f}")

    # Save
    result = {
        "theta_critical": theta_critical, "theta_model": theta_model,
        "jenks": thr_jenks, "corr_break": thr_corr, "corr_onset": thr_corr_onset,
        "corr_scan": corr_results[:50], "gmm_info": gmm_info,
        "stress_zone_n": stress.get("n", 0), "critical_zone_n": stress_crit.get("n", 0),
    }
    with open(os.path.join(OUTPUT_DIR, "theta_params.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    # Figure
    make_figure(filt, corr_results, thr_jenks, thr_corr, thr_gmm, theta_model)
    print(f"\n  [DONE] theta_params.json, figures/theta_diagnostic.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
