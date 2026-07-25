"""
step1.6_cstr_refinement.py — A-line CSTR formula refinement (N-CSTR + delay + var-area + wall-release)
======================================================================================================
Self-contained ablation experiment. No dependency on step1.3 or step1_shared_utils.
Reads clean_data.csv directly. All params defined locally.

Refinements:
  A1: N-stage CSTR-in-series + transport delay delta (filter-to-clearwell pipe)
  A2: Flow-dependent effective area (high flow -> short-circuiting -> smaller A_eff)
  A3: Wall sediment release on flow up-ramp

Output:
  output/cstr_refinement_ablation.csv   — full ablation matrix
  output/cstr_refinement_best.json      — best config
  output/figures/cstr_refinement.png    — ablation bar chart
"""

import os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
EPS = 1e-6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")
DELTA_T = 2.0
TIER_THRESHOLDS = [0.05, 0.15]

# ================================================================
#  Data loading
# ================================================================
def load_data():
    df = pd.read_csv(CLEAN_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])

    out = {}
    for col in ["FILT_NTU", "NTU", "CW_WELL_LEVEL", "TW_FLOW"]:
        out[col] = df[col].values.astype(np.float64)
    out["Q_max"] = float(out["TW_FLOW"].max())
    out["N"] = len(df)

    # Tier masks
    out["tier"] = np.zeros(out["N"], dtype=int)
    out["tier"][out["FILT_NTU"] <= TIER_THRESHOLDS[0]] = 1
    out["tier"][(out["FILT_NTU"] > TIER_THRESHOLDS[0]) & (out["FILT_NTU"] <= TIER_THRESHOLDS[1])] = 2
    out["tier"][out["FILT_NTU"] > TIER_THRESHOLDS[1]] = 3
    return out

# ================================================================
#  Core CSTR N-series predictor
# ================================================================
def predict_n_cstr(filt, ntu, cw, tw, N, delta, A0, k_A=0.0, C_base=0.0, Q_max=55.0):
    """
    N-stage CSTR-in-series with transport delay, variable area, wall release.

    Parameters
    ----------
    N : int          number of CSTR stages in series (>=1)
    delta : int      transport delay steps from filter to clearwell (0,1,2)
    A0 : float       baseline clearwell area (m2), original optimum = 141.3
    k_A : float      flow compression coefficient for A_eff (0..0.5)
    C_base : float   wall sediment release intensity (NTU)
    Q_max : float    max outflow for normalization

    Returns
    -------
    pred_ntu : np.ndarray   predicted NTU series (same length as input)
    """
    n = len(filt)
    pred = np.zeros(n)
    if n == 0:
        return pred
    pred[0] = ntu[0]

    # Build cascade stages to process FILT contribution (N-1 stages feeding the final)
    # All stages carry over their own predicted previous value
    n_cascade = max(1, N - 1)
    C = np.zeros((n_cascade, n))
    for s in range(n_cascade):
        C[s, 0] = ntu[0]

    for t in range(1, n):
        # ---- effective area ----
        tw_prev = tw[t - 1]
        if k_A > 0 and Q_max > 0:
            A_eff = A0 * max(0.2, 1.0 - k_A * tw_prev / Q_max)
        else:
            A_eff = A0

        # ---- theta & per-stage beta (beta_N = beta_2^N) ----
        cw_prev = cw[t - 1]
        theta = A_eff * cw_prev / max(tw_prev, 1.0)
        theta = max(theta, 0.02)
        beta_N = np.exp(-N * DELTA_T / theta)
        beta_N = np.clip(beta_N, 0.001, 0.999)

        # ---- FILT with transport delay ----
        filt_idx = max(0, t - delta)
        filt_in = filt[filt_idx]

        # ---- Process FILT through cascade (N-1 autoregressive stages) ----
        if N == 1:
            # Single-stage: FILT feeds output directly, no cascade
            cascade_feed = filt_in
        else:
            # Stage 0: receives FILT feed
            C[0, t] = beta_N * C[0, t - 1] + (1.0 - beta_N) * filt_in
            # Stages 1..N-2: cascade forward
            for s in range(1, n_cascade):
                C[s, t] = beta_N * C[s, t - 1] + (1.0 - beta_N) * C[s - 1, t]
            cascade_feed = C[n_cascade - 1, t]

        # ---- Final output: TRUE carryover + cascade-processed feed ----
        pred[t] = beta_N * ntu[t - 1] + (1.0 - beta_N) * cascade_feed

        # ---- Wall release ----
        if C_base > 0:
            tw_curr = tw[t] if t < n else tw[-1]
            dQ_ratio = (tw_curr - tw_prev) / max(tw_prev, 1.0)
            pred[t] += C_base * max(0.0, dQ_ratio)

    return np.clip(pred, 0.0, np.inf)


# ================================================================
#  Metrics
# ================================================================
def compute_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"r2": round(float(r2), 4), "rmse": round(float(rmse), 4), "mae": round(float(mae), 4)}


def evaluate_config(data, N, delta, A0, k_A, C_base):
    pred = predict_n_cstr(
        data["FILT_NTU"], data["NTU"],
        data["CW_WELL_LEVEL"], data["TW_FLOW"],
        N, delta, A0, k_A, C_base, data["Q_max"],
    )
    ntu = data["NTU"]
    tier = data["tier"]
    results = {}

    # Full
    m_all = compute_metrics(ntu, pred)
    results["R2_all"] = m_all["r2"]
    results["RMSE_all"] = m_all["rmse"]

    # Per tier
    for t_id, t_name in [(1, "T1"), (2, "T2"), (3, "T3")]:
        mask = tier == t_id
        if mask.sum() < 10:
            results[f"R2_{t_name}"] = None
            results[f"RMSE_{t_name}"] = None
        else:
            m = compute_metrics(ntu[mask], pred[mask])
            results[f"R2_{t_name}"] = m["r2"]
            results[f"RMSE_{t_name}"] = m["rmse"]

    # Extreme zone (FILT > 0.5)
    ext = data["FILT_NTU"] > 0.5
    if ext.sum() >= 10:
        m_ext = compute_metrics(ntu[ext], pred[ext])
        results["R2_ext"] = m_ext["r2"]
        results["RMSE_ext"] = m_ext["rmse"]
        results["n_ext"] = int(ext.sum())
    else:
        results["R2_ext"] = None
        results["RMSE_ext"] = None
        results["n_ext"] = 0

    results["N"] = N
    results["delta"] = delta
    results["A0"] = A0
    results["k_A"] = k_A
    results["C_base"] = C_base
    return results, pred


# ================================================================
#  Main
# ================================================================
def main():
    print("=" * 70)
    print("  step1.6 — CSTR Formula Refinement (A-line Ablation)")
    print("=" * 70)

    data = load_data()
    print(f"\n  Loaded {data['N']} samples, Q_max = {data['Q_max']:.2f}")

    rows = []

    # ==================== BASELINE ====================
    print("\n[Baseline] Single-stage CSTR, A=141.3, delta=0 ...")
    r_bl, _ = evaluate_config(data, N=1, delta=0, A0=141.3, k_A=0.0, C_base=0.0)
    r_bl["config"] = "Baseline"
    rows.append(r_bl)
    print(f"  Full R2={r_bl['R2_all']:.4f}  RMSE={r_bl['RMSE_all']:.4f}  "
          f"T3_R2={r_bl['R2_T3']:.4f}  ext_R2={r_bl['R2_ext']:.4f}")

    # ==================== PHASE A1: N-CSTR + delay ====================
    print("\n[Phase A1] N-CSTR + transport delay (delta)")
    print("  " + "-" * 60)
    best_a1_r2 = -999
    best_a1 = None
    for N in [1, 2, 3, 4, 5]:
        for delta in [0, 1, 2]:
            r, _ = evaluate_config(data, N=N, delta=delta, A0=141.3, k_A=0.0, C_base=0.0)
            r["config"] = f"A1_N={N}_d={delta}"
            rows.append(r)
            marker = ""
            if r["R2_all"] is not None and r["R2_all"] > best_a1_r2:
                best_a1_r2 = r["R2_all"]
                best_a1 = (N, delta)
                marker = " ***"
            print(f"    N={N} delta={delta}  R2={r['R2_all']:.4f}  "
                  f"T3_R2={r['R2_T3']:.4f}  ext_R2={r['R2_ext']:.4f}{marker}")

    N_best, d_best = best_a1
    print(f"\n  [A1 best] N={N_best}, delta={d_best}, R2={best_a1_r2:.4f}")

    # ==================== PHASE A2: Variable area ====================
    print(f"\n[Phase A2] Variable area (N={N_best}, delta={d_best})")
    print("  " + "-" * 60)
    best_a2_r2 = -999
    best_k = 0.0
    for k_A in [0.0, 0.05, 0.10, 0.15, 0.20]:
        r, _ = evaluate_config(data, N=N_best, delta=d_best, A0=141.3, k_A=k_A, C_base=0.0)
        r["config"] = f"A2_kA={k_A:.2f}"
        rows.append(r)
        marker = ""
        if r["R2_all"] is not None and r["R2_all"] > best_a2_r2:
            best_a2_r2 = r["R2_all"]
            best_k = k_A
            marker = " ***"
        print(f"    k_A={k_A:.2f}  R2={r['R2_all']:.4f}  "
              f"T3_R2={r['R2_T3']:.4f}  ext_R2={r['R2_ext']:.4f}{marker}")

    print(f"\n  [A2 best] k_A={best_k:.2f}, R2={best_a2_r2:.4f}")

    # ==================== PHASE A3: Wall release ====================
    print(f"\n[Phase A3] Wall release (N={N_best}, delta={d_best}, k_A={best_k:.2f})")
    print("  " + "-" * 60)
    best_a3_r2 = -999
    best_C = 0.0
    for C_base in [0.0, 0.01, 0.02, 0.05, 0.10]:
        r, _ = evaluate_config(data, N=N_best, delta=d_best, A0=141.3, k_A=best_k, C_base=C_base)
        r["config"] = f"A3_Cb={C_base:.2f}"
        rows.append(r)
        marker = ""
        if r["R2_all"] is not None and r["R2_all"] > best_a3_r2:
            best_a3_r2 = r["R2_all"]
            best_C = C_base
            marker = " ***"
        print(f"    C_base={C_base:.2f}  R2={r['R2_all']:.4f}  "
              f"T3_R2={r['R2_T3']:.4f}  ext_R2={r['R2_ext']:.4f}{marker}")

    print(f"\n  [A3 best] C_base={best_C:.2f}, R2={best_a3_r2:.4f}")

    # ==================== SAVE ====================
    df_ab = pd.DataFrame(rows)
    col_order = ["config", "N", "delta", "A0", "k_A", "C_base",
                 "R2_all", "RMSE_all", "R2_T1", "R2_T2", "R2_T3", "R2_ext",
                 "RMSE_T1", "RMSE_T2", "RMSE_T3", "RMSE_ext", "n_ext"]
    df_ab = df_ab[[c for c in col_order if c in df_ab.columns]]
    csv_path = os.path.join(OUTPUT_DIR, "cstr_refinement_ablation.csv")
    df_ab.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] {csv_path}")

    # Best config
    best_row = max(rows, key=lambda x: x.get("R2_all", -999))
    best_path = os.path.join(OUTPUT_DIR, "cstr_refinement_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({k: best_row[k] for k in ["N", "delta", "A0", "k_A", "C_base",
                                              "R2_all", "R2_T3", "R2_ext"]}, f, indent=2)
    print(f"[DONE] {best_path}")

    # ==================== SUMMARY ====================
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    bl_r2 = rows[0]["R2_all"]
    print(f"  Baseline:        R2={bl_r2:.4f}  T3_R2={rows[0]['R2_T3']:.4f}  ext_R2={rows[0]['R2_ext']:.4f}")
    print(f"  Best A1+A2+A3:   R2={best_a3_r2:.4f}  T3_R2={best_row['R2_T3']:.4f}  ext_R2={best_row['R2_ext']:.4f}")
    print(f"  Delta:           dR2={best_a3_r2 - bl_r2:+.4f}  dT3={best_row['R2_T3'] - rows[0]['R2_T3']:+.4f}  dExt={best_row['R2_ext'] - rows[0]['R2_ext']:+.4f}")
    print(f"  Best params:     N={best_row['N']}  delta={best_row['delta']}  k_A={best_row['k_A']}  C_base={best_row['C_base']}")

    # ==================== FIGURE ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    phases = [("A1", [r for r in rows if r["config"].startswith("A1_")]),
              ("A2", [r for r in rows if r["config"].startswith("A2_")]),
              ("A3", [r for r in rows if r["config"].startswith("A3_")])]

    for ax, (label, phase_rows) in zip(axes, phases):
        if not phase_rows:
            continue
        configs = [r["config"].replace("A1_","").replace("A2_","").replace("A3_","") for r in phase_rows]
        r2s = [r["R2_all"] for r in phase_rows]
        t3s = [r["R2_T3"] for r in phase_rows]
        exts = [r["R2_ext"] for r in phase_rows]
        x = np.arange(len(configs))
        w = 0.25
        ax.bar(x - w, r2s, w, label="R2_all", color="steelblue", alpha=0.85)
        ax.bar(x, t3s, w, label="R2_T3", color="darkorange", alpha=0.85)
        ax.bar(x + w, exts, w, label="R2_ext(>0.5)", color="firebrick", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8)
        ax.set_title(f"Phase {label}")
        ax.legend(fontsize=7)
        ax.axhline(y=bl_r2, color="gray", ls="--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "cstr_refinement.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {fig_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
