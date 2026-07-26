"""
step3.4_sensitivity.py — Saltelli Sobol global sensitivity for CSTR model
==========================================================================
Spec: N=1024, D=5, 12,288 evaluations. Variables:
  FILT_NTU, eta_coag, TW_FLOW, CW_WELL, RW_NTU

Outputs: S_i bar chart, S_Ti bar chart, S_Ti-S_i interaction heatmap
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "savefig.dpi": 300,
                      "savefig.bbox": "tight"})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")
os.makedirs(FIG_DIR, exist_ok=True)

N_SAMPLES = 1024  # Saltelli base N
DELTA_T = 1.0     # 1h step (per spec)


def load_data_bounds():
    """Extract empirical bounds from 2025 data for each Sobol variable."""
    df = pd.read_csv(CLEAN_CSV)
    filt = df["FILT_NTU"].values.astype(float)
    rw = df["RW_NTU"].values.astype(float)
    tw = df["TW_FLOW"].values.astype(float)
    cw = df["CW_WELL_LEVEL"].values.astype(float)
    eta = (rw - filt) / np.maximum(rw, 1e-6)
    eta = np.clip(eta, 0, 1)
    return {
        "FILT_NTU": [float(np.percentile(filt, 1)), float(np.percentile(filt, 99))],
        "eta_coag": [float(np.percentile(eta[~np.isnan(eta)], 1)), float(np.percentile(eta[~np.isnan(eta)], 99))],
        "TW_FLOW": [float(np.percentile(tw, 1)), float(np.percentile(tw, 99))],
        "CW_WELL": [float(np.percentile(cw, 1)), float(np.percentile(cw, 99))],
        "RW_NTU": [float(np.percentile(rw, 1)), float(np.percentile(rw, 99))],
    }


def cstr_1h_step(filt_arr, ntu_init, cw_arr, q_arr):
    """CSTR model at 1h resolution (spec: beta = exp(-1/theta))."""
    A_T1, A_T2, A_T3 = 400, 250, 30
    A_same, A_diff = 100, 20
    RL_med, Q_med = 6.09, 44.0
    T1_THR, T2_THR = 0.05, 0.15

    pred = np.zeros(len(filt_arr))
    pred[0] = ntu_init
    for t in range(1, len(filt_arr)):
        H = max(cw_arr[t - 1], 0.1)
        Qv = max(q_arr[t - 1], 1.0)
        ft = filt_arr[t]
        if ft <= T1_THR:
            A0 = A_T1
        elif ft <= T2_THR:
            A0 = A_T2
        else:
            A0 = A_T3
        theta = A0 * H / Qv
        beta = np.clip(np.exp(-DELTA_T / max(theta, 0.02)), 0.001, 0.999)
        pred[t] = beta * pred[t - 1] + (1.0 - beta) * ft
        pred[t] = np.clip(pred[t], 0, ft + 0.05)
    return pred


def sobol_model(x):
    """Sobol model wrapper: 5 variables -> scalar output (mean NTU over 13 steps)."""
    FILT, eta_coag, TW_FLOW, CW_WELL, RW_NTU = x

    # Effective FILT: blend raw FILT with coagulation-derived FILT
    filt_coag = RW_NTU * max(1 - eta_coag, 0.001)
    FILT_eff = 0.5 * FILT + 0.5 * filt_coag
    FILT_eff = np.clip(FILT_eff, 0.001, None)

    n_steps = 13
    ntu_init = 0.18  # Jan 2026 monthly mean
    f_arr = np.full(n_steps, FILT_eff)
    c_arr = np.full(n_steps, max(CW_WELL, 0.1))
    q_arr = np.full(n_steps, max(TW_FLOW, 1.0))

    pred = cstr_1h_step(f_arr, ntu_init, c_arr, q_arr)
    return float(np.mean(pred))


def main():
    print("=" * 60)
    print("  Sobol Sensitivity: Saltelli N=1024 D=5")
    print("=" * 60)

    bounds_dict = load_data_bounds()
    problem = {
        "num_vars": 5,
        "names": ["FILT_NTU", "eta_coag", "TW_FLOW", "CW_WELL", "RW_NTU"],
        "bounds": [bounds_dict[n] for n in ["FILT_NTU", "eta_coag", "TW_FLOW", "CW_WELL", "RW_NTU"]],
    }

    print(f"  Variable bounds:")
    for i, name in enumerate(problem["names"]):
        print(f"    {name}: [{problem['bounds'][i][0]:.4f}, {problem['bounds'][i][1]:.4f}]")

    print(f"\n  Sampling: Saltelli N={N_SAMPLES}, D={problem['num_vars']}")
    print(f"  Total evaluations: {N_SAMPLES * (2 * problem['num_vars'] + 2)}")
    param_values = saltelli.sample(problem, N_SAMPLES)

    print(f"  Evaluating model ({len(param_values)} runs)...")
    Y = np.array([sobol_model(row) for row in param_values])

    print(f"  Analyzing Sobol indices...")
    Si = sobol.analyze(problem, Y, calc_second_order=True, conf_level=0.95)

    # ---- Save results ----
    out = {
        "N": N_SAMPLES,
        "D": problem["num_vars"],
        "variable_bounds": {n: b for n, b in zip(problem["names"], problem["bounds"])},
        "S1": {name: round(float(Si["S1"][i]), 5) for i, name in enumerate(problem["names"])},
        "S1_conf": {name: round(float(Si["S1_conf"][i]), 5) for i, name in enumerate(problem["names"])},
        "ST": {name: round(float(Si["ST"][i]), 5) for i, name in enumerate(problem["names"])},
        "ST_conf": {name: round(float(Si["ST_conf"][i]), 5) for i, name in enumerate(problem["names"])},
    }
    with open(os.path.join(OUTPUT_DIR, "q3_sensitivity_sobol.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  [DONE] q3_sensitivity_sobol.json")

    # ---- Figure 1: S_i first-order bar chart ----
    fig, ax = plt.subplots(figsize=(8, 4))
    names_short = ["FILT_NTU", "eta_coag", "TW_FLOW", "CW_WELL", "RW_NTU"]
    x = np.arange(len(names_short))
    ax.bar(x, Si["S1"], yerr=Si["S1_conf"], capsize=4, color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names_short, fontsize=9, rotation=20)
    ax.set_ylabel("S_i (first-order)")
    ax.set_title("Sobol First-Order Sensitivity of CSTR NTU Prediction")
    ax.axhline(0, color="gray", lw=0.5)
    for i, v in enumerate(Si["S1"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3_sobol_s1.png"))
    plt.close(fig)
    print(f"  [DONE] q3_sobol_s1.png")

    # ---- Figure 2: S_Ti total-order bar chart ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, Si["ST"], yerr=Si["ST_conf"], capsize=4, color="darkorange", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names_short, fontsize=9, rotation=20)
    ax.set_ylabel("S_Ti (total-order)")
    ax.set_title("Sobol Total-Order Sensitivity of CSTR NTU Prediction")
    ax.axhline(0, color="gray", lw=0.5)
    for i, v in enumerate(Si["ST"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3_sobol_st.png"))
    plt.close(fig)
    print(f"  [DONE] q3_sobol_st.png")

    # ---- Figure 3: Interaction heatmap (S_Ti - S_i) ----
    if Si.get("S2") is not None and len(Si["S2"]) > 0:
        inter = Si["ST"] - Si["S1"]
        inter = np.clip(inter, 0, None)

        fig, ax = plt.subplots(figsize=(6, 5))
        n_v = len(names_short)
        im = ax.imshow(np.diag(inter), cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(n_v))
        ax.set_yticks(range(n_v))
        ax.set_xticklabels(names_short, fontsize=8, rotation=30)
        ax.set_yticklabels(names_short, fontsize=8)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("S_Ti - S_i (interaction)")
        ax.set_title("Sobol Interaction Effects (diagonal)")
        for i in range(n_v):
            ax.text(i, i, f"{inter[i]:.3f}", ha="center", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "q3_sobol_interaction.png"))
        plt.close(fig)
        print(f"  [DONE] q3_sobol_interaction.png")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SOBOL RESULTS")
    print(f"{'='*60}")
    print(f"  {'Variable':<15s} {'S_i':>8s} {'S_i_conf':>10s} {'S_Ti':>8s} {'S_Ti_conf':>10s}")
    for i, name in enumerate(problem["names"]):
        print(f"  {name:<15s} {Si['S1'][i]:>8.4f} {Si['S1_conf'][i]:>10.4f} {Si['ST'][i]:>8.4f} {Si['ST_conf'][i]:>10.4f}")

    ranking = np.argsort(Si["ST"])[::-1]
    print(f"\n  Ranking (by S_Ti):")
    for i, idx in enumerate(ranking):
        print(f"    {i+1}. {problem['names'][idx]}: S_Ti={Si['ST'][idx]:.4f}, S_i={Si['S1'][idx]:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
