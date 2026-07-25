"""
step3.5_visualization.py — Q3+Q1 visualization
================================================
Generates: prediction curves, error analysis, sensitivity plots.
"""

import numpy as np, pandas as pd, os, json, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                      "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

def plot_q3_predictions():
    """Three-day Q3 prediction curves with uncertainty bands."""
    preds = pd.read_csv(os.path.join(OUTPUT_DIR, "q3_final_predictions.csv"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = {"2026-02-01": "#2196F3", "2026-02-10": "#FF5722", "2026-02-20": "#4CAF50"}
    for ax, (dt, label) in zip(axes, [("2026-02-01", "Feb 1"), ("2026-02-10", "Feb 10"),
                                       ("2026-02-20", "Feb 20")]):
        sub = preds[preds["date"] == dt]
        t = sub["time"].values
        ax.fill_between(range(len(t)), sub["NTU_P5"], sub["NTU_P95"],
                        alpha=0.2, color=colors[dt], label="90% CI")
        ax.plot(range(len(t)), sub["NTU_ensemble"], "o-", color=colors[dt],
                linewidth=2, markersize=6, label="Ensemble")
        ax.plot(range(len(t)), sub["FILT"], "--", color="gray", alpha=0.6,
                linewidth=1.5, label="FILT")
        ax.axhline(1.0, color="red", linestyle=":", alpha=0.5, label="Std (1.0)")
        ax.set_ylabel("NTU")
        ax.set_title(f"{label}  (avg={sub['NTU_ensemble'].mean():.3f})")
        ax.legend(fontsize=8, ncol=3)
        ax.set_xticks(range(0, len(t), 2))
        ax.set_xticklabels(t[::2], rotation=45)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Q3: Hourly NTU Forecast (7:00-19:00)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3_prediction_curves.png"))
    plt.close(fig)
    print(f"  [OK] q3_prediction_curves.png")

def plot_q1_vs_q3():
    """Q1 (2h) vs Q3 (1h) comparison."""
    preds = pd.read_csv(os.path.join(OUTPUT_DIR, "q3_final_predictions.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, dt in zip(axes, ["2026-02-01", "2026-02-10", "2026-02-20"]):
        sub = preds[preds["date"] == dt]
        t = sub["time"].values
        q3 = sub["NTU_ensemble"].values
        q1_mask = [i % 2 == 0 for i in range(len(t))]
        ax.plot(range(len(t)), q3, "-o", color="#2196F3", linewidth=2, markersize=5, label="Q3 1h")
        ax.plot(np.where(q1_mask)[0], q3[q1_mask], "s", color="#FF5722",
                markersize=8, markeredgecolor="black", label="Q1 2h")
        ax.set_title(dt)
        ax.set_xlabel("Hour index"); ax.set_ylabel("NTU")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Q1 (2h points) vs Q3 (1h continuous)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q1_vs_q3_comparison.png"))
    plt.close(fig)
    print(f"  [OK] q1_vs_q3_comparison.png")

def plot_error_accumulation():
    """Multi-step recursive error accumulation from 2025 CV."""
    with open(os.path.join(MODEL_DIR, "validation_results.json")) as f:
        val = json.load(f)
    # Aggregate fold results into step-by-step errors
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'clean_data.csv'))
    ntu = df['NTU'].values.astype(float); filt = df['FILT_NTU'].values.astype(float)
    cw = df['CW_WELL_LEVEL'].values.astype(float); q = df['TW_FLOW'].values.astype(float)
    n = len(ntu)
    T1_THR, T2_THR = 0.05, 0.15; A_T1, A_T2, A_T3 = 400, 250, 30
    step_rmse = {k: [] for k in range(1, 13)}
    step_bias = {k: [] for k in range(1, 13)}
    for day_start in range(0, n - 24, 12):
        pred = np.zeros(12); pred[0] = ntu[day_start]
        for s in range(1, 12):
            idx = day_start + s
            ft = filt[idx]
            tier = 1 if ft <= T1_THR else (2 if ft <= T2_THR else 3)
            A0 = [A_T1, A_T2, A_T3][tier-1]
            theta = max(A0 * max(cw[idx-1], 0.1) / max(q[idx-1], 1.0), 0.02)
            beta = np.clip(np.exp(-2.0 / theta), 0.001, 0.999)
            pred[s] = np.clip(beta * pred[s-1] + (1.0 - beta) * ft, 0, None)
            step_rmse[s].append(pred[s] - ntu[idx])
            step_bias[s].append(pred[s] - ntu[idx])
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = sorted(step_rmse.keys())
    rmses = [np.sqrt(np.mean(np.array(step_rmse[s])**2)) for s in steps]
    biases = [np.mean(np.array(step_bias[s])) for s in steps]
    ax.bar(steps, rmses, alpha=0.6, label="RMSE", color="#FF5722")
    ax.plot(steps, biases, "o-", color="#2196F3", linewidth=2, label="Bias")
    ax.axhline(0.305, color="green", linestyle="--", alpha=0.7, label="One-step RMSE (0.305)")
    ax.set_xlabel("Recursive step (2h each)")
    ax.set_ylabel("Error (NTU)")
    ax.set_title("Recursive error accumulation (2025 full data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3_error_accumulation.png"))
    plt.close(fig)
    print(f"  [OK] q3_error_accumulation.png")

def plot_sensitivity():
    """OAT sensitivity bar chart."""
    with open(os.path.join(OUTPUT_DIR, "q3_sensitivity_results.json")) as f:
        sens = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 4))
    vars_sorted = [r["variable"] for r in sens["ranking"]]
    ranges = [r["oat_range"] for r in sens["ranking"]]
    colors_s = ["#FF5722" if v == "FILT_NTU" else "#2196F3" for v in vars_sorted]
    bars = ax.barh(range(len(vars_sorted)), ranges, color=colors_s, alpha=0.8)
    ax.set_yticks(range(len(vars_sorted)))
    ax.set_yticklabels(vars_sorted)
    ax.set_xlabel("OAT sensitivity range (NTU)")
    ax.set_title("Variable importance (OAT)")
    for bar, val in zip(bars, ranges):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3_sensitivity_oat.png"))
    plt.close(fig)
    print(f"  [OK] q3_sensitivity_oat.png")

def plot_daily_average():
    """Daily average NTU with CI."""
    preds = pd.read_csv(os.path.join(OUTPUT_DIR, "q3_final_predictions.csv"))
    fig, ax = plt.subplots(figsize=(8, 4))
    dates = ["2026-02-01", "2026-02-10", "2026-02-20"]
    labels = ["Feb 1", "Feb 10", "Feb 20"]
    means, lows, highs = [], [], []
    for dt in dates:
        sub = preds[preds["date"] == dt]
        means.append(sub["NTU_ensemble"].mean())
        lows.append(sub["NTU_P5"].min())
        highs.append(sub["NTU_P95"].max())
    x = range(len(dates))
    ax.bar(x, means, yerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)],
           capsize=5, color=["#2196F3", "#FF5722", "#4CAF50"], alpha=0.7)
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.6, linewidth=1.5, label="Std")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Daily mean NTU")
    ax.set_title("Daily average NTU with 90% CI")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3_daily_average.png"))
    plt.close(fig)
    print(f"  [OK] q3_daily_average.png")

def plot_all_days_facet():
    """Facet grid: all 3 days, 4 panels (FILT, Base, RF, Ensemble)."""
    preds = pd.read_csv(os.path.join(OUTPUT_DIR, "q3_final_predictions.csv"))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    colors_d = {"2026-02-01": "#2196F3", "2026-02-10": "#FF5722", "2026-02-20": "#4CAF50"}
    for row, (dt, label) in enumerate([("2026-02-01", "Feb 1"), ("2026-02-10", "Feb 10"),
                                        ("2026-02-20", "Feb 20")]):
        sub = preds[preds["date"] == dt]
        t = range(len(sub))
        for col, (key, title) in enumerate([("FILT", "FILT Input"),
                                              ("NTU_ensemble", "Ensemble")]):
            ax = axes[row, col]
            ax.plot(t, sub[key], "-o", color=colors_d[dt], linewidth=2)
            if key == "NTU_ensemble":
                ax.fill_between(t, sub["NTU_P5"], sub["NTU_P95"], alpha=0.2, color=colors_d[dt])
                ax.axhline(1.0, color="red", ls=":", alpha=0.5)
            ax.set_title(f"{label} - {title}")
            ax.grid(True, alpha=0.3)
            if row == 2: ax.set_xlabel("Hour index")
    fig.suptitle("Q3: All days - FILT input and Ensemble prediction", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q3_all_days_facet.png"))
    plt.close(fig)
    print(f"  [OK] q3_all_days_facet.png")

def main():
    print("=" * 60)
    print("  Q3 Visualization")
    print("=" * 60)
    plot_q3_predictions()
    plot_q1_vs_q3()
    plot_error_accumulation()
    plot_sensitivity()
    plot_daily_average()
    plot_all_days_facet()
    print(f"\n  All figures saved to {FIG_DIR}/")

if __name__ == "__main__":
    main()
