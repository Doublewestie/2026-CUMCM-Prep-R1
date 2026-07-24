"""
step1.5_visualization.py — Q1 Three-Tier Greybox Visualization
===============================================================
Output: output/figures/*.png
"""
import numpy as np, pandas as pd, os, json, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from q1_data_utils import load_clean_data, add_tier_labels
from step0_config import *
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = load_clean_data()
df = add_tier_labels(df)
filt, ntu = df["FILT_NTU"].values, df["NTU"].values
tier = df["tier"].values

# ========== 1. Tier Distribution ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (name, mk, color) in enumerate([
    ("T1: ≤0.05", tier==1, "#4ECDC4"),
    ("T2: 0.05~0.15", tier==2, "#FFE66D"),
    ("T3: >0.15", tier==3, "#FF6B6B"),
]):
    axes[0].hist(filt[mk], bins=50, alpha=0.6, color=color, label=f"{name} (n={mk.sum()})", density=True)
axes[0].axvline(x=0.05, color="gray", ls="--", lw=1)
axes[0].axvline(x=0.15, color="gray", ls="--", lw=1)
axes[0].set_xlabel("FILT_NTU"); axes[0].set_ylabel("Density")
axes[0].set_title("FILT_NTU Distribution by Tier"); axes[0].legend(fontsize=8)

counts = [(tier==1).sum(), (tier==2).sum(), (tier==3).sum()]
bars = axes[1].bar(["T1 (≤0.05)", "T2 (0.05~0.15)", "T3 (>0.15)"], counts,
                   color=["#4ECDC4", "#FFE66D", "#FF6B6B"], edgecolor="white")
for bar, c in zip(bars, counts):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                 f"{c}\n({c/len(filt)*100:.1f}%)", ha="center", fontsize=10)
axes[1].set_ylabel("Sample Count"); axes[1].set_title("Three-Tier Partition")
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "q1_tier_distribution.png"), dpi=200, bbox_inches="tight")
plt.close(); print("[FIG] q1_tier_distribution.png")

# ========== 2. T3 Feature Importance ==========
try:
    fi = pd.read_csv(OUT_TIER_FACTOR)
    top12 = fi.head(12)
    colors = plt.cm.RdYlGn_r(top12["robust"] / top12["robust"].max())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top12)), top12["robust"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top12))); ax.set_yticklabels(top12["feature"].values)
    ax.set_xlabel("Robust Importance (SHAP×Perm)²"); ax.set_title("T3 Feature Importance (Top 12)")
    for i, v in enumerate(top12["robust"]):
        ax.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "q1_t3_feature_importance.png"), dpi=200, bbox_inches="tight")
    plt.close(); print("[FIG] q1_t3_feature_importance.png")
except: print("[SKIP] q1_t3_feature_importance (no data)")

# ========== 3. CSTR Prediction vs Actual (NTU, per tier) ==========
pred = np.zeros_like(ntu); pred[0] = ntu[0]
cw, tw = df["CW_WELL_LEVEL"].values, df["TW_FLOW"].values
for t in range(1, len(ntu)):
    th = 141.3 * cw[t-1] / max(tw[t-1], 1)
    b2 = np.exp(-2.0 / max(th, 0.1))
    pred[t] = b2 * ntu[t-1] + (1 - b2) * filt[t]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors_tier = {1: "#4ECDC4", 2: "#FFE66D", 3: "#FF6B6B"}
names_tier = {1: "T1 (≤0.05)", 2: "T2 (0.05~0.15)", 3: "T3 (>0.15)"}
for i, t_id in enumerate([1, 2, 3]):
    mk = tier == t_id
    axes[i].scatter(ntu[mk], pred[mk], c=colors_tier[t_id], alpha=0.3, s=5)
    lims = [0, max(ntu[mk].max(), pred[mk].max()) * 1.1]
    axes[i].plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    axes[i].set_xlim(lims); axes[i].set_ylim(lims)
    axes[i].set_xlabel("True NTU"); axes[i].set_ylabel("Predicted NTU")
    from sklearn.metrics import r2_score
    r2 = r2_score(ntu[mk], pred[mk])
    axes[i].set_title(f"{names_tier[t_id]} (R²={r2:.3f}, n={mk.sum()})")
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "q1_cstr_pred_vs_actual.png"), dpi=200, bbox_inches="tight")
plt.close(); print("[FIG] q1_cstr_pred_vs_actual.png")

# ========== 4. T1 Empirical Distribution ==========
fig, ax = plt.subplots(figsize=(8, 5))
mask1 = tier == 1
vals = filt[mask1]
unique, counts = np.unique(vals, return_counts=True)
ax.bar(unique, counts / counts.sum(), width=0.008, color="#4ECDC4", edgecolor="white", alpha=0.8)
# Gaussian overlay
from scipy.stats import norm
x_range = np.linspace(0.015, 0.055, 100)
gauss = norm.pdf(x_range, vals.mean(), vals.std())
gauss = gauss / gauss.sum() * (len(x_range) / 0.04 * 0.008)
ax.plot(x_range, gauss * (counts.sum() * 0.008), "r--", lw=2, label="Gaussian fit")
ax.set_xlabel("FILT_NTU"); ax.set_ylabel("Frequency"); ax.set_title("T1 FILT_NTU: Empirical vs Gaussian")
ax.legend()
for u, c in zip(unique, counts):
    ax.text(u, c / counts.sum() + 0.01, f"{c/counts.sum()*100:.1f}%", ha="center", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "q1_t1_empirical_dist.png"), dpi=200, bbox_inches="tight")
plt.close(); print("[FIG] q1_t1_empirical_dist.png")

# ========== 5. Overall Summary ==========
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
summary_text = (
    "Q1 Three-Tier Greybox Scheme Summary\n"
    "═══════════════════════════════════\n\n"
    "Model: NTU(t) = b2 * NTU(t-1) + (1-b2) * FILT(t)\n"
    "  b2 = exp(-2h / theta), theta = A * CW_WELL_LEVEL / TW_FLOW\n\n"
    "Overall NTU R2 = 0.727 (vs original XGBoost 0.34)\n"
    "  T1 (<=0.05, 49%): Empirical sampling, R2=0.86\n"
    "  T2 (0.05~0.15, 30%): Log-compressed greybox, R2=0.76\n"
    "  T3 (>0.15, 21%): CSTR+feedback, R2=0.67\n\n"
    "T3 Key Factors:\n"
    "  1. eta_coag (0.335) - Removal efficiency\n"
    "  2. FILT_NTU_mean6 (0.242) - Recent filter trend\n"
    "  3. TW_FLOW (0.053) - Clearwell outflow\n"
    "  4. day_cos (0.048) - Seasonal\n"
    "  5. RIVER_LEVEL (0.041) - River level\n\n"
    "tau_1 learned via softmax: peak at 4h"
)
ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="center", family="monospace")
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "q1_summary.png"), dpi=200, bbox_inches="tight")
plt.close(); print("[FIG] q1_summary.png")

print(f"\nAll figures saved to {FIG_DIR}")
