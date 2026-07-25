"""
q1_generate_figures.py — Generate Q1 final figures (3-tier CSTR model)
Model: NTU(t) = beta2(t)*NTU(t-1) + (1-beta2(t))*FILT(t)
       beta2(t) = exp(-2h/theta), theta = A_tier * CW_WELL(t-1) / TW_FLOW(t-1)
       A_tier = 400 (T1), 250 (T2), 30 (T3)
All labels in English
Uses clean_data.csv from step0_preprocess (consistent with step1.7 results)
"""

import os, numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12})

BASE = r'C:\Users\lenovo\2026-CUMCM-Prep-R1'
OUT_FIG = os.path.join(BASE, 'results', 'figures')
OUT_TAB = os.path.join(BASE, 'results', 'tables')
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
EPS = 1e-6
DT = 2.0

# Load clean data from step0 (produces R2=0.7827, consistent with step1.7)
data = pd.read_csv(os.path.join(BASE, 'output', 'clean_data.csv'))
data = data.dropna(subset=['NTU','FILT_NTU','CW_WELL_LEVEL','TW_FLOW']).reset_index(drop=True)
n = len(data)
print(f"Data loaded from clean_data.csv: {n} samples")

filt = data['FILT_NTU'].values.astype(float)
ntu = data['NTU'].values.astype(float)
cw = data['CW_WELL_LEVEL'].values.astype(float)
tw = data['TW_FLOW'].values.astype(float)

# 3-tier CSTR model with Balance Detector (same as step1.7_final_cstr.py Phase 5)
A_T1, A_T2, A_T3 = 400, 250, 30
T1_THR, T2_THR = 0.05, 0.15

# Balance detector params (from Phase 5 best: T3-only median)
A_same, A_diff = 100, 20
rl = data['RIVER_LEVEL'].values.astype(float)
mT3 = filt > T2_THR
rl_t3 = rl[mT3]
rl_t3 = rl_t3[~np.isnan(rl_t3)]
RL_med = float(np.median(rl_t3)) if len(rl_t3) > 0 else 6.09
Q_med = float(np.median(tw[mT3]))

def predict_ntu(use_balance=True):
    n = len(ntu)
    pred = np.zeros(n)
    pred[0] = ntu[0]
    for t in range(1, n):
        H = max(cw[t-1], 0.1)
        Qv = max(tw[t-1], 1.0)
        ft = filt[t]
        if ft <= T1_THR:
            A0 = A_T1
        elif ft <= T2_THR:
            A0 = A_T2
        else:
            # T3: balance detector (match step1.7 Phase 5: uses Q[t] not Q[t-1])
            if use_balance:
                rl_t = rl[t]
                if not np.isnan(rl_t):
                    bal = (rl_t - RL_med) * (tw[t] - Q_med)
                    A0 = A_same if bal > 0 else A_diff
                else:
                    A0 = A_T3
            else:
                A0 = A_T3
        theta = A0 * H / Qv
        theta = max(theta, 0.02)
        beta = np.exp(-DT / max(theta, 0.02))
        beta = np.clip(beta, 0.001, 0.999)
        pred[t] = beta * ntu[t-1] + (1.0 - beta) * ft
    return np.clip(pred, 0, np.inf)

# Evaluate final model with balance detector
pred = predict_ntu(use_balance=True)
pred_base = predict_ntu(use_balance=False)
r2_all = r2_score(ntu, pred)
rmse_all = np.sqrt(mean_squared_error(ntu, pred))

# Tier breakdown
t1_m = filt <= T1_THR
t2_m = (filt > T1_THR) & (filt <= T2_THR)
t3_m = filt > T2_THR

t1_r2 = r2_score(ntu[t1_m], pred[t1_m]) if t1_m.sum() > 5 else np.nan
t2_r2 = r2_score(ntu[t2_m], pred[t2_m]) if t2_m.sum() > 5 else np.nan
t3_r2 = r2_score(ntu[t3_m], pred[t3_m]) if t3_m.sum() > 5 else np.nan
t1_rmse = np.sqrt(mean_squared_error(ntu[t1_m], pred[t1_m])) if t1_m.sum() > 5 else np.nan
t2_rmse = np.sqrt(mean_squared_error(ntu[t2_m], pred[t2_m])) if t2_m.sum() > 5 else np.nan
t3_rmse = np.sqrt(mean_squared_error(ntu[t3_m], pred[t3_m])) if t3_m.sum() > 5 else np.nan

r2_base = r2_score(ntu, pred_base)
print(f"CSTR 3-Tier + Balance Detector (A_same={A_same}, A_diff={A_diff})")
print(f"  Baseline (uniform A): R2_all={r2_base:.4f}")
print(f"  Balance Detector:     R2_all={r2_all:.4f}, RMSE={rmse_all:.4f}")
print(f"  T1: R2={t1_r2:.4f}, RMSE={t1_rmse:.4f} (n={t1_m.sum()})")
print(f"  T2: R2={t2_r2:.4f}, RMSE={t2_rmse:.4f} (n={t2_m.sum()})")
print(f"  T3: R2={t3_r2:.4f}, RMSE={t3_rmse:.4f} (n={t3_m.sum()})")

# ================================================================
# Fig 1: Pred vs Actual (in-sample)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(ntu, pred, s=3, alpha=0.3, c='steelblue', edgecolors='none')
ax.plot([0, 10], [0, 10], 'r--', linewidth=1.5, alpha=0.7, label='y = x')
ax.set_xlabel('True NTU')
ax.set_ylabel('Predicted NTU')
ax.set_title(f'3-Tier CSTR: Full Data (R2={r2_all:.4f}, RMSE={rmse_all:.4f})')
ax.legend(loc='lower right')
ax.set_xlim(0, 3); ax.set_ylim(0, 3)
ax.grid(True, alpha=0.3)

ax = axes[1]
colors_tier = ['#2c7bb6', '#fdae61', '#d7191c']
labels_tier = ['T1 (FILT<=0.05)', 'T2 (0.05-0.15)', 'T3 (FILT>0.15)']
for i, (mask, lbl, c) in enumerate(zip([t1_m, t2_m, t3_m], labels_tier, colors_tier)):
    ax.scatter(ntu[mask], pred[mask], s=3, alpha=0.3, c=c, edgecolors='none', label=lbl)
ax.plot([0, 10], [0, 10], 'r--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('True NTU')
ax.set_ylabel('Predicted NTU')
ax.set_title('3-Tier CSTR: Colored by FILT Tier')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 3); ax.set_ylim(0, 3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q1_pred_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: q1_pred_vs_actual.png")

# ================================================================
# Fig 2: Tier comparison bar chart
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
t_names = ['T1\n(FILT<=0.05)', 'T2\n(0.05-0.15)', 'T3\n(FILT>0.15)', 'All']
r2s = [t1_r2, t2_r2, t3_r2, r2_all]
colors = ['#2c7bb6', '#fdae61', '#d7191c', '#636363']
ax.bar(t_names, r2s, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_ylabel('R2')
ax.set_title('CSTR 3-Tier: R2 by FILT Tier')
for i, v in enumerate(r2s):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
rms = [t1_rmse, t2_rmse, t3_rmse, rmse_all]
ax.bar(t_names, rms, color=colors, alpha=0.85, edgecolor='white')
ax.set_ylabel('RMSE')
ax.set_title('CSTR 3-Tier: RMSE by FILT Tier')
for i, v in enumerate(rms):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q1_tier_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: q1_tier_comparison.png")

# ================================================================
# Fig 3: CSTR parameters and formula
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: A_tier parameter table as text
ax = axes[0]
ax.axis('off')
param_text = f"""
CSTR + BALANCE DETECTOR
{'='*30}
A_T1 (FILT <= 0.05)    = {A_T1}
A_T2 (0.05-0.15)       = {A_T2}
A_same (RL x Q > 0)    = {A_same}
A_diff (RL x Q < 0)    = {A_diff}
RL_med / Q_med         = {RL_med:.1f} / {Q_med:.1f}
{'='*30}
beta2(t) = exp(-2h / theta(t))
theta(t) = A_eff * CW_WELL(t-1) 
                  / TW_FLOW(t-1)
{'='*30}
NTU(t) = beta2 * NTU(t-1)
      + (1-beta2) * FILT(t)
{'='*30}
Balance = (RL - RL_med) 
        x (TW_FLOW - Q_med)
{'='*30}
R2_all = {r2_all:.4f}  (base={r2_base:.4f})
RMSE   = {rmse_all:.4f}
n      = {n}
"""
ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')
ax.set_title('CSTR Model Configuration', fontsize=12)

# Middle: beta2(t) distribution
ax = axes[1]
cw_s = np.roll(cw, 1); cw_s[0] = cw[0]
tw_s = np.roll(tw, 1); tw_s[0] = tw[0]
# Compute per-tier beta2
beta2_all = np.zeros(n)
for t in range(n):
    ft = filt[t]
    if ft <= T1_THR: A0 = A_T1
    elif ft <= T2_THR: A0 = A_T2
    else: A0 = A_T3
    H = max(cw_s[t], 0.1); Q = max(tw_s[t], 1.0)
    th = A0 * H / Q
    beta2_all[t] = np.clip(np.exp(-DT / max(th, 0.02)), 0.001, 0.999)

for i, (mask, name, c) in enumerate(zip([t1_m, t2_m, t3_m], ['T1', 'T2', 'T3'], colors)):
    ax.hist(beta2_all[mask], bins=30, alpha=0.5, color=c, label=name, density=True)
ax.set_xlabel('beta2 (mixing coefficient)')
ax.set_ylabel('Density')
ax.set_title('beta2 Distribution by Tier')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Per-tier performance scatter
ax = axes[2]
ax.bar(['T1', 'T2', 'T3'], [A_T1, A_T2, A_T3], color=colors, alpha=0.85, edgecolor='white')
ax.set_ylabel('Effective Area A')
ax.set_title('Per-Tier Effective Area')
for i, v in enumerate([A_T1, A_T2, A_T3]):
    ax.text(i, v + 10, str(v), ha='center', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q1_cstr_parameters.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: q1_cstr_parameters.png")

# ================================================================
# Fig 4: Time series comparison (first 300 points)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 5))
idx = np.arange(min(300, n))
ax.plot(idx, ntu[idx], 'b-', linewidth=0.6, label='True NTU', alpha=0.7)
ax.plot(idx, pred[idx], 'r-', linewidth=0.6, label='CSTR Predicted', alpha=0.7)
ax.fill_between(idx, ntu[idx], pred[idx], alpha=0.1, color='gray')
ax.set_xlabel('Time step (2h)')
ax.set_ylabel('NTU')
ax.set_title(f'CSTR 3-Tier: Time Series (first {min(300, n)} steps, R2={r2_all:.4f})')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q1_time_series.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: q1_time_series.png")

# ================================================================
# Save metrics table
# ================================================================
metrics = pd.DataFrame({
    'Tier': ['T1 (FILT<=0.05)', 'T2 (0.05-0.15)', 'T3 (FILT>0.15)', 'All'],
    'N': [int(t1_m.sum()), int(t2_m.sum()), int(t3_m.sum()), n],
    'R2': [f'{t1_r2:.4f}', f'{t2_r2:.4f}', f'{t3_r2:.4f}', f'{r2_all:.4f}'],
    'RMSE': [f'{t1_rmse:.4f}', f'{t2_rmse:.4f}', f'{t3_rmse:.4f}', f'{rmse_all:.4f}'],
    'A_eff': [A_T1, A_T2, A_T3, '-'],
})
metrics.to_csv(os.path.join(OUT_TAB, 'q1_cstr_metrics.csv'), index=False, encoding='utf-8-sig')
print(f"Saved: q1_cstr_metrics.csv")

print(f"\n{'='*60}")
print(f"  Q1 Final Results Summary")
print(f"{'='*60}")
print(f"  Model:       3-Tier CSTR + Balance Detector")
print(f"  A_T1={A_T1}, A_T2={A_T2}, A_same={A_same}, A_diff={A_diff}")
print(f"  Balance: sign((RL-RL_med)*(Q-Q_med)) > 0 ? A_same : A_diff")
print(f"  RL_med={RL_med:.1f}, Q_med={Q_med:.1f}")
print(f"  R2_all:      {r2_all:.4f}  (vs baseline {r2_base:.4f})")
print(f"  RMSE:        {rmse_all:.4f}")
print(f"{'='*60}")
print(f"  Tier breakdown:")
print(f"    T1 (FILT<=0.05, n={t1_m.sum():>4d}): R2={t1_r2:.4f}, RMSE={t1_rmse:.4f}")
print(f"    T2 (0.05-0.15,  n={t2_m.sum():>4d}): R2={t2_r2:.4f}, RMSE={t2_rmse:.4f}")
print(f"    T3 (FILT>0.15,  n={t3_m.sum():>4d}): R2={t3_r2:.4f}, RMSE={t3_rmse:.4f}")
print(f"{'='*60}")
print(f"  Figures: results/figures/q1_*.png (4 files)")
print(f"  Tables:  results/tables/q1_cstr_metrics.csv")
