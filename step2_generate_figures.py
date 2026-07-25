"""
step2_generate_figures.py — Generate Q2 final figures
Model: log(FILT+eps) AR(6) + RidgeCV
All labels in English
"""

import os, numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12})

BASE = r'C:\Users\lenovo\2026-CUMCM-Prep-R1'
OUT_FIG = os.path.join(BASE, 'results', 'figures')
OUT_TAB = os.path.join(BASE, 'results', 'tables')
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

# Load data
for d in os.listdir(os.path.join(BASE, 'data', '2025')):
    fp = os.path.join(BASE, 'data', '2025', d)
    if os.path.isdir(fp): raw_dir = fp; break

FILES = sorted([f for f in os.listdir(raw_dir) if f.endswith('.xlsx')])
RENAME = {'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU','R/W CLR':'RW_CLR','FILT. NTU':'FILT_NTU','C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW','ALUM':'ALUM','NTU':'NTU','R/W PH':'RW_PH'}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_WELL_LEVEL','TW_FLOW','ALUM','NTU']
data_all = []
for fname in FILES:
    fp = os.path.join(raw_dir, fname)
    dfm = pd.read_excel(fp, skiprows=1 if 'Jan' in fname else 0)
    dfm.rename(columns={k:v for k,v in RENAME.items() if k in dfm.columns}, inplace=True)
    newcols = []
    for c in dfm.columns:
        if isinstance(c, str): newcols.append(c.strip().replace('.','_').replace(' ','_'))
        else: newcols.append(str(c))
    dfm.columns = newcols
    for c in NUM_COLS:
        if c in dfm.columns: dfm[c] = pd.to_numeric(dfm[c], errors='coerce')
    data_all.append(dfm)
data = pd.concat(data_all, ignore_index=True)
data = data.dropna(subset=['FILT_NTU']).reset_index(drop=True)
n = len(data)
EPS = 1e-3

filt = data['FILT_NTU'].values.astype(float)
log_filt = np.log(filt + EPS)

def ar_lags(y, k):
    X = np.zeros((len(y), k))
    for lag in range(1, k+1):
        X[lag:, lag-1] = y[:-lag]
        X[:lag, lag-1] = y[0]
    return X

X_ar6 = ar_lags(log_filt, 6)

# Full model
m_all = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]).fit(X_ar6[6:], log_filt[6:])
p_all_log = m_all.predict(X_ar6)
p_all = np.exp(p_all_log) - EPS
r2_all = r2_score(filt[6:], p_all[6:])
rmse_all = np.sqrt(mean_squared_error(filt[6:], p_all[6:]))

# TS-CV (per-fold evaluation)
tscv = TimeSeriesSplit(n_splits=5)
p_cv = np.full(n - 6, np.nan)
r2_folds, rmse_folds = [], []
for tr, va in tscv.split(X_ar6[6:]):
    mc = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]).fit(X_ar6[6:][tr], log_filt[6:][tr])
    p_va = np.exp(mc.predict(X_ar6[6:][va])) - EPS
    t_va = filt[6:][va]
    r2_folds.append(r2_score(t_va, p_va))
    rmse_folds.append(np.sqrt(mean_squared_error(t_va, p_va)))
    p_cv[va] = p_va

r2_cv_mean, r2_cv_std = np.mean(r2_folds), np.std(r2_folds)
rmse_cv_mean = np.mean(rmse_folds)

# Also compute global (for scatter plot display only, not as metric)
p_cv_f = p_cv
cv_ok = ~np.isnan(p_cv_f)
residual = np.full(n - 6, np.nan)
residual[cv_ok] = filt[6:][cv_ok] - p_cv_f[cv_ok]

# Tier-specific metrics with per-fold correction
cv_mask = cv_ok
t1_idx = cv_mask & (filt[6:] <= 0.05)
t2_idx = cv_mask & ((filt[6:] > 0.05) & (filt[6:] <= 0.15))
t3_idx = cv_mask & (filt[6:] > 0.15)

print("=== Q2 Final Results ===")
print(f"In-sample:       R2={r2_all:.4f}, RMSE={rmse_all:.4f}")
print(f"TS-CV (per-fold): R2={r2_cv_mean:.4f}+-{r2_cv_std:.4f}, RMSE={rmse_cv_mean:.4f}")
print(f"  Folds: {[f'{r:.4f}' for r in r2_folds]}")
print(f"Alpha:      {m_all.alpha_:.2f}")

# ================================================================
# Fig 1: Pred vs Actual (scatter)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.scatter(filt[6:][cv_mask], p_cv_f[cv_mask], s=3, alpha=0.3, c='steelblue', edgecolors='none')
ax.plot([0, 10], [0, 10], 'r--', linewidth=1.5, alpha=0.7, label='y = x')
ax.set_xlabel('True FILT.NTU')
ax.set_ylabel('Predicted FILT.NTU')
ax.set_title(f'TS-CV Prediction (R2={r2_cv_mean:.4f}+-{r2_cv_std:.4f}, RMSE={rmse_cv_mean:.4f})')
ax.legend(loc='lower right')
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(filt[6:], p_all[6:], s=3, alpha=0.3, c='steelblue', edgecolors='none')
ax.plot([0, 10], [0, 10], 'r--', linewidth=1.5, alpha=0.7, label='y = x')
ax.set_xlabel('True FILT.NTU')
ax.set_ylabel('Predicted FILT.NTU')
ax.set_title(f'In-Sample Fit (R2={r2_all:.4f}, RMSE={rmse_all:.4f})')
ax.legend(loc='lower right')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q2_pred_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: q2_pred_vs_actual.png")

# ================================================================
# Fig 2: Time series + Residual diagnostics
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 2a: Time series (first 500 points)
ax = axes[0, 0]
idx = np.arange(min(500, n-6))
ok_idx = idx[cv_mask[idx]]
ax.plot(ok_idx, filt[6:][ok_idx], 'b-', linewidth=0.5, label='True FILT', alpha=0.7)
ax.plot(ok_idx, p_cv_f[ok_idx], 'r-', linewidth=0.5, label='Predicted FILT', alpha=0.7)
ax.set_xlabel('Time step (2h)')
ax.set_ylabel('FILT.NTU')
ax.set_title('TS-CV: True vs Predicted (first 500 pts)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2b: Residual distribution
ax = axes[0, 1]
res_plot = residual[cv_mask]
ax.hist(res_plot, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Residual (True - Predicted)')
ax.set_ylabel('Count')
ax.set_title(f'Residual Distribution (std={np.std(res_plot):.4f})')
ax.grid(True, alpha=0.3)

# 2c: QQ plot
ax = axes[0, 2]
from scipy import stats
stats.probplot(res_plot, dist='norm', plot=ax)
ax.get_lines()[0].set_markersize(2)
ax.get_lines()[0].set_color('steelblue')
ax.get_lines()[1].set_color('red')
ax.set_title('Q-Q Plot (vs Normal)')
ax.grid(True, alpha=0.3)

# 2d: Residual vs Predicted
ax = axes[1, 0]
ax.scatter(p_cv_f[cv_mask], res_plot, s=2, alpha=0.2, c='steelblue', edgecolors='none')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Predicted FILT.NTU')
ax.set_ylabel('Residual')
ax.set_title('Residuals vs Predicted')
ax.grid(True, alpha=0.3)

# 2e: Residual ACF
ax = axes[1, 1]
acf = np.correlate(res_plot - np.mean(res_plot), res_plot - np.mean(res_plot), mode='full')
acf = acf / acf.max()
acf = acf[len(acf)//2:]
ax.bar(range(min(30, len(acf))), acf[:min(30, len(acf))], width=0.6, color='steelblue', alpha=0.7)
ax.axhline(y=1.96/np.sqrt(len(res_plot)), color='r', linestyle=':', alpha=0.5)
ax.axhline(y=-1.96/np.sqrt(len(res_plot)), color='r', linestyle=':', alpha=0.5)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('Residual ACF (first 30 lags)')
ax.grid(True, alpha=0.3)

# 2f: Residual by month
ax = axes[1, 2]
month_vals = np.ones(n-6)
cum = 0
for fi, fname in enumerate(FILES):
    fp = os.path.join(raw_dir, fname)
    tmp = pd.read_excel(fp, skiprows=1 if 'Jan' in fname else 0)
    tmp.rename(columns={k:v for k,v in RENAME.items() if k in tmp.columns}, inplace=True)
    newcols2 = []
    for c in tmp.columns:
        if isinstance(c, str): newcols2.append(c.strip().replace('.','_').replace(' ','_'))
        else: newcols2.append(str(c))
    tmp.columns = newcols2
    nf = tmp['FILT_NTU'].notna().sum() if 'FILT_NTU' in tmp.columns else len(tmp)
    start_idx = max(6, cum + 6)
    end_idx = min(n, cum + nf)
    if start_idx < end_idx:
        month_vals[max(0, start_idx-6):end_idx-6] = fi % 12 + 1
    cum += nf
month_vals = month_vals[cv_mask]
month_res = pd.DataFrame({'month': month_vals.astype(int), 'residual': res_plot})
month_res.boxplot(column='residual', by='month', ax=ax, grid=False, patch_artist=True,
                  boxprops=dict(facecolor='steelblue', alpha=0.5))
ax.set_title('Residual by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Residual')
fig.suptitle('')

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q2_residual_diagnostics.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: q2_residual_diagnostics.png")

# ================================================================
# Fig 3: Feature importance (Ridge coefficients)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))
coefs = m_all.coef_
names = ['AR-lag-1', 'AR-lag-2', 'AR-lag-3', 'AR-lag-4', 'AR-lag-5', 'AR-lag-6']
colors = ['#2c7bb6' if c >= 0 else '#d7191c' for c in coefs]
bars = ax.barh(names, coefs, color=colors, alpha=0.8, edgecolor='white')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_xlabel('Ridge Coefficient')
ax.set_ylabel('Feature')
ax.set_title(f'AR(6) Ridge Coefficients (alpha={m_all.alpha_:.2f})')
ax.grid(True, alpha=0.3, axis='x')
ax.text(0.95, 0.95, f'$R^2={r2_cv_mean:.4f}$', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q2_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: q2_feature_importance.png")

# ================================================================
# Fig 4: Tier comparison bar chart
# ================================================================
t1_r2 = r2_score(filt[6:][t1_idx], p_cv_f[t1_idx]) if t1_idx.sum() > 5 else np.nan
t2_r2 = r2_score(filt[6:][t2_idx], p_cv_f[t2_idx]) if t2_idx.sum() > 5 else np.nan
t3_r2 = r2_score(filt[6:][t3_idx], p_cv_f[t3_idx]) if t3_idx.sum() > 5 else np.nan
t1_rmse = np.sqrt(mean_squared_error(filt[6:][t1_idx], p_cv_f[t1_idx])) if t1_idx.sum() > 5 else np.nan
t2_rmse = np.sqrt(mean_squared_error(filt[6:][t2_idx], p_cv_f[t2_idx])) if t2_idx.sum() > 5 else np.nan
t3_rmse = np.sqrt(mean_squared_error(filt[6:][t3_idx], p_cv_f[t3_idx])) if t3_idx.sum() > 5 else np.nan

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
tier_names = ['T1\n(FILT<=0.05)', 'T2\n(0.05-0.15)', 'T3\n(FILT>0.15)', 'All']

ax = axes[0]
r2s = [t1_r2, t2_r2, t3_r2, r2_cv_mean]
ax.bar(tier_names, r2s, color=['#2c7bb6', '#fdae61', '#d7191c', '#636363'], alpha=0.8, edgecolor='white')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_ylabel('R2')
ax.set_title('TS-CV R2 by FILT Tier')
for i, v in enumerate(r2s):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
rms = [t1_rmse, t2_rmse, t3_rmse, rmse_cv_mean]
ax.bar(tier_names, rms, color=['#2c7bb6', '#fdae61', '#d7191c', '#636363'], alpha=0.8, edgecolor='white')
ax.set_ylabel('RMSE')
ax.set_title('TS-CV RMSE by FILT Tier')
for i, v in enumerate(rms):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUT_FIG, 'q2_tier_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: q2_tier_comparison.png")

# ================================================================
# Save metrics table
# ================================================================
metrics = pd.DataFrame({
    'Tier': ['T1 (FILT<=0.05)', 'T2 (0.05-0.15)', 'T3 (FILT>0.15)', 'All (per-fold mean)'],
    'N': [t1_idx.sum(), t2_idx.sum(), t3_idx.sum(), n-6],
    'R2': [f'{t1_r2:.4f}', f'{t2_r2:.4f}', f'{t3_r2:.4f}', f'{r2_cv_mean:.4f}'],
    'RMSE': [f'{t1_rmse:.4f}', f'{t2_rmse:.4f}', f'{t3_rmse:.4f}', f'{rmse_cv_mean:.4f}'],
})
metrics.to_csv(os.path.join(OUT_TAB, 'q2_tier_metrics.csv'), index=False, encoding='utf-8-sig')
print(f"\nSaved: q2_tier_metrics.csv")

ar6_coefs = pd.DataFrame({'Feature': names, 'Coefficient': coefs})
ar6_coefs.to_csv(os.path.join(OUT_TAB, 'q2_ar6_coefficients.csv'), index=False, encoding='utf-8-sig')
print(f"Saved: q2_ar6_coefficients.csv")

# Final summary
print(f"\n{'='*60}")
print(f"  Q2 Final Results Summary")
print(f"{'='*60}")
print(f"  Model:      log(FILT+1e-3) AR(6) + RidgeCV(alpha={m_all.alpha_:.2f})")
print(f"  In-sample:       R2={r2_all:.4f}, RMSE={rmse_all:.4f}")
print(f"  TS-CV (per-fold): R2={r2_cv_mean:.4f}+-{r2_cv_std:.4f}, RMSE={rmse_cv_mean:.4f}")
print(f"{'='*60}")
print(f"  Tier (global CV predictions):")
print(f"    T1 (FILT<=0.05,   n={t1_idx.sum():>4d}): R2={t1_r2:.4f}, RMSE={t1_rmse:.4f}")
print(f"    T2 (0.05-0.15,    n={t2_idx.sum():>4d}): R2={t2_r2:.4f}, RMSE={t2_rmse:.4f}")
print(f"    T3 (FILT>0.15,    n={t3_idx.sum():>4d}): R2={t3_r2:.4f}, RMSE={t3_rmse:.4f}")
print(f"{'='*60}")
print(f"  Figures saved to: results/figures/")
print(f"  Tables saved to:  results/tables/")
