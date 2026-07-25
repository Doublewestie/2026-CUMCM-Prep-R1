"""
q2_two_tier_verify.py — Complete verification of the 2-tier Q2 framework
========================================================================
Tier 1 (Normal): empirical distribution when no extreme warning
Tier 2 (Extreme): AR(6) prediction when extreme warning triggered
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import json, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', '2025')

# ============================================================
# 1. LOAD
# ============================================================
data_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
data_dir = os.path.join(DATA_DIR, data_dirs[0])
FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])
RENAME = {
    'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU',
    'R/W CLR':'RW_CLR','R/W PH':'RW_PH','FILT. NTU':'FILT_NTU',
    'C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW','ALUM':'ALUM','NTU':'NTU',
}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU',
            'CW_WELL_LEVEL','TW_FLOW','ALUM','NTU']
dfs = []
for f in FILES:
    fp = os.path.join(data_dir, f)
    df = pd.read_excel(fp, skiprows=1 if 'Jan' in f else 0)
    df.rename(columns={k:v for k,v in RENAME.items() if k in df.columns}, inplace=True)
    for c in NUM_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=['FILT_NTU']).reset_index(drop=True)
for c in ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','ALUM']:
    if c in data.columns:
        med = data[c].median()
        data[c] = data[c].fillna(med if not pd.isna(med) else 0)

n = len(data)
FILT = data['FILT_NTU'].values
print(f"N={n}")
print()

# ============================================================
# 2. Define extreme warning condition
# ============================================================
EXTREME_NTU_THR = np.percentile(data['RW_NTU'].dropna(), 95)   # P95
EXTREME_RIVER_THR = np.percentile(data['RIVER_LEVEL'].dropna(), 97)  # P97

data['extreme'] = ((data['RW_NTU'] > EXTREME_NTU_THR) | (data['RIVER_LEVEL'] > EXTREME_RIVER_THR)).astype(int)
n_extreme = data['extreme'].sum()
n_normal = n - n_extreme
print(f"Extreme threshold: RW_NTU > {EXTREME_NTU_THR:.1f} or RIVER > {EXTREME_RIVER_THR:.1f}")
print(f"Normal: {n_normal} ({n_normal/n*100:.1f}%)   Extreme: {n_extreme} ({n_extreme/n*100:.1f}%)")

# ============================================================
# VERIFY A: FILT distribution in normal (non-extreme) conditions
# ============================================================
print(f"\n{'='*80}")
print(f"  VERIFY A: FILT distribution under NORMAL conditions")
print(f"  (no extreme warning triggered)")
print(f"{'='*80}")

normal_filt = FILT[data['extreme'] == 0]
print(f"  N = {len(normal_filt)}")
print(f"  Mean = {normal_filt.mean():.4f}")
print(f"  Std  = {normal_filt.std():.4f}")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:>2d} = {np.percentile(normal_filt, p):.4f}")
print(f"  FILT <= 0.03: {(normal_filt <= 0.03).mean():.1%}")
print(f"  FILT 0.03-0.05: {((normal_filt > 0.03) & (normal_filt <= 0.05)).mean():.1%}")
print(f"  FILT 0.05-0.08: {((normal_filt > 0.05) & (normal_filt <= 0.08)).mean():.1%}")
print(f"  FILT 0.08-0.15: {((normal_filt > 0.08) & (normal_filt <= 0.15)).mean():.1%}")
print(f"  FILT > 0.15 (T3): {(normal_filt > 0.15).mean():.1%}")
print(f"  FILT > 0.08:      {(normal_filt > 0.08).mean():.1%}")

# Test the proposed range
print(f"\n  Proposed normal range: [0.03, 0.08]")
in_range = ((normal_filt >= 0.03) & (normal_filt <= 0.08)).mean()
below = (normal_filt < 0.03).mean()
above = (normal_filt > 0.08).mean()
print(f"    Within range:  {in_range:.1%}")
print(f"    Below 0.03:    {below:.1%}")
print(f"    Above 0.08:    {above:.1%}")

# Also check: what about a wider range?
for lo, hi, label in [(0.02, 0.08, '0.02-0.08'), (0.02, 0.10, '0.02-0.10'), (0.03, 0.12, '0.03-0.12')]:
    pct = ((normal_filt >= lo) & (normal_filt <= hi)).mean()
    print(f"    Range [{label}]: covers {pct:.1%}")

# ============================================================
# VERIFY B: AR(6) in extreme conditions
# ============================================================
print(f"\n{'='*80}")
print(f"  VERIFY B: AR(6) performance by zone")
print(f"{'='*80}")

# Build lags
ar_cols = []
for lag in range(1, 7):
    data[f'FILT_lag{lag}'] = data['FILT_NTU'].shift(lag)
    ar_cols.append(f'FILT_lag{lag}')
data = data.dropna(subset=ar_cols).reset_index(drop=True)
n = len(data)
FILT = data['FILT_NTU'].values
extreme = data['extreme'].values

# Split by zone
normal_mask = extreme == 0
extreme_mask = extreme == 1
normal_filt = FILT[normal_mask]

# Global AR(6)
X_ar = data[ar_cols].values
X_aug = np.column_stack([np.ones(n), X_ar])
theta_global = np.linalg.lstsq(X_aug, FILT, rcond=None)[0]
pred_global = X_aug @ theta_global
rmse_global = np.sqrt(mean_squared_error(FILT, pred_global))
r2_global = r2_score(FILT, pred_global)

# Normal zone AR(6) - trained on normal only
X_norm = X_aug[normal_mask]; y_norm = FILT[normal_mask]
theta_norm = np.linalg.lstsq(X_norm, y_norm, rcond=None)[0]
pred_norm_all = X_aug @ theta_norm  # applied everywhere

# Extreme zone AR(6) - trained on extreme only
X_ext = X_aug[extreme_mask]; y_ext = FILT[extreme_mask]
if extreme_mask.sum() > 20:
    theta_ext = np.linalg.lstsq(X_ext, y_ext, rcond=None)[0]
    pred_ext_all = X_aug @ theta_ext
else:
    theta_ext = theta_global
    pred_ext_all = pred_global

print(f"  {'Model':<35} {'RMSE':>8} {'R2':>8} {'N':>6}")
print(f"  {'-'*60}")
print(f"  {'Global AR(6) all data':<35} {rmse_global:>8.4f} {r2_global:>8.4f} {n:>6d}")

# Zone-specific evaluation
for name, mask, pred_func in [
    ('Global on normal', normal_mask, pred_global[normal_mask]),
    ('Global on extreme', extreme_mask, pred_global[extreme_mask]),
    ('Zone-AR on normal', normal_mask, pred_norm_all[normal_mask]),
    ('Zone-AR on extreme', extreme_mask, pred_ext_all[extreme_mask]),
]:
    if mask.sum() < 10: continue
    rmse = np.sqrt(mean_squared_error(FILT[mask], pred_func))
    r2 = r2_score(FILT[mask], pred_func)
    print(f"  {name:<35} {rmse:>8.4f} {r2:>8.4f} {mask.sum():>6d}")

# === Delta-FILT evaluation (corrected metric: predicting CHANGES, not levels) ===
print(f"\n  {'─'*60}")
print(f"  Corrected metric: predicting ΔFILT = FILT(t) - FILT(t-1)")
print(f"  {'─'*60}")

delta_filt = np.diff(FILT, prepend=FILT[0])
delta_ar_feats = data[ar_cols].values
# AR(6) on delta
X_delta = np.column_stack([np.ones(n), delta_ar_feats])
theta_delta = np.linalg.lstsq(X_delta, delta_filt, rcond=None)[0]
pred_delta = X_delta @ theta_delta
rmse_delta = np.sqrt(mean_squared_error(delta_filt, pred_delta))
r2_delta = r2_score(delta_filt, pred_delta)

# Separate by zone
for mask, name in [(normal_mask, 'Normal zone'), (extreme_mask, 'Extreme zone')]:
    if mask.sum() < 10: continue
    delta_seg = delta_filt[mask]
    pred_seg = pred_delta[mask]
    r = np.sqrt(mean_squared_error(delta_seg, pred_seg))
    r2 = r2_score(delta_seg, pred_seg)
    print(f"  {'AR(6) on ' + name:<35} {r:>8.4f} {r2:>8.4f} {mask.sum():>6d}")

# Zone-specific AR(6) on delta
for mask, name in [(normal_mask, 'Normal zone ΔAR'), (extreme_mask, 'Extreme zone ΔAR')]:
    if mask.sum() < 10: continue
    X_delta_zone = np.column_stack([np.ones(mask.sum()), delta_ar_feats[mask]])
    y_delta_zone = delta_filt[mask]
    theta_zone = np.linalg.lstsq(X_delta_zone, y_delta_zone, rcond=None)[0]
    pred_zone = X_delta_zone @ theta_zone
    r = np.sqrt(mean_squared_error(y_delta_zone, pred_zone))
    r2 = r2_score(y_delta_zone, pred_zone)
    print(f"  {name:<35} {r:>8.4f} {r2:>8.4f} {mask.sum():>6d}")

# ============================================================
# VERIFY C: 2-tier combined error
# ============================================================
print(f"\n{'='*80}")
print(f"  VERIFY C: 2-tier combined vs global AR(6)")
print(f"{'='*80}")

# 2-tier: normal uses empirical sampling, extreme uses zone AR
np.random.seed(42)
pred_2tier = np.zeros(n)
for i in range(n):
    if extreme[i] == 0:
        # Empirical sampling from normal FILT distribution
        pred_2tier[i] = np.random.choice(normal_filt, size=1)[0]
    else:
        # Zone AR(6) for extreme
        pred_2tier[i] = pred_ext_all[i]

rmse_2tier = np.sqrt(mean_squared_error(FILT, pred_2tier))
r2_2tier = r2_score(FILT, pred_2tier)

print(f"  Global AR(6):       RMSE={rmse_global:.4f}  R2={r2_global:.4f}")
print(f"  2-Tier (emp+extAR): RMSE={rmse_2tier:.4f}  R2={r2_2tier:.4f}")
delta = rmse_2tier - rmse_global
print(f"  Delta: {delta:+.4f} ({'improves' if delta < 0 else 'worse than'} baseline)")

# Also try: AR(6) on both zones (2 separate AR models)
pred_2ar = np.zeros(n)
pred_2ar[normal_mask] = pred_norm_all[normal_mask]
pred_2ar[extreme_mask] = pred_ext_all[extreme_mask]
rmse_2ar = np.sqrt(mean_squared_error(FILT, pred_2ar))
r2_2ar = r2_score(FILT, pred_2ar)
print(f"  2-AR (normal+extreme): RMSE={rmse_2ar:.4f}  R2={r2_2ar:.4f}")

# ============================================================
# VERIFY D: τ-aligned AR(6) in extreme zone
# ============================================================
print(f"\n{'='*80}")
print(f"  VERIFY D: τ-aligned AR(6) in extreme zone")
print(f"{'='*80}")

data['NTU_tau2'] = data['RW_NTU'].shift(2).fillna(data['RW_NTU'].median())
data['RIVER_tau1'] = data['RIVER_LEVEL'].shift(1).fillna(data['RIVER_LEVEL'].median())
data['ALUM_tau2'] = data['ALUM'].shift(2).fillna(data['ALUM'].median())
data['LOAD_tau'] = data['NTU_tau2'] * data['RW_CLR']

tau_feats = ['FILT_lag1','FILT_lag2','FILT_lag3','FILT_lag4','FILT_lag5','FILT_lag6',
             'NTU_tau2','RIVER_tau1','ALUM_tau2','LOAD_tau']
data = data.dropna(subset=tau_feats).reset_index(drop=True)
n2 = len(data)
FILT2 = data['FILT_NTU'].values
extreme2 = data['extreme'].values

ext_mask2 = extreme2 == 1
if ext_mask2.sum() > 20:
    # Plain AR(6) on extreme
    X_ar_ext = np.column_stack([np.ones(ext_mask2.sum())] + [data.loc[ext_mask2, f'FILT_lag{lag}'].values for lag in range(1,7)])
    X_ar_ext = np.column_stack([np.ones(ext_mask2.sum()), data[ar_cols].values[ext_mask2]])
    y_ext2 = FILT2[ext_mask2]
    theta_ar = np.linalg.lstsq(X_ar_ext, y_ext2, rcond=None)[0]
    pred_ar_ext = X_ar_ext @ theta_ar
    rmse_ar_ext = np.sqrt(mean_squared_error(y_ext2, pred_ar_ext))
    r2_ar_ext = r2_score(y_ext2, pred_ar_ext)
    
    # AR(6) + tau features on extreme
    X_tau_ext = np.column_stack([
        np.ones(ext_mask2.sum()),
        data.loc[ext_mask2, ar_cols].values,
        data.loc[ext_mask2, ['NTU_tau2','RIVER_tau1','ALUM_tau2','LOAD_tau']].values,
    ])
    theta_tau = np.linalg.lstsq(X_tau_ext, y_ext2, rcond=None)[0]
    pred_tau_ext = X_tau_ext @ theta_tau
    rmse_tau_ext = np.sqrt(mean_squared_error(y_ext2, pred_tau_ext))
    r2_tau_ext = r2_score(y_ext2, pred_tau_ext)
    
    print(f"  Extreme samples: {ext_mask2.sum()}")
    print(f"  {'Model':<35} {'RMSE':>8} {'R2':>8}")
    print(f"  {'-'*55}")
    print(f"  {'AR(6) on extreme':<35} {rmse_ar_ext:>8.4f} {r2_ar_ext:>8.4f}")
    print(f"  {'AR(6)+tau features':<35} {rmse_tau_ext:>8.4f} {r2_tau_ext:>8.4f}")
    delta = rmse_tau_ext - rmse_ar_ext
    print(f"  {'Tau improvement':<35} {delta:>+8.4f} {r2_tau_ext - r2_ar_ext:>+8.4f}")
else:
    print(f"  Too few extreme samples: {ext_mask2.sum()}")

# ============================================================
# 5-fold CV of the full 2-tier model
# ============================================================
print(f"\n{'='*80}")
print(f"  CV of 2-TIER MODEL (5-fold TimeSeriesSplit)")
print(f"{'='*80}")

tscv = TimeSeriesSplit(n_splits=5)
cv_global = []; cv_2tier = []; cv_2ar = []
ar_values = data[ar_cols].values

for fold, (tr, va) in enumerate(tscv.split(data)):
    y_tr, y_va = FILT2[tr], FILT2[va]
    ext_tr = extreme2[tr]
    ext_va = extreme2[va]
    n_tr = len(tr); n_va = len(va)
    
    # Global AR on training
    X_tr_g = np.column_stack([np.ones(n_tr), ar_values[tr]])
    X_va_g = np.column_stack([np.ones(n_va), ar_values[va]])
    theta_g = np.linalg.lstsq(X_tr_g, y_tr, rcond=None)[0]
    pred_g = X_va_g @ theta_g
    cv_global.append(np.sqrt(mean_squared_error(y_va, pred_g)))
    
    # Zone ARs on training
    tr_norm = tr[ext_tr == 0]; tr_ext = tr[ext_tr == 1]
    if len(tr_ext) > 10:
        X_tr_ext = np.column_stack([np.ones(len(tr_ext)), ar_values[tr_ext]])
        theta_ext_cv = np.linalg.lstsq(X_tr_ext, y_tr[ext_tr == 1], rcond=None)[0]
    else:
        theta_ext_cv = theta_g
    if len(tr_norm) > 10:
        X_tr_norm = np.column_stack([np.ones(len(tr_norm)), ar_values[tr_norm]])
        theta_norm_cv = np.linalg.lstsq(X_tr_norm, y_tr[ext_tr == 0], rcond=None)[0]
    else:
        theta_norm_cv = theta_g
    
    # Apply zone ARs
    pred_2ar = np.zeros(n_va)
    for j in range(n_va):
        Xj = np.hstack([[1.0], ar_values[va[j]]])  # shape (7,)
        if ext_va[j] == 0:
            pred_2ar[j] = np.dot(Xj, theta_norm_cv)
        else:
            pred_2ar[j] = np.dot(Xj, theta_ext_cv)
    cv_2ar.append(np.sqrt(mean_squared_error(y_va, pred_2ar)))
    
    # 2-tier: normal = empirical from training normal
    normal_vals = y_tr[ext_tr == 0]
    pred_2t = np.zeros(n_va)
    np.random.seed(42 + fold)
    for j in range(n_va):
        if ext_va[j] == 0:
            pred_2t[j] = np.random.choice(normal_vals)
        else:
            Xj = np.hstack([[1.0], ar_values[va[j]]])
            pred_2t[j] = np.dot(Xj, theta_ext_cv)
    cv_2tier.append(np.sqrt(mean_squared_error(y_va, pred_2t)))

print(f"  {'Model':<35} {'CV_RMSE_mean':>12} {'CV_RMSE_std':>12}")
print(f"  {'-'*60}")
print(f"  {'Global AR(6)':<35} {np.mean(cv_global):>12.4f} {np.std(cv_global):>12.4f}")
print(f"  {'2-Tier (emp+extAR)':<35} {np.mean(cv_2tier):>12.4f} {np.std(cv_2tier):>12.4f}")
print(f"  {'2-AR (normal+ext)':<35} {np.mean(cv_2ar):>12.4f} {np.std(cv_2ar):>12.4f}")

print(f"\n{'='*80}")
print(f"  SAVING RESULTS")
print(f"{'='*80}")

out = {
    'n_total': int(n),
    'extreme_thresholds': {'rw_ntu_p95': float(EXTREME_NTU_THR), 'river_p97': float(EXTREME_RIVER_THR)},
    'tier_split': {'normal_n': int(n_normal), 'extreme_n': int(n_extreme)},
    'normal_filt_dist': {
        'p5': float(np.percentile(normal_filt, 5)),
        'p25': float(np.percentile(normal_filt, 25)),
        'p50': float(np.percentile(normal_filt, 50)),
        'p75': float(np.percentile(normal_filt, 75)),
        'p95': float(np.percentile(normal_filt, 95)),
        'pct_in_003_008': float(((normal_filt >= 0.03) & (normal_filt <= 0.08)).mean()),
        'pct_below_003': float((normal_filt < 0.03).mean()),
        'pct_above_008': float((normal_filt > 0.08).mean()),
    },
    'global_ar6': {'rmse': float(rmse_global), 'r2': float(r2_global)},
    'zone_ar6': {
        'ar_on_normal': {'rmse': float(np.sqrt(mean_squared_error(FILT[normal_mask], pred_norm_all[normal_mask]))), 'r2': float(r2_score(FILT[normal_mask], pred_norm_all[normal_mask]))},
        'ar_on_extreme': {'rmse': float(np.sqrt(mean_squared_error(FILT[extreme_mask], pred_ext_all[extreme_mask]))), 'r2': float(r2_score(FILT[extreme_mask], pred_ext_all[extreme_mask]))},
    },
    'two_tier': {
        'emp_plus_extar': {'rmse': float(rmse_2tier), 'r2': float(r2_2tier)},
        'two_ar': {'rmse': float(rmse_2ar), 'r2': float(r2_2ar)},
    },
    'tau_aligned': {
        'extreme_ar6': {'rmse': float(rmse_ar_ext), 'r2': float(r2_ar_ext)} if ext_mask2.sum() > 20 else {},
        'extreme_ar6_plus_tau': {'rmse': float(rmse_tau_ext), 'r2': float(r2_tau_ext)} if ext_mask2.sum() > 20 else {},
    },
    'cv': {
        'global_ar6': {'rmse_mean': float(np.mean(cv_global)), 'rmse_std': float(np.std(cv_global))},
        'two_tier_emp': {'rmse_mean': float(np.mean(cv_2tier)), 'rmse_std': float(np.std(cv_2tier))},
        'two_ar': {'rmse_mean': float(np.mean(cv_2ar)), 'rmse_std': float(np.std(cv_2ar))},
    },
}
out_path = os.path.join(BASE, 'output', 'q2_two_tier_result.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  Saved to {out_path}")
print(f"{'='*80}")
