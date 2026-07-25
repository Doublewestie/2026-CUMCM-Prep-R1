"""
step2.6_binning_compare.py — 方案B(log空间分箱) vs 方案C(原始3-tier) vs 统一log-AR(6)基线
全部在TS-CV下评估，最终FILT空间对比R2/RMSE
分箱切换: 使用上一步真实FILT值(t-1)决定当前步(t)用哪个模型
"""

import os, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

from step2_shared import load_raw_filt_data
data = load_raw_filt_data()
n = len(data)
EPS = 1e-3

filt = data['FILT_NTU'].values.astype(float)
ntu_v = data['NTU'].values.astype(float)
rw_v = data['RW_NTU'].values.astype(float)
rw_clr = data['RW_CLR'].values.astype(float)
cw_v = data['CW_WELL_LEVEL'].values.astype(float)
tw_v = data['TW_FLOW'].values.astype(float)
rl_v = data['RIVER_LEVEL'].values.astype(float)
clr_v = data['CLR'].values.astype(float)
rw_flow_v = data['RW_FLOW'].values.astype(float)
alum_v = data['ALUM'].values.astype(float)

for v in [rl_v, alum_v, clr_v, rw_clr]:
    v[np.isnan(v)] = np.nanmedian(v)

RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

print("=" * 70)
print("  Q2 分箱方案对比: B(log空间) vs C(3-tier) vs 基线")
print("=" * 70)

def ar_lags(y, k):
    X = np.zeros((len(y), k))
    for lag in range(1, k+1):
        X[lag:, lag-1] = y[:-lag]
        X[:lag, lag-1] = y[0]
    return X

def roll_safe(arr, lag):
    s = np.roll(arr, lag); s[:lag] = arr[0]; return s

tscv = TimeSeriesSplit(n_splits=5)
log_filt = np.log(filt + EPS)

# Precompute feature matrices
X_arlog6 = ar_lags(log_filt, 6)          # AR(6) on log(FILT)
X_arlog3 = ar_lags(log_filt, 3)          # AR(3) on log(FILT)
X_arlev6 = ar_lags(filt, 6)              # AR(6) on FILT level
X_ext_t3 = np.column_stack([             # rich features for T3
    X_arlev6,
    roll_safe(ntu_v, 1), roll_safe(ntu_v, 2),
    roll_safe(rw_v, 2), roll_safe(rw_v, 3),
    roll_safe(alum_v, 3),
    rl_v, roll_safe(tw_v, 1), roll_safe(cw_v, 1),
    rw_clr, clr_v,
])
X_ext_l3 = np.column_stack([             # rich features for L3 (log space)
    X_arlog6,
    roll_safe(ntu_v, 1), roll_safe(ntu_v, 2),
    roll_safe(rw_v, 2),
    roll_safe(alum_v, 3),
    rl_v, roll_safe(tw_v, 1),
])
X_ext_t3 = np.nan_to_num(X_ext_t3, nan=0)
X_ext_l3 = np.nan_to_num(X_ext_l3, nan=0)

# ================================================================
# 基线: 统一 log-AR(6)
# ================================================================
print("\n--- 基线: 统一 log-AR(6) ---")
start = 6
r2_bs, rmse_bs = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_arlog6[start:], log_filt[start:]
    m = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xv[tr], yv[tr])
    p_ly = m.predict(Xv[va])
    p_f = np.exp(p_ly) - EPS
    t_f = filt[start:][va]
    r2_bs.append(r2_score(t_f, p_f))
    rmse_bs.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  CV R2={np.mean(r2_bs):.4f}+-{np.std(r2_bs):.4f}, RMSE={np.mean(rmse_bs):.4f}")

# ================================================================
# 方案B: log空间分箱
# ================================================================
print("\n" + "=" * 70)
print("  方案B: log(FILT)空间分箱")
print("=" * 70)

log_b1, log_b2 = -2.5, -1.9
print(f"  边界: L1<{log_b1}, L2={log_b1}~{log_b2}, L3>{log_b2}")
for nm, lo, hi in [("L1", -99, log_b1), ("L2", log_b1, log_b2), ("L3", log_b2, 99)]:
    m2 = (log_filt > lo) & (log_filt <= hi)
    print(f"  {nm}: n={m2.sum()}({m2.sum()/n*100:.0f}%) filt_m={np.median(filt[m2]):.4f}")

# Per-zone model definitions: (X_matrix, n_lags, alphas, use_log)
zone_models_b = {
    1: (X_arlog3, 3, [1.0, 10.0, 100.0], True),     # L1: log-AR(3) heavy ridge
    2: (X_arlog6, 6, [0.1, 1.0, 10.0], True),         # L2: log-AR(6) medium ridge
    3: (X_ext_l3, 6, [0.01, 0.1, 1.0], True),          # L3: log-AR(6)+ext weak ridge
}

def get_zone_b(lf_val):
    if lf_val < log_b1: return 1
    if lf_val < log_b2: return 2
    return 3

r2_b, rmse_b = [], []
for fold_idx, (tr, va) in enumerate(tscv.split(np.arange(start, n))):
    tr_full = tr + start
    va_full = va + start
    
    # Train zone-specific models
    models = {}
    for zid, (Xmat, nl, alphas, _) in zone_models_b.items():
        z_mask = np.zeros(n, dtype=bool)
        for t_idx in tr_full:
            if get_zone_b(log_filt[t_idx-1]) == zid:
                z_mask[t_idx] = True
        z_tr = z_mask.copy()
        z_tr[:nl] = False
        if z_tr.sum() >= 10:
            z_idx = np.where(z_tr)[0]
            models[zid] = RidgeCV(alphas=alphas).fit(Xmat[nl:][z_idx - nl], log_filt[nl:][z_idx])
    
    # Predict validation
    all_pred_f = np.zeros(len(va_full))
    all_true_f = np.zeros(len(va_full))
    
    for i, t_idx in enumerate(va_full):
        zid = get_zone_b(log_filt[t_idx-1])
        all_true_f[i] = filt[t_idx]
        
        if zid in models:
            Xmat, nl, _, use_log = zone_models_b[zid]
            x_va = Xmat[t_idx].reshape(1, -1)
            p_ly = models[zid].predict(x_va)[0]
            all_pred_f[i] = np.exp(p_ly) - EPS
        else:
            # fallback: global log-AR(6)
            m_fb = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_arlog6[start:][tr], log_filt[start:][tr])
            p_fb = m_fb.predict(X_arlog6[t_idx].reshape(1, -1))[0]
            all_pred_f[i] = np.exp(p_fb) - EPS
    
    r2_b.append(r2_score(all_true_f, all_pred_f))
    rmse_b.append(np.sqrt(mean_squared_error(all_true_f, all_pred_f)))
    print(f"  Fold {fold_idx}: R2={r2_b[-1]:.4f}, RMSE={rmse_b[-1]:.4f} (zones trained: {list(models.keys())})")

print(f"  方案B: CV R2={np.mean(r2_b):.4f}+-{np.std(r2_b):.4f}, RMSE={np.mean(rmse_b):.4f}")

# ================================================================
# 方案C: 原始FILT 3-tier
# ================================================================
print("\n" + "=" * 70)
print("  方案C: FILT原始3-tier")
print("=" * 70)

t1_thr, t2_thr = 0.05, 0.15
print(f"  边界: T1<={t1_thr}, T2={t1_thr}~{t2_thr}, T3>{t2_thr}")

t1_filt = filt[filt <= t1_thr]
t1_median = np.median(t1_filt)
print(f"  T1: n={len(t1_filt)} median={t1_median:.4f}")

# Zone models for C
zone_models_c = {
    1: (None, None, None, 'empirical'),   # T1: empirical median
    2: (X_arlog6, 6, [0.1, 1.0, 10.0], 'log'),  # T2: log-AR(6)
    3: (X_ext_t3, 12, [0.01, 0.1, 1.0], 'level'), # T3: level AR(6)+ext
}

def get_zone_c(f_val):
    if f_val <= t1_thr: return 1
    if f_val <= t2_thr: return 2
    return 3

r2_c, rmse_c = [], []
for fold_idx, (tr, va) in enumerate(tscv.split(np.arange(start, n))):
    tr_full = tr + start
    va_full = va + start
    
    # Train T2 and T3 models
    models_c = {}
    for zid in [2, 3]:
        Xmat, nl, alphas, mtype = zone_models_c[zid]
        z_mask = np.zeros(n, dtype=bool)
        for t_idx in tr_full:
            if get_zone_c(filt[t_idx-1]) == zid:
                z_mask[t_idx] = True
        z_tr = z_mask.copy()
        z_tr[:nl] = False
        z_tr_idx = np.where(z_tr)[0]
        if len(z_tr_idx) >= 10:
            if mtype == 'log':
                models_c[zid] = RidgeCV(alphas=alphas).fit(Xmat[nl:][z_tr_idx - nl], log_filt[nl:][z_tr_idx])
            else:
                models_c[zid] = RidgeCV(alphas=alphas).fit(Xmat[nl:][z_tr_idx - nl], filt[nl:][z_tr_idx])
    
    # Predict validation
    all_pred_c = np.zeros(len(va_full))
    all_true_c = np.zeros(len(va_full))
    
    for i, t_idx in enumerate(va_full):
        zid = get_zone_c(filt[t_idx-1])
        all_true_c[i] = filt[t_idx]
        
        if zid == 1:
            all_pred_c[i] = t1_median
        elif zid in models_c:
            Xmat, nl, _, mtype = zone_models_c[zid]
            x_va = Xmat[t_idx].reshape(1, -1)
            if mtype == 'log':
                p_ly = models_c[zid].predict(x_va)[0]
                all_pred_c[i] = np.exp(p_ly) - EPS
            else:
                all_pred_c[i] = models_c[zid].predict(x_va)[0]
        else:
            m_fb = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_arlog6[start:][tr], log_filt[start:][tr])
            p_fb = m_fb.predict(X_arlog6[t_idx].reshape(1, -1))[0]
            all_pred_c[i] = np.exp(p_fb) - EPS
    
    r2_c.append(r2_score(all_true_c, all_pred_c))
    rmse_c.append(np.sqrt(mean_squared_error(all_true_c, all_pred_c)))
    print(f"  Fold {fold_idx}: R2={r2_c[-1]:.4f}, RMSE={rmse_c[-1]:.4f} (models: T2={2 in models_c}, T3={3 in models_c})")

print(f"  方案C: CV R2={np.mean(r2_c):.4f}+-{np.std(r2_c):.4f}, RMSE={np.mean(rmse_c):.4f}")

# ================================================================
# 汇总
# ================================================================
print("\n" + "=" * 70)
print("  最终对比")
print("=" * 70)
print(f"{'方案':<35s}  {'CV R2':>12s}  {'RMSE':>8s}")
print("-" * 60)
print(f"{'基线: log-AR(6) 统一':<35s}  {np.mean(r2_bs):>6.4f}±{np.std(r2_bs):<5.4f}  {np.mean(rmse_bs):>8.4f}")
print(f"{'方案B: log空间分箱':<35s}  {np.mean(r2_b):>6.4f}±{np.std(r2_b):<5.4f}  {np.mean(rmse_b):>8.4f}")
print(f"{'方案C: 原始3-tier':<35s}  {np.mean(r2_c):>6.4f}±{np.std(r2_c):<5.4f}  {np.mean(rmse_c):>8.4f}")
print("=" * 70)
