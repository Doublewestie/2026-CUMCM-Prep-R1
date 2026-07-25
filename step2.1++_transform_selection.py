"""
step2.1++_transform_selection.py — FILT变换方案系统评测 v2
测试: identity, log, log1p, sqrt + AR阶数扩展 + 额外特征
"""

import os, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

from step2_shared import load_raw_filt_data
data = load_raw_filt_data()
for c in ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','ALUM','CW_WELL_LEVEL','TW_FLOW','NTU']:
    if c in data.columns:
        data[c] = data[c].fillna(data[c].median())

filt = data['FILT_NTU'].values.astype(float)
rw = data['RW_NTU'].values.astype(float)
ntu_vals = data['NTU'].values.astype(float)
rl = data['RIVER_LEVEL'].values.astype(float)
alum = data['ALUM'].values.astype(float)
cw = data['CW_WELL_LEVEL'].values.astype(float)
tw = data['TW_FLOW'].values.astype(float)
n = len(data)
EPS = 1e-3

def ar_lags(y, k):
    X = np.zeros((len(y), k))
    for lag in range(1, k+1):
        X[lag:, lag-1] = y[:-lag]
        X[:lag, lag-1] = y[0]
    return X

def roll_safe(arr, lag):
    s = np.roll(arr, lag)
    s[:lag] = arr[0]
    return s

# Test AR order expansion + additional features on FILT levels
print("=" * 100)
print(f"{'方案':<30s}  {'In-Sample':>22s}  {'TS-CV(5fold)':>22s}  {'80/20 Seq':>22s}")
print("=" * 100)

for name, make_features in [
    ("AR(6) baseline", lambda: ar_lags(filt, 6)),
    ("AR(12)", lambda: ar_lags(filt, 12)),
    ("AR(24)", lambda: ar_lags(filt, 24)),
    ("AR(6)+NTU(t-1)", lambda: np.column_stack([ar_lags(filt, 6), roll_safe(ntu_vals, 1)])),
    ("AR(6)+NTU(t-1,t-2)", lambda: np.column_stack([ar_lags(filt, 6), roll_safe(ntu_vals, 1), roll_safe(ntu_vals, 2)])),
    ("AR(6)+RW(t-2)", lambda: np.column_stack([ar_lags(filt, 6), roll_safe(rw, 2)])),
    ("AR(6)+ALUM(t-3)", lambda: np.column_stack([ar_lags(filt, 6), roll_safe(alum, 3)])),
    ("AR(6)+RL+TW", lambda: np.column_stack([ar_lags(filt, 6), rl, roll_safe(tw, 1)])),
    ("AR(6)+all_ext", lambda: np.column_stack([
        ar_lags(filt, 6),
        roll_safe(ntu_vals, 1), roll_safe(ntu_vals, 2),
        roll_safe(rw, 2),
        roll_safe(alum, 3),
        rl, roll_safe(tw, 1), roll_safe(cw, 1),
    ])),
]:
    X = make_features()
    if X.ndim == 1: X = X.reshape(-1, 1)
    start = max(6, 12) if '12' in name else 6
    if '24' in name: start = 24
    
    Xv, yv = X[start:], filt[start:]
    ok = ~(np.isnan(Xv).any(1) | np.isnan(yv) | np.isinf(yv))
    Xv, yv = Xv[ok], yv[ok]
    if len(Xv) < 20: continue
    
    # In-sample
    m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(Xv, yv)
    pred = m.predict(Xv)
    r2_is = r2_score(yv, pred); rmse_is = np.sqrt(mean_squared_error(yv, pred))
    
    # TS-CV
    tscv = TimeSeriesSplit(n_splits=5)
    r2_cv, rmse_cv = [], []
    for tr, va in tscv.split(Xv):
        mc = Ridge(alpha=m.alpha_).fit(Xv[tr], yv[tr])
        pc = mc.predict(Xv[va])
        r2_cv.append(r2_score(yv[va], pc))
        rmse_cv.append(np.sqrt(mean_squared_error(yv[va], pc)))
    
    # 80/20
    s80 = int(len(Xv)*0.8)
    m80 = Ridge(alpha=m.alpha_).fit(Xv[:s80], yv[:s80])
    p80 = m80.predict(Xv[s80:])
    r2_80 = r2_score(yv[s80:], p80); rmse_80 = np.sqrt(mean_squared_error(yv[s80:], p80))
    
    print(f"  {name:<30s}  InS R2={r2_is:.4f} RMSE={rmse_is:.4f}  "
          f"CV R2={np.mean(r2_cv):.4f}+-{np.std(r2_cv):.4f}  "
          f"80/20 R2={r2_80:.4f} RMSE={rmse_80:.4f}")
    
    if name == "AR(6)+all_ext":
        print(f"  {'(best alpha='+str(round(m.alpha_,2))+')':>30s}  {'N='+str(len(Xv)):>22s}")

# Also test: what if we use RidgeCV automatically optimizing alpha per fold?
print(f"\n--- RidgeCV with auto-alpha per fold (best configuration) ---")
X_best = np.column_stack([
    ar_lags(filt, 6),
    roll_safe(ntu_vals, 1), roll_safe(ntu_vals, 2),
    roll_safe(rw, 2),
    roll_safe(alum, 3),
    rl, roll_safe(tw, 1), roll_safe(cw, 1),
])
start = 6
Xv, yv = X_best[start:], filt[start:]
ok = ~(np.isnan(Xv).any(1) | np.isnan(yv) | np.isinf(yv))
Xv, yv = Xv[ok], yv[ok]
print(f"  Samples: {len(Xv)}, FILT std={yv.std():.4f}")

tscv = TimeSeriesSplit(n_splits=5)
r2_cv, rmse_cv = [], []
for tr, va in tscv.split(Xv):
    mc = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xv[tr], yv[tr])
    pc = mc.predict(Xv[va])
    r2_cv.append(r2_score(yv[va], pc))
    rmse_cv.append(np.sqrt(mean_squared_error(yv[va], pc)))
    print(f"  Fold alpha={mc.alpha_:.2f}: R2={r2_cv[-1]:.4f}, RMSE={rmse_cv[-1]:.4f}")
print(f"  CV mean: R2={np.mean(r2_cv):.4f}+-{np.std(r2_cv):.4f}, RMSE={np.mean(rmse_cv):.4f}")
