"""
step2.5+_logar_optimize.py — log-AR(6) 统一模型优化
测试: 两阶段校正 / 细α网格 / 模型平均 / AR(12) / Huber混合
"""

import os, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

from step2_shared import load_raw_filt_data
data = load_raw_filt_data()
n = len(data)
EPS = 1e-3

filt = data['FILT_NTU'].values.astype(float)
for v in [data['RIVER_LEVEL'].values.astype(float), data['ALUM'].values.astype(float)]:
    v[np.isnan(v)] = np.nanmedian(v)

log_filt = np.log(filt + EPS)
tscv = TimeSeriesSplit(n_splits=5)
ALPHAS_WIDE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

print("=" * 100)
print("  log-AR(6) 优化: 基准 + 6种优化策略")
print("=" * 100)

def ar_lags(y, k):
    X = np.zeros((len(y), k))
    for lag in range(1, k+1):
        X[lag:, lag-1] = y[:-lag]
        X[:lag, lag-1] = y[0]
    return X

# ================================================================
# 0. 基准
# ================================================================
print("\n--- 0. 基准: log-AR(6) RidgeCV ---")
X_ar6 = ar_lags(log_filt, 6)
start = 6
r2_0, rmse_0 = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    m = RidgeCV(alphas=ALPHAS_WIDE).fit(Xv[tr], yv[tr])
    p_f = np.exp(m.predict(Xv[va])) - EPS
    t_f = filt[start:][va]
    r2_0.append(r2_score(t_f, p_f))
    rmse_0.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  CV R2={np.mean(r2_0):.4f}+-{np.std(r2_0):.4f}, RMSE={np.mean(rmse_0):.4f}")

# ================================================================
# 1. 两阶段校正
# ================================================================
print("\n--- 1. 两阶段校正 ---")
r2_1, rmse_1 = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    # Stage 1: log-AR Ridge
    m1 = RidgeCV(alphas=ALPHAS_WIDE).fit(Xv[tr], yv[tr])
    p1_log = m1.predict(Xv[tr])
    p1_f = np.exp(p1_log) - EPS
    # Stage 2: a*p1_f + b in FILT space
    yraw_tr = filt[start:][tr]
    a = np.cov(p1_f, yraw_tr)[0, 1] / (np.var(p1_f) + 1e-10)
    b = np.mean(yraw_tr) - a * np.mean(p1_f)
    # Predict
    p1_log_va = m1.predict(Xv[va])
    p_f = a * (np.exp(p1_log_va) - EPS) + b
    t_f = filt[start:][va]
    r2_1.append(r2_score(t_f, p_f))
    rmse_1.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  CV R2={np.mean(r2_1):.4f}+-{np.std(r2_1):.4f}, RMSE={np.mean(rmse_1):.4f}")

# ================================================================
# 2a. 细α网格
# ================================================================
print("\n--- 2a. RidgeCV(细α=11) ---")
ALPHAS_FINE = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
r2_2a, rmse_2a = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    m = RidgeCV(alphas=ALPHAS_FINE).fit(Xv[tr], yv[tr])
    p_f = np.exp(m.predict(Xv[va])) - EPS
    t_f = filt[start:][va]
    r2_2a.append(r2_score(t_f, p_f))
    rmse_2a.append(np.sqrt(mean_squared_error(t_f, p_f)))
    if tr[0] == 0:  # first fold only
        print(f"    最佳α={m.alpha_:.4f}")
print(f"  CV R2={np.mean(r2_2a):.4f}+-{np.std(r2_2a):.4f}, RMSE={np.mean(rmse_2a):.4f}")

# ================================================================
# 2b-2d. ElasticNet
# ================================================================
for l1r, label in [(0.3, 'ElasticNet l1=0.3'), (0.5, 'ElasticNet l1=0.5'), (0.7, 'ElasticNet l1=0.7')]:
    r2_l, rmse_l = [], []
    for tr, va in tscv.split(np.arange(start, n)):
        Xv, yv = X_ar6[start:], log_filt[start:]
        m = ElasticNetCV(l1_ratio=l1r, alphas=[0.001, 0.01, 0.1, 1.0], max_iter=10000, cv=3).fit(Xv[tr], yv[tr])
        p_f = np.exp(m.predict(Xv[va])) - EPS
        t_f = filt[start:][va]
        r2_l.append(r2_score(t_f, p_f))
        rmse_l.append(np.sqrt(mean_squared_error(t_f, p_f)))
    
    if l1r == 0.3:
        r2_2b, rmse_2b = r2_l, rmse_l
    elif l1r == 0.5:
        r2_2c, rmse_2c = r2_l, rmse_l
    else:
        r2_2d, rmse_2d = r2_l, rmse_l
    print(f"  {label}: CV R2={np.mean(r2_l):.4f}+-{np.std(r2_l):.4f}, RMSE={np.mean(rmse_l):.4f}")

# ================================================================
# 3. 模型平均
# ================================================================
print("\n--- 3. 模型平均(Ridge+EN+Huber) ---")
r2_3, rmse_3 = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    m1 = RidgeCV(alphas=ALPHAS_WIDE).fit(Xv[tr], yv[tr])
    m2 = ElasticNetCV(l1_ratio=0.3, alphas=[0.001, 0.01, 0.1, 1.0], max_iter=10000, cv=3).fit(Xv[tr], yv[tr])
    m3 = HuberRegressor(alpha=0.1, max_iter=500).fit(Xv[tr], yv[tr])
    p_f = (np.exp(m1.predict(Xv[va])) - EPS + 
           np.exp(m2.predict(Xv[va])) - EPS + 
           np.exp(m3.predict(Xv[va])) - EPS) / 3
    t_f = filt[start:][va]
    r2_3.append(r2_score(t_f, p_f))
    rmse_3.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  CV R2={np.mean(r2_3):.4f}+-{np.std(r2_3):.4f}, RMSE={np.mean(rmse_3):.4f}")

# ================================================================
# 4. AR(12)
# ================================================================
print("\n--- 4. log-AR(12) ---")
X_ar12 = ar_lags(log_filt, 12)
start12 = 12
r2_4, rmse_4 = [], []
for tr, va in tscv.split(np.arange(start12, n)):
    Xv, yv = X_ar12[start12:], log_filt[start12:]
    m = RidgeCV(alphas=ALPHAS_WIDE).fit(Xv[tr], yv[tr])
    p_f = np.exp(m.predict(Xv[va])) - EPS
    t_f = filt[start12:][va]
    r2_4.append(r2_score(t_f, p_f))
    rmse_4.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  CV R2={np.mean(r2_4):.4f}+-{np.std(r2_4):.4f}, RMSE={np.mean(rmse_4):.4f}")

# ================================================================
# 5. Huber残差修正
# ================================================================
print("\n--- 5. Huber残差修正 ---")
r2_5, rmse_5 = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    m_ridge = RidgeCV(alphas=ALPHAS_WIDE).fit(Xv[tr], yv[tr])
    resid_tr = yv[tr] - m_ridge.predict(Xv[tr])
    m_huber = HuberRegressor(alpha=0.01, max_iter=500).fit(Xv[tr], resid_tr)
    p_log = m_ridge.predict(Xv[va]) + m_huber.predict(Xv[va])
    p_f = np.exp(p_log) - EPS
    t_f = filt[start:][va]
    r2_5.append(r2_score(t_f, p_f))
    rmse_5.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  CV R2={np.mean(r2_5):.4f}+-{np.std(r2_5):.4f}, RMSE={np.mean(rmse_5):.4f}")

# ================================================================
# 6. 最佳组合: 两阶段 + 细α + 模型平均
# ================================================================
print("\n--- 6. 最佳组合(全部) ---")
r2_6, rmse_6 = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    # 三个基模型
    m1 = RidgeCV(alphas=ALPHAS_FINE).fit(Xv[tr], yv[tr])
    m2 = ElasticNetCV(l1_ratio=0.3, alphas=[0.001, 0.01, 0.1, 1.0], max_iter=10000, cv=3).fit(Xv[tr], yv[tr])
    m3 = HuberRegressor(alpha=0.1, max_iter=500).fit(Xv[tr], yv[tr])
    
    # 两阶段校正 (在训练集上学 尺缩系数)
    p1_tr = np.exp(m1.predict(Xv[tr])) - EPS
    p2_tr = np.exp(m2.predict(Xv[tr])) - EPS
    p3_tr = np.exp(m3.predict(Xv[tr])) - EPS
    avg_tr = (p1_tr + p2_tr + p3_tr) / 3
    yraw_tr = filt[start:][tr]
    a = np.cov(avg_tr, yraw_tr)[0, 1] / (np.var(avg_tr) + 1e-10)
    b = np.mean(yraw_tr) - a * np.mean(avg_tr)
    
    # 预测
    p_f = a * ((np.exp(m1.predict(Xv[va])) - EPS + 
                np.exp(m2.predict(Xv[va])) - EPS + 
                np.exp(m3.predict(Xv[va])) - EPS) / 3) + b
    t_f = filt[start:][va]
    r2_6.append(r2_score(t_f, p_f))
    rmse_6.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  CV R2={np.mean(r2_6):.4f}+-{np.std(r2_6):.4f}, RMSE={np.mean(rmse_6):.4f}")

# ================================================================
# 汇总
# ================================================================
print("\n" + "=" * 100)
print(f"{'方案':<35s}  {'CV R2':>12s}  {'RMSE':>8s}")
print("-" * 60)
results = [
    ("0. 基准 log-AR(6) RidgeCV", r2_0, rmse_0),
    ("1. 两阶段校正", r2_1, rmse_1),
    ("2a. RidgeCV(细α=16)", r2_2a, rmse_2a),
    ("2b. ElasticNet l1=0.3", r2_2b, rmse_2b),
    ("2c. ElasticNet l1=0.5", r2_2c, rmse_2c),
    ("2d. ElasticNet l1=0.7", r2_2d, rmse_2d),
    ("3. 模型平均(R+E+H)", r2_3, rmse_3),
    ("4. log-AR(12)", r2_4, rmse_4),
    ("5. Huber残差修正", r2_5, rmse_5),
    ("6. 最佳组合(全部)", r2_6, rmse_6),
]
for name, rr, rms in results:
    print(f"  {name:<35s}  {np.mean(rr):>6.4f}±{np.std(rr):<5.4f}  {np.mean(rms):>8.4f}")
print("=" * 100)

best_idx = np.argmax([np.mean(r) for _, r, _ in results])
best_name, best_r, best_rmse = results[best_idx]
print(f"\n  最佳方案: {best_name} (R2={np.mean(best_r):.4f}, RMSE={np.mean(best_rmse):.4f})")
if np.mean(best_r) >= 0.70:
    print(f"  ✅ 超过 0.70!")
else:
    diff = 0.70 - np.mean(best_r)
    print(f"  ❌ 未达到 0.70 (差 {diff:.4f})")
