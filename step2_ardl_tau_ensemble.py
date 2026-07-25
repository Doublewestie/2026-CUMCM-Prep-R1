"""
step2_ardl_tau_ensemble.py — 带τ时滞对齐的 ARDL + 模型平均 (Q2最终方案)
ARDL: log(FILT(t)) = AR(6) + RW_NTU(t-2) + ALUM(t-3) + RW_FLOW(t-1) 
                      + RW_PH(t-1) + RIVER_LEVEL + TW_FLOW(t-1) + seasonal
模型: RidgeCV + ElasticNet + Huber (3模型平均)
评估: 5-fold TS-CV, FILT空间 R2/RMSE
"""

import os, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV, ElasticNetCV, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

BASE = r'C:\Users\lenovo\2026-CUMCM-Prep-R1'
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

# 提取数据
filt = data['FILT_NTU'].values.astype(float)
rw_ntu = data['RW_NTU'].values.astype(float)
alum = data['ALUM'].values.astype(float)
rw_flow = data['RW_FLOW'].values.astype(float)
rw_ph = data['RW_PH'].values.astype(float)
river_lv = data['RIVER_LEVEL'].values.astype(float)
tw_flow = data['TW_FLOW'].values.astype(float)

# 填充NaN（用中位数）
for v in [rw_ntu, alum, rw_flow, rw_ph, river_lv, tw_flow]:
    v[np.isnan(v)] = np.nanmedian(v)

# 时间编码
hour = data['TIME'].values.astype(float)
hour = (hour // 100).astype(int) % 24
# 月份（按月数据拼接顺序判断，1-12）
# 使用文件名顺序
month = np.zeros(n, dtype=int)
cum = 0
for fi, fname in enumerate(FILES):
    fp = os.path.join(raw_dir, fname)
    tmp = pd.read_excel(fp, skiprows=1 if 'Jan' in fname else 0)
    tmp.rename(columns={k:v for k,v in RENAME.items() if k in tmp.columns}, inplace=True)
    newcols = []
    for c in tmp.columns:
        if isinstance(c, str): newcols.append(c.strip().replace('.','_').replace(' ','_'))
        else: newcols.append(str(c))
    tmp.columns = newcols
    if 'FILT_NTU' in tmp.columns:
        n_m = tmp['FILT_NTU'].notna().sum()
    else:
        n_m = len(tmp)
    month[cum:cum + n_m] = fi % 12 + 1
    cum += n_m

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
mon_sin = np.sin(2 * np.pi * month / 12)
mon_cos = np.cos(2 * np.pi * month / 12)

log_filt = np.log(filt + EPS)
tscv = TimeSeriesSplit(n_splits=5)
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# ================================================================
# 常用函数
# ================================================================
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

# ================================================================
# 特征构造
# ================================================================
# 基准: AR(6) on log(FILT)
X_ar6 = ar_lags(log_filt, 6)

# ARDL: AR(6) + τ对齐外部变量 + 时间编码
X_ardl = np.column_stack([
    X_ar6,                           # AR(6) on log(FILT)
    roll_safe(np.log(rw_ntu + 1e-3), 2),   # RW_NTU(t-2)  τ=2
    roll_safe(alum, 3),              # ALUM(t-3)     τ=3
    roll_safe(rw_flow, 1),           # RW_FLOW(t-1)  τ=1
    roll_safe(rw_ph, 1),             # RW_PH(t-1)    τ=1
    river_lv,                        # RIVER_LEVEL(t)
    roll_safe(tw_flow, 1),           # TW_FLOW(t-1)
    hour_sin, hour_cos,              # 日内周期
    mon_sin, mon_cos,                # 季节周期
])

# 填充NaN (roll_safe和log变换可能产生的)
X_ardl = np.nan_to_num(X_ardl, nan=0)

print("=" * 80)
print("  Q2 ARDL(τ) + 模型平均: 对比测试")
print("=" * 80)
print(f"\n  特征维度: {X_ardl.shape[1]}")
print(f"  AR(6) lags + 7外部变量 + 4季节编码")

# ================================================================
# A. 基准: 纯 log-AR(6) RidgeCV
# ================================================================
print("\n--- A. 基准: log-AR(6) RidgeCV ---")
start = 6
r2_a, rmse_a = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    m = RidgeCV(alphas=ALPHAS).fit(Xv[tr], yv[tr])
    p_f = np.exp(m.predict(Xv[va])) - EPS
    t_f = filt[start:][va]
    r2_a.append(r2_score(t_f, p_f))
    rmse_a.append(np.sqrt(mean_squared_error(t_f, p_f)))
    print(f"  α={m.alpha_:.1f}  Fold: R2={r2_a[-1]:.4f}  RMSE={rmse_a[-1]:.4f}")
print(f"  => CV R2={np.mean(r2_a):.4f}+-{np.std(r2_a):.4f}, RMSE={np.mean(rmse_a):.4f}")

# ================================================================
# B. ARDL(τ) + RidgeCV
# ================================================================
print("\n--- B. ARDL(τ) + RidgeCV ---")
r2_b, rmse_b = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ardl[start:], log_filt[start:]
    m = RidgeCV(alphas=ALPHAS).fit(Xv[tr], yv[tr])
    p_f = np.exp(m.predict(Xv[va])) - EPS
    t_f = filt[start:][va]
    r2_b.append(r2_score(t_f, p_f))
    rmse_b.append(np.sqrt(mean_squared_error(t_f, p_f)))
    print(f"  α={m.alpha_:.1f}  Fold: R2={r2_b[-1]:.4f}  RMSE={rmse_b[-1]:.4f}")
print(f"  => CV R2={np.mean(r2_b):.4f}+-{np.std(r2_b):.4f}, RMSE={np.mean(rmse_b):.4f}")

# ================================================================
# C. ARDL(τ) + 模型平均(Ridge+EN+Huber)
# ================================================================
print("\n--- C. ARDL(τ) + 模型平均(Ridge+EN+Huber) ---")
r2_c, rmse_c = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ardl[start:], log_filt[start:]
    
    m1 = RidgeCV(alphas=ALPHAS).fit(Xv[tr], yv[tr])
    m2 = ElasticNetCV(l1_ratio=0.5, alphas=[0.001, 0.01, 0.1, 1.0], 
                      max_iter=10000, cv=3).fit(Xv[tr], yv[tr])
    m3 = HuberRegressor(alpha=0.1, max_iter=500).fit(Xv[tr], yv[tr])
    
    p1 = np.exp(m1.predict(Xv[va])) - EPS
    p2 = np.exp(m2.predict(Xv[va])) - EPS
    p3 = np.exp(m3.predict(Xv[va])) - EPS
    p_f = (p1 + p2 + p3) / 3
    
    t_f = filt[start:][va]
    r2_c.append(r2_score(t_f, p_f))
    rmse_c.append(np.sqrt(mean_squared_error(t_f, p_f)))
    print(f"  Fold: R2={r2_c[-1]:.4f}  RMSE={rmse_c[-1]:.4f}")
print(f"  => CV R2={np.mean(r2_c):.4f}+-{np.std(r2_c):.4f}, RMSE={np.mean(rmse_c):.4f}")

# ================================================================
# D. 纯 AR(6) + 模型平均 (去掉外部变量, 只保留AR+seasonal)
# ================================================================
print("\n--- D. log-AR(6) + 模型平均(无外部变量) ---")
r2_d, rmse_d = [], []
for tr, va in tscv.split(np.arange(start, n)):
    Xv, yv = X_ar6[start:], log_filt[start:]
    m1 = RidgeCV(alphas=ALPHAS).fit(Xv[tr], yv[tr])
    m2 = ElasticNetCV(l1_ratio=0.5, alphas=[0.001, 0.01, 0.1, 1.0], 
                      max_iter=10000, cv=3).fit(Xv[tr], yv[tr])
    m3 = HuberRegressor(alpha=0.1, max_iter=500).fit(Xv[tr], yv[tr])
    p_f = (np.exp(m1.predict(Xv[va])) - EPS +
           np.exp(m2.predict(Xv[va])) - EPS +
           np.exp(m3.predict(Xv[va])) - EPS) / 3
    t_f = filt[start:][va]
    r2_d.append(r2_score(t_f, p_f))
    rmse_d.append(np.sqrt(mean_squared_error(t_f, p_f)))
print(f"  => CV R2={np.mean(r2_d):.4f}+-{np.std(r2_d):.4f}, RMSE={np.mean(rmse_d):.4f}")

# ================================================================
# 汇总
# ================================================================
print("\n" + "=" * 80)
print(f"{'方案':<40s}  {'CV R2':>12s}  {'RMSE':>8s}")
print("-" * 65)
print(f"{'A. 基准 log-AR(6) RidgeCV':<40s}  {np.mean(r2_a):>6.4f}±{np.std(r2_a):<5.4f}  {np.mean(rmse_a):>8.4f}")
print(f"{'B. ARDL(τ) + RidgeCV':<40s}  {np.mean(r2_b):>6.4f}±{np.std(r2_b):<5.4f}  {np.mean(rmse_b):>8.4f}")
print(f"{'C. ARDL(τ) + 模型平均':<40s}  {np.mean(r2_c):>6.4f}±{np.std(r2_c):<5.4f}  {np.mean(rmse_c):>8.4f}")
print(f"{'D. log-AR(6) + 模型平均':<40s}  {np.mean(r2_d):>6.4f}±{np.std(r2_d):<5.4f}  {np.mean(rmse_d):>8.4f}")
print("=" * 80)

# τ重要性: 看Ridge系数
print("\n--- ARDL(τ) Ridge系数 (外部变量重要性) ---")
m_demo = RidgeCV(alphas=ALPHAS).fit(X_ardl[start:], log_filt[start:])
coef_names = ['ARlag1','ARlag2','ARlag3','ARlag4','ARlag5','ARlag6',
              'RW_NTU(t-2)','ALUM(t-3)','RW_FLOW(t-1)','RW_PH(t-1)',
              'RIVER_LEVEL','TW_FLOW(t-1)','hour_sin','hour_cos','mon_sin','mon_cos']
coefs = list(zip(coef_names, m_demo.coef_))
coefs_sorted = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)
for name, c in coefs_sorted:
    print(f"  {name:<20s}  {c:+.6f}")
