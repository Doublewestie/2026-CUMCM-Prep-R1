"""run Q1 ablation only"""
import sys, os
sys.path.insert(0, r"c:\Users\lenovo\Desktop\第二阶段-第一次\DW-0-0\Code")
os.chdir(sys.path[0])

import numpy as np, pandas as pd, warnings, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from step0_config import *

warnings.filterwarnings("ignore")

def boxcox_inverse(y_trans, lam):
    y_t = np.asarray(y_trans, dtype=np.float64).copy()
    if abs(lam) < 1e-6: return np.expm1(y_t)
    if lam < 0: y_t = np.minimum(y_t, 0.99 / abs(lam))
    else: y_t = np.maximum(y_t, -0.99 / lam)
    return (y_t * lam + 1) ** (1.0 / lam) - EPS

def classify_layers(feature_names):
    layers = {"L1":[],"L2":[],"L2+":[],"L3":[],"L4":[],"L5":[]}
    for i, name in enumerate(feature_names):
        n = str(name)
        if n in ["PI_load","GAMMA_alum","PSI_alum","OMEGA_night"]: layers["L5"].append(i); continue
        if "_lag" in n: layers["L3"].append(i); continue
        if any(s in n for s in ["_mean","_std","_max","_delta"]): layers["L4"].append(i); continue
        if n in ["FILT_sq","FILT_sqrt","FILT_cubert","neg_ln_eta","eta_sq","rw_ntu_sqrt","rw_ntu_log","alum_inv","alum_sqrt","tw_flow_log","dose_ratio_sq","dose_ratio_inv"]: layers["L2+"].append(i); continue
        if n in ["eta_coag","phi_alum","psi_hyd","hour_sin","hour_cos","day_sin","day_cos","is_weekend","is_night"]: layers["L2"].append(i); continue
        layers["L1"].append(i)
    return layers

X = np.load(OUT_X_ALL).astype(np.float64)
y = np.load(OUT_Y_ALL).astype(np.float64)
names = list(np.load(OUT_FEATURE_NAMES, allow_pickle=True))
lam = joblib.load(OUT_LAMBDA_NTU)

layers = classify_layers(names)

def gid(keys):
    idxs = []
    for k in keys: idxs.extend(layers.get(k,[]))
    return sorted(set(idxs))

configs = [
    ("L1 only", gid(["L1"])),
    ("L1+L2", gid(["L1","L2"])),
    ("L1+L2+L3", gid(["L1","L2","L3"])),
    ("L1+L2+L3+L4", gid(["L1","L2","L3","L4"])),
    ("+L5", gid(["L1","L2","L3","L4","L5"])),
    ("+L2+", gid(["L1","L2","L2+","L3","L4","L5"])),
]

# remove FILT_NTU raw
filt_idx = [i for i,n in enumerate(names) if str(n)=="FILT_NTU"]
if filt_idx:
    idx = [i for i in range(len(names)) if i!=filt_idx[0]]
    configs.append(("remove FILT_NTU(raw)", idx))

# remove ALL FILT-related
filt_all = [i for i,n in enumerate(names) if "FILT" in str(n).upper()]
if filt_all:
    idx = [i for i in range(len(names)) if i not in filt_all]
    configs.append((f"remove ALL_FILT({len(filt_all)})", idx))

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
results = []

for cfg_name, feat_idx in configs:
    X_sub = X[:,feat_idx]
    rmses,r2s,mapes = [],[],[]
    for tr,va in tscv.split(X_sub):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X_sub[tr], y[tr])
        pt = m.predict(X_sub[va])
        yva = boxcox_inverse(y[va], lam)
        pr = boxcox_inverse(pt, lam)
        rmses.append(np.sqrt(mean_squared_error(yva,pr)))
        r2s.append(r2_score(yva,pr))
        mapes.append(mean_absolute_percentage_error(yva,pr)*100)
    results.append({"Config":cfg_name,"n_feat":len(feat_idx),
        "RMSE":f"{np.mean(rmses):.4f}+/-{np.std(rmses):.3f}",
        "R2":f"{np.mean(r2s):.4f}+/-{np.std(r2s):.3f}",
        "MAPE":f"{np.mean(mapes):.1f}+/-{np.std(mapes):.1f}",
        "R2_m":np.mean(r2s)})

pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR,"q1_ablation_full.csv"),index=False,encoding="utf-8-sig")

print(f"\n{'='*80}")
print(f"  Q1 Ablation (5-fold XGBoost)")
print(f"{'='*80}")
print(f"  {'Config':<25s} {'n':>4s}  {'RMSE':<18s}  {'R2':<18s}  {'MAPE(%)':<14s}")
print(f"  {'-'*75}")
best = max(r["R2_m"] for r in results)
for r in results:
    star = " *" if r["R2_m"]==best else ""
    print(f"  {r['Config']:<25s} {r['n_feat']:>4d}  {r['RMSE']:<18s}  {r['R2']:<18s}  {r['MAPE']:<14s}{star}")

print(f"\n  R2 增量:")
prev = None
for r in results:
    cur = r["R2_m"]
    if prev is not None:
        d = cur-prev; bar="+"*max(1,int(abs(d)*100))
        print(f"    {r['Config']:<25s} dR2={d:+.4f}  {bar}")
    prev = cur

# 重度消融单独说明
filt_row = [r for r in results if "ALL_FILT" in r["Config"]]
if filt_row:
    all_idx = [r for r in results if "+L2+" in r["Config"]]
    if all_idx:
        drop = all_idx[0]["R2_m"] - filt_row[0]["R2_m"]
        print(f"\n  移除全部 FILT_NTU 系列特征: R2 从 {all_idx[0]['R2_m']:.4f} 降到 {filt_row[0]['R2_m']:.4f}")
        print(f"  -> FILT_NTU 系列特征的净贡献 = {drop:.4f} ({drop/all_idx[0]['R2_m']*100:.0f}% 的 R2)")

print(f"{'='*80}")
