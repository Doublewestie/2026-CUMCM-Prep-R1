"""
step5.0_ablation.py — Q1 特征层级消融 + Q2 模型消融
===========================================================
Q1 消融 (5个问题):
  ① L2衍生特征增益 (物理公式 vs 原始传感器)
  ② L3滞后特征增益 (历史信息预测力)
  ③ L4聚合特征增益 (趋势/波动信息)
  ④ L2+幂次项增益 (物理泰勒展开价值)
  ⑤ FILT_NTU 主导性 (核心特征移除→模型崩溃？)

Q2 消融 (8组对比, log(FILT) AR):
  1. log-AR(3) + RidgeCV
  2. log-AR(6) + RidgeCV
  3. log-AR(12) + RidgeCV
  4. FILT-level AR(6) + RidgeCV
  5. log-AR(6) + ElasticNet
  6. log-AR(6) + Huber
  7. log-AR(6) + Ensemble-3
  8. log-AR(6) + ARDL-tau + Ensemble

输出:
  Q1: output/q1_ablation_results.csv
  Q2: output/q2_ablation.csv
"""

import os, json, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from step0_config import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================
# Q1 辅助函数 (复用原有逻辑)
# ==============================
def boxcox_inverse(y_trans, lam):
    y_t = np.asarray(y_trans, dtype=np.float64).copy()
    if abs(lam) < 1e-6:
        return np.expm1(y_t)
    if lam < 0:
        y_t = np.minimum(y_t, 0.99 / abs(lam))
    else:
        y_t = np.maximum(y_t, -0.99 / lam)
    return (y_t * lam + 1) ** (1.0 / lam) - EPS


def classify_layers(feature_names):
    layers = {"L1": [], "L2": [], "L2+": [], "L3": [], "L4": [], "L5": []}
    for i, name in enumerate(feature_names):
        n = str(name)
        if n in ["PI_load", "GAMMA_alum", "PSI_alum", "OMEGA_night"]:
            layers["L5"].append(i); continue
        if "_lag" in n:
            layers["L3"].append(i); continue
        if any(s in n for s in ["_mean", "_std", "_max", "_delta"]):
            layers["L4"].append(i); continue
        if n in ["FILT_sq","FILT_sqrt","FILT_cubert","neg_ln_eta","eta_sq",
                 "rw_ntu_sqrt","rw_ntu_log","alum_inv","alum_sqrt",
                 "tw_flow_log","dose_ratio_sq","dose_ratio_inv"]:
            layers["L2+"].append(i); continue
        if n in ["eta_coag","phi_alum","psi_hyd","hour_sin","hour_cos",
                 "day_sin","day_cos","is_weekend","is_night"]:
            layers["L2"].append(i); continue
        layers["L1"].append(i)
    return layers


def run_q1_ablation(X, y, feature_names):
    """Q1 特征层级消融"""
    import joblib
    lam = joblib.load(OUT_LAMBDA_NTU)
    layers = classify_layers(feature_names)

    def get_indices(layer_keys):
        idxs = []
        for k in layer_keys:
            idxs.extend(layers[k])
        return sorted(set(idxs))

    L_all_keys = ["L1","L2","L2+","L3","L4","L5"]
    ablation_configs = [
        ("L1 only",           get_indices(["L1"])),
        ("L1+L2",             get_indices(["L1","L2"])),
        ("L1+L2+L3",          get_indices(["L1","L2","L3"])),
        ("L1+L2+L3+L4",       get_indices(["L1","L2","L3","L4"])),
        ("+L5(交互)",         get_indices(["L1","L2","L3","L4","L5"])),
        ("+L2+(幂次)",       get_indices(L_all_keys)),
    ]

    # 轻度消融：仅移除 FILT_NTU 原值（滞后/聚合版本仍在）
    filt_idx = None
    for i, name in enumerate(feature_names):
        if str(name) == "FILT_NTU":
            filt_idx = i; break
    if filt_idx is not None:
        no_filt_idx = [i for i in range(len(feature_names)) if i != filt_idx]
        ablation_configs.append(("remove FILT_NTU(raw)", no_filt_idx))

    # 重度消融：移除全部 FILT_NTU 相关特征（原值+滞后+聚合+幂次）
    filt_related = []
    for i, name in enumerate(feature_names):
        n = str(name).upper()
        if "FILT" in n:
            filt_related.append(i)
    if filt_related:
        no_filt_all_idx = [i for i in range(len(feature_names)) if i not in filt_related]
        n_removed = len(filt_related)
        ablation_configs.append((f"remove ALL_FILT({n_removed})", no_filt_all_idx))

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    for config_name, feat_indices in ablation_configs:
        X_sub = X[:, feat_indices]
        fold_rmses, fold_r2s, fold_mapes = [], [], []
        for tr_idx, va_idx in tscv.split(X_sub):
            m = XGBRegressor(**XGB_PARAMS)
            m.fit(X_sub[tr_idx], y[tr_idx])
            pred_t = m.predict(X_sub[va_idx])
            yva_r = boxcox_inverse(y[va_idx], lam)
            pred_r = boxcox_inverse(pred_t, lam)
            fold_rmses.append(np.sqrt(mean_squared_error(yva_r, pred_r)))
            fold_r2s.append(r2_score(yva_r, pred_r))
            fold_mapes.append(mean_absolute_percentage_error(yva_r, pred_r) * 100)
        results.append({
            "消融配置": config_name, "特征数": len(feat_indices),
            "RMSE_mean": np.mean(fold_rmses), "RMSE_std": np.std(fold_rmses),
            "R2_mean": np.mean(fold_r2s), "R2_std": np.std(fold_r2s),
            "MAPE_mean": np.mean(fold_mapes), "MAPE_std": np.std(fold_mapes),
        })
    return pd.DataFrame(results)



# ==============================
# Q2 log-AR 消融
# ==============================
def run_q2_logar_ablation():
    """Q2 log(FILT) AR(6) 消融: 比较变换方案/模型/阶数/特征"""
    print("\n[Q2 log-AR Ablation]")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load data
    for d in os.listdir(os.path.join(BASE_DIR, 'data', '2025')):
        fp = os.path.join(BASE_DIR, 'data', '2025', d)
        if os.path.isdir(fp): raw_dir = fp; break
    FILES = sorted([f for f in os.listdir(raw_dir) if f.endswith('.xlsx')])
    RENAME = {'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU','R/W CLR':'RW_CLR',
              'FILT. NTU':'FILT_NTU','C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW',
              'ALUM':'ALUM','NTU':'NTU','R/W PH':'RW_PH'}
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
    
    filt = data['FILT_NTU'].values.astype(float)
    n = len(filt)
    EPS = 1e-3
    log_filt = np.log(filt + EPS)
    
    def ar_lags(y, k):
        X = np.zeros((len(y), k))
        for lag in range(1, k+1):
            X[lag:, lag-1] = y[:-lag]
            X[:lag, lag-1] = y[0]
        return X
    
    def roll_safe(arr, lag):
        s = np.roll(arr, lag); s[:lag] = arr[0]; return s
    
    tscv = TimeSeriesSplit(n_splits=5)
    ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Build external feature matrix
    for col in ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','RW_PH','ALUM','CW_WELL_LEVEL','TW_FLOW','NTU']:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    rw_ntu = data['RW_NTU'].values.astype(float)
    rl = data['RIVER_LEVEL'].values.astype(float)
    tw = data['TW_FLOW'].values.astype(float)
    alum = data['ALUM'].values.astype(float)
    
    X_log6 = ar_lags(log_filt, 6)
    X_log3 = ar_lags(log_filt, 3)
    X_filt6 = ar_lags(filt, 6)
    X_log12 = ar_lags(log_filt, 12)
    
    X_ardl = np.column_stack([
        X_log6,
        roll_safe(np.log(rw_ntu + EPS), 2),
        roll_safe(alum, 3),
        rl, roll_safe(tw, 1),
    ])
    X_ardl = np.nan_to_num(X_ardl, nan=0)
    
    from sklearn.linear_model import RidgeCV, ElasticNetCV, HuberRegressor
    
    def cv_eval(Xmat, y_log, start, is_log=True):
        r2s, rms = [], []
        for tr, va in tscv.split(Xmat[start:]):
            m = RidgeCV(alphas=ALPHAS).fit(Xmat[start:][tr], y_log[start:][tr])
            if is_log:
                p = np.exp(m.predict(Xmat[start:][va])) - EPS
            else:
                p = m.predict(Xmat[start:][va])
            t = filt[start:][va]
            r2s.append(r2_score(t, p))
            rms.append(np.sqrt(mean_squared_error(t, p)))
        return np.mean(r2s), np.std(r2s), np.mean(rms)
    
    
    def cv_eval_ensemble(Xmat, y_log, start):
        r2s, rms = [], []
        for tr, va in tscv.split(Xmat[start:]):
            Xv, yv = Xmat[start:], y_log[start:]
            m1 = RidgeCV(alphas=ALPHAS).fit(Xv[tr], yv[tr])
            m2 = ElasticNetCV(l1_ratio=0.5, alphas=[0.001, 0.01, 0.1, 1.0], max_iter=10000, cv=3).fit(Xv[tr], yv[tr])
            m3 = HuberRegressor(alpha=0.1, max_iter=500).fit(Xv[tr], yv[tr])
            p = (np.exp(m1.predict(Xv[va])) - EPS +
                 np.exp(m2.predict(Xv[va])) - EPS +
                 np.exp(m3.predict(Xv[va])) - EPS) / 3
            t = filt[start:][va]
            r2s.append(r2_score(t, p))
            rms.append(np.sqrt(mean_squared_error(t, p)))
        return np.mean(r2s), np.std(r2s), np.mean(rms)
    
    configs = [
        ("1.log-AR(3)+RidgeCV", lambda: cv_eval(X_log3, log_filt, 3)),
        ("2.log-AR(6)+RidgeCV", lambda: cv_eval(X_log6, log_filt, 6)),
        ("3.log-AR(12)+RidgeCV", lambda: cv_eval(X_log12, log_filt, 12)),
        ("4.FILT-level AR(6)+RidgeCV", lambda: cv_eval(X_filt6, filt, 6, is_log=False)),
        ("5.log-AR(6)+ElasticNet", lambda: cv_eval(X_log6, log_filt, 6)),
        ("6.log-AR(6)+Huber", lambda: cv_eval(X_log6, log_filt, 6)),
        ("7.log-AR(6)+Ensemble-3", lambda: cv_eval_ensemble(X_log6, log_filt, 6)),
        ("8.log-AR(6)+ARDL-tau+Ensemble", lambda: cv_eval_ensemble(X_ardl, log_filt, 6)),
    ]
    
    results = []
    for name, func in configs:
        r2_m, r2_s, rmse = func()
        results.append({"config": name, "R2_mean": r2_m, "R2_std": r2_s, "RMSE": rmse})
        print(f"  {name:<40s} R2={r2_m:.4f}+-{r2_s:.4f}  RMSE={rmse:.4f}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "q2_logar_ablation.csv"), index=False, encoding="utf-8-sig")
    
    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    configs_s = [r["config"] for r in results]
    r2s_s = [r["R2_mean"] for r in results]
    rms_s = [r["RMSE"] for r in results]
    
    axes[0].barh(range(len(configs_s)), r2s_s, color='steelblue', alpha=0.85, edgecolor='white')
    axes[0].set_yticks(range(len(configs_s)))
    axes[0].set_yticklabels(configs_s, fontsize=9)
    axes[0].set_xlabel('CV R2')
    axes[0].invert_yaxis()
    axes[0].axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    axes[1].barh(range(len(configs_s)), rms_s, color='steelblue', alpha=0.85, edgecolor='white')
    axes[1].set_yticks(range(len(configs_s)))
    axes[1].set_yticklabels(configs_s, fontsize=9)
    axes[1].set_xlabel('CV RMSE')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "q2_logar_ablation_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [DONE] q2_logar_ablation_bar.png saved")
    return df


# ==============================
# 主流程
# ==============================
def main():
    print("=" * 60)
    print("  step5.0 — 跨题消融实验汇总")
    print("=" * 60)

    # ---- Q1 消融 ----
    if os.path.exists(OUT_X_ALL) and os.path.exists(OUT_LAMBDA_NTU):
        print("\n[Q1] Feature-level ablation")
        X = np.load(OUT_X_ALL).astype(np.float64)
        y = np.load(OUT_Y_ALL).astype(np.float64)
        feature_names = list(np.load(OUT_FEATURE_NAMES, allow_pickle=True))
        print(f"  X={X.shape}, y={y.shape}")

        df_q1 = run_q1_ablation(X, y, feature_names)
        print(f"\n{'='*75}")
        print(f"  Q1 Ablation ({N_SPLITS}-fold, XGBoost)")
        print(f"{'='*75}")
        best_r2 = df_q1["R2_mean"].max()
        for _, row in df_q1.iterrows():
            marker = " *" if row["R2_mean"] == best_r2 else ""
            print(f"  {row['消融配置']:<20s} {row['特征数']:>5d} "
                  f"RMSE={row['RMSE_mean']:.4f} R2={row['R2_mean']:.4f}{marker}")
        df_q1.to_csv(os.path.join(OUTPUT_DIR, "q1_ablation_results.csv"),
                     index=False, encoding="utf-8-sig")
        print("[step5.0] q1_ablation_results.csv saved")
    else:
        print("\n[Q1] Skipped: preprocessed data not found (run step0_preprocess.py first)")

    # ---- Q2 log-AR 消融 (新增) ----
    print(f"\n{'='*60}")
    print("  [Q2] log-AR 模型消融")
    print(f"{'='*60}")
    df_q2_logar = run_q2_logar_ablation()
    if df_q2_logar is not None:
        best_row = df_q2_logar.loc[df_q2_logar["R2_mean"].idxmax()]
        print(f"\n  Best config: {best_row['config']}  R2={best_row['R2_mean']:.4f}  RMSE={best_row['RMSE']:.4f}")
    
    print(f"\n[step5.0] Done.")


if __name__ == "__main__":
    main()
