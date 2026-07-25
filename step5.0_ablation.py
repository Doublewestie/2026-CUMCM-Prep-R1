"""
step5.0_ablation.py — Q1 CSTR 组件消融 + Q2 log-AR 消融 + Feb 2026 预测对比
===========================================================================
仅修改此文件，不触碰 step0/step1/step2 原始代码。

物理背景: 清水池 CSTR 模型
  NTU(t) = beta(t)*NTU(t-1) + (1-beta(t))*FILT(t)  ✓ FILT(t) 是滤后水入池浓度
  beta(t) = exp(-2h/theta), theta = A_eff * CW_WELL / TW_FLOW

Q1 CSTR 消融 (6 组):
  ① Base (3-Tier + Balance): A_T1=400, A_T2=250, A_same=100, A_diff=20
  ② No_Balance: A_T1=400, A_T2=250, A_T3=30（取消平衡规则）
  ③ No_Tier: 全量用 A=141.3（即原单 CSTR 基线）
  ④ Pure_Persistence: NTU(t)=NTU(t-1)，无混合
  ⑤ Direct_FILT: NTU(t)=FILT(t)，无 CSTR 缓冲
  ⑥ Mean: 常数均值预测

Q2 log-AR 消融 (7 组, 修复 RidgeCV bug):
  ① log-AR(3)+RidgeCV  ② log-AR(6)+RidgeCV  ③ log-AR(12)+RidgeCV
  ④ FILT-level AR(6)+RidgeCV  ⑤ log-AR(6)+ElasticNet  ⑥ log-AR(6)+Huber
  ⑦ log-AR(6)+Ensemble-3

Q2 Feb 2026 预测对比:
  Global AR(6) vs T2-only AR(6) vs Hybrid Switching

输出:
  output/q1_cstr_ablation.csv
  output/q2_logar_ablation.csv
  output/q2_feb2026_compare.csv
  results/tables/ablation_summary.csv
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, ElasticNetCV, HuberRegressor, LinearRegression
from step0_config import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

EPS = 1e-6

# ================================================================
#  全局: 数据加载 (统一从 clean_data.csv 加载)
# ================================================================

def load_clean():
    df = pd.read_csv(OUT_CLEAN_DATA)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


# ================================================================
#  Q1: CSTR 模型组件消融
# ================================================================

def cstr_eval(df, A_T1, A_T2, A_T3=30, A_same=None, A_diff=None,
              RL_med=None, Q_med=None, use_balance=True):
    """CSTR 段2评估器: NTU(t)=beta*NTU(t-1)+(1-beta)*FILT(t)
    FILT(t) 为 t 时刻滤后水浊度（清水池入流），物理上可提前获取。
    """
    filt = df["FILT_NTU"].values.astype(float)
    ntu  = df["NTU"].values.astype(float)
    cw   = df["CW_WELL_LEVEL"].values.astype(float)
    tw   = df["TW_FLOW"].values.astype(float)
    rl   = df.get("RIVER_LEVEL", pd.Series(np.full(len(df), np.nan))).values.astype(float)
    n = len(ntu)
    pred = np.zeros(n)
    pred[0] = ntu[0]
    for t in range(1, n):
        H = max(cw[t-1], 0.1)
        Qv = max(tw[t-1], 1.0)
        ft = filt[t]
        if ft <= TIER_THRESHOLDS[0]:
            A0 = A_T1
        elif ft <= TIER_THRESHOLDS[1]:
            A0 = A_T2
        else:
            if use_balance and A_same is not None and A_diff is not None:
                rv = rl[t]
                if not np.isnan(rv) and RL_med is not None and Q_med is not None:
                    A0 = A_same if (rv - RL_med) * (tw[t] - Q_med) > 0 else A_diff
                else:
                    A0 = A_T3
            else:
                A0 = A_T3
        theta = A0 * H / Qv
        theta = max(theta, 0.02)
        beta = np.exp(-2.0 / theta)
        beta = np.clip(beta, 0.001, 0.999)
        pred[t] = beta * ntu[t-1] + (1 - beta) * ft
    pred = np.clip(pred, 0, np.inf)
    ssr = np.sum((ntu - pred) ** 2)
    sst = np.sum((ntu - ntu.mean()) ** 2)
    r2 = round(1 - ssr / (sst + EPS), 4)
    rmse = round(float(np.sqrt(np.mean((ntu - pred) ** 2))), 4)
    mask_t3 = filt > TIER_THRESHOLDS[1]
    if mask_t3.sum() >= 10:
        s_t3 = np.sum((ntu[mask_t3] - pred[mask_t3]) ** 2)
        t_t3 = np.sum((ntu[mask_t3] - ntu[mask_t3].mean()) ** 2)
        r2_t3 = round(1 - s_t3 / (t_t3 + EPS), 4)
    else:
        r2_t3 = None
    return {"R2_all": r2, "RMSE_all": rmse, "R2_T3": r2_t3}


def run_q1_cstr_ablation(df):
    """Q1 CSTR 组件消融 (5 组)"""
    print("\n" + "=" * 65)
    print("  [Q1] CSTR 组件消融")
    print("=" * 65)

    # 平衡检测器参数 (复用 step1.7 Phase 5 逻辑)
    filt = df["FILT_NTU"].values.astype(float)
    rl = df["RIVER_LEVEL"].values.astype(float)
    tw = df["TW_FLOW"].values.astype(float)
    m_t3 = filt > TIER_THRESHOLDS[1]
    rl_t3 = rl[m_t3][~np.isnan(rl[m_t3])]
    RL_med = float(np.median(rl_t3)) if len(rl_t3) > 0 else 6.09
    Q_med = float(np.median(tw[m_t3])) if m_t3.sum() > 0 else 44.0

    ntu = df["NTU"].values.astype(float)
    filt = df["FILT_NTU"].values.astype(float)

    configs = [
        ("1.Base(Balance)", {
            "A_T1": 400, "A_T2": 250, "A_T3": 30,
            "A_same": 100, "A_diff": 20,
            "RL_med": RL_med, "Q_med": Q_med,
            "use_balance": True,
        }),
        ("2.No_Balance(A_T3=30)", {
            "A_T1": 400, "A_T2": 250, "A_T3": 30,
            "use_balance": False,
        }),
        ("3.No_Tier(A=141.3)", {
            "A_T1": 141.3, "A_T2": 141.3, "A_T3": 141.3,
            "use_balance": False,
        }),
        ("4.Pure_Persistence", {"_mode": "persist"}),
        ("5.Direct_FILT", {"_mode": "direct"}),
        ("6.Mean_Pred", {"_mean": True}),
    ]

    results = []
    for name, params in configs:
        if params.get("_mean"):
            pred = np.full(len(df), ntu.mean())
            ssr = np.sum((ntu - pred) ** 2)
            sst = np.sum((ntu - ntu.mean()) ** 2)
            r2 = round(1 - ssr / (sst + EPS), 4)
            rmse = round(float(np.sqrt(np.mean((ntu - pred) ** 2))), 4)
            r2_t3 = None
        elif params.get("_mode") == "persist":
            pred = np.roll(ntu, 1); pred[0] = ntu[0]
            ssr = np.sum((ntu - pred) ** 2)
            sst = np.sum((ntu - ntu.mean()) ** 2)
            r2 = round(1 - ssr / (sst + EPS), 4)
            rmse = round(float(np.sqrt(np.mean((ntu - pred) ** 2))), 4)
            mask_t3 = filt > TIER_THRESHOLDS[1]
            if mask_t3.sum() >= 10:
                s_t3 = np.sum((ntu[mask_t3] - pred[mask_t3]) ** 2)
                t_t3 = np.sum((ntu[mask_t3] - ntu[mask_t3].mean()) ** 2)
                r2_t3 = round(1 - s_t3 / (t_t3 + EPS), 4)
            else:
                r2_t3 = None
        elif params.get("_mode") == "direct":
            pred = filt.copy()
            ssr = np.sum((ntu - pred) ** 2)
            sst = np.sum((ntu - ntu.mean()) ** 2)
            r2 = round(1 - ssr / (sst + EPS), 4)
            rmse = round(float(np.sqrt(np.mean((ntu - pred) ** 2))), 4)
            mask_t3 = filt > TIER_THRESHOLDS[1]
            if mask_t3.sum() >= 10:
                s_t3 = np.sum((ntu[mask_t3] - pred[mask_t3]) ** 2)
                t_t3 = np.sum((ntu[mask_t3] - ntu[mask_t3].mean()) ** 2)
                r2_t3 = round(1 - s_t3 / (t_t3 + EPS), 4)
            else:
                r2_t3 = None
        else:
            m = cstr_eval(df, **params)
            r2, rmse, r2_t3 = m["R2_all"], m["RMSE_all"], m["R2_T3"]
        results.append({"config": name, "R2_all": r2, "RMSE_all": rmse, "R2_T3": r2_t3})
        print(f"  {name:<30s}  R2={r2:.4f}  RMSE={rmse:.4f}")

    df_out = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "q1_cstr_ablation.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [DONE] {out_path}")

    # 柱状图
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [r["config"] for r in results]
    r2s = [r["R2_all"] for r in results]
    colors = ["#2c7bb6" if "Base" in n else
              "#fdae61" if "No_Balance" in n else
              "#fdae61" if "No_Tier" in n else
              "#969696" for n in names]
    bars = ax.bar(range(len(names)), r2s, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(y=0, color="gray", lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Q1 CSTR Component Ablation")
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.015, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "q1_cstr_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"  [DONE] figures/q1_cstr_ablation.png")
    return df_out


# ================================================================
#  Q2: log-AR 模型消融
# ================================================================

def _load_2025_raw():
    """从原始 Excel 加载 2025 数据"""
    from step2_shared import load_raw_filt_data
    return load_raw_filt_data()


def _ar_lags(y, k):
    X = np.zeros((len(y), k))
    for lag in range(1, k+1):
        X[lag:, lag-1] = y[:-lag]
        X[:lag, lag-1] = y[0]
    return X


def _cv_ridge(Xmat, y_log, start, filt_raw, alphas, tscv):
    r2s, rms = [], []
    for tr, va in tscv.split(Xmat[start:]):
        m = RidgeCV(alphas=alphas).fit(Xmat[start:][tr], y_log[start:][tr])
        p = np.exp(m.predict(Xmat[start:][va])) - 1e-3
        t = filt_raw[start:][va]
        r2s.append(r2_score(t, p))
        rms.append(np.sqrt(mean_squared_error(t, p)))
    return np.mean(r2s), np.std(r2s), np.mean(rms)


def _cv_en(Xmat, y_log, start, filt_raw, tscv):
    r2s, rms = [], []
    for tr, va in tscv.split(Xmat[start:]):
        m = ElasticNetCV(l1_ratio=0.5, alphas=[0.001, 0.01, 0.1, 1.0],
                         max_iter=10000, cv=3).fit(Xmat[start:][tr], y_log[start:][tr])
        p = np.exp(m.predict(Xmat[start:][va])) - 1e-3
        t = filt_raw[start:][va]
        r2s.append(r2_score(t, p))
        rms.append(np.sqrt(mean_squared_error(t, p)))
    return np.mean(r2s), np.std(r2s), np.mean(rms)


def _cv_huber(Xmat, y_log, start, filt_raw, tscv):
    r2s, rms = [], []
    for tr, va in tscv.split(Xmat[start:]):
        m = HuberRegressor(alpha=0.1, max_iter=500).fit(Xmat[start:][tr], y_log[start:][tr])
        p = np.exp(m.predict(Xmat[start:][va])) - 1e-3
        t = filt_raw[start:][va]
        r2s.append(r2_score(t, p))
        rms.append(np.sqrt(mean_squared_error(t, p)))
    return np.mean(r2s), np.std(r2s), np.mean(rms)


def _cv_ensemble(Xmat, y_log, start, filt_raw, tscv, alphas):
    r2s, rms = [], []
    for tr, va in tscv.split(Xmat[start:]):
        Xv, yv = Xmat[start:], y_log[start:]
        m1 = RidgeCV(alphas=alphas).fit(Xv[tr], yv[tr])
        m2 = ElasticNetCV(l1_ratio=0.5, alphas=[0.001, 0.01, 0.1, 1.0],
                          max_iter=10000, cv=3).fit(Xv[tr], yv[tr])
        m3 = HuberRegressor(alpha=0.1, max_iter=500).fit(Xv[tr], yv[tr])
        p = (np.exp(m1.predict(Xv[va])) + np.exp(m2.predict(Xv[va])) + np.exp(m3.predict(Xv[va]))) / 3 - 1e-3
        t = filt_raw[start:][va]
        r2s.append(r2_score(t, p))
        rms.append(np.sqrt(mean_squared_error(t, p)))
    return np.mean(r2s), np.std(r2s), np.mean(rms)


def run_q2_logar_ablation():
    """Q2 log(FILT) 消融 — 7 组配置, 各用正确回归器"""
    print("\n" + "=" * 65)
    print("  [Q2] log-AR 模型消融")
    print("=" * 65)

    data = _load_2025_raw()
    filt = data["FILT_NTU"].values.astype(float)
    n = len(filt)
    log_filt = np.log(filt + 1e-3)

    X_log3 = _ar_lags(log_filt, 3)
    X_log6 = _ar_lags(log_filt, 6)
    X_log12 = _ar_lags(log_filt, 12)
    X_filt6 = _ar_lags(filt, 6)

    tscv = TimeSeriesSplit(n_splits=5)
    ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    configs = [
        ("1.log-AR(3)+RidgeCV",  lambda: _cv_ridge(X_log3, log_filt, 3, filt, ALPHAS, tscv)),
        ("2.log-AR(6)+RidgeCV",  lambda: _cv_ridge(X_log6, log_filt, 6, filt, ALPHAS, tscv)),
        ("3.log-AR(12)+RidgeCV", lambda: _cv_ridge(X_log12, log_filt, 12, filt, ALPHAS, tscv)),
        ("4.FILT-level AR(6)",    lambda: _cv_ridge(X_filt6, filt, 6, filt, ALPHAS, tscv)),
        ("5.log-AR(6)+ElasticNet", lambda: _cv_en(X_log6, log_filt, 6, filt, tscv)),
        ("6.log-AR(6)+Huber",    lambda: _cv_huber(X_log6, log_filt, 6, filt, tscv)),
        ("7.log-AR(6)+Ensemble", lambda: _cv_ensemble(X_log6, log_filt, 6, filt, tscv, ALPHAS)),
    ]

    results = []
    for name, func in configs:
        r2_m, r2_s, rmse = func()
        results.append({"config": name, "R2_mean": r2_m, "R2_std": r2_s, "RMSE": rmse})
        print(f"  {name:<35s}  R2={r2_m:.4f}+-{r2_s:.4f}  RMSE={rmse:.4f}")

    df_out = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "q2_logar_ablation.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [DONE] {out_path}")

    # 柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [r["config"] for r in results]
    r2s = [r["R2_mean"] for r in results]
    rms = [r["RMSE"] for r in results]
    axes[0].barh(range(len(labels)), r2s, color="steelblue", alpha=0.85, edgecolor="white")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("CV R2")
    axes[0].invert_yaxis()
    axes[0].axvline(x=0, color="gray", lw=0.5)
    axes[1].barh(range(len(labels)), rms, color="steelblue", alpha=0.85, edgecolor="white")
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("CV RMSE")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "q2_logar_ablation_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  [DONE] figures/q2_logar_ablation_bar.png")
    return df_out


# ================================================================
#  Q2: Feb 2026 预测方法对比
# ================================================================

def _load_2026_data():
    """加载 Jan+Feb 2026 数据"""
    jan_path = os.path.join(DATA_DIR_2026, "2026年1月.xls")
    feb_path = os.path.join(DATA_DIR_2026, "2026年2月.xls")
    col_map = {
        "TIME ": "TIME", "RIVER LEVEL": "RIVER_LEVEL",
        "R/W PUMP DUTY": "RW_PUMP_DUTY", "R/W FLOW": "RW_FLOW",
        "R/W NTU": "RW_NTU", "R/W CLR": "RW_CLR", "R/W PH": "RW_PH",
        "FILT. NTU": "FILT_NTU", "C/W WELL LEVEL": "CW_WELL_LEVEL",
        "F/RIDE": "F_RIDE", "T/W PUMP DUTY": "TW_PUMP_DUTY",
        "T/W FLOW": "TW_FLOW", "18ML LEVEL": "18ML_LEVEL",
        "18ML FLOW": "18ML_FLOW",
    }
    df_jan = pd.read_excel(jan_path).rename(columns=col_map)
    df_feb = pd.read_excel(feb_path).rename(columns=col_map)
    for c in ["FILT_NTU", "RW_NTU"]:
        df_jan[c] = pd.to_numeric(df_jan[c], errors="coerce")
        df_feb[c] = pd.to_numeric(df_feb[c], errors="coerce")
    return df_jan, df_feb


def _train_tier_ar(data, tier_col, tier_id, ar_order=6):
    """训练单级 AR"""
    mask = data[tier_col] == tier_id
    sub = data["FILT_NTU"].values[mask.values]
    n_t = len(sub)
    if n_t <= ar_order + 5:
        return {"coef": np.zeros(ar_order), "intercept": float(np.mean(sub))}
    X = np.column_stack([np.roll(sub, i) for i in range(1, ar_order + 1)])[ar_order:]
    y = sub[ar_order:]
    m = LinearRegression().fit(X, y)
    return {"coef": m.coef_, "intercept": m.intercept_}


def run_q2_feb2026_compare(df_2025):
    """三种方法在 Feb 2026 数据上的预测对比"""
    print("\n" + "=" * 65)
    print("  [Q2] Feb 2026 预测方法对比")
    print("=" * 65)

    df_jan, df_feb = _load_2026_data()
    filt = df_2025["FILT_NTU"].values.astype(float)
    n_all = len(filt)
    AR_ORDER = 6
    TAU_RW = 2

    # 准备种子
    seed = pd.to_numeric(df_jan["FILT_NTU"], errors="coerce").values[-AR_ORDER:].copy()
    if np.any(np.isnan(seed)):
        print("  [WARNING] Jan FILT_NTU has NaN in last 6 rows, using fallback")
        seed = np.full(AR_ORDER, 0.1)
    feb_actual = pd.to_numeric(df_feb["FILT_NTU"], errors="coerce").values
    n_pred = len(df_feb)
    rw_feb = pd.to_numeric(df_feb["RW_NTU"], errors="coerce").values

    # 三级标签
    tiers = np.ones(n_all, dtype=int)
    tiers[filt > TIER_THRESHOLDS[0]] = 2
    tiers[filt > TIER_THRESHOLDS[1]] = 3

    # 训练模型
    tier_models = {t: _train_tier_ar(df_2025.assign(tier=tiers), "tier", t, AR_ORDER)
                   for t in [1, 2, 3]}

    # 全局 AR
    X_all = np.column_stack([np.roll(filt, i) for i in range(1, AR_ORDER + 1)])[AR_ORDER:]
    y_all = filt[AR_ORDER:]
    m_global = LinearRegression().fit(X_all, y_all)

    # -------- 预测 ----------
    def predict_global(seed_vec, steps):
        vals = seed_vec.copy()
        out = np.zeros(steps)
        for t in range(steps):
            out[t] = m_global.intercept_ + np.dot(m_global.coef_, vals[::-1])
            vals = np.roll(vals, -1)
            vals[-1] = out[t]
        return out

    def predict_t2only(seed_vec, steps):
        vals = seed_vec.copy()
        m = tier_models[2]
        out = np.zeros(steps)
        for t in range(steps):
            out[t] = m["intercept"] + np.dot(m["coef"], vals[::-1])
            vals = np.roll(vals, -1)
            vals[-1] = out[t]
        return out

    def predict_hybrid(seed_vec, steps, rw_arr):
        vals = seed_vec.copy()
        m = tier_models[2]
        t2_ss = m["intercept"] / (1 - sum(m["coef"]) + 1e-8)
        out = np.zeros(steps)
        for t in range(steps):
            base = m["intercept"] + np.dot(m["coef"], vals[::-1])
            correction = max(0, (rw_arr[t - TAU_RW] - 25) / 2000.0) if t >= TAU_RW else 0
            filt_last = seed[-1] if t == 0 else out[t-1]
            if filt_last < TIER_THRESHOLDS[0]:
                scale = 0.3
            elif filt_last > TIER_THRESHOLDS[1]:
                scale = 1.3
            else:
                scale = 1.0
            out[t] = base * scale + correction
            out[t] = np.clip(out[t], 0.01, 0.35)
            vals = np.roll(vals, -1)
            vals[-1] = out[t]
        return out

    pred_g = predict_global(seed, n_pred)
    pred_t2 = predict_t2only(seed, n_pred)
    pred_h = predict_hybrid(seed, n_pred, rw_feb)

    valid = ~np.isnan(feb_actual)
    methods = {
        "Global AR(6)": pred_g,
        "T2-only AR(6)": pred_t2,
        "Hybrid Switching": pred_h,
    }

    results = []
    print(f"\n  {'Method':<25s}  {'R2':>8s}  {'RMSE':>8s}  {'MAE':>8s}")
    print(f"  {'-'*51}")
    for name, pred in methods.items():
        pv = pred[valid]
        av = feb_actual[valid]
        r2 = r2_score(av, pv)
        rmse = np.sqrt(mean_squared_error(av, pv))
        mae = np.mean(np.abs(av - pv))
        results.append({"method": name, "R2": round(r2, 4),
                        "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
        print(f"  {name:<25s}  {r2:>8.4f}  {rmse:>8.4f}  {mae:>8.4f}")

    df_out = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "q2_feb2026_compare.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [DONE] {out_path}")
    return df_out


# ================================================================
#  统一消融汇总表
# ================================================================

def run_ablation_summary(q1_cstr, q2_logar, q2_feb):
    print("\n" + "=" * 65)
    print("  [Summary] 统一消融汇总表")
    print("=" * 65)

    rows = []
    # Q1 CSTR
    for _, r in q1_cstr.iterrows():
        rows.append({"question": "Q1", "ablation": r["config"],
                      "R2": r["R2_all"], "RMSE": r["RMSE_all"]})

    # Q2 log-AR
    for _, r in q2_logar.iterrows():
        rows.append({"question": "Q2.AR", "ablation": r["config"],
                      "R2": r["R2_mean"], "RMSE": r["RMSE"]})

    # Q2 Feb 2026
    for _, r in q2_feb.iterrows():
        rows.append({"question": "Q2.Feb2026", "ablation": r["method"],
                      "R2": r["R2"], "RMSE": r["RMSE"]})

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(BASE_DIR, "results", "tables", "ablation_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n  {'Question':<15s} {'Ablation':<35s} {'R2':>8s} {'RMSE':>8s}")
    print(f"  {'-'*66}")
    for _, r in df_out.iterrows():
        print(f"  {r['question']:<15s} {r['ablation']:<35s} {r['R2']:>8.4f} {r['RMSE']:>8.4f}")
    print(f"\n  [DONE] {out_path}")
    return df_out


# ================================================================
#  主流程
# ================================================================

def main():
    print("=" * 65)
    print("  step5.0 — 消融实验 (CSTR 组件 + log-AR + Feb 2026)")
    print("=" * 65)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ---- Q1 CSTR 消融 ----
    df_clean = load_clean()
    q1_res = run_q1_cstr_ablation(df_clean)

    # ---- Q2 log-AR 消融 ----
    q2_res = run_q2_logar_ablation()

    # ---- Q2 Feb 2026 预测对比 ----
    q2_feb_res = run_q2_feb2026_compare(df_clean)

    # ---- 统一汇总 ----
    summary = run_ablation_summary(q1_res, q2_res, q2_feb_res)

    print(f"\n{'='*65}")
    print(f"  step5.0 全部完成。")
    print(f"{'='*65}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    main()
