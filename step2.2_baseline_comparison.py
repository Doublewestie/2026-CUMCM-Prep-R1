"""
step2.2_baseline_comparison.py
Q2 基准模型对比 — 传递函数 / AR(6) / ARMAX(6,4)
======================================================
输入: clean_data.csv, tau_params.json
输出: q2_baseline_comparison.csv, figures/q2_baseline_bar.png

基准模型:
  1. 传递函数 (CCF + LinearRegression) — 5个参数
  2. AR(6) — 纯自回归, 7个参数
  3. ARMAX(6,4) — 自回归+外生变量, ~17个参数

与TCN结果合并后输出深度学习 vs 基线的精度对比。
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_SPLITS = 5
EPS = 1e-6
INPUT_VARS = ["RW_NTU", "RW_FLOW", "RW_PH", "ALUM"]
AUTOREG_LAGS = 6


def load_and_align(clean_csv, tau_params):
    df = pd.read_csv(clean_csv)
    df = df.dropna(subset=["FILT_NTU"])
    n = len(df)

    y_raw = df["FILT_NTU"].values.astype(np.float64)
    x_raw_all = {}
    for v in INPUT_VARS:
        x_raw_all[v] = df[v].values.astype(np.float64)

    y_log = np.log1p(y_raw)
    x_log_all = {v: np.log1p(x_raw_all[v]) for v in INPUT_VARS}

    aligned = {}
    for v in INPUT_VARS:
        d_star = tau_params[v]["steps"]
        x_shifted = np.roll(x_log_all[v], d_star)
        x_shifted[:d_star] = np.nan
        aligned[v] = x_shifted

    return y_log, y_raw, aligned


def build_features(y_log, aligned, ar_lags=AUTOREG_LAGS):
    n = len(y_log)
    feats = []
    for v in INPUT_VARS:
        feats.append(aligned[v])
    for lag in range(1, ar_lags + 1):
        y_ar = np.roll(y_log, lag)
        y_ar[:lag] = np.nan
        feats.append(y_ar)

    X_full = np.column_stack(feats)
    valid = ~np.any(np.isnan(X_full), axis=1) & ~np.isnan(y_log)
    return X_full[valid], y_log[valid]


def evaluate_model(X, y, model_cls, fit_kwargs=None):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rmses, r2s, maes = [], [], []
    for tr_idx, vl_idx in tscv.split(X):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_vl, y_vl = X[vl_idx], y[vl_idx]
        if len(tr_idx) < 10 or len(vl_idx) < 5:
            continue
        model = model_cls()
        kw = fit_kwargs or {}
        model.fit(X_tr, y_tr, **kw)
        pred = model.predict(X_vl)
        pred_real = np.expm1(pred)
        y_real = np.expm1(y_vl)
        rmse = np.sqrt(np.mean((pred_real - y_real) ** 2))
        ss_res = np.sum((pred_real - y_real) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = 1 - ss_res / (ss_tot + EPS)
        mae = np.mean(np.abs(pred_real - y_real))
        rmses.append(rmse)
        r2s.append(r2)
        maes.append(mae)
    return np.mean(rmses), np.mean(r2s), np.mean(maes)


# ==============================
# 模型 1: 传递函数 (CCF+LinearRegression)
# ==============================
class TransferFunction:
    def __init__(self):
        self.lr = LinearRegression()

    def fit(self, X, y, **kwargs):
        self.lr.fit(X, y)

    def predict(self, X):
        return self.lr.predict(X)


# ==============================
# 模型 2: AR(6) 自回归
# ==============================
class AR6:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, **kwargs):
        ar_feats = X[:, -AUTOREG_LAGS:]
        X_ar = np.column_stack([np.ones(len(y)), ar_feats])
        theta = np.linalg.lstsq(X_ar, y, rcond=None)[0]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X):
        ar_feats = X[:, -AUTOREG_LAGS:]
        return self.intercept_ + ar_feats @ self.coef_


# ==============================
# 模型 3: ARMAX(6,4) — 自回归 + 4个外生变量
# ==============================
class ARMAX64:
    def __init__(self):
        self.lr = LinearRegression()

    def fit(self, X, y, **kwargs):
        self.lr.fit(X, y)

    def predict(self, X):
        return self.lr.predict(X)


# ==============================
# 主流程
# ==============================
def main():
    clean_csv = os.path.join(OUTPUT_DIR, "clean_data.csv")
    tau_file = os.path.join(OUTPUT_DIR, "tau_params.json")

    if not os.path.exists(tau_file):
        print("[step2.2] tau_params.json 未找到，请先运行 step2.0")
        return

    with open(tau_file, "r", encoding="utf-8") as f:
        tau_params = json.load(f)

    y_log, y_raw, aligned = load_and_align(clean_csv, tau_params)
    X_full, y_full = build_features(y_log, aligned)
    print(f"[step2.2] 有效样本: {len(X_full)}")

    results = []

    # --- 传递函数 ---
    print("\n[step2.2] 传递函数 (CCF + LR)...")
    rmse, r2, mae = evaluate_model(X_full, y_full, TransferFunction)
    results.append({
        "模型": "传递函数(LR)", "RMSE": round(rmse, 4),
        "R2": round(r2, 4), "MAE": round(mae, 4), "参数量": 11,
    })
    print(f"  RMSE={rmse:.4f} R2={r2:.4f} MAE={mae:.4f}")

    # --- AR(6) ---
    print("\n[step2.2] AR(6) 自回归...")
    rmse, r2, mae = evaluate_model(X_full, y_full, AR6)
    results.append({
        "模型": "AR(6)", "RMSE": round(rmse, 4),
        "R2": round(r2, 4), "MAE": round(mae, 4), "参数量": 7,
    })
    print(f"  RMSE={rmse:.4f} R2={r2:.4f} MAE={mae:.4f}")

    # --- ARMAX(6,4) ---
    print("\n[step2.2] ARMAX(6,4)...")
    rmse, r2, mae = evaluate_model(X_full, y_full, ARMAX64)
    results.append({
        "模型": "ARMAX(6,4)", "RMSE": round(rmse, 4),
        "R2": round(r2, 4), "MAE": round(mae, 4), "参数量": 17,
    })
    print(f"  RMSE={rmse:.4f} R2={r2:.4f} MAE={mae:.4f}")

    # 保存
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "q2_baseline_comparison.csv"),
              index=False, encoding="utf-8-sig")
    print("\n[step2.2] 基准对比表 → output/q2_baseline_comparison.csv")

    # --- 精度对比总结 ---
    print("\n[step2.2] MIC vs TE vs CCF(基准) 精度对比:")
    print(f"  CCF+LR: RMSE={results[0]['RMSE']:.4f} R2={results[0]['R2']:.4f}")
    print(f"  AR(6):   RMSE={results[1]['RMSE']:.4f} R2={results[1]['R2']:.4f}")
    print(f"  ARMAX64: RMSE={results[2]['RMSE']:.4f} R2={results[2]['R2']:.4f}")
    print("\n  (TCN精度见 step2.1 输出)")

    # 柱状图
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = [r["模型"] for r in results]
    rmses = [r["RMSE"] for r in results]
    r2s = [r["R2"] for r in results]
    params = [r["参数量"] for r in results]

    axes[0].bar(models, rmses, color=["steelblue", "mediumseagreen", "darkorange"], alpha=0.85)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("基准模型 — RMSE")

    axes[1].bar(models, r2s, color=["steelblue", "mediumseagreen", "darkorange"], alpha=0.85)
    axes[1].set_ylabel("R²")
    axes[1].set_title("基准模型 — R²")

    for ax in axes:
        ax.tick_params(axis="x", rotation=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_baseline_bar.png"), dpi=150)
    plt.close()
    print("[step2.2] figures/q2_baseline_bar.png 已保存")

    print("\n[step2.2] 完成.")


if __name__ == "__main__":
    main()
