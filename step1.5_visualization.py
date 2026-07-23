"""
step1.5_visualization.py — Q1 统一出图
========================================
输出到 output/figures/:
  1. SHAP beeswarm 图（前 30 个最重要特征）
  2. 特征重要性柱状图（按 robust_score 降序）
  3. 预测 vs 实际散点图（四模型颜色区分）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import shap, joblib, os, warnings
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from step0_config import *

warnings.filterwarnings("ignore")

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def boxcox_inverse(y_trans, lam):
    y_t = np.asarray(y_trans, dtype=np.float64).copy()
    if abs(lam) < 1e-6:
        return np.expm1(y_t)
    if lam < 0:
        upper = 0.99 / abs(lam)
        y_t = np.minimum(y_t, upper)
    else:
        lower = -0.99 / lam
        y_t = np.maximum(y_t, lower)
    return (y_t * lam + 1) ** (1.0 / lam) - EPS


def plot_shap_beeswarm(X, y, feature_names):
    """SHAP beeswarm — 展示每个特征对每条样本的影响大小+方向"""
    print("  绘制 SHAP beeswarm...")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X, y)

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)

    # 取前 30 个最重要的特征
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_k = min(30, len(feature_names))
    top_idx = np.argsort(mean_abs_shap)[-top_k:][::-1]

    X_top = X[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]
    shap_top = shap_values[:, top_idx]

    fig, ax = plt.subplots(figsize=(14, 10))
    shap.summary_plot(shap_top, X_top, feature_names=names_top,
                      show=False, max_display=top_k)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q1_shap_beeswarm.png"), dpi=300)
    plt.close()
    print("  → q1_shap_beeswarm.png")


def plot_feature_importance():
    """特征重要性柱状图"""
    print("  绘制特征重要性柱状图...")
    csv_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    if not os.path.exists(csv_path):
        print("  [WARNING] feature_importance.csv 未找到，跳过")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values("robust_score", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#2E86AB" if s else "#D64045" for s in df["selected"]]
    bars = ax.barh(range(len(df)), df["robust_score"], color=colors)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("Robust Importance Score", fontsize=12)
    ax.set_title("Q1: 特征重要性 (SHAP + Permutation 鲁棒融合)", fontsize=14)

    # 图例
    from matplotlib.patches import Patch
    legend = [Patch(color="#2E86AB", label="保留"),
              Patch(color="#D64045", label="剔除")]
    ax.legend(handles=legend, loc="lower right", fontsize=10)

    # 标注阈值线
    ax.axvline(SHAP_THRESHOLD, color="gray", linestyle="--", alpha=0.5)
    ax.text(SHAP_THRESHOLD + 0.002, len(df) - 1,
            f"阈值={SHAP_THRESHOLD}", fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q1_feature_importance.png"), dpi=300)
    plt.close()
    print("  → q1_feature_importance.png")


def plot_predictions_vs_actual():
    """预测 vs 实际散点图（最后一折）"""
    print("  绘制预测 vs 实际...")

    X = np.load(OUT_X_ALL).astype(np.float64)
    y_trans = np.load(OUT_Y_ALL).astype(np.float64)
    lambda_ntu = joblib.load(OUT_LAMBDA_NTU)

    selected_idx_path = os.path.join(OUTPUT_DIR, "selected_indices.npy")
    if os.path.exists(selected_idx_path):
        selected_idx = np.load(selected_idx_path)
        X = X[:, selected_idx]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    # 取最后一折
    splits = list(tscv.split(X))
    tr_idx, val_idx = splits[-1]

    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr_t, y_val_t = y_trans[tr_idx], y_trans[val_idx]
    y_val = boxcox_inverse(y_val_t, lambda_ntu)

    models = {
        "XGBoost": XGBRegressor(**XGB_PARAMS),
        "LightGBM": LGBMRegressor(**LGB_PARAMS),
        "RandomForest": RandomForestRegressor(**RF_PARAMS),
    }

    colors = {"XGBoost": "#E63946", "LightGBM": "#457B9D",
              "RandomForest": "#2A9D8F"}

    fig, ax = plt.subplots(figsize=(8, 6))

    # 理想线
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
            "k--", alpha=0.3, label="理想 y=x")

    for name, model in models.items():
        model.fit(X_tr, y_tr_t)
        y_pred_t = model.predict(X_val)
        y_pred = boxcox_inverse(y_pred_t, lambda_ntu)

        ax.scatter(y_val, y_pred, s=15, alpha=0.5,
                   color=colors[name], label=f"{name}")

    # 集成
    preds = {}
    for name, model in models.items():
        preds[name] = boxcox_inverse(model.predict(X_val), lambda_ntu)

    var_means = {n: np.var(preds[n]) for n in models}
    total = sum(1.0 / max(v, 1e-6) for v in var_means.values())
    weights = {n: (1.0 / max(v, 1e-6)) / total for n, v in var_means.items()}

    y_ens = sum(weights[n] * preds[n] for n in models)
    ax.scatter(y_val, y_ens, s=25, alpha=0.7,
               color="black", marker="D", label="集成(倒方差)")

    ax.set_xlabel("实际 NTU", fontsize=12)
    ax.set_ylabel("预测 NTU", fontsize=12)
    ax.set_title("Q1: 预测值 vs 实际值 (最后一折验证集)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q1_pred_vs_actual.png"), dpi=300)
    plt.close()
    print("  → q1_pred_vs_actual.png")


def main():
    print("=" * 60)
    print("  step1.5 — Q1 可视化")
    print("=" * 60)

    # 加载数据
    X = np.load(OUT_X_ALL).astype(np.float64)
    y = np.load(OUT_Y_ALL).astype(np.float64)
    feature_names = list(np.load(OUT_FEATURE_NAMES, allow_pickle=True))

    # 1. SHAP beeswarm
    plot_shap_beeswarm(X, y, feature_names)

    # 2. 特征重要性柱状图
    plot_feature_importance()

    # 3. 预测 vs 实际
    plot_predictions_vs_actual()

    print(f"\n{'='*60}")
    print(f"  [DONE] 图表输出至 {FIG_DIR}/")
    print(f"    q1_shap_beeswarm.png")
    print(f"    q1_feature_importance.png")
    print(f"    q1_pred_vs_actual.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
