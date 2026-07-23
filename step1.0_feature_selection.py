"""
step1.0_feature_selection.py — SHAP + Permutation 双重特征筛选
================================================================
输入：X_all.npy, y_all.npy, feature_names.npy
输出：output/feature_importance.csv
方法：
  1. XGBoost 基模型 → SHAP TreeExplainer → φ_j（Shapley 全局重要性）
  2. Permutation Importance ×10 → Ĩ_j（信息破坏后性能下降）
  3. I_j^robust = √(φ_j_norm × Ĩ_j_norm)（几何平均融合）
  4. 保留 I_j^robust > THRESHOLD 的特征
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from step0_config import *

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("  step1.0 — SHAP + Permutation 特征筛选")
    print("=" * 60)

    # --- 加载数据 ---
    X = np.load(OUT_X_ALL).astype(np.float64)
    y = np.load(OUT_Y_ALL).astype(np.float64)
    feature_names = list(np.load(OUT_FEATURE_NAMES, allow_pickle=True))

    print(f"\n[输入] X.shape={X.shape}, y.shape={y.shape}")
    print(f"[输入] 特征维度={len(feature_names)}")

    # --- 1. 训练 XGBoost ---
    print("\n[1/3] 训练 XGBoost...")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X, y)

    # --- 2. SHAP 重要性 ---
    print("[2/3] 计算 SHAP 重要性...")
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    # 全局重要性 = 绝对 SHAP 值均值
    phi_j = np.abs(shap_values).mean(axis=0)
    phi_j_norm = phi_j / (phi_j.sum() + EPS)

    # --- 3. Permutation 重要性 ×10 ---
    print("[3/3] 计算 Permutation 重要性 (×10)...")
    perm = permutation_importance(
        xgb, X, y,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    I_j = perm.importances_mean        # 正值 = RMSE 增大 = 重要
    I_j = np.maximum(I_j, 0)           # 负值→0（无信息特征）
    I_j_norm = I_j / (I_j.sum() + EPS)

    # --- 4. 鲁棒融合 ---
    I_robust = np.sqrt(phi_j_norm * I_j_norm)

    # --- 5. 筛选 ---
    selected = I_robust > SHAP_THRESHOLD

    # --- 6. 输出表格 ---
    df_importance = pd.DataFrame({
        "rank": range(1, len(feature_names) + 1),
        "feature": feature_names,
        "shap_phi": phi_j.round(6),
        "shap_norm": phi_j_norm.round(4),
        "perm_I": I_j.round(6),
        "perm_norm": I_j_norm.round(4),
        "robust_score": I_robust.round(4),
        "selected": selected,
    }).sort_values("robust_score", ascending=False).reset_index(drop=True)
    df_importance["rank"] = range(1, len(df_importance) + 1)

    out_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    df_importance.to_csv(out_path, index=False, encoding="utf-8-sig")

    # --- 终端输出 ---
    print(f"\n{'='*60}")
    print(f"  筛选结果: {selected.sum()}/{len(feature_names)} 个特征保留\n")
    print(f"  {'Rank':<6} {'特征':<28} {'SHAP_norm':<12} {'Perm_norm':<12} {'Robust':<10} {'保留'}")
    print(f"  {'-'*70}")
    for _, row in df_importance.head(20).iterrows():
        flag = "Y" if row["selected"] else "N"
        print(f"  {row['rank']:<6} {row['feature']:<28} "
              f"{row['shap_norm']:<12.4f} {row['perm_norm']:<12.4f} "
              f"{row['robust_score']:<10.4f} {flag}")
    print(f"\n  [输出] {out_path}")
    print(f"{'='*60}")

    # 返回选中的特征索引（供后续使用）
    selected_indices = np.where(selected)[0]
    np.save(os.path.join(OUTPUT_DIR, "selected_indices.npy"), selected_indices)
    print(f"\n  selected_indices.npy → {len(selected_indices)} 个索引")


if __name__ == "__main__":
    main()
