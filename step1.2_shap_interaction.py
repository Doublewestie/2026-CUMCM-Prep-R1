"""
step1.2_shap_interaction.py — L5 交互特征二阶段检验
=====================================================
输入：X_all.npy, feature_names.npy
方法：
  阶段一：物理合理性（人工判断，L5_CANDIDATES 已预筛过）
  阶段二：SHAP 交互效应显著性（统计检验）
    - 计算 shap_interaction_values
    - 对每个 L5 候选特征，提取其与所有其他特征的交互效应均值
    - 若交互效应不显著（低于阈值），建议删除
输出：output/l5_validation_report.csv
"""

import numpy as np
import pandas as pd
import shap, joblib, os, warnings
from xgboost import XGBRegressor
from step0_config import *

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("  step1.2 — L5 交互特征二阶段检验")
    print("=" * 60)

    X = np.load(OUT_X_ALL).astype(np.float64)
    y = np.load(OUT_Y_ALL).astype(np.float64)
    feature_names = list(np.load(OUT_FEATURE_NAMES, allow_pickle=True))

    print(f"\nX.shape={X.shape}, 特征数={len(feature_names)}")

    # 找到 L5 特征在特征列表中的位置
    l5_indices = {}
    for cand in L5_CANDIDATES:
        for i, name in enumerate(feature_names):
            if name.upper() == cand.upper():
                l5_indices[cand] = i
                break

    if not l5_indices:
        print("\n[WARNING] 未找到任何 L5 候选特征在特征矩阵中")
        return

    print(f"\n找到 {len(l5_indices)} 个 L5 特征:")
    for cand, idx in l5_indices.items():
        print(f"  {cand} (索引 {idx})")

    # --- 阶段一：物理合理性（已在本文件对应的文档中论证，此处仅记录） ---
    physics_check = {
        "PI_load":    {"meaning": "污染物总通量 = RW_NTU × RW_FLOW", "pass": True},
        "GAMMA_alum": {"meaning": "混凝剂充足度 = ALUM / RW_NTU",   "pass": True},
        "PSI_alum":   {"meaning": "总矾投加速率 = ALUM × RW_FLOW",  "pass": True},
        "OMEGA_night": {"meaning": "夜间原水浊度突变检测",          "pass": True},
    }

    # --- 阶段二：SHAP 交互显著性 ---
    print("\n[阶段二] 计算 SHAP 交互效应...")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X, y)

    explainer = shap.TreeExplainer(xgb)
    # shap_interaction: (n_samples, n_features, n_features)
    shap_interaction = explainer.shap_interaction_values(X)

    interaction_threshold = 0.005  # 交互效应显著性阈值

    results = []
    for cand, idx in l5_indices.items():
        # 对角线 = 主效应，非对角线 = 交互效应
        # 取该特征列的所有非对角线值的绝对值均值
        col_interactions = np.abs(shap_interaction[:, idx, :]).mean(axis=0)
        # 去掉自交互（对角线）
        diag_val = col_interactions[idx]
        other_mask = np.ones(col_interactions.shape, dtype=bool)
        other_mask[idx] = False
        avg_interaction = col_interactions[other_mask].mean()

        passed = avg_interaction > interaction_threshold

        results.append({
            "feature": cand,
            "index": idx,
            "physics_pass": physics_check.get(cand, {}).get("pass", True),
            "shap_main_effect": round(float(diag_val), 6),
            "shap_avg_interaction": round(float(avg_interaction), 6),
            "interaction_pass": passed,
            "final_verdict": "保留" if (physics_check.get(cand, {}).get("pass", True) and passed) else "删除",
        })

    df_results = pd.DataFrame(results)

    # 输出
    out_path = os.path.join(OUTPUT_DIR, "l5_validation_report.csv")
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  L5 Interaction Feature Validation Results:\n")
    for _, row in df_results.iterrows():
        print(f"  {row['feature']:<16} | main={row['shap_main_effect']:.6f} "
              f"| inter={row['shap_avg_interaction']:.6f} | {row['final_verdict']}")

    print(f"\n  [OUTPUT] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
