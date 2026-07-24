# Phase Q1 — 实现路径 + 接口约束 + 验收标准

> 2026-07-24

---

## 1. 已完成

| 步骤 | 内容 | 文件 | 验收 |
|------|------|------|:---:|
| Step 0 | 数据清洗(4380→4357行) + L1-L5特征工程(101维) | step0_config.py, step0_preprocess.py | ✅ |
| Step 0 | Format A/B兼容(两套Excel), 缺失列LGBM插补(ALUM R2=0.75) | step0_preprocess.py | ✅ |
| Step 0 | Box-Cox→log1p降级(λ=-0.45→0), TimeSeriesSplit(5折) | step0_preprocess.py | ✅ |
| Step 1.0 | SHAP + Permutation×10 鲁棒融合, 25/101特征保留 | step1.0_feature_selection.py | ✅ |
| Step 1.1 | 渐进式建模: Linear(R2=-0.56)→GAM(0.07)→XGB(0.34)→LGB(0.27)→RF(0.32) | step1.1_model_comparison.py | ✅ |
| Step 1.1 | Ridge显式公式(R2=0.70) + GAM偏依赖(top5, R2=0.51) | step1.1_model_comparison.py | ✅ |
| Step 1.2 | L5交互特征二阶段检验: 4/4不显著 | step1.2_shap_interaction.py | ✅ |
| Step 1.5 | 可视化: SHAP beeswarm + 重要性bar + 预测vs实际 + GAM偏依赖 | step1.5_visualization.py | ✅ |
| Step 5.0 | Q1消融: 8组逐层+重度(FILT_NTU占R2的42%) | step5.0_ablation.py | ✅ |

---

## 2. 待完成

| 任务 | 依赖 | 优先级 |
|------|:---:|:---:|
| Q1.2 2026年三天预测 | step3 2026数据加载 | P0 |
| 消融柱状图可视化 | step5.0结果 | P1 |

---

## 3. 接口约束

| 接口 | 形状 | 说明 |
|------|------|------|
| X_all.npy | (4357, 101) | 全量特征矩阵, float32 |
| y_all.npy | (4357,) | log1p变换后NTU |
| selected_indices.npy | (25,) | step1.0筛选后的特征索引 |
| lambda_ntu.pkl | scalar(0.0) | log1p反变换用 |

---

## 4. 验收标准

| 指标 | 目标 | 实际 | 判定 |
|------|:---:|:---:|:---:|
| 数据行数 | 4380→≥4300(清洗后) | 4357 | ✅ |
| XGBoost R2 | ≥0.25 | 0.34 | ✅ |
| LGBM R2 | ≥0 | 0.27(调参后:lr=0.01+L1L2+采样) | ✅ |
| 消融R2递进 | L1<L1+L2<Full | -0.14<0.30<0.36 | ✅ |
| 函数公式R2 | ≥0.5 | 0.70(Ridge) | ✅ |
| TimeSeriesSplit严格性 | 训练永远在过去 | 5折时序递增 | ✅ |
| FILT_NTU主导性 | 删全部FILT→R2显著下降 | 0.35→0.20(重度消融) | ✅ |
| L2+结论 | 幂次项对XGB无效 | ΔR2=-0.012 | ✅ |
| L3结论 | 历史滞后对树模型无增益 | ΔR2=+0.005 | ✅ |
| 集成有效性 | 倒方差加权≥最佳单模型 | 否(集成RMSE=0.44>XGB=0.437) | ⚠️ 论文中说明 |

---

*变更记录: 2026-07-24 Q1完成*
