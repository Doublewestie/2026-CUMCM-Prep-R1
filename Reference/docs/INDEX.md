# INDEX.md — 项目全映射

> 论文公式 → 代码文件 → 行号映射（占位，随编码逐步填充）

---

## 全局映射

| 公式/方法 | 代码文件 | 行号 | 状态 |
|------|------|:---:|:---:|
| L1-L5五级特征金字塔 | step0_preprocess.py | 350-445 | ✅ 已实现 |
| Box-Cox变换(log1p降级) | step0_preprocess.py | 327-344 | ✅ 已实现 |
| 物理幂次项展开(L2+) | step0_preprocess.py | 396-434 | ✅ 已实现 |
| SHAP TreeExplainer | step1.0_feature_selection.py | 37-48 | ✅ 已实现 |
| Permutation Importance | step1.0_feature_selection.py | 50-64 | ✅ 已实现 |
| XGBoost/LGBM/RF集成 | step1.1_model_comparison.py | 60-130 | ✅ 已实现 |
| 渐进式建模(Linear→GAM→XGB→Ensemble) | step1.1_model_comparison.py | 89-200 | ✅ 已实现 |
| Ridge显式公式提取 | step1.1_model_comparison.py | 243-295 | ✅ 已实现 |
| GAM偏依赖分析 | step1.1_model_comparison.py | 300-325 | ✅ 已实现 |
| SHAP交互效应 | step1.2_shap_interaction.py | 35-90 | ✅ 已实现 |
| 特征层级消融(8组) | step5.0_ablation.py | 85-130 | ✅ 已实现 |
| FILT_NTU重度消融 | step5.0_ablation.py | 110-118 | ✅ 已实现 |
| MIC计算 | step2.0_time_delay_estimation.py | — | 待实现 |
| 传递熵TE | step2.0_time_delay_estimation.py | — | 待实现 |
| TCN因果膨胀卷积 | step2.1_tcn_dynamic_model.py | — | 待实现 |
| TCN→GRU集成模型 | step3.0_source_a_multivariate.py | — | 待实现 |
| N-BEATS基函数分解 | step3.1_source_b_univariate.py | — | 待实现 |
| TimesFM零样本预测 | step3.1_source_b_univariate.py | — | 待实现 |
| RF元学习器条件推理 | step3.2_meta_feature_matrix.py | — | 待实现 |
| Sobol敏感性(Saltelli) | step3.3_sobol_sensitivity.py | — | 待实现 |
| 三维风险评分 | step4.0_risk_scoring.py | — | 待实现 |
| H5 熵权法 | step4.0_risk_scoring.py | — | 待实现 |
| Jenks自然断点法 | step4.1_jenks_classification.py | — | 待实现 |
| FCE模糊综合评价 | step4.2_dual_validation.py | — | 待实现 |
| Kappa一致性检验 | step4.2_dual_validation.py | — | 待实现 |
| Bootstrap CI | step4.2_dual_validation.py | — | 待实现 |
| 消融矩阵 | step5.0_ablation.py | 85-130 | ✅ Q1消融完成(8组) |
| TimesFM独立基线 | step5.1_timesfm_baseline.py | — | 待实现 |

---

## Phase → 文档路径

| Phase | 文档 | 状态 |
|:---:|------|:---:|
| Q1 | Code/docs/sums/sum_1_题目分析与建模方案.md | ✅ 完成 |
| Q1 | Code/docs/sums/sum_3_Q1实验结果与函数关系.md | ✅ 完成 |
| Q1 | Reference/docs/PhaseQ1/paper_code_summary.md | ✅ 完成 |
| Q1 | Reference/docs/PhaseQ1/reference_analysis.md | ✅ 完成 |
| Q1 | Reference/docs/PhaseQ1/guidance.md | ✅ 完成 |
| Q2 | Reference/docs/PhaseQ2/paper_code_summary.md | 占位 |
| Q2 | Reference/docs/PhaseQ2/reference_analysis.md | 占位 |
| Q2 | Reference/docs/PhaseQ2/guidance.md | 占位 |
| Q3 | Reference/docs/PhaseQ3/paper_code_summary.md | 占位 |
| Q3 | Reference/docs/PhaseQ3/reference_analysis.md | 占位 |
| Q3 | Reference/docs/PhaseQ3/guidance.md | 占位 |
| Q4 | Reference/docs/PhaseQ4/paper_code_summary.md | 占位 |
| Q4 | Reference/docs/PhaseQ4/reference_analysis.md | 占位 |
| Q4 | Reference/docs/PhaseQ4/guidance.md | 占位 |

---

_变更记录: 2026-07-23 初始创建(占位)_
