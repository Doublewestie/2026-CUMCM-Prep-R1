# INDEX.md — 论文公式→代码→文档全映射

> 创建: 2026-07-24 | 最后更新: 2026-07-24

---

## 文档路径索引

### specs (设计文档)

| 文档 | 覆盖范围 |
|------|------|
| `Code/docs/specs/2026-07-23-architecture-design.md` | 原始双源RF架构设计 (参考) |

### sums (决策历史, `Code/docs/sums/`)

| Sum | 主题 | 关键结论 |
|:---:|------|------|
| sum_1 | 题目分析与建模方案 | 选B题, 双源RF元学习器 |
| sum_2 | F/RIDE排除决策 | 六项独立证据排除F/RIDE |
| sum_3 | Q1 XGBoost原始方案结果 | R²=0.34, FILT_NTU#1, 高值区低估40%+ |
| sum_4 | Q2时滞估计+TCN | TCN R²=-0.15, AR(6)R²=0.52 |
| sum_4b | 灰箱重构+双模阈值 (队友版) | V4_DualMode R²=0.53, 阈值0.15 |
| **sum_5** | **Q1三级分层灰箱 (当前方案)** | **CSTR NTU R²=0.727, T3 η_coag#1** |

### Reference/sums/ (方法论学习, agent面向)

| Sum | 主题 | 核心方法 |
|:---:|------|------|
| sum_1 | Q1特征筛选+XGBoost学习总结 | SHAP+Permutation融合, 树模型对比 |
| sum_2 | Q2时滞估计+TCN失败学习总结 | CCF/MIC/TE全失效, 深度学习vs AR |
| sum_3 | Q1三级分层灰箱学习总结 | CSTR物理模型, 三级分区策略 |
| sum_4 | Q2双模阈值诊断学习总结 | Jenks/CorrBreak/GMM, 操作员反馈 |
| sum_5 | Q3-Q5方法论前瞻 | 已验证发现+约束+教训清单 |

### logs (`Code/docs/logs/`)

| Log | 覆盖期间 | 主题 |
|:---:|------|------|
| latest_0 | 项目初始化 | 初始日志 |
| latest_1 | Stage 0+1 | 数据清洗+101维XGBoost |
| latest_2 | Q1调参 | 误差分析+LGBM调参 |
| latest_3 | Q2旧方案 | TCN+消融, 统计方法全失败 |
| latest_4 | Q1三级灰箱重构 | 三级分区+CSTR+反馈扫参 |

---

## 代码文件索引

### 共享基础设施

| 文件 | 功能 |
|------|------|
| `step0_config.py` | 全局参数: 灰箱配置+三级参数+兼容常量 |
| `step0_preprocess.py` | 数据清洗(Format A/B), ~12维精简特征 |
| `q1_data_utils.py` | 共享: 数据加载+灰箱函数+评估工具 |

### Q1 三级分层灰箱方案 (当前, step1.x)

| 文件 | 功能 | 行数 |
|------|------|:---:|
| `step1.0_tier_classifier.py` | 三级分类器: C1(T1 vs rest)+C2(T2 vs T3), Logistic | ~110 |
| `step1.1_tier1_noise.py` | T1(≤0.05): 经验频率采样+JS散度验证 | ~100 |
| `step1.2_tier2_experiment.py` | T2(0.05~0.15): 经验分布 vs 对数压缩灰箱双路径 | ~170 |
| `step1.3_tier3_greybox.py` | T3(>0.15): CSTR+线性反馈+τ₁可学习+λ₃扫参 | ~280 |
| `step1.4_feature_importance.py` | T3特征重要性: SHAP+Permutation, η_coag#1 | ~100 |
| `step1.5_visualization.py` | 三级可视化: 分布+特征重要性+CSTR预测+T1分布 | ~130 |
| `run_q1_full.py` | 全流程汇总表 | ~130 |

**核心结果**: NTU(t)=β₂·NTU(t-1)+(1-β₂)·FILT(t), 全量R²=**0.727** (vs 原XGBoost 0.34)

### Q2 双模诊断方案 (队友, step2.x)

| 文件 | 功能 | 行数 |
|------|------|:---:|
| `step2.0_greybox_diagnostic.py` | Jenks/CorrBreak/GMM三法阈值检测 | ~245 |
| `step2.1_stress_tcn.py` | 应力区2层TCN, 滞后权重提取 | ~340 |
| `step2.1+_closed_loop_decompose.py` | 操作员策略OLS分解(失败) | ~220 |
| `step2.2_baseline_comparison.py` | 应力区AR(6)/ARMAX基线 | ~90 |
| `step2.3_comfort_report.py` | 舒适区统计报告 | ~55 |
| `step2.5_visualization.py` | 双模分区+操作员策略图 | ~140 |

### 未启动 (Q3-Q5)

| 文件 | 功能 | 状态 |
|------|------|:---:|
| `step3.0_source_a_multivariate.py` | Q3源A: TCN→GRU | 空骨架 |
| `step3.1_source_b_univariate.py` | Q3源B: N-BEATS/TimesFM | 空骨架 |
| `step3.2_meta_feature_matrix.py` | Q3: RF元学习器 | 空骨架 |
| `step3.3_sobol_sensitivity.py` | Q3: Sobol敏感性 | 空骨架 |
| `step3.5_visualization.py` | Q3: 图表 | 空骨架 |
| `step4.0~step4.5` | Q4: 风险评分+Jenks+FCE | 空骨架 |
| `step5.0~step5.1` | 跨题消融+TimesFM基线 | 部分完成 |

---

## 关键数据流

```
clean_data.csv (4375行)
     ↓
  ┌─ q1_data_utils.py ─→ step1.0 → tier_labels.npy
  │       ↓              step1.1 → tier1_report.json
  │    tier_classifier    step1.2 → tier2_comparison.json
  │       ↓              step1.3 → tier3_sweep_results.csv
  │  T1/T2/T3 分配        step1.4 → tier3_factor_importance.csv
  │       ↓
  │  T3: CSTR段2
  │  NTU(t)=β₂·NTU₋₁+(1-β₂)·FILT(t)  R²=0.727
  │
  └─ step2.0  → theta_params.json (阈值θ=0.15)
     step2.1  → q2_lag_weights.json (滞后权重)
     step2.1+ → q2_operator_policy.json (闭环分解)
```

## 关键发现编号

| # | 发现 | 验证方式 | 来源 |
|:---:|------|------|:---:|
| F1 | FILT_NTU三级分界: 0.05 / 0.15 | 经验分布+相关分析 | step1.0/1.1 |
| F2 | CSTR适用于NTU, 不适用于FILT | 手动测试+扫参: NTU R²=0.727 | step1.3 |
| F3 | η_coag为T3应力区#1因素 | SHAP+Perm (Robust=0.335) | step1.4 |
| F4 | τ₁=4h (RW_NTU→FILT时滞) | softmax可学习, 峰值在lag=2 | step1.3 |
| F5 | T1分布离散: 仅4个值 {0.02,0.03,0.04,0.05} | 经验频率: 0.4%/22.4%/51.8%/25.4% | step1.1 |
| F6 | 舒适区r(FILT,NTU)=0.03, 应力区=0.79 | 分层相关分析 | step2.3 |
| F7 | 操作员策略R²=0.0067(线性不可表示) | OLS分解 | step2.1+ |
