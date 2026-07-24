# INDEX.md — 论文公式→代码→文档全映射

> 创建: 2026-07-24 | 状态: 框架, 待填充Phase具体内容

---

## 文档路径索引

### specs (设计文档)

| Phase | 文档 |
|------|------|
| Q1+Q2重构 | `Code/docs/specs/2026-07-24-q1-greybox-redesign.md` |
| 原始架构 | `Code/docs/specs/2026-07-23-architecture-design.md` |

### sums (决策历史)

| Sum | 主题 | 关键结论 |
|:---:|------|------|
| sum_1 | 题目分析与建模方案 | 选B题, 双源RF元学习器, 五级特征金字塔 |
| sum_2 | F/RIDE排除决策 | 六项独立证据排除F/RIDE |
| sum_3 | (空缺) | — |
| sum_4 | Q2时滞估计与TCN动态模型 | TCN全量R²=-0.15, AR(6)R²=0.52 |
| sum_5 | 灰箱重构与双模阈值发现 | V4_DualMode R²=0.53, 阈值0.15 |

### logs

| Log | 覆盖期间 |
|:---:|------|
| latest_0 | 项目初始化 |
| latest_1 | Stage 0+1 原始方案 (101维XGBoost) |
| latest_2 | Q1调参+误差分析 |
| latest_3 | Q2时滞估计+TCN+消融 (旧方案) |
| latest_4 | Q1+Q2全面重构 (灰箱+双模) |

---

## 代码文件索引

### Q1 灰箱 (step1+.x)

| 文件 | 公式/功能 | 行数 |
|------|------|:---:|
| `step1+.0_greybox_segment1.py` | FILT(t)=β₁·FILT(t-1)+(1-β₁)·RW_NTU(t)·[1-η] | ~230 |
| `step1+.1_greybox_segment2.py` | NTU(t)=β₂·NTU(t-1)+(1-β₂)·FILT(t) | ~180 |
| `step1+.2_greybox_joint.py` | 4变体对比(V1-V4), 输出最优参数 | ~470 |
| `step1+.5_greybox_output.py` | 函数公式(q1_formula.txt) + 因子分析JSON | ~275 |

### Q2 双模诊断 (step2.x)

| 文件 | 功能 | 行数 |
|------|------|:---:|
| `step2.0_greybox_diagnostic.py` | Jenks/CorrBreak/GMM三法阈值检测 | ~180 |
| `step2.1_stress_tcn.py` | 应力区2层TCN, shift-1约束, 滞后权重 | ~340 |
| `step2.1+closed_loop_decompose.py` | 操作员策略分解尝试(负面结果) | ~220 |
| `step2.2_baseline_comparison.py` | 应力区AR(6)/ARMAX基线 | ~90 |
| `step2.3_comfort_report.py` | 舒适区统计报告 | ~55 |
| `step2.5_visualization.py` | 双模分区+操作员策略图 | ~135 |

### 共享

| 文件 | 功能 |
|------|------|
| `step0_config.py` | 全局参数, 灰箱配置, 特征定义, 向后兼容常量 |
| `step0_preprocess.py` | 数据清洗(Format A/B), 泵count压缩, 15维特征 |

### 废弃 (旧方案)

| 文件 | 废弃原因 |
|------|------|
| `step2.0-_time_delay_estimation.py` | CCF/MIC/TE全部d*=0 |
| `step2.1-_tcn_dynamic_model.py` | 4层TCN R²=-0.15 |
| `step1.0 ~ step1.5` (原始) | 101维XGBoost方案已废弃 |

### 未启动

| 文件 | 功能 | 状态 |
|------|------|:---:|
| `step3.0_source_a_multivariate.py` | Q3源A: TCN→GRU | 空骨架 |
| `step3.1_source_b_univariate.py` | Q3源B: N-BEATS/TimesFM | 空骨架 |
| `step3.2_meta_feature_matrix.py` | Q3: RF元学习器 | 空骨架 |
| `step3.3_sobol_sensitivity.py` | Q3: Sobol敏感性 | 空骨架 |
| `step4.0~step4.5` | Q4三维评分+Jenks+FCE | 空骨架 |
| `step5.0~step5.1` | 跨题消融+TimesFM基线 | 部分完成 |

---

## 关键数据流

```
clean_data.csv → step1+.2 → q1_greybox_params.json (灰箱参数)
                                    ↓
          q1_greybox_ntu_pred.npy → [Q3预测输入]

clean_data.csv → step2.1 → q2_lag_weights.json (滞后权重)
                              ↓
                     [Q3注意力soft prior]

clean_data.csv → step2.0 → theta_params.json (阈值)
                              ↓
                     [Q1+Q3双模切换]
```

## 关键发现编号

| # | 发现 | 验证方式 |
|:---:|------|------|
| F1 | FILT_NTU阈值0.15分隔舒适/应力区 | Jenks + CorrBreak + 分层相关分析 |
| F2 | 舒适区r(FILT,NTU)=0.03, 应力区=0.79 | step2.3_comfort_report |
| F3 | 段1全参数触底(β₁→1, K_m→0, α→0) | step1+.0 L-BFGS-B收敛到边界 |
| F4 | V4_DualMode CV R²=+0.30 vs V1=-0.69 | step1+.2 4变体对比 |
| F5 | 操作员策略R²=0.0067(线性不可表示) | step2.1+closed_loop OLS |
| F6 | ΔFILT应力区不可预测(所有模型R²≤0) | step2.1+2.2 三模型比较 |
