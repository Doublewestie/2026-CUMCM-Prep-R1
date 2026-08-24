# INDEX.md — 论文公式→代码→文档全映射

> 创建: 2026-07-24 | 最后更新: 2026-07-27 (sum_13 口径统一, 新增诚实验证工具)

---

## 文档路径索引

### specs (设计文档)

| 文档 | 覆盖范围 |
|------|------|
| `Code/docs/specs/2026-07-23-architecture-design.md` | 原始双源RF架构设计 (参考) |
| `Code/docs/specs/2026-07-24-Q1三级分层灰箱-design.md` | Q1三级灰箱方案 (当前方案) |

### sums (决策历史, `Code/docs/sums/`)

| Sum | 主题 | 关键结论 |
|:---:|------|------|
| sum_1 | 题目分析与建模方案 | 选B题, 双源RF元学习器 |
| sum_2 | F/RIDE排除决策 | 六项独立证据排除F/RIDE |
| sum_3 | Q1 XGBoost原始方案结果 | R²=0.34, FILT_NTU#1, 高值区低估40%+ |
| sum_4 | Q2时滞估计+TCN | TCN R²=-0.15, AR(6)R²=0.52 |
| sum_4b | 灰箱重构+双模阈值 (队友版) | V4_DualMode R²=0.53, 阈值0.15 |
| **sum_5_Q1** | **Q1三级分层灰箱 (当前方案)** | **CSTR NTU R²=0.727 (旧全量), T3 η_coag#1** |
| sum_5_Q2 | Q2时滞验证与最终结论 (队友) | 事件CCF中水位τ=4h ✅ |
| sum_6_Q1 | Q1阈值敏感性分析 | 0.05/0.15 数据驱动验证为全局最优 |
| sum_6_Q2 | Q2物理结构化时滞辨识+闭环掩蔽 (队友) | Langmuir 7参数扫τ全失效 |
| **sum_7** | **A线CSTR物理模型细化全历程** | **分tier A: in-sample R²=0.807, A={400,250,30}+平衡(6.09/44)** |
| **sum_8** | **Q2 log(FILT) AR(6)+RidgeCV 终版** | **log-AR(6) CV R²=0.6955, tau=4h/6h/2h** |
| sum_9 | Q4三维风险评分与四级分类 | 3D评分+分区Jenks (回顾口径: 捕获率88.76%) |
| sum_10 | 项目清理与重构 | 统一数据加载器, TCN代码移除 |
| sum_11 | 模型改进探索全历程 | Q2/Q3五次尝试全失败, 闭环掩蔽效应五重证据链 |
| sum_11_Q3 | Q3融合策略对比与模型收敛 (队友) | 双RF / 线性融合 / ablations |
| **sum_12** | **Q3四层条件路由最终方案** | **A/B/C_strong/C_weak; ⚠️ 0.602 为 oracle 口径 (FILT真值已知)** |
| **sum_13** | **诚实验证与数字口径统一 (2026-07-27)** | **⚠️ 论文唯一数字源: results/number_census.csv; Q1=0.807/0.737, Q3=0.485/0.617, Q4 双口径** |

### Reference/sums/ (方法论学习, agent面向)

| Sum | 主题 | 核心方法 |
|:---:|------|------|
| sum_1 | Q1特征筛选+XGBoost学习总结 | SHAP+Permutation融合, 树模型对比 |
| sum_2 | Q2时滞估计+TCN失败学习总结 | CCF/MIC/TE全失效, 深度学习vs AR |
| sum_3 | Q1三级分层灰箱学习总结 | CSTR物理模型, 三级分区策略 |
| sum_4 | Q2双模阈值诊断学习总结 | Jenks/CorrBreak/GMM, 操作员反馈 |
| sum_5 | Q3-Q5方法论前瞻 | 已验证发现+约束+教训清单 |
| sum_6 | Q1阈值敏感性分析 | 离散分布特征, 相关性断层扫描 |
| sum_6_Q2 | Q2伪数据验证与Langmuir模型学习 (队友) | 伪数据τ识别验证方法正确性 |
| **sum_7** | **A线CSTR物理模型细化** | **分tier A发现, N-CSTR/Fr/加速度消融, 5项失败+1项突破** |
| **sum_8** | **B线负反馈控制回路探索** | **Granger/IRF/eta 隐藏反馈/状态空间/三项修正** |
| sum_10 | 闭环掩蔽效应五重证据链 | Agent认知: 五次改进全历程 + 失败模式分类 + 设计原则 |
| **sum_11** | **Q3四层条件路由方法论** | **Q1→Q3迁移验证, γ阻尼, C_th扫描, 失败模式分类** |

### 论文材料 (docs/materials/, 2026-07-27 初始化)

| 文档 | 内容 |
|:---:|------|
| `docs/materials/00_素材总览.md` | 材料区唯一根入口（类别登记/核心发现速览/口径红线） |
| `docs/materials/02_核心文档/口径声明大全.md` | M1-M4 修正历史 + 主/伴随口径 + 数字黑名单 |
| `docs/materials/02_核心文档/数字总表.md` | 派生自 results/number_census.csv（论文引用数字） |
| `docs/materials/02_核心文档/路线叙事_要点式.md` | 六转折点故事线 |
| `docs/materials/02_核心文档/图素材清单.md` | 10 图优先级 + 图注三件套 |
| `docs/materials/02_核心文档/论文大纲.md` | 九章结构 + 摘要三段式 |
| `docs/materials/02_核心文档/Q1章节初稿.md` | Q1 章节初稿（评审自审记录） |
| `docs/materials/02_核心文档/Q2章节初稿.md` | Q2 章节初稿（评审自审记录） |

### 论文材料 (docs/)

| 文档 | 内容 |
|:---:|------|
| `Code/docs/代码手→论文手交接说明.md` | 创新点★★★10条+叙事结构+图表优先级+数字引用 (论文手交接) |

### logs (`Code/docs/logs/`)

| Log | 覆盖期间 | 主题 |
|:---:|------|------|
| latest_0~12 | 项目初始化~Q2 | 见旧版 INDEX (sum_10 记录) |
| latest_13 | Q2/Q3改进探索 | 8次尝试全部记录 |
| latest_14 | 命名规范审查 | math-name 全项目审计 |
| latest_15 | Q3终局开发 | 四层条件路由收敛, 文档整理 |
| **latest_16** | **遗留收编 (2026-07-27)** | **诚实CV验证 + 口径统一 + TimesFM收尾 + FAIL归档** |

---

## 代码文件索引

### 共享基础设施

| 文件 | 功能 |
|------|------|
| `step0_config.py` | 全局参数: 灰箱配置+三级参数+兼容常量 |
| `step0_preprocess.py` | 数据清洗(Format A/B), ~12维精简特征 |
| `step1_shared.py` | 共享: 数据加载+灰箱函数+评估工具 |

### Q1 三级分层灰箱方案 (当前, step1.x)

| 文件 | 功能 | 行数 |
|------|------|:---:|
| `step1.0_tier_classifier.py` | 三级分类器: C1(T1 vs rest)+C2(T2 vs T3), Logistic | ~110 |
| `step1.1_tier1_noise.py` | T1(≤0.05): 经验频率采样+JS散度验证 | ~100 |
| `step1.2_tier2_experiment.py` | T2(0.05~0.15): 经验分布 vs 对数压缩灰箱双路径 | ~170 |
| `step1.3_tier3_greybox.py` | T3(>0.15): CSTR+线性反馈+τ₁可学习+λ₃扫参 | ~280 |
| `step1.4_feature_importance.py` | T3特征重要性: SHAP+Permutation, η_coag#1 | ~100 |
| `step1.5_visualization.py` | 三级可视化 + CSTR预测图 (分tier A) | ~135 |
| `step1.6_cstr_refinement.py` | N-CSTR/延迟/变面积/管壁释放 消融 (探索档案) | ~330 |
| `step1.7_final_cstr.py` | **分tier A + 四阶段扫参 (权威 in-sample R²=0.8072)** | ~400 |
| `step1.7+_cstr_figures.py` | Q1 图表生成 | — |
| **`step1.7+_tscv_validation.py`** | **⚠️ Q1 诚实 TS-CV (sum_13 新增): 固定物理参数(先验)+5折TS-CV, 默认 RL_med=6.09/Q_med=44 → TS-CV R²=0.7369; 参数可配置 (--rl-med/--q-med)** | ~230 |
| `step1.9_pysical_reconstruct.py` | 物理重构 (已淘汰, R²=-0.29) | — |
| `step1.9+_summary_report.py` | 全流程汇总表 (in-sample 0.8072) | ~116 |
| `archive/step1.10_learnable_beta.py` | NN 学习可学习 β/θ (FAIL: 0.588<0.689, 2026-07-27 归档) | ~380 |

**核心结果 (step1.7_final)**: A = {T1:400, T2:250, T3:30} + Balance (RL_med=6.09/Q_med=44), in-sample R²=**0.8072**, **诚实 TS-CV R²=0.7369** (step1.7+_tscv_validation). 论文报双口径.

### Q2 双模诊断方案 (队友, step2.x)

| 文件 | 功能 | 行数 |
|------|------|:---:|
| `step2.0_greybox_diagnostic.py` | Jenks/CorrBreak/GMM三法阈值检测 | ~245 |
| `step2.1_stress_tcn.py` | 应力区2层TCN, 滞后权重提取 | ~340 |
| `step2.1+_closed_loop_decompose.py` | 操作员策略OLS分解(失败) | ~220 |
| `step2.2_baseline_comparison.py` | 应力区AR(6)/ARMAX基线 | ~90 |
| `step2.3_comfort_report.py` | 舒适区统计报告 | ~55 |
| `step2.5_logar_final.py` | **log-AR(6)+RidgeCV 最终模型 (CV R²=0.6955)** | — |
| `step2.5_visualization.py` | 双模分区+操作员策略图 | ~140 |
| `step2.7_generate_figures.py` | Q2 图表生成 | — |
| `step2_shared.py` | 共享: 数据加载 + AR6Predictor 类 (备用, 全量系数, 勿用于诚实CV) | — |

### Q3 四层条件路由方案 (step3.8)

| 文件 | 功能 | 行数 |
|------|------|:---:|
| `step3.8_final_stratified.py` | **Q3 终版 (oracle 口径): 四层分类+全局参数+CV+消融表; CV R²=0.6017 (FILT真值已知+偏置表); 2026 Feb 预测 (2025同日期proxy)** | ~320 |
| **`step3.8+_forecast_cstr.py`** | **⚠️ Q3 诚实部署口径 (sum_13 新增): AR(6)预测FILT→CSTR链; forecast R²=0.4853, oracle 0.6165, 代价0.131** | ~340 |
| `step3.9_diagnostics.py` | 路由/损失诊断 (blend/residual/log/huber 全负) | — |
| `archive/step3.9_learnable_routing.py` | NN softmax 连续混合 (FAIL: 0.09<if-else 0.617, 2026-07-27 归档) | — |

**核心结果 (sum_13 口径)**: oracle (FILT真值) CV R²=**0.617**; **部署口径 (AR(6) FILT) CV R²=0.485**; 2参数+1阈值 (α=0.34, γ=0.25, C_th=1.0)

**旧版 (archive/)**:
| 文件 | 功能 |
|------|------|
| `archive/step3.0_pipeline_main.py` | 原CSTR两段链式预测 |
| `archive/step3.1_residual_train.py` | 原残差RF+物理RF训练 |
| `archive/step3.2_forecast_2026.py` | 原2026预测 |
| `archive/step3.3_validation.py` | 原CV验证脚本 |
| `archive/step3.4_sensitivity.py` | Sobol敏感性分析 |
| `archive/step3.5_visualization.py` | 原Q3可视化 |

### Q4-Q5 (已完成)

| 文件 | 功能 | 状态 |
|------|------|:---:|
| `step4.0_risk_scoring.py` | 三维风险评分+熵权法; **USE_ACTUAL_NTU 双模式 (前瞻CSTR预测NTU / 回顾实际NTU)** | ✅ |
| `step4.1_jenks_classification.py` | 分区独立Jenks → 统一四级 | ✅ |
| `step4.2_dual_validation.py` | FCE+Jenks Kappa/Bootstrap/事件回溯 | ✅ |
| `step4.4_predict_2026.py` | 2026-03 逐日风险分类 → Excel | ✅ |
| `step4.5_visualization.py` | Q4 图表 | ✅ |
| `step5.0_ablation.py` | Q2 log-AR 消融 (8配置) | ✅ |
| `step5.1_timesfm_baseline.py` | **TimesFM 2.5 零样本基线 (2026-07-27 收尾: 预测均值~0.094, 失败证毕; 需在 mathorcup 环境运行)** | ✅ |
| `step5.2_final_summary.py` | 终版汇总 | ✅ |
| `step5.3_package_figures.py` | 论文图表 (300dpi) | ✅ |

### 诚实验证工具 (sum_13 新增, 论文数字源)

| 文件 | 功能 |
|------|------|
| `results/number_census.csv` | **论文唯一数字总表 (每行含指标/口径/参数/复现命令/文档引用)** |
| `step1.7+_tscv_validation.py` | Q1 诚实 TS-CV (R²=0.737 @ 6.09/44) |
| `step3.8+_forecast_cstr.py` | Q3 诚实部署口径 (forecast R²=0.485) |
| `step3.9_diagnostics.py` | Q3 路由/损失诊断 (routing 最优) |

---

## 关键数据流

```
clean_data.csv (4375行)
     ↓
  ┌─ step1_shared.py ─→ step1.0 → tier_labels.npy
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
| F8 | Q2最终: log-AR(6)+RidgeCV R2=0.6955 CV | TS-CV, FILT空间 | step2.5_logar_final.py |
| F9 | 清池A为regime-dependent: T1→400, T3→30 | 分tier扫参 + 40组联合扫 | step1.7+ |
| F10 | **Q1诚实TS-CV=0.737 (in-sample 0.807), 非0.732** | step1.7+_tscv_validation (6.09/44) | sum_13 |
| F11 | **Q3部署口径=0.485, oracle=0.617 (0.602为oracle+偏置)** | step3.8+_forecast_cstr | sum_13 |
| F12 | **NN可学习β/θ/路由全部FAIL (0.588/0.09 < 手调)** | step1.10/step3.9_learnable (归档) | sum_13 |
| F13 | **TimesFM零样本基线失败 (均值~0.094 < 实际0.21)** | step5.1 (mathorcup环境) | sum_13 |
