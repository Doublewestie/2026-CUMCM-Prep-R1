# Sum 13: 诚实验证与数字口径统一

> 2026-07-27 | 收编 cb255fb 之后的遗留探索, 统一全项目数字口径, 为论文写作提供唯一数字源

---

## 1. 问题起点

2026-07-26 最后一次提交 (cb255fb) 之后, 工作区出现了 5 个未入库探索脚本 + 2 个已修改脚本, 且文档体系 (migration_prompt/INDEX/PLAN) 未覆盖。同时发现三处数字口径冲突, 直接威胁论文可信度:

| 冲突 | 表现 |
|---|---|
| Q1: "CV 5折=0.732" 口径过时 | 0.732 实为**旧单参数 A=141.3 的 CV** (sum_5), 分 tier A + balance 的诚实 TS-CV 从未被文档记录 |
| Q1: 平衡检测器参数分裂 | step1.7_final/sum_7 用 **RL_med=6.09, Q_med=44** (R²=0.8072); step3.8/CONSTITUTION 用 **8.0/48** (Q3 优化) |
| Q3: 0.602 是 oracle 口径 | step3.8 的 CSTR 链使用**真实 FILT(t)** (传递函数口径), 部署时 7:00-19:00 的 FILT 未知 |
| Q4: 前瞻 vs 回顾 | step4.0 新前瞻模式 (CSTR预测NTU) 与 sum_9 回顾模式 (实际NTU) 指标差异巨大 |

## 2. 方案与过程

### 2.1 审计 5 个探索脚本 (均复现验证)

| 脚本 | 内容 | 结论 |
|---|---|---|
| `step1.7+_tscv_validation.py` | Q1 诚实 TS-CV (固定物理参数, 仅验证折评估) | ✅ 正式验证工具, 保留 |
| `step1.10_learnable_beta.py` | NN 学习可学习 β/θ (2-5 组实验) | ❌ FAIL: NN 0.588 < 手调 0.689 → **归档** |
| `step3.8+_forecast_cstr.py` | Q3 AR(6) 预测 FILT → CSTR 链 (oracle/forecast/persist 三模式) | ✅ 部署口径验证, 保留 |
| `step3.9_diagnostics.py` | 路由/损失诊断 (blend/residual/log/huber) | ✅ 诊断证据, 保留 |
| `step3.9_learnable_routing.py` | NN softmax 连续混合权重 | ❌ FAIL: NN 0.09 < if-else 0.617 → **归档** |
| `step2_shared.py` (+47) | AR6Predictor 类 (全量训练系数) | 备用工具, 保留; **不可用于诚实 CV** (非 per-fold) |
| `step4.0_risk_scoring.py` (+101) | USE_ACTUAL_NTU 前瞻模式 (评审 Issue#3 循环论证修复) | ✅ 保留, 双模式 |

两个 FAIL 与 sum_11"闭环掩蔽效应"叙事一致: 可学习参数化 (β/θ/混合权重) 均无法超越手调物理参数。

### 2.2 Q1 口径统一 (决策 D1: 采用 6.09/44, 与已提交 0.8072 同源)

`step1.7+_tscv_validation.py` 参数可配置化 (`--rl-med`/`--q-med`), 默认 6.09/44:

| 口径 | RL_med/Q_med | in-sample R² | 5-fold TS-CV R² |
|---|---|---|---|
| **论文主口径** | **6.09 / 44** | **0.8023** | **0.7369 ± 0.1009** |
| Q3 优化备选 | 8.0 / 48 | 0.7553 | 0.6887 ± 0.1773 |

- in-sample 0.8023 vs step1.9+ 报告 0.8072 差 0.005 (边界处理微差), 论文报 step1.7 权威值 0.807
- **"CV 5折=0.732" 全项目更正为 0.737** (6.09/44 口径诚实 TS-CV)

### 2.3 Q3 口径统一 (决策 D2: forecast 0.485 为主口径, oracle 0.617 为上限)

`step3.8+_forecast_cstr.py` 三模式 5-fold TS-CV:

| 模式 | R² | 含义 |
|---|---|---|
| Oracle (真 FILT + persist CW/Q) | **0.6165 ± 0.2715** | 信息上限 (FILT 已知) |
| **Forecast (AR(6) FILT + persist CW/Q)** | **0.4853 ± 0.3044** | **可部署主口径** |
| Persistence (全天持久) | -3.3488 | 仅诊断, 涉及负值极端; **与 Q1 单步 persist 0.607 不可比** |

- FILT 预测代价 = 0.6165 - 0.4853 = **0.1312** — 论文分解点: CSTR 链共享 Q1 物理, 误差主要来自 FILT 未来不可知
- step3.8 (已提交) 的 0.602 保留为"oracle+偏置表"参考, 论文主口径以 step3.8+ forecast 为准
- persistence 为 step3.8+ 自定义"全天持续" (7 步), 与 Q1 单步 persist 定义不同 → 论文不可混用

### 2.4 Q4 前瞻 vs 回顾 (决策 D3: 双口径并列, 主报前瞻)

| 指标 | 前瞻 (CSTR预测NTU) | 回顾 (实际NTU, sum_9) |
|---|---|---|
| 熵权 w1/w2/w3 | 0.1273/0.6129/0.2599 | 0.087/0.806/0.106 |
| 超标捕获率 | **57.99%** | 88.76% |
| 虚警率 | 1.21% | 0% |
| Kappa 全量 / 舒适区 | 0.2961 / 0.8774 | 0.522 / 0.900 |
| 平均提前预警 | 9.3 步 | 10.1 步 (20h) |
| 2026-03 分级 | G1×1, G2×11, G3/G4×0 | 同 (G1×1, G2×11) |

- 前瞻模式消除循环论证 (评审 Issue#3), 但 CSTR 预测 NTU 与真值 corr 仅 0.67, 指标全面下降
- **前瞻是"可用性"口径, 回顾是"诊断力"口径** — 论文建议: 主报前瞻 (风险评分用于前瞻), 回顾数字作上限参考并解释差异来源

### 2.5 TimesFM 收尾 (PLAN Stage 5 完成)

- 在 mathorcup 环境 (D:\Anaconda\envs\mathorcup) 运行 `step5.1_timesfm_baseline.py` (base 环境无 timesfm, 用户指定)
- 结果: 2026-02-01/10/20 零样本预测均值 0.0951/0.0946/0.0925, 远低于实际 FILT 均值 0.21
- 结论: 200M 参数零样本大模型无物理知识纯外推失败 — 反衬 5 参数 CSTR 优越性 (与 sum_11 一致)

## 3. 关键发现

1. **0.807 是 in-sample, 0.737 才是诚实 CV** — 论文必须双口径报告 (Q1: 0.807/0.737, Q3: 0.485/0.617)
2. **0.602 不是部署性能** — 是"FILT 真值已知"的传递函数上限, 论文如用必须标注口径
3. **FAIL 探索是闭环掩蔽的补充证据** — 可学习 β/θ/路由权重全部不敌手调物理参数
4. **Q4 前瞻/回顾差异巨大** — 循环论证修复的代价是捕获率 -31pp, 这是"信息可得性"的物理代价, 论文应如实呈现
5. **TimesFM 基线完成** — PLAN Stage 5 全部 ✅

## 4. 比对与决策记录

| 决策 | 选项 | 采用 | 理由 |
|---|---|---|---|
| D1 Q1 参数 | 6.09/44 vs 8.0/48 | **6.09/44** | 与已提交 0.8072/sum_7 同源; 8.0/48 属 Q3 优化口径 |
| D2 Q3 主口径 | 0.602 vs 0.485 | **forecast 0.485 主报 + oracle 0.617 上限** | 部署诚实性; FILT 代价 0.131 是叙事点 |
| D3 Q4 主口径 | 前瞻 vs 回顾 | **双口径并列, 主报前瞻** | 前瞻无循环论证; 回顾作上限参考 |
| D4 FAIL 脚本 | archive vs 删除 | **archive/** | 保留证据可复现, 遵循 sum_10 惯例 |

**对论文的影响**: 数字总表 `results/number_census.csv` 为唯一数字源, 每行含复现命令与文档引用; 所有旧文档 (README/PLAN/CONSTITUTION/INDEX/migration_prompt) 中过时数字全部更正。

## 5. 产出文件

| 文件 | 用途 |
|---|---|
| `results/number_census.csv` | 数字总表 (唯一数字源, 30+ 行) |
| `results/q1_tscv_validation_rl6.09_q44.json` | Q1 诚实 TS-CV (论文口径) |
| `results/q1_tscv_validation_rl8_q48.json` | Q1 诚实验证 (备选参数) |
| `results/q3_forecast_cv_results.json` | Q3 三模式 CV 结果 |
| `results/tables/timesfm_baseline.csv` + `results/timesfm_summary.json` | TimesFM 基线 |
| `docs/logs/latest_16.log` | 本会话时间线 |
| `archive/step1.10_learnable_beta.py` + `archive/step3.9_learnable_routing.py` | FAIL 探索归档 |

## 6. 下一步

- 论文写作: 以 number_census.csv 为数字源, 按交接说明 §七 叙事结构推进
- 已知限制: output/ 不入 git, clone 后需先跑 step0 系列生成 clean_data.csv 等依赖