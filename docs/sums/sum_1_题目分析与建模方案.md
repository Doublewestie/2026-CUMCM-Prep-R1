# Sum 1: 题目分析与建模方案

> 基于全队讨论 + 数据探索 + 多技能协同分析，2026-07-23 确立

---

## 1. 题目选择与依据

**选择：B题「自来水厂水质预测与评估」**

| 维度 | 分析 |
|------|------|
| 队伍能力 | 主攻机器学习（Python），B题全程围绕特征筛选→时序预测→混合建模→风险评价，完全在能力主轴上 |
| A题对比 | 偏热力学物理建模，ML仅为辅助角色 |
| C题对比 | 偏运筹学评价优化，ML使用度低 |
| 数据规模 | 2025年4380行（12月×~30天×12次/天），2026年36行，适合ML+需小样本策略 |
| 获奖匹配 | 预测类+机理混合选题成熟，创新空间大，适合国一冲刺 |

---

## 2. 小问划分与逻辑关系

### 2.1 题型分类

| 小问 | 题型 | 子类型 | 直接目标 | 隐含目标 |
|:---:|------|--------|----------|----------|
| Q1 | 预测类 | 回归 + 特征选择 | 筛选NTU主因，建模预测2/1, 2/10, 2/20的NTU | 验证非线性模型在极右偏数据上的适用性，为Q2/Q3提供特征基础 |
| Q2 | 机理分析类 + 预测类 | NARX系统辨识 | 建立FILT.NTU动态模型 + 各输入变时滞参数 | 揭示工艺链因果延迟，为Q3提供时滞对齐参数 |
| Q3 | 预测类 + 机理分析类 | 多步超前 + 机理-数据混合 | 双源RF元学习器预测6-12h出厂NTU + Sobol敏感性分析 | 验证"多变量因果+单变量自回归+物理约束"条件推理范式 |
| Q4 | 评价类 | 指标体系 + 分类 | 三维评分+Jenks断点→四等级+统计 | 验证Q3预测在风险评价中的实际应用 |

### 2.2 依赖链

```
Q1(特征集 L1-L5) ──→ Q2(时滞参数 τ₁-τ₄) ──→ Q3(输入时间对齐) ──→ Q4(风险评价输入)
     │                         │                        │
     └── 特征重要性共享 ────────┴── 特征子集+时滞共享 ───┘
```

---

## 3. 数据总览

### 3.1 核心统计（2025年训练集，4380行）

| 变量 | mean | std | min | max | skew | 关键特性 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| R/W NTU | 40.74 | 42.42 | 2 | 456 | 3.24 | 极右偏，暴雨时暴增 |
| FILT.NTU | 0.21 | 0.63 | 0.02 | 9.8 | 7.80 | 通常<0.3，7-9月升高 |
| NTU(出厂) | 0.45 | 0.66 | 0.08 | 11.9 | 6.92 | 超标率3.9%，7-9月最高 |
| ALUM | 0.054 | 0.005 | 0.04 | 0.08 | 0.37 | 控制变量，操作员主动调节 |
| R/W FLOW | 49.06 | 3.46 | 4.7 | 63.4 | -1.54 | 左偏 |

### 3.2 关键物理发现

| 发现 | 数值 | 建模启示 |
|------|:---:|------|
| FILT ≤ R/W 违规率 | 0/4379 = 0% | 过滤不增浊度硬约束天然成立 |
| NTU自相关 lag1/lag12 | 0.88/0.46 | 短程+长程依赖并存，验证TCN+GRU互补 |
| R/W NTU→NTU直接相关 | 0.04-0.07 | 线性模型必然失败——处理系统极其有效 |
| 连续超标事件 | 27次，均值12.5h，max=112h | Q4中D_run需指数衰减避免线性外推 |
| 7-9月FILT均值 | 0.94/0.38/0.81 vs 其他月<0.3 | 季节性特征至关重要 |

### 3.3 数据质量问题

| 问题 | 解决方案 |
|------|------|
| T/W PUMP DUTY含"2,4"字符串 | 编码时检查上下文（是否伴随流量变化），取均值或拆分 |
| 18ML LEVEL/FLOW缺失率25.8% | Q1中SHAP判断重要性后决定是否保留 |
| F/RIDE含"-" | 转为0（未投加矾） |
| 2026年测试集仅36行 | 全部模型验证在2025数据上做TimeSeriesSplit，2026仅做最终预测演示 |

---

## 4. 核心假设（全局适用）

| H# | 假设 | 数据支撑 | 用途 |
|:---:|------|------|------|
| H1 | 系统因果：t时刻水质仅依赖t及之前的输入 | TCN因果卷积 + FILT>R/W违规率0%验证 | 所有时序模型 |
| H2 | 一级反应动力学近似：dC/dt = -kC | 去除率恒定99.1%-99.3% | Q2物理正则化 |
| H3 | 清水池=N个串联CSTR | RTD从水位/流量推算 | Q3物理约束 |
| H4 | 忽略设备故障离散冲击（但REMARKS作为异常标记加入） | 平衡简化与信息保留 | 全局 |
| H5 | 浊度不瞬时跳变：\|ΔNTU\|<ε_max | P99=0.805 | 平滑性Loss |

---

## 5. 逐小问模型方案

### 5.1 Q1：特征筛选 + NTU预测

**选定方案**：XGBoost/LGBM/RF三模型集成 + 五级特征金字塔(L1-L5) + SHAP二重验证

| 步骤 | 内容 |
|------|------|
| 预处理 | Box-Cox(极右偏→正态) + Log1p + Sin/Cos循环编码 |
| 特征工程 | L1(原始20维)→L2(衍生:η_coag,φ_alum,ψ_hyd)→L3(滞后:lag1/3/6)→L4(聚合:μ,σ,M,Δ)→L5(交互:Π_load,Γ_alum,Ψ_alum,Ω_night) |
| 筛选 | SHAP TreeExplainer → φ_j, Permutation Importance×10 → Ĩ_j, I_j^robust=√(φ_j·Ĩ_j), 保留>0.05 |
| 集成 | XGBoost+LGBM+RF → 方差倒数加权 → ŷ_ens |

### 5.2 Q2：FILT.NTU动态时滞模型

**选定方案**：TCN因果膨胀卷积 + MIC/传递熵双重时滞估计 + 物理嵌入式Loss

| 步骤 | 内容 |
|------|------|
| 时滞 | MIC(非线性统计依赖) + TE(因果方向性) → 双重验证确定τ* |
| 建模 | TCN(4层膨胀卷积, RF=31步=62h) → 时滞注意力 → ŷ_filt |
| Loss | Huber(δ=1.0) + 0.1·Σ\|Δ\|(平滑) + 0.5·ΣReLU(ŷ-X_raw)(物理上界) |

**关键决策**：ALUM的时滞不通过传递熵估计（矾是控制变量，信号被操作逻辑掩盖），改为工艺先验固定值6h。

### 5.3 Q3：NTU 6-12h混合预测（核心创新）

**选定方案**：双源信息 + RF元学习器条件推理

```
源A(主力): TCN(局部特征)→GRU(长期依赖) → ŷ_multi, h_multi
           输入: L1-L5全特征(~60维)
           训练: HuberLoss + λ₁平滑 + λ₂上界ReLU
           
源B(参照): 单变量自回归插槽 → ŷ_uni, u_uni, trend_uni
           默认: N-BEATS(~1M参数，趋势+周期+残差分解)
           对比: TimesFM 2.5(200M，消融对比)
           输入: 仅NTU自身历史序列

RF元学习器: 输入~40维元特征矩阵
           ├ 预测特征: ŷ_multi, ŷ_uni, Δŷ(分歧)
           ├ 表示特征: h_multi[0:8], attn_weights, u_uni
           ├ 物理一致度: d_multi, d_uni, d_neg, safety
           └ 环境上下文: month, hour, volatility, gate_output
           → 条件推理: "此刻该信A还是B还是保守靠近物理边界"
           → 硬约束: clip(ŷ, 0, X_raw(t-τ₁))

消融矩阵: 空(无B源) vs N-BEATS vs TimesFM vs 仅N-BEATS vs 仅TimesFM
```

**TimesFM双重角色**：消融矩阵中的对比项 + 纯零样本独立基线（不参与融合架构）

### 5.4 Q4：水质风险评价

**选定方案**：三维风险评分 + Jenks自然断点 + 双重验证

| 步骤 | 内容 |
|------|------|
| 维度1 | f₁ = max(0, (NTU-1.0)/1.0) / P99 —— 超标幅度 |
| 维度2 | f₂ = 1-exp(-γ·D_run), γ取决于半衰期 —— 持续时长(指数饱和) |
| 维度3 | f₃ = ReLU(ΔNTU/Δt) —— 恶化趋势 |
| 赋权 | 熵权法客观确定w₁,w₂,w₃ |
| 分级 | Jenks自然断点法(最小化组内方差) → 四级{安全,低,中,高} |
| 验证 | FCE对比 → Kappa一致性 + Bootstrap 1000次CI |

---

## 6. 物理约束溶解路径（不出现在架构图中作为独立组件）

| 层级 | 物理形式 | 数学表达 |
|------|------|------|
| 数据层 | 合理性验证 | FILT≤R/W, NTU≥0, \|ΔNTU\|分布检查 |
| 特征层 | 物理定义 | η_coag=(R/W-FILT)/R/W(质量守恒), Π_load=NTU·Flow(通量) |
| 模型层(Loss) | 处罚违反 | λ₁Σ\|Δ\|(平滑) + λ₂ΣReLU(ŷ-X_raw)(上界) + λ₃ΣReLU(-ŷ)(非负) |
| 模型层(架构) | 结构约束 | TCN因果卷积, GRU遗忘门正偏置(b_f=+1.0,水质惯性) |
| 元特征层 | 违规检测 | d_multi=max(0,ŷ_multi-X_raw), d_uni, d_neg, safety |
| 输出层 | 硬裁剪 | clip(ŷ, 0, X_raw(t-τ₁)) |
| 训练数据 | 降权 | y_true > X_raw → 样本权重×0.1 |

---

## 7. 创新点汇总（论文素材）

| 小问 | 创新点 | 论文卖点叙事 |
|:---:|------|------|
| Q1 | L5交互特征(物理派生) + SHAP二重验证 | "交互特征并非随意相乘——每个交互项对应一条物理守恒律" |
| Q2 | 传递熵因果方向 + 物理嵌入式Loss | "传递熵不仅回答'相关吗'，更回答'谁导致谁'——这是相关性分析无法做到的" |
| Q3 | 双源RF条件推理 + 消融矩阵 | "融合不是加权平均——RF元学习器在40维上下文空间中学到了'信任的条件'" |
| Q4 | 三维评分+Jenks数据驱动边界+双重验证 | "风险边界不是人定的，是数据自己揭示的；两种方法的Kappa一致性检验排除了人为偏差" |

---

## 8. 文件组织（math-name命名规范）

```
Code/
├── step0_config.py              # 全局配置 + 物理常数
├── step0_preprocess.py          # 数据清洗 + L1-L5特征工程
├── step1.0_feature_selection.py # Q1基线: XGBoost+SHAP
├── step1.1_model_comparison.py  # Q1增强: 三模型集成
├── step1.2_shap_interaction.py  # Q1补充: SHAP交互效应
├── step1.5_visualization.py     # Q1汇总: 图表+Excel
├── step2.0_time_delay_estimation.py # Q2基线: MIC+TE
├── step2.1_tcn_dynamic_model.py # Q2增强: TCN+物理Loss
├── step2.5_visualization.py     # Q2汇总
├── step3.0_source_a_multivariate.py # Q3源A: TCN→GRU
├── step3.1_source_b_univariate.py   # Q3源B: N-BEATS/TimesFM
├── step3.2_meta_feature_matrix.py   # Q3融合: RF元学习器
├── step3.3_sobol_sensitivity.py     # Q3补充: Sobol分析
├── step3.5_visualization.py     # Q3汇总
├── step4.0_risk_scoring.py      # Q4: 三维评分+熵权
├── step4.1_jenks_classification.py  # Q4: Jenks断点
├── step4.2_dual_validation.py   # Q4: FCE+Bootstrap
├── step4.5_visualization.py     # Q4汇总
├── step5.0_ablation.py          # 跨题消融
└── step5.1_timesfm_baseline.py  # TimesFM独立基线
```

**执行原则**：stepN.0先跑通基线 → stepN.1/N.2增强 → stepN.5出图 → step5.0全流程消融。

---

## 9. 待细化项

| 项 | 当前状态 | 细化时机 |
|----|------|:---:|
| CSTR级数N估计（KS距离拟合） | 已定N=10(硬界)+N=2(参考) | 编码时 |
| 传递熵surrogate test（p<0.05） | 已定方法，待写shuffle代码 | 编码时 |
| Sobol一阶/总阶效应分解 | 已定Saltelli采样 | 编码时 |
| Huber δ = 1.5×MAD | 待从训练集残差估计 | 编码时 |
| T/W PUMP DUTY "2,4"解析 | 待检查上下文 | 编码时 |
| 18ML列SHAP重要性 | 待Q1跑通后判断 | Q1编码后 |

---

## 10. 参考文献

[1] Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions
[2] Reshef et al. (2011) — MIC: Detecting Novel Associations in Large Data Sets
[3] Schreiber (2000) — Transfer Entropy: Measuring Information Transfer
[4] Bai et al. (2018) — TCN: An Empirical Evaluation of Generic Convolutional and Recurrent Networks
[5] Oreshkin et al. (2020) — N-BEATS: Neural Basis Expansion Analysis for Time Series
[6] Das et al. (2024) — TimesFM: A Decoder-Only Foundation Model for Time-Series Forecasting
[7] Jenks (1967) — The Data Model Concept in Statistical Mapping
[8] Saltelli et al. (2010) — Variance Based Sensitivity Analysis of Model Output
