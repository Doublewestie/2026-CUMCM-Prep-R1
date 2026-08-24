# 2026-CUMCM-Prep-R1

2026年全国大学生数模竞赛，队伍第一轮备赛训练

**选题**：B题「自来水厂水质预测与评估」

**核心架构**：三级FILT分区 + CSTR物理模型 + 经验/压缩/反馈三策略

> ⚠️ **数字口径 (sum_13, 2026-07-27)**: 论文唯一数字源 = `results/number_census.csv`。
> Q1: in-sample 0.807 / 诚实TS-CV **0.737**; Q3: oracle( FILT真值已知) 0.617 / 部署口径 **0.485**; 过时数字 0.732/0.602 引用须标注口径。

---

## 项目结构

```
Code/
├── PLAN.md                          # 分阶段实施计划
├── docs/
│   ├── logs/                        # 工作日志 (latest_0~16)
│   ├── sums/sum_*                   # 实验报告 (1~13)
│   ├── specs/                       # 架构规格
│   └── 代码手→论文手交接说明.md      # 论文手交接材料
├── step0_config.py                  # [完成] 全局配置+三级参数
├── step0_preprocess.py              # [完成] 数据预处理+特征工程
├── step1.0_tier_classifier.py       # [完成] Q1: 三级分类器
├── step1.1_tier1_noise.py           # [完成] Q1: T1经验采样
├── step1.2_tier2_experiment.py      # [完成] Q1: T2双路径对比
├── step1.3_tier3_greybox.py         # [完成] Q1: T3 CSTR+反馈+τ₁
├── step1.4_feature_importance.py    # [完成] Q1: T3特征重要性
├── step1.7_final_cstr.py            # [完成] Q1: 最终公式 (in-sample R²=0.807)
├── step1.7+_tscv_validation.py      # [完成] Q1: 诚实TS-CV (R²=0.737)
├── step1.9+_summary_report.py       # [完成] Q1: 汇总表
├── step2.*.py                       # [完成] Q2动态时滞建模
├── step3.8_final_stratified.py      # [完成] Q3: oracle口径 (R²=0.602)
├── step3.8+_forecast_cstr.py        # [完成] Q3: 诚实部署口径 (R²=0.485)
├── step4.*.py                       # [完成] Q4风险评价
├── step5.*.py                       # [完成] 消融+TimesFM+图表
└── results/number_census.csv        # 论文唯一数字总表
```

## Q1 核心结果

| 等级 | 阈值 | 占比 | 策略 | NTU R² |
|:---:|---:|---:|---|---:|
| T1 | ≤0.05 | 49% | 经验频率采样 | 0.862 |
| T2 | 0.05~0.15 | 30% | 对数压缩灰箱 | 0.757 |
| T3 | >0.15 | 21% | CSTR+反馈 | 0.788 |
| **全量** | — | — | **CSTR段2**: NTU(t)=β₂·NTU(t-1)+(1-β₂)·FILT(t), 分tier A + Balance Detector | **0.807** (in-sample) / **0.737** (诚实TS-CV) |

> 原XGBoost (101维) R²=0.34 → CSTR物理模型 in-sample R²=0.807, 提升+0.47
> T3核心因素: η_coag(0.335) > FILT_NTU_mean6(0.242) > TW_FLOW(0.053)

## Q3 核心结果 (sum_13 口径)

| 口径 | CV R² | 说明 |
|---|---|---|
| Oracle (FILT真值已知) | 0.617 | 信息上限 |
| **部署口径 (AR(6)预测FILT)** | **0.485** | 论文主口径 |
| legacy step3.8 (oracle+偏置表) | 0.602 | 引用须标注 |

## Git

```
https://github.com/Doublewestie/2026-CUMCM-Prep-R1
```
