# 2026-CUMCM-Prep-R1

2026年全国大学生数模竞赛，队伍第一轮备赛训练

**选题**：B题「自来水厂水质预测与评估」

**核心架构**：三级FILT分区 + CSTR物理模型 + 经验/压缩/反馈三策略

---

## 项目结构

```
Code/
├── PLAN.md                          # 分阶段实施计划
├── docs/
│   ├── logs/                        # 工作日志 (latest_0~4)
│   ├── sums/sum_*                   # 实验报告 (1~5)
│   └── specs/                       # 架构规格
├── step0_config.py                  # [完成] 全局配置+三级参数
├── step0_preprocess.py              # [完成] 数据预处理+特征工程
├── step1.0_tier_classifier.py       # [新建] Q1: 三级分类器
├── step1.1_tier1_noise.py           # [新建] Q1: T1经验采样
├── step1.2_tier2_experiment.py      # [新建] Q1: T2双路径对比
├── step1.3_tier3_greybox.py         # [新建] Q1: T3 CSTR+反馈+τ₁
├── step1.4_feature_importance.py    # [新建] Q1: T3特征重要性
├── q1_data_utils.py                 # [新建] Q1: 共享工具函数
├── run_q1_full.py                   # [新建] Q1: 汇总表
├── step2.*.py                       # [完成] Q2动态时滞建模
├── step3.*.py                       # [待实现] Q3混合预测
├── step4.*.py                       # [待实现] Q4风险评价
└── step5.*.py                       # [部分完成] 消融实验
```

## Q1 核心结果

| 等级 | 阈值 | 占比 | 策略 | NTU R² |
|:---:|---:|---:|---|---:|
| T1 | ≤0.05 | 49% | 经验频率采样 | 0.862 |
| T2 | 0.05~0.15 | 30% | 对数压缩灰箱 | 0.757 |
| T3 | >0.15 | 21% | CSTR+反馈 | 0.668 |
| **全量** | — | — | **CSTR段2**: NTU(t)=β₂·NTU(t-1)+(1-β₂)·FILT(t) | **0.727** |

> 原XGBoost (101维) R²=0.34 → CSTR物理模型 R²=0.727, 提升+0.39
> T3核心因素: η_coag(0.335) > FILT_NTU_mean6(0.242) > TW_FLOW(0.053)

## Git

```
https://github.com/Doublewestie/2026-CUMCM-Prep-R1
```
