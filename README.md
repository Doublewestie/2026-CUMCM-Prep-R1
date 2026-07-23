# 2026-CUMCM-Prep-R1

2026年全国大学生数模竞赛，队伍第一轮备赛训练

**选题**：B题「自来水厂水质预测与评估」

**核心架构**：双源信息 + RF元学习器条件推理 + 物理约束溶解嵌入

---

## 项目结构

```
Code/
├── PLAN.md                          # 分阶段实施计划
├── docs/
│   ├── logs/latest_0.log            # 工作日志
│   ├── sums/sum_1_题目分析与建模方案.md  # 题目分析+模型选择
│   ├── specs/2026-07-23-architecture-design.md  # 完整架构规格
│   └── migration_prompt.md          # Agent会话入口
├── step0_config.py                  # [待实现] 全局配置
├── step0_preprocess.py              # [待实现] 数据预处理
├── step1.*.py                       # [待实现] Q1特征筛选+预测
├── step2.*.py                       # [待实现] Q2动态时滞建模
├── step3.*.py                       # [待实现] Q3混合预测
├── step4.*.py                       # [待实现] Q4风险评价
└── step5.*.py                       # [待实现] 消融实验
```

## Git

```
https://github.com/Doublewestie/2026-CUMCM-Prep-R1
```
