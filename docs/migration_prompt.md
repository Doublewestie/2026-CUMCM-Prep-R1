# Migration Prompt — B题项目入口

## Step 1: 加载 project-reference skill

加载 `agent-memory`，了解`Code/docs/`下的`sums/`、`specs/`、`logs/`目录结构。

---

## Step 2: 阅读全部关键文档

### 必读（项目状态）
- `Code/PLAN.md` — 分阶段实施计划
- `Code/PLAN-details.md` — 完整数学推导(800行+)
- `Code/docs/specs/2026-07-23-architecture-design.md` — 完整架构规格（旧）
- **`Code/docs/specs/2026-07-24-q1-greybox-redesign.md`** — Q1+Q2灰箱重构设计规格

### 必读（决策历史）
- `Code/docs/sums/sum_1_题目分析与建模方案.md` — 题目分析、模型选择、创新点
- `Code/docs/sums/sum_2_F_RIDE数据质量审查与排除决策.md` — F/RIDE排除依据
- `Code/docs/sums/sum_4_Q2时滞估计与动态建模实验结果.md` — 原Q2 TCN失败记录
- **`Code/docs/sums/sum_5_灰箱模型重构与双模态阈值发现.md`** — 2026-07-24重构全记录

### 代码现状检查
- Q1: `step1+.x` 系列(灰箱模型, ±done); `step1.x` 系列(旧, 废弃)
- Q2: `step2.x` 系列(双模诊断, ±done); `step2.x-_` 后缀文件(旧, 废弃)
- Q3: `step3.x` 全部空骨架
- Q4: `step4.x` 全部空骨架
- 读取`step0_config.py`获取全局参数(含灰箱常数和THETA_COMFORT)

---

## Step 3: 恢复当前任务上下文

### 已完成
- 题目选定：B题「自来水厂水质预测与评估」
- Stage 0: 数据预处理(4350行清洗, 15维精简特征)
- **Stage 1+: Q1灰箱重构已完成** — 双模CSTR, V4_DualMode R²=0.53(CV+0.30), Stress_R²=0.68
- **Stage 2: Q2双模诊断已完成** — FILT_NTU阈值0.15, 滞后权重(6h/8h/10h), 闭环分解验证(失败但记录)
- 旧Q1(101维XGBoost, 废弃), 旧Q2(4层TCN全量, 废弃)

### 待完成（按优先级）
1. **Q1遗留**: 2026年三天NTU预测(归Q3)
2. **Q2遗留**: step5.0消融更新(应力区消融尚未整合)
3. **Stage 3**: Q3双源架构+RF元学习器（step3.0-3.5, 全未启动）
4. **Stage 4**: Q4三维评分+Jenks+双重验证（step4.0-4.5, 全未启动）
5. **Stage 5**: 跨题消融+TimesFM基线（step5.0-5.1, 仅Q1+Q2消融部分完成）
6. **文档体系**: Reference/docs/ CONSTITUTION + INDEX 未创建

### 当前聚焦
**Stage 3 Q3准备** — 需基于Q1双模CSTR公式+Q2滞后权重, 设计融入双模逻辑的Q3架构

---

## Step 4: 硬约束速查

1. 物理模型不出现在架构图中作为独立组件——溶解在7层路径
2. 融合范式：RF元学习器在40维特征空间做条件推理，非输出加权
3. 全部模型验证在2025年4380行数据上用TimeSeriesSplit，2026年36行仅做最终预测
4. TCN因果卷积保证时间不泄露；GRU遗忘门b_f=+1.0
5. Loss函数每项有物理意义：Huber+平滑+上界ReLU+非负ReLU
6. 特征工程仅L5交互特征需二阶段物理检验（物理合理性+SHAP交互显著性）
7. math-name命名规范：step{N}.{M}{suffix}_{description}.py

---

## Step 5: 回退策略

1. 先读 `Reference/docs/Phase*/` 三文件（占位，待编码后填充）
2. 再看 `Code/docs/specs/` 对应架构设计
3. 仍不确定 → 检查 `Code/docs/sums/sum_1` 中的具体决策
4. 仍不确定 → 直接提问
