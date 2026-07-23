# Migration Prompt — B题项目入口

## Step 1: 加载 project-reference skill

加载 `agent-memory`，了解`Code/docs/`下的`sums/`、`specs/`、`logs/`目录结构。

---

## Step 2: 阅读全部关键文档

### 必读（项目状态）
- `Code/PLAN.md` — 分阶段实施计划
- `Code/PLAN-details.md` — 完整数学推导(800行+): 变量定义表、公式推导、假设论证、Mermaid流程图、消融矩阵
- `Code/docs/specs/2026-07-23-architecture-design.md` — 完整架构规格

### 必读（决策历史）
- `Code/docs/sums/sum_1_题目分析与建模方案.md` — 题目分析、模型选择、创新点
- `Code/docs/sums/sum_2_F_RIDE数据质量审查与排除决策.md` — F/RIDE排除依据(六项独立证据)

### 必读（硬约束）
- `Reference/docs/CONSTITUTION.md` — 全局参数、物理约束、测试纪律
- `Reference/docs/INDEX.md` — 论文公式→代码位置全映射

### 速读（了解近期动态）
- `Code/docs/logs/` 最新 3 篇

### 代码现状检查
- `Code/`下`step*.py`文件均为骨架（无实现），需从Stage 0开始逐步实现
- 读取`step0_config.py`获取当前全局参数

---

## Step 3: 恢复当前任务上下文

### 已完成
- 题目选定：B题「自来水厂水质预测与评估」
- 建模方案确立：双源RF元学习器 + 物理溶解7路径 + L1-L5五级特征金字塔
- 文档体系初始化：sum_1 + specs + PLAN.md 已落笔
- 代码骨架：20个step*.py空文件已创建

### 待完成（按优先级）
1. **Stage 0**: step0_config.py + step0_preprocess.py（全题共享数据预处理）
2. **Stage 1**: Q1特征筛选与三模型集成（step1.0-1.5）
3. **Stage 2**: Q2传递熵时滞+TCN动态模型（step2.0-2.5）
4. **Stage 3**: Q3双源架构+RF元学习器（step3.0-3.5）
5. **Stage 4**: Q4三维评分+Jenks+双重验证（step4.0-4.5）
6. **Stage 5**: 跨题消融+TimesFM基线（step5.0-5.1）

### 当前聚焦
**Stage 0** — 数据预处理与五级特征工程（唯一的前置依赖，阻塞所有后续Stage）

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
