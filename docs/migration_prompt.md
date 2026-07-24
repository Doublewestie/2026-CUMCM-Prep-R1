# Migration Prompt — B题项目入口

## Step 1: 加载 project-reference skill

加载 `agent-memory`，了解 `Code/docs/` 下的 `sums/`、`specs/`、`logs/` 目录结构。

---

## Step 2: 阅读全部关键文档

### 必读（项目状态）
- `Code/PLAN.md` — 分阶段实施计划
- `Code/PLAN-details.md` — 完整数学推导(800行+)
- `Code/docs/specs/2026-07-23-architecture-design.md` — 原始架构设计（参考）
- **`Code/docs/specs/2026-07-24-Q1三级分层灰箱-design.md`** — Q1三级灰箱方案规格（当前方案）
- **`Code/README.md`** — 项目核心结果速览（NTU R²=0.727, T3 η_coag#1）

### 必读（决策历史）
- `Code/docs/sums/sum_1_题目分析与建模方案.md` — 题目分析
- `Code/docs/sums/sum_2_F_RIDE数据质量审查与排除决策.md` — F/RIDE排除
- `Code/docs/sums/sum_3_Q1实验结果与函数关系.md` — 旧XGBoost方案(R²=0.34)
- `Code/docs/sums/sum_4_Q2时滞估计与动态建模实验结果.md` — 旧TCN方案(R²=-0.15)
- `Code/docs/sums/sum_4b_灰箱模型重构与双模态阈值发现.md` — 双模CSTR重构(队友)
- **`Code/docs/sums/sum_5_Q1三级分层灰箱建模.md`** — **三级方案（当前, NTU R²=0.727）**

### 必读（方法论学习，agent面向）
- `Code/Reference/sums/sum_1_Q1特征筛选学习总结.md`
- `Code/Reference/sums/sum_2_Q2时滞估计学习总结.md`
- `Code/Reference/sums/sum_3_Q1三级分层灰箱学习总结.md`
- `Code/Reference/sums/sum_4_Q2双模阈值诊断学习总结.md`
- `Code/Reference/sums/sum_5_Q3Q5方法论前瞻.md`
- `Code/Reference/docs/CONSTITUTION.md`
- `Code/Reference/docs/INDEX.md`

### 速读（了解近期动态）
- `Code/docs/logs/latest_4.log` — Q1三级灰箱开发日志
- `Code/docs/logs/latest_3.log` — Q2旧方案开发日志

### 必读（代码现状）
- 运行 `python step1.4_feature_importance.py` 获取T3特征重要性
- 运行 `python run_q1_full.py` 获取完整汇总表

---

## Step 3: 恢复当前任务上下文

### 已完成（按Phase）
- **Phase 1 (Q1)**: 三级分层灰箱方案已闭环。CSTR段2 NTU全量 R²=0.727。T3特征重要性: η_coag#1(0.335)。τ₁=4h softmax学习。
- **Phase 2 (Q2)**: 双模阈值诊断已闭环。CCF/MIC/TE三方法全部失效。AR(6) R²=0.52 > TCN R²=-0.15。
- **Phase 3-5 (Q3/Q4/Q5)**: 代码骨架已创建，内容待实现。

### 待完成（按优先级）
1. **Stage 3**: Q3双源架构+RF元学习器（step3.0-3.5, 全未启动）
2. **Stage 4**: Q4三维评分+Jenks+双重验证（step4.0-4.5, 全未启动）
3. **Stage 5**: 跨题消融+TimesFM基线（step5.0-5.1, 部分完成）
4. **2026年NTU预测**: 归Q3统一交付

### 当前聚焦
**Stage 3 Q3准备** — 基于CSTR段2公式(NTU=β₂·NTU₋₁+(1-β₂)·FILT) + 三级分区策略, 设计双源RF元学习器架构

---

## Step 4: 硬约束速查

1. `Reference/` 位于 git 根目录内 (`Code/Reference/`), 随代码一同版本控制
2. CSTR段2公式: NTU(t)=β₂·NTU(t-1)+(1-β₂)·FILT(t), β₂=exp(-2h/θ)
3. 三级分区: T1(≤0.05, 经验采样), T2(0.05~0.15, 对数压缩), T3(>0.15, CSTR+反馈)
4. 物理约束只在违规出现时激活；否则用硬裁剪
5. 全部模型在2025年数据上用TimeSeriesSplit验证, 2026年仅做最终预测
6. 时滞参数不硬传给Q3——用注意力机制自适应学习
7. math-name命名规范: `step{N}.{M}_{description}.py`

---

## Step 5: 回退策略

1. 先读 `Reference/docs/Phase*/` 三文件
2. 再看 `Reference/sums/` 对应学习记录
3. 仍不确定 → 检查 `Code/docs/sums/sum_5_Q1三级分层灰箱建模.md`
4. 仍不确定 → 直接提问
