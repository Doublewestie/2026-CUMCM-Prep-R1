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
- **`Code/docs/sums/sum_5_Q1三级分层灰箱建模.md`** — **三级方案 (NTU R²=0.727)**
- `Code/docs/sums/sum_6_Q1阈值敏感性分析.md` — 阈值 0.05/0.15 验证为全局最优
- **`Code/docs/sums/sum_7_A线CSTR物理模型细化.md`** — **A线终点: 分tier A + 平衡检测器, R²=0.807**
- `Code/docs/sums/sum_8_Q2_log_AR_final.md` — Q2 log-AR(6)+RidgeCV 终版
- `Code/docs/sums/sum_9_Q4三维风险评分与四级分类.md` — Q4 方案
- `Code/docs/sums/sum_10_project_cleanup.md` — 项目重构记录
- **`Code/docs/sums/sum_11_模型改进探索全历程.md`** — **Q2/Q3改进探索: 五次尝试全失败, 闭环掩蔽效应五重证据链**

### 必读（方法论学习，agent面向）
- `Code/Reference/sums/sum_1_Q1特征筛选学习总结.md`
- `Code/Reference/sums/sum_2_Q2时滞估计学习总结.md`
- `Code/Reference/sums/sum_3_Q1三级分层灰箱学习总结.md`
- `Code/Reference/sums/sum_4_Q2双模阈值诊断学习总结.md`
- `Code/Reference/sums/sum_5_Q3Q5方法论前瞻.md`
- `Code/Reference/sums/sum_6_Q1阈值敏感性分析.md`
- `Code/Reference/sums/sum_7_A线CSTR物理模型细化全历程.md` — **A线终点: 分tier A, R²=0.783**
- `Code/Reference/sums/sum_8_B线负反馈控制回路探索全历程.md` — **B线终点: 负反馈存在但不可量化建模**
- **`Code/Reference/sums/sum_10_闭环掩蔽效应五重证据链.md`** — **Agent认知: 五次改进全历程 + 失败模式分类 + 设计原则**
- `Code/docs/logs/latest_8.log` — 墙(η_coag)模型全历程 + RL×Q规则发现 (替代已废弃的Reference/sum_9, 内容合并至sum_7+交接说明)
- `Code/Reference/docs/CONSTITUTION.md`
- `Code/Reference/docs/INDEX.md`

### 速读（了解近期动态）
- `Code/docs/logs/latest_13.log` — Q2/Q3模型改进探索全历程 (2026-07-26)
- `Code/docs/logs/latest_12.log` — Q3管线验证 (2026-07-26)

### 必读（代码现状）
- 运行 `python step1.4_feature_importance.py` 获取T3特征重要性
- 运行 `python step1.9+_summary_report.py` 获取完整汇总表

---

## Step 3: 恢复当前任务上下文

### 已完成（按Phase）
- **Phase 1 (Q1)**: 三级分层灰箱 + A线物理细化已闭环。最终公式 (step1.7_final): 分tier CSTR + Balance Detector, A={T1:400, T2:250, T3:30} + A_same=100/A_diff=20, 全量 R²=**0.807**。T3特征重要性: η_coag#1(0.335)。
- **Phase 2 (Q2)**: 双模阈值诊断已闭环。CCF/MIC/TE三方法全部失效。AR(6) R²=0.52 > TCN R²=-0.15。最终: log-AR(6)+RidgeCV, CV R²=0.6955。**五次改进尝试(方向感知/加权/去惯性等)全失败——详见 sum_11 + Reference sum_10。**
- **Phase 3 (Q3)**: 链式预测(CSTR递归+bias+RF+ensemble)已实现。CV R²=0.267; oracle R²=0.503理论天花板。Sobol敏感性: η_coag S_Ti=0.596最显著。TimesFM零样本基线: 预测均值~0.093(纯外推,无物理知识)。**Q3偏离原始spec(双源架构未实现), 当前架构为简化链式方案。**
- **Phase 4 (Q4)**: 三维风险评分(融合Q1/Q2) + 分区独立Jenks + 事件回溯验证已闭环。超标捕获率88.76%, 虚警率0%, 提前预警20h。f₂维度已接入per-tier CSTR β₂。

### 待完成（按优先级）
1. **论文写作**: 基于现有代码+sums+五重证据链撰写全文
2. **Step1.8 模型对比**: 物理重构模型(Langmuir+CSTR)已运行, R²=-0.29被淘汰, 对比表已产出

### 当前聚焦
**论文写作** — 全部模型已稳定, 消融+基线+敏感性+图表均已完成

---

## Step 4: 硬约束速查

1. `Reference/` 位于 git 根目录内 (`Code/Reference/`), 随代码一同版本控制
2. CSTR段2公式 (分tier A, step1.7+): NTU(t)=β₂·NTU(t-1)+(1-β₂)·FILT(t), β₂=exp(-2h/θ), θ=A_tier·CW_WELL(t-1)/TW_FLOW(t-1), A_tier={400(T1), 250(T2), 30(T3)}
3. 三级分区: T1(≤0.05, 经验采样), T2(0.05~0.15, 对数压缩), T3(>0.15, CSTR+反馈) — 不可调 (sum_6 验证)
4. 物理约束只在违规出现时激活；否则用硬裁剪
5. 全部模型在2025年数据上用TimeSeriesSplit验证, 2026年仅做最终预测
6. 时滞参数不硬传给Q3——用注意力机制自适应学习
7. math-name命名规范: `step{N}.{M}_{description}.py`
8. **闭环掩蔽效应——禁止再尝试"方向感知/去惯性/流态修正"类改进**: 五重证据链已证明闭环水处理系统中物理诊断信号在预测层面不可利用。CSTR公式已完整编码流态效应。所有"基于物理直觉的CSTR结构修正"均已被验证为无效(详见 sum_11 + Reference sum_10)。Q2 log-AR(6)已达本架构天花板(CV R²=0.696)。

---

## Step 5: 回退策略

1. 先读 `Reference/docs/Phase*/` 三文件
2. 再看 `Reference/sums/` 对应学习记录
3. 仍不确定 → 检查 `Code/docs/sums/sum_5_Q1三级分层灰箱建模.md`
4. 仍不确定 → 直接提问
