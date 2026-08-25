# Migration Prompt — B题项目入口

## Step 1: 加载 project-reference skill

加载 `agent-memory`，了解 `Code/docs/` 下的 `sums/`、`specs/`、`logs/` 目录结构。

> **时期 C（论文材料）**: 建模完成后启动 **math-paper-production**（.dsh/skills/，
> 时期 C 编排器：六步链/DoD/守卫任务/条款分级/人工核验点）——auto 只管 B 期，
> 材料期由其独立承接；速查手册 `.dsh/math系列速查手册.md` 第七节为接线导航。
>
> **表达与合规**：表达纪律/去 AI 味 → **math-expression**（.dsh/skills/，材料/日志/
> 对话档：先定档再动手，去味不洗稿；语言层调全局 humanizer-zh）；递交/提交前 →
> **math-compliance**（26 条自查 + AI 声明，消费 audit_N 与复查报告）。

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
- **`Code/docs/sums/sum_5_Q1三级分层灰箱建模.md`** — **三级方案 (CSTR 全量 R²=0.727, CV=0.732 属旧单A口径, 勿引用)**
- `Code/docs/sums/sum_6_Q1阈值敏感性分析.md` — 阈值 0.05/0.15 验证为全局最优
- **`Code/docs/sums/sum_7_A线CSTR物理模型细化.md`** — **A线终点: 分tier A + 平衡检测器, R²=0.807 (in-sample)**
- `Code/docs/sums/sum_8_Q2_log_AR_final.md` — Q2 log-AR(6)+RidgeCV 终版
- `Code/docs/sums/sum_9_Q4三维风险评分与四级分类.md` — Q4 方案（回顾口径）
- `Code/docs/sums/sum_10_project_cleanup.md` — 项目重构记录
- **`Code/docs/sums/sum_11_模型改进探索全历程.md`** — **Q2/Q3改进探索: 五次尝试全失败, 闭环掩蔽效应五重证据链**
- **`Code/docs/sums/sum_12_Q3四层条件路由最终方案.md`** — **Q3终局: 四层条件路由（注意: 0.602 为 oracle 口径, FILT 真值已知; 部署口径见 sum_13）**
- **`Code/docs/sums/sum_13_诚实验证与数字口径统一.md`** — **⚠️ 必需: 数字口径总表 (Q1: 0.807/0.737, Q3: 0.485/0.617, Q4 前瞻/回顾双口径), 论文唯一数字源 `results/number_census.csv`**
- `Code/docs/代码手→论文手交接说明.md` — 论文手交接材料（创新点+叙事+图表清单）
- `Code/docs/sums/sum_11_Q3模型收敛.md` — Q3融合策略对比(队友)

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
- **`Code/Reference/sums/sum_11_Q3四层条件路由方法论.md`** — **Q3方法论深度分析 + Q1对比 + 闭环掩蔽验证**
- `Code/docs/logs/latest_8.log` — 墙(η_coag)模型全历程 + RL×Q规则发现 (替代已废弃的Reference/sum_9, 内容合并至sum_7+交接说明)
- `Code/Reference/docs/CONSTITUTION.md`
- `Code/Reference/docs/INDEX.md`

### 速读（了解近期动态）
- `Code/docs/logs/latest_16.log` — 遗留收编: 诚实CV验证 + 口径统一 + TimesFM收尾 (2026-07-27)
- `Code/docs/logs/latest_15.log` — Q3终局开发 + 文档整理 (2026-07-26)

### 必读（论文材料区 — 唯一契约点）
- **`Code/docs/materials/00_素材总览.md`** — 材料区唯一契约点：**到达后必须按其中 §1 必读（口径声明/数字总表/audit/研究思路/路线叙事/defense）→ §2 速读 → §3 按需 完整执行**，红线文档不可跳过；材料区全部 26 份文件由该总览导航，此处不再逐条列出

### 必读（守卫）
- **`Code/tests/test_core_guards.py`** — 论文必引数字守卫 G1-G10（10/10 全绿）；修改产物后运行 `python tests/test_core_guards.py` 验证

### 必读（代码现状）
- 运行 `python step3.8_final_stratified.py` 获取Q3完整CV+消融报告（oracle口径）
- 运行 `python step3.8+_forecast_cstr.py` 获取Q3诚实部署口径 (forecast R²=0.485)
- 运行 `python step1.7+_tscv_validation.py` 获取Q1诚实TS-CV（仿论文主口径: R²=0.737）
- 运行 `python step1.9+_summary_report.py` 获取完整汇总表
- 数字总表: `results/number_census.csv` — 论文唯一数字源

---

## Step 3: 恢复当前任务上下文

### 已完成（按Phase）
- **Phase 1 (Q1)**: 三级分层灰箱 + A线物理细化已闭环。最终公式 (step1.7_final): 分tier CSTR + Balance Detector, A={T1:400, T2:250, T3:30} + A_same=100/A_diff=20 (RL_med=6.09/Q_med=44), 全量 in-sample R²=**0.807** (step1.7权威值), 诚实 TS-CV R²=**0.737** (step1.7+_tscv_validation)。T3特征重要性: η_coag#1(0.335)。
- **Phase 2 (Q2)**: 双模阈值诊断已闭环。CCF/MIC/TE三方法全部失效。AR(6) R²=0.52 > TCN R²=-0.15。最终: log-AR(6)+RidgeCV, CV R²=0.6955。**五次改进尝试(方向感知/加权/去惯性等)全失败——详见 sum_11 + Reference sum_10。**
- **Phase 3 (Q3)**: 四层条件路由已闭环。final: step3.8 (oracle口径 R²=0.602, FILT真值已知) + step3.8+ (诚实部署口径: **forecast R²=0.485**, oracle上限 0.617, FILT预测代价 0.131)。天型 A/B/C_strong/C_weak 分策略预测，CSTR链从NTU(1:00)启动。参数: α_B=0.34, γ_W=0.25, C_th=1.0。探索历程: 0.195→0.251→0.397→0.485→0.576→0.602。详见 sum_12 + Reference sum_11 + **sum_13**。
- **Phase 4 (Q4)**: 三维风险评分(融合Q1/Q2) + 分区独立Jenks + 事件回溯验证已闭环。**双口径**: 前瞻模式(CSTR预测NTU, 无循环论证): 捕获率57.99%, 虚警1.21%, Kappa舒适区0.877; 回顾模式(实际NTU): 捕获率88.76%, 虚警0%, 提前预警20h。f₂维度已接入per-tier CSTR β₂。
- **Phase 5 (跨题)**: 消融汇总 step5.0 ✅, TimesFM零样本基线 step5.1 ✅ (失败基线证毕, 均值~0.094), 终版汇总 step5.2 ✅, 论文图表 step5.3 ✅

### 待完成（按优先级）
1. **论文写作**: 基于现有代码+sums+五重证据链撰写全文; **数字源 = results/number_census.csv (sum_13)**; 材料区 26 份文档已就绪(Q1/Q2初稿已有, Q3/Q4待写)
2. **文献人工核验**: 6 篇引用卡 DOI/期刊/卷期页/IF (03_文献库, 需浏览器确认); TimesFM 预印本是否正式引用待决策
3. 无守卫覆盖数字 6 项 (audit_1 遗留): 熵权权重/消融链/2026预测值/阶跃值/负面项 — 引用前人工复读产物
4. Step1.8 模型对比: 物理重构模型(Langmuir+CSTR)已运行, R²=-0.29被淘汰, 对比表已产出
5. 协议草稿层审核: `二阶段-题目1/论文材料自主生产_prompt_v2.md` (库外草稿, 等用户审核)

### 当前聚焦
**论文写作** — 全部模型已稳定, 材料六步链已闭环 (创新档案/素材/审计/文献/输入包/评审轮 18/18)

---

## Step 4: 硬约束速查

1. `Reference/` 位于 git 根目录内 (`Code/Reference/`), 随代码一同版本控制
2. CSTR段2公式 (分tier A, step1.7+): NTU(t)=β₂·NTU(t-1)+(1-β₂)·FILT(t), β₂=exp(-2h/θ), θ=A_tier·CW_WELL(t-1)/TW_FLOW(t-1), A_tier={400(T1), 250(T2), 30(T3)}
3. 三级分区: T1(≤0.05, 经验采样), T2(0.05~0.15, 对数压缩), T3(>0.15, CSTR+反馈) — 不可调 (sum_6 验证)
4. 物理约束只在违规出现时激活；否则用硬裁剪
5. 全部模型在2025年数据上用TimeSeriesSplit验证, 2026年仅做最终预测
6. 时滞参数不硬传给Q3——用注意力机制自适应学习
7. math-name命名规范: `step{N}.{M}_{description}.py`
8. **闭环掩蔽效应——禁止再尝试"方向感知/去惯性/流态修正"类改进**: 五重证据链已证明闭环水处理系统中物理诊断信号在预测层面不可利用。CSTR公式已完整编码流态效应。所有"基于物理直觉的CSTR结构修正"均已被验证为无效(详见 sum_11 + Reference sum_10)。Q2 log-AR(6)已达本架构天花板(CV R²=0.696)。**step1.10/step3.9 NN可学习实验同样失败 (sum_13), 已归档。**
9. **数字口径纪律 (sum_13)**: 论文数字一律以 `results/number_census.csv` 为准; Q1 报 in-sample 0.807 + TS-CV 0.737 (RL_med=6.09/Q_med=44); Q3 报 forecast 0.485 主口径 + oracle 0.617 上限 (0.602 为 legacy oracle 口径, 引用必须标注); Q4 主报前瞻口径, 回顾口径作上限参考; "0.732" 已废弃 (旧单A=141.3口径)
10. **守卫纪律 (时期C)**: `tests/test_core_guards.py` G1-G10 锚定论文必引数字; 产物重跑/参数调整后必须运行守卫; τ/参数类数字一律入守卫 (audit_1 A1 教训: 交接说明 τ=4h vs 产物 6h 冲突被守卫捕获)
11. **文献引用纪律 (时期C)**: 引用卡库 6 卡中 6 篇 DOI 均为【人工核验】待办 — 未经确认不得写入论文参考文献表; 排除记账 (黑箱ML/模糊投药/图像监测等) 作为方法学讨论素材

---

## Step 5: 回退策略

1. 先读 `Reference/docs/Phase*/` 三文件
2. 再看 `Reference/sums/` 对应学习记录
3. 仍不确定 → 检查 `Code/docs/sums/sum_5_Q1三级分层灰箱建模.md` + `sum_13_诚实验证与数字口径统一.md`
4. 仍不确定 → 直接提问
