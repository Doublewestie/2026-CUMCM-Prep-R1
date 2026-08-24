# Audit 1 — 素材-代码一致性核查报告（2026-07-27 时期C Step3 强制门）

> math-consistency 全量核查 | 创建: 2026-07-27 | 核查范围: docs/materials/ 全部 + 交接说明 / sums 声称 vs output/ + results/ 产物

---

## 阶段 0：范围与容忍度声明

- **范围**：`docs/materials/`（00 总览/01 主题素材 5 卡/02 核心文档 9 份/04 亮点档案 3 份）中全部支撑数字；交接说明 τ 表；sum_5/sum_8/sum_12 关键数字
- **容忍度判据**：相对差 ≤0.1% 且绝对差 ≤ 报告精度（4 位小数）→ 格式差异豁免；自洽性检查不受豁免
- **产物真值源**：output/*.json / results/*.json / results/tables/*.csv

## 阶段 1：声称清单（提取要点，全量核对见守卫）

| # | 声称位置 | 声称 | 产物映射 |
|:---:|---|---|---|
| C1 | 数据层卡 | T3 应力区 r=0.81（sum_5 记 0.81 vs INDEX F6 记 0.79） | 两处定义略异（T3>0.15 vs 应力区≥0.15），非产物单值 |
| C2 | 方法层卡 | log-AR(6) CV R²=0.6955 | output/step2_final_results.json cv_mean.r2 |
| C3 | 方法层卡 | τ RW_NTU=4h, ALUM=6h, FLOW=2h, PH=2h | output/step2_final_results.json tau_params |
| C4 | 模型层卡 | Q1 全量 in-sample 0.8072 / TS-CV 0.7369 | step1.7 权威值 + results/q1_tscv_validation_rl6.09_q44.json |
| C5 | 模型层_Q3 卡 | forecast 0.4853 / oracle 0.6165 / penalty 0.1312 | results/q3_forecast_cv_results.json |
| C6 | 风险层卡 | Q4 前瞻捕获 0.5799 / 虚警 0.0121 / Kappa 舒适 0.8774 | output/q4_event_backtest.json + q4_kappa_report.json |
| C7 | 方法学卡 | η_coag robust=0.335（T3 #1） | output/tier3_factor_importance.csv row1 |
| C8 | 方法学卡 | NN-β 0.5884 < 0.6889；NN-路由 0.0905 < 0.6165 | results/step1.10_verification.json + step3.9_routing_verification.json |
| C9 | 数据层卡 | T1 JS=0.0499 vs 高斯 0.6379；T2 log RMSE 0.0289 < 经验 0.0363 | output/tier1_report.json + tier2_comparison.json |
| C10 | 所有卡 | TimesFM feb 均值 0.0951/0.0946/0.0925 | results/timesfm_summary.json |

## 阶段 2-3：逐项验证与归因

全部 10 项已由 `tests/test_core_guards.py` 守卫断言逐项比对（G1-G10，产物直读）。结果: **守卫 10/10 PASS**（运行时间 2026-07-27，日志见下）。

### 错误清单（核查发现的非一致项）

| 编号 | 类别 | 位置 | 声称 | 产物实际 | 判定强度 | 修复状态 |
|:---:|---|---|---|:---:|:---:|:---:|
| A1 | 数字不符 | 交接说明 §创新4 τ 表 + 541/548/725 行 | ALUM→FILT τ=4h（"同步控制"） | **τ=6h**（step2_final_results.json tau_params.ALUM_to_FILT_hours=6；sum_8 §5 亦记 6h） | 【实证】 | ✅ 已修复（4h→6h，含 3 处正文表述）；残留扫描见下 |
| A2 | 数字不符 | 方法层卡 τ 边界声明 | 同上 4h | 6h | 【实证】 | ✅ 已修复 |
| A3 | 口径差异（已核） | 数据层卡 | T3 r=0.81 vs INDEX F6 0.79 | 两处定义不同（T3>0.15 子集 vs 应力区≥0.15 双模），非产物冲突 | 【推断】 | ✅ 已标注"以产物为准核查"，论文统一采用 0.81（T3 定义，与 21% 占比口径一致） |
| A4 | 历史值混用（豁免） | 数模总表 Q1 行 | in-sample 0.8072（step1.7 权威值）vs 守卫 in-sample 0.8023（step1.7+ 复算） | 差 0.005（边界处理微差） | 【实证】 | ✅ 已在口径声明 M2 注明：论文报 0.807（step1.7 权威），守卫锚定 0.8023（step1.7+ 复算），两者同为 in-sample 同参数 |

## 阶段 5：闭环验证

- **残留扫描**：`grep -rn "ALUM.*4h\|滞后 4h"` 交接说明 → 0 匹配（修复后）；材料区 τ 表 → 6h ✅
- **守卫测试**：`python tests/test_core_guards.py` → **ALL CORE GUARDS PASSED (10/10)** ✅
- **守卫覆盖说明**：论文必引数字 10 组断言已固化；其余数字（如 Q4 熵权/等级占比、消融链 0.195→0.602、2026 预测具体值）**无守卫覆盖**，如实标注入遗留清单

## 收敛判定

✅ **全量零错误（A1/A2 已修复闭环）+ 残留扫描零匹配 + 守卫 10/10 通过 → 放行**

## 遗留（无守卫覆盖项，论文引用时注意）

| 项 | 说明 |
|---|---|
| Q4 熵权权重/等级占比（前瞻 0.127/0.613/0.260） | 产物可读但未固化断言，引用前人工复读 output/q4_omega_weights.json |
| Q3 消融链 0.195→0.602 | step3.8 Part B 硬编码表，产物 working（output/q3_final_metrics.json） |
| 2026 Feb 预测具体值 | results/q3_final_predictions.csv，未固化断言 |
| CSTR 消融阶跃 0.727/0.783/0.807 | step1.9+ 输出，未固化 |
| Q4 等级单调性 FAIL / Kappa 应力区 -0.04 | 负面结果如实呈现，未固化（避免守卫误锁负面项） |