# CONSTITUTION.md — B题项目硬约束速查

> 不可变设计决策 | 违反任一条即视为架构级错误

---

## §1 物理约束（不可绕过）

| # | 约束 | 数学形式 | 可调？ |
|:---:|------|------|:---:|
| C1 | 出厂浊度非负 | ŷ ≥ 0 | 不可 |
| C2 | 过滤不增浊度 | FILT ≤ R/W NTU(t-τ₁) | 不可 |
| C3 | 浊度不瞬时跳变 | |ΔNTU| ≤ ε_max (P99≈0.805) | 不可 |
| C4 | 时间因果性 | t时刻预测仅用≤t的数据 | 不可 |
| C5 | 物理不作为独立输出 | 物理约束不出现在架构图中作为预测值 | 不可 |

---

## §2 全局参数（编码前锁定）

| 参数 | 值 | 用途 | 可调？ |
|------|:---:|------|:---:|
| NTU_STANDARD | 1.0 | 国标限值 | 不可 |
| DELTA_T | 2h | 采样间隔 | 不可 |
| CSTR_N_HARD | 10 | CSTR硬边界级数 | 可调(灵敏度分析) |
| CSTR_N_REF | 2 | CSTR参考级数 | 可调 |
| ALUM_FIXED_LAG | 3 steps (6h) | ALUM固定时滞 | 不可(工艺先验) |
| RISK_T_HALF | 3 steps (6h) | Q4风险时长半衰期 | 可调 |
| HUBER_DELTA | 1.0 | Huber损失δ | 从MAD估计后可调 |

---

## §3 模型超参（编码时选定，非不可变）

| 参数 | 选定值 | 调参范围 | 备注 |
|------|:---:|------|------|
| XGB_n_estimators | 200 | [100, 500] | Q1已确认为最优 |
| XGB_max_depth | 6 | [4, 8] | Q1已确认为最优 |
| XGB_learning_rate | 0.05 | [0.01, 0.07] | Q1已确认为最优 |
| LGB_n_estimators | 400 | [200, 500] | Q1: lr=0.01,L1L2=0.01,subsample=0.9 |
| LGB_learning_rate | 0.01 | [0.01, 0.05] | Q1调参: 0.05→0.01(R2: -0.01→0.27) |
| LGB_reg_alpha | 0.01 | [0, 1] | Q1调参新增 |
| LGB_reg_lambda | 0.01 | [0, 1] | Q1调参新增 |
| RF_n_estimators | 100 | [50, 500] | Q1已确认为最优 |
| RF_max_depth | 8 | [5, 12] | Q1已确认为最优 |
| GRU_HIDDEN | 64 | [32, 128] | Q3用 |
| TCN_LAYERS | 4 | [2, 6] | Q2/Q3用 |
| TCN_KERNEL | 3 | [3, 5] | Q2/Q3用 |
| RF_TREES | 50 | [30, 100] | Q3用 |
| RF_DEPTH | 6 | [4, 10] | Q3用 |
| LAMBDA_SMOOTH | 0.1 | [0.01, 1.0] | Q2用 |
| LAMBDA_UPPER | 0.5 | [0.1, 2.0] | Q2用 |
| LAMBDA_NONNEG | 0.5 | [0.1, 2.0] | Q3用 |
| LAMBDA_CONSIST | 0.01 | [0.001, 0.1] | Q3用 |

---

## §4 测试纪律

| # | 规则 |
|:---:|------|
| T1 | 每写完一个step，立即运行并检查输出形状/类型 |
| T2 | 所有模型验证在2025年数据上做TimeSeriesSplit(n_splits=5) |
| T3 | 2026年36行数据仅做一次最终预测，不参与任何训练/调参 |
| T4 | 模型对比统一使用：RMSE(首选), R², Huber Loss(训练), MAPE(补充) |
| T5 | 论文中明确声明2026年数据量限制 |
| T6 | 消融实验在2025验证集上完成，结论不依赖2026年测试结果 |

---

## §5 接口形状约定

| 接口 | 形状 | 说明 |
|------|------|------|
| X_L1L5 | [n_samples, ~60] | 五级特征矩阵 |
| y_out | [n_samples] | 出厂NTU |
| y_filt | [n_samples] | 滤后NTU |
| X_3d_tcn | [n_samples, n_channels, seq_len] | TCN输入(3D) |
| X_3d_gru | [n_samples, seq_len, n_features] | GRU输入(3D) |
| M_meta | [n_samples, ~40] | 元特征矩阵 |
| ŷ_final | [n_samples, horizon] | 多步预测输出 |

---

## §6 不可变设计决策

1. **RF元学习器条件推理**：不是输出加权avg，是40维特征空间中的条件P(Y|MetaFeatures)
2. **源A(多变量因果)+源B(单变量自回归)**：两源本质互补，信息输入完全不同
3. **物理溶解7层**：数据→特征→Loss→架构→元特征→输出→训练数据
4. **L5交互特征二阶段准入**：物理合理性论证+SHAP交互显著性，任一不过→删除
5. **TimesFM双重角色**：消融矩阵中的对比项+独立基线（不参与融合架构）
6. **N-BEATS默认首选**：源B插槽中~1M参数，TimesFM仅做对比
7. **Q1 L2+幂次项用于Ridge可解释性，不用于XGB/LGB/RF**：树模型内部已学非线性，预计算幂次项反降精度(ΔR2=-0.012)
8. **Q1 最优特征集: 89维(L1+L2+L3+L4+L5)**：无需L2+幂次项
9. **FILT_NTU系列占R2的42%(重度消融验证)**：不可移除，Q2/Q3需以此为输入
10. **Q1 集成不优于单模型(已确认)**：论文中如实说明，建议XGBoost为推荐单模型

---

## §7 Q1实验固定参数(已锁定)

| 参数 | 值 | 来源 |
|------|:---:|------|
| XGB n_estimators | 200(默认最优, 5组调参确认) | step1.1 |
| XGB max_depth | 6(默认最优) | step1.1 |
| XGB learning_rate | 0.05(默认最优) | step1.1 |
| LGB learning_rate | 0.01(从0.05下调, 修复过拟合) | step1.1 |
| LGB reg_alpha/reg_lambda | 0.01(新加) | step1.1 |
| LGB subsample/colsample | 0.9(新加) | step1.1 |
| RF n_estimators | 100(默认最优, 5组调参确认) | step1.1 |
| RF max_depth | 8(默认最优) | step1.1 |
| SHAP threshold | 0.005(筛选25/101特征) | step1.0 |
| Box-Cox λ | 0(log1p降级, 数值稳定) | step0 |

---

_变更记录: 2026-07-23 初始创建; 2026-07-24 Q1实验完成后更新§6-§7_
