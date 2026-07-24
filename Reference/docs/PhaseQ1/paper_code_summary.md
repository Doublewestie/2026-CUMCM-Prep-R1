# Phase Q1 — 论文公式→代码位置映射

> B题 特征筛选与出厂浊度预测 | 2026-07-24

---

## 论文关键公式

| 公式 | 数学表达 | 代码文件 | 行号 |
|------|------|------|:---:|
| Box-Cox变换 | $x' = (x^\lambda - 1)/\lambda$, λ由MLE估计 | step0_preprocess.py | 20-35 |
| 混凝去除效率η | $\eta = (R/W - FILT)/(R/W + \epsilon)$ | step0_preprocess.py | 372 |
| 单位矾耗φ | $\phi = ALUM/(R/W + \epsilon)$ | step0_preprocess.py | 376 |
| 污染物总通量Π | $\Pi = R/W \cdot FLOW$ | step0_preprocess.py | 416 |
| 时间循环编码 | $\sin(2\pi h/24), \cos(2\pi h/24)$ | step0_preprocess.py | 386-389 |
| SHAP Shapley值 | $\phi_j = \sum |\phi_j^{(t)}|/n$ | step1.0_feature_selection.py | 41-43 |
| Permutation重要性 | $\tilde{I}_j = \Delta RMSE \times 10$ | step1.0_feature_selection.py | 50-64 |
| 鲁棒融合重要性 | $I_j^{robust} = \sqrt{\phi_j' \cdot \tilde{I}_j'}$ | step1.0_feature_selection.py | 68 |
| 方差倒数加权 | $w_m = 1/\sigma_m^2$ | step1.1_model_comparison.py | 177-180 |
| Huber Loss | $\ell(r) = \{r^2/2, |r|\le\delta; \delta(|r|-\delta/2), |r|>\delta\}$ | step0_config.py | HUBER_DELTA=1.0 |
| 一级反应动力学 | $-\ln(\eta_{coag})$ | step0_preprocess.py | 400 |
| GAM偏依赖 | $\hat{y} = \sum s_i(x_i)$ | step1.1_model_comparison.py | 300-325 |
| Ridge显式公式 | $\log(1+NTU) = \beta_0 + \sum \beta_i x_i$ | step1.1_model_comparison.py | 243-295 |

---

## 特征工程映射

| 层级 | 维度 | 构造位置 |
|:---:|:---:|------|
| L1 原始传感器 | 22 | step0_preprocess.py L354-366 |
| L2 物理衍生 | 9 | step0_preprocess.py L368-394 |
| L2+ 幂次展开 | 12 | step0_preprocess.py L396-434 |
| L3 历史滞后 | 15 | step0_preprocess.py L441-450 |
| L4 趋势聚合 | 40 | step0_preprocess.py L455-464 |
| L5 交互特征 | 4 | step0_preprocess.py L469-478 |

---

## 消融实验映射

| 消融组 | 使用特征索引 | step5.0_ablation.py |
|------|:---:|:---:|
| L1 only | [0:22) | 93 |
| L1+L2 | L1∪L2 | 94 |
| L1+L2+L3 | L1∪L2∪L3 | 95 |
| L1+L2+L3+L4 | L1∪L2∪L3∪L4 | 96 |
| +L5 | 全L1-L5 | 97 |
| +L2+ | 全L1-L5+L2+ | 98 |
| remove FILT_NTU(raw) | 全量-{FILT_NTU} | 105-107 |
| remove ALL_FILT(17) | 全量-17FILT系列 | 110-114 |

---

*变更记录: 2026-07-24 Q1完成, 填充映射*
