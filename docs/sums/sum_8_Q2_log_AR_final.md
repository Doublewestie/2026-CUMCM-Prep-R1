# Sum 8: Q2 Final — log(FILT) AR(6) + RidgeCV

> log(FILT+eps) transform + AR(6) autoregression + RidgeCV with automatic alpha selection | 2026-07-25

## 1. Problem

Q2 requires building a dynamic model of FILT.NTU (filtered turbidity) and estimating each input variable's time-delay parameter. Core difficulty: 99.7% removal efficiency drowns causal signals.

## 2. Approach Evolution

| Phase | Method | Best R2 | Status |
|-------|--------|:-------:|:------:|
| 1 | CCF/MIC/TE statistical lag estimation | all failed | abandoned |
| 2 | TCN + attention + PINN physics loss | -0.15 | abandoned |
| 3 | LRV = log10(RW/FILT) + negative feedback decoupling | 0.38-0.52 | abandoned |
| 4 | CSTR inverse formula | -0.69 | unstable |
| 5 | **log(FILT+1e-3) AR(6) + RidgeCV** | **0.6955** | **FINAL** |

## 3. Final Model

```
log(FILT(t) + 1e-3) = c + phi_1*log(FILT(t-1) + 1e-3) + ...
                         + phi_6*log(FILT(t-6) + 1e-3) + epsilon(t)
```

- Estimator: RidgeCV (alpha candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
- Optimal alpha: 1.0 (selected automatically by CV)
- All 5 folds select alpha=1.0, indicating stable regularization needs

## 4. Precision

| Metric | In-Sample | TS-CV (5-fold) |
|--------|:---------:|:--------------:|
| R2 | 0.7903 | 0.6955 +- 0.084 |
| RMSE | 0.3023 | 0.2412 |
| MAE | — | 0.0626 |

## 5. Time-Delay Parameters (from physics prior + event CCF)

| Input | tau | Evidence |
|-------|:---:|----------|
| RW_NTU -> FILT | 4h (2 steps) | Event CCF under median water level |
| ALUM -> FILT | 6h (3 steps) | Coagulation-flocculation chain |
| RW_FLOW -> FILT | 2h (1 step) | Hydraulic residence time |
| RW_PH -> FILT | 2h (1 step) | pH affects coagulation instantly |

## 6. Ablation Matrix (from step5.0)

| Config | CV R2 | CV RMSE |
|--------|:----:|:-------:|
| log-AR(3) + RidgeCV | 0.6961 | 0.2413 |
| **log-AR(6) + RidgeCV** | **0.6955** | **0.2412** |
| log-AR(12) + RidgeCV | 0.6846 | 0.2491 |
| FILT-level AR(6) | 0.5725 | 0.2349 |
| log-AR(6) + ElasticNet | 0.6955 | 0.2412 |
| log-AR(6) + Huber | 0.6955 | 0.2412 |
| log-AR(6) + Ensemble-3 | 0.7019 | 0.2302 |
| log-AR(6) + ARDL-tau + Ensemble | 0.7043 | 0.2312 |

Key findings:
- log transform is critical: FILT-level AR(6) R2=0.572 vs log-AR(6) R2=0.695
- AR(3) equals AR(6) performance: no benefit from longer lags
- External variables (ARDL) add only +0.009 R2: 98%+ variance from FILT self-lags
- Ensemble adds +0.006 R2: marginal gain, baseline RidgeCV is already near-optimal

## 7. Output Files

| File | Purpose |
|------|---------|
| step2_final_model.py | Final model + answer output |
| step2_final_results.json | Structured results (tau, metrics, coefficients) |
| step2_generate_figures.py | Figure generation script |
| results/figures/q2_pred_vs_actual.png | TS-CV + in-sample scatter |
| results/figures/q2_residual_diagnostics.png | 6-panel residual diagnostics |
| results/figures/q2_feature_importance.png | Ridge coefficients |
| results/figures/q2_tier_comparison.png | R2/RMSE by FILT tier |
| results/tables/q2_tier_metrics.csv | Per-tier metrics |
| results/tables/q2_ar6_coefficients.csv | Ridge coefficients |
| step5.0_ablation.py | Q2 log-AR ablation (8 configs) |

## 8. Next Steps

- Q3: Hybrid prediction (multivariate TCN + univariate N-BEATS + RF meta-learner)
- Q4: Risk evaluation with 3D scoring + Jenks classification
- Ablation: run step0_preprocess.py first to enable Q1 feature ablation
