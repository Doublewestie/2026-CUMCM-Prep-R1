# Sum 9: Project Cleanup, Refactor, and Verification

> TCN ablation removal, unified data loader, archive cleanup, naming convention restoration | 2026-07-25

## 1. Problem

Three issues flagged after Q1/Q2 closure:
- **Data loading inconsistency**: Each q2_*.py script loaded raw Excel with different NaN/column handling
- **Dead TCN code**: step5.0_ablation.py still carried 278 lines of PyTorch TCN code (N_SPLITS_Q2=2, undefined config refs)
- **Archive overreach**: 14 step1.x/step2.x files accidentally moved to archive, breaking step numbering continuity

## 2. Fixes

### 2a. Unified Data Loader

Created `data_loader.py` with three functions:
- `load_clean_data()` — loads clean_data.csv from step0_preprocess with consistent NaN handling
- `get_time_features()` — unified hour/month sin/cos encoding
- `get_arrays()` — standardized numpy array extraction

All future Q2 scripts should import from here.

### 2b. TCN Surgical Removal

Removed from step5.0_ablation.py:
- 5 PyTorch model classes (CausalConv1d, TCNBlock, TemporalAttention, Q2TCN, Q2GRU)
- 3 helper functions (huber_loss, train_q2_model, load_and_align, build_sequences)
- 1 ablation runner (run_q2_ablation with N_SPLITS_Q2=2)
- All torch imports

Kept: Q1 XGBoost ablation + new log-AR ablation (8 configs, 5-fold CV).

### 2c. Archive Cleanup

| Action | Files |
|--------|-------|
| Restored to root | 14 step1.x/step2.x/q1_*.py files |
| Kept in archive | 15 truly obsolete scripts (q2_balance_test, q2_extreme_rule, etc.) |
| Total root py files | 26 (step0.0 through step5.0, continuous) |

### 2d. eta_coag Annotation

Added detailed comment in step1.4_feature_importance.py explaining that FILT_NTU is a legitimate physical input to the CSTR chain, not data leakage.

## 3. Verification (per verification-before-completion skill)

| Check | Result |
|-------|--------|
| Step file continuity | 26 files, step0.0->step5.0 |
| Q1 figures | 4 png, English labels, R2=0.8072 |
| Q2 figures | 4 png, English labels, R2=0.6955 |
| Tables | 10 csv, all current |
| data_loader.py | 4375 rows loaded successfully |
| clean_data.csv | 474KB present |
| git status | clean, 9 commits ahead of origin |
| Errors | 0 |

## 4. Git History (9 commits ahead of origin)

```
14dc095 fix: restore step1.x/step2.x files to root
b2a90a4 refactor: remove old TCN/XGBoost, create data_loader.py
a49a55f docs(tables): update Q1/Q2 result tables
7b8268b feat(q1): Balance Detector R2=0.807
be78f3b fix(q1): correct R2 to 0.7827
a6b97b9 refactor(q1): CSTR figures, remove old XGBoost plots
112ae73 fix(q2): TS-CV R2 to per-fold mean 0.6955
7bb02d5 feat(q2): log-AR(6)+RidgeCV final model
c31fea7 feat(Q2): final log-AR(6)+RidgeCV + full exploration
```
