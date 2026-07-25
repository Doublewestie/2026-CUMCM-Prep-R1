"""
q2_predict_feb2026.py — 预测 2026年2月 FILT.NTU 缺失值
=========================================================
方法: AR(6) 自回归模型 (Q2已验证 R²=0.52)
      用 2025 全年数据训练, Jan 2026 最后6步做种子
"""
import numpy as np
import pandas as pd
import os
import sys
import io
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_2025 = os.path.join(OUTPUT_DIR, "clean_data.csv")
DATA_DIR_2026 = os.path.join(BASE_DIR, "..", "题目", "第一次模拟训练题目",
                              "B题", "B题附件", "附件2  2026数据集")

# ================================================================
#  加载 2025 数据 + 训练 AR(6)
# ================================================================
print("=" * 60)
print("  Q2: FILT.NTU Feb 2026 Prediction")
print("=" * 60)

df_2025 = pd.read_csv(DATA_2025)
df_2025["DATE"] = pd.to_datetime(df_2025["DATE"])
df_2025 = df_2025.sort_values("DATE").reset_index(drop=True)

filt = df_2025["FILT_NTU"].values
print(f"\n  2025 data: {len(df_2025)} rows")
print(f"  FILT_NTU: mean={filt.mean():.4f}, std={filt.std():.4f}")

# AR(6) 训练
AR_ORDER = 6
p_order = AR_ORDER
n = len(filt)
X_ar = np.column_stack([np.roll(filt, i) for i in range(1, p_order + 1)])
X_ar = X_ar[p_order:, :]
y_ar = filt[p_order:]

ar_model = LinearRegression()
ar_model.fit(X_ar, y_ar)

# 训练集 R²
y_pred_train = np.zeros(n)
y_pred_train[:p_order] = filt[:p_order]
for t in range(p_order, n):
    y_pred_train[t] = ar_model.intercept_ + np.dot(ar_model.coef_, filt[t - p_order:t][::-1])

train_r2 = r2_score(filt[p_order:], y_pred_train[p_order:])
train_rmse = np.sqrt(mean_squared_error(filt[p_order:], y_pred_train[p_order:]))

print(f"  AR({AR_ORDER}) training: R2 = {train_r2:.4f}, RMSE = {train_rmse:.4f}")
print(f"  Coefs: {[round(c, 4) for c in ar_model.coef_]}")
print(f"  Intercept: {ar_model.intercept_:.4f}")

# ================================================================
#  加载 2026 年1月数据 (Feb的种子在Jan的末尾)
# ================================================================
# 读取2026年1月.xls
jan_path = os.path.join(DATA_DIR_2026, "2026年1月.xls")
feb_path = os.path.join(DATA_DIR_2026, "2026年2月.xls")

df_jan = pd.read_excel(jan_path)
df_feb = pd.read_excel(feb_path)

# 标准化列名
col_map = {
    "TIME ": "TIME", "RIVER LEVEL": "RIVER_LEVEL",
    "R/W PUMP DUTY": "RW_PUMP_DUTY", "R/W FLOW": "RW_FLOW",
    "R/W NTU": "RW_NTU", "R/W CLR": "RW_CLR", "R/W PH": "RW_PH",
    "FILT. NTU": "FILT_NTU", "C/W WELL LEVEL": "CW_WELL_LEVEL",
    "F/RIDE": "F_RIDE", "T/W PUMP DUTY": "TW_PUMP_DUTY",
    "T/W FLOW": "TW_FLOW", "18ML LEVEL": "18ML_LEVEL",
    "18ML FLOW": "18ML_FLOW",
}
for df in [df_jan, df_feb]:
    df.rename(columns=col_map, inplace=True)

# 提取 Jan 的 FILT_NTU (有值, 12个)
filt_jan = pd.to_numeric(df_jan["FILT_NTU"], errors="coerce").values
print(f"\n  Jan 2026 FILT_NTU: {[round(v, 4) for v in filt_jan]}")

# 提取 Feb 的 TIME (仅有 TIME 列, FILT 全 NaN)
time_feb = df_feb["TIME"].values
filt_feb_actual = pd.to_numeric(df_feb["FILT_NTU"], errors="coerce").values
print(f"  Feb 2026 TIME: {list(time_feb)}")
print(f"  Feb 2026 FILT_NTU (observed, all NaN?): {filt_feb_actual}")

# ================================================================
#  预测 Feb 2026 的 FILT_NTU
# ================================================================
# 种子: Jan 2026 的最后 6 个 FILT_NTU 值
seed = filt_jan[-AR_ORDER:]
print(f"\n  AR({AR_ORDER}) seed (last 6 of Jan): {[round(v, 4) for v in seed]}")

n_pred = len(time_feb)
pred = np.zeros(n_pred)
pred_vals = seed.copy()

for t in range(n_pred):
    pred[t] = ar_model.intercept_ + np.dot(ar_model.coef_, pred_vals[::-1])
    pred_vals = np.roll(pred_vals, -1)
    pred_vals[-1] = pred[t]

print(f"\n  --- Feb 2026 FILT_NTU Predictions ---")
print(f"  {'TIME':<8} {'FILT_NTU_pred':<16}")
print(f"  {'-'*24}")
for i, tm in enumerate(time_feb):
    print(f"  {str(tm).strip():<8} {pred[i]:<16.4f}")

# 保存
out_table = pd.DataFrame({
    "TIME": [str(t).strip() for t in time_feb],
    "FILT_NTU_pred": pred,
})
out_path = os.path.join(OUTPUT_DIR, "q2_feb2026_predictions.csv")
out_table.to_csv(out_path, index=False, encoding="utf-8-sig")

# 完整的 2026 FILT 预测报表
print(f"\n  预测已保存至: {out_path}")

# ================================================================
#  预测精度评估 (与真实值对比)
# ================================================================
# Feb 实际 FILT_NTU 有值 (NTU缺失, FILT有)
valid_mask = ~np.isnan(filt_feb_actual)
if valid_mask.sum() > 0:
    actual_valid = filt_feb_actual[valid_mask]
    pred_valid = pred[valid_mask]
    rmse = np.sqrt(mean_squared_error(actual_valid, pred_valid))
    r2 = r2_score(actual_valid, pred_valid)
    mae = np.mean(np.abs(actual_valid - pred_valid))
else:
    rmse, r2, mae = np.nan, np.nan, np.nan

print(f"\n  --- Prediction vs Actual (Feb 2026) ---")
print(f"  {'TIME':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
print(f"  {'-'*44}")
for i in range(len(time_feb)):
    a = filt_feb_actual[i]
    pv = pred[i]
    err = a - pv if not np.isnan(a) else np.nan
    if not np.isnan(a):
        print(f"  {str(time_feb[i]).strip():<8} {a:<12.4f} {pred[i]:<12.4f} {err:<12.4f}")
    else:
        print(f"  {str(time_feb[i]).strip():<8} {'NaN':<12} {pred[i]:<12.4f} {'N/A':<12}")

print(f"\n  Prediction Accuracy:")
print(f"    RMSE = {rmse:.4f}")
print(f"    R²   = {r2:.4f}")
print(f"    MAE  = {mae:.4f}")

# ================================================================
#  综合报告
# ================================================================
print(f"""
{'='*60}
  Q2 Final Report: FILT.NTU Dynamic Model & Prediction
{'='*60}

  1. Dynamic Model: AR(6) on {n} training rows
     训练 R² = {train_r2:.4f},  RMSE = {train_rmse:.4f}
     系数: [{', '.join(f'{c:.4f}' for c in ar_model.coef_)}]
     截距: {ar_model.intercept_:.4f}

  2. Feb 2026 Prediction vs Actual
     RMSE = {rmse:.4f}  |  R² = {r2:.4f}  |  MAE = {mae:.4f}

  3. Time Delay Parameters
     tau_RW_NTU -> FILT  = 2 steps (4h)  [Q1 softmax + engineering prior]
     tau_ALUM   -> FILT  = 2 steps (4h)  [same control loop]
     tau_FLOW   -> FILT  = 1 step  (2h)  [hydraulic propagation]
     tau_PH     -> FILT  = 1 step  (2h)  [immediate effect]

  4. Key Physical Findings
     - CCF/MIC/TE all failed: control loop masks causal signals
     - Physics-structured scan: 所有7路R²<0, tau不可辨识
     - FILT is ~98% AR(1) driven; exogenous contribution <2%
     - Pseudo-data verification: method works when signal exists
     - C_phys = 0.00273 -> physical segment removal = 99.73%

   5. Feb 2026 FILT_NTU Predictions (AR(6) seeded from Jan)
""")

for i, tm in enumerate(time_feb):
    print(f"     {str(tm).strip():8s}  {pred[i]:.4f} NTU")

print(f"""
   6. Prediction Method
      Model: AR({AR_ORDER}) linear autoregression
      Training: 2025 FILT_NTU time series (n={n})
      Seed: Jan 2026 FILT_NTU last {AR_ORDER} values
      Rationale: AR outperformed TCN, CCF, MIC, TE, and physics models
                 on FILT data due to closed-loop signal suppression

   7. Data Flow to Q3
      tau*_total = 2 steps (4h) -> source A input alignment
      C_phys     = 0.00273       -> physics-informed loss
      Zone labels (comfort/stress) -> dual-mode prediction strategy
""")
print(f"{'='*60}")
