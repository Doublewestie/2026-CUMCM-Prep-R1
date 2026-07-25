"""
q2_predict_feb2026_tiered.py — Q1 分区思路改进 Q2 预测
===========================================================
方法: 三级分箱 AR(6) × 动态模型切换
  T1 (≤0.05): 噪声区, AR(6)
  T2 (0.05~0.15): 过渡区, AR(6)
  T3 (>0.15): 应力区, AR(6)
  预测时根据上一步值动态切换模型
"""
import numpy as np
import pandas as pd
import os
import sys
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_2025 = os.path.join(OUTPUT_DIR, "clean_data.csv")
DATA_DIR_2026 = os.path.join(BASE_DIR, "data", "2026")

TIER_THRESHOLDS = [0.05, 0.15]
AR_ORDER = 6

print("=" * 60)
print("  Q2 Tiered: 三级分箱 AR 预测 FILT.NTU")
print("=" * 60)

# ================================================================
#  加载 2025 数据 + 分箱训练
# ================================================================
df_2025 = pd.read_csv(DATA_2025)
df_2025["DATE"] = pd.to_datetime(df_2025["DATE"])
df_2025 = df_2025.sort_values("DATE").reset_index(drop=True)

filt = df_2025["FILT_NTU"].values

# 分箱
df_2025["tier"] = 1
df_2025.loc[df_2025["FILT_NTU"] > TIER_THRESHOLDS[0], "tier"] = 2
df_2025.loc[df_2025["FILT_NTU"] > TIER_THRESHOLDS[1], "tier"] = 3

print(f"\n  2025 data: {len(df_2025)} rows")
print(f"  T1 (<=0.05): {(df_2025.tier==1).sum()} ({100*(df_2025.tier==1).sum()/len(df_2025):.1f}%)")
print(f"  T2 (0.05~0.15): {(df_2025.tier==2).sum()} ({100*(df_2025.tier==2).sum()/len(df_2025):.1f}%)")
print(f"  T3 (>0.15): {(df_2025.tier==3).sum()} ({100*(df_2025.tier==3).sum()/len(df_2025):.1f}%)")

# 为每个 tier 训练 AR(6)
tier_models = {}
tier_r2 = {}
for tier_id in [1, 2, 3]:
    mask = df_2025["tier"] == tier_id
    sub = filt[mask.values]
    if len(sub) <= AR_ORDER + 5:
        print(f"  T{tier_id}: 样本不足 ({len(sub)}), 跳过AR训练，使用零模型")
        tier_models[tier_id] = {"coef": np.zeros(AR_ORDER), "intercept": float(np.mean(sub))}
        tier_r2[tier_id] = float("-inf")
        continue

    n_t = len(sub)
    X_t = np.column_stack([np.roll(sub, i) for i in range(1, AR_ORDER + 1)])
    X_t = X_t[AR_ORDER:, :]
    y_t = sub[AR_ORDER:]

    m = LinearRegression()
    m.fit(X_t, y_t)

    yp_t = np.zeros(n_t)
    yp_t[:AR_ORDER] = sub[:AR_ORDER]
    for t in range(AR_ORDER, n_t):
        yp_t[t] = m.intercept_ + np.dot(m.coef_, sub[t - AR_ORDER:t][::-1])

    r2_t = r2_score(sub[AR_ORDER:], yp_t[AR_ORDER:])
    rmse_t = np.sqrt(mean_squared_error(sub[AR_ORDER:], yp_t[AR_ORDER:]))

    tier_models[tier_id] = {"coef": m.coef_, "intercept": m.intercept_}
    tier_r2[tier_id] = r2_t
    print(f"  T{tier_id} AR({AR_ORDER}): n={n_t}, R2={r2_t:.4f}, RMSE={rmse_t:.4f}")
    print(f"    Coefs: {[f'{c:.4f}' for c in m.coef_]}, Intercept={m.intercept_:.4f}")

# ================================================================
#  全局 AR(6) 作为对比基线
# ================================================================
n_all = len(filt)
X_all = np.column_stack([np.roll(filt, i) for i in range(1, AR_ORDER + 1)])
X_all = X_all[AR_ORDER:, :]
y_all = filt[AR_ORDER:]
m_all = LinearRegression()
m_all.fit(X_all, y_all)

yp_all = np.zeros(n_all)
yp_all[:AR_ORDER] = filt[:AR_ORDER]
for t in range(AR_ORDER, n_all):
    yp_all[t] = m_all.intercept_ + np.dot(m_all.coef_, filt[t - AR_ORDER:t][::-1])

r2_all = r2_score(filt[AR_ORDER:], yp_all[AR_ORDER:])
rmse_all = np.sqrt(mean_squared_error(filt[AR_ORDER:], yp_all[AR_ORDER:]))
print(f"\n  全局 AR({AR_ORDER}): R2={r2_all:.4f}, RMSE={rmse_all:.4f}")

# ================================================================
#  2026 Feb 预测 (动态模型切换)
# ================================================================
jan_path = os.path.join(DATA_DIR_2026, "2026年1月.xls")
feb_path = os.path.join(DATA_DIR_2026, "2026年2月.xls")

df_jan = pd.read_excel(jan_path)
df_feb = pd.read_excel(feb_path)

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

filt_jan = pd.to_numeric(df_jan["FILT_NTU"], errors="coerce").values
time_feb = df_feb["TIME"].values
filt_feb_actual = pd.to_numeric(df_feb["FILT_NTU"], errors="coerce").values

# Tier-dynamic prediction
seed = filt_jan[-AR_ORDER:].copy()
n_pred = len(time_feb)

pred_tiered = np.zeros(n_pred)
pred_vals = seed.copy()

# 确定初始 tier
current_vals = seed[-1] if len(seed) > 0 else 0.10
if current_vals <= TIER_THRESHOLDS[0]:
    current_tier = 1
elif current_vals <= TIER_THRESHOLDS[1]:
    current_tier = 2
else:
    current_tier = 3

tier_history = [current_tier]

for t in range(n_pred):
    model = tier_models.get(current_tier, tier_models.get(2))
    pred_tiered[t] = model["intercept"] + np.dot(model["coef"], pred_vals[::-1])

    # 更新 pred_vals
    pred_vals = np.roll(pred_vals, -1)
    pred_vals[-1] = pred_tiered[t]

    # 根据预测值切换 tier
    if pred_tiered[t] <= TIER_THRESHOLDS[0]:
        current_tier = 1
    elif pred_tiered[t] <= TIER_THRESHOLDS[1]:
        current_tier = 2
    else:
        current_tier = 3
    tier_history.append(current_tier)

# 也是全局 AR(6) 预测作对比
pred_global = np.zeros(n_pred)
pred_gv = seed.copy()
for t in range(n_pred):
    pred_global[t] = m_all.intercept_ + np.dot(m_all.coef_, pred_gv[::-1])
    pred_gv = np.roll(pred_gv, -1)
    pred_gv[-1] = pred_global[t]

# ================================================================
#  精度对比
# ================================================================
valid = ~np.isnan(filt_feb_actual)
a = filt_feb_actual[valid]

pt = pred_tiered[valid]
r2_tiered = r2_score(a, pt)
rmse_tiered = np.sqrt(mean_squared_error(a, pt))

pg = pred_global[valid]
r2_global = r2_score(a, pg)
rmse_global = np.sqrt(mean_squared_error(a, pg))

print(f"\n{'='*60}")
print(f"  Feb 2026 预测结果对比")
print(f"{'='*60}")
print(f"\n  {'TIME':<8} {'Actual':<8} {'Global AR':<10} {'Tiered AR':<10} {'Tier at t':<10}")
print(f"  {'-'*46}")

tier_labels = {1: "T1", 2: "T2", 3: "T3"}
for i in range(n_pred):
    a_str = f"{filt_feb_actual[i]:.4f}" if not np.isnan(filt_feb_actual[i]) else "NaN"
    print(f"  {str(time_feb[i]).strip():<8} {a_str:<8} {pred_global[i]:<10.4f} "
          f"{pred_tiered[i]:<10.4f} {tier_labels.get(tier_history[i], '?'):<10}")

print(f"\n  精度对比:")
print(f"  {'Method':<20} {'R²':<10} {'RMSE':<10} {'Δ vs Global':<15}")
print(f"  {'-'*55}")
print(f"  {'Global AR(6)':<20} {r2_global:<10.4f} {rmse_global:<10.4f} {'—':<15}")
print(f"  {'Tiered AR(6)':<20} {r2_tiered:<10.4f} {rmse_tiered:<10.4f} "
      f"{'+' if r2_tiered > r2_global else ''}{r2_tiered - r2_global:<+.4f}")

print(f"\n  三级模型使用情况:")
for t in range(1, 4):
    count = tier_history.count(t)
    print(f"    T{t} ({tier_labels[t]}): 使用了 {count} 步")

# ================================================================
#  输出
# ================================================================
out = pd.DataFrame({
    "TIME": [str(t).strip() for t in time_feb],
    "Actual": [float(filt_feb_actual[i]) for i in range(n_pred)],
    "Global_AR": pred_global,
    "Tiered_AR": pred_tiered,
    "Tier_used": [tier_history[i] for i in range(n_pred)],
})
out_path = os.path.join(OUTPUT_DIR, "q2_feb2026_tiered_predictions.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n  预测保存: {out_path}")

print(f"""
{'='*60}
  结论: 分箱 AR 预测 vs 全局 AR 预测
{'='*60}

  全局 AR(6): 单一模型, 不考虑 FILT 的级区差异
  分箱 AR(6): 在 T1/T2/T3 各区间训练独立的 AR 模型
             预测时根据当前值的级区动态切换
            
  预期: 当 FILT 在级区间大幅穿越时, 分箱模型应更准确
        (因各区的 AR 系数已捕捉到该区特有的自回归模式)
  
  实际改善: 见上方 R² 和 RMSE 对比
  {'='*60}
""")
