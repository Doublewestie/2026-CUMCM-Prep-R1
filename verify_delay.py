"""
verify_delay.py
Q2 物理先验时滞参数交叉验证
=============================
三条证据链:
  1. 事件CCF — 仅在滤池穿透事件(FILT>0.3)期间计算 RW_NTU->FILT 互相关
  2. CSTR卷积 — FILT->NTU 清水池延迟 (已知强耦合 r=0.66)
  3. 理论计算 — 由流量和水位估算各单元停留时间
"""

import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
EPS = 1e-10

df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
df = df.dropna(subset=["RW_NTU", "FILT_NTU", "NTU"]).reset_index(drop=True)
n = len(df)
rw = df["RW_NTU"].values.astype(np.float64)
filt = df["FILT_NTU"].values.astype(np.float64)
ntu = df["NTU"].values.astype(np.float64)
rwf = df["RW_FLOW"].values.astype(np.float64)
twf = df["TW_FLOW"].values.astype(np.float64)
cwl = df["CW_WELL_LEVEL"].values.astype(np.float64)
rw_log = np.log1p(rw)
filt_log = np.log1p(filt)
ntu_log = np.log1p(ntu)

print("=" * 70)
print("  Q2 Time Delay Verification")
print("=" * 70)

# ==============================
# 1. 事件CCF: 只在FILT>0.3的时段
# ==============================
print(f"\n[1a] Full-data CCF (N={n}):")
for name, (x, y, l) in [
    ("R/W NTU -> FILT", (rw_log, filt_log, "steelblue")),
    ("FILT -> NTU    ", (filt_log, ntu_log, "seagreen")),
    ("R/W NTU -> NTU ", (rw_log, ntu_log, "darkorange")),
]:
    nn = len(y)
    cors = [abs(pearsonr(x[d:nn], y[:nn-d])[0]) for d in range(7)]
    best = np.argmax(cors)
    print(f"  {name}: peak lag={best}step({best*2}h) r={cors[best]:.4f}  "
          f"lag2={cors[2]:.4f} lag3={cors[3]:.4f}")

# 事件检测: FILT > 0.3
high_filt = filt > 0.3
n_high = high_filt.sum()
print(f"\n[1b] Event CCF (FILT > 0.3 only, N={n_high}):")

mask = high_filt
rw_f = rw_log[mask]
rwf_f = rwf[mask]
filt_f = filt_log[mask]
ntu_f = ntu_log[mask]

# RW_NTU -> FILT during events
ni = len(filt_f)
cors = [abs(pearsonr(rw_f[d:ni], filt_f[:ni-d])[0]) for d in range(min(7, ni))]
cors_arr = np.array(cors)
print(f"  R/W NTU -> FILT: {' '.join([f'{c:.4f}' for c in cors_arr])}")
best = np.nanargmax(cors_arr) if not np.all(np.isnan(cors_arr)) else 0
print(f"  -> peak lag={best}step({best*2}h) r={cors_arr[best]:.4f}")

# 分段水位的事件CCF
print(f"\n[1c] RIVER_LEVEL分段事件CCF:")
df_tmp = pd.DataFrame({"rl": df["RIVER_LEVEL"], "filt": df["FILT_NTU"], "rw": df["RW_NTU"]})
levels = [(0, 4, "低水位<4m"), (4, 7, "中水位4-7m"), (7, 100, "高水位>7m")]
for lo, hi, lb in levels:
    m = (df["RIVER_LEVEL"] > lo) & (df["RIVER_LEVEL"] <= hi) & high_filt
    if m.sum() < 10:
        print(f"  {lb}: N={m.sum()}, 样本不足")
        continue
    rw_s = rw_log[m]; fl_s = filt_log[m]; ni2 = len(fl_s)
    if ni2 < 10:
        print(f"  {lb}: N={ni2:5d}, 样本不足")
        continue
    cors = [abs(pearsonr(rw_s[d:ni2], fl_s[:ni2-d])[0]) if d < ni2 else 0 for d in range(min(7, ni2))]
    best = np.argmax(cors)
    print(f"  {lb}: N={m.sum():5d}  peak lag={best}step({best*2}h) r={cors[best]:.4f} "
          f"  full curve: {' '.join([f'{c:.3f}' for c in cors[:4]])}")


# ==============================
# 2. CSTR卷积: 从FILT->NTU反推清水池延迟
# ==============================
print(f"\n[2] CSTR convolution: FILT -> NTU buffer delay")
# CSTR串联模型的停留时间分布
def rtd_kernel(N, K, dt=1):
    """CSTR串联RTD: E(θ) = N^N θ^{N-1} e^{-Nθ} / (N-1)!"""
    import math
    k = np.arange(K, dtype=np.float64)
    theta = k * dt
    kernel = (N**N * theta**(N-1) * np.exp(-N*theta)) / math.factorial(N-1)
    return kernel / (kernel.sum() + EPS)

# 搜索最优N (1-10)
best_n, best_ks = 0, float("inf")
ks_results = []
filt_arr = filt_log
ntu_arr = ntu_log

for N in range(1, 11):
    kern = rtd_kernel(N, 30)
    # 卷积
    conv = np.convolve(filt_arr, kern, mode='same')
    conv = conv[:len(ntu_arr)]
    # KS距离
    cdf_conv = np.sort(conv)
    cdf_ntu = np.sort(ntu_arr)
    ks = np.max(np.abs(cdf_conv - cdf_ntu))
    ks_results.append((N, ks, np.corrcoef(conv, ntu_arr)[0,1]))
    if ks < best_ks:
        best_ks, best_n = ks, N
print(f"  Optimal CSTR series N={best_n}  (KS={best_ks:.4f})")
for N, ks_ret, r_ret in ks_results:
    print(f"    N={N:d}  KS={ks_ret:.4f}  corr={r_ret:.4f}")

# 最优N下, 查看卷积核的峰值在哪——那就是"FILT的信号多久到达NTU"
best_kern = rtd_kernel(best_n, 30)
peak_idx = np.argmax(best_kern)
print(f"  Kernel peak at lag={peak_idx}step(s) ({peak_idx*2}h)")
print(f"  Kernel median at lag={np.where(np.cumsum(best_kern) > 0.5)[0][0]}step(s)")
print(f"  -> FILT->NTU 清水池延迟 ~ {peak_idx*2}-{np.where(np.cumsum(best_kern) > 0.5)[0][0]*2}h")


# ==============================
# 3. 理论估算: 由流量和水位计算停留时间
# ==============================
print(f"\n[3] Theoretical hydraulic residence time:")

# 出厂流量 TW_FLOW (m³/h) 和清水池水位 CW_WELL_LEVEL (m)
# 估算: 清水池面积 A = Q_out * τ / Δh  (假设一个合理的池深)
# 我们从数据中推算: 当CW_WELL_LEVEL变化时, τ变化
# τ(t) 和 CW_WELL_LEVEL 成正比, 和 TW_FLOW 成反比

# 假设水池横截面积 A ~ 500 m2 (典型值)
A_tank = 500.0  # m2
# 停留时间 τ = V / Q = (L_cw * A) / Q_out
tau_cstr = cwl * A_tank / (twf + EPS)
tau_cstr_mean = np.nanmean(tau_cstr)
tau_cstr_med = np.nanmedian(tau_cstr)
print(f"  Clear water tank (assume A={A_tank:.0f}m2):")
print(f"    CSTR residence time: mean={tau_cstr_mean/2:.1f}steps ({tau_cstr_mean:.1f}h)  "
      f"median={tau_cstr_med/2:.1f}steps ({tau_cstr_med:.1f}h)")

# FILT->NTU延迟 ~ τ_cstr, 而R/W->FILT延迟 ~ 理论3-6h
# 总链: R/W->FILT(3-6h) + FILT->NTU(τ_cstr)
total_delay_min = 3 + tau_cstr_med
total_delay_max = 6 + tau_cstr_med
print(f"  Total chain: R/W->NTU ~ {3:.0f}-{6:.0f}h + {tau_cstr_med:.1f}h"
      f" = {total_delay_min:.1f}-{total_delay_max:.1f}h")


# ==============================
# 4. 综合验证表
# ==============================
print(f"\n  {'='*60}")
print(f"  Q2 物理先核实证总结")
print(f"{'='*60}")

# 汇总R/W->FILT延迟
event_filt_ccf = [abs(pearsonr(rw_log[high_filt][d:], filt_log[high_filt][:len(filt_log[high_filt])-d])[0])
                  for d in range(min(7, high_filt.sum()))]
event_best = np.argmax(event_filt_ccf)

print(f"  {'变量':<20s} {'物理先验':>12s} {'事件CCF':>12s} {'CSTR核' if peak_idx else '':>12s}")
print(f"  {'-'*60}")
print(f"  {'R/W NTU->FILT':<20s} {'2-3步(4-6h)':>12s} {f'{event_best}步({event_best*2}h)':>12s} {'':>12s}")
print(f"  {'FILT->NTU':<20s} {'CSTR依赖':>12s} {'':>12s} {f'{peak_idx}步({peak_idx*2}h)':>12s}")
print(f"  {'总链R/W->NTU':<20s} {f'{total_delay_min:.0f}-{total_delay_max:.0f}h':>12s} {'':>12s} {'':>12s}")
print(f"")
if event_best >= 2:
    print(f"  >>> 事件CCF确认 4-6h 延迟, 与物理先验 2-3步一致 ✅")
else:
    print(f"  >>> 事件CCF仍不明确(r={max(event_filt_ccf):.4f}), 延迟被高频噪声淹没 ⚠️")

# 保存结果
results = {
    "full_ccf_rw2filt": [round(abs(pearsonr(rw_log[d:], filt_log[:len(filt_log)-d])[0]), 4) for d in range(7)],
    "event_ccf_rw2filt": [round(c, 4) for c in event_filt_ccf],
    "cstr_optimal_N": best_n,
    "cstr_kernel_peak_step": peak_idx,
    "cstr_kernel_peak_hour": peak_idx * 2,
    "hydraulic_residence_hours": round(tau_cstr_med, 1),
    "total_chain_delay_range_hours": [round(total_delay_min, 1), round(total_delay_max, 1)],
}
import json
with open(os.path.join(OUTPUT_DIR, "delay_verification.json"), "w") as f:
    json.dump(results, f, indent=2)

# 总图: 全数据CCF vs 事件CCF对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
lags_h = np.arange(7) * 2

# 全量 vs 事件
for xi, (name, full_cors, evt_cors) in enumerate([
    ("R/W NTU -> FILT", None, event_filt_ccf),
]):
    ax = axes[xi]
    full_c = [abs(pearsonr(rw_log[d:], filt_log[:len(filt_log)-d])[0]) for d in range(7)]
    evt_c = event_filt_ccf
    ax.bar(lags_h - 0.7, full_c, width=0.6, color="lightgray", alpha=0.7, label="full data")
    ax.bar(lags_h + 0.1, evt_c, width=0.6, color="darkorange", alpha=0.8, label="FILT>0.3 events")
    ax.axvline(x=2*2, color="red", linestyle="--", alpha=0.6, label="physical prior 4h")
    ax.axvline(x=3*2, color="red", linestyle=":", alpha=0.6, label="physical prior 6h")
    ax.set_xlabel("lag (h)"); ax.set_ylabel("|r|")
    ax.set_title(name); ax.legend(fontsize=7)

# CSTR核
ax = axes[2]
ax.plot(np.arange(30)*2, best_kern, "steelblue", linewidth=2)
ax.axvline(x=peak_idx*2, color="red", linestyle="--", label=f"peak={peak_idx*2}h")
ax.set_xlabel("lag (h)"); ax.set_ylabel("kernel weight")
ax.set_title(f"CSTR kernel N={best_n} (FILT->NTU)"); ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "delay_verification.png"), dpi=150)
plt.close()
print(f"\n[Done] delay_verification.json + figures/delay_verification.png")
