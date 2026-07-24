"""
step0_config.py — 全局参数 + 命名约定
===========================================
用途：所有后续 step 共用此文件中的参数。
2026-07-24 重构：Q1改为灰箱模型，删L3/L4/L5特征工程参数。
"""

import numpy as np
import os

# ==============================
# 路径
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_2025 = os.path.join(BASE_DIR, "..", "题目", "第一次模拟训练题目",
                              "B题", "B题附件", "附件1  2025数据集")
DATA_DIR_2026 = os.path.join(BASE_DIR, "..", "题目", "第一次模拟训练题目",
                              "B题", "B题附件", "附件2  2026数据集")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 物理常数（不可调）
# ==============================
NTU_STANDARD = 1.0      # 国标限值 (NTU)
DELTA_T       = 2.0      # 采样间隔 (小时)
EPS           = 1e-6     # 防除零

# ==============================
# TIME 编码
# ==============================
TIME_VALUES = [
    "0700", "0900", "1100", "1300", "1500", "1700",
    "1900", "2100", "2300", "0100", "0300", "0500"
]

# ==============================
# 标准列名（统一 Format A 和 Format B 后）
# ==============================
STD_COLS = [
    "DATE", "TIME", "RIVER_LEVEL", "RW_PUMP_DUTY",
    "RW_FLOW", "RW_NTU", "RW_CLR", "RW_PH",
    "FILT_NTU", "CW_WELL_LEVEL", "PH", "NTU",
    "CLR", "CL2", "F_RIDE", "ALUM",
    "TW_PUMP_DUTY", "TW_FLOW", "18ML_LEVEL", "18ML_FLOW",
    "REMARKS"
]

# Format B (5-8月) 缺失的列 — 需要插补
COLS_MISSING_IN_FORMAT_B = [
    "RW_PUMP_DUTY", "RW_PH", "PH", "CL2",
    "F_RIDE", "ALUM", "TW_PUMP_DUTY",
    "18ML_LEVEL", "18ML_FLOW", "REMARKS"
]

# Format A → 标准列名映射
RENAME_MAP_FORMAT_A = {
    "RIVER LEVEL":     "RIVER_LEVEL",
    "R/W PUMP DUTY":  "RW_PUMP_DUTY",
    "R/W FLOW":       "RW_FLOW",
    "R/W NTU":        "RW_NTU",
    "R/W CLR":        "RW_CLR",
    "R/W PH":         "RW_PH",
    "FILT. NTU":      "FILT_NTU",
    "C/W WELL LEVEL": "CW_WELL_LEVEL",
    "F/RIDE":          "F_RIDE",
    "T/W PUMP DUTY":  "TW_PUMP_DUTY",
    "T/W FLOW":       "TW_FLOW",
    "18ML LEVEL":     "18ML_LEVEL",
    "18ML FLOW":      "18ML_FLOW",
}

# Format B → 标准列名映射
RENAME_MAP_FORMAT_B = {
    "Data":           "DATE",
    "Time":           "TIME",
    "River Level":    "RIVER_LEVEL",
    "R/W FLOW":       "RW_FLOW",
    "R/W NTU":        "RW_NTU",
    "R/W CLR":        "RW_CLR",
    "FILT. NTU":      "FILT_NTU",
    "C/W WELL LEVEL": "CW_WELL_LEVEL",
    "T/W FLOW":       "TW_FLOW",
}

# 死列（直接丢弃）
DEAD_COLS = ["18ML_LEVEL", "18ML_FLOW"]

# F_RIDE 直接丢弃（sum_2 决定：83.1%缺失、零预测力、物理不合理）
DROP_F_RIDE = True

# ==============================
# 2025 文件清单（按月份排序）
# ==============================
FILES_2025 = [
    "JBALB_Jan2025.xlsx", "JBALB_Feb2025.xlsx", "JBALB_Mar2025.xlsx",
    "JBALB_Apr2025.xlsx", "JBALB_May2025.xlsx", "JBALB_Jun2025.xlsx",
    "JBALB_July2025.xlsx","JBALB_Aug2025.xlsx", "JBALB_Sep2025.xlsx",
    "JBALB_Oct2025.xlsx", "JBALB_Nov2025.xlsx", "JBALB_Dec2025.xlsx",
]

# 2025 文件名 → 月份编号 (1-12)
MONTH_FROM_FILE = {
    f"JBALB_{abbr}2025.xlsx": i for i, abbr in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
    )
}

# ==============================
# 泵编码 — 压缩模式
# ==============================
PUMP_COMPRESS = True  # True→整数count(0-5), False→5维0/1展开

# ==============================
# 特征工程参数（精简后，2026-07-24重构）
# ==============================
# L1保留的特征列（排除出厂同期泄露变量PH/CLR/CL2/TW_FLOW）
FEATURE_COLS_L1 = [
    "RIVER_LEVEL", "RW_FLOW", "RW_NTU", "RW_CLR", "RW_PH",
    "FILT_NTU", "CW_WELL_LEVEL", "ALUM",
]
# 泵压缩后的列名
PUMP_COLS = ["rw_pump_count", "tw_pump_count"]

# ==============================
# Q1 灰箱模型参数（2026-07-24新方案）
# ==============================
# 段1: FILT(t) = β1*FILT(t-1) + (1-β1)*RW_NTU(t)*[1-η(t)]
#   η(t) = ALUM(t) / (ALUM(t) + K_m(t) + α*CLR(t))
#   K_m(t) = K_m0 + K_m1*day_sin(t) + K_m2*day_cos(t)
# 段2: NTU(t) = β2(t)*NTU(t-1) + (1-β2(t))*FILT(t)
#   β2(t) = exp(-Δt/θ(t)), θ(t) = A*CW_WELL_LEVEL(t-1) / TW_FLOW(t-1+ε)

GREYBOX_PARAM_INIT = {
    "beta1":   0.85,    # 工艺链时间常数 (0,1)
    "Km0":     0.05,    # 基础矾需量 >0
    "Km1":     0.01,    # K_m季节正弦分量
    "Km2":     0.01,    # K_m季节余弦分量
    "alpha":   0.005,   # CLR竞争系数 >0
    "A":       100.0,   # 清水池等效底面积 m²
    "FILT0":   0.2,     # FILT递推初始值
    "NTU0":    0.3,     # NTU递推初始值
}

GREYBOX_PARAM_BOUNDS = {
    "beta1":  (0.01, 0.99),
    "Km0":    (0.001, 0.5),
    "Km1":    (-0.1, 0.1),
    "Km2":    (-0.1, 0.1),
    "alpha":  (0.0, 0.1),
    "A":      (1.0, 1000.0),
    "FILT0":  (0.001, 10.0),
    "NTU0":   (0.001, 10.0),
}

# 损失权重
GREYBOX_LAMBDA = {
    "filter_upper": 0.5,   # λ₁: FILT ≤ RW_NTU
    "nonneg":       0.1,   # λ₂: 浊度非负
    "cstr_upper":   0.5,   # λ₃: NTU ≤ FILT
    "km_pos":       0.01,  # λ₄: K_m保持正值
}

# 优化
GREYBOX_N_RESTARTS = 8     # 多起点次数
GREYBOX_MAX_ITER = 2000    # L-BFGS-B 最大迭代

# 双模态阈值（Q2确定, 2026-07-24）
THETA_COMFORT = 0.15

# ==============================
# 通用：TimeSeriesSplit
# ==============================
N_SPLITS = 5

# ==============================
# 2026 年目标预测日期
# ==============================
TARGET_DATES_2026 = ["2026-02-01", "2026-02-10", "2026-02-20"]

# ==============================
# 输出文件路径
# ==============================
OUT_CLEAN_DATA    = os.path.join(OUTPUT_DIR, "clean_data.csv")
OUT_X_ALL         = os.path.join(OUTPUT_DIR, "X_all.npy")
OUT_Y_ALL         = os.path.join(OUTPUT_DIR, "y_all.npy")
OUT_FEATURE_NAMES = os.path.join(OUTPUT_DIR, "feature_names.npy")
OUT_IMPUTE_REPORT = os.path.join(OUTPUT_DIR, "impute_report.json")

# Q1 灰箱模型输出
OUT_GREYBOX_PARAMS    = os.path.join(OUTPUT_DIR, "q1_greybox_params.json")
OUT_GREYBOX_METRICS   = os.path.join(OUTPUT_DIR, "q1_greybox_metrics.csv")
OUT_GREYBOX_PRED_FILT = os.path.join(OUTPUT_DIR, "q1_greybox_filt_pred.npy")
OUT_GREYBOX_PRED_NTU  = os.path.join(OUTPUT_DIR, "q1_greybox_ntu_pred.npy")

# ==============================
# [LEGACY] 旧Q1/Q2代码向后兼容 — 2026-07-24后仅保留用于旧脚本
# ==============================
XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
    "random_state": 42, "verbosity": 0,
}
LGB_PARAMS = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.01,
    "num_leaves": 31, "min_child_samples": 10,
    "subsample": 0.9, "colsample_bytree": 0.9,
    "reg_alpha": 0.01, "reg_lambda": 0.01,
    "random_state": 42, "verbose": -1,
}
RF_PARAMS = {
    "n_estimators": 100, "max_depth": 8,
    "random_state": 42, "n_jobs": -1,
}
HUBER_DELTA = 1.0
SHAP_THRESHOLD = 0.005
GAM_TOP_K = 5
GAM_N_SPLINES = 10
OUT_LAMBDA_NTU = os.path.join(OUTPUT_DIR, "lambda_ntu.pkl")

print(f"[step0_config] 配置加载完毕。输出目录: {OUTPUT_DIR}")
