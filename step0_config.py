"""
step0_config.py — Q1 全局参数 + 命名约定
===========================================
用途：所有后续 step 共用此文件中的参数。
优先只配置 Q1 所需参数，Q2/Q3/Q4 参数后续按需追加。
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
# 特征工程参数
# ==============================
LAG_STEPS       = [1, 3, 6]          # L3 滞后步数
WINDOW_SIZES    = [3, 6, 12]         # L4 滚动窗口
L5_CANDIDATES   = [                  # L5 交互特征候选
    "PI_load",                       # RW_NTU * RW_FLOW (污染物总通量)
    "GAMMA_alum",                    # ALUM / RW_NTU   (混凝剂充足度)
    "PSI_alum",                      # ALUM * RW_FLOW  (总矾投加速率)
    "OMEGA_night",                   # is_night * ΔRW_NTU_clip+ (夜间突变)
]

# ==============================
# Q1 模型超参
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

# Huber Loss δ
HUBER_DELTA = 1.0

# SHAP 特征保留阈值
SHAP_THRESHOLD = 0.005  # 特征保留阈值（几何平均后值域很小）

# TimeSeriesSplit 折数
N_SPLITS = 5

# ==============================
# GAM 参数（Q1.1 函数关系可视化用）
# ==============================
GAM_TOP_K = 5        # 取 SHAP 排名前 5 的特征做 GAM
GAM_N_SPLINES = 10   # 每个变量的样条基函数数

# ==============================
# 输出文件路径
# ==============================
OUT_CLEAN_DATA    = os.path.join(OUTPUT_DIR, "clean_data.csv")
OUT_FEATURES_ALL  = os.path.join(OUTPUT_DIR, "features_all.csv")
OUT_X_ALL         = os.path.join(OUTPUT_DIR, "X_all.npy")
OUT_Y_ALL         = os.path.join(OUTPUT_DIR, "y_all.npy")
OUT_FEATURE_NAMES = os.path.join(OUTPUT_DIR, "feature_names.npy")
OUT_LAMBDA_NTU    = os.path.join(OUTPUT_DIR, "lambda_ntu.pkl")
OUT_BOXCOX_SCALER = os.path.join(OUTPUT_DIR, "boxcox_scaler.pkl")
OUT_IMPUTE_REPORT = os.path.join(OUTPUT_DIR, "impute_report.json")

# 2026 年目标预测日期
TARGET_DATES_2026 = ["2026-02-01", "2026-02-10", "2026-02-20"]

print(f"[step0_config] 配置加载完毕。输出目录: {OUTPUT_DIR}")
