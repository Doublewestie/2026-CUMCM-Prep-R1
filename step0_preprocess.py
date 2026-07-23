"""
step0_preprocess.py — 数据清洗 + L1-L5 特征工程 + Box-Cox 变换
================================================================
输入：12 个 2025 年月度 Excel（Format A ×8 + Format B ×4）
输出：
  - output/clean_data.csv       清洗后的完整数据
  - output/features_all.csv     L1-L5 全部特征 + NTU 标签
  - output/X_all.npy            特征矩阵 (numpy)
  - output/y_all.npy            目标向量 (numpy, Box-Cox 空间)
  - output/feature_names.npy    特征名列表
  - output/lambda_ntu.pkl       Box-Cox 的 λ
  - output/impute_report.json   插补质量报告
"""

import numpy as np
import pandas as pd
import os, re, json, warnings, joblib
from step0_config import *

# 自定义 Box-Cox 变换（避免 scipy/numpy 版本冲突）
def boxcox_custom(y):
    """Box-Cox: 使正偏分布趋近正态。用最大似然估计 λ"""
    from scipy.optimize import minimize_scalar
    y = np.asarray(y).ravel()
    y = y[y > 0]  # 严格正

    def neg_log_lik(lam):
        if abs(lam) < 1e-8:
            yt = np.log(y)
        else:
            yt = (y**lam - 1) / lam
        n = len(y)
        var = np.var(yt, ddof=1)
        return n / 2 * np.log(var) - (lam - 1) * np.sum(np.log(y))

    res = minimize_scalar(neg_log_lik, bounds=(-2, 2), method="bounded")
    lam = res.x
    if abs(lam) < 1e-8:
        yt = np.log(y)
    else:
        yt = (y**lam - 1) / lam
    return yt, lam

warnings.filterwarnings("ignore")

# ================================================================
#  第一部分：数据加载与列名统一
# ================================================================

def is_format_b(df):
    """判断 DataFrame 是否为 Format B（意大利文表头）"""
    return "Data" in df.columns


def normalize_columns(df, month_label=""):
    """统一列名到 STD_COLS 命名空间"""
    if is_format_b(df):
        # Format B: 意大利文表头
        df = df.rename(columns=RENAME_MAP_FORMAT_B)
        # 删除 Unnamed 幽灵列
        ghost = [c for c in df.columns if c.startswith("Unnamed")]
        df = df.drop(columns=ghost, errors="ignore")
        # 缺失列补 NaN
        for col in COLS_MISSING_IN_FORMAT_B:
            if col not in df.columns:
                df[col] = np.nan
    else:
        # Format A: 英文表头
        df = df.rename(columns=RENAME_MAP_FORMAT_A)

    # 统一列名中的空格和点
    df.columns = [c.strip().replace(".", "_").replace(" ", "_") for c in df.columns]

    # 保留标准列
    keep_cols = [c for c in STD_COLS if c in df.columns]
    df = df[keep_cols].copy()
    return df


def load_2025_all():
    """读取 12 个月的数据，拼接，标注月份"""
    dfs = []
    for fname in FILES_2025:
        fpath = os.path.join(DATA_DIR_2025, fname)
        if not os.path.exists(fpath):
            print(f"  [WARNING] 文件不存在: {fpath}")
            continue
        df = pd.read_excel(fpath)
        month = MONTH_FROM_FILE.get(fname, 0)
        df = normalize_columns(df, month_label=fname)
        df["MONTH"] = month
        dfs.append(df)
        print(f"  {fname}: {len(df)} 行 × {len(df.columns)-1} 列 | Format {'B' if month in [5,6,7,8] else 'A'}")
    data = pd.concat(dfs, ignore_index=True)
    print(f"\n拼接完成: {len(data)} 行 × {len(data.columns)} 列")
    return data


# ================================================================
#  第二部分：异常数据清洗
# ================================================================

def parse_date(val):
    """DATE 列三种编码统一为 datetime"""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (int, float)):
        if 40000 < val < 50000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(val))
    if isinstance(val, str):
        for fmt in ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]:
            try:
                return pd.to_datetime(val, format=fmt)
            except ValueError:
                continue
    return pd.to_datetime(val, errors="coerce")


def fix_dates_and_sort(df):
    """解析 DATE，修补错位日期，按时间排序"""
    df["DATE"] = df["DATE"].apply(parse_date)

    # 找到日期错位的行：日期与 MONTH 不匹配 + 日期间隔回退 > 100 天
    df["_month_from_date"] = df["DATE"].dt.month
    df["_date_diff"] = df["DATE"].diff().dt.days

    # 标记假日期（日期大幅回退，说明是前面残留的假行）
    mask_bad = df["_date_diff"] < -100
    df.loc[mask_bad, "DATE"] = pd.NaT

    # 用前一行 + 2h 递推填充
    for i in range(3):  # 迭代几轮确保连锁填充
        na_idx = df[df["DATE"].isna()].index
        for idx in na_idx:
            if idx > 0 and not pd.isna(df.at[idx-1, "DATE"]):
                df.at[idx, "DATE"] = df.at[idx-1, "DATE"] + pd.Timedelta(hours=DELTA_T)

    df = df.dropna(subset=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # 清理辅助列
    df = df.drop(columns=["_month_from_date", "_date_diff"], errors="ignore")
    return df


def parse_pump_combo(val, prefix="pump"):
    """泵组合字符串 → 5 维 0/1 向量
       '2,4' → {pump1:0, pump2:1, pump3:0, pump4:1, pump5:0}
    """
    pumps = {f"{prefix}{i}": 0 for i in range(1, 6)}
    if pd.isna(val) or str(val).strip() in ["-", "", "nan"]:
        return pumps
    s = str(val).strip()
    # 修正常见 typo
    s = s.replace("&", ",").replace("+", ",").replace("/", ",")
    s = re.sub(r",,+", ",", s).rstrip(",")
    if s == "244":
        s = "2,4"
    nums = re.findall(r"\d+", s)
    for n in nums:
        idx = int(n)
        if 1 <= idx <= 5:
            pumps[f"{prefix}{idx}"] = 1
    return pumps


def clean_string_anomalies(df):
    """处理数值列中的字符串异常"""

    # --- T/W PUMP DUTY → 5 维 0/1 ---
    if "TW_PUMP_DUTY" in df.columns:
        pump_df = df["TW_PUMP_DUTY"].apply(parse_pump_combo, prefix="tw_p").apply(pd.Series)
        df = pd.concat([df, pump_df], axis=1)
        df = df.drop(columns=["TW_PUMP_DUTY"])

    # --- R/W PUMP DUTY → 5 维 0/1 ---
    if "RW_PUMP_DUTY" in df.columns:
        pump_df = df["RW_PUMP_DUTY"].apply(parse_pump_combo, prefix="rw_p").apply(pd.Series)
        df = pd.concat([df, pump_df], axis=1)
        df = df.drop(columns=["RW_PUMP_DUTY"])

    # --- F_RIDE: '-' → 0 (然后整体删列) ---
    if "F_RIDE" in df.columns:
        df["F_RIDE"] = pd.to_numeric(df["F_RIDE"].replace("-", "0"), errors="coerce")

    # --- 其他 '-' → NaN（RIVER_LEVEL, RW_FLOW, RW_NTU, RW_CLR, CL2）---
    dash_cols = ["RIVER_LEVEL", "RW_FLOW", "RW_NTU", "RW_CLR", "CL2"]
    for col in dash_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).replace("-", np.nan), errors="coerce")

    # --- '46/5' 笔误 ---
    if "TW_FLOW" in df.columns:
        df["TW_FLOW"] = df["TW_FLOW"].replace("46/5", "46.5")
        df["TW_FLOW"] = pd.to_numeric(df["TW_FLOW"], errors="coerce")

    return df


def drop_dead_columns(df):
    """删除死列和 F_RIDE"""
    for c in DEAD_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    if DROP_F_RIDE and "F_RIDE" in df.columns:
        df = df.drop(columns=["F_RIDE"])
    return df


# ================================================================
#  第三部分：缺失列插补（Format B 的 10 个缺失列）
# ================================================================

def impute_missing_columns(df):
    """
    对 Format B (5-8月) 缺失的列，用 LightGBM 从共同特征推断。
    为什么用 LightGBM？
      - 能抓非线性关系（ALUM 与原水浊度之间是操作员策略，非线性）
      - 原生处理特征中的 NaN（训练特征中 ALUM 以外的列也有少量 NaN）
      - 训练快，适合这个规模的数据
    """
    from lightgbm import LGBMRegressor
    from sklearn.metrics import r2_score

    # 需要插补的关键列
    cols_to_impute = ["ALUM", "CL2"]
    # RW_PH / PH 是离散值{7.0,7.1,7.3,7.7}，用众数填充
    discrete_cols = ["RW_PH", "PH"]

    # 共同特征列（Format A 和 Format B 都有的数值列，不含 NTU 目标变量）
    common_features = [
        c for c in ["RIVER_LEVEL", "RW_FLOW", "RW_NTU", "RW_CLR",
                     "FILT_NTU", "CW_WELL_LEVEL", "CLR", "TW_FLOW"]
        if c in df.columns
    ]
    # 确保所有共同特征为数值类型
    for c in common_features:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    impute_report = {}

    # 数值列 → LightGBM 插补
    for target_col in cols_to_impute:
        if target_col not in df.columns:
            continue
        known = df[df[target_col].notna()]
        unknown = df[df[target_col].isna()]
        if len(unknown) == 0:
            continue

        X_known = known[common_features].copy()
        for c in common_features:
            X_known[c] = pd.to_numeric(X_known[c], errors="coerce")
        X_known = X_known.fillna(X_known.median())
        y_known = known[target_col]

        X_unknown = unknown[common_features].copy()
        for c in common_features:
            X_unknown[c] = pd.to_numeric(X_unknown[c], errors="coerce")
        X_unknown = X_unknown.fillna(X_known.median())  # 用训练集的中位数

        # 处理 CL2 的特殊情况（可能有很多 NaN 在 Format A 内部）
        if y_known.isna().sum() > 0:
            known_valid = known[known[target_col].notna()]
            X_known = known_valid[common_features].copy()
            for c in common_features:
                X_known[c] = pd.to_numeric(X_known[c], errors="coerce")
            X_known = X_known.fillna(X_known.median())
            y_known = known_valid[target_col]

        if len(y_known) == 0:
            continue

        model = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X_known, y_known)

        pred = model.predict(X_unknown)
        df.loc[df[target_col].isna(), target_col] = pred
        impute_report[target_col] = {
            "method": "LightGBM",
            "imputed_rows": len(unknown),
            "train_R2": round(r2_score(y_known, model.predict(X_known)), 4),
        }
        print(f"  [插补] {target_col}: {len(unknown)} 行, 训练 R2={impute_report[target_col]['train_R2']}")

    # 离散列 → 众数填充
    for col in discrete_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            fill = mode_val.iloc[0] if len(mode_val) > 0 else 7.0
            df[col] = df[col].fillna(fill)
            impute_report[col] = {"method": "mode", "fill_value": fill}

    return df, impute_report


# ================================================================
#  第四部分：小量 NaN 修补
# ================================================================

def fill_small_gaps(df):
    """对少量 NaN 做前向填充 + 中位数后备"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        na_count = df[col].isna().sum()
        if na_count == 0:
            continue
        if na_count <= 10:
            # 极少量：前向填充（时序相关，上一个有效值是最佳估计）
            df[col] = df[col].fillna(method="ffill")
        elif na_count <= 50:
            # 中等量：线性插值
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        else:
            # 大量不处理（应该是 Format B 的插补已完成）
            pass
    return df


# ================================================================
#  第五部分：目标变量处理
# ================================================================

def handle_target_variable(df):
    """处理 NTU 目标变量：删 NTU 为 NaN 的行，Box-Cox 变换"""
    # 删除 NTU 为空的行（目标变量不能插补）
    before = len(df)
    df = df.dropna(subset=["NTU"])
    print(f"  删除 NTU 为空的 {before - len(df)} 行 (目标变量不可插补)")

    # Box-Cox 变换（自动选 λ）
    ntu_values = df["NTU"].values
    ntu_pos = ntu_values + EPS  # Box-Cox 要求严格 >0

    y_transformed, lambda_ntu = boxcox_custom(ntu_pos)

    if lambda_ntu < 0:
        print(f"  [WARNING] lambda={lambda_ntu:.4f}<0, fallback to log1p (more stable)")
        lambda_ntu = 0.0
        y_transformed = np.log1p(ntu_values)
    else:
        print(f"  Box-Cox lambda = {lambda_ntu:.4f}")

    print(f"  NTU raw: mean={ntu_values.mean():.3f}, skew={pd.Series(ntu_values).skew():.2f}")
    print(f"  NTU transformed: mean={y_transformed.mean():.3f}, skew={pd.Series(y_transformed).skew():.2f}")

    df["NTU_transformed"] = y_transformed
    return df, lambda_ntu


# ================================================================
#  第六部分：L1-L5 特征构造
# ================================================================

def build_features(df):
    """构造五级特征金字塔"""

    # ===== L1：原始层（传感器直接读数）=====
    l1_cols = [
        c for c in df.columns
        if c not in ["DATE", "TIME", "NTU", "NTU_transformed", "REMARKS", "MONTH"]
        and df[c].dtype in [np.float64, np.int64, np.int32]
    ]
    X_l1 = df[l1_cols].copy()

    # ===== L2：衍生层（物理公式）=====
    X_l2 = pd.DataFrame(index=df.index)

    # 混凝去除效率
    X_l2["eta_coag"] = (df["RW_NTU"] - df["FILT_NTU"]) / (df["RW_NTU"] + EPS)

    # 单位矾耗
    if "ALUM" in df.columns:
        X_l2["phi_alum"] = df["ALUM"] / (df["RW_NTU"] + EPS)

    # 水力负荷率（流量归一化，用中位数做基准）
    flow_ref = df["RW_FLOW"].median()
    X_l2["psi_hyd"] = df["RW_FLOW"] / (flow_ref + EPS)

    # 时间循环编码
    # TIME 列可能是 str('0700') 或 int(700) 或 float(700.0)
    time_vals = pd.to_numeric(df["TIME"], errors="coerce").fillna(0).astype(int)
    hour = time_vals // 100  # 700 → 7, 2300 → 23
    X_l2["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X_l2["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X_l2["day_sin"] = np.sin(2 * np.pi * df["DATE"].dt.dayofyear / 365)
    X_l2["day_cos"] = np.cos(2 * np.pi * df["DATE"].dt.dayofyear / 365)

    # 周末/夜间标记
    X_l2["is_weekend"] = df["DATE"].dt.weekday.isin([5, 6]).astype(int)
    X_l2["is_night"] = ((hour >= 22) | (hour <= 6)).astype(int)

    # ===== L2+ ：物理幂次项（泰勒展开思想，幂次→线性可加）=====
    # 核心原理：非线性关系 f(x) 在特征空间展开为 x,x²,√x,lnx,1/x
    # 使得 Ridge 回归也能捕捉非线性，XGBoost 获得更丰富的信息

    # --- FILT_NTU 展开（最核心特征，r=0.70）---
    f = df["FILT_NTU"].clip(0.001, 20)
    X_l2["FILT_sq"] = f ** 2          # 二次项：冲击时NTU非线性放大
    X_l2["FILT_sqrt"] = np.sqrt(f)    # 平方根：低值区灵敏度
    X_l2["FILT_cubert"] = f ** (1/3)  # 立方根：进一步压缩尺度

    # --- eta_coag 展开（物理去除效率，r=-0.68）---
    e = X_l2["eta_coag"].clip(0.001, 1.0)
    X_l2["neg_ln_eta"] = -np.log(e)   # -ln(η): 一级反应动力学 dC/dt=-kC → C=C0*e^{-kt}
    X_l2["eta_sq"] = e ** 2           # 效率的二次非线性

    # --- RW_NTU 展开（原水浊度，直接r≈0但通过eta间接起作用）---
    rw = df["RW_NTU"].clip(1, 500)
    X_l2["rw_ntu_sqrt"] = np.sqrt(rw) # 平方根：物理上浓度效应往往与√浓度成正比（扩散定律）
    X_l2["rw_ntu_log"] = np.log1p(rw) # 对数：处理效果是浓度的对数函数

    # --- ALUM 展开（控制变量，r≈0但导数效应存在）---
    if "ALUM" in df.columns:
        al = df["ALUM"].clip(0.01, 1.0)
        X_l2["alum_inv"] = 1.0 / al                 # 倒数：矾不足→断崖恶化
        X_l2["alum_sqrt"] = np.sqrt(al)          # 平方根：边际效应递减

    # --- TW_FLOW 展开（出厂流量，r=0.12）---
    tf = df["TW_FLOW"].clip(1, 100)
    X_l2["tw_flow_log"] = np.log1p(tf)           # 对数：流量对停留时间的影响是指数关系

    # --- 交互项的高阶展开 ---
    # 矾投加量 vs 原水浊度的匹配度（concept: 矾不足时水质急剧恶化）
    if "ALUM" in df.columns:
        dose_ratio = df["ALUM"] / (df["RW_NTU"] + EPS)
        X_l2["dose_ratio_sq"] = dose_ratio ** 2  # 匹配度的非线性
        X_l2["dose_ratio_inv"] = 1.0 / (dose_ratio + EPS)  # 矾严重不足信号

    # ===== L3：滞后层（lag=1/3/6 步）=====
    key_vars = ["RW_NTU", "RW_FLOW", "FILT_NTU", "CW_WELL_LEVEL"]
    if "ALUM" in df.columns:
        key_vars.append("ALUM")

    X_l3 = pd.DataFrame(index=df.index)
    for var in key_vars:
        if var not in df.columns:
            continue
        for lag in LAG_STEPS:
            X_l3[f"{var}_lag{lag}"] = df[var].shift(lag)

    # ===== L4：聚合层（滚动窗口 μ/σ/M）=====
    agg_vars = ["RW_NTU", "RW_FLOW", "FILT_NTU", "CW_WELL_LEVEL"]

    X_l4 = pd.DataFrame(index=df.index)
    for var in agg_vars:
        if var not in df.columns:
            continue
        for w in WINDOW_SIZES:
            X_l4[f"{var}_mean{w}"] = df[var].rolling(w, min_periods=1).mean()
            X_l4[f"{var}_std{w}"] = df[var].rolling(w, min_periods=1).std()
            X_l4[f"{var}_max{w}"] = df[var].rolling(w, min_periods=1).max()
        X_l4[f"{var}_delta"] = df[var].diff()

    # ===== L5：交互层（物理量乘积）=====
    X_l5 = pd.DataFrame(index=df.index)

    # 污染物总通量
    X_l5["PI_load"] = df["RW_NTU"] * df["RW_FLOW"]

    # 混凝剂充足度
    if "ALUM" in df.columns:
        X_l5["GAMMA_alum"] = df["ALUM"] / (df["RW_NTU"] + EPS)
        X_l5["PSI_alum"] = df["ALUM"] * df["RW_FLOW"]

    # 夜间突变检测
    rw_ntu_delta = df["RW_NTU"].diff().clip(lower=0)  # 只取上升
    X_l5["OMEGA_night"] = X_l2["is_night"] * rw_ntu_delta

    # ===== 合并 =====
    X = pd.concat([X_l1, X_l2, X_l3, X_l4, X_l5], axis=1)

    # 删除全 NaN 列和常量列
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    # 扔掉 shift/rolling 产生的头部 NaN 行
    valid_start = max(LAG_STEPS) + max(WINDOW_SIZES)  # 保守取 12 + 6 = 18
    X = X.iloc[valid_start:].reset_index(drop=True)

    y = df["NTU_transformed"].iloc[valid_start:].reset_index(drop=True)

    # 剩余 NaN 用 0 填充（shift 操作的残留，数量极少的尾部）
    X = X.fillna(0)

    print(f"\n  特征矩阵: {X.shape[1]} 维 × {X.shape[0]} 行")
    print(f"  L1={len(l1_cols)}, L2={len(X_l2.columns)}, L3={len(X_l3.columns)}, "
          f"L4={len(X_l4.columns)}, L5={len(X_l5.columns)}")
    return X, y


# ================================================================
#  第七部分：主函数
# ================================================================

def main():
    print("=" * 60)
    print("  step0_preprocess.py — 数据预处理与特征工程")
    print("=" * 60)

    # --- 1. 加载 ---
    print("\n[1/5] 加载数据...")
    data = load_2025_all()

    # --- 2. 清洗 ---
    print("\n[2/5] 清洗异常数据...")
    data = drop_dead_columns(data)              # 删死列 + F_RIDE
    data = clean_string_anomalies(data)         # 字符串异常 + 泵组合解析
    data = fix_dates_and_sort(data)             # DATE 标准化

    # --- 3. 插补 Format B 缺失列 ---
    print("\n[3/5] 插补缺失列 (Format B)...")
    data, impute_report = impute_missing_columns(data)

    # --- 4. 小量 NaN 修补 + 目标变量处理 ---
    print("\n[4/5] 修补小量 NaN + Box-Cox...")
    data = fill_small_gaps(data)

    # 目标变量处理之前，确保数值列全部正确
    for col in data.select_dtypes(include=["object"]).columns:
        if col not in ["DATE", "TIME", "REMARKS"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data, lambda_ntu = handle_target_variable(data)

    # --- 5. 构造特征 ---
    print("\n[5/5] 构造 L1-L5 特征...")
    X, y = build_features(data)

    # --- 保存 ---
    print("\n保存输出...")
    # 清洗后数据 CSV
    data.to_csv(OUT_CLEAN_DATA, index=False, encoding="utf-8-sig")

    # 特征矩阵
    feature_names = X.columns.tolist()
    np.save(OUT_X_ALL, X.values.astype(np.float32))
    np.save(OUT_Y_ALL, y.values.astype(np.float32))
    np.save(OUT_FEATURE_NAMES, np.array(feature_names))

    # Box-Cox λ
    joblib.dump(lambda_ntu, OUT_LAMBDA_NTU)

    # 插补报告
    with open(OUT_IMPUTE_REPORT, "w", encoding="utf-8") as f:
        json.dump(impute_report, f, indent=2, ensure_ascii=False)

    # features_all.csv（特征 + NTU 标签，方便查看）
    df_out = X.copy()
    df_out["NTU_transformed"] = y.values
    df_out.to_csv(OUT_FEATURES_ALL, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  [DONE] 全部输出至 {OUTPUT_DIR}/")
    print(f"    clean_data.csv       → {data.shape[0]}行 × {data.shape[1]}列")
    print(f"    X_all.npy             → {X.shape}")
    print(f"    y_all.npy             → {y.shape}")
    print(f"    lambda_ntu.pkl        → λ = {lambda_ntu:.4f}")
    print(f"    impute_report.json    → 插补质量报告")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
