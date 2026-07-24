"""
step0_preprocess.py — 数据清洗 + L1-L2特征工程（精简后）
================================================================
2026-07-24 重构：
  - 泵编码从10维0/1展开 → 2维整数count
  - 删除L2+/L3/L4/L5全部特征层
  - 特征矩阵从101维 → ~12维（服务于Q3源A）
  - Q1灰箱模型直接读clean_data.csv原始列，不依赖特征矩阵

输入：12个2025年月度Excel（Format A×8 + Format B×4）
输出：
  - output/clean_data.csv       清洗后的完整数据
  - output/X_all.npy             精简特征矩阵 (numpy)
  - output/y_all.npy             目标向量 (numpy, 原始NTU空间)
  - output/feature_names.npy     特征名列表
  - output/impute_report.json    插补质量报告
"""

import numpy as np
import pandas as pd
import os, re, json, warnings
from step0_config import *

warnings.filterwarnings("ignore")

# ================================================================
#  第一部分：数据加载与列名统一
# ================================================================

def is_format_b(df):
    """判断 DataFrame 是否为 Format B（意大利文表头）"""
    return "Data" in df.columns


def normalize_columns(df):
    """统一列名到 STD_COLS 命名空间"""
    if is_format_b(df):
        df = df.rename(columns=RENAME_MAP_FORMAT_B)
        ghost = [c for c in df.columns if c.startswith("Unnamed")]
        df = df.drop(columns=ghost, errors="ignore")
        for col in COLS_MISSING_IN_FORMAT_B:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = df.rename(columns=RENAME_MAP_FORMAT_A)

    df.columns = [c.strip().replace(".", "_").replace(" ", "_") for c in df.columns]
    keep_cols = [c for c in STD_COLS if c in df.columns]
    df = df[keep_cols].copy()
    return df


def load_2025_all():
    """读取12个月的数据，拼接，标注月份"""
    dfs = []
    for fname in FILES_2025:
        fpath = os.path.join(DATA_DIR_2025, fname)
        if not os.path.exists(fpath):
            print(f"  [WARNING] 文件不存在: {fpath}")
            continue
        df = pd.read_excel(fpath)
        month = MONTH_FROM_FILE.get(fname, 0)
        df = normalize_columns(df)
        df["MONTH"] = month
        dfs.append(df)
        fmt = "B" if month in [5, 6, 7, 8] else "A"
        print(f"  {fname}: {len(df)}行 × {len(df.columns)-1}列 | Format {fmt}")
    data = pd.concat(dfs, ignore_index=True)
    print(f"\n拼接完成: {len(data)}行 × {len(data.columns)}列")
    return data


# ================================================================
#  第二部分：异常数据清洗
# ================================================================

def parse_date(val):
    """DATE列三种编码统一为datetime"""
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
    """解析DATE，修补错位日期，按时间排序"""
    df["DATE"] = df["DATE"].apply(parse_date)
    df["_month_from_date"] = df["DATE"].dt.month
    df["_date_diff"] = df["DATE"].diff().dt.days

    mask_bad = df["_date_diff"] < -100
    df.loc[mask_bad, "DATE"] = pd.NaT

    for _ in range(3):
        na_idx = df[df["DATE"].isna()].index
        for idx in na_idx:
            if idx > 0 and not pd.isna(df.at[idx - 1, "DATE"]):
                df.at[idx, "DATE"] = df.at[idx - 1, "DATE"] + pd.Timedelta(hours=DELTA_T)

    df = df.dropna(subset=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.drop(columns=["_month_from_date", "_date_diff"], errors="ignore")
    return df


def parse_pump_combo(val):
    """泵组合字符串 → 运行台数 (0-5)
       '2,4' → 2台, '3' → 1台, '-' → 0台
       typo修复: '&' '+' '/' → ','  '244' → '2,4'
    """
    if pd.isna(val) or str(val).strip() in ["-", "", "nan"]:
        return 0
    s = str(val).strip()
    s = s.replace("&", ",").replace("+", ",").replace("/", ",")
    s = re.sub(r",,+", ",", s).rstrip(",")
    if s == "244":
        s = "2,4"
    nums = re.findall(r"\d+", s)
    return len(set(nums))  # 去重数量 = 运行台数


def clean_string_anomalies(df):
    """处理数值列中的字符串异常（泵压缩为count）"""
    if "TW_PUMP_DUTY" in df.columns:
        df["tw_pump_count"] = df["TW_PUMP_DUTY"].apply(parse_pump_combo)
        df = df.drop(columns=["TW_PUMP_DUTY"])

    if "RW_PUMP_DUTY" in df.columns:
        df["rw_pump_count"] = df["RW_PUMP_DUTY"].apply(parse_pump_combo)
        df = df.drop(columns=["RW_PUMP_DUTY"])

    if "F_RIDE" in df.columns:
        df["F_RIDE"] = pd.to_numeric(df["F_RIDE"].replace("-", "0"), errors="coerce")

    dash_cols = ["RIVER_LEVEL", "RW_FLOW", "RW_NTU", "RW_CLR", "CL2"]
    for col in dash_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).replace("-", np.nan), errors="coerce")

    if "TW_FLOW" in df.columns:
        df["TW_FLOW"] = df["TW_FLOW"].replace("46/5", "46.5")
        df["TW_FLOW"] = pd.to_numeric(df["TW_FLOW"], errors="coerce")

    return df


def drop_dead_columns(df):
    """删除死列和F_RIDE"""
    for c in DEAD_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    if DROP_F_RIDE and "F_RIDE" in df.columns:
        df = df.drop(columns=["F_RIDE"])
    return df


# ================================================================
#  第三部分：缺失列插补（Format B 的10个缺失列）
# ================================================================

def impute_missing_columns(df):
    """对Format B(5-8月)缺失的列，用LightGBM从共同特征推断"""
    from lightgbm import LGBMRegressor
    from sklearn.metrics import r2_score

    cols_to_impute = ["ALUM", "CL2"]
    discrete_cols = ["RW_PH", "PH"]

    common_features = [
        c for c in ["RIVER_LEVEL", "RW_FLOW", "RW_NTU", "RW_CLR",
                     "FILT_NTU", "CW_WELL_LEVEL", "CLR", "TW_FLOW"]
        if c in df.columns
    ]
    for c in common_features:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    impute_report = {}

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
        X_unknown = X_unknown.fillna(X_known.median())

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
        print(f"  [插补] {target_col}: {len(unknown)}行, 训练R2={impute_report[target_col]['train_R2']}")

    for col in discrete_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            fill = mode_val.iloc[0] if len(mode_val) > 0 else 7.0
            df[col] = df[col].fillna(fill)
            impute_report[col] = {"method": "mode", "fill_value": fill}

    return df, impute_report


# ================================================================
#  第四部分：小量NaN修补
# ================================================================

def fill_small_gaps(df):
    """对少量NaN做前向填充+中位数后备"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        na_count = df[col].isna().sum()
        if na_count == 0:
            continue
        if na_count <= 10:
            df[col] = df[col].fillna(method="ffill")
        elif na_count <= 50:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
    return df


# ================================================================
#  第五部分：目标变量处理（原始NTU空间，无需Box-Cox）
# ================================================================

def handle_target_variable(df):
    """删NTU为NaN的行，保留原始NTU值（灰箱模型在原始空间工作）"""
    before = len(df)
    df = df.dropna(subset=["NTU"])
    print(f"  删除NTU为空的{before - len(df)}行 (目标变量不可插补)")
    print(f"  NTU raw: mean={df['NTU'].mean():.3f}, skew={df['NTU'].skew():.2f}")
    return df


# ================================================================
#  第六部分：精简特征构造（仅L1+L2时间编码，~12维）
# ================================================================

def build_features(df):
    """构造精简特征矩阵（服务于Q3源A，Q1灰箱不依赖此矩阵）"""

    # L1: 保留灰箱模型中出现的变量 + 泵压缩列
    l1_cols = [c for c in FEATURE_COLS_L1 if c in df.columns]
    pump_cols = [c for c in PUMP_COLS if c in df.columns]
    X_l1 = df[l1_cols + pump_cols].copy()

    # 时间编码
    time_vals = pd.to_numeric(df["TIME"], errors="coerce").fillna(0).astype(int)
    hour = time_vals // 100

    time_features = pd.DataFrame(index=df.index)
    time_features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    time_features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    time_features["day_sin"] = np.sin(2 * np.pi * df["DATE"].dt.dayofyear / 365)
    time_features["day_cos"] = np.cos(2 * np.pi * df["DATE"].dt.dayofyear / 365)

    # TW_FLOW前向平移1步（避免同期泄露：不能用t时刻的出厂流量预测t时刻的NTU）
    # 灰箱模型中θ(t) = A*CW_WELL(t-1)/TW_FLOW(t-1)，此处统一做shift-1
    if "TW_FLOW" in df.columns:
        time_features["TW_FLOW_shift1"] = df["TW_FLOW"].shift(1)

    # 合并
    X = pd.concat([X_l1, time_features], axis=1)

    # 删除头部NaN（shift和递推产生的）
    X = X.iloc[2:].reset_index(drop=True)
    y = df["NTU"].iloc[2:].reset_index(drop=True)

    # 剩余NaN填充
    X = X.fillna(X.median())

    print(f"\n  特征矩阵: {X.shape[1]}维 × {X.shape[0]}行")
    print(f"  L1={len(l1_cols)+len(pump_cols)}, 时间编码={len(time_features.columns)}")
    return X, y


# ================================================================
#  第七部分：主函数
# ================================================================

def main():
    print("=" * 60)
    print("  step0_preprocess.py — 数据预处理与特征工程（精简版）")
    print("=" * 60)

    print("\n[1/5] 加载数据...")
    data = load_2025_all()

    print("\n[2/5] 清洗异常数据...")
    data = drop_dead_columns(data)
    data = clean_string_anomalies(data)
    data = fix_dates_and_sort(data)

    print("\n[3/5] 插补缺失列 (Format B)...")
    data, impute_report = impute_missing_columns(data)

    print("\n[4/5] 修补小量NaN + 目标变量...")
    data = fill_small_gaps(data)

    for col in data.select_dtypes(include=["object"]).columns:
        if col not in ["DATE", "TIME", "REMARKS"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = handle_target_variable(data)

    print("\n[5/5] 构造特征...")
    X, y = build_features(data)

    print("\n保存输出...")
    data.to_csv(OUT_CLEAN_DATA, index=False, encoding="utf-8-sig")

    feature_names = X.columns.tolist()
    np.save(OUT_X_ALL, X.values.astype(np.float32))
    np.save(OUT_Y_ALL, y.values.astype(np.float32))
    np.save(OUT_FEATURE_NAMES, np.array(feature_names))

    with open(OUT_IMPUTE_REPORT, "w", encoding="utf-8") as f:
        json.dump(impute_report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  [DONE] 全部输出至 {OUTPUT_DIR}/")
    print(f"    clean_data.csv        → {data.shape[0]}行 × {data.shape[1]}列")
    print(f"    X_all.npy              → {X.shape}")
    print(f"    y_all.npy              → {y.shape}")
    print(f"    feature_names.npy      → {len(feature_names)}个特征")
    print(f"    impute_report.json     → 插补质量报告")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
