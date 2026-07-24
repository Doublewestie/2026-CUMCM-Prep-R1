"""Shared data utilities for Q1 three-tier scheme"""
import numpy as np, pandas as pd, os, json, joblib
from scipy.optimize import minimize
from step0_config import *

def load_clean_data():
    df = pd.read_csv(OUT_CLEAN_DATA)
    df["DATE"] = pd.to_datetime(df["DATE"])
    time_vals = pd.to_numeric(df["TIME"], errors="coerce").fillna(0).astype(int)
    df["hour"] = time_vals // 100
    df["day_sin"] = np.sin(2 * np.pi * df["DATE"].dt.dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["DATE"].dt.dayofyear / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    return df

def add_tier_labels(df):
    filt = df["FILT_NTU"].values
    labels = np.zeros(len(df), dtype=int)
    labels[filt <= TIER_THRESHOLDS[0]] = 1
    labels[(filt > TIER_THRESHOLDS[0]) & (filt <= TIER_THRESHOLDS[1])] = 2
    labels[filt > TIER_THRESHOLDS[1]] = 3
    df["tier"] = labels
    return df

def boxcox_custom(y):
    from scipy.optimize import minimize_scalar
    y = np.asarray(y).ravel()
    y = y[y > 0]
    def neg_log_lik(lam):
        if abs(lam) < 1e-8:
            yt = np.log(y)
        else:
            yt = (y**lam - 1) / lam
        n = len(y)
        var = np.var(yt, ddof=1)
        return n / 2 * np.log(var) - (lam - 1) * np.sum(np.log(y))
    res = minimize_scalar(neg_log_lik, bounds=(-2, 2), method="bounded")
    return res.x

def boxcox_transform(y, lam):
    y = np.asarray(y, dtype=np.float64)
    if abs(lam) < 1e-8:
        return np.log(y + EPS)
    return ((y + EPS) ** lam - 1) / lam

def boxcox_inverse(y, lam):
    if abs(lam) < 1e-8:
        return np.exp(y) - EPS
    return (y * lam + 1) ** (1.0 / lam) - EPS

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": round(rmse, 4), "r2": round(r2, 4), "mae": round(mae, 4)}

def seg1_greybox(filt, rw_ntu, alum, clr, day_sin, day_cos, params, tau=0):
    beta1 = params["beta1"]
    Km0, Km1, Km2 = params["Km0"], params["Km1"], params["Km2"]
    alpha = params["alpha"]
    if tau > 0:
        rw_ntu_aligned = np.concatenate([np.full(tau, rw_ntu[0]), rw_ntu[:-tau]])
    else:
        rw_ntu_aligned = rw_ntu
    pred = np.zeros_like(filt)
    pred[0] = params.get("FILT0", 0.2)
    for t in range(1, len(filt)):
        Km_t = max(Km0 + Km1 * day_sin[t] + Km2 * day_cos[t], 0.001)
        eta_t = alum[t] / (alum[t] + Km_t + max(alpha * clr[t], 0) + 1e-8)
        eta_t = np.clip(eta_t, 0.8, 1.0)
        pred[t] = beta1 * pred[t-1] + (1 - beta1) * rw_ntu_aligned[t] * (1 - eta_t)
    return pred

def seg1_loss(params, filt, rw_ntu, alum, clr, day_sin, day_cos):
    pred = seg1_greybox(filt, rw_ntu, alum, clr, day_sin, day_cos, params)
    errors = filt - pred
    huber = np.where(np.abs(errors) < 1.0, 0.5 * errors**2, np.abs(errors) - 0.5)
    smooth = 0.1 * np.mean(np.abs(np.diff(pred)))
    upper = 0.5 * np.mean(np.maximum(0, pred - rw_ntu))
    return np.mean(huber) + smooth + upper

def fit_seg1(filt, rw_ntu, alum, clr, day_sin, day_cos):
    init = [GREYBOX_PARAM_INIT[k] for k in ["beta1","Km0","Km1","Km2","alpha"]]
    bounds = [GREYBOX_PARAM_BOUNDS[k] for k in ["beta1","Km0","Km1","Km2","alpha"]]
    def f(p):
        p_dict = {k: p[i] for i, k in enumerate(["beta1","Km0","Km1","Km2","alpha"])}
        p_dict["FILT0"] = 0.2
        return seg1_loss(p_dict, filt, rw_ntu, alum, clr, day_sin, day_cos)
    best_res = None
    best_val = float("inf")
    for _ in range(GREYBOX_N_RESTARTS):
        x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        res = minimize(f, x0, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
    p_dict = {k: best_res.x[i] for i, k in enumerate(["beta1","Km0","Km1","Km2","alpha"])}
    p_dict["FILT0"] = 0.2
    return p_dict, best_res.fun
