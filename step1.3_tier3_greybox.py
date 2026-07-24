"""
step1.3_tier3_greybox.py — T3 Greybox (corrected: AR for FILT, CSTR for NTU)
====================================================================
Model:
  FILT(t) = AR(6) on FILT history
  NTU(t)  = β₂·NTU(t-1) + (1-β₂)·FILT(t)  (CSTR)
  β₂      = exp(-2h / θ), θ = A·CW_WELL(t-1)/TW_FLOW(t-1)
  ε_alum  = ALUM - ALUM_policy (feedback residual)
  
τ₁ learned via softmax: RW_NTU_aligned = Σ w_d · RW_NTU(t-d)
"""
import numpy as np, pandas as pd, os, json, warnings
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from q1_data_utils import load_clean_data, add_tier_labels, compute_metrics
from step0_config import *
warnings.filterwarnings("ignore")

def build_operator_features(df_slice):
    X = pd.DataFrame(index=df_slice.index)
    X["RW_NTU"] = df_slice["RW_NTU"]
    X["log_RW_NTU"] = np.log1p(df_slice["RW_NTU"])
    X["FILT_lag1"] = df_slice["FILT_NTU"].shift(1).fillna(df_slice["FILT_NTU"].median())
    X["month_sin"] = df_slice["month_sin"]
    X["month_cos"] = df_slice["month_cos"]
    return X.fillna(X.median())

def compute_tau_weights(s_logits):
    s = np.array(s_logits) - max(s_logits)
    exp_s = np.exp(s)
    return exp_s / exp_s.sum()

def run_experiment(df, fb_type, gamma_type, lambda3):
    mask = df["tier"] == 3
    sub = df[mask].copy()
    n = len(sub)
    if n < 100:
        return None

    filt, ntu = sub["FILT_NTU"].values, sub["NTU"].values
    rw = sub["RW_NTU"].values
    alum = sub["ALUM"].values
    cw = sub["CW_WELL_LEVEL"].values
    tw = sub["TW_FLOW"].values
    ds, dc = sub["day_sin"].values, sub["day_cos"].values

    # Build operator policy
    if fb_type == "linear":
        Xp = build_operator_features(sub)
        m = LinearRegression().fit(Xp, alum)
        alum_policy = m.predict(Xp)
    elif fb_type == "gbdt":
        from lightgbm import LGBMRegressor
        Xp = build_operator_features(sub)
        m = LGBMRegressor(n_estimators=50, max_depth=3, verbose=-1, random_state=42)
        m.fit(Xp, alum)
        alum_policy = m.predict(Xp)
    else:
        alum_policy = alum.copy()
    eps_alum = alum - alum_policy

    # Parameters to optimize
    opt_keys = ["beta2_opt"]  # if we use scalar beta2; if gamma_type has params, add
    opt_init = [0.5]
    opt_bounds = [(0.01, 0.99)]

    # η_filter: sensitivity to feedback → NTU
    if gamma_type == "fixed":
        opt_keys += ["gamma"]
        opt_init += [0.05]
        opt_bounds += [(0.0, 0.5)]
    elif gamma_type == "linear_rw":
        opt_keys += ["gamma_a", "gamma_b"]
        opt_init += [0.001, 0.02]
        opt_bounds += [(-0.01, 0.01), (-0.1, 0.1)]
    elif gamma_type == "sigmoid":
        opt_keys += ["gamma_w", "gamma_b"]
        opt_init += [0.005, 0.0]
        opt_bounds += [(-0.05, 0.05), (-0.5, 0.5)]

    # τ₁ learnable (softmax over lags 0-6)
    n_tau = TIER3_TAU_MAX_LAG
    tau_keys = [f"tau_s{d}" for d in range(n_tau + 1)]
    opt_keys += tau_keys
    opt_init += [0.0] * (n_tau + 1)
    opt_bounds += [(-3.0, 3.0)] * (n_tau + 1)

    # CSTR constant A (if we use A instead of beta2)
    opt_keys += ["A_cstr"]
    opt_init += [50.0]
    opt_bounds += [(1.0, 2000.0)]

    meta = {"fb_type": fb_type, "gamma_type": gamma_type, "lambda3": lambda3, "n_tau": n_tau}

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=min(N_SPLITS, n // 20))
    fold_results = []
    all_pred_ntu = np.zeros(n)

    for tr, va in tscv.split(sub):
        def f_obj(p):
            pd_ct = {k: p[i] for i, k in enumerate(opt_keys)}
            # τ₁ softmax weights
            tau_w = compute_tau_weights([pd_ct[k] for k in tau_keys])

            # NTU: CSTR with TRUE FILT (available at same timestamp)
            n_tr = len(tr)
            pred_n = np.zeros(n_tr)
            pred_n[0] = ntu[tr[0]]
            loss = 0.0
            for t in range(1, n_tr):
                if gamma_type == "fixed":
                    gamma_t = pd_ct["gamma"]
                elif gamma_type == "linear_rw":
                    gamma_t = pd_ct["gamma_a"] * rw[tr[t]] + pd_ct["gamma_b"]
                    gamma_t = np.clip(gamma_t, -0.5, 0.5)
                elif gamma_type == "sigmoid":
                    z = pd_ct["gamma_w"] * rw[tr[t]] + pd_ct["gamma_b"]
                    gamma_t = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10))) - 0.5
                else:
                    gamma_t = 0.0

                theta = pd_ct["A_cstr"] * cw[tr[t-1]] / max(tw[tr[t-1]], 1)
                b2 = np.exp(-2.0 / max(theta, 0.1))
                fb_effect = gamma_t * eps_alum[tr[t]]

                # Use TRUE FILT(t) and TRUE NTU(t-1) (递推 with ground truth available)
                pred_n[t] = b2 * ntu[tr[t-1]] + (1-b2) * filt[tr[t]] + fb_effect

                err = ntu[tr[t]] - pred_n[t]
                loss += np.where(np.abs(err) < 0.5, 0.5 * err**2, 0.5 * np.abs(err) - 0.125)

            loss /= n_tr
            smooth = 0.05 * np.mean(np.abs(np.diff(pred_n)))
            fb_reg = lambda3 * np.mean(np.abs(eps_alum[tr]))
            return loss + smooth + fb_reg

        best_val, best_x = float("inf"), None
        for _ in range(min(GREYBOX_N_RESTARTS, 3)):
            x0 = np.array([np.random.uniform(l, u) for l, u in opt_bounds])
            res = minimize(f_obj, x0, bounds=opt_bounds,
                          method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        if best_x is None:
            continue
        pd_ct = {k: best_x[i] for i, k in enumerate(opt_keys)}
        tau_w = compute_tau_weights([pd_ct[k] for k in tau_keys])

        # Predict on validation set (using TRUE FILT and TRUE prev NTU)
        n_va = len(va)
        pn = np.zeros(n_va)
        pn[0] = ntu[va[0]]
        for t in range(1, n_va):
            if gamma_type == "fixed":
                gamma_t = pd_ct["gamma"]
            elif gamma_type == "linear_rw":
                gamma_t = pd_ct["gamma_a"] * rw[va[t]] + pd_ct["gamma_b"]
                gamma_t = np.clip(gamma_t, -0.5, 0.5)
            else:
                gamma_t = 0.0
            theta = pd_ct["A_cstr"] * cw[va[t-1]] / max(tw[va[t-1]], 1)
            b2 = np.exp(-2.0 / max(theta, 0.1))
            fb_effect = gamma_t * eps_alum[va[t]]
            pn[t] = b2 * ntu[va[t-1]] + (1-b2) * filt[va[t]] + fb_effect

        all_pred_ntu[va] = pn
        m = compute_metrics(ntu[va], pn)
        fold_results.append(m)

    if not fold_results:
        return None
    avg_rmse = np.mean([r["rmse"] for r in fold_results])
    avg_r2 = np.mean([r["r2"] for r in fold_results])
    overall_r2 = r2_score(ntu, all_pred_ntu)

    # Get tau weights from last fold
    tau_w_final = compute_tau_weights([pd_ct.get(k, 0.0) for k in tau_keys]) if 'pd_ct' in dir() else np.ones(n_tau + 1) / (n_tau + 1)
    tau_peak = int(np.argmax(tau_w_final))

    return {
        "fb_type": fb_type, "gamma_type": gamma_type, "lambda3": lambda3,
        "cv_rmse_mean": round(avg_rmse, 4), "cv_r2_mean": round(avg_r2, 4),
        "overall_r2": round(overall_r2, 4), "n_samples": n,
        "tau_weights": [round(float(w), 4) for w in tau_w],
        "tau_peak_lag": tau_peak,
    }

def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("=" * 60)
    print("  step1.3 — T3 Greybox (CSTR for NTU, AR for FILT)")
    print("=" * 60)
    df = load_clean_data()
    df = add_tier_labels(df)
    results = []
    count = 0

    # Phase 1: fb × gamma, lambda3=0.05
    print("\n--- Phase 1: fb × gamma @ lambda3=0.05 ---")
    for fb in TIER3_FB_TYPES:
        for gm in TIER3_GAMMA_TYPES:
            count += 1
            print(f"  [{count}/9] fb={fb} gamma={gm} lam3=0.05")
            r = run_experiment(df, fb, gm, 0.05)
            if r:
                results.append(r)
                print(f"    → NTU R²={r['overall_r2']:.4f} RMSE={r['cv_rmse_mean']:.4f}")

    best_ph1 = max(results, key=lambda x: x.get("overall_r2", -999))
    bf, bg = best_ph1["fb_type"], best_ph1["gamma_type"]
    print(f"\n  Phase 1 best: fb={bf} gamma={bg} R2={best_ph1['overall_r2']:.4f}")

    # Phase 2: lam3 sweep
    print(f"\n--- Phase 2: lam3 sweep (fb={bf}, gamma={bg}) ---")
    for lam in TIER3_LAMBDA3_VALUES:
        if lam == 0.05:
            continue
        r = run_experiment(df, bf, bg, lam)
        if r:
            results.append(r)
            print(f"    lam3={lam:.2f} -> NTU R2={r['overall_r2']:.4f} RMSE={r['cv_rmse_mean']:.4f}")

    pd.DataFrame(results).to_csv(OUT_TIER3_SWEEP, index=False, encoding="utf-8-sig")

    best = max(results, key=lambda x: x.get("overall_r2", -999))
    print(f"\n{'='*60}")
    print(f"  BEST: fb={best['fb_type']} gamma={best['gamma_type']} lam3={best['lambda3']}")
    print(f"  NTU R2={best['overall_r2']:.4f}  RMSE={best['cv_rmse_mean']:.4f}")
    print(f"  tau1 weights: {best['tau_weights']}")
    print(f"  tau1 peak: {best['tau_peak_lag']*2}h")

    json.dump({k: best.get(k) for k in ["fb_type","gamma_type","lambda3","tau_weights",
        "tau_peak_lag","overall_r2","cv_rmse_mean"]}, open(OUT_TIER3_BEST, "w"), indent=2)

    # Summary table
    print(f"\n{'='*100}")
    print(f"  {'fb':<10} {'gamma':<12} {'lam3':<6} {'R2':<10} {'RMSE':<10} {'tau_peak(h)':<10} {'n':<6}")
    print(f"  {'-'*90}")
    for r in sorted(results, key=lambda x: x.get("overall_r2", -999), reverse=True):
        print(f"  {r['fb_type']:<10} {r['gamma_type']:<12} {r['lambda3']:<6.2f} "
              f"{r['overall_r2']:<10.4f} {r['cv_rmse_mean']:<10.4f} "
              f"{r['tau_peak_lag']*2:<10} {r['n_samples']:<6}")
    print(f"\n[DONE] {OUT_TIER3_SWEEP}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
