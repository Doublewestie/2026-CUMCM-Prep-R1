"""
tests/test_core_guards.py — 论文核心数字守卫（math-consistency Step3 附加任务）
===============================================================================
固化论文必引的核心数字断言，防止产物重跑/口径变更后素材失配。
断言与产物 JSON 逐项核对（审计: docs/materials/05_审查与总账/audit_1.md, 2026-07-27）。

运行:  python tests/test_core_guards.py   （或 pytest tests/test_core_guards.py）
全部断言通过 → 输出 "ALL CORE GUARDS PASSED (N/N)"。

守卫清单（≤10 断言组）:
  G1 Q1 诚实 TS-CV (RL_med=6.09/Q_med=44): r2=0.7369 ± tol, in-sample 0.8023
  G2 Q3 部署口径: forecast 0.4853 / oracle 0.6165 / FILT 代价 0.1312
  G3 Q2 log-AR(6): cv r2=0.6955 (rmse 0.2412)
  G4 Q4 前瞻: 捕获率 0.5799 / 虚警 0.0121 / Kappa舒适区 0.8774
  G5 T3 特征重要性: η_coag robust=0.335 (#1)
  G6 T1 经验采样: JS=0.0499 < 高斯 0.6379; T2 对数压缩 RMSE 0.0289 < 经验 0.0363
  G7 τ 参数表: RW_NTU=4h, ALUM=6h, RW_FLOW=2h, RW_PH=2h (产物为准)
  G8 NN-β FAIL: 0.5884 < 手调 tier-A 0.6889
  G9 NN-路由 FAIL: 0.0905 < if-else 0.6165
  G10 TimesFM 零样本失败基线: feb 均值 ≈0.0951/0.0946/0.0925
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 0.0005  # 容忍度: 相对差<=0.1% 且绝对差<=报告精度(4位)


def _load(rel):
    with open(os.path.join(BASE, rel), encoding="utf-8") as f:
        return json.load(f)


def _check(name, got, expected, tol=TOL):
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got} expected={expected}")
    return ok


def test_g1_q1_tscv():
    """G1: Q1 诚实 TS-CV (6.09/44 口径) — 论文 Q1 主验证数字"""
    d = _load("results/q1_tscv_validation_rl6.09_q44.json")
    cv = d["cv_mean"]
    return (
        _check("Q1 TS-CV r2", cv["r2_cstr"], 0.7369)
        and _check("Q1 TS-CV std", cv["r2_cstr_std"], 0.1009)
        and _check("Q1 in-sample r2", d["in_sample_reference"]["r2_cstr"], 0.8023)
    )


def test_g2_q3_deploy():
    """G2: Q3 部署口径 vs oracle — 论文 Q3 主口径"""
    cv = _load("results/q3_forecast_cv_results.json")["cv_mean"]
    return (
        _check("Q3 forecast r2", cv["r2_forecast"], 0.4853)
        and _check("Q3 oracle r2", cv["r2_oracle"], 0.6165)
        and _check("Q3 FILT penalty", cv["delta_filt_forecast_penalty"], 0.1312)
    )


def test_g3_q2_logar():
    """G3: Q2 log-AR(6)+RidgeCV — 论文 Q2 主数字"""
    d = _load("output/step2_final_results.json")
    return (
        _check("Q2 cv r2", d["cv_mean"]["r2"], 0.6955)
        and _check("Q2 cv rmse", d["cv_mean"]["rmse"], 0.2412)
        and _check("Q2 in-sample r2", d["in_sample"]["r2"], 0.7903)
    )


def test_g4_q4_prospective():
    """G4: Q4 前瞻口径 — 论文 Q4 主口径"""
    eb = _load("output/q4_event_backtest.json")
    kap = _load("output/q4_kappa_report.json")
    return (
        _check("Q4 capture (prosp)", eb["exceed_capture_rate"], 0.5799)
        and _check("Q4 false alarm (prosp)", eb["false_alarm_rate"], 0.0121)
        and _check("Q4 kappa comfort", kap["kappa_comfort"], 0.8774)
    )


def test_g5_t3_importance():
    """G5: T3 特征重要性 η_coag #1"""
    import csv
    with open(os.path.join(BASE, "output", "tier3_factor_importance.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    top = rows[0]
    return (
        _check(f"T3 #{top['rank']} feature = {top['feature']}", float(top["robust"]), 0.335)
        and top["feature"] == "eta_coag"
    )


def test_g6_tiers():
    """G6: T1 经验采样 JS 与 T2 对数压缩 RMSE"""
    t1 = _load("output/tier1_report.json")
    t2 = _load("output/tier2_comparison.json")
    return (
        _check("T1 JS empirical", t1["validation_js_divergence"], 0.0499)
        and _check("T1 JS gauss", t1["gauss_js_divergence"], 0.6379)
        and _check("T2 log rmse", t2["path_b_log"]["filt_rmse"], 0.0289)
        and _check("T2 emp rmse", t2["path_a"]["filt_rmse"], 0.0363)
    )


def test_g7_tau():
    """G7: τ 参数表（产物为准 — ALUM=6h 非 4h）"""
    tau = _load("output/step2_final_results.json")["tau_params"]
    return (
        _check("tau RW_NTU", tau["RW_NTU_to_FILT_hours"], 4)
        and _check("tau ALUM (6h, NOT 4h)", tau["ALUM_to_FILT_hours"], 6)
        and _check("tau RW_FLOW", tau["RW_FLOW_to_FILT_hours"], 2)
        and _check("tau RW_PH", tau["RW_PH_to_FILT_hours"], 2)
    )


def test_g8_nn_beta_fail():
    """G8: NN 可学习 β/θ FAIL（闭环掩蔽补充证据）"""
    d = _load("results/step1.10_verification.json")
    ok = d["verdict"] == "FAIL" and d["best_nn_r2"] < d["baseline_tier"]["r2"]
    print(f"  [{'PASS' if ok else 'FAIL'}] NN-beta: best={d['best_nn_r2']} < tier={d['baseline_tier']['r2']}")
    return ok and _check("NN-beta best", d["best_nn_r2"], 0.5884)


def test_g9_nn_routing_fail():
    """G9: NN 路由 FAIL"""
    d = _load("results/step3.9_routing_verification.json")["cv_results"]
    ok = d["routing_ifelse"]["r2"] > d.get("nn_2way", {}).get("r2", -1)
    nn2 = d.get("nn_2way", {}).get("r2", None)
    if nn2 is None:
        print("  [FAIL] NN-2way missing in routing verification json")
        return False
    return ok and _check("NN-2way r2", nn2, 0.0905)


def test_g10_timesfm():
    """G10: TimesFM 零样本失败基线"""
    d = _load("results/timesfm_summary.json")
    return (
        _check("TimesFM feb1", d["feb1_mean"], 0.0951)
        and _check("TimesFM feb10", d["feb10_mean"], 0.0946)
        and _check("TimesFM feb20", d["feb20_mean"], 0.0925)
    )


ALL = [
    test_g1_q1_tscv,
    test_g2_q3_deploy,
    test_g3_q2_logar,
    test_g4_q4_prospective,
    test_g5_t3_importance,
    test_g6_tiers,
    test_g7_tau,
    test_g8_nn_beta_fail,
    test_g9_nn_routing_fail,
    test_g10_timesfm,
]


def main():
    print("=" * 60)
    print("  Core Guards: paper-critical numbers vs artifacts")
    print("=" * 60)
    passed = 0
    for t in ALL:
        try:
            if t():
                passed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {e}")
    total = len(ALL)
    print("=" * 60)
    if passed == total:
        print(f"  ALL CORE GUARDS PASSED ({passed}/{total})")
    else:
        print(f"  GUARD FAILURES: {total - passed} of {total}")
        sys.exit(1)


if __name__ == "__main__":
    main()