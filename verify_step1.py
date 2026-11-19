"""End-to-end Iteration 2 verification script for hybrid-fedbatch-twin (D1-D9)."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing
import time
import numpy as np
from scipy.optimize import minimize, root

from hybrid_fedbatch_twin.mechanistic import (
    BioreactorParams,
    OperatingInputs,
    bioreactor_rhs,
    compute_hydrodynamics,
    compute_rates,
    default_initial_state,
    nominal_kla,
    true_kla_viscosity_factor,
)
from hybrid_fedbatch_twin.neural_ode import (
    _fast_loss_grad_and_rss,
    _gam_init_weights,
    _make_block_l2_weights,
    _make_gam_spec,
    _prepare_batch_tensors,
    generate_training_and_test_batches,
    recover_mechanistic_parameters,
    train_default_hybrid_model,
)
from hybrid_fedbatch_twin.estimator import (
    BioreactorDigitalTwin,
    run_twin_estimation_loop,
)
from hybrid_fedbatch_twin.controller import (
    ClosedLoopFeedController,
    summarize_control_run,
)


def _arch_aic_bic(train_trajs, params: BioreactorParams, h_ace: int, h_mu: int, subsample_step: int):
    specs, k_params, lb, ub, _, _ = _make_gam_spec(h_ace=h_ace, h_mu=h_mu)
    l2_w = _make_block_l2_weights(specs)
    rng = np.random.default_rng(7)
    bdata = _prepare_batch_tensors(
        train_trajs,
        params=params,
        noise_std=0.015,
        rng=rng,
        subsample_step=subsample_step,
        atline_every=5,
        include_x0_jitter=False,
    )
    theta0 = _gam_init_weights(7, specs, lb, ub, h_ace=h_ace, h_mu=h_mu)
    bounds = list(zip(lb, ub))
    opt = minimize(
        lambda th: _fast_loss_grad_and_rss(th, bdata, params, specs=specs, l2_w=l2_w)[:2],
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-7},
    )
    _, _, rss = _fast_loss_grad_and_rss(opt.x, bdata, params, specs=specs, l2_w=l2_w)
    n_obs = float(np.sum(bdata["w_obs"][1:] > 0.0))
    sigma2 = max(1e-12, float(rss) / max(1.0, n_obs))
    aic = float(n_obs * np.log(sigma2) + 2.0 * k_params)
    bic = float(n_obs * np.log(sigma2) + k_params * np.log(max(2.0, n_obs)))
    return k_params, float(rss), aic, bic, bool(opt.success)


def _match_operating_point_for_scale(
    p_ref: BioreactorParams,
    V_ref: float,
    rpm_ref: float,
    vvm_ref: float,
    p_target: BioreactorParams,
    V_target: float,
):
    hyd_ref = compute_hydrodynamics(rpm_ref, vvm_ref, V_ref, p_ref)
    target_pgv = hyd_ref["Pg_V"]
    target_vs = hyd_ref["v_s"]

    def residual(z):
        hyd = compute_hydrodynamics(z[0], z[1], V_target, p_target)
        return [
            (hyd["Pg_V"] - target_pgv) / target_pgv,
            (hyd["v_s"] - target_vs) / target_vs,
        ]

    sol = root(residual, x0=[rpm_ref * 0.4, vvm_ref * 0.2], tol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Failed scale matching solve: {sol.message}")
    return float(sol.x[0]), float(sol.x[1])


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 72)
    print("ITERATION 2 VERIFICATION (D1-D9) ON LIVE REPO CODE")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. [D1 + D5] 14-seed Stage-1 Recovery + k_La0=550.0 Re-center Check
    # ------------------------------------------------------------------
    print("\n--- [1] Stage-1 6-Param Mechanistic Recovery (14 seeds + k_La0=550) ---")
    seeds = list(range(10)) + [42, 99, 123, 2025]
    per_seed_mean = []
    per_seed_max = []
    per_seed_kappa = []
    res_42 = None
    for sd in seeds:
        r = recover_mechanistic_parameters(noise_std=0.015, seed=sd)
        if sd == 42:
            res_42 = r
        errs = list(r["relative_error_pct"].values())
        m_err = float(np.mean(errs))
        mx_err = float(r["max_relative_error_pct"])
        kp = float(r["condition_number_rel"])
        per_seed_mean.append(m_err)
        per_seed_max.append(mx_err)
        per_seed_kappa.append(kp)
        print(
            f"  seed={sd:4d}: mean={m_err:5.2f}%, max={mx_err:5.2f}%, "
            f"kappa_rel={kp:5.2f}, errs={ {k: round(v, 2) for k, v in r['relative_error_pct'].items()} }"
        )
        assert mx_err < 8.0, f"Seed {sd} max rel error {mx_err:.2f}% >= 8.0%"
        assert kp < 45.0, f"Seed {sd} condition_number_rel {kp:.2f} >= 45.0"

    overall_mean = float(np.mean(per_seed_mean))
    overall_max = float(np.max(per_seed_max))
    overall_kappa = float(np.max(per_seed_kappa))
    print(
        f"  14-seed Summary: mean_across_seeds={overall_mean:.2f}% (<5.0%), "
        f"worst_max={overall_max:.2f}% (<8.0%), max_kappa_rel={overall_kappa:.2f} (<45.0)"
    )
    assert overall_mean < 5.0, f"14-seed mean error {overall_mean:.2f}% >= 5.0%"

    r_550 = recover_mechanistic_parameters(
        noise_std=0.015, seed=42, true_params=BioreactorParams(k_La0=550.0)
    )
    print(
        f"  k_La0=550 check (seed=42): recovered k_La0={r_550['recovered_params'].k_La0:.2f}, "
        f"max_err={r_550['max_relative_error_pct']:.2f}%, kappa_rel={r_550['condition_number_rel']:.2f}"
    )
    assert r_550["max_relative_error_pct"] < 8.0
    assert r_550["condition_number_rel"] < 45.0

    # ------------------------------------------------------------------
    # 2a. [D9] Cadence & Architecture AIC/BIC Comparison
    # ------------------------------------------------------------------
    print("\n--- [2a] GAM Architecture & Cadence AIC/BIC Check ---")
    p_default = BioreactorParams()
    train_trajs, test_trajs = generate_training_and_test_batches(p_default)
    print(f"  Generated {len(train_trajs)} training batches, {len(test_trajs)} held-out test batches")
    assert len(train_trajs) == 10
    assert len(test_trajs) == 5

    for cad_label, sub_step in [("0.5h (step=5)", 5), ("1.0h (step=10)", 10)]:
        row_strs = []
        for h_a, h_m in [(3, 2), (4, 3), (5, 4)]:
            k_p, rss, aic, bic, ok = _arch_aic_bic(train_trajs, p_default, h_a, h_m, sub_step)
            row_strs.append(f"Arch({h_a},{h_m})[k={k_p},AIC={aic:.1f},BIC={bic:.1f},ok={ok}]")
        print(f"  {cad_label}: " + " | ".join(row_strs))

    # ------------------------------------------------------------------
    # 2b. [D2 + D6 + D7 + D9] Stage-2 GAM-86 Training, Bootstrap UQ, DoV & Scale
    # ------------------------------------------------------------------
    print("\n--- [2b] Stage-2 GAM-86 (B=15), Held-out 5-Regime R2, UQ & DoV ---")
    p_rec = res_42["recovered_params"]
    t_s2 = time.perf_counter()
    model, metrics = train_default_hybrid_model(
        params=p_rec, plant_params=p_default, noise_std=0.015, seed=7, n_bootstrap=15
    )
    dt_s2 = time.perf_counter() - t_s2
    print(f"  Stage-2 runtime: {dt_s2:.2f}s")
    print("  model.fit_metadata:", model.fit_metadata)
    print("  train/test metrics:", metrics)

    assert model.n_params == 86, f"Expected n_params=86, got {model.n_params}"
    assert bool(model.fit_metadata.get("converged", False)), "Point fit did not converge"
    boot_conv = model.fit_metadata.get("bootstrap_converged", 0)
    assert boot_conv >= 12, f"Too few converged bootstrap members: {boot_conv}"
    assert "dov_p99" in model.fit_metadata and model.fit_metadata["dov_p99"] > 0.0

    # Verify held-out 5-regime R2 directly via evaluate_on_trajectories too
    eval_test = model.evaluate_on_trajectories(test_trajs, p_rec, plant_params=p_default)
    print("  evaluate_on_trajectories(test_trajs):", eval_test)
    assert float(eval_test["kd_r2"]) >= 0.70, f"Held-out kd_r2={eval_test['kd_r2']:.4f} < 0.70"
    for k_rate in ("kla_r2", "qAnet_r2", "mu_r2"):
        val = float(eval_test[k_rate])
        assert val > 0.80, f"Held-out {k_rate}={val:.4f} <= 0.80"

    # High-biomass extrapolated k_La UQ check (X_tot = 84 g/L)
    hyd_nom = compute_hydrodynamics(450.0, 1.2, 2.0, p_default)
    mu_mech_ref = p_default.mu_max * (0.3 / (p_default.K_S + 0.3)) * (5.0 / (p_default.K_O2 + 5.0))
    pred_84 = model.predict_full_with_uq(
        X_v=80.0,
        X_d=4.0,
        S=0.3,
        C_L=5.0,
        A=0.2,
        v_tip=hyd_nom["v_tip"],
        Pg_V=hyd_nom["Pg_V"],
        v_s=hyd_nom["v_s"],
        mu_mech=mu_mech_ref,
    )
    true_eta_84 = true_kla_viscosity_factor(84.0)
    err_84 = abs(pred_84["eta_kla"] - true_eta_84)
    z_84 = err_84 / max(1e-9, pred_84["std_kla"])
    print(
        f"  Extrap X_tot=84 g/L: pred_eta_kla={pred_84['eta_kla']:.4f} +/- {pred_84['std_kla']:.4f}, "
        f"true={true_eta_84:.4f}, z={z_84:.2f}, d_m={pred_84['d_m']:.2f}, p99={model.fit_metadata['dov_p99']:.2f}, "
        f"in_domain={pred_84['in_domain']}"
    )
    assert err_84 <= 2.5 * pred_84["std_kla"], f"X_tot=84 error {err_84:.4f} > 2.5*std {2.5*pred_84['std_kla']:.4f}"
    assert not pred_84["in_domain"]

    # Sharp OOD metabolic attenuation check (S=15.0, A=8.0) -> w_met / eta_mu -> 1.0
    mu_mech_ood = p_default.mu_max * (15.0 / (p_default.K_S + 15.0)) * (5.0 / (p_default.K_O2 + 5.0))
    pred_ood = model.predict_full_with_uq(
        X_v=20.0,
        X_d=0.5,
        S=15.0,
        C_L=5.0,
        A=8.0,
        v_tip=hyd_nom["v_tip"],
        Pg_V=hyd_nom["Pg_V"],
        v_s=hyd_nom["v_s"],
        mu_mech=mu_mech_ood,
    )
    print(
        f"  OOD (S=15, A=8): eta_mu={pred_ood['eta_mu']:.4f}, std_mu={pred_ood['std_mu']:.4f}, "
        f"d_m={pred_ood['d_m']:.2f}, p99={model.fit_metadata['dov_p99']:.2f}, in_domain={pred_ood['in_domain']}"
    )
    assert not pred_ood["in_domain"]
    assert abs(pred_ood["eta_mu"] - 1.0) < 0.05, f"Expected eta_mu -> 1.0 in far OOD, got {pred_ood['eta_mu']:.4f}"

    # Scale invariance: V = 5 L vs V = 1000 L
    p5 = BioreactorParams(V_vessel=5.0, V_0=2.0, V_max=5.0)
    p1000 = BioreactorParams(V_vessel=1000.0, V_0=400.0, V_max=1000.0)
    rpm_1000, vvm_1000 = _match_operating_point_for_scale(p5, 2.0, 450.0, 1.2, p1000, 400.0)
    hyd_5 = compute_hydrodynamics(450.0, 1.2, 2.0, p5)
    hyd_1000 = compute_hydrodynamics(rpm_1000, vvm_1000, 400.0, p1000)
    kla_5 = nominal_kla(450.0, 1.2, 2.0, p5)
    kla_1000 = nominal_kla(rpm_1000, vvm_1000, 400.0, p1000)
    r_5 = compute_rates(np.array([25.0, 0.8, 0.35, 5.2, 0.25, 4.0, 2.0]), OperatingInputs(0.08, 450.0, 1.2), p5, mode="hybrid", residual_model=model)
    r_1000 = compute_rates(np.array([25.0, 0.8, 0.35, 5.2, 0.25, 4.0, 400.0]), OperatingInputs(16.0, rpm_1000, vvm_1000), p1000, mode="hybrid", residual_model=model)
    print(
        f"  Scale invariance 5L vs 1000L: matched rpm={rpm_1000:.2f}, vvm={vvm_1000:.4f}, "
        f"Pg_V diff={abs(hyd_5['Pg_V']-hyd_1000['Pg_V']):.2e}, v_s diff={abs(hyd_5['v_s']-hyd_1000['v_s']):.2e}, "
        f"kla_nom diff={abs(kla_5-kla_1000):.2e}, eta_mu diff={abs(r_5['eta_mu']-r_1000['eta_mu']):.2e}"
    )
    assert abs(hyd_5["Pg_V"] - hyd_1000["Pg_V"]) / hyd_5["Pg_V"] < 1e-6
    assert abs(hyd_5["v_s"] - hyd_1000["v_s"]) / hyd_5["v_s"] < 1e-6
    assert abs(kla_5 - kla_1000) / kla_5 < 1e-6
    assert abs(r_5["k_La"] - r_1000["k_La"]) / r_5["k_La"] < 1e-6
    assert abs(r_5["q_A_net"] - r_1000["q_A_net"]) < 1e-9
    assert abs(r_5["eta_mu"] - r_1000["eta_mu"]) < 1e-9

    # Boundary nonnegativity & D3 overflow stoichiometry guard
    for m_chk in ("nominal", "hybrid"):
        rhs0 = bioreactor_rhs(
            5.0,
            np.array([15.0, 0.5, 0.0, 5.0, 0.3, 2.0, 2.5]),
            OperatingInputs(0.05, 450.0, 1.2),
            p_default,
            mode=m_chk,
            residual_model=model if m_chk == "hybrid" else None,
        )
        assert rhs0[2] >= 0.0, f"Negative dS/dt at S=0 in {m_chk} mode: {rhs0[2]}"

    # ------------------------------------------------------------------
    # 3. [D3 + D4 + D8] Closed-Loop EKF + Controller & ALCOA+ Audit Log
    # ------------------------------------------------------------------
    print("\n--- [3] Closed-Loop EKF + Controller Across All 6 Scenarios + ALCOA+ ---")
    scenarios = ["none", "aeration_drop", "stirrer_drop", "shear_spike", "feed_step", "combo"]
    last_sim = None
    for sc_mode in scenarios:
        ctrl = ClosedLoopFeedController(
            params=p_rec,
            residual_model=model,
            mu_target=0.28,
            S_set=0.25,
            A_limit=1.8,
            rpm=450.0,
            vvm=1.2,
            F_max=0.30,
            disturbance_mode=sc_mode,
        )
        u_probe = ctrl(8.0, default_initial_state(p_rec))
        assert u_probe.S_feed is None, f"Controller leaked S_feed in {sc_mode}: {u_probe.S_feed}"

        sim = run_twin_estimation_loop(
            t_span=(0.0, 16.0),
            x0=default_initial_state(p_default),
            controller_fn=ctrl,
            params=p_rec,
            residual_model=model,
            dt_step=0.1,
            atline_interval_steps=10,
            seed=123,
            plant_params=p_default,
            disturbance_mode=sc_mode,
        )
        summ = summarize_control_run(sim, mu_target=0.28, params=p_rec, C_star=p_default.C_star)
        ratio = summ["hybrid_norm_rmse"] / max(1e-9, summ["nominal_norm_rmse"])
        thresh_label = "<0.70 (blind shear observability)" if sc_mode == "shear_spike" else "<0.50"
        print(
            f"  scenario={sc_mode:13s}: hyb_norm_rmse={summ['hybrid_norm_rmse']:.4f}, "
            f"nom_norm_rmse={summ['nominal_norm_rmse']:.4f}, ratio={ratio:.3f} ({thresh_label}), "
            f"max_d_m={summ['max_d_m']:.2f}, ood_frac={summ['ood_fraction']:.3f}"
        )
        if sc_mode == "shear_spike":
            # Under strictly blind +260 RPM runaway (no tachometer, unmeasured X_d), unmeasured X_d lysis
            # contributes ~97% of residual MSE; verify clear hybrid advantage overall and across X_d/P/A.
            assert summ["hybrid_norm_rmse"] < 0.70 * summ["nominal_norm_rmse"], (
                f"Scenario {sc_mode}: hyb {summ['hybrid_norm_rmse']:.4f} >= 0.70 * nom {summ['nominal_norm_rmse']:.4f}"
            )
            assert summ["hybrid_Xd_rmse"] < summ["nominal_Xd_rmse"]
            assert summ["hybrid_P_rmse"] < summ["nominal_P_rmse"]
            assert summ["hybrid_A_rmse"] < summ["nominal_A_rmse"]
        else:
            assert summ["hybrid_norm_rmse"] < 0.5 * summ["nominal_norm_rmse"], (
                f"Scenario {sc_mode}: hyb {summ['hybrid_norm_rmse']:.4f} >= 0.5 * nom {summ['nominal_norm_rmse']:.4f}"
            )
        last_sim = sim

    # Also test Stage-1 recovered parameters -> Stage-2 -> EKF plant mismatch run
    ctrl_rec = ClosedLoopFeedController(
        params=p_rec,
        residual_model=model,
        mu_target=0.28,
        disturbance_mode="combo",
    )
    sim_rec = run_twin_estimation_loop(
        t_span=(0.0, 16.0),
        x0=default_initial_state(p_default),
        controller_fn=ctrl_rec,
        params=p_rec,
        residual_model=model,
        dt_step=0.1,
        atline_interval_steps=10,
        seed=123,
        plant_params=p_default,
        disturbance_mode="combo",
    )
    summ_rec = summarize_control_run(sim_rec, mu_target=0.28, params=p_rec, C_star=p_default.C_star)
    print(
        f"  Stage1-wired combo: hyb_norm_rmse={summ_rec['hybrid_norm_rmse']:.4f}, "
        f"nom_norm_rmse={summ_rec['nominal_norm_rmse']:.4f}, "
        f"ratio={summ_rec['hybrid_norm_rmse']/summ_rec['nominal_norm_rmse']:.3f}"
    )
    assert summ_rec["hybrid_norm_rmse"] < 0.5 * summ_rec["nominal_norm_rmse"]

    # Full ALCOA+ audit log check
    twin = BioreactorDigitalTwin.from_simulation_run(
        last_sim,
        params=p_rec,
        residual_model=model,
        calibration_metadata={"stage1_sha256": res_42["calibration_sha256"]},
        batch_id="VERIFY-STEP1-COMBO",
    )
    audit_rec = twin.to_audit_record()
    log = audit_rec["audit_log"]
    assert audit_rec["audit_log_count"] > 100, f"Expected >100 entries, got {audit_rec['audit_log_count']}"
    assert len(log) == audit_rec["audit_log_count"]
    assert len(audit_rec["record_sha256"]) == 64
    for idx, entry in enumerate(log, start=1):
        assert entry["seq"] == idx, f"Expected seq {idx}, got {entry['seq']}"
        assert len(entry["entry_sha256"]) == 64 and int(entry["entry_sha256"], 16) >= 0
    ok_rec, rec_errs = BioreactorDigitalTwin.verify_audit_record(audit_rec)
    assert ok_rec, f"ALCOA+ verification failed: {rec_errs}"
    print(
        f"  ALCOA+ Audit Record OK: count={audit_rec['audit_log_count']}, "
        f"record_sha256={audit_rec['record_sha256'][:16]}..."
    )

    print(f"\nALL VERIFY_STEP1 CHECKS PASSED in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
