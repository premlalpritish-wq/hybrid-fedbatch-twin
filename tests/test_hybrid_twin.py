"""Comprehensive unit, physical-invariance, and anti-reward-hacking tests."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

import app
import stress_test_uhlendorff as stu
from hybrid_fedbatch_twin import (
    BioreactorDigitalTwin,
    BioreactorParams,
    ClosedLoopFeedController,
    HybridExtendedKalmanFilter,
    ManualFeedController,
    NominalMechanisticController,
    OpenLoopExponentialController,
    OperatingInputs,
    Q_DEFAULT,
    benchmark_controllers,
    bioreactor_rhs,
    compute_hydrodynamics,
    compute_rates,
    default_initial_state,
    default_operating_inputs,
    disturbance_profile,
    recover_mechanistic_parameters,
    run_twin_estimation_loop,
    simulate_bioreactor,
    summarize_control_run,
    train_default_hybrid_model,
    verify_audit_chain,
)
from hybrid_fedbatch_twin.estimator import GENESIS_ENTRY_SHA256
from hybrid_fedbatch_twin.mechanistic import true_kla_viscosity_factor
from hybrid_fedbatch_twin.neural_ode import (
    PhysicsInformedNeuralODE,
    _gam_eval_heads,
    _gam_init_weights,
    _prepare_batch_tensors,
    _trajectory_loss_and_grad,
    generate_training_and_test_batches,
)


@pytest.fixture(scope="module")
def calibrated_twin_assets():
    """Calibrate Stage-1 parameters and train Stage-2 GAM-86 (B=15) wired sequentially."""
    true_params = BioreactorParams()
    rec = recover_mechanistic_parameters(noise_std=0.015, seed=42, true_params=true_params)
    p_rec = rec["recovered_params"]
    model, metrics = train_default_hybrid_model(
        params=p_rec,
        plant_params=true_params,
        noise_std=0.015,
        seed=7,
        n_bootstrap=15,
    )
    return true_params, rec, p_rec, model, metrics


@pytest.mark.unit
def test_boundary_nonnegativity_and_overflow_stoichiometry():
    """Verify non-negative S=0 / A=0 boundaries and coupled overflow carbon/oxygen sinks."""
    p = BioreactorParams()
    x_zero = default_initial_state(p).copy()
    x_zero[2] = 0.0
    x_zero[4] = 0.0
    u0 = OperatingInputs(F=0.0, rpm=450.0, vvm=1.2)

    m_hyb = PhysicsInformedNeuralODE(n_bootstrap=1)
    for mode in ("ground_truth", "nominal", "hybrid"):
        rhs = bioreactor_rhs(
            0.0,
            x_zero,
            u0,
            p,
            mode=mode,
            residual_model=(m_hyb if mode == "hybrid" else None),
        )
        assert rhs[2] >= 0.0
        assert rhs[4] >= 0.0

    # Overflow condition: glucose surplus triggers acetate production carbon sink in dS/dt
    x_ov = np.array([10.0, 0.2, 2.5, 5.5, 0.2, 1.0, 2.5])
    r_gt = compute_rates(x_ov, u0, p, mode="ground_truth")
    r_nom = compute_rates(x_ov, u0, p, mode="nominal")
    assert r_gt["q_A_prod"] > 0.01
    rhs_gt = bioreactor_rhs(0.0, x_ov, u0, p, mode="ground_truth")
    assert np.all(np.isfinite(rhs_gt))
    assert r_nom["q_A_prod"] == 0.0

    # Simulate extreme hypoxia/starvation stress and assert all 7 states stay non-negative
    sim = simulate_bioreactor(
        (0.0, 8.0),
        default_initial_state(p),
        lambda t, x: OperatingInputs(F=0.0, rpm=200.0, vvm=0.25),
        params=p,
        mode="ground_truth",
    )
    assert np.all(np.isfinite(sim["states"]))
    assert np.all(sim["states"] >= 0.0)


@pytest.mark.slow
def test_scale_invariance_1000L(calibrated_twin_assets):
    """Verify intensive scale invariance (zero V leakage) and geometric 1000 L scale-up."""
    _, _, p_rec, model, _ = calibrated_twin_assets

    p5 = p_rec
    p1000 = BioreactorParams(
        **{**p_rec.to_dict(), "V_vessel": 1000.0, "V_0": 400.0, "V_max": 1000.0}
    )
    v5 = 3.5
    scale = 1000.0 / 5.0
    v1000 = v5 * scale
    rpm5, vvm5 = 520.0, 1.25
    hyd5 = compute_hydrodynamics(rpm5, vvm5, v5, p5)

    vvm1000 = vvm5 * (5.0 / 1000.0) ** (1.0 / 3.0)
    rpm1000 = brentq(
        lambda r: compute_hydrodynamics(r, vvm1000, v1000, p1000)["Pg_V"] - hyd5["Pg_V"],
        40.0,
        400.0,
    )

    x5 = np.array([12.0, 0.8, 0.35, 3.2, 0.45, 2.5, v5])
    x1000 = x5.copy()
    x1000[6] = v1000
    u5 = OperatingInputs(F=0.12, rpm=rpm5, vvm=vvm5)
    u1000 = OperatingInputs(F=0.12 * scale, rpm=float(rpm1000), vvm=vvm1000)

    r5 = compute_rates(x5, u5, p5, mode="hybrid", residual_model=model)
    r1000 = compute_rates(x1000, u1000, p1000, mode="hybrid", residual_model=model)

    assert abs(r5["Pg_V"] - r1000["Pg_V"]) < 1e-8
    assert abs(r5["v_s"] - r1000["v_s"]) < 1e-12
    assert abs(r5["k_La"] - r1000["k_La"]) < 1e-8
    assert r1000["v_tip"] > 1.5 * r5["v_tip"]

    # Check exact equality of neural outputs when queried with identical intensive inputs
    mu_m = p5.mu_max * (x5[2] / (p5.K_S + x5[2])) * (x5[3] / (p5.K_O2 + x5[3]))
    q_a = model.predict_full_with_uq(
        X_v=12.0, X_d=0.8, S=0.35, C_L=3.2, A=0.45,
        v_tip=1.5, Pg_V=hyd5["Pg_V"], v_s=hyd5["v_s"], mu_mech=mu_m, V=3.5,
    )
    q_b = model.predict_full_with_uq(
        X_v=12.0, X_d=0.8, S=0.35, C_L=3.2, A=0.45,
        v_tip=1.5, Pg_V=hyd5["Pg_V"], v_s=hyd5["v_s"], mu_mech=mu_m, V=700.0,
    )
    for k in ("delta_kd", "eta_kla", "q_A_prod", "q_A_cons", "eta_mu", "d_m"):
        assert abs(q_a[k] - q_b[k]) < 1e-12

    # Full 1000 L closed-loop simulation completes physically without divergence
    ctrl_1000 = ClosedLoopFeedController(
        p1000, model, rpm=float(rpm1000), vvm=vvm1000, F_max=0.30 * scale
    )
    sim_1000 = run_twin_estimation_loop(
        (0.0, 12.0), default_initial_state(p1000), ctrl_1000, p1000, model, plant_params=p1000
    )
    assert np.all(np.isfinite(sim_1000["est_hyb"]))
    assert np.all(sim_1000["est_hyb"] >= 0.0)


@pytest.mark.slow
def test_stage1_parameter_recovery_and_conditioning(calibrated_twin_assets):
    """Verify multi-batch iDoE Stage-1 recovers all 6 kinetic & transfer parameters accurately."""
    _, rec, _, _, _ = calibrated_twin_assets
    assert rec["param_names"] == ["mu_max", "K_S", "Y_XS", "alpha", "beta", "k_La0"]
    assert len(rec["recovered_values"]) == 6
    assert "k_La0" in rec["recovered_values"]
    assert rec["max_relative_error_pct"] < 8.0
    assert rec["condition_number_rel"] < 45.0
    assert len(rec["calibration_sha256"]) == 64
    assert isinstance(rec["recovered_params"], BioreactorParams)


@pytest.mark.slow
def test_gam86_indirect_training_no_xd_leakage_and_uq(calibrated_twin_assets):
    """Verify zero X_d supervision in training tensors, held-out R2/rollout, and DoV/UQ."""
    _, _, p_rec, model, metrics = calibrated_twin_assets

    assert model.n_params == 86
    assert len(model.ensemble_weights) == 15
    assert bool(metrics["converged"]) is True
    assert metrics["kd_r2"] >= 0.70
    assert metrics["kla_r2"] > 0.90
    assert metrics["qAnet_r2"] > 0.85
    assert metrics["mu_r2"] > 0.95
    assert metrics["hybrid_norm_rmse"] < 0.35 * metrics["nominal_norm_rmse"]

    # Prove _prepare_batch_tensors puts ZERO weight on unmeasured X_d (index 1)
    train_b, _ = generate_training_and_test_batches(p_rec)
    bdata = _prepare_batch_tensors(train_b, p_rec, noise_std=0.015, rng=np.random.default_rng(7))
    assert float(np.max(bdata["w_obs"][:, :, 1])) == 0.0

    # Verify Mahalanobis DoV triggers OOD, inflates bootstrap std, and relaxes prior sharply
    in_dom = model.predict_full_with_uq(X_v=10.0, X_d=0.4, S=0.4, C_L=4.5, A=0.3, v_tip=1.2, Pg_V=900.0, v_s=0.008, mu_mech=0.30)
    ood = model.predict_full_with_uq(X_v=55.0, X_d=6.0, S=18.0, C_L=0.1, A=14.0, v_tip=4.2, Pg_V=9000.0, v_s=0.05, mu_mech=0.40)
    assert bool(ood["in_domain"]) is False
    assert ood["d_m"] > in_dom["d_m"]
    assert ood["std_kla"] > in_dom["std_kla"]
    assert ood["std_kd"] > 0.0

    # High-biomass 84 g/L rheology prediction within 2.5-sigma calibrated uncertainty
    hi_x = model.predict_full_with_uq(X_v=80.0, X_d=4.0, S=0.3, C_L=4.5, A=0.2, v_tip=1.5, Pg_V=900.0, v_s=0.01, mu_mech=0.30)
    assert abs(hi_x["eta_kla"] - true_kla_viscosity_factor(84.0)) <= 2.5 * hi_x["std_kla"]

    # Sharp metabolic OOD prior relaxation at extreme substrate & acetate
    ood_met = model.predict_full_with_uq(X_v=10.0, X_d=0.3, S=15.0, C_L=4.0, A=8.0, v_tip=1.5, Pg_V=900.0, v_s=0.01, mu_mech=0.30)
    assert bool(ood_met["in_domain"]) is False
    assert abs(ood_met["eta_mu"] - 1.0) < 0.01
    assert abs(ood_met["q_A_prod"] - ood_met["q_A_cons"]) < 1e-3


@pytest.mark.unit
def test_vjp_gradient_matches_central_fd():
    """Verify reverse-mode VJP across all 86 parameters against central finite differences."""
    p = BioreactorParams()
    train_b, _ = generate_training_and_test_batches(p)
    rng = np.random.default_rng(99)
    bdata = _prepare_batch_tensors(train_b[:2], p, noise_std=0.015, rng=rng, subsample_step=2)
    b20 = {
        "x0": bdata["x0"],
        "F": bdata["F"][:20],
        "S_feed": bdata["S_feed"][:20],
        "kla_nom": bdata["kla_nom"][:20],
        "v_tip": bdata["v_tip"][:20],
        "y_obs": bdata["y_obs"][:21],
        "w_obs": bdata["w_obs"][:21],
        "inv_var_obs": bdata["inv_var_obs"][:21],
        "dt": bdata["dt"],
    }
    th = _gam_init_weights(7) + rng.normal(0.0, 0.05, size=86)
    _, grad_a = _trajectory_loss_and_grad(th, b20, p)
    eps = 1e-6
    grad_fd = np.zeros_like(th)
    for i in range(len(th)):
        tp = th.copy(); tp[i] += eps
        tm = th.copy(); tm[i] -= eps
        grad_fd[i] = (_trajectory_loss_and_grad(tp, b20, p)[0] - _trajectory_loss_and_grad(tm, b20, p)[0]) / (2.0 * eps)
    denom = np.maximum(1e-5, np.maximum(np.abs(grad_a), np.abs(grad_fd)))
    max_rel = float(np.max(np.abs(grad_a - grad_fd) / denom))
    assert max_rel < 1e-3


@pytest.mark.slow
def test_ekf_expm_validation_and_sensor_parity(calibrated_twin_assets):
    """Verify unclipped expm, 1D/2D Q_diag validation, and fair EKF parity."""
    true_params, _, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)

    # 1D and 2D symmetric PSD Q_diag acceptance / rejection
    ekf_1d = HybridExtendedKalmanFilter(x0, p_rec, mode="hybrid", residual_model=model, Q_diag=Q_DEFAULT)
    ekf_2d = HybridExtendedKalmanFilter(x0, p_rec, mode="hybrid", residual_model=model, Q_diag=np.diag(Q_DEFAULT))
    assert np.allclose(ekf_1d.Q_base, ekf_2d.Q_base)

    with pytest.raises(ValueError):
        HybridExtendedKalmanFilter(x0, p_rec, mode="hybrid", residual_model=model, Q_diag=-Q_DEFAULT)
    bad_asym = np.diag(Q_DEFAULT)
    bad_asym[0, 1] = 0.01
    with pytest.raises(ValueError):
        HybridExtendedKalmanFilter(x0, p_rec, mode="hybrid", residual_model=model, Q_diag=bad_asym)

    # Verify identical Q_DEFAULT across nominal and hybrid modes and non-zero epistemic Q
    ekf_nom = HybridExtendedKalmanFilter(x0, p_rec, mode="nominal")
    assert np.allclose(ekf_1d.Q_base, ekf_nom.Q_base)

    u_stiff = OperatingInputs(F=0.05, rpm=650.0, vvm=1.8)
    ekf_1d.predict(u_stiff, dt=0.1)
    assert float(np.max(ekf_1d.last_epistemic_cov)) > 0.0
    # Verify unclipped expm handles eigenvalues with |F_mat*dt| > 20 smoothly
    F_mat = ekf_1d._jacobian(ekf_1d.x, u_stiff)
    assert float(np.max(np.abs(F_mat * 0.1))) > 20.0
    phi_exact = expm(F_mat * 0.1)
    assert np.all(np.isfinite(phi_exact))

    # Sensor parity check inside full closed-loop run
    ctrl = ClosedLoopFeedController(p_rec, model)
    sim = run_twin_estimation_loop(
        (0.0, 16.0), x0, ctrl, params=p_rec, residual_model=model, plant_params=true_params
    )
    kpis = summarize_control_run(sim, mu_target=0.28, params=p_rec)
    assert kpis["hybrid_norm_rmse"] < 0.5 * kpis["nominal_norm_rmse"]


@pytest.mark.slow
def test_benchmark_controllers_and_disturbance_rejection(calibrated_twin_assets):
    """Verify all 3 strategies respect parity, no hardcoded acetate cap, and hybrid advantage."""
    true_params, _, p_rec, model, _ = calibrated_twin_assets

    src = inspect.getsource(ClosedLoopFeedController.__call__)
    assert "8.0 / (8.0 + A)" not in src

    bench = benchmark_controllers(
        params=p_rec,
        residual_model=model,
        duration_h=16.0,
        mu_target=0.28,
        plant_params=true_params,
    )
    s_hyb = bench["hybrid_closed_loop"]["summary"]
    s_nom = bench["nominal_closed_loop"]["summary"]
    s_ol = bench["open_loop_exp"]["summary"]

    for s in (s_hyb, s_nom, s_ol):
        assert s["hybrid_norm_rmse"] < 0.5 * s["nominal_norm_rmse"]

    assert s_hyb["peak_acetate_g_L"] < s_nom["peak_acetate_g_L"]
    assert s_hyb["peak_acetate_g_L"] < s_ol["peak_acetate_g_L"]
    assert s_hyb["min_DO_pct"] > s_nom["min_DO_pct"]
    assert s_hyb["min_DO_pct"] > s_ol["min_DO_pct"]

    # Disturbance test (aeration drop & unmeasured feed_step dilution) stays physical and blind to S_feed fault
    ctrl_dist = ClosedLoopFeedController(p_rec, model, disturbance_mode="aeration_drop")
    sim_d = run_twin_estimation_loop(
        (0.0, 14.0), default_initial_state(p_rec), ctrl_dist, p_rec, model, plant_params=true_params
    )
    assert np.all(np.isfinite(sim_d["true_states"]))
    assert np.all(sim_d["true_states"] >= 0.0)

    ctrl_feed = ClosedLoopFeedController(p_rec, model, disturbance_mode="feed_step")
    assert ctrl_feed(8.0, default_initial_state(p_rec)).S_feed is None
    sim_f = run_twin_estimation_loop(
        (0.0, 14.0), default_initial_state(p_rec), ctrl_feed, p_rec, model, plant_params=true_params
    )
    kpis_f = summarize_control_run(sim_f, mu_target=0.28, params=p_rec)
    assert kpis_f["hybrid_norm_rmse"] < kpis_f["nominal_norm_rmse"]


@pytest.mark.slow
def test_digital_twin_delayed_lims_and_operator_ui_df(calibrated_twin_assets):
    """Verify asynchronous LIMS replay, ALCOA+ provenance, and oracle-free Operator DataFrame."""
    _, rec, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)
    twin = BioreactorDigitalTwin(x0, p_rec, model, calibration_metadata={"sha": rec["calibration_sha256"]})
    ctrl = ClosedLoopFeedController(p_rec, model)

    for step in range(1, 11):
        t_h = 0.1 * step
        act = twin.compute_control_action(t_h, ctrl)
        twin.ingest_online_pat(
            t_h, 1.5 + 0.06 * step, 7.1, float(x0[6]), act["F_cmd"], act["rpm_cmd"], act["vvm_cmd"]
        )

    x_before = twin.ekf.x.copy()
    replay_out = twin.ingest_atline_assay(t_sample_h=0.4, y_S=0.30, y_A=0.55, y_P=0.20)
    assert replay_out["replayed_steps"] == 6
    assert float(np.linalg.norm(twin.ekf.x - x_before)) > 1e-4

    audit = twin.to_audit_record()
    assert len(audit["params_sha256"]) == 64
    assert len(audit["model_sha256"]) == 64
    assert len(audit["record_sha256"]) == 64
    assert audit["audit_log_count"] >= 20

    # Verify full untruncated ALCOA+ replay and Operator UI DataFrame hides *_true when include_truth=False
    sim_small = run_twin_estimation_loop((0.0, 3.0), x0, ctrl, p_rec, model, dt_step=0.1)
    twin_full = BioreactorDigitalTwin.from_simulation_run(sim_small, p_rec, model)
    audit_full = twin_full.to_audit_record()
    assert audit_full["audit_log_count"] > 25
    assert len(audit_full["audit_log"]) == audit_full["audit_log_count"]

    df_op = app.build_dataframe(sim_small, include_truth=False)
    assert "V_hyb" in df_op.columns
    assert "X_v_nom" in df_op.columns and "X_v_std" in df_op.columns
    assert "d_m" in df_op.columns and "in_domain" in df_op.columns
    assert not any(col.endswith("_true") for col in df_op.columns)


@pytest.mark.slow
def test_output_space_uq_ddof1_and_calibrated_predict_residuals(calibrated_twin_assets):
    """Verify output-space ensemble mean, ddof=1 empirical variance, and calibrated predict_residuals."""
    _, _, p_rec, model, _ = calibrated_twin_assets

    fit_src = inspect.getsource(PhysicsInformedNeuralODE.fit)
    assert "slice(2, 8)" not in fit_src
    assert model.calibrated_params is not None

    # Compare predict_batch_with_uq & predict_point_fast against explicit output-space mean across B=15
    X_v, X_d, S, C_L, A = 11.5, 0.35, 0.42, 4.1, 0.28
    v_tip, Pg_V, v_s, mu_m = 1.35, 920.0, 0.0085, 0.31
    uq = model.predict_full_with_uq(
        X_v=X_v, X_d=X_d, S=S, C_L=C_L, A=A, v_tip=v_tip, Pg_V=Pg_V, v_s=v_s, mu_mech=mu_m
    )
    fast = model.predict_point_fast(
        X_v=X_v, X_d=X_d, S=S, C_L=C_L, A=A, v_tip=v_tip, Pg_V=Pg_V, v_s=v_s, mu_mech=mu_m
    )
    for key in ("delta_kd", "eta_kla", "q_A_prod", "q_A_cons", "eta_mu"):
        assert abs(uq[key] - fast[key]) < 1e-12

    # In-domain point: output-space mean and Bessel-corrected ddof=1 variance
    xt_arr = np.array([X_v + X_d])
    heads = [
        _gam_eval_heads(
            th, xt_arr, np.array([S]), np.array([C_L]), np.array([A]), np.array([v_tip]), np.array([mu_m])
        )
        for th in model.ensemble_weights
    ]
    dkd_samples = np.array([h[0][0] for h in heads])
    ek_samples = np.array([h[1][0] for h in heads])
    qp_samples = np.array([h[2][0] for h in heads])
    qc_samples = np.array([h[3][0] for h in heads])
    emu_samples = np.array([h[4][0] for h in heads])

    assert bool(uq["in_domain"]) is True
    assert abs(uq["delta_kd"] - float(np.mean(dkd_samples))) < 1e-12
    assert abs(uq["eta_kla"] - float(np.mean(ek_samples))) < 1e-12
    assert abs(uq["q_A_prod"] - float(np.mean(qp_samples))) < 1e-12
    assert abs(uq["q_A_cons"] - float(np.mean(qc_samples))) < 1e-12
    assert abs(uq["eta_mu"] - float(np.mean(emu_samples))) < 1e-12

    # Check empirical variance uses ddof=1 rather than ddof=0
    emp_var_ek = float(np.var(ek_samples, ddof=1))
    emp_var_ek_ddof0 = float(np.var(ek_samples, ddof=0))
    recovered_var_ek = float(uq["std_kla"] ** 2 - 0.025 ** 2)
    assert abs(recovered_var_ek - emp_var_ek) < 1e-12
    assert abs(recovered_var_ek - emp_var_ek_ddof0) > 1e-10

    # Verify predict_residuals binds to calibrated_params rather than default BioreactorParams()
    m_custom = copy.copy(model)
    p_custom = BioreactorParams(
        **{**p_rec.to_dict(), "mu_max": 0.62, "K_S": 0.08, "V_vessel": 50.0, "V_0": 20.0, "V_max": 50.0}
    )
    m_custom.calibrated_params = p_custom
    res_bound = m_custom.predict_residuals(X_v=10.0, S=0.3, C_L=4.0, A=0.5, rpm=250.0, vvm=0.7, V=25.0)
    res_explicit = m_custom.predict_residuals(
        X_v=10.0, S=0.3, C_L=4.0, A=0.5, rpm=250.0, vvm=0.7, V=25.0, params=p_custom
    )
    res_default = m_custom.predict_residuals(
        X_v=10.0, S=0.3, C_L=4.0, A=0.5, rpm=250.0, vvm=0.7, V=25.0, params=BioreactorParams()
    )
    assert res_bound == pytest.approx(res_explicit, abs=1e-12)
    assert np.max(np.abs(np.array(res_bound) - np.array(res_default))) > 1e-5


@pytest.mark.slow
def test_dual_ood_residual_gate_and_controller_safe_fallback(calibrated_twin_assets):
    """Verify dual Mahalanobis + NIS/SPE gate sets undecided and triggers ML-free fallback + escalation."""
    _, _, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)

    # 1. Pure NIS/SPE residual breach while state remains in Mahalanobis DoV
    ekf = HybridExtendedKalmanFilter(x0, params=p_rec, mode="hybrid", residual_model=model)
    ekf.predict(OperatingInputs(F=0.04, rpm=450.0, vvm=1.2), dt=0.1)
    gate_nom = ekf.evaluate_gate()
    assert gate_nom["dov_in_domain"] is True
    assert gate_nom["in_domain"] is True
    assert gate_nom["undecided"] is False

    # Inject severe unmodeled DO/biomass discrepancy while remaining inside Mahalanobis DoV
    y_anomalous = np.array([ekf.x[0], max(0.2, ekf.x[3] - 4.5), ekf.x[6]], dtype=float)
    ekf.update(y_anomalous, [0, 3, 6], np.array([0.12, 0.08, 0.015]) ** 2)
    gate_res = ekf.evaluate_gate()
    assert gate_res["dov_in_domain"] is True
    assert gate_res["residual_breach"] is True
    assert gate_res["in_domain"] is False
    assert gate_res["undecided"] is True
    assert "residual" in gate_res["gate_reason"]

    # 2. ClosedLoopFeedController fallback on residual breach ("safe_nominal" and "hold")
    x_nom = np.array([10.0, 0.3, 0.30, 5.2, 0.25, 2.0, 2.6], dtype=float)
    ctrl_safe = ClosedLoopFeedController(p_rec, model, fallback_policy="safe_nominal")
    u_adap = ctrl_safe(4.0, x_nom, gate_status=gate_nom)
    assert ctrl_safe.last_fallback_active is False
    assert ctrl_safe.last_undecided is False
    assert ctrl_safe.last_escalation_event is None

    u_fallback = ctrl_safe(4.1, x_nom, gate_status=gate_res)
    assert ctrl_safe.last_fallback_active is True
    assert ctrl_safe.last_undecided is True
    assert ctrl_safe.last_control_mode == "safe_fallback_safe_nominal"
    assert ctrl_safe.last_escalation_event is not None
    assert ctrl_safe.last_escalation_event["event_type"] == "CONTROL_FALLBACK_ESCALATION"
    assert u_fallback.F <= u_adap.F

    ctrl_hold = ClosedLoopFeedController(p_rec, model, fallback_policy="hold", F_hold=0.01)
    u_hold = ctrl_hold(4.1, x_nom, gate_status=gate_res)
    assert ctrl_hold.last_fallback_active is True
    assert ctrl_hold.last_control_mode == "safe_fallback_hold"
    assert u_hold.F == pytest.approx(0.01)

    # 3. Mahalanobis OOD triggers undecided + fallback + escalation even without external gate_status
    x_ood = np.array([55.0, 6.0, 15.0, 4.5, 10.0, 5.0, 3.0], dtype=float)
    u_ood = ctrl_safe(8.0, x_ood)
    assert ctrl_safe.last_fallback_active is True
    assert ctrl_safe.last_undecided is True
    assert u_ood.F == 0.0

    # 4. BioreactorDigitalTwin logs CONTROL_ESCALATION into tamper-evident ALCOA+ audit chain
    twin = BioreactorDigitalTwin(x0, p_rec, model)
    twin.ingest_online_pat(0.1, float(x0[0]), 0.5, float(x0[6]), 0.02, 450.0, 1.2)
    act = twin.compute_control_action(0.1, ctrl_safe)
    assert act["undecided"] is True
    assert act["fallback_active"] is True
    assert act["escalation_event"] is not None
    assert any(e["event_type"] == "CONTROL_ESCALATION" for e in twin.audit_log)
    ok_chain, errs = twin.verify_audit_chain()
    assert ok_chain is True and not errs


@pytest.mark.slow
def test_alcoa_hash_chain_tamper_detection_and_lims_replay_deltas(calibrated_twin_assets):
    """Verify full SHA-256 prev_entry_sha256 linking, LIMS replay deltas, and tamper detection."""
    _, _, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)
    twin = BioreactorDigitalTwin(x0, p_rec, model)
    ctrl = ClosedLoopFeedController(p_rec, model)

    for step in range(1, 6):
        t_h = 0.1 * step
        act = twin.compute_control_action(t_h, ctrl)
        twin.ingest_online_pat(
            t_h, 1.5 + 0.08 * step, 7.4, float(x0[6]), act["F_cmd"], act["rpm_cmd"], act["vvm_cmd"]
        )

    replay = twin.ingest_atline_assay(t_sample_h=0.2, y_S=0.25, y_A=0.60, y_P=0.25)
    assert replay["replayed_steps"] == 3
    assert replay["state_delta_l2"] > 1e-4
    assert np.allclose(replay["post_replay_state"] - replay["pre_replay_state"], replay["state_delta"])

    replay_entries = [e for e in twin.audit_log if e["event_type"] == "ATLINE_LIMS_REPLAY"]
    assert len(replay_entries) == 1
    det = replay_entries[0]["detail"]
    assert len(det["replayed_step_deltas"]) == 3
    assert len(det["pre_replay_state"]) == 7 and len(det["post_replay_state"]) == 7
    assert det["state_delta_l2"] == pytest.approx(replay["state_delta_l2"])

    # Full 64-char SHA-256 chain verification
    assert twin.audit_log[0]["prev_entry_sha256"] == GENESIS_ENTRY_SHA256
    for i, entry in enumerate(twin.audit_log):
        assert len(entry["entry_sha256"]) == 64
        assert len(entry["prev_entry_sha256"]) == 64
        assert len(entry["params_sha256"]) == 64
        if i > 0:
            assert entry["prev_entry_sha256"] == twin.audit_log[i - 1]["entry_sha256"]

    ok_chain, errs = twin.verify_audit_chain()
    assert ok_chain is True and not errs
    rec_audit = twin.to_audit_record()
    assert rec_audit["chain_valid"] is True
    ok_rec, rec_errs = BioreactorDigitalTwin.verify_audit_record(rec_audit)
    assert ok_rec is True and not rec_errs

    # Tamper tests: (a) detail mutation, (b) broken prev link, (c) entry deletion, (d) record tamper
    tampered_detail = copy.deepcopy(twin.audit_log)
    tampered_detail[2]["detail"]["y_Xv"] += 1.0
    ok_a, errs_a = verify_audit_chain(tampered_detail)
    assert ok_a is False and len(errs_a) > 0

    tampered_link = copy.deepcopy(twin.audit_log)
    tampered_link[3]["prev_entry_sha256"] = "f" * 64
    ok_b, _ = verify_audit_chain(tampered_link)
    assert ok_b is False

    tampered_del = copy.deepcopy(twin.audit_log)
    del tampered_del[1]
    ok_c, _ = verify_audit_chain(tampered_del)
    assert ok_c is False

    tampered_rec = copy.deepcopy(rec_audit)
    tampered_rec["t_now_h"] = 999.0
    ok_d, _ = BioreactorDigitalTwin.verify_audit_record(tampered_rec)
    assert ok_d is False


@pytest.mark.slow
def test_blind_actuator_disturbance_parity_and_fault_detection(calibrated_twin_assets):
    """Verify controllers/EKFs remain blind to actuator faults and dual gate catches blind aeration drop."""
    true_params, _, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)

    # All 4 controllers must command nominal (rpm, vvm) and S_feed=None even under disturbance_mode
    controllers = [
        ClosedLoopFeedController(p_rec, model, rpm=450.0, vvm=1.2, disturbance_mode="aeration_drop"),
        NominalMechanisticController(p_rec, rpm=450.0, vvm=1.2, disturbance_mode="stirrer_drop"),
        OpenLoopExponentialController(p_rec, x0, rpm=450.0, vvm=1.2, disturbance_mode="shear_spike"),
        ManualFeedController(p_rec, F_set=0.05, rpm=450.0, vvm=1.2, disturbance_mode="combo"),
    ]
    for ctrl in controllers:
        u_8 = ctrl(8.0, x0)
        assert u_8.rpm == pytest.approx(450.0)
        assert u_8.vvm == pytest.approx(1.2)
        assert u_8.S_feed is None

    # Nominal run (disturbance_mode="none") must yield 0 false-positive residual/fallback trips
    ctrl_nom = ClosedLoopFeedController(p_rec, model, disturbance_mode="none")
    sim_nom = run_twin_estimation_loop(
        (0.0, 16.0), x0, ctrl_nom, params=p_rec, residual_model=model, seed=123, plant_params=true_params
    )
    kpis_nom = summarize_control_run(sim_nom, mu_target=0.28, params=p_rec)
    assert kpis_nom["undecided_fraction"] == 0.0
    assert kpis_nom["fallback_fraction"] == 0.0

    # Blind aeration_drop: model/inputs stay at vvm=1.2 while plant drops vvm; dual gate catches it & escalates
    ctrl_air = ClosedLoopFeedController(p_rec, model, disturbance_mode="aeration_drop")
    sim_air = run_twin_estimation_loop(
        (0.0, 14.0), x0, ctrl_air, params=p_rec, residual_model=model, seed=123, plant_params=true_params
    )
    assert np.allclose(sim_air["inputs"][:, 2], 1.2)
    assert float(np.min(sim_air["plant_inputs"][:, 2])) < 0.6
    kpis_air = summarize_control_run(sim_air, mu_target=0.28, params=p_rec)
    assert kpis_air["undecided_fraction"] > 0.0
    assert kpis_air["fallback_fraction"] > 0.0
    assert kpis_air["escalation_count"] >= 1.0
    assert kpis_air["hybrid_norm_rmse"] < 0.5 * kpis_air["nominal_norm_rmse"]


@pytest.mark.contract
def test_stress_test_cleanliness_and_honest_maturity_labels():
    """Verify stress test has no clipped expm or hardcoded campaign/metric literals and docs are honest."""
    src_ekf = inspect.getsource(stu.run_pade_ekf_stress_test)
    assert "np.clip(f_jac" not in src_ekf
    assert "expm(f_jac * dt)" in src_ekf
    assert "EXP011" not in src_ekf

    src_stu = inspect.getsource(stu)
    assert '"unremediated": {' not in src_stu
    assert "24.991" not in src_stu
    for fn in (
        stu.load_uhlendorff_dataset,
        stu.compute_duplicate_noise_floor,
        stu.PreparedCHOBatch.from_runs,
        stu.run_pade_ekf_stress_test,
        stu.main,
    ):
        assert "EXP011" not in inspect.getsource(fn)

    # Data-driven anomaly detection on synthetic CHORun objects (no hardcoded campaign IDs)
    def _synth_run(exp_id: str, rep_id: str, vcd_vals: np.ndarray, amm_vals: np.ndarray) -> stu.CHORun:
        m = len(vcd_vals)
        t_eval = np.linspace(0.0, 48.0, m)
        y_obs = np.column_stack([vcd_vals, np.full(m, 4.0), np.full(m, 0.5), amm_vals])
        t_inl = np.linspace(0.0, 48.0, 20)
        return stu.CHORun(
            exp_id=exp_id,
            rep_id=rep_id,
            mode="batch",
            t_eval=t_eval,
            y_obs=y_obs,
            v_grid=np.full(m, 0.6),
            f_in_offline=np.zeros(m),
            t_feed_start=1e9,
            c_glc_feed=0.0,
            pat_features=np.zeros((m, 10)),
            pat_valid_7=np.ones(m, dtype=bool),
            t_inline=t_inl,
            pat_inline=np.zeros((20, 4)),
            pat_inline_valid=np.ones(20, dtype=bool),
            f_in_inline=np.zeros(20),
            v_inline=np.full(20, 0.6),
        )

    clean_r1 = _synth_run("SYN01", "R01", np.array([0.5, 2.0, 4.0, 3.5]), np.array([1.0, 2.0, 3.0, 4.0]))
    clean_r2 = _synth_run("SYN01", "R02", np.array([0.5, 2.1, 3.9, 3.6]), np.array([1.1, 2.0, 3.1, 4.0]))
    ok_r1 = _synth_run("SYN99", "R01", np.array([0.5, 2.0, 4.0, 3.5]), np.array([1.0, 2.0, 3.0, 4.0]))
    bad_r2 = _synth_run("SYN99", "R02", np.array([0.0, 0.0, 0.0, 0.0]), np.array([45.0, 46.0, 48.0, 49.0]))

    assert stu.is_biophysical_anomaly_run(clean_r1) is False
    assert stu.is_biophysical_anomaly_run(bad_r2) is True
    sig_clean = stu.compute_duplicate_noise_floor([clean_r1, clean_r2, ok_r1, bad_r2], exclude_anomalous=True)
    sig_all = stu.compute_duplicate_noise_floor([clean_r1, clean_r2, ok_r1, bad_r2], exclude_anomalous=False)
    assert float(sig_clean[3]) < 0.2
    assert float(sig_all[3]) > 20.0

    # Verify honest Portela maturity claims across README, app.py, and estimator.py
    root = Path(__file__).resolve().parent.parent
    for rel_path in ("README.md", "app.py", "hybrid_fedbatch_twin/estimator.py"):
        text = (root / rel_path).read_text(encoding="utf-8")
        assert "Portela Level 3+" not in text
        assert "Portela Level 3-Ready" not in text
        assert "Portela Level 3 Digital Twin" not in text
        assert "9.2 / 10" not in text


@pytest.mark.slow
def test_stiff_van_loan_ekf_nis_and_do_crash_detection(calibrated_twin_assets):
    """Verify stiff Van Loan Q_d avoids DO covariance inflation and flags severe DO crash via NIS."""
    _, _, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)
    ekf = HybridExtendedKalmanFilter(x0, params=p_rec, mode="hybrid", residual_model=model)
    ekf.predict(OperatingInputs(F=0.04, rpm=450.0, vvm=1.2), dt=0.1)

    std_cl = float(np.sqrt(max(0.0, ekf.P[3, 3])))
    assert std_cl < 1.0, f"Expected calibrated std_CL < 1.0 mg/L under Van Loan Q_d, got {std_cl:.3f}"
    assert np.allclose(ekf.last_discrete_Q, ekf.last_discrete_Q.T, atol=1e-10)
    assert float(np.min(np.linalg.eigvalsh(ekf.last_discrete_Q))) >= -1e-12

    y_crash = np.array([ekf.x[0], max(0.2, ekf.x[3] - 4.5), ekf.x[6]], dtype=float)
    ekf.update(y_crash, [0, 3, 6], np.array([0.12, 0.08, 0.015]) ** 2)
    gate = ekf.evaluate_gate()
    assert gate["nis_breach"] is True
    assert gate["spe_breach"] is True
    assert gate["residual_breach"] is True
    assert gate["nis"] > gate["nis_threshold"]


@pytest.mark.slow
def test_truncated_audit_record_verification_and_tamper_resistance(calibrated_twin_assets):
    """Verify to_audit_record(max_entries=...) marks truncation honestly and detects tampering."""
    _, _, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)
    twin = BioreactorDigitalTwin(x0, p_rec, model)
    ctrl = ClosedLoopFeedController(p_rec, model)

    for step in range(1, 5):
        t_h = 0.1 * step
        act = twin.compute_control_action(t_h, ctrl)
        twin.ingest_online_pat(
            t_h, 1.5 + 0.06 * step, 7.2, float(x0[6]), act["F_cmd"], act["rpm_cmd"], act["vvm_cmd"]
        )

    rec_full = twin.to_audit_record()
    assert rec_full["audit_log_truncated"] is False
    assert BioreactorDigitalTwin.verify_audit_record(rec_full) == (True, [])

    rec_trunc = twin.to_audit_record(max_entries=3)
    assert rec_trunc["audit_log_truncated"] is True
    assert len(rec_trunc["audit_log"]) == 3
    assert rec_trunc["audit_log_count"] == len(twin.audit_log)
    assert BioreactorDigitalTwin.verify_audit_record(rec_trunc) == (True, [])

    bad_mut = copy.deepcopy(rec_trunc)
    bad_mut["audit_log"][0]["detail"]["tampered"] = True
    ok_mut, _ = BioreactorDigitalTwin.verify_audit_record(bad_mut)
    assert ok_mut is False

    bad_flag = copy.deepcopy(rec_trunc)
    bad_flag["audit_log_truncated"] = False
    ok_flag, _ = BioreactorDigitalTwin.verify_audit_record(bad_flag)
    assert ok_flag is False

    bad_del = copy.deepcopy(rec_full)
    bad_del["audit_log"] = bad_del["audit_log"][-3:]
    ok_del, _ = BioreactorDigitalTwin.verify_audit_record(bad_del)
    assert ok_del is False


@pytest.mark.slow
def test_app_audit_export_sealed_and_operator_truth_isolated(calibrated_twin_assets):
    """Verify app.py ALCOA+ export remains hash-valid and omits truth KPIs when include_truth=False."""
    true_params, rec, p_rec, model, _ = calibrated_twin_assets
    x0 = default_initial_state(p_rec)
    ctrl = ClosedLoopFeedController(p_rec, model)
    sim_small = run_twin_estimation_loop(
        (0.0, 2.0), x0, ctrl, p_rec, model, dt_step=0.1, plant_params=true_params
    )
    kpis = summarize_control_run(sim_small, mu_target=0.28, params=p_rec)

    rec_op = app.build_audit_export_payload(
        sim=sim_small,
        kpis=kpis,
        params=p_rec,
        residual_model=model,
        calibration_metadata={"calibration_sha256": rec["calibration_sha256"]},
        include_truth=False,
    )
    assert BioreactorDigitalTwin.verify_audit_record(rec_op) == (True, [])
    op_meta = rec_op["calibration_metadata"]
    assert op_meta["includes_oracle_truth"] is False
    op_kpis = op_meta["run_kpis"]
    assert "est_final_Xv_g_L" in op_kpis and "max_d_m" in op_kpis
    for truth_key in (
        "final_Xv_g_L",
        "final_Xd_g_L",
        "final_P_g_L",
        "peak_acetate_g_L",
        "min_DO_pct",
        "mu_rmse",
        "hybrid_norm_rmse",
        "nominal_norm_rmse",
    ):
        assert truth_key not in op_kpis

    rec_val = app.build_audit_export_payload(
        sim=sim_small,
        kpis=kpis,
        params=p_rec,
        residual_model=model,
        calibration_metadata={"calibration_sha256": rec["calibration_sha256"]},
        include_truth=True,
    )
    assert BioreactorDigitalTwin.verify_audit_record(rec_val) == (True, [])
    assert rec_val["calibration_metadata"]["includes_oracle_truth"] is True
    assert "hybrid_norm_rmse" in rec_val["calibration_metadata"]["run_kpis"]


@pytest.mark.slow
def test_1000L_default_operating_inputs_and_disturbance_scaling(calibrated_twin_assets):
    """Verify 1000 L twin defaults initialize in-domain and disturbance_profile scales physically."""
    _, _, p_rec, model, _ = calibrated_twin_assets
    p1000 = BioreactorParams(
        **{**p_rec.to_dict(), "V_vessel": 1000.0, "V_0": 400.0, "V_max": 1000.0}
    )
    u1000 = default_operating_inputs(p1000)
    assert 120.0 <= u1000.rpm <= 150.0
    assert 0.15 <= u1000.vvm <= 0.25

    twin_1000 = BioreactorDigitalTwin(default_initial_state(p1000), p1000, model)
    assert twin_1000.last_u.rpm == pytest.approx(u1000.rpm)
    assert twin_1000.last_u.vvm == pytest.approx(u1000.vvm)
    gate0 = twin_1000.ekf.evaluate_gate()
    assert gate0["in_domain"] is True
    assert gate0["undecided"] is False
    assert gate0["gate_reason"] == "nominal"

    _, vvm_air, _ = disturbance_profile("aeration_drop", 7.0, u1000.rpm, u1000.vvm, p1000.S_feed, V_vessel=1000.0)
    assert vvm_air < u1000.vvm
    rpm_stir, _, _ = disturbance_profile("stirrer_drop", 8.0, u1000.rpm, u1000.vvm, p1000.S_feed, V_vessel=1000.0)
    assert rpm_stir < u1000.rpm
    rpm_shear, _, _ = disturbance_profile("shear_spike", 9.0, u1000.rpm, u1000.vvm, p1000.S_feed, V_vessel=1000.0)
    assert rpm_shear > u1000.rpm
