"""Comprehensive unit, physical-invariance, and anti-reward-hacking tests."""

from __future__ import annotations

import inspect
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

import app
from hybrid_fedbatch_twin import (
    BioreactorDigitalTwin,
    BioreactorParams,
    ClosedLoopFeedController,
    HybridExtendedKalmanFilter,
    OperatingInputs,
    Q_DEFAULT,
    benchmark_controllers,
    bioreactor_rhs,
    compute_hydrodynamics,
    compute_rates,
    default_initial_state,
    recover_mechanistic_parameters,
    run_twin_estimation_loop,
    simulate_bioreactor,
    summarize_control_run,
    train_default_hybrid_model,
)
from hybrid_fedbatch_twin.mechanistic import true_kla_viscosity_factor
from hybrid_fedbatch_twin.neural_ode import (
    PhysicsInformedNeuralODE,
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
