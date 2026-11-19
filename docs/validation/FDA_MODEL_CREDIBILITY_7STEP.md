# FDA 7-Step Model Credibility Assessment (Bioprocess Hybrid Digital Twin)

**Model Name:** `hybrid-fedbatch-twin` (6-parameter iDoE Monod/Luedeking-Piret mechanistic backbone + `B=15` bootstrap `GAM-86` Physics-Informed Neural ODE + Continuous-Discrete Padé `expm` EKF)  
**Target Regulatory Framing:** FDA *Assessing the Credibility of Computational Modeling and Simulation in Medical Device Submissions* / FDA *AI/ML-Enabled Biomanufacturing & Process Analytical Technology (PAT)* 7-step credibility methodology (adapted here for upstream bioprocess soft-sensing and advisory supervisory feed control).

---

## Step 1 — Question of Interest

> *Can a grey-box hybrid digital twin calibrated solely on benchtop (`5 L`) measurable PAT/at-line trajectories accurately estimate unmeasured bioprocess rates (`k_d`, `k_La`, `q_A,net`, `mu`), reconstruct full 7-state dynamics across delayed LIMS assays, transfer across scales (`5 L` -> `1000 L`), and fail safe to ML-free control when operating out of domain or under unmodeled plant faults?*

---

## Step 2 — Context of Use (CoU) & Portela Maturity Boundary

- **Intended Use:**
  1. Offline *in-silico* bioprocess simulation, control-strategy benchmarking, and operator decision-support training (`5 L`, `50 L`, `1000 L`).
  2. Advisory state estimation (`X_v`, `X_d`, `S`, `C_L`, `A`, `P`, `V`) via Continuous-Discrete EKF (`hybrid_fedbatch_twin/estimator.py`) ingesting online PAT and retrodictively replaying delayed at-line LIMS samples.
  3. Advisory closed-loop `mu-stat` feed recommendation (`hybrid_fedbatch_twin/controller.py`) with automatic ML-free fallback (`safe_nominal` or `hold`) whenever dual Mahalanobis Domain-of-Validity (`DoV`) or innovation chi-square (`NIS`/`SPE`) gates trigger `"undecided"`.
  4. Grounded deviation/CAPA investigation drafting (`agents/deviation_graph.py`) staged exclusively for human 2FA QA review (`agents/part11.py`).
- **Explicit CoU Classification:** **Portela Level 1 Offline Digital Model / Level 2-Ready Soft-Sensor Architecture**. Live automated PLC/DCS closed-loop actuation in a GMP cleanroom is explicitly **outside** the validated Context of Use until facility IQ/OQ/PQ and wet-lab validation complete (see [`OPEN_ASSUMPTIONS_AND_GAPS.md`](OPEN_ASSUMPTIONS_AND_GAPS.md)).

---

## Step 3 — Model Risk Assessment

- **Model Influence:** **Medium-to-High (Advisory Supervisory Control & Deviation Investigation)** — model outputs inform feed setpoint recommendations, OOD alarms, and QA deviation CAPA drafts, though a human QA/operator gate remains in the decision loop.
- **Decision Consequence:** **High** if unguarded ML residuals extrapolated during oxygen-transfer collapse, sensor failure, or substrate overload, or if unverified AI CAPA claims bypassed QA review.
- **Resulting Credibility Risk Tier:** **Tier 2 / High Assurance Required** — mitigated architecturally by:
  - Physical positivity & non-negative mass balances (`bioreactor_rhs`, Patankar rollout),
  - Dual Mahalanobis `DoV` (`d_m`) + innovation `NIS`/`SPE` `"undecided"` gating (`HybridExtendedKalmanFilter.evaluate_gate`),
  - Strictly ML-free fallback feed (`ClosedLoopFeedController._compute_safe_fallback_feed`),
  - Deterministic CAPA guardrails (`validate_capa_draft`) and single-use 2FA QA sign-off (`build_part11_decision_record`).

---

## Step 4 — Verification (Code & Numerical Calculation Integrity)

All mathematical and numerical implementations are verified offline via `tests/test_hybrid_twin.py` and `tests/test_contracts_and_roundtrips.py`:
- **Exact Discrete Adjoint Gradients:** Reverse-mode VJP across all 86 `GAM-86` parameters (`_trajectory_loss_and_grad`) matches central finite differences (`test_vjp_gradient_matches_central_fd`).
- **Optimization Tolerances:** Stage-1 bounded Trust-Region Reflective `least_squares` runs with `ftol=1e-10` (`hybrid_fedbatch_twin/neural_ode.py:299`); Stage-2 L-BFGS-B runs with `ftol=1e-12` (`hybrid_fedbatch_twin/neural_ode.py:1567`).
- **Zero Unmeasured State Supervision:** `_prepare_batch_tensors` enforces zero training observation weight on unmeasured dead biomass `X_d` (`w_obs[:, :, 1] == 0.0`), verified by `test_gam86_indirect_training_no_xd_leakage_and_uq`.
- **Stiff Covariance Discretization:** `_discretize_linear_gaussian` computes exact `Phi = expm(F * dt)` and scaling-and-squaring Simpson Van Loan integral `Q_d`, verified symmetric PSD across stiff steps (`|F*dt| > 20`) without covariance explosion (`std_CL < 1.0 mg/L`) in `test_stiff_van_loan_ekf_nis_and_do_crash_detection`.
- **Output-Space Ensemble Statistics:** `PhysicsInformedNeuralODE.predict_batch_with_uq` averages predicted rate heads across `B=15` bootstrap members in output space with Bessel-corrected `ddof=1` empirical variance (`hybrid_fedbatch_twin/neural_ode.py:1299-1303`), verified in `test_output_space_uq_ddof1_and_calibrated_predict_residuals`.

---

## Step 5 — Validation Evidence (Synthetic Multi-Regime & External Empirical Wet-Lab)

### 5.1 Stage-1 Parameter Recovery & Conditioning (`results_summary.json`, `docs/assets/twin_bundle.json`)
- **6 Recovered Mechanistic Parameters:** `mu_max`, `K_S`, `Y_XS`, `alpha`, `beta`, `k_La0`.
- **Seed-42 Recovery:** `condition_number_rel = 38.18` (`38.1764`), `max_relative_error_pct = 4.39%` (`4.3934%` on `K_S`).
- **14-Seed Distribution:** `mean_err_pct = 1.59%` (`1.5857%`), `worst_max_err_pct = 7.84%`, `max_kappa_rel = 40.50`.

### 5.2 Stage-2 `GAM-86` (`B=15`) Held-Out Accuracy (`results_summary.json`)
- **Parsimony & Convergence:** `n_params = 86`, `n_bootstrap = 15` (`15/15` converged), `aic = 2194.81`, `bic = 2670.71`, `dov_p99 = 5.0118`.
- **Held-Out 5-Regime Rate Metrics:**
  - `kd_r2 = 0.7320` (`0.73204`, `kd_rmse = 0.009757`)
  - `kla_r2 = 0.9806` (`0.98058`, `kla_rmse = 27.4279`)
  - `qAnet_r2 = 0.8803` (`0.88032`)
  - `mu_r2 = 0.9925` (`0.99254`)
  - `hybrid_norm_rmse = 0.05146` vs. `nominal_norm_rmse = 0.47728`

### 5.3 Geometric Scale Transfer (`5 L` -> `1000 L`, `docs/assets/twin_bundle.json`)
- **Matched Intensive Operating Point (`1000 L`):** `rpm = 134.80 RPM`, `vvm = 0.2052 vvm`, `P_g/V = 584.90 W/m^3`, `v_s = 0.002354 m/s`.
- **1000 L Closed-Loop Estimation Benchmark:** `hybrid_norm_rmse = 0.03687` vs. `nominal_norm_rmse = 0.39081`.

### 5.4 External Empirical Validation: Uhlendorff et al. (2026) CHO-K1 (`artifacts/uhlendorff_2026_validation.json`)
- **Dataset:** Zenodo `10.5281/zenodo.20829178` — 12 campaigns (`EXP001..EXP012`, 24 replicate runs; 9 dev campaigns, 3 held-out test campaigns `EXP008`, `EXP009`, `EXP012`; 378 offline rows, 28 incomplete 7-sensor rows, 5 biophysical anomaly rows in `EXP011_R02`).
- **Scorecard Verdict (`remediated`):** **`8.61 / 10.00 -> GO (PASS)`**
  - Sub-scores: `generalization = 1.61`, `identifiability = 2.0`, `fedbatch_positivity = 1.0`, `shift_and_anomaly = 2.0`, `ci_and_ekf = 2.0`.
- **Key Empirical Metrics:**
  - `pooled_test_wmse = 23.3272` (`EXP008 = 8.5087`, `EXP009 = 9.9639`, `EXP012 = 51.5089`)
  - `ci_coverage_pct = 80.21%`
  - `ekf_vcd_rmse = 1.76`, `ekf_dropout_steps = 38`, `ekf_nans = 0`, `min_x = 0.0`

---

## Step 6 — Uncertainty Quantification (UQ) & Error Propagation

1. **Parametric Epistemic UQ:** Output-space `B=15` ensemble dispersion (`ddof=1`) combined with calibrated floor uncertainty (`std_kd`, `std_kla`, `std_qp`, `std_qc`, `std_mu`) mapped into process covariance `Q_ep = G * Sigma_nn * G^T` inside `HybridExtendedKalmanFilter._epistemic_process_cov`.
2. **State Covariance Propagation:** Continuous-discrete Van Loan propagation `P_{k|k-1} = Phi * P_{k-1} * Phi^T + Q_d` + Joseph-form measurement update `P = (I - KH) P (I - KH)^T + K R K^T`.
3. **Domain-of-Validity & Innovation Monitoring:** Dual gating combines subspace Mahalanobis distance `d_m` (`kd`, `kla`, `met`, `global`) with dynamic degrees-of-freedom (`df=2` when `y_P=None`, `df=3` when `y_P` or PAT) chi-square `NIS` (`alpha = 0.999`) and persistent `SPE` thresholds (`spe_persist_steps = 2`).

---

## Step 7 — Adequacy Assessment & Explicit Operational Boundaries

- **Adequacy Verdict:** Credible **within validated envelope** for advisory soft-sensing, delayed-LIMS replay, *in-silico* scale-up studies (`5 L` -> `1000 L`), and human-gated deviation drafting.
- **Explicit Operational Boundaries:**
  1. **Organism / Kinetics Scope:** Synthetic Stage-1/Stage-2 twin is calibrated for *E. coli* BL21 fed-batch kinetics; external empirical stress validation (`8.61 / 10.00 -> GO`) demonstrated structural generalization on CHO-K1 batch/fed-batch data (`Uhlendorff et al., 2026`), **not** on in-house Eli Lilly cell lines or manufacturing batches.
  2. **OOD & Residual Envelope:** Whenever `in_domain == False` (`undecided == True`), ML residual corrections relax toward mechanistic priors and `ClosedLoopFeedController` mandates ML-free `safe_nominal` or `hold` fallback plus `CONTROL_ESCALATION` audit logging.
  3. **Physical Range Boundaries Enforced at API/ETL:** `t_h in [0, 500] h`, `X_v in [0, 200] g/L`, `C_L in [0, 25] mg/L`, `V in (0, 50000] L`, `rpm in [50, 1200]`, `vvm in [0.01, 5.0]`.
  4. **Human Authority Boundary:** Neither MCP tools (`stage_capa_draft`) nor LangGraph workflows may finalize QA records or push live actuator setpoints; human QA 2FA approval (`agents/part11.py`) is mandatory.
