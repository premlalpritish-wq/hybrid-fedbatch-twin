# ICH Q14 & ICH Q12 Lifecycle Governance Framework

**System:** `hybrid-fedbatch-twin`  
**Regulatory Alignment:** ICH Q14 (*Analytical Procedure Development*, adapted for PAT Soft-Sensor & Hybrid Model Development) + ICH Q12 (*Technical and Regulatory Considerations for Pharmaceutical Product Lifecycle Management* — Established Conditions & Post-Approval Change Management Protocol / PACMP).

---

## 1. Analytical Target Profile (ATP) / Process Target Profile (PTP)

| Target Dimension | Specified Requirement / Target Performance | Code & Artifact Verification |
| --- | --- | --- |
| **Biomass & State Soft-Sensing** | Track viable biomass `X_v`, unmeasured dead biomass `X_d`, glucose `S`, DO `C_L`, acetate `A`, product `P`, and volume `V` without negative excursions (`min_x >= 0.0`) or `NaN` EKF states (`ekf_nans == 0`). | `hybrid_fedbatch_twin/estimator.py`, `artifacts/uhlendorff_2026_validation.json` |
| **Rate Estimation Accuracy (In-Silico Held-Out)** | `kd_r2 >= 0.70` (verified `0.7320`), `kla_r2 >= 0.90` (verified `0.9806`), `mu_r2 >= 0.95` (verified `0.9925`), `qAnet_r2 >= 0.85` (verified `0.8803`). | `results_summary.json`, `mlops/registry.py` (`PromotionCriteria`) |
| **External Empirical Stress Benchmark** | CHO-K1 external validation composite score `>= 8.0 / 10.0` with `GO` verdict (`8.61 / 10.00 -> GO (PASS)`), `ci_coverage_pct` verified (`80.21%`), zero EKF `NaN`s. | `stress_test_uhlendorff.py`, `artifacts/uhlendorff_2026_validation.json` |
| **Supervisory Control & Fault Safety** | Maintain growth tracking (`mu_target`) below DO/acetate safety ceilings; revert strictly to ML-free fallback (`_compute_safe_fallback_feed`) within 1 control tick on OOD or `NIS`/`SPE` breach. | `hybrid_fedbatch_twin/controller.py` |
| **Data Integrity & Auditability** | 100% SHA-256 hash-chained ALCOA+ event logging (`prev_entry_sha256` -> `entry_sha256`), zero silent row drops in ETL/adapters, and 4-part Part 11 records for AI CAPA decisions. | `api/adapters.py`, `api/db.py`, `mlops/medallion.py`, `agents/part11.py` |

---

## 2. Method Operable Design Region (MODR) & Domain of Validity

- **Geometric Scale Envelope:** Calibrated at `5 L` and validated for intensive hydrodynamic transfer across `5 L`, `50 L`, and `1000 L` (`compute_hydrodynamics`, `scale_params`), using intensive invariants `(P_g/V, v_tip, v_s)` so vessel volume `V` never leaks directly into ML rate heads.
- **Data-Driven MODR (Subspace Mahalanobis DoV):** `PhysicsInformedNeuralODE.dov_stats` enforces empirical 99th-percentile Mahalanobis boundaries (`dov_p99 = 5.0118` global) across `kd`, `kla`, `met`, and `global` feature subspaces.
- **Innovation MODR (Dynamic EKF Residuals):** `HybridExtendedKalmanFilter` monitors `NIS` (`chi2.ppf(0.999, df=obs_dim)`) and `SPE` (`10 * chi2` single-step, `6.8 * chi2` across `spe_persist_steps = 2`). Exiting either DoV or `NIS`/`SPE` exits the MODR (`in_domain = False`, `undecided = True`).

---

## 3. ICH Q12 Established Conditions (ECs)

### Core ECs (Require Formal PACMP + Human QA 2FA Sign-Off to Modify)
1. **Mechanistic Stoichiometry & Balances:** `BioreactorParams` defaults, overflow carbon/oxygen coupling, and `bioreactor_rhs` differential structure (`hybrid_fedbatch_twin/mechanistic.py`).
2. **Model Architecture & Calibration Integrity:** `GAM-86` (`n_params = 86`, `n_bootstrap = 15`), `params_sha256`, `weights_sha256`, and `dov_stats` (`docs/assets/twin_bundle.json`).
3. **EKF Discretization & Safety Gating Logic:** Van Loan `_discretize_linear_gaussian`, dual Mahalanobis + `NIS`/`SPE` thresholds (`gate_alpha = 0.999`, `spe_persist_steps = 2`), and ML-free fallback equations (`ClosedLoopFeedController._compute_safe_fallback_feed`).
4. **Governance & Security Boundaries:** `MCP_TOOL_POLICY` (`read_only` / `draft_staging` only), deterministic guardrail checks (`validate_capa_draft`), `ServerEvidenceContextRow` enforcement, single-use 2FA e-signatures (`UsedTwoFactorChallengeRow`), and truth isolation (`show_truth=False`).

### Non-EC / Operational Configuration Parameters (Routine Change Control Within Verified Ranges)
- API logging/telemetry scrape intervals (`GET /metrics` via `mlops/telemetry.py`),
- Batch identifiers, operator IDs, and `?max_entries` audit export slice size,
- Re-usable declarative `DataQualityPolicy` thresholds strictly tighter than physical ceilings (`mlops/medallion.py`).

---

## 4. Offline-Retraining Post-Approval Change Management Protocol (PACMP)

> [!CAUTION]
> **Prohibition on Autonomous Online GMP Weight Updates:** Online self-updating or autonomous reinforcement-learning updates to `GAM-86` weights or `BioreactorParams` in production are strictly prohibited. All model retraining occurs **offline** under this staged PACMP.

1. **Stage 1 — Bronze -> Silver -> Gold Data Curation (`mlops/medallion.py`):**
   - Ingest raw OT/LIMS batch histories to immutable Bronze JSONL (`ingest_bronze`).
   - Execute declarative `DataQualityPolicy` gates (`promote_bronze_to_silver`), quarantining anomalous/non-monotonic/oracle-leaking rows with explicit reason codes and preserving `silver + quarantine == bronze`.
   - Build Gold features (`build_gold_features`) with strict batch-grouped train/hold-out separation (`train_batches & holdout_batches == empty`) and SHA-256 manifest verification (`verify_medallion_lineage`).
2. **Stage 2 — Offline Re-Calibration & Quantitative Gate Evaluation (`mlops/registry.py`):**
   - Run `run_offline_training_and_registration_flow` linking `train_split_sha256`, `holdout_split_sha256`, `params_sha256`, and `weights_sha256`.
   - Evaluate candidate against `PromotionCriteria` (`kd_r2 >= 0.70`, `kla_r2 >= 0.90`, `mu_r2 >= 0.95`, `uhlendorff_score >= 8.0` + `GO`, `ekf_nans == 0`, `min_x >= 0.0`, valid lineage, zero non-finite hold-out outputs).
   - Failing candidates transition to `stage="REJECTED"` with explicit `failed_criteria`; passing candidates transition ONLY to **`stage="PROMOTED_STAGING"`** inside the SHA-256 hash-chained `registry_ledger.jsonl`.
3. **Stage 3 — Shadow Verification & Human QA 2FA Release (`agents/shadow_eval.py`, `agents/part11.py`):**
   - Execute `run_shadow_evaluation_suite` and `verify_repo_hygiene.py` + `generate_rtm.py --check` + `pytest` suites offline.
   - Require human `QA_APPROVER` / `QA_LEAD` / `QP_RELEASE` 2-factor electronic signature (`Part11AiDecisionRecord`) before releasing a `PROMOTED_STAGING` bundle to a production deployment artifact (`docs/assets/twin_bundle.json`).
4. **Stage 4 — Post-Deployment Continuous Process Verification (`mlops/telemetry.py`):**
   - Monitor `twin_ekf_nis`, `twin_ekf_spe`, `twin_ekf_mahalanobis_dm`, `twin_control_fallback_total`, `twin_medallion_quarantined_rows_total`, and `twin_agent_guardrail_rejections_total` via `/metrics` and OTel spans to trigger PACMP re-evaluation if process drift elevates fallback rates.
