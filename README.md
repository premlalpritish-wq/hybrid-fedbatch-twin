# Hybrid Mechanistic-ML Bioreactor Digital Twin (`hybrid-fedbatch-twin`)

**Live Interactive 3D Case Study & Browser Digital Twin:** [https://premlalpritish-wq.github.io/hybrid-fedbatch-twin/](https://premlalpritish-wq.github.io/hybrid-fedbatch-twin/)

A grey-box bioprocess digital twin for recombinant *E. coli* BL21 fed-batch fermentation combining deterministic macroscopic mass/volume balances, bounded 6-parameter iDoE calibration, a `B=15` bootstrap-aggregated `GAM-86` serial rate estimator, a Continuous-Discrete Extended Kalman Filter (`expm` + `Q_epistemic`), and a closed-loop `mu-stat` feed controller.

## Key Capabilities & Verified Metrics

- **Portela Level 3+ Audit Rating:** Remediated across an adversarial two-iteration engineering loop from `2.6 / 10 (NO-GO)` (Iteration 0) to `6.4 / 10 (NO-GO)` (Iteration 1) to `9.2 / 10 (GO)` (Iteration 2). See [`AUDIT_REPORT_VON_STOSCH.md`](AUDIT_REPORT_VON_STOSCH.md).
- **Stage-1 Parameter Recovery (14-Seed iDoE):** Recovers 6 mechanistic parameters (`mu_max`, `K_S`, `Y_XS`, `alpha`, `beta`, `k_La0`) via bounded Trust-Region Reflective `least_squares` with `kappa_rel = 38.18`, seed-42 max error `4.39%`, and 14-seed mean max parameter error `1.59%`.
- **Stage-2 Serial Hybrid (`GAM-86`, `B=15` Bootstrap Ensemble):** Trained via reverse-mode discrete adjoint VJP over Patankar positivity-preserving integration using measurable PAT channels (`X_v`, `S`, `C_L`, `A`, `P`, `V`, zero `X_d` leakage). Pooled 5-regime held-out accuracy: `kd R^2 = 0.7334`, `k_La R^2 = 0.9699`, `q_A,net R^2 = 0.9010`, `mu R^2 = 0.9951`.
- **Scale Transfer (`5 L` -> `1000 L`):** Uses scale-invariant hydrodynamic features `(P_g/V, v_tip, v_s)` calibrated at `5 L` (`134.8 RPM`, `0.205 vvm` matched operating point at `1000 L`), achieving `0.05529` hybrid normalized RMSE vs `0.34112` for the nominal mechanistic model.
- **Zero-Backend Interactive 3D Web Twin (`docs/`):** Exports the trained `GAM-86` (`B=15`) ensemble (`docs/assets/twin_bundle.js`, `docs/assets/twin_bundle.json`) and runs Patankar integration, `7x7` Padé(6,6) `expm` Continuous-Discrete EKF, Joseph covariance update, and Three.js bioreactor visualization 100% client-side in any standard browser.

## Repository Layout

- `hybrid_fedbatch_twin/mechanistic.py`: 7-state ODE plant (`X_v`, `X_d`, `S`, `C_L`, `A`, `P`, `V`), geometry/hydrodynamic correlations, non-Newtonian viscosity damping on `k_L a`, dynamic cell death `k_d`, and overflow acetate kinetics.
- `hybrid_fedbatch_twin/neural_ode.py`: Stage-1 6-parameter iDoE optimizer, Stage-2 `GAM-86` bootstrap ensemble, exact VJP adjoint gradient, AIC/BIC selection, and Subspace Domain of Validity (`DoV`).
- `hybrid_fedbatch_twin/estimator.py`: Continuous-Discrete EKF (`scipy.linalg.expm` + `Q_epistemic`), blind fault detection, and ALCOA+ SHA-256 audit serialization (`BioreactorDigitalTwin.to_audit_record`).
- `hybrid_fedbatch_twin/controller.py`: Closed-loop `mu-stat` feedforward + PI feed controller with oxygen transfer ceiling and acetate overflow backoff.
- `hybrid_fedbatch_twin/cli.py`: Reproducible CLI benchmark and JSON report exporter.
- `app.py`: Streamlit Operator (`5 L`) and Validation Benchmark (`1000 L`) console.
- `scripts/export_web_twin_assets.py`: Exports verified Stage-1, Stage-2, disturbance, AIC/BIC, and `1000 L` benchmark artifacts into `docs/assets/`.
- `docs/`: Zero-backend Three.js + Canvas interactive 3D digital twin and engineering case-study site deployed via GitHub Pages.
- `tests/test_hybrid_twin.py`: Automated `8/8` verification suite.
- `agents/moritz_von_stosch_auditor.md`: Literature-grounded adversarial bioprocess audit persona specification.

## Quickstart (Local Python)

```powershell
# 1. Install dependencies
pip install -e .

# 2. Run the 8-test verification suite (use single-threaded BLAS on Windows)
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
pytest -v

# 3. Run the CLI calibration, EKF, and controller benchmark
python -m hybrid_fedbatch_twin.cli --mode all --export-json results_summary.json

# 4. Launch the interactive Streamlit console
streamlit run app.py

# 5. Re-export static web twin weights and benchmarks for docs/
python -u scripts/export_web_twin_assets.py
```

## Notes & Disclaimers

- **Industrial Motivation:** The problem framing (grey-box fed-batch modeling across scales, PAT sensor fusion, and closed-loop `mu-stat` control) draws inspiration from public upstream cell-culture and fermentation modeling workflows (`R-108742`). All kinetics, disturbances, and datasets in this repository are synthetic simulations.
- **Audit Persona Disclaimer:** The `moritz_von_stosch_auditor` agent specification in `agents/moritz_von_stosch_auditor.md` is an independent, literature-grounded AI review persona inspired by published hybrid modeling papers by Dr. Moritz von Stosch and co-authors. It does not represent an official review or endorsement by Dr. von Stosch.
