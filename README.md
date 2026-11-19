# Hybrid Mechanistic-ML Bioreactor Digital Twin (`hybrid-fedbatch-twin`)

**Live Interactive 3D Case Study & Browser Digital Twin:** [https://premlalpritish-wq.github.io/hybrid-fedbatch-twin/](https://premlalpritish-wq.github.io/hybrid-fedbatch-twin/)

A grey-box bioprocess digital twin for recombinant *E. coli* BL21 fed-batch fermentation combining deterministic macroscopic mass/volume balances, bounded 6-parameter iDoE calibration, a `B=15` bootstrap-aggregated `GAM-86` serial rate estimator, a Continuous-Discrete Extended Kalman Filter (`expm` + `Q_epistemic`), and a closed-loop `mu-stat` feed controller.

## Key Capabilities & Verified Metrics

- **Digital Twin Maturity Classification (Portela et al., 2021):** **Portela Level 1 Offline Digital Model / Level 2-Ready Soft-Sensor Architecture** (*in-silico* plant harness and FastAPI/MCP service; true Level 3 requires validated GMP OT/PLC closed-loop integration). Remediated via adversarial engineering verification; see [`AUDIT_REPORT_VON_STOSCH.md`](AUDIT_REPORT_VON_STOSCH.md).
- **Stage-1 Parameter Recovery (14-Seed iDoE):** Recovers 6 mechanistic parameters (`mu_max`, `K_S`, `Y_XS`, `alpha`, `beta`, `k_La0`) via bounded Trust-Region Reflective `least_squares` with `kappa_rel = 38.18`, seed-42 max error `4.39%`, and 14-seed mean max parameter error `1.59%`.
- **Stage-2 Serial Hybrid (`GAM-86`, `B=15` Bootstrap Ensemble):** Trained via reverse-mode discrete adjoint VJP over Patankar positivity-preserving integration using measurable PAT channels (`X_v`, `S`, `C_L`, `A`, `P`, `V`, zero `X_d` leakage). Pooled 5-regime held-out accuracy: `kd R^2 = 0.7320`, `k_La R^2 = 0.9806`, `q_A,net R^2 = 0.8803`, `mu R^2 = 0.9925`.
- **Scale Transfer (`5 L` -> `1000 L`):** Uses scale-invariant hydrodynamic features `(P_g/V, v_tip, v_s)` calibrated at `5 L` (`134.8 RPM`, `0.205 vvm` matched operating point at `1000 L`), achieving `0.03687` hybrid normalized RMSE vs `0.39081` for the nominal mechanistic model.
- **Zero-Backend Interactive 3D Web Twin (`docs/`):** Exports the trained `GAM-86` (`B=15`) ensemble (`docs/assets/twin_bundle.js`, `docs/assets/twin_bundle.json`) and runs Patankar integration, `7x7` Padé(6,6) `expm` Continuous-Discrete EKF, Joseph covariance update, and Three.js bioreactor visualization 100% client-side in any standard browser.

## Repository Layout

- `hybrid_fedbatch_twin/mechanistic.py`: 7-state ODE plant (`X_v`, `X_d`, `S`, `C_L`, `A`, `P`, `V`), geometry/hydrodynamic correlations, non-Newtonian viscosity damping on `k_L a`, dynamic cell death `k_d`, and overflow acetate kinetics.
- `hybrid_fedbatch_twin/neural_ode.py`: Stage-1 6-parameter iDoE optimizer, Stage-2 `GAM-86` bootstrap ensemble, exact VJP adjoint gradient, AIC/BIC selection, and Subspace Domain of Validity (`DoV`).
- `hybrid_fedbatch_twin/estimator.py`: Continuous-Discrete EKF (`scipy.linalg.expm` + `Q_epistemic`), blind fault detection, and ALCOA+ SHA-256 audit serialization (`BioreactorDigitalTwin.to_audit_record`).
- `hybrid_fedbatch_twin/controller.py`: Closed-loop `mu-stat` feedforward + PI feed controller with oxygen transfer ceiling and acetate overflow backoff.
- `hybrid_fedbatch_twin/cli.py`: Reproducible CLI benchmark and JSON report exporter.
- `api/`: Production FastAPI + Pydantic v2 Digital Twin service (`api/main.py`, `api/schemas.py`, `api/runtime.py`) backed by SQLAlchemy ALCOA+ persistence (`api/db.py`, configurable via `DATABASE_URL` for PostgreSQL or local SQLite). Exposes twin initialization (`5 L`/`50 L`/`1000 L`), online PAT ingestion, delayed LIMS replay, closed-loop control, ALCOA+ SHA-256 audit retrieval, and CAPA deviation investigation (`POST /twins/{session_id}/investigate`).
- `mcp_server/server.py`: Model Context Protocol (`MCPServer`) server exposing deterministic bioprocess tools (`simulate_fedbatch_run`, `ingest_pat_and_estimate_state`, `replay_delayed_lims_assay`, `search_uhlendorff_campaigns`) so LLM agents delegate all ODE, EKF, and CHO-K1 empirical lookups to verified solvers.
- `agents/deviation_graph.py` & `agents/observability.py`: 4-step stateful Bioprocess Deviation & CAPA Investigator graph (`detect_excursion` -> `run_counterfactual_controls` -> `retrieve_empirical_precedent` -> `synthesize_capa_report`) with structured node/tool latency tracing and deterministic faithfulness evaluation (`evaluate_capa_report`).
- `app.py`: Streamlit Operator (`5 L`) and Validation Benchmark (`1000 L`) console.
- `scripts/export_web_twin_assets.py` & `scripts/verify_repo_hygiene.py`: Export verified Stage-1, Stage-2, disturbance, AIC/BIC, and `1000 L` benchmark artifacts into `docs/assets/` and verify `.gitignore`, bundle SHA-256, Uhlendorff JSON/JS parity, and ALCOA+ provenance consistency.
- `docs/`: Zero-backend Three.js + Canvas interactive 3D digital twin and Distill-style engineering monograph deployed via GitHub Pages.
- `tests/`: Automated verification harness partitioned by pytest markers (`hygiene`, `unit`, `contract`, `agent`, `slow`) across `tests/test_repo_hygiene.py`, `tests/test_contracts_and_roundtrips.py`, `tests/test_service_and_mcp.py`, `tests/test_agent_evals.py`, and `tests/test_hybrid_twin.py`.
- `Dockerfile` & `.github/workflows/ci.yml`: Reproducible container build and automated CI verification (fast hygiene + `not slow` suite on every push/PR; full `slow` + Von Stosch benchmark suite via `workflow_dispatch`).
- `agents/moritz_von_stosch_auditor.md`: Literature-grounded adversarial bioprocess audit persona specification.

## Quickstart (Local Python, API, MCP & Docker)

```powershell
# 1. Install core + service dependencies
pip install -e ".[service,dev]"

# 2. Run repository hygiene & artifact consistency check
python scripts/verify_repo_hygiene.py

# 3. Run fast CI check suite (hygiene + unit + contract + agent evals, skips heavy training)
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
pytest -m "not slow" -v

# Targeted fast marker subsets:
pytest -m hygiene -v
pytest -m contract -v

# 4. Run full scientific training/simulation/benchmark suite (multi-minute)
pytest -m slow -v
pytest -v
pytest agents/von_stosch_twin/test_calibration.py -v

# 4. Launch the FastAPI + ALCOA+ Digital Twin service
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Run the MCP server (stdio)
python -m mcp_server.server

# 6. Or build and run via Docker
docker build -t hybrid-fedbatch-twin .
docker run -p 8000:8000 hybrid-fedbatch-twin

# 7. Run the CLI benchmark or Streamlit console
python -m hybrid_fedbatch_twin.cli --mode all --export-json results_summary.json
streamlit run app.py
```

## Notes & Disclaimers

- **Industrial Motivation:** The problem framing (grey-box fed-batch modeling across scales, PAT sensor fusion, and closed-loop `mu-stat` control) draws inspiration from public upstream cell-culture and fermentation modeling workflows (`R-108742`). All kinetics, disturbances, and datasets in this repository are synthetic simulations.
- **Audit Persona Disclaimer:** The `moritz_von_stosch_auditor` agent specification in `agents/moritz_von_stosch_auditor.md` is an independent, literature-grounded AI review persona inspired by published hybrid modeling papers by Dr. Moritz von Stosch and co-authors. It does not represent an official review or endorsement by Dr. von Stosch.
