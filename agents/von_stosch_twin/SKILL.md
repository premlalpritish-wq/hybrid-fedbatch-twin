---
name: von-stosch-twin
description: Reusable hybrid bioprocess modeling sub-agent implementing von Stosch's serial hybrid ODE, forward sensitivity equations, Willis & von Stosch (2017) sparse structure discrimination, and Pinto et al. (2019) bootstrap-aggregated calibration.
---

# Von Stosch Hybrid Digital Twin Sub-Agent

## Invocation Rules
1. **Pure NumPy / SciPy Only**: Never import PyTorch, TensorFlow, JAX, or scikit-learn. Use `agents.von_stosch_twin.core_math` (`SerialHybridODE`, `SparseRateDiscriminator`, `BaggedHybridTrainer`).
2. **Bounded Sensitivity Optimization**: Always bound neural network weights in `L-BFGS-B` (`param_bounds=(-3.0, 3.0)`) and cap `maxiter=25` per bootstrap candidate (`b_cand=12`, `b_ens=8`) with `RK45` (`rtol=1e-3, atol=1e-4`) and automatic finite-difference sensitivity fallback if coupled integration exceeds `0.1s` per batch.
3. **Unbuffered Streaming**: Always pass `flush=True` to `print()` statements and run verification via `python -u -m pytest agents/von_stosch_twin/test_calibration.py -s`.

## The 5 Von Stosch Verification Gates
1. **Gate 1 — White-Box Macroscopic Conservation**: Reactor volume `dV/dt = F_in` and dilution `D = F_in / V` are strictly white-box with zero trainable weights.
2. **Gate 2 — Monod Positivity Guard**: Specific substrate uptake `q_S` is multiplied by `S / (S + 0.05)` so `min(S_pred) >= 0.0` across all rollouts.
3. **Gate 3 — Sparse Structure Recovery & Identifiability (Willis & von Stosch, 2017)**: Forward OLS-ERR with BIC stopping (`BIC = N * ln(RSS/N) + k * ln(N)`) must recover the true kinetic structure with zero decoy terms and collinearity index `gamma_col < 20`.
4. **Gate 4 — Batch-Level Bagging & OOB Early Stopping (Pinto et al., 2019)**: Resample at the batch/campaign level (~67% train with replacement, ~33% OOB validation), restore weights at minimum OOB WMSE, rank by `0.5 * train_WMSE + 0.5 * val_WMSE`, and verify ensemble aggregation reduces Test WMSE vs single-model (`n=1`) worst/median WMSE.
5. **Gate 5 — Epistemic Uncertainty & Sensitivity Diagnostics**: Report 95% Epistemic Credible Interval coverage (`mean ± 1.96 * std`) on unseen DoE test batches and top-1 model collinearity index `gamma_col`.
