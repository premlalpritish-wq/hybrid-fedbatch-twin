"""
auditor.py

Automated verification and benchmarking runner for the Von Stosch Digital Twin
sub-agent, enforcing the 5 Von Stosch Verification Gates across:
  - Benchmark 1: Willis & von Stosch (2017) Sparse Structure Recovery
  - Benchmark 2: Pinto, de Azevedo, Oliveira, & von Stosch (2019) Fed-Batch DoE Bagging
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from agents.von_stosch_twin.core_math import (
    BaggedHybridTrainer,
    BatchData,
    SerialHybridODE,
    SparseRateDiscriminator,
    compute_collinearity_index,
)


def run_benchmark_willis_2017(seed: int = 2017) -> Dict[str, object]:
    """Benchmark 1: Willis & von Stosch (2017) Sparse Structure Recovery."""
    print("=" * 72, flush=True)
    print("BENCHMARK 1: Willis & von Stosch (2017) Sparse Structure Recovery", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)
    n_samples = 200
    s_vals = rng.uniform(0.05, 20.0, size=n_samples)
    p_vals = rng.uniform(0.0, 30.0, size=n_samples)

    mu_true = 0.45 * (s_vals / (s_vals + 0.5)) * (1.0 / (1.0 + p_vals / 15.0))
    mu_obs = mu_true * (1.0 + rng.normal(0.0, 0.05, size=n_samples))

    term_names = [
        "1",
        "S/(S+0.5)",
        "1/(1+P/15)",
        "(S/(S+0.5))/(1+P/15)",
        "P/(P+5)",
        "decoy_1",
        "decoy_2",
        "decoy_3",
    ]
    phi = np.column_stack(
        [
            np.ones(n_samples),
            s_vals / (s_vals + 0.5),
            1.0 / (1.0 + p_vals / 15.0),
            (s_vals / (s_vals + 0.5)) / (1.0 + p_vals / 15.0),
            p_vals / (p_vals + 5.0),
            rng.normal(0.0, 1.0, size=n_samples),
            rng.normal(0.0, 1.0, size=n_samples),
            rng.normal(0.0, 1.0, size=n_samples),
        ]
    )

    disc = SparseRateDiscriminator().fit(phi, mu_obs, term_names=term_names)

    print(
        f"Observations N = {n_samples}, Candidate basis terms = {len(term_names)}",
        flush=True,
    )
    print("\nSelected basis terms (in order of ERR):", flush=True)
    for step, (name, err, bic) in enumerate(
        zip(disc.selected_names_, disc.err_history_, disc.bic_history_), start=1
    ):
        coef = disc.coef_[step - 1] if disc.coef_ is not None else float("nan")
        print(
            f"  Step {step}: {name:<24s} | ERR = {err:.6f} | BIC = {bic:10.4f} | coef = {coef:.6f}",
            flush=True,
        )

    if len(disc.bic_history_) > len(disc.selected_names_):
        print(
            f"  Step {len(disc.selected_names_) + 1}: [STOPPED]                | "
            f"BIC = {disc.bic_history_[-1]:10.4f} (did not improve)",
            flush=True,
        )

    print(f"\nBIC progression: {[round(b, 4) for b in disc.bic_history_]}", flush=True)
    print(f"Collinearity index gamma_col = {disc.gamma_col_:.4f}", flush=True)

    has_true = "(S/(S+0.5))/(1+P/15)" in disc.selected_names_
    n_decoys = sum(1 for name in disc.selected_names_ if name.startswith("decoy_"))
    exact_match = disc.selected_names_ == ["(S/(S+0.5))/(1+P/15)"]
    identifiable = disc.gamma_col_ < 20.0
    passed = bool(has_true and exact_match and n_decoys == 0 and identifiable)

    print(
        f"Gate 3 Check: true_term={has_true}, exact_recovery={exact_match}, "
        f"decoys={n_decoys}, gamma_col<20={identifiable} -> {'PASS' if passed else 'FAIL'}\n",
        flush=True,
    )
    return {
        "passed": passed,
        "selected_names": disc.selected_names_,
        "err_history": disc.err_history_,
        "bic_history": disc.bic_history_,
        "gamma_col": disc.gamma_col_,
        "n_decoys": n_decoys,
    }


def _ground_truth_rhs(t: float, x: np.ndarray, f_in: float, c_s_feed: float) -> np.ndarray:
    """Ground-truth fed-batch E. coli ODE for Benchmark 2."""
    x_bio, s_conc, p_conc, vol = x
    s_pos = max(s_conc, 0.0)
    p_pos = max(p_conc, 0.0)

    mu = 0.35 * (s_pos / (s_pos + 0.4)) * (1.0 / (1.0 + p_pos / 20.0))
    q_s = -(mu / 0.50 + 0.02 * (s_pos / (s_pos + 0.1)))
    q_p = 0.30 * mu + 0.015 * (s_pos / (s_pos + 0.4))

    d_rate = f_in / max(vol, 1e-8)
    dx = mu * x_bio - d_rate * x_bio
    ds = q_s * x_bio + d_rate * (c_s_feed - s_conc)
    dp = q_p * x_bio - d_rate * p_conc
    dv = f_in
    return np.array([dx, ds, dp, dv], dtype=float)


def _simulate_ground_truth_batch(
    s0: float,
    f_feed: float,
    rng: np.random.Generator,
    sigma_y: np.ndarray,
    name: str,
) -> BatchData:
    """Generate a single synthetic fed-batch dataset with noiseless truth and noisy obs."""
    t_eval = np.linspace(0.0, 24.0, 13)
    t_feed_start = 4.0
    c_s_feed = 100.0
    x0 = np.array([0.2, s0, 0.0, 1.0], dtype=float)

    mask1 = t_eval <= t_feed_start
    mask2 = t_eval > t_feed_start
    sol1 = solve_ivp(
        lambda t, x: _ground_truth_rhs(t, x, 0.0, c_s_feed),
        (0.0, t_feed_start),
        x0,
        method="BDF",
        t_eval=t_eval[mask1],
        rtol=1e-7,
        atol=1e-9,
    )
    sol2 = solve_ivp(
        lambda t, x: _ground_truth_rhs(t, x, f_feed, c_s_feed),
        (t_feed_start, 24.0),
        sol1.y[:, -1],
        method="BDF",
        t_eval=t_eval[mask2],
        rtol=1e-7,
        atol=1e-9,
    )
    x_true = np.vstack([sol1.y.T, sol2.y.T])
    y_true = x_true[:, :3]
    y_obs = np.maximum(y_true + rng.normal(0.0, sigma_y[None, :], size=y_true.shape), 0.0)

    return BatchData(
        t_eval=t_eval,
        y_obs=y_obs,
        x0=x0,
        f_feed=f_feed,
        t_feed_start=t_feed_start,
        c_s_feed=c_s_feed,
        y_true=y_true,
        name=name,
    )


def run_benchmark_pinto_2019(seed: int = 2019) -> Dict[str, object]:
    """Benchmark 2: Pinto & von Stosch (2019) Fed-Batch DoE Bagging Benchmark."""
    print("=" * 72, flush=True)
    print("BENCHMARK 2: Pinto & von Stosch (2019) Fed-Batch DoE Bagging Benchmark", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)
    sigma_y = np.array([0.15, 0.25, 0.10], dtype=float)

    dev_batches: List[BatchData] = []
    for s0 in [5.0, 10.0, 15.0]:
        for f_in in [0.02, 0.05, 0.08]:
            dev_batches.append(
                _simulate_ground_truth_batch(
                    s0, f_in, rng, sigma_y, name=f"Dev(S0={s0},F={f_in})"
                )
            )

    test_specs: List[Tuple[float, float]] = [
        (7.5, 0.035),
        (12.5, 0.065),
        (16.0, 0.085),
        (4.5, 0.015),
    ]
    test_batches: List[BatchData] = [
        _simulate_ground_truth_batch(
            s0, f_in, rng, sigma_y, name=f"Test(S0={s0},F={f_in})"
        )
        for s0, f_in in test_specs
    ]

    all_dev_y = np.vstack([b.y_obs for b in dev_batches])
    x_mean = np.mean(all_dev_y, axis=0)
    x_std = np.std(all_dev_y, axis=0)

    ode_model = SerialHybridODE(
        n_hidden=2,
        n_states=4,
        x_mean=x_mean,
        x_std=x_std,
        sigma_y=sigma_y,
        monod_guard_k=0.05,
        dt=0.2,
    )
    trainer = BaggedHybridTrainer(
        ode_model=ode_model,
        b_cand=12,
        b_ens=8,
        maxiter=25,
        l2_reg=0.0,
        oob_outlier_factor=2.5,
        seed=seed,
        verbose=True,
    )

    t_start = time.perf_counter()
    trainer.fit(dev_batches)
    elapsed = time.perf_counter() - t_start
    print(
        f"Trained B_cand = {trainer.b_cand} candidates (top B_ens = {trainer.b_ens}) "
        f"with H = {ode_model.n_hidden} hidden nodes in {elapsed:.2f}s",
        flush=True,
    )

    preds = trainer.predict_ensemble(test_batches, n_models=trainer.b_ens)
    per_model_trajs = preds["per_model_trajs"]
    mean_trajs = preds["mean_traj"]
    std_trajs = preds["std_model_traj"]
    wmse_obs_by_n = preds["wmse_by_n"]
    wmse_true_by_n = preds["wmse_true_by_n"]
    min_s_pred = float(preds["min_substrate"])

    single_model_obs_wmses = []
    single_model_true_wmses = []
    weights_y = ode_model.weights_y
    for m_idx in range(len(trainer.ensemble_)):
        sq_obs = 0.0
        sq_true = 0.0
        n_obs = 0
        for b_idx, batch in enumerate(test_batches):
            pred_y = per_model_trajs[m_idx][b_idx][:, :3]
            err_obs = pred_y - batch.y_obs
            err_true = pred_y - batch.y_true
            sq_obs += float(np.sum((err_obs ** 2) * weights_y[None, :]))
            sq_true += float(np.sum((err_true ** 2) * weights_y[None, :]))
            n_obs += err_obs.size
        single_model_obs_wmses.append(sq_obs / n_obs)
        single_model_true_wmses.append(sq_true / n_obs)

    covered = 0
    total_pts = 0
    for b_idx, batch in enumerate(test_batches):
        mu_pred = mean_trajs[b_idx][:, :3]
        sd_pred = std_trajs[b_idx][:, :3]
        half_width = 1.96 * sd_pred
        in_ci = np.abs(batch.y_true - mu_pred) <= half_width
        covered += int(np.sum(in_ci))
        total_pts += in_ci.size
    coverage_pct = 100.0 * covered / total_pts

    top1_theta = trainer.ensemble_[0].theta
    s_mat = ode_model.compute_sensitivity_matrix(top1_theta, dev_batches)
    gamma_col_top1 = compute_collinearity_index(s_mat)

    print("\n--- Benchmark 2 Summary Table ---", flush=True)
    print(
        f"{'Metric':<44s} | {'Noisy Obs WMSE':>14s} | {'True State WMSE':>15s}",
        flush=True,
    )
    print("-" * 79, flush=True)
    print(
        f"{'Single-Model (n=1) Median across B_ens':<44s} | "
        f"{np.median(single_model_obs_wmses):14.4f} | "
        f"{np.median(single_model_true_wmses):15.4f}",
        flush=True,
    )
    print(
        f"{'Single-Model (n=1) Worst across B_ens':<44s} | "
        f"{np.max(single_model_obs_wmses):14.4f} | "
        f"{np.max(single_model_true_wmses):15.4f}",
        flush=True,
    )
    for n_agg in [1, 2, 5, 8]:
        print(
            f"{f'Bagged Ensemble (n={n_agg})':<44s} | "
            f"{wmse_obs_by_n[n_agg]:14.4f} | "
            f"{wmse_true_by_n[n_agg]:15.4f}",
            flush=True,
        )
    print("-" * 79, flush=True)
    print(
        f"95% Epistemic Credible Interval Coverage (vs True): {coverage_pct:.2f}% ({covered}/{total_pts})",
        flush=True,
    )
    print(
        f"Minimum Predicted Substrate min(S_pred)         : {min_s_pred:.6f} g/L (guard >= 0: {min_s_pred >= 0.0})",
        flush=True,
    )
    print(f"Top-1 Model Collinearity Index gamma_col        : {gamma_col_top1:.4f}", flush=True)
    print("=" * 72, flush=True)

    return {
        "elapsed_s": elapsed,
        "single_median_obs": float(np.median(single_model_obs_wmses)),
        "single_worst_obs": float(np.max(single_model_obs_wmses)),
        "single_median_true": float(np.median(single_model_true_wmses)),
        "single_worst_true": float(np.max(single_model_true_wmses)),
        "wmse_obs_by_n": wmse_obs_by_n,
        "wmse_true_by_n": wmse_true_by_n,
        "coverage_pct": coverage_pct,
        "min_s_pred": min_s_pred,
        "gamma_col_top1": gamma_col_top1,
    }
