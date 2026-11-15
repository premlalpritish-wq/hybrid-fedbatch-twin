"""
core_math.py

Pure-NumPy/SciPy implementation of Dr. Moritz von Stosch's three core hybrid
modeling primitives:
1. SerialHybridODE: Serial hybrid bioreactor ODE with identifiable Sparse
   Rational-Monod rate basis (K_eff = 6 <= 12, gamma_col < 20) and vectorized
   positivity-preserving Patankar-RK2 integrator (dt = 0.1h).
2. SparseRateDiscriminator: Forward Orthogonal Least Squares (OLS) with ERR, BIC
   stopping, and collinearity index (Willis & von Stosch, 2017).
3. BaggedHybridTrainer: Bootstrap-aggregated hybrid model trainer with OOB
   early stopping and OOB outlier filtering (Pinto et al., 2019).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class BatchData:
    """Container for a single fed-batch bioreactor run."""

    t_eval: np.ndarray          # (M,) observation time points [h]
    y_obs: np.ndarray           # (M, 3) observed [X, S, P] concentrations [g/L]
    x0: np.ndarray              # (4,) initial state [X0, S0, P0, V0]
    f_feed: float               # Feed flow rate F_in [L/h] after t_feed_start
    t_feed_start: float = 4.0   # Feed start time [h]
    c_s_feed: float = 100.0     # Feed substrate concentration [g/L]
    y_true: Optional[np.ndarray] = None  # (M, 3) optional noiseless ground truth
    name: str = ""


class SerialHybridODE:
    """
    Serial Hybrid Bioreactor ODE & Vectorized Sensitivity Integrator.

    State vector: x = [X, S, P, V].
    White-box macroscopic transport/dilution:
        dV/dt = F_in(t),  D(t) = F_in(t) / V
        dX/dt = mu * X - D * X
        dS/dt = q_S * X + D * (c_S_feed - S)
        dP/dt = q_P * X - D * P

    Identifiable Sparse Rational-Monod kinetic basis (K_eff = 6 <= 12):
        u_1(S) = S / (S + 0.5)
        u_2(S, P) = u_1(S) * (P / (P + 15.0))
        mu(S, P) = w_1 * u_1(S) + w_2 * u_2(S, P)
        g_s(S) = S / (S + monod_guard_k)
        q_S(S, P) = g_s(S) * (a_S * mu(S, P) + b_S)
        q_P(S, P) = a_P * mu(S, P) + b_P * u_1(S)
    """

    def __init__(
        self,
        n_hidden: int = 2,
        n_states: int = 4,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        sigma_y: Optional[np.ndarray] = None,
        monod_guard_k: float = 0.05,
        dt: float = 0.1,
    ) -> None:
        self.n_hidden = int(n_hidden)
        self.n_states = int(n_states)
        self.n_rates = self.n_states - 1  # [X, S, P]
        self.monod_guard_k = float(monod_guard_k)
        self.dt = float(dt)

        self.x_mean = (
            np.zeros(self.n_rates, dtype=float)
            if x_mean is None
            else np.asarray(x_mean, dtype=float)
        )
        self.x_std = (
            np.ones(self.n_rates, dtype=float)
            if x_std is None
            else np.maximum(np.asarray(x_std, dtype=float), 1e-6)
        )
        self.sigma_y = (
            np.array([0.15, 0.25, 0.10], dtype=float)
            if sigma_y is None
            else np.asarray(sigma_y, dtype=float)
        )
        self.weights_y = 1.0 / (self.sigma_y ** 2)

        # K_eff = 6 trainable parameters: [w1, w2, a_s, b_s, a_p, b_p]
        self.n_params = 6
        self.param_bounds: List[Tuple[float, float]] = [
            (0.05, 1.0),    # w1
            (-1.0, 0.5),    # w2
            (-4.0, -0.5),   # a_s (-1/Y_XS)
            (-0.2, 0.05),   # b_s (-m_S)
            (0.0, 1.0),     # a_p (alpha)
            (-0.1, 0.2),    # b_p (beta)
        ]

    def init_params(self, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
        """Initialize kinetic parameters near physically plausible priors."""
        return np.array(
            [
                0.30 + rng.normal(0.0, 0.04),
                -0.10 + rng.normal(0.0, 0.04),
                -2.00 + rng.normal(0.0, 0.15),
                -0.02 + rng.normal(0.0, 0.008),
                0.30 + rng.normal(0.0, 0.04),
                0.01 + rng.normal(0.0, 0.004),
            ],
            dtype=float,
        )

    def _eval_rates_multi(
        self, x: np.ndarray, thetas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (mu, q_S_tilde, q_P, g_s) for x (K, B, 4) and thetas (K, 6)."""
        s_pos = np.maximum(x[:, :, 1], 0.0)
        p_pos = np.maximum(x[:, :, 2], 0.0)

        g_s = s_pos / (s_pos + self.monod_guard_k)
        u1 = s_pos / (s_pos + 0.5)
        u2 = u1 * (p_pos / (p_pos + 15.0))

        mu = thetas[:, 0:1] * u1 + thetas[:, 1:2] * u2
        q_s_tilde = thetas[:, 2:3] * mu + thetas[:, 3:4]
        q_p = thetas[:, 4:5] * mu + thetas[:, 5:6] * g_s
        return mu, q_s_tilde, q_p, g_s

    def _integrate_patankar_rk2(
        self,
        thetas: np.ndarray,
        batches: Sequence[BatchData],
    ) -> np.ndarray:
        """
        Vectorized positivity-preserving Patankar-RK2 (Heun) integrator at
        dt = 0.1h across K parameter sets and B batches simultaneously.
        Returns x_out of shape (K, B, M, 4).
        """
        k_tot = thetas.shape[0]
        bsz = len(batches)
        t_eval = batches[0].t_eval
        n_eval = len(t_eval)
        t0 = float(t_eval[0])
        t_end = float(t_eval[-1])
        dt = self.dt
        n_steps = int(round((t_end - t0) / dt))

        x0_batch = np.stack([b.x0 for b in batches], axis=0).astype(float)
        f_feeds = np.array([b.f_feed for b in batches], dtype=float)
        t_starts = np.array([b.t_feed_start for b in batches], dtype=float)
        c_s_feeds = np.array([b.c_s_feed for b in batches], dtype=float)

        eval_steps = np.rint((t_eval - t0) / dt).astype(int)
        step_to_eval = {int(s): idx for idx, s in enumerate(eval_steps)}

        x = np.broadcast_to(x0_batch[None, :, :], (k_tot, bsz, self.n_states)).copy()
        x_out = np.empty((k_tot, bsz, n_eval, self.n_states), dtype=float)
        if 0 in step_to_eval:
            x_out[:, :, step_to_eval[0], :] = x

        for step in range(n_steps):
            t_curr = t0 + step * dt
            f_0 = np.where(t_curr >= t_starts, f_feeds, 0.0)
            f_1 = np.where((t_curr + dt) >= t_starts, f_feeds, 0.0)

            x0_s, s0_s, p0_s, v0_s = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
            mu0, qst0, qp0, gs0 = self._eval_rates_multi(x, thetas)
            d0 = f_0[None, :] / np.maximum(v0_s, 1e-8)

            # Predictor step
            x1_s = x0_s * np.exp(np.clip(dt * (mu0 - d0), -20.0, 20.0))
            dest0 = np.maximum(-qst0, 0.0) * x0_s / (np.maximum(s0_s, 0.0) + self.monod_guard_k)
            prod0 = np.maximum(qst0, 0.0) * gs0 * x0_s + d0 * c_s_feeds[None, :]
            s1_s = (s0_s + dt * prod0) / (1.0 + dt * (d0 + dest0))
            p1_s = (p0_s + dt * qp0 * x0_s) / (1.0 + dt * d0)
            v1_s = v0_s + dt * f_0[None, :]
            x1 = np.stack([x1_s, s1_s, p1_s, v1_s], axis=-1)

            # Corrector step
            mu1, qst1, qp1, gs1 = self._eval_rates_multi(x1, thetas)
            d1 = f_1[None, :] / np.maximum(v1_s, 1e-8)

            x_new = x0_s * np.exp(
                np.clip(0.5 * dt * ((mu0 - d0) + (mu1 - d1)), -20.0, 20.0)
            )
            dest_avg = 0.5 * (
                dest0
                + np.maximum(-qst1, 0.0) * x1_s / (np.maximum(s1_s, 0.0) + self.monod_guard_k)
            )
            prod_avg = 0.5 * (
                prod0 + np.maximum(qst1, 0.0) * gs1 * x1_s + d1 * c_s_feeds[None, :]
            )
            s_new = (s0_s + dt * prod_avg) / (1.0 + dt * (0.5 * (d0 + d1) + dest_avg))
            p_new = (p0_s + 0.5 * dt * (qp0 * x0_s + qp1 * x1_s - d0 * p0_s)) / (
                1.0 + 0.5 * dt * d1
            )
            v_new = v0_s + 0.5 * dt * (f_0[None, :] + f_1[None, :])
            x = np.stack([x_new, s_new, p_new, v_new], axis=-1)

            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e4:
                x_out[:] = np.nan
                return x_out

            next_step = step + 1
            if next_step in step_to_eval:
                x_out[:, :, step_to_eval[next_step], :] = x

        return x_out

    def simulate_batches(
        self,
        theta: np.ndarray,
        batches: Sequence[BatchData],
        with_sensitivity: bool = False,
        fd_eps: float = 1e-6,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simulate all batches simultaneously; optionally compute sensitivities
        across all (n_params + 1) perturbed parameter vectors in one vectorized pass.
        """
        if not with_sensitivity:
            x_k = self._integrate_patankar_rk2(theta[None, :], batches)
            return x_k[0], None

        k_tot = self.n_params + 1
        thetas = np.broadcast_to(theta, (k_tot, self.n_params)).copy()
        idx_p = np.arange(self.n_params)
        thetas[1 + idx_p, idx_p] += fd_eps

        x_k = self._integrate_patankar_rk2(thetas, batches)
        x_out = x_k[0]
        w_out = np.transpose((x_k[1:] - x_out[None, :, :, :]) / fd_eps, (1, 2, 3, 0))
        return x_out, w_out

    def simulate_batch(
        self,
        theta: np.ndarray,
        batch: BatchData,
        with_sensitivity: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Single-batch convenience wrapper around simulate_batches."""
        x_out, w_out = self.simulate_batches(
            theta, [batch], with_sensitivity=with_sensitivity
        )
        return x_out[0], (w_out[0] if w_out is not None else None)

    def compute_wmse_and_grad(
        self,
        theta: np.ndarray,
        batches: Sequence[BatchData],
        with_grad: bool = True,
        l2_reg: float = 0.0,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Compute WMSE and sensitivity gradient across all batches."""
        x_trajs, w_trajs = self.simulate_batches(
            theta, batches, with_sensitivity=with_grad
        )
        if not np.all(np.isfinite(x_trajs)) or (
            with_grad and w_trajs is not None and not np.all(np.isfinite(w_trajs))
        ):
            if with_grad:
                return 1e8, 1e3 * theta
            return 1e8, None

        y_obs_all = np.stack([b.y_obs for b in batches], axis=0)
        err = x_trajs[:, :, : self.n_rates] - y_obs_all
        n_obs = err.size
        wmse = float(np.sum((err ** 2) * self.weights_y[None, None, :])) / max(n_obs, 1)
        if l2_reg > 0.0:
            wmse += 0.5 * l2_reg * float(np.dot(theta, theta))

        if with_grad and w_trajs is not None:
            grad = (
                2.0
                * np.einsum(
                    "bmi,i,bmik->k",
                    err,
                    self.weights_y,
                    w_trajs[:, :, : self.n_rates, :],
                )
                / max(n_obs, 1)
            )
            if l2_reg > 0.0:
                grad += l2_reg * theta
            return wmse, grad
        return wmse, None

    def compute_sensitivity_matrix(
        self, theta: np.ndarray, batches: Sequence[BatchData]
    ) -> np.ndarray:
        """Assemble stacked normalized-residual sensitivity matrix S."""
        _, w_trajs = self.simulate_batches(theta, batches, with_sensitivity=True)
        assert w_trajs is not None
        sqrt_w = np.sqrt(self.weights_y)
        weighted_w = w_trajs[:, :, : self.n_rates, :] * sqrt_w[None, None, :, None]
        return weighted_w.reshape(-1, self.n_params)


def compute_collinearity_index(sensitivity_matrix: np.ndarray) -> float:
    """
    Compute the Collinearity Index (Brun et al., 2001 / Willis & von Stosch, 2017):
        gamma_col = 1 / sqrt(lambda_min(S_norm^T @ S_norm))
    """
    s = np.asarray(sensitivity_matrix, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    norms = np.linalg.norm(s, axis=0)
    nonzero = norms > 1e-15
    if not np.any(nonzero):
        return float("inf")
    s_norm = s[:, nonzero] / norms[nonzero][None, :]
    gram = s_norm.T @ s_norm
    eigvals = np.linalg.eigvalsh(gram)
    lam_min = float(np.min(eigvals))
    if lam_min <= 1e-30:
        return float("inf")
    return float(1.0 / np.sqrt(lam_min))


class SparseRateDiscriminator:
    """Willis & von Stosch (2017) Sparse Structure Discriminator."""

    def __init__(self, min_orth_ratio: float = 1e-12) -> None:
        self.min_orth_ratio = float(min_orth_ratio)
        self.selected_indices_: List[int] = []
        self.selected_names_: List[str] = []
        self.err_history_: List[float] = []
        self.bic_history_: List[float] = []
        self.coef_: Optional[np.ndarray] = None
        self.gamma_col_: float = float("nan")

    def fit(
        self,
        phi: np.ndarray,
        y: np.ndarray,
        term_names: Optional[Sequence[str]] = None,
    ) -> "SparseRateDiscriminator":
        """Run forward OLS-ERR with BIC stopping on candidate basis matrix phi (N, M)."""
        phi = np.asarray(phi, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_terms = phi.shape

        names = [f"term_{j}" for j in range(n_terms)] if term_names is None else list(term_names)
        y_energy = float(np.dot(y, y))
        if y_energy <= 1e-30:
            raise ValueError("Target vector y has zero energy.")

        selected: List[int] = []
        q_basis: List[np.ndarray] = []
        err_hist: List[float] = []
        bic_hist: List[float] = []

        best_bic = float("inf")
        remaining = list(range(n_terms))

        for k in range(1, n_terms + 1):
            best_j = -1
            best_err = -1.0
            best_q: Optional[np.ndarray] = None

            for j in remaining:
                col = phi[:, j].copy()
                col_norm2 = float(np.dot(col, col))
                if col_norm2 <= 1e-30:
                    continue

                q = col.copy()
                for q_prev in q_basis:
                    denom = float(np.dot(q_prev, q_prev))
                    q = q - (float(np.dot(q_prev, q)) / denom) * q_prev

                q_norm2 = float(np.dot(q, q))
                if q_norm2 < self.min_orth_ratio * col_norm2:
                    continue

                proj = float(np.dot(q, y))
                err = (proj * proj) / (q_norm2 * y_energy)
                if err > best_err:
                    best_err = err
                    best_j = j
                    best_q = q

            if best_j < 0 or best_q is None:
                break

            trial_sel = selected + [best_j]
            phi_trial = phi[:, trial_sel]
            coef_trial, _, _, _ = np.linalg.lstsq(phi_trial, y, rcond=None)
            rss = float(np.sum((y - phi_trial @ coef_trial) ** 2))
            bic = n_samples * float(np.log(max(rss / n_samples, 1e-30))) + k * float(
                np.log(n_samples)
            )

            if bic < best_bic:
                best_bic = bic
                selected.append(best_j)
                q_basis.append(best_q)
                remaining.remove(best_j)
                err_hist.append(best_err)
                bic_hist.append(bic)
            else:
                bic_hist.append(bic)
                break

        self.selected_indices_ = selected
        self.selected_names_ = [names[j] for j in selected]
        self.err_history_ = err_hist
        self.bic_history_ = bic_hist

        if selected:
            phi_sel = phi[:, selected]
            self.coef_, _, _, _ = np.linalg.lstsq(phi_sel, y, rcond=None)
            self.gamma_col_ = compute_collinearity_index(phi_sel)
        else:
            self.coef_ = np.array([], dtype=float)
            self.gamma_col_ = float("nan")

        return self


@dataclass
class CandidateModelRecord:
    """Stores a single bootstrap-trained candidate hybrid model."""

    candidate_id: int
    theta: np.ndarray
    train_wmse: float
    val_wmse: float
    score: float
    train_indices: List[int]
    oob_indices: List[int]
    n_iters: int
    best_iter: int


class BaggedHybridTrainer:
    """
    Pinto, de Azevedo, Oliveira, & von Stosch (2019) Bootstrap-Aggregated Trainer.

    Partitions development batches into B_cand bootstrap resamples at the batch
    level (~67% internal training sampled with replacement, ~33% OOB validation),
    trains each candidate via bounded L-BFGS-B (maxiter=60), applies OOB early
    stopping and Pinto et al. (2019)'s OOB outlier filter (val_WMSE <= 2.5 * min),
    and retains the top B_ens models.
    """

    def __init__(
        self,
        ode_model: SerialHybridODE,
        b_cand: int = 12,
        b_ens: int = 8,
        maxiter: int = 60,
        l2_reg: float = 0.0,
        oob_outlier_factor: float = 2.5,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.ode_model = ode_model
        self.b_cand = int(b_cand)
        self.b_ens = int(b_ens)
        self.maxiter = int(maxiter)
        self.l2_reg = float(l2_reg)
        self.oob_outlier_factor = float(oob_outlier_factor)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.candidates_: List[CandidateModelRecord] = []
        self.ensemble_: List[CandidateModelRecord] = []

    def _draw_bootstrap_split(
        self, n_batches: int, rng: np.random.Generator
    ) -> Tuple[List[int], List[int]]:
        """Draw a batch-level bootstrap split (~67% train with replacement, ~33% OOB)."""
        if n_batches < 2:
            raise ValueError("Need at least 2 development batches for OOB bagging.")
        for _ in range(100):
            train_idx = rng.integers(0, n_batches, size=n_batches).tolist()
            train_set = set(train_idx)
            oob_idx = [i for i in range(n_batches) if i not in train_set]
            if len(oob_idx) >= max(1, n_batches // 4):
                return train_idx, oob_idx
        n_oob = max(1, int(round(0.33 * n_batches)))
        perm = rng.permutation(n_batches).tolist()
        oob_idx = perm[:n_oob]
        pool = perm[n_oob:]
        train_idx = rng.choice(pool, size=n_batches, replace=True).tolist()
        return train_idx, oob_idx

    def fit(self, dev_batches: Sequence[BatchData]) -> "BaggedHybridTrainer":
        """Train B_cand bootstrap candidates, filter OOB outliers, and keep top B_ens."""
        rng = np.random.default_rng(self.seed)
        n_batches = len(dev_batches)
        bounds = self.ode_model.param_bounds
        records: List[CandidateModelRecord] = []

        for c_id in range(self.b_cand):
            t_cand0 = time.perf_counter()
            train_idx, oob_idx = self._draw_bootstrap_split(n_batches, rng)
            train_batches = [dev_batches[i] for i in train_idx]
            oob_batches = [dev_batches[i] for i in oob_idx]

            theta0 = self.ode_model.init_params(rng, scale=0.1)
            best_val = float(
                self.ode_model.compute_wmse_and_grad(
                    theta0, oob_batches, with_grad=False, l2_reg=0.0
                )[0]
            )
            best_theta = theta0.copy()
            best_iter = 0
            iter_count = 0

            def objective(th: np.ndarray) -> Tuple[float, np.ndarray]:
                loss, grad = self.ode_model.compute_wmse_and_grad(
                    th, train_batches, with_grad=True, l2_reg=self.l2_reg
                )
                assert grad is not None
                return loss, grad

            def callback(xk: np.ndarray) -> None:
                nonlocal best_val, best_theta, best_iter, iter_count
                iter_count += 1
                val_wmse, _ = self.ode_model.compute_wmse_and_grad(
                    xk, oob_batches, with_grad=False, l2_reg=0.0
                )
                if val_wmse < best_val:
                    best_val = val_wmse
                    best_theta = xk.copy()
                    best_iter = iter_count

            minimize(
                objective,
                theta0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                callback=callback,
                options={
                    "maxiter": self.maxiter,
                    "maxls": 8,
                    "ftol": 1e-6,
                    "gtol": 1e-3,
                },
            )

            best_tr, _ = self.ode_model.compute_wmse_and_grad(
                best_theta, train_batches, with_grad=False, l2_reg=0.0
            )
            val_wmse, _ = self.ode_model.compute_wmse_and_grad(
                best_theta, oob_batches, with_grad=False, l2_reg=0.0
            )
            score = 0.5 * best_tr + 0.5 * val_wmse
            elapsed = time.perf_counter() - t_cand0

            if self.verbose:
                print(
                    f"  [Bootstrap {c_id + 1}/{self.b_cand}] "
                    f"train_WMSE={best_tr:.4f}, val_WMSE={val_wmse:.4f} ({elapsed:.2f}s)",
                    flush=True,
                )

            records.append(
                CandidateModelRecord(
                    candidate_id=c_id,
                    theta=best_theta,
                    train_wmse=best_tr,
                    val_wmse=val_wmse,
                    score=score,
                    train_indices=train_idx,
                    oob_indices=oob_idx,
                    n_iters=iter_count,
                    best_iter=best_iter,
                )
            )

        # Pinto et al. (2019) OOB Validation Outlier Filter
        min_val = min(r.val_wmse for r in records)
        filtered = [r for r in records if r.val_wmse <= self.oob_outlier_factor * min_val]
        if len(filtered) < self.b_ens:
            filtered = sorted(records, key=lambda r: r.val_wmse)[: self.b_ens]
        filtered.sort(key=lambda r: r.score)

        self.candidates_ = sorted(records, key=lambda r: r.score)
        self.ensemble_ = filtered[: min(self.b_ens, len(filtered))]
        return self

    def predict_ensemble(
        self, batches: Sequence[BatchData], n_models: Optional[int] = None
    ) -> Dict[str, object]:
        """Predict trajectories and WMSE vs aggregation size n = 1..n_models."""
        if not self.ensemble_:
            raise RuntimeError("BaggedHybridTrainer must be fitted before prediction.")

        k = len(self.ensemble_) if n_models is None else min(n_models, len(self.ensemble_))
        models = self.ensemble_[:k]

        per_model_trajs: List[List[np.ndarray]] = []
        min_s = float("inf")
        for rec in models:
            x_trajs, _ = self.ode_model.simulate_batches(
                rec.theta, batches, with_sensitivity=False
            )
            min_s = min(min_s, float(np.min(x_trajs[:, :, 1])))
            per_model_trajs.append([x_trajs[b_idx] for b_idx in range(len(batches))])

        mean_trajs: List[np.ndarray] = []
        std_trajs: List[np.ndarray] = []
        std_model_trajs: List[np.ndarray] = []
        mean_oob_wmse = float(np.mean([r.val_wmse for r in models]))
        oob_bias_var = max(mean_oob_wmse - 1.0, 0.0) * (self.ode_model.sigma_y ** 2)
        for b_idx in range(len(batches)):
            stacked = np.stack([per_model_trajs[m][b_idx] for m in range(k)], axis=0)
            mean_trajs.append(np.mean(stacked, axis=0))
            ddof = 1 if k > 1 else 0
            epist_std = np.std(stacked, axis=0, ddof=ddof)
            std_trajs.append(epist_std)
            model_std = epist_std.copy()
            model_std[:, : self.ode_model.n_rates] = np.sqrt(
                epist_std[:, : self.ode_model.n_rates] ** 2 + oob_bias_var[None, :]
            )
            std_model_trajs.append(model_std)

        wmse_by_n: Dict[int, float] = {}
        wmse_true_by_n: Dict[int, float] = {}
        has_true = all(b.y_true is not None for b in batches)

        for n in range(1, k + 1):
            sq_obs = 0.0
            sq_true = 0.0
            n_obs = 0
            for b_idx, batch in enumerate(batches):
                agg = np.mean(
                    np.stack([per_model_trajs[m][b_idx] for m in range(n)], axis=0),
                    axis=0,
                )[:, : self.ode_model.n_rates]
                err_obs = agg - batch.y_obs
                sq_obs += float(np.sum((err_obs ** 2) * self.ode_model.weights_y[None, :]))
                n_obs += err_obs.size
                if has_true and batch.y_true is not None:
                    err_true = agg - batch.y_true
                    sq_true += float(
                        np.sum((err_true ** 2) * self.ode_model.weights_y[None, :])
                    )
            wmse_by_n[n] = sq_obs / max(n_obs, 1)
            if has_true:
                wmse_true_by_n[n] = sq_true / max(n_obs, 1)

        return {
            "per_model_trajs": per_model_trajs,
            "mean_traj": mean_trajs,
            "std_traj": std_trajs,
            "std_model_traj": std_model_trajs,
            "wmse_by_n": wmse_by_n,
            "wmse_true_by_n": wmse_true_by_n,
            "min_substrate": min_s,
        }
