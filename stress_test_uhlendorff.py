"""
stress_test_uhlendorff.py

Adversarial Wet-Lab Stress Test of `hybrid-fedbatch-twin` using `@von_stosch_twin`
on the Uhlendorff et al. (2026) CHO-K1 dataset (12 campaigns EXP001-EXP012,
24 bioreactor runs, 48 inline/offline CSVs).

Vectorized pure-NumPy Patankar-RK2 integrator across (K_perturbations, B_runs, 4)
for < 2s per bootstrap and < 30s total runtime.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize

from agents.von_stosch_twin.core_math import (
    SparseRateDiscriminator,
    compute_collinearity_index,
)

DATA_DIR = os.path.join("data", "uhlendorff_2026", "Time-axis aligned raw data tables")
STATE_NAMES = ["VCD (1e6/mL)", "Glc (g/L)", "Lac (g/L)", "Amm (mM)"]
SENSORS_7 = [
    "DeltaEps (pF/cm)",
    "fc (kHz)",
    "Alpha",
    "Permittivity (pF/cm)",
    "Conductivity (mS/cm)",
    "Transmission (arb. Unit)",
    "Reflection (arb. Unit)",
]
PAT_EXTRA = ["Dissolved O2 (% air sat)", "Dissolved CO2 (%-sat)", "pH"]

DEV_EXPS = ["EXP001", "EXP002", "EXP003", "EXP004", "EXP005", "EXP006", "EXP007", "EXP010", "EXP011"]
TEST_EXPS = ["EXP008", "EXP009", "EXP012"]


@dataclass
class CHORun:
    """Single bioreactor cultivation run from Uhlendorff et al. (2026)."""

    exp_id: str
    rep_id: str
    mode: str                   # "batch" or "fed-batch"
    t_eval: np.ndarray          # (M,) offline timestamps [h]
    y_obs: np.ndarray           # (M, 4) [VCD_1e6, Glc, Lac, Amm]
    v_grid: np.ndarray          # (M,) liquid volume [L] at offline times
    f_in_offline: np.ndarray    # (M,) causal feed rate [L/h] at offline times
    t_feed_start: float         # 48.0h for fed-batch, 1e9 for batch
    c_glc_feed: float           # Feed glucose concentration [g/L]
    pat_features: np.ndarray    # (M, 10) causal [t-0.5h, t] means (NaN if dropout)
    pat_valid_7: np.ndarray     # (M,) bool: True iff all 7 core PAT sensors complete in [t-0.5h, t]
    t_inline: np.ndarray        # High-frequency inline timestamps [h]
    pat_inline: np.ndarray      # (N_inl, 4) [DeltaEps, Permittivity, Transmission, Reflection]
    pat_inline_valid: np.ndarray  # (N_inl,) bool: all 7 core sensors valid at inline step
    f_in_inline: np.ndarray     # (N_inl,) causal feed rate [L/h]
    v_inline: np.ndarray        # (N_inl,) liquid volume [L]


def detect_biophysical_anomaly_mask(
    y_obs: np.ndarray,
    vcd_thresh: float = 0.05,
    amm_thresh: float = 25.0,
) -> np.ndarray:
    """Detect sensor/assay collapse rows via biophysical invariants (zero viable cells with high ammonium)."""
    arr = np.asarray(y_obs, dtype=float)
    return (arr[:, 0] <= float(vcd_thresh)) & (arr[:, 3] > float(amm_thresh))


def is_biophysical_anomaly_run(run: CHORun, min_anomalous_points: int = 2) -> bool:
    """Return True if a cultivation run fails the biophysical sensor-consistency check."""
    return int(np.sum(detect_biophysical_anomaly_mask(run.y_obs))) >= int(min_anomalous_points)


def load_uhlendorff_dataset(c_glc_feed_est: float = 18.0) -> Tuple[List[CHORun], List[CHORun], Dict[str, int]]:
    """Ingest all 24 runs (48 CSVs) with strictly causal [t-0.5h, t] inline PAT alignment."""
    dev_runs: List[CHORun] = []
    test_runs: List[CHORun] = []
    total_rows = 0
    incomplete_7_rows = 0
    anomaly_rows = 0

    for exp in DEV_EXPS + TEST_EXPS:
        mode = (
            "fed-batch"
            if os.path.isfile(os.path.join(DATA_DIR, f"{exp}_fed-batch_R01_offline.csv"))
            else "batch"
        )
        for rep in ("R01", "R02"):
            off = pd.read_csv(os.path.join(DATA_DIR, f"{exp}_{mode}_{rep}_offline.csv"))
            inl = pd.read_csv(os.path.join(DATA_DIR, f"{exp}_{mode}_{rep}_inline.csv"))

            t_eval = off["Time (h)"].values.astype(float)
            vcd = off["VCD Mean (cells/mL)"].values.astype(float) / 1e6
            glc = off["Glucose (g/L)"].values.astype(float)
            lac = off["Lactate (g/L)"].values.astype(float)
            amm = off["Ammonium (mM)"].values.astype(float)
            y_obs = np.column_stack([vcd, glc, lac, amm])
            anomaly_rows += int(np.sum(detect_biophysical_anomaly_mask(y_obs)))

            t_inl = inl["Time (h)"].values.astype(float)
            v_inl = np.maximum(inl["Liquid volume (L)"].values.astype(float), 0.2)
            fw_inl = inl["Cumulative feed weight (g)"].fillna(0.0).values.astype(float)

            f_in_inl = np.zeros_like(t_inl)
            t_feed_start = 1e9
            if mode == "fed-batch":
                nz_feed = np.where(fw_inl > 0.5)[0]
                t_feed_start = float(t_inl[int(nz_feed[0])]) if len(nz_feed) else 1e9
                dt_inl = np.maximum(np.diff(t_inl, prepend=t_inl[0] - 1.0 / 60.0), 1e-4)
                dfw = np.maximum(np.diff(fw_inl, prepend=fw_inl[0]), 0.0)
                raw_f = (dfw * 0.001) / dt_inl
                kernel = np.ones(30) / 30.0
                f_smooth = np.convolve(raw_f, kernel, mode="full")[: len(raw_f)]
                f_in_inl = np.where(t_inl >= t_feed_start, f_smooth, 0.0)

            pat_cols_all = SENSORS_7 + PAT_EXTRA
            pat_df = inl[pat_cols_all]
            s7_df = inl[SENSORS_7]
            pat_inl_valid = ~s7_df.isna().any(axis=1).values
            pat_inl_4 = inl[
                ["DeltaEps (pF/cm)", "Permittivity (pF/cm)", "Transmission (arb. Unit)", "Reflection (arb. Unit)"]
            ].values.astype(float)

            m_pts = len(t_eval)
            pat_feat = np.full((m_pts, len(pat_cols_all)), np.nan, dtype=float)
            pat_valid_7 = np.zeros(m_pts, dtype=bool)
            v_off = np.zeros(m_pts, dtype=float)
            f_off = np.zeros(m_pts, dtype=float)

            for i, t_val in enumerate(t_eval):
                total_rows += 1
                win_mask = (t_inl >= t_val - 0.5) & (t_inl <= t_val)
                if not np.any(win_mask):
                    incomplete_7_rows += 1
                    v_off[i] = v_inl[0]
                    continue

                v_win = v_inl[win_mask]
                v_off[i] = float(np.nanmean(v_win)) if np.any(np.isfinite(v_win)) else float(v_inl[0])
                f_off[i] = float(np.nanmean(f_in_inl[win_mask]))

                s7_win = s7_df.loc[win_mask]
                if len(s7_win) == 0 or s7_win.isna().any().any():
                    incomplete_7_rows += 1
                else:
                    pat_valid_7[i] = True
                    pat_feat[i, :] = pat_df.loc[win_mask].mean(axis=0).values

            run = CHORun(
                exp_id=exp,
                rep_id=rep,
                mode=mode,
                t_eval=t_eval,
                y_obs=y_obs,
                v_grid=v_off,
                f_in_offline=f_off,
                t_feed_start=t_feed_start if mode == "fed-batch" else 1e9,
                c_glc_feed=c_glc_feed_est if mode == "fed-batch" else 0.0,
                pat_features=pat_feat,
                pat_valid_7=pat_valid_7,
                t_inline=t_inl,
                pat_inline=pat_inl_4,
                pat_inline_valid=pat_inl_valid,
                f_in_inline=f_in_inl,
                v_inline=v_inl,
            )
            if exp in DEV_EXPS:
                dev_runs.append(run)
            else:
                test_runs.append(run)

    return dev_runs, test_runs, {
        "total_offline_rows": total_rows,
        "incomplete_7_sensor_rows": incomplete_7_rows,
        "anomaly_rows": anomaly_rows,
    }


def compute_duplicate_noise_floor(
    runs: Sequence[CHORun],
    exclude_exps: Sequence[str] = (),
    exclude_anomalous: bool = True,
) -> np.ndarray:
    """Compute Sigma_y = sqrt(0.5 * E[(x_R01 - x_R02)^2]) across clean duplicate pairs."""
    anomalous_exps = {r.exp_id for r in runs if exclude_anomalous and is_biophysical_anomaly_run(r)}
    excluded = set(exclude_exps) | anomalous_exps
    by_exp: Dict[str, Dict[str, CHORun]] = {}
    for r in runs:
        if r.exp_id in excluded:
            continue
        by_exp.setdefault(r.exp_id, {})[r.rep_id] = r
    diffs = []
    for _, reps in sorted(by_exp.items()):
        if "R01" in reps and "R02" in reps:
            y1, y2 = reps["R01"].y_obs, reps["R02"].y_obs
            n = min(len(y1), len(y2))
            diffs.append(y1[:n] - y2[:n])
    return np.sqrt(0.5 * np.nanmean(np.vstack(diffs) ** 2, axis=0))


# =============================================================================
# Vectorized Patankar-RK2 Integrator across (K, B, 4)
# =============================================================================

@dataclass
class PreparedCHOBatch:
    """Pre-aligned tensor batch for ultra-fast vectorized Patankar-RK2 rollouts."""

    n_runs: int
    n_steps: int
    dt: float
    x0: np.ndarray          # (B, 4)
    v_grid: np.ndarray      # (B, n_steps + 1) true causal Liquid volume (L)
    c_glc_feed: np.ndarray  # (B,)
    f_grid: np.ndarray      # (B, n_steps + 1)
    y_obs_pad: np.ndarray   # (B, M_max, 4)
    obs_mask: np.ndarray    # (B, M_max) bool
    idx_lo: np.ndarray      # (B, M_max) int
    idx_hi: np.ndarray      # (B, M_max) int
    alpha: np.ndarray       # (B, M_max) float
    n_obs_scalars: int

    @classmethod
    def from_runs(cls, runs: Sequence[CHORun], dt: float = 1.0, mask_anomalies: bool = False) -> "PreparedCHOBatch":
        b_cnt = len(runs)
        t_max = max(float(r.t_eval[-1]) for r in runs)
        n_steps = int(np.ceil(t_max / dt))
        t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)
        m_max = max(len(r.t_eval) for r in runs)

        x0 = np.stack([np.maximum(r.y_obs[0], 0.0) for r in runs], axis=0)
        v_grid = np.stack([np.maximum(np.interp(t_grid, r.t_inline, r.v_inline), 0.2) for r in runs], axis=0)
        c_glc = np.array([float(r.c_glc_feed) for r in runs], dtype=float)
        f_grid = np.stack([np.interp(t_grid, r.t_inline, r.f_in_inline) for r in runs], axis=0)

        y_pad = np.zeros((b_cnt, m_max, 4), dtype=float)
        mask = np.zeros((b_cnt, m_max), dtype=bool)
        idx_lo = np.zeros((b_cnt, m_max), dtype=int)
        idx_hi = np.zeros((b_cnt, m_max), dtype=int)
        alpha = np.zeros((b_cnt, m_max), dtype=float)

        for b_i, r in enumerate(runs):
            m = len(r.t_eval)
            y_pad[b_i, :m, :] = r.y_obs
            row_ok = np.all(np.isfinite(r.y_obs), axis=1)
            if mask_anomalies:
                if is_biophysical_anomaly_run(r):
                    row_ok[:] = False
            mask[b_i, :m] = row_ok
            pos = np.clip(r.t_eval / dt, 0.0, n_steps)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, n_steps)
            idx_lo[b_i, :m] = lo
            idx_hi[b_i, :m] = hi
            alpha[b_i, :m] = pos - lo

        return cls(
            n_runs=b_cnt,
            n_steps=n_steps,
            dt=dt,
            x0=x0,
            v_grid=v_grid,
            c_glc_feed=c_glc,
            f_grid=f_grid,
            y_obs_pad=y_pad,
            obs_mask=mask,
            idx_lo=idx_lo,
            idx_hi=idx_hi,
            alpha=alpha,
            n_obs_scalars=int(np.sum(mask)) * 4,
        )


def rollout_vectorized_patankar(
    batch: PreparedCHOBatch,
    rate_fn_multi,
    thetas: np.ndarray,
) -> np.ndarray:
    """
    Integrate K parameter vectors across all B runs simultaneously.
    Returns predicted states at offline timestamps: shape (K, B, M_max, 4).
    """
    k_tot = thetas.shape[0]
    bsz = batch.n_runs
    dt = batch.dt

    x = np.broadcast_to(batch.x0[None, :, :], (k_tot, bsz, 4)).copy()
    traj_grid = np.empty((k_tot, bsz, batch.n_steps + 1, 4), dtype=float)
    traj_grid[:, :, 0, :] = x

    c_feed = batch.c_glc_feed[None, :]
    for s in range(batch.n_steps):
        f0 = batch.f_grid[:, s][None, :]
        f1 = batch.f_grid[:, s + 1][None, :]
        d0 = f0 / batch.v_grid[:, s][None, :]

        vcd0, glc0, lac0, amm0 = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
        mu0, kd0, qg0, qlp0, qlc0, qa0 = rate_fn_multi(x, thetas)

        vcd1 = vcd0 * np.exp(np.clip(dt * (mu0 - kd0 - d0), -10.0, 10.0))
        dest_g0 = np.maximum(-qg0, 0.0) * vcd0 / (np.maximum(glc0, 0.0) + 0.05)
        prod_g0 = np.maximum(qg0, 0.0) * (glc0 / (np.maximum(glc0, 0.0) + 0.05)) * vcd0 + d0 * c_feed
        glc1 = (glc0 + dt * prod_g0) / (1.0 + dt * (d0 + dest_g0))

        dest_l0 = np.maximum(qlc0, 0.0) * vcd0 / (np.maximum(lac0, 0.0) + 0.1)
        lac1 = (lac0 + dt * np.maximum(qlp0, 0.0) * vcd0) / (1.0 + dt * (d0 + dest_l0))
        dest_a0 = np.maximum(-qa0, 0.0) * vcd0 / (np.maximum(amm0, 0.0) + 0.1)
        amm1 = (amm0 + dt * np.maximum(qa0, 0.0) * vcd0) / (1.0 + dt * (d0 + dest_a0))
        x1 = np.stack([vcd1, glc1, lac1, amm1], axis=-1)

        d1 = f1 / batch.v_grid[:, s + 1][None, :]
        mu1, kd1, qg1, qlp1, qlc1, qa1 = rate_fn_multi(x1, thetas)

        vcd_n = vcd0 * np.exp(np.clip(0.5 * dt * ((mu0 - kd0 - d0) + (mu1 - kd1 - d1)), -10.0, 10.0))
        dest_g_avg = 0.5 * (dest_g0 + np.maximum(-qg1, 0.0) * vcd1 / (np.maximum(glc1, 0.0) + 0.05))
        prod_g_avg = 0.5 * (prod_g0 + np.maximum(qg1, 0.0) * (glc1 / (np.maximum(glc1, 0.0) + 0.05)) * vcd1 + d1 * c_feed)
        glc_n = (glc0 + dt * prod_g_avg) / (1.0 + dt * (0.5 * (d0 + d1) + dest_g_avg))

        dest_l_avg = 0.5 * (dest_l0 + np.maximum(qlc1, 0.0) * vcd1 / (np.maximum(lac1, 0.0) + 0.1))
        prod_l_avg = 0.5 * (np.maximum(qlp0, 0.0) * vcd0 + np.maximum(qlp1, 0.0) * vcd1)
        lac_n = (lac0 + dt * prod_l_avg) / (1.0 + dt * (0.5 * (d0 + d1) + dest_l_avg))

        prod_a_avg = 0.5 * (np.maximum(qa0, 0.0) * vcd0 + np.maximum(qa1, 0.0) * vcd1)
        amm_n = (amm0 + dt * prod_a_avg) / (1.0 + dt * 0.5 * (d0 + d1))

        x = np.maximum(np.stack([vcd_n, glc_n, lac_n, amm_n], axis=-1), 0.0)
        if not np.all(np.isfinite(x)) or np.max(x) > 1e4:
            return np.full((k_tot, bsz, batch.n_steps + 1, 4), np.nan)
        traj_grid[:, :, s + 1, :] = x

    # Gather & interpolate onto exact offline evaluation points: (K, B, M_max, 4)
    b_idx = np.arange(bsz)[None, :, None]
    k_idx = np.arange(k_tot)[:, None, None]
    v_lo = traj_grid[k_idx, b_idx, batch.idx_lo[None, :, :], :]
    v_hi = traj_grid[k_idx, b_idx, batch.idx_hi[None, :, :], :]
    a = batch.alpha[None, :, :, None]
    return (1.0 - a) * v_lo + a * v_hi


# =============================================================================
# SYSTEM A: Our `hybrid-fedbatch-twin` (Mechanistic + MLP Residual + B=15)
# =============================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def _huber_loss_and_grad_coeff(
    err: np.ndarray,
    sigma_y: np.ndarray,
    mask: np.ndarray,
    delta: float = 3.0,
) -> Tuple[float, np.ndarray]:
    """
    Smooth Huber loss on standardized residuals r = err / sigma_y with threshold delta=3.0.
    Returns (mean_huber_loss, d_loss_d_err_scaled_before_sens).
    """
    r = err / sigma_y[None, None, :]
    abs_r = np.abs(r)
    quad = abs_r <= delta
    loss_elem = np.where(quad, 0.5 * r * r, delta * (abs_r - 0.5 * delta)) * mask[:, :, None]
    n_obs = max(int(np.sum(mask)) * 4, 1)
    loss = float(np.sum(loss_elem)) / n_obs
    d_r = np.where(quad, r, delta * np.sign(r)) * mask[:, :, None]
    d_err = d_r / (sigma_y[None, None, :] * n_obs)
    return loss, d_err


def mech_cho_rates_multi(x: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Vectorized Stage-1 mechanistic CHO-K1 rates for x (K, B, 4) and p (K, 9).
    Locked priors: K_G=0.5 g/L, K_I_Lac=5.0 g/L, K_I_Amm=10.0 mM, K_d_Amm=5.0 mM, K_d_Lac=4.0 g/L.
    """
    vcd = np.maximum(x[:, :, 0], 0.0)
    g = np.maximum(x[:, :, 1], 0.0)
    l = np.maximum(x[:, :, 2], 0.0)
    a = np.maximum(x[:, :, 3], 0.0)
    mu = p[:, 0:1] * (g / (g + 0.5)) / ((1.0 + l / 5.0) * (1.0 + a / 10.0))
    kd = (
        p[:, 1:2] * (a / (a + 5.0) + l / (l + 4.0)) * (vcd / (vcd + 1.0))
        + p[:, 2:3] * (0.2 / (g + 0.2))
    )
    qg_tilde = -(mu / p[:, 3:4] + p[:, 4:5])
    ql_prod = p[:, 5:6] * mu
    ql_cons = p[:, 6:7] / (1.0 + g / 0.5)
    qa = p[:, 7:8] * mu + p[:, 8:9]
    return mu, kd, qg_tilde, ql_prod, ql_cons, qa


class HybridFedBatchTwinCHO:
    """
    Remediated `hybrid-fedbatch-twin`:
    - Parsimonious decoupled mechanistic + rational/RBF residual (K_eff=9, gamma_col < 20)
    - CHO-K1 late-phase toxicity + starvation apoptosis kinetics
    - Internal biophysical anomaly gate + Huber loss (delta = 3.0 * Sigma_y)
    - Regime-stratified B=15 bootstrap + OOB epistemic variance calibration
    """

    def __init__(self, sigma_y: np.ndarray, b_ens: int = 15, h_nodes: int = 4, seed: int = 42) -> None:
        self.sigma_y = np.asarray(sigma_y, dtype=float)
        self.weights_y = 1.0 / (self.sigma_y ** 2)
        self.b_ens = int(b_ens)
        self.h_nodes = int(h_nodes)
        self.seed = int(seed)
        self.p_mech: Optional[np.ndarray] = None
        self.n_ml = 9  # K_eff = 9 decoupled active parameters (Stage-1 priors locked)
        self.ensemble_thetas: List[np.ndarray] = []
        self.oob_var: np.ndarray = self.sigma_y ** 2

    def hybrid_rates_multi(self, x: np.ndarray, th_ml: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Vectorized decoupled hybrid rates for x (K, B, 4) and th_ml (K, 9)."""
        assert self.p_mech is not None
        p_b = np.broadcast_to(self.p_mech[None, :], (th_ml.shape[0], 9))
        mu_m, kd_m, qg_m, qlp_m, qlc_m, qa_m = mech_cho_rates_multi(x, p_b)

        vcd = np.maximum(x[:, :, 0], 0.0)
        g = np.maximum(x[:, :, 1], 0.0)
        l = np.maximum(x[:, :, 2], 0.0)
        a = np.maximum(x[:, :, 3], 0.0)

        # Orthogonalized / state-separated decoupled residual multipliers
        rbf_gl = np.exp(-((g / 5.0 - 0.5) ** 2 + (l / 5.0 - 0.5) ** 2))
        mu = mu_m * np.exp(np.clip(0.5 * th_ml[:, 0:1] + 0.4 * th_ml[:, 1:2] * rbf_gl, -1.5, 1.5))

        tox_gate = (a / (a + 5.0) + l / (l + 4.0)) * (vcd / (vcd + 1.0))
        starv_gate = 0.2 / (g + 0.2)
        kd = np.maximum(
            0.0,
            kd_m + 0.025 * th_ml[:, 2:3] * tox_gate + 0.04 * th_ml[:, 3:4] * starv_gate,
        )

        qg_tilde = qg_m * np.exp(np.clip(0.5 * th_ml[:, 4:5], -1.5, 1.5))
        ql_prod = np.maximum(0.0, qlp_m * np.exp(np.clip(0.5 * th_ml[:, 5:6], -1.5, 1.5)))
        ql_cons = np.maximum(0.0, qlc_m * np.exp(np.clip(0.5 * th_ml[:, 6:7], -1.5, 1.5)))
        qa = np.maximum(0.0, qa_m * np.exp(np.clip(0.5 * th_ml[:, 7:8], -1.5, 1.5)) + 0.005 * th_ml[:, 8:9])
        return mu, kd, qg_tilde, ql_prod, ql_cons, qa

    def fit_stage1(self, dev_runs: Sequence[CHORun]) -> np.ndarray:
        batch = PreparedCHOBatch.from_runs(dev_runs, dt=1.0, mask_anomalies=True)
        p0 = np.array([0.045, 0.015, 0.025, 2.2, 0.004, 0.45, 0.015, 0.5, 0.003])
        lb = np.array([0.015, 0.002, 0.002, 1.0, 0.0005, 0.05, 0.0, 0.05, 0.0])
        ub = np.array([0.10, 0.08, 0.12, 3.5, 0.008, 2.5, 0.12, 2.5, 0.04])
        sqrt_w = np.sqrt(self.weights_y)

        def residuals(p: np.ndarray) -> np.ndarray:
            pred = rollout_vectorized_patankar(batch, mech_cho_rates_multi, p[None, :])[0]
            if not np.all(np.isfinite(pred)):
                return np.full(batch.n_obs_scalars, 1e3)
            r = (pred - batch.y_obs_pad) * sqrt_w[None, None, :]
            # Huberize residuals at 3.0 * Sigma_y for Stage-1 robustness
            r_hub = np.where(np.abs(r) <= 3.0, r, np.sign(r) * np.sqrt(np.maximum(6.0 * np.abs(r) - 9.0, 0.0)))
            return r_hub[batch.obs_mask].ravel()

        sol = least_squares(residuals, p0, bounds=(lb, ub), method="trf", max_nfev=60)
        self.p_mech = sol.x.copy()
        return self.p_mech

    def fit_stage2_bagged(self, dev_runs: Sequence[CHORun]) -> None:
        assert self.p_mech is not None
        rng = np.random.default_rng(self.seed)
        batch_exps = sorted(set(r.exp_id for r in dev_runs if r.mode == "batch"))
        fed_exps = sorted(set(r.exp_id for r in dev_runs if r.mode == "fed-batch"))
        all_exps = batch_exps + fed_exps
        runs_by_exp = {e: [r for r in dev_runs if r.exp_id == e] for e in all_exps}
        bounds = [(-1.5, 1.5)] * self.n_ml
        self.ensemble_thetas = []
        oob_sq_errs: List[np.ndarray] = []
        eps = 1e-4

        for b_idx in range(self.b_ens):
            t0 = time.perf_counter()
            # Regime-stratified bootstrap: always represent both Batch and Fed-Batch campaigns
            for _ in range(30):
                s_batch = rng.choice(batch_exps, size=len(batch_exps), replace=True).tolist()
                s_fed = rng.choice(fed_exps, size=len(fed_exps), replace=True).tolist() if fed_exps else []
                sampled_exps = s_batch + s_fed
                oob_exps = [e for e in all_exps if e not in set(sampled_exps)]
                if len(oob_exps) >= 1:
                    break

            boot_runs = [r for e in sampled_exps for r in runs_by_exp[e]]
            boot_batch = PreparedCHOBatch.from_runs(boot_runs, dt=1.0, mask_anomalies=True)
            theta0 = rng.normal(0.0, 0.05, size=self.n_ml)

            def obj_and_grad(th: np.ndarray) -> Tuple[float, np.ndarray]:
                k_tot = self.n_ml + 1
                ths = np.broadcast_to(th, (k_tot, self.n_ml)).copy()
                ths[1 + np.arange(self.n_ml), np.arange(self.n_ml)] += eps
                preds = rollout_vectorized_patankar(boot_batch, self.hybrid_rates_multi, ths)
                if not np.all(np.isfinite(preds)):
                    return 1e8, 1e3 * th
                err = preds[0] - boot_batch.y_obs_pad
                loss, d_err = _huber_loss_and_grad_coeff(err, self.sigma_y, boot_batch.obs_mask, delta=3.0)
                loss += 1e-3 * float(np.dot(th, th))
                w_sens = (preds[1:] - preds[0:1]) / eps  # (P, B, M, 4)
                grad = np.einsum("bmi,pbmi->p", d_err, w_sens) + 2e-3 * th
                return loss, grad

            sol = minimize(
                obj_and_grad,
                theta0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 18, "ftol": 1e-5, "gtol": 1e-3, "maxls": 6},
            )
            th_opt = sol.x.copy()
            self.ensemble_thetas.append(th_opt)

            if oob_exps:
                oob_runs = [r for e in oob_exps for r in runs_by_exp[e]]
                oob_batch = PreparedCHOBatch.from_runs(oob_runs, dt=0.5, mask_anomalies=True)
                p_oob = rollout_vectorized_patankar(oob_batch, self.hybrid_rates_multi, th_opt[None, :])[0]
                e_oob = (p_oob - oob_batch.y_obs_pad)[oob_batch.obs_mask]
                if len(e_oob) > 0:
                    oob_sq_errs.append(e_oob ** 2)

            print(
                f"  [hybrid-fedbatch-twin B={b_idx + 1}/{self.b_ens}] "
                f"boot_Huber={sol.fun:.4f} ({time.perf_counter() - t0:.2f}s)",
                flush=True,
            )

        if oob_sq_errs:
            pooled_oob = np.vstack(oob_sq_errs)
            # Robust OOB residual variance (winsorized at 95th percentile to guard OOB spikes)
            q95 = np.percentile(pooled_oob, 95.0, axis=0)
            self.oob_var = np.maximum(np.mean(np.minimum(pooled_oob, q95[None, :]), axis=0), self.sigma_y ** 2)
        else:
            self.oob_var = self.sigma_y ** 2

    def predict(self, runs: Sequence[CHORun]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        batch = PreparedCHOBatch.from_runs(runs, dt=0.5)
        ths = np.stack(self.ensemble_thetas, axis=0)
        preds_all = rollout_vectorized_patankar(batch, self.hybrid_rates_multi, ths)  # (B_ens, B, M_max, 4)
        means, stds = [], []
        for b_i, r in enumerate(runs):
            m = len(r.t_eval)
            sub = preds_all[:, b_i, :m, :]
            means.append(np.mean(sub, axis=0))
            var_ens = np.var(sub, axis=0, ddof=1)
            stds.append(np.sqrt(var_ens + self.oob_var[None, :]))
        return means, stds

    def compute_collinearity(self, runs: Sequence[CHORun]) -> float:
        batch = PreparedCHOBatch.from_runs(runs, dt=1.0, mask_anomalies=True)
        th = self.ensemble_thetas[0]
        eps = 1e-4
        ths = np.broadcast_to(th, (self.n_ml + 1, self.n_ml)).copy()
        ths[1 + np.arange(self.n_ml), np.arange(self.n_ml)] += eps
        preds = rollout_vectorized_patankar(batch, self.hybrid_rates_multi, ths)
        sqrt_w = np.sqrt(self.weights_y)
        w_sens = ((preds[1:] - preds[0:1]) / eps) * sqrt_w[None, None, None, :]
        s_mat = w_sens[:, batch.obs_mask, :].reshape(self.n_ml, -1).T
        return compute_collinearity_index(s_mat)


# =============================================================================
# SYSTEM B: `@von_stosch_twin` Parsimonious Sparse-BIC + OOB Bagged Baseline
# =============================================================================

class VonStoschCHOBaseline:
    """`@von_stosch_twin` Adversarial Benchmark (OLS-ERR + BIC + K_eff=9 OOB Bagging)."""

    def __init__(self, sigma_y: np.ndarray, b_cand: int = 12, b_ens: int = 8, seed: int = 2019) -> None:
        self.sigma_y = np.asarray(sigma_y, dtype=float)
        self.weights_y = 1.0 / (self.sigma_y ** 2)
        self.b_cand = int(b_cand)
        self.b_ens = int(b_ens)
        self.seed = int(seed)
        self.n_params = 9
        self.bounds = [
            (0.01, 0.12), (0.5, 20.0), (1.0, 40.0), (0.0005, 0.03),
            (0.3, 6.0), (0.001, 0.15), (0.05, 3.0), (0.0, 0.15), (0.05, 3.0),
        ]
        self.ensemble_thetas: List[np.ndarray] = []

    @staticmethod
    def rates_multi(x: np.ndarray, th: np.ndarray) -> Tuple[np.ndarray, ...]:
        g = np.maximum(x[:, :, 1], 0.0)
        l = np.maximum(x[:, :, 2], 0.0)
        a = np.maximum(x[:, :, 3], 0.0)
        mu = th[:, 0:1] * (g / (g + 0.3)) / ((1.0 + l / th[:, 1:2]) * (1.0 + a / th[:, 2:3]))
        kd = np.broadcast_to(th[:, 3:4], mu.shape)
        qg_tilde = -(mu / th[:, 4:5] + th[:, 5:6])
        ql_prod = th[:, 6:7] * mu
        ql_cons = th[:, 7:8] / (1.0 + g / 0.5)
        qa = th[:, 8:9] * mu + 0.002
        return mu, kd, qg_tilde, ql_prod, ql_cons, qa

    def run_structure_discrimination(self, dev_runs: Sequence[CHORun]) -> SparseRateDiscriminator:
        rng = np.random.default_rng(self.seed)
        g_list, l_list, a_list, mu_list = [], [], [], []
        for r in dev_runs:
            vcd = np.maximum(r.y_obs[:, 0], 0.05)
            t = r.t_eval
            for i in range(1, len(t) - 1):
                dt = t[i + 1] - t[i - 1]
                d_rate = r.f_in_offline[i] / max(r.v_grid[i], 0.2)
                mu_emp = (np.log(vcd[i + 1]) - np.log(vcd[i - 1])) / dt + d_rate
                if np.isfinite(mu_emp) and -0.05 < mu_emp < 0.15:
                    g_list.append(max(r.y_obs[i, 1], 0.0))
                    l_list.append(max(r.y_obs[i, 2], 0.0))
                    a_list.append(max(r.y_obs[i, 3], 0.0))
                    mu_list.append(mu_emp)

        g, l, a, mu_obs = np.array(g_list), np.array(l_list), np.array(a_list), np.array(mu_list)
        n = len(mu_obs)
        names = ["1", "G/(G+0.3)", "(G/(G+0.3))/(1+L/3)", "(G/(G+0.3))/((1+L/3)*(1+A/10))", "A/(A+5)", "decoy_1", "decoy_2"]
        phi = np.column_stack([
            np.ones(n), g / (g + 0.3), (g / (g + 0.3)) / (1.0 + l / 3.0),
            (g / (g + 0.3)) / ((1.0 + l / 3.0) * (1.0 + a / 10.0)),
            a / (a + 5.0), rng.normal(0.0, 1.0, size=n), rng.normal(0.0, 1.0, size=n),
        ])
        return SparseRateDiscriminator().fit(phi, mu_obs, term_names=names)

    def fit_bagged(self, dev_runs: Sequence[CHORun]) -> None:
        rng = np.random.default_rng(self.seed)
        exps = sorted(set(r.exp_id for r in dev_runs))
        runs_by_exp = {e: [r for r in dev_runs if r.exp_id == e] for e in exps}
        sqrt_w = np.sqrt(self.weights_y)
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        p0_base = np.array([0.045, 4.0, 8.0, 0.005, 1.5, 0.015, 0.6, 0.015, 0.5])

        records = []
        for c_id in range(self.b_cand):
            t0 = time.perf_counter()
            for _ in range(50):
                tr_exps = rng.choice(exps, size=len(exps), replace=True).tolist()
                oob_exps = [e for e in exps if e not in set(tr_exps)]
                if len(oob_exps) >= 2:
                    break
            tr_runs = [r for e in tr_exps for r in runs_by_exp[e]]
            oob_runs = [r for e in oob_exps for r in runs_by_exp[e]]
            tr_batch = PreparedCHOBatch.from_runs(tr_runs, dt=1.0)
            oob_batch = PreparedCHOBatch.from_runs(oob_runs, dt=0.5)
            th0 = np.clip(p0_base * rng.uniform(0.85, 1.15, size=self.n_params), lb + 1e-4, ub - 1e-4)

            def tr_res(th: np.ndarray) -> np.ndarray:
                pred = rollout_vectorized_patankar(tr_batch, self.rates_multi, th[None, :])[0]
                if not np.all(np.isfinite(pred)):
                    return np.full(tr_batch.n_obs_scalars, 1e3)
                err = (pred - tr_batch.y_obs_pad) * sqrt_w[None, None, :]
                return err[tr_batch.obs_mask].ravel()

            sol = least_squares(tr_res, th0, bounds=(lb, ub), method="trf", max_nfev=35)
            th_opt = sol.x.copy()

            p_tr = rollout_vectorized_patankar(tr_batch, self.rates_multi, th_opt[None, :])[0]
            p_oob = rollout_vectorized_patankar(oob_batch, self.rates_multi, th_opt[None, :])[0]
            e_tr = (p_tr - tr_batch.y_obs_pad) * tr_batch.obs_mask[:, :, None]
            e_oob = (p_oob - oob_batch.y_obs_pad) * oob_batch.obs_mask[:, :, None]
            tr_w = float(np.sum((e_tr ** 2) * self.weights_y[None, None, :])) / tr_batch.n_obs_scalars
            oob_w = float(np.sum((e_oob ** 2) * self.weights_y[None, None, :])) / oob_batch.n_obs_scalars
            records.append((0.5 * tr_w + 0.5 * oob_w, tr_w, oob_w, th_opt))
            print(
                f"  [@von_stosch_twin Boot {c_id + 1}/{self.b_cand}] "
                f"train_WMSE={tr_w:.4f}, oob_WMSE={oob_w:.4f} ({time.perf_counter() - t0:.2f}s)",
                flush=True,
            )

        min_oob = min(r[2] for r in records)
        kept = [r for r in records if r[2] <= 2.5 * min_oob]
        if len(kept) < self.b_ens:
            kept = sorted(records, key=lambda r: r[2])[: self.b_ens]
        kept.sort(key=lambda r: r[0])
        self.ensemble_thetas = [r[3] for r in kept[: self.b_ens]]

    def predict(self, runs: Sequence[CHORun]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        batch = PreparedCHOBatch.from_runs(runs, dt=0.5)
        ths = np.stack(self.ensemble_thetas, axis=0)
        preds_all = rollout_vectorized_patankar(batch, self.rates_multi, ths)
        means, stds = [], []
        for b_i, r in enumerate(runs):
            m = len(r.t_eval)
            sub = preds_all[:, b_i, :m, :]
            means.append(np.mean(sub, axis=0))
            stds.append(np.std(sub, axis=0, ddof=1))
        return means, stds

    def compute_collinearity(self, runs: Sequence[CHORun]) -> float:
        batch = PreparedCHOBatch.from_runs(runs, dt=1.0)
        th = self.ensemble_thetas[0]
        eps = 1e-4
        ths = np.broadcast_to(th, (self.n_params + 1, self.n_params)).copy()
        ths[1 + np.arange(self.n_params), np.arange(self.n_params)] += eps
        preds = rollout_vectorized_patankar(batch, self.rates_multi, ths)
        sqrt_w = np.sqrt(self.weights_y)
        w_sens = ((preds[1:] - preds[0:1]) / eps) * sqrt_w[None, None, None, :]
        s_mat = w_sens[:, batch.obs_mask, :].reshape(self.n_params, -1).T
        return compute_collinearity_index(s_mat)


# =============================================================================
# Padé EKF + Causal Inline PAT Observer
# =============================================================================

def run_pade_ekf_stress_test(
    twin: HybridFedBatchTwinCHO,
    dev_runs: Sequence[CHORun],
    test_runs: Sequence[CHORun],
) -> Dict[str, float]:
    """Causal PAT -> VCD ridge observer + Continuous-Discrete unclipped Padé expm EKF."""
    x_pat_list, y_vcd_list = [], []
    for r in dev_runs:
        if is_biophysical_anomaly_run(r):
            continue
        valid = r.pat_valid_7
        if np.any(valid):
            x_pat_list.append(r.pat_features[valid][:, [0, 3, 5, 6]])
            y_vcd_list.append(r.y_obs[valid, 0])

    x_pat = np.vstack(x_pat_list)
    y_vcd = np.concatenate(y_vcd_list)
    pat_mean = np.mean(x_pat, axis=0)
    pat_std = np.maximum(np.std(x_pat, axis=0), 1e-6)
    x_pat_norm = np.column_stack([np.ones(len(x_pat)), (x_pat - pat_mean) / pat_std])
    w_pat = np.linalg.solve(x_pat_norm.T @ x_pat_norm + np.eye(5), x_pat_norm.T @ y_vcd)
    pat_rmse_dev = float(np.sqrt(np.mean((x_pat_norm @ w_pat - y_vcd) ** 2)))

    ths_ens = np.stack(twin.ensemble_thetas, axis=0)
    q_base = np.diag([0.04, 0.02, 0.02, 0.04])
    r_meas = np.array([[max(pat_rmse_dev ** 2, 1.0)]])
    h_mat = np.array([[1.0, 0.0, 0.0, 0.0]])

    vcd_sq_err, vcd_pts, dropout_steps, nan_count, max_p_trace = 0.0, 0, 0, 0, 0.0
    ekf_trajectories: Dict[str, Dict[str, List[float]]] = {}

    for r in test_runs:
        x = np.maximum(r.y_obs[0].copy(), 0.0)
        p_cov = np.diag([0.25, 0.1, 0.1, 0.1])
        vol = float(r.v_grid[0])
        t_rec, vcd_rec, drop_rec = [0.0], [float(x[0])], [0]

        t_steps = np.arange(0.0, float(r.t_eval[-1]) + 0.1, 2.0)
        for idx_s in range(len(t_steps) - 1):
            t0, t1 = t_steps[idx_s], t_steps[idx_s + 1]
            dt = t1 - t0
            win_f = (r.t_inline >= t0) & (r.t_inline <= t1)
            f_in = float(np.nanmean(r.f_in_inline[win_f])) if np.any(win_f) else 0.0

            def rhs(xv: np.ndarray) -> np.ndarray:
                x_b = np.broadcast_to(xv[None, None, :], (len(ths_ens), 1, 4))
                mu_k, kd_k, qg_k, qlp_k, qlc_k, qa_k = twin.hybrid_rates_multi(x_b, ths_ens)
                mu = float(np.mean(mu_k))
                kd = float(np.mean(kd_k))
                qg_t = float(np.mean(qg_k))
                ql_p = float(np.mean(qlp_k))
                ql_c = float(np.mean(qlc_k))
                qa = float(np.mean(qa_k))
                d_r = f_in / max(vol, 0.2)
                g_s = max(xv[1], 0.0) / (max(xv[1], 0.0) + 0.05)
                l_s = max(xv[2], 0.0) / (max(xv[2], 0.0) + 0.1)
                return np.array([
                    (mu - kd - d_r) * xv[0],
                    qg_t * g_s * xv[0] + d_r * (r.c_glc_feed - xv[1]),
                    (ql_p - ql_c * l_s) * xv[0] - d_r * xv[2],
                    qa * xv[0] - d_r * xv[3],
                ])

            k1 = rhs(x)
            k2 = rhs(np.maximum(x + dt * k1, 0.0))
            x_pred = np.maximum(x + 0.5 * dt * (k1 + k2), 0.0)
            x_mid = 0.5 * (x + x_pred)

            f_jac = np.zeros((4, 4))
            for j in range(4):
                eps = 1e-4 * max(1.0, abs(x_mid[j]))
                xp, xm = x_mid.copy(), x_mid.copy()
                xp[j] += eps
                xm[j] = max(0.0, xm[j] - eps)
                f_jac[:, j] = (rhs(xp) - rhs(xm)) / max(xp[j] - xm[j], 1e-8)

            phi = expm(f_jac * dt)
            p_cov = 0.5 * (phi @ p_cov @ phi.T + q_base * dt + (phi @ p_cov @ phi.T + q_base * dt).T)
            x = x_pred
            vol = float(np.interp(t1, r.t_inline, r.v_inline))

            pat_win = (r.t_inline >= t1 - 0.5) & (r.t_inline <= t1)
            is_drop = 0
            if np.any(pat_win) and np.all(r.pat_inline_valid[pat_win]):
                feat_4 = np.mean(r.pat_inline[pat_win], axis=0)
                vcd_pat = max(0.0, float(np.dot(w_pat, np.concatenate([[1.0], (feat_4 - pat_mean) / pat_std]))))
                s_cov = h_mat @ p_cov @ h_mat.T + r_meas
                k_gain = np.linalg.solve(s_cov, h_mat @ p_cov).T
                x = np.maximum(x + (k_gain @ (np.array([vcd_pat]) - h_mat @ x)).ravel(), 0.0)
                ikh = np.eye(4) - k_gain @ h_mat
                p_cov = 0.5 * (ikh @ p_cov @ ikh.T + k_gain @ r_meas @ k_gain.T + (ikh @ p_cov @ ikh.T + k_gain @ r_meas @ k_gain.T).T)
            else:
                dropout_steps += 1
                is_drop = 1

            t_rec.append(round(float(t1), 2))
            vcd_rec.append(round(float(x[0]), 4))
            drop_rec.append(is_drop)

            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p_cov)):
                nan_count += 1
            else:
                max_p_trace = max(max_p_trace, float(np.trace(p_cov)))

            diffs_t = np.abs(r.t_eval - t1)
            if np.min(diffs_t) <= 1.0:
                vcd_sq_err += (x[0] - r.y_obs[int(np.argmin(diffs_t)), 0]) ** 2
                vcd_pts += 1

        ekf_trajectories[f"{r.exp_id}_{r.rep_id}"] = {
            "t": t_rec,
            "vcd": vcd_rec,
            "dropout": drop_rec,
        }

    return {
        "pat_soft_sensor_dev_rmse": pat_rmse_dev,
        "ekf_vcd_test_rmse": float(np.sqrt(vcd_sq_err / max(vcd_pts, 1))),
        "ekf_dropout_steps_survived": float(dropout_steps),
        "ekf_nan_count": float(nan_count),
        "ekf_max_p_trace": max_p_trace,
        "ekf_trajectories": ekf_trajectories,
    }


def compute_metrics(preds: Sequence[np.ndarray], runs: Sequence[CHORun], weights_y: np.ndarray) -> Dict[str, object]:
    y_all = np.vstack([r.y_obs for r in runs])
    p_all = np.vstack(preds)
    err = p_all - y_all
    wmse = float(np.mean((err ** 2) * weights_y[None, :]))
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_all - np.mean(y_all, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {"wmse": wmse, "rmse": rmse, "r2": r2, "min_x": float(np.min(p_all))}


def main() -> None:
    t_wall0 = time.perf_counter()
    print("=" * 78, flush=True)
    print("ADVERSARIAL WET-LAB STRESS TEST: hybrid-fedbatch-twin vs @von_stosch_twin", flush=True)
    print("Dataset: Uhlendorff et al. (2026) CHO-K1 (12 EXPs, 24 Runs, 48 CSVs)", flush=True)
    print("=" * 78, flush=True)

    dev_runs, test_runs, stats = load_uhlendorff_dataset(c_glc_feed_est=18.0)
    sigma_y_clean = compute_duplicate_noise_floor(dev_runs, exclude_anomalous=True)
    sigma_y_unmasked = compute_duplicate_noise_floor(dev_runs, exclude_anomalous=False)
    anomalous_run_ids = [f"{r.exp_id}_{r.rep_id}" for r in dev_runs if is_biophysical_anomaly_run(r)]
    dev_clean = [r for r in dev_runs if not is_biophysical_anomaly_run(r)]

    print(f"[Gate 1] Dev Campaigns ({len(DEV_EXPS)} EXPs = {len(dev_runs)} runs): {DEV_EXPS}", flush=True)
    print(f"[Gate 1] Held-Out Test ({len(TEST_EXPS)} EXPs = {len(test_runs)} runs): {TEST_EXPS}", flush=True)
    print(f"[Gate 1] Duplicate Noise Floor Sigma_y (clean Dev): {np.round(sigma_y_clean, 4)} -> [VCD, Glc, Lac, Amm]", flush=True)
    print(f"[Gate 1] Duplicate Noise Floor Sigma_y (unmasked): {np.round(sigma_y_unmasked, 4)}", flush=True)
    print(
        f"[Gate 2] Offline rows={stats['total_offline_rows']}, "
        f"Incomplete 7-sensor inline windows [t-0.5h,t]={stats['incomplete_7_sensor_rows']}, "
        f"Flagged anomaly runs={anomalous_run_ids} (rows={stats['anomaly_rows']})\n",
        flush=True,
    )

    print("-" * 78, flush=True)
    print("STEP 2: Training `hybrid-fedbatch-twin` (Decoupled Mech + RBF Residual, K_eff=9, B=15)", flush=True)
    print("-" * 78, flush=True)
    twin_clean = HybridFedBatchTwinCHO(sigma_y=sigma_y_clean, b_ens=15, h_nodes=4, seed=42)
    twin_clean.fit_stage1(dev_clean)
    print(f"Stage-1 Mechanistic params (9): {np.round(twin_clean.p_mech, 4)}", flush=True)
    twin_clean.fit_stage2_bagged(dev_clean)
    print(f"OOB Epistemic Std Floor: {np.round(np.sqrt(twin_clean.oob_var), 4)}", flush=True)

    print(f"\nTraining unmasked `hybrid-fedbatch-twin` (with anomalous runs {anomalous_run_ids}, B=5)...", flush=True)
    twin_unmasked = HybridFedBatchTwinCHO(sigma_y=sigma_y_clean, b_ens=5, h_nodes=4, seed=42)
    twin_unmasked.fit_stage1(dev_runs)
    twin_unmasked.fit_stage2_bagged(dev_runs)

    print("\n" + "-" * 78, flush=True)
    print("STEP 3: Training `@von_stosch_twin` Parsimonious Baseline (OLS-ERR + BIC + OOB Bagging)", flush=True)
    print("-" * 78, flush=True)
    vs_base = VonStoschCHOBaseline(sigma_y=sigma_y_clean, b_cand=12, b_ens=8, seed=2019)
    disc = vs_base.run_structure_discrimination(dev_clean)
    print(f"OLS-ERR + BIC Selected Terms: {disc.selected_names_}", flush=True)
    print(f"OLS-ERR Values: {[round(e, 4) for e in disc.err_history_]}", flush=True)
    print(f"OLS-ERR BIC Progression: {[round(b, 2) for b in disc.bic_history_]}", flush=True)
    print(f"OLS-ERR Collinearity gamma_col: {disc.gamma_col_:.4f}", flush=True)
    vs_base.fit_bagged(dev_clean)

    ht_means, ht_stds = twin_clean.predict(test_runs)
    ht_unm_means, _ = twin_unmasked.predict(test_runs)
    vs_means, vs_stds = vs_base.predict(test_runs)

    m_ht = compute_metrics(ht_means, test_runs, twin_clean.weights_y)
    m_ht_unm = compute_metrics(ht_unm_means, test_runs, twin_clean.weights_y)
    m_vs = compute_metrics(vs_means, test_runs, twin_clean.weights_y)

    # Dynamic Stage-1 mechanistic-only baseline on held-out test set
    test_batch_eval = PreparedCHOBatch.from_runs(test_runs, dt=0.5)
    stage1_test_pred_pad = rollout_vectorized_patankar(test_batch_eval, mech_cho_rates_multi, twin_clean.p_mech[None, :])[0]
    stage1_test_preds = [stage1_test_pred_pad[i, : len(r.t_eval), :] for i, r in enumerate(test_runs)]
    m_stage1 = compute_metrics(stage1_test_preds, test_runs, twin_clean.weights_y)

    gamma_ht = twin_clean.compute_collinearity(dev_clean)
    gamma_vs = vs_base.compute_collinearity(dev_clean)

    per_campaign_ht: Dict[str, Dict[str, object]] = {}
    per_campaign_vs: Dict[str, Dict[str, object]] = {}
    for exp_id in sorted({r.exp_id for r in test_runs}):
        idxs = [i for i, r in enumerate(test_runs) if r.exp_id == exp_id]
        per_campaign_ht[exp_id] = compute_metrics([ht_means[i] for i in idxs], [test_runs[i] for i in idxs], twin_clean.weights_y)
        per_campaign_vs[exp_id] = compute_metrics([vs_means[i] for i in idxs], [test_runs[i] for i in idxs], twin_clean.weights_y)

    fedbatch_test_exps = sorted({r.exp_id for r in test_runs if r.mode == "fed-batch"})
    batch_test_exps = sorted({r.exp_id for r in test_runs if r.mode == "batch"})
    fb_key = fedbatch_test_exps[0] if fedbatch_test_exps else sorted(per_campaign_ht.keys())[-1]
    shift_key = batch_test_exps[-1] if batch_test_exps else sorted(per_campaign_ht.keys())[0]
    m_ht_fb = per_campaign_ht[fb_key]
    m_vs_fb = per_campaign_vs[fb_key]
    m_ht_shift = per_campaign_ht[shift_key]
    m_vs_shift = per_campaign_vs[shift_key]

    in_ci_ht = np.concatenate([(np.abs(r.y_obs - mu) <= 1.96 * sd).ravel() for r, mu, sd in zip(test_runs, ht_means, ht_stds)])
    in_ci_vs = np.concatenate([(np.abs(r.y_obs - mu) <= 1.96 * sd).ravel() for r, mu, sd in zip(test_runs, vs_means, vs_stds)])
    cov_ht_pct = 100.0 * float(np.mean(in_ci_ht))
    cov_vs_pct = 100.0 * float(np.mean(in_ci_vs))

    ekf_res = run_pade_ekf_stress_test(twin_clean, dev_clean, test_runs)

    print("\n" + "=" * 78, flush=True)
    print(f"ADVERSARIAL STRESS-TEST REPORT (`@von_stosch_twin` Referee) [{time.perf_counter() - t_wall0:.1f}s]", flush=True)
    print("=" * 78, flush=True)

    print(f"\n[Stress Test 1] Held-Out Generalization ({len(test_runs)} Test Runs: {', '.join(TEST_EXPS)})", flush=True)
    print(f"{'State':<14s} | {'Sigma_y':>8s} | {'HT RMSE':>8s} | {'HT R2':>7s} | {'VS RMSE':>8s} | {'VS R2':>7s}", flush=True)
    print("-" * 68, flush=True)
    for i, nm in enumerate(STATE_NAMES):
        print(
            f"{nm:<14s} | {sigma_y_clean[i]:8.3f} | {m_ht['rmse'][i]:8.3f} | {m_ht['r2'][i]:7.3f} | "
            f"{m_vs['rmse'][i]:8.3f} | {m_vs['r2'][i]:7.3f}",
            flush=True,
        )
    print("-" * 68, flush=True)
    print(f"Pooled Test WMSE: hybrid-fedbatch-twin = {m_ht['wmse']:.4f} | @von_stosch_twin = {m_vs['wmse']:.4f}", flush=True)

    print("\n[Stress Test 2] Identifiability & Collinearity Gate (gamma_col < 20)", flush=True)
    print(f"  hybrid-fedbatch-twin: K_eff = {twin_clean.n_ml} (decoupled Stage-2; Stage-1 priors locked), gamma_col = {gamma_ht:.2f} -> {'PASS' if gamma_ht < 20 else 'FAIL (OVER-PARAMETERIZED)'}", flush=True)
    print(f"  @von_stosch_twin    : K_eff = {vs_base.n_params}, gamma_col = {gamma_vs:.2f} -> {'PASS' if gamma_vs < 20 else 'FAIL'}", flush=True)

    print(f"\n[Stress Test 3] Fed-Batch Transition ({fb_key}) & Positivity", flush=True)
    print(f"  min(x_pred) across all rollouts: hybrid-fedbatch-twin = {m_ht['min_x']:.4f} | @von_stosch_twin = {m_vs['min_x']:.4f}", flush=True)
    print(f"  {fb_key} Test WMSE             : hybrid-fedbatch-twin = {m_ht_fb['wmse']:.4f} | @von_stosch_twin = {m_vs_fb['wmse']:.4f}", flush=True)
    print(f"  {fb_key} Glc/Lac RMSE          : HT = {np.round(m_ht_fb['rmse'][1:3], 3)} | VS = {np.round(m_vs_fb['rmse'][1:3], 3)}", flush=True)

    print(f"\n[Stress Test 4] {shift_key} Shift & Anomaly Survival ({anomalous_run_ids})", flush=True)
    print(f"  {shift_key} Shifted Batch WMSE    : hybrid-fedbatch-twin = {m_ht_shift['wmse']:.4f} | @von_stosch_twin = {m_vs_shift['wmse']:.4f}", flush=True)
    print(f"  Anomaly Masked WMSE            : {m_ht['wmse']:.4f}", flush=True)
    print(f"  Anomaly Unmasked WMSE          : {m_ht_unm['wmse']:.4f} (degradation: {m_ht_unm['wmse'] - m_ht['wmse']:+.4f})", flush=True)

    print("\n[Stress Test 5] Epistemic B=15 Calibration & Pade EKF Dropout Resilience", flush=True)
    print(f"  95% CI Coverage (vs Test Obs)  : hybrid-fedbatch-twin = {cov_ht_pct:.2f}% | @von_stosch_twin = {cov_vs_pct:.2f}%", flush=True)
    print(f"  Causal PAT->VCD Dev RMSE       : {ekf_res['pat_soft_sensor_dev_rmse']:.3f} x 1e6 cells/mL", flush=True)
    print(f"  Pade EKF Test VCD RMSE         : {ekf_res['ekf_vcd_test_rmse']:.3f} x 1e6 cells/mL", flush=True)
    print(f"  EKF Dropout Steps Survived     : {int(ekf_res['ekf_dropout_steps_survived'])} (NaNs={int(ekf_res['ekf_nan_count'])}, max tr(P)={ekf_res['ekf_max_p_trace']:.3f})", flush=True)

    s_gen = max(0.0, min(2.0, 2.0 * (1.0 - max(0.0, (m_ht["wmse"] - m_vs["wmse"]) / max(m_vs["wmse"], 1.0)))))
    s_id = 2.0 if gamma_ht < 20.0 else (1.0 if gamma_ht < 100.0 else 0.3)
    s_pos = 2.0 if (m_ht["min_x"] >= 0.0 and m_ht_fb["wmse"] < 30.0) else 1.0
    s_anom = 2.0 if abs(m_ht_unm["wmse"] - m_ht["wmse"]) < 2.0 else (1.0 if m_ht_unm["wmse"] < 50.0 else 0.5)
    s_ekf = 2.0 if (ekf_res["ekf_nan_count"] == 0 and cov_ht_pct >= 50.0) else 1.0
    total_score = s_gen + s_id + s_pos + s_anom + s_ekf

    print("\n" + "=" * 78, flush=True)
    print("@von_stosch_twin ITEMIZED SURVIVAL SCORECARD (0.0 - 10.0)", flush=True)
    print("=" * 78, flush=True)
    print(f"  1. Held-Out Generalization vs Sparse Baseline : {s_gen:.2f} / 2.00", flush=True)
    print(f"  2. Structural Identifiability (gamma_col < 20): {s_id:.2f} / 2.00 (gamma_col={gamma_ht:.1f})", flush=True)
    print(f"  3. Fed-Batch Mass Balance & Positivity        : {s_pos:.2f} / 2.00", flush=True)
    print(f"  4. Shift & Biophysical Anomaly Survival       : {s_anom:.2f} / 2.00", flush=True)
    print(f"  5. Epistemic B=15 CI & Pade EKF Dropouts      : {s_ekf:.2f} / 2.00", flush=True)
    print("-" * 78, flush=True)
    verdict = "GO (PASS)" if total_score >= 7.0 else "CONDITIONAL / NO-GO (REMEDIATE COLLINEARITY & ANOMALY GUARD)"
    print(f"  FINAL SURVIVAL VERDICT: {total_score:.2f} / 10.00 -> {verdict}", flush=True)
    print("=" * 78, flush=True)

    # Export deterministic wet-lab validation JSON + JS bundle computed strictly from observed runs
    vol_series = {}
    clean_fb_r1 = [r for r in (dev_clean + test_runs) if r.mode == "fed-batch" and r.rep_id == "R01"]
    for r1 in clean_fb_r1:
        t_ds = np.linspace(0.0, float(r1.t_eval[-1]), 40)
        v_true = np.interp(t_ds, r1.t_inline, r1.v_inline)
        f_ds = np.interp(t_ds, r1.t_inline, r1.f_in_inline)
        dt_ds = t_ds[1] - t_ds[0]
        v_naive = float(r1.v_grid[0]) + np.cumsum(np.r_[0.0, 0.5 * (f_ds[:-1] + f_ds[1:]) * dt_ds])
        vol_series[r1.exp_id] = {
            "t": np.round(t_ds, 2).tolist(),
            "v_true": np.round(v_true, 4).tolist(),
            "v_naive": np.round(v_naive, 4).tolist(),
        }

    # Dynamically compute mass balance & dev fit diagnostics from actual runs and model
    dev_ht_means, _ = twin_clean.predict(dev_clean)
    m_dev_after = compute_metrics(dev_ht_means, dev_clean, twin_clean.weights_y)
    dev_batch_eval = PreparedCHOBatch.from_runs(dev_clean, dt=0.5)
    dev_s1_pad = rollout_vectorized_patankar(dev_batch_eval, mech_cho_rates_multi, twin_clean.p_mech[None, :])[0]
    dev_s1_preds = [dev_s1_pad[i, : len(r.t_eval), :] for i, r in enumerate(dev_clean)]
    m_dev_before = compute_metrics(dev_s1_preds, dev_clean, twin_clean.weights_y)

    fb_ref = clean_fb_r1[0]
    dv_inl = np.diff(fb_ref.v_inline)
    v_naive_ref = vol_series[fb_ref.exp_id]["v_naive"][-1]
    mass_balance_summary = {
        "reference_campaign": fb_ref.exp_id,
        "v0": round(float(fb_ref.v_inline[0]), 4),
        "v_end_true": round(float(fb_ref.v_inline[-1]), 4),
        "v_end_naive": round(float(v_naive_ref), 4),
        "sum_pos_dv": round(float(np.sum(dv_inl[dv_inl > 0.0])), 4),
        "sum_neg_dv": round(float(np.sum(dv_inl[dv_inl < 0.0])), 4),
        "n_samples": int(len(fb_ref.t_eval)),
        "c_glc_feed_calibrated_g_per_L": float(fb_ref.c_glc_feed),
        "dev_wmse_before": round(float(m_dev_before["wmse"]), 4),
        "dev_wmse_after": round(float(m_dev_after["wmse"]), 4),
        "dev_vcd_r2_after": round(float(m_dev_after["r2"][0]), 4),
        "dev_glc_r2_after": round(float(m_dev_after["r2"][1]), 4),
        "volume_series": vol_series,
    }

    held_out_traj = {}
    for exp_id in TEST_EXPS:
        idxs = [i for i, r in enumerate(test_runs) if r.exp_id == exp_id]
        i1 = next(i for i in idxs if test_runs[i].rep_id == "R01")
        i2 = next(i for i in idxs if test_runs[i].rep_id == "R02")
        r1, r2 = test_runs[i1], test_runs[i2]
        held_out_traj[exp_id] = {
            "mode": r1.mode,
            "t_feed_start": None if r1.mode == "batch" else float(r1.t_feed_start),
            "t_off_r01": np.round(r1.t_eval, 2).tolist(),
            "obs_r01": np.round(r1.y_obs, 4).tolist(),
            "t_off_r02": np.round(r2.t_eval, 2).tolist(),
            "obs_r02": np.round(r2.y_obs, 4).tolist(),
            "ht_mean": np.round(ht_means[i1], 4).tolist(),
            "ht_std": np.round(ht_stds[i1], 4).tolist(),
            "vs_mean": np.round(vs_means[i1], 4).tolist(),
            "ekf_r01": ekf_res["ekf_trajectories"][f"{exp_id}_R01"],
        }

    payload = {
        "dataset_metadata": {
            "source": "Uhlendorff et al. (2026) CHO-K1 (10.5281/zenodo.20829178)",
            "dev_campaigns": DEV_EXPS,
            "test_campaigns": TEST_EXPS,
            "anomalous_runs": anomalous_run_ids,
            "sigma_y_clean": np.round(sigma_y_clean, 4).tolist(),
            "sigma_y_unmasked": np.round(sigma_y_unmasked, 4).tolist(),
            "total_offline_rows": int(stats["total_offline_rows"]),
            "incomplete_7_sensor_rows": int(stats["incomplete_7_sensor_rows"]),
            "anomaly_rows": int(stats["anomaly_rows"]),
        },
        "mass_balance_discovery": mass_balance_summary,
        "benchmark_comparison": {
            "stage1_mechanistic_only": {
                "k_eff": int(len(twin_clean.p_mech)),
                "pooled_test_wmse": round(float(m_stage1["wmse"]), 4),
                "rmse": np.round(m_stage1["rmse"], 3).tolist(),
                "r2": np.round(m_stage1["r2"], 3).tolist(),
            },
            "remediated": {
                "k_eff": int(twin_clean.n_ml),
                "gamma_col": round(float(gamma_ht), 2),
                "pooled_test_wmse": round(float(m_ht["wmse"]), 4),
                "rmse": np.round(m_ht["rmse"], 3).tolist(),
                "r2": np.round(m_ht["r2"], 3).tolist(),
                "min_x": round(float(m_ht["min_x"]), 4),
                "per_campaign_test_wmse": {
                    eid: round(float(met["wmse"]), 4) for eid, met in per_campaign_ht.items()
                },
                "fedbatch_test_wmse": round(float(m_ht_fb["wmse"]), 4),
                "shifted_batch_wmse": round(float(m_ht_shift["wmse"]), 4),
                "anomaly_unmasked_wmse": round(float(m_ht_unm["wmse"]), 4),
                "anomaly_unmasked_deg": round(float(m_ht_unm["wmse"] - m_ht["wmse"]), 4),
                "ci_coverage_pct": round(float(cov_ht_pct), 2),
                "pat_dev_rmse": round(float(ekf_res["pat_soft_sensor_dev_rmse"]), 3),
                "ekf_vcd_rmse": round(float(ekf_res["ekf_vcd_test_rmse"]), 3),
                "ekf_dropout_steps": int(ekf_res["ekf_dropout_steps_survived"]),
                "ekf_nans": int(ekf_res["ekf_nan_count"]),
                "ekf_max_p_trace": round(float(ekf_res["ekf_max_p_trace"]), 3),
                "scorecard": {
                    "generalization": round(s_gen, 2),
                    "identifiability": round(s_id, 2),
                    "fedbatch_positivity": round(s_pos, 2),
                    "shift_and_anomaly": round(s_anom, 2),
                    "ci_and_ekf": round(s_ekf, 2),
                },
                "score": round(float(total_score), 2),
                "verdict": verdict,
            },
            "von_stosch_referee": {
                "k_eff": int(vs_base.n_params),
                "ols_err_terms": disc.selected_names_,
                "ols_err_gamma_col": round(float(disc.gamma_col_), 4),
                "ode_gamma_col": round(float(gamma_vs), 2),
                "pooled_test_wmse": round(float(m_vs["wmse"]), 4),
                "rmse": np.round(m_vs["rmse"], 3).tolist(),
                "r2": np.round(m_vs["r2"], 3).tolist(),
                "per_campaign_test_wmse": {
                    eid: round(float(met["wmse"]), 4) for eid, met in per_campaign_vs.items()
                },
                "fedbatch_test_wmse": round(float(m_vs_fb["wmse"]), 4),
                "shifted_batch_wmse": round(float(m_vs_shift["wmse"]), 4),
                "ci_coverage_pct": round(float(cov_vs_pct), 2),
            },
        },
        "held_out_trajectories": held_out_traj,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", "uhlendorff_2026_validation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.makedirs(os.path.join("docs", "assets"), exist_ok=True)
    with open(os.path.join("docs", "assets", "uhlendorff_2026_validation.js"), "w", encoding="utf-8") as f:
        f.write("window.UHLENDORFF_2026_VALIDATION = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print("Exported artifacts/uhlendorff_2026_validation.json and docs/assets/uhlendorff_2026_validation.js", flush=True)


if __name__ == "__main__":
    main()
