"""Stage-1 iDoE parameter recovery and Stage-2 bootstrap-aggregated GAM-86 Neural ODE."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import least_squares, minimize

from hybrid_fedbatch_twin.mechanistic import (
    BioreactorParams,
    OperatingInputs,
    compute_hydrodynamics,
    default_initial_state,
    nominal_kla,
    simulate_bioreactor,
)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _phi1_and_grad(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate phi1(z) = (1 - exp(-z)) / z and d(phi1)/dz smoothly near z = 0."""
    z_arr = np.asarray(z, dtype=float)
    small = np.abs(z_arr) <= 1e-5
    z_safe = np.where(small, 1.0, z_arr)
    exp_neg = np.exp(-np.clip(z_safe, -50.0, 80.0))

    phi_main = -np.expm1(-np.clip(z_safe, -50.0, 80.0)) / z_safe
    dphi_main = ((1.0 + z_safe) * exp_neg - 1.0) / (z_safe ** 2)

    phi_small = 1.0 - 0.5 * z_arr + (z_arr ** 2) / 6.0 - (z_arr ** 3) / 24.0
    dphi_small = -0.5 + z_arr / 3.0 - (z_arr ** 2) / 8.0

    return np.where(small, phi_small, phi_main), np.where(small, dphi_small, dphi_main)


# =============================================================================
# Stage 1: Multi-Step Intensified DoE (iDoE) 6-Parameter Recovery (V17)
# =============================================================================

_STAGE1_THETA_INIT = np.array([0.33, 0.20, 0.60, 0.25, 0.033, 500.0], dtype=float)
_STAGE1_LOWER = np.array([0.15, 0.02, 0.20, 0.05, 0.001, 200.0], dtype=float)
_STAGE1_UPPER = np.array([0.95, 1.00, 0.90, 1.00, 0.150, 1400.0], dtype=float)
_STAGE1_PARAM_NAMES = ["mu_max", "K_S", "Y_XS", "alpha", "beta", "k_La0"]


def _stage1_exp_rosenbrock_rollout(
    t_grid: np.ndarray,
    x0: np.ndarray,
    F_arr: np.ndarray,
    kla_ref: np.ndarray,
    S_feed: float,
    params: BioreactorParams,
    n_substeps: int = 4,
) -> np.ndarray:
    """Fast honest Stage-1 Exponential-Rosenbrock rollout (uses params.k_d0 only)."""
    n = len(t_grid)
    states = np.zeros((n, 7), dtype=float)
    states[0] = np.maximum(0.0, np.asarray(x0, dtype=float))
    x = states[0].copy()
    dt_arr = np.diff(t_grid) / max(1, int(n_substeps))

    k_d = float(params.k_d0)
    inv_Yxs = 1.0 / params.Y_XS
    inv_Yxo = 1.0 / params.Y_XO
    inv_Yps = 1.0 / params.Y_PS
    kla_ratio = params.k_La0 / 650.0

    for i in range(n - 1):
        dt = float(dt_arr[i])
        F_in = float(F_arr[i])
        kla_base = float(kla_ref[i]) * kla_ratio
        for _ in range(n_substeps):
            Xv, Xd, S, CL, A, P, V = (
                max(0.0, float(x[0])),
                max(0.0, float(x[1])),
                max(0.0, float(x[2])),
                max(0.0, float(x[3])),
                max(0.0, float(x[4])),
                max(0.0, float(x[5])),
                max(0.2, float(x[6])),
            )
            F = 0.0 if V >= params.V_max else max(0.0, F_in)
            D = F / V
            kla = kla_base

            dk = params.K_S + S
            do = params.K_O2 + CL
            s_m = S / dk
            o_m = CL / do
            d_sm = params.K_S / (dk * dk)
            d_om = params.K_O2 / (do * do)

            mu = params.mu_max * s_m * o_m
            dmu_dS = params.mu_max * d_sm * o_m
            dmu_dCL = params.mu_max * s_m * d_om

            sa = S / (0.002 + S)
            dsa = 0.002 / ((0.002 + S) ** 2)
            oa = CL / (0.005 + CL)
            doa = 0.005 / ((0.005 + CL) ** 2)

            qP = (params.alpha * mu + params.beta) * sa * oa
            dqP_dS = params.alpha * dmu_dS * sa * oa + (params.alpha * mu + params.beta) * dsa * oa
            qS = mu * inv_Yxs + params.m_s * sa + qP * inv_Yps
            dqS_dS = dmu_dS * inv_Yxs + params.m_s * dsa + dqP_dS * inv_Yps
            our = 1000.0 * (mu * inv_Yxo + params.m_o * oa) * Xv
            dour_dCL = 1000.0 * (dmu_dCL * inv_Yxo + params.m_o * doa) * Xv

            z2 = (D + Xv * dqS_dS) * dt
            phi2 = 1.0 if abs(z2) < 1e-6 else (1.0 - np.exp(-z2)) / z2
            z3 = (kla + D + dour_dCL) * dt
            phi3 = 1.0 if abs(z3) < 1e-6 else (1.0 - np.exp(-z3)) / z3

            x[0] = max(0.0, Xv + dt * (mu - k_d - D) * Xv)
            x[1] = max(0.0, Xd + dt * (k_d * Xv - D * Xd))
            x[2] = max(0.0, S + dt * phi2 * (D * (S_feed - S) - qS * Xv))
            x[3] = min(
                params.C_star * 1.05,
                max(0.0, CL + dt * phi3 * (kla * (params.C_star - CL) - our - D * CL)),
            )
            x[4] = max(0.0, A * (1.0 - dt * D))
            x[5] = max(0.0, P + dt * (qP * Xv - D * P))
            x[6] = min(params.V_max, V + dt * F)
        states[i + 1] = x
    return states


def _build_stage1_idoe_batches(
    p_true: BioreactorParams,
    noise_std: float,
    rng: np.random.Generator,
    sp_stride: int = 5,
) -> List[Dict[str, object]]:
    """Generate 3 orthogonal low-acetate V17 iDoE batches exciting all 6 kinetic/mass-transfer parameters."""
    p_ref = replace(p_true, k_La0=650.0)
    specs = [
        (
            "B1_high_do",
            (0.0, 9.5),
            np.array([0.80, 0.01, 0.36, 1.15, 0.0, 0.0, p_true.V_0], dtype=float),
            lambda t, s: OperatingInputs(
                F=float(0.0040 * np.exp(0.295 * t) if t < 7.5 else 0.036),
                rpm=175.0 if t < 1.1 else (240.0 if t < 2.7 else 520.0),
                vvm=0.24 if t < 1.1 else (0.52 if t < 2.7 else 3.5),
            ),
        ),
        (
            "B2_low_mu",
            (0.0, 14.5),
            np.array([5.00, 0.01, 0.060, p_true.C_star * 0.95, 0.0, 0.0, p_true.V_0], dtype=float),
            lambda t, s: OperatingInputs(
                F=float(0.0124 * np.exp(0.103 * t)),
                rpm=540.0,
                vvm=3.5,
            ),
        ),
        (
            "B3_ks_idoe",
            (0.0, 11.5),
            np.array([1.10, 0.01, 0.30, 1.40, 0.0, 0.0, p_true.V_0], dtype=float),
            lambda t, s: OperatingInputs(
                F=float(
                    0.0028
                    if t < 1.2
                    else (0.0080 * np.exp(0.215 * (t - 1.2)) if t < 5.5 else (0.025 if t < 8.5 else 0.024))
                ),
                rpm=185.0 if t < 1.1 else (255.0 if t < 2.7 else 520.0),
                vvm=0.25 if t < 1.1 else (0.58 if t < 2.7 else 3.5),
            ),
        ),
    ]

    batches: List[Dict[str, object]] = []
    for b_idx, (name, t_span, x0, fn) in enumerate(specs):
        sim = simulate_bioreactor(
            t_span=t_span,
            x0=x0,
            control_fn=fn,
            params=p_true,
            mode="ground_truth",
            dt_eval=0.1,
        )
        t_grid = sim["t"]
        st = sim["states"]
        u_series = [fn(t_grid[k], st[k]) for k in range(len(t_grid))]
        F_arr = np.array([max(0.0, float(u.F)) for u in u_series[:-1]], dtype=float)
        kla_ref = np.array(
            [nominal_kla(u_series[k].rpm, u_series[k].vvm, st[k, 6], p_ref) for k in range(len(t_grid) - 1)],
            dtype=float,
        )

        idx_xv = np.arange(0, len(t_grid), 2, dtype=int)
        idx_sp = np.arange(0, len(t_grid), sp_stride, dtype=int)
        idx_cl = np.arange(0, min(35, len(t_grid)), 2, dtype=int) if b_idx in (0, 2) else np.array([], dtype=int)

        sc_xv = max(1.0, float(np.max(st[:, 0])))
        sc_s = max(0.25, float(np.max(st[:, 2])))
        sc_p = max(0.5, float(np.max(st[:, 5])))
        sc_cl = float(p_true.C_star)

        y_xv = np.maximum(0.0, st[idx_xv, 0] + rng.normal(0.0, noise_std, len(idx_xv)) * sc_xv)
        y_s = np.maximum(0.0, st[idx_sp, 2] + rng.normal(0.0, noise_std, len(idx_sp)) * sc_s)
        y_p = np.maximum(0.0, st[idx_sp, 5] + rng.normal(0.0, noise_std, len(idx_sp)) * sc_p)
        y_cl = (
            np.clip(st[idx_cl, 3] + rng.normal(0.0, noise_std, len(idx_cl)) * sc_cl, 0.0, sc_cl)
            if len(idx_cl)
            else np.array([], dtype=float)
        )

        batches.append(
            {
                "name": name,
                "t": t_grid,
                "x0": x0,
                "F_arr": F_arr,
                "kla_ref": kla_ref,
                "idx_xv": idx_xv,
                "idx_sp": idx_sp,
                "idx_cl": idx_cl,
                "y_xv": y_xv,
                "y_s": y_s,
                "y_p": y_p,
                "y_cl": y_cl,
                "scales": (sc_xv, sc_s, sc_p, sc_cl),
            }
        )
    return batches


def recover_mechanistic_parameters(
    noise_std: float = 0.015,
    seed: int = 42,
    true_params: Optional[BioreactorParams] = None,
    sp_stride: int = 5,
) -> Dict[str, object]:
    """Recover all 6 mechanistic/transfer parameters via V17 multi-step iDoE from a fixed agnostic guess."""
    rng = np.random.default_rng(seed)
    p_true = true_params or BioreactorParams()
    batches = _build_stage1_idoe_batches(p_true, noise_std=noise_std, rng=rng, sp_stride=sp_stride)
    noise_eff = max(1e-4, float(noise_std))

    theta_true = np.array(
        [p_true.mu_max, p_true.K_S, p_true.Y_XS, p_true.alpha, p_true.beta, p_true.k_La0],
        dtype=float,
    )
    theta_init = _STAGE1_THETA_INIT.copy()

    def residuals(theta: np.ndarray) -> np.ndarray:
        cand = replace(
            p_true,
            mu_max=float(theta[0]),
            K_S=float(theta[1]),
            Y_XS=float(theta[2]),
            alpha=float(theta[3]),
            beta=float(theta[4]),
            k_La0=float(theta[5]),
        )
        parts = []
        for b in batches:
            pred = _stage1_exp_rosenbrock_rollout(
                b["t"], b["x0"], b["F_arr"], b["kla_ref"], p_true.S_feed, cand, n_substeps=4
            )
            sc_xv, sc_s, sc_p, sc_cl = b["scales"]
            ix, isp, icl = b["idx_xv"], b["idx_sp"], b["idx_cl"]
            parts.append((pred[ix, 0] - b["y_xv"]) / (sc_xv * noise_eff))
            parts.append((pred[isp, 2] - b["y_s"]) / (sc_s * noise_eff))
            parts.append((pred[isp, 5] - b["y_p"]) / (sc_p * noise_eff))
            if len(icl):
                parts.append((pred[icl, 3] - b["y_cl"]) / (sc_cl * noise_eff))
        return np.concatenate(parts)

    res = least_squares(
        residuals,
        theta_init,
        bounds=(_STAGE1_LOWER, _STAGE1_UPPER),
        method="trf",
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        max_nfev=300,
    )
    theta_rec = res.x
    names = list(_STAGE1_PARAM_NAMES)
    rel_err_pct = 100.0 * np.abs(theta_rec - theta_true) / theta_true

    j_rel = res.jac * theta_rec[None, :]
    svals = np.linalg.svd(j_rel, compute_uv=False)
    kappa_rel = float(svals[0] / max(1e-12, svals[-1])) if len(svals) else float("inf")
    cov = np.linalg.pinv(res.jac.T @ res.jac)
    diag = np.sqrt(np.maximum(1e-18, np.diag(cov)))
    dof = max(1, len(res.fun) - len(names))
    mse = max(1e-12, 2.0 * float(res.cost) / dof)
    se = diag * np.sqrt(mse)
    corr = cov / np.outer(diag, diag)

    recovered_params = replace(
        p_true,
        mu_max=float(theta_rec[0]),
        K_S=float(theta_rec[1]),
        Y_XS=float(theta_rec[2]),
        alpha=float(theta_rec[3]),
        beta=float(theta_rec[4]),
        k_La0=float(theta_rec[5]),
    )
    sha = hashlib.sha256(
        json.dumps(recovered_params.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "param_names": names,
        "true_values": {n: float(v) for n, v in zip(names, theta_true)},
        "initial_guess": {n: float(v) for n, v in zip(names, theta_init)},
        "recovered_values": {n: float(v) for n, v in zip(names, theta_rec)},
        "standard_errors": {n: float(s) for n, s in zip(names, se)},
        "relative_error_pct": {n: float(e) for n, e in zip(names, rel_err_pct)},
        "max_relative_error_pct": float(np.max(rel_err_pct)),
        "recovered_params": recovered_params,
        "fit_cost": float(res.cost),
        "nfev": int(res.nfev),
        "condition_number_rel": kappa_rel,
        "corr_matrix": corr.tolist(),
        "calibration_sha256": sha,
    }


# =============================================================================
# Stage 2: Parsimonious GAM-86 (h_ace=5, h_mu=4) with Power-Law Rheology & VJP
# =============================================================================

_VTIP_ONSET = 1.65
_KLA_SCALES = np.array([10.0, 25.0], dtype=float)
_HEALTHY_S = np.array([0.12, 0.40, 1.20, 2.00, 0.50], dtype=float)
_HEALTHY_CL = np.array([3.50, 6.50, 7.50, 4.50, 2.50], dtype=float)
_HEALTHY_A = np.array([0.00, 0.00, 0.05, 0.15, 0.05], dtype=float)
_HEALTHY_VTIP = np.array([1.00, 1.50, 1.60, 1.20, 1.65], dtype=float)


def _make_gam_spec(
    h_ace: int = 5,
    h_mu: int = 4,
) -> Tuple[List[Tuple[str, Tuple[int, ...], float, float, float, float]], int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    specs: List[Tuple[str, Tuple[int, ...], float, float, float, float]] = [
        ("kd_w1", (4, 2), 0.0, 25.0, 6.0, 1.0),
        ("kd_b1", (4, 2), -12.0, 2.0, 4.0, 0.0),
        ("kd_w2", (4, 2), 0.0, 4.0, 1.5, 1.0),
        ("kla_c", (2,), 0.0, 5.0, 0.3, 0.0),
        ("kla_p", (2,), 0.8, 2.8, 0.3, 0.0),
        ("ace_W1", (h_ace, 4), -10.0, 10.0, 1.8, 1.0),
        ("ace_b1", (h_ace,), -10.0, 10.0, 1.8, 0.0),
        ("ace_W2", (2, h_ace), -10.0, 10.0, 1.4, 1.0),
        ("ace_b2", (2,), -10.0, 10.0, 1.4, 0.0),
        ("mu_W1", (h_mu, 3), -10.0, 10.0, 0.8, 1.0),
        ("mu_b1", (h_mu,), -10.0, 10.0, 0.8, 0.0),
        ("mu_w2", (h_mu,), -10.0, 10.0, 0.6, 1.0),
        ("mu_b2", (1,), -10.0, 10.0, 0.6, 0.0),
    ]
    lowers, uppers, scales, l2_mask = [], [], [], []
    for _, shape, lo, hi, sc, mv in specs:
        sz = int(np.prod(shape))
        lowers.append(np.full(sz, lo, dtype=float))
        uppers.append(np.full(sz, hi, dtype=float))
        scales.append(np.full(sz, sc, dtype=float))
        l2_mask.append(np.full(sz, mv, dtype=float))
    return (
        specs,
        sum(len(a) for a in lowers),
        np.concatenate(lowers),
        np.concatenate(uppers),
        np.concatenate(scales),
        np.concatenate(l2_mask),
    )


def _make_block_l2_weights(
    specs: List[Tuple[str, Tuple[int, ...], float, float, float, float]],
    w_kd: float = 2e-6,
    w_ace: float = 6.0e-3,
    w_mu: float = 1.5e-2,
) -> np.ndarray:
    vecs = []
    for nm, shape, *_ in specs:
        sz = int(np.prod(shape))
        if nm in ("kd_w1", "kd_w2"):
            vecs.append(np.full(sz, w_kd, dtype=float))
        elif nm in ("ace_W1", "ace_W2"):
            vecs.append(np.full(sz, w_ace, dtype=float))
        elif nm in ("mu_W1", "mu_w2"):
            vecs.append(np.full(sz, w_mu, dtype=float))
        else:
            vecs.append(np.zeros(sz, dtype=float))
    return np.sqrt(np.concatenate(vecs))


_GAM_SPECS, _N_PARAMS, _LB, _UB, _D_SCALE, _L2_MASK = _make_gam_spec(h_ace=5, h_mu=4)
_L2_W = _make_block_l2_weights(_GAM_SPECS)


def _unpack_theta(
    theta: np.ndarray,
    specs: Sequence[Tuple[str, Tuple[int, ...], float, float, float, float]] = _GAM_SPECS,
) -> Tuple[np.ndarray, ...]:
    idx = 0
    out = []
    for _, shape, *_ in specs:
        size = int(np.prod(shape))
        arr = theta[idx : idx + size]
        out.append(arr.reshape(shape) if len(shape) > 1 else arr)
        idx += size
    return tuple(out)


def _pack_theta(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])


def _gam_init_weights(
    seed: int,
    specs: Sequence[Tuple[str, Tuple[int, ...], float, float, float, float]] = _GAM_SPECS,
    lb: np.ndarray = _LB,
    ub: np.ndarray = _UB,
    h_ace: int = 5,
    h_mu: int = 4,
) -> np.ndarray:
    """Initialize GAM weights with independent per-head streams and zero basal acetate leak."""
    del specs
    rng_kd = np.random.default_rng(seed + 101)
    rng_kl = np.random.default_rng(seed + 202)
    rng_ac = np.random.default_rng(seed + 303)
    rng_mu = np.random.default_rng(seed + 404)

    kd_w1 = rng_kd.uniform(3.0, 6.0, size=(4, 2))
    kd_b1 = np.array([[-3.5, -5.0], [-3.5, -5.0], [-2.5, -4.0], [-2.0, -3.5]], dtype=float)
    kd_w2 = rng_kd.uniform(0.15, 0.35, size=(4, 2))
    kla_c = np.array([0.22, 0.30], dtype=float) + rng_kl.uniform(-0.05, 0.08, size=2)
    kla_p = np.array([1.40, 1.80], dtype=float) + rng_kl.uniform(-0.15, 0.15, size=2)
    ace_W1 = rng_ac.normal(0.0, 0.25, size=(h_ace, 4))
    ace_b1 = np.zeros(h_ace, dtype=float)
    ace_W2 = rng_ac.normal(0.0, 0.20, size=(2, h_ace))
    ace_b2 = np.array([-4.8, -4.8], dtype=float)
    mu_W1 = rng_mu.normal(0.0, 0.20, size=(h_mu, 3))
    mu_b1 = np.zeros(h_mu, dtype=float)
    mu_w2 = rng_mu.normal(0.0, 0.20, size=(h_mu,))
    mu_b2 = np.array([2.4], dtype=float)
    theta = _pack_theta(
        kd_w1, kd_b1, kd_w2, kla_c, kla_p, ace_W1, ace_b1, ace_W2, ace_b2, mu_W1, mu_b1, mu_w2, mu_b2
    )
    return np.clip(theta, lb + 1e-4, ub - 1e-4)


# Backward-compatible alias used in tests/external callers
_gam76_init_weights = _gam_init_weights


def _gam_eval_heads(
    theta: np.ndarray,
    X_tot: np.ndarray,
    S: np.ndarray,
    C_L: np.ndarray,
    A: np.ndarray,
    v_tip: np.ndarray,
    mu_mech: np.ndarray,
    specs: Sequence[Tuple[str, Tuple[int, ...], float, float, float, float]] = _GAM_SPECS,
    vtip_onset: float = _VTIP_ONSET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate all 5 intensive biological/rheological outputs for array inputs of shape (M,)."""
    (
        kd_w1, kd_b1, kd_w2,
        kla_c, kla_p,
        ace_W1, ace_b1, ace_W2, ace_b2,
        mu_W1, mu_b1, mu_w2, mu_b2,
    ) = _unpack_theta(theta, specs)

    s_pos = np.maximum(0.0, S)
    cl_pos = np.maximum(0.0, C_L)
    a_pos = np.maximum(0.0, A)
    xt_pos = np.maximum(0.0, X_tot)

    s_avail = s_pos / (0.002 + s_pos)
    o2_avail = cl_pos / (0.005 + cl_pos)
    a_avail = a_pos / (0.25 + a_pos)

    z_kd = np.column_stack(
        [
            1.0 / (1.0 + s_pos / 0.05),
            1.0 / (1.0 + cl_pos / 0.50),
            2.0 * ((a_pos / (1.5 + a_pos)) ** 2),
            np.maximum(0.0, v_tip - vtip_onset),
        ]
    )
    phi_kd = _sigmoid(z_kd[:, :, None] * kd_w1[None, :, :] + kd_b1[None, :, :]) - _sigmoid(kd_b1)[None, :, :]
    delta_kd = np.maximum(0.0, 0.05 * np.sum(phi_kd * kd_w2[None, :, :], axis=(1, 2)))

    u_kla = 1.0 + xt_pos[:, None] / _KLA_SCALES[None, :]
    psi_kla = np.sum(kla_c[None, :] * ((u_kla ** kla_p[None, :]) - 1.0), axis=1)
    eta_kla = 1.0 / (1.0 + np.maximum(0.0, psi_kla))

    z_mu = np.column_stack([s_pos / (0.5 + s_pos), np.log1p(s_pos * 0.5) * 0.4, a_pos / (3.0 + a_pos)])
    eta_mu = 0.05 + _sigmoid(np.tanh(z_mu @ mu_W1.T + mu_b1) @ mu_w2 + mu_b2[0])
    mu_hyb = mu_mech * eta_mu

    z_ace = np.column_stack(
        [
            mu_hyb / 0.45,
            np.log1p(s_pos * 10.0) / 3.0,
            np.log1p(cl_pos * 2.0) / 3.0,
            np.log1p(a_pos * 2.0) / 3.0,
        ]
    )
    raw_ace = np.tanh(z_ace @ ace_W1.T + ace_b1) @ ace_W2.T + ace_b2
    q_a_prod = 0.08 * _softplus(raw_ace[:, 0]) * s_avail
    q_a_cons = 0.08 * _softplus(raw_ace[:, 1]) * a_avail * o2_avail
    return delta_kd, eta_kla, q_a_prod, q_a_cons, eta_mu


def _fast_loss_grad_and_rss(
    theta: np.ndarray,
    batch_data: Dict[str, np.ndarray],
    params: BioreactorParams,
    specs: Sequence[Tuple[str, Tuple[int, ...], float, float, float, float]] = _GAM_SPECS,
    l2_w: np.ndarray = _L2_W,
    flux_l2: float = 2e-3,
    ov_reg: float = 3e-4,
    hreg: float = 250.0,
    vtip_onset: float = _VTIP_ONSET,
) -> Tuple[float, np.ndarray, float]:
    """Preallocated exact VJP rollout returning (total_loss, grad, unweighted_std_rss_obs)."""
    (
        kd_w1, kd_b1, kd_w2,
        kla_c, kla_p,
        ace_W1, ace_b1, ace_W2, ace_b2,
        mu_W1, mu_b1, mu_w2, mu_b2,
    ) = _unpack_theta(theta, specs)

    x0 = batch_data["x0"]
    F_arr = batch_data["F"]
    Sf_arr = batch_data["S_feed"]
    kla_nom_arr = batch_data["kla_nom"]
    v_tip_arr = batch_data["v_tip"]
    y_obs = batch_data["y_obs"]
    w_obs = batch_data["w_obs"]
    inv_var_obs = batch_data.get("inv_var_obs", w_obs)
    dt = float(batch_data["dt"])
    T_steps, M = F_arr.shape

    x_hist = np.empty((T_steps + 1, M, 7), dtype=float)
    x_hist[0] = np.maximum(0.0, x0)

    inv_Yxs = 1.0 / params.Y_XS
    inv_Yxo = 1.0 / params.Y_XO
    inv_Yps = 1.0 / params.Y_PS
    inv_Yas = 1.0 / params.Y_AS
    inv_Yao = 1.0 / params.Y_AO
    sig_kdb = _sigmoid(kd_b1)
    kla_scales = _KLA_SCALES

    z_kd_all = np.empty((T_steps, M, 4), dtype=float)
    z_mu_all = np.empty((T_steps, M, 3), dtype=float)
    z_ace_all = np.empty((T_steps, M, 4), dtype=float)
    h_mu_all = np.empty((T_steps, M, len(mu_b1)), dtype=float)
    h_ace_all = np.empty((T_steps, M, len(ace_b1)), dtype=float)
    sig_kd_all = np.empty((T_steps, M, 4, 2), dtype=float)
    u_kla_all = np.empty((T_steps, M, 2), dtype=float)
    u_pow_all = np.empty((T_steps, M, 2), dtype=float)
    phi_kla_all = np.empty((T_steps, M, 2), dtype=float)

    eta_kla_all = np.empty((T_steps, M), dtype=float)
    k_La_all = np.empty((T_steps, M), dtype=float)
    mask_kla_all = np.empty((T_steps, M), dtype=float)
    mask_dkd_all = np.empty((T_steps, M), dtype=float)
    sig_mu_all = np.empty((T_steps, M), dtype=float)
    eta_mu_all = np.empty((T_steps, M), dtype=float)
    mu_mech_all = np.empty((T_steps, M), dtype=float)
    mu_all = np.empty((T_steps, M), dtype=float)
    s_mon_all = np.empty((T_steps, M), dtype=float)
    o2_mon_all = np.empty((T_steps, M), dtype=float)
    dsm_all = np.empty((T_steps, M), dtype=float)
    dom_all = np.empty((T_steps, M), dtype=float)
    sav_all = np.empty((T_steps, M), dtype=float)
    dsav_all = np.empty((T_steps, M), dtype=float)
    oav_all = np.empty((T_steps, M), dtype=float)
    doav_all = np.empty((T_steps, M), dtype=float)
    aav_all = np.empty((T_steps, M), dtype=float)
    daav_all = np.empty((T_steps, M), dtype=float)
    sp0_all = np.empty((T_steps, M), dtype=float)
    sp1_all = np.empty((T_steps, M), dtype=float)
    sig_a0_all = np.empty((T_steps, M), dtype=float)
    sig_a1_all = np.empty((T_steps, M), dtype=float)
    qap_all = np.empty((T_steps, M), dtype=float)
    qac_all = np.empty((T_steps, M), dtype=float)
    kd_all = np.empty((T_steps, M), dtype=float)
    qP_all = np.empty((T_steps, M), dtype=float)
    qS_all = np.empty((T_steps, M), dtype=float)
    our_all = np.empty((T_steps, M), dtype=float)
    D_all = np.empty((T_steps, M), dtype=float)
    d_mu_dS_all = np.empty((T_steps, M), dtype=float)
    d_mu_dCL_all = np.empty((T_steps, M), dtype=float)
    d_qS_dS_all = np.empty((T_steps, M), dtype=float)
    d_our_dCL_all = np.empty((T_steps, M), dtype=float)
    phi2_all = np.empty((T_steps, M), dtype=float)
    dphi2_all = np.empty((T_steps, M), dtype=float)
    f2_all = np.empty((T_steps, M), dtype=float)
    phi3_all = np.empty((T_steps, M), dtype=float)
    dphi3_all = np.empty((T_steps, M), dtype=float)
    f3_all = np.empty((T_steps, M), dtype=float)
    mask_clamp_all = np.empty((T_steps, M, 7), dtype=float)

    loss_flux = 0.0
    loss_ov = 0.0
    eps_sp = 1e-4
    cl_cap = params.C_star * 1.05
    v_tip_excess = np.maximum(0.0, v_tip_arr - vtip_onset)

    for t in range(T_steps):
        x = x_hist[t]
        Xv, Xd, S, CL, A, P = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        V = np.maximum(0.2, x[:, 6])
        F_eff = np.where(V >= params.V_max, 0.0, np.maximum(0.0, F_arr[t]))
        D = F_eff / V
        D_all[t] = D
        S_feed, kla_nom = Sf_arr[t], kla_nom_arr[t]

        denom_ks, denom_ko = params.K_S + S, params.K_O2 + CL
        s_monod, o2_monod = S / denom_ks, CL / denom_ko
        d_sm, d_om = params.K_S / (denom_ks * denom_ks), params.K_O2 / (denom_ko * denom_ko)
        mu_mech = params.mu_max * s_monod * o2_monod
        s_mon_all[t], o2_mon_all[t], dsm_all[t], dom_all[t], mu_mech_all[t] = (
            s_monod, o2_monod, d_sm, d_om, mu_mech
        )

        sa = S / (0.002 + S)
        dsa = 0.002 / ((0.002 + S) ** 2)
        oa = CL / (0.005 + CL)
        doa = 0.005 / ((0.005 + CL) ** 2)
        aa = A / (0.25 + A)
        daa = 0.25 / ((0.25 + A) ** 2)
        sav_all[t], dsav_all[t], oav_all[t], doav_all[t], aav_all[t], daav_all[t] = (
            sa, dsa, oa, doa, aa, daa
        )

        zk = z_kd_all[t]
        zk[:, 0] = 1.0 / (1.0 + S * 20.0)
        zk[:, 1] = 1.0 / (1.0 + CL * 2.0)
        a_rat = A / (1.5 + A)
        zk[:, 2] = 2.0 * a_rat * a_rat
        zk[:, 3] = v_tip_excess[t]
        skd = _sigmoid(zk[:, :, None] * kd_w1[None, :, :] + kd_b1[None, :, :])
        sig_kd_all[t] = skd
        dk_raw = 0.05 * np.sum((skd - sig_kdb[None, :, :]) * kd_w2[None, :, :], axis=(1, 2))
        sq_kd = np.sqrt(dk_raw * dk_raw + 1e-10)
        delta_kd = 0.5 * (dk_raw + sq_kd)
        mask_dkd_all[t] = 0.5 * (1.0 + dk_raw / sq_kd)

        X_tot = Xv + Xd
        uk = 1.0 + X_tot[:, None] / kla_scales[None, :]
        up = uk ** kla_p[None, :]
        pk = up - 1.0
        u_kla_all[t], u_pow_all[t], phi_kla_all[t] = uk, up, pk
        psi_raw = pk @ kla_c
        eta_kla = 1.0 / (1.0 + psi_raw)
        eta_kla_all[t] = eta_kla
        k_La_raw = kla_nom * eta_kla
        k_La = np.maximum(5.0, k_La_raw)
        k_La_all[t] = k_La
        mask_kla_all[t] = (k_La_raw > 5.0).astype(float)

        zm = z_mu_all[t]
        zm[:, 0] = S / (0.5 + S)
        zm[:, 1] = np.log1p(S * 0.5) * 0.4
        zm[:, 2] = A / (3.0 + A)
        hm = np.tanh(zm @ mu_W1.T + mu_b1)
        h_mu_all[t] = hm
        smu = _sigmoid(hm @ mu_w2 + mu_b2[0])
        sig_mu_all[t] = smu
        eta_mu = 0.05 + smu
        eta_mu_all[t] = eta_mu
        mu = mu_mech * eta_mu
        mu_all[t] = mu

        za = z_ace_all[t]
        za[:, 0] = mu / 0.45
        za[:, 1] = np.log1p(S * 10.0) / 3.0
        za[:, 2] = np.log1p(CL * 2.0) / 3.0
        za[:, 3] = np.log1p(A * 2.0) / 3.0
        ha = np.tanh(za @ ace_W1.T + ace_b1)
        h_ace_all[t] = ha
        ra = ha @ ace_W2.T + ace_b2
        sp0, sp1 = _softplus(ra[:, 0]), _softplus(ra[:, 1])
        sp0_all[t], sp1_all[t] = sp0, sp1
        sig_a0_all[t], sig_a1_all[t] = _sigmoid(ra[:, 0]), _sigmoid(ra[:, 1])
        qap = 0.08 * sp0 * sa
        qac = 0.08 * sp1 * aa * oa
        qap_all[t], qac_all[t] = qap, qac
        loss_flux += float(np.sum(qap * qap + qac * qac))
        loss_ov += float(np.sum(qap * qac))

        k_d = params.k_d0 + delta_kd
        kd_all[t] = k_d
        q_P = (params.alpha * mu + params.beta) * sa * oa
        q_S = mu * inv_Yxs + params.m_s * sa + q_P * inv_Yps + qap * inv_Yas
        our = 1000.0 * (mu * inv_Yxo + params.m_o * oa + qac * inv_Yao) * Xv
        qP_all[t], qS_all[t], our_all[t] = q_P, q_S, our

        d_mu_dS = params.mu_max * d_sm * o2_monod * eta_mu
        d_qP_dS = params.alpha * d_mu_dS * sa * oa + (params.alpha * mu + params.beta) * dsa * oa
        d_qS_dS = d_mu_dS * inv_Yxs + params.m_s * dsa + d_qP_dS * inv_Yps
        phi2, dphi2 = _phi1_and_grad((D + Xv * d_qS_dS) * dt)
        f2 = D * (S_feed - S) - q_S * Xv
        d_mu_dCL = params.mu_max * s_monod * d_om * eta_mu
        d_our_dCL = 1000.0 * (d_mu_dCL * inv_Yxo + params.m_o * doa) * Xv
        phi3, dphi3 = _phi1_and_grad((k_La + D + d_our_dCL) * dt)
        f3 = k_La * (params.C_star - CL) - our - D * CL

        d_mu_dS_all[t], d_mu_dCL_all[t] = d_mu_dS, d_mu_dCL
        d_qS_dS_all[t], d_our_dCL_all[t] = d_qS_dS, d_our_dCL
        phi2_all[t], dphi2_all[t], f2_all[t] = phi2, dphi2, f2
        phi3_all[t], dphi3_all[t], f3_all[t] = phi3, dphi3, f3

        raw_next = np.empty((M, 7), dtype=float)
        raw_next[:, 0] = Xv + dt * (mu - k_d - D) * Xv
        raw_next[:, 1] = Xd + dt * (k_d * Xv - D * Xd)
        raw_next[:, 2] = S + dt * phi2 * f2
        raw_next[:, 3] = CL + dt * phi3 * f3
        raw_next[:, 4] = A + dt * ((qap - qac) * Xv - D * A)
        raw_next[:, 5] = P + dt * (q_P * Xv - D * P)
        raw_next[:, 6] = V + dt * F_eff

        sq_rn = np.sqrt(raw_next[:, :6] ** 2 + eps_sp ** 2)
        xn = x_hist[t + 1]
        xn[:, :6] = 0.5 * (raw_next[:, :6] + sq_rn)
        mc = mask_clamp_all[t]
        mc[:, :6] = 0.5 * (1.0 + raw_next[:, :6] / sq_rn)
        mask_cl = (xn[:, 3] < cl_cap).astype(float)
        xn[:, 3] = np.minimum(xn[:, 3], cl_cap)
        xn[:, 6] = np.minimum(raw_next[:, 6], params.V_max)
        mc[:, 3] *= mask_cl
        mc[:, 6] = (raw_next[:, 6] < params.V_max).astype(float)

    diff = x_hist - y_obs
    loss_data = 0.5 * float(np.sum(w_obs * (diff * diff)))
    std_rss_obs = float(np.sum(inv_var_obs * (diff * diff)))
    c_flux = flux_l2 / (2.0 * T_steps * M)
    c_ov = ov_reg / (T_steps * M)

    zh_kd = np.column_stack(
        [
            1.0 / (1.0 + _HEALTHY_S / 0.05),
            1.0 / (1.0 + _HEALTHY_CL / 0.50),
            2.0 * ((_HEALTHY_A / (1.5 + _HEALTHY_A)) ** 2),
            np.maximum(0.0, _HEALTHY_VTIP - vtip_onset),
        ]
    )
    sigh_kd = _sigmoid(zh_kd[:, :, None] * kd_w1[None, :, :] + kd_b1[None, :, :])
    phih_kd = sigh_kd - sig_kdb[None, :, :]
    dh = 0.05 * np.sum(phih_kd * kd_w2[None, :, :], axis=(1, 2))
    loss_hreg = 0.5 * hreg * float(np.mean(dh * dh))
    loss_l2 = 0.5 * float(np.sum((l2_w * theta) ** 2))
    total_loss = loss_data + c_flux * loss_flux + c_ov * loss_ov + loss_hreg + loss_l2

    g_kd_w1, g_kd_b1, g_kd_w2 = np.zeros_like(kd_w1), np.zeros_like(kd_b1), np.zeros_like(kd_w2)
    g_kla_c, g_kla_p = np.zeros_like(kla_c), np.zeros_like(kla_p)
    g_ace_W1, g_ace_b1 = np.zeros_like(ace_W1), np.zeros_like(ace_b1)
    g_ace_W2, g_ace_b2 = np.zeros_like(ace_W2), np.zeros_like(ace_b2)
    g_mu_W1, g_mu_b1 = np.zeros_like(mu_W1), np.zeros_like(mu_b1)
    g_mu_w2, g_mu_b2 = np.zeros_like(mu_w2), np.zeros_like(mu_b2)

    g_dh = (hreg / len(_HEALTHY_S)) * dh
    g_phi_h = (0.05 * g_dh[:, None, None]) * kd_w2[None, :, :]
    g_kd_w2 += 0.05 * np.sum(g_dh[:, None, None] * phih_kd, axis=0)
    g_argh = g_phi_h * sigh_kd * (1.0 - sigh_kd)
    g_kd_w1 += np.sum(g_argh * zh_kd[:, :, None], axis=0)
    g_kd_b1 += np.sum(g_argh, axis=0) - np.sum(g_phi_h, axis=0) * sig_kdb * (1.0 - sig_kdb)

    adj_x = w_obs[T_steps] * diff[T_steps]
    g_raw_ace = np.empty((M, 2), dtype=float)
    for t in range(T_steps - 1, -1, -1):
        x = x_hist[t]
        Xv, S, CL, A = x[:, 0], x[:, 2], x[:, 3], x[:, 4]
        D = D_all[t]
        mu, k_d, q_P, q_S = mu_all[t], kd_all[t], qP_all[t], qS_all[t]
        qap, qac = qap_all[t], qac_all[t]
        sa, dsa, oa, doa, aa, daa = (
            sav_all[t], dsav_all[t], oav_all[t], doav_all[t], aav_all[t], daav_all[t]
        )
        denom_ks, denom_ko = params.K_S + S, params.K_O2 + CL
        s_monod, o2_monod, d_sm, d_om = s_mon_all[t], o2_mon_all[t], dsm_all[t], dom_all[t]
        mu_mech, eta_mu, smu = mu_mech_all[t], eta_mu_all[t], sig_mu_all[t]
        phi2, dphi2, f2 = phi2_all[t], dphi2_all[t], f2_all[t]
        phi3, dphi3, f3 = phi3_all[t], dphi3_all[t], f3_all[t]
        d_mu_dS, d_mu_dCL = d_mu_dS_all[t], d_mu_dCL_all[t]
        d_qS_dS, d_our_dCL = d_qS_dS_all[t], d_our_dCL_all[t]
        k_La, eta_kla = k_La_all[t], eta_kla_all[t]

        g_raw = adj_x * mask_clamp_all[t]
        g_Xv_nxt, g_Xd_nxt, g_S_nxt, g_CL_nxt, g_A_nxt, g_P_nxt, g_V_nxt = (
            g_raw[:, 0], g_raw[:, 1], g_raw[:, 2], g_raw[:, 3], g_raw[:, 4], g_raw[:, 5], g_raw[:, 6]
        )

        g_Xv = (
            g_Xv_nxt * (1.0 + dt * (mu - k_d - D))
            + g_Xd_nxt * (dt * k_d)
            + g_A_nxt * (dt * (qap - qac))
            + g_P_nxt * (dt * q_P)
        )
        g_Xd, g_S, g_CL = g_Xd_nxt * (1.0 - dt * D), g_S_nxt, g_CL_nxt
        g_A, g_P, g_V = g_A_nxt * (1.0 - dt * D), g_P_nxt * (1.0 - dt * D), g_V_nxt

        g_f2, g_z2 = g_S_nxt * dt * phi2, g_S_nxt * dt * f2 * dphi2
        g_S -= g_f2 * D
        g_qS = -g_f2 * Xv
        g_Xv += -g_f2 * q_S + g_z2 * dt * d_qS_dS
        g_d_qS_dS = g_z2 * dt * Xv

        g_f3, g_z3 = g_CL_nxt * dt * phi3, g_CL_nxt * dt * f3 * dphi3
        g_kLa = g_f3 * (params.C_star - CL) + g_z3 * dt
        g_CL -= g_f3 * (k_La + D)
        g_our, g_d_our_dCL = -g_f3, g_z3 * dt

        g_mu = g_Xv_nxt * dt * Xv + g_qS * inv_Yxs + 1000.0 * (g_our * Xv) * inv_Yxo
        g_qP = g_P_nxt * dt * Xv + g_qS * inv_Yps
        g_delta_kd = (-g_Xv_nxt + g_Xd_nxt) * dt * Xv * mask_dkd_all[t]

        g_q_a_prod = g_A_nxt * dt * Xv + g_qS * inv_Yas + 2.0 * c_flux * qap + c_ov * qac
        g_q_a_cons = -g_A_nxt * dt * Xv + 1000.0 * g_our * Xv * inv_Yao + 2.0 * c_flux * qac + c_ov * qap
        g_Xv += (
            1000.0 * g_our * (mu * inv_Yxo + params.m_o * oa + qac * inv_Yao)
            + 1000.0 * g_d_our_dCL * (d_mu_dCL * inv_Yxo + params.m_o * doa)
        )

        g_d_mu_dCL = 1000.0 * g_d_our_dCL * Xv * inv_Yxo
        g_o2avail_direct = g_qP * (params.alpha * mu + params.beta) * sa + 1000.0 * g_our * Xv * params.m_o
        g_d_o2avail = 1000.0 * g_d_our_dCL * Xv * params.m_o

        g_mu += g_qP * params.alpha * sa * oa
        g_savail_direct = g_qP * (params.alpha * mu + params.beta) * oa + g_qS * params.m_s

        g_d_mu_dS = g_d_qS_dS * inv_Yxs + (g_d_qS_dS * inv_Yps) * params.alpha * sa * oa
        g_d_savail = g_d_qS_dS * params.m_s + (g_d_qS_dS * inv_Yps) * (params.alpha * mu + params.beta) * oa
        g_mu += (g_d_qS_dS * inv_Yps) * params.alpha * dsa * oa
        g_savail_direct += (g_d_qS_dS * inv_Yps) * params.alpha * d_mu_dS * oa
        g_o2avail_direct += (g_d_qS_dS * inv_Yps) * (
            params.alpha * d_mu_dS * sa + (params.alpha * mu + params.beta) * dsa
        )

        g_sp0, g_sp1 = g_q_a_prod * 0.08 * sa, g_q_a_cons * 0.08 * aa * oa
        g_savail_direct += g_q_a_prod * 0.08 * sp0_all[t]
        g_aavail = g_q_a_cons * 0.08 * sp1_all[t] * oa
        g_o2avail_direct += g_q_a_cons * 0.08 * sp1_all[t] * aa

        g_raw_ace[:, 0] = g_sp0 * sig_a0_all[t]
        g_raw_ace[:, 1] = g_sp1 * sig_a1_all[t]
        ha = h_ace_all[t]
        g_ace_W2 += g_raw_ace.T @ ha
        g_ace_b2 += np.sum(g_raw_ace, axis=0)
        g_act_ace = (g_raw_ace @ ace_W2) * (1.0 - ha * ha)
        g_ace_W1 += g_act_ace.T @ z_ace_all[t]
        g_ace_b1 += np.sum(g_act_ace, axis=0)
        g_z_ace = g_act_ace @ ace_W1
        g_mu += g_z_ace[:, 0] / 0.45
        g_S += g_z_ace[:, 1] / (3.0 * (0.1 + S))
        g_CL += g_z_ace[:, 2] / (3.0 * (0.5 + CL))
        g_A += g_z_ace[:, 3] / (3.0 * (0.5 + A))

        g_eta_mu = (
            g_mu * mu_mech
            + g_d_mu_dS * (params.mu_max * d_sm * o2_monod)
            + g_d_mu_dCL * (params.mu_max * s_monod * d_om)
        )
        g_mu_mech = g_mu * eta_mu
        g_s_monod = g_mu_mech * params.mu_max * o2_monod + g_d_mu_dCL * params.mu_max * d_om * eta_mu
        g_o2_monod = g_mu_mech * params.mu_max * s_monod + g_d_mu_dS * params.mu_max * d_sm * eta_mu
        g_d_smonod_dS = g_d_mu_dS * params.mu_max * o2_monod * eta_mu
        g_d_o2monod_dCL = g_d_mu_dCL * params.mu_max * s_monod * eta_mu
        g_S += g_s_monod * d_sm - 2.0 * g_d_smonod_dS * d_sm / denom_ks
        g_CL += g_o2_monod * d_om - 2.0 * g_d_o2monod_dCL * d_om / denom_ko

        g_raw_mu = g_eta_mu * smu * (1.0 - smu)
        hm = h_mu_all[t]
        g_mu_w2 += hm.T @ g_raw_mu
        g_mu_b2[0] += float(np.sum(g_raw_mu))
        g_act_mu = np.outer(g_raw_mu, mu_w2) * (1.0 - hm * hm)
        g_mu_W1 += g_act_mu.T @ z_mu_all[t]
        g_mu_b1 += np.sum(g_act_mu, axis=0)
        g_z_mu = g_act_mu @ mu_W1
        g_S += g_z_mu[:, 0] * 0.5 / ((0.5 + S) ** 2) + g_z_mu[:, 1] / (2.5 * (2.0 + S))
        g_A += g_z_mu[:, 2] * 3.0 / ((3.0 + A) ** 2)

        g_S += g_savail_direct * dsa - 2.0 * g_d_savail * dsa / (0.002 + S)
        g_CL += g_o2avail_direct * doa - 2.0 * g_d_o2avail * doa / (0.005 + CL)
        g_A += g_aavail * daa

        g_eta_kla = g_kLa * mask_kla_all[t] * kla_nom_arr[t]
        g_psi = -g_eta_kla * (eta_kla * eta_kla)
        pk = phi_kla_all[t]
        uk = u_kla_all[t]
        up = u_pow_all[t]
        g_kla_c += np.sum(g_psi[:, None] * pk, axis=0)
        g_kla_p += np.sum(g_psi[:, None] * kla_c[None, :] * up * np.log(uk), axis=0)
        u_pm1 = uk ** (kla_p[None, :] - 1.0)
        g_xtot = np.sum(
            g_psi[:, None] * kla_c[None, :] * (kla_p[None, :] / kla_scales[None, :]) * u_pm1,
            axis=1,
        )
        g_Xv += g_xtot
        g_Xd += g_xtot

        skd = sig_kd_all[t]
        g_phi_kd = (0.05 * g_delta_kd[:, None, None]) * kd_w2[None, :, :]
        g_kd_w2 += 0.05 * np.sum(g_delta_kd[:, None, None] * (skd - sig_kdb[None, :, :]), axis=0)
        g_arg_kd = g_phi_kd * skd * (1.0 - skd)
        g_kd_w1 += np.sum(g_arg_kd * z_kd_all[t][:, :, None], axis=0)
        g_kd_b1 += np.sum(g_arg_kd, axis=0) - np.sum(g_phi_kd, axis=0) * sig_kdb * (1.0 - sig_kdb)
        g_z_kd = np.sum(g_arg_kd * kd_w1[None, :, :], axis=2)
        g_S += -g_z_kd[:, 0] * 20.0 / ((1.0 + S * 20.0) ** 2)
        g_CL += -g_z_kd[:, 1] * 2.0 / ((1.0 + CL * 2.0) ** 2)
        g_A += g_z_kd[:, 2] * (4.0 * (A / (1.5 + A)) * 1.5 / ((1.5 + A) ** 2))

        adj_x = np.column_stack([g_Xv, g_Xd, g_S, g_CL, g_A, g_P, g_V]) + w_obs[t] * diff[t]

    grad = (
        _pack_theta(
            g_kd_w1, g_kd_b1, g_kd_w2,
            g_kla_c, g_kla_p,
            g_ace_W1, g_ace_b1, g_ace_W2, g_ace_b2,
            g_mu_W1, g_mu_b1, g_mu_w2, g_mu_b2,
        )
        + (l2_w * l2_w) * theta
    )
    return total_loss, grad, std_rss_obs


def _trajectory_loss_and_grad(
    theta: np.ndarray,
    batch_data: Dict[str, np.ndarray],
    params: BioreactorParams,
    flux_l2: float = 2e-3,
    ov_reg: float = 3e-4,
    hreg: float = 250.0,
) -> Tuple[float, np.ndarray]:
    """Evaluate exact trajectory loss and reverse-mode VJP gradient (2-tuple API)."""
    loss, grad, _ = _fast_loss_grad_and_rss(
        theta=theta,
        batch_data=batch_data,
        params=params,
        flux_l2=flux_l2,
        ov_reg=ov_reg,
        hreg=hreg,
    )
    return loss, grad


def _prepare_batch_tensors(
    trajectories: List[Dict[str, np.ndarray]],
    params: BioreactorParams,
    noise_std: float,
    rng: np.random.Generator,
    subsample_step: int = 2,
    atline_every: int = 5,
    include_x0_jitter: bool = False,
    batch_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Assemble strictly observable training tensors (ZERO unmeasured X_d observations)."""
    selected = trajectories if batch_indices is None else [trajectories[int(i)] for i in batch_indices]
    m_batches = len(selected)
    sl_t = selected[0]["t"][::subsample_step]
    t_steps = len(sl_t) - 1
    dt = float(sl_t[1] - sl_t[0])

    x0 = np.zeros((m_batches, 7), dtype=float)
    F_arr = np.zeros((t_steps, m_batches), dtype=float)
    Sf_arr = np.full((t_steps, m_batches), params.S_feed, dtype=float)
    kla_nom_arr = np.zeros((t_steps, m_batches), dtype=float)
    v_tip_arr = np.zeros((t_steps, m_batches), dtype=float)
    y_obs = np.zeros((t_steps + 1, m_batches, 7), dtype=float)
    w_obs = np.zeros((t_steps + 1, m_batches, 7), dtype=float)
    inv_var_obs = np.zeros((t_steps + 1, m_batches, 7), dtype=float)

    scales_obs = {0: 25.0, 2: 4.0, 3: 8.2, 4: 3.0, 5: 12.0}
    chw = {0: 1.2, 2: 1.0, 3: 0.8, 4: 1.8, 5: 1.0}
    idx_at = np.arange(0, t_steps + 1, atline_every, dtype=int)
    idx_on = np.arange(0, t_steps + 1, 1, dtype=int)
    noise_eff = max(1e-4, float(noise_std))

    for m, sim in enumerate(selected):
        st_sub = sim["states"][::subsample_step]
        f_sub = sim["F"][::subsample_step]
        rpm_sub = sim["rpm"][::subsample_step]
        vvm_sub = sim["vvm"][::subsample_step]
        v_sub = st_sub[:, 6]

        x0[m] = st_sub[0].copy()
        F_arr[:, m] = f_sub[:t_steps]
        for t in range(t_steps):
            kla_nom_arr[t, m] = nominal_kla(rpm_sub[t], vvm_sub[t], max(0.2, float(v_sub[t])), params)
            v_tip_arr[t, m] = compute_hydrodynamics(
                rpm_sub[t], vvm_sub[t], max(0.2, float(v_sub[t])), params
            )["v_tip"]

        for c in (0, 3):
            sig = noise_eff * scales_obs[c]
            y_obs[idx_on, m, c] = np.maximum(0.0, st_sub[idx_on, c] + rng.normal(0.0, sig, size=len(idx_on)))
            if c == 3:
                y_obs[idx_on, m, c] = np.minimum(y_obs[idx_on, m, c], params.C_star)
            if include_x0_jitter:
                y_obs[idx_on, m, c] = np.maximum(
                    0.0, y_obs[idx_on, m, c] + rng.normal(0.0, 0.004 * scales_obs[c], size=len(idx_on))
                )
            w_obs[idx_on, m, c] = chw[c] / (scales_obs[c] ** 2) * (100.0 / (len(idx_on) * m_batches))
            inv_var_obs[idx_on, m, c] = 1.0 / (sig ** 2)

        for c in (2, 4, 5):
            sig = noise_eff * scales_obs[c]
            y_obs[idx_at, m, c] = np.maximum(0.0, st_sub[idx_at, c] + rng.normal(0.0, sig, size=len(idx_at)))
            if include_x0_jitter:
                y_obs[idx_at, m, c] = np.maximum(
                    0.0, y_obs[idx_at, m, c] + rng.normal(0.0, 0.004 * scales_obs[c], size=len(idx_at))
                )
            w_obs[idx_at, m, c] = chw[c] / (scales_obs[c] ** 2) * (100.0 / (len(idx_at) * m_batches))
            inv_var_obs[idx_at, m, c] = 1.0 / (sig ** 2)

    return {
        "x0": x0,
        "F": F_arr,
        "S_feed": Sf_arr,
        "kla_nom": kla_nom_arr,
        "v_tip": v_tip_arr,
        "y_obs": y_obs,
        "w_obs": w_obs,
        "inv_var_obs": inv_var_obs,
        "dt": np.array(dt, dtype=float),
    }


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _fit_bootstrap_member(args: Tuple[Any, ...]) -> Tuple[int, np.ndarray, bool, int]:
    """Train one Dirichlet-weighted + residual-perturbed bootstrap ensemble member."""
    b_idx, th_base, lb, ub, ds, l2_w, specs, bdata, p_dict, boot_seed, h_ace, h_mu, maxiter = args
    p = BioreactorParams(**p_dict)
    rng = np.random.default_rng(boot_seed)
    th_fresh = _gam_init_weights(1000 + 37 * boot_seed + 13 * b_idx, specs, lb, ub, h_ace=h_ace, h_mu=h_mu)
    m_batches = bdata["F"].shape[1]
    dir_w = rng.dirichlet(np.full(m_batches, 1.3)) * m_batches
    w_boot = bdata["w_obs"] * dir_w[None, :, None]
    mask_obs = bdata["w_obs"] > 0.0
    sig_grid = np.where(mask_obs, 1.0 / np.sqrt(np.maximum(1e-12, bdata["inv_var_obs"])), 0.0)
    y_boot = np.maximum(0.0, bdata["y_obs"] + rng.normal(0.0, 0.85 * sig_grid))
    b_boot = {**bdata, "w_obs": w_boot, "y_obs": y_boot}

    th0 = 0.88 * th_base + 0.12 * th_fresh + rng.normal(0.0, 0.08, size=len(th_base)) * ds
    th0 = np.clip(th0, lb + 1e-4, ub - 1e-4)
    bounds_u = list(zip(lb / ds, ub / ds))

    def obj_u(u: np.ndarray) -> Tuple[float, np.ndarray]:
        val, grad, _ = _fast_loss_grad_and_rss(u * ds, b_boot, p, specs=specs, l2_w=l2_w)
        return val, grad * ds

    res = minimize(
        obj_u,
        th0 / ds,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_u,
        options={"maxiter": int(maxiter), "ftol": 5e-8, "gtol": 1.2e-2, "maxcor": 15},
    )
    return b_idx, res.x * ds, bool(res.success), int(res.nit)


def _compute_cov_inv_and_p99(
    Z: np.ndarray, reg: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute standardized subspace Mahalanobis statistics and 99th-percentile threshold."""
    mean = np.mean(Z, axis=0)
    std = np.maximum(1e-6, np.std(Z, axis=0))
    zn = (Z - mean[None, :]) / std[None, :]
    cov = np.cov(zn, rowvar=False) if zn.shape[1] > 1 else np.array([[1.0]], dtype=float)
    cov_reg = np.atleast_2d(cov) + reg * np.eye(zn.shape[1])
    inv_cov = np.linalg.inv(cov_reg)
    d2 = np.sum((zn @ inv_cov) * zn, axis=1)
    p99 = float(np.quantile(np.sqrt(np.maximum(0.0, d2)), 0.99))
    return mean, inv_cov, p99, std


class PhysicsInformedNeuralODE:
    """Parsimonious bootstrap-aggregated GAM-86 Neural ODE with power-law rheology and subspace DoV."""

    def __init__(self, hidden_dim: int = 5, seed: int = 7, n_bootstrap: int = 15) -> None:
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.n_bootstrap = max(1, int(n_bootstrap))
        self.h_ace = 5
        self.h_mu = 4
        self.specs = _GAM_SPECS
        self.n_params = _N_PARAMS
        self.weights = _gam_init_weights(seed)
        self._unpacked_weights = _unpack_theta(self.weights, self.specs)
        self._unpacked_src = self.weights
        self.ensemble_weights: List[np.ndarray] = [self.weights.copy()]
        self.dov_stats: Dict[str, Tuple[np.ndarray, ...]] = {}
        self.training_history: List[float] = []
        self.fit_metadata: Dict[str, float] = {}
        self.is_trained = False
        self.weights_sha256 = ""

    def _update_checksum(self) -> None:
        arr = np.vstack(self.ensemble_weights) if self.ensemble_weights else self.weights[None, :]
        self.weights_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()
        self._unpacked_weights = _unpack_theta(self.weights, self.specs)
        self._unpacked_src = self.weights

    @staticmethod
    def _mahalanobis(
        z: np.ndarray,
        mu: np.ndarray,
        cov_inv: np.ndarray,
        std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        diff = z - mu[None, :]
        if std is not None:
            diff = diff / std[None, :]
        d2 = np.sum((diff @ cov_inv) * diff, axis=1)
        return np.sqrt(np.maximum(0.0, d2))

    def _compute_dov_vectorized(
        self,
        X_tot: np.ndarray,
        S: np.ndarray,
        C_L: np.ndarray,
        A: np.ndarray,
        v_tip: np.ndarray,
        Pg_V: np.ndarray,
        v_s: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        del Pg_V, v_s  # Strict scale invariance: intensive reactor state & tip speed only
        n = len(X_tot)
        if not self.dov_stats:
            return {
                "d_m": np.zeros(n),
                "d_m_kla": np.zeros(n),
                "d_m_kd": np.zeros(n),
                "d_m_met": np.zeros(n),
                "in_domain": np.ones(n, dtype=bool),
            }

        s_pos = np.maximum(0.0, S)
        cl_pos = np.maximum(0.0, C_L)
        a_pos = np.maximum(0.0, A)

        z_kd = np.column_stack(
            [
                1.0 / (1.0 + s_pos / 0.05),
                1.0 / (1.0 + cl_pos / 0.50),
                2.0 * ((a_pos / (1.5 + a_pos)) ** 2),
                np.maximum(0.0, v_tip - _VTIP_ONSET),
            ]
        )
        z_kla = np.maximum(0.0, X_tot)[:, None]
        z_met = np.column_stack(
            [
                np.log1p(s_pos / 0.1) / 3.0,
                np.log1p(cl_pos / 0.5) / 3.0,
                np.log1p(a_pos / 0.5) / 3.0,
                s_pos / 5.0,
                a_pos / 3.0,
            ]
        )

        mu_kd, ic_kd, _, sd_kd = self.dov_stats["kd"]
        mu_kl, ic_kl, _, sd_kl = self.dov_stats["kla"]
        mu_mt, ic_mt, _, sd_mt = self.dov_stats["met"]

        dm_kd = self._mahalanobis(z_kd, mu_kd, ic_kd, sd_kd)
        dm_kla = self._mahalanobis(z_kla, mu_kl, ic_kl, sd_kl)
        dm_met = self._mahalanobis(z_met, mu_mt, ic_mt, sd_mt)
        d_m = np.maximum.reduce([dm_kd, dm_kla, dm_met])
        p99_eff = float(self.dov_stats["global"][2])
        return {
            "d_m": d_m,
            "d_m_kla": dm_kla,
            "d_m_kd": dm_kd,
            "d_m_met": dm_met,
            "in_domain": d_m <= p99_eff,
        }

    def predict_batch_with_uq(
        self,
        X_v: np.ndarray,
        X_d: np.ndarray,
        S: np.ndarray,
        C_L: np.ndarray,
        A: np.ndarray,
        v_tip: np.ndarray,
        Pg_V: np.ndarray,
        v_s: np.ndarray,
        mu_mech: np.ndarray,
        V: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Vectorized bootstrap-aggregated prediction, calibrated epistemic std, and subspace DoV."""
        del V  # Strict scale invariance: zero reactor volume bypass
        X_tot = np.maximum(0.0, np.asarray(X_v, dtype=float)) + np.maximum(0.0, np.asarray(X_d, dtype=float))
        S_a = np.maximum(0.0, np.asarray(S, dtype=float))
        CL_a = np.maximum(0.0, np.asarray(C_L, dtype=float))
        A_a = np.maximum(0.0, np.asarray(A, dtype=float))
        vt_a = np.asarray(v_tip, dtype=float)
        pg_a = np.asarray(Pg_V, dtype=float)
        vs_a = np.asarray(v_s, dtype=float)
        mum_a = np.asarray(mu_mech, dtype=float)

        members = self.ensemble_weights if self.ensemble_weights else [self.weights]
        B = len(members)
        n = len(X_tot)
        dkd_l = np.empty((B, n), dtype=float)
        ek_l = np.empty((B, n), dtype=float)
        qp_l = np.empty((B, n), dtype=float)
        qc_l = np.empty((B, n), dtype=float)
        emu_l = np.empty((B, n), dtype=float)
        for b, th in enumerate(members):
            dkd_l[b], ek_l[b], qp_l[b], qc_l[b], emu_l[b] = _gam_eval_heads(
                th, X_tot, S_a, CL_a, A_a, vt_a, mum_a, specs=self.specs
            )

        dkd_m, ek_m, qp_m, qc_m, emu_m = _gam_eval_heads(
            self.weights, X_tot, S_a, CL_a, A_a, vt_a, mum_a, specs=self.specs
        )

        var_kd = np.var(dkd_l, axis=0) if B > 1 else np.zeros(n, dtype=float)
        var_ek = np.var(ek_l, axis=0) if B > 1 else np.zeros(n, dtype=float)
        var_qp = np.var(qp_l, axis=0) if B > 1 else np.zeros(n, dtype=float)
        var_qc = np.var(qc_l, axis=0) if B > 1 else np.zeros(n, dtype=float)
        var_mu = np.var(emu_l, axis=0) if B > 1 else np.zeros(n, dtype=float)

        dov = self._compute_dov_vectorized(X_tot, S_a, CL_a, A_a, vt_a, pg_a, vs_a)
        if self.dov_stats:
            p99_kd = float(self.dov_stats["kd"][2])
            p99_kla = float(self.dov_stats["kla"][2])
            p99_met = float(self.dov_stats["met"][2])

            ex_kd = np.maximum(0.0, dov["d_m_kd"] - p99_kd)
            ex_kla = np.maximum(0.0, dov["d_m_kla"] - p99_kla)
            ex_met = np.maximum(0.0, dov["d_m_met"] - p99_met)

            w_kd = np.exp(-0.5 * (ex_kd / max(0.5, 0.5 * p99_kd)) ** 2)
            w_met = np.exp(-0.5 * (ex_met / max(0.35, 0.35 * p99_met)) ** 2)

            dkd_m = w_kd * dkd_m
            qp_m = w_met * qp_m
            qc_m = w_met * qc_m
            emu_m = w_met * emu_m + (1.0 - w_met)

            r_kd = np.clip(ex_kd / max(1.0, p99_kd), 0.0, 2.0)
            r_kla = np.clip(ex_kla / max(1.0, p99_kla), 0.0, 2.0)
            r_met = np.clip(ex_met / max(1.0, p99_met), 0.0, 2.0)

            std_kd = np.sqrt(var_kd + (0.002 + 0.015 * r_kd) ** 2)
            std_kla = np.sqrt(var_ek + (0.025 + 0.065 * np.minimum(1.0, r_kla) * (1.0 - ek_m)) ** 2)
            std_qp = np.sqrt(var_qp + (0.002 + 0.015 * r_met) ** 2)
            std_qc = np.sqrt(var_qc + (0.002 + 0.015 * r_met) ** 2)
            std_mu = np.sqrt(var_mu + (0.015 + 0.12 * r_met) ** 2)
        else:
            std_kd = np.sqrt(var_kd + 0.002 ** 2)
            std_kla = np.sqrt(var_ek + 0.025 ** 2)
            std_qp = np.sqrt(var_qp + 0.002 ** 2)
            std_qc = np.sqrt(var_qc + 0.002 ** 2)
            std_mu = np.sqrt(var_mu + 0.015 ** 2)

        return {
            "delta_kd": dkd_m,
            "eta_kla": ek_m,
            "q_A_prod": qp_m,
            "q_A_cons": qc_m,
            "eta_mu": emu_m,
            "std_kd": std_kd,
            "std_kla": std_kla,
            "std_qp": std_qp,
            "std_qc": std_qc,
            "std_mu": std_mu,
            "d_m": dov["d_m"],
            "d_m_kla": dov["d_m_kla"],
            "in_domain": dov["in_domain"],
        }

    def predict_point_fast(
        self,
        X_v: float,
        X_d: float = 0.0,
        S: float = 0.2,
        C_L: float = 4.0,
        A: float = 0.0,
        v_tip: float = 1.2,
        Pg_V: float = 1000.0,
        v_s: float = 0.01,
        mu_mech: float = 0.0,
        V: Optional[float] = None,
    ) -> Dict[str, float]:
        """Fast deterministic single-point GAM-86 prediction for Radau ODE RHS evaluations."""
        del Pg_V, v_s, V
        if getattr(self, "_unpacked_src", None) is not self.weights:
            self._unpacked_weights = _unpack_theta(self.weights, self.specs)
            self._unpacked_src = self.weights

        (
            kd_w1, kd_b1, kd_w2,
            kla_c, kla_p,
            ace_W1, ace_b1, ace_W2, ace_b2,
            mu_W1, mu_b1, mu_w2, mu_b2,
        ) = self._unpacked_weights

        s_pos = max(0.0, float(S))
        cl_pos = max(0.0, float(C_L))
        a_pos = max(0.0, float(A))
        xt_pos = max(0.0, float(X_v)) + max(0.0, float(X_d))
        vt = float(v_tip)

        s_avail = s_pos / (0.002 + s_pos)
        o2_avail = cl_pos / (0.005 + cl_pos)
        a_avail = a_pos / (0.25 + a_pos)

        zk = np.array(
            [
                1.0 / (1.0 + s_pos / 0.05),
                1.0 / (1.0 + cl_pos / 0.50),
                2.0 * ((a_pos / (1.5 + a_pos)) ** 2),
                max(0.0, vt - _VTIP_ONSET),
            ],
            dtype=float,
        )
        phi_kd = _sigmoid(zk[:, None] * kd_w1 + kd_b1) - _sigmoid(kd_b1)
        dkd_val = max(0.0, float(0.05 * np.sum(phi_kd * kd_w2)))

        u0 = 1.0 + xt_pos / 10.0
        u1 = 1.0 + xt_pos / 25.0
        psi_kla = kla_c[0] * (u0 ** kla_p[0] - 1.0) + kla_c[1] * (u1 ** kla_p[1] - 1.0)
        ek_val = float(1.0 / (1.0 + max(0.0, psi_kla)))

        zm = np.array([s_pos / (0.5 + s_pos), np.log1p(s_pos * 0.5) * 0.4, a_pos / (3.0 + a_pos)], dtype=float)
        emu_val = float(0.05 + _sigmoid(float(np.tanh(mu_W1 @ zm + mu_b1) @ mu_w2 + mu_b2[0])))
        mu_hyb = float(mu_mech) * emu_val

        za = np.array(
            [
                mu_hyb / 0.45,
                np.log1p(s_pos * 10.0) / 3.0,
                np.log1p(cl_pos * 2.0) / 3.0,
                np.log1p(a_pos * 2.0) / 3.0,
            ],
            dtype=float,
        )
        raw_ace = ace_W2 @ np.tanh(ace_W1 @ za + ace_b1) + ace_b2
        qp_val = float(0.08 * _softplus(raw_ace[0]) * s_avail)
        qc_val = float(0.08 * _softplus(raw_ace[1]) * a_avail * o2_avail)

        if self.dov_stats:
            z_met = np.array(
                [
                    np.log1p(s_pos / 0.1) / 3.0,
                    np.log1p(cl_pos / 0.5) / 3.0,
                    np.log1p(a_pos / 0.5) / 3.0,
                    s_pos / 5.0,
                    a_pos / 3.0,
                ],
                dtype=float,
            )
            mu_kd, ic_kd, p99_kd, sd_kd = self.dov_stats["kd"]
            mu_mt, ic_mt, p99_met, sd_mt = self.dov_stats["met"]
            d_kd = (zk - mu_kd) / sd_kd
            dm_kd = float(np.sqrt(max(0.0, float(d_kd @ ic_kd @ d_kd))))
            d_mt = (z_met - mu_mt) / sd_mt
            dm_met = float(np.sqrt(max(0.0, float(d_mt @ ic_mt @ d_mt))))
            w_kd = float(np.exp(-0.5 * (max(0.0, dm_kd - float(p99_kd)) / max(0.5, 0.5 * float(p99_kd))) ** 2))
            w_met = float(np.exp(-0.5 * (max(0.0, dm_met - float(p99_met)) / max(0.35, 0.35 * float(p99_met))) ** 2))
            dkd_val *= w_kd
            qp_val *= w_met
            qc_val *= w_met
            emu_val = w_met * emu_val + (1.0 - w_met)

        return {
            "delta_kd": float(dkd_val),
            "eta_kla": float(ek_val),
            "q_A_prod": float(qp_val),
            "q_A_cons": float(qc_val),
            "eta_mu": float(emu_val),
        }

    def predict_full_with_uq(
        self,
        X_v: float,
        X_d: float = 0.0,
        S: float = 0.2,
        C_L: float = 4.0,
        A: float = 0.0,
        v_tip: Optional[float] = None,
        Pg_V: Optional[float] = None,
        v_s: Optional[float] = None,
        mu_mech: float = 0.0,
        rpm: float = 450.0,
        vvm: float = 1.2,
        V: float = 2.0,
        params: Optional[BioreactorParams] = None,
    ) -> Dict[str, float]:
        """Single-point intensive prediction with calibrated epistemic UQ and subspace DoV."""
        if v_tip is None or Pg_V is None or v_s is None:
            hyd = compute_hydrodynamics(rpm, vvm, V, params or BioreactorParams())
            v_tip = hyd["v_tip"]
            Pg_V = hyd["Pg_V"]
            v_s = hyd["v_s"]

        out = self.predict_batch_with_uq(
            np.array([X_v]),
            np.array([X_d]),
            np.array([S]),
            np.array([C_L]),
            np.array([A]),
            np.array([v_tip]),
            np.array([Pg_V]),
            np.array([v_s]),
            np.array([mu_mech]),
        )
        return {
            k: (bool(v[0]) if k == "in_domain" else float(v[0]))
            for k, v in out.items()
        }

    def predict_residuals(
        self,
        X_v: float,
        S: float,
        C_L: float,
        A: float,
        rpm: float,
        vvm: float,
        V: float,
        X_d: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Backward-compatible 3-tuple predictor returning (delta_kd, eta_kla, q_A_net)."""
        p = BioreactorParams()
        hyd = compute_hydrodynamics(rpm, vvm, V, p)
        s_pos = max(0.0, float(S))
        cl_pos = max(0.0, float(C_L))
        mu_m = p.mu_max * (s_pos / (p.K_S + s_pos)) * (cl_pos / (p.K_O2 + cl_pos))
        out = self.predict_full_with_uq(
            X_v=X_v,
            X_d=X_d,
            S=S,
            C_L=C_L,
            A=A,
            v_tip=hyd["v_tip"],
            Pg_V=hyd["Pg_V"],
            v_s=hyd["v_s"],
            mu_mech=mu_m,
        )
        return float(out["delta_kd"]), float(out["eta_kla"]), float(out["q_A_prod"] - out["q_A_cons"])

    def fit(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        params: BioreactorParams,
        noise_std: float = 0.015,
        maxiter: int = 800,
        n_bootstrap: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train indirect trajectory-integrated GAM-86 + converged bootstrap ensemble (zero X_d)."""
        rng = np.random.default_rng(self.seed + 17)
        t0_wall = time.perf_counter()
        n_boot = self.n_bootstrap if n_bootstrap is None else max(1, int(n_bootstrap))

        batch_data = _prepare_batch_tensors(
            trajectories,
            params,
            noise_std=noise_std,
            rng=rng,
            subsample_step=2,
            atline_every=5,
            include_x0_jitter=False,
        )

        self.training_history = []
        bounds_u = list(zip(_LB / _D_SCALE, _UB / _D_SCALE))
        u0 = self.weights / _D_SCALE

        def obj_u(u: np.ndarray) -> Tuple[float, np.ndarray]:
            val, grad_theta, _ = _fast_loss_grad_and_rss(u * _D_SCALE, batch_data, params)
            self.training_history.append(val)
            return val, grad_theta * _D_SCALE

        res_base = minimize(
            obj_u,
            u0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_u,
            options={"maxiter": int(maxiter), "ftol": 1e-9, "gtol": 1e-2, "maxcor": 20},
        )
        base_theta = res_base.x * _D_SCALE
        _, grad_base, std_rss_obs = _fast_loss_grad_and_rss(base_theta, batch_data, params)
        self.weights = base_theta.copy()

        # Fit standardized subspace DoV ellipsoids (zero volume bypass)
        z_kla_l, z_kd_l, z_met_l = [], [], []
        for sim in trajectories:
            st = sim["states"][::2]
            xv, xd, s, cl, a, v_arr = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4], st[:, 6]
            rpm_arr, vvm_arr = sim["rpm"][::2], sim["vvm"][::2]
            xt = np.maximum(0.0, xv + xd)
            v_tip = np.array(
                [compute_hydrodynamics(r, vm, vl, params)["v_tip"] for r, vm, vl in zip(rpm_arr, vvm_arr, v_arr)]
            )
            z_kd_l.append(
                np.column_stack(
                    [
                        1.0 / (1.0 + np.maximum(0.0, s) / 0.05),
                        1.0 / (1.0 + np.maximum(0.0, cl) / 0.50),
                        2.0 * ((np.maximum(0.0, a) / (1.5 + np.maximum(0.0, a))) ** 2),
                        np.maximum(0.0, v_tip - _VTIP_ONSET),
                    ]
                )
            )
            z_kla_l.append(xt[:, None])
            z_met_l.append(
                np.column_stack(
                    [
                        np.log1p(np.maximum(0.0, s) / 0.1) / 3.0,
                        np.log1p(np.maximum(0.0, cl) / 0.5) / 3.0,
                        np.log1p(np.maximum(0.0, a) / 0.5) / 3.0,
                        np.maximum(0.0, s) / 5.0,
                        np.maximum(0.0, a) / 3.0,
                    ]
                )
            )

        dov_kd = _compute_cov_inv_and_p99(np.vstack(z_kd_l))
        dov_kla = _compute_cov_inv_and_p99(np.vstack(z_kla_l))
        dov_met = _compute_cov_inv_and_p99(np.vstack(z_met_l))
        dov_p99 = float(max(dov_kd[2], dov_kla[2], dov_met[2]))
        self.dov_stats = {
            "kd": dov_kd,
            "kla": dov_kla,
            "met": dov_met,
            "global": (np.zeros(1, dtype=float), np.eye(1, dtype=float), dov_p99, np.ones(1, dtype=float)),
        }

        # Train B Dirichlet-weighted + residual-perturbed bootstrap ensemble members
        self.ensemble_weights = []
        boot_converged = 1
        if n_boot == 1:
            self.ensemble_weights = [base_theta.copy()]
            boot_converged = int(bool(res_base.success))
        else:
            p_dict = params.to_dict()
            tasks = [
                (
                    b,
                    base_theta,
                    _LB,
                    _UB,
                    _D_SCALE,
                    _L2_W,
                    _GAM_SPECS,
                    batch_data,
                    p_dict,
                    self.seed + b,
                    self.h_ace,
                    self.h_mu,
                    180,
                )
                for b in range(n_boot)
            ]
            n_workers = min(4, max(1, (os.cpu_count() or 4) // 2), n_boot)
            main_mod = sys.modules.get("__main__")
            main_file = getattr(main_mod, "__file__", None)
            can_spawn = bool(
                main_file
                and str(main_file).endswith(".py")
                and os.path.isfile(str(main_file))
                and "pytest" not in sys.modules
                and "streamlit" not in sys.modules
            )
            if can_spawn and n_workers > 1:
                try:
                    ctx = multiprocessing.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=n_workers,
                        mp_context=ctx,
                        initializer=_init_worker,
                    ) as pool:
                        results = list(pool.map(_fit_bootstrap_member, tasks))
                except Exception:
                    results = [_fit_bootstrap_member(t) for t in tasks]
            else:
                results = [_fit_bootstrap_member(t) for t in tasks]

            self.ensemble_weights = [th for _, th, _, _ in results]
            boot_converged = sum(1 for _, _, conv, _ in results if conv)
            ens_mean = np.mean(self.ensemble_weights, axis=0)
            self.weights = ens_mean.copy()
            for sl in (slice(2, 8), slice(10, 16), slice(18, 24)):
                self.weights[sl] = base_theta[sl]

        self.is_trained = True
        self._update_checksum()

        n_obs = int(np.sum(batch_data["w_obs"] > 0.0))
        rss = max(1e-12, float(std_rss_obs))
        aic = float(n_obs * np.log(rss / max(1, n_obs)) + 2.0 * _N_PARAMS)
        bic = float(n_obs * np.log(rss / max(1, n_obs)) + _N_PARAMS * np.log(max(2, n_obs)))
        wall_s = float(time.perf_counter() - t0_wall)

        grad_inf = float(np.max(np.abs(grad_base * _D_SCALE)))
        self.fit_metadata = {
            "final_loss": float(res_base.fun),
            "std_rss_obs": float(std_rss_obs),
            "iterations": int(res_base.nit),
            "samples": n_obs,
            "n_params": _N_PARAMS,
            "n_bootstrap": len(self.ensemble_weights),
            "bootstrap_converged": int(boot_converged),
            "aic": aic,
            "bic": bic,
            "grad_inf_norm": grad_inf,
            "converged": bool(res_base.success or grad_inf < 1e-2),
            "dov_p99": dov_p99,
            "train_wall_s": wall_s,
        }
        return dict(self.fit_metadata)

    def evaluate_on_trajectories(
        self,
        test_trajectories: List[Dict[str, np.ndarray]],
        params: BioreactorParams,
        plant_params: Optional[BioreactorParams] = None,
    ) -> Dict[str, float]:
        """Compute R2 on dynamic rates and Radau rollout RMSE on given trajectories."""
        p_plant = plant_params or params
        kd_true_all, kd_pred_all = [], []
        kla_true_all, kla_pred_all = [], []
        qa_true_all, qa_pred_all = [], []
        mu_true_all, mu_pred_all = [], []
        hybrid_state_errs, nominal_state_errs = [], []

        for sim in test_trajectories:
            t = sim["t"]
            hyd_vtip = np.array(
                [compute_hydrodynamics(r, vm, vl, params)["v_tip"] for r, vm, vl in zip(sim["rpm"], sim["vvm"], sim["V"])]
            )
            hyd_pgv = np.array(
                [compute_hydrodynamics(r, vm, vl, params)["Pg_V"] for r, vm, vl in zip(sim["rpm"], sim["vvm"], sim["V"])]
            )
            hyd_vs = np.array(
                [compute_hydrodynamics(r, vm, vl, params)["v_s"] for r, vm, vl in zip(sim["rpm"], sim["vvm"], sim["V"])]
            )
            mu_mech_arr = (
                params.mu_max
                * (sim["S"] / (params.K_S + sim["S"]))
                * (sim["C_L"] / (params.K_O2 + sim["C_L"]))
            )
            out = self.predict_batch_with_uq(
                sim["X_v"], sim["X_d"], sim["S"], sim["C_L"], sim["A"],
                hyd_vtip, hyd_pgv, hyd_vs, mu_mech_arr,
            )
            kd_pred = params.k_d0 + out["delta_kd"]
            kla_pred = np.array(
                [nominal_kla(r, vm, vl, params) for r, vm, vl in zip(sim["rpm"], sim["vvm"], sim["V"])]
            ) * out["eta_kla"]
            qa_pred = out["q_A_prod"] - out["q_A_cons"]
            mu_pred = mu_mech_arr * out["eta_mu"]

            kd_true_all.append(sim["k_d"])
            kd_pred_all.append(kd_pred)
            kla_true_all.append(sim["k_La"])
            kla_pred_all.append(kla_pred)
            qa_true_all.append(sim["q_A_net"])
            qa_pred_all.append(qa_pred)
            mu_true_all.append(sim["mu"])
            mu_pred_all.append(mu_pred)

            def replay_u(t_query: float, state: np.ndarray, _sim=sim) -> OperatingInputs:
                del state
                idx = int(np.clip(np.searchsorted(_sim["t"], t_query), 0, len(_sim["t"]) - 1))
                return OperatingInputs(
                    F=float(_sim["F"][idx]),
                    rpm=float(_sim["rpm"][idx]),
                    vvm=float(_sim["vvm"][idx]),
                    S_feed=p_plant.S_feed,
                )

            dt = float(t[1] - t[0])
            sim_hyb = simulate_bioreactor(
                (t[0], t[-1]), sim["states"][0], replay_u, params=params, mode="hybrid", residual_model=self, dt_eval=dt
            )
            sim_nom = simulate_bioreactor(
                (t[0], t[-1]), sim["states"][0], replay_u, params=params, mode="nominal", dt_eval=dt
            )
            scales = np.array([15.0, 3.0, 4.0, 8.2, 2.0, 8.0])
            hybrid_state_errs.append(np.sqrt(np.mean(((sim_hyb["states"][:, :6] - sim["states"][:, :6]) / scales) ** 2)))
            nominal_state_errs.append(np.sqrt(np.mean(((sim_nom["states"][:, :6] - sim["states"][:, :6]) / scales) ** 2)))

        def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
            return 1.0 - ss_res / ss_tot

        kd_t, kd_p = np.concatenate(kd_true_all), np.concatenate(kd_pred_all)
        kla_t, kla_p = np.concatenate(kla_true_all), np.concatenate(kla_pred_all)
        qa_t, qa_p = np.concatenate(qa_true_all), np.concatenate(qa_pred_all)
        mu_t, mu_p = np.concatenate(mu_true_all), np.concatenate(mu_pred_all)

        return {
            "kd_r2": float(r2_score(kd_t, kd_p)),
            "kla_r2": float(r2_score(kla_t, kla_p)),
            "qAnet_r2": float(r2_score(qa_t, qa_p)),
            "mu_r2": float(r2_score(mu_t, mu_p)),
            "kd_rmse": float(np.sqrt(np.mean((kd_t - kd_p) ** 2))),
            "kla_rmse": float(np.sqrt(np.mean((kla_t - kla_p) ** 2))),
            "hybrid_norm_rmse": float(np.mean(hybrid_state_errs)),
            "nominal_norm_rmse": float(np.mean(nominal_state_errs)),
        }


def generate_training_and_test_batches(
    params: Optional[BioreactorParams] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Generate 10-batch orthogonal QbD training suite and 5 strictly held-out test regimes."""
    p = params or BioreactorParams()

    def make_x0(xv: float, xd: float, s: float, a: float = 0.0) -> np.ndarray:
        x = default_initial_state(p)
        x[0] = xv
        x[1] = xd
        x[2] = s
        x[4] = a
        return x

    train_specs = [
        ("tr_std", make_x0(1.2, 0.02, 0.6), lambda t, s: OperatingInputs(F=float(0.015 * np.exp(0.16 * t)), rpm=460.0, vvm=1.3)),
        ("tr_over", make_x0(1.5, 0.02, 2.2), lambda t, s: OperatingInputs(F=float(0.032 * np.exp(0.10 * t)), rpm=460.0, vvm=1.2)),
        ("tr_hypox", make_x0(6.0, 0.05, 0.45), lambda t, s: OperatingInputs(F=float(0.010 * np.exp(0.08 * t)), rpm=270.0 if t > 1.5 else 450.0, vvm=0.30 if t > 1.5 else 1.2)),
        ("tr_hypox_rec", make_x0(5.5, 0.05, 0.65), lambda t, s: OperatingInputs(F=float(0.018 if t < 5.0 else 0.008), rpm=280.0 if (2.5 <= t < 7.0) else 490.0, vvm=0.35 if (2.5 <= t < 7.0) else 1.4)),
        ("tr_hdens", make_x0(6.5, 0.10, 0.7), lambda t, s: OperatingInputs(F=float(0.024 * np.exp(0.12 * t)), rpm=480.0, vvm=1.25)),
        ("tr_shear", make_x0(3.5, 0.05, 1.0), lambda t, s: OperatingInputs(F=float(0.022 * np.exp(0.11 * t)), rpm=800.0 if t > 1.5 else 450.0, vvm=1.5)),
        ("tr_starve", make_x0(5.5, 0.05, 0.4), lambda t, s: OperatingInputs(F=float(0.002 if t > 2.5 else 0.018), rpm=460.0, vvm=1.4)),
        ("tr_acetate", make_x0(4.5, 0.05, 0.45, a=3.2), lambda t, s: OperatingInputs(F=float(0.012 * np.exp(0.08 * t)), rpm=480.0, vvm=1.4)),
        ("tr_over_ace", make_x0(1.8, 0.03, 1.8, a=1.2), lambda t, s: OperatingInputs(F=float(0.038 * np.exp(0.08 * t)), rpm=520.0, vvm=1.4)),
        (
            "tr_idoe",
            make_x0(2.2, 0.03, 0.9),
            lambda t, s: OperatingInputs(
                F=float(0.030 if (2.0 <= t < 5.0 or 9.0 <= t < 12.0) else 0.007),
                rpm=700.0 if (6.0 <= t < 10.0) else 440.0,
                vvm=0.65 if t >= 10.0 else 1.4,
            ),
        ),
    ]
    test_specs = [
        ("test_val", make_x0(1.5, 0.03, 0.7), lambda t, s: OperatingInputs(F=float(0.026 * np.exp(0.13 * t)), rpm=600.0 if t > 7.0 else 440.0, vvm=0.85 if t > 6.0 else 1.3)),
        ("unseen_shear_step", make_x0(3.0, 0.05, 0.7), lambda t, s: OperatingInputs(F=float(0.022 * np.exp(0.10 * t)), rpm=720.0 if t >= 5.0 else 440.0, vvm=1.3)),
        ("unseen_hypox_dip", make_x0(4.0, 0.05, 0.8), lambda t, s: OperatingInputs(F=float(0.020 * np.exp(0.08 * t)), rpm=300.0 if (4.0 <= t < 8.0) else 460.0, vvm=0.45 if (4.0 <= t < 8.0) else 1.3)),
        ("unseen_starve_pulse", make_x0(3.5, 0.05, 0.5), lambda t, s: OperatingInputs(F=float(0.001 if (4.0 <= t < 7.0) else 0.022 * np.exp(0.08 * t)), rpm=460.0, vvm=1.3)),
        ("unseen_high_overflow", make_x0(1.5, 0.03, 1.5, a=1.0), lambda t, s: OperatingInputs(F=float(0.035 * np.exp(0.09 * t)), rpm=480.0, vvm=1.2)),
    ]

    train_batches = [
        simulate_bioreactor((0.0, 14.0), x0, fn, params=p, mode="ground_truth", dt_eval=0.1)
        for _, x0, fn in train_specs
    ]
    test_batches = [
        simulate_bioreactor((0.0, 14.0), x0, fn, params=p, mode="ground_truth", dt_eval=0.1)
        for _, x0, fn in test_specs
    ]
    return train_batches, test_batches


def train_default_hybrid_model(
    params: Optional[BioreactorParams] = None,
    noise_std: float = 0.015,
    seed: int = 7,
    plant_params: Optional[BioreactorParams] = None,
    n_bootstrap: int = 15,
) -> Tuple[PhysicsInformedNeuralODE, Dict[str, float]]:
    """Train and evaluate bootstrap-aggregated GAM-86 hybrid model on strictly held-out test batches."""
    p_model = params or BioreactorParams()
    p_plant = plant_params or BioreactorParams()
    train_batches, test_batches = generate_training_and_test_batches(p_plant)

    model = PhysicsInformedNeuralODE(seed=seed, n_bootstrap=n_bootstrap)
    fit_stats = model.fit(train_batches, p_model, noise_std=noise_std, maxiter=800, n_bootstrap=n_bootstrap)

    test_stats = model.evaluate_on_trajectories(test_batches, p_model, plant_params=p_plant)
    train_stats = model.evaluate_on_trajectories(train_batches, p_model, plant_params=p_plant)

    eval_stats = {
        "kd_r2": test_stats["kd_r2"],
        "kla_r2": test_stats["kla_r2"],
        "qAnet_r2": test_stats["qAnet_r2"],
        "mu_r2": test_stats["mu_r2"],
        "kd_rmse": test_stats["kd_rmse"],
        "kla_rmse": test_stats["kla_rmse"],
        "hybrid_norm_rmse": test_stats["hybrid_norm_rmse"],
        "nominal_norm_rmse": test_stats["nominal_norm_rmse"],
        "train_kd_r2": train_stats["kd_r2"],
        "train_kla_r2": train_stats["kla_r2"],
        "train_qAnet_r2": train_stats["qAnet_r2"],
        "train_mu_r2": train_stats["mu_r2"],
        "train_hybrid_norm_rmse": train_stats["hybrid_norm_rmse"],
        "train_nominal_norm_rmse": train_stats["nominal_norm_rmse"],
    }
    return model, {**fit_stats, **eval_stats}
