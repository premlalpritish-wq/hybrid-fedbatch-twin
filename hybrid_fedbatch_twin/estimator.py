"""Extended Kalman Filter (EKF) state estimator and Portela Level 3 Digital Twin session."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
from scipy.linalg import expm

from .mechanistic import (
    BioreactorParams,
    ControlCallback,
    OperatingInputs,
    ResidualCorrectorProtocol,
    STATE_NAMES,
    bioreactor_rhs,
    compute_rates,
    disturbance_profile,
    integrate_step,
)

# Identical continuous-time base process-noise spectral density shared by hybrid and nominal EKFs
Q_DEFAULT = np.array([0.04, 5e-4, 0.04, 0.18, 0.04, 0.04, 1e-4], dtype=float)
P0_DEFAULT = np.array([0.04, 0.01, 0.05, 0.10, 0.01, 0.02, 1e-4], dtype=float)


def _canonical_sha256(payload: Any) -> str:
    """Compute deterministic SHA-256 digest of JSON-serializable metadata."""
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_cov_matrix(
    spec: Optional[Sequence[float]],
    default_diag: np.ndarray,
    name: str,
) -> np.ndarray:
    """Parse a 1D variance diagonal or 2D symmetric PSD (7, 7) covariance matrix."""
    if spec is None:
        return np.diag(np.asarray(default_diag, dtype=float))
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (7,) or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(f"{name} 1D array must have 7 finite non-negative entries.")
        return np.diag(arr)
    if arr.ndim == 2:
        if arr.shape != (7, 7) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} 2D matrix must be finite with shape (7, 7).")
        if not np.allclose(arr, arr.T, atol=1e-10):
            raise ValueError(f"{name} 2D matrix must be symmetric.")
        sym = 0.5 * (arr + arr.T)
        if float(np.min(np.linalg.eigvalsh(sym))) < -1e-12:
            raise ValueError(f"{name} 2D matrix must be positive semi-definite.")
        return sym
    raise ValueError(f"{name} must be 1D (7,) or 2D (7, 7).")


class HybridExtendedKalmanFilter:
    """Continuous-discrete EKF pairing stiff Radau prediction with epistemic covariance propagation."""

    def __init__(
        self,
        x0: np.ndarray,
        params: Optional[BioreactorParams] = None,
        mode: Literal["hybrid", "nominal"] = "hybrid",
        residual_model: Optional[ResidualCorrectorProtocol] = None,
        P0_diag: Optional[Sequence[float]] = None,
        Q_diag: Optional[Sequence[float]] = None,
    ) -> None:
        if mode == "hybrid" and residual_model is None:
            raise ValueError("residual_model must be supplied for mode='hybrid'.")
        self.params = params or BioreactorParams()
        self.mode = mode
        self.residual_model = residual_model
        self.x = np.maximum(0.0, np.asarray(x0, dtype=float).copy())
        self.P = _parse_cov_matrix(P0_diag, P0_DEFAULT, "P0_diag")
        self.Q_base = _parse_cov_matrix(Q_diag, Q_DEFAULT, "Q_diag")
        self.Q = self.Q_base
        self.last_epistemic_cov = np.zeros((7, 7), dtype=float)
        self.last_rates: Dict[str, float] = {}
        self.last_nis: float = 0.0

    def _jacobian(self, x: np.ndarray, u: OperatingInputs) -> np.ndarray:
        """Central finite-difference Jacobian F = d(bioreactor_rhs)/dx."""
        n = len(x)
        F_mat = np.zeros((n, n), dtype=float)
        for j in range(n):
            eps = 1e-5 * max(1.0, abs(float(x[j])))
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] = max(0.0, xm[j] - eps)
            denom = xp[j] - xm[j]
            if denom <= 0.0:
                continue
            fp = bioreactor_rhs(0.0, xp, u, self.params, mode=self.mode, residual_model=self.residual_model)
            fm = bioreactor_rhs(0.0, xm, u, self.params, mode=self.mode, residual_model=self.residual_model)
            F_mat[:, j] = (fp - fm) / denom
        return F_mat

    def _epistemic_process_cov(self, x: np.ndarray, rates: Dict[str, float]) -> np.ndarray:
        """Map bootstrap rate std [delta_kd, eta_kla, q_A_prod, q_A_cons, eta_mu] via stoichiometry G."""
        if self.mode != "hybrid":
            return np.zeros((7, 7), dtype=float)

        X_v = max(0.0, float(x[0]))
        S = max(0.0, float(x[2]))
        C_L = max(0.0, float(x[3]))
        A = max(0.0, float(x[4]))

        s_av = S / (0.002 + S)
        o2_av = C_L / (0.005 + C_L)
        a_av = A / (0.25 + A)
        mu_m = float(rates["mu_mech"])
        k_la_nom = float(rates["k_La_nom"])

        G = np.zeros((7, 5), dtype=float)
        # Col 0: delta_kd (1/h)
        G[0, 0] = -X_v
        G[1, 0] = +X_v
        # Col 1: eta_kla (-)
        G[3, 1] = k_la_nom * (self.params.C_star - C_L)
        # Col 2: q_A_prod (g A / (g X * h))
        G[2, 2] = -X_v / self.params.Y_AS
        G[4, 2] = +X_v
        # Col 3: q_A_cons (g A / (g X * h))
        G[3, 3] = -1000.0 * X_v / self.params.Y_AO
        G[4, 3] = -X_v
        # Col 4: eta_mu (-)
        G[0, 4] = mu_m * X_v
        G[2, 4] = -(mu_m / self.params.Y_XS + self.params.alpha * mu_m * s_av * o2_av / self.params.Y_PS) * X_v
        G[3, 4] = -1000.0 * (mu_m / self.params.Y_XO) * X_v
        G[5, 4] = self.params.alpha * mu_m * s_av * o2_av * X_v

        std_vec = np.array(
            [
                float(rates.get("std_kd", 0.0)),
                float(rates.get("std_kla", 0.0)),
                float(rates.get("std_qp", 0.0)),
                float(rates.get("std_qc", 0.0)),
                float(rates.get("std_mu", 0.0)),
            ],
            dtype=float,
        )
        Sigma_nn = np.diag(std_vec ** 2)
        Q_ep = G @ Sigma_nn @ G.T
        return 0.5 * (Q_ep + Q_ep.T)

    def predict(self, u: OperatingInputs, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state via stiff Radau ODE and covariance via unclipped expm(F * dt)."""
        dt_val = float(dt)
        if dt_val <= 0.0:
            return self.x.copy(), self.P.copy()

        x_pred = integrate_step(
            0.0,
            dt_val,
            self.x,
            u,
            self.params,
            mode=self.mode,
            residual_model=self.residual_model,
            rtol=1e-6,
            atol=1e-8,
        )
        x_mid = 0.5 * (self.x + x_pred)
        F_mat = self._jacobian(x_mid, u)
        Phi = expm(F_mat * dt_val)

        rates_mid = compute_rates(
            x_mid,
            u,
            self.params,
            mode=self.mode,
            residual_model=self.residual_model,
        )
        Q_ep = self._epistemic_process_cov(x_mid, rates_mid)
        self.last_epistemic_cov = Q_ep
        self.last_rates = rates_mid

        self.P = Phi @ self.P @ Phi.T + (self.Q_base + Q_ep) * max(dt_val, 1e-4)
        self.P = 0.5 * (self.P + self.P.T)
        self.x = x_pred
        return self.x.copy(), self.P.copy()

    def update(
        self,
        y: np.ndarray,
        obs_indices: Sequence[int],
        R_diag: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assimilate measurements y at obs_indices using Joseph-stabilized Kalman update."""
        idx = list(obs_indices)
        m = len(idx)
        n = len(self.x)
        H = np.zeros((m, n), dtype=float)
        for row, state_idx in enumerate(idx):
            H[row, state_idx] = 1.0

        y_vec = np.asarray(y, dtype=float)
        R = np.diag(np.asarray(R_diag, dtype=float))
        S_cov = H @ self.P @ H.T + R
        S_cov = 0.5 * (S_cov + S_cov.T)

        K = np.linalg.solve(S_cov, H @ self.P).T
        innov = y_vec - H @ self.x
        try:
            self.last_nis = float(innov.T @ np.linalg.solve(S_cov, innov))
        except np.linalg.LinAlgError:
            self.last_nis = 0.0

        self.x = self.x + K @ innov
        self.x = np.maximum(0.0, self.x)
        self.x[3] = min(self.x[3], self.params.C_star * 1.05)
        self.x[6] = max(0.2, min(self.x[6], self.params.V_max + 0.05))

        IKH = np.eye(n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy()

    def current_rates(self, u: OperatingInputs) -> Dict[str, float]:
        """Return biological, mass-transfer, UQ, and DoV rates at current state estimate."""
        return compute_rates(
            self.x,
            u,
            self.params,
            mode=self.mode,
            residual_model=self.residual_model,
        )


def _resolve_operating_inputs(
    controller_fn: ControlCallback,
    t_val: float,
    x_state: np.ndarray,
) -> OperatingInputs:
    """Query controller at (t_val, x_state) so disturbance rpm/vvm apply with zero lag."""
    return controller_fn(float(t_val), np.asarray(x_state, dtype=float))


def run_twin_estimation_loop(
    t_span: Tuple[float, float],
    x0: np.ndarray,
    controller_fn: ControlCallback,
    params: BioreactorParams,
    residual_model: ResidualCorrectorProtocol,
    dt_step: float = 0.1,
    atline_interval_steps: int = 10,
    seed: int = 123,
    noise_scale: float = 1.0,
    plant_params: Optional[BioreactorParams] = None,
    disturbance_mode: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Run closed-loop ground-truth plant alongside sensor-parity Hybrid & Nominal EKFs."""
    rng = np.random.default_rng(seed)
    p_plant = plant_params or params
    dist_mode = (
        str(getattr(controller_fn, "disturbance_mode", "none"))
        if disturbance_mode is None
        else str(disturbance_mode)
    )

    t0, t1 = float(t_span[0]), float(t_span[1])
    n_steps = max(1, int(round((t1 - t0) / dt_step)))
    t_grid = np.linspace(t0, t1, n_steps + 1)

    true_states = np.zeros((n_steps + 1, 7), dtype=float)
    est_hyb = np.zeros((n_steps + 1, 7), dtype=float)
    est_nom = np.zeros((n_steps + 1, 7), dtype=float)
    std_hyb = np.zeros((n_steps + 1, 7), dtype=float)
    std_nom = np.zeros((n_steps + 1, 7), dtype=float)
    meas_states = np.full((n_steps + 1, 7), np.nan, dtype=float)
    inputs = np.zeros((n_steps + 1, 3), dtype=float)

    mu_true = np.zeros(n_steps + 1, dtype=float)
    mu_hyb = np.zeros(n_steps + 1, dtype=float)
    mu_nom = np.zeros(n_steps + 1, dtype=float)
    mu_mech = np.zeros(n_steps + 1, dtype=float)
    eta_mu_hyb = np.zeros(n_steps + 1, dtype=float)
    kd_true = np.zeros(n_steps + 1, dtype=float)
    kd_hyb = np.zeros(n_steps + 1, dtype=float)
    kla_true = np.zeros(n_steps + 1, dtype=float)
    kla_hyb = np.zeros(n_steps + 1, dtype=float)
    qAnet_true = np.zeros(n_steps + 1, dtype=float)
    qAnet_hyb = np.zeros(n_steps + 1, dtype=float)
    d_m_arr = np.zeros(n_steps + 1, dtype=float)
    d_m_kla_arr = np.zeros(n_steps + 1, dtype=float)
    in_domain_arr = np.ones(n_steps + 1, dtype=float)
    nis_hyb = np.zeros(n_steps + 1, dtype=float)
    nis_nom = np.zeros(n_steps + 1, dtype=float)
    obs_mask_atline = np.zeros(n_steps + 1, dtype=bool)

    x_init = np.maximum(0.0, np.asarray(x0, dtype=float))
    true_states[0] = x_init
    est_hyb[0] = x_init
    est_nom[0] = x_init

    # Equal base Q_DEFAULT and equal P0 across hybrid and nominal EKFs
    ekf_hyb = HybridExtendedKalmanFilter(
        x_init, params=params, mode="hybrid", residual_model=residual_model
    )
    ekf_nom = HybridExtendedKalmanFilter(
        x_init, params=params, mode="nominal", residual_model=None
    )
    std_hyb[0] = np.sqrt(np.maximum(0.0, np.diag(ekf_hyb.P)))
    std_nom[0] = np.sqrt(np.maximum(0.0, np.diag(ekf_nom.P)))

    online_idx = [0, 3, 6]
    sig_online = np.array([0.12, 0.08, 0.015], dtype=float) * noise_scale
    atline_idx = [0, 2, 3, 4, 6]
    sig_atline = np.array([0.12, 0.06, 0.08, 0.05, 0.015], dtype=float) * noise_scale

    # Record initial measurement for operator parity
    y0 = np.maximum(0.0, true_states[0, atline_idx] + rng.normal(0.0, sig_atline))
    meas_states[0, atline_idx] = y0
    obs_mask_atline[0] = True

    for i in range(n_steps):
        u_i = _resolve_operating_inputs(controller_fn, t_grid[i], ekf_hyb.x)
        inputs[i] = [u_i.F, u_i.rpm, u_i.vvm]
        _, _, s_feed_dist = disturbance_profile(
            dist_mode, float(t_grid[i]), u_i.rpm, u_i.vvm, p_plant.S_feed
        )
        s_feed_plant = (
            s_feed_dist
            if (dist_mode or "none").lower() != "none" or u_i.S_feed is None
            else float(u_i.S_feed)
        )
        u_plant = OperatingInputs(
            F=u_i.F, rpm=u_i.rpm, vvm=u_i.vvm, S_feed=s_feed_plant
        )
        u_model = OperatingInputs(
            F=u_i.F, rpm=u_i.rpm, vvm=u_i.vvm, S_feed=None
        )

        rt_i = compute_rates(true_states[i], u_plant, p_plant, mode="ground_truth")
        rh_i = ekf_hyb.current_rates(u_model)
        rn_i = ekf_nom.current_rates(u_model)

        mu_true[i] = rt_i["mu"]
        mu_hyb[i] = rh_i["mu"]
        mu_nom[i] = rn_i["mu"]
        mu_mech[i] = rh_i["mu_mech"]
        eta_mu_hyb[i] = rh_i["eta_mu"]
        kd_true[i] = rt_i["k_d"]
        kd_hyb[i] = rh_i["k_d"]
        kla_true[i] = rt_i["k_La"]
        kla_hyb[i] = rh_i["k_La"]
        qAnet_true[i] = rt_i["q_A_net"]
        qAnet_hyb[i] = rh_i["q_A_net"]
        d_m_arr[i] = rh_i["d_m"]
        d_m_kla_arr[i] = rh_i["d_m_kla"]
        in_domain_arr[i] = rh_i["in_domain"]

        dt_i = float(t_grid[i + 1] - t_grid[i])
        true_states[i + 1] = integrate_step(
            t_grid[i],
            t_grid[i + 1],
            true_states[i],
            u_plant,
            p_plant,
            mode="ground_truth",
        )
        ekf_hyb.predict(u_model, dt_i)
        ekf_nom.predict(u_model, dt_i)

        is_atline = ((i + 1) % max(1, int(atline_interval_steps))) == 0
        obs_mask_atline[i + 1] = is_atline

        if is_atline:
            y_vec = np.maximum(0.0, true_states[i + 1, atline_idx] + rng.normal(0.0, sig_atline))
            y_vec[1] = max(0.0, y_vec[1])
            y_vec[3] = max(0.0, y_vec[3])
            meas_states[i + 1, atline_idx] = y_vec
            ekf_hyb.update(y_vec, atline_idx, sig_atline ** 2)
            ekf_nom.update(y_vec, atline_idx, sig_atline ** 2)
        else:
            y_vec = np.maximum(0.0, true_states[i + 1, online_idx] + rng.normal(0.0, sig_online))
            meas_states[i + 1, online_idx] = y_vec
            ekf_hyb.update(y_vec, online_idx, sig_online ** 2)
            ekf_nom.update(y_vec, online_idx, sig_online ** 2)

        est_hyb[i + 1] = ekf_hyb.x
        est_nom[i + 1] = ekf_nom.x
        std_hyb[i + 1] = np.sqrt(np.maximum(0.0, np.diag(ekf_hyb.P)))
        std_nom[i + 1] = np.sqrt(np.maximum(0.0, np.diag(ekf_nom.P)))
        nis_hyb[i + 1] = ekf_hyb.last_nis
        nis_nom[i + 1] = ekf_nom.last_nis

    u_end = _resolve_operating_inputs(controller_fn, t_grid[-1], ekf_hyb.x)
    inputs[-1] = [u_end.F, u_end.rpm, u_end.vvm]
    _, _, s_feed_end_dist = disturbance_profile(
        dist_mode, float(t_grid[-1]), u_end.rpm, u_end.vvm, p_plant.S_feed
    )
    s_feed_end_plant = (
        s_feed_end_dist
        if (dist_mode or "none").lower() != "none" or u_end.S_feed is None
        else float(u_end.S_feed)
    )
    u_end_plant = OperatingInputs(
        F=u_end.F, rpm=u_end.rpm, vvm=u_end.vvm, S_feed=s_feed_end_plant
    )
    u_end_model = OperatingInputs(
        F=u_end.F, rpm=u_end.rpm, vvm=u_end.vvm, S_feed=None
    )
    rt_e = compute_rates(true_states[-1], u_end_plant, p_plant, mode="ground_truth")
    rh_e = ekf_hyb.current_rates(u_end_model)
    rn_e = ekf_nom.current_rates(u_end_model)
    mu_true[-1] = rt_e["mu"]
    mu_hyb[-1] = rh_e["mu"]
    mu_nom[-1] = rn_e["mu"]
    mu_mech[-1] = rh_e["mu_mech"]
    eta_mu_hyb[-1] = rh_e["eta_mu"]
    kd_true[-1] = rt_e["k_d"]
    kd_hyb[-1] = rh_e["k_d"]
    kla_true[-1] = rt_e["k_La"]
    kla_hyb[-1] = rh_e["k_La"]
    qAnet_true[-1] = rt_e["q_A_net"]
    qAnet_hyb[-1] = rh_e["q_A_net"]
    d_m_arr[-1] = rh_e["d_m"]
    d_m_kla_arr[-1] = rh_e["d_m_kla"]
    in_domain_arr[-1] = rh_e["in_domain"]

    return {
        "t": t_grid,
        "true_states": true_states,
        "est_hyb": est_hyb,
        "est_nom": est_nom,
        "std_hyb": std_hyb,
        "std_nom": std_nom,
        "meas_states": meas_states,
        "inputs": inputs,
        "mu_true": mu_true,
        "mu_hyb": mu_hyb,
        "mu_nom": mu_nom,
        "mu_mech": mu_mech,
        "eta_mu_hyb": eta_mu_hyb,
        "kd_true": kd_true,
        "kd_hyb": kd_hyb,
        "kla_true": kla_true,
        "kla_hyb": kla_hyb,
        "qAnet_true": qAnet_true,
        "qAnet_hyb": qAnet_hyb,
        "d_m": d_m_arr,
        "d_m_kla": d_m_kla_arr,
        "in_domain": in_domain_arr,
        "nis_hyb": nis_hyb,
        "nis_nom": nis_nom,
        "obs_mask_atline": obs_mask_atline,
    }


class BioreactorDigitalTwin:
    """Portela Level 3-ready stateful Digital Twin (in-silico plant harness) with delayed-LIMS replay and ALCOA+ provenance."""

    def __init__(
        self,
        x0: np.ndarray,
        params: BioreactorParams,
        residual_model: ResidualCorrectorProtocol,
        calibration_metadata: Optional[Dict[str, Any]] = None,
        P0_diag: Optional[Sequence[float]] = None,
        Q_diag: Optional[Sequence[float]] = None,
        operator_id: str = "bioprocess_engineer",
        batch_id: str = "FB-BL21-001",
    ) -> None:
        self.params = params
        self.residual_model = residual_model
        self.calibration_metadata = dict(calibration_metadata or {})
        self.operator_id = operator_id
        self.batch_id = batch_id

        self.params_sha256 = _canonical_sha256(params.to_dict())
        model_sha = getattr(residual_model, "weights_sha256", "")
        if not model_sha and hasattr(residual_model, "weights"):
            model_sha = _canonical_sha256(np.asarray(residual_model.weights).tolist())
        self.model_sha256 = str(model_sha)

        self.ekf = HybridExtendedKalmanFilter(
            x0=x0,
            params=params,
            mode="hybrid",
            residual_model=residual_model,
            P0_diag=P0_diag,
            Q_diag=Q_diag,
        )
        self.t_now = 0.0
        self.last_u = OperatingInputs(F=0.0, rpm=450.0, vvm=1.2)
        self._events: List[Dict[str, Any]] = [
            {
                "kind": "init",
                "t_h": 0.0,
                "x_post": self.ekf.x.copy(),
                "P_post": self.ekf.P.copy(),
                "u": self.last_u,
            }
        ]
        self.audit_log: List[Dict[str, Any]] = []
        self._append_audit("INIT", 0.0, {"state_names": list(STATE_NAMES)})

    def _append_audit(self, event_type: str, process_t_h: float, detail: Dict[str, Any]) -> None:
        entry = {
            "seq": len(self.audit_log) + 1,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_id": self.batch_id,
            "operator_id": self.operator_id,
            "event_type": event_type,
            "process_t_h": round(float(process_t_h), 6),
            "params_sha256": self.params_sha256[:12],
            "model_sha256": self.model_sha256[:12],
            "detail": detail,
        }
        entry["entry_sha256"] = _canonical_sha256(entry)[:16]
        self.audit_log.append(entry)

    def _build_ekf_at(self, x_state: np.ndarray, P_mat: np.ndarray) -> HybridExtendedKalmanFilter:
        ekf = HybridExtendedKalmanFilter(
            x0=x_state,
            params=self.params,
            mode="hybrid",
            residual_model=self.residual_model,
            P0_diag=P_mat,
            Q_diag=self.ekf.Q_base,
        )
        return ekf

    def ingest_online_pat(
        self,
        t_h: float,
        y_Xv: float,
        y_CL: float,
        y_V: float,
        F_act: float,
        rpm_act: float,
        vvm_act: float,
        sig_online: Tuple[float, float, float] = (0.12, 0.08, 0.015),
        S_feed_act: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Propagate EKF to t_h under actual actuators and assimilate online PAT measurements."""
        t_val = float(t_h)
        if t_val < self.t_now - 1e-9:
            raise ValueError(f"Online PAT timestamp {t_val} precedes current twin clock {self.t_now}.")

        u_act = OperatingInputs(
            F=float(F_act),
            rpm=float(rpm_act),
            vvm=float(vvm_act),
            S_feed=S_feed_act,
        )
        dt = max(0.0, t_val - self.t_now)
        if dt > 0.0:
            self.ekf.predict(u_act, dt)
        obs_idx = [0, 3, 6]
        y_vec = np.maximum(0.0, np.array([y_Xv, y_CL, y_V], dtype=float))
        r_diag = np.asarray(sig_online, dtype=float) ** 2
        self.ekf.update(y_vec, obs_idx, r_diag)

        self.t_now = t_val
        self.last_u = u_act
        rates = self.ekf.current_rates(u_act)
        x_std = np.sqrt(np.maximum(0.0, np.diag(self.ekf.P)))

        self._events.append(
            {
                "kind": "online",
                "t_h": t_val,
                "u": u_act,
                "y_vec": y_vec.copy(),
                "obs_idx": list(obs_idx),
                "r_diag": r_diag.copy(),
                "x_post": self.ekf.x.copy(),
                "P_post": self.ekf.P.copy(),
            }
        )
        self._append_audit(
            "ONLINE_PAT",
            t_val,
            {
                "y_Xv": float(y_Xv),
                "y_CL": float(y_CL),
                "y_V": float(y_V),
                "F_act": float(F_act),
                "rpm_act": float(rpm_act),
                "vvm_act": float(vvm_act),
                "d_m": float(rates["d_m"]),
                "in_domain": bool(rates["in_domain"] > 0.5),
            },
        )
        return {
            "t_h": t_val,
            "x_hat": self.ekf.x.copy(),
            "x_std": x_std,
            "mu_hat": float(rates["mu"]),
            "k_La_hat": float(rates["k_La"]),
            "k_d_hat": float(rates["k_d"]),
            "q_A_net_hat": float(rates["q_A_net"]),
            "d_m": float(rates["d_m"]),
            "d_m_kla": float(rates["d_m_kla"]),
            "in_domain": bool(rates["in_domain"] > 0.5),
            "nis": float(self.ekf.last_nis),
        }

    def ingest_atline_assay(
        self,
        t_sample_h: float,
        y_S: float,
        y_A: float,
        y_P: Optional[float] = None,
        sig_atline: Tuple[float, float, float] = (0.06, 0.05, 0.10),
    ) -> Dict[str, Any]:
        """Assimilate a potentially delayed at-line LIMS sample at t_sample_h and replay forward."""
        t_s = float(t_sample_h)
        if t_s < -1e-9 or t_s > self.t_now + 1e-9:
            raise ValueError(
                f"Sample time {t_s} out of valid range [0.0, {self.t_now}]."
            )
        t_s = max(0.0, min(t_s, self.t_now))

        obs_idx = [2, 4] if y_P is None else [2, 4, 5]
        y_list = [y_S, y_A] if y_P is None else [y_S, y_A, y_P]
        sig_list = list(sig_atline[: len(obs_idx)])
        y_vec = np.maximum(0.0, np.asarray(y_list, dtype=float))
        r_diag = np.asarray(sig_list, dtype=float) ** 2

        if abs(t_s - self.t_now) <= 1e-9:
            self.ekf.update(y_vec, obs_idx, r_diag)
            self._events.append(
                {
                    "kind": "atline",
                    "t_h": t_s,
                    "u": self.last_u,
                    "y_vec": y_vec.copy(),
                    "obs_idx": list(obs_idx),
                    "r_diag": r_diag.copy(),
                    "x_post": self.ekf.x.copy(),
                    "P_post": self.ekf.P.copy(),
                }
            )
            replayed_steps = 0
        else:
            # Find latest checkpoint with timestamp <= t_s
            anchor_idx = 0
            for idx_ev, ev in enumerate(self._events):
                if ev["t_h"] <= t_s + 1e-9:
                    anchor_idx = idx_ev
                else:
                    break

            anchor = self._events[anchor_idx]
            ekf_replay = self._build_ekf_at(anchor["x_post"], anchor["P_post"])
            later_events = self._events[anchor_idx + 1 :]
            if abs(t_s - anchor["t_h"]) > 1e-9:
                u_bridge = later_events[0]["u"] if later_events else anchor["u"]
                ekf_replay.predict(u_bridge, t_s - anchor["t_h"])
            ekf_replay.update(y_vec, obs_idx, r_diag)

            lims_event = {
                "kind": "atline",
                "t_h": t_s,
                "u": anchor["u"],
                "y_vec": y_vec.copy(),
                "obs_idx": list(obs_idx),
                "r_diag": r_diag.copy(),
                "x_post": ekf_replay.x.copy(),
                "P_post": ekf_replay.P.copy(),
            }

            t_cursor = t_s
            updated_later: List[Dict[str, Any]] = []
            for ev in later_events:
                dt_ev = float(ev["t_h"] - t_cursor)
                if dt_ev > 1e-9:
                    ekf_replay.predict(ev["u"], dt_ev)
                ekf_replay.update(ev["y_vec"], ev["obs_idx"], ev["r_diag"])
                ev["x_post"] = ekf_replay.x.copy()
                ev["P_post"] = ekf_replay.P.copy()
                updated_later.append(ev)
                t_cursor = float(ev["t_h"])

            self._events = self._events[: anchor_idx + 1] + [lims_event] + updated_later
            self.ekf = ekf_replay
            replayed_steps = len(later_events)

        rates = self.ekf.current_rates(self.last_u)
        x_std = np.sqrt(np.maximum(0.0, np.diag(self.ekf.P)))
        self._append_audit(
            "ATLINE_LIMS_REPLAY" if replayed_steps > 0 else "ATLINE_LIMS",
            self.t_now,
            {
                "t_sample_h": t_s,
                "y_S": float(y_S),
                "y_A": float(y_A),
                "y_P": None if y_P is None else float(y_P),
                "replayed_steps": replayed_steps,
                "d_m": float(rates["d_m"]),
            },
        )
        return {
            "t_now": self.t_now,
            "t_sample_h": t_s,
            "replayed_steps": replayed_steps,
            "x_hat": self.ekf.x.copy(),
            "x_std": x_std,
            "mu_hat": float(rates["mu"]),
            "d_m": float(rates["d_m"]),
            "in_domain": bool(rates["in_domain"] > 0.5),
        }

    def compute_control_action(self, t_h: float, controller: ControlCallback) -> Dict[str, Any]:
        """Compute closed-loop supervisory feed setpoint strictly from current EKF state."""
        u_cmd = controller(float(t_h), self.ekf.x.copy())
        rates = self.ekf.current_rates(u_cmd)
        x_std = np.sqrt(np.maximum(0.0, np.diag(self.ekf.P)))
        self._append_audit(
            "CONTROL_ACTION",
            float(t_h),
            {
                "F_cmd": float(u_cmd.F),
                "rpm_cmd": float(u_cmd.rpm),
                "vvm_cmd": float(u_cmd.vvm),
                "mu_hat": float(rates["mu"]),
                "d_m": float(rates["d_m"]),
                "in_domain": bool(rates["in_domain"] > 0.5),
            },
        )
        return {
            "t_h": float(t_h),
            "F_cmd": float(u_cmd.F),
            "rpm_cmd": float(u_cmd.rpm),
            "vvm_cmd": float(u_cmd.vvm),
            "mu_hat": float(rates["mu"]),
            "k_La_hat": float(rates["k_La"]),
            "d_m": float(rates["d_m"]),
            "in_domain": bool(rates["in_domain"] > 0.5),
            "x_hat": self.ekf.x.copy(),
            "x_std": x_std,
        }

    @classmethod
    def from_simulation_run(
        cls,
        sim_result: Dict[str, np.ndarray],
        params: BioreactorParams,
        residual_model: ResidualCorrectorProtocol,
        calibration_metadata: Optional[Dict[str, Any]] = None,
        P0_diag: Optional[Sequence[float]] = None,
        Q_diag: Optional[Sequence[float]] = None,
        operator_id: str = "bioprocess_engineer",
        batch_id: str = "FB-BL21-001",
    ) -> "BioreactorDigitalTwin":
        """Reconstruct and replay a full Digital Twin ALCOA+ audit trail from a closed-loop run."""
        t_grid = np.asarray(sim_result["t"], dtype=float)
        meas_states = np.asarray(sim_result["meas_states"], dtype=float)
        inputs = np.asarray(sim_result["inputs"], dtype=float)
        est_hyb = np.asarray(sim_result["est_hyb"], dtype=float)
        obs_mask = np.asarray(
            sim_result.get("obs_mask_atline", np.zeros(len(t_grid), dtype=bool)),
            dtype=bool,
        )
        twin = cls(
            x0=est_hyb[0],
            params=params,
            residual_model=residual_model,
            calibration_metadata=calibration_metadata,
            P0_diag=P0_diag,
            Q_diag=Q_diag,
            operator_id=operator_id,
            batch_id=batch_id,
        )
        for k in range(1, len(t_grid)):
            u_prev = inputs[k - 1]
            m_k = meas_states[k]
            twin.ingest_online_pat(
                t_h=float(t_grid[k]),
                y_Xv=float(m_k[0]),
                y_CL=float(m_k[3]),
                y_V=float(m_k[6]),
                F_act=float(u_prev[0]),
                rpm_act=float(u_prev[1]),
                vvm_act=float(u_prev[2]),
            )
            if obs_mask[k] and np.isfinite(m_k[2]) and np.isfinite(m_k[4]):
                twin.ingest_atline_assay(
                    t_sample_h=float(t_grid[k]),
                    y_S=float(m_k[2]),
                    y_A=float(m_k[4]),
                    y_P=None,
                )
        return twin

    def to_audit_record(self, max_entries: Optional[int] = None) -> Dict[str, Any]:
        """Return ALCOA+ compliant provenance summary with integrity SHA-256 checksums."""
        x_std = np.sqrt(np.maximum(0.0, np.diag(self.ekf.P)))
        exported_log = (
            list(self.audit_log)
            if max_entries is None
            else list(self.audit_log[-max(1, int(max_entries)) :])
        )
        record: Dict[str, Any] = {
            "alcoa_standard": "ALCOA+ / EU GMP Annex 11 / 21 CFR Part 11 Style Trace",
            "batch_id": self.batch_id,
            "operator_id": self.operator_id,
            "t_now_h": float(self.t_now),
            "params_sha256": self.params_sha256,
            "model_sha256": self.model_sha256,
            "calibration_metadata": self.calibration_metadata,
            "current_state": {
                name: float(self.ekf.x[i]) for i, name in enumerate(STATE_NAMES)
            },
            "current_std": {
                name: float(x_std[i]) for i, name in enumerate(STATE_NAMES)
            },
            "audit_log_count": len(self.audit_log),
            "audit_log": exported_log,
        }
        record["record_sha256"] = _canonical_sha256(record)
        return record
