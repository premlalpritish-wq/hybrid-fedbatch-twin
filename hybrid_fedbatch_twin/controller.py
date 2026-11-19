"""Closed-loop & open-loop fed-batch feed controllers and benchmarking utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

from .estimator import HybridExtendedKalmanFilter, run_twin_estimation_loop
from .mechanistic import (
    BioreactorParams,
    OperatingInputs,
    ResidualCorrectorProtocol,
    compute_rates,
    default_initial_state,
    disturbance_profile,
    integrate_step,
    nominal_kla,
)

# Fixed physical normalization scales for dimensionless 7-state RMSE comparison
STATE_NORM_SCALES = np.array([15.0, 2.0, 2.0, 8.2, 2.5, 5.0, 3.0], dtype=float)


class ClosedLoopFeedController:
    """Hybrid-Twin supervisory feed controller maintaining mu_target while guarding DO, acetate, and OOD/residual gates."""

    def __init__(
        self,
        params: BioreactorParams,
        residual_model: ResidualCorrectorProtocol,
        mu_target: float = 0.28,
        S_set: float = 0.25,
        A_limit: float = 1.8,
        rpm: float = 450.0,
        vvm: float = 1.2,
        F_max: float = 0.30,
        disturbance_mode: str = "none",
        fallback_policy: Literal["safe_nominal", "hold"] = "safe_nominal",
        F_hold: Optional[float] = None,
    ) -> None:
        if fallback_policy not in ("safe_nominal", "hold"):
            raise ValueError(f"Unsupported fallback_policy: {fallback_policy!r}. Must be 'safe_nominal' or 'hold'.")
        self.params = params
        self.residual_model = residual_model
        self.mu_target = float(mu_target)
        self.S_set = float(S_set)
        self.A_limit = float(A_limit)
        self.rpm = float(rpm)
        self.vvm = float(vvm)
        self.F_max = float(F_max)
        self.disturbance_mode = disturbance_mode
        self.fallback_policy = str(fallback_policy)
        self.F_hold = None if F_hold is None else float(F_hold)

        self.last_control_mode: str = "hybrid_adaptive"
        self.last_fallback_active: bool = False
        self.last_undecided: bool = False
        self.last_escalation_event: Optional[Dict[str, Any]] = None
        self.escalation_events: List[Dict[str, Any]] = []
        self._last_safe_F: Optional[float] = None

    def _compute_safe_fallback_feed(
        self,
        X_v: float,
        S: float,
        C_L: float,
        A: float,
        V: float,
        rpm_now: float,
        vvm_now: float,
    ) -> float:
        """Compute strictly ML-free conservative fallback feed setpoint on OOD or residual breach."""
        if V >= self.params.V_max * 0.995 or C_L <= 1.6 or A >= self.A_limit:
            return 0.0
        if self.fallback_policy == "hold":
            hold_val = 0.0 if self.F_hold is None else self.F_hold
            return float(np.clip(hold_val, 0.0, 0.5 * self.F_max))
        if S >= 2.0 * self.S_set:
            return 0.0

        o2_av = C_L / (0.005 + C_L)
        k_la_nom = nominal_kla(rpm_now, vvm_now, V, self.params)
        otr_avail = k_la_nom * max(0.0, self.params.C_star - max(C_L, 2.5)) / 1000.0
        mu_otr_cap_nom = max(
            0.02,
            self.params.Y_XO * (otr_avail / max(0.1, X_v) - self.params.m_o * o2_av),
        )
        mu_safe = min(0.5 * self.mu_target, 0.5 * self.params.mu_max, mu_otr_cap_nom)
        s_safe = min(self.S_set, 0.15)
        s_av_safe = s_safe / (0.002 + s_safe)
        q_p_safe = (self.params.alpha * mu_safe + self.params.beta) * s_av_safe * o2_av
        q_s_safe = (
            mu_safe / self.params.Y_XS
            + self.params.m_s * s_av_safe
            + q_p_safe / self.params.Y_PS
        )
        demand_safe = q_s_safe * X_v * V
        f_nom_safe = float(np.clip(demand_safe / max(10.0, self.params.S_feed - S), 0.0, 0.5 * self.F_max))
        if self._last_safe_F is not None:
            return min(f_nom_safe, self._last_safe_F)
        return f_nom_safe

    def __call__(
        self,
        t: float,
        x_hat: np.ndarray,
        gate_status: Optional[Dict[str, Any]] = None,
    ) -> OperatingInputs:
        # Controller remains strictly blind to plant-side actuator/feed disturbances
        rpm_now, vvm_now = self.rpm, self.vvm

        X_v = max(0.05, float(x_hat[0]))
        S = max(0.0, float(x_hat[2]))
        C_L = max(0.0, float(x_hat[3]))
        A = max(0.0, float(x_hat[4]))
        V = max(0.2, float(x_hat[6]))

        u_probe = OperatingInputs(F=0.0, rpm=rpm_now, vvm=vvm_now, S_feed=None)
        rates_hat = compute_rates(
            x_hat, u_probe, self.params, mode="hybrid", residual_model=self.residual_model
        )

        dov_breach = bool(rates_hat.get("in_domain", 1.0) < 0.5)
        gate_undecided = (
            bool(gate_status.get("undecided", not gate_status.get("in_domain", True)))
            if gate_status is not None
            else False
        )
        residual_breach = (
            bool(gate_status.get("residual_breach", False))
            if gate_status is not None
            else False
        )
        dov_in_domain_gate = (
            bool(gate_status.get("dov_in_domain", not dov_breach))
            if gate_status is not None
            else (not dov_breach)
        )
        undecided = bool(dov_breach or gate_undecided or residual_breach or (not dov_in_domain_gate))

        if undecided:
            d_m_val = (
                float(gate_status.get("d_m", rates_hat.get("d_m", 0.0)))
                if gate_status is not None
                else float(rates_hat.get("d_m", 0.0))
            )
            nis_val = float(gate_status.get("nis", 0.0)) if gate_status is not None else 0.0
            spe_val = float(gate_status.get("spe", 0.0)) if gate_status is not None else 0.0
            if gate_status is not None and gate_status.get("gate_reason") not in (None, "", "nominal"):
                gate_reason = str(gate_status["gate_reason"])
            elif dov_breach and residual_breach:
                gate_reason = "dual_ood_and_residual_breach"
            elif dov_breach:
                gate_reason = "mahalanobis_ood"
            elif residual_breach:
                gate_reason = "residual_breach"
            else:
                gate_reason = "undecided_gate"

            F_safe = self._compute_safe_fallback_feed(X_v, S, C_L, A, V, rpm_now, vvm_now)
            self.last_fallback_active = True
            self.last_undecided = True
            self.last_control_mode = f"safe_fallback_{self.fallback_policy}"
            esc_event: Dict[str, Any] = {
                "event_type": "CONTROL_FALLBACK_ESCALATION",
                "t_h": float(t),
                "policy": self.fallback_policy,
                "gate_reason": gate_reason,
                "d_m": d_m_val,
                "nis": nis_val,
                "spe": spe_val,
                "F_safe": float(F_safe),
                "requires_operator_review": True,
            }
            self.last_escalation_event = esc_event
            self.escalation_events.append(esc_event)
            return OperatingInputs(F=float(F_safe), rpm=rpm_now, vvm=vvm_now, S_feed=None)

        if V >= self.params.V_max * 0.995:
            self.last_fallback_active = False
            self.last_undecided = False
            self.last_control_mode = "volume_limit_stop"
            self.last_escalation_event = None
            return OperatingInputs(F=0.0, rpm=rpm_now, vvm=vvm_now, S_feed=None)

        mu_eff = float(rates_hat["mu"])
        eta_mu_hat = float(rates_hat.get("eta_mu", 1.0))

        inhib_o2 = C_L / (self.params.K_O2 + C_L)
        mu_cap = max(mu_eff + 0.04, self.params.mu_max * inhib_o2 * eta_mu_hat)

        ace_throttle = max(
            0.35, 1.0 - max(0.0, A - 0.55 * self.A_limit) / max(0.25, self.A_limit)
        )
        o2_guard = float(np.clip((C_L - 1.0) / 1.8, 0.35, 1.0))
        o2_av = C_L / (0.005 + C_L)
        k_la_hat = float(rates_hat["k_La"])
        otr_avail = k_la_hat * max(0.0, self.params.C_star - max(C_L, 2.2)) / 1000.0
        mu_otr_cap = max(
            0.03,
            self.params.Y_XO * (otr_avail / max(0.1, X_v) - self.params.m_o * o2_av),
        )
        mu_sp = min(self.mu_target * ace_throttle * o2_guard, mu_cap, mu_otr_cap)

        s_sp = max(
            0.08,
            self.S_set * max(0.35, 1.0 - max(0.0, A - 0.7 * self.A_limit) / max(0.5, self.A_limit)),
        )
        s_av_sp = s_sp / (0.002 + s_sp)
        q_P_sp = (self.params.alpha * mu_sp + self.params.beta) * s_av_sp * o2_av
        q_S_sp = (
            mu_sp / self.params.Y_XS
            + self.params.m_s * s_av_sp
            + q_P_sp / self.params.Y_PS
            + float(rates_hat["q_A_prod"]) / self.params.Y_AS
        )

        gain_s = 1.6 if S <= s_sp else 2.4
        demand_g_h = q_S_sp * X_v * V + gain_s * (s_sp - S) * V
        F_cmd = demand_g_h / max(10.0, self.params.S_feed - S)
        F_cmd = float(np.clip(F_cmd, 0.0, self.F_max))
        self._last_safe_F = F_cmd
        self.last_fallback_active = False
        self.last_undecided = False
        self.last_control_mode = "hybrid_adaptive"
        self.last_escalation_event = None
        return OperatingInputs(F=F_cmd, rpm=rpm_now, vvm=vvm_now, S_feed=None)


class NominalMechanisticController:
    """Baseline mechanistic-only closed-loop controller (blind to residual kinetics & viscosity)."""

    def __init__(
        self,
        params: BioreactorParams,
        mu_target: float = 0.28,
        S_set: float = 0.25,
        rpm: float = 450.0,
        vvm: float = 1.2,
        F_max: float = 0.30,
        disturbance_mode: str = "none",
    ) -> None:
        self.params = params
        self.mu_target = float(mu_target)
        self.S_set = float(S_set)
        self.rpm = float(rpm)
        self.vvm = float(vvm)
        self.F_max = float(F_max)
        self.disturbance_mode = disturbance_mode

    def __call__(self, t: float, x_hat: np.ndarray) -> OperatingInputs:
        del t
        rpm_now, vvm_now = self.rpm, self.vvm

        X_v = max(0.05, float(x_hat[0]))
        S = max(0.0, float(x_hat[2]))
        C_L = max(0.0, float(x_hat[3]))
        V = max(0.2, float(x_hat[6]))

        if V >= self.params.V_max * 0.995:
            return OperatingInputs(F=0.0, rpm=rpm_now, vvm=vvm_now, S_feed=None)

        s_av = self.S_set / (0.002 + self.S_set)
        o2_av = C_L / (0.005 + C_L)
        k_la_nom = nominal_kla(rpm_now, vvm_now, V, self.params)
        otr_avail = k_la_nom * max(0.0, self.params.C_star - max(C_L, 2.2)) / 1000.0
        mu_otr_cap = max(
            0.03,
            self.params.Y_XO * (otr_avail / max(0.1, X_v) - self.params.m_o * o2_av),
        )
        mu_sp = min(self.mu_target, self.params.mu_max * 0.95, mu_otr_cap)
        q_P_sp = (self.params.alpha * mu_sp + self.params.beta) * s_av * o2_av
        q_S_sp = (
            mu_sp / self.params.Y_XS
            + self.params.m_s * s_av
            + q_P_sp / self.params.Y_PS
        )
        demand_g_h = q_S_sp * X_v * V + 1.5 * (self.S_set - S) * V
        F_cmd = demand_g_h / max(10.0, self.params.S_feed - S)
        F_cmd = float(np.clip(F_cmd, 0.0, self.F_max))
        return OperatingInputs(F=F_cmd, rpm=rpm_now, vvm=vvm_now, S_feed=None)


class OpenLoopExponentialController:
    """Classical pre-programmed exponential feed profile (no feedback from sensors)."""

    def __init__(
        self,
        params: BioreactorParams,
        x0: np.ndarray,
        mu_target: float = 0.28,
        rpm: float = 450.0,
        vvm: float = 1.2,
        F_max: float = 0.30,
        t_batch_delay_h: float = 3.0,
        disturbance_mode: str = "none",
    ) -> None:
        self.params = params
        self.X0 = max(0.1, float(x0[0]))
        self.V0 = max(0.2, float(x0[6]))
        self.mu_target = float(mu_target)
        self.rpm = float(rpm)
        self.vvm = float(vvm)
        self.F_max = float(F_max)
        self.t_batch_delay_h = float(t_batch_delay_h)
        self.disturbance_mode = disturbance_mode

    def __call__(self, t: float, x_hat: np.ndarray) -> OperatingInputs:
        del x_hat
        rpm_now, vvm_now = self.rpm, self.vvm
        if t < self.t_batch_delay_h:
            return OperatingInputs(F=0.0, rpm=rpm_now, vvm=vvm_now, S_feed=None)
        dt_feed = float(t - self.t_batch_delay_h)
        x_expected = self.X0 * self.V0 * np.exp(self.mu_target * min(t, 12.0))
        q_s_nom = self.mu_target / self.params.Y_XS + self.params.m_s
        F_cmd = (q_s_nom * x_expected / max(10.0, self.params.S_feed)) * np.exp(
            0.05 * min(dt_feed, 8.0)
        )
        return OperatingInputs(
            F=float(np.clip(F_cmd, 0.0, self.F_max)),
            rpm=rpm_now,
            vvm=vvm_now,
            S_feed=None,
        )


class ManualFeedController:
    """Manual constant feed setpoint controller with interactive rpm & vvm sliders."""

    def __init__(
        self,
        params: BioreactorParams,
        F_set: float = 0.05,
        rpm: float = 450.0,
        vvm: float = 1.2,
        disturbance_mode: str = "none",
    ) -> None:
        self.params = params
        self.F_set = float(F_set)
        self.rpm = float(rpm)
        self.vvm = float(vvm)
        self.disturbance_mode = disturbance_mode

    def __call__(self, t: float, x_hat: np.ndarray) -> OperatingInputs:
        del t
        rpm_now, vvm_now = self.rpm, self.vvm
        V = float(x_hat[6])
        F_cmd = 0.0 if V >= self.params.V_max * 0.995 else max(0.0, self.F_set)
        return OperatingInputs(F=F_cmd, rpm=rpm_now, vvm=vvm_now, S_feed=None)


def _run_nominal_controller_loop(
    t_span: Tuple[float, float],
    x0: np.ndarray,
    nom_ctrl: NominalMechanisticController,
    params: BioreactorParams,
    residual_model: ResidualCorrectorProtocol,
    dt_step: float = 0.1,
    atline_interval_steps: int = 10,
    seed: int = 123,
    plant_params: Optional[BioreactorParams] = None,
) -> Dict[str, np.ndarray]:
    """Run NominalMechanisticController closed-loop strictly on its own nominal EKF."""
    rng = np.random.default_rng(seed)
    p_plant = plant_params or params

    t0, t1 = float(t_span[0]), float(t_span[1])
    n_steps = max(1, int(round((t1 - t0) / dt_step)))
    t_grid = np.linspace(t0, t1, n_steps + 1)

    true_states = np.zeros((n_steps + 1, 7), dtype=float)
    est_hyb = np.zeros((n_steps + 1, 7), dtype=float)
    est_nom = np.zeros((n_steps + 1, 7), dtype=float)
    std_hyb = np.zeros((n_steps + 1, 7), dtype=float)
    inputs = np.zeros((n_steps + 1, 3), dtype=float)
    mu_true = np.zeros(n_steps + 1, dtype=float)
    mu_hyb = np.zeros(n_steps + 1, dtype=float)
    mu_nom = np.zeros(n_steps + 1, dtype=float)
    d_m_arr = np.zeros(n_steps + 1, dtype=float)
    in_domain_arr = np.ones(n_steps + 1, dtype=float)

    x_init = np.maximum(0.0, np.asarray(x0, dtype=float))
    true_states[0] = x_init
    est_hyb[0] = x_init
    est_nom[0] = x_init

    ekf_hyb = HybridExtendedKalmanFilter(
        x_init, params=params, mode="hybrid", residual_model=residual_model
    )
    ekf_nom = HybridExtendedKalmanFilter(
        x_init, params=params, mode="nominal", residual_model=None
    )
    std_hyb[0] = np.sqrt(np.maximum(0.0, np.diag(ekf_hyb.P)))

    online_idx = [0, 3, 6]
    sig_online = np.array([0.12, 0.08, 0.015], dtype=float)
    atline_idx = [0, 2, 3, 4, 6]
    sig_atline = np.array([0.12, 0.06, 0.08, 0.05, 0.015], dtype=float)

    dist_mode = str(getattr(nom_ctrl, "disturbance_mode", "none"))
    for i in range(n_steps):
        u_i = nom_ctrl(t_grid[i], ekf_nom.x)
        inputs[i] = [u_i.F, u_i.rpm, u_i.vvm]
        rpm_plant, vvm_plant, s_feed_plant = disturbance_profile(
            dist_mode, float(t_grid[i]), u_i.rpm, u_i.vvm, p_plant.S_feed, V_vessel=p_plant.V_vessel
        )
        u_plant = OperatingInputs(
            F=u_i.F, rpm=rpm_plant, vvm=vvm_plant, S_feed=s_feed_plant
        )
        u_model = OperatingInputs(
            F=u_i.F, rpm=u_i.rpm, vvm=u_i.vvm, S_feed=None
        )
        rt = compute_rates(true_states[i], u_plant, p_plant, mode="ground_truth")
        rh = ekf_hyb.current_rates(u_model)
        rn = ekf_nom.current_rates(u_model)
        mu_true[i] = rt["mu"]
        mu_hyb[i] = rh["mu"]
        mu_nom[i] = rn["mu"]
        d_m_arr[i] = rh["d_m"]
        in_domain_arr[i] = rh["in_domain"]

        dt_i = float(t_grid[i + 1] - t_grid[i])
        true_states[i + 1] = integrate_step(
            t_grid[i], t_grid[i + 1], true_states[i], u_plant, p_plant, mode="ground_truth"
        )
        ekf_hyb.predict(u_model, dt_i)
        ekf_nom.predict(u_model, dt_i)

        if ((i + 1) % max(1, int(atline_interval_steps))) == 0:
            y_vec = np.maximum(0.0, true_states[i + 1, atline_idx] + rng.normal(0.0, sig_atline))
            ekf_hyb.update(y_vec, atline_idx, sig_atline ** 2)
            ekf_nom.update(y_vec, atline_idx, sig_atline ** 2)
        else:
            y_vec = np.maximum(0.0, true_states[i + 1, online_idx] + rng.normal(0.0, sig_online))
            ekf_hyb.update(y_vec, online_idx, sig_online ** 2)
            ekf_nom.update(y_vec, online_idx, sig_online ** 2)

        est_hyb[i + 1] = ekf_hyb.x
        est_nom[i + 1] = ekf_nom.x
        std_hyb[i + 1] = np.sqrt(np.maximum(0.0, np.diag(ekf_hyb.P)))

    u_e = nom_ctrl(t_grid[-1], ekf_nom.x)
    inputs[-1] = [u_e.F, u_e.rpm, u_e.vvm]
    rpm_e, vvm_e, s_feed_e = disturbance_profile(
        dist_mode, float(t_grid[-1]), u_e.rpm, u_e.vvm, p_plant.S_feed, V_vessel=p_plant.V_vessel
    )
    u_e_plant = OperatingInputs(F=u_e.F, rpm=rpm_e, vvm=vvm_e, S_feed=s_feed_e)
    u_e_model = OperatingInputs(F=u_e.F, rpm=u_e.rpm, vvm=u_e.vvm, S_feed=None)
    rt_e = compute_rates(true_states[-1], u_e_plant, p_plant, mode="ground_truth")
    rh_e = ekf_hyb.current_rates(u_e_model)
    rn_e = ekf_nom.current_rates(u_e_model)
    mu_true[-1] = rt_e["mu"]
    mu_hyb[-1] = rh_e["mu"]
    mu_nom[-1] = rn_e["mu"]
    d_m_arr[-1] = rh_e["d_m"]
    in_domain_arr[-1] = rh_e["in_domain"]

    return {
        "t": t_grid,
        "true_states": true_states,
        "est_hyb": est_hyb,
        "est_nom": est_nom,
        "std_hyb": std_hyb,
        "inputs": inputs,
        "mu_true": mu_true,
        "mu_hyb": mu_hyb,
        "mu_nom": mu_nom,
        "d_m": d_m_arr,
        "in_domain": in_domain_arr,
    }


def summarize_control_run(
    sim_result: Dict[str, np.ndarray],
    mu_target: float,
    params: Optional[BioreactorParams] = None,
    C_star: Optional[float] = None,
) -> Dict[str, float]:
    """Compute both plant benchmark metrics and oracle-free Operator Twin KPIs."""
    c_star = float(C_star if C_star is not None else (params.C_star if params is not None else 8.2))
    t = sim_result["t"]
    true_x = sim_result["true_states"]
    est_hyb = sim_result["est_hyb"]
    est_nom = sim_result["est_nom"]
    std_hyb = sim_result.get("std_hyb", np.zeros_like(est_hyb))
    mu_true = sim_result["mu_true"]
    mu_hyb = sim_result["mu_hyb"]

    fed_mask = t >= 2.0
    if not np.any(fed_mask):
        fed_mask = np.ones_like(t, dtype=bool)

    err_hyb = est_hyb - true_x
    err_nom = est_nom - true_x
    norm_hyb = float(np.sqrt(np.mean((err_hyb / STATE_NORM_SCALES[None, :]) ** 2)))
    norm_nom = float(np.sqrt(np.mean((err_nom / STATE_NORM_SCALES[None, :]) ** 2)))

    d_m = sim_result.get("d_m", np.zeros_like(t))
    in_domain = sim_result.get("in_domain", np.ones_like(t))
    undecided = sim_result.get("undecided", 1.0 - np.asarray(in_domain, dtype=float))
    fallback_active = sim_result.get("fallback_active", np.zeros_like(t, dtype=bool))
    esc_events = sim_result.get("escalation_events", [])
    nis_hyb = sim_result.get("nis_hyb", np.zeros_like(t))
    spe_hyb = sim_result.get("spe_hyb", np.zeros_like(t))

    return {
        # Ground-truth validation benchmark KPIs
        "final_Xv_g_L": float(true_x[-1, 0]),
        "final_Xd_g_L": float(true_x[-1, 1]),
        "final_P_g_L": float(true_x[-1, 5]),
        "peak_acetate_g_L": float(np.max(true_x[:, 4])),
        "min_substrate_g_L": float(np.min(true_x[:, 2])),
        "min_DO_pct": float(100.0 * np.min(true_x[:, 3]) / max(1e-6, c_star)),
        "mu_rmse": float(np.sqrt(np.mean((mu_true[fed_mask] - mu_target) ** 2))),
        "hybrid_norm_rmse": norm_hyb,
        "nominal_norm_rmse": norm_nom,
        "hybrid_Xv_rmse": float(np.sqrt(np.mean(err_hyb[:, 0] ** 2))),
        "nominal_Xv_rmse": float(np.sqrt(np.mean(err_nom[:, 0] ** 2))),
        "hybrid_Xd_rmse": float(np.sqrt(np.mean(err_hyb[:, 1] ** 2))),
        "nominal_Xd_rmse": float(np.sqrt(np.mean(err_nom[:, 1] ** 2))),
        "hybrid_S_rmse": float(np.sqrt(np.mean(err_hyb[:, 2] ** 2))),
        "nominal_S_rmse": float(np.sqrt(np.mean(err_nom[:, 2] ** 2))),
        "hybrid_A_rmse": float(np.sqrt(np.mean(err_hyb[:, 4] ** 2))),
        "nominal_A_rmse": float(np.sqrt(np.mean(err_nom[:, 4] ** 2))),
        "hybrid_P_rmse": float(np.sqrt(np.mean(err_hyb[:, 5] ** 2))),
        "nominal_P_rmse": float(np.sqrt(np.mean(err_nom[:, 5] ** 2))),
        # Oracle-free Operator Twin KPIs (strictly derived from est_hyb + UQ)
        "est_final_Xv_g_L": float(est_hyb[-1, 0]),
        "est_final_Xv_std": float(std_hyb[-1, 0]),
        "est_final_Xd_g_L": float(est_hyb[-1, 1]),
        "est_final_Xd_std": float(std_hyb[-1, 1]),
        "est_final_P_g_L": float(est_hyb[-1, 5]),
        "est_final_P_std": float(std_hyb[-1, 5]),
        "est_peak_acetate_g_L": float(np.max(est_hyb[:, 4])),
        "est_min_substrate_g_L": float(np.min(est_hyb[:, 2])),
        "est_min_DO_pct": float(100.0 * np.min(est_hyb[:, 3]) / max(1e-6, c_star)),
        "est_mu_rmse": float(np.sqrt(np.mean((mu_hyb[fed_mask] - mu_target) ** 2))),
        "max_d_m": float(np.max(d_m)),
        "ood_fraction": float(np.mean(in_domain < 0.5)),
        "undecided_fraction": float(np.mean(np.asarray(undecided, dtype=float) > 0.5)),
        "fallback_fraction": float(np.mean(np.asarray(fallback_active, dtype=bool))),
        "escalation_count": float(len(esc_events)),
        "max_nis_hyb": float(np.max(nis_hyb)),
        "max_spe_hyb": float(np.max(spe_hyb)),
    }


def benchmark_controllers(
    params: BioreactorParams,
    residual_model: ResidualCorrectorProtocol,
    duration_h: float = 16.0,
    mu_target: float = 0.28,
    rpm: float = 450.0,
    vvm: float = 1.2,
    F_max: float = 0.30,
    plant_params: Optional[BioreactorParams] = None,
) -> Dict[str, Dict[str, Any]]:
    """Benchmark hybrid closed-loop, nominal closed-loop, and open-loop exponential control."""
    p_plant = plant_params or params
    x0 = default_initial_state(p_plant)

    cl_hyb = ClosedLoopFeedController(
        params=params,
        residual_model=residual_model,
        mu_target=mu_target,
        rpm=rpm,
        vvm=vvm,
        F_max=F_max,
    )
    res_hyb = run_twin_estimation_loop(
        (0.0, duration_h),
        x0,
        cl_hyb,
        params=params,
        residual_model=residual_model,
        seed=123,
        plant_params=p_plant,
    )

    cl_nom = NominalMechanisticController(
        params=params,
        mu_target=mu_target,
        rpm=rpm,
        vvm=vvm,
        F_max=F_max,
    )
    res_nom = _run_nominal_controller_loop(
        (0.0, duration_h),
        x0,
        cl_nom,
        params=params,
        residual_model=residual_model,
        seed=123,
        plant_params=p_plant,
    )

    ol_exp = OpenLoopExponentialController(
        params=params,
        x0=x0,
        mu_target=mu_target,
        rpm=rpm,
        vvm=vvm,
        F_max=F_max,
    )
    res_ol = run_twin_estimation_loop(
        (0.0, duration_h),
        x0,
        ol_exp,
        params=params,
        residual_model=residual_model,
        seed=123,
        plant_params=p_plant,
    )

    return {
        "hybrid_closed_loop": {
            "simulation": res_hyb,
            "summary": summarize_control_run(res_hyb, mu_target, params=params),
        },
        "nominal_closed_loop": {
            "simulation": res_nom,
            "summary": summarize_control_run(res_nom, mu_target, params=params),
        },
        "open_loop_exp": {
            "simulation": res_ol,
            "summary": summarize_control_run(res_ol, mu_target, params=params),
        },
    }
