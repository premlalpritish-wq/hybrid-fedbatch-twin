"""Mechanistic Monod/Luedeking-Piret model, intensive hydrodynamics, and fed-batch plant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Literal, Optional, Protocol, Tuple
import numpy as np
from scipy.integrate import solve_ivp


STATE_NAMES = ("X_v", "X_d", "S", "C_L", "A", "P", "V")


@dataclass
class BioreactorParams:
    """Biokinetic, stoichiometric, mass-transfer, and vessel geometry parameters."""

    mu_max: float = 0.45        # Maximum specific growth rate (1/h)
    K_S: float = 0.15           # Glucose Monod half-saturation constant (g/L)
    K_O2: float = 0.25          # Dissolved oxygen Monod constant (mg/L)
    Y_XS: float = 0.48          # Biomass-on-glucose yield (g X / g S)
    Y_XO: float = 1.15          # Biomass-on-oxygen yield (g X / g O2)
    Y_PS: float = 0.85          # Product-on-glucose yield (g P / g S)
    Y_AS: float = 0.667         # Overflow acetate-on-glucose yield (2 M_A / M_S, g A / g S)
    Y_AO: float = 0.938         # Acetate-on-oxygen yield (M_A / 2 M_O2, g A / g O2)
    m_s: float = 0.025          # Glucose maintenance coefficient (g S / (g X * h))
    m_o: float = 0.018          # Oxygen maintenance coefficient (g O2 / (g X * h))
    alpha: float = 0.35         # Luedeking-Piret growth-associated product constant (g P / g X)
    beta: float = 0.025         # Luedeking-Piret non-growth-associated constant (g P / (g X * h))
    k_d0: float = 0.005         # Nominal basal cell death rate (1/h)
    k_La0: float = 650.0        # Reference k_L a at 400 RPM, 1.0 vvm in 5 L vessel at V_0 (1/h)
    C_star: float = 8.2         # Oxygen saturation concentration in broth (mg/L)
    S_feed: float = 300.0       # Feed glucose concentration (g/L)
    V_0: float = 2.0            # Reference initial working volume (L)
    V_max: float = 5.0          # Maximum vessel working volume (L)
    V_vessel: float = 5.0       # Rated vessel working volume for geometric scaling (L)
    H_to_T_nominal: float = 2.0 # Nominal liquid height-to-tank-diameter aspect ratio
    D_to_T: float = 0.35        # Impeller-diameter-to-tank-diameter ratio
    N_p: float = 10.0           # Dual-Rushton impeller turbulent power number
    broth_density: float = 1020.0  # Broth density (kg/m3)
    v_tip_crit_ref: float = 1.3478734461592616  # Critical shear tip speed (500 RPM @ 5 L, m/s)
    v_tip_span_ref: float = 0.8087240676955569  # Characteristic shear tip-speed span (300 RPM @ 5 L, m/s)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class OperatingInputs:
    """Manipulated and operating variables at time t."""

    F: float       # Feed flow rate (L/h)
    rpm: float     # Impeller agitation speed (RPM)
    vvm: float     # Aeration rate (vessel volumes per minute)
    S_feed: Optional[float] = None  # Optional override for feed glucose disturbance (g/L)


class ResidualCorrectorProtocol(Protocol):
    """Protocol implemented by PhysicsInformedNeuralODE."""

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
        """Return (delta_kd, eta_kla, q_A_net)."""
        ...


ControlCallback = Callable[[float, np.ndarray], OperatingInputs]


def default_initial_state(params: Optional[BioreactorParams] = None) -> np.ndarray:
    """Return standard initial state [X_v, X_d, S, C_L, A, P, V]."""
    p = params or BioreactorParams()
    return np.array([1.5, 0.02, 1.2, p.C_star * 0.95, 0.05, 0.0, p.V_0], dtype=float)


def vessel_geometry(params: BioreactorParams) -> Dict[str, float]:
    """Compute geometrically similar cylindrical stirred-tank dimensions from V_vessel."""
    v_ves_m3 = 1e-3 * max(1e-4, float(params.V_vessel))
    tank_d = (4.0 * v_ves_m3 / (np.pi * max(0.5, float(params.H_to_T_nominal)))) ** (1.0 / 3.0)
    impeller_d = float(params.D_to_T) * tank_d
    area_tank = 0.25 * np.pi * (tank_d ** 2)
    return {
        "V_ves_m3": float(v_ves_m3),
        "T_m": float(tank_d),
        "D_i_m": float(impeller_d),
        "A_tank_m2": float(area_tank),
    }


def compute_hydrodynamics(
    rpm: float,
    vvm: float,
    V: float,
    params: BioreactorParams,
) -> Dict[str, float]:
    """Compute intensive gassed power input P_g/V (W/m3), superficial gas velocity v_s, and v_tip."""
    geom = vessel_geometry(params)
    n_rps = max(50.0, float(rpm)) / 60.0
    v_m3 = 1e-3 * max(0.05, float(V))
    q_g = max(0.01, float(vvm)) * v_m3 / 60.0
    d_i = geom["D_i_m"]
    n_ae = q_g / (n_rps * (d_i ** 3) + 1e-12)

    p0 = float(params.N_p) * float(params.broth_density) * (n_rps ** 3) * (d_i ** 5)
    pg = p0 / (1.0 + 3.2 * (n_ae ** 0.7))
    pg_v = pg / v_m3
    v_s = q_g / geom["A_tank_m2"]
    v_tip = np.pi * n_rps * d_i
    return {
        "N_rps": float(n_rps),
        "Q_g_m3_s": float(q_g),
        "N_Ae": float(n_ae),
        "P_g_W": float(pg),
        "Pg_V": float(pg_v),
        "v_s": float(v_s),
        "v_tip": float(v_tip),
    }


_REF_PARAMS_5L = BioreactorParams()
_REF_HYDRO_5L = compute_hydrodynamics(400.0, 1.0, _REF_PARAMS_5L.V_0, _REF_PARAMS_5L)


def nominal_kla(rpm: float, vvm: float, V: float, params: BioreactorParams) -> float:
    """Scale-invariant Van't Riet correlation k_L a ~ (P_g / V)^0.40 * v_s^0.50."""
    hyd = compute_hydrodynamics(rpm, vvm, V, params)
    pg_ratio = max(1e-6, hyd["Pg_V"] / _REF_HYDRO_5L["Pg_V"])
    vs_ratio = max(1e-6, hyd["v_s"] / _REF_HYDRO_5L["v_s"])
    return float(params.k_La0 * (pg_ratio ** 0.40) * (vs_ratio ** 0.50))


def true_kla_viscosity_factor(X_total: float) -> float:
    """Non-Newtonian broth viscosity damping factor on k_L a from total solids X_v + X_d."""
    x_pos = max(0.0, float(X_total))
    return float(1.0 / (1.0 + 0.85 * (x_pos / 18.0) ** 1.6))


def true_dynamic_death_rate(
    S: float,
    C_L: float,
    A: float,
    rpm: float = 450.0,
    params: Optional[BioreactorParams] = None,
    V: Optional[float] = None,
    v_tip: Optional[float] = None,
) -> float:
    """True cell death rate (1/h) from acetate toxicity, hypoxia, shear tip speed, and starvation."""
    p = params or BioreactorParams()
    s_pos = max(0.0, float(S))
    cl_pos = max(0.0, float(C_L))
    a_pos = max(0.0, float(A))

    if v_tip is None:
        v_ref = p.V_0 if V is None else float(V)
        v_tip_val = compute_hydrodynamics(rpm, 1.0, v_ref, p)["v_tip"]
    else:
        v_tip_val = float(v_tip)

    shear_excess = max(0.0, (v_tip_val - p.v_tip_crit_ref) / max(1e-6, p.v_tip_span_ref))
    kd_acetate = 0.045 * (a_pos ** 2) / (3.5 + a_pos ** 2)
    kd_hypoxia = 0.055 * np.exp(-cl_pos / 0.40)
    kd_shear = 0.030 * (shear_excess ** 2)
    kd_starvation = 0.020 * np.exp(-s_pos / 0.020)
    return float(p.k_d0 + kd_acetate + kd_hypoxia + kd_shear + kd_starvation)


def true_acetate_rates(mu: float, S: float, C_L: float, A: float) -> Tuple[float, float]:
    """Return (q_A_prod, q_A_cons) in g A / (g X * h) with smooth starvation/hypoxia gates."""
    s_pos = max(0.0, float(S))
    cl_pos = max(0.0, float(C_L))
    a_pos = max(0.0, float(A))

    s_avail = s_pos / (0.002 + s_pos)
    o2_avail = cl_pos / (0.005 + cl_pos)
    a_avail = a_pos / (0.25 + a_pos)

    q_prod = (
        0.38 * max(0.0, float(mu) - 0.30)
        + 0.055 * (s_pos / (0.5 + s_pos)) * (0.6 / (0.6 + cl_pos))
    ) * s_avail
    q_cons = (
        0.040
        * (a_pos / (0.5 + a_pos))
        * (0.25 / (0.25 + s_pos))
        * (cl_pos / (0.5 + cl_pos))
        * a_avail
        * o2_avail
    )
    return float(q_prod), float(q_cons)


def compute_rates(
    x: np.ndarray,
    u: OperatingInputs,
    params: BioreactorParams,
    mode: Literal["ground_truth", "nominal", "hybrid"] = "ground_truth",
    residual_model: Optional[ResidualCorrectorProtocol] = None,
    include_uq: bool = True,
) -> Dict[str, float]:
    """Compute growth rate, death rate, k_L a, and coupled byproduct/product fluxes."""
    X_v = max(0.0, float(x[0]))
    X_d = max(0.0, float(x[1]))
    S = max(0.0, float(x[2]))
    C_L = max(0.0, float(x[3]))
    A = max(0.0, float(x[4]))
    V = max(0.2, float(x[6]))

    hyd = compute_hydrodynamics(u.rpm, u.vvm, V, params)
    kla_nom = nominal_kla(u.rpm, u.vvm, V, params)
    s_avail = S / (0.002 + S)
    o2_avail = C_L / (0.005 + C_L)
    a_avail = A / (0.25 + A)

    # Identical pure Monod backbone shared by nominal and hybrid modes
    mu_mech = (
        params.mu_max
        * (S / (params.K_S + S))
        * (C_L / (params.K_O2 + C_L))
    )

    std_kd = 0.0
    std_kla = 0.0
    std_qp = 0.0
    std_qc = 0.0
    std_mu = 0.0
    d_m = 0.0
    d_m_kla = 0.0
    in_domain = 1.0
    eta_mu = 1.0

    if mode == "nominal":
        mu = mu_mech
        k_d = params.k_d0
        k_La = kla_nom
        q_A_prod = 0.0
        q_A_cons = 0.0
    elif mode == "ground_truth":
        mu = (
            params.mu_max
            * (S / (params.K_S + S + (S ** 2) / 60.0))
            * (C_L / (params.K_O2 + C_L))
            * (8.0 / (8.0 + A))
        )
        eta_mu = mu / max(1e-12, mu_mech) if mu_mech > 1e-12 else 1.0
        k_d = true_dynamic_death_rate(S, C_L, A, u.rpm, params, V=V, v_tip=hyd["v_tip"])
        k_La = kla_nom * true_kla_viscosity_factor(X_v + X_d)
        q_A_prod, q_A_cons = true_acetate_rates(mu, S, C_L, A)
    elif mode == "hybrid":
        if residual_model is None:
            raise ValueError("residual_model must be provided when mode='hybrid'.")
        if not include_uq and hasattr(residual_model, "predict_point_fast"):
            out = residual_model.predict_point_fast(
                X_v=X_v,
                X_d=X_d,
                S=S,
                C_L=C_L,
                A=A,
                v_tip=hyd["v_tip"],
                Pg_V=hyd["Pg_V"],
                v_s=hyd["v_s"],
                mu_mech=mu_mech,
            )
            delta_kd = float(out["delta_kd"])
            eta_kla = float(out["eta_kla"])
            q_A_prod = max(0.0, float(out["q_A_prod"]))
            q_A_cons = max(0.0, float(out["q_A_cons"]))
            eta_mu = float(out["eta_mu"])
            mu = max(0.0, mu_mech * eta_mu)
        elif hasattr(residual_model, "predict_full_with_uq"):
            out = residual_model.predict_full_with_uq(
                X_v=X_v,
                X_d=X_d,
                S=S,
                C_L=C_L,
                A=A,
                v_tip=hyd["v_tip"],
                Pg_V=hyd["Pg_V"],
                v_s=hyd["v_s"],
                mu_mech=mu_mech,
            )
            delta_kd = float(out["delta_kd"])
            eta_kla = float(out["eta_kla"])
            q_A_prod = max(0.0, float(out["q_A_prod"]))
            q_A_cons = max(0.0, float(out["q_A_cons"]))
            eta_mu = float(out["eta_mu"])
            mu = max(0.0, mu_mech * eta_mu)
            std_kd = float(out.get("std_kd", 0.0))
            std_kla = float(out.get("std_kla", 0.0))
            std_qp = float(out.get("std_qp", 0.0))
            std_qc = float(out.get("std_qc", 0.0))
            std_mu = float(out.get("std_mu", 0.0))
            d_m = float(out.get("d_m", 0.0))
            d_m_kla = float(out.get("d_m_kla", 0.0))
            in_domain = 1.0 if out.get("in_domain", True) else 0.0
        else:
            try:
                delta_kd, eta_kla, q_A_net = residual_model.predict_residuals(
                    X_v=X_v, S=S, C_L=C_L, A=A, rpm=u.rpm, vvm=u.vvm, V=V, X_d=X_d
                )
            except TypeError:
                delta_kd, eta_kla, q_A_net = residual_model.predict_residuals(
                    X_v=X_v, S=S, C_L=C_L, A=A, rpm=u.rpm, vvm=u.vvm, V=V
                )
            mu = mu_mech
            q_A_prod = max(0.0, float(q_A_net)) * s_avail
            q_A_cons = max(0.0, -float(q_A_net)) * a_avail * o2_avail
        k_d = max(0.0, params.k_d0 + float(delta_kd))
        k_La = max(5.0, kla_nom * float(eta_kla))
    else:
        raise ValueError(f"Unsupported simulation mode: {mode}")

    q_P = (params.alpha * mu + params.beta) * s_avail * o2_avail
    q_A_net = q_A_prod - q_A_cons
    return {
        "mu": float(mu),
        "mu_mech": float(mu_mech),
        "eta_mu": float(eta_mu),
        "k_d": float(k_d),
        "k_La": float(k_La),
        "k_La_nom": float(kla_nom),
        "q_A_prod": float(q_A_prod),
        "q_A_cons": float(q_A_cons),
        "q_A_net": float(q_A_net),
        "q_P": float(q_P),
        "std_kd": float(std_kd),
        "std_kla": float(std_kla),
        "std_qp": float(std_qp),
        "std_qc": float(std_qc),
        "std_mu": float(std_mu),
        "d_m": float(d_m),
        "d_m_kla": float(d_m_kla),
        "in_domain": float(in_domain),
        "Pg_V": float(hyd["Pg_V"]),
        "v_s": float(hyd["v_s"]),
        "v_tip": float(hyd["v_tip"]),
    }


def bioreactor_rhs(
    t: float,
    x: np.ndarray,
    u: OperatingInputs,
    params: BioreactorParams,
    mode: Literal["ground_truth", "nominal", "hybrid"] = "ground_truth",
    residual_model: Optional[ResidualCorrectorProtocol] = None,
) -> np.ndarray:
    """Right-hand side dx/dt for the 7-state stoichiometrically coupled fed-batch bioreactor."""
    del t
    X_v = max(0.0, float(x[0]))
    X_d = max(0.0, float(x[1]))
    S = max(0.0, float(x[2]))
    C_L = max(0.0, float(x[3]))
    A = max(0.0, float(x[4]))
    P = max(0.0, float(x[5]))
    V = max(0.2, float(x[6]))

    F = 0.0 if V >= params.V_max else max(0.0, float(u.F))
    D = F / V
    S_feed = params.S_feed if u.S_feed is None else float(u.S_feed)

    rates = compute_rates(
        x, u, params, mode=mode, residual_model=residual_model, include_uq=False
    )
    mu = rates["mu"]
    k_d = rates["k_d"]
    k_La = rates["k_La"]
    q_A_prod = rates["q_A_prod"]
    q_A_cons = rates["q_A_cons"]
    q_P = rates["q_P"]

    s_avail = S / (0.002 + S)
    o2_avail = C_L / (0.005 + C_L)

    dXv_dt = (mu - k_d - D) * X_v
    dXd_dt = k_d * X_v - D * X_d
    q_S = (
        mu / params.Y_XS
        + params.m_s * s_avail
        + q_P / params.Y_PS
        + q_A_prod / params.Y_AS
    )
    dS_dt = D * (S_feed - S) - q_S * X_v
    our = 1000.0 * (
        mu / params.Y_XO
        + params.m_o * o2_avail
        + q_A_cons / params.Y_AO
    ) * X_v
    otr = k_La * (params.C_star - C_L)
    dCL_dt = otr - our - D * C_L
    dA_dt = (q_A_prod - q_A_cons) * X_v - D * A
    dP_dt = q_P * X_v - D * P
    dV_dt = F

    return np.array([dXv_dt, dXd_dt, dS_dt, dCL_dt, dA_dt, dP_dt, dV_dt], dtype=float)


def integrate_step(
    t0: float,
    t1: float,
    x0: np.ndarray,
    u: OperatingInputs,
    params: BioreactorParams,
    mode: Literal["ground_truth", "nominal", "hybrid"] = "ground_truth",
    residual_model: Optional[ResidualCorrectorProtocol] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> np.ndarray:
    """Integrate one control interval [t0, t1] with stiff Radau solver."""
    sol = solve_ivp(
        fun=lambda t, y: bioreactor_rhs(t, y, u, params, mode=mode, residual_model=residual_model),
        t_span=(t0, t1),
        y0=np.maximum(0.0, np.asarray(x0, dtype=float)),
        method="Radau",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed on [{t0}, {t1}]: {sol.message}")
    y_next = np.maximum(0.0, sol.y[:, -1])
    y_next[3] = min(y_next[3], params.C_star * 1.05)
    y_next[6] = min(y_next[6], params.V_max + 0.05)
    return y_next


def simulate_bioreactor(
    t_span: Tuple[float, float],
    x0: np.ndarray,
    control_fn: ControlCallback,
    params: Optional[BioreactorParams] = None,
    mode: Literal["ground_truth", "nominal", "hybrid"] = "ground_truth",
    residual_model: Optional[ResidualCorrectorProtocol] = None,
    dt_eval: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Simulate a full fed-batch run using zero-order-hold control steps."""
    p = params or BioreactorParams()
    t_start, t_end = float(t_span[0]), float(t_span[1])
    n_steps = max(1, int(round((t_end - t_start) / dt_eval)))
    t_grid = np.linspace(t_start, t_end, n_steps + 1)

    states = np.zeros((n_steps + 1, 7), dtype=float)
    states[0] = np.maximum(0.0, np.asarray(x0, dtype=float))

    feeds = np.zeros(n_steps + 1, dtype=float)
    rpms = np.zeros(n_steps + 1, dtype=float)
    vvms = np.zeros(n_steps + 1, dtype=float)
    mus = np.zeros(n_steps + 1, dtype=float)
    kds = np.zeros(n_steps + 1, dtype=float)
    klas = np.zeros(n_steps + 1, dtype=float)
    kla_noms = np.zeros(n_steps + 1, dtype=float)
    q_as = np.zeros(n_steps + 1, dtype=float)

    for i in range(n_steps):
        u = control_fn(t_grid[i], states[i])
        r = compute_rates(states[i], u, p, mode=mode, residual_model=residual_model)
        feeds[i] = 0.0 if states[i, 6] >= p.V_max else max(0.0, float(u.F))
        rpms[i] = float(u.rpm)
        vvms[i] = float(u.vvm)
        mus[i] = r["mu"]
        kds[i] = r["k_d"]
        klas[i] = r["k_La"]
        kla_noms[i] = r["k_La_nom"]
        q_as[i] = r["q_A_net"]

        states[i + 1] = integrate_step(
            t_grid[i], t_grid[i + 1], states[i], u, p, mode=mode, residual_model=residual_model
        )

    u_last = control_fn(t_grid[-1], states[-1])
    r_last = compute_rates(states[-1], u_last, p, mode=mode, residual_model=residual_model)
    feeds[-1] = 0.0 if states[-1, 6] >= p.V_max else max(0.0, float(u_last.F))
    rpms[-1] = float(u_last.rpm)
    vvms[-1] = float(u_last.vvm)
    mus[-1] = r_last["mu"]
    kds[-1] = r_last["k_d"]
    klas[-1] = r_last["k_La"]
    kla_noms[-1] = r_last["k_La_nom"]
    q_as[-1] = r_last["q_A_net"]

    return {
        "t": t_grid,
        "states": states,
        "X_v": states[:, 0],
        "X_d": states[:, 1],
        "S": states[:, 2],
        "C_L": states[:, 3],
        "DO_pct": 100.0 * states[:, 3] / p.C_star,
        "A": states[:, 4],
        "P": states[:, 5],
        "V": states[:, 6],
        "F": feeds,
        "rpm": rpms,
        "vvm": vvms,
        "mu": mus,
        "k_d": kds,
        "k_La": klas,
        "k_La_nom": kla_noms,
        "q_A_net": q_as,
    }


def disturbance_profile(
    disturbance_mode: str,
    t: float,
    rpm_nom: float,
    vvm_nom: float,
    S_feed_nom: float,
) -> Tuple[float, float, float]:
    """Return active (rpm, vvm, S_feed) after injecting optional real-plant disturbances."""
    rpm = float(rpm_nom)
    vvm = float(vvm_nom)
    s_feed = float(S_feed_nom)
    mode = (disturbance_mode or "none").lower()

    if mode == "none":
        return rpm, vvm, s_feed
    if mode in ("aeration_drop", "do_drop") and 6.0 <= t <= 9.5:
        vvm = max(0.35, 0.45 * vvm_nom)
    elif mode in ("stirrer_drop", "agitation_drop") and 7.0 <= t <= 10.5:
        rpm = max(180.0, 0.55 * rpm_nom)
    elif mode == "shear_spike" and 8.0 <= t <= 12.0:
        rpm = min(900.0, rpm_nom + 260.0)
    elif mode in ("feed_step", "feed_dilution") and t >= 7.0:
        s_feed = 0.70 * S_feed_nom
    elif mode == "combo":
        if 5.5 <= t <= 8.5:
            vvm = max(0.40, 0.55 * vvm_nom)
        if t >= 9.0:
            s_feed = 0.80 * S_feed_nom
    return rpm, vvm, s_feed
