"""Interactive Hybrid Mechanistic-ML Bioreactor Digital Twin Dashboard (Streamlit + Altair)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from hybrid_fedbatch_twin import (
    BioreactorDigitalTwin,
    BioreactorParams,
    ClosedLoopFeedController,
    ManualFeedController,
    NominalMechanisticController,
    OpenLoopExponentialController,
    OperatingInputs,
    STATE_NAMES,
    benchmark_controllers,
    compute_hydrodynamics,
    default_initial_state,
    recover_mechanistic_parameters,
    run_twin_estimation_loop,
    summarize_control_run,
    train_default_hybrid_model,
)

THEMES: Dict[str, Dict[str, str]] = {
    "Clean Biopharma Light": {
        "bg": "#F7F9FC",
        "card": "#FFFFFF",
        "border": "#DCE4EE",
        "text": "#132238",
        "muted": "#51627A",
        "accent": "#0B6E99",
        "hyb": "#0B6E99",
        "nom": "#D97706",
        "truth": "#059669",
        "meas": "#7C3AED",
        "band": "#0B6E99",
        "warn": "#DC2626",
        "ok": "#059669",
        "liquid": "#CFE8F3",
    },
    "Dark Process Room": {
        "bg": "#0E1525",
        "card": "#162033",
        "border": "#26344D",
        "text": "#E6EDF7",
        "muted": "#9AB0CE",
        "accent": "#38BDF8",
        "hyb": "#38BDF8",
        "nom": "#FBBF24",
        "truth": "#34D399",
        "meas": "#C084FC",
        "band": "#38BDF8",
        "warn": "#F87171",
        "ok": "#34D399",
        "liquid": "#1E3A5F",
    },
    "Editorial Clinical": {
        "bg": "#FAFAF8",
        "card": "#FFFFFF",
        "border": "#E5E2DC",
        "text": "#1C1917",
        "muted": "#57534E",
        "accent": "#1D4ED8",
        "hyb": "#1D4ED8",
        "nom": "#B45309",
        "truth": "#047857",
        "meas": "#9333EA",
        "band": "#1D4ED8",
        "warn": "#B91C1C",
        "ok": "#047857",
        "liquid": "#DBEAFE",
    },
}

VESSEL_SCALES: Dict[str, float] = {
    "5.0 L (Benchtop)": 5.0,
    "50.0 L (Pilot)": 50.0,
    "1000.0 L (Manufacturing)": 1000.0,
}


def inject_theme_css(theme: Dict[str, str]) -> None:
    """Inject clean theme-coherent CSS for KPI cards, badges, and layout."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme['bg']};
            color: {theme['text']};
        }}
        .kpi-card {{
            background: {theme['card']};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            padding: 12px 14px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }}
        .kpi-label {{
            font-size: 0.78rem;
            color: {theme['muted']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .kpi-val {{
            font-size: 1.32rem;
            font-weight: 700;
            color: {theme['text']};
            margin-top: 2px;
        }}
        .kpi-sub {{
            font-size: 0.76rem;
            color: {theme['muted']};
            margin-top: 3px;
        }}
        .dov-badge-ok {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background-color: {theme['ok']}22;
            border: 1px solid {theme['ok']};
            color: {theme['ok']};
            font-weight: 700;
            font-size: 0.78rem;
        }}
        .dov-badge-warn {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background-color: {theme['warn']}22;
            border: 1px solid {theme['warn']};
            color: {theme['warn']};
            font-weight: 700;
            font-size: 0.78rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bioreactor_canvas(
    Xv: float,
    DO_mgL: float,
    V: float,
    V_max: float,
    rpm: float,
    vvm: float,
    C_star: float,
    d_m: float,
    dov_p99: float,
    hyd: Dict[str, float],
    theme: Dict[str, str],
    source_label: str,
) -> None:
    """Render SVG fed-batch vessel schematic annotated strictly from allowed twin estimates."""
    fill_frac = float(np.clip(V / max(1e-6, V_max), 0.12, 0.95))
    liq_h = int(170 * fill_frac)
    liq_y = 210 - liq_h
    do_pct = 100.0 * max(0.0, DO_mgL) / max(1e-6, C_star)
    in_dom = d_m <= dov_p99
    status_col = theme["ok"] if in_dom else theme["warn"]
    status_txt = f"IN-DOMAIN (D_M {d_m:.2f} <= {dov_p99:.2f})" if in_dom else f"OOD ALERT (D_M {d_m:.2f} > {dov_p99:.2f})"

    svg = f"""
    <div style="background:{theme['card']};border:1px solid {theme['border']};border-radius:12px;padding:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-weight:700;color:{theme['text']};font-size:0.95rem;">
          Fed-Batch Bioreactor Schematic ({source_label})
        </span>
        <span style="color:{status_col};font-weight:700;font-size:0.78rem;">{status_txt}</span>
      </div>
      <svg viewBox="0 0 420 245" width="100%" height="230" xmlns="http://www.w3.org/2000/svg">
        <rect x="40" y="25" width="140" height="190" rx="18" fill="none" stroke="{theme['muted']}" stroke-width="3"/>
        <rect x="43" y="{liq_y}" width="134" height="{liq_h}" rx="12" fill="{theme['liquid']}" opacity="0.88"/>
        <line x1="110" y1="15" x2="110" y2="195" stroke="{theme['text']}" stroke-width="3"/>
        <rect x="82" y="135" width="56" height="7" rx="3" fill="{theme['accent']}"/>
        <rect x="82" y="175" width="56" height="7" rx="3" fill="{theme['accent']}"/>
        <circle cx="95" cy="{min(200, liq_y + 25)}" r="3.5" fill="{theme['accent']}" opacity="0.7"/>
        <circle cx="122" cy="{min(200, liq_y + 45)}" r="3" fill="{theme['accent']}" opacity="0.6"/>
        <circle cx="105" cy="{min(200, liq_y + 65)}" r="4" fill="{theme['accent']}" opacity="0.5"/>
        <text x="110" y="232" text-anchor="middle" fill="{theme['muted']}" font-size="11">
          V = {V:.2f} / {V_max:.1f} L
        </text>
        <g font-family="sans-serif" font-size="12" fill="{theme['text']}">
          <text x="205" y="48" font-weight="bold" fill="{theme['accent']}">Intensive Hydrodynamics</text>
          <text x="205" y="70">Agitation: {rpm:.0f} RPM (v_tip = {hyd['v_tip']:.2f} m/s)</text>
          <text x="205" y="90">Aeration: {vvm:.2f} vvm (v_s = {hyd['v_s']*1000:.2f} mm/s)</text>
          <text x="205" y="110">Power/Vol: P_g/V = {hyd['Pg_V']:.0f} W/m3</text>
          <text x="205" y="142" font-weight="bold" fill="{theme['accent']}">Estimated Broth State</text>
          <text x="205" y="164">Viable Biomass: {Xv:.2f} g/L</text>
          <text x="205" y="184">Dissolved O2: {DO_mgL:.2f} mg/L ({do_pct:.1f}% sat)</text>
          <text x="205" y="204">Mahalanobis DoV: D_M = {d_m:.2f} (p99 = {dov_p99:.2f})</text>
        </g>
      </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Calibrating Stage-1 iDoE parameters & training B=15 GAM-86 bootstrap ensemble...")
def load_calibrated_digital_twin() -> Tuple[Dict[str, Any], Any, Dict[str, float]]:
    """Train and cache Stage-1 iDoE recovery and Stage-2 GAM-86 ensemble once per server."""
    true_params = BioreactorParams()
    rec = recover_mechanistic_parameters(noise_std=0.015, seed=42, true_params=true_params)
    params_rec = rec["recovered_params"]
    model, neural_metrics = train_default_hybrid_model(
        params=params_rec,
        plant_params=true_params,
        noise_std=0.015,
        seed=7,
        n_bootstrap=15,
    )
    return rec, model, neural_metrics


@st.cache_data(show_spinner=False)
def run_cached_simulation(
    strategy: str,
    v_vessel: float,
    mu_target: float,
    s_set: float,
    f_manual: float,
    rpm: float,
    vvm: float,
    disturbance_mode: str,
    duration_h: float,
) -> Dict[str, np.ndarray]:
    """Execute closed-loop / open-loop digital twin run wired to Stage-1 recovered params."""
    rec, model, _ = load_calibrated_digital_twin()
    params_rec = rec["recovered_params"]
    true_params = BioreactorParams()

    scale = float(v_vessel) / 5.0
    p_twin = BioreactorParams(
        **{
            **params_rec.to_dict(),
            "V_vessel": float(v_vessel),
            "V_0": 2.0 * scale,
            "V_max": 5.0 * scale,
        }
    )
    p_plant = BioreactorParams(
        **{
            **true_params.to_dict(),
            "V_vessel": float(v_vessel),
            "V_0": 2.0 * scale,
            "V_max": 5.0 * scale,
        }
    )
    f_max = 0.30 * scale
    x0 = default_initial_state(p_twin)

    if strategy == "Closed-Loop Hybrid Controller":
        ctrl = ClosedLoopFeedController(
            params=p_twin,
            residual_model=model,
            mu_target=mu_target,
            S_set=s_set,
            rpm=rpm,
            vvm=vvm,
            F_max=f_max,
            disturbance_mode=disturbance_mode,
        )
    elif strategy == "Nominal Mechanistic Controller":
        ctrl = NominalMechanisticController(
            params=p_twin,
            mu_target=mu_target,
            S_set=s_set,
            rpm=rpm,
            vvm=vvm,
            F_max=f_max,
            disturbance_mode=disturbance_mode,
        )
    elif strategy == "Open-Loop Exponential Feed":
        ctrl = OpenLoopExponentialController(
            params=p_twin,
            x0=x0,
            mu_target=mu_target,
            rpm=rpm,
            vvm=vvm,
            F_max=f_max,
            disturbance_mode=disturbance_mode,
        )
    else:
        ctrl = ManualFeedController(
            params=p_twin,
            F_set=f_manual * scale,
            rpm=rpm,
            vvm=vvm,
            disturbance_mode=disturbance_mode,
        )

    return run_twin_estimation_loop(
        (0.0, float(duration_h)),
        x0,
        ctrl,
        params=p_twin,
        residual_model=model,
        seed=123,
        plant_params=p_plant,
    )


def make_state_chart(
    t: np.ndarray,
    hyb: np.ndarray,
    nom: np.ndarray,
    std: np.ndarray,
    meas: Optional[np.ndarray],
    truth: Optional[np.ndarray],
    y_label: str,
    theme: Dict[str, str],
    show_truth: bool,
    height: int = 235,
) -> alt.Chart:
    """Build layered Altair state plot with 95% bootstrap/EKF confidence band and sparse PAT dots."""
    df_band = pd.DataFrame(
        {
            "t": t,
            "lo": np.maximum(0.0, hyb - 1.96 * std),
            "hi": hyb + 1.96 * std,
        }
    )
    band = (
        alt.Chart(df_band)
        .mark_area(opacity=0.22, color=theme["band"])
        .encode(x=alt.X("t:Q", title="Batch Time (h)"), y=alt.Y("lo:Q", title=y_label), y2="hi:Q")
    )

    frames = [
        pd.DataFrame({"t": t, "value": hyb, "Series": "Hybrid Twin EKF"}),
        pd.DataFrame({"t": t, "value": nom, "Series": "Nominal EKF"}),
    ]
    domain = ["Hybrid Twin EKF", "Nominal EKF"]
    colors = [theme["hyb"], theme["nom"]]
    if show_truth and truth is not None:
        frames.append(pd.DataFrame({"t": t, "value": truth, "Series": "Plant Ground Truth"}))
        domain.append("Plant Ground Truth")
        colors.append(theme["truth"])

    df_lines = pd.concat(frames, ignore_index=True)
    lines = (
        alt.Chart(df_lines)
        .mark_line(strokeWidth=2.3)
        .encode(
            x="t:Q",
            y="value:Q",
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.condition(
                alt.datum.Series == "Nominal EKF",
                alt.value([5, 4]),
                alt.value([1, 0]),
            ),
        )
    )

    layers: List[alt.Chart] = [band, lines]
    if meas is not None:
        valid = np.isfinite(meas)
        if np.any(valid):
            df_pts = pd.DataFrame(
                {"t": t[valid], "value": meas[valid], "Sensor": "Measured PAT/LIMS"}
            )
            pts = (
                alt.Chart(df_pts)
                .mark_point(size=36, filled=True, color=theme["meas"], opacity=0.85)
                .encode(x="t:Q", y="value:Q", tooltip=["t:Q", "value:Q"])
            )
            layers.append(pts)

    return alt.layer(*layers).properties(height=height).interactive()


def build_dataframe(sim: Dict[str, np.ndarray], include_truth: bool = False) -> pd.DataFrame:
    """Assemble full EKF state, uncertainty, DoV, and optional validation truth DataFrame."""
    data: Dict[str, Any] = {
        "t_h": sim["t"],
        "F_L_h": sim["inputs"][:, 0],
        "rpm": sim["inputs"][:, 1],
        "vvm": sim["inputs"][:, 2],
    }
    for idx, name in enumerate(STATE_NAMES):
        data[f"{name}_hyb"] = sim["est_hyb"][:, idx]
        data[f"{name}_nom"] = sim["est_nom"][:, idx]
        data[f"{name}_std"] = sim["std_hyb"][:, idx]
        if include_truth:
            data[f"{name}_true"] = sim["true_states"][:, idx]

    data["mu_hyb"] = sim["mu_hyb"]
    data["mu_nom"] = sim["mu_nom"]
    data["kd_hyb"] = sim["kd_hyb"]
    data["kla_hyb"] = sim["kla_hyb"]
    data["qAnet_hyb"] = sim["qAnet_hyb"]
    data["d_m"] = sim["d_m"]
    data["d_m_kla"] = sim["d_m_kla"]
    data["in_domain"] = sim["in_domain"]
    if include_truth:
        data["mu_true"] = sim["mu_true"]
        data["kd_true"] = sim["kd_true"]
        data["kla_true"] = sim["kla_true"]
    return pd.DataFrame(data)


def _render_kpi_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-val">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Hybrid Fed-Batch Digital Twin (E. coli BL21)",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Digital Twin Controls")
        theme_name = st.selectbox("UI Theme", list(THEMES.keys()), index=0)
        display_mode = st.radio(
            "Display Mode",
            [
                "Operator Twin View (No Oracle Truth)",
                "Validation Benchmark (With Ground Truth)",
            ],
            index=0,
        )
        show_truth = display_mode == "Validation Benchmark (With Ground Truth)"

        scale_label = st.selectbox("Vessel Scale (V_vessel)", list(VESSEL_SCALES.keys()), index=0)
        v_vessel = VESSEL_SCALES[scale_label]
        scale = v_vessel / 5.0

        strategy = st.selectbox(
            "Feed Control Strategy",
            [
                "Closed-Loop Hybrid Controller",
                "Nominal Mechanistic Controller",
                "Open-Loop Exponential Feed",
                "Manual Constant Feed",
            ],
            index=0,
        )
        mu_target = st.slider("Target Specific Growth Rate mu (1/h)", 0.12, 0.38, 0.28, 0.01)
        s_set = st.slider("Glucose Setpoint S_set (g/L)", 0.08, 0.80, 0.25, 0.02)
        f_manual = st.slider("Manual Feed Rate Base (L/h @ 5L eq)", 0.01, 0.25, 0.06, 0.01)

        default_rpm = float(np.clip(round(450.0 * (5.0 / v_vessel) ** (2.0 / 9.0)), 80.0, 850.0))
        default_vvm = float(np.clip(round(1.2 * (5.0 / v_vessel) ** (1.0 / 3.0), 2), 0.15, 2.2))
        rpm = st.slider("Impeller Agitation (RPM)", 80.0, 850.0, default_rpm, 10.0)
        vvm = st.slider("Aeration Rate (vvm)", 0.15, 2.20, default_vvm, 0.05)

        disturbance_mode = st.selectbox(
            "Plant Disturbance Scenario",
            ["none", "aeration_drop", "stirrer_drop", "shear_spike", "feed_step", "combo"],
            index=0,
        )
        duration_h = st.slider("Batch Duration (h)", 8.0, 20.0, 16.0, 1.0)

    theme = THEMES[theme_name]
    inject_theme_css(theme)

    rec, model, neural_metrics = load_calibrated_digital_twin()
    params_rec = rec["recovered_params"]
    true_params = BioreactorParams()
    p_twin = BioreactorParams(
        **{
            **params_rec.to_dict(),
            "V_vessel": float(v_vessel),
            "V_0": 2.0 * scale,
            "V_max": 5.0 * scale,
        }
    )
    p_plant = BioreactorParams(
        **{
            **true_params.to_dict(),
            "V_vessel": float(v_vessel),
            "V_0": 2.0 * scale,
            "V_max": 5.0 * scale,
        }
    )

    sim = run_cached_simulation(
        strategy=strategy,
        v_vessel=v_vessel,
        mu_target=mu_target,
        s_set=s_set,
        f_manual=f_manual,
        rpm=rpm,
        vvm=vvm,
        disturbance_mode=disturbance_mode,
        duration_h=duration_h,
    )
    kpis = summarize_control_run(sim, mu_target=mu_target, params=p_twin)
    dov_p99 = float(neural_metrics["dov_p99"])
    in_dom_run = kpis["max_d_m"] <= dov_p99

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.title("Hybrid Mechanistic-ML Bioreactor Digital Twin")
        st.caption(
            f"Portela Level 3-Ready Closed-Loop E. coli BL21 State Estimator (In-Silico Plant Harness) | "
            f"GAM-{int(neural_metrics['n_params'])} ({int(neural_metrics['n_params'])} params, "
            f"B={int(neural_metrics['n_bootstrap'])} bootstraps) | Scale: {scale_label}"
        )
    with hdr_r:
        badge_cls = "dov-badge-ok" if in_dom_run else "dov-badge-warn"
        badge_txt = "IN-DOMAIN OPERATION" if in_dom_run else "OUT-OF-DOMAIN ALERT"
        st.markdown(
            f'<div style="text-align:right;padding-top:14px;">'
            f'<span class="{badge_cls}">{badge_txt}<br/>max D_M = {kpis["max_d_m"]:.2f} / p99 = {dov_p99:.2f}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Top Schematic + KPI Row (Strictly Oracle-Free in Operator Twin View)
    end_hyd = compute_hydrodynamics(
        float(sim["inputs"][-1, 1]),
        float(sim["inputs"][-1, 2]),
        float(sim["est_hyb"][-1, 6]),
        p_twin,
    )
    col_svg, col_kpi = st.columns([1.35, 1.65])
    with col_svg:
        render_bioreactor_canvas(
            Xv=float(sim["est_hyb"][-1, 0]),
            DO_mgL=float(sim["est_hyb"][-1, 3]),
            V=float(sim["est_hyb"][-1, 6]),
            V_max=p_twin.V_max,
            rpm=float(sim["inputs"][-1, 1]),
            vvm=float(sim["inputs"][-1, 2]),
            C_star=p_twin.C_star,
            d_m=float(sim["d_m"][-1]),
            dov_p99=dov_p99,
            hyd=end_hyd,
            theme=theme,
            source_label="Hybrid EKF Estimate",
        )
    with col_kpi:
        r1 = st.columns(3)
        r2 = st.columns(3)
        if not show_truth:
            with r1[0]:
                _render_kpi_card(
                    "Est. Viable Biomass X_v",
                    f"{kpis['est_final_Xv_g_L']:.2f} g/L",
                    f"±{1.96 * kpis['est_final_Xv_std']:.2f} g/L (95% CI)",
                )
            with r1[1]:
                _render_kpi_card(
                    "Est. Product Titer P",
                    f"{kpis['est_final_P_g_L']:.2f} g/L",
                    f"±{1.96 * kpis['est_final_P_std']:.2f} g/L (95% CI)",
                )
            with r1[2]:
                _render_kpi_card(
                    "Est. Dead Biomass X_d",
                    f"{kpis['est_final_Xd_g_L']:.2f} g/L",
                    f"±{1.96 * kpis['est_final_Xd_std']:.2f} g/L (95% CI)",
                )
            with r2[0]:
                _render_kpi_card(
                    "Est. Peak Acetate A_max",
                    f"{kpis['est_peak_acetate_g_L']:.2f} g/L",
                    f"Min S_hat = {kpis['est_min_substrate_g_L']:.2f} g/L",
                )
            with r2[1]:
                _render_kpi_card(
                    "Est. Min Dissolved O2",
                    f"{kpis['est_min_DO_pct']:.1f}% sat",
                    f"C* = {p_twin.C_star:.1f} mg/L",
                )
            with r2[2]:
                _render_kpi_card(
                    "Tracked mu_RMSE & DoV",
                    f"{kpis['est_mu_rmse']:.3f} 1/h",
                    f"OOD time = {100.0 * kpis['ood_fraction']:.1f}%",
                )
        else:
            with r1[0]:
                _render_kpi_card(
                    "True Final X_v",
                    f"{kpis['final_Xv_g_L']:.2f} g/L",
                    f"Est {kpis['est_final_Xv_g_L']:.2f} | RMSE {kpis['hybrid_Xv_rmse']:.3f}",
                )
            with r1[1]:
                _render_kpi_card(
                    "True Final Product P",
                    f"{kpis['final_P_g_L']:.2f} g/L",
                    f"Est {kpis['est_final_P_g_L']:.2f} | RMSE {kpis['hybrid_P_rmse']:.3f}",
                )
            with r1[2]:
                _render_kpi_card(
                    "Peak Acetate (True)",
                    f"{kpis['peak_acetate_g_L']:.2f} g/L",
                    f"A_RMSE = {kpis['hybrid_A_rmse']:.3f} g/L",
                )
            with r2[0]:
                _render_kpi_card(
                    "Min DO (True)",
                    f"{kpis['min_DO_pct']:.1f}% sat",
                    f"Min S = {kpis['min_substrate_g_L']:.3f} g/L",
                )
            with r2[1]:
                _render_kpi_card(
                    "True Growth mu_RMSE",
                    f"{kpis['mu_rmse']:.4f} 1/h",
                    f"Setpoint {mu_target:.2f} 1/h",
                )
            with r2[2]:
                _render_kpi_card(
                    "7-State Parity nRMSE",
                    f"Hyb {kpis['hybrid_norm_rmse']:.4f}",
                    f"Nom {kpis['nominal_norm_rmse']:.4f} (fair sensors)",
                )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. State & Mass-Transfer Estimation",
            "2. Closed-Loop Control Benchmark",
            f"3. GAM-{int(neural_metrics['n_params'])} & Stage-1 Identifiability",
            "4. Telemetry & ALCOA+ Export",
        ]
    )

    with tab1:
        t = sim["t"]
        meas = sim["meas_states"]
        truth = sim["true_states"] if show_truth else None

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                make_state_chart(
                    t, sim["est_hyb"][:, 0], sim["est_nom"][:, 0], sim["std_hyb"][:, 0],
                    meas[:, 0], truth[:, 0] if truth is not None else None,
                    "Viable Biomass X_v (g/L)", theme, show_truth,
                ),
                width="stretch",
            )
            st.altair_chart(
                make_state_chart(
                    t, sim["est_hyb"][:, 2], sim["est_nom"][:, 2], sim["std_hyb"][:, 2],
                    meas[:, 2], truth[:, 2] if truth is not None else None,
                    "Glucose Substrate S (g/L)", theme, show_truth,
                ),
                width="stretch",
            )
            st.altair_chart(
                make_state_chart(
                    t, sim["est_hyb"][:, 4], sim["est_nom"][:, 4], sim["std_hyb"][:, 4],
                    meas[:, 4], truth[:, 4] if truth is not None else None,
                    "Acetate Overflow A (g/L)", theme, show_truth,
                ),
                width="stretch",
            )
        with c2:
            st.altair_chart(
                make_state_chart(
                    t, sim["est_hyb"][:, 3], sim["est_nom"][:, 3], sim["std_hyb"][:, 3],
                    meas[:, 3], truth[:, 3] if truth is not None else None,
                    "Dissolved Oxygen C_L (mg/L)", theme, show_truth,
                ),
                width="stretch",
            )
            st.altair_chart(
                make_state_chart(
                    t, sim["est_hyb"][:, 5], sim["est_nom"][:, 5], sim["std_hyb"][:, 5],
                    None, truth[:, 5] if truth is not None else None,
                    "Recombinant Product P (g/L)", theme, show_truth,
                ),
                width="stretch",
            )
            st.altair_chart(
                make_state_chart(
                    t, sim["est_hyb"][:, 1], sim["est_nom"][:, 1], sim["std_hyb"][:, 1],
                    None, truth[:, 1] if truth is not None else None,
                    "Dead Biomass X_d (g/L)", theme, show_truth,
                ),
                width="stretch",
            )

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            df_rate = pd.DataFrame(
                {
                    "t": t,
                    "Feed F (L/h)": sim["inputs"][:, 0],
                    "mu_hyb (1/h)": sim["mu_hyb"],
                    "mu_nom (1/h)": sim["mu_nom"],
                }
            )
            st.line_chart(df_rate.set_index("t")[["mu_hyb (1/h)", "mu_nom (1/h)"]])
        with r_col2:
            df_kla_dov = pd.DataFrame(
                {
                    "t": t,
                    "k_La_hyb (1/h)": sim["kla_hyb"],
                    "Mahalanobis D_M": sim["d_m"],
                    "DoV p99": np.full_like(t, dov_p99),
                }
            )
            st.line_chart(df_kla_dov.set_index("t")[["Mahalanobis D_M", "DoV p99"]])

    with tab2:
        st.subheader("Three-Way Sensor-Parity Controller Benchmark")
        st.caption(
            f"Compares Hybrid Closed-Loop vs Nominal Closed-Loop vs Open-Loop Exponential "
            f"under active parameters (scale={scale_label}, rpm={rpm:.0f}, vvm={vvm:.2f})."
        )
        if st.button("Run 3-Way Controller Benchmark at Current Operating Point"):
            with st.spinner("Running 3-way closed-loop / open-loop comparison..."):
                bench = benchmark_controllers(
                    params=p_twin,
                    residual_model=model,
                    duration_h=float(duration_h),
                    mu_target=float(mu_target),
                    rpm=float(rpm),
                    vvm=float(vvm),
                    F_max=0.30 * scale,
                    plant_params=p_plant,
                )
            rows = []
            for strat_key, entry in bench.items():
                s = entry["summary"]
                row = {
                    "Strategy": strat_key,
                    "Est Xv_end": round(s["est_final_Xv_g_L"], 2),
                    "Est P_end": round(s["est_final_P_g_L"], 2),
                    "Est A_peak": round(s["est_peak_acetate_g_L"], 3),
                    "Est Min DO%": round(s["est_min_DO_pct"], 1),
                    "Max D_M": round(s["max_d_m"], 2),
                }
                if show_truth:
                    row.update(
                        {
                            "True Xv_end": round(s["final_Xv_g_L"], 2),
                            "True P_end": round(s["final_P_g_L"], 2),
                            "True A_peak": round(s["peak_acetate_g_L"], 3),
                            "True Min DO%": round(s["min_DO_pct"], 1),
                            "Hyb nRMSE": round(s["hybrid_norm_rmse"], 4),
                            "Nom nRMSE": round(s["nominal_norm_rmse"], 4),
                        }
                    )
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("Click the button above to execute the full 3-way benchmark at the current vessel scale and agitation/aeration settings.")

    with tab3:
        st.subheader(f"Stage-1 iDoE Calibration & Stage-2 GAM-{int(neural_metrics['n_params'])} Uncertainty Structure")
        m_cols = st.columns(5)
        m_cols[0].metric(f"GAM-{int(neural_metrics['n_params'])} Params", int(neural_metrics["n_params"]))
        m_cols[1].metric("Bootstrap B", int(neural_metrics["n_bootstrap"]))
        m_cols[2].metric("AIC / BIC", f"{neural_metrics['aic']:.0f} / {neural_metrics['bic']:.0f}")
        m_cols[3].metric("Rel. FIM Cond", f"{rec['condition_number_rel']:.2f}")
        m_cols[4].metric("Rollout Hyb/Nom", f"{neural_metrics['hybrid_norm_rmse']:.3f} vs {neural_metrics['nominal_norm_rmse']:.3f}")

        st1_rows = []
        for pname in rec["param_names"]:
            row_p = {
                "Parameter": pname,
                "Recovered Value": round(rec["recovered_values"][pname], 4),
                "Std Error (SE)": round(rec["standard_errors"][pname], 4),
            }
            if show_truth:
                row_p["True Value"] = round(rec["true_values"][pname], 4)
                row_p["Rel Error (%)"] = round(rec["relative_error_pct"][pname], 2)
            st1_rows.append(row_p)
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Stage-1 Multi-Batch iDoE Recovery**")
            st.dataframe(pd.DataFrame(st1_rows), width="stretch")
            st.markdown(
                f"**Held-Out Rate R2**: `k_d = {neural_metrics['kd_r2']:.4f}` | "
                f"`k_La = {neural_metrics['kla_r2']:.4f}` | "
                f"`q_A_net = {neural_metrics['qAnet_r2']:.4f}` | "
                f"`mu = {neural_metrics['mu_r2']:.4f}`"
            )
        with c_right:
            st.markdown("**Learned Broth Viscosity Damping eta_kla(X_tot) ± 95% CI**")
            x_grid = np.linspace(0.0, 45.0, 90)
            zeros = np.zeros_like(x_grid)
            hyd_grid = compute_hydrodynamics(rpm, vvm, p_twin.V_0, p_twin)
            uq_visc = model.predict_batch_with_uq(
                X_v=x_grid,
                X_d=zeros,
                S=np.full_like(x_grid, 0.3),
                C_L=np.full_like(x_grid, 5.0),
                A=zeros,
                v_tip=np.full_like(x_grid, hyd_grid["v_tip"]),
                Pg_V=np.full_like(x_grid, hyd_grid["Pg_V"]),
                v_s=np.full_like(x_grid, hyd_grid["v_s"]),
                mu_mech=np.full_like(x_grid, 0.3),
            )
            df_visc = pd.DataFrame(
                {
                    "X_total": x_grid,
                    "eta_kla": uq_visc["eta_kla"],
                    "lo": np.maximum(0.0, uq_visc["eta_kla"] - 1.96 * uq_visc["std_kla"]),
                    "hi": np.minimum(1.05, uq_visc["eta_kla"] + 1.96 * uq_visc["std_kla"]),
                }
            )
            band_v = (
                alt.Chart(df_visc)
                .mark_area(opacity=0.25, color=theme["band"])
                .encode(x=alt.X("X_total:Q", title="Total Biomass X_v + X_d (g/L)"), y=alt.Y("lo:Q", title="eta_kla (-)"), y2="hi:Q")
            )
            line_v = (
                alt.Chart(df_visc)
                .mark_line(strokeWidth=2.5, color=theme["hyb"])
                .encode(x="X_total:Q", y="eta_kla:Q")
            )
            st.altair_chart(alt.layer(band_v, line_v).properties(height=270), width="stretch")

    with tab4:
        st.subheader("Batch Telemetry & ALCOA+ Audit Record")
        df_export = build_dataframe(sim, include_truth=show_truth)
        st.dataframe(df_export.head(20), width="stretch")

        twin_session = BioreactorDigitalTwin.from_simulation_run(
            sim,
            params=p_twin,
            residual_model=model,
            calibration_metadata={
                "max_relative_error_pct": rec["max_relative_error_pct"],
                "condition_number_rel": rec["condition_number_rel"],
                "calibration_sha256": rec["calibration_sha256"],
                "scale_vessel_L": v_vessel,
            },
        )
        audit_payload = twin_session.to_audit_record()
        audit_payload["run_kpis"] = {k: round(float(v), 6) for k, v in kpis.items()}

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Telemetry CSV",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name=f"hybrid_twin_telemetry_{int(v_vessel)}L.csv",
                mime="text/csv",
            )
        with dl2:
            st.download_button(
                "Download ALCOA+ Audit Provenance JSON",
                data=json.dumps(audit_payload, indent=2).encode("utf-8"),
                file_name=f"hybrid_twin_alcoa_{int(v_vessel)}L.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
