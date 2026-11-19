"""MCP Server exposing hybrid-fedbatch-twin physics, EKF, ALCOA+ replay, and Uhlendorff CHO-K1 data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

from api.db import create_db_engine_and_sessionmaker, load_twin_session, save_twin_session
from api.runtime import (
    ROOT_DIR,
    create_twin_instance,
    load_calibrated_assets,
    scale_feed_cap,
    scale_params,
    state_dict_from_vector,
)
from hybrid_fedbatch_twin.controller import ClosedLoopFeedController
from hybrid_fedbatch_twin.mechanistic import (
    default_initial_state,
    default_operating_inputs,
    simulate_bioreactor,
)

UHLENDORFF_DATA_DIR = ROOT_DIR / "data" / "uhlendorff_2026" / "Time-axis aligned raw data tables"
UHLENDORFF_ARTIFACT = ROOT_DIR / "artifacts" / "uhlendorff_2026_validation.json"

mcp = MCPServer("hybrid-fedbatch-twin")

_DB_CACHE: Dict[str, Any] = {}


def _get_sessionmaker(database_url: Optional[str] = None):
    url = database_url or os.environ.get("DATABASE_URL") or "sqlite:///./hybrid_twin_ledger.db"
    if url not in _DB_CACHE:
        engine, SessionLocal = create_db_engine_and_sessionmaker(url)
        _DB_CACHE[url] = (engine, SessionLocal)
    return _DB_CACHE[url][1]


@mcp.tool()
def simulate_fedbatch_run(
    vessel_scale_L: Literal[5, 50, 1000] = 5,
    duration_h: float = 16.0,
    mode: Literal["hybrid", "nominal", "ground_truth"] = "hybrid",
    mu_target: float = 0.28,
    rpm: Optional[float] = None,
    vvm: Optional[float] = None,
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none",
    dt_eval: float = 0.2,
) -> Dict[str, Any]:
    """Run deterministic fed-batch ODE simulation using GAM-86 / Monod kinetics at 5L, 50L, or 1000L."""
    base_params, model = load_calibrated_assets()
    p_scaled = scale_params(vessel_scale_L, base_params)
    x0 = default_initial_state(p_scaled)
    u_def = default_operating_inputs(p_scaled)
    rpm_val = float(rpm) if rpm is not None else float(u_def.rpm)
    vvm_val = float(vvm) if vvm is not None else float(u_def.vvm)
    ctrl = ClosedLoopFeedController(
        params=p_scaled,
        residual_model=model,
        mu_target=mu_target,
        rpm=rpm_val,
        vvm=vvm_val,
        F_max=scale_feed_cap(vessel_scale_L),
        disturbance_mode=disturbance_mode,
    )
    sim = simulate_bioreactor(
        t_span=(0.0, duration_h),
        x0=x0,
        control_fn=ctrl,
        params=p_scaled,
        mode=mode,
        residual_model=model if mode == "hybrid" else None,
        dt_eval=dt_eval,
    )
    return {
        "vessel_scale_L": int(vessel_scale_L),
        "mode": mode,
        "duration_h": float(duration_h),
        "n_points": int(len(sim["t"])),
        "final_state": {
            "X_v_g_L": float(sim["X_v"][-1]),
            "X_d_g_L": float(sim["X_d"][-1]),
            "S_g_L": float(sim["S"][-1]),
            "DO_pct": float(sim["DO_pct"][-1]),
            "A_g_L": float(sim["A"][-1]),
            "P_g_L": float(sim["P"][-1]),
            "V_L": float(sim["V"][-1]),
        },
        "peak_acetate_g_L": float(np.max(sim["A"])),
        "min_DO_pct": float(np.min(sim["DO_pct"])),
        "trajectory_tail": [
            {
                "t_h": float(sim["t"][i]),
                "X_v": float(sim["X_v"][i]),
                "S": float(sim["S"][i]),
                "DO_pct": float(sim["DO_pct"][i]),
                "A": float(sim["A"][i]),
                "P": float(sim["P"][i]),
                "V": float(sim["V"][i]),
            }
            for i in range(max(0, len(sim["t"]) - 5), len(sim["t"]))
        ],
    }


@mcp.tool()
def ingest_pat_and_estimate_state(
    t_h: float,
    y_Xv: float,
    y_CL: float,
    y_V: float,
    F_act: float,
    rpm_act: Optional[float] = None,
    vvm_act: Optional[float] = None,
    session_id: Optional[str] = None,
    vessel_scale_L: Literal[5, 50, 1000] = 5,
    batch_id: str = "FB-BL21-MCP-001",
    operator_id: str = "mcp_agent",
    database_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Continuous-Discrete expm EKF update on online PAT telemetry and return Mahalanobis DoV status."""
    SessionLocal = _get_sessionmaker(database_url)
    _, model = load_calibrated_assets()
    sid = session_id or uuid4().hex[:16]

    with SessionLocal() as db:
        loaded = load_twin_session(db, sid, model)
        if loaded is None:
            twin = create_twin_instance(
                vessel_scale_L=vessel_scale_L,
                batch_id=batch_id,
                operator_id=operator_id,
            )
            scale_l = int(vessel_scale_L)
        else:
            twin = loaded["twin"]
            scale_l = int(loaded["vessel_scale_L"])

        rpm_val = float(rpm_act) if rpm_act is not None else float(twin.last_u.rpm)
        vvm_val = float(vvm_act) if vvm_act is not None else float(twin.last_u.vvm)
        res = twin.ingest_online_pat(
            t_h=t_h,
            y_Xv=y_Xv,
            y_CL=y_CL,
            y_V=y_V,
            F_act=F_act,
            rpm_act=rpm_val,
            vvm_act=vvm_val,
        )
        save_twin_session(db, sid, scale_l, twin)

    return {
        "session_id": sid,
        "vessel_scale_L": scale_l,
        "t_h": float(res["t_h"]),
        "x_hat": state_dict_from_vector(res["x_hat"]),
        "x_std": state_dict_from_vector(res["x_std"]),
        "mu_hat": float(res["mu_hat"]),
        "k_La_hat": float(res["k_La_hat"]),
        "k_d_hat": float(res["k_d_hat"]),
        "q_A_net_hat": float(res["q_A_net_hat"]),
        "d_m": float(res["d_m"]),
        "in_domain": bool(res["in_domain"]),
        "undecided": bool(res["undecided"]),
        "residual_breach": bool(res["residual_breach"]),
        "gate_reason": str(res["gate_reason"]),
        "nis": float(res["nis"]),
        "spe": float(res["spe"]),
    }


@mcp.tool()
def replay_delayed_lims_assay(
    session_id: str,
    t_sample_h: float,
    y_S: float,
    y_A: float,
    y_P: Optional[float] = None,
    database_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute retrodictive EKF replay for a delayed at-line LIMS sample and return signed ALCOA+ record."""
    SessionLocal = _get_sessionmaker(database_url)
    _, model = load_calibrated_assets()

    with SessionLocal() as db:
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise ValueError(f"Twin session '{session_id}' does not exist in persistence ledger.")
        twin = loaded["twin"]
        replay = twin.ingest_atline_assay(
            t_sample_h=t_sample_h,
            y_S=y_S,
            y_A=y_A,
            y_P=y_P,
        )
        save_twin_session(db, session_id, loaded["vessel_scale_L"], twin)
        audit = twin.to_audit_record()

    return {
        "session_id": session_id,
        "t_now": float(replay["t_now"]),
        "t_sample_h": float(replay["t_sample_h"]),
        "replayed_steps": int(replay["replayed_steps"]),
        "x_hat": state_dict_from_vector(replay["x_hat"]),
        "x_std": state_dict_from_vector(replay["x_std"]),
        "d_m": float(replay["d_m"]),
        "in_domain": bool(replay["in_domain"]),
        "undecided": bool(replay["undecided"]),
        "residual_breach": bool(replay["residual_breach"]),
        "gate_reason": str(replay["gate_reason"]),
        "nis": float(replay["nis"]),
        "spe": float(replay["spe"]),
        "state_delta_l2": float(replay["state_delta_l2"]),
        "alcoa_record": audit,
    }


def _resolve_campaign_wmse(
    exp_id: str,
    bench: Dict[str, Any],
    anomalous_runs: Sequence[str],
) -> Optional[float]:
    """Resolve per-campaign held-out wMSE from current or legacy Uhlendorff validation artifacts."""
    per_camp = bench.get("per_campaign_test_wmse", {})
    if isinstance(per_camp, dict) and exp_id in per_camp:
        return float(per_camp[exp_id])

    is_anom = any(str(a) == exp_id or str(a).startswith(f"{exp_id}_") for a in anomalous_runs)
    if is_anom and "anomaly_unmasked_wmse" in bench:
        return float(bench["anomaly_unmasked_wmse"])

    exp_lower = exp_id.lower()
    if f"{exp_lower}_wmse" in bench:
        return float(bench[f"{exp_lower}_wmse"])
    if f"{exp_lower}_unmasked_wmse" in bench:
        return float(bench[f"{exp_lower}_unmasked_wmse"])
    return None


@mcp.tool()
def search_uhlendorff_campaigns(
    mode: Optional[Literal["batch", "fed-batch"]] = None,
    campaign_id: Optional[str] = None,
    split: Optional[Literal["dev", "test"]] = None,
    max_wmse: Optional[float] = None,
) -> Dict[str, Any]:
    """Query the 24 wet-lab CHO-K1 runs in data/uhlendorff_2026/ and artifacts/uhlendorff_2026_validation.json."""
    validation: Dict[str, Any] = {}
    if UHLENDORFF_ARTIFACT.exists():
        with UHLENDORFF_ARTIFACT.open("r", encoding="utf-8") as f:
            validation = json.load(f)

    meta = validation.get("dataset_metadata", {})
    dev_campaigns = set(meta.get("dev_campaigns", []))
    test_campaigns = set(meta.get("test_campaigns", []))
    anomalous_runs = list(meta.get("anomalous_runs", []))
    bench = validation.get("benchmark_comparison", {}).get("remediated", {})
    held_out = validation.get("held_out_trajectories", {})
    vol_series = validation.get("mass_balance_discovery", {}).get("volume_series", {})

    runs: List[Dict[str, Any]] = []
    if UHLENDORFF_DATA_DIR.exists():
        offline_files = sorted(UHLENDORFF_DATA_DIR.glob("*_offline.csv"))
        for path in offline_files:
            parts = path.name.split("_")
            exp_id = parts[0]
            run_mode = "fed-batch" if "fed-batch" in path.name else "batch"
            rep_id = "R01" if "_R01_" in path.name else "R02"
            run_split = "test" if exp_id in test_campaigns else "dev"

            if campaign_id and exp_id.upper() != campaign_id.upper():
                continue
            if mode and run_mode != mode:
                continue
            if split and run_split != split:
                continue

            wmse_val = _resolve_campaign_wmse(exp_id, bench, anomalous_runs)

            if max_wmse is not None and (wmse_val is None or wmse_val > max_wmse):
                continue

            df = pd.read_csv(path)
            t_col = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
            vcd_col = next((c for c in df.columns if "vcd" in c.lower()), None)
            glc_col = next((c for c in df.columns if "glc" in c.lower()), None)
            lac_col = next((c for c in df.columns if "lac" in c.lower()), None)

            runs.append(
                {
                    "campaign_id": exp_id,
                    "replicate": rep_id,
                    "mode": run_mode,
                    "split": run_split,
                    "offline_file": path.name,
                    "n_offline_rows": int(len(df)),
                    "duration_h": float(df[t_col].dropna().max()) if len(df) else 0.0,
                    "peak_vcd_1e6_mL": float(df[vcd_col].dropna().max()) if vcd_col else None,
                    "max_glc_g_L": float(df[glc_col].dropna().max()) if glc_col else None,
                    "peak_lac_g_L": float(df[lac_col].dropna().max()) if lac_col else None,
                    "validation_wmse": wmse_val,
                    "has_heldout_trajectory": exp_id in held_out,
                }
            )
    else:
        # Clean-checkout fallback: derive all 24 campaign-replicates from tracked validation artifact
        all_exps = sorted(dev_campaigns | test_campaigns) or [f"EXP{i:03d}" for i in range(1, 13)]
        for exp_id in all_exps:
            ho_entry = held_out.get(exp_id, {})
            if ho_entry.get("mode"):
                run_mode = str(ho_entry["mode"])
            elif isinstance(vol_series, dict) and exp_id in vol_series:
                run_mode = "fed-batch"
            else:
                run_mode = "fed-batch" if exp_id in {"EXP010", "EXP011", "EXP012"} else "batch"
            run_split = "test" if exp_id in test_campaigns else "dev"

            if campaign_id and exp_id.upper() != campaign_id.upper():
                continue
            if mode and run_mode != mode:
                continue
            if split and run_split != split:
                continue

            wmse_val = _resolve_campaign_wmse(exp_id, bench, anomalous_runs)

            if max_wmse is not None and (wmse_val is None or wmse_val > max_wmse):
                continue

            for rep_id, t_key, obs_key in (("R01", "t_off_r01", "obs_r01"), ("R02", "t_off_r02", "obs_r02")):
                t_arr = ho_entry.get(t_key, [])
                obs_arr = ho_entry.get(obs_key, [])
                runs.append(
                    {
                        "campaign_id": exp_id,
                        "replicate": rep_id,
                        "mode": run_mode,
                        "split": run_split,
                        "offline_file": f"{exp_id}_{run_mode}_{rep_id}_offline.csv",
                        "n_offline_rows": len(t_arr),
                        "duration_h": float(max(t_arr)) if t_arr else None,
                        "peak_vcd_1e6_mL": float(max(r[0] for r in obs_arr)) if obs_arr else None,
                        "max_glc_g_L": float(max(r[1] for r in obs_arr)) if obs_arr else None,
                        "peak_lac_g_L": float(max(r[2] for r in obs_arr)) if obs_arr else None,
                        "validation_wmse": wmse_val,
                        "has_heldout_trajectory": exp_id in held_out,
                    }
                )

    return {
        "source": meta.get("source", "Uhlendorff et al. (2026) CHO-K1"),
        "matched_runs_count": len(runs),
        "remediated_pooled_test_wmse": bench.get("pooled_test_wmse"),
        "remediated_ci_coverage_pct": bench.get("ci_coverage_pct"),
        "runs": runs,
    }


if __name__ == "__main__":
    mcp.run()
