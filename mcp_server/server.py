"""Least-privilege read-only & draft-staging MCP Server for hybrid-fedbatch-twin."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

from api.db import (
    create_db_engine_and_sessionmaker,
    is_decision_record_finalized,
    load_twin_session,
    save_staged_capa_draft,
)
from api.runtime import (
    ROOT_DIR,
    create_twin_instance,
    load_calibrated_assets,
    scale_feed_cap,
    scale_params,
    state_dict_from_vector,
)
from api.schemas import contains_forbidden_oracle_keys, sanitize_truth_payload
from hybrid_fedbatch_twin.controller import ClosedLoopFeedController
from hybrid_fedbatch_twin.estimator import BioreactorDigitalTwin
from hybrid_fedbatch_twin.mechanistic import (
    default_initial_state,
    default_operating_inputs,
    simulate_bioreactor,
)
from mlops.telemetry import record_ekf_step, record_mcp_privilege_rejection

UHLENDORFF_DATA_DIR = ROOT_DIR / "data" / "uhlendorff_2026" / "Time-axis aligned raw data tables"
UHLENDORFF_ARTIFACT = ROOT_DIR / "artifacts" / "uhlendorff_2026_validation.json"

mcp = MCPServer("hybrid-fedbatch-twin")

_DB_CACHE: Dict[str, Any] = {}
_SHADOW_SANDBOX: Dict[Tuple[str, str], Dict[str, Any]] = {}


class MCPPrivilegeError(PermissionError):
    """Raised when an MCP caller attempts a prohibited live, setpoint, truth, or QA-finalization mutation."""


MCP_TOOL_POLICY: Dict[str, Literal["read_only", "draft_staging"]] = {
    "simulate_fedbatch_run": "read_only",
    "ingest_pat_and_estimate_state": "read_only",
    "replay_delayed_lims_assay": "read_only",
    "search_uhlendorff_campaigns": "read_only",
    "get_twin_state_readonly": "read_only",
    "get_audit_trail_readonly": "read_only",
    "stage_capa_draft": "draft_staging",
    "list_mcp_privilege_policy": "read_only",
}

FORBIDDEN_MUTATION_OPERATIONS = frozenset(
    {
        "set_bioreactor_setpoint",
        "mutate_live_setpoint",
        "mutate_live_ekf_state",
        "overwrite_ekf_state",
        "finalize_qa_approval",
        "approve_capa",
        "modify_finalized_qa_record",
        "delete_audit_entry",
        "expose_oracle_truth",
    }
)

_FORBIDDEN_DRAFT_STATUSES = frozenset(
    {"APPROVED", "APPROVED_AND_FINALIZED", "FINALIZED", "SIGNED", "PRODUCTION", "RELEASED"}
)


def _resolve_db_url(database_url: Optional[str] = None) -> str:
    return database_url or os.environ.get("DATABASE_URL") or "sqlite:///./hybrid_twin_ledger.db"


def _get_sessionmaker(database_url: Optional[str] = None):
    url = _resolve_db_url(database_url)
    if url not in _DB_CACHE:
        engine, SessionLocal = create_db_engine_and_sessionmaker(url)
        _DB_CACHE[url] = (engine, SessionLocal)
    return _DB_CACHE[url][1]


def clear_shadow_sandbox(database_url: Optional[str] = None) -> None:
    """Clear in-memory MCP shadow clones (for isolated test runs)."""
    if database_url is None:
        _SHADOW_SANDBOX.clear()
        return
    url = _resolve_db_url(database_url)
    for k in [k for k in _SHADOW_SANDBOX if k[0] == url]:
        _SHADOW_SANDBOX.pop(k, None)


def evict_shadow_session(session_id: str, database_url: Optional[str] = None) -> None:
    """Evict a single shadow clone so the next MCP call forks cleanly from live DB."""
    url = _resolve_db_url(database_url)
    _SHADOW_SANDBOX.pop((url, session_id), None)


def enforce_mcp_least_privilege(
    operation: str,
    target_scope: str = "shadow",
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Enforce read-only / draft-staging boundaries and reject live or finalized QA mutations."""
    if operation in FORBIDDEN_MUTATION_OPERATIONS:
        record_mcp_privilege_rejection(operation, target_scope)
        raise MCPPrivilegeError(
            f"MCP least-privilege boundary blocks operation {operation!r} on scope {target_scope!r}."
        )
    if target_scope in {"live", "finalized_qa", "bioreactor_actuator"}:
        record_mcp_privilege_rejection(operation, target_scope)
        raise MCPPrivilegeError(
            f"MCP tools are read-only or draft-staging only; cannot mutate scope {target_scope!r} via {operation!r}."
        )
    if isinstance(payload, dict):
        oracle_keys = contains_forbidden_oracle_keys(payload)
        if oracle_keys:
            record_mcp_privilege_rejection("expose_oracle_truth", target_scope)
            raise MCPPrivilegeError(f"Truth isolation violation: forbidden oracle keys {oracle_keys}.")
        raw_status = str(payload.get("status", "")).strip().upper()
        if raw_status in _FORBIDDEN_DRAFT_STATUSES:
            record_mcp_privilege_rejection("finalize_qa_approval", "finalized_qa")
            raise MCPPrivilegeError(
                f"MCP draft staging cannot set status={raw_status!r}; human 2FA QA approval is required."
            )
        raw_qa_status = str(payload.get("qa_status", "")).strip().upper()
        if raw_qa_status in (_FORBIDDEN_DRAFT_STATUSES | {"REJECTED_BY_QA"}):
            record_mcp_privilege_rejection("finalize_qa_approval", "finalized_qa")
            raise MCPPrivilegeError(
                f"MCP draft staging cannot set qa_status={raw_qa_status!r}; human 2FA QA approval is required."
            )
        if payload.get("two_factor_esig") is not None or payload.get("finalized") is True:
            record_mcp_privilege_rejection("finalize_qa_approval", "finalized_qa")
            raise MCPPrivilegeError("MCP cannot attach two_factor_esig or finalize QA decision records.")
        if payload.get("apply_setpoint_live") is True or payload.get("live_setpoint_override") is not None:
            record_mcp_privilege_rejection("mutate_live_setpoint", "bioreactor_actuator")
            raise MCPPrivilegeError("MCP cannot push live setpoint mutations to bioreactor actuators.")


def _get_live_head_sha(twin: BioreactorDigitalTwin) -> str:
    return str(twin.audit_log[-1]["entry_sha256"]) if twin.audit_log else ""


def _get_or_init_shadow_twin(
    *,
    db_url: str,
    session_id: Optional[str],
    vessel_scale_L: Literal[5, 50, 1000],
    batch_id: str,
    operator_id: str,
    create_if_missing: bool,
) -> Tuple[str, BioreactorDigitalTwin, int, Optional[str]]:
    """Return an isolated in-memory twin clone for shadow MCP simulation without mutating live DB."""
    sid = session_id or f"shadow_{uuid4().hex[:12]}"
    cache_key = (db_url, sid)
    _, model = load_calibrated_assets()
    SessionLocal = _get_sessionmaker(db_url)

    live_twin: Optional[BioreactorDigitalTwin] = None
    live_scale: Optional[int] = None
    live_head_sha: Optional[str] = None
    live_record_sha: Optional[str] = None

    with SessionLocal() as db:
        loaded = load_twin_session(db, sid, model)
        if loaded is not None:
            live_twin = loaded["twin"]
            live_scale = int(loaded["vessel_scale_L"])
            live_head_sha = _get_live_head_sha(live_twin)
            live_record_sha = str(live_twin.to_audit_record()["record_sha256"])

    if live_twin is not None:
        cached = _SHADOW_SANDBOX.get(cache_key)
        if cached is not None and cached.get("live_head_sha") == live_head_sha:
            return sid, cached["twin"], int(cached["vessel_scale_L"]), live_record_sha
        # Always work on the freshly hydrated in-memory instance; never save back to live DB
        _SHADOW_SANDBOX[cache_key] = {
            "twin": live_twin,
            "vessel_scale_L": int(live_scale or vessel_scale_L),
            "live_head_sha": live_head_sha,
            "live_record_sha": live_record_sha,
        }
        return sid, live_twin, int(live_scale or vessel_scale_L), live_record_sha

    if cache_key in _SHADOW_SANDBOX:
        cached = _SHADOW_SANDBOX[cache_key]
        return sid, cached["twin"], int(cached["vessel_scale_L"]), cached.get("live_record_sha")

    if not create_if_missing:
        raise ValueError(f"Twin session '{sid}' does not exist in persistence ledger or shadow sandbox.")

    twin = create_twin_instance(
        vessel_scale_L=vessel_scale_L,
        batch_id=batch_id,
        operator_id=operator_id,
    )
    _SHADOW_SANDBOX[cache_key] = {
        "twin": twin,
        "vessel_scale_L": int(vessel_scale_L),
        "live_head_sha": None,
        "live_record_sha": None,
    }
    return sid, twin, int(vessel_scale_L), None


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
    show_truth: bool = False,
) -> Dict[str, Any]:
    """Run read-only deterministic fed-batch ODE simulation (ground_truth blocked unless show_truth=True)."""
    if mode == "ground_truth" and not show_truth:
        record_mcp_privilege_rejection("expose_oracle_truth", "oracle_simulation")
        raise MCPPrivilegeError(
            "Truth isolation violation: simulate_fedbatch_run(mode='ground_truth') requires show_truth=True."
        )

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
    out = {
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
    return sanitize_truth_payload(out, show_truth=show_truth)


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
    persist_to_live: bool = False,
) -> Dict[str, Any]:
    """Run shadow-sandbox EKF state estimation on PAT telemetry without mutating live persisted EKF state."""
    if persist_to_live:
        enforce_mcp_least_privilege("mutate_live_ekf_state", target_scope="live")

    db_url = _resolve_db_url(database_url)
    sid, twin, scale_l, live_record_sha = _get_or_init_shadow_twin(
        db_url=db_url,
        session_id=session_id,
        vessel_scale_L=vessel_scale_L,
        batch_id=batch_id,
        operator_id=operator_id,
        create_if_missing=True,
    )

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
    record_ekf_step(
        session_id=sid,
        nis=float(res["nis"]),
        spe=float(res["spe"]),
        d_m=float(res["d_m"]),
        residual_breach=bool(res["residual_breach"]),
        undecided=bool(res["undecided"]),
        gate_reason=str(res["gate_reason"]),
    )

    out = {
        "session_id": sid,
        "vessel_scale_L": scale_l,
        "sandbox_only": True,
        "live_state_mutated": False,
        "live_alcoa_record_sha256": live_record_sha,
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
    return sanitize_truth_payload(out, show_truth=False)


@mcp.tool()
def replay_delayed_lims_assay(
    session_id: str,
    t_sample_h: float,
    y_S: float,
    y_A: float,
    y_P: Optional[float] = None,
    database_url: Optional[str] = None,
    persist_to_live: bool = False,
) -> Dict[str, Any]:
    """Run shadow-sandbox retrodictive EKF replay for delayed LIMS assay without mutating live DB state."""
    if persist_to_live:
        enforce_mcp_least_privilege("mutate_live_ekf_state", target_scope="live")

    db_url = _resolve_db_url(database_url)
    sid, twin, _, live_record_sha = _get_or_init_shadow_twin(
        db_url=db_url,
        session_id=session_id,
        vessel_scale_L=5,
        batch_id="SHADOW",
        operator_id="mcp_agent",
        create_if_missing=False,
    )

    replay = twin.ingest_atline_assay(
        t_sample_h=t_sample_h,
        y_S=y_S,
        y_A=y_A,
        y_P=y_P,
    )
    record_ekf_step(
        session_id=sid,
        nis=float(replay["nis"]),
        spe=float(replay["spe"]),
        d_m=float(replay["d_m"]),
        residual_breach=bool(replay["residual_breach"]),
        undecided=bool(replay["undecided"]),
        gate_reason=str(replay["gate_reason"]),
    )
    audit = twin.to_audit_record()

    out = {
        "session_id": sid,
        "sandbox_only": True,
        "live_state_mutated": False,
        "live_alcoa_record_sha256": live_record_sha,
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
    return sanitize_truth_payload(out, show_truth=False)


@mcp.tool()
def get_twin_state_readonly(
    session_id: str,
    database_url: Optional[str] = None,
    show_truth: bool = False,
) -> Dict[str, Any]:
    """Read-only inspection of live persisted EKF twin state, gate status, and ALCOA+ head digest."""
    if show_truth:
        record_mcp_privilege_rejection("expose_oracle_truth", "live")
        raise MCPPrivilegeError("MCP get_twin_state_readonly forbids show_truth=True.")

    SessionLocal = _get_sessionmaker(database_url)
    _, model = load_calibrated_assets()
    with SessionLocal() as db:
        loaded = load_twin_session(db, session_id, model)
    if loaded is None:
        raise ValueError(f"Twin session '{session_id}' not found.")

    twin: BioreactorDigitalTwin = loaded["twin"]
    rates = twin.ekf.current_rates(twin.last_u)
    gate = twin.ekf.evaluate_gate(rates=rates)
    x_std = twin.ekf.P.diagonal().clip(min=0.0) ** 0.5
    audit_rec = twin.to_audit_record(max_entries=1)
    out = {
        "session_id": session_id,
        "batch_id": twin.batch_id,
        "operator_id": twin.operator_id,
        "vessel_scale_L": int(loaded["vessel_scale_L"]),
        "t_now_h": float(twin.t_now),
        "current_state": state_dict_from_vector(twin.ekf.x),
        "current_std": state_dict_from_vector(x_std),
        "d_m": float(gate["d_m"]),
        "nis": float(gate["nis"]),
        "spe": float(gate["spe"]),
        "in_domain": bool(gate["in_domain"]),
        "undecided": bool(gate["undecided"]),
        "residual_breach": bool(gate["residual_breach"]),
        "gate_reason": str(gate["gate_reason"]),
        "chain_head_sha256": str(audit_rec["chain_head_sha256"]),
        "alcoa_record_sha256": str(audit_rec["record_sha256"]),
        "audit_log_count": int(audit_rec["audit_log_count"]),
        "read_only": True,
    }
    return sanitize_truth_payload(out, show_truth=False)


@mcp.tool()
def get_audit_trail_readonly(
    session_id: str,
    max_entries: Optional[int] = None,
    database_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Read-only export and cryptographic verification of live persisted ALCOA+ audit entries."""
    SessionLocal = _get_sessionmaker(database_url)
    _, model = load_calibrated_assets()
    with SessionLocal() as db:
        loaded = load_twin_session(db, session_id, model)
    if loaded is None:
        raise ValueError(f"Twin session '{session_id}' not found.")

    twin: BioreactorDigitalTwin = loaded["twin"]
    rec = twin.to_audit_record(max_entries=max_entries)
    ok_rec, errs = BioreactorDigitalTwin.verify_audit_record(rec)
    out = {
        "session_id": session_id,
        "verified": bool(ok_rec),
        "verification_errors": errs,
        **rec,
    }
    return sanitize_truth_payload(out, show_truth=False)


@mcp.tool()
def stage_capa_draft(
    session_id: str,
    draft_report: Dict[str, Any],
    author_id: str = "mcp_agent",
    target_decision_record_id: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage a pending CAPA draft for human QA review; rejects QA finalization or live mutations."""
    enforce_mcp_least_privilege("stage_capa_draft", target_scope="draft_staging", payload=draft_report)
    if not isinstance(draft_report, dict) or not draft_report:
        raise ValueError("draft_report must be a non-empty JSON object.")

    sanitized_draft = sanitize_truth_payload(dict(draft_report), show_truth=False)
    draft_id = f"DRAFT-{uuid4().hex[:12]}"
    draft_sha256 = hashlib.sha256(
        json.dumps(sanitized_draft, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    SessionLocal = _get_sessionmaker(database_url)
    with SessionLocal() as db:
        if target_decision_record_id and is_decision_record_finalized(db, target_decision_record_id):
            record_mcp_privilege_rejection("modify_finalized_qa_record", "finalized_qa")
            raise MCPPrivilegeError(
                f"Cannot modify or overwrite finalized Part 11 QA record '{target_decision_record_id}'."
            )
        staged = save_staged_capa_draft(
            db,
            draft_id=draft_id,
            session_id=session_id,
            author_id=author_id,
            draft_payload=sanitized_draft,
            draft_sha256=draft_sha256,
            target_decision_record_id=target_decision_record_id,
        )

    return {
        **staged,
        "live_state_mutated": False,
        "qa_finalized": False,
    }


@mcp.tool()
def list_mcp_privilege_policy() -> Dict[str, Any]:
    """Return declared least-privilege scopes for all MCP tools and blocked mutation operations."""
    return {
        "server_name": "hybrid-fedbatch-twin",
        "allowed_scopes": ["read_only", "draft_staging"],
        "tool_policies": dict(MCP_TOOL_POLICY),
        "forbidden_operations": sorted(FORBIDDEN_MUTATION_OPERATIONS),
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
