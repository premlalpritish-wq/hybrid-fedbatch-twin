"""Structured observability tracing and deterministic faithfulness evaluation for CAPA agent runs."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from api.db import create_db_engine_and_sessionmaker, load_twin_session
from api.runtime import load_calibrated_assets


class ToolCallSpan(BaseModel):
    tool_name: str
    node_name: str
    arguments: Dict[str, Any]
    output: Dict[str, Any]
    duration_ms: float
    status: str = "ok"
    error: Optional[str] = None


class NodeExecutionSpan(BaseModel):
    node_name: str
    started_utc: str
    completed_utc: str
    duration_ms: float
    state_keys_updated: List[str]


class ExecutionTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    session_id: str
    started_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_duration_ms: float = 0.0
    execution_mode: str = "deterministic"
    node_spans: List[NodeExecutionSpan] = Field(default_factory=list)
    tool_spans: List[ToolCallSpan] = Field(default_factory=list)


class EvalCheckResult(BaseModel):
    name: str
    passed: bool
    score: float
    detail: str


class CapaEvaluationReport(BaseModel):
    overall_passed: bool
    overall_score: float
    tool_trajectory_completeness: EvalCheckResult
    numerical_grounding_faithfulness: EvalCheckResult
    alcoa_traceability: EvalCheckResult


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def evaluate_capa_report(
    report: Any,
    trace: ExecutionTrace,
    database_url: Optional[str] = None,
) -> CapaEvaluationReport:
    """Deterministically score CAPA report on tool completeness, numerical grounding, and ALCOA+ linkage."""
    # 1. Tool-Trajectory Completeness
    called_tools = [s.tool_name for s in trace.tool_spans if s.status == "ok"]
    has_state_tool = any(
        t in called_tools for t in ("ingest_pat_and_estimate_state", "replay_delayed_lims_assay")
    )
    has_sim_hyb = any(
        s.tool_name == "simulate_fedbatch_run" and s.arguments.get("mode") == "hybrid"
        for s in trace.tool_spans
        if s.status == "ok"
    )
    has_sim_nom = any(
        s.tool_name == "simulate_fedbatch_run" and s.arguments.get("mode") == "nominal"
        for s in trace.tool_spans
        if s.status == "ok"
    )
    has_uhlendorff = "search_uhlendorff_campaigns" in called_tools

    checks_1 = [has_state_tool, has_sim_hyb, has_sim_nom, has_uhlendorff]
    score_1 = sum(1 for c in checks_1 if c) / len(checks_1)
    missing_tools = []
    if not has_state_tool:
        missing_tools.append("ingest_pat_and_estimate_state|replay_delayed_lims_assay")
    if not has_sim_hyb:
        missing_tools.append("simulate_fedbatch_run(hybrid)")
    if not has_sim_nom:
        missing_tools.append("simulate_fedbatch_run(nominal)")
    if not has_uhlendorff:
        missing_tools.append("search_uhlendorff_campaigns")

    traj_eval = EvalCheckResult(
        name="tool_trajectory_completeness",
        passed=all(checks_1),
        score=score_1,
        detail="All required MCP physics/EKF/empirical tools executed."
        if all(checks_1)
        else f"Missing required tool calls: {', '.join(missing_tools)}",
    )

    # 2. Numerical Grounding / Zero-Hallucination Faithfulness
    grounding_failures: List[str] = []
    grounding_checks = 0
    grounding_passed = 0

    def _check_close(label: str, cited: float, observed: float, tol: float = 1e-6) -> None:
        nonlocal grounding_checks, grounding_passed
        grounding_checks += 1
        if abs(float(cited) - float(observed)) <= tol:
            grounding_passed += 1
        else:
            grounding_failures.append(f"{label}: cited={cited} != observed={observed}")

    # Match d_m & in_domain against latest state tool output
    state_spans = [
        s
        for s in trace.tool_spans
        if s.tool_name in ("replay_delayed_lims_assay", "ingest_pat_and_estimate_state")
        and s.status == "ok"
    ]
    if state_spans:
        latest_state = state_spans[-1].output
        _check_close("d_m", report.d_m, latest_state["d_m"])
        grounding_checks += 1
        if bool(report.in_domain) == bool(latest_state["in_domain"]):
            grounding_passed += 1
        else:
            grounding_failures.append(
                f"in_domain: cited={report.in_domain} != observed={latest_state['in_domain']}"
            )
    else:
        grounding_checks += 1
        grounding_failures.append("No state estimation tool span to ground d_m / in_domain.")

    # Match counterfactual simulation numbers
    hyb_spans = [
        s for s in trace.tool_spans if s.tool_name == "simulate_fedbatch_run" and s.arguments.get("mode") == "hybrid"
    ]
    nom_spans = [
        s for s in trace.tool_spans if s.tool_name == "simulate_fedbatch_run" and s.arguments.get("mode") == "nominal"
    ]
    if hyb_spans and nom_spans:
        hyb_out = hyb_spans[-1].output
        nom_out = nom_spans[-1].output
        cf = report.counterfactual_comparison
        _check_close("final_P_hybrid_g_L", cf.final_P_hybrid_g_L, hyb_out["final_state"]["P_g_L"])
        _check_close("final_P_nominal_g_L", cf.final_P_nominal_g_L, nom_out["final_state"]["P_g_L"])
        _check_close(
            "titer_delta_g_L",
            cf.titer_delta_g_L,
            hyb_out["final_state"]["P_g_L"] - nom_out["final_state"]["P_g_L"],
        )
        _check_close("min_do_hybrid_pct", cf.min_do_hybrid_pct, hyb_out["min_DO_pct"])
        _check_close("min_do_nominal_pct", cf.min_do_nominal_pct, nom_out["min_DO_pct"])
        _check_close("peak_acetate_hybrid_g_L", cf.peak_acetate_hybrid_g_L, hyb_out["peak_acetate_g_L"])
        _check_close("peak_acetate_nominal_g_L", cf.peak_acetate_nominal_g_L, nom_out["peak_acetate_g_L"])
    else:
        grounding_checks += 1
        grounding_failures.append("Missing counterfactual simulation spans.")

    # Match cited Uhlendorff campaigns
    uhl_spans = [s for s in trace.tool_spans if s.tool_name == "search_uhlendorff_campaigns"]
    if uhl_spans:
        observed_campaigns = {r["campaign_id"] for r in uhl_spans[-1].output.get("runs", [])}
        cited_set = set(report.cited_uhlendorff_campaigns)
        grounding_checks += 1
        if cited_set and cited_set.issubset(observed_campaigns):
            grounding_passed += 1
        else:
            grounding_failures.append(
                f"cited_uhlendorff_campaigns={sorted(cited_set)} not subset of {sorted(observed_campaigns)}"
            )
    else:
        grounding_checks += 1
        grounding_failures.append("No Uhlendorff campaign span.")

    score_2 = grounding_passed / max(1, grounding_checks)
    grounding_eval = EvalCheckResult(
        name="numerical_grounding_faithfulness",
        passed=len(grounding_failures) == 0,
        score=score_2,
        detail="All cited EKF, counterfactual, and empirical metrics match MCP tool outputs."
        if not grounding_failures
        else "; ".join(grounding_failures),
    )

    # 3. ALCOA+ Traceability
    alcoa_ok = False
    alcoa_detail = ""
    if not _SHA256_RE.match(report.alcoa_record_sha256):
        alcoa_detail = f"Invalid SHA-256 format: {report.alcoa_record_sha256}"
    else:
        engine, SessionLocal = create_db_engine_and_sessionmaker(database_url)
        try:
            _, model = load_calibrated_assets()
            with SessionLocal() as db:
                loaded = load_twin_session(db, report.session_id, model)
        finally:
            engine.dispose()
        if loaded is None:
            alcoa_detail = f"Session '{report.session_id}' not found in persistence ledger."
        else:
            ledger_record = loaded["twin"].to_audit_record()
            if ledger_record["record_sha256"] == report.alcoa_record_sha256:
                alcoa_ok = True
                alcoa_detail = (
                    f"Verified SHA-256 chain linkage ({report.alcoa_record_sha256[:12]}...) "
                    f"across {ledger_record['audit_log_count']} ledger entries."
                )
            else:
                alcoa_detail = (
                    f"SHA-256 mismatch: report={report.alcoa_record_sha256[:12]} "
                    f"vs ledger={ledger_record['record_sha256'][:12]}"
                )

    alcoa_eval = EvalCheckResult(
        name="alcoa_traceability",
        passed=alcoa_ok,
        score=1.0 if alcoa_ok else 0.0,
        detail=alcoa_detail,
    )

    overall_score = round((traj_eval.score + grounding_eval.score + alcoa_eval.score) / 3.0, 4)
    guardrail_ok = bool(getattr(report, "guardrail_passed", True))
    overall_passed = traj_eval.passed and grounding_eval.passed and alcoa_eval.passed and guardrail_ok

    return CapaEvaluationReport(
        overall_passed=overall_passed,
        overall_score=overall_score,
        tool_trajectory_completeness=traj_eval,
        numerical_grounding_faithfulness=grounding_eval,
        alcoa_traceability=alcoa_eval,
    )
