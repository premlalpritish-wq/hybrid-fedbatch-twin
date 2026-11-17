"""Stateful Bioprocess Deviation & CAPA Investigator graph orchestrating MCP physics tools."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field

from api.db import create_db_engine_and_sessionmaker, load_twin_session
from api.runtime import load_calibrated_assets
from mcp_server.server import (
    ingest_pat_and_estimate_state,
    replay_delayed_lims_assay,
    search_uhlendorff_campaigns,
    simulate_fedbatch_run,
)

from .observability import ExecutionTrace, NodeExecutionSpan, ToolCallSpan

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"


class CounterfactualComparison(BaseModel):
    final_P_hybrid_g_L: float
    final_P_nominal_g_L: float
    titer_delta_g_L: float
    min_do_hybrid_pct: float
    min_do_nominal_pct: float
    peak_acetate_hybrid_g_L: float
    peak_acetate_nominal_g_L: float


class DeviationCapaReport(BaseModel):
    session_id: str
    vessel_scale_L: int
    root_cause_classification: Literal[
        "IN_DOMAIN_NOMINAL",
        "OOD_EXCURSION",
        "HYPOXIA_RISK",
        "ACETATE_OVERFLOW_RISK",
        "COMBINED_PROCESS_DEVIATION",
    ]
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    d_m: float
    in_domain: bool
    recommended_action: str
    counterfactual_comparison: CounterfactualComparison
    cited_uhlendorff_campaigns: List[str] = Field(default_factory=list)
    remediated_pooled_test_wmse: Optional[float] = None
    alcoa_record_sha256: str
    alcoa_audit_log_count: int
    execution_mode: Literal["deterministic", "llm_assisted"] = "deterministic"
    narrative_summary: str


class DeviationInvestigationState(TypedDict, total=False):
    session_id: str
    vessel_scale_L: Literal[5, 50, 1000]
    database_url: Optional[str]
    pat_tick: Optional[Dict[str, Any]]
    delayed_lims: Optional[Dict[str, Any]]
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"]
    duration_h: float
    mu_target: float
    use_llm: bool
    excursion_result: Dict[str, Any]
    lims_replay_result: Optional[Dict[str, Any]]
    alcoa_record: Dict[str, Any]
    counterfactual_hybrid: Dict[str, Any]
    counterfactual_nominal: Dict[str, Any]
    empirical_precedents: Dict[str, Any]
    capa_report: DeviationCapaReport
    trace: ExecutionTrace


class _FallbackCompiledGraph:
    """Minimal typed state-machine runner compatible with LangGraph node/edge semantics."""

    def __init__(self, nodes: Dict[str, Callable[[DeviationInvestigationState], Dict[str, Any]]], order: List[str]):
        self._nodes = nodes
        self._order = order

    def invoke(self, initial_state: DeviationInvestigationState) -> DeviationInvestigationState:
        state: DeviationInvestigationState = dict(initial_state)
        for name in self._order:
            updates = self._nodes[name](state)
            state.update(updates)
        return state


def _invoke_mcp_with_trace(
    trace: ExecutionTrace,
    node_name: str,
    tool_fn: Callable[..., Dict[str, Any]],
    tool_name: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out = tool_fn(**kwargs)
        dur = (time.perf_counter() - t0) * 1000.0
        trace.tool_spans.append(
            ToolCallSpan(
                tool_name=tool_name,
                node_name=node_name,
                arguments={k: v for k, v in kwargs.items() if k != "database_url"},
                output=out,
                duration_ms=round(dur, 3),
                status="ok",
            )
        )
        return out
    except Exception as exc:
        dur = (time.perf_counter() - t0) * 1000.0
        trace.tool_spans.append(
            ToolCallSpan(
                tool_name=tool_name,
                node_name=node_name,
                arguments={k: v for k, v in kwargs.items() if k != "database_url"},
                output={},
                duration_ms=round(dur, 3),
                status="error",
                error=str(exc),
            )
        )
        raise


def _run_node(
    node_name: str,
    state: DeviationInvestigationState,
    fn: Callable[[DeviationInvestigationState, ExecutionTrace], Dict[str, Any]],
) -> Dict[str, Any]:
    trace = state["trace"]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    updates = fn(state, trace)
    dur = (time.perf_counter() - t0) * 1000.0
    trace.node_spans.append(
        NodeExecutionSpan(
            node_name=node_name,
            started_utc=started,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(dur, 3),
            state_keys_updated=sorted(updates.keys()),
        )
    )
    return updates


def _detect_excursion_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        sid = st["session_id"]
        db_url = st.get("database_url")
        scale_l = st.get("vessel_scale_L", 5)
        pat_tick = st.get("pat_tick")
        delayed_lims = st.get("delayed_lims")

        excursion_out: Dict[str, Any] = {}
        lims_out: Optional[Dict[str, Any]] = None
        alcoa_rec: Dict[str, Any] = {}

        if pat_tick is not None:
            pat_args = {
                k: pat_tick[k]
                for k in ("t_h", "y_Xv", "y_CL", "y_V", "F_act", "rpm_act", "vvm_act")
                if k in pat_tick
            }
            excursion_out = _invoke_mcp_with_trace(
                trace,
                "detect_excursion",
                ingest_pat_and_estimate_state,
                "ingest_pat_and_estimate_state",
                {
                    "session_id": sid,
                    "vessel_scale_L": scale_l,
                    "database_url": db_url,
                    **pat_args,
                },
            )

        if delayed_lims is not None:
            lims_args = {
                k: delayed_lims[k]
                for k in ("t_sample_h", "y_S", "y_A", "y_P")
                if k in delayed_lims
            }
            lims_out = _invoke_mcp_with_trace(
                trace,
                "detect_excursion",
                replay_delayed_lims_assay,
                "replay_delayed_lims_assay",
                {
                    "session_id": sid,
                    "database_url": db_url,
                    **lims_args,
                },
            )
            alcoa_rec = lims_out["alcoa_record"]
            excursion_out = lims_out

        # If no PAT tick was passed, still inspect current persisted twin state & ALCOA+ chain
        if not excursion_out or not alcoa_rec:
            _, SessionLocal = create_db_engine_and_sessionmaker(db_url)
            _, model = load_calibrated_assets()
            with SessionLocal() as db:
                loaded = load_twin_session(db, sid, model)
            if loaded is None:
                raise ValueError(f"Twin session '{sid}' not found and no pat_tick provided.")
            twin = loaded["twin"]
            if not alcoa_rec:
                alcoa_rec = twin.to_audit_record()
            if not excursion_out:
                # Re-ingest current state via EKF snapshot or latest PAT
                rates = twin.ekf.current_rates(twin.last_u)
                excursion_out = _invoke_mcp_with_trace(
                    trace,
                    "detect_excursion",
                    ingest_pat_and_estimate_state,
                    "ingest_pat_and_estimate_state",
                    {
                        "session_id": sid,
                        "vessel_scale_L": loaded["vessel_scale_L"],
                        "t_h": float(twin.t_now),
                        "y_Xv": float(twin.ekf.x[0]),
                        "y_CL": float(twin.ekf.x[3]),
                        "y_V": float(twin.ekf.x[6]),
                        "F_act": float(twin.last_u.F),
                        "rpm_act": float(twin.last_u.rpm),
                        "vvm_act": float(twin.last_u.vvm),
                        "database_url": db_url,
                    },
                )
                # Refresh ALCOA+ record after PAT ingestion
                with SessionLocal() as db:
                    reloaded = load_twin_session(db, sid, model)
                    if reloaded is not None:
                        alcoa_rec = reloaded["twin"].to_audit_record()

        return {
            "excursion_result": excursion_out,
            "lims_replay_result": lims_out,
            "alcoa_record": alcoa_rec,
        }

    return _run_node("detect_excursion", state, _impl)


def _run_counterfactual_controls_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        scale_l = st.get("vessel_scale_L", 5)
        duration_h = float(st.get("duration_h", 12.0))
        mu_target = float(st.get("mu_target", 0.28))
        dist_mode = st.get("disturbance_mode", "none")

        hyb = _invoke_mcp_with_trace(
            trace,
            "run_counterfactual_controls",
            simulate_fedbatch_run,
            "simulate_fedbatch_run",
            {
                "vessel_scale_L": scale_l,
                "duration_h": duration_h,
                "mode": "hybrid",
                "mu_target": mu_target,
                "disturbance_mode": dist_mode,
                "dt_eval": 0.25,
            },
        )
        nom = _invoke_mcp_with_trace(
            trace,
            "run_counterfactual_controls",
            simulate_fedbatch_run,
            "simulate_fedbatch_run",
            {
                "vessel_scale_L": scale_l,
                "duration_h": duration_h,
                "mode": "nominal",
                "mu_target": mu_target,
                "disturbance_mode": dist_mode,
                "dt_eval": 0.25,
            },
        )
        return {"counterfactual_hybrid": hyb, "counterfactual_nominal": nom}

    return _run_node("run_counterfactual_controls", state, _impl)


def _retrieve_empirical_precedent_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        precedents = _invoke_mcp_with_trace(
            trace,
            "retrieve_empirical_precedent",
            search_uhlendorff_campaigns,
            "search_uhlendorff_campaigns",
            {"mode": "fed-batch"},
        )
        return {"empirical_precedents": precedents}

    return _run_node("retrieve_empirical_precedent", state, _impl)


def _synthesize_capa_report_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        exc = st["excursion_result"]
        hyb = st["counterfactual_hybrid"]
        nom = st["counterfactual_nominal"]
        emp = st["empirical_precedents"]
        alcoa = st["alcoa_record"]

        d_m = float(exc["d_m"])
        in_domain = bool(exc["in_domain"])
        x_hat = exc["x_hat"]
        cl_now = float(x_hat.get("C_L", 0.0))
        ace_now = float(x_hat.get("A", 0.0))

        hypoxia_risk = cl_now < 1.64 or float(nom["min_DO_pct"]) < 20.0
        acetate_risk = ace_now > 1.5 or float(nom["peak_acetate_g_L"]) > 1.8

        if not in_domain and (hypoxia_risk or acetate_risk):
            root_cause = "COMBINED_PROCESS_DEVIATION"
            severity = "CRITICAL"
        elif not in_domain:
            root_cause = "OOD_EXCURSION"
            severity = "CRITICAL"
        elif hypoxia_risk and acetate_risk:
            root_cause = "COMBINED_PROCESS_DEVIATION"
            severity = "WARNING"
        elif hypoxia_risk:
            root_cause = "HYPOXIA_RISK"
            severity = "WARNING"
        elif acetate_risk:
            root_cause = "ACETATE_OVERFLOW_RISK"
            severity = "WARNING"
        else:
            root_cause = "IN_DOMAIN_NOMINAL"
            severity = "INFO"

        p_hyb = float(hyb["final_state"]["P_g_L"])
        p_nom = float(nom["final_state"]["P_g_L"])
        titer_delta = p_hyb - p_nom

        if root_cause == "IN_DOMAIN_NOMINAL":
            action = (
                "Maintain ClosedLoopFeedController (mu-stat backoff active); "
                "no manual feed intervention required."
            )
        else:
            action = (
                "Engage ClosedLoopFeedController with oxygen-transfer ceiling and acetate overflow backoff; "
                "verify dissolved oxygen probe and schedule confirmatory at-line LIMS glucose/acetate assay."
            )

        cited_campaigns = sorted({r["campaign_id"] for r in emp.get("runs", [])})
        wmse = emp.get("remediated_pooled_test_wmse")

        narrative = (
            f"[{severity}] {root_cause} at {st.get('vessel_scale_L', 5)} L scale "
            f"(Mahalanobis D_M={d_m:.3f}, in_domain={in_domain}). "
            f"Counterfactual closed-loop hybrid control projects P={p_hyb:.2f} g/L vs nominal P={p_nom:.2f} g/L "
            f"(delta={titer_delta:+.2f} g/L, min_DO={hyb['min_DO_pct']:.1f}% vs {nom['min_DO_pct']:.1f}%). "
            f"Grounded in Uhlendorff et al. (2026) fed-batch campaigns {', '.join(cited_campaigns)} "
            f"and ALCOA+ SHA-256 {alcoa['record_sha256'][:12]}."
        )

        exec_mode: Literal["deterministic", "llm_assisted"] = "deterministic"
        if st.get("use_llm") and (os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            exec_mode = "llm_assisted"
        trace.execution_mode = exec_mode

        report = DeviationCapaReport(
            session_id=st["session_id"],
            vessel_scale_L=int(st.get("vessel_scale_L", 5)),
            root_cause_classification=root_cause,
            severity=severity,
            d_m=d_m,
            in_domain=in_domain,
            recommended_action=action,
            counterfactual_comparison=CounterfactualComparison(
                final_P_hybrid_g_L=p_hyb,
                final_P_nominal_g_L=p_nom,
                titer_delta_g_L=titer_delta,
                min_do_hybrid_pct=float(hyb["min_DO_pct"]),
                min_do_nominal_pct=float(nom["min_DO_pct"]),
                peak_acetate_hybrid_g_L=float(hyb["peak_acetate_g_L"]),
                peak_acetate_nominal_g_L=float(nom["peak_acetate_g_L"]),
            ),
            cited_uhlendorff_campaigns=cited_campaigns,
            remediated_pooled_test_wmse=None if wmse is None else float(wmse),
            alcoa_record_sha256=str(alcoa["record_sha256"]),
            alcoa_audit_log_count=int(alcoa["audit_log_count"]),
            execution_mode=exec_mode,
            narrative_summary=narrative,
        )
        return {"capa_report": report}

    return _run_node("synthesize_capa_report", state, _impl)


def build_deviation_graph():
    """Build and compile the 4-step Deviation Investigation state graph."""
    nodes = {
        "detect_excursion": _detect_excursion_node,
        "run_counterfactual_controls": _run_counterfactual_controls_node,
        "retrieve_empirical_precedent": _retrieve_empirical_precedent_node,
        "synthesize_capa_report": _synthesize_capa_report_node,
    }
    order = [
        "detect_excursion",
        "run_counterfactual_controls",
        "retrieve_empirical_precedent",
        "synthesize_capa_report",
    ]
    if _HAS_LANGGRAPH:
        graph = StateGraph(DeviationInvestigationState)
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(START, order[0])
        for a, b in zip(order[:-1], order[1:]):
            graph.add_edge(a, b)
        graph.add_edge(order[-1], END)
        return graph.compile()
    return _FallbackCompiledGraph(nodes, order)


def run_deviation_investigation(
    session_id: str,
    vessel_scale_L: Literal[5, 50, 1000] = 5,
    database_url: Optional[str] = None,
    pat_tick: Optional[Dict[str, Any]] = None,
    delayed_lims: Optional[Dict[str, Any]] = None,
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none",
    duration_h: float = 12.0,
    mu_target: float = 0.28,
    use_llm: bool = False,
) -> Tuple[DeviationCapaReport, ExecutionTrace]:
    """Execute the full CAPA Deviation Investigation graph and return (report, trace)."""
    trace = ExecutionTrace(session_id=session_id)
    t0 = time.perf_counter()
    compiled = build_deviation_graph()
    final_state = compiled.invoke(
        {
            "session_id": session_id,
            "vessel_scale_L": vessel_scale_L,
            "database_url": database_url,
            "pat_tick": pat_tick,
            "delayed_lims": delayed_lims,
            "disturbance_mode": disturbance_mode,
            "duration_h": duration_h,
            "mu_target": mu_target,
            "use_llm": use_llm,
            "trace": trace,
        }
    )
    trace.total_duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return final_state["capa_report"], trace
