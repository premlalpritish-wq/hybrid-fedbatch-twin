"""Stateful Bioprocess Deviation & CAPA Investigator LangGraph workflow with deterministic guardrails."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, TypedDict

from pydantic import BaseModel, Field

from api.db import (
    create_db_engine_and_sessionmaker,
    get_latest_ai_decision_hash,
    load_twin_session,
    save_ai_decision_record,
    save_server_evidence_context,
)
from api.runtime import load_calibrated_assets
from mcp_server.server import (
    evict_shadow_session,
    get_audit_trail_readonly,
    get_twin_state_readonly,
    ingest_pat_and_estimate_state,
    replay_delayed_lims_assay,
    search_uhlendorff_campaigns,
    simulate_fedbatch_run,
    stage_capa_draft,
)
from mlops.telemetry import record_agent_guardrail_rejection, record_agent_investigation

from .guardrails import (
    GroundedClaim,
    GuardrailResult,
    build_evidence_context,
    validate_capa_draft,
)
from .observability import ExecutionTrace, NodeExecutionSpan, ToolCallSpan
from .part11 import (
    LlmExecutionMetadata,
    Part11AiDecisionRecord,
    Part1RawModelOutput,
    build_part11_decision_record,
    build_part1_raw_output,
    canonical_sha256,
)

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"


PROMPT_TEMPLATE_VERSION = "bioprocess-capa-v2.0"
PROMPT_TEMPLATE_TEXT = (
    "Synthesize a grounded bioprocess deviation CAPA draft solely from verified EKF telemetry, "
    "counterfactual hybrid vs nominal trajectories, empirical Uhlendorff CHO-K1 precedents, and ALCOA+ hashes."
)
PROMPT_SHA256 = canonical_sha256(
    {"version": PROMPT_TEMPLATE_VERSION, "template": PROMPT_TEMPLATE_TEXT}
)


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
    nis: float = 0.0
    spe: float = 0.0
    recommended_action: str
    counterfactual_comparison: CounterfactualComparison
    cited_uhlendorff_campaigns: List[str] = Field(default_factory=list)
    remediated_pooled_test_wmse: Optional[float] = None
    alcoa_record_sha256: str
    alcoa_audit_log_count: int
    execution_mode: Literal["deterministic", "llm_assisted"] = "deterministic"
    narrative_summary: str
    claims: List[GroundedClaim] = Field(default_factory=list)
    guardrail_passed: bool = True
    guardrail_violations: List[str] = Field(default_factory=list)
    qa_status: Literal[
        "PENDING_QA_APPROVAL",
        "APPROVED_AND_FINALIZED",
        "REJECTED_BY_QA",
        "GUARDRAIL_REJECTED",
    ] = "PENDING_QA_APPROVAL"
    context_sha256: str = ""
    llm_execution_metadata: Optional[Dict[str, Any]] = None


class CapaWorkflowOutcome(BaseModel):
    """Complete outcome of the 5-step CAPA workflow including guardrail & Part 11 artifacts."""

    capa_report: DeviationCapaReport
    trace: ExecutionTrace
    evidence_context: Dict[str, Any]
    guardrail_result: GuardrailResult
    part1_raw_output: Part1RawModelOutput
    staged_draft: Optional[Dict[str, Any]] = None
    part11_record: Optional[Part11AiDecisionRecord] = None


class CapaDraftProvider(Protocol):
    """Protocol for deterministic CI or LLM-backed CAPA draft generators."""

    def draft_capa(
        self,
        state: "DeviationInvestigationState",
        evidence_context: Dict[str, Any],
    ) -> Tuple[DeviationCapaReport, LlmExecutionMetadata]:
        ...


class DeterministicCiCapaProvider:
    """Zero-secret deterministic CI provider synthesizing grounded CAPA drafts from evidence."""

    def __init__(self, provider_name: str = "deterministic_ci_provider") -> None:
        self.provider_name = provider_name

    def draft_capa(
        self,
        state: "DeviationInvestigationState",
        evidence_context: Dict[str, Any],
    ) -> Tuple[DeviationCapaReport, LlmExecutionMetadata]:
        t0 = time.perf_counter()
        exc = state["excursion_result"]
        hyb = state["counterfactual_hybrid"]
        nom = state["counterfactual_nominal"]
        emp = state["empirical_precedents"]
        alcoa = state["alcoa_record"]

        d_m = float(exc["d_m"])
        nis = float(exc.get("nis", 0.0))
        spe = float(exc.get("spe", 0.0))
        in_domain = bool(exc["in_domain"])
        x_hat = exc["x_hat"]
        cl_now = float(x_hat.get("C_L", 0.0))
        ace_now = float(x_hat.get("A", 0.0))

        hypoxia_risk = cl_now < 1.64 or float(nom["min_DO_pct"]) < 20.0
        acetate_risk = ace_now > 1.5 or float(nom["peak_acetate_g_L"]) > 1.8

        if not in_domain and (hypoxia_risk or acetate_risk):
            root_cause: Literal[
                "IN_DOMAIN_NOMINAL",
                "OOD_EXCURSION",
                "HYPOXIA_RISK",
                "ACETATE_OVERFLOW_RISK",
                "COMBINED_PROCESS_DEVIATION",
            ] = "COMBINED_PROCESS_DEVIATION"
            severity: Literal["INFO", "WARNING", "CRITICAL"] = "CRITICAL"
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
        min_do_hyb = float(hyb["min_DO_pct"])
        min_do_nom = float(nom["min_DO_pct"])
        peak_ace_hyb = float(hyb["peak_acetate_g_L"])
        peak_ace_nom = float(nom["peak_acetate_g_L"])

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

        cited_campaigns = sorted({str(r["campaign_id"]) for r in emp.get("runs", []) if "campaign_id" in r})
        wmse = emp.get("remediated_pooled_test_wmse")
        alcoa_sha = str(alcoa["record_sha256"])

        narrative = (
            f"[{severity}] {root_cause} at {state.get('vessel_scale_L', 5)} L scale "
            f"(Mahalanobis D_M={d_m:.3f}, NIS={nis:.2f}, SPE={spe:.2f}, in_domain={in_domain}). "
            f"Counterfactual closed-loop hybrid control projects P={p_hyb:.2f} g/L vs nominal P={p_nom:.2f} g/L "
            f"(delta={titer_delta:+.2f} g/L, min_DO={min_do_hyb:.1f}% vs {min_do_nom:.1f}%). "
            f"Grounded in Uhlendorff et al. (2026) fed-batch campaigns {', '.join(cited_campaigns)} "
            f"and ALCOA+ SHA-256 {alcoa_sha[:12]}."
        )

        claims: List[GroundedClaim] = [
            GroundedClaim(
                claim_id="CLM-EKF-DM",
                statement=f"Mahalanobis DoV distance D_M={d_m:.4f} (in_domain={in_domain}).",
                citation_ref="ekf:gate_status",
                metric_name="d_m",
                cited_value=d_m,
                unit="dimensionless",
            ),
            GroundedClaim(
                claim_id="CLM-EKF-NIS",
                statement=f"EKF Normalized Innovation Squared NIS={nis:.4f}.",
                citation_ref="ekf:gate_status",
                metric_name="nis",
                cited_value=nis,
                unit="chi2",
            ),
            GroundedClaim(
                claim_id="CLM-EKF-SPE",
                statement=f"EKF Squared Prediction Error SPE={spe:.4f}.",
                citation_ref="ekf:gate_status",
                metric_name="spe",
                cited_value=spe,
                unit="chi2",
            ),
            GroundedClaim(
                claim_id="CLM-CF-HYB-P",
                statement=f"Counterfactual hybrid closed-loop projects final product titer {p_hyb:.4f} g/L.",
                citation_ref="counterfactual:hybrid",
                metric_name="final_P_hybrid_g_L",
                cited_value=p_hyb,
                unit="g/L",
            ),
            GroundedClaim(
                claim_id="CLM-CF-NOM-P",
                statement=f"Counterfactual nominal closed-loop projects final product titer {p_nom:.4f} g/L.",
                citation_ref="counterfactual:nominal",
                metric_name="final_P_nominal_g_L",
                cited_value=p_nom,
                unit="g/L",
            ),
            GroundedClaim(
                claim_id="CLM-CF-HYB-DO",
                statement=f"Counterfactual hybrid minimum DO is {min_do_hyb:.2f}%.",
                citation_ref="counterfactual:hybrid",
                metric_name="min_do_hybrid_pct",
                cited_value=min_do_hyb,
                unit="%",
            ),
            GroundedClaim(
                claim_id="CLM-ALCOA-LEDGER",
                statement=f"Report linked to tamper-evident ALCOA+ ledger SHA-256 {alcoa_sha[:12]}.",
                citation_ref="alcoa:record_sha256",
            ),
        ]
        if wmse is not None:
            claims.append(
                GroundedClaim(
                    claim_id="CLM-UHL-WMSE",
                    statement=f"Empirical CHO-K1 Uhlendorff pooled test wMSE={float(wmse):.4f}.",
                    citation_ref="uhlendorff:pooled_test_wmse",
                    metric_name="remediated_pooled_test_wmse",
                    cited_value=float(wmse),
                    unit="wMSE",
                )
            )
        elif cited_campaigns:
            claims.append(
                GroundedClaim(
                    claim_id="CLM-UHL-CAMP",
                    statement=f"Grounded against Uhlendorff campaign {cited_campaigns[0]}.",
                    citation_ref=f"uhlendorff:{cited_campaigns[0]}",
                )
            )

        # DeterministicCiCapaProvider always synthesizes from verified deterministic templates
        exec_mode: Literal["deterministic", "llm_assisted"] = "deterministic"

        dur_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        llm_meta = LlmExecutionMetadata(
            provider=self.provider_name,
            model_id="deterministic-bioprocess-capa-v2",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            prompt_sha256=PROMPT_SHA256,
            context_sha256=str(evidence_context["context_sha256"]),
            temperature=0.0,
            seed=42,
            latency_ms=dur_ms,
            token_usage={"prompt_tokens": len(PROMPT_TEMPLATE_TEXT.split()), "completion_tokens": len(narrative.split())},
            deterministic_ci_mode=True,
        )

        report = DeviationCapaReport(
            session_id=state["session_id"],
            vessel_scale_L=int(state.get("vessel_scale_L", 5)),
            root_cause_classification=root_cause,
            severity=severity,
            d_m=d_m,
            in_domain=in_domain,
            nis=nis,
            spe=spe,
            recommended_action=action,
            counterfactual_comparison=CounterfactualComparison(
                final_P_hybrid_g_L=p_hyb,
                final_P_nominal_g_L=p_nom,
                titer_delta_g_L=titer_delta,
                min_do_hybrid_pct=min_do_hyb,
                min_do_nominal_pct=min_do_nom,
                peak_acetate_hybrid_g_L=peak_ace_hyb,
                peak_acetate_nominal_g_L=peak_ace_nom,
            ),
            cited_uhlendorff_campaigns=cited_campaigns,
            remediated_pooled_test_wmse=None if wmse is None else float(wmse),
            alcoa_record_sha256=alcoa_sha,
            alcoa_audit_log_count=int(alcoa["audit_log_count"]),
            execution_mode=exec_mode,
            narrative_summary=narrative,
            claims=claims,
            context_sha256=str(evidence_context["context_sha256"]),
            llm_execution_metadata=llm_meta.model_dump(),
        )
        return report, llm_meta


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
    provider: Any
    qa_review_input: Optional[Dict[str, Any]]
    excursion_result: Dict[str, Any]
    lims_replay_result: Optional[Dict[str, Any]]
    alcoa_record: Dict[str, Any]
    counterfactual_hybrid: Dict[str, Any]
    counterfactual_nominal: Dict[str, Any]
    empirical_precedents: Dict[str, Any]
    evidence_context: Dict[str, Any]
    capa_report: DeviationCapaReport
    llm_execution_metadata: LlmExecutionMetadata
    guardrail_result: GuardrailResult
    part1_raw_output: Part1RawModelOutput
    staged_draft: Optional[Dict[str, Any]]
    part11_record: Optional[Part11AiDecisionRecord]
    trace: ExecutionTrace


class _FallbackCompiledGraph:
    """Typed state-machine runner supporting linear and conditional LangGraph edge semantics."""

    def __init__(
        self,
        nodes: Dict[str, Callable[[DeviationInvestigationState], Dict[str, Any]]],
        edges: Dict[str, str],
        conditional_edges: Dict[str, Tuple[Callable[[DeviationInvestigationState], str], Dict[str, str]]],
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges

    def invoke(self, initial_state: DeviationInvestigationState) -> DeviationInvestigationState:
        state: DeviationInvestigationState = dict(initial_state)
        cur = self._edges[START]
        hops = 0
        while cur != END:
            if hops > 32:
                raise RuntimeError("Deviation investigation graph exceeded maximum traversal hops.")
            hops += 1
            updates = self._nodes[cur](state)
            state.update(updates)
            if cur in self._conditional_edges:
                router_fn, mapping = self._conditional_edges[cur]
                route_key = router_fn(state)
                cur = mapping[route_key]
            else:
                cur = self._edges[cur]
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

        # Start a fresh shadow clone from persisted live DB without mutating live DB
        evict_shadow_session(sid, db_url)

        live_alcoa: Optional[Dict[str, Any]] = None
        try:
            live_alcoa = _invoke_mcp_with_trace(
                trace,
                "detect_excursion",
                get_audit_trail_readonly,
                "get_audit_trail_readonly",
                {"session_id": sid, "database_url": db_url},
            )
        except ValueError:
            live_alcoa = None

        excursion_out: Dict[str, Any] = {}
        lims_out: Optional[Dict[str, Any]] = None

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
            excursion_out = lims_out

        if not excursion_out:
            live_state = _invoke_mcp_with_trace(
                trace,
                "detect_excursion",
                get_twin_state_readonly,
                "get_twin_state_readonly",
                {"session_id": sid, "database_url": db_url, "show_truth": False},
            )
            engine, SessionLocal = create_db_engine_and_sessionmaker(db_url)
            try:
                _, model = load_calibrated_assets()
                with SessionLocal() as db:
                    loaded = load_twin_session(db, sid, model)
                if loaded is None:
                    raise ValueError(f"Twin session '{sid}' not found.")
                twin = loaded["twin"]
                excursion_out = _invoke_mcp_with_trace(
                    trace,
                    "detect_excursion",
                    ingest_pat_and_estimate_state,
                    "ingest_pat_and_estimate_state",
                    {
                        "session_id": sid,
                        "vessel_scale_L": live_state["vessel_scale_L"],
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
            finally:
                engine.dispose()

        alcoa_rec = live_alcoa if live_alcoa is not None else (lims_out["alcoa_record"] if lims_out else {})
        if not alcoa_rec:
            raise ValueError(f"Twin session '{sid}' has no verifiable ALCOA+ audit record.")

        return {
            "excursion_result": excursion_out,
            "lims_replay_result": lims_out,
            "alcoa_record": alcoa_rec,
        }

    return _run_node("detect_excursion", state, _impl)


def _retrieve_evidence_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        scale_l = st.get("vessel_scale_L", 5)
        duration_h = float(st.get("duration_h", 12.0))
        mu_target = float(st.get("mu_target", 0.28))
        dist_mode = st.get("disturbance_mode", "none")

        hyb = _invoke_mcp_with_trace(
            trace,
            "retrieve_evidence",
            simulate_fedbatch_run,
            "simulate_fedbatch_run",
            {
                "vessel_scale_L": scale_l,
                "duration_h": duration_h,
                "mode": "hybrid",
                "mu_target": mu_target,
                "disturbance_mode": dist_mode,
                "dt_eval": 0.25,
                "show_truth": False,
            },
        )
        nom = _invoke_mcp_with_trace(
            trace,
            "retrieve_evidence",
            simulate_fedbatch_run,
            "simulate_fedbatch_run",
            {
                "vessel_scale_L": scale_l,
                "duration_h": duration_h,
                "mode": "nominal",
                "mu_target": mu_target,
                "disturbance_mode": dist_mode,
                "dt_eval": 0.25,
                "show_truth": False,
            },
        )
        precedents = _invoke_mcp_with_trace(
            trace,
            "retrieve_evidence",
            search_uhlendorff_campaigns,
            "search_uhlendorff_campaigns",
            {"mode": "fed-batch"},
        )
        evidence_ctx = build_evidence_context(
            excursion_result=st["excursion_result"],
            counterfactual_hybrid=hyb,
            counterfactual_nominal=nom,
            empirical_precedents=precedents,
            alcoa_record=st["alcoa_record"],
        )
        engine, SessionLocal = create_db_engine_and_sessionmaker(st.get("database_url"))
        try:
            with SessionLocal() as db:
                save_server_evidence_context(db, st["session_id"], evidence_ctx)
        finally:
            engine.dispose()
        return {
            "counterfactual_hybrid": hyb,
            "counterfactual_nominal": nom,
            "empirical_precedents": precedents,
            "evidence_context": evidence_ctx,
        }

    return _run_node("retrieve_evidence", state, _impl)


def _draft_capa_report_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        provider: CapaDraftProvider = st.get("provider") or DeterministicCiCapaProvider()
        report, llm_meta = provider.draft_capa(st, st["evidence_context"])
        trace.execution_mode = report.execution_mode
        return {
            "capa_report": report,
            "llm_execution_metadata": llm_meta,
        }

    return _run_node("draft_capa_report", state, _impl)


def _validate_capa_guardrail_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        del trace
        report = st["capa_report"]
        evidence_ctx = st["evidence_context"]
        expected_alcoa_sha = str(st["alcoa_record"].get("record_sha256", ""))

        guardrail_res = validate_capa_draft(
            report,
            evidence_ctx,
            expected_alcoa_sha256=expected_alcoa_sha,
        )
        violation_msgs = [f"{v.code} ({v.field}): {v.detail}" for v in guardrail_res.violations]
        violation_codes = [v.code for v in guardrail_res.violations]

        part1 = build_part1_raw_output(
            raw_output_payload=report.model_dump(),
            context_sha256=str(evidence_ctx["context_sha256"]),
            guardrail_passed=guardrail_res.passed,
            guardrail_violations=violation_msgs,
        )

        if not guardrail_res.passed:
            record_agent_guardrail_rejection(st["session_id"], violation_codes)
            updated_report = report.model_copy(
                update={
                    "guardrail_passed": False,
                    "guardrail_violations": violation_msgs,
                    "qa_status": "GUARDRAIL_REJECTED",
                    "context_sha256": str(evidence_ctx["context_sha256"]),
                }
            )
        else:
            updated_report = report.model_copy(
                update={
                    "guardrail_passed": True,
                    "guardrail_violations": [],
                    "context_sha256": str(evidence_ctx["context_sha256"]),
                }
            )

        return {
            "capa_report": updated_report,
            "guardrail_result": guardrail_res,
            "part1_raw_output": part1,
        }

    return _run_node("validate_capa_guardrail", state, _impl)


def _route_after_guardrail(state: DeviationInvestigationState) -> str:
    return "pass" if state["guardrail_result"].passed else "reject"


def _human_qa_approval_gate_node(state: DeviationInvestigationState) -> Dict[str, Any]:
    def _impl(st: DeviationInvestigationState, trace: ExecutionTrace) -> Dict[str, Any]:
        sid = st["session_id"]
        db_url = st.get("database_url")
        report = st["capa_report"]
        part1 = st["part1_raw_output"]
        llm_meta = st["llm_execution_metadata"]
        qa_input = st.get("qa_review_input")

        if qa_input is None:
            staged = _invoke_mcp_with_trace(
                trace,
                "human_qa_approval_gate",
                stage_capa_draft,
                "stage_capa_draft",
                {
                    "session_id": sid,
                    "draft_report": report.model_dump(),
                    "author_id": "capa_deviation_graph",
                    "database_url": db_url,
                },
            )
            updated_report = report.model_copy(update={"qa_status": "PENDING_QA_APPROVAL"})
            return {
                "capa_report": updated_report,
                "staged_draft": staged,
                "part11_record": None,
            }

        redlines = qa_input["redlines"]
        esig = qa_input["two_factor_esig"]
        engine, SessionLocal = create_db_engine_and_sessionmaker(db_url)
        try:
            with SessionLocal() as db:
                next_seq, prev_sha = get_latest_ai_decision_hash(db, sid)
                p11_record = build_part11_decision_record(
                    session_id=sid,
                    part1=part1,
                    reviewer_id=redlines["reviewer_id"],
                    decision=redlines["decision"],
                    revised_overrides=redlines.get("revised_overrides"),
                    rationales=redlines.get("rationales"),
                    signer_id=esig["signer_id"],
                    signer_name=esig["signer_name"],
                    signer_role=esig["signer_role"],
                    signing_meaning=esig["signing_meaning"],
                    factor_1_type=esig["factor_1_type"],
                    factor_2_type=esig["factor_2_type"],
                    factor_2_challenge_id=esig["factor_2_challenge_id"],
                    linked_twin_alcoa_sha256=report.alcoa_record_sha256,
                    llm_execution_metadata=llm_meta,
                    seq=next_seq,
                    prev_entry_sha256=prev_sha,
                    decision_record_id=qa_input.get("decision_record_id"),
                )
                save_ai_decision_record(db, p11_record)
        finally:
            engine.dispose()

        updated_report = report.model_copy(update={"qa_status": p11_record.status})
        return {
            "capa_report": updated_report,
            "staged_draft": None,
            "part11_record": p11_record,
        }

    return _run_node("human_qa_approval_gate", state, _impl)


def build_deviation_graph():
    """Build and compile the 5-step Deviation & CAPA Investigator LangGraph workflow."""
    nodes = {
        "detect_excursion": _detect_excursion_node,
        "retrieve_evidence": _retrieve_evidence_node,
        "draft_capa_report": _draft_capa_report_node,
        "validate_capa_guardrail": _validate_capa_guardrail_node,
        "human_qa_approval_gate": _human_qa_approval_gate_node,
    }
    if _HAS_LANGGRAPH:
        graph = StateGraph(DeviationInvestigationState)
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(START, "detect_excursion")
        graph.add_edge("detect_excursion", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "draft_capa_report")
        graph.add_edge("draft_capa_report", "validate_capa_guardrail")
        graph.add_conditional_edges(
            "validate_capa_guardrail",
            _route_after_guardrail,
            {"pass": "human_qa_approval_gate", "reject": END},
        )
        graph.add_edge("human_qa_approval_gate", END)
        return graph.compile()

    edges = {
        START: "detect_excursion",
        "detect_excursion": "retrieve_evidence",
        "retrieve_evidence": "draft_capa_report",
        "draft_capa_report": "validate_capa_guardrail",
        "human_qa_approval_gate": END,
    }
    cond_edges = {
        "validate_capa_guardrail": (
            _route_after_guardrail,
            {"pass": "human_qa_approval_gate", "reject": END},
        )
    }
    return _FallbackCompiledGraph(nodes, edges, cond_edges)


def run_governed_capa_workflow(
    session_id: str,
    vessel_scale_L: Literal[5, 50, 1000] = 5,
    database_url: Optional[str] = None,
    pat_tick: Optional[Dict[str, Any]] = None,
    delayed_lims: Optional[Dict[str, Any]] = None,
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none",
    duration_h: float = 12.0,
    mu_target: float = 0.28,
    use_llm: bool = False,
    provider: Optional[CapaDraftProvider] = None,
    qa_review_input: Optional[Dict[str, Any]] = None,
) -> CapaWorkflowOutcome:
    """Execute the full 5-step CAPA workflow and return report, trace, guardrails, and Part 11 data."""
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
            "provider": provider,
            "qa_review_input": qa_review_input,
            "trace": trace,
        }
    )
    trace.total_duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    report = final_state["capa_report"]
    record_agent_investigation(session_id, report.qa_status, trace.total_duration_ms)
    return CapaWorkflowOutcome(
        capa_report=report,
        trace=trace,
        evidence_context=final_state["evidence_context"],
        guardrail_result=final_state["guardrail_result"],
        part1_raw_output=final_state["part1_raw_output"],
        staged_draft=final_state.get("staged_draft"),
        part11_record=final_state.get("part11_record"),
    )


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
    provider: Optional[CapaDraftProvider] = None,
    qa_review_input: Optional[Dict[str, Any]] = None,
) -> Tuple[DeviationCapaReport, ExecutionTrace]:
    """Execute the CAPA Deviation Investigation graph and return (report, trace)."""
    outcome = run_governed_capa_workflow(
        session_id=session_id,
        vessel_scale_L=vessel_scale_L,
        database_url=database_url,
        pat_tick=pat_tick,
        delayed_lims=delayed_lims,
        disturbance_mode=disturbance_mode,
        duration_h=duration_h,
        mu_target=mu_target,
        use_llm=use_llm,
        provider=provider,
        qa_review_input=qa_review_input,
    )
    return outcome.capa_report, outcome.trace
