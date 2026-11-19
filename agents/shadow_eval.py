"""Zero-secret shadow-mode evaluation harness measuring Unsupported Claim Rate (UCR) & 95% CIs."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from api.db import create_db_engine_and_sessionmaker, load_twin_session, save_twin_session
from api.runtime import create_twin_instance, load_calibrated_assets
from hybrid_fedbatch_twin.mechanistic import default_operating_inputs
from mcp_server.server import clear_shadow_sandbox

from .deviation_graph import (
    CapaDraftProvider,
    CapaWorkflowOutcome,
    DeterministicCiCapaProvider,
    DeviationCapaReport,
    DeviationInvestigationState,
    run_governed_capa_workflow,
)
from .guardrails import GroundedClaim
from .part11 import LlmExecutionMetadata


def wilson_ci_95(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Compute exact 95% Wilson score confidence interval for binomial proportion k / n."""
    if n <= 0:
        return 0.0, 1.0
    k_clamped = max(0, min(int(k), int(n)))
    p_hat = float(k_clamped) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    half_width = (
        z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n))) / float(n))
    ) / denom
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    if k_clamped == 0:
        low = 0.0
    if k_clamped == n:
        high = 1.0
    return round(low, 6), round(high, 6)


class ShadowExcursionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    vessel_scale_L: Literal[5, 50, 1000] = 5
    seed_pat_ticks: List[Dict[str, Any]] = Field(default_factory=list)
    shadow_pat_tick: Optional[Dict[str, Any]] = None
    delayed_lims: Optional[Dict[str, Any]] = None
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none"
    duration_h: float = 4.0
    mu_target: float = 0.28


class ShadowCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    session_id: str
    vessel_scale_L: int
    root_cause_classification: str
    severity: str
    qa_status: str
    guardrail_passed: bool
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    ucr: float
    live_state_mutated: bool
    duration_ms: float


class ShadowEvalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_cases: int
    n_guardrail_passed: int
    n_guardrail_rejected: int
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    ucr: float
    ucr_ci_95_low: float
    ucr_ci_95_high: float
    any_live_mutation: bool
    mean_latency_ms: float
    case_results: List[ShadowCaseResult] = Field(default_factory=list)


def default_shadow_excursion_cases() -> List[ShadowExcursionCase]:
    """Return 8 varied bioprocess excursion scenarios across 5 L, 50 L, and 1000 L scales."""
    return [
        ShadowExcursionCase(
            case_id="nominal_5L_in_domain",
            vessel_scale_L=5,
            seed_pat_ticks=[
                {"t_h": 0.2, "y_Xv": 1.60, "y_CL": 7.60, "y_V": 2.01, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2}
            ],
            shadow_pat_tick={"t_h": 0.4, "y_Xv": 1.72, "y_CL": 7.45, "y_V": 2.02, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2},
            disturbance_mode="none",
        ),
        ShadowExcursionCase(
            case_id="hypoxia_aeration_drop_1000L",
            vessel_scale_L=1000,
            seed_pat_ticks=[
                {"t_h": 0.2, "y_Xv": 1.60, "y_CL": 7.50, "y_V": 401.6, "F_act": 8.0, "rpm_act": 134.8, "vvm_act": 0.205}
            ],
            shadow_pat_tick={"t_h": 0.6, "y_Xv": 1.85, "y_CL": 1.10, "y_V": 405.0, "F_act": 10.0, "rpm_act": 134.8, "vvm_act": 0.10},
            delayed_lims={"t_sample_h": 0.2, "y_S": 0.40, "y_A": 0.45, "y_P": 0.12},
            disturbance_mode="aeration_drop",
        ),
        ShadowExcursionCase(
            case_id="acetate_overflow_50L",
            vessel_scale_L=50,
            seed_pat_ticks=[
                {"t_h": 0.2, "y_Xv": 1.65, "y_CL": 7.20, "y_V": 20.1, "F_act": 0.40, "rpm_act": 265.0, "vvm_act": 0.56}
            ],
            shadow_pat_tick={"t_h": 0.4, "y_Xv": 2.20, "y_CL": 5.50, "y_V": 20.3, "F_act": 0.50, "rpm_act": 265.0, "vvm_act": 0.56},
            delayed_lims={"t_sample_h": 0.2, "y_S": 2.50, "y_A": 2.20, "y_P": 0.18},
            disturbance_mode="feed_step",
        ),
        ShadowExcursionCase(
            case_id="delayed_lims_df2_nis_breach_5L",
            vessel_scale_L=5,
            seed_pat_ticks=[
                {"t_h": 0.2, "y_Xv": 1.60, "y_CL": 7.60, "y_V": 2.01, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2}
            ],
            shadow_pat_tick={"t_h": 0.4, "y_Xv": 1.70, "y_CL": 7.30, "y_V": 2.02, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2},
            delayed_lims={"t_sample_h": 0.2, "y_S": 3.00, "y_A": 2.20, "y_P": None},
            disturbance_mode="none",
        ),
        ShadowExcursionCase(
            case_id="persistent_spe_drift_5L",
            vessel_scale_L=5,
            seed_pat_ticks=[
                {"t_h": 0.2, "y_Xv": 1.62, "y_CL": 6.60, "y_V": 2.01, "F_act": 0.02, "rpm_act": 450.0, "vvm_act": 1.2}
            ],
            shadow_pat_tick={"t_h": 0.4, "y_Xv": 1.70, "y_CL": 5.90, "y_V": 2.02, "F_act": 0.02, "rpm_act": 450.0, "vvm_act": 1.2},
            disturbance_mode="aeration_drop",
        ),
        ShadowExcursionCase(
            case_id="mahalanobis_ood_high_biomass_5L",
            vessel_scale_L=5,
            seed_pat_ticks=[],
            shadow_pat_tick={"t_h": 0.2, "y_Xv": 60.0, "y_CL": 7.40, "y_V": 2.02, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2},
            disturbance_mode="none",
        ),
        ShadowExcursionCase(
            case_id="combined_ood_hypoxia_acetate_50L",
            vessel_scale_L=50,
            seed_pat_ticks=[
                {"t_h": 0.2, "y_Xv": 2.00, "y_CL": 6.80, "y_V": 20.1, "F_act": 0.40, "rpm_act": 265.0, "vvm_act": 0.56}
            ],
            shadow_pat_tick={"t_h": 0.5, "y_Xv": 55.0, "y_CL": 0.85, "y_V": 20.4, "F_act": 0.45, "rpm_act": 265.0, "vvm_act": 0.20},
            delayed_lims={"t_sample_h": 0.2, "y_S": 4.20, "y_A": 3.10, "y_P": 0.25},
            disturbance_mode="aeration_drop",
        ),
        ShadowExcursionCase(
            case_id="feed_step_disturbance_1000L",
            vessel_scale_L=1000,
            seed_pat_ticks=[
                {"t_h": 0.2, "y_Xv": 1.62, "y_CL": 7.55, "y_V": 401.5, "F_act": 8.0, "rpm_act": 134.8, "vvm_act": 0.205}
            ],
            shadow_pat_tick={"t_h": 0.4, "y_Xv": 1.80, "y_CL": 7.10, "y_V": 403.5, "F_act": 12.0, "rpm_act": 134.8, "vvm_act": 0.205},
            delayed_lims={"t_sample_h": 0.2, "y_S": 0.80, "y_A": 0.35, "y_P": 0.10},
            disturbance_mode="feed_step",
        ),
    ]


class FaultInjectingCapaProvider:
    """Adversarial test provider injecting ungrounded citations or numeric mismatches to verify guardrails."""

    def __init__(
        self,
        fault_mode: Literal[
            "missing_citation",
            "invented_campaign",
            "claim_numeric_mismatch",
            "incomplete_schema",
            "alcoa_mismatch",
            "mixed",
        ] = "mixed",
    ) -> None:
        self.fault_mode = fault_mode
        self._base = DeterministicCiCapaProvider(provider_name="fault_injecting_eval_provider")
        self._call_idx = 0

    def draft_capa(
        self,
        state: DeviationInvestigationState,
        evidence_context: Dict[str, Any],
    ) -> Tuple[DeviationCapaReport, LlmExecutionMetadata]:
        report, meta = self._base.draft_capa(state, evidence_context)
        modes = (
            "missing_citation",
            "invented_campaign",
            "claim_numeric_mismatch",
            "incomplete_schema",
            "alcoa_mismatch",
        )
        mode = modes[self._call_idx % len(modes)] if self.fault_mode == "mixed" else self.fault_mode
        self._call_idx += 1

        claims = [c.model_copy() for c in report.claims]
        if mode == "missing_citation" and claims:
            claims[0] = claims[0].model_copy(update={"citation_ref": ""})
            return report.model_copy(update={"claims": claims}), meta

        if mode == "invented_campaign":
            if claims:
                claims[-1] = GroundedClaim(
                    claim_id="CLM-FAKE-CAMP",
                    statement="Fabricated precedent EXP999 showed 99 g/L titer.",
                    citation_ref="uhlendorff:EXP999",
                )
            return report.model_copy(
                update={
                    "cited_uhlendorff_campaigns": [*report.cited_uhlendorff_campaigns, "EXP999"],
                    "claims": claims,
                }
            ), meta

        if mode == "claim_numeric_mismatch" and claims:
            claims[0] = claims[0].model_copy(update={"cited_value": float(report.d_m) + 25.0})
            return report.model_copy(update={"d_m": float(report.d_m) + 25.0, "claims": claims}), meta

        if mode == "incomplete_schema":
            return report.model_copy(update={"recommended_action": "", "claims": []}), meta

        # alcoa_mismatch
        if claims:
            claims[-1] = claims[-1].model_copy(update={"citation_ref": "uhlendorff:UNKNOWN"})
        return report.model_copy(
            update={"alcoa_record_sha256": "f" * 64, "claims": claims}
        ), meta


def run_shadow_evaluation_suite(
    database_url: str,
    cases: Optional[Sequence[ShadowExcursionCase]] = None,
    provider: Optional[CapaDraftProvider] = None,
) -> ShadowEvalSummary:
    """Execute offline shadow-mode evaluation across varied excursion cases and compute UCR + 95% CI."""
    eval_cases = list(cases) if cases is not None else default_shadow_excursion_cases()
    active_provider = provider or DeterministicCiCapaProvider()
    _, model = load_calibrated_assets()
    engine, SessionLocal = create_db_engine_and_sessionmaker(database_url)

    results: List[ShadowCaseResult] = []
    total_claims = 0
    supported_claims = 0
    unsupported_claims = 0
    n_passed = 0
    n_rejected = 0
    any_mutated = False
    latencies: List[float] = []

    try:
        for case in eval_cases:
            sid = f"shd_{uuid4().hex[:10]}"
            twin = create_twin_instance(
                vessel_scale_L=case.vessel_scale_L,
                batch_id=f"SHADOW-{case.case_id}",
                operator_id="shadow_eval_harness",
            )
            u_def = default_operating_inputs(twin.params)
            for tick in case.seed_pat_ticks:
                twin.ingest_online_pat(
                    t_h=float(tick["t_h"]),
                    y_Xv=float(tick["y_Xv"]),
                    y_CL=float(tick["y_CL"]),
                    y_V=float(tick["y_V"]),
                    F_act=float(tick["F_act"]),
                    rpm_act=float(tick.get("rpm_act", u_def.rpm)),
                    vvm_act=float(tick.get("vvm_act", u_def.vvm)),
                )
            with SessionLocal() as db:
                save_twin_session(db, sid, case.vessel_scale_L, twin)
                before_rec = twin.to_audit_record()
                before_sha = str(before_rec["record_sha256"])
                before_count = int(before_rec["audit_log_count"])

            outcome: CapaWorkflowOutcome = run_governed_capa_workflow(
                session_id=sid,
                vessel_scale_L=case.vessel_scale_L,
                database_url=database_url,
                pat_tick=case.shadow_pat_tick,
                delayed_lims=case.delayed_lims,
                disturbance_mode=case.disturbance_mode,
                duration_h=case.duration_h,
                mu_target=case.mu_target,
                use_llm=False,
                provider=active_provider,
                qa_review_input=None,
            )

            with SessionLocal() as db:
                reloaded = load_twin_session(db, sid, model)
            if reloaded is None:
                live_mutated = True
            else:
                after_rec = reloaded["twin"].to_audit_record()
                live_mutated = (
                    str(after_rec["record_sha256"]) != before_sha
                    or int(after_rec["audit_log_count"]) != before_count
                )

            any_mutated = any_mutated or live_mutated
            gr = outcome.guardrail_result
            if gr.passed:
                n_passed += 1
            else:
                n_rejected += 1

            total_claims += gr.total_claims
            supported_claims += gr.supported_claims
            unsupported_claims += gr.unsupported_claims
            latencies.append(float(outcome.trace.total_duration_ms))

            results.append(
                ShadowCaseResult(
                    case_id=case.case_id,
                    session_id=sid,
                    vessel_scale_L=case.vessel_scale_L,
                    root_cause_classification=outcome.capa_report.root_cause_classification,
                    severity=outcome.capa_report.severity,
                    qa_status=outcome.capa_report.qa_status,
                    guardrail_passed=gr.passed,
                    total_claims=gr.total_claims,
                    supported_claims=gr.supported_claims,
                    unsupported_claims=gr.unsupported_claims,
                    ucr=gr.unsupported_claim_rate,
                    live_state_mutated=live_mutated,
                    duration_ms=float(outcome.trace.total_duration_ms),
                )
            )
    finally:
        clear_shadow_sandbox(database_url)
        engine.dispose()

    ucr = 0.0 if total_claims == 0 else float(unsupported_claims) / float(total_claims)
    ci_low, ci_high = wilson_ci_95(unsupported_claims, total_claims)
    return ShadowEvalSummary(
        n_cases=len(results),
        n_guardrail_passed=n_passed,
        n_guardrail_rejected=n_rejected,
        total_claims=total_claims,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        ucr=round(ucr, 6),
        ucr_ci_95_low=ci_low,
        ucr_ci_95_high=ci_high,
        any_live_mutation=any_mutated,
        mean_latency_ms=round(float(np.mean(latencies)) if latencies else 0.0, 3),
        case_results=results,
    )
