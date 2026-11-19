"""Deterministic citation, telemetry numeric matching, and schema completeness guardrails for CAPA drafts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set
from pydantic import BaseModel, ConfigDict, Field

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class GroundedClaim(BaseModel):
    """Single verifiable claim inside a CAPA draft linked to an evidence citation."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=64)
    statement: str = Field(min_length=1, max_length=1024)
    citation_ref: str = Field(default="", max_length=128)
    metric_name: Optional[str] = Field(default=None, max_length=64)
    cited_value: Optional[float] = None
    unit: Optional[str] = Field(default=None, max_length=32)


class GuardrailViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    field: str
    detail: str


class GuardrailResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    violations: List[GuardrailViolation] = Field(default_factory=list)
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    unsupported_claim_rate: float = 0.0


def build_evidence_context(
    excursion_result: Dict[str, Any],
    counterfactual_hybrid: Dict[str, Any],
    counterfactual_nominal: Dict[str, Any],
    empirical_precedents: Dict[str, Any],
    alcoa_record: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble an immutable grounded evidence bundle and SHA-256 context digest."""
    x_hat = dict(excursion_result.get("x_hat", {}))
    hyb_final = dict(counterfactual_hybrid.get("final_state", {}))
    nom_final = dict(counterfactual_nominal.get("final_state", {}))

    p_hyb = float(hyb_final.get("P_g_L", 0.0))
    p_nom = float(nom_final.get("P_g_L", 0.0))
    wmse = empirical_precedents.get("remediated_pooled_test_wmse")

    telemetry_metrics: Dict[str, float] = {
        "d_m": float(excursion_result.get("d_m", 0.0)),
        "nis": float(excursion_result.get("nis", 0.0)),
        "spe": float(excursion_result.get("spe", 0.0)),
        "X_v": float(x_hat.get("X_v", 0.0)),
        "S": float(x_hat.get("S", 0.0)),
        "C_L": float(x_hat.get("C_L", 0.0)),
        "A": float(x_hat.get("A", 0.0)),
        "V": float(x_hat.get("V", 0.0)),
        "final_P_hybrid_g_L": p_hyb,
        "final_P_nominal_g_L": p_nom,
        "titer_delta_g_L": p_hyb - p_nom,
        "min_do_hybrid_pct": float(counterfactual_hybrid.get("min_DO_pct", 0.0)),
        "min_do_nominal_pct": float(counterfactual_nominal.get("min_DO_pct", 0.0)),
        "peak_acetate_hybrid_g_L": float(counterfactual_hybrid.get("peak_acetate_g_L", 0.0)),
        "peak_acetate_nominal_g_L": float(counterfactual_nominal.get("peak_acetate_g_L", 0.0)),
    }
    if wmse is not None:
        telemetry_metrics["remediated_pooled_test_wmse"] = float(wmse)

    observed_campaigns = sorted(
        {str(r["campaign_id"]) for r in empirical_precedents.get("runs", []) if "campaign_id" in r}
    )
    allowed_citations: Set[str] = {
        "ekf:latest_state",
        "ekf:gate_status",
        "counterfactual:hybrid",
        "counterfactual:nominal",
        "counterfactual:delta",
        "uhlendorff:pooled_test_wmse",
        "alcoa:record_sha256",
    }
    for cid in observed_campaigns:
        allowed_citations.add(f"uhlendorff:{cid}")

    context: Dict[str, Any] = {
        "telemetry_metrics": telemetry_metrics,
        "observed_booleans": {
            "in_domain": bool(excursion_result.get("in_domain", False)),
            "undecided": bool(excursion_result.get("undecided", False)),
            "residual_breach": bool(excursion_result.get("residual_breach", False)),
        },
        "observed_campaigns": observed_campaigns,
        "remediated_pooled_test_wmse": None if wmse is None else float(wmse),
        "alcoa_record_sha256": str(alcoa_record.get("record_sha256", "")),
        "alcoa_audit_log_count": int(alcoa_record.get("audit_log_count", 0)),
        "allowed_citations": sorted(allowed_citations),
    }
    context["context_sha256"] = _canonical_sha256(context)
    return context


def _parse_narrative_mentions(narrative: str) -> List[tuple[str, float, float]]:
    """Extract all explicit metric=value patterns from prose for cross-checking."""
    mentions: List[tuple[str, float, float]] = []
    for m_dm in re.finditer(r"\bD_M\s*=\s*(-?\d+(?:\.\d+)?)", narrative):
        mentions.append(("d_m", float(m_dm.group(1)), 0.015))
    for m_nis in re.finditer(r"\bNIS\s*=\s*(-?\d+(?:\.\d+)?)", narrative):
        mentions.append(("nis", float(m_nis.group(1)), 0.05))
    for m_spe in re.finditer(r"\bSPE\s*=\s*(-?\d+(?:\.\d+)?)", narrative):
        mentions.append(("spe", float(m_spe.group(1)), 0.05))
    for m_p in re.finditer(
        r"projects\s+P\s*=\s*(-?\d+(?:\.\d+)?)\s*g/L\s+vs\s+nominal\s+P\s*=\s*(-?\d+(?:\.\d+)?)",
        narrative,
    ):
        mentions.append(("final_P_hybrid_g_L", float(m_p.group(1)), 0.02))
        mentions.append(("final_P_nominal_g_L", float(m_p.group(2)), 0.02))
    for m_pnom in re.finditer(r"\bnominal\s+P\s*=\s*(-?\d+(?:\.\d+)?)", narrative):
        mentions.append(("final_P_nominal_g_L", float(m_pnom.group(1)), 0.02))
    for m_do in re.finditer(r"\bmin_DO\s*=\s*(-?\d+(?:\.\d+)?)%\s*vs\s*(-?\d+(?:\.\d+)?)%", narrative):
        mentions.append(("min_do_hybrid_pct", float(m_do.group(1)), 0.06))
        mentions.append(("min_do_nominal_pct", float(m_do.group(2)), 0.06))
    for m_do1 in re.finditer(r"\bmin_DO\s*=\s*(-?\d+(?:\.\d+)?)%", narrative):
        mentions.append(("min_do_hybrid_pct", float(m_do1.group(1)), 0.06))
    for m_delta in re.finditer(r"\bdelta\s*=\s*([+-]?\d+(?:\.\d+)?)\s*g/L", narrative):
        mentions.append(("titer_delta_g_L", float(m_delta.group(1)), 0.02))
    for m_wmse in re.finditer(r"\bwMSE\s*=\s*(-?\d+(?:\.\d+)?)", narrative, re.IGNORECASE):
        mentions.append(("remediated_pooled_test_wmse", float(m_wmse.group(1)), 0.02))
    return mentions


def validate_capa_draft(
    report: Any,
    evidence_context: Dict[str, Any],
    expected_alcoa_sha256: Optional[str] = None,
    numeric_tol: float = 1e-4,
) -> GuardrailResult:
    """Deterministically reject CAPA drafts with missing citations, numeric mismatches, or incomplete schemas."""
    data = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    violations: List[GuardrailViolation] = []

    metrics: Dict[str, float] = dict(evidence_context.get("telemetry_metrics", {}))
    obs_bools: Dict[str, bool] = dict(evidence_context.get("observed_booleans", {}))
    obs_campaigns: Set[str] = set(evidence_context.get("observed_campaigns", []))
    allowed_citations: Set[str] = set(evidence_context.get("allowed_citations", []))
    exp_alcoa = expected_alcoa_sha256 or str(evidence_context.get("alcoa_record_sha256", ""))

    # 1. Schema completeness
    rec_action = str(data.get("recommended_action", "")).strip()
    if len(rec_action) < 10:
        violations.append(
            GuardrailViolation(
                code="INCOMPLETE_SCHEMA",
                field="recommended_action",
                detail="recommended_action must contain a substantive non-empty engineering directive.",
            )
        )
    narrative = str(data.get("narrative_summary", "")).strip()
    if len(narrative) < 15:
        violations.append(
            GuardrailViolation(
                code="INCOMPLETE_SCHEMA",
                field="narrative_summary",
                detail="narrative_summary must be present and non-trivial.",
            )
        )

    alcoa_sha = str(data.get("alcoa_record_sha256", ""))
    if not _SHA256_RE.match(alcoa_sha):
        violations.append(
            GuardrailViolation(
                code="INVALID_ALCOA_HASH",
                field="alcoa_record_sha256",
                detail=f"Invalid SHA-256 digest format: {alcoa_sha!r}.",
            )
        )
    elif exp_alcoa and alcoa_sha != exp_alcoa:
        violations.append(
            GuardrailViolation(
                code="ALCOA_HASH_MISMATCH",
                field="alcoa_record_sha256",
                detail=f"Draft ALCOA+ SHA-256 {alcoa_sha[:12]} != ledger {exp_alcoa[:12]}.",
            )
        )

    # 2. Structured numeric & boolean grounding against telemetry and counterfactuals
    def _chk_num(field_name: str, observed_key: str, val: Any, tol: float = numeric_tol) -> None:
        if observed_key not in metrics:
            return
        try:
            fval = float(val)
        except (TypeError, ValueError):
            violations.append(
                GuardrailViolation(
                    code="NUMERIC_MISMATCH",
                    field=field_name,
                    detail=f"{field_name}={val!r} is not numeric.",
                )
            )
            return
        obs_val = float(metrics[observed_key])
        if not math.isfinite(fval) or abs(fval - obs_val) > tol:
            violations.append(
                GuardrailViolation(
                    code="NUMERIC_MISMATCH",
                    field=field_name,
                    detail=f"{field_name}={fval} != observed {obs_val}.",
                )
            )

    for req_key in ("d_m", "nis", "spe"):
        if req_key not in data or data.get(req_key) is None:
            violations.append(
                GuardrailViolation(
                    code="INCOMPLETE_SCHEMA",
                    field=req_key,
                    detail=f"Missing required telemetry field {req_key!r}.",
                )
            )
        else:
            _chk_num(req_key, req_key, data.get(req_key))

    if "in_domain" not in data or not isinstance(data.get("in_domain"), bool):
        violations.append(
            GuardrailViolation(
                code="INCOMPLETE_SCHEMA",
                field="in_domain",
                detail="Missing or non-boolean required field 'in_domain'.",
            )
        )
    elif bool(data.get("in_domain")) != bool(obs_bools.get("in_domain", False)):
        violations.append(
            GuardrailViolation(
                code="BOOLEAN_MISMATCH",
                field="in_domain",
                detail=f"in_domain={data.get('in_domain')} != observed {obs_bools.get('in_domain')}.",
            )
        )

    cf = data.get("counterfactual_comparison")
    if not isinstance(cf, dict):
        violations.append(
            GuardrailViolation(
                code="INCOMPLETE_SCHEMA",
                field="counterfactual_comparison",
                detail="counterfactual_comparison object is missing.",
            )
        )
    else:
        for key in (
            "final_P_hybrid_g_L",
            "final_P_nominal_g_L",
            "titer_delta_g_L",
            "min_do_hybrid_pct",
            "min_do_nominal_pct",
            "peak_acetate_hybrid_g_L",
            "peak_acetate_nominal_g_L",
        ):
            if key not in cf:
                violations.append(
                    GuardrailViolation(
                        code="INCOMPLETE_SCHEMA",
                        field=f"counterfactual_comparison.{key}",
                        detail=f"Missing required field {key}.",
                    )
                )
            else:
                _chk_num(f"counterfactual_comparison.{key}", key, cf[key])

    # 3. Empirical campaign citations
    cited_campaigns = data.get("cited_uhlendorff_campaigns") or []
    if not cited_campaigns:
        violations.append(
            GuardrailViolation(
                code="MISSING_CITATION",
                field="cited_uhlendorff_campaigns",
                detail="At least one verified Uhlendorff campaign citation is required.",
            )
        )
    else:
        bad_camps = sorted(set(str(c) for c in cited_campaigns) - obs_campaigns)
        if bad_camps:
            violations.append(
                GuardrailViolation(
                    code="UNGROUNDED_CAMPAIGN_CITATION",
                    field="cited_uhlendorff_campaigns",
                    detail=f"Campaigns not present in retrieved empirical evidence: {bad_camps}",
                )
            )

    exp_wmse = evidence_context.get("remediated_pooled_test_wmse")
    if exp_wmse is not None and data.get("remediated_pooled_test_wmse") is not None:
        _chk_num("remediated_pooled_test_wmse", "remediated_pooled_test_wmse", data.get("remediated_pooled_test_wmse"))

    # 4. Narrative inline number cross-check
    for metric_name, cited_num, tol in _parse_narrative_mentions(narrative):
        if metric_name in metrics:
            obs_num = float(metrics[metric_name])
            if abs(cited_num - obs_num) > tol:
                violations.append(
                    GuardrailViolation(
                        code="NARRATIVE_NUMERIC_MISMATCH",
                        field="narrative_summary",
                        detail=f"Narrative states {metric_name}={cited_num} vs observed {obs_num:.4f}.",
                    )
                )

    # 5. Claim-level citation & numeric verification (used for UCR)
    raw_claims = data.get("claims")
    claims_list: Sequence[Any] = raw_claims if isinstance(raw_claims, list) else []
    total_claims = len(claims_list)
    unsupported_claims = 0

    if total_claims == 0:
        violations.append(
            GuardrailViolation(
                code="MISSING_CLAIMS",
                field="claims",
                detail="Draft must provide explicit GroundedClaim entries with citations.",
            )
        )
    else:
        for idx, raw_claim in enumerate(claims_list):
            claim_ok = True
            try:
                claim = (
                    raw_claim
                    if isinstance(raw_claim, GroundedClaim)
                    else GroundedClaim.model_validate(raw_claim)
                )
            except Exception as exc:
                claim_ok = False
                violations.append(
                    GuardrailViolation(
                        code="INVALID_CLAIM_SCHEMA",
                        field=f"claims[{idx}]",
                        detail=str(exc),
                    )
                )
                unsupported_claims += 1
                continue

            cit = claim.citation_ref.strip()
            if not cit:
                claim_ok = False
                violations.append(
                    GuardrailViolation(
                        code="MISSING_CITATION",
                        field=f"claims[{idx}].citation_ref",
                        detail=f"Claim {claim.claim_id!r} has an empty citation_ref.",
                    )
                )
            elif cit not in allowed_citations:
                claim_ok = False
                violations.append(
                    GuardrailViolation(
                        code="UNGROUNDED_CITATION",
                        field=f"claims[{idx}].citation_ref",
                        detail=f"Citation {cit!r} is not in verified evidence citations.",
                    )
                )

            if claim.metric_name is not None:
                if claim.metric_name not in metrics:
                    claim_ok = False
                    violations.append(
                        GuardrailViolation(
                            code="UNKNOWN_METRIC_NAME",
                            field=f"claims[{idx}].metric_name",
                            detail=f"Unknown telemetry metric {claim.metric_name!r}.",
                        )
                    )
                elif claim.cited_value is None:
                    claim_ok = False
                    violations.append(
                        GuardrailViolation(
                            code="MISSING_CLAIM_VALUE",
                            field=f"claims[{idx}].cited_value",
                            detail=f"Claim {claim.claim_id!r} specifies metric_name without cited_value.",
                        )
                    )
                else:
                    obs_v = float(metrics[claim.metric_name])
                    if not math.isfinite(float(claim.cited_value)) or abs(float(claim.cited_value) - obs_v) > numeric_tol:
                        claim_ok = False
                        violations.append(
                            GuardrailViolation(
                                code="CLAIM_NUMERIC_MISMATCH",
                                field=f"claims[{idx}].cited_value",
                                detail=f"Claim {claim.claim_id!r} cites {claim.metric_name}={claim.cited_value} != observed {obs_v}.",
                            )
                        )
            elif claim.cited_value is not None:
                claim_ok = False
                violations.append(
                    GuardrailViolation(
                        code="MISSING_METRIC_NAME",
                        field=f"claims[{idx}].metric_name",
                        detail=f"Claim {claim.claim_id!r} supplies cited_value without metric_name.",
                    )
                )

            for stmt_metric, stmt_val, stmt_tol in _parse_narrative_mentions(claim.statement):
                if stmt_metric in metrics:
                    obs_sv = float(metrics[stmt_metric])
                    if abs(stmt_val - obs_sv) > stmt_tol:
                        claim_ok = False
                        violations.append(
                            GuardrailViolation(
                                code="CLAIM_STATEMENT_MISMATCH",
                                field=f"claims[{idx}].statement",
                                detail=f"Claim {claim.claim_id!r} statement cites {stmt_metric}={stmt_val} != observed {obs_sv:.4f}.",
                            )
                        )

            if not claim_ok:
                unsupported_claims += 1

    supported_claims = max(0, total_claims - unsupported_claims)
    ucr = 1.0 if total_claims == 0 else float(unsupported_claims) / float(total_claims)
    return GuardrailResult(
        passed=(len(violations) == 0),
        violations=violations,
        total_claims=total_claims,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        unsupported_claim_rate=round(ucr, 6),
    )
