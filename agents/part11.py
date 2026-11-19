"""21 CFR Part 11 4-part AI Decision Record, QA redline diffing, 2FA e-signature, and tamper verification."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

GENESIS_PART11_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AGENT_SIGNER_PREFIXES = ("mcp_", "agent_", "llm_", "bot_", "automated_")


class FinalizedRecordImmutabilityError(PermissionError):
    """Raised when any caller attempts to mutate or overwrite a finalized QA Part 11 record."""


class TwoFactorReplayError(PermissionError):
    """Raised when a single-use 2FA challenge ID is replayed across QA e-signatures."""


def validate_human_qa_id(actor_id: str, field_name: str = "reviewer_id") -> str:
    """Reject automated agent/LLM/MCP identifiers from acting as human QA reviewers or signers."""
    low = str(actor_id).strip().lower()
    if not low or low.startswith(_AGENT_SIGNER_PREFIXES):
        raise ValueError(
            f"Non-human or agent {field_name} {actor_id!r} cannot perform human QA review or e-signature."
        )
    return str(actor_id)


def canonical_sha256(payload: Any) -> str:
    """Compute deterministic SHA-256 digest of JSON-serializable payloads."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _validate_iso8601(ts: str, field_name: str) -> str:
    try:
        datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp, got {ts!r}.") from exc
    return str(ts)


class LlmExecutionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=128)
    prompt_template_version: str = Field(min_length=1, max_length=64)
    prompt_sha256: str = Field(pattern=_SHA256_RE)
    context_sha256: str = Field(pattern=_SHA256_RE)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: Optional[int] = 42
    latency_ms: float = Field(default=0.0, ge=0.0)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    deterministic_ci_mode: bool = True


class FieldRedlineDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_path: str = Field(min_length=1, max_length=128)
    original_value: Any
    revised_value: Any
    rationale: str = Field(min_length=1, max_length=512)


class Part1RawModelOutput(BaseModel):
    """Part 1: Immutable raw AI/CAPA draft output bound to its context SHA-256."""

    model_config = ConfigDict(extra="forbid")

    raw_output_payload: Dict[str, Any]
    raw_output_sha256: str = Field(pattern=_SHA256_RE)
    context_sha256: str = Field(pattern=_SHA256_RE)
    guardrail_passed: bool
    guardrail_violations: List[str] = Field(default_factory=list)
    generated_utc: str

    @field_validator("generated_utc")
    @classmethod
    def _chk_ts(cls, v: str) -> str:
        return _validate_iso8601(v, "generated_utc")


class Part2HumanQaRedlines(BaseModel):
    """Part 2: Structured human QA review decision and field-level redlines/diffs."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(min_length=1, max_length=128)
    decision: Literal["APPROVED", "APPROVED_WITH_REDLINES", "REJECTED"]
    redline_diffs: List[FieldRedlineDiff] = Field(default_factory=list)
    approved_payload: Dict[str, Any]
    redline_sha256: str = Field(pattern=_SHA256_RE)
    reviewed_utc: str

    @field_validator("reviewer_id")
    @classmethod
    def _chk_human_reviewer(cls, v: str) -> str:
        return validate_human_qa_id(v, "reviewer_id")

    @field_validator("reviewed_utc")
    @classmethod
    def _chk_ts(cls, v: str) -> str:
        return _validate_iso8601(v, "reviewed_utc")


class Part3TwoFactorESig(BaseModel):
    """Part 3: 21 CFR Part 11 2-factor electronic signature metadata bound to Part 1 & Part 2."""

    model_config = ConfigDict(extra="forbid")

    signer_id: str = Field(min_length=1, max_length=128)
    signer_name: str = Field(min_length=1, max_length=128)
    signer_role: Literal["QA_APPROVER", "QA_LEAD", "QP_RELEASE"]
    signing_meaning: str = Field(min_length=3, max_length=256)
    factor_1_type: Literal["PASSWORD", "SSO_OIDC", "SMARTCARD_PIV"]
    factor_1_verified_utc: str
    factor_2_type: Literal["TOTP", "HARDWARE_TOKEN", "WEBAUTHN_FIDO2"]
    factor_2_challenge_id: str = Field(min_length=6, max_length=128)
    factor_2_verified_utc: str
    signed_utc: str
    signature_binding_sha256: str = Field(pattern=_SHA256_RE)

    @field_validator("signer_id")
    @classmethod
    def _chk_human_signer(cls, v: str) -> str:
        return validate_human_qa_id(v, "signer_id")

    @field_validator("factor_1_verified_utc", "factor_2_verified_utc", "signed_utc")
    @classmethod
    def _chk_esig_ts(cls, v: str, info) -> str:
        return _validate_iso8601(v, info.field_name)


class Part4AuditChainRecord(BaseModel):
    """Part 4: SHA-256 hash-chained audit record including LLM execution metadata."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=1)
    session_id: str = Field(min_length=1, max_length=64)
    decision_record_id: str = Field(min_length=4, max_length=64)
    utc_timestamp: str
    prev_entry_sha256: str = Field(pattern=_SHA256_RE)
    linked_twin_alcoa_sha256: str = Field(pattern=_SHA256_RE)
    llm_execution_metadata: LlmExecutionMetadata
    part1_sha256: str = Field(pattern=_SHA256_RE)
    part2_sha256: str = Field(pattern=_SHA256_RE)
    part3_sha256: str = Field(pattern=_SHA256_RE)
    entry_sha256: str = Field(pattern=_SHA256_RE)

    @field_validator("utc_timestamp")
    @classmethod
    def _chk_ts(cls, v: str) -> str:
        return _validate_iso8601(v, "utc_timestamp")


class Part11AiDecisionRecord(BaseModel):
    """Complete 4-part 21 CFR Part 11 AI decision record for CAPA governance."""

    model_config = ConfigDict(extra="forbid")

    decision_record_id: str = Field(min_length=4, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    status: Literal["APPROVED_AND_FINALIZED", "REJECTED_BY_QA"]
    finalized: bool = True
    part1_raw_model_output: Part1RawModelOutput
    part2_human_qa_redlines: Part2HumanQaRedlines
    part3_two_factor_esig: Part3TwoFactorESig
    part4_audit_chain_record: Part4AuditChainRecord


def _get_nested(payload: Dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_nested(payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = payload
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def compute_structured_redlines(
    raw_payload: Dict[str, Any],
    revised_overrides: Optional[Dict[str, Any]] = None,
    rationales: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[FieldRedlineDiff]]:
    """Apply QA overrides onto a deep copy of raw_payload and emit structured field-level diffs."""
    approved = copy.deepcopy(raw_payload)
    overrides = revised_overrides or {}
    rat_map = rationales or {}
    diffs: List[FieldRedlineDiff] = []

    for path in sorted(overrides.keys()):
        new_val = overrides[path]
        orig_val = _get_nested(raw_payload, path)
        _set_nested(approved, path, copy.deepcopy(new_val))
        if orig_val != new_val:
            diffs.append(
                FieldRedlineDiff(
                    field_path=path,
                    original_value=orig_val,
                    revised_value=new_val,
                    rationale=str(rat_map.get(path, "QA editorial/technical redline.")),
                )
            )
    return approved, diffs


def build_part1_raw_output(
    raw_output_payload: Dict[str, Any],
    context_sha256: str,
    guardrail_passed: bool,
    guardrail_violations: Optional[Sequence[str]] = None,
    generated_utc: Optional[str] = None,
) -> Part1RawModelOutput:
    """Create Part 1 immutable raw AI draft output."""
    raw_copy = copy.deepcopy(raw_output_payload)
    return Part1RawModelOutput(
        raw_output_payload=raw_copy,
        raw_output_sha256=canonical_sha256(raw_copy),
        context_sha256=str(context_sha256),
        guardrail_passed=bool(guardrail_passed),
        guardrail_violations=list(guardrail_violations or []),
        generated_utc=generated_utc or datetime.now(timezone.utc).isoformat(),
    )


def _compute_part2_redline_sha(
    reviewer_id: str,
    decision: str,
    diffs: Sequence[FieldRedlineDiff],
    approved_payload: Dict[str, Any],
) -> str:
    return canonical_sha256(
        {
            "reviewer_id": reviewer_id,
            "decision": decision,
            "redline_diffs": [d.model_dump() for d in diffs],
            "approved_payload": approved_payload,
        }
    )


def _compute_part3_binding_sha(
    part1_sha256: str,
    part2_sha256: str,
    signer_id: str,
    signer_role: str,
    signing_meaning: str,
    factor_1_type: str,
    factor_2_type: str,
    factor_2_challenge_id: str,
    signed_utc: str,
) -> str:
    return canonical_sha256(
        {
            "part1_sha256": part1_sha256,
            "part2_sha256": part2_sha256,
            "signer_id": signer_id,
            "signer_role": signer_role,
            "signing_meaning": signing_meaning,
            "factor_1_type": factor_1_type,
            "factor_2_type": factor_2_type,
            "factor_2_challenge_id": factor_2_challenge_id,
            "signed_utc": signed_utc,
        }
    )


def build_part11_decision_record(
    *,
    session_id: str,
    part1: Part1RawModelOutput,
    reviewer_id: str,
    decision: Literal["APPROVED", "APPROVED_WITH_REDLINES", "REJECTED"],
    revised_overrides: Optional[Dict[str, Any]],
    rationales: Optional[Dict[str, str]],
    signer_id: str,
    signer_name: str,
    signer_role: Literal["QA_APPROVER", "QA_LEAD", "QP_RELEASE"],
    signing_meaning: str,
    factor_1_type: Literal["PASSWORD", "SSO_OIDC", "SMARTCARD_PIV"],
    factor_2_type: Literal["TOTP", "HARDWARE_TOKEN", "WEBAUTHN_FIDO2"],
    factor_2_challenge_id: str,
    linked_twin_alcoa_sha256: str,
    llm_execution_metadata: LlmExecutionMetadata,
    seq: int = 1,
    prev_entry_sha256: str = GENESIS_PART11_SHA256,
    decision_record_id: Optional[str] = None,
) -> Part11AiDecisionRecord:
    """Assemble and cryptographically bind all 4 parts of a 21 CFR Part 11 AI decision record."""
    if not part1.guardrail_passed and decision != "REJECTED":
        raise ValueError(
            "Cannot QA-approve a CAPA draft that failed deterministic citation/numeric/schema guardrails."
        )
    if str(factor_1_type) == str(factor_2_type):
        raise ValueError("Two-factor e-signature requires distinct factor_1_type and factor_2_type.")

    now_utc = datetime.now(timezone.utc).isoformat()
    rec_id = decision_record_id or f"P11-{uuid4().hex[:12]}"

    approved_payload, diffs = compute_structured_redlines(
        part1.raw_output_payload,
        revised_overrides=revised_overrides,
        rationales=rationales,
    )
    eff_decision = decision
    if eff_decision == "APPROVED" and diffs:
        eff_decision = "APPROVED_WITH_REDLINES"

    part2 = Part2HumanQaRedlines(
        reviewer_id=reviewer_id,
        decision=eff_decision,  # type: ignore[arg-type]
        redline_diffs=diffs,
        approved_payload=approved_payload,
        redline_sha256=_compute_part2_redline_sha(reviewer_id, eff_decision, diffs, approved_payload),
        reviewed_utc=now_utc,
    )

    p1_sha = canonical_sha256(part1.model_dump())
    p2_sha = canonical_sha256(part2.model_dump())
    sig_bind = _compute_part3_binding_sha(
        p1_sha,
        p2_sha,
        signer_id,
        signer_role,
        signing_meaning,
        factor_1_type,
        factor_2_type,
        factor_2_challenge_id,
        now_utc,
    )
    part3 = Part3TwoFactorESig(
        signer_id=signer_id,
        signer_name=signer_name,
        signer_role=signer_role,
        signing_meaning=signing_meaning,
        factor_1_type=factor_1_type,
        factor_1_verified_utc=now_utc,
        factor_2_type=factor_2_type,
        factor_2_challenge_id=factor_2_challenge_id,
        factor_2_verified_utc=now_utc,
        signed_utc=now_utc,
        signature_binding_sha256=sig_bind,
    )
    p3_sha = canonical_sha256(part3.model_dump())

    part4_body = {
        "seq": int(seq),
        "session_id": session_id,
        "decision_record_id": rec_id,
        "utc_timestamp": now_utc,
        "prev_entry_sha256": prev_entry_sha256,
        "linked_twin_alcoa_sha256": linked_twin_alcoa_sha256,
        "llm_execution_metadata": llm_execution_metadata.model_dump(),
        "part1_sha256": p1_sha,
        "part2_sha256": p2_sha,
        "part3_sha256": p3_sha,
    }
    part4 = Part4AuditChainRecord(
        **part4_body,
        entry_sha256=canonical_sha256(part4_body),
    )

    status: Literal["APPROVED_AND_FINALIZED", "REJECTED_BY_QA"] = (
        "REJECTED_BY_QA" if eff_decision == "REJECTED" else "APPROVED_AND_FINALIZED"
    )
    return Part11AiDecisionRecord(
        decision_record_id=rec_id,
        session_id=session_id,
        status=status,
        finalized=True,
        part1_raw_model_output=part1,
        part2_human_qa_redlines=part2,
        part3_two_factor_esig=part3,
        part4_audit_chain_record=part4,
    )


def verify_part11_decision_record(
    record: Union[Part11AiDecisionRecord, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Verify integrity, 2FA binding, and SHA-256 digests across all 4 parts of a Part 11 record."""
    errors: List[str] = []
    try:
        rec = (
            record
            if isinstance(record, Part11AiDecisionRecord)
            else Part11AiDecisionRecord.model_validate(record)
        )
    except Exception as exc:
        return False, [f"Invalid Part 11 record schema: {exc}"]

    if rec.finalized is not True:
        errors.append("Part 11 decision record must have finalized=True.")

    p1 = rec.part1_raw_model_output
    p2 = rec.part2_human_qa_redlines
    p3 = rec.part3_two_factor_esig
    p4 = rec.part4_audit_chain_record

    expected_status = "REJECTED_BY_QA" if p2.decision == "REJECTED" else "APPROVED_AND_FINALIZED"
    if rec.status != expected_status:
        errors.append(
            f"Record status {rec.status!r} mismatches Part 2 human QA decision {p2.decision!r} (expected {expected_status!r})."
        )

    if canonical_sha256(p1.raw_output_payload) != p1.raw_output_sha256:
        errors.append("Part 1 raw_output_sha256 mismatch (raw output payload tampered).")
    if not p1.guardrail_passed and p2.decision != "REJECTED":
        errors.append("Part 1 guardrail_passed=False cannot accompany QA approval.")

    exp_redline_sha = _compute_part2_redline_sha(
        p2.reviewer_id, p2.decision, p2.redline_diffs, p2.approved_payload
    )
    if exp_redline_sha != p2.redline_sha256:
        errors.append("Part 2 redline_sha256 mismatch (QA redlines/approved_payload tampered).")

    p1_sha = canonical_sha256(p1.model_dump())
    p2_sha = canonical_sha256(p2.model_dump())
    p3_sha = canonical_sha256(p3.model_dump())

    if str(p3.factor_1_type) == str(p3.factor_2_type):
        errors.append("Part 3 2FA requires distinct factor_1_type and factor_2_type.")
    exp_sig_bind = _compute_part3_binding_sha(
        p1_sha,
        p2_sha,
        p3.signer_id,
        p3.signer_role,
        p3.signing_meaning,
        p3.factor_1_type,
        p3.factor_2_type,
        p3.factor_2_challenge_id,
        p3.signed_utc,
    )
    if exp_sig_bind != p3.signature_binding_sha256:
        errors.append("Part 3 signature_binding_sha256 mismatch (e-signature or bound content tampered).")

    if p4.session_id != rec.session_id or p4.decision_record_id != rec.decision_record_id:
        errors.append("Part 4 session_id/decision_record_id mismatch.")
    if p4.part1_sha256 != p1_sha:
        errors.append("Part 4 part1_sha256 mismatch.")
    if p4.part2_sha256 != p2_sha:
        errors.append("Part 4 part2_sha256 mismatch.")
    if p4.part3_sha256 != p3_sha:
        errors.append("Part 4 part3_sha256 mismatch.")

    p4_body = {k: v for k, v in p4.model_dump().items() if k != "entry_sha256"}
    if canonical_sha256(p4_body) != p4.entry_sha256:
        errors.append("Part 4 entry_sha256 mismatch (audit chain entry tampered).")

    return len(errors) == 0, errors


def verify_part11_decision_chain(
    records: Sequence[Union[Part11AiDecisionRecord, Dict[str, Any]]],
) -> Tuple[bool, List[str]]:
    """Verify every individual Part 11 record AND prev_entry_sha256 hash chaining across records."""
    if not records:
        return True, []
    errors: List[str] = []
    expected_prev = GENESIS_PART11_SHA256
    for idx, raw_rec in enumerate(records):
        ok_rec, rec_errs = verify_part11_decision_record(raw_rec)
        if not ok_rec:
            errors.extend(f"Record[{idx}]: {e}" for e in rec_errs)
            continue
        rec = (
            raw_rec
            if isinstance(raw_rec, Part11AiDecisionRecord)
            else Part11AiDecisionRecord.model_validate(raw_rec)
        )
        p4 = rec.part4_audit_chain_record
        if p4.seq != idx + 1:
            errors.append(f"Record[{idx}]: expected seq={idx + 1}, got {p4.seq}.")
        if p4.prev_entry_sha256 != expected_prev:
            errors.append(
                f"Record[{idx}]: broken prev_entry_sha256 link ({p4.prev_entry_sha256[:12]} != {expected_prev[:12]})."
            )
        expected_prev = p4.entry_sha256
    return len(errors) == 0, errors
