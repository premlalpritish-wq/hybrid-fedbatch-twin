"""Verification, guardrail, 21 CFR Part 11, and shadow UCR evaluation tests for CAPA workflows."""

from __future__ import annotations

import copy
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agents.deviation_graph import run_deviation_investigation, run_governed_capa_workflow
from agents.guardrails import GroundedClaim, validate_capa_draft
from agents.observability import evaluate_capa_report
from agents.part11 import verify_part11_decision_chain, verify_part11_decision_record
from agents.shadow_eval import (
    FaultInjectingCapaProvider,
    run_shadow_evaluation_suite,
    wilson_ci_95,
)
from api.main import create_app
import mcp_server.server as mcp_srv
from mcp_server.server import MCPPrivilegeError, clear_shadow_sandbox, stage_capa_draft

pytestmark = pytest.mark.agent


@pytest.fixture()
def temp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test_capa_eval.db'}"


def _cleanup_db(app, db_url: str) -> None:
    clear_shadow_sandbox(db_url)
    if db_url in mcp_srv._DB_CACHE:
        eng, _ = mcp_srv._DB_CACHE.pop(db_url)
        eng.dispose()
    if app is not None:
        app.state.db_engine.dispose()


def test_deviation_graph_and_faithfulness_eval_offline(temp_db_url: str) -> None:
    """Verify 5-step CAPA graph, MCP tool trajectory, numerical grounding, and ALCOA+ linkage."""
    app = create_app(database_url=temp_db_url)
    try:
        client = TestClient(app)

        r_init = client.post(
            "/twins",
            json={"batch_id": "FB-1000L-CAPA-01", "operator_id": "qa_eng", "vessel_scale_L": 1000},
        )
        assert r_init.status_code == 201
        session_id = r_init.json()["session_id"]

        r_pat = client.post(
            f"/twins/{session_id}/pat",
            json={
                "t_h": 0.2,
                "y_Xv": 1.6,
                "y_CL": 7.5,
                "y_V": 402.0,
                "F_act": 8.0,
                "rpm_act": 135.0,
                "vvm_act": 0.205,
            },
        )
        assert r_pat.status_code == 200
        live_audit_before = client.get(f"/twins/{session_id}/audit").json()

        # Run CAPA investigation with an aeration_drop disturbance & delayed LIMS assay
        report, trace = run_deviation_investigation(
            session_id=session_id,
            vessel_scale_L=1000,
            database_url=temp_db_url,
            pat_tick={
                "t_h": 0.6,
                "y_Xv": 1.85,
                "y_CL": 1.2,
                "y_V": 406.0,
                "F_act": 12.0,
                "rpm_act": 135.0,
                "vvm_act": 0.12,
            },
            delayed_lims={"t_sample_h": 0.2, "y_S": 0.45, "y_A": 0.65, "y_P": 0.12},
            disturbance_mode="aeration_drop",
            duration_h=6.0,
        )

        assert report.execution_mode == "deterministic"
        assert report.guardrail_passed is True
        assert report.qa_status == "PENDING_QA_APPROVAL"
        assert len(report.claims) >= 7
        assert len(trace.node_spans) == 5
        assert [n.node_name for n in trace.node_spans] == [
            "detect_excursion",
            "retrieve_evidence",
            "draft_capa_report",
            "validate_capa_guardrail",
            "human_qa_approval_gate",
        ]
        assert len(report.cited_uhlendorff_campaigns) > 0

        # Shadow investigation did not mutate live DB twin state
        live_audit_after = client.get(f"/twins/{session_id}/audit").json()
        assert live_audit_after["record_sha256"] == live_audit_before["record_sha256"]
        assert live_audit_after["audit_log_count"] == live_audit_before["audit_log_count"]

        eval_res = evaluate_capa_report(report, trace, database_url=temp_db_url)
        assert eval_res.overall_passed is True
        assert eval_res.overall_score == pytest.approx(1.0)
        assert eval_res.tool_trajectory_completeness.passed is True
        assert eval_res.numerical_grounding_faithfulness.passed is True
        assert eval_res.alcoa_traceability.passed is True

        # Adversarial check: tampered metric or SHA-256 must fail faithfulness/ALCOA+ checks
        tampered = report.model_copy(
            update={
                "d_m": report.d_m + 42.0,
                "alcoa_record_sha256": "0" * 64,
            }
        )
        eval_bad = evaluate_capa_report(tampered, trace, database_url=temp_db_url)
        assert eval_bad.overall_passed is False
        assert eval_bad.numerical_grounding_faithfulness.passed is False
        assert eval_bad.alcoa_traceability.passed is False
    finally:
        _cleanup_db(app, temp_db_url)


def test_fastapi_investigate_endpoint(temp_db_url: str) -> None:
    """Verify POST /twins/{session_id}/investigate returns CAPA report, trace, guardrails, and staged draft."""
    app = create_app(database_url=temp_db_url)
    try:
        client = TestClient(app)

        r_init = client.post(
            "/twins",
            json={"batch_id": "FB-5L-INV-01", "operator_id": "op_blr", "vessel_scale_L": 5},
        )
        assert r_init.status_code == 201
        session_id = r_init.json()["session_id"]

        r_inv = client.post(
            f"/twins/{session_id}/investigate",
            json={
                "pat_tick": {
                    "t_h": 0.4,
                    "y_Xv": 1.75,
                    "y_CL": 6.8,
                    "y_V": 2.04,
                    "F_act": 0.05,
                    "rpm_act": 450.0,
                    "vvm_act": 1.2,
                },
                "disturbance_mode": "none",
                "duration_h": 4.0,
            },
        )
        assert r_inv.status_code == 200
        payload = r_inv.json()
        assert payload["evaluation"]["overall_passed"] is True
        assert payload["evaluation"]["overall_score"] == pytest.approx(1.0)
        assert payload["guardrail_result"]["passed"] is True
        assert payload["staged_draft"]["status"] == "DRAFT_PENDING_QA_REVIEW"
        assert len(payload["trace"]["node_spans"]) == 5
        assert len(payload["capa_report"]["alcoa_record_sha256"]) == 64
    finally:
        _cleanup_db(app, temp_db_url)


def test_deterministic_guardrails_reject_ungrounded_and_mismatched_drafts(temp_db_url: str) -> None:
    """Verify deterministic guardrails block drafts with missing citations, bad numbers, or incomplete schemas."""
    app = create_app(database_url=temp_db_url)
    try:
        client = TestClient(app)
        r_init = client.post(
            "/twins",
            json={"batch_id": "FB-GRD-01", "operator_id": "qa_eng", "vessel_scale_L": 5},
        )
        sid = r_init.json()["session_id"]
        client.post(
            f"/twins/{sid}/pat",
            json={"t_h": 0.2, "y_Xv": 1.6, "y_CL": 7.5, "y_V": 2.01, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2},
        )

        for mode in (
            "missing_citation",
            "invented_campaign",
            "claim_numeric_mismatch",
            "incomplete_schema",
            "alcoa_mismatch",
        ):
            outcome = run_governed_capa_workflow(
                session_id=sid,
                vessel_scale_L=5,
                database_url=temp_db_url,
                pat_tick={"t_h": 0.4, "y_Xv": 1.75, "y_CL": 1.2, "y_V": 2.02, "F_act": 0.05, "rpm_act": 450.0, "vvm_act": 1.2},
                duration_h=4.0,
                provider=FaultInjectingCapaProvider(fault_mode=mode),  # type: ignore[arg-type]
            )
            assert outcome.guardrail_result.passed is False
            assert outcome.capa_report.guardrail_passed is False
            assert outcome.capa_report.qa_status == "GUARDRAIL_REJECTED"
            assert len(outcome.capa_report.guardrail_violations) > 0
            # Conditional edge routes rejected drafts straight to END without touching QA gate
            assert [n.node_name for n in outcome.trace.node_spans] == [
                "detect_excursion",
                "retrieve_evidence",
                "draft_capa_report",
                "validate_capa_guardrail",
            ]
            assert outcome.staged_draft is None
            assert outcome.part11_record is None

        # Direct guardrail bypass vectors: repeated narrative mentions, omitted dict keys, ungrounded claim prose
        ok_outcome = run_governed_capa_workflow(
            session_id=sid,
            vessel_scale_L=5,
            database_url=temp_db_url,
            pat_tick={"t_h": 0.4, "y_Xv": 1.75, "y_CL": 1.2, "y_V": 2.02, "F_act": 0.05, "rpm_act": 450.0, "vvm_act": 1.2},
            duration_h=4.0,
        )
        base_dict = ok_outcome.capa_report.model_dump()
        ctx = ok_outcome.evidence_context

        rep_narr = copy.deepcopy(base_dict)
        rep_narr["narrative_summary"] += " Secondary note: D_M=999.000."
        res_narr = validate_capa_draft(rep_narr, ctx)
        assert res_narr.passed is False
        assert any(v.code == "NARRATIVE_NUMERIC_MISMATCH" for v in res_narr.violations)

        rep_omit = copy.deepcopy(base_dict)
        for omit_k in ("nis", "spe", "in_domain"):
            rep_omit.pop(omit_k, None)
        res_omit = validate_capa_draft(rep_omit, ctx)
        assert res_omit.passed is False
        assert any(v.code == "INCOMPLETE_SCHEMA" for v in res_omit.violations)

        rep_stmt = copy.deepcopy(base_dict)
        rep_stmt["claims"][0] = GroundedClaim(
            claim_id="CLM-PROSE-FAKE",
            statement="Observed Mahalanobis D_M=999.000 within limits.",
            citation_ref="ekf:gate_status",
        ).model_dump()
        res_stmt = validate_capa_draft(rep_stmt, ctx)
        assert res_stmt.passed is False
        assert any(v.code == "CLAIM_STATEMENT_MISMATCH" for v in res_stmt.violations)
    finally:
        _cleanup_db(app, temp_db_url)


def test_part11_4part_ai_decision_record_redlines_2fa_and_tamper_resistance(temp_db_url: str) -> None:
    """Verify 4-part 21 CFR Part 11 record creation, QA redlines, 2FA e-sig, immutability, and tamper detection."""
    app = create_app(database_url=temp_db_url)
    try:
        client = TestClient(app)
        r_init = client.post(
            "/twins",
            json={"batch_id": "FB-P11-01", "operator_id": "qa_eng", "vessel_scale_L": 5},
        )
        sid = r_init.json()["session_id"]
        client.post(
            f"/twins/{sid}/pat",
            json={"t_h": 0.2, "y_Xv": 1.6, "y_CL": 7.5, "y_V": 2.01, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2},
        )

        # 1. Investigate -> stages draft awaiting QA review
        r_inv = client.post(
            f"/twins/{sid}/investigate",
            json={
                "pat_tick": {"t_h": 0.4, "y_Xv": 1.75, "y_CL": 1.1, "y_V": 2.03, "F_act": 0.05, "rpm_act": 450.0, "vvm_act": 1.2},
                "disturbance_mode": "aeration_drop",
                "duration_h": 4.0,
            },
        )
        assert r_inv.status_code == 200
        inv = r_inv.json()
        draft_id = inv["staged_draft"]["draft_id"]

        # 2. Reject invalid 2FA (agent signer_id or invalid role)
        r_bad_signer = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "staged_draft_id": draft_id,
                "redlines": {"reviewer_id": "qa_lead_blr", "decision": "APPROVED"},
                "two_factor_esig": {
                    "signer_id": "mcp_agent",
                    "signer_name": "Autonomous Agent",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "Attempted automated approval",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "WEBAUTHN_FIDO2",
                    "factor_2_challenge_id": "chal-001122",
                },
            },
        )
        assert r_bad_signer.status_code == 422

        # Reject automated agent reviewer_id as well
        r_bad_reviewer = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "staged_draft_id": draft_id,
                "redlines": {"reviewer_id": "mcp_agent", "decision": "APPROVED"},
                "two_factor_esig": {
                    "signer_id": "qa_lead_blr",
                    "signer_name": "Dr. Ananya Rao",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "QA Deviation CAPA Approval",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "WEBAUTHN_FIDO2",
                    "factor_2_challenge_id": "chal-bad-rev-01",
                },
            },
        )
        assert r_bad_reviewer.status_code == 422

        # Reject forged direct capa_report, forged MCP-staged draft, and forged client evidence_context
        live_alcoa_sha = client.get(f"/twins/{sid}/audit").json()["record_sha256"]
        forged_report = {
            "session_id": sid,
            "vessel_scale_L": 5,
            "root_cause_classification": "IN_DOMAIN_NOMINAL",
            "severity": "INFO",
            "d_m": 99999.0,
            "nis": 0.0,
            "spe": 0.0,
            "in_domain": True,
            "recommended_action": "Fabricated action.",
            "counterfactual_comparison": inv["capa_report"]["counterfactual_comparison"],
            "cited_uhlendorff_campaigns": ["EXP999"],
            "alcoa_record_sha256": live_alcoa_sha,
            "alcoa_audit_log_count": 2,
            "narrative_summary": "Forged report claiming D_M=99999.000.",
            "claims": [{"claim_id": "FAKE-1", "statement": "Fake", "citation_ref": ""}],
            "guardrail_passed": True,
            "context_sha256": inv["capa_report"]["context_sha256"],
        }
        r_forged_direct = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "capa_report": forged_report,
                "redlines": {"reviewer_id": "qa_lead_blr", "decision": "APPROVED"},
                "two_factor_esig": {
                    "signer_id": "qa_lead_blr",
                    "signer_name": "Dr. Ananya Rao",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "Attempt forged approval",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "WEBAUTHN_FIDO2",
                    "factor_2_challenge_id": "chal-forged-01",
                },
            },
        )
        assert r_forged_direct.status_code == 422

        forged_staged = stage_capa_draft(
            session_id=sid,
            draft_report=forged_report,
            database_url=temp_db_url,
        )
        r_forged_staged = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "staged_draft_id": forged_staged["draft_id"],
                "redlines": {"reviewer_id": "qa_lead_blr", "decision": "APPROVED"},
                "two_factor_esig": {
                    "signer_id": "qa_lead_blr",
                    "signer_name": "Dr. Ananya Rao",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "Attempt forged staged approval",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "WEBAUTHN_FIDO2",
                    "factor_2_challenge_id": "chal-forged-02",
                },
            },
        )
        assert r_forged_staged.status_code == 422

        r_forged_ctx = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "capa_report": {**forged_report, "context_sha256": "a" * 64},
                "evidence_context": {
                    "telemetry_metrics": {"d_m": 99999.0},
                    "observed_booleans": {"in_domain": True},
                    "observed_campaigns": ["EXP999"],
                    "allowed_citations": ["ekf:gate_status"],
                    "alcoa_record_sha256": live_alcoa_sha,
                    "context_sha256": "a" * 64,
                },
                "redlines": {"reviewer_id": "qa_lead_blr", "decision": "APPROVED"},
                "two_factor_esig": {
                    "signer_id": "qa_lead_blr",
                    "signer_name": "Dr. Ananya Rao",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "Attempt forged context approval",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "WEBAUTHN_FIDO2",
                    "factor_2_challenge_id": "chal-forged-03",
                },
            },
        )
        assert r_forged_ctx.status_code == 422

        # 3. Approve with structured QA redlines + valid human 2FA e-signature (Record #1)
        r_ok1 = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "decision_record_id": "P11-REC-001",
                "staged_draft_id": draft_id,
                "redlines": {
                    "reviewer_id": "qa_lead_blr",
                    "decision": "APPROVED",
                    "revised_overrides": {
                        "recommended_action": "Hold feed <= 0.02 L/h, inspect sparger line, and pull confirmatory LIMS glucose/acetate sample."
                    },
                    "rationales": {
                        "recommended_action": "Added explicit sparger line inspection per SOP-MFG-042."
                    },
                },
                "two_factor_esig": {
                    "signer_id": "qa_lead_blr",
                    "signer_name": "Dr. Ananya Rao",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "QA Deviation CAPA Approval & Release",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "WEBAUTHN_FIDO2",
                    "factor_2_challenge_id": "fido2-chal-998877",
                },
            },
        )
        assert r_ok1.status_code == 200
        res1 = r_ok1.json()
        assert res1["chain_valid"] is True
        rec1 = res1["decision_record"]
        assert rec1["finalized"] is True
        assert rec1["status"] == "APPROVED_AND_FINALIZED"
        # Part 1 keeps raw model output immutable; Part 2 captures structured redline diff
        assert rec1["part1_raw_model_output"]["raw_output_payload"]["recommended_action"] != (
            rec1["part2_human_qa_redlines"]["approved_payload"]["recommended_action"]
        )
        assert rec1["part2_human_qa_redlines"]["decision"] == "APPROVED_WITH_REDLINES"
        assert len(rec1["part2_human_qa_redlines"]["redline_diffs"]) == 1
        assert rec1["part2_human_qa_redlines"]["redline_diffs"][0]["field_path"] == "recommended_action"
        assert rec1["part3_two_factor_esig"]["factor_1_type"] == "SSO_OIDC"
        assert rec1["part3_two_factor_esig"]["factor_2_type"] == "WEBAUTHN_FIDO2"
        assert rec1["part4_audit_chain_record"]["llm_execution_metadata"]["provider"] == "deterministic_ci_provider"

        # 4. Finalized immutability: HTTP 409 & MCPPrivilegeError on overwrite attempts
        r_dup = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "decision_record_id": "P11-REC-001",
                "staged_draft_id": draft_id,
                "redlines": {"reviewer_id": "qa_lead_blr", "decision": "APPROVED"},
                "two_factor_esig": {
                    "signer_id": "qa_lead_blr",
                    "signer_name": "Dr. Ananya Rao",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "Duplicate overwrite attempt",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "TOTP",
                    "factor_2_challenge_id": "totp-chal-112233",
                },
            },
        )
        assert r_dup.status_code == 409

        # 4b. Single-use 2FA challenge replay prevention -> HTTP 409
        r_replay_chal = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "decision_record_id": "P11-REC-REPLAY",
                "staged_draft_id": draft_id,
                "redlines": {"reviewer_id": "qa_lead_blr", "decision": "APPROVED"},
                "two_factor_esig": {
                    "signer_id": "qa_lead_blr",
                    "signer_name": "Dr. Ananya Rao",
                    "signer_role": "QA_LEAD",
                    "signing_meaning": "Replay 2FA challenge attempt",
                    "factor_1_type": "SSO_OIDC",
                    "factor_2_type": "WEBAUTHN_FIDO2",
                    "factor_2_challenge_id": "fido2-chal-998877",
                },
            },
        )
        assert r_replay_chal.status_code == 409

        with pytest.raises(MCPPrivilegeError):
            stage_capa_draft(
                session_id=sid,
                draft_report={"summary": "Attempt overwrite finalized QA"},
                target_decision_record_id="P11-REC-001",
                database_url=temp_db_url,
            )

        # 5. Append second decision record (Record #2) and verify full hash chain
        r_ok2 = client.post(
            f"/twins/{sid}/qa/approve",
            json={
                "decision_record_id": "P11-REC-002",
                "staged_draft_id": draft_id,
                "redlines": {"reviewer_id": "qp_rel_blr", "decision": "REJECTED"},
                "two_factor_esig": {
                    "signer_id": "qp_rel_blr",
                    "signer_name": "Vikram Mehta",
                    "signer_role": "QP_RELEASE",
                    "signing_meaning": "Superseded CAPA disposition",
                    "factor_1_type": "SMARTCARD_PIV",
                    "factor_2_type": "HARDWARE_TOKEN",
                    "factor_2_challenge_id": "piv-hw-445566",
                },
            },
        )
        assert r_ok2.status_code == 200
        rec2 = r_ok2.json()["decision_record"]
        assert rec2["status"] == "REJECTED_BY_QA"
        assert rec2["part4_audit_chain_record"]["seq"] == 2
        assert rec2["part4_audit_chain_record"]["prev_entry_sha256"] == rec1["part4_audit_chain_record"]["entry_sha256"]

        r_list = client.get(f"/twins/{sid}/qa/records")
        assert r_list.status_code == 200
        q_list = r_list.json()
        assert q_list["record_count"] == 2
        assert q_list["chain_valid"] is True

        # 6. Tamper checks across all 4 parts
        assert verify_part11_decision_record(rec1) == (True, [])
        assert verify_part11_decision_chain([rec1, rec2]) == (True, [])

        bad_p1 = copy.deepcopy(rec1)
        bad_p1["part1_raw_model_output"]["raw_output_payload"]["severity"] = "INFO"
        assert verify_part11_decision_record(bad_p1)[0] is False

        bad_p2 = copy.deepcopy(rec1)
        bad_p2["part2_human_qa_redlines"]["approved_payload"]["recommended_action"] = "Tampered"
        assert verify_part11_decision_record(bad_p2)[0] is False

        bad_p3 = copy.deepcopy(rec1)
        bad_p3["part3_two_factor_esig"]["factor_2_challenge_id"] = "tampered-chal"
        assert verify_part11_decision_record(bad_p3)[0] is False

        bad_p4 = copy.deepcopy(rec1)
        bad_p4["part4_audit_chain_record"]["llm_execution_metadata"]["model_id"] = "tampered-model"
        assert verify_part11_decision_record(bad_p4)[0] is False

        bad_chain = copy.deepcopy([rec1, rec2])
        bad_chain[1]["part4_audit_chain_record"]["prev_entry_sha256"] = "a" * 64
        assert verify_part11_decision_chain(bad_chain)[0] is False

        bad_status = copy.deepcopy(rec2)
        bad_status["status"] = "APPROVED_AND_FINALIZED"
        assert verify_part11_decision_record(bad_status)[0] is False
    finally:
        _cleanup_db(app, temp_db_url)


def test_shadow_evaluation_harness_ucr_and_wilson_ci_offline(
    temp_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify zero-secret deterministic CI provider + shadow UCR & Wilson 95% CI across 8 excursion cases."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Unit checks for Wilson 95% CI math
    assert wilson_ci_95(0, 0) == (0.0, 1.0)
    lo_0, hi_0 = wilson_ci_95(0, 50)
    assert lo_0 == 0.0 and 0.05 < hi_0 < 0.10
    lo_half, hi_half = wilson_ci_95(25, 50)
    assert 0.35 < lo_half < 0.50 < hi_half < 0.65

    # Clean deterministic provider -> 0% UCR across all 8 scenarios, zero live mutations
    summary_ok = run_shadow_evaluation_suite(database_url=temp_db_url)
    assert summary_ok.n_cases == 8
    assert summary_ok.n_guardrail_passed == 8
    assert summary_ok.n_guardrail_rejected == 0
    assert summary_ok.total_claims >= 40
    assert summary_ok.unsupported_claims == 0
    assert summary_ok.ucr == 0.0
    assert summary_ok.ucr_ci_95_low == 0.0
    assert 0.0 < summary_ok.ucr_ci_95_high < 0.15
    assert summary_ok.any_live_mutation is False

    # Fault-injecting provider -> UCR > 0 with non-zero 95% CI and 100% guardrail catch rate
    summary_fault = run_shadow_evaluation_suite(
        database_url=temp_db_url,
        provider=FaultInjectingCapaProvider(fault_mode="mixed"),
    )
    assert summary_fault.n_cases == 8
    assert summary_fault.n_guardrail_rejected == 8
    assert summary_fault.n_guardrail_passed == 0
    assert summary_fault.unsupported_claims > 0
    assert summary_fault.ucr > 0.0
    assert 0.0 < summary_fault.ucr_ci_95_low < summary_fault.ucr < summary_fault.ucr_ci_95_high <= 1.0
    assert summary_fault.any_live_mutation is False
