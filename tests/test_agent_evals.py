"""Verification and faithfulness evaluation tests for the CAPA Deviation Investigation graph."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agents.deviation_graph import run_deviation_investigation
from agents.observability import evaluate_capa_report
from api.main import create_app

pytestmark = pytest.mark.agent


@pytest.fixture()
def temp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test_capa_eval.db'}"


def test_deviation_graph_and_faithfulness_eval_offline(temp_db_url: str) -> None:
    """Verify 4-step CAPA graph, MCP tool trajectory, numerical grounding, and ALCOA+ linkage."""
    app = create_app(database_url=temp_db_url)
    client = TestClient(app)

    r_init = client.post(
        "/twins",
        json={"batch_id": "FB-1000L-CAPA-01", "operator_id": "qa_eng", "vessel_scale_L": 1000},
    )
    assert r_init.status_code == 201
    session_id = r_init.json()["session_id"]

    # Step 1: nominal PAT tick at 0.2h
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
    assert len(trace.node_spans) == 4
    assert [n.node_name for n in trace.node_spans] == [
        "detect_excursion",
        "run_counterfactual_controls",
        "retrieve_empirical_precedent",
        "synthesize_capa_report",
    ]
    assert len(report.cited_uhlendorff_campaigns) > 0

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


def test_fastapi_investigate_endpoint(temp_db_url: str) -> None:
    """Verify POST /twins/{session_id}/investigate returns CAPA report, trace, and eval=PASS."""
    app = create_app(database_url=temp_db_url)
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
    assert len(payload["trace"]["node_spans"]) == 4
    assert len(payload["capa_report"]["alcoa_record_sha256"]) == 64
