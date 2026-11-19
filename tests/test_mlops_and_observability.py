"""Offline tests for Medallion Bronze->Silver->Gold ETL, Prefect/MLflow-compatible registry, and OTel/Prometheus hooks."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import numpy as np
import pytest

from agents.deviation_graph import run_governed_capa_workflow
from agents.shadow_eval import FaultInjectingCapaProvider
from api.main import create_app
import mcp_server.server as mcp_srv
from mcp_server.server import MCPPrivilegeError, clear_shadow_sandbox, ingest_pat_and_estimate_state
from mlops.medallion import _read_jsonl, run_medallion_pipeline, verify_medallion_lineage
from mlops.registry import (
    LocalMlflowCompatibleTracker,
    PromotionCriteria,
    run_offline_training_and_registration_flow,
)
from mlops.telemetry import get_telemetry_recorder, reset_telemetry

pytestmark = pytest.mark.mlops


def _build_mixed_medallion_records():
    valid = [
        # Batch B01 (train)
        {"batch_id": "B01", "t_h": 0.0, "X_v": 1.5, "S": 1.2, "C_L": 7.8, "A": 0.05, "P": 0.0, "V": 2.0, "F": 0.02, "rpm": 450.0, "vvm": 1.2},
        {"batch_id": "B01", "t_h": 0.5, "X_v": 2.1, "S": 0.8, "C_L": 7.2, "A": 0.10, "P": 0.2, "V": 2.05, "F": 0.04, "rpm": 450.0, "vvm": 1.2},
        # Batch B02 (train)
        {"batch_id": "B02", "t_h": 0.0, "X_v": 1.6, "S": 1.1, "C_L": 7.7, "A": 0.06, "P": 0.0, "V": 20.0, "F": 0.20, "rpm": 265.0, "vvm": 0.56},
        {"batch_id": "B02", "t_h": 0.5, "X_v": 2.3, "S": 0.7, "C_L": 7.0, "A": 0.12, "P": 0.3, "V": 20.2, "F": 0.40, "rpm": 265.0, "vvm": 0.56},
        # Batch B03 (train)
        {"batch_id": "B03", "t_h": 0.0, "X_v": 1.5, "S": 1.2, "C_L": 7.9, "A": 0.04, "P": 0.0, "V": 400.0, "F": 8.0, "rpm": 135.0, "vvm": 0.205},
        {"batch_id": "B03", "t_h": 1.0, "X_v": 2.6, "S": 0.6, "C_L": 6.9, "A": 0.15, "P": 0.5, "V": 408.0, "F": 10.0, "rpm": 135.0, "vvm": 0.205},
        # Batch B04 (holdout)
        {"batch_id": "B04", "t_h": 0.0, "X_v": 1.55, "S": 1.15, "C_L": 7.8, "A": 0.05, "P": 0.0, "V": 2.0, "F": 0.03, "rpm": 450.0, "vvm": 1.2},
        {"batch_id": "B04", "t_h": 0.5, "X_v": 2.25, "S": 0.75, "C_L": 7.1, "A": 0.11, "P": 0.25, "V": 2.04, "F": 0.05, "rpm": 450.0, "vvm": 1.2},
    ]
    invalid = [
        # 1. Missing required column C_L
        {"batch_id": "B01", "t_h": 1.0, "X_v": 2.4, "V": 2.08, "F": 0.04, "rpm": 450.0, "vvm": 1.2},
        # 2. Non-finite NaN value
        {"batch_id": "B01", "t_h": 1.5, "X_v": float("nan"), "C_L": 7.0, "V": 2.1, "F": 0.04, "rpm": 450.0, "vvm": 1.2},
        # 3. Out of physical bounds (C_L = 99.0 > 25.0 and V = 0.0)
        {"batch_id": "B02", "t_h": 1.0, "X_v": 2.5, "C_L": 99.0, "V": 0.0, "F": 0.4, "rpm": 265.0, "vvm": 0.56},
        # 4. Non-monotonic timestamp in B01 (0.2 < 0.5)
        {"batch_id": "B01", "t_h": 0.2, "X_v": 1.8, "C_L": 7.4, "V": 2.02, "F": 0.04, "rpm": 450.0, "vvm": 1.2},
        # 5. Duplicate timestamp in B04 (0.5 == 0.5)
        {"batch_id": "B04", "t_h": 0.5, "X_v": 2.25, "S": 0.75, "C_L": 7.1, "A": 0.11, "P": 0.25, "V": 2.04, "F": 0.05, "rpm": 450.0, "vvm": 1.2},
        # 6. Oracle truth leakage (u_plant, including nested payload)
        {"batch_id": "B03", "t_h": 2.0, "X_v": 3.1, "C_L": 6.5, "V": 415.0, "u_plant": {"F": 10.0}},
        {"batch_id": "B03", "t_h": 2.5, "X_v": 3.2, "C_L": 6.4, "V": 416.0, "meta": {"u_plant": {"F": 10.0}}},
        # 7. Malformed non-dict record
        "not_a_json_object",
    ]
    return valid + invalid, len(valid), len(invalid)


def test_medallion_bronze_silver_gold_quarantine_and_lineage_tamper_detection(tmp_path: Path) -> None:
    """Verify Bronze->Silver->Gold DQ gates, zero silent drops, batch split safety, and lineage tamper detection."""
    reset_telemetry()
    records, n_valid, n_invalid = _build_mixed_medallion_records()
    med_dir = tmp_path / "medallion"

    out = run_medallion_pipeline(
        records=records,
        output_dir=med_dir,
        run_id="run_dq_001",
        holdout_batches=["B04"],
        source_uri="unit_test://synthetic_mixed",
    )
    manifest = out["manifest"]
    counts = manifest["row_counts"]

    # Strict row conservation: zero silent drops
    assert counts["bronze"] == n_valid + n_invalid
    assert counts["silver"] == n_valid
    assert counts["quarantine"] == n_invalid
    assert counts["silver"] + counts["quarantine"] == counts["bronze"]
    assert counts["gold"] == n_valid
    assert counts["train"] + counts["holdout"] == n_valid

    # Batch-grouped split has zero leakage
    assert set(manifest["train_batches"]) == {"B01", "B02", "B03"}
    assert set(manifest["holdout_batches"]) == {"B04"}
    train_rows = _read_jsonl(Path(manifest["train_split_path"]))
    holdout_rows = _read_jsonl(Path(manifest["holdout_split_path"]))
    assert {r["batch_id"] for r in train_rows}.isdisjoint({r["batch_id"] for r in holdout_rows})

    # Verify every invalid record has explicit quarantine reason codes
    q_rows = _read_jsonl(Path(manifest["quarantine_path"]))
    assert len(q_rows) == n_invalid
    all_reasons = {code for row in q_rows for code in row["reason_codes"]}
    assert {
        "MISSING_REQUIRED_COLUMN",
        "NON_FINITE_VALUE",
        "OUT_OF_PHYSICAL_BOUNDS",
        "NON_MONOTONIC_TIMESTAMP",
        "DUPLICATE_TIMESTAMP",
        "FORBIDDEN_ORACLE_FIELD",
        "MALFORMED_RECORD",
    }.issubset(all_reasons)

    # Verify Gold analytical features
    b01_rows = [r for r in train_rows if r["batch_id"] == "B01"]
    assert b01_rows[1]["DO_pct"] == pytest.approx(100.0 * 7.2 / 8.2, rel=1e-4)
    assert b01_rows[1]["dilution_rate_D"] == pytest.approx(0.04 / 2.05, rel=1e-4)
    assert b01_rows[1]["dXv_dt"] == pytest.approx((2.1 - 1.5) / 0.5, rel=1e-4)

    ok_lin, errs_lin = verify_medallion_lineage(out["manifest_path"])
    assert ok_lin is True and errs_lin == []

    # Tamper detection on Silver, Quarantine, and Manifest
    tamper_dir = tmp_path / "medallion_tampered"
    shutil.copytree(med_dir, tamper_dir)
    # Fix path pointers inside copied manifest to tamper_dir and recompute manifest_sha256
    t_manifest_path = tamper_dir / "gold" / "run_dq_001_manifest.json"
    m_obj = json.loads(t_manifest_path.read_text(encoding="utf-8"))
    for k in (
        "bronze_path",
        "silver_path",
        "quarantine_path",
        "gold_path",
        "train_split_path",
        "holdout_split_path",
    ):
        m_obj[k] = str(Path(m_obj[k]).parent.parent / Path(m_obj[k]).parent.name / Path(m_obj[k]).name).replace(
            str(med_dir), str(tamper_dir)
        )
    # Tamper directly on original silver file in tamper_dir without updating manifest
    s_file = tamper_dir / "silver" / "run_dq_001.jsonl"
    s_lines = s_file.read_text(encoding="utf-8").splitlines()
    s_first = json.loads(s_lines[0])
    s_first["normalized_payload"]["X_v"] = 99.99
    s_lines[0] = json.dumps(s_first, sort_keys=True)
    s_file.write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    t_manifest_path.write_text(json.dumps(m_obj, indent=2), encoding="utf-8")
    ok_bad, errs_bad = verify_medallion_lineage(t_manifest_path)
    assert ok_bad is False and len(errs_bad) > 0


def test_offline_prefect_mlflow_training_registry_promotion_and_rejection(tmp_path: Path) -> None:
    """Verify offline training/registry workflow hashes datasets, evaluates metrics, and gates promotions."""
    records, _, _ = _build_mixed_medallion_records()
    med_dir = tmp_path / "medallion_reg"
    med_out = run_medallion_pipeline(records, med_dir, run_id="reg_001", holdout_batches=["B04"])
    manifest_path = med_out["manifest_path"]
    tracking_dir = tmp_path / "mlops_tracking"

    # 1. Verified calibrated run -> PROMOTED_STAGING (requires human QA for PRODUCTION)
    flow_ok = run_offline_training_and_registration_flow(
        manifest_path=manifest_path,
        tracking_dir=tracking_dir,
        run_name="baseline_gam86_b15",
    )
    assert flow_ok["lineage"]["lineage_valid"] is True
    assert flow_ok["lineage"]["train_split_sha256"] == med_out["manifest"]["train_split_sha256"]
    assert flow_ok["lineage"]["holdout_split_sha256"] == med_out["manifest"]["holdout_split_sha256"]
    assert flow_ok["metrics"]["kd_r2"] == pytest.approx(0.73204, abs=1e-4)
    assert flow_ok["metrics"]["uhlendorff_score"] == pytest.approx(8.61)
    assert flow_ok["metrics"]["holdout_eval_samples"] == 2
    assert flow_ok["metrics"]["holdout_non_finite_outputs"] == 0
    assert flow_ok["promotion"]["passed"] is True
    assert flow_ok["promotion"]["stage"] == "PROMOTED_STAGING"
    assert flow_ok["promotion"]["requires_human_qa_for_production"] is True
    assert flow_ok["ledger_valid"] is True
    assert len(flow_ok["task_trace"]) == 3

    # 2. Degraded model candidate -> REJECTED with explicit failed criteria
    flow_bad = run_offline_training_and_registration_flow(
        manifest_path=manifest_path,
        tracking_dir=tracking_dir,
        run_name="degraded_candidate",
        candidate_metrics_override={"kd_r2": 0.42, "uhlendorff_score": 6.1, "ekf_nans": 3},
        criteria=PromotionCriteria(),
    )
    assert flow_bad["promotion"]["passed"] is False
    assert flow_bad["promotion"]["stage"] == "REJECTED"
    failed_text = " ".join(flow_bad["promotion"]["failed_criteria"])
    assert "kd_r2=" in failed_text
    assert "uhlendorff_score=" in failed_text
    assert "ekf_nans=" in failed_text
    assert flow_bad["ledger_valid"] is True

    # 3. Tampered run manifest JSON detected even if registry_ledger.jsonl is untouched
    tracker = LocalMlflowCompatibleTracker(tracking_dir)
    assert tracker.verify_registry_ledger() == (True, [])
    run_file = Path(flow_ok["run_path"])
    orig_run_text = run_file.read_text(encoding="utf-8")
    tampered_run = json.loads(orig_run_text)
    tampered_run["stage"] = "PRODUCTION"
    tampered_run["metrics"]["kd_r2"] = 0.01
    run_file.write_text(json.dumps(tampered_run, indent=2), encoding="utf-8")
    ok_run_tamper, errs_run_tamper = tracker.verify_registry_ledger()
    assert ok_run_tamper is False and len(errs_run_tamper) > 0
    run_file.write_text(orig_run_text, encoding="utf-8")

    # 4. Tampered registry ledger detected
    ledger_lines = tracker.ledger_path.read_text(encoding="utf-8").splitlines()
    entry0 = json.loads(ledger_lines[0])
    entry0["stage"] = "PRODUCTION"
    ledger_lines[0] = json.dumps(entry0, sort_keys=True)
    tracker.ledger_path.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
    ok_ledger, errs_ledger = tracker.verify_registry_ledger()
    assert ok_ledger is False and len(errs_ledger) > 0


def test_opentelemetry_prometheus_telemetry_hooks_and_metrics_endpoint(tmp_path: Path) -> None:
    """Verify real runtime telemetry hooks populate Prometheus /metrics and offline OTel spans."""
    reset_telemetry()
    db_url = f"sqlite:///{tmp_path / 'telemetry.db'}"
    app = create_app(database_url=db_url)
    try:
        client = TestClient(app)
        r_init = client.post(
            "/twins",
            json={"batch_id": "FB-TEL-01", "operator_id": "tel_eng", "vessel_scale_L": 5},
        )
        sid = r_init.json()["session_id"]

        # Normal PAT tick + extreme OOD/DO-crash PAT tick -> triggers EKF breach + control fallback
        client.post(
            f"/twins/{sid}/pat",
            json={"t_h": 0.2, "y_Xv": 1.6, "y_CL": 7.5, "y_V": 2.01, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2},
        )
        client.post(
            f"/twins/{sid}/pat",
            json={"t_h": 0.4, "y_Xv": 60.0, "y_CL": 1.1, "y_V": 2.03, "F_act": 0.05, "rpm_act": 450.0, "vvm_act": 1.2},
        )
        r_ctrl = client.post(f"/twins/{sid}/control", json={"t_h": 0.4, "mu_target": 0.28})
        assert r_ctrl.json()["fallback_active"] is True

        # Trigger MCP privilege rejection
        with pytest.raises(MCPPrivilegeError):
            ingest_pat_and_estimate_state(
                t_h=0.6,
                y_Xv=1.8,
                y_CL=7.0,
                y_V=2.04,
                F_act=0.05,
                session_id=sid,
                database_url=db_url,
                persist_to_live=True,
            )

        # Trigger agent guardrail rejection
        run_governed_capa_workflow(
            session_id=sid,
            vessel_scale_L=5,
            database_url=db_url,
            pat_tick={"t_h": 0.6, "y_Xv": 1.85, "y_CL": 1.2, "y_V": 2.04, "F_act": 0.05, "rpm_act": 450.0, "vvm_act": 1.2},
            duration_h=4.0,
            provider=FaultInjectingCapaProvider("missing_citation"),
        )

        # Trigger Medallion quarantine metric
        records, _, _ = _build_mixed_medallion_records()
        run_medallion_pipeline(records, tmp_path / "med_tel", run_id="tel_dq")

        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert "text/plain" in r_metrics.headers["content-type"]
        body = r_metrics.text
        for required_metric in (
            "twin_ekf_nis",
            "twin_ekf_spe",
            "twin_ekf_mahalanobis_dm",
            "twin_ekf_residual_breach_total",
            "twin_control_fallback_total",
            "twin_control_escalation_total",
            "twin_api_requests_total",
            "twin_api_request_duration_ms_bucket",
            "twin_agent_guardrail_rejections_total",
            "twin_mcp_privilege_rejections_total",
            "twin_medallion_quarantined_rows_total",
        ):
            assert required_metric in body, f"Missing expected Prometheus metric: {required_metric}"

        spans = get_telemetry_recorder().export_otel_spans()
        span_names = {s["name"] for s in spans}
        assert {
            "http.server.request",
            "ekf.update",
            "controller.compute_action",
            "mcp.privilege_reject",
            "agent.guardrail_reject",
            "mlops.medallion_quarantine",
        }.issubset(span_names)
    finally:
        clear_shadow_sandbox(db_url)
        if db_url in mcp_srv._DB_CACHE:
            eng, _ = mcp_srv._DB_CACHE.pop(db_url)
            eng.dispose()
        app.state.db_engine.dispose()
