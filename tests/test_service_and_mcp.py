"""Fast verification suite for FastAPI service, SQLAlchemy ALCOA+ ledger, and MCP server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.stats import chi2

from api.db import create_db_engine_and_sessionmaker, load_twin_session
from api.main import create_app
from api.runtime import load_calibrated_assets
from hybrid_fedbatch_twin.estimator import BioreactorDigitalTwin
import mcp_server.server as mcp_srv
from mcp_server.server import (
    ingest_pat_and_estimate_state,
    mcp,
    replay_delayed_lims_assay,
    search_uhlendorff_campaigns,
    simulate_fedbatch_run,
)

pytestmark = pytest.mark.contract


@pytest.fixture()
def temp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test_alcoa_ledger.db'}"


def test_fastapi_twin_lifecycle_and_alcoa_persistence_across_reload(temp_db_url: str) -> None:
    """Verify 5L/1000L twin lifecycle, PAT/LIMS replay, control, and SHA-256 persistence across restart."""
    app1 = create_app(database_url=temp_db_url)
    client1 = TestClient(app1)

    r_health = client1.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["n_bootstrap"] == 15

    # Initialize 1000 L manufacturing-scale twin
    r_init = client1.post(
        "/twins",
        json={
            "batch_id": "FB-1000L-GMP-01",
            "operator_id": "qa_lead_blr",
            "vessel_scale_L": 1000,
        },
    )
    assert r_init.status_code == 201
    init_data = r_init.json()
    session_id = init_data["session_id"]
    assert init_data["vessel_scale_L"] == 1000
    assert init_data["current_state"]["V"] == pytest.approx(400.0)
    assert len(init_data["params_sha256"]) == 64

    # Ingest 3 online PAT ticks
    for step in range(1, 4):
        t_h = 0.2 * step
        r_pat = client1.post(
            f"/twins/{session_id}/pat",
            json={
                "t_h": t_h,
                "y_Xv": 1.5 + 0.12 * step,
                "y_CL": 7.4,
                "y_V": 400.0 + 2.0 * step,
                "F_act": 10.0,
                "rpm_act": 135.0,
                "vvm_act": 0.205,
            },
        )
        assert r_pat.status_code == 200
        pat_data = r_pat.json()
        assert pat_data["t_h"] == pytest.approx(t_h)
        assert isinstance(pat_data["in_domain"], bool)

    # Delayed at-line LIMS assay at t=0.2h triggers retrodictive EKF replay across 2 later steps
    r_lims = client1.post(
        f"/twins/{session_id}/atline",
        json={"t_sample_h": 0.2, "y_S": 0.32, "y_A": 0.45, "y_P": 0.15},
    )
    assert r_lims.status_code == 200
    lims_data = r_lims.json()
    assert lims_data["replayed_steps"] == 2

    # Compute closed-loop control action
    r_ctrl = client1.post(
        f"/twins/{session_id}/control",
        json={"t_h": 0.6, "mu_target": 0.28, "rpm": 135.0, "vvm": 0.205},
    )
    assert r_ctrl.status_code == 200
    assert r_ctrl.json()["F_cmd"] >= 0.0

    r_audit1 = client1.get(f"/twins/{session_id}/audit")
    assert r_audit1.status_code == 200
    audit1 = r_audit1.json()
    assert audit1["audit_log_count"] >= 6
    assert len(audit1["record_sha256"]) == 64

    # Simulate full service restart with a brand-new FastAPI app pointing to the same SQLite DB
    app2 = create_app(database_url=temp_db_url)
    client2 = TestClient(app2)

    r_get2 = client2.get(f"/twins/{session_id}")
    assert r_get2.status_code == 200
    assert r_get2.json()["t_now_h"] == pytest.approx(0.6)

    r_audit2 = client2.get(f"/twins/{session_id}/audit")
    assert r_audit2.status_code == 200
    audit2 = r_audit2.json()
    assert audit2["record_sha256"] == audit1["record_sha256"]
    assert audit2["audit_log_count"] == audit1["audit_log_count"]
    assert audit2["current_state"] == audit1["current_state"]


def test_fastapi_simulate_and_benchmark_endpoints(temp_db_url: str) -> None:
    """Verify /simulate and /benchmark endpoints at 5 L scale."""
    app = create_app(database_url=temp_db_url)
    client = TestClient(app)

    r_sim = client.post(
        "/simulate",
        json={"vessel_scale_L": 5, "duration_h": 4.0, "dt_eval": 0.5, "mode": "hybrid"},
    )
    assert r_sim.status_code == 200
    sim = r_sim.json()
    assert sim["n_points"] == 9
    assert all(v >= 0.0 for v in sim["S"])
    assert sim["X_v"][-1] > sim["X_v"][0]

    r_bench = client.post(
        "/benchmark",
        json={"vessel_scale_L": 5, "duration_h": 4.0},
    )
    assert r_bench.status_code == 200
    strategies = r_bench.json()["strategies"]
    assert set(strategies.keys()) == {"hybrid_closed_loop", "nominal_closed_loop", "open_loop_exp"}


def test_mcp_server_tools_and_uhlendorff_grounding(temp_db_url: str) -> None:
    """Verify MCP tool registration, deterministic physics/EKF/ALCOA+ tools, and CHO-K1 lookup."""
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert {
        "simulate_fedbatch_run",
        "ingest_pat_and_estimate_state",
        "replay_delayed_lims_assay",
        "search_uhlendorff_campaigns",
    }.issubset(tool_names)

    sim_out = simulate_fedbatch_run(vessel_scale_L=50, duration_h=4.0, mode="hybrid", dt_eval=0.5)
    assert sim_out["vessel_scale_L"] == 50
    assert sim_out["final_state"]["V_L"] >= 20.0
    assert sim_out["min_DO_pct"] > 0.0

    # Ingest 2 PAT steps via MCP, then replay a delayed LIMS sample
    p1 = ingest_pat_and_estimate_state(
        t_h=0.2, y_Xv=1.6, y_CL=7.2, y_V=2.03, F_act=0.05, database_url=temp_db_url
    )
    sid = p1["session_id"]
    p2 = ingest_pat_and_estimate_state(
        t_h=0.4, y_Xv=1.7, y_CL=7.0, y_V=2.05, F_act=0.05, session_id=sid, database_url=temp_db_url
    )
    assert p2["session_id"] == sid
    assert isinstance(p2["in_domain"], bool)

    replay = replay_delayed_lims_assay(
        session_id=sid, t_sample_h=0.2, y_S=0.35, y_A=0.40, database_url=temp_db_url
    )
    assert replay["replayed_steps"] == 1
    assert len(replay["alcoa_record"]["record_sha256"]) == 64

    # Empirical Uhlendorff CHO-K1 dataset grounding (24 runs across EXP001-EXP012)
    all_runs = search_uhlendorff_campaigns()
    assert all_runs["matched_runs_count"] == 24

    test_fb = search_uhlendorff_campaigns(mode="fed-batch", split="test", campaign_id="EXP012")
    assert test_fb["matched_runs_count"] == 2
    assert all(r["campaign_id"] == "EXP012" and r["mode"] == "fed-batch" for r in test_fb["runs"])
    assert test_fb["remediated_pooled_test_wmse"] is not None

    for sid_cleanup in (p1["session_id"],):
        assert sid_cleanup
    if temp_db_url in mcp_srv._DB_CACHE:
        eng_cleanup, _ = mcp_srv._DB_CACHE.pop(temp_db_url)
        eng_cleanup.dispose()


def test_db_gate_roundtrip_lims_breach_df2_and_spe_streak(tmp_path: Path) -> None:
    """Verify full EKF gate state (at-line breach, df=2 NIS threshold, NIS+SPE, SPE streak) persists across DB reloads."""
    db_url = f"sqlite:///{tmp_path / 'gate_roundtrip.db'}"
    _, model = load_calibrated_assets()

    # Case A: At-line df=2 severe breach -> simultaneous NIS+SPE -> reload -> /control triggers fallback
    app1 = create_app(database_url=db_url)
    client1 = TestClient(app1)
    r_init = client1.post(
        "/twins",
        json={"batch_id": "FB-5L-GATE-A", "operator_id": "qa_eng", "vessel_scale_L": 5},
    )
    assert r_init.status_code == 201
    sid_a = r_init.json()["session_id"]

    r_pat = client1.post(
        f"/twins/{sid_a}/pat",
        json={"t_h": 0.2, "y_Xv": 1.6, "y_CL": 7.6, "y_V": 2.01, "F_act": 0.04, "rpm_act": 450.0, "vvm_act": 1.2},
    )
    assert r_pat.status_code == 200
    assert r_pat.json()["residual_breach"] is False

    # Severe 2-channel (y_P=None -> df=2) LIMS excursion
    r_atline = client1.post(
        f"/twins/{sid_a}/atline",
        json={"t_sample_h": 0.2, "y_S": 3.0, "y_A": 2.2},
    )
    assert r_atline.status_code == 200
    atline_data = r_atline.json()
    assert atline_data["residual_breach"] is True
    assert atline_data["undecided"] is True
    assert atline_data["gate_reason"] == "residual_nis_and_spe_breach"

    # Reload from SQLite with fresh FastAPI app
    app1.state.db_engine.dispose()
    app2 = create_app(database_url=db_url)
    client2 = TestClient(app2)

    r_get2 = client2.get(f"/twins/{sid_a}")
    assert r_get2.status_code == 200
    get2 = r_get2.json()
    assert get2["residual_breach"] is True
    assert get2["undecided"] is True
    assert get2["in_domain"] is False
    assert get2["gate_reason"] == "residual_nis_and_spe_breach"

    with app2.state.session_local() as db:
        loaded = load_twin_session(db, sid_a, model)
        assert loaded is not None
        ekf_reloaded = loaded["twin"].ekf
        assert ekf_reloaded.last_obs_dim == 2
        assert ekf_reloaded.last_nis_threshold == pytest.approx(float(chi2.ppf(0.999, df=2)))
        assert ekf_reloaded.last_nis_breach is True
        assert ekf_reloaded.last_spe_breach is True
        assert ekf_reloaded.last_residual_breach is True

    r_ctrl2 = client2.post(f"/twins/{sid_a}/control", json={"t_h": 0.2, "mu_target": 0.28})
    assert r_ctrl2.status_code == 200
    ctrl2 = r_ctrl2.json()
    assert ctrl2["residual_breach"] is True
    assert ctrl2["undecided"] is True
    assert ctrl2["fallback_active"] is True
    assert ctrl2["gate_reason"] == "residual_nis_and_spe_breach"
    assert ctrl2["escalation_event"] is not None

    # Case B: Moderate SPE between persist threshold (6.8*chi2) and single-step threshold (10*chi2) persists streak=1 across reload
    r_init_b = client2.post(
        "/twins",
        json={"batch_id": "FB-5L-GATE-B", "operator_id": "qa_eng", "vessel_scale_L": 5},
    )
    sid_b = r_init_b.json()["session_id"]
    with app2.state.session_local() as db:
        loaded_b = load_twin_session(db, sid_b, model)
        twin_b = loaded_b["twin"]
        twin_b.ekf.predict(twin_b.last_u, 0.2)
        x_pred_b = twin_b.ekf.x.copy()
    # Innov in DO ~ 0.08 * sqrt(130) -> SPE ~ 130 (between 110.6 and 162.66 for df=3) while NIS < 16.27
    y_cl_mod = float(x_pred_b[3] - 0.08 * np.sqrt(130.0))
    r_pat_b1 = client2.post(
        f"/twins/{sid_b}/pat",
        json={
            "t_h": 0.2,
            "y_Xv": float(x_pred_b[0]),
            "y_CL": y_cl_mod,
            "y_V": float(x_pred_b[6]),
            "F_act": 0.0,
            "rpm_act": 450.0,
            "vvm_act": 1.2,
        },
    )
    assert r_pat_b1.status_code == 200
    assert r_pat_b1.json()["residual_breach"] is False
    assert r_pat_b1.json()["spe"] > 110.0

    app2.state.db_engine.dispose()
    app3 = create_app(database_url=db_url)
    client3 = TestClient(app3)
    with app3.state.session_local() as db:
        loaded_b2 = load_twin_session(db, sid_b, model)
        assert loaded_b2["twin"].ekf._spe_persist_streak == 1
        loaded_b2["twin"].ekf.predict(loaded_b2["twin"].last_u, 0.2)
        x_pred_b2 = loaded_b2["twin"].ekf.x.copy()

    r_pat_b2 = client3.post(
        f"/twins/{sid_b}/pat",
        json={
            "t_h": 0.4,
            "y_Xv": float(x_pred_b2[0]),
            "y_CL": float(x_pred_b2[3] - 0.08 * np.sqrt(130.0)),
            "y_V": float(x_pred_b2[6]),
            "F_act": 0.0,
            "rpm_act": 450.0,
            "vvm_act": 1.2,
        },
    )
    assert r_pat_b2.status_code == 200
    assert r_pat_b2.json()["residual_breach"] is True
    assert r_pat_b2.json()["gate_reason"] == "residual_spe_breach"
    app3.state.db_engine.dispose()


def test_legacy_unchained_db_migration_flags_chain_invalid(tmp_path: Path) -> None:
    """Verify legacy DB migration without prev_entry_sha256 does not fabricate GENESIS hashes."""
    db_path = tmp_path / "legacy_unchained.db"
    db_url = f"sqlite:///{db_path}"
    base_params, model = load_calibrated_assets()

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE twin_sessions (
            session_id VARCHAR(64) PRIMARY KEY,
            batch_id VARCHAR(128) NOT NULL,
            operator_id VARCHAR(128) NOT NULL,
            vessel_scale_L INTEGER NOT NULL,
            params_json TEXT NOT NULL,
            calibration_metadata_json TEXT NOT NULL,
            params_sha256 VARCHAR(64) NOT NULL,
            model_sha256 VARCHAR(64) NOT NULL,
            t_now_h FLOAT NOT NULL,
            last_u_json TEXT NOT NULL,
            created_at DATETIME,
            updated_at DATETIME
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE twin_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id VARCHAR(64) NOT NULL,
            seq_idx INTEGER NOT NULL,
            kind VARCHAR(32) NOT NULL,
            t_h FLOAT NOT NULL,
            u_json TEXT NOT NULL,
            y_vec_json TEXT,
            obs_idx_json TEXT,
            r_diag_json TEXT,
            x_post_json TEXT NOT NULL,
            P_post_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE alcoa_audit_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id VARCHAR(64) NOT NULL,
            seq INTEGER NOT NULL,
            utc_timestamp VARCHAR(64) NOT NULL,
            event_type VARCHAR(64) NOT NULL,
            process_t_h FLOAT NOT NULL,
            params_sha256 VARCHAR(64) NOT NULL,
            model_sha256 VARCHAR(64) NOT NULL,
            detail_json TEXT NOT NULL,
            entry_sha256 VARCHAR(64) NOT NULL
        )
        """
    )
    x0 = [1.5, 0.02, 1.2, 7.79, 0.05, 0.0, 2.0]
    p0 = [[0.04 if i == j else 0.0 for j in range(7)] for i in range(7)]
    conn.execute(
        "INSERT INTO twin_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
        (
            "leg001",
            "LEGACY-B1",
            "op_legacy",
            5,
            json.dumps(base_params.to_dict()),
            "{}",
            "a" * 64,
            "b" * 64,
            0.0,
            json.dumps({"F": 0.0, "rpm": 450.0, "vvm": 1.2, "S_feed": None}),
        ),
    )
    conn.execute(
        "INSERT INTO twin_events (session_id, seq_idx, kind, t_h, u_json, x_post_json, P_post_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("leg001", 0, "init", 0.0, json.dumps({"F": 0.0, "rpm": 450.0, "vvm": 1.2, "S_feed": None}), json.dumps(x0), json.dumps(p0)),
    )
    for seq in (1, 2):
        conn.execute(
            "INSERT INTO alcoa_audit_entries (session_id, seq, utc_timestamp, event_type, process_t_h, params_sha256, model_sha256, detail_json, entry_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("leg001", seq, "2026-01-01T00:00:00+00:00", "INIT", 0.0, "a" * 64, "b" * 64, "{}", "deadbeef12345678"),
        )
    conn.commit()
    conn.close()

    engine, SessionLocal = create_db_engine_and_sessionmaker(db_url)
    try:
        with SessionLocal() as db:
            loaded = load_twin_session(db, "leg001", model)
            assert loaded is not None
            twin = loaded["twin"]
            assert twin.audit_log[0]["prev_entry_sha256"] == ""
            rec = twin.to_audit_record()
            assert rec["chain_valid"] is False
            ok_rec, errs = BioreactorDigitalTwin.verify_audit_record(rec)
            assert ok_rec is False and len(errs) > 0
    finally:
        engine.dispose()


def test_1000L_api_defaults_and_uhlendorff_mcp_schema_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 1000 L API initializes & controls in-domain and MCP search_uhlendorff_campaigns supports updated schema."""
    db_url = f"sqlite:///{tmp_path / 'api_1000l.db'}"
    app = create_app(database_url=db_url)
    client = TestClient(app)

    r_init = client.post(
        "/twins",
        json={"batch_id": "FB-1000L-NOM", "operator_id": "op_mfg", "vessel_scale_L": 1000},
    )
    assert r_init.status_code == 201
    init_data = r_init.json()
    assert init_data["in_domain"] is True
    assert init_data["undecided"] is False
    assert init_data["gate_reason"] == "nominal"

    sid = init_data["session_id"]
    r_ctrl = client.post(f"/twins/{sid}/control", json={"t_h": 0.0, "mu_target": 0.28})
    assert r_ctrl.status_code == 200
    ctrl_data = r_ctrl.json()
    assert ctrl_data["in_domain"] is True
    assert ctrl_data["undecided"] is False
    assert ctrl_data["fallback_active"] is False
    assert ctrl_data["escalation_event"] is None
    assert 120.0 <= ctrl_data["rpm_cmd"] <= 150.0
    assert 0.15 <= ctrl_data["vvm_cmd"] <= 0.25
    app.state.db_engine.dispose()

    # MCP search_uhlendorff_campaigns with updated schema (both with local CSVs and clean checkout)
    filtered = search_uhlendorff_campaigns(split="test", max_wmse=25.0)
    assert filtered["matched_runs_count"] >= 2
    assert all(r["validation_wmse"] is not None and r["validation_wmse"] <= 25.0 for r in filtered["runs"])

    exp11 = search_uhlendorff_campaigns(campaign_id="EXP011")
    assert exp11["matched_runs_count"] == 2
    assert all(r["validation_wmse"] is not None for r in exp11["runs"])

    monkeypatch.setattr(mcp_srv, "UHLENDORFF_DATA_DIR", tmp_path / "missing_uhlendorff_dir")
    clean = search_uhlendorff_campaigns()
    assert clean["matched_runs_count"] == 24
    modes_by_exp = {r["campaign_id"]: r["mode"] for r in clean["runs"]}
    assert modes_by_exp["EXP010"] == "fed-batch"
    assert modes_by_exp["EXP012"] == "fed-batch"
    assert modes_by_exp["EXP001"] == "batch"
    wmse_by_exp = {r["campaign_id"]: r["validation_wmse"] for r in clean["runs"]}
    assert wmse_by_exp["EXP012"] is not None and wmse_by_exp["EXP009"] is not None
