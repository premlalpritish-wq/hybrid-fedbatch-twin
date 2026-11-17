"""Fast verification suite for FastAPI service, SQLAlchemy ALCOA+ ledger, and MCP server."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from api.main import create_app
from mcp_server.server import (
    ingest_pat_and_estimate_state,
    mcp,
    replay_delayed_lims_assay,
    search_uhlendorff_campaigns,
    simulate_fedbatch_run,
)


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
