"""Fast contract, persistence round-trip, schema compatibility, truth-isolation, and OOD/fallback tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict

from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.stats import chi2
from streamlit.testing.v1 import AppTest

import app as st_app
from api.db import create_db_engine_and_sessionmaker, load_twin_session, save_twin_session
from api.main import create_app
from api.runtime import load_calibrated_assets
from hybrid_fedbatch_twin.controller import (
    ClosedLoopFeedController,
    ManualFeedController,
    NominalMechanisticController,
    OpenLoopExponentialController,
    summarize_control_run,
)
from hybrid_fedbatch_twin.estimator import (
    GENESIS_ENTRY_SHA256,
    BioreactorDigitalTwin,
    HybridExtendedKalmanFilter,
    Q_DEFAULT,
    _canonical_sha256,
    run_twin_estimation_loop,
)
from hybrid_fedbatch_twin.mechanistic import (
    BioreactorParams,
    OperatingInputs,
    STATE_NAMES,
    default_initial_state,
    default_operating_inputs,
    disturbance_profile,
)
import mcp_server.server as mcp_srv
from mcp_server.server import (
    _resolve_campaign_wmse,
    ingest_pat_and_estimate_state,
    search_uhlendorff_campaigns,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


def _run_app_fast_stub() -> None:
    import json as _json
    from pathlib import Path as _Path

    import app as _app
    from api.runtime import load_calibrated_assets as _load_assets
    from hybrid_fedbatch_twin.controller import ClosedLoopFeedController as _Ctrl
    from hybrid_fedbatch_twin.estimator import run_twin_estimation_loop as _run_loop
    from hybrid_fedbatch_twin.mechanistic import (
        BioreactorParams as _Params,
        default_initial_state as _x0,
    )

    bundle = _json.loads(
        (_Path(_app.__file__).resolve().parent / "docs" / "assets" / "twin_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    p_rec, model = _load_assets()
    rec = dict(bundle["stage1_recovery"])
    rec["recovered_params"] = p_rec
    nm = dict(bundle["gam86"]["heldout_metrics"])
    _app.load_calibrated_digital_twin = lambda: (rec, model, nm)
    _app.run_cached_simulation = lambda **kw: _run_loop(
        (0.0, 1.0),
        _x0(p_rec),
        _Ctrl(p_rec, model),
        params=p_rec,
        residual_model=model,
        seed=123,
        plant_params=_Params(),
    )
    _app.main()


@pytest.mark.contract
def test_db_gate_custom_thresholds_df2_boundary_and_streak_reset(tmp_path: Path) -> None:
    """Verify api/db.py persists custom gate params, df=2 boundary NIS, and SPE streak set/reset."""
    db_url = f"sqlite:///{tmp_path / 'gate_contracts.db'}"
    p_rec, model = load_calibrated_assets()
    engine1, SessionLocal1 = create_db_engine_and_sessionmaker(db_url)
    engine2 = None
    engine3 = None
    try:
        # 1. Custom thresholds + gate_alpha + spe_persist_steps round-trip
        twin_custom = BioreactorDigitalTwin(
            default_initial_state(p_rec),
            p_rec,
            model,
            batch_id="B-CUSTOM-01",
            operator_id="qa_contract",
        )
        twin_custom.ekf.gate_alpha = 0.99
        twin_custom.ekf.spe_persist_steps = 3
        twin_custom.ekf._nis_threshold_override = 12.5
        twin_custom.ekf._spe_threshold_override = 95.0
        twin_custom.ekf._spe_persist_threshold_override = 55.0

        with SessionLocal1() as db:
            save_twin_session(db, "sess_custom", 5, twin_custom)

        # 2. Exact 2-channel LIMS df=2 boundary (NIS=15.0 in (chi2_df2=13.82, chi2_df3=16.27), no SPE breach)
        twin_df2 = BioreactorDigitalTwin(
            default_initial_state(p_rec),
            p_rec,
            model,
            batch_id="B-DF2-01",
            operator_id="qa_contract",
        )
        twin_df2.ingest_online_pat(
            t_h=0.2, y_Xv=1.6, y_CL=7.5, y_V=2.01, F_act=0.04, rpm_act=450.0, vvm_act=1.2
        )
        H = np.zeros((2, 7), dtype=float)
        H[0, 2] = 1.0
        H[1, 4] = 1.0
        sig_atline = (0.30, 0.30, 0.30)
        R = np.diag(np.asarray(sig_atline[:2], dtype=float) ** 2)
        S_cov = 0.5 * (H @ twin_df2.ekf.P @ H.T + R + (H @ twin_df2.ekf.P @ H.T + R).T)
        inv_S = np.linalg.inv(S_cov)
        dS = float(np.sqrt(15.0 / inv_S[0, 0]))
        y_S = float(twin_df2.ekf.x[2] + dS)
        y_A = float(twin_df2.ekf.x[4])

        res_df2 = twin_df2.ingest_atline_assay(
            t_sample_h=0.2, y_S=y_S, y_A=y_A, y_P=None, sig_atline=sig_atline
        )
        assert twin_df2.ekf.last_obs_dim == 2
        assert res_df2["nis"] == pytest.approx(15.0, rel=1e-4)
        assert chi2.ppf(0.999, df=2) < res_df2["nis"] < chi2.ppf(0.999, df=3)
        assert twin_df2.ekf.last_nis_breach is True
        assert twin_df2.ekf.last_spe_breach is False
        assert res_df2["residual_breach"] is True
        assert res_df2["gate_reason"] == "residual_nis_breach"

        with SessionLocal1() as db:
            save_twin_session(db, "sess_df2", 5, twin_df2)

        # 3. SPE persistence streak increment (streak=1) -> persist -> nominal PAT resets streak to 0
        twin_streak = BioreactorDigitalTwin(
            default_initial_state(p_rec),
            p_rec,
            model,
            batch_id="B-STREAK-01",
            operator_id="qa_contract",
        )
        probe = twin_streak._build_ekf_at(twin_streak.ekf.x, twin_streak.ekf.P)
        probe.predict(OperatingInputs(F=0.0, rpm=450.0, vvm=1.2), 0.2)
        y_cl_mod = float(probe.x[3] - 0.08 * np.sqrt(130.0))
        pat_mod = twin_streak.ingest_online_pat(
            t_h=0.2,
            y_Xv=float(probe.x[0]),
            y_CL=y_cl_mod,
            y_V=float(probe.x[6]),
            F_act=0.0,
            rpm_act=450.0,
            vvm_act=1.2,
        )
        assert pat_mod["spe"] > 110.0
        assert pat_mod["residual_breach"] is False
        assert twin_streak.ekf._spe_persist_streak == 1

        with SessionLocal1() as db:
            save_twin_session(db, "sess_streak", 5, twin_streak)

        engine1.dispose()
        engine2, SessionLocal2 = create_db_engine_and_sessionmaker(db_url)
        with SessionLocal2() as db:
            loaded_custom = load_twin_session(db, "sess_custom", model)
            assert loaded_custom is not None
            ekf_c = loaded_custom["twin"].ekf
            assert ekf_c.gate_alpha == pytest.approx(0.99)
            assert ekf_c.spe_persist_steps == 3
            assert ekf_c._nis_threshold_override == pytest.approx(12.5)
            assert ekf_c._spe_threshold_override == pytest.approx(95.0)
            assert ekf_c._spe_persist_threshold_override == pytest.approx(55.0)

            loaded_df2 = load_twin_session(db, "sess_df2", model)
            assert loaded_df2 is not None
            ekf_d = loaded_df2["twin"].ekf
            assert ekf_d.last_obs_dim == 2
            assert ekf_d.last_nis_threshold == pytest.approx(float(chi2.ppf(0.999, df=2)))
            assert ekf_d.last_nis == pytest.approx(15.0, rel=1e-4)
            assert ekf_d.last_nis_breach is True
            assert ekf_d.last_spe_breach is False
            assert ekf_d.last_residual_breach is True
            assert ekf_d.evaluate_gate()["gate_reason"] == "residual_nis_breach"

            loaded_streak = load_twin_session(db, "sess_streak", model)
            assert loaded_streak is not None
            twin_s2 = loaded_streak["twin"]
            assert twin_s2.ekf._spe_persist_streak == 1
            assert twin_s2.ekf.last_spe_breach is False

            # Nominal PAT tick resets _spe_persist_streak back to 0
            probe2 = twin_s2._build_ekf_at(twin_s2.ekf.x, twin_s2.ekf.P)
            probe2.predict(OperatingInputs(F=0.0, rpm=450.0, vvm=1.2), 0.2)
            pat_nom = twin_s2.ingest_online_pat(
                t_h=0.4,
                y_Xv=float(probe2.x[0]),
                y_CL=float(probe2.x[3]),
                y_V=float(probe2.x[6]),
                F_act=0.0,
                rpm_act=450.0,
                vvm_act=1.2,
            )
            assert pat_nom["residual_breach"] is False
            assert twin_s2.ekf._spe_persist_streak == 0
            save_twin_session(db, "sess_streak", 5, twin_s2)

        engine2.dispose()
        engine3, SessionLocal3 = create_db_engine_and_sessionmaker(db_url)
        with SessionLocal3() as db:
            loaded_streak_after = load_twin_session(db, "sess_streak", model)
            assert loaded_streak_after is not None
            assert loaded_streak_after["twin"].ekf._spe_persist_streak == 0
            assert loaded_streak_after["twin"].ekf.last_residual_breach is False
    finally:
        for eng in (engine1, engine2, engine3):
            if eng is not None:
                eng.dispose()


@pytest.mark.contract
def test_db_legacy_null_ekf_gate_state_reconstruction(tmp_path: Path) -> None:
    """Verify migration adds ekf_gate_state_json and reconstructs gate flags from legacy audit rows."""
    db_path = tmp_path / "legacy_gate_null.db"
    db_url = f"sqlite:///{db_path}"
    p_rec, model = load_calibrated_assets()

    conn = sqlite3.connect(db_path)
    try:
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
                prev_entry_sha256 VARCHAR(64) NOT NULL,
                detail_json TEXT NOT NULL,
                entry_sha256 VARCHAR(64) NOT NULL
            )
            """
        )

        x0 = [1.5, 0.02, 1.2, 7.79, 0.05, 0.0, 2.0]
        p0 = [[0.04 if i == j else 0.0 for j in range(7)] for i in range(7)]
        u_json = json.dumps({"F": 0.0, "rpm": 450.0, "vvm": 1.2, "S_feed": None})

        sessions = [
            (
                "leg_df2",
                "ATLINE_LIMS",
                {"t_sample_h": 0.2, "y_S": 3.0, "y_A": 0.2, "y_P": None, "nis": 15.0, "spe": 10.0},
            ),
            (
                "leg_replay",
                "ATLINE_LIMS_REPLAY",
                {
                    "t_sample_h": 0.2,
                    "y_S": 0.3,
                    "y_A": 0.2,
                    "y_P": 0.5,
                    "nis": 5.0,
                    "spe": 20.0,
                    "residual_breach": True,
                },
            ),
            (
                "leg_pat",
                "ONLINE_PAT",
                {
                    "y_Xv": 1.6,
                    "y_CL": 7.0,
                    "y_V": 2.0,
                    "nis": 4.0,
                    "spe": 120.0,
                    "spe_persist_streak": 1,
                    "residual_breach": False,
                },
            ),
        ]

        for sid, ev_type, detail in sessions:
            conn.execute(
                "INSERT INTO twin_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    sid,
                    f"BATCH-{sid}",
                    "op_legacy",
                    5,
                    json.dumps(p_rec.to_dict()),
                    "{}",
                    "a" * 64,
                    "b" * 64,
                    0.2,
                    u_json,
                ),
            )
            conn.execute(
                "INSERT INTO twin_events (session_id, seq_idx, kind, t_h, u_json, x_post_json, P_post_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, 0, "init", 0.0, u_json, json.dumps(x0), json.dumps(p0)),
            )
            conn.execute(
                "INSERT INTO alcoa_audit_entries (session_id, seq, utc_timestamp, event_type, process_t_h, params_sha256, model_sha256, prev_entry_sha256, detail_json, entry_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    1,
                    "2026-01-01T00:00:00+00:00",
                    ev_type,
                    0.2,
                    "a" * 64,
                    "b" * 64,
                    GENESIS_ENTRY_SHA256,
                    json.dumps(detail),
                    "c" * 64,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    engine, SessionLocal = create_db_engine_and_sessionmaker(db_url)
    try:
        # Confirm migration added column and left legacy rows NULL
        chk = sqlite3.connect(db_path)
        try:
            cols = {row[1] for row in chk.execute("PRAGMA table_info(twin_sessions)").fetchall()}
            assert "ekf_gate_state_json" in cols
            null_cnt = chk.execute(
                "SELECT COUNT(*) FROM twin_sessions WHERE ekf_gate_state_json IS NULL"
            ).fetchone()[0]
            assert null_cnt == 3
        finally:
            chk.close()

        with SessionLocal() as db:
            # Session A: ATLINE_LIMS with y_P=None -> df=2, NIS=15.0 > chi2(df=2)
            l_a = load_twin_session(db, "leg_df2", model)
            assert l_a is not None
            ekf_a = l_a["twin"].ekf
            assert ekf_a.last_obs_dim == 2
            assert ekf_a.last_nis_threshold == pytest.approx(float(chi2.ppf(0.999, df=2)))
            assert ekf_a.last_nis_breach is True
            assert ekf_a.last_spe_breach is False
            assert ekf_a.last_residual_breach is True

            # Session B: ATLINE_LIMS_REPLAY with y_P=0.5 -> df=3, residual_breach=True & not nis_breach -> spe_breach=True
            l_b = load_twin_session(db, "leg_replay", model)
            assert l_b is not None
            ekf_b = l_b["twin"].ekf
            assert ekf_b.last_obs_dim == 3
            assert ekf_b.last_nis_threshold == pytest.approx(float(chi2.ppf(0.999, df=3)))
            assert ekf_b.last_nis_breach is False
            assert ekf_b.last_spe_breach is True
            assert ekf_b.last_residual_breach is True

            # Session C: ONLINE_PAT -> df=3, spe_persist_streak=1, residual_breach=False
            l_c = load_twin_session(db, "leg_pat", model)
            assert l_c is not None
            ekf_c = l_c["twin"].ekf
            assert ekf_c.last_obs_dim == 3
            assert ekf_c._spe_persist_streak == 1
            assert ekf_c.last_nis_breach is False
            assert ekf_c.last_spe_breach is False
            assert ekf_c.last_residual_breach is False
    finally:
        engine.dispose()


@pytest.mark.contract
def test_db_audit_max_entries_slicing_hashing_and_api_query(tmp_path: Path) -> None:
    """Verify max_entries slicing/hashing, HTTP ?max_entries query, and tamper detection after DB reload."""
    db_url = f"sqlite:///{tmp_path / 'audit_slicing.db'}"
    _, model = load_calibrated_assets()
    app1 = create_app(database_url=db_url)
    app2 = None
    try:
        client1 = TestClient(app1)
        r_init = client1.post(
            "/twins",
            json={"batch_id": "FB-5L-SLICE", "operator_id": "qa_slice", "vessel_scale_L": 5},
        )
        assert r_init.status_code == 201
        sid = r_init.json()["session_id"]

        for step in range(1, 6):
            r_pat = client1.post(
                f"/twins/{sid}/pat",
                json={
                    "t_h": 0.2 * step,
                    "y_Xv": 1.5 + 0.08 * step,
                    "y_CL": 7.4,
                    "y_V": 2.0 + 0.02 * step,
                    "F_act": 0.04,
                    "rpm_act": 450.0,
                    "vvm_act": 1.2,
                },
            )
            assert r_pat.status_code == 200

        r_lims = client1.post(
            f"/twins/{sid}/atline",
            json={"t_sample_h": 0.4, "y_S": 0.30, "y_A": 0.45, "y_P": 0.12},
        )
        assert r_lims.status_code == 200

        r_ctrl = client1.post(f"/twins/{sid}/control", json={"t_h": 1.0, "mu_target": 0.28})
        assert r_ctrl.status_code == 200

        app1.state.db_engine.dispose()
        app2 = create_app(database_url=db_url)
        client2 = TestClient(app2)

        r_full = client2.get(f"/twins/{sid}/audit")
        assert r_full.status_code == 200
        full_payload = r_full.json()
        assert full_payload["audit_log_truncated"] is False
        assert len(full_payload["audit_log"]) == full_payload["audit_log_count"]
        assert full_payload["audit_log_count"] == 8
        full_rec = {k: v for k, v in full_payload.items() if k != "session_id"}
        assert BioreactorDigitalTwin.verify_audit_record(full_rec) == (True, [])

        r_slice = client2.get(f"/twins/{sid}/audit", params={"max_entries": 3})
        assert r_slice.status_code == 200
        slice_payload = r_slice.json()
        assert slice_payload["audit_log_truncated"] is True
        assert slice_payload["audit_log_count"] == full_payload["audit_log_count"]
        assert len(slice_payload["audit_log"]) == 3
        assert slice_payload["audit_log"][0]["seq"] == slice_payload["audit_log_count"] - 2
        assert slice_payload["chain_head_sha256"] == slice_payload["audit_log"][-1]["entry_sha256"]
        slice_rec = {k: v for k, v in slice_payload.items() if k != "session_id"}
        assert BioreactorDigitalTwin.verify_audit_record(slice_rec) == (True, [])

        r_over = client2.get(f"/twins/{sid}/audit", params={"max_entries": 100})
        assert r_over.status_code == 200
        assert r_over.json()["audit_log_truncated"] is False
        assert len(r_over.json()["audit_log"]) == full_payload["audit_log_count"]

        r_bad_q = client2.get(f"/twins/{sid}/audit", params={"max_entries": 0})
        assert r_bad_q.status_code == 422

        with app2.state.session_local() as db:
            loaded = load_twin_session(db, sid, model)
            assert loaded is not None
            twin = loaded["twin"]
            rec_trunc = twin.to_audit_record(max_entries=3)
            rec_full_obj = twin.to_audit_record()
            assert BioreactorDigitalTwin.verify_audit_record(rec_trunc) == (True, [])

            # Tamper (a): mutate detail inside truncated slice
            bad_mut = copy.deepcopy(rec_trunc)
            bad_mut["audit_log"][0]["detail"]["tampered"] = True
            assert BioreactorDigitalTwin.verify_audit_record(bad_mut)[0] is False

            # Tamper (b): flip audit_log_truncated flag to False
            bad_flag = copy.deepcopy(rec_trunc)
            bad_flag["audit_log_truncated"] = False
            assert BioreactorDigitalTwin.verify_audit_record(bad_flag)[0] is False

            # Tamper (c): forge prev_entry_sha256 = GENESIS on truncated suffix and recompute hashes
            bad_genesis = copy.deepcopy(rec_trunc)
            e0 = bad_genesis["audit_log"][0]
            e0["prev_entry_sha256"] = GENESIS_ENTRY_SHA256
            e0["entry_sha256"] = _canonical_sha256({k: v for k, v in e0.items() if k != "entry_sha256"})
            for idx in range(1, len(bad_genesis["audit_log"])):
                ei = bad_genesis["audit_log"][idx]
                ei["prev_entry_sha256"] = bad_genesis["audit_log"][idx - 1]["entry_sha256"]
                ei["entry_sha256"] = _canonical_sha256(
                    {k: v for k, v in ei.items() if k != "entry_sha256"}
                )
            bad_genesis["chain_head_sha256"] = bad_genesis["audit_log"][-1]["entry_sha256"]
            bad_genesis["record_sha256"] = _canonical_sha256(
                {k: v for k, v in bad_genesis.items() if k != "record_sha256"}
            )
            ok_g, errs_g = BioreactorDigitalTwin.verify_audit_record(bad_genesis)
            assert ok_g is False
            assert any("genesis" in e.lower() for e in errs_g)

            # Tamper (d): empty audit_log with audit_log_truncated=True
            bad_empty = copy.deepcopy(rec_trunc)
            bad_empty["audit_log"] = []
            bad_empty["record_sha256"] = _canonical_sha256(
                {k: v for k, v in bad_empty.items() if k != "record_sha256"}
            )
            assert BioreactorDigitalTwin.verify_audit_record(bad_empty)[0] is False

            # Tamper (e): drop prefix entries from full record
            bad_del = copy.deepcopy(rec_full_obj)
            bad_del["audit_log"] = bad_del["audit_log"][-3:]
            assert BioreactorDigitalTwin.verify_audit_record(bad_del)[0] is False
    finally:
        app1.state.db_engine.dispose()
        if app2 is not None:
            app2.state.db_engine.dispose()


@pytest.mark.contract
def test_uhlendorff_schema_contract_with_mcp_live_synthetic_and_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify MCP search_uhlendorff_campaigns matches stress_test_uhlendorff.py, synthetic, and legacy schemas."""
    artifact_path = ROOT_DIR / "artifacts" / "uhlendorff_2026_validation.json"
    val = json.loads(artifact_path.read_text(encoding="utf-8"))
    bench = val["benchmark_comparison"]["remediated"]
    meta = val["dataset_metadata"]
    anomalous_exps = {str(a).split("_")[0] for a in meta["anomalous_runs"]}

    # Clean-checkout fallback mode
    monkeypatch.setattr(mcp_srv, "UHLENDORFF_DATA_DIR", tmp_path / "absent_uhlendorff_dir")
    clean_res = search_uhlendorff_campaigns()
    assert clean_res["matched_runs_count"] == 24
    assert clean_res["remediated_pooled_test_wmse"] == pytest.approx(bench["pooled_test_wmse"])
    assert clean_res["remediated_ci_coverage_pct"] == pytest.approx(bench["ci_coverage_pct"])

    for run in clean_res["runs"]:
        cid = run["campaign_id"]
        if cid in bench["per_campaign_test_wmse"]:
            assert run["validation_wmse"] == pytest.approx(bench["per_campaign_test_wmse"][cid])
        elif cid in anomalous_exps:
            assert run["validation_wmse"] == pytest.approx(bench["anomaly_unmasked_wmse"])

    # If local raw CSVs exist on this machine, verify live CSV mode matches clean-checkout metadata
    live_csv_dir = ROOT_DIR / "data" / "uhlendorff_2026" / "Time-axis aligned raw data tables"
    if live_csv_dir.exists():
        monkeypatch.setattr(mcp_srv, "UHLENDORFF_DATA_DIR", live_csv_dir)
        live_res = search_uhlendorff_campaigns()
        assert live_res["matched_runs_count"] == 24
        key_fields = (
            "campaign_id",
            "replicate",
            "mode",
            "split",
            "validation_wmse",
            "has_heldout_trajectory",
        )
        live_map = {(r["campaign_id"], r["replicate"]): {k: r[k] for k in key_fields} for r in live_res["runs"]}
        clean_map = {(r["campaign_id"], r["replicate"]): {k: r[k] for k in key_fields} for r in clean_res["runs"]}
        assert live_map == clean_map

    # Synthetic stress_test_uhlendorff.py schema with non-EXP campaign IDs (no hardcoded EXP* literals)
    synth_artifact = tmp_path / "synth_uhlendorff.json"
    synth_payload: Dict[str, Any] = {
        "dataset_metadata": {
            "source": "Synthetic CHO-K1 Contract Fixture",
            "dev_campaigns": ["SYN001", "SYN002", "SYN003", "SYN004"],
            "test_campaigns": ["SYN005", "SYN006"],
            "anomalous_runs": ["SYN004_R02"],
        },
        "mass_balance_discovery": {
            "volume_series": {
                "SYN006": {"t": [0.0, 1.0], "v_true": [0.6, 0.7], "v_naive": [0.6, 0.8]},
            }
        },
        "benchmark_comparison": {
            "remediated": {
                "pooled_test_wmse": 12.34,
                "ci_coverage_pct": 91.5,
                "per_campaign_test_wmse": {"SYN005": 11.1, "SYN006": 13.5},
                "anomaly_unmasked_wmse": 22.2,
            }
        },
        "held_out_trajectories": {
            "SYN005": {
                "mode": "batch",
                "t_off_r01": [0.0, 12.0],
                "obs_r01": [[0.5, 4.0, 0.1, 0.2], [3.0, 1.0, 1.2, 1.5]],
                "t_off_r02": [0.0, 12.0],
                "obs_r02": [[0.5, 4.1, 0.1, 0.2], [2.9, 1.1, 1.3, 1.6]],
            },
            "SYN006": {
                "mode": "fed-batch",
                "t_off_r01": [0.0, 24.0],
                "obs_r01": [[0.5, 4.0, 0.1, 0.2], [4.5, 2.0, 0.9, 1.8]],
                "t_off_r02": [0.0, 24.0],
                "obs_r02": [[0.5, 4.0, 0.1, 0.2], [4.4, 2.1, 1.0, 1.9]],
            },
        },
    }
    synth_artifact.write_text(json.dumps(synth_payload), encoding="utf-8")
    monkeypatch.setattr(mcp_srv, "UHLENDORFF_DATA_DIR", tmp_path / "missing_dir")
    monkeypatch.setattr(mcp_srv, "UHLENDORFF_ARTIFACT", synth_artifact)

    synth_all = search_uhlendorff_campaigns()
    assert synth_all["matched_runs_count"] == 12
    assert synth_all["remediated_pooled_test_wmse"] == pytest.approx(12.34)
    assert synth_all["remediated_ci_coverage_pct"] == pytest.approx(91.5)
    by_cid = {r["campaign_id"]: r for r in synth_all["runs"]}
    assert by_cid["SYN006"]["mode"] == "fed-batch"
    assert by_cid["SYN006"]["split"] == "test"
    assert by_cid["SYN006"]["validation_wmse"] == pytest.approx(13.5)
    assert by_cid["SYN005"]["mode"] == "batch"
    assert by_cid["SYN005"]["validation_wmse"] == pytest.approx(11.1)
    assert by_cid["SYN004"]["validation_wmse"] == pytest.approx(22.2)

    assert search_uhlendorff_campaigns(mode="fed-batch")["matched_runs_count"] == 2
    assert search_uhlendorff_campaigns(split="test")["matched_runs_count"] == 4
    assert search_uhlendorff_campaigns(campaign_id="syn005")["matched_runs_count"] == 2
    assert search_uhlendorff_campaigns(max_wmse=12.0)["matched_runs_count"] == 2

    # Legacy schema fallback (_resolve_campaign_wmse)
    legacy_bench = {"exp012_wmse": 8.2, "exp009_wmse": 14.7, "exp011_unmasked_wmse": 24.5}
    assert _resolve_campaign_wmse("EXP012", legacy_bench, ["EXP011_R02"]) == pytest.approx(8.2)
    assert _resolve_campaign_wmse("EXP009", legacy_bench, ["EXP011_R02"]) == pytest.approx(14.7)
    assert _resolve_campaign_wmse("EXP011", legacy_bench, ["EXP011_R02"]) == pytest.approx(24.5)


@pytest.mark.contract
def test_fastapi_schema_validation_and_clean_checkout_investigate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Pydantic extra='forbid'/404/422 guards, clean-checkout /investigate, and 1000L MCP defaults."""
    db_url = f"sqlite:///{tmp_path / 'schema_and_clean.db'}"
    mcp_db_url = f"sqlite:///{tmp_path / 'mcp_1000l.db'}"
    app = create_app(database_url=db_url)
    try:
        client = TestClient(app)

        # extra="forbid" on all request schemas
        r_init_bad = client.post("/twins", json={"batch_id": "X", "unexpected_field": 123})
        assert r_init_bad.status_code == 422
        assert client.post("/twins", json={"vessel_scale_L": 250}).status_code == 422
        assert client.post("/simulate", json={"mode": "hybrid", "bogus": True}).status_code == 422
        assert client.post("/benchmark", json={"duration_h": 4.0, "bogus": True}).status_code == 422

        # Missing session -> 404
        assert client.get("/twins/does_not_exist").status_code == 404
        assert client.get("/twins/does_not_exist/audit").status_code == 404

        r_init = client.post(
            "/twins",
            json={"batch_id": "FB-SCHEMA-01", "operator_id": "qa_schema", "vessel_scale_L": 5},
        )
        assert r_init.status_code == 201
        sid = r_init.json()["session_id"]

        assert (
            client.post(
                f"/twins/{sid}/pat",
                json={
                    "t_h": 0.2,
                    "y_Xv": 1.6,
                    "y_CL": 7.4,
                    "y_V": 2.02,
                    "F_act": 0.04,
                    "rpm_act": 450.0,
                    "vvm_act": 1.2,
                    "extra_pat": 1,
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/twins/{sid}/pat",
                json={
                    "t_h": 0.2,
                    "y_Xv": 1.6,
                    "y_CL": 7.4,
                    "y_V": 2.02,
                    "F_act": 0.04,
                    "rpm_act": 450.0,
                    "vvm_act": 1.2,
                },
            ).status_code
            == 200
        )
        # Out-of-order PAT timestamp
        assert (
            client.post(
                f"/twins/{sid}/pat",
                json={
                    "t_h": 0.1,
                    "y_Xv": 1.6,
                    "y_CL": 7.4,
                    "y_V": 2.02,
                    "F_act": 0.04,
                    "rpm_act": 450.0,
                    "vvm_act": 1.2,
                },
            ).status_code
            == 422
        )
        # Out-of-range LIMS timestamp and extra field
        assert (
            client.post(
                f"/twins/{sid}/atline",
                json={"t_sample_h": 0.1, "y_S": 0.3, "y_A": 0.1, "extra_lims": 1},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/twins/{sid}/atline",
                json={"t_sample_h": 1.5, "y_S": 0.3, "y_A": 0.1},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/twins/{sid}/control",
                json={"t_h": 0.2, "extra_ctrl": 1},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/twins/{sid}/investigate",
                json={"duration_h": 4.0, "extra_inv": 1},
            ).status_code
            == 422
        )

        # Clean-checkout /investigate without raw data/ CSVs
        monkeypatch.setattr(mcp_srv, "UHLENDORFF_DATA_DIR", tmp_path / "missing_uhlendorff_dir")
        r_inv = client.post(
            f"/twins/{sid}/investigate",
            json={
                "pat_tick": {
                    "t_h": 0.4,
                    "y_Xv": 1.7,
                    "y_CL": 6.9,
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
        inv = r_inv.json()
        assert inv["evaluation"]["overall_passed"] is True
        assert inv["evaluation"]["overall_score"] == pytest.approx(1.0)
        assert inv["capa_report"]["cited_uhlendorff_campaigns"] == ["EXP010", "EXP011", "EXP012"]
        assert inv["capa_report"]["remediated_pooled_test_wmse"] is not None

        # MCP ingest_pat_and_estimate_state at 1000 L with omitted rpm_act/vvm_act uses matched defaults
        mcp_out = ingest_pat_and_estimate_state(
            t_h=0.2,
            y_Xv=1.6,
            y_CL=7.7,
            y_V=401.6,
            F_act=8.0,
            vessel_scale_L=1000,
            database_url=mcp_db_url,
        )
        assert mcp_out["vessel_scale_L"] == 1000
        assert mcp_out["in_domain"] is True
        assert mcp_out["undecided"] is False
        assert mcp_out["gate_reason"] == "nominal"
    finally:
        app.state.db_engine.dispose()
        if mcp_db_url in mcp_srv._DB_CACHE:
            eng_m, _ = mcp_srv._DB_CACHE.pop(mcp_db_url)
            eng_m.dispose()


@pytest.mark.contract
def test_app_truth_isolation_dataframe_charts_and_audit_export() -> None:
    """Verify app.py DataFrame, Altair charts, and ALCOA+ audit export isolate oracle truth when show_truth=False."""
    p_rec, model = load_calibrated_assets()
    true_params = BioreactorParams()
    x0 = default_initial_state(p_rec)
    ctrl = ClosedLoopFeedController(p_rec, model)
    sim = run_twin_estimation_loop(
        (0.0, 1.0), x0, ctrl, params=p_rec, residual_model=model, dt_step=0.1, plant_params=true_params
    )
    kpis = summarize_control_run(sim, mu_target=0.28, params=p_rec)

    df_op = st_app.build_dataframe(sim, include_truth=False)
    assert not any(col.endswith("_true") for col in df_op.columns)

    df_val = st_app.build_dataframe(sim, include_truth=True)
    for state_name in STATE_NAMES:
        assert f"{state_name}_true" in df_val.columns
    for rate_col in ("mu_true", "kd_true", "kla_true"):
        assert rate_col in df_val.columns

    theme = st_app.THEMES["Clean Biopharma Light"]
    chart_op = st_app.make_state_chart(
        sim["t"],
        sim["est_hyb"][:, 0],
        sim["est_nom"][:, 0],
        sim["std_hyb"][:, 0],
        sim["meas_states"][:, 0],
        sim["true_states"][:, 0],
        "Viable Biomass X_v (g/L)",
        theme,
        show_truth=False,
    ).to_dict()
    assert "Plant Ground Truth" not in json.dumps(chart_op)

    chart_val = st_app.make_state_chart(
        sim["t"],
        sim["est_hyb"][:, 0],
        sim["est_nom"][:, 0],
        sim["std_hyb"][:, 0],
        sim["meas_states"][:, 0],
        sim["true_states"][:, 0],
        "Viable Biomass X_v (g/L)",
        theme,
        show_truth=True,
    ).to_dict()
    assert "Plant Ground Truth" in json.dumps(chart_val)

    raw_meta = {
        "calibration_sha256": "a" * 64,
        "condition_number_rel": 38.18,
        "max_relative_error_pct": 4.39,
        "relative_error_pct": {"mu_max": 1.2},
        "true_values": {"mu_max": 0.45},
    }
    rec_op = st_app.build_audit_export_payload(
        sim=sim,
        kpis=kpis,
        params=p_rec,
        residual_model=model,
        calibration_metadata=raw_meta,
        include_truth=False,
    )
    assert BioreactorDigitalTwin.verify_audit_record(rec_op) == (True, [])
    op_meta = rec_op["calibration_metadata"]
    assert op_meta["includes_oracle_truth"] is False
    assert set(op_meta["run_kpis"].keys()).issubset(set(st_app.ORACLE_FREE_KPI_KEYS))
    for oracle_meta_key in st_app.ORACLE_METADATA_KEYS:
        assert oracle_meta_key not in op_meta

    rec_val = st_app.build_audit_export_payload(
        sim=sim,
        kpis=kpis,
        params=p_rec,
        residual_model=model,
        calibration_metadata=raw_meta,
        include_truth=True,
    )
    assert BioreactorDigitalTwin.verify_audit_record(rec_val) == (True, [])
    assert rec_val["calibration_metadata"]["includes_oracle_truth"] is True
    assert "hybrid_norm_rmse" in rec_val["calibration_metadata"]["run_kpis"]
    assert rec_val["calibration_metadata"]["max_relative_error_pct"] == pytest.approx(4.39)


@pytest.mark.contract
def test_app_streamlit_main_truth_isolation_end_to_end() -> None:
    """Drive app.main() via Streamlit AppTest and assert show_truth=False vs True isolation."""
    orig_loader = st_app.load_calibrated_digital_twin
    orig_runner = st_app.run_cached_simulation
    try:
        at = AppTest.from_function(_run_app_fast_stub, default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        md_op = "\n".join(m.value for m in at.markdown)
        assert "Est. Viable Biomass X_v" in md_op
        assert "True Final X_v" not in md_op
        assert "7-State Parity nRMSE" not in md_op

        st1_df_op = at.dataframe[0].value
        assert "True Value" not in st1_df_op.columns
        assert "Rel Error (%)" not in st1_df_op.columns

        telem_df_op = at.dataframe[1].value
        assert not any(col.endswith("_true") for col in telem_df_op.columns)

        at.sidebar.radio[0].set_value("Validation Benchmark (With Ground Truth)").run()
        assert len(at.exception) == 0

        md_val = "\n".join(m.value for m in at.markdown)
        assert "True Final X_v" in md_val
        assert "7-State Parity nRMSE" in md_val

        st1_df_val = at.dataframe[0].value
        assert "True Value" in st1_df_val.columns
        assert "Rel Error (%)" in st1_df_val.columns

        telem_df_val = at.dataframe[1].value
        assert "X_v_true" in telem_df_val.columns
        assert "mu_true" in telem_df_val.columns
    finally:
        st_app.load_calibrated_digital_twin = orig_loader
        st_app.run_cached_simulation = orig_runner


@pytest.mark.contract
def test_ood_and_residual_gate_fallback_control_contracts() -> None:
    """Verify OOD/residual gate fallback policies, interlocks, reset semantics, and ALCOA+ escalation."""
    p_rec, model = load_calibrated_assets()
    x0 = default_initial_state(p_rec)

    with pytest.raises(ValueError):
        ClosedLoopFeedController(p_rec, model, fallback_policy="invalid_policy")  # type: ignore[arg-type]

    x_nom = np.array([10.0, 0.3, 0.20, 5.2, 0.25, 2.0, 2.6], dtype=float)
    ctrl_safe = ClosedLoopFeedController(p_rec, model, fallback_policy="safe_nominal")

    # In-domain control populates _last_safe_F
    u_nom = ctrl_safe(4.0, x_nom, gate_status={"in_domain": True, "undecided": False, "residual_breach": False, "dov_in_domain": True})
    assert ctrl_safe.last_fallback_active is False
    assert ctrl_safe.last_undecided is False
    assert ctrl_safe.last_control_mode == "hybrid_adaptive"
    assert ctrl_safe.last_escalation_event is None
    prev_safe_f = float(u_nom.F)

    # Mahalanobis OOD without external gate_status at high S/A -> F == 0.0
    x_ood_high = np.array([55.0, 6.0, 15.0, 4.5, 10.0, 5.0, 3.0], dtype=float)
    u_ood_high = ctrl_safe(4.5, x_ood_high)
    assert u_ood_high.F == 0.0
    assert ctrl_safe.last_fallback_active is True
    assert ctrl_safe.last_undecided is True
    assert ctrl_safe.last_control_mode == "safe_fallback_safe_nominal"
    assert ctrl_safe.last_escalation_event is not None
    assert ctrl_safe.last_escalation_event["gate_reason"] == "mahalanobis_ood"

    # Mahalanobis OOD with moderate S/A and extreme X_v -> bounded ML-free conservative feed
    x_ood_mod = np.array([90.0, 5.0, 0.20, 5.2, 0.25, 2.0, 2.6], dtype=float)
    u_ood_mod = ctrl_safe(4.6, x_ood_mod)
    assert 0.0 <= u_ood_mod.F <= min(0.5 * ctrl_safe.F_max, prev_safe_f) + 1e-12
    assert ctrl_safe.last_fallback_active is True
    assert ctrl_safe.last_escalation_event is not None
    assert ctrl_safe.last_escalation_event["gate_reason"] == "mahalanobis_ood"

    # Pure residual breach (dov_in_domain=True, residual_breach=True)
    gate_res = {
        "d_m": 1.2,
        "dov_in_domain": True,
        "residual_breach": True,
        "in_domain": False,
        "undecided": True,
        "nis": 25.0,
        "spe": 12.0,
        "gate_reason": "residual_nis_breach",
    }
    u_res_safe = ctrl_safe(4.7, x_nom, gate_status=gate_res)
    assert u_res_safe.F <= u_nom.F
    assert ctrl_safe.last_fallback_active is True
    assert ctrl_safe.last_escalation_event is not None
    assert ctrl_safe.last_escalation_event["gate_reason"] == "residual_nis_breach"

    # Hold policy contracts: explicit F_hold, None -> 0.0, clamp to 0.5*F_max, and safety interlocks
    ctrl_hold = ClosedLoopFeedController(p_rec, model, fallback_policy="hold", F_hold=0.04)
    assert ctrl_hold(4.0, x_nom, gate_status=gate_res).F == pytest.approx(0.04)
    ctrl_hold_none = ClosedLoopFeedController(p_rec, model, fallback_policy="hold", F_hold=None)
    assert ctrl_hold_none(4.0, x_nom, gate_status=gate_res).F == 0.0
    ctrl_hold_hi = ClosedLoopFeedController(p_rec, model, fallback_policy="hold", F_hold=0.99, F_max=0.30)
    assert ctrl_hold_hi(4.0, x_nom, gate_status=gate_res).F == pytest.approx(0.15)

    for bad_state in (
        np.array([10.0, 0.3, 0.20, 1.5, 0.25, 2.0, 2.6]),  # C_L <= 1.6
        np.array([10.0, 0.3, 0.20, 5.2, 1.85, 2.0, 2.6]),  # A >= A_limit
        np.array([10.0, 0.3, 0.20, 5.2, 0.25, 2.0, 4.98]),  # V >= 0.995 * V_max
    ):
        assert ctrl_hold(4.1, bad_state, gate_status=gate_res).F == 0.0

    # Dual OOD + residual breach
    ctrl_safe(4.8, x_ood_high, gate_status={"residual_breach": True, "undecided": True, "in_domain": False})
    assert ctrl_safe.last_escalation_event is not None
    assert ctrl_safe.last_escalation_event["gate_reason"] == "dual_ood_and_residual_breach"

    # Recovery to nominal in-domain state resets active fallback flags while retaining escalation history
    n_esc_before = len(ctrl_safe.escalation_events)
    assert n_esc_before >= 4
    ctrl_safe(4.9, x_nom, gate_status={"in_domain": True, "undecided": False, "residual_breach": False, "dov_in_domain": True})
    assert ctrl_safe.last_fallback_active is False
    assert ctrl_safe.last_undecided is False
    assert ctrl_safe.last_control_mode == "hybrid_adaptive"
    assert ctrl_safe.last_escalation_event is None
    assert len(ctrl_safe.escalation_events) == n_esc_before

    # In-domain volume limit stop
    x_vol_stop = np.array([10.0, 0.3, 0.20, 5.2, 0.25, 2.0, 4.98], dtype=float)
    u_vstop = ctrl_safe(5.0, x_vol_stop, gate_status={"in_domain": True, "undecided": False, "residual_breach": False, "dov_in_domain": True})
    assert u_vstop.F == 0.0
    assert ctrl_safe.last_control_mode == "volume_limit_stop"
    assert ctrl_safe.last_fallback_active is False

    # Digital twin logs CONTROL_ESCALATION and maintains valid SHA-256 chain
    twin = BioreactorDigitalTwin(x0, p_rec, model)
    twin.ingest_online_pat(0.1, float(x0[0]), 0.4, float(x0[6]), 0.02, 450.0, 1.2)
    act = twin.compute_control_action(0.1, ctrl_safe)
    assert act["undecided"] is True
    assert act["fallback_active"] is True
    assert act["escalation_event"] is not None
    assert any(e["event_type"] == "CONTROL_ESCALATION" for e in twin.audit_log)
    assert twin.verify_audit_chain() == (True, [])


@pytest.mark.contract
def test_stiff_van_loan_ekf_blind_disturbance_and_api_ood_fallback(tmp_path: Path) -> None:
    """Verify Van Loan EKF invariants, blind disturbance parity, 1000L defaults, and FastAPI OOD escalation."""
    p_rec, model = load_calibrated_assets()
    x0 = default_initial_state(p_rec)

    # 1. Q_diag 1D/2D validation, Van Loan Qd PSD, non-zero epistemic cov, and DO crash NIS+SPE
    ekf_1d = HybridExtendedKalmanFilter(x0, params=p_rec, mode="hybrid", residual_model=model, Q_diag=Q_DEFAULT)
    ekf_2d = HybridExtendedKalmanFilter(x0, params=p_rec, mode="hybrid", residual_model=model, Q_diag=np.diag(Q_DEFAULT))
    assert np.allclose(ekf_1d.Q_base, ekf_2d.Q_base)
    with pytest.raises(ValueError):
        HybridExtendedKalmanFilter(x0, params=p_rec, mode="hybrid", residual_model=model, Q_diag=-Q_DEFAULT)
    bad_asym = np.diag(Q_DEFAULT)
    bad_asym[0, 1] = 0.02
    with pytest.raises(ValueError):
        HybridExtendedKalmanFilter(x0, params=p_rec, mode="hybrid", residual_model=model, Q_diag=bad_asym)

    ekf_1d.predict(OperatingInputs(F=0.04, rpm=450.0, vvm=1.2), dt=0.1)
    assert float(np.max(ekf_1d.last_epistemic_cov)) > 0.0
    assert float(np.sqrt(max(0.0, ekf_1d.P[3, 3]))) < 1.0
    assert np.allclose(ekf_1d.last_discrete_Q, ekf_1d.last_discrete_Q.T, atol=1e-10)
    assert float(np.min(np.linalg.eigvalsh(ekf_1d.last_discrete_Q))) >= -1e-12

    ekf_1d.update(np.array([ekf_1d.x[0], max(0.2, ekf_1d.x[3] - 4.5), ekf_1d.x[6]]), [0, 3, 6], np.array([0.12, 0.08, 0.015]) ** 2)
    gate_crash = ekf_1d.evaluate_gate()
    assert gate_crash["nis_breach"] is True
    assert gate_crash["spe_breach"] is True
    assert gate_crash["residual_breach"] is True

    # 2. Blind disturbance parity across all 4 controllers & 1000L scale defaults
    for ctrl in (
        ClosedLoopFeedController(p_rec, model, rpm=450.0, vvm=1.2, disturbance_mode="aeration_drop"),
        NominalMechanisticController(p_rec, rpm=450.0, vvm=1.2, disturbance_mode="stirrer_drop"),
        OpenLoopExponentialController(p_rec, x0, rpm=450.0, vvm=1.2, disturbance_mode="shear_spike"),
        ManualFeedController(p_rec, F_set=0.05, rpm=450.0, vvm=1.2, disturbance_mode="combo"),
    ):
        u_cmd = ctrl(8.0, x0)
        assert u_cmd.rpm == pytest.approx(450.0)
        assert u_cmd.vvm == pytest.approx(1.2)
        assert u_cmd.S_feed is None

    p1000 = BioreactorParams(**{**p_rec.to_dict(), "V_vessel": 1000.0, "V_0": 400.0, "V_max": 1000.0})
    u1000 = default_operating_inputs(p1000)
    assert 120.0 <= u1000.rpm <= 150.0
    assert 0.15 <= u1000.vvm <= 0.25
    _, vvm_air, _ = disturbance_profile("aeration_drop", 7.0, u1000.rpm, u1000.vvm, p1000.S_feed, V_vessel=1000.0)
    assert vvm_air < u1000.vvm

    # 3. FastAPI OOD PAT -> /control safe fallback & CONTROL_ESCALATION audit trail
    db_url = f"sqlite:///{tmp_path / 'api_ood_fallback.db'}"
    app = create_app(database_url=db_url)
    try:
        client = TestClient(app)
        r_init = client.post(
            "/twins",
            json={"batch_id": "FB-5L-OOD-API", "operator_id": "qa_ood", "vessel_scale_L": 5},
        )
        assert r_init.status_code == 201
        sid = r_init.json()["session_id"]

        r_pat = client.post(
            f"/twins/{sid}/pat",
            json={
                "t_h": 0.2,
                "y_Xv": 60.0,
                "y_CL": 7.4,
                "y_V": 2.02,
                "F_act": 0.04,
                "rpm_act": 450.0,
                "vvm_act": 1.2,
            },
        )
        assert r_pat.status_code == 200
        assert r_pat.json()["in_domain"] is False
        assert r_pat.json()["undecided"] is True

        r_ctrl = client.post(f"/twins/{sid}/control", json={"t_h": 0.2, "mu_target": 0.28})
        assert r_ctrl.status_code == 200
        ctrl_out = r_ctrl.json()
        assert ctrl_out["fallback_active"] is True
        assert ctrl_out["undecided"] is True
        assert ctrl_out["control_mode"] == "safe_fallback_safe_nominal"
        assert ctrl_out["escalation_event"] is not None

        r_audit = client.get(f"/twins/{sid}/audit")
        assert r_audit.status_code == 200
        audit_payload = r_audit.json()
        assert any(e["event_type"] == "CONTROL_ESCALATION" for e in audit_payload["audit_log"])
        audit_rec = {k: v for k, v in audit_payload.items() if k != "session_id"}
        assert BioreactorDigitalTwin.verify_audit_record(audit_rec) == (True, [])
    finally:
        app.state.db_engine.dispose()
