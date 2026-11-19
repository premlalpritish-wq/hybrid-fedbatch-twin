"""SQLAlchemy persistence layer for BioreactorDigitalTwin sessions and ALCOA+ SHA-256 audit trails."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import chi2
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, delete, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from hybrid_fedbatch_twin.estimator import GENESIS_ENTRY_SHA256, BioreactorDigitalTwin
from hybrid_fedbatch_twin.mechanistic import BioreactorParams, ResidualCorrectorProtocol

from .runtime import deserialize_operating_inputs, serialize_operating_inputs

DEFAULT_DATABASE_URL = "sqlite:///./hybrid_twin_ledger.db"


class Base(DeclarativeBase):
    """Base ORM class for the ALCOA+ ledger."""


class TwinSessionRow(Base):
    __tablename__ = "twin_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(128), nullable=False)
    operator_id: Mapped[str] = mapped_column(String(128), nullable=False)
    vessel_scale_L: Mapped[int] = mapped_column(Integer, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    calibration_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    params_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    t_now_h: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_u_json: Mapped[str] = mapped_column(Text, nullable=False)
    ekf_gate_state_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class TwinEventRow(Base):
    __tablename__ = "twin_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("twin_sessions.session_id", ondelete="CASCADE"), index=True
    )
    seq_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    t_h: Mapped[float] = mapped_column(Float, nullable=False)
    u_json: Mapped[str] = mapped_column(Text, nullable=False)
    y_vec_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    obs_idx_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    r_diag_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    x_post_json: Mapped[str] = mapped_column(Text, nullable=False)
    P_post_json: Mapped[str] = mapped_column(Text, nullable=False)


class AlcoaAuditRow(Base):
    __tablename__ = "alcoa_audit_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("twin_sessions.session_id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    utc_timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    process_t_h: Mapped[float] = mapped_column(Float, nullable=False)
    params_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    prev_entry_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, default=GENESIS_ENTRY_SHA256
    )
    detail_json: Mapped[str] = mapped_column(Text, nullable=False)
    entry_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


def create_db_engine_and_sessionmaker(database_url: Optional[str] = None):
    """Create SQLAlchemy engine & sessionmaker (defaults to local SQLite)."""
    url = database_url or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    audit_cols = {col["name"] for col in inspector.get_columns("alcoa_audit_entries")}
    if "prev_entry_sha256" not in audit_cols:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE alcoa_audit_entries ADD COLUMN prev_entry_sha256 VARCHAR(64) NOT NULL DEFAULT ''"
                )
            )
    session_cols = {col["name"] for col in inspector.get_columns("twin_sessions")}
    if "ekf_gate_state_json" not in session_cols:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE twin_sessions ADD COLUMN ekf_gate_state_json TEXT")
            )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return engine, SessionLocal


def save_twin_session(
    db: Session,
    session_id: str,
    vessel_scale_L: int,
    twin: BioreactorDigitalTwin,
) -> None:
    """Persist or update a BioreactorDigitalTwin session, its EKF events, and ALCOA+ audit log."""
    row = db.get(TwinSessionRow, session_id)
    params_json = json.dumps(twin.params.to_dict(), sort_keys=True)
    meta_json = json.dumps(twin.calibration_metadata, sort_keys=True, default=str)
    last_u_json = json.dumps(serialize_operating_inputs(twin.last_u), sort_keys=True)
    gate_state_json = json.dumps(
        {
            "last_nis": float(twin.ekf.last_nis),
            "last_spe": float(twin.ekf.last_spe),
            "last_obs_dim": int(twin.ekf.last_obs_dim),
            "last_nis_threshold": float(twin.ekf.last_nis_threshold),
            "last_spe_threshold": float(twin.ekf.last_spe_threshold),
            "last_nis_breach": bool(twin.ekf.last_nis_breach),
            "last_spe_breach": bool(twin.ekf.last_spe_breach),
            "last_residual_breach": bool(twin.ekf.last_residual_breach),
            "spe_persist_streak": int(twin.ekf._spe_persist_streak),
            "gate_alpha": float(twin.ekf.gate_alpha),
            "nis_threshold_override": twin.ekf._nis_threshold_override,
            "spe_threshold_override": twin.ekf._spe_threshold_override,
            "spe_persist_threshold_override": twin.ekf._spe_persist_threshold_override,
            "spe_persist_steps": int(twin.ekf.spe_persist_steps),
        },
        sort_keys=True,
    )

    if row is None:
        row = TwinSessionRow(
            session_id=session_id,
            batch_id=twin.batch_id,
            operator_id=twin.operator_id,
            vessel_scale_L=int(vessel_scale_L),
            params_json=params_json,
            calibration_metadata_json=meta_json,
            params_sha256=twin.params_sha256,
            model_sha256=twin.model_sha256,
            t_now_h=float(twin.t_now),
            last_u_json=last_u_json,
            ekf_gate_state_json=gate_state_json,
        )
        db.add(row)
    else:
        row.t_now_h = float(twin.t_now)
        row.last_u_json = last_u_json
        row.ekf_gate_state_json = gate_state_json
        row.updated_at = datetime.now(timezone.utc)

    # Replace events (atline replay can insert & update later events deterministically)
    db.execute(delete(TwinEventRow).where(TwinEventRow.session_id == session_id))
    for idx, ev in enumerate(twin._events):
        db.add(
            TwinEventRow(
                session_id=session_id,
                seq_idx=idx,
                kind=str(ev["kind"]),
                t_h=float(ev["t_h"]),
                u_json=json.dumps(serialize_operating_inputs(ev["u"]), sort_keys=True),
                y_vec_json=None if "y_vec" not in ev else json.dumps(np.asarray(ev["y_vec"]).tolist()),
                obs_idx_json=None if "obs_idx" not in ev else json.dumps(list(ev["obs_idx"])),
                r_diag_json=None if "r_diag" not in ev else json.dumps(np.asarray(ev["r_diag"]).tolist()),
                x_post_json=json.dumps(np.asarray(ev["x_post"]).tolist()),
                P_post_json=json.dumps(np.asarray(ev["P_post"]).tolist()),
            )
        )

    # Append only new ALCOA+ audit entries (preserving original UTC timestamps and entry hashes)
    existing_seqs = {
        s
        for (s,) in db.execute(
            select(AlcoaAuditRow.seq).where(AlcoaAuditRow.session_id == session_id)
        ).all()
    }
    for entry in twin.audit_log:
        if int(entry["seq"]) not in existing_seqs:
            db.add(
                AlcoaAuditRow(
                    session_id=session_id,
                    seq=int(entry["seq"]),
                    utc_timestamp=str(entry["utc_timestamp"]),
                    event_type=str(entry["event_type"]),
                    process_t_h=float(entry["process_t_h"]),
                    params_sha256=str(entry["params_sha256"]),
                    model_sha256=str(entry["model_sha256"]),
                    prev_entry_sha256=str(entry.get("prev_entry_sha256", "")),
                    detail_json=json.dumps(entry["detail"], sort_keys=True, default=str),
                    entry_sha256=str(entry["entry_sha256"]),
                )
            )
    db.commit()


def load_twin_session(
    db: Session,
    session_id: str,
    residual_model: ResidualCorrectorProtocol,
) -> Optional[Dict[str, Any]]:
    """Rehydrate a BioreactorDigitalTwin from SQLite/PostgreSQL with exact EKF state and ALCOA+ chain."""
    row = db.get(TwinSessionRow, session_id)
    if row is None:
        return None

    params = BioreactorParams(**json.loads(row.params_json))
    cal_meta = json.loads(row.calibration_metadata_json)

    ev_rows: List[TwinEventRow] = list(
        db.execute(
            select(TwinEventRow)
            .where(TwinEventRow.session_id == session_id)
            .order_by(TwinEventRow.seq_idx.asc())
        ).scalars()
    )
    audit_rows: List[AlcoaAuditRow] = list(
        db.execute(
            select(AlcoaAuditRow)
            .where(AlcoaAuditRow.session_id == session_id)
            .order_by(AlcoaAuditRow.seq.asc())
        ).scalars()
    )

    events: List[Dict[str, Any]] = []
    for ev in ev_rows:
        item: Dict[str, Any] = {
            "kind": ev.kind,
            "t_h": float(ev.t_h),
            "u": deserialize_operating_inputs(json.loads(ev.u_json)),
            "x_post": np.asarray(json.loads(ev.x_post_json), dtype=float),
            "P_post": np.asarray(json.loads(ev.P_post_json), dtype=float),
        }
        if ev.y_vec_json is not None:
            item["y_vec"] = np.asarray(json.loads(ev.y_vec_json), dtype=float)
        if ev.obs_idx_json is not None:
            item["obs_idx"] = list(json.loads(ev.obs_idx_json))
        if ev.r_diag_json is not None:
            item["r_diag"] = np.asarray(json.loads(ev.r_diag_json), dtype=float)
        events.append(item)

    x0 = events[0]["x_post"] if events else np.zeros(7, dtype=float)
    twin = BioreactorDigitalTwin(
        x0=x0,
        params=params,
        residual_model=residual_model,
        calibration_metadata=cal_meta,
        operator_id=row.operator_id,
        batch_id=row.batch_id,
    )
    twin.t_now = float(row.t_now_h)
    twin.last_u = deserialize_operating_inputs(json.loads(row.last_u_json))
    twin._events = events
    if events:
        twin.ekf = twin._build_ekf_at(events[-1]["x_post"], events[-1]["P_post"])

    twin.audit_log = [
        {
            "seq": int(a.seq),
            "utc_timestamp": a.utc_timestamp,
            "batch_id": row.batch_id,
            "operator_id": row.operator_id,
            "event_type": a.event_type,
            "process_t_h": float(a.process_t_h),
            "params_sha256": a.params_sha256,
            "model_sha256": a.model_sha256,
            "prev_entry_sha256": a.prev_entry_sha256,
            "detail": json.loads(a.detail_json),
            "entry_sha256": a.entry_sha256,
        }
        for a in audit_rows
    ]
    if row.ekf_gate_state_json is not None:
        gate_state = json.loads(row.ekf_gate_state_json)
        if "gate_alpha" in gate_state:
            twin.ekf.gate_alpha = float(gate_state["gate_alpha"])
        twin.ekf._nis_threshold_override = gate_state.get("nis_threshold_override")
        twin.ekf._spe_threshold_override = gate_state.get("spe_threshold_override")
        twin.ekf._spe_persist_threshold_override = gate_state.get("spe_persist_threshold_override")
        if "spe_persist_steps" in gate_state:
            twin.ekf.spe_persist_steps = max(1, int(gate_state["spe_persist_steps"]))
        twin.ekf.last_nis = float(gate_state.get("last_nis", 0.0))
        twin.ekf.last_spe = float(gate_state.get("last_spe", 0.0))
        twin.ekf.last_obs_dim = int(gate_state.get("last_obs_dim", 0))
        twin.ekf.last_nis_threshold = float(gate_state.get("last_nis_threshold", twin.ekf.last_nis_threshold))
        twin.ekf.last_spe_threshold = float(gate_state.get("last_spe_threshold", twin.ekf.last_spe_threshold))
        twin.ekf.last_nis_breach = bool(gate_state.get("last_nis_breach", False))
        twin.ekf.last_spe_breach = bool(gate_state.get("last_spe_breach", False))
        twin.ekf.last_residual_breach = bool(
            gate_state.get(
                "last_residual_breach",
                twin.ekf.last_nis_breach or twin.ekf.last_spe_breach,
            )
        )
        twin.ekf._spe_persist_streak = int(gate_state.get("spe_persist_streak", 0))
    else:
        for entry in reversed(twin.audit_log):
            detail = entry.get("detail", {})
            if isinstance(detail, dict) and "nis" in detail:
                twin.ekf.last_nis = float(detail["nis"])
                twin.ekf.last_spe = float(detail.get("spe", 0.0))
                if "obs_dim" in detail and int(detail["obs_dim"]) > 0:
                    obs_dim = int(detail["obs_dim"])
                elif entry.get("event_type") == "ONLINE_PAT":
                    obs_dim = 3
                elif entry.get("event_type") in ("ATLINE_LIMS", "ATLINE_LIMS_REPLAY"):
                    obs_dim = 2 if detail.get("y_P") is None else 3
                else:
                    obs_dim = 3
                twin.ekf.last_obs_dim = obs_dim
                chi2_crit = float(chi2.ppf(twin.ekf.gate_alpha, df=max(1, obs_dim)))
                nis_thresh = (
                    chi2_crit
                    if twin.ekf._nis_threshold_override is None
                    else twin.ekf._nis_threshold_override
                )
                spe_thresh = (
                    (10.0 * chi2_crit)
                    if twin.ekf._spe_threshold_override is None
                    else twin.ekf._spe_threshold_override
                )
                twin.ekf.last_nis_threshold = float(nis_thresh)
                twin.ekf.last_spe_threshold = float(spe_thresh)
                nis_breach = (
                    bool(detail["nis_breach"])
                    if "nis_breach" in detail
                    else bool(twin.ekf.last_nis > nis_thresh)
                )
                spe_breach = (
                    bool(detail["spe_breach"])
                    if "spe_breach" in detail
                    else bool(
                        twin.ekf.last_spe > spe_thresh
                        or (bool(detail.get("residual_breach", False)) and not nis_breach)
                    )
                )
                res_breach = bool(detail.get("residual_breach", False) or nis_breach or spe_breach)
                twin.ekf.last_nis_breach = nis_breach
                twin.ekf.last_spe_breach = spe_breach
                twin.ekf.last_residual_breach = res_breach
                twin.ekf._spe_persist_streak = int(detail.get("spe_persist_streak", 0))
                break
    return {"twin": twin, "vessel_scale_L": int(row.vessel_scale_L)}
