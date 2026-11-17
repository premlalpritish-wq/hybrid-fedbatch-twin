"""FastAPI Digital Twin service wrapping BioreactorDigitalTwin, simulate_bioreactor, and benchmark_controllers."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from agents.deviation_graph import run_deviation_investigation
from agents.observability import evaluate_capa_report
from hybrid_fedbatch_twin.controller import ClosedLoopFeedController, benchmark_controllers
from hybrid_fedbatch_twin.mechanistic import (
    OperatingInputs,
    default_initial_state,
    simulate_bioreactor,
)

from .db import create_db_engine_and_sessionmaker, load_twin_session, save_twin_session
from .runtime import (
    create_twin_instance,
    load_calibrated_assets,
    scale_feed_cap,
    scale_params,
    state_dict_from_vector,
)
from .schemas import (
    AtlineAssayRequest,
    AtlineAssayResponse,
    AuditRecordResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    ControlActionRequest,
    ControlActionResponse,
    InvestigateRequest,
    InvestigateResponse,
    PatIngestRequest,
    PatIngestResponse,
    SimulateRequest,
    SimulateResponse,
    TwinInitRequest,
    TwinStateResponse,
)


def create_app(database_url: Optional[str] = None) -> FastAPI:
    """Create FastAPI app backed by SQLAlchemy ALCOA+ persistence."""
    app = FastAPI(
        title="hybrid-fedbatch-twin Digital Twin Service",
        version="0.1.0",
        description="GMP-oriented EKF state estimation, delayed-LIMS replay, and ALCOA+ ledger API.",
    )
    engine, SessionLocal = create_db_engine_and_sessionmaker(database_url)
    app.state.database_url = database_url
    app.state.db_engine = engine
    app.state.session_local = SessionLocal

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @app.get("/health")
    def health() -> dict:
        base_params, model = load_calibrated_assets()
        return {
            "status": "ok",
            "model_sha256": model.weights_sha256[:12] if model.weights_sha256 else "untrained",
            "n_bootstrap": len(model.ensemble_weights),
            "default_scale_L": int(base_params.V_vessel),
        }

    @app.post("/twins", response_model=TwinStateResponse, status_code=201)
    def init_twin(req: TwinInitRequest, db: Session = Depends(get_db)) -> TwinStateResponse:
        session_id = uuid4().hex[:16]
        twin = create_twin_instance(
            vessel_scale_L=req.vessel_scale_L,
            batch_id=req.batch_id,
            operator_id=req.operator_id,
            calibration_metadata=req.calibration_metadata,
        )
        save_twin_session(db, session_id, req.vessel_scale_L, twin)
        rates = twin.ekf.current_rates(twin.last_u)
        x_std = twin.ekf.P.diagonal().clip(min=0.0) ** 0.5
        return TwinStateResponse(
            session_id=session_id,
            batch_id=twin.batch_id,
            operator_id=twin.operator_id,
            vessel_scale_L=req.vessel_scale_L,
            t_now_h=float(twin.t_now),
            current_state=state_dict_from_vector(twin.ekf.x),
            current_std=state_dict_from_vector(x_std),
            params_sha256=twin.params_sha256,
            model_sha256=twin.model_sha256,
            d_m=float(rates["d_m"]),
            in_domain=bool(rates["in_domain"] > 0.5),
        )

    @app.get("/twins/{session_id}", response_model=TwinStateResponse)
    def get_twin(session_id: str, db: Session = Depends(get_db)) -> TwinStateResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        rates = twin.ekf.current_rates(twin.last_u)
        x_std = twin.ekf.P.diagonal().clip(min=0.0) ** 0.5
        return TwinStateResponse(
            session_id=session_id,
            batch_id=twin.batch_id,
            operator_id=twin.operator_id,
            vessel_scale_L=loaded["vessel_scale_L"],
            t_now_h=float(twin.t_now),
            current_state=state_dict_from_vector(twin.ekf.x),
            current_std=state_dict_from_vector(x_std),
            params_sha256=twin.params_sha256,
            model_sha256=twin.model_sha256,
            d_m=float(rates["d_m"]),
            in_domain=bool(rates["in_domain"] > 0.5),
        )

    @app.post("/twins/{session_id}/pat", response_model=PatIngestResponse)
    def ingest_pat(
        session_id: str, req: PatIngestRequest, db: Session = Depends(get_db)
    ) -> PatIngestResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        try:
            res = twin.ingest_online_pat(
                t_h=req.t_h,
                y_Xv=req.y_Xv,
                y_CL=req.y_CL,
                y_V=req.y_V,
                F_act=req.F_act,
                rpm_act=req.rpm_act,
                vvm_act=req.vvm_act,
                sig_online=req.sig_online,
                S_feed_act=req.S_feed_act,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        save_twin_session(db, session_id, loaded["vessel_scale_L"], twin)
        return PatIngestResponse(
            session_id=session_id,
            t_h=res["t_h"],
            x_hat=state_dict_from_vector(res["x_hat"]),
            x_std=state_dict_from_vector(res["x_std"]),
            mu_hat=res["mu_hat"],
            k_La_hat=res["k_La_hat"],
            k_d_hat=res["k_d_hat"],
            q_A_net_hat=res["q_A_net_hat"],
            d_m=res["d_m"],
            d_m_kla=res["d_m_kla"],
            in_domain=res["in_domain"],
            nis=res["nis"],
        )

    @app.post("/twins/{session_id}/atline", response_model=AtlineAssayResponse)
    def ingest_atline(
        session_id: str, req: AtlineAssayRequest, db: Session = Depends(get_db)
    ) -> AtlineAssayResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        try:
            res = twin.ingest_atline_assay(
                t_sample_h=req.t_sample_h,
                y_S=req.y_S,
                y_A=req.y_A,
                y_P=req.y_P,
                sig_atline=req.sig_atline,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        save_twin_session(db, session_id, loaded["vessel_scale_L"], twin)
        return AtlineAssayResponse(
            session_id=session_id,
            t_now=res["t_now"],
            t_sample_h=res["t_sample_h"],
            replayed_steps=res["replayed_steps"],
            x_hat=state_dict_from_vector(res["x_hat"]),
            x_std=state_dict_from_vector(res["x_std"]),
            mu_hat=res["mu_hat"],
            d_m=res["d_m"],
            in_domain=res["in_domain"],
        )

    @app.post("/twins/{session_id}/control", response_model=ControlActionResponse)
    def compute_control(
        session_id: str, req: ControlActionRequest, db: Session = Depends(get_db)
    ) -> ControlActionResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        f_max = req.F_max if req.F_max is not None else scale_feed_cap(loaded["vessel_scale_L"])
        ctrl = ClosedLoopFeedController(
            params=twin.params,
            residual_model=model,
            mu_target=req.mu_target,
            S_set=req.S_set,
            A_limit=req.A_limit,
            rpm=req.rpm,
            vvm=req.vvm,
            F_max=f_max,
            disturbance_mode=req.disturbance_mode,
        )
        act = twin.compute_control_action(req.t_h, ctrl)
        save_twin_session(db, session_id, loaded["vessel_scale_L"], twin)
        return ControlActionResponse(
            session_id=session_id,
            t_h=act["t_h"],
            F_cmd=act["F_cmd"],
            rpm_cmd=act["rpm_cmd"],
            vvm_cmd=act["vvm_cmd"],
            mu_hat=act["mu_hat"],
            k_La_hat=act["k_La_hat"],
            d_m=act["d_m"],
            in_domain=act["in_domain"],
            x_hat=state_dict_from_vector(act["x_hat"]),
            x_std=state_dict_from_vector(act["x_std"]),
        )

    @app.get("/twins/{session_id}/audit", response_model=AuditRecordResponse)
    def get_audit(session_id: str, db: Session = Depends(get_db)) -> AuditRecordResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        record = loaded["twin"].to_audit_record()
        return AuditRecordResponse(session_id=session_id, **record)

    @app.post("/twins/{session_id}/investigate", response_model=InvestigateResponse)
    def investigate_deviation(
        session_id: str, req: InvestigateRequest, db: Session = Depends(get_db)
    ) -> InvestigateResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        report, trace = run_deviation_investigation(
            session_id=session_id,
            vessel_scale_L=loaded["vessel_scale_L"],
            database_url=app.state.database_url,
            pat_tick=None if req.pat_tick is None else req.pat_tick.model_dump(),
            delayed_lims=None if req.delayed_lims is None else req.delayed_lims.model_dump(),
            disturbance_mode=req.disturbance_mode,
            duration_h=req.duration_h,
            mu_target=req.mu_target,
            use_llm=req.use_llm,
        )
        evaluation = evaluate_capa_report(report, trace, database_url=app.state.database_url)
        return InvestigateResponse(
            capa_report=report.model_dump(),
            trace=trace.model_dump(),
            evaluation=evaluation.model_dump(),
        )

    @app.post("/simulate", response_model=SimulateResponse)
    def run_simulation(req: SimulateRequest) -> SimulateResponse:
        base_params, model = load_calibrated_assets()
        p_scaled = scale_params(req.vessel_scale_L, base_params)
        x0 = default_initial_state(p_scaled)
        f_max = req.F_max if req.F_max is not None else scale_feed_cap(req.vessel_scale_L)
        ctrl = ClosedLoopFeedController(
            params=p_scaled,
            residual_model=model,
            mu_target=req.mu_target,
            rpm=req.rpm,
            vvm=req.vvm,
            F_max=f_max,
            disturbance_mode=req.disturbance_mode,
        )
        sim = simulate_bioreactor(
            t_span=(0.0, req.duration_h),
            x0=x0,
            control_fn=ctrl,
            params=p_scaled,
            mode=req.mode,
            residual_model=model if req.mode == "hybrid" else None,
            dt_eval=req.dt_eval,
        )
        return SimulateResponse(
            vessel_scale_L=req.vessel_scale_L,
            mode=req.mode,
            n_points=len(sim["t"]),
            t=sim["t"].tolist(),
            X_v=sim["X_v"].tolist(),
            X_d=sim["X_d"].tolist(),
            S=sim["S"].tolist(),
            C_L=sim["C_L"].tolist(),
            DO_pct=sim["DO_pct"].tolist(),
            A=sim["A"].tolist(),
            P=sim["P"].tolist(),
            V=sim["V"].tolist(),
            F=sim["F"].tolist(),
            mu=sim["mu"].tolist(),
            k_La=sim["k_La"].tolist(),
        )

    @app.post("/benchmark", response_model=BenchmarkResponse)
    def run_benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
        base_params, model = load_calibrated_assets()
        p_scaled = scale_params(req.vessel_scale_L, base_params)
        f_max = req.F_max if req.F_max is not None else scale_feed_cap(req.vessel_scale_L)
        bench = benchmark_controllers(
            params=p_scaled,
            residual_model=model,
            duration_h=req.duration_h,
            mu_target=req.mu_target,
            rpm=req.rpm,
            vvm=req.vvm,
            F_max=f_max,
            plant_params=p_scaled,
        )
        strategies = {
            name: {k: float(v) for k, v in entry["summary"].items()}
            for name, entry in bench.items()
        }
        return BenchmarkResponse(
            vessel_scale_L=req.vessel_scale_L,
            duration_h=req.duration_h,
            strategies=strategies,
        )

    return app


app = create_app()
