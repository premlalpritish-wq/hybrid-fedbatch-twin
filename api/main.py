"""FastAPI Digital Twin service wrapping BioreactorDigitalTwin, OT/LIMS adapters, CAPA workflow, and Part 11 QA."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from agents.deviation_graph import PROMPT_SHA256, PROMPT_TEMPLATE_VERSION, run_governed_capa_workflow
from agents.guardrails import validate_capa_draft
from agents.observability import evaluate_capa_report
from agents.part11 import (
    FinalizedRecordImmutabilityError,
    LlmExecutionMetadata,
    TwoFactorReplayError,
    build_part11_decision_record,
    build_part1_raw_output,
    canonical_sha256,
    verify_part11_decision_chain,
)
from hybrid_fedbatch_twin.controller import ClosedLoopFeedController, benchmark_controllers
from hybrid_fedbatch_twin.mechanistic import (
    default_initial_state,
    default_operating_inputs,
    simulate_bioreactor,
)
from mlops.telemetry import (
    get_telemetry_recorder,
    record_api_request,
    record_control_fallback,
    record_ekf_step,
)

from .adapters import ingest_adapter_into_twin
from .db import (
    create_db_engine_and_sessionmaker,
    get_latest_ai_decision_hash,
    get_staged_capa_draft,
    is_decision_record_finalized,
    load_ai_decision_records,
    load_server_evidence_context,
    load_twin_session,
    save_ai_decision_record,
    save_twin_session,
)
from .runtime import (
    create_twin_instance,
    load_calibrated_assets,
    scale_feed_cap,
    scale_params,
    state_dict_from_vector,
)
from .schemas import (
    AdapterIngestRequest,
    AdapterIngestResponse,
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
    QaApprovalRequest,
    QaApprovalResponse,
    SimulateRequest,
    SimulateResponse,
    TwinInitRequest,
    TwinStateResponse,
    sanitize_truth_payload,
)


def create_app(database_url: Optional[str] = None) -> FastAPI:
    """Create FastAPI app backed by SQLAlchemy ALCOA+ persistence and Part 11 QA governance."""
    app = FastAPI(
        title="hybrid-fedbatch-twin Digital Twin Service",
        version="0.2.0",
        description="GMP-oriented EKF state estimation, OT/LIMS adapters, delayed-LIMS replay, and ALCOA+/Part 11 API.",
    )
    engine, SessionLocal = create_db_engine_and_sessionmaker(database_url)
    app.state.database_url = database_url
    app.state.db_engine = engine
    app.state.session_local = SessionLocal

    @app.middleware("http")
    async def telemetry_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        route_obj = request.scope.get("route")
        route_path = getattr(route_obj, "path", request.url.path)
        record_api_request(request.method, str(route_path), response.status_code, dur_ms)
        return response

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

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        """Expose Prometheus-formatted offline telemetry metrics."""
        return PlainTextResponse(
            get_telemetry_recorder().render_prometheus_text(),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/twins", response_model=TwinStateResponse, status_code=201)
    def init_twin(req: TwinInitRequest, db: Session = Depends(get_db)) -> TwinStateResponse:
        session_id = uuid4().hex[:16]
        twin = create_twin_instance(
            vessel_scale_L=req.vessel_scale_L,
            batch_id=req.batch_id,
            operator_id=req.operator_id,
            calibration_metadata=sanitize_truth_payload(req.calibration_metadata, show_truth=False),
        )
        save_twin_session(db, session_id, req.vessel_scale_L, twin)
        rates = twin.ekf.current_rates(twin.last_u)
        gate = twin.ekf.evaluate_gate(rates=rates)
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
            d_m=float(gate["d_m"]),
            in_domain=bool(gate["in_domain"]),
            undecided=bool(gate["undecided"]),
            residual_breach=bool(gate["residual_breach"]),
            gate_reason=str(gate["gate_reason"]),
        )

    @app.get("/twins/{session_id}", response_model=TwinStateResponse)
    def get_twin(session_id: str, db: Session = Depends(get_db)) -> TwinStateResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        rates = twin.ekf.current_rates(twin.last_u)
        gate = twin.ekf.evaluate_gate(rates=rates)
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
            d_m=float(gate["d_m"]),
            in_domain=bool(gate["in_domain"]),
            undecided=bool(gate["undecided"]),
            residual_breach=bool(gate["residual_breach"]),
            gate_reason=str(gate["gate_reason"]),
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
        record_ekf_step(
            session_id=session_id,
            nis=float(res["nis"]),
            spe=float(res["spe"]),
            d_m=float(res["d_m"]),
            residual_breach=bool(res["residual_breach"]),
            undecided=bool(res["undecided"]),
            gate_reason=str(res["gate_reason"]),
        )
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
            undecided=res["undecided"],
            residual_breach=res["residual_breach"],
            gate_reason=res["gate_reason"],
            nis=res["nis"],
            spe=res["spe"],
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
        record_ekf_step(
            session_id=session_id,
            nis=float(res["nis"]),
            spe=float(res["spe"]),
            d_m=float(res["d_m"]),
            residual_breach=bool(res["residual_breach"]),
            undecided=bool(res["undecided"]),
            gate_reason=str(res["gate_reason"]),
        )
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
            undecided=res["undecided"],
            residual_breach=res["residual_breach"],
            gate_reason=res["gate_reason"],
            nis=res["nis"],
            spe=res["spe"],
            state_delta_l2=res["state_delta_l2"],
        )

    @app.post("/twins/{session_id}/ingest", response_model=AdapterIngestResponse)
    def ingest_adapter_payload(
        session_id: str, req: AdapterIngestRequest, db: Session = Depends(get_db)
    ):
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        res = ingest_adapter_into_twin(twin, req.protocol, req.payload)
        save_twin_session(db, session_id, loaded["vessel_scale_L"], twin)

        payload_out = AdapterIngestResponse(
            session_id=session_id,
            protocol=res["protocol"],
            status=res["status"],
            payload_kind=res["payload_kind"],
            raw_payload_sha256=res["raw_payload_sha256"],
            alcoa_entry_sha256=res["alcoa_entry_sha256"],
            alcoa_prev_entry_sha256=res["alcoa_prev_entry_sha256"],
            unit_conversions=res["unit_conversions"],
            quarantine_reason_codes=res["quarantine_reason_codes"],
            normalized_payload=res["normalized_payload"],
            ekf_result=res["ekf_result"],
        )
        if res["status"] == "quarantined":
            return JSONResponse(
                status_code=422,
                content={**payload_out.model_dump(), "detail": res.get("detail", "Payload quarantined.")},
            )

        ekf_res = res["ekf_result"] or {}
        record_ekf_step(
            session_id=session_id,
            nis=float(ekf_res.get("nis", 0.0)),
            spe=float(ekf_res.get("spe", 0.0)),
            d_m=float(ekf_res.get("d_m", 0.0)),
            residual_breach=bool(ekf_res.get("residual_breach", False)),
            undecided=bool(ekf_res.get("undecided", False)),
            gate_reason=str(ekf_res.get("gate_reason", "nominal")),
        )
        return payload_out

    @app.get("/twins/{session_id}/quarantine")
    def get_quarantine_entries(session_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        q_entries = sanitize_truth_payload(
            [e for e in twin.audit_log if e.get("event_type") == "ADAPTER_QUARANTINE"],
            show_truth=False,
        )
        chain_ok, errs = twin.verify_audit_chain()
        return {
            "session_id": session_id,
            "quarantine_count": len(q_entries),
            "chain_valid": chain_ok,
            "chain_errors": errs,
            "quarantine_entries": q_entries,
        }

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
        rpm_val = float(req.rpm) if req.rpm is not None else float(twin.last_u.rpm)
        vvm_val = float(req.vvm) if req.vvm is not None else float(twin.last_u.vvm)
        ctrl = ClosedLoopFeedController(
            params=twin.params,
            residual_model=model,
            mu_target=req.mu_target,
            S_set=req.S_set,
            A_limit=req.A_limit,
            rpm=rpm_val,
            vvm=vvm_val,
            F_max=f_max,
            disturbance_mode=req.disturbance_mode,
        )
        act = twin.compute_control_action(req.t_h, ctrl)
        save_twin_session(db, session_id, loaded["vessel_scale_L"], twin)
        record_control_fallback(
            session_id=session_id,
            fallback_active=bool(act["fallback_active"]),
            control_mode=str(act["control_mode"]),
            gate_reason=str(act["gate_reason"]),
            escalated=act["escalation_event"] is not None,
        )
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
            undecided=act["undecided"],
            residual_breach=act["residual_breach"],
            gate_reason=act["gate_reason"],
            fallback_active=act["fallback_active"],
            control_mode=act["control_mode"],
            escalation_event=act["escalation_event"],
            nis=act["nis"],
            spe=act["spe"],
            x_hat=state_dict_from_vector(act["x_hat"]),
            x_std=state_dict_from_vector(act["x_std"]),
        )

    @app.get("/twins/{session_id}/audit", response_model=AuditRecordResponse)
    def get_audit(
        session_id: str,
        max_entries: Optional[int] = Query(default=None, ge=1),
        db: Session = Depends(get_db),
    ) -> AuditRecordResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        record = loaded["twin"].to_audit_record(max_entries=max_entries)
        return AuditRecordResponse(session_id=session_id, **sanitize_truth_payload(record, show_truth=False))

    @app.post("/twins/{session_id}/investigate", response_model=InvestigateResponse)
    def investigate_deviation(
        session_id: str, req: InvestigateRequest, db: Session = Depends(get_db)
    ) -> InvestigateResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        outcome = run_governed_capa_workflow(
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
        evaluation = evaluate_capa_report(
            outcome.capa_report, outcome.trace, database_url=app.state.database_url
        )
        return InvestigateResponse(
            capa_report=outcome.capa_report.model_dump(),
            trace=outcome.trace.model_dump(),
            evaluation=evaluation.model_dump(),
            guardrail_result=outcome.guardrail_result.model_dump(),
            staged_draft=outcome.staged_draft,
        )

    @app.post("/twins/{session_id}/qa/approve", response_model=QaApprovalResponse)
    def approve_capa_decision(
        session_id: str, req: QaApprovalRequest, db: Session = Depends(get_db)
    ) -> QaApprovalResponse:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        twin = loaded["twin"]
        live_alcoa_sha = str(twin.to_audit_record()["record_sha256"])

        if req.decision_record_id and is_decision_record_finalized(db, req.decision_record_id):
            raise HTTPException(
                status_code=409,
                detail=f"Finalized Part 11 record '{req.decision_record_id}' is immutable and cannot be overwritten.",
            )

        draft_payload: Optional[Dict[str, Any]] = None
        if req.staged_draft_id:
            staged = get_staged_capa_draft(db, req.staged_draft_id)
            if staged is None or staged["session_id"] != session_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Staged draft '{req.staged_draft_id}' not found for session '{session_id}'.",
                )
            draft_payload = staged["draft_payload"]
        elif req.capa_report is not None:
            draft_payload = dict(req.capa_report)
        else:
            raise HTTPException(
                status_code=422,
                detail="Either staged_draft_id or capa_report must be provided for QA approval.",
            )

        ctx_sha = str(draft_payload.get("context_sha256", ""))
        server_ctx = load_server_evidence_context(db, session_id, ctx_sha)
        if server_ctx is None:
            raise HTTPException(
                status_code=422,
                detail="CAPA draft must reference a server-verified investigation evidence context; run /investigate first.",
            )
        if req.evidence_context is not None and canonical_sha256(req.evidence_context) != canonical_sha256(server_ctx):
            raise HTTPException(
                status_code=422,
                detail="Client-supplied evidence_context does not match server-verified context.",
            )

        gr = validate_capa_draft(
            draft_payload,
            server_ctx,
            expected_alcoa_sha256=live_alcoa_sha,
        )
        guardrail_ok = gr.passed
        guardrail_violations = [f"{v.code} ({v.field}): {v.detail}" for v in gr.violations]

        if not guardrail_ok and req.redlines.decision != "REJECTED":
            raise HTTPException(
                status_code=422,
                detail=f"Cannot QA-approve draft failing deterministic guardrails: {guardrail_violations}",
            )

        raw_llm_meta = req.llm_execution_metadata or draft_payload.get("llm_execution_metadata")
        try:
            llm_meta = (
                LlmExecutionMetadata.model_validate(raw_llm_meta)
                if raw_llm_meta
                else LlmExecutionMetadata(
                    provider="deterministic_ci_provider",
                    model_id="deterministic-bioprocess-capa-v2",
                    prompt_template_version=PROMPT_TEMPLATE_VERSION,
                    prompt_sha256=PROMPT_SHA256,
                    context_sha256=ctx_sha,
                    deterministic_ci_mode=True,
                )
            )
            part1 = build_part1_raw_output(
                raw_output_payload=draft_payload,
                context_sha256=ctx_sha,
                guardrail_passed=guardrail_ok,
                guardrail_violations=guardrail_violations,
            )
            next_seq, prev_sha = get_latest_ai_decision_hash(db, session_id)
            record = build_part11_decision_record(
                session_id=session_id,
                part1=part1,
                reviewer_id=req.redlines.reviewer_id,
                decision=req.redlines.decision,
                revised_overrides=req.redlines.revised_overrides,
                rationales=req.redlines.rationales,
                signer_id=req.two_factor_esig.signer_id,
                signer_name=req.two_factor_esig.signer_name,
                signer_role=req.two_factor_esig.signer_role,
                signing_meaning=req.two_factor_esig.signing_meaning,
                factor_1_type=req.two_factor_esig.factor_1_type,
                factor_2_type=req.two_factor_esig.factor_2_type,
                factor_2_challenge_id=req.two_factor_esig.factor_2_challenge_id,
                linked_twin_alcoa_sha256=live_alcoa_sha,
                llm_execution_metadata=llm_meta,
                seq=next_seq,
                prev_entry_sha256=prev_sha,
                decision_record_id=req.decision_record_id,
            )
            save_ai_decision_record(db, record)
            all_records = load_ai_decision_records(db, session_id)
            chain_ok, errs = verify_part11_decision_chain(all_records)
        except (FinalizedRecordImmutabilityError, TwoFactorReplayError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return QaApprovalResponse(
            decision_record=record.model_dump(),
            chain_valid=chain_ok,
            verification_errors=errs,
        )

    @app.get("/twins/{session_id}/qa/records")
    def get_qa_decision_records(session_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
        _, model = load_calibrated_assets()
        loaded = load_twin_session(db, session_id, model)
        if loaded is None:
            raise HTTPException(status_code=404, detail=f"Twin session '{session_id}' not found.")
        records = load_ai_decision_records(db, session_id)
        chain_ok, errs = verify_part11_decision_chain(records)
        return {
            "session_id": session_id,
            "record_count": len(records),
            "chain_valid": chain_ok,
            "verification_errors": errs,
            "records": records,
        }

    @app.post("/simulate", response_model=SimulateResponse)
    def run_simulation(req: SimulateRequest) -> SimulateResponse:
        if req.mode == "ground_truth" and not req.show_truth:
            raise HTTPException(
                status_code=403,
                detail="Truth isolation violation: mode='ground_truth' requires show_truth=True.",
            )
        base_params, model = load_calibrated_assets()
        p_scaled = scale_params(req.vessel_scale_L, base_params)
        x0 = default_initial_state(p_scaled)
        u_def = default_operating_inputs(p_scaled)
        rpm_val = float(req.rpm) if req.rpm is not None else float(u_def.rpm)
        vvm_val = float(req.vvm) if req.vvm is not None else float(u_def.vvm)
        f_max = req.F_max if req.F_max is not None else scale_feed_cap(req.vessel_scale_L)
        ctrl = ClosedLoopFeedController(
            params=p_scaled,
            residual_model=model,
            mu_target=req.mu_target,
            rpm=rpm_val,
            vvm=vvm_val,
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
        u_def = default_operating_inputs(p_scaled)
        rpm_val = float(req.rpm) if req.rpm is not None else float(u_def.rpm)
        vvm_val = float(req.vvm) if req.vvm is not None else float(u_def.vvm)
        f_max = req.F_max if req.F_max is not None else scale_feed_cap(req.vessel_scale_L)
        bench = benchmark_controllers(
            params=p_scaled,
            residual_model=model,
            duration_h=req.duration_h,
            mu_target=req.mu_target,
            rpm=rpm_val,
            vvm=vvm_val,
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
