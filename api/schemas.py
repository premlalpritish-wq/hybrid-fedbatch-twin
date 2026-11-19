"""Pydantic v2 request and response schemas for the Bioreactor Digital Twin API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class TwinInitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(default="FB-BL21-001", min_length=1, max_length=128)
    operator_id: str = Field(default="bioprocess_engineer", min_length=1, max_length=128)
    vessel_scale_L: Literal[5, 50, 1000] = Field(default=5)
    calibration_metadata: Optional[Dict[str, Any]] = None


class TwinStateResponse(BaseModel):
    session_id: str
    batch_id: str
    operator_id: str
    vessel_scale_L: int
    t_now_h: float
    current_state: Dict[str, float]
    current_std: Dict[str, float]
    params_sha256: str
    model_sha256: str
    d_m: float
    in_domain: bool
    undecided: bool = False
    residual_breach: bool = False
    gate_reason: str = "nominal"


class PatIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_h: float = Field(ge=0.0)
    y_Xv: float = Field(ge=0.0)
    y_CL: float = Field(ge=0.0)
    y_V: float = Field(gt=0.0)
    F_act: float = Field(ge=0.0)
    rpm_act: float = Field(ge=50.0, le=1200.0)
    vvm_act: float = Field(ge=0.01, le=5.0)
    sig_online: Tuple[float, float, float] = (0.12, 0.08, 0.015)
    S_feed_act: Optional[float] = Field(default=None, ge=0.0)


class PatIngestResponse(BaseModel):
    session_id: str
    t_h: float
    x_hat: Dict[str, float]
    x_std: Dict[str, float]
    mu_hat: float
    k_La_hat: float
    k_d_hat: float
    q_A_net_hat: float
    d_m: float
    d_m_kla: float
    in_domain: bool
    undecided: bool = False
    residual_breach: bool = False
    gate_reason: str = "nominal"
    nis: float
    spe: float = 0.0


class AtlineAssayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_sample_h: float = Field(ge=0.0)
    y_S: float = Field(ge=0.0)
    y_A: float = Field(ge=0.0)
    y_P: Optional[float] = Field(default=None, ge=0.0)
    sig_atline: Tuple[float, float, float] = (0.06, 0.05, 0.10)


class AtlineAssayResponse(BaseModel):
    session_id: str
    t_now: float
    t_sample_h: float
    replayed_steps: int
    x_hat: Dict[str, float]
    x_std: Dict[str, float]
    mu_hat: float
    d_m: float
    in_domain: bool
    undecided: bool = False
    residual_breach: bool = False
    gate_reason: str = "nominal"
    nis: float = 0.0
    spe: float = 0.0
    state_delta_l2: float = 0.0


class ControlActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_h: float = Field(ge=0.0)
    mu_target: float = Field(default=0.28, gt=0.0, le=0.8)
    S_set: float = Field(default=0.25, ge=0.0, le=10.0)
    A_limit: float = Field(default=1.8, gt=0.0, le=20.0)
    rpm: Optional[float] = Field(default=None, ge=50.0, le=1200.0)
    vvm: Optional[float] = Field(default=None, ge=0.01, le=5.0)
    F_max: Optional[float] = Field(default=None, gt=0.0)
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none"


class ControlActionResponse(BaseModel):
    session_id: str
    t_h: float
    F_cmd: float
    rpm_cmd: float
    vvm_cmd: float
    mu_hat: float
    k_La_hat: float
    d_m: float
    in_domain: bool
    undecided: bool = False
    residual_breach: bool = False
    gate_reason: str = "nominal"
    fallback_active: bool = False
    control_mode: str = "hybrid_adaptive"
    escalation_event: Optional[Dict[str, Any]] = None
    nis: float = 0.0
    spe: float = 0.0
    x_hat: Dict[str, float]
    x_std: Dict[str, float]


class AuditRecordResponse(BaseModel):
    session_id: str
    alcoa_standard: str
    batch_id: str
    operator_id: str
    t_now_h: float
    params_sha256: str
    model_sha256: str
    chain_head_sha256: str = ""
    chain_valid: bool = True
    record_sha256: str
    calibration_metadata: Dict[str, Any]
    current_state: Dict[str, float]
    current_std: Dict[str, float]
    audit_log_count: int
    audit_log_truncated: bool = False
    audit_log: List[Dict[str, Any]]


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vessel_scale_L: Literal[5, 50, 1000] = 5
    duration_h: float = Field(default=16.0, gt=0.0, le=72.0)
    dt_eval: float = Field(default=0.2, gt=0.01, le=2.0)
    mode: Literal["hybrid", "nominal", "ground_truth"] = "hybrid"
    mu_target: float = Field(default=0.28, gt=0.0, le=0.8)
    rpm: Optional[float] = Field(default=None, ge=50.0, le=1200.0)
    vvm: Optional[float] = Field(default=None, ge=0.01, le=5.0)
    F_max: Optional[float] = Field(default=None, gt=0.0)
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none"


class SimulateResponse(BaseModel):
    vessel_scale_L: int
    mode: str
    n_points: int
    t: List[float]
    X_v: List[float]
    X_d: List[float]
    S: List[float]
    C_L: List[float]
    DO_pct: List[float]
    A: List[float]
    P: List[float]
    V: List[float]
    F: List[float]
    mu: List[float]
    k_La: List[float]


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vessel_scale_L: Literal[5, 50, 1000] = 5
    duration_h: float = Field(default=12.0, gt=1.0, le=48.0)
    mu_target: float = Field(default=0.28, gt=0.0, le=0.8)
    rpm: Optional[float] = Field(default=None, ge=50.0, le=1200.0)
    vvm: Optional[float] = Field(default=None, ge=0.01, le=5.0)
    F_max: Optional[float] = Field(default=None, gt=0.0)


class BenchmarkResponse(BaseModel):
    vessel_scale_L: int
    duration_h: float
    strategies: Dict[str, Dict[str, float]]


class InvestigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pat_tick: Optional[PatIngestRequest] = None
    delayed_lims: Optional[AtlineAssayRequest] = None
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none"
    duration_h: float = Field(default=12.0, gt=1.0, le=48.0)
    mu_target: float = Field(default=0.28, gt=0.0, le=0.8)
    use_llm: bool = False


class InvestigateResponse(BaseModel):
    capa_report: Dict[str, Any]
    trace: Dict[str, Any]
    evaluation: Dict[str, Any]
