"""Pydantic v2 request, canonical PAT/LIMS, adapter, QA, and response schemas for the Bioreactor Digital Twin API."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

C_STAR_DEFAULT_MG_L = 8.2
GLUCOSE_MW_G_MOL = 180.156
ACETATE_MW_G_MOL = 60.052

FORBIDDEN_ORACLE_KEYS = frozenset(
    {
        "u_plant",
        "plant_inputs",
        "true_states",
        "p_true",
        "mu_true",
        "kd_true",
        "kla_true",
        "qAnet_true",
        "X_v_true",
        "X_d_true",
        "S_true",
        "C_L_true",
        "A_true",
        "P_true",
        "V_true",
    }
)


def contains_forbidden_oracle_keys(obj: Any, prefix: str = "") -> List[str]:
    """Recursively scan dicts/lists for hidden oracle truth keys (u_plant, true_states, *_true)."""
    found: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            path = f"{prefix}.{ks}" if prefix else ks
            if ks in FORBIDDEN_ORACLE_KEYS or ks.endswith("_true"):
                found.append(path)
            found.extend(contains_forbidden_oracle_keys(v, path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            found.extend(contains_forbidden_oracle_keys(item, f"{prefix}[{idx}]"))
    return sorted(set(found))


def sanitize_truth_payload(payload: Any, show_truth: bool = False) -> Any:
    """Strip oracle truth keys recursively whenever show_truth is False."""
    if show_truth:
        return payload
    if isinstance(payload, dict):
        clean: Dict[str, Any] = {}
        for k, v in payload.items():
            ks = str(k)
            if ks in FORBIDDEN_ORACLE_KEYS or ks.endswith("_true"):
                continue
            clean[k] = sanitize_truth_payload(v, show_truth=False)
        return clean
    if isinstance(payload, list):
        return [sanitize_truth_payload(item, show_truth=False) for item in payload]
    return payload


_AGENT_ACTOR_PREFIXES = ("mcp_", "agent_", "llm_", "bot_", "automated_")


def _validate_human_actor_id(val: str, field_name: str) -> str:
    low = str(val).strip().lower()
    if not low or low.startswith(_AGENT_ACTOR_PREFIXES):
        raise ValueError(
            f"Automated or agent {field_name} {val!r} cannot perform human QA review or e-signature."
        )
    return str(val)


def _assert_finite(val: float, name: str) -> float:
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {val!r}.")
    return fval


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

    t_h: float = Field(ge=0.0, le=500.0)
    y_Xv: float = Field(ge=0.0, le=200.0)
    y_CL: float = Field(ge=0.0, le=25.0)
    y_V: float = Field(gt=0.0, le=50000.0)
    F_act: float = Field(ge=0.0, le=5000.0)
    rpm_act: float = Field(ge=50.0, le=1200.0)
    vvm_act: float = Field(ge=0.01, le=5.0)
    sig_online: Tuple[float, float, float] = (0.12, 0.08, 0.015)
    S_feed_act: Optional[float] = Field(default=None, ge=0.0, le=1000.0)

    @field_validator("t_h", "y_Xv", "y_CL", "y_V", "F_act", "rpm_act", "vvm_act")
    @classmethod
    def _validate_finite_scalars(cls, v: float, info) -> float:
        return _assert_finite(v, info.field_name)

    @field_validator("S_feed_act")
    @classmethod
    def _validate_optional_s_feed(cls, v: Optional[float]) -> Optional[float]:
        return None if v is None else _assert_finite(v, "S_feed_act")

    @field_validator("sig_online")
    @classmethod
    def _validate_sig_online(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        vals = tuple(_assert_finite(x, "sig_online") for x in v)
        if any(x <= 0.0 for x in vals):
            raise ValueError("sig_online entries must all be strictly positive.")
        return (vals[0], vals[1], vals[2])


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

    t_sample_h: float = Field(ge=0.0, le=500.0)
    y_S: float = Field(ge=0.0, le=100.0)
    y_A: float = Field(ge=0.0, le=50.0)
    y_P: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    sig_atline: Tuple[float, float, float] = (0.06, 0.05, 0.10)

    @field_validator("t_sample_h", "y_S", "y_A")
    @classmethod
    def _validate_finite_lims(cls, v: float, info) -> float:
        return _assert_finite(v, info.field_name)

    @field_validator("y_P")
    @classmethod
    def _validate_optional_yp(cls, v: Optional[float]) -> Optional[float]:
        return None if v is None else _assert_finite(v, "y_P")

    @field_validator("sig_atline")
    @classmethod
    def _validate_sig_atline(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        vals = tuple(_assert_finite(x, "sig_atline") for x in v)
        if any(x <= 0.0 for x in vals):
            raise ValueError("sig_atline entries must all be strictly positive.")
        return (vals[0], vals[1], vals[2])


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


class CanonicalPatPayload(BaseModel):
    """Canonical unit-normalized online PAT telemetry payload with truth-isolation guards."""

    model_config = ConfigDict(extra="forbid")

    t_h: float = Field(ge=0.0, le=500.0)
    y_Xv: float = Field(ge=0.0)
    y_Xv_unit: Literal["g/L", "kg/m3"] = "g/L"
    y_CL: float = Field(ge=0.0)
    y_CL_unit: Literal["mg/L", "%DO"] = "mg/L"
    y_V: float = Field(gt=0.0)
    y_V_unit: Literal["L", "mL", "m3"] = "L"
    F_act: float = Field(ge=0.0)
    F_unit: Literal["L/h", "mL/h", "mL/min"] = "L/h"
    rpm_act: float = Field(ge=50.0, le=1200.0)
    vvm_act: float = Field(ge=0.01, le=5.0)
    S_feed_act: Optional[float] = Field(default=None, ge=0.0, le=1000.0)
    sig_online: Tuple[float, float, float] = (0.12, 0.08, 0.015)

    @model_validator(mode="before")
    @classmethod
    def _reject_oracle_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            oracle = contains_forbidden_oracle_keys(data)
            if oracle:
                raise ValueError(f"Truth isolation violation: forbidden oracle keys {oracle}.")
        return data

    @field_validator("t_h", "y_Xv", "y_CL", "y_V", "F_act", "rpm_act", "vvm_act")
    @classmethod
    def _validate_finite(cls, v: float, info) -> float:
        return _assert_finite(v, info.field_name)

    @field_validator("S_feed_act")
    @classmethod
    def _validate_opt_sfeed(cls, v: Optional[float]) -> Optional[float]:
        return None if v is None else _assert_finite(v, "S_feed_act")

    @field_validator("sig_online")
    @classmethod
    def _validate_sig(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        vals = tuple(_assert_finite(x, "sig_online") for x in v)
        if any(x <= 0.0 for x in vals):
            raise ValueError("sig_online entries must be positive.")
        return (vals[0], vals[1], vals[2])

    def to_normalized_pat_request(self, c_star: float = C_STAR_DEFAULT_MG_L) -> Tuple[PatIngestRequest, Dict[str, str]]:
        """Convert units to canonical (g/L, mg/L, L, L/h) and validate physical ranges."""
        conversions: Dict[str, str] = {}

        xv_g_l = float(self.y_Xv)
        if self.y_Xv_unit == "kg/m3":
            conversions["y_Xv"] = "kg/m3->g/L (x1.0)"

        if self.y_CL_unit == "%DO":
            if self.y_CL > 250.0:
                raise ValueError(f"y_CL in %DO ({self.y_CL}) exceeds physical ceiling 250%.")
            cl_mg_l = float(self.y_CL) * float(c_star) / 100.0
            conversions["y_CL"] = f"%DO->mg/L (C_star={c_star})"
        else:
            cl_mg_l = float(self.y_CL)

        if self.y_V_unit == "mL":
            vol_l = float(self.y_V) / 1000.0
            conversions["y_V"] = "mL->L (/1000)"
        elif self.y_V_unit == "m3":
            vol_l = float(self.y_V) * 1000.0
            conversions["y_V"] = "m3->L (x1000)"
        else:
            vol_l = float(self.y_V)

        if self.F_unit == "mL/h":
            f_lh = float(self.F_act) / 1000.0
            conversions["F_act"] = "mL/h->L/h (/1000)"
        elif self.F_unit == "mL/min":
            f_lh = float(self.F_act) * 60.0 / 1000.0
            conversions["F_act"] = "mL/min->L/h (x0.06)"
        else:
            f_lh = float(self.F_act)

        req = PatIngestRequest(
            t_h=float(self.t_h),
            y_Xv=xv_g_l,
            y_CL=cl_mg_l,
            y_V=vol_l,
            F_act=f_lh,
            rpm_act=float(self.rpm_act),
            vvm_act=float(self.vvm_act),
            sig_online=self.sig_online,
            S_feed_act=self.S_feed_act,
        )
        return req, conversions


class CanonicalLimsAssayPayload(BaseModel):
    """Canonical unit-normalized at-line LIMS assay payload with truth-isolation guards."""

    model_config = ConfigDict(extra="forbid")

    t_sample_h: float = Field(ge=0.0, le=500.0)
    y_S: float = Field(ge=0.0)
    y_S_unit: Literal["g/L", "mmol/L"] = "g/L"
    y_A: float = Field(ge=0.0)
    y_A_unit: Literal["g/L", "mmol/L"] = "g/L"
    y_P: Optional[float] = Field(default=None, ge=0.0)
    y_P_unit: Literal["g/L", "mg/L"] = "g/L"
    sig_atline: Tuple[float, float, float] = (0.06, 0.05, 0.10)

    @model_validator(mode="before")
    @classmethod
    def _reject_oracle_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            oracle = contains_forbidden_oracle_keys(data)
            if oracle:
                raise ValueError(f"Truth isolation violation: forbidden oracle keys {oracle}.")
        return data

    @field_validator("t_sample_h", "y_S", "y_A")
    @classmethod
    def _validate_finite(cls, v: float, info) -> float:
        return _assert_finite(v, info.field_name)

    @field_validator("y_P")
    @classmethod
    def _validate_opt_yp(cls, v: Optional[float]) -> Optional[float]:
        return None if v is None else _assert_finite(v, "y_P")

    @field_validator("sig_atline")
    @classmethod
    def _validate_sig(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        vals = tuple(_assert_finite(x, "sig_atline") for x in v)
        if any(x <= 0.0 for x in vals):
            raise ValueError("sig_atline entries must be positive.")
        return (vals[0], vals[1], vals[2])

    def to_normalized_atline_request(self) -> Tuple[AtlineAssayRequest, Dict[str, str]]:
        """Convert substrate, acetate, and titer units into canonical g/L."""
        conversions: Dict[str, str] = {}

        if self.y_S_unit == "mmol/L":
            s_gl = float(self.y_S) * GLUCOSE_MW_G_MOL / 1000.0
            conversions["y_S"] = f"mmol/L->g/L (MW={GLUCOSE_MW_G_MOL})"
        else:
            s_gl = float(self.y_S)

        if self.y_A_unit == "mmol/L":
            a_gl = float(self.y_A) * ACETATE_MW_G_MOL / 1000.0
            conversions["y_A"] = f"mmol/L->g/L (MW={ACETATE_MW_G_MOL})"
        else:
            a_gl = float(self.y_A)

        if self.y_P is None:
            p_gl = None
        elif self.y_P_unit == "mg/L":
            p_gl = float(self.y_P) / 1000.0
            conversions["y_P"] = "mg/L->g/L (/1000)"
        else:
            p_gl = float(self.y_P)

        req = AtlineAssayRequest(
            t_sample_h=float(self.t_sample_h),
            y_S=s_gl,
            y_A=a_gl,
            y_P=p_gl,
            sig_atline=self.sig_atline,
        )
        return req, conversions


class AdapterIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["allotrope_asm", "opcua", "mqtt", "lims_batch_assay"]
    payload: Dict[str, Any]


class AdapterIngestResponse(BaseModel):
    session_id: str
    protocol: str
    status: Literal["ok", "quarantined"]
    payload_kind: Optional[Literal["pat", "lims"]] = None
    raw_payload_sha256: str
    alcoa_entry_sha256: str
    alcoa_prev_entry_sha256: str
    unit_conversions: Dict[str, str] = Field(default_factory=dict)
    quarantine_reason_codes: List[str] = Field(default_factory=list)
    normalized_payload: Optional[Dict[str, Any]] = None
    ekf_result: Optional[Dict[str, Any]] = None


class ControlActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_h: float = Field(ge=0.0, le=500.0)
    mu_target: float = Field(default=0.28, gt=0.0, le=0.8)
    S_set: float = Field(default=0.25, ge=0.0, le=10.0)
    A_limit: float = Field(default=1.8, gt=0.0, le=20.0)
    rpm: Optional[float] = Field(default=None, ge=50.0, le=1200.0)
    vvm: Optional[float] = Field(default=None, ge=0.01, le=5.0)
    F_max: Optional[float] = Field(default=None, gt=0.0, le=5000.0)
    disturbance_mode: Literal["none", "aeration_drop", "feed_step"] = "none"

    @field_validator("t_h", "mu_target", "S_set", "A_limit")
    @classmethod
    def _validate_finite_ctrl(cls, v: float, info) -> float:
        return _assert_finite(v, info.field_name)

    @field_validator("rpm", "vvm", "F_max")
    @classmethod
    def _validate_opt_ctrl(cls, v: Optional[float], info) -> Optional[float]:
        return None if v is None else _assert_finite(v, info.field_name)


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
    show_truth: bool = Field(default=False)
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
    guardrail_result: Optional[Dict[str, Any]] = None
    staged_draft: Optional[Dict[str, Any]] = None


class QaRedlineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(min_length=1, max_length=128)
    decision: Literal["APPROVED", "APPROVED_WITH_REDLINES", "REJECTED"]
    revised_overrides: Dict[str, Any] = Field(default_factory=dict)
    rationales: Dict[str, str] = Field(default_factory=dict)

    @field_validator("reviewer_id")
    @classmethod
    def _chk_human_reviewer(cls, v: str) -> str:
        return _validate_human_actor_id(v, "reviewer_id")


class TwoFactorESigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signer_id: str = Field(min_length=1, max_length=128)
    signer_name: str = Field(min_length=1, max_length=128)
    signer_role: Literal["QA_APPROVER", "QA_LEAD", "QP_RELEASE"]
    signing_meaning: str = Field(min_length=3, max_length=256)
    factor_1_type: Literal["PASSWORD", "SSO_OIDC", "SMARTCARD_PIV"] = "SSO_OIDC"
    factor_2_type: Literal["TOTP", "HARDWARE_TOKEN", "WEBAUTHN_FIDO2"] = "WEBAUTHN_FIDO2"
    factor_2_challenge_id: str = Field(min_length=6, max_length=128)

    @field_validator("signer_id")
    @classmethod
    def _chk_human_signer(cls, v: str) -> str:
        return _validate_human_actor_id(v, "signer_id")


class QaApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_record_id: Optional[str] = Field(default=None, min_length=4, max_length=64)
    staged_draft_id: Optional[str] = Field(default=None, min_length=4, max_length=64)
    capa_report: Optional[Dict[str, Any]] = None
    evidence_context: Optional[Dict[str, Any]] = None
    llm_execution_metadata: Optional[Dict[str, Any]] = None
    redlines: QaRedlineInput
    two_factor_esig: TwoFactorESigInput


class QaApprovalResponse(BaseModel):
    decision_record: Dict[str, Any]
    chain_valid: bool
    verification_errors: List[str] = Field(default_factory=list)
