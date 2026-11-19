"""Offline-safe OpenTelemetry and Prometheus telemetry hooks for the Bioreactor Digital Twin."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import uuid4

try:
    from opentelemetry import metrics as _otel_metrics  # type: ignore
    from opentelemetry import trace as _otel_trace  # type: ignore

    _HAS_OTEL_API = True
except ImportError:  # pragma: no cover - optional dependency
    _otel_metrics = None
    _otel_trace = None
    _HAS_OTEL_API = False

try:
    import prometheus_client as _prom_client  # type: ignore

    _HAS_PROMETHEUS_CLIENT = True
except ImportError:
    _prom_client = None
    _HAS_PROMETHEUS_CLIENT = False


DEFAULT_LATENCY_BUCKETS_MS: Tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0)


def _escape_prom_label(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: Tuple[Tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    joined = ",".join(f'{k}="{_escape_prom_label(v)}"' for k, v in labels)
    return f"{{{joined}}}"


def _norm_labels(labels: Optional[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


@dataclass
class _HistogramAccumulator:
    buckets: Tuple[float, ...]
    counts: List[int] = field(default_factory=list)
    total_sum: float = 0.0
    total_count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        val = float(value)
        if not math.isfinite(val):
            return
        self.total_sum += val
        self.total_count += 1
        for idx, bound in enumerate(self.buckets):
            if val <= bound:
                self.counts[idx] += 1
        self.counts[-1] += 1


class TwinTelemetryRecorder:
    """Thread-safe offline Prometheus & OpenTelemetry span/metric recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metric_help: Dict[str, str] = {}
        self._metric_types: Dict[str, str] = {}
        self._counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(dict)
        self._gauges: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[Tuple[Tuple[str, str], ...], _HistogramAccumulator]] = defaultdict(dict)
        self._spans: List[Dict[str, Any]] = []

        self._otel_tracer = _otel_trace.get_tracer("hybrid_fedbatch_twin") if _HAS_OTEL_API else None
        self._otel_meter = _otel_metrics.get_meter("hybrid_fedbatch_twin") if _HAS_OTEL_API else None
        self._otel_instruments: Dict[str, Any] = {}
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        defs = (
            ("twin_ekf_nis", "gauge", "Latest EKF Normalized Innovation Squared (NIS) statistic."),
            ("twin_ekf_spe", "gauge", "Latest EKF Squared Prediction Error (SPE) statistic."),
            ("twin_ekf_mahalanobis_dm", "gauge", "Latest GAM-86 Mahalanobis Domain-of-Validity distance D_M."),
            ("twin_ekf_steps_total", "counter", "Total EKF assimilation updates executed."),
            ("twin_ekf_residual_breach_total", "counter", "Total EKF NIS/SPE residual gate breaches."),
            ("twin_ekf_undecided_total", "counter", "Total EKF evaluations entering undecided gate state."),
            ("twin_control_fallback_total", "counter", "Total closed-loop invocations entering ML-free safe fallback."),
            ("twin_control_escalation_total", "counter", "Total closed-loop operator escalation events emitted."),
            ("twin_api_requests_total", "counter", "Total FastAPI HTTP requests processed."),
            ("twin_api_request_duration_ms", "histogram", "FastAPI HTTP request latency in milliseconds."),
            ("twin_agent_guardrail_rejections_total", "counter", "Total CAPA agent drafts rejected by deterministic guardrails."),
            ("twin_agent_investigations_total", "counter", "Total CAPA agent investigation runs by completion status."),
            ("twin_mcp_privilege_rejections_total", "counter", "Total MCP calls rejected by least-privilege boundary."),
            ("twin_medallion_quarantined_rows_total", "counter", "Total rows quarantined during Bronze-to-Silver ETL."),
        )
        for name, mtype, help_text in defs:
            self._metric_types[name] = mtype
            self._metric_help[name] = help_text

    def _emit_otel_metric(self, kind: str, name: str, value: float, labels: Dict[str, Any]) -> None:
        if self._otel_meter is None:
            return
        try:
            inst = self._otel_instruments.get(name)
            if inst is None:
                if kind == "counter":
                    inst = self._otel_meter.create_counter(name=name, description=self._metric_help.get(name, ""))
                elif kind == "histogram":
                    inst = self._otel_meter.create_histogram(name=name, description=self._metric_help.get(name, ""))
                else:
                    return
                self._otel_instruments[name] = inst
            attrs = {str(k): str(v) for k, v in labels.items()}
            if kind == "counter":
                inst.add(value, attributes=attrs)
            elif kind == "histogram":
                inst.record(value, attributes=attrs)
        except Exception:
            # Never break process execution if optional OTel meter backend is unconfigured
            return

    def inc_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, Any]] = None, help_text: str = "") -> None:
        key = _norm_labels(labels)
        with self._lock:
            self._metric_types.setdefault(name, "counter")
            if help_text:
                self._metric_help.setdefault(name, help_text)
            cur = self._counters[name].get(key, 0.0)
            self._counters[name][key] = cur + float(amount)
        self._emit_otel_metric("counter", name, float(amount), labels or {})

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None, help_text: str = "") -> None:
        val = float(value)
        if not math.isfinite(val):
            return
        key = _norm_labels(labels)
        with self._lock:
            self._metric_types.setdefault(name, "gauge")
            if help_text:
                self._metric_help.setdefault(name, help_text)
            self._gauges[name][key] = val

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, Any]] = None,
        buckets: Tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_MS,
        help_text: str = "",
    ) -> None:
        val = float(value)
        if not math.isfinite(val):
            return
        key = _norm_labels(labels)
        with self._lock:
            self._metric_types.setdefault(name, "histogram")
            if help_text:
                self._metric_help.setdefault(name, help_text)
            acc = self._histograms[name].get(key)
            if acc is None:
                acc = _HistogramAccumulator(buckets=buckets)
                self._histograms[name][key] = acc
            acc.observe(val)
        self._emit_otel_metric("histogram", name, val, labels or {})

    def record_span(
        self,
        name: str,
        start_time_ns: int,
        end_time_ns: int,
        attributes: Optional[Dict[str, Any]] = None,
        status: str = "OK",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        dur_ms = max(0.0, (int(end_time_ns) - int(start_time_ns)) / 1_000_000.0)
        span_rec = {
            "trace_id": trace_id or uuid4().hex,
            "span_id": span_id or uuid4().hex[:16],
            "name": str(name),
            "start_time_ns": int(start_time_ns),
            "end_time_ns": int(end_time_ns),
            "duration_ms": round(dur_ms, 4),
            "attributes": {str(k): v for k, v in (attributes or {}).items()},
            "status": str(status),
        }
        with self._lock:
            self._spans.append(span_rec)
        return span_rec

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        attrs = dict(attributes or {})
        t0 = time.time_ns()
        otel_span = None
        if self._otel_tracer is not None:
            try:
                otel_span = self._otel_tracer.start_span(name)
                for k, v in attrs.items():
                    if isinstance(v, (bool, str, int, float)):
                        otel_span.set_attribute(str(k), v)
            except Exception:
                otel_span = None
        status = "OK"
        span_holder: Dict[str, Any] = {"name": name, "attributes": attrs}
        try:
            yield span_holder
        except Exception as exc:
            status = "ERROR"
            attrs["error.type"] = type(exc).__name__
            attrs["error.message"] = str(exc)
            raise
        finally:
            t1 = time.time_ns()
            if otel_span is not None:
                try:
                    otel_span.end()
                except Exception:
                    pass
            recorded = self.record_span(
                name=name,
                start_time_ns=t0,
                end_time_ns=t1,
                attributes=attrs,
                status=status,
            )
            span_holder.update(recorded)

    def export_otel_spans(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(s, attributes=dict(s["attributes"])) for s in self._spans]

    def get_counter_value(self, name: str, labels: Optional[Dict[str, Any]] = None) -> float:
        key = _norm_labels(labels)
        with self._lock:
            if labels is not None:
                return float(self._counters.get(name, {}).get(key, 0.0))
            return float(sum(self._counters.get(name, {}).values()))

    def get_gauge_value(self, name: str, labels: Optional[Dict[str, Any]] = None) -> Optional[float]:
        key = _norm_labels(labels)
        with self._lock:
            val = self._gauges.get(name, {}).get(key)
            return None if val is None else float(val)

    def render_prometheus_text(self) -> str:
        lines: List[str] = []
        with self._lock:
            all_names = sorted(
                set(self._counters.keys()) | set(self._gauges.keys()) | set(self._histograms.keys())
            )
            for name in all_names:
                mtype = self._metric_types.get(name, "gauge")
                mhelp = self._metric_help.get(name, f"Twin telemetry metric {name}.")
                lines.append(f"# HELP {name} {mhelp}")
                lines.append(f"# TYPE {name} {mtype}")

                if name in self._counters:
                    for lbls, val in sorted(self._counters[name].items()):
                        lines.append(f"{name}{_format_labels(lbls)} {val}")
                if name in self._gauges:
                    for lbls, val in sorted(self._gauges[name].items()):
                        lines.append(f"{name}{_format_labels(lbls)} {val}")
                if name in self._histograms:
                    for lbls, acc in sorted(self._histograms[name].items()):
                        base_lbls = dict(lbls)
                        for idx, bound in enumerate(acc.buckets):
                            b_lbls = _norm_labels({**base_lbls, "le": str(bound)})
                            lines.append(f"{name}_bucket{_format_labels(b_lbls)} {acc.counts[idx]}")
                        inf_lbls = _norm_labels({**base_lbls, "le": "+Inf"})
                        lines.append(f"{name}_bucket{_format_labels(inf_lbls)} {acc.counts[-1]}")
                        lines.append(f"{name}_sum{_format_labels(lbls)} {acc.total_sum}")
                        lines.append(f"{name}_count{_format_labels(lbls)} {acc.total_count}")
        return ("\n".join(lines) + "\n") if lines else "# No twin telemetry recorded yet.\n"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "otel_api_available": _HAS_OTEL_API,
                "prometheus_client_available": _HAS_PROMETHEUS_CLIENT,
                "counters": {
                    name: {str(dict(lbls)): val for lbls, val in entries.items()}
                    for name, entries in self._counters.items()
                },
                "gauges": {
                    name: {str(dict(lbls)): val for lbls, val in entries.items()}
                    for name, entries in self._gauges.items()
                },
                "histograms": {
                    name: {
                        str(dict(lbls)): {"count": acc.total_count, "sum": acc.total_sum}
                        for lbls, acc in entries.items()
                    }
                    for name, entries in self._histograms.items()
                },
                "span_count": len(self._spans),
            }

    def clear(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._spans.clear()


_DEFAULT_RECORDER = TwinTelemetryRecorder()


def get_telemetry_recorder() -> TwinTelemetryRecorder:
    """Return the process-wide offline-safe telemetry recorder."""
    return _DEFAULT_RECORDER


def reset_telemetry() -> None:
    """Clear all recorded metrics and spans (primarily for isolated tests)."""
    _DEFAULT_RECORDER.clear()


def record_ekf_step(
    session_id: str,
    nis: float,
    spe: float,
    d_m: float,
    residual_breach: bool,
    undecided: bool,
    gate_reason: str,
) -> None:
    """Record EKF innovation (NIS/SPE), Mahalanobis distance, and gate breaches."""
    rec = get_telemetry_recorder()
    lbls = {"session_id": session_id}
    rec.set_gauge("twin_ekf_nis", float(nis), labels=lbls)
    rec.set_gauge("twin_ekf_spe", float(spe), labels=lbls)
    rec.set_gauge("twin_ekf_mahalanobis_dm", float(d_m), labels=lbls)
    rec.inc_counter("twin_ekf_steps_total", 1.0, labels={"gate_reason": gate_reason})
    if residual_breach:
        rec.inc_counter("twin_ekf_residual_breach_total", 1.0, labels={"gate_reason": gate_reason})
    if undecided:
        rec.inc_counter("twin_ekf_undecided_total", 1.0, labels={"gate_reason": gate_reason})
    t_now = time.time_ns()
    rec.record_span(
        name="ekf.update",
        start_time_ns=t_now,
        end_time_ns=t_now,
        attributes={
            "twin.session_id": session_id,
            "twin.nis": float(nis),
            "twin.spe": float(spe),
            "twin.d_m": float(d_m),
            "twin.residual_breach": bool(residual_breach),
            "twin.undecided": bool(undecided),
            "twin.gate_reason": str(gate_reason),
        },
        status="WARN" if undecided else "OK",
    )


def record_control_fallback(
    session_id: str,
    fallback_active: bool,
    control_mode: str,
    gate_reason: str,
    escalated: bool,
) -> None:
    """Record closed-loop safe fallback activations and operator escalations."""
    rec = get_telemetry_recorder()
    if fallback_active:
        rec.inc_counter(
            "twin_control_fallback_total",
            1.0,
            labels={"control_mode": control_mode, "gate_reason": gate_reason},
        )
    if escalated:
        rec.inc_counter(
            "twin_control_escalation_total",
            1.0,
            labels={"gate_reason": gate_reason},
        )
    t_now = time.time_ns()
    rec.record_span(
        name="controller.compute_action",
        start_time_ns=t_now,
        end_time_ns=t_now,
        attributes={
            "twin.session_id": session_id,
            "twin.fallback_active": bool(fallback_active),
            "twin.control_mode": str(control_mode),
            "twin.gate_reason": str(gate_reason),
            "twin.escalated": bool(escalated),
        },
        status="WARN" if fallback_active else "OK",
    )


def record_api_request(method: str, route: str, status_code: int, duration_ms: float) -> None:
    """Record FastAPI request counter, latency histogram, and HTTP OTel span."""
    rec = get_telemetry_recorder()
    lbls = {"method": method.upper(), "route": route, "status_code": str(int(status_code))}
    rec.inc_counter("twin_api_requests_total", 1.0, labels=lbls)
    rec.observe_histogram("twin_api_request_duration_ms", float(duration_ms), labels={"method": method.upper(), "route": route})
    end_ns = time.time_ns()
    start_ns = max(0, end_ns - int(max(0.0, float(duration_ms)) * 1_000_000.0))
    rec.record_span(
        name="http.server.request",
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes={
            "http.method": method.upper(),
            "http.route": route,
            "http.status_code": int(status_code),
            "http.duration_ms": float(duration_ms),
        },
        status="ERROR" if int(status_code) >= 500 else "OK",
    )


def record_agent_guardrail_rejection(session_id: str, reason_codes: Sequence[str]) -> None:
    """Record CAPA draft guardrail rejections by deterministic reason code."""
    rec = get_telemetry_recorder()
    codes = list(reason_codes) or ["UNSPECIFIED_GUARDRAIL_VIOLATION"]
    for code in codes:
        rec.inc_counter("twin_agent_guardrail_rejections_total", 1.0, labels={"reason_code": str(code)})
    t_now = time.time_ns()
    rec.record_span(
        name="agent.guardrail_reject",
        start_time_ns=t_now,
        end_time_ns=t_now,
        attributes={"twin.session_id": session_id, "agent.reason_codes": ",".join(str(c) for c in codes)},
        status="ERROR",
    )


def record_agent_investigation(session_id: str, status: str, duration_ms: float) -> None:
    """Record CAPA agent investigation execution outcome."""
    rec = get_telemetry_recorder()
    rec.inc_counter("twin_agent_investigations_total", 1.0, labels={"status": str(status)})
    end_ns = time.time_ns()
    start_ns = max(0, end_ns - int(max(0.0, float(duration_ms)) * 1_000_000.0))
    rec.record_span(
        name="agent.deviation_investigation",
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        attributes={"twin.session_id": session_id, "agent.status": str(status), "agent.duration_ms": float(duration_ms)},
        status="ERROR" if status == "GUARDRAIL_REJECTED" else "OK",
    )


def record_mcp_privilege_rejection(operation: str, target_scope: str) -> None:
    """Record blocked MCP mutation or truth-leakage attempts."""
    rec = get_telemetry_recorder()
    rec.inc_counter(
        "twin_mcp_privilege_rejections_total",
        1.0,
        labels={"operation": str(operation), "target_scope": str(target_scope)},
    )
    t_now = time.time_ns()
    rec.record_span(
        name="mcp.privilege_reject",
        start_time_ns=t_now,
        end_time_ns=t_now,
        attributes={"mcp.operation": str(operation), "mcp.target_scope": str(target_scope)},
        status="ERROR",
    )


def record_medallion_quarantine(batch_id: str, reason_codes: Sequence[str]) -> None:
    """Record quarantined rows during Bronze -> Silver data quality gating."""
    rec = get_telemetry_recorder()
    codes = list(reason_codes) or ["UNSPECIFIED_DQ_FAILURE"]
    for code in codes:
        rec.inc_counter("twin_medallion_quarantined_rows_total", 1.0, labels={"reason_code": str(code)})
    t_now = time.time_ns()
    rec.record_span(
        name="mlops.medallion_quarantine",
        start_time_ns=t_now,
        end_time_ns=t_now,
        attributes={"mlops.batch_id": str(batch_id), "mlops.reason_codes": ",".join(str(c) for c in codes)},
        status="WARN",
    )
