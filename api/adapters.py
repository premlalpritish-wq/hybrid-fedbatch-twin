"""OT and LIMS ingestion adapters normalizing payloads into the ALCOA+ SHA-256 ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import ValidationError

from hybrid_fedbatch_twin.estimator import BioreactorDigitalTwin

from .schemas import (
    C_STAR_DEFAULT_MG_L,
    CanonicalLimsAssayPayload,
    CanonicalPatPayload,
    contains_forbidden_oracle_keys,
    sanitize_truth_payload,
)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class AdapterResult:
    """Structured normalization and quarantine result from an OT/LIMS adapter."""

    protocol: str
    status: Literal["ok", "quarantined"]
    payload_kind: Optional[Literal["pat", "lims"]]
    raw_payload_sha256: str
    normalized: Optional[Dict[str, Any]] = None
    unit_conversions: Dict[str, str] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    quarantine_reason_codes: List[str] = field(default_factory=list)
    detail: str = ""


class AllotropeASMAdapter:
    """Adapter for Allotrope Simple Model (ASM) JSON telemetry and assay documents."""

    protocol = "allotrope_asm"

    def parse(self, payload: Dict[str, Any], c_star: float = C_STAR_DEFAULT_MG_L) -> AdapterResult:
        raw_sha = _canonical_sha256(payload)
        oracle = contains_forbidden_oracle_keys(payload)
        if oracle:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind=None,
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["FORBIDDEN_ORACLE_FIELD"],
                detail=f"Truth isolation violation: {oracle}",
            )

        manifest = payload.get("$asm.manifest")
        if not isinstance(manifest, str) or "allotrope" not in manifest.lower():
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind=None,
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["ASM_MISSING_MANIFEST"],
                detail="Missing or invalid '$asm.manifest' header.",
            )

        agg = payload.get("measurement aggregate document")
        if not isinstance(agg, dict):
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind=None,
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["ASM_INVALID_STRUCTURE"],
                detail="Missing 'measurement aggregate document' object.",
            )

        docs_raw = agg.get("measurement document")
        if isinstance(docs_raw, list) and len(docs_raw) > 0:
            doc = docs_raw[0]
        elif isinstance(docs_raw, dict):
            doc = docs_raw
        else:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind=None,
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["ASM_INVALID_STRUCTURE"],
                detail="Missing 'measurement document' entry.",
            )

        if not isinstance(doc, dict):
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind=None,
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["ASM_INVALID_STRUCTURE"],
                detail="Measurement document must be a JSON object.",
            )

        def _extract_vu(key_aliases: Tuple[str, ...]) -> Optional[Tuple[Any, Optional[str]]]:
            for alias in key_aliases:
                if alias in doc:
                    item = doc[alias]
                    if isinstance(item, dict):
                        return item.get("value"), item.get("unit")
                    return item, None
            return None

        source_meta = {
            "asm_manifest": manifest,
            "measurement_identifier": doc.get("measurement identifier"),
            "sample_identifier": doc.get("sample identifier"),
        }

        is_lims = any(k in doc for k in ("glucose", "y_S", "acetate", "y_A")) and not any(
            k in doc for k in ("viable cell density", "y_Xv")
        )
        t_entry = _extract_vu(("measurement time", "t_h", "t_sample_h", "batch age"))
        t_val = t_entry[0] if t_entry else None

        try:
            if is_lims:
                glc = _extract_vu(("glucose", "y_S"))
                ace = _extract_vu(("acetate", "y_A"))
                pro = _extract_vu(("product titer", "product", "y_P"))
                if t_val is None or glc is None or ace is None:
                    return AdapterResult(
                        protocol=self.protocol,
                        status="quarantined",
                        payload_kind="lims",
                        raw_payload_sha256=raw_sha,
                        source_metadata=source_meta,
                        quarantine_reason_codes=["ASM_MISSING_FIELDS"],
                        detail="ASM LIMS document missing measurement time, glucose, or acetate.",
                    )
                canon = CanonicalLimsAssayPayload(
                    t_sample_h=t_val,
                    y_S=glc[0],
                    y_S_unit=glc[1] or "g/L",
                    y_A=ace[0],
                    y_A_unit=ace[1] or "g/L",
                    y_P=None if pro is None else pro[0],
                    y_P_unit=(pro[1] if pro and pro[1] else "g/L"),
                )
                req, conv = canon.to_normalized_atline_request()
                return AdapterResult(
                    protocol=self.protocol,
                    status="ok",
                    payload_kind="lims",
                    raw_payload_sha256=raw_sha,
                    normalized=req.model_dump(),
                    unit_conversions=conv,
                    source_metadata=source_meta,
                )

            xv = _extract_vu(("viable cell density", "y_Xv"))
            cl = _extract_vu(("dissolved oxygen", "y_CL"))
            vol = _extract_vu(("working volume", "y_V"))
            feed = _extract_vu(("feed flow rate", "F_act"))
            rpm = _extract_vu(("agitation speed", "rpm_act"))
            vvm = _extract_vu(("aeration rate", "vvm_act"))
            s_feed = _extract_vu(("feed glucose concentration", "S_feed_act"))
            if None in (t_val, xv, cl, vol, feed, rpm, vvm):
                return AdapterResult(
                    protocol=self.protocol,
                    status="quarantined",
                    payload_kind="pat",
                    raw_payload_sha256=raw_sha,
                    source_metadata=source_meta,
                    quarantine_reason_codes=["ASM_MISSING_FIELDS"],
                    detail="ASM PAT document missing one or more required PAT/actuator fields.",
                )
            canon_pat = CanonicalPatPayload(
                t_h=t_val,
                y_Xv=xv[0],
                y_Xv_unit=xv[1] or "g/L",
                y_CL=cl[0],
                y_CL_unit=cl[1] or "mg/L",
                y_V=vol[0],
                y_V_unit=vol[1] or "L",
                F_act=feed[0],
                F_unit=feed[1] or "L/h",
                rpm_act=rpm[0],
                vvm_act=vvm[0],
                S_feed_act=None if s_feed is None else s_feed[0],
            )
            req_pat, conv_pat = canon_pat.to_normalized_pat_request(c_star=c_star)
            return AdapterResult(
                protocol=self.protocol,
                status="ok",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                normalized=req_pat.model_dump(),
                unit_conversions=conv_pat,
                source_metadata=source_meta,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="lims" if is_lims else "pat",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["SCHEMA_VALIDATION_ERROR"],
                detail=str(exc),
            )


class OpcUaSnapshotAdapter:
    """Adapter for OPC-UA server node snapshot payloads with per-node StatusCode quality gating."""

    protocol = "opcua"

    _NODE_ALIASES: Dict[str, Tuple[str, ...]] = {
        "t_h": ("t_h", "BatchAge_h", "Bioreactor.BatchAge_h"),
        "y_Xv": ("y_Xv", "X_v", "Bioreactor.PAT.X_v"),
        "y_CL": ("y_CL", "C_L", "DO", "Bioreactor.PAT.DO"),
        "y_V": ("y_V", "V", "Volume", "Bioreactor.PAT.Volume"),
        "F_act": ("F_act", "F", "FeedFlow", "Bioreactor.Actuators.FeedFlow"),
        "rpm_act": ("rpm_act", "rpm", "Agitation", "Bioreactor.Actuators.Agitation"),
        "vvm_act": ("vvm_act", "vvm", "Aeration", "Bioreactor.Actuators.Aeration"),
    }

    def parse(self, payload: Dict[str, Any], c_star: float = C_STAR_DEFAULT_MG_L) -> AdapterResult:
        raw_sha = _canonical_sha256(payload)
        oracle = contains_forbidden_oracle_keys(payload)
        if oracle:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["FORBIDDEN_ORACLE_FIELD"],
                detail=f"Truth isolation violation: {oracle}",
            )

        ns_uri = payload.get("namespace_uri")
        srv_ts = payload.get("server_timestamp_utc")
        nodes = payload.get("nodes")
        if not isinstance(ns_uri, str) or not ns_uri.strip() or not isinstance(srv_ts, str) or not srv_ts.strip():
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["OPCUA_MISSING_METADATA"],
                detail="OPC-UA snapshot requires non-empty 'namespace_uri' and 'server_timestamp_utc'.",
            )
        if not isinstance(nodes, dict) or not nodes:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["OPCUA_MISSING_NODES"],
                detail="OPC-UA snapshot requires a non-empty 'nodes' mapping.",
            )

        source_meta = {"namespace_uri": ns_uri, "server_timestamp_utc": srv_ts}

        bad_nodes: List[str] = []
        for node_id, node_val in nodes.items():
            if not isinstance(node_val, dict):
                bad_nodes.append(f"{node_id}:not_object")
                continue
            status = str(node_val.get("status_code", "")).strip()
            if status not in ("Good", "0x00000000"):
                bad_nodes.append(f"{node_id}:{status or 'MISSING'}")

        if bad_nodes:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["OPCUA_BAD_QUALITY"],
                detail=f"OPC-UA nodes failed Good StatusCode gate: {bad_nodes}",
            )

        def _find_node(canonical_key: str) -> Optional[Dict[str, Any]]:
            for alias in self._NODE_ALIASES[canonical_key]:
                if alias in nodes:
                    return nodes[alias]
            return None

        t_h = payload.get("batch_age_h")
        if t_h is None and _find_node("t_h") is not None:
            t_h = _find_node("t_h").get("value")  # type: ignore[union-attr]

        xv = _find_node("y_Xv")
        cl = _find_node("y_CL")
        vol = _find_node("y_V")
        feed = _find_node("F_act")
        rpm = _find_node("rpm_act")
        vvm = _find_node("vvm_act")
        if None in (t_h, xv, cl, vol, feed, rpm, vvm):
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["OPCUA_MISSING_REQUIRED_TAGS"],
                detail="Missing one or more required OPC-UA PAT/actuator nodes.",
            )

        try:
            canon = CanonicalPatPayload(
                t_h=t_h,  # type: ignore[arg-type]
                y_Xv=xv["value"],  # type: ignore[index]
                y_Xv_unit=xv.get("engineering_unit") or "g/L",  # type: ignore[union-attr]
                y_CL=cl["value"],  # type: ignore[index]
                y_CL_unit=cl.get("engineering_unit") or "mg/L",  # type: ignore[union-attr]
                y_V=vol["value"],  # type: ignore[index]
                y_V_unit=vol.get("engineering_unit") or "L",  # type: ignore[union-attr]
                F_act=feed["value"],  # type: ignore[index]
                F_unit=feed.get("engineering_unit") or "L/h",  # type: ignore[union-attr]
                rpm_act=rpm["value"],  # type: ignore[index]
                vvm_act=vvm["value"],  # type: ignore[index]
            )
            req, conv = canon.to_normalized_pat_request(c_star=c_star)
            return AdapterResult(
                protocol=self.protocol,
                status="ok",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                normalized=req.model_dump(),
                unit_conversions=conv,
                source_metadata=source_meta,
            )
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["SCHEMA_VALIDATION_ERROR"],
                detail=str(exc),
            )


class MqttTelemetryAdapter:
    """Adapter for MQTT Sparkplug/JSON bioreactor PAT telemetry envelopes."""

    protocol = "mqtt"

    def parse(self, payload: Dict[str, Any], c_star: float = C_STAR_DEFAULT_MG_L) -> AdapterResult:
        raw_sha = _canonical_sha256(payload)
        oracle = contains_forbidden_oracle_keys(payload)
        if oracle:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["FORBIDDEN_ORACLE_FIELD"],
                detail=f"Truth isolation violation: {oracle}",
            )

        topic = payload.get("topic")
        qos = payload.get("qos")
        ts = payload.get("timestamp_utc")
        metrics = payload.get("metrics")
        units = payload.get("units") or {}

        if (
            not isinstance(topic, str)
            or len([seg for seg in topic.split("/") if seg]) < 3
            or "#" in topic
            or "+" in topic
            or isinstance(qos, bool)
            or qos not in (0, 1, 2)
            or not isinstance(ts, str)
            or not ts.strip()
        ):
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["MQTT_INVALID_ENVELOPE"],
                detail="MQTT envelope requires valid topic (>=3 segments, no wildcards), qos in {0,1,2}, and timestamp_utc.",
            )

        if not isinstance(metrics, dict) or not isinstance(units, dict):
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                source_metadata={"topic": topic, "qos": qos, "timestamp_utc": ts},
                quarantine_reason_codes=["MQTT_INVALID_METRICS"],
                detail="MQTT payload requires 'metrics' object and optional 'units' object.",
            )

        source_meta = {"topic": topic, "qos": qos, "timestamp_utc": ts}
        required_keys = ("t_h", "y_Xv", "y_CL", "y_V", "F_act", "rpm_act", "vvm_act")
        if any(k not in metrics for k in required_keys):
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["MQTT_MISSING_METRICS"],
                detail=f"MQTT metrics missing one or more required keys: {required_keys}",
            )

        try:
            canon = CanonicalPatPayload(
                t_h=metrics["t_h"],
                y_Xv=metrics["y_Xv"],
                y_Xv_unit=units.get("y_Xv", "g/L"),
                y_CL=metrics["y_CL"],
                y_CL_unit=units.get("y_CL", "mg/L"),
                y_V=metrics["y_V"],
                y_V_unit=units.get("y_V", "L"),
                F_act=metrics["F_act"],
                F_unit=units.get("F_act", "L/h"),
                rpm_act=metrics["rpm_act"],
                vvm_act=metrics["vvm_act"],
                S_feed_act=metrics.get("S_feed_act"),
            )
            req, conv = canon.to_normalized_pat_request(c_star=c_star)
            return AdapterResult(
                protocol=self.protocol,
                status="ok",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                normalized=req.model_dump(),
                unit_conversions=conv,
                source_metadata=source_meta,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="pat",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["SCHEMA_VALIDATION_ERROR"],
                detail=str(exc),
            )


class LimsBatchAssayAdapter:
    """Adapter for at-line LIMS batch assay certificates with per-analyte QC flagging."""

    protocol = "lims_batch_assay"

    _ANALYTE_MAP = {
        "glucose": "y_S",
        "glc": "y_S",
        "s": "y_S",
        "y_s": "y_S",
        "acetate": "y_A",
        "ace": "y_A",
        "a": "y_A",
        "y_a": "y_A",
        "product": "y_P",
        "titer": "y_P",
        "p": "y_P",
        "y_p": "y_P",
    }

    def parse(self, payload: Dict[str, Any], c_star: float = C_STAR_DEFAULT_MG_L) -> AdapterResult:
        del c_star
        raw_sha = _canonical_sha256(payload)
        oracle = contains_forbidden_oracle_keys(payload)
        if oracle:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="lims",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["FORBIDDEN_ORACLE_FIELD"],
                detail=f"Truth isolation violation: {oracle}",
            )

        required_meta = ("lims_system", "sample_id", "batch_id", "analyst_id", "sampled_utc", "batch_age_h")
        if any(not payload.get(k) and payload.get(k) != 0 for k in required_meta):
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="lims",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["LIMS_MISSING_METADATA"],
                detail=f"LIMS payload requires {required_meta}.",
            )

        analytes = payload.get("analytes")
        if not isinstance(analytes, list) or not analytes:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="lims",
                raw_payload_sha256=raw_sha,
                quarantine_reason_codes=["LIMS_MISSING_ANALYTES"],
                detail="LIMS payload requires non-empty 'analytes' list.",
            )

        source_meta = {
            "lims_system": payload["lims_system"],
            "sample_id": payload["sample_id"],
            "batch_id": payload["batch_id"],
            "analyst_id": payload["analyst_id"],
            "sampled_utc": payload["sampled_utc"],
        }

        qc_failures: List[str] = []
        extracted: Dict[str, Tuple[Any, str]] = {}
        for item in analytes:
            if not isinstance(item, dict):
                qc_failures.append("malformed_analyte_row")
                continue
            name = str(item.get("name", "")).strip()
            qc_flag = str(item.get("qc_flag", "")).strip().upper()
            if qc_flag != "PASS":
                qc_failures.append(f"{name or 'unknown'}:{qc_flag or 'MISSING'}")
            canon_key = self._ANALYTE_MAP.get(name.lower())
            if canon_key is not None:
                extracted[canon_key] = (item.get("value"), str(item.get("unit") or "g/L"))

        if qc_failures:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="lims",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["LIMS_QC_FLAG_FAILED"],
                detail=f"LIMS analytes failed QC gate: {qc_failures}",
            )

        if "y_S" not in extracted or "y_A" not in extracted:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="lims",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["LIMS_MISSING_REQUIRED_ANALYTE"],
                detail="LIMS batch assay requires both glucose and acetate analytes.",
            )

        try:
            canon = CanonicalLimsAssayPayload(
                t_sample_h=payload["batch_age_h"],
                y_S=extracted["y_S"][0],
                y_S_unit=extracted["y_S"][1],  # type: ignore[arg-type]
                y_A=extracted["y_A"][0],
                y_A_unit=extracted["y_A"][1],  # type: ignore[arg-type]
                y_P=None if "y_P" not in extracted else extracted["y_P"][0],
                y_P_unit="g/L" if "y_P" not in extracted else extracted["y_P"][1],  # type: ignore[arg-type]
            )
            req, conv = canon.to_normalized_atline_request()
            return AdapterResult(
                protocol=self.protocol,
                status="ok",
                payload_kind="lims",
                raw_payload_sha256=raw_sha,
                normalized=req.model_dump(),
                unit_conversions=conv,
                source_metadata=source_meta,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            return AdapterResult(
                protocol=self.protocol,
                status="quarantined",
                payload_kind="lims",
                raw_payload_sha256=raw_sha,
                source_metadata=source_meta,
                quarantine_reason_codes=["SCHEMA_VALIDATION_ERROR"],
                detail=str(exc),
            )


_ADAPTER_REGISTRY = {
    "allotrope_asm": AllotropeASMAdapter(),
    "opcua": OpcUaSnapshotAdapter(),
    "mqtt": MqttTelemetryAdapter(),
    "lims_batch_assay": LimsBatchAssayAdapter(),
}


def normalize_adapter_payload(
    protocol: str,
    raw_payload: Dict[str, Any],
    c_star: float = C_STAR_DEFAULT_MG_L,
) -> AdapterResult:
    """Parse and validate a raw OT/LIMS payload into a canonical PAT or LIMS model."""
    adapter = _ADAPTER_REGISTRY.get(protocol)
    if adapter is None:
        return AdapterResult(
            protocol=protocol,
            status="quarantined",
            payload_kind=None,
            raw_payload_sha256=_canonical_sha256(raw_payload),
            quarantine_reason_codes=["UNSUPPORTED_PROTOCOL"],
            detail=f"Unsupported OT/LIMS adapter protocol: {protocol!r}.",
        )
    return adapter.parse(raw_payload, c_star=c_star)


def ingest_adapter_into_twin(
    twin: BioreactorDigitalTwin,
    protocol: str,
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize an OT/LIMS payload and append EKF + provenance or quarantine into the ALCOA+ hash chain."""
    res = normalize_adapter_payload(
        protocol=protocol,
        raw_payload=raw_payload,
        c_star=float(twin.params.C_star),
    )

    if res.status == "quarantined" or res.normalized is None:
        twin._append_audit(
            "ADAPTER_QUARANTINE",
            float(twin.t_now),
            {
                "protocol": protocol,
                "payload_kind": res.payload_kind,
                "raw_payload_sha256": res.raw_payload_sha256,
                "quarantine_reason_codes": list(res.quarantine_reason_codes),
                "source_metadata": res.source_metadata,
                "detail": res.detail,
            },
        )
        head = twin.audit_log[-1]
        return {
            "protocol": protocol,
            "status": "quarantined",
            "payload_kind": res.payload_kind,
            "raw_payload_sha256": res.raw_payload_sha256,
            "alcoa_entry_sha256": str(head["entry_sha256"]),
            "alcoa_prev_entry_sha256": str(head["prev_entry_sha256"]),
            "unit_conversions": res.unit_conversions,
            "quarantine_reason_codes": list(res.quarantine_reason_codes),
            "normalized_payload": None,
            "ekf_result": None,
            "detail": res.detail,
        }

    if res.payload_kind == "pat":
        norm = res.normalized
        if float(norm["y_V"]) > 1.05 * float(twin.params.V_max):
            twin._append_audit(
                "ADAPTER_QUARANTINE",
                float(twin.t_now),
                {
                    "protocol": protocol,
                    "payload_kind": "pat",
                    "raw_payload_sha256": res.raw_payload_sha256,
                    "quarantine_reason_codes": ["VESSEL_VOLUME_EXCEEDS_MAX"],
                    "source_metadata": res.source_metadata,
                    "detail": f"y_V={norm['y_V']} exceeds 1.05 * V_max={1.05 * float(twin.params.V_max):.2f} L.",
                },
            )
            head = twin.audit_log[-1]
            return {
                "protocol": protocol,
                "status": "quarantined",
                "payload_kind": "pat",
                "raw_payload_sha256": res.raw_payload_sha256,
                "alcoa_entry_sha256": str(head["entry_sha256"]),
                "alcoa_prev_entry_sha256": str(head["prev_entry_sha256"]),
                "unit_conversions": res.unit_conversions,
                "quarantine_reason_codes": ["VESSEL_VOLUME_EXCEEDS_MAX"],
                "normalized_payload": None,
                "ekf_result": None,
                "detail": f"y_V={norm['y_V']} exceeds vessel V_max={twin.params.V_max}.",
            }
        try:
            ekf_raw = twin.ingest_online_pat(**norm)
        except ValueError as exc:
            twin._append_audit(
                "ADAPTER_QUARANTINE",
                float(twin.t_now),
                {
                    "protocol": protocol,
                    "payload_kind": "pat",
                    "raw_payload_sha256": res.raw_payload_sha256,
                    "quarantine_reason_codes": ["NON_MONOTONIC_TIMESTAMP"],
                    "source_metadata": res.source_metadata,
                    "detail": str(exc),
                },
            )
            head = twin.audit_log[-1]
            return {
                "protocol": protocol,
                "status": "quarantined",
                "payload_kind": "pat",
                "raw_payload_sha256": res.raw_payload_sha256,
                "alcoa_entry_sha256": str(head["entry_sha256"]),
                "alcoa_prev_entry_sha256": str(head["prev_entry_sha256"]),
                "unit_conversions": res.unit_conversions,
                "quarantine_reason_codes": ["NON_MONOTONIC_TIMESTAMP"],
                "normalized_payload": None,
                "ekf_result": None,
                "detail": str(exc),
            }
    else:
        norm = res.normalized
        try:
            ekf_raw = twin.ingest_atline_assay(**norm)
        except ValueError as exc:
            twin._append_audit(
                "ADAPTER_QUARANTINE",
                float(twin.t_now),
                {
                    "protocol": protocol,
                    "payload_kind": "lims",
                    "raw_payload_sha256": res.raw_payload_sha256,
                    "quarantine_reason_codes": ["LIMS_SAMPLE_TIME_OUT_OF_RANGE"],
                    "source_metadata": res.source_metadata,
                    "detail": str(exc),
                },
            )
            head = twin.audit_log[-1]
            return {
                "protocol": protocol,
                "status": "quarantined",
                "payload_kind": "lims",
                "raw_payload_sha256": res.raw_payload_sha256,
                "alcoa_entry_sha256": str(head["entry_sha256"]),
                "alcoa_prev_entry_sha256": str(head["prev_entry_sha256"]),
                "unit_conversions": res.unit_conversions,
                "quarantine_reason_codes": ["LIMS_SAMPLE_TIME_OUT_OF_RANGE"],
                "normalized_payload": None,
                "ekf_result": None,
                "detail": str(exc),
            }

    twin._append_audit(
        "ADAPTER_PROVENANCE",
        float(twin.t_now),
        {
            "protocol": protocol,
            "payload_kind": res.payload_kind,
            "raw_payload_sha256": res.raw_payload_sha256,
            "unit_conversions": res.unit_conversions,
            "source_metadata": res.source_metadata,
        },
    )
    head = twin.audit_log[-1]

    ekf_serializable: Dict[str, Any] = {}
    for k, v in ekf_raw.items():
        if hasattr(v, "tolist"):
            ekf_serializable[k] = [float(x) for x in v.tolist()]
        elif isinstance(v, bool):
            ekf_serializable[k] = v
        elif isinstance(v, (int, float)):
            ekf_serializable[k] = float(v)
        else:
            ekf_serializable[k] = v

    return {
        "protocol": protocol,
        "status": "ok",
        "payload_kind": res.payload_kind,
        "raw_payload_sha256": res.raw_payload_sha256,
        "alcoa_entry_sha256": str(head["entry_sha256"]),
        "alcoa_prev_entry_sha256": str(head["prev_entry_sha256"]),
        "unit_conversions": res.unit_conversions,
        "quarantine_reason_codes": [],
        "normalized_payload": res.normalized,
        "ekf_result": sanitize_truth_payload(ekf_serializable, show_truth=False),
    }
