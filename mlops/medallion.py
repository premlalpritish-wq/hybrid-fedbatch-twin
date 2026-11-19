"""Bronze -> Silver -> Gold Medallion ETL pipeline with declarative DQ gates and SHA-256 lineage."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .telemetry import record_medallion_quarantine

DEFAULT_FORBIDDEN_ORACLE_FIELDS: Tuple[str, ...] = (
    "u_plant",
    "plant_inputs",
    "true_states",
    "p_true",
    "mu_true",
    "kd_true",
    "kla_true",
    "qAnet_true",
)


def _canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ColumnBound:
    """Declarative numeric range and nullability policy for a single column."""

    min_val: float
    max_val: float
    allow_null: bool = False


@dataclass
class DataQualityPolicy:
    """Declarative Bronze -> Silver quality gate configuration."""

    required_columns: Tuple[str, ...] = ("batch_id", "t_h", "X_v", "C_L", "V")
    numeric_bounds: Dict[str, ColumnBound] = field(
        default_factory=lambda: {
            "t_h": ColumnBound(0.0, 500.0, allow_null=False),
            "X_v": ColumnBound(0.0, 200.0, allow_null=False),
            "X_d": ColumnBound(0.0, 100.0, allow_null=True),
            "S": ColumnBound(0.0, 100.0, allow_null=True),
            "C_L": ColumnBound(0.0, 25.0, allow_null=False),
            "A": ColumnBound(0.0, 50.0, allow_null=True),
            "P": ColumnBound(0.0, 100.0, allow_null=True),
            "V": ColumnBound(1e-6, 50000.0, allow_null=False),
            "F": ColumnBound(0.0, 5000.0, allow_null=True),
            "rpm": ColumnBound(50.0, 1200.0, allow_null=True),
            "vvm": ColumnBound(0.01, 5.0, allow_null=True),
        }
    )
    enforce_monotonic_timestamp: bool = True
    enforce_truth_isolation: bool = True
    min_completeness_ratio: float = 0.80
    forbidden_oracle_fields: Tuple[str, ...] = DEFAULT_FORBIDDEN_ORACLE_FIELDS


def _find_oracle_keys(record: Any, forbidden: Tuple[str, ...], prefix: str = "") -> List[str]:
    """Recursively scan dicts/lists for forbidden oracle truth keys."""
    found: List[str] = []
    forbidden_set = set(forbidden)
    if isinstance(record, dict):
        for k, v in record.items():
            ks = str(k)
            path = f"{prefix}.{ks}" if prefix else ks
            if ks in forbidden_set or ks.endswith("_true"):
                found.append(path)
            found.extend(_find_oracle_keys(v, forbidden, path))
    elif isinstance(record, list):
        for idx, item in enumerate(record):
            found.extend(_find_oracle_keys(item, forbidden, f"{prefix}[{idx}]"))
    return sorted(set(found))


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line_s = line.strip()
        if line_s:
            rows.append(json.loads(line_s))
    return rows


def ingest_bronze(
    records: Iterable[Any],
    output_dir: str | Path,
    run_id: str,
    source_uri: str = "in_memory",
) -> Dict[str, Any]:
    """Persist immutable raw telemetry/assay records into the Bronze layer with row SHA-256 digests."""
    base_dir = Path(output_dir)
    bronze_path = base_dir / "bronze" / f"{run_id}.jsonl"
    ingested_utc = datetime.now(timezone.utc).isoformat()

    bronze_rows: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records):
        row_obj: Dict[str, Any] = {
            "row_index": idx,
            "run_id": run_id,
            "source_uri": source_uri,
            "ingested_utc": ingested_utc,
            "raw_payload": raw,
        }
        row_obj["raw_row_sha256"] = _canonical_sha256(
            {"row_index": idx, "run_id": run_id, "raw_payload": raw}
        )
        bronze_rows.append(row_obj)

    _write_jsonl(bronze_path, bronze_rows)
    bronze_sha = _file_sha256(bronze_path)
    return {
        "run_id": run_id,
        "bronze_path": str(bronze_path),
        "bronze_sha256": bronze_sha,
        "row_count": len(bronze_rows),
        "source_uri": source_uri,
    }


def promote_bronze_to_silver(
    bronze_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    policy: Optional[DataQualityPolicy] = None,
) -> Dict[str, Any]:
    """Apply declarative schema, bounds, completeness, truth-isolation, and monotonicity gates."""
    dq_policy = policy or DataQualityPolicy()
    b_path = Path(bronze_path)
    base_dir = Path(output_dir)
    silver_path = base_dir / "silver" / f"{run_id}.jsonl"
    quarantine_path = base_dir / "quarantine" / f"{run_id}.jsonl"

    bronze_rows = _read_jsonl(b_path)
    silver_rows: List[Dict[str, Any]] = []
    quarantine_rows: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    last_t_by_batch: Dict[str, float] = {}

    quarantined_utc = datetime.now(timezone.utc).isoformat()
    silver_utc = quarantined_utc

    checked_fields = sorted(set(dq_policy.required_columns) | set(dq_policy.numeric_bounds.keys()))

    for b_row in bronze_rows:
        row_index = int(b_row.get("row_index", -1))
        bronze_row_sha = str(b_row.get("raw_row_sha256", ""))
        raw = b_row.get("raw_payload")

        reason_codes: List[str] = []
        failed_rules: List[str] = []

        if not isinstance(raw, dict):
            reason_codes.append("MALFORMED_RECORD")
            failed_rules.append("Record payload must be a JSON object/dict.")
        else:
            if dq_policy.enforce_truth_isolation:
                oracle_keys = _find_oracle_keys(raw, dq_policy.forbidden_oracle_fields)
                if oracle_keys:
                    reason_codes.append("FORBIDDEN_ORACLE_FIELD")
                    failed_rules.append(f"Oracle truth keys prohibited in Medallion ETL: {oracle_keys}")

            missing_req = [
                col for col in dq_policy.required_columns if col not in raw or raw[col] is None or raw[col] == ""
            ]
            if missing_req:
                reason_codes.append("MISSING_REQUIRED_COLUMN")
                failed_rules.append(f"Missing required columns: {missing_req}")

            batch_id_val = raw.get("batch_id")
            if "batch_id" not in missing_req and (not isinstance(batch_id_val, str) or not batch_id_val.strip()):
                reason_codes.append("INVALID_BATCH_ID")
                failed_rules.append("batch_id must be a non-empty string.")

            present_count = sum(1 for col in checked_fields if raw.get(col) is not None and raw.get(col) != "")
            completeness = present_count / max(1, len(checked_fields))
            if completeness < dq_policy.min_completeness_ratio:
                reason_codes.append("COMPLETENESS_BELOW_THRESHOLD")
                failed_rules.append(
                    f"Completeness {completeness:.3f} < threshold {dq_policy.min_completeness_ratio:.3f}."
                )

            normalized: Dict[str, Any] = {
                "batch_id": batch_id_val.strip() if isinstance(batch_id_val, str) else batch_id_val,
            }
            for col, bound in dq_policy.numeric_bounds.items():
                val = raw.get(col)
                if val is None:
                    if not bound.allow_null and col not in missing_req:
                        reason_codes.append("NULL_NON_NULLABLE_COLUMN")
                        failed_rules.append(f"Column {col!r} cannot be null.")
                    normalized[col] = None
                    continue
                if isinstance(val, bool):
                    reason_codes.append("INVALID_NUMERIC_TYPE")
                    failed_rules.append(f"Column {col!r} cannot be boolean.")
                    continue
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    reason_codes.append("INVALID_NUMERIC_TYPE")
                    failed_rules.append(f"Column {col!r} value {val!r} is not numeric.")
                    continue
                if not math.isfinite(num):
                    reason_codes.append("NON_FINITE_VALUE")
                    failed_rules.append(f"Column {col!r} is non-finite ({val!r}).")
                    continue
                if num < bound.min_val or num > bound.max_val:
                    reason_codes.append("OUT_OF_PHYSICAL_BOUNDS")
                    failed_rules.append(
                        f"Column {col!r}={num} outside [{bound.min_val}, {bound.max_val}]."
                    )
                    continue
                normalized[col] = num

            if dq_policy.enforce_monotonic_timestamp and isinstance(normalized.get("batch_id"), str):
                b_id = str(normalized["batch_id"])
                t_val = normalized.get("t_h")
                if isinstance(t_val, (int, float)) and math.isfinite(float(t_val)):
                    prev_t = last_t_by_batch.get(b_id)
                    if prev_t is not None:
                        if float(t_val) < prev_t - 1e-12:
                            reason_codes.append("NON_MONOTONIC_TIMESTAMP")
                            failed_rules.append(
                                f"Batch {b_id!r} timestamp t_h={t_val} precedes previous {prev_t}."
                            )
                        elif abs(float(t_val) - prev_t) <= 1e-12:
                            reason_codes.append("DUPLICATE_TIMESTAMP")
                            failed_rules.append(
                                f"Batch {b_id!r} duplicate timestamp t_h={t_val}."
                            )

        unique_reasons = sorted(set(reason_codes))
        if unique_reasons:
            for code in unique_reasons:
                reason_counts[code] += 1
            batch_label = (
                str(raw.get("batch_id", "UNKNOWN"))
                if isinstance(raw, dict)
                else "MALFORMED"
            )
            record_medallion_quarantine(batch_label, unique_reasons)
            q_entry: Dict[str, Any] = {
                "row_index": row_index,
                "run_id": run_id,
                "bronze_row_sha256": bronze_row_sha,
                "raw_payload": raw,
                "reason_codes": unique_reasons,
                "failed_rules": failed_rules,
                "quarantined_utc": quarantined_utc,
            }
            q_entry["quarantine_entry_sha256"] = _canonical_sha256(q_entry)
            quarantine_rows.append(q_entry)
        else:
            b_id = str(normalized["batch_id"])
            last_t_by_batch[b_id] = float(normalized["t_h"])
            s_entry: Dict[str, Any] = {
                "row_index": row_index,
                "run_id": run_id,
                "bronze_row_sha256": bronze_row_sha,
                "normalized_payload": normalized,
                "promoted_utc": silver_utc,
            }
            s_entry["silver_row_sha256"] = _canonical_sha256(s_entry)
            silver_rows.append(s_entry)

    if len(silver_rows) + len(quarantine_rows) != len(bronze_rows):
        raise AssertionError("Row conservation violated between Bronze, Silver, and Quarantine.")

    _write_jsonl(silver_path, silver_rows)
    _write_jsonl(quarantine_path, quarantine_rows)
    return {
        "run_id": run_id,
        "silver_path": str(silver_path),
        "silver_sha256": _file_sha256(silver_path),
        "quarantine_path": str(quarantine_path),
        "quarantine_sha256": _file_sha256(quarantine_path),
        "bronze_count": len(bronze_rows),
        "silver_count": len(silver_rows),
        "quarantine_count": len(quarantine_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def build_gold_features(
    silver_path: str | Path,
    quarantine_path: str | Path,
    bronze_info: Dict[str, Any],
    silver_info: Dict[str, Any],
    output_dir: str | Path,
    run_id: str,
    holdout_batches: Optional[Sequence[str]] = None,
    C_star: float = 8.2,
) -> Dict[str, Any]:
    """Compute Gold ML features and deterministic batch-grouped train/holdout splits."""
    base_dir = Path(output_dir)
    s_path = Path(silver_path)
    q_path = Path(quarantine_path)

    silver_rows = _read_jsonl(s_path)
    by_batch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s_row in silver_rows:
        payload = s_row["normalized_payload"]
        by_batch[str(payload["batch_id"])].append(s_row)

    gold_rows: List[Dict[str, Any]] = []
    for batch_id in sorted(by_batch.keys()):
        batch_items = sorted(by_batch[batch_id], key=lambda item: float(item["normalized_payload"]["t_h"]))
        prev_payload: Optional[Dict[str, Any]] = None
        for item in batch_items:
            p = item["normalized_payload"]
            t_h = float(p["t_h"])
            xv = float(p["X_v"])
            cl = float(p["C_L"])
            vol = float(p["V"])
            f_in = float(p["F"]) if p.get("F") is not None else 0.0
            do_pct = 100.0 * cl / max(1e-6, float(C_star))
            dilution_d = f_in / max(1e-6, vol)

            if prev_payload is None:
                dxv_dt = 0.0
                dv_dt = 0.0
            else:
                dt = max(1e-6, t_h - float(prev_payload["t_h"]))
                dxv_dt = (xv - float(prev_payload["X_v"])) / dt
                dv_dt = (vol - float(prev_payload["V"])) / dt

            feat_row: Dict[str, Any] = {
                "run_id": run_id,
                "row_index": int(item["row_index"]),
                "silver_row_sha256": str(item["silver_row_sha256"]),
                "bronze_row_sha256": str(item["bronze_row_sha256"]),
                **p,
                "DO_pct": round(do_pct, 6),
                "dilution_rate_D": round(dilution_d, 8),
                "dXv_dt": round(dxv_dt, 8),
                "dV_dt": round(dv_dt, 8),
            }
            feat_row["gold_row_sha256"] = _canonical_sha256(feat_row)
            gold_rows.append(feat_row)
            prev_payload = p

    all_batches = sorted(by_batch.keys())
    if holdout_batches is not None:
        holdout_set = {str(b) for b in holdout_batches} & set(all_batches)
    elif len(all_batches) >= 2:
        n_hold = max(1, int(round(0.25 * len(all_batches))))
        ranked = sorted(all_batches, key=lambda b: hashlib.sha256(b.encode("utf-8")).hexdigest())
        holdout_set = set(ranked[:n_hold])
    else:
        holdout_set = set()
    train_set = set(all_batches) - holdout_set

    train_rows = [r for r in gold_rows if r["batch_id"] in train_set]
    holdout_rows = [r for r in gold_rows if r["batch_id"] in holdout_set]

    gold_path = base_dir / "gold" / f"{run_id}_features.jsonl"
    train_path = base_dir / "gold" / f"{run_id}_train.jsonl"
    holdout_path = base_dir / "gold" / f"{run_id}_holdout.jsonl"
    manifest_path = base_dir / "gold" / f"{run_id}_manifest.json"

    _write_jsonl(gold_path, gold_rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(holdout_path, holdout_rows)

    manifest: Dict[str, Any] = {
        "schema_version": "medallion_v1",
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bronze_path": str(bronze_info["bronze_path"]),
        "bronze_sha256": str(bronze_info["bronze_sha256"]),
        "silver_path": str(s_path),
        "silver_sha256": str(silver_info["silver_sha256"]),
        "quarantine_path": str(q_path),
        "quarantine_sha256": str(silver_info["quarantine_sha256"]),
        "gold_path": str(gold_path),
        "gold_sha256": _file_sha256(gold_path),
        "train_split_path": str(train_path),
        "train_split_sha256": _file_sha256(train_path),
        "holdout_split_path": str(holdout_path),
        "holdout_split_sha256": _file_sha256(holdout_path),
        "row_counts": {
            "bronze": int(silver_info["bronze_count"]),
            "silver": int(silver_info["silver_count"]),
            "quarantine": int(silver_info["quarantine_count"]),
            "gold": len(gold_rows),
            "train": len(train_rows),
            "holdout": len(holdout_rows),
        },
        "train_batches": sorted(train_set),
        "holdout_batches": sorted(holdout_set),
        "dq_summary": {
            "reason_counts": dict(silver_info.get("reason_counts", {})),
            "quarantine_rate": round(
                float(silver_info["quarantine_count"]) / max(1, int(silver_info["bronze_count"])),
                6,
            ),
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def run_medallion_pipeline(
    records: Iterable[Any],
    output_dir: str | Path,
    run_id: str,
    policy: Optional[DataQualityPolicy] = None,
    holdout_batches: Optional[Sequence[str]] = None,
    source_uri: str = "in_memory",
) -> Dict[str, Any]:
    """Run the full Bronze -> Silver -> Gold Medallion ETL pipeline with quarantine & lineage."""
    bronze_info = ingest_bronze(records, output_dir, run_id, source_uri=source_uri)
    silver_info = promote_bronze_to_silver(bronze_info["bronze_path"], output_dir, run_id, policy=policy)
    gold_info = build_gold_features(
        silver_info["silver_path"],
        silver_info["quarantine_path"],
        bronze_info,
        silver_info,
        output_dir,
        run_id,
        holdout_batches=holdout_batches,
    )
    return {
        "bronze": bronze_info,
        "silver": silver_info,
        "gold": gold_info,
        "manifest_path": gold_info["manifest_path"],
        "manifest": gold_info["manifest"],
    }


def verify_medallion_lineage(manifest_path: str | Path) -> Tuple[bool, List[str]]:
    """Verify SHA-256 digests, row conservation, batch split disjointness, and per-row hashes."""
    errors: List[str] = []
    m_path = Path(manifest_path)
    if not m_path.exists():
        return False, [f"Manifest not found: {m_path}"]

    try:
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"Invalid manifest JSON: {exc}"]

    stored_m_sha = str(manifest.get("manifest_sha256", ""))
    calc_m_sha = _canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if stored_m_sha != calc_m_sha:
        errors.append(f"manifest_sha256 mismatch ({stored_m_sha[:12]} != {calc_m_sha[:12]}).")

    file_checks = (
        ("bronze_path", "bronze_sha256"),
        ("silver_path", "silver_sha256"),
        ("quarantine_path", "quarantine_sha256"),
        ("gold_path", "gold_sha256"),
        ("train_split_path", "train_split_sha256"),
        ("holdout_split_path", "holdout_split_sha256"),
    )
    for path_key, sha_key in file_checks:
        p = Path(str(manifest.get(path_key, "")))
        if not p.exists():
            errors.append(f"Missing lineage artifact {path_key}: {p}")
            continue
        actual_sha = _file_sha256(p)
        expected_sha = str(manifest.get(sha_key, ""))
        if actual_sha != expected_sha:
            errors.append(f"{sha_key} mismatch for {p.name} ({actual_sha[:12]} != {expected_sha[:12]}).")

    counts = manifest.get("row_counts", {})
    bronze_rows = _read_jsonl(Path(str(manifest.get("bronze_path", ""))))
    silver_rows = _read_jsonl(Path(str(manifest.get("silver_path", ""))))
    quarantine_rows = _read_jsonl(Path(str(manifest.get("quarantine_path", ""))))
    gold_rows = _read_jsonl(Path(str(manifest.get("gold_path", ""))))
    train_rows = _read_jsonl(Path(str(manifest.get("train_split_path", ""))))
    holdout_rows = _read_jsonl(Path(str(manifest.get("holdout_split_path", ""))))

    if len(bronze_rows) != counts.get("bronze"):
        errors.append(f"Bronze row count mismatch ({len(bronze_rows)} != {counts.get('bronze')}).")
    if len(silver_rows) != counts.get("silver"):
        errors.append(f"Silver row count mismatch ({len(silver_rows)} != {counts.get('silver')}).")
    if len(quarantine_rows) != counts.get("quarantine"):
        errors.append(f"Quarantine row count mismatch ({len(quarantine_rows)} != {counts.get('quarantine')}).")
    if len(silver_rows) + len(quarantine_rows) != len(bronze_rows):
        errors.append(
            f"Row conservation failed: silver({len(silver_rows)}) + quarantine({len(quarantine_rows)}) != bronze({len(bronze_rows)})."
        )
    if len(gold_rows) != len(silver_rows):
        errors.append(f"Gold row count ({len(gold_rows)}) != Silver row count ({len(silver_rows)}).")
    if len(train_rows) + len(holdout_rows) != len(gold_rows):
        errors.append("Train + Holdout row counts do not equal Gold row count.")

    train_batches: Set[str] = {str(r.get("batch_id")) for r in train_rows}
    holdout_batches: Set[str] = {str(r.get("batch_id")) for r in holdout_rows}
    overlap = sorted(train_batches & holdout_batches)
    if overlap:
        errors.append(f"Batch leakage detected across train/holdout split: {overlap}")

    for b_row in bronze_rows:
        calc_b = _canonical_sha256(
            {
                "row_index": int(b_row.get("row_index", -1)),
                "run_id": b_row.get("run_id"),
                "raw_payload": b_row.get("raw_payload"),
            }
        )
        if b_row.get("raw_row_sha256") != calc_b:
            errors.append(f"Bronze row {b_row.get('row_index')} raw_row_sha256 mismatch.")
            break

    for s_row in silver_rows:
        payload_s = {k: v for k, v in s_row.items() if k != "silver_row_sha256"}
        if s_row.get("silver_row_sha256") != _canonical_sha256(payload_s):
            errors.append(f"Silver row {s_row.get('row_index')} silver_row_sha256 mismatch.")
            break

    for q_row in quarantine_rows:
        payload_q = {k: v for k, v in q_row.items() if k != "quarantine_entry_sha256"}
        if q_row.get("quarantine_entry_sha256") != _canonical_sha256(payload_q):
            errors.append(f"Quarantine row {q_row.get('row_index')} quarantine_entry_sha256 mismatch.")
            break

    for split_name, split_rows in (
        ("Gold", gold_rows),
        ("Gold train", train_rows),
        ("Gold holdout", holdout_rows),
    ):
        for g_row in split_rows:
            payload_g = {k: v for k, v in g_row.items() if k != "gold_row_sha256"}
            if g_row.get("gold_row_sha256") != _canonical_sha256(payload_g):
                errors.append(f"{split_name} row {g_row.get('row_index')} gold_row_sha256 mismatch.")
                break

    return len(errors) == 0, errors
