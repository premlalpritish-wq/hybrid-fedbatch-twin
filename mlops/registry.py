"""Offline-capable Prefect & MLflow-compatible training, evaluation, and model registry workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from api.runtime import ROOT_DIR, load_calibrated_assets
from hybrid_fedbatch_twin.mechanistic import compute_hydrodynamics

from .medallion import _read_jsonl, verify_medallion_lineage

try:
    import prefect as _prefect  # type: ignore

    _HAS_PREFECT = True
except ImportError:
    _prefect = None
    _HAS_PREFECT = False

try:
    import mlflow as _mlflow  # type: ignore

    _HAS_MLFLOW = True
except ImportError:
    _mlflow = None
    _HAS_MLFLOW = False

GENESIS_REGISTRY_SHA256 = "0" * 64


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class TaskRunRecord:
    """Captured execution metadata for an offline Prefect-compatible task."""

    task_name: str
    status: str
    started_utc: str
    completed_utc: str
    duration_ms: float
    error: Optional[str] = None


def task_compat(
    name: Optional[str] = None,
    use_native_prefect: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator mirroring Prefect @task offline without requiring a Prefect server."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        task_label = name or fn.__name__
        wrapped_fn = _prefect.task(name=task_label)(fn) if (_HAS_PREFECT and use_native_prefect) else fn

        @wraps(fn)
        def runner(*args: Any, task_trace: Optional[List[TaskRunRecord]] = None, **kwargs: Any) -> Any:
            started = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            try:
                out = wrapped_fn(*args, **kwargs)
                dur_ms = round((time.perf_counter() - t0) * 1000.0, 4)
                if task_trace is not None:
                    task_trace.append(
                        TaskRunRecord(
                            task_name=task_label,
                            status="COMPLETED",
                            started_utc=started,
                            completed_utc=datetime.now(timezone.utc).isoformat(),
                            duration_ms=dur_ms,
                        )
                    )
                return out
            except Exception as exc:
                dur_ms = round((time.perf_counter() - t0) * 1000.0, 4)
                if task_trace is not None:
                    task_trace.append(
                        TaskRunRecord(
                            task_name=task_label,
                            status="FAILED",
                            started_utc=started,
                            completed_utc=datetime.now(timezone.utc).isoformat(),
                            duration_ms=dur_ms,
                            error=str(exc),
                        )
                    )
                raise

        return runner

    return decorator


def flow_compat(
    name: Optional[str] = None,
    use_native_prefect: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator mirroring Prefect @flow offline without requiring a Prefect server."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        flow_label = name or fn.__name__
        wrapped_fn = _prefect.flow(name=flow_label)(fn) if (_HAS_PREFECT and use_native_prefect) else fn

        @wraps(fn)
        def runner(*args: Any, **kwargs: Any) -> Any:
            return wrapped_fn(*args, **kwargs)

        runner.flow_name = flow_label  # type: ignore[attr-defined]
        return runner

    return decorator


@dataclass(frozen=True)
class PromotionCriteria:
    """Quantitative model promotion thresholds prior to human QA release."""

    min_kd_r2: float = 0.70
    min_kla_r2: float = 0.90
    min_mu_r2: float = 0.95
    min_uhlendorff_score: float = 8.0
    require_uhlendorff_go: bool = True
    max_ekf_nans: int = 0
    min_x_nonnegative: bool = True
    require_lineage_valid: bool = True


class LocalMlflowCompatibleTracker:
    """Offline JSON + SHA-256 hash-chained model experiment and registry store."""

    def __init__(
        self,
        tracking_dir: str | Path,
        enable_native_mlflow: bool = False,
    ) -> None:
        self.tracking_dir = Path(tracking_dir)
        self.runs_dir = self.tracking_dir / "runs"
        self.ledger_path = self.tracking_dir / "registry_ledger.jsonl"
        self.enable_native_mlflow = bool(enable_native_mlflow and _HAS_MLFLOW)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _read_ledger(self) -> List[Dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                entries.append(json.loads(s))
        return entries

    def log_and_register_run(
        self,
        run_name: str,
        model_name: str,
        stage: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        tags: Dict[str, Any],
        dataset_lineage: Dict[str, Any],
        promotion_evaluation: Dict[str, Any],
        task_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist run manifest JSON and append a SHA-256 chained registry ledger entry."""
        run_id = uuid4().hex[:16]
        created_utc = datetime.now(timezone.utc).isoformat()
        existing = self._read_ledger()
        seq = len(existing) + 1
        prev_sha = str(existing[-1]["entry_sha256"]) if existing else GENESIS_REGISTRY_SHA256

        run_payload: Dict[str, Any] = {
            "schema_version": "mlops_registry_v1",
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "model_version": seq,
            "stage": stage,
            "created_utc": created_utc,
            "params": params,
            "metrics": metrics,
            "tags": tags,
            "dataset_lineage": dataset_lineage,
            "promotion_evaluation": promotion_evaluation,
            "task_trace": task_trace,
        }
        run_payload["run_manifest_sha256"] = _canonical_sha256(run_payload)

        run_path = self.runs_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

        ledger_entry: Dict[str, Any] = {
            "seq": seq,
            "utc_timestamp": created_utc,
            "run_id": run_id,
            "model_name": model_name,
            "model_version": seq,
            "stage": stage,
            "model_sha256": str(tags.get("model_sha256", "")),
            "params_sha256": str(tags.get("params_sha256", "")),
            "train_split_sha256": str(dataset_lineage.get("train_split_sha256", "")),
            "holdout_split_sha256": str(dataset_lineage.get("holdout_split_sha256", "")),
            "run_manifest_sha256": str(run_payload["run_manifest_sha256"]),
            "prev_entry_sha256": prev_sha,
        }
        ledger_entry["entry_sha256"] = _canonical_sha256(ledger_entry)

        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(ledger_entry, sort_keys=True) + "\n")

        if self.enable_native_mlflow and _mlflow is not None:
            mlruns_uri = (self.tracking_dir / "mlruns").resolve().as_uri()
            _mlflow.set_tracking_uri(mlruns_uri)
            with _mlflow.start_run(run_name=run_name):
                _mlflow.log_params({str(k): str(v) for k, v in params.items()})
                _mlflow.log_metrics({str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
                _mlflow.set_tags({str(k): str(v) for k, v in tags.items()})

        return {
            "run_id": run_id,
            "run_path": str(run_path),
            "ledger_path": str(self.ledger_path),
            "ledger_entry": ledger_entry,
            "run_manifest": run_payload,
        }

    def verify_registry_ledger(self) -> Tuple[bool, List[str]]:
        """Verify SHA-256 chain linkage AND companion runs/<run_id>.json manifest digests."""
        entries = self._read_ledger()
        if not entries:
            return False, ["Registry ledger is empty."]
        errors: List[str] = []
        expected_prev = GENESIS_REGISTRY_SHA256
        for idx, entry in enumerate(entries):
            expected_seq = idx + 1
            if entry.get("seq") != expected_seq:
                errors.append(f"Registry entry {idx}: expected seq={expected_seq}, got {entry.get('seq')!r}.")
            if entry.get("prev_entry_sha256") != expected_prev:
                errors.append(f"Registry entry {expected_seq}: broken prev_entry_sha256 link.")
            stored_sha = str(entry.get("entry_sha256", ""))
            recomputed = _canonical_sha256({k: v for k, v in entry.items() if k != "entry_sha256"})
            if stored_sha != recomputed:
                errors.append(f"Registry entry {expected_seq}: entry_sha256 mismatch.")

            run_id = str(entry.get("run_id", ""))
            run_path = self.runs_dir / f"{run_id}.json"
            if not run_path.exists():
                errors.append(f"Registry entry {expected_seq}: missing run manifest {run_path.name}.")
            else:
                try:
                    run_obj = json.loads(run_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"Registry entry {expected_seq}: invalid run manifest JSON ({exc}).")
                else:
                    calc_run_sha = _canonical_sha256(
                        {k: v for k, v in run_obj.items() if k != "run_manifest_sha256"}
                    )
                    if run_obj.get("run_manifest_sha256") != calc_run_sha:
                        errors.append(
                            f"Registry entry {expected_seq}: run manifest {run_path.name} run_manifest_sha256 mismatch."
                        )
                    if entry.get("run_manifest_sha256") != calc_run_sha:
                        errors.append(
                            f"Registry entry {expected_seq}: ledger run_manifest_sha256 disagrees with {run_path.name}."
                        )
                    if (
                        run_obj.get("run_id") != run_id
                        or run_obj.get("stage") != entry.get("stage")
                        or run_obj.get("model_version") != entry.get("model_version")
                    ):
                        errors.append(
                            f"Registry entry {expected_seq}: run manifest identity/stage disagrees with ledger."
                        )

            expected_prev = stored_sha
        return len(errors) == 0, errors


@task_compat(name="verify_dataset_lineage")
def _task_verify_dataset_lineage(manifest_path: str | Path) -> Dict[str, Any]:
    ok, errs = verify_medallion_lineage(manifest_path)
    m_path = Path(manifest_path)
    manifest = json.loads(m_path.read_text(encoding="utf-8")) if m_path.exists() else {}
    return {
        "lineage_valid": bool(ok),
        "lineage_errors": list(errs),
        "manifest_path": str(m_path),
        "manifest_sha256": str(manifest.get("manifest_sha256", "")),
        "bronze_sha256": str(manifest.get("bronze_sha256", "")),
        "silver_sha256": str(manifest.get("silver_sha256", "")),
        "quarantine_sha256": str(manifest.get("quarantine_sha256", "")),
        "train_split_sha256": str(manifest.get("train_split_sha256", "")),
        "holdout_split_sha256": str(manifest.get("holdout_split_sha256", "")),
        "holdout_split_path": str(manifest.get("holdout_split_path", "")),
        "row_counts": dict(manifest.get("row_counts", {})),
    }


@task_compat(name="evaluate_calibrated_model")
def _task_evaluate_calibrated_model(
    lineage_info: Dict[str, Any],
    results_summary_path: Optional[Path] = None,
    uhlendorff_path: Optional[Path] = None,
    candidate_metrics_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    p_rec, model = load_calibrated_assets()

    res_path = results_summary_path or (ROOT_DIR / "results_summary.json")
    uhl_path = uhlendorff_path or (ROOT_DIR / "artifacts" / "uhlendorff_2026_validation.json")
    results_data = json.loads(res_path.read_text(encoding="utf-8"))
    uhl_data = json.loads(uhl_path.read_text(encoding="utf-8"))

    neural_metrics = dict(results_data.get("neural_ode", {}))
    recovery_data = dict(results_data.get("recovery", {}))
    rem = dict(uhl_data.get("benchmark_comparison", {}).get("remediated", {}))

    # Execute live forward inference across Gold holdout rows to confirm finite runtime evaluation
    holdout_path = Path(str(lineage_info.get("holdout_split_path", "")))
    holdout_rows = _read_jsonl(holdout_path) if holdout_path.exists() else []
    eval_samples = 0
    non_finite_outputs = 0
    for row in holdout_rows:
        xv = float(row.get("X_v", 1.5))
        xd = float(row.get("X_d") if row.get("X_d") is not None else 0.02)
        sub = float(row.get("S") if row.get("S") is not None else 0.3)
        cl = float(row.get("C_L", 7.0))
        ace = float(row.get("A") if row.get("A") is not None else 0.1)
        vol = float(row.get("V", p_rec.V_0))
        rpm = float(row.get("rpm") if row.get("rpm") is not None else 450.0)
        vvm = float(row.get("vvm") if row.get("vvm") is not None else 1.2)
        hyd = compute_hydrodynamics(rpm, vvm, vol, p_rec)
        mu_m = p_rec.mu_max * (sub / (p_rec.K_S + sub)) * (cl / (p_rec.K_O2 + cl))
        pred = model.predict_full_with_uq(
            X_v=xv,
            X_d=xd,
            S=sub,
            C_L=cl,
            A=ace,
            v_tip=hyd["v_tip"],
            Pg_V=hyd["Pg_V"],
            v_s=hyd["v_s"],
            mu_mech=mu_m,
            V=vol,
        )
        eval_samples += 1
        if not all(
            math.isfinite(float(pred[k]))
            for k in ("delta_kd", "eta_kla", "q_A_prod", "q_A_cons", "eta_mu", "d_m")
        ):
            non_finite_outputs += 1

    metrics: Dict[str, Any] = {
        "kd_r2": float(neural_metrics["kd_r2"]),
        "kla_r2": float(neural_metrics["kla_r2"]),
        "qAnet_r2": float(neural_metrics["qAnet_r2"]),
        "mu_r2": float(neural_metrics["mu_r2"]),
        "hybrid_norm_rmse": float(neural_metrics["hybrid_norm_rmse"]),
        "nominal_norm_rmse": float(neural_metrics["nominal_norm_rmse"]),
        "stage1_max_rel_err_pct": float(recovery_data["max_relative_error_pct"]),
        "stage1_kappa_rel": float(recovery_data["condition_number_rel"]),
        "uhlendorff_score": float(rem["score"]),
        "uhlendorff_verdict": str(rem["verdict"]),
        "uhlendorff_pooled_test_wmse": float(rem["pooled_test_wmse"]),
        "uhlendorff_ci_coverage_pct": float(rem["ci_coverage_pct"]),
        "ekf_nans": int(rem["ekf_nans"]),
        "min_x": float(rem["min_x"]),
        "holdout_eval_samples": eval_samples,
        "holdout_non_finite_outputs": non_finite_outputs,
    }
    if candidate_metrics_override:
        metrics.update(candidate_metrics_override)

    tags = {
        "params_sha256": _canonical_sha256(p_rec.to_dict()),
        "model_sha256": str(model.weights_sha256),
        "calibration_sha256": str(recovery_data.get("calibration_sha256", "")),
        "n_params": int(model.n_params),
        "n_bootstrap": int(len(model.ensemble_weights)),
    }
    return {"metrics": metrics, "tags": tags}


@task_compat(name="evaluate_promotion_and_register")
def _task_evaluate_promotion_and_register(
    tracker: LocalMlflowCompatibleTracker,
    run_name: str,
    model_name: str,
    lineage_info: Dict[str, Any],
    eval_info: Dict[str, Any],
    criteria: PromotionCriteria,
    task_trace: List[TaskRunRecord],
) -> Dict[str, Any]:
    m = eval_info["metrics"]
    failed_criteria: List[str] = []

    if criteria.require_lineage_valid and not lineage_info.get("lineage_valid", False):
        failed_criteria.append(f"lineage_invalid: {lineage_info.get('lineage_errors', [])}")
    if float(m.get("kd_r2", 0.0)) < criteria.min_kd_r2:
        failed_criteria.append(f"kd_r2={m.get('kd_r2')} < {criteria.min_kd_r2}")
    if float(m.get("kla_r2", 0.0)) < criteria.min_kla_r2:
        failed_criteria.append(f"kla_r2={m.get('kla_r2')} < {criteria.min_kla_r2}")
    if float(m.get("mu_r2", 0.0)) < criteria.min_mu_r2:
        failed_criteria.append(f"mu_r2={m.get('mu_r2')} < {criteria.min_mu_r2}")
    if float(m.get("uhlendorff_score", 0.0)) < criteria.min_uhlendorff_score:
        failed_criteria.append(
            f"uhlendorff_score={m.get('uhlendorff_score')} < {criteria.min_uhlendorff_score}"
        )
    verdict = str(m.get("uhlendorff_verdict", ""))
    if criteria.require_uhlendorff_go and "GO" not in verdict.upper():
        failed_criteria.append(f"uhlendorff_verdict={verdict!r} is not GO")
    if int(m.get("ekf_nans", 999)) > criteria.max_ekf_nans:
        failed_criteria.append(f"ekf_nans={m.get('ekf_nans')} > {criteria.max_ekf_nans}")
    if criteria.min_x_nonnegative and float(m.get("min_x", -1.0)) < 0.0:
        failed_criteria.append(f"min_x={m.get('min_x')} < 0.0")
    if int(m.get("holdout_non_finite_outputs", 0)) > 0:
        failed_criteria.append(
            f"holdout_non_finite_outputs={m.get('holdout_non_finite_outputs')} > 0"
        )

    passed = len(failed_criteria) == 0
    # Model stays in PROMOTED_STAGING until human QA signs off under 21 CFR Part 11
    stage = "PROMOTED_STAGING" if passed else "REJECTED"
    promotion_eval = {
        "passed": passed,
        "stage": stage,
        "requires_human_qa_for_production": True,
        "criteria": asdict(criteria),
        "failed_criteria": failed_criteria,
    }

    trace_dicts = [asdict(t) for t in task_trace]
    reg_out = tracker.log_and_register_run(
        run_name=run_name,
        model_name=model_name,
        stage=stage,
        params={
            "n_params": eval_info["tags"]["n_params"],
            "n_bootstrap": eval_info["tags"]["n_bootstrap"],
            "offline_mode": True,
        },
        metrics=m,
        tags=eval_info["tags"],
        dataset_lineage=lineage_info,
        promotion_evaluation=promotion_eval,
        task_trace=trace_dicts,
    )
    return {
        **reg_out,
        "promotion_evaluation": promotion_eval,
    }


@flow_compat(name="offline_hybrid_twin_training_and_registration")
def run_offline_training_and_registration_flow(
    manifest_path: str | Path,
    tracking_dir: str | Path,
    run_name: str = "gam86_b15_offline_qualification",
    model_name: str = "hybrid_fedbatch_gam86_b15",
    criteria: Optional[PromotionCriteria] = None,
    candidate_metrics_override: Optional[Dict[str, Any]] = None,
    results_summary_path: Optional[Path] = None,
    uhlendorff_path: Optional[Path] = None,
    enable_native_mlflow: bool = False,
) -> Dict[str, Any]:
    """Run the offline Prefect/MLflow-compatible lineage, evaluation, and model registry flow."""
    crit = criteria or PromotionCriteria()
    tracker = LocalMlflowCompatibleTracker(
        tracking_dir=tracking_dir,
        enable_native_mlflow=enable_native_mlflow,
    )
    task_trace: List[TaskRunRecord] = []

    lineage_info = _task_verify_dataset_lineage(manifest_path, task_trace=task_trace)
    eval_info = _task_evaluate_calibrated_model(
        lineage_info,
        results_summary_path=results_summary_path,
        uhlendorff_path=uhlendorff_path,
        candidate_metrics_override=candidate_metrics_override,
        task_trace=task_trace,
    )
    reg_result = _task_evaluate_promotion_and_register(
        tracker,
        run_name,
        model_name,
        lineage_info,
        eval_info,
        crit,
        task_trace,
        task_trace=task_trace,
    )
    ledger_ok, ledger_errs = tracker.verify_registry_ledger()
    return {
        "flow_name": "offline_hybrid_twin_training_and_registration",
        "prefect_installed": _HAS_PREFECT,
        "mlflow_installed": _HAS_MLFLOW,
        "lineage": lineage_info,
        "metrics": eval_info["metrics"],
        "tags": eval_info["tags"],
        "promotion": reg_result["promotion_evaluation"],
        "run_id": reg_result["run_id"],
        "run_path": reg_result["run_path"],
        "ledger_path": reg_result["ledger_path"],
        "ledger_valid": ledger_ok,
        "ledger_errors": ledger_errs,
        "task_trace": [asdict(t) for t in task_trace],
    }
