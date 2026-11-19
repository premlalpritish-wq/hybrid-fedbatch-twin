"""Automated repository hygiene, artifact schema, and ALCOA+ provenance verification."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.runtime import _load_bundle_model
from hybrid_fedbatch_twin.estimator import BioreactorDigitalTwin
from hybrid_fedbatch_twin.mechanistic import BioreactorParams, STATE_NAMES

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_ARTIFACT_REL_PATHS = (
    "docs/assets/twin_bundle.json",
    "docs/assets/twin_bundle.js",
    "artifacts/uhlendorff_2026_validation.json",
    "docs/assets/uhlendorff_2026_validation.js",
    "results_summary.json",
)

FORBIDDEN_PATH_PARTS = {
    "data",
    "papers",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".venv",
    "venv",
    "build",
    "dist",
    ".ipynb_checkpoints",
}

FORBIDDEN_EXTENSIONS = {
    ".db",
    ".db-wal",
    ".db-shm",
    ".db-journal",
    ".sqlite",
    ".sqlite3",
    ".pyc",
    ".pyo",
    ".tmp",
    ".pdf",
    ".zip",
    ".xlsx",
}

FORBIDDEN_FILENAMES = {
    ".DS_Store",
    "Thumbs.db",
    ".env",
}

SAMPLE_IGNORED_PATHS = (
    "data/uhlendorff_2026/Dataset.zip",
    "papers/Von Stosch et al 2014.pdf",
    "hybrid_twin_ledger.db",
    "local.db-wal",
    "local.sqlite3",
    "__pycache__/mod.pyc",
    ".pytest_cache/README.md",
    "hybrid_fedbatch_twin.egg-info/PKG-INFO",
    "build/lib/x.py",
    "dist/x.whl",
    ".venv/bin/python",
    ".env",
    "pytest_runner_tmp/test.json",
)


def _run_git(root_dir: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(root_dir),
        capture_output=True,
        text=True,
        check=False,
    )


def _match_forbidden_rel_path(rel_path: str) -> Optional[str]:
    p = Path(rel_path)
    for part in p.parts[:-1]:
        if (
            part in FORBIDDEN_PATH_PARTS
            or part.endswith(".egg-info")
            or fnmatch.fnmatchcase(part.lower(), "*pytest*tmp*")
        ):
            return f"forbidden path component '{part}'"
    name = p.name
    if (
        name in FORBIDDEN_PATH_PARTS
        or name.endswith(".egg-info")
        or fnmatch.fnmatchcase(name.lower(), "*pytest*tmp*")
    ):
        return f"forbidden directory/file '{name}'"
    if name in FORBIDDEN_FILENAMES or name.startswith(".env."):
        return f"forbidden filename '{name}'"
    suffix = p.suffix.lower()
    if suffix in FORBIDDEN_EXTENSIONS:
        return f"forbidden extension '{suffix}'"
    return None


def _load_json_file(path: Path, label: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Failed to parse {label} ({path}): {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{label} must decode to a JSON object.")
        return None
    return data


def _verify_js_wrapper(
    js_path: Path,
    prefix: str,
    expected_obj: Dict[str, Any],
    label: str,
    errors: List[str],
) -> None:
    if not js_path.exists():
        errors.append(f"Missing {label}: {js_path}")
        return
    text = js_path.read_text(encoding="utf-8").strip()
    if not text.startswith(prefix):
        errors.append(f"{label} must start with {prefix!r}.")
        return
    payload_text = text[len(prefix) :].rstrip()
    if payload_text.endswith(";"):
        payload_text = payload_text[:-1].rstrip()
    try:
        js_obj = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        errors.append(f"{label} contains invalid JSON payload: {exc}")
        return
    if js_obj != expected_obj:
        errors.append(f"{label} payload has drifted from its companion JSON file.")


def _collect_defined_test_names(root_dir: Path) -> set[str]:
    names: set[str] = set()
    tests_dir = root_dir / "tests"
    if not tests_dir.exists():
        return names
    for test_file in sorted(tests_dir.glob("*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(node.name)
    return names


def check_git_hygiene(root_dir: Path) -> List[str]:
    """Verify .gitignore coverage, tracked files, untracked clutter, and required artifacts."""
    errors: List[str] = []

    if not (root_dir / ".gitignore").exists():
        errors.append("Missing .gitignore at repository root.")

    for rel in REQUIRED_ARTIFACT_REL_PATHS:
        if not (root_dir / rel).exists():
            errors.append(f"Required tracked artifact missing: {rel}")

    if not (root_dir / ".git").exists() or shutil.which("git") is None:
        return errors

    cached_ignored = _run_git(root_dir, ["ls-files", "--cached", "-i", "--exclude-standard"])
    if cached_ignored.returncode != 0:
        errors.append(f"git ls-files --cached -i failed: {cached_ignored.stderr.strip()}")
    elif cached_ignored.stdout.strip():
        files = [line.strip() for line in cached_ignored.stdout.splitlines() if line.strip()]
        errors.append(f"Tracked files match .gitignore rules: {files}")

    ls_tracked = _run_git(root_dir, ["ls-files"])
    if ls_tracked.returncode != 0:
        errors.append(f"git ls-files failed: {ls_tracked.stderr.strip()}")
    else:
        for rel in (line.strip() for line in ls_tracked.stdout.splitlines() if line.strip()):
            reason = _match_forbidden_rel_path(rel)
            if reason is not None:
                errors.append(f"Tracked file violates hygiene rule ({reason}): {rel}")

    for sample in SAMPLE_IGNORED_PATHS:
        chk = _run_git(root_dir, ["check-ignore", sample])
        if chk.returncode != 0:
            errors.append(f".gitignore does not ignore sample local artifact path: {sample}")

    untracked = _run_git(root_dir, ["ls-files", "--others", "--exclude-standard"])
    if untracked.returncode != 0:
        errors.append(f"git ls-files --others failed: {untracked.stderr.strip()}")
    else:
        for rel in (line.strip() for line in untracked.stdout.splitlines() if line.strip()):
            reason = _match_forbidden_rel_path(rel)
            if reason is not None:
                errors.append(f"Untracked non-ignored clutter ({reason}): {rel}")

    return errors


def check_twin_bundle_artifact(
    root_dir: Path,
    bundle_json_path: Optional[Path] = None,
    bundle_js_path: Optional[Path] = None,
) -> List[str]:
    """Validate docs/assets/twin_bundle.json, twin_bundle.js, GAM-86 weights, DoV, and ALCOA+ preview."""
    errors: List[str] = []
    json_path = bundle_json_path or (root_dir / "docs" / "assets" / "twin_bundle.json")
    js_path = bundle_js_path or (root_dir / "docs" / "assets" / "twin_bundle.js")

    bundle = _load_json_file(json_path, "twin_bundle.json", errors)
    if bundle is None:
        return errors

    required_top = {
        "schema_version",
        "generated_utc",
        "constants",
        "p_true",
        "p_rec",
        "stage1_recovery",
        "gam86",
        "benchmarks",
        "audit_progression",
        "pytest_suite",
        "alcoa_sample",
        "bundle_sha256",
    }
    missing_top = sorted(required_top - set(bundle.keys()))
    if missing_top:
        errors.append(f"twin_bundle.json missing top-level keys: {missing_top}")
        return errors

    stored_sha = str(bundle.get("bundle_sha256", ""))
    calc_sha = hashlib.sha256(
        json.dumps({k: v for k, v in bundle.items() if k != "bundle_sha256"}, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    if stored_sha != calc_sha:
        errors.append(
            f"twin_bundle.json bundle_sha256 mismatch ({stored_sha[:12]} != {calc_sha[:12]})."
        )

    _verify_js_wrapper(
        js_path,
        "window.HYBRID_TWIN_BUNDLE = ",
        bundle,
        "twin_bundle.js",
        errors,
    )

    constants = bundle.get("constants", {})
    if constants.get("state_names") != list(STATE_NAMES):
        errors.append(f"twin_bundle constants.state_names mismatch: {constants.get('state_names')!r}")

    for p_key in ("p_true", "p_rec"):
        try:
            BioreactorParams(**bundle[p_key])
        except Exception as exc:
            errors.append(f"Invalid BioreactorParams in twin_bundle[{p_key!r}]: {exc}")

    gam = bundle.get("gam86", {})
    if gam.get("n_params") != 86:
        errors.append(f"gam86.n_params expected 86, got {gam.get('n_params')!r}.")
    if gam.get("n_bootstrap") != 15:
        errors.append(f"gam86.n_bootstrap expected 15, got {gam.get('n_bootstrap')!r}.")
    if len(gam.get("weights", [])) != 86:
        errors.append(f"gam86.weights length expected 86, got {len(gam.get('weights', []))}.")
    ens = gam.get("ensemble_weights", [])
    if len(ens) != 15 or any(len(w) != 86 for w in ens):
        errors.append("gam86.ensemble_weights must contain 15 vectors of length 86.")

    dov_stats = gam.get("dov_stats", {})
    expected_dov_keys = {"kd", "kla", "met", "global"}
    if not expected_dov_keys.issubset(set(dov_stats.keys())):
        errors.append(f"gam86.dov_stats missing keys: {sorted(expected_dov_keys - set(dov_stats.keys()))}")
    else:
        for dkey in sorted(expected_dov_keys):
            entry = dov_stats[dkey]
            missing_d = {"mean", "cov_inv", "p99", "std"} - set(entry.keys())
            if missing_d:
                errors.append(f"gam86.dov_stats[{dkey!r}] missing keys: {sorted(missing_d)}")

    try:
        _, hydrated_model = _load_bundle_model(json_path)
        if not hydrated_model.is_trained:
            errors.append("Hydrated GAM-86 model reports is_trained=False.")
        if hydrated_model.weights_sha256 != gam.get("weights_sha256"):
            errors.append(
                f"Hydrated weights_sha256 mismatch ({hydrated_model.weights_sha256[:12]} != {str(gam.get('weights_sha256'))[:12]})."
            )
    except Exception as exc:
        errors.append(f"Failed to hydrate bundle model via _load_bundle_model: {exc}")

    bench = bundle.get("benchmarks", {})
    ctrl_5l = bench.get("control_strategies_5L", {})
    expected_strats = {"hybrid_closed_loop", "nominal_closed_loop", "open_loop_exp"}
    if not expected_strats.issubset(set(ctrl_5l.keys())):
        errors.append(
            f"benchmarks.control_strategies_5L missing keys: {sorted(expected_strats - set(ctrl_5l.keys()))}"
        )
    if not isinstance(bench.get("disturbances_5L"), list) or not bench["disturbances_5L"]:
        errors.append("benchmarks.disturbances_5L must be a non-empty list.")
    scale_1000 = bench.get("scale_1000L", {})
    for sk in ("rpm_matched", "vvm_matched", "Pg_V_matched", "v_s_matched", "summary"):
        if sk not in scale_1000:
            errors.append(f"benchmarks.scale_1000L missing key {sk!r}.")

    alcoa = bundle.get("alcoa_sample", {})
    required_alcoa = {
        "alcoa_standard",
        "batch_id",
        "operator_id",
        "t_now_h",
        "params_sha256",
        "model_sha256",
        "chain_head_sha256",
        "chain_valid",
        "calibration_metadata",
        "current_state",
        "current_std",
        "audit_log_count",
        "audit_log_truncated",
        "record_sha256",
        "audit_log_preview",
    }
    missing_alcoa = sorted(required_alcoa - set(alcoa.keys()))
    if missing_alcoa:
        errors.append(f"twin_bundle.alcoa_sample missing keys: {missing_alcoa}")
    else:
        if alcoa.get("chain_valid") is not True:
            errors.append("twin_bundle.alcoa_sample.chain_valid must be True.")
        for sha_field in ("params_sha256", "model_sha256", "chain_head_sha256", "record_sha256"):
            if not _SHA256_RE.match(str(alcoa.get(sha_field, ""))):
                errors.append(f"twin_bundle.alcoa_sample.{sha_field} is not a 64-char hex SHA-256.")
        if alcoa.get("model_sha256") != gam.get("weights_sha256"):
            errors.append("twin_bundle.alcoa_sample.model_sha256 does not match gam86.weights_sha256.")

    suite = bundle.get("pytest_suite", [])
    if not isinstance(suite, list) or not suite:
        errors.append("twin_bundle.pytest_suite must be a non-empty list.")
    else:
        defined_tests = _collect_defined_test_names(root_dir)
        for idx, item in enumerate(suite):
            if not isinstance(item, dict) or not {"name", "status", "focus"}.issubset(item.keys()):
                errors.append(f"twin_bundle.pytest_suite[{idx}] missing required keys.")
                continue
            if item["status"] != "PASSED":
                errors.append(f"twin_bundle.pytest_suite[{idx}] ({item['name']}) status != PASSED.")
            if defined_tests and item["name"] not in defined_tests:
                errors.append(
                    f"twin_bundle.pytest_suite references missing test function: {item['name']}"
                )

    return errors


def check_results_summary_and_audit(
    root_dir: Path,
    summary_path: Optional[Path] = None,
    bundle_json_path: Optional[Path] = None,
) -> List[str]:
    """Validate results_summary.json schema, ALCOA+ verify_audit_record, and bundle parity."""
    errors: List[str] = []
    sum_path = summary_path or (root_dir / "results_summary.json")
    bun_path = bundle_json_path or (root_dir / "docs" / "assets" / "twin_bundle.json")

    summary = _load_json_file(sum_path, "results_summary.json", errors)
    bundle = _load_json_file(bun_path, "twin_bundle.json", errors)
    if summary is None:
        return errors

    required_top = {"recovery", "neural_ode", "control", "alcoa_provenance"}
    missing_top = sorted(required_top - set(summary.keys()))
    if missing_top:
        errors.append(f"results_summary.json missing top-level keys: {missing_top}")
        return errors

    ctrl_keys = set(summary["control"].keys())
    expected_ctrl = {"hybrid_closed_loop", "nominal_closed_loop", "open_loop_exp"}
    if not expected_ctrl.issubset(ctrl_keys):
        errors.append(f"results_summary.control missing keys: {sorted(expected_ctrl - ctrl_keys)}")

    alcoa_rec = summary["alcoa_provenance"]
    ok_rec, rec_errs = BioreactorDigitalTwin.verify_audit_record(alcoa_rec)
    if not ok_rec:
        errors.append(f"results_summary.alcoa_provenance failed verify_audit_record: {rec_errs}")
    if alcoa_rec.get("audit_log_truncated") is not False:
        errors.append("results_summary.alcoa_provenance must not be truncated.")
    if len(alcoa_rec.get("audit_log", [])) != alcoa_rec.get("audit_log_count"):
        errors.append("results_summary.alcoa_provenance audit_log length != audit_log_count.")

    if bundle is not None:
        if summary["recovery"].get("calibration_sha256") != bundle.get("stage1_recovery", {}).get(
            "calibration_sha256"
        ):
            errors.append("Stage-1 calibration_sha256 differs between results_summary and twin_bundle.")
        if alcoa_rec.get("model_sha256") != bundle.get("gam86", {}).get("weights_sha256"):
            errors.append("alcoa_provenance.model_sha256 differs from twin_bundle.gam86.weights_sha256.")
        sample = bundle.get("alcoa_sample", {})
        for key in ("record_sha256", "chain_head_sha256", "audit_log_count"):
            if alcoa_rec.get(key) != sample.get(key):
                errors.append(
                    f"results_summary.alcoa_provenance.{key} differs from twin_bundle.alcoa_sample.{key}."
                )
        expected_preview = alcoa_rec.get("audit_log", [])[:6] + alcoa_rec.get("audit_log", [])[-6:]
        if sample.get("audit_log_preview") != expected_preview:
            errors.append(
                "twin_bundle.alcoa_sample.audit_log_preview does not match results_summary audit_log slices."
            )

    return errors


def check_uhlendorff_validation_artifact(
    root_dir: Path,
    artifact_json_path: Optional[Path] = None,
    artifact_js_path: Optional[Path] = None,
) -> List[str]:
    """Validate artifacts/uhlendorff_2026_validation.json and its companion JS bundle."""
    errors: List[str] = []
    json_path = artifact_json_path or (root_dir / "artifacts" / "uhlendorff_2026_validation.json")
    js_path = artifact_js_path or (root_dir / "docs" / "assets" / "uhlendorff_2026_validation.js")

    val = _load_json_file(json_path, "uhlendorff_2026_validation.json", errors)
    if val is None:
        return errors

    required_top = {
        "dataset_metadata",
        "mass_balance_discovery",
        "benchmark_comparison",
        "held_out_trajectories",
    }
    missing_top = sorted(required_top - set(val.keys()))
    if missing_top:
        errors.append(f"uhlendorff_2026_validation.json missing top-level keys: {missing_top}")
        return errors

    _verify_js_wrapper(
        js_path,
        "window.UHLENDORFF_2026_VALIDATION = ",
        val,
        "uhlendorff_2026_validation.js",
        errors,
    )

    meta = val["dataset_metadata"]
    req_meta = {
        "source",
        "dev_campaigns",
        "test_campaigns",
        "anomalous_runs",
        "sigma_y_clean",
        "sigma_y_unmasked",
        "total_offline_rows",
        "incomplete_7_sensor_rows",
        "anomaly_rows",
    }
    missing_meta = sorted(req_meta - set(meta.keys()))
    if missing_meta:
        errors.append(f"dataset_metadata missing keys: {missing_meta}")
        return errors

    dev_set = set(meta["dev_campaigns"])
    test_set = set(meta["test_campaigns"])
    if dev_set & test_set:
        errors.append(f"dev_campaigns and test_campaigns overlap: {sorted(dev_set & test_set)}")
    if len(dev_set | test_set) != 12:
        errors.append(f"Expected 12 total campaigns, got {len(dev_set | test_set)}.")

    for sig_key in ("sigma_y_clean", "sigma_y_unmasked"):
        vals = meta.get(sig_key, [])
        if len(vals) != 4 or not all(
            isinstance(x, (int, float)) and math.isfinite(x) and x > 0.0 for x in vals
        ):
            errors.append(f"dataset_metadata.{sig_key} must have 4 finite positive numbers.")

    mb = val["mass_balance_discovery"]
    for mb_key in ("reference_campaign", "v0", "v_end_true", "v_end_naive", "volume_series"):
        if mb_key not in mb:
            errors.append(f"mass_balance_discovery missing key {mb_key!r}.")
    exp10 = mb.get("volume_series", {}).get("EXP010")
    if not isinstance(exp10, dict) or not {"t", "v_true", "v_naive"}.issubset(exp10.keys()):
        errors.append("mass_balance_discovery.volume_series.EXP010 missing required series keys.")
    elif not (len(exp10["t"]) == len(exp10["v_true"]) == len(exp10["v_naive"]) > 0):
        errors.append("mass_balance_discovery.volume_series.EXP010 arrays must share non-zero length.")

    bc = val["benchmark_comparison"]
    if "unremediated" in bc:
        errors.append("benchmark_comparison must not contain legacy 'unremediated' section.")
    for bc_key in ("stage1_mechanistic_only", "remediated", "von_stosch_referee"):
        if bc_key not in bc:
            errors.append(f"benchmark_comparison missing key {bc_key!r}.")

    rem = bc.get("remediated", {})
    req_rem = {
        "k_eff",
        "gamma_col",
        "pooled_test_wmse",
        "rmse",
        "r2",
        "min_x",
        "per_campaign_test_wmse",
        "fedbatch_test_wmse",
        "shifted_batch_wmse",
        "anomaly_unmasked_wmse",
        "ci_coverage_pct",
        "ekf_nans",
        "scorecard",
        "score",
        "verdict",
    }
    missing_rem = sorted(req_rem - set(rem.keys()))
    if missing_rem:
        errors.append(f"benchmark_comparison.remediated missing keys: {missing_rem}")
    else:
        if rem["ekf_nans"] != 0:
            errors.append(f"benchmark_comparison.remediated.ekf_nans must be 0, got {rem['ekf_nans']}.")
        if float(rem["min_x"]) < 0.0:
            errors.append(f"benchmark_comparison.remediated.min_x must be >= 0, got {rem['min_x']}.")
        if set(rem["per_campaign_test_wmse"].keys()) != test_set:
            errors.append(
                f"per_campaign_test_wmse keys {sorted(rem['per_campaign_test_wmse'].keys())} != test_campaigns {sorted(test_set)}."
            )

    ho = val["held_out_trajectories"]
    if set(ho.keys()) != test_set:
        errors.append(f"held_out_trajectories keys {sorted(ho.keys())} != test_campaigns {sorted(test_set)}.")
    req_traj = {
        "mode",
        "t_feed_start",
        "t_off_r01",
        "obs_r01",
        "t_off_r02",
        "obs_r02",
        "ht_mean",
        "ht_std",
        "vs_mean",
        "ekf_r01",
    }
    for exp_id, tr in ho.items():
        missing_tr = sorted(req_traj - set(tr.keys()))
        if missing_tr:
            errors.append(f"held_out_trajectories[{exp_id!r}] missing keys: {missing_tr}")
            continue
        ekf_r01 = tr.get("ekf_r01", {})
        if not {"t", "vcd", "dropout"}.issubset(set(ekf_r01.keys())):
            errors.append(f"held_out_trajectories[{exp_id!r}].ekf_r01 missing t/vcd/dropout.")

    return errors


def check_honest_maturity_labels(root_dir: Path) -> List[str]:
    """Ensure README.md, app.py, and estimator.py do not carry prohibited Portela Level 3 claims."""
    errors: List[str] = []
    forbidden_phrases = (
        "Portela Level 3+",
        "Portela Level 3-Ready",
        "Portela Level 3 Digital Twin",
        "9.2 / 10",
    )
    for rel in ("README.md", "app.py", "hybrid_fedbatch_twin/estimator.py"):
        path = root_dir / rel
        if not path.exists():
            errors.append(f"Missing required documentation/code file: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            if phrase in text:
                errors.append(f"Prohibited maturity claim {phrase!r} found in {rel}.")
    return errors


def run_all_checks(root_dir: Path = ROOT_DIR) -> Dict[str, List[str]]:
    """Run all repository hygiene and artifact consistency checks."""
    return {
        "git_hygiene": check_git_hygiene(root_dir),
        "twin_bundle": check_twin_bundle_artifact(root_dir),
        "results_summary_and_audit": check_results_summary_and_audit(root_dir),
        "uhlendorff_validation": check_uhlendorff_validation_artifact(root_dir),
        "honest_maturity_labels": check_honest_maturity_labels(root_dir),
    }


def main() -> int:
    results = run_all_checks(ROOT_DIR)
    total_errors = 0
    for name, errs in results.items():
        if errs:
            total_errors += len(errs)
            print(f"[FAIL] {name} ({len(errs)} error(s))")
            for err in errs:
                print(f"  - {err}")
        else:
            print(f"[PASS] {name}")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
