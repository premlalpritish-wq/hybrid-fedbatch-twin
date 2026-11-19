"""Automated repository hygiene, artifact schema, and ALCOA+ provenance verification tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.verify_repo_hygiene import (
    ROOT_DIR,
    check_results_summary_and_audit,
    check_twin_bundle_artifact,
    check_uhlendorff_validation_artifact,
    run_all_checks,
)

pytestmark = pytest.mark.hygiene


def test_repo_git_and_artifact_hygiene_passes() -> None:
    """Verify gitignore, tracked/untracked hygiene, bundles, Uhlendorff validation, and maturity labels."""
    results = run_all_checks(ROOT_DIR)
    assert set(results.keys()) == {
        "git_hygiene",
        "twin_bundle",
        "results_summary_and_audit",
        "uhlendorff_validation",
        "honest_maturity_labels",
    }
    for check_name, errs in results.items():
        assert errs == [], f"{check_name} failed with errors: {errs}"


def test_repo_hygiene_cli_exits_zero() -> None:
    """Verify scripts/verify_repo_hygiene.py CLI runs cleanly and exits with status 0."""
    proc = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "verify_repo_hygiene.py")],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    for label in (
        "git_hygiene",
        "twin_bundle",
        "results_summary_and_audit",
        "uhlendorff_validation",
        "honest_maturity_labels",
    ):
        assert f"[PASS] {label}" in proc.stdout


def test_artifact_validators_detect_tampering(tmp_path: Path) -> None:
    """Verify artifact validators detect SHA tampering, JS drift, audit corruption, and schema breaks."""
    bundle_json = tmp_path / "twin_bundle.json"
    bundle_js = tmp_path / "twin_bundle.js"
    summary_json = tmp_path / "results_summary.json"
    uhl_json = tmp_path / "uhlendorff_2026_validation.json"
    uhl_js = tmp_path / "uhlendorff_2026_validation.js"

    shutil.copy2(ROOT_DIR / "docs" / "assets" / "twin_bundle.json", bundle_json)
    shutil.copy2(ROOT_DIR / "docs" / "assets" / "twin_bundle.js", bundle_js)
    shutil.copy2(ROOT_DIR / "results_summary.json", summary_json)
    shutil.copy2(ROOT_DIR / "artifacts" / "uhlendorff_2026_validation.json", uhl_json)
    shutil.copy2(ROOT_DIR / "docs" / "assets" / "uhlendorff_2026_validation.js", uhl_js)

    assert check_twin_bundle_artifact(ROOT_DIR, bundle_json, bundle_js) == []
    assert check_results_summary_and_audit(ROOT_DIR, summary_json, bundle_json) == []
    assert check_uhlendorff_validation_artifact(ROOT_DIR, uhl_json, uhl_js) == []

    # 1. Mutate twin_bundle.json without updating bundle_sha256
    orig_bundle = json.loads(bundle_json.read_text(encoding="utf-8"))
    bad_bundle = json.loads(json.dumps(orig_bundle))
    bad_bundle["p_rec"]["mu_max"] = float(bad_bundle["p_rec"]["mu_max"]) + 0.05
    bundle_json.write_text(json.dumps(bad_bundle, indent=2), encoding="utf-8")
    errs_sha = check_twin_bundle_artifact(ROOT_DIR, bundle_json, bundle_js)
    assert any("bundle_sha256 mismatch" in e for e in errs_sha)

    # Restore twin_bundle.json, then drift twin_bundle.js
    bundle_json.write_text(json.dumps(orig_bundle, indent=2), encoding="utf-8")
    drifted_bundle = json.loads(json.dumps(orig_bundle))
    drifted_bundle["schema_version"] = "drifted"
    bundle_js.write_text(
        f"window.HYBRID_TWIN_BUNDLE = {json.dumps(drifted_bundle)};\n", encoding="utf-8"
    )
    errs_js = check_twin_bundle_artifact(ROOT_DIR, bundle_json, bundle_js)
    assert any("drifted" in e for e in errs_js)
    shutil.copy2(ROOT_DIR / "docs" / "assets" / "twin_bundle.js", bundle_js)

    # 2. Mutate results_summary.json alcoa_provenance -> verify_audit_record fails
    orig_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    bad_summary = json.loads(json.dumps(orig_summary))
    bad_summary["alcoa_provenance"]["t_now_h"] = 999.0
    summary_json.write_text(json.dumps(bad_summary, indent=2), encoding="utf-8")
    errs_audit = check_results_summary_and_audit(ROOT_DIR, summary_json, bundle_json)
    assert any("verify_audit_record" in e for e in errs_audit)
    summary_json.write_text(json.dumps(orig_summary, indent=2), encoding="utf-8")

    # 3. Corrupt alcoa_sample.record_sha256 relative to results_summary.json
    cross_bundle = json.loads(json.dumps(orig_bundle))
    cross_bundle["alcoa_sample"]["record_sha256"] = "0" * 64
    no_sha = {k: v for k, v in cross_bundle.items() if k != "bundle_sha256"}
    cross_bundle["bundle_sha256"] = hashlib.sha256(
        json.dumps(no_sha, sort_keys=True).encode("utf-8")
    ).hexdigest()
    bundle_json.write_text(json.dumps(cross_bundle, indent=2), encoding="utf-8")
    errs_cross = check_results_summary_and_audit(ROOT_DIR, summary_json, bundle_json)
    assert any("differs from twin_bundle.alcoa_sample.record_sha256" in e for e in errs_cross)
    shutil.copy2(ROOT_DIR / "docs" / "assets" / "twin_bundle.json", bundle_json)

    # 4. Uhlendorff tampering: legacy unremediated key, missing campaign in per_campaign_test_wmse, JS drift
    orig_uhl = json.loads(uhl_json.read_text(encoding="utf-8"))
    bad_uhl_unrem = json.loads(json.dumps(orig_uhl))
    bad_uhl_unrem["benchmark_comparison"]["unremediated"] = {"pooled_test_wmse": 99.0}
    uhl_json.write_text(json.dumps(bad_uhl_unrem, indent=2), encoding="utf-8")
    errs_unrem = check_uhlendorff_validation_artifact(ROOT_DIR, uhl_json, uhl_js)
    assert any("unremediated" in e for e in errs_unrem)

    bad_uhl_miss = json.loads(json.dumps(orig_uhl))
    first_test_camp = bad_uhl_miss["dataset_metadata"]["test_campaigns"][0]
    bad_uhl_miss["benchmark_comparison"]["remediated"]["per_campaign_test_wmse"].pop(
        first_test_camp
    )
    uhl_json.write_text(json.dumps(bad_uhl_miss, indent=2), encoding="utf-8")
    errs_miss = check_uhlendorff_validation_artifact(ROOT_DIR, uhl_json, uhl_js)
    assert any("per_campaign_test_wmse keys" in e for e in errs_miss)

    uhl_json.write_text(json.dumps(orig_uhl, indent=2), encoding="utf-8")
    bad_uhl_js = json.loads(json.dumps(orig_uhl))
    bad_uhl_js["dataset_metadata"]["source"] = "tampered"
    uhl_js.write_text(
        f"window.UHLENDORFF_2026_VALIDATION = {json.dumps(bad_uhl_js)};\n", encoding="utf-8"
    )
    errs_uhl_js = check_uhlendorff_validation_artifact(ROOT_DIR, uhl_json, uhl_js)
    assert any("drifted" in e for e in errs_uhl_js)
