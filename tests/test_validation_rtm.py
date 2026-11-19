"""Validation tests for the automated GxP RTM, AST/test links, and validation documentation integrity."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from scripts.generate_rtm import ROOT_DIR, validate_rtm

pytestmark = pytest.mark.validation

_FILE_REF_RE = re.compile(
    r"`((?:hybrid_fedbatch_twin|api|mcp_server|agents|mlops|scripts|tests|docs|artifacts)/[A-Za-z0-9_./-]+\.(?:py|json|md|yml)|(?:app\.py|stress_test_uhlendorff\.py|results_summary\.json))`"
)


def test_rtm_generation_check_and_tamper_detection(tmp_path: Path) -> None:
    """Verify RTM AST/test links, zero drift via CLI --check, and explicit tamper/orphan rejection."""
    ok, errs, _ = validate_rtm(ROOT_DIR, check_rendered_drift=True)
    assert ok is True and errs == [], f"RTM validation failed: {errs}"

    proc = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "generate_rtm.py"), "--check"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"generate_rtm.py --check failed:\n{proc.stdout}\n{proc.stderr}"
    assert "[PASS] RTM verified" in proc.stdout

    # Negative tampering checks against temporary copies
    manifest_src = ROOT_DIR / "docs" / "validation" / "requirements_manifest.json"
    rtm_src = ROOT_DIR / "docs" / "validation" / "RTM.md"
    orig_manifest = json.loads(manifest_src.read_text(encoding="utf-8"))

    # 1. Broken test link
    bad_test = json.loads(json.dumps(orig_manifest))
    bad_test["requirements"][0]["test_refs"] = ["tests/test_hybrid_twin.py::test_nonexistent_xyz"]
    p_bad_test = tmp_path / "m_bad_test.json"
    p_bad_test.write_text(json.dumps(bad_test), encoding="utf-8")
    ok_t, errs_t, _ = validate_rtm(ROOT_DIR, manifest_path=p_bad_test)
    assert ok_t is False and any("missing test function" in e for e in errs_t)

    # 2. Broken AST source symbol
    bad_sym = json.loads(json.dumps(orig_manifest))
    bad_sym["requirements"][0]["source_refs"][0]["symbols"].append("missing_symbol_abc")
    p_bad_sym = tmp_path / "m_bad_sym.json"
    p_bad_sym.write_text(json.dumps(bad_sym), encoding="utf-8")
    ok_s, errs_s, _ = validate_rtm(ROOT_DIR, manifest_path=p_bad_sym)
    assert ok_s is False and any("broken symbol link" in e for e in errs_s)

    # 3. Orphaned requirement (zero test_refs)
    bad_orphan = json.loads(json.dumps(orig_manifest))
    bad_orphan["requirements"][0]["test_refs"] = []
    p_bad_orphan = tmp_path / "m_bad_orphan.json"
    p_bad_orphan.write_text(json.dumps(bad_orphan), encoding="utf-8")
    ok_o, errs_o, _ = validate_rtm(ROOT_DIR, manifest_path=p_bad_orphan)
    assert ok_o is False and any("orphaned requirement" in e for e in errs_o)

    # 4. Stale / drifted RTM.md
    drifted_rtm = tmp_path / "RTM_drifted.md"
    shutil.copy2(rtm_src, drifted_rtm)
    drifted_rtm.write_text(drifted_rtm.read_text(encoding="utf-8") + "\n<!-- drifted -->\n", encoding="utf-8")
    ok_d, errs_d, _ = validate_rtm(ROOT_DIR, output_path=drifted_rtm, check_rendered_drift=True)
    assert ok_d is False and any("drift" in e.lower() for e in errs_d)


def test_validation_docs_citations_and_honest_maturity() -> None:
    """Verify every validation doc and cited file path exists on disk with honest maturity framing."""
    val_dir = ROOT_DIR / "docs" / "validation"
    expected_docs = (
        "requirements_manifest.json",
        "RTM.md",
        "FDA_CSA_GAMP5_RISK_GRID.md",
        "FDA_MODEL_CREDIBILITY_7STEP.md",
        "ICH_Q14_Q12_LIFECYCLE_GOVERNANCE.md",
        "OPEN_ASSUMPTIONS_AND_GAPS.md",
    )
    for name in expected_docs:
        assert (val_dir / name).exists(), f"Missing validation artifact: docs/validation/{name}"

    prohibited = (
        "Portela Level " + "3+",
        "Portela Level " + "3-Ready",
        "Portela Level " + "3 Digital Twin",
        "9.2" + " / 10",
    )

    for md_path in sorted(val_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for phrase in prohibited:
            assert phrase not in text, f"Prohibited maturity claim {phrase!r} found in {md_path.name}"
        for match in _FILE_REF_RE.findall(text):
            assert (ROOT_DIR / match).exists(), f"Cited path {match!r} in {md_path.name} does not exist."

    cred_text = (val_dir / "FDA_MODEL_CREDIBILITY_7STEP.md").read_text(encoding="utf-8")
    for expected_fact in ("0.7320", "8.61 / 10.00 -> GO (PASS)", "38.18", "4.39%", "ddof=1", "ftol=1e-12"):
        assert expected_fact in cred_text
