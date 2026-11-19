"""Automated GxP Requirements Traceability Matrix (RTM) generator and AST link verifier."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = ROOT_DIR / "docs" / "validation" / "requirements_manifest.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "docs" / "validation" / "RTM.md"


def _collect_ast_symbols(py_path: Path) -> Set[str]:
    """Collect module-level symbols and Class.method names via Python AST."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    symbols: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(f"{node.name}.{item.name}")
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    symbols.add(f"{node.name}.{item.target.id}")
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name):
                            symbols.add(f"{node.name}.{tgt.id}")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    symbols.add(tgt.id)
    return symbols


def _collect_test_functions(py_path: Path) -> Set[str]:
    """Collect pytest test_* function names in a test module via Python AST."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.add(node.name)
    return names


def render_rtm_markdown(manifest: Dict[str, Any]) -> str:
    """Render deterministic Markdown RTM table from a verified requirements manifest."""
    reqs: List[Dict[str, Any]] = list(manifest.get("requirements", []))
    total_src_refs = sum(len(r.get("source_refs", [])) for r in reqs)
    total_test_refs = sum(len(r.get("test_refs", [])) for r in reqs)

    by_subsystem: Dict[str, Dict[str, int]] = {}
    for req in reqs:
        sub = str(req["subsystem"])
        tier = str(req["risk_tier"])
        bucket = by_subsystem.setdefault(sub, {"count": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0})
        bucket["count"] += 1
        if tier in bucket:
            bucket[tier] += 1

    lines: List[str] = [
        "# Automated GxP Requirements Traceability Matrix (RTM)",
        "",
        f"- **System:** `{manifest.get('system_name', 'hybrid-fedbatch-twin')}`",
        f"- **Schema Version:** `{manifest.get('schema_version', 'rtm_v1')}`",
        f"- **Traceability Source:** `docs/validation/requirements_manifest.json`",
        f"- **Generator / Verifier:** `scripts/generate_rtm.py` (AST-verified symbols & test links)",
        f"- **Total Requirements:** {len(reqs)}",
        f"- **Verified Source Modules Referenced:** {total_src_refs}",
        f"- **Verified Automated Test Links:** {total_test_refs}",
        "- **Orphaned Requirements:** 0",
        "- **Broken AST / Test Links:** 0",
        "",
        "## Subsystem Risk & Assurance Summary",
        "",
        "| Subsystem | Requirements | HIGH Risk | MEDIUM Risk | LOW Risk |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for sub in sorted(by_subsystem.keys()):
        b = by_subsystem[sub]
        lines.append(f"| {sub} | {b['count']} | {b['HIGH']} | {b['MEDIUM']} | {b['LOW']} |")

    lines.extend(
        [
            "",
            "## Full Requirements Traceability Matrix",
            "",
            "| Req ID | Subsystem | Risk Tier | CSA Assurance | Description | Verified Code Symbols | Verified Test Cases |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for req in reqs:
        src_parts = []
        for sr in req["source_refs"]:
            syms = ", ".join(f"`{s}`" for s in sr.get("symbols", []))
            src_parts.append(f"`{sr['path']}` ({syms})" if syms else f"`{sr['path']}`")
        src_cell = "<br>".join(src_parts)
        test_cell = "<br>".join(f"`{t}`" for t in req["test_refs"])
        desc = str(req["description"]).replace("|", "\\|")
        lines.append(
            f"| `{req['req_id']}` | {req['subsystem']} | {req['risk_tier']} | {req['csa_assurance_level']} | {desc} | {src_cell} | {test_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def validate_rtm(
    root_dir: Path = ROOT_DIR,
    manifest_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    check_rendered_drift: bool = False,
) -> Tuple[bool, List[str], str]:
    """Validate requirements manifest AST symbols/test links and optionally check RTM.md drift."""
    m_path = manifest_path or (root_dir / "docs" / "validation" / "requirements_manifest.json")
    o_path = output_path or (root_dir / "docs" / "validation" / "RTM.md")
    errors: List[str] = []

    if not m_path.exists():
        return False, [f"Missing requirements manifest: {m_path}"], ""

    try:
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"Failed to parse manifest JSON ({m_path}): {exc}"], ""

    reqs = manifest.get("requirements")
    if not isinstance(reqs, list) or not reqs:
        return False, ["Manifest must contain a non-empty 'requirements' list."], ""

    symbol_cache: Dict[Path, Set[str]] = {}
    test_cache: Dict[Path, Set[str]] = {}
    seen_ids: Set[str] = set()

    for idx, req in enumerate(reqs):
        if not isinstance(req, dict):
            errors.append(f"Requirement[{idx}] must be an object.")
            continue
        req_id = str(req.get("req_id", "")).strip()
        if not req_id:
            errors.append(f"Requirement[{idx}] missing req_id.")
            continue
        if req_id in seen_ids:
            errors.append(f"Duplicate req_id: {req_id}")
        seen_ids.add(req_id)

        for field in ("subsystem", "risk_tier", "csa_assurance_level", "description"):
            if not str(req.get(field, "")).strip():
                errors.append(f"{req_id}: missing required field {field!r}.")

        src_refs = req.get("source_refs")
        if not isinstance(src_refs, list) or not src_refs:
            errors.append(f"{req_id}: orphaned requirement with zero source_refs.")
        else:
            for sref in src_refs:
                rel = str(sref.get("path", "")).strip()
                fpath = root_dir / rel
                if not rel or not fpath.exists():
                    errors.append(f"{req_id}: missing source file {rel!r}.")
                    continue
                syms = sref.get("symbols") or []
                if not syms:
                    errors.append(f"{req_id}: source_ref {rel!r} must declare at least one symbol.")
                    continue
                if fpath.suffix == ".py":
                    if fpath not in symbol_cache:
                        symbol_cache[fpath] = _collect_ast_symbols(fpath)
                    avail = symbol_cache[fpath]
                    for sym in syms:
                        if sym not in avail:
                            errors.append(f"{req_id}: broken symbol link {sym!r} in {rel}.")

        test_refs = req.get("test_refs")
        if not isinstance(test_refs, list) or not test_refs:
            errors.append(f"{req_id}: orphaned requirement with zero test_refs.")
        else:
            for tref in test_refs:
                if "::" not in str(tref):
                    errors.append(f"{req_id}: invalid test_ref format {tref!r} (expected path::test_name).")
                    continue
                t_rel, t_name = str(tref).split("::", 1)
                t_path = root_dir / t_rel
                if not t_path.exists():
                    errors.append(f"{req_id}: missing test file {t_rel!r}.")
                    continue
                if t_path not in test_cache:
                    test_cache[t_path] = _collect_test_functions(t_path)
                if t_name not in test_cache[t_path]:
                    errors.append(f"{req_id}: missing test function {t_name!r} in {t_rel}.")

    rendered = render_rtm_markdown(manifest) if not errors else ""
    if check_rendered_drift and not errors:
        if not o_path.exists():
            errors.append(f"Generated RTM file does not exist: {o_path}")
        else:
            existing = o_path.read_text(encoding="utf-8")
            if existing != rendered:
                errors.append(
                    f"RTM drift detected: {o_path} differs from deterministic output of {m_path.name}."
                )

    return len(errors) == 0, errors, rendered


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and verify GxP Requirements Traceability Matrix.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all AST/test links and fail if RTM.md is missing or out of date.",
    )
    args = parser.parse_args(argv)

    ok, errors, rendered = validate_rtm(
        root_dir=ROOT_DIR,
        manifest_path=args.manifest,
        output_path=args.output,
        check_rendered_drift=args.check,
    )
    if not ok:
        print(f"[FAIL] RTM verification failed with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    if args.check:
        print(f"[PASS] RTM verified ({args.output})")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"[PASS] Generated RTM at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
