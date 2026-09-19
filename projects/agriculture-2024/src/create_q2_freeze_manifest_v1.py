#!/usr/bin/env python3
"""Create or verify the Human-Gate-approved immutable Q2 baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from q2_validate import verify_q1_freeze  # noqa: E402


MANIFEST_PATH = PROJECT_ROOT / "outputs" / "q2_freeze_manifest_v1.json"
RESEARCH_PATH = PROJECT_ROOT / "research" / "q2_freeze_manifest_v1.md"
FIXED_PATHS = (
    "outputs/result2.xlsx",
    "outputs/q1_freeze_manifest_v2.json",
    "src/q2_config.py",
    "src/q2_model.py",
    "src/q2_outputs.py",
    "src/q2_uncertainty.py",
    "src/q2_validate.py",
    "src/fill_q2_workbook.py",
    "scripts/run_q2.py",
    "scripts/run_q2_deadline_closure.py",
    "scripts/verify_q2_delivery.py",
    "scripts/inspect_result2_template.mjs",
    "research/q2_mathematical_formulation.md",
    "research/q2_model_design_v1.md",
    "research/q2_model_review.md",
    "research/q2_modeling_decisions_v1.md",
    "research/q2_open_decisions.md",
    "research/q2_uncertainty_map.md",
    "research/q2_validation.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def authoritative_paths() -> list[str]:
    q2_files = [
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in sorted((PROJECT_ROOT / "outputs" / "q2").rglob("*"))
        if path.is_file()
    ]
    return sorted(set(q2_files).union(FIXED_PATHS))


def create_manifest() -> dict[str, object]:
    q1 = verify_q1_freeze(PROJECT_ROOT)
    if not q1["passed"] or q1["checked"] != 25:
        raise RuntimeError(f"Q1 v2 freeze gate failed: {q1}")
    hashes = {}
    for relative in authoritative_paths():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        hashes[relative] = sha256(path)
    payload: dict[str, object] = {
        "schema_version": "q2-freeze-manifest-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "human_gate_status": "Q2 APPROVED FOR FREEZE",
        "immutable_scope_rule": (
            "All listed bytes are immutable during Q3 unless a new Critical "
            "validation failure is discovered and Human Gate authorizes repair."
        ),
        "file_count": len(hashes),
        "hashes_sha256": hashes,
        "q1_freeze_at_creation": q1,
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Q2冻结清单 v1",
        "",
        f"- 冻结文件数：{len(hashes)}",
        f"- 机器清单：`outputs/q2_freeze_manifest_v1.json`",
        "- Human Gate：Q2 APPROVED FOR FREEZE",
        "- Q1 v2冻结：25/25 PASS",
        "- 规则：Q3期间不得修改任何列入清单的字节，除非发现新的Critical验证失败并获得新的Human Gate授权。",
        "",
        "## SHA-256",
        "",
        "| 路径 | SHA-256 |",
        "|---|---|",
    ]
    lines.extend(f"| `{path}` | `{digest}` |" for path, digest in hashes.items())
    RESEARCH_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def verify_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in payload["hashes_sha256"].items():
        path = PROJECT_ROOT / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    q1 = verify_q1_freeze(PROJECT_ROOT)
    return {
        "passed": not failures and q1["passed"] and q1["checked"] == 25,
        "checked": len(payload["hashes_sha256"]),
        "failures": failures,
        "q1_freeze": q1,
        "manifest_sha256": sha256(MANIFEST_PATH),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    result = create_manifest() if args.create else verify_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.create and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
