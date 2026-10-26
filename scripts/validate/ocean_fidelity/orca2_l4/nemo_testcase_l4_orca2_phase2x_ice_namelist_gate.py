#!/usr/bin/env python3
"""Field-complete ORCA1 ice-deck override gate for the Lane-4 variant."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ASSIGNMENT = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)(?:\s*!.*)?$")
RESOLVED_ASSIGNMENT = re.compile(r"([A-Za-z][A-Za-z0-9_%]*)\s*=\s*(.*?)(?:,\s*)?$")
ALLOWED = {("namini", "nn_iceini_file"): ("1", "0", "ORCA1-grid Ice_initialization is not an ORCA2 input; use NEMO analytic SST initialization")}


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _parse_cfg(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    block: str | None = None
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("&"):
            block = stripped.split()[0][1:].lower()
            continue
        if stripped.startswith("/"):
            block = None
            continue
        if not block or not stripped or stripped.startswith("!"):
            continue
        match = ASSIGNMENT.match(stripped)
        if match:
            key = (block, match.group(1).lower())
            if key in rows:
                raise ValueError(f"duplicate active assignment {key} in {path}")
            rows[key] = {"value": match.group(2).strip(), "line": line_no}
    return rows


def _parse_resolved(path: Path) -> dict[tuple[str, str], str]:
    rows: dict[tuple[str, str], str] = {}
    block: str | None = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("&"):
            block = stripped[1:].lower()
            continue
        if stripped.startswith("/"):
            block = None
            continue
        if not block:
            continue
        match = RESOLVED_ASSIGNMENT.match(stripped)
        if match and "%" not in match.group(1):
            rows[(block, match.group(1).lower())] = match.group(2).strip()
    return rows


def validate(reference: Path, candidate: Path, old_resolved: Path, plant: str | None) -> dict[str, object]:
    expected = _parse_cfg(reference)
    actual = _parse_cfg(candidate)
    resolved = _parse_resolved(old_resolved)
    if plant:
        targets = [key for key in actual if key[1] == plant.lower()]
        if len(targets) != 1:
            raise ValueError(f"plant key is not unique: {plant}")
        actual[targets[0]] = {**actual[targets[0]], "value": ".true." if _norm(str(actual[targets[0]]["value"])) != ".true." else ".false."}

    report: list[dict[str, object]] = []
    failures: list[str] = []
    for key, source in expected.items():
        if key not in actual:
            failures.append(f"missing {key[0]}.{key[1]}")
            continue
        source_value = str(source["value"])
        candidate_value = str(actual[key]["value"])
        if _norm(source_value) == _norm(candidate_value):
            status, reason = "EXACT_ORCA1_OVERRIDE", None
        elif key in ALLOWED and (_norm(source_value), _norm(candidate_value)) == ALLOWED[key][:2]:
            status, reason = "DELIBERATE_ORCA2_INPUT_DIFFERENCE", ALLOWED[key][2]
        else:
            status, reason = "UNREGISTERED_DIFFERENCE", None
            failures.append(f"unregistered difference {key[0]}.{key[1]}: {source_value} -> {candidate_value}")
        report.append({
            "block": key[0], "field": key[1], "orca1_value": source_value,
            "candidate_value": candidate_value, "old_variant_resolved": resolved.get(key),
            "source_line": source["line"], "candidate_line": actual[key]["line"],
            "status": status, "reason": reason,
        })
    extras = sorted(set(actual) - set(expected))
    if extras:
        failures.extend(f"candidate-only assignment {block}.{field}" for block, field in extras)
    if failures:
        raise ValueError("; ".join(failures))
    deliberate = [row for row in report if row["status"] == "DELIBERATE_ORCA2_INPUT_DIFFERENCE"]
    return {
        "reference": str(reference), "candidate": str(candidate),
        "old_resolved": str(old_resolved), "orca1_assignment_count": len(expected),
        "exact_orca1_rows": len(report) - len(deliberate),
        "deliberate_difference_count": len(deliberate),
        "candidate_only_count": len(extras), "rows": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--old-resolved", type=Path, required=True)
    parser.add_argument("--plant")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = validate(args.reference, args.candidate, args.old_resolved, args.plant)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
