#!/usr/bin/env python3
"""Decision-38 Rule-12 gate for the Round-85 GYRE momentum bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from legoesm.ocean.fidelity.ulp_move_gate import (
    certified_rows,
    compare_gate_reports,
    load_residual_artifact,
    plant_at_bar_to_debt,
)

TARGETS = ("GYRE-zco.kt2.before.u", "GYRE-zco.kt2.before.v")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plant_target_worse(fields: dict[str, dict[str, np.ndarray]]) -> None:
    payload = fields[TARGETS[0]]
    index = int(np.argmax(payload["residual"]))
    old = np.float64(payload["residual"][index])
    step = max(old, np.spacing(np.float64(1.0)))
    direction = np.float64(1.0 if payload["candidate"][index] >= payload["oracle"][index]
                           else -1.0)
    payload["candidate"][index] = payload["oracle"][index] + direction * (old + step)
    payload["residual"][index] = abs(
        payload["candidate"][index] - payload["oracle"][index])


def evaluate(
    reference: dict,
    candidate: dict,
    reference_fields: dict[str, dict[str, np.ndarray]],
    candidate_fields: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Apply Decision 38 while retaining the canonical structural checks."""
    ordinary = compare_gate_reports(
        reference,
        candidate,
        reference_fields=reference_fields,
        candidate_fields=candidate_fields,
    )
    # Decision 38 supersedes only the old two-ULP rejection of downstream
    # worsening. Every other canonical violation remains binding.
    violations = [
        item for item in ordinary["violations"]
        if "worsened against NEMO by" not in item
    ]
    before_rows = certified_rows(reference)
    after_rows = certified_rows(candidate)
    target_rows = []
    for name in TARGETS:
        if name not in before_rows or name not in after_rows:
            violations.append(f"missing registered target row {name}")
            continue
        if name not in reference_fields or name not in candidate_fields:
            violations.append(f"missing target residual payload {name}")
            continue
        old = float(np.max(reference_fields[name]["residual"], initial=0.0))
        new = float(np.max(candidate_fields[name]["residual"], initial=0.0))
        moved = not np.array_equal(
            reference_fields[name]["candidate"], candidate_fields[name]["candidate"])
        improved = new < old
        target_rows.append({
            "row": name,
            "reference_max_residual": old,
            "candidate_max_residual": new,
            "field_moved": bool(moved),
            "moved_toward_bar": bool(improved),
        })
        if not moved or not improved:
            violations.append(
                f"{name}: must move at field level toward the bar; "
                f"moved={moved}, residual={old:.17e}->{new:.17e}")

    moved_rows = [
        row for row in ordinary["field_moves"]
        if row["max_previous_legoesm_field_move"] > 0.0
    ]
    return {
        "format": "gyre-round85-decision38-bundle-gate-v1",
        "worktree": worktree_stamp(),
        "status": "PASS" if not violations else "FAIL",
        "criterion": "decision38_bundle",
        "decision38_relaxation": (
            "downstream worsening is registered rather than rejected only when "
            "both first-over-bar kt2 momentum rows move toward the bar"
        ),
        "targets": target_rows,
        "first_over_bar_reference": ordinary["first_over_bar_reference"],
        "first_over_bar_candidate": ordinary["first_over_bar_candidate"],
        "row_status_changes": ordinary["row_status_changes"],
        "moved_rows": moved_rows,
        "n_moved_rows": len(moved_rows),
        "largest_oracle_residual_worsening_ulps": (
            ordinary["largest_oracle_residual_worsening_ulps"]),
        "canonical_two_ulp_status": ordinary["status"],
        "canonical_two_ulp_violations": ordinary["violations"],
        "violations": violations,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plant", choices=("target-u-worse", "at-bar-to-debt", "earlier-first-over-bar"))
    args = parser.parse_args(argv)

    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    reference_fields = load_residual_artifact(reference, args.reference)
    candidate_fields = load_residual_artifact(candidate, args.candidate)
    candidate = copy.deepcopy(candidate)
    candidate_fields = {
        name: {key: np.array(value, copy=True) for key, value in payload.items()}
        for name, payload in candidate_fields.items()
    }
    if args.plant == "target-u-worse":
        _plant_target_worse(candidate_fields)
    elif args.plant == "at-bar-to-debt":
        plant_at_bar_to_debt(candidate)
    elif args.plant == "earlier-first-over-bar":
        candidate["first_over_bar"] = {"kt": 1, "fields": ["u"]}

    result = evaluate(reference, candidate, reference_fields, candidate_fields)
    result.update({
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_sha256": _sha256(args.reference),
        "candidate_sha256": _sha256(args.candidate),
        "plant": args.plant,
        "worktree": worktree_stamp(),
    })
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"ROUND85_DECISION38_BUNDLE {result['status']}: "
        f"targets={[(r['reference_max_residual'], r['candidate_max_residual']) for r in result['targets']]} "
        f"moved_rows={result['n_moved_rows']} plant={args.plant!r}")
    expected = "PASS" if args.plant is None else "FAIL"
    if result["status"] != expected:
        return 2
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
