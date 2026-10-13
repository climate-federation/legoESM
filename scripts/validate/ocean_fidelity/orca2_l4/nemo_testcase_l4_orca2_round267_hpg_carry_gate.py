#!/usr/bin/env python3
"""Gate OMT-4's source-exact stage-2 HPG accumulator carry."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round266_rhs_operator_gate as r266,
)

LABELS = r266.LABELS
FLOOR = r266.FLOOR
NEXT_ORDER = ("after_vorticity_u", "after_keg_u", "after_advection_u")
PLANTS = (
    "none", "label-coverage", "record-identity", "baseline-reproduction",
    "passivity", "source-order", "hpg-carry", "next-boundary",
    "completed-rhs", "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-267 predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            label: str, expect_commit: str) -> dict[str, object]:
    baseline = r266.measure(
        deck_root, frame_root, record_root, label, expect_commit,
        literal_hpg_accumulator=False)
    candidate = r266.measure(
        deck_root, frame_root, record_root, label, expect_commit,
        literal_hpg_accumulator=True)
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round267-hpg-carry-v1",
        "status": "PASS_R267_MEASUREMENT",
        "label": label,
        "baseline": baseline,
        "candidate": candidate,
        "one_bit_control": r266._score(one, np.nextafter(one, np.inf)),
    }


def _first_next(report: dict[str, object]) -> str | None:
    rows = report["accumulator_self_replay"]
    return next((name for name in NEXT_ORDER if rows[name]["max_abs"] > FLOOR), None)


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both labels are required")
    independent = by_label["independent"]
    if plant == "record-identity":
        independent["candidate"]["record_identity"] = {"record": "0" * 64}
    elif plant == "baseline-reproduction":
        independent["baseline"]["accumulator_self_replay"]["after_hpg_u"]["max_abs"] = 0.0
    elif plant == "passivity":
        independent["candidate"]["passivity"][next(iter(independent["candidate"]["passivity"]))] = False
    elif plant == "source-order":
        independent["candidate"]["boundary_order"][1:3] = reversed(
            independent["candidate"]["boundary_order"][1:3])
    elif plant == "hpg-carry":
        independent["candidate"]["boundary_rows"]["after_hpg_u"]["complete"]["max_abs"] = 1.0
    elif plant == "next-boundary":
        independent["candidate"]["accumulator_self_replay"]["after_keg_u"]["max_abs"] = 0.0
    elif plant == "completed-rhs":
        independent["candidate"]["boundary_rows"]["after_advection_u"]["complete"]["max_abs"] = 0.0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    identity = by_label[LABELS[0]]["baseline"]["record_identity"]
    next_rows = {}
    summaries = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R267_MEASUREMENT", f"{label}: status moved")
        baseline = report["baseline"]
        candidate = report["candidate"]
        require(baseline["literal_hpg_accumulator"] is False,
                f"{label}: baseline arm moved")
        require(candidate["literal_hpg_accumulator"] is True,
                f"{label}: candidate arm moved")
        require(baseline["record_identity"] == identity
                and candidate["record_identity"] == identity,
                f"{label}: record identity moved")
        require(all(baseline["passivity"].values())
                and all(candidate["passivity"].values()),
                f"{label}: trace is not passive")
        require(candidate["boundary_order"] == list(r266.BOUNDARY_ORDER),
                f"{label}: source order moved")
        require(all(candidate["matched_stage2"][name]["max_abs"] <= FLOOR
                    for name in ("T", "S")),
                f"{label}: matched stage-2 state moved")
        require(baseline["accumulator_self_replay"]["after_hpg_u"]["max_abs"]
                > FLOOR, f"{label}: round-266 baseline did not reproduce")
        hpg_complete = candidate["boundary_rows"]["after_hpg_u"]["complete"]
        hpg_carry = candidate["accumulator_self_replay"]["after_hpg_u"]
        require(hpg_complete["max_abs"] <= FLOOR and hpg_carry["max_abs"] <= FLOOR,
                f"{label}: literal HPG carry did not close")
        first = _first_next(candidate)
        require(first is not None, f"{label}: no registered next boundary")
        next_rows[label] = first
        completed = candidate["boundary_rows"]["after_advection_u"]["complete"]
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summaries[label] = {
            "baseline_hpg": baseline["boundary_rows"]["after_hpg_u"]["complete"],
            "candidate_hpg": hpg_complete,
            "first_next_boundary": first,
            "next_self_replay": candidate["accumulator_self_replay"][first],
            "completed_rhs": completed,
            "matched_stage2": candidate["matched_stage2"],
        }
    require(len(set(next_rows.values())) == 1, "labels disagree on next boundary")
    first = next_rows[LABELS[0]]
    completed_flags = {
        label: bool(summaries[label]["completed_rhs"]["max_abs"] <= FLOOR)
        for label in LABELS
    }
    require(len(set(completed_flags.values())) == 1,
            "labels disagree on completed-RHS closure")
    completed_closes = completed_flags[LABELS[0]]
    return {
        "format": "nemo-testcase-l4-orca2-round267-hpg-classification-v1",
        "status": ("PASS_R267_COMPLETED_RHS" if completed_closes
                   else "PASS_R267_HPG_CARRY_NEXT_BOUNDARY"),
        "first_next_boundary": first,
        "completed_rhs_closes": completed_closes,
        "labels": summaries,
        "predictions": {
            "R267-P1": "CONFIRMED",
            "R267-P2": "CONFIRMED",
            "R267-P3": "CONFIRMED_KEG" if first == "after_keg_u" else "REFUTED",
            "R267-P4": "READY_FOR_TRAJECTORY" if completed_closes else "HELD_NEXT_BOUNDARY",
            "R267-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure(args.deck_root, args.frame_root, args.record_root,
                             args.label, args.expect_commit)
        else:
            require(args.classify is not None, "classification reports required")
            result = classify(
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.classify], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, r266.GateError, OSError, KeyError,
            TypeError, ValueError, IndexError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
