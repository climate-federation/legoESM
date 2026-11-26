#!/usr/bin/env python3
"""Classify round 271's private written-order EEN transfer measurement."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

FLOOR = 2.0e-10
LABELS = ("independent", "given_nemo_entry")


class GateError(RuntimeError):
    """A frozen round-271 measurement predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(before_reports: list[dict], candidate_reports: list[dict],
             *, plant: bool = False) -> dict:
    before = {report["label"]: report for report in before_reports}
    candidate = {report["label"]: report for report in candidate_reports}
    require(set(before) == set(candidate) == set(LABELS),
            "both labels are required in both arms")
    candidate = copy.deepcopy(candidate)
    if plant:
        candidate["independent"]["boundary_rows"]["after_vorticity_u"][
            "complete"]["max_abs"] = 0.0

    rows = {}
    for label in LABELS:
        old = before[label]["boundary_rows"]["after_vorticity_u"]["complete"]
        new = candidate[label]["boundary_rows"]["after_vorticity_u"]["complete"]
        self_replay = candidate[label]["accumulator_self_replay"][
            "after_vorticity_u"]
        entry = candidate[label]["matched_stage1"]
        require(candidate[label]["status"] == "PASS_R266_MEASUREMENT",
                f"{label}: passive measurement status moved")
        require(all(row["max_abs"] <= FLOOR for row in entry.values()),
                f"{label}: matched stage-1 entry left the floor")
        require(self_replay["max_abs"] <= FLOOR,
                f"{label}: vorticity accumulator no longer self-replays")
        require(new["max_abs"] <= old["max_abs"],
                f"{label}: written-order pair moved the maximum away")
        require(new["max_abs"] > FLOOR,
                f"{label}: residual unexpectedly closed; this HELD gate is stale")
        require(new["argmax"] == [19, 33, 3],
                f"{label}: interior argmax moved")
        rows[label] = {
            "before": old,
            "candidate": new,
            "max_delta": float(new["max_abs"] - old["max_abs"]),
            "accumulator_self_replay": self_replay,
            "matched_stage1": entry,
        }

    require(rows["independent"]["candidate"] ==
            rows["given_nemo_entry"]["candidate"],
            "labels disagree at the matched stage-2 boundary")
    return {
        "format": "nemo-testcase-l4-orca2-round271-een-transfer-v1",
        "status": "PASS_R271_HELD_WRITTEN_ORDER_NOT_CLOSED",
        "floor": FLOOR,
        "labels": rows,
        "first_surviving_statement":
            "combined Coriolis-plus-relative-vorticity EEN statement",
        "next_measurement":
            "ff_f, relative circulation, r1_e1e2f, live e3f, transports, triads",
        "predictions": {
            "R271-P1": "CONFIRMED_PRIVATE_IMPORT",
            "R271-P2": "REFUTED_RESIDUAL_ABOVE_FLOOR",
            "R271-P3": "NOT_REACHED_NO_CLOSURE",
            "R271-P4": "NOT_REACHED_NO_LANDING",
            "R271-P5": "CONFIRMED" if not plant else "PLANT",
            "R271-P6": "CONFIRMED_BRANCH_LANDED_AFTER_MEASUREMENT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs=2, required=True, type=Path)
    parser.add_argument("--candidate", nargs=2, required=True, type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(
            [load(path) for path in args.before],
            [load(path) for path in args.candidate], plant=args.plant)
        require(not args.plant, "closure plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"REFUSE: {error}")
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
