#!/usr/bin/env python3
"""Classify round 268's G1 transfer and matched EEN boundary."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

FLOOR = 2.0e-10
LABELS = ("independent", "given_nemo_entry")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate(report: dict) -> dict:
    return report.get("candidate", report)


def classify(before_paths, transfer_paths, entry_paths, curl_path, plant=False):
    before = {report["label"]: report for report in map(load, before_paths)}
    transfer = {report["label"]: report for report in map(load, transfer_paths)}
    entry = {report["label"]: report for report in map(load, entry_paths)}
    require(set(before) == set(transfer) == set(entry) == set(LABELS),
            "both labels are required in every lane")
    entry = copy.deepcopy(entry)
    if plant:
        candidate(entry["independent"])["matched_stage1"]["eta"]["max_abs"] = 1.0

    rows = {}
    for label in LABELS:
        old = candidate(before[label])["boundary_rows"]["after_vorticity_u"]["complete"]
        new = candidate(transfer[label])["boundary_rows"]["after_vorticity_u"]["complete"]
        carried = candidate(entry[label])["matched_stage1"]
        require(all(row["max_abs"] <= FLOOR for row in carried.values()),
                f"{label}: matched stage-1 entry left the floor")
        require(new["max_abs"] < old["max_abs"],
                f"{label}: G1 transfer did not improve vorticity maximum")
        require(new["max_abs"] > FLOOR,
                f"{label}: preregistered closure unexpectedly changed")
        rows[label] = {
            "before_vorticity": old,
            "transferred_vorticity": new,
            "matched_stage1": carried,
        }

    curl = candidate(load(curl_path))["boundary_rows"]["after_vorticity_u"]
    require(curl["complete"]["max_abs"] == rows["independent"]["transferred_vorticity"]["max_abs"],
            "literal relative-curl arm moved the headline maximum")
    require(curl["fold_band"]["max_abs"] > candidate(transfer["independent"])["boundary_rows"]["after_vorticity_u"]["fold_band"]["max_abs"],
            "relative-curl refutation no longer worsens the fold row")
    return {
        "format": "nemo-testcase-l4-orca2-round268-g1-transfer-v1",
        "status": "PASS_R268_HELD_EEN_STATEMENT",
        "labels": rows,
        "relative_curl_arm": curl,
        "first_non_floor_statement": "stage-2 EEN vorticity accumulator",
        "predictions": {
            "R268-P1": "CONFIRMED_TRANSFER_EXECUTED",
            "R268-P2": "REFUTED_NOT_AT_FLOOR",
            "R268-P3": "CONFIRMED_EEN_CREATES_BOUNDARY",
            "R268-P4": "REFUTED_RELATIVE_CURL_NULL",
            "R268-P5": "NOT_REACHED_NO_LANDING",
            "R268-P6": "CONFIRMED" if not plant else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs=2, type=Path, required=True)
    parser.add_argument("--transfer", nargs=2, type=Path, required=True)
    parser.add_argument("--entry", nargs=2, type=Path, required=True)
    parser.add_argument("--curl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = classify(args.before, args.transfer, args.entry, args.curl, args.plant)
        require(not args.plant, "matched-entry plant stayed green")
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
