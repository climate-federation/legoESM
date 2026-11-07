#!/usr/bin/env python3
"""Reconcile the ordinary current-tip ladder with round 79b's artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLANTS = ("none", "candidate-row", "producer", "round80-log", "claim")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(reference_path: Path, candidate_path: Path, round80_log: Path) -> dict:
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    require(reference["status"] == candidate["status"] == "LADDER_MEASURED",
            "one ladder is not measured")
    require(reference["worktree"]["commit"] ==
            "9b27d1b3ad198a275b639018ea4ce007f786e9eb",
            "round-79b producer changed")
    old_rows = reference["candidate_trajectory"]["checkpoints"]
    new_rows = candidate["candidate_trajectory"]["checkpoints"]
    require(len(old_rows) == len(new_rows) == 40, "ladder checkpoint count changed")
    moved = []
    comparisons = 0
    for old, new in zip(old_rows, new_rows, strict=True):
        require((old["kt"], old["checkpoint"]) == (new["kt"], new["checkpoint"]),
                "ladder checkpoint registry changed")
        for field in ("T", "S", "u", "v", "ssh"):
            comparisons += 1
            if old["rows"][field] != new["rows"][field]:
                moved.append({"kt": old["kt"], "checkpoint": old["checkpoint"],
                              "field": field})
    log_text = round80_log.read_text()
    require("ordinary trajectory differs from landed round-79b reference" in log_text,
            "round-80 observer-path refusal is absent")
    kt10 = [row for row in new_rows
            if row["kt"] == 10 and row["checkpoint"] == "stage3"]
    require(len(kt10) == 1, "candidate has no unique kt10 stage3 row")
    return {
        "format": "nemo-testcase-l4-orca2-round81-ladder-reconciliation-v1",
        "claim": "ordinary current-tip ladder reproduces round 79b exactly",
        "claim_label": "given NEMO's entry",
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path),
                      "producer_commit": reference["worktree"]["commit"]},
        "candidate": {"path": str(candidate_path), "sha256": sha256(candidate_path),
                      "producer_commit": candidate["worktree"]["commit"]},
        "round80_observer_refusal": {"path": str(round80_log),
                                     "sha256": sha256(round80_log)},
        "checkpoint_rows_compared": comparisons,
        "moved_rows": moved,
        "first_non_bit_statement_equal": (
            reference["candidate_trajectory"]["first_non_bit_statement"]
            == candidate["candidate_trajectory"]["first_non_bit_statement"]),
        "kt10_stage3_rows": kt10[0]["rows"],
    }


def classify(report: dict, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "candidate-row":
        report["moved_rows"] = [{"kt": 10, "checkpoint": "stage3", "field": "T"}]
    elif plant == "producer":
        report["reference"]["producer_commit"] = "0" * 40
    elif plant == "round80-log":
        report["round80_observer_refusal"]["sha256"] = "0" * 64
    elif plant == "claim":
        report["claim_label"] = "independent"
    require(report["claim_label"] == "given NEMO's entry", "claim label changed")
    require(report["reference"]["producer_commit"] ==
            "9b27d1b3ad198a275b639018ea4ce007f786e9eb", "reference producer changed")
    require(report["round80_observer_refusal"]["sha256"] != "0" * 64,
            "round-80 refusal evidence changed")
    require(report["checkpoint_rows_compared"] == 200, "row census changed")
    require(report["moved_rows"] == [], "ordinary ladder has moved rows")
    require(report["first_non_bit_statement_equal"], "first non-bit statement moved")
    report["status"] = "PASS_ROUND81_LADDER_RECONCILIATION"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--round80-log", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(not any((args.reference, args.candidate, args.round80_log)),
                    "--classify-json cannot be combined with evidence inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "plants classify existing JSON")
            require(all((args.reference, args.candidate, args.round80_log)),
                    "run mode requires all evidence inputs")
            raw = measure(args.reference, args.candidate, args.round80_log)
        result = classify(raw, args.plant)
    except (GateError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND81_LADDER_RECONCILIATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
