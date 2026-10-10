#!/usr/bin/env python3
"""Apply Decision 96 to the round-215 OMT-1 atomic vector pair."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


class GateError(RuntimeError):
    """The registered atomic pair is incomplete or ineligible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _rows(document: dict) -> tuple[list[dict], object]:
    for label in ("independent", "given_nemo_entry"):
        if label in document:
            document = document[label]
            break
    return document["rows"], document.get("first_non_bit_checkpoint")


def _comparison(base: dict, candidate: dict) -> dict:
    old_rows, old_first = _rows(base)
    new_rows, new_first = _rows(candidate)
    require(len(old_rows) == 160 and len(new_rows) == 160,
            "OMT-1 ladder must contain exactly 160 rows")
    key = lambda row: (row["kt"], row["checkpoint"], row["field"])
    old = {key(row): row for row in old_rows}
    new = {key(row): row for row in new_rows}
    require(old.keys() == new.keys(), "candidate row key set moved")
    require(old_first == new_first, "first non-bit checkpoint moved")
    moved, losses = [], []
    for row_key in sorted(old):
        before, after = old[row_key], new[row_key]
        if before == after:
            continue
        if before["bit_identical"] and not after["bit_identical"]:
            losses.append(row_key)
        metrics = {}
        for name in ("rms", "max_abs", "mean_abs_over_unequal"):
            delta = float(after[name]) - float(before[name])
            metrics[name] = {
                "before": before[name], "after": after[name], "delta": delta,
                "direction": "toward" if delta < 0 else "away" if delta > 0 else "equal",
            }
        moved.append({
            "kt": row_key[0], "checkpoint": row_key[1], "field": row_key[2],
            "bit_identical_before": before["bit_identical"],
            "bit_identical_after": after["bit_identical"],
            "unequal_before": before["unequal"],
            "unequal_after": after["unequal"],
            "first_unequal_index_before": before["first_unequal_index"],
            "first_unequal_index_after": after["first_unequal_index"],
            "metrics": metrics,
        })
    require(not losses, f"previously exact rows left the bar: {losses}")
    return {
        "row_count": len(old), "moved_row_count": len(moved),
        "unchanged_row_count": len(old) - len(moved),
        "first_non_bit_before": old_first, "first_non_bit_after": new_first,
        "bit_identical_losses": losses, "moved_rows": moved,
    }


def _row(document: dict, target: tuple[int, str, str]) -> dict:
    rows, _ = _rows(document)
    for row in rows:
        if (row["kt"], row["checkpoint"], row["field"]) == target:
            return row
    raise GateError(f"missing ladder row {target}")


def _direction(before: dict, after: dict, metric: str) -> str:
    delta = float(after[metric]) - float(before[metric])
    return "toward" if delta < 0 else "away" if delta > 0 else "equal"


def classify(base: dict, candidate: dict, pair: dict, *, plant: str = "none") -> dict:
    base = copy.deepcopy(base)
    candidate = copy.deepcopy(candidate)
    pair = copy.deepcopy(pair)
    if plant == "pair-closure":
        pair["replays"]["pair"]["bit_exact"] = False
    elif plant == "exact-loss":
        base_rows, _ = _rows(base)
        candidate_rows, _ = _rows(candidate)
        base_rows[0]["bit_identical"] = True
        base_rows[0]["unequal"] = 0
        candidate_rows[0]["bit_identical"] = False
        candidate_rows[0]["unequal"] = 1

    require(pair["status"] == "PASS_R215_OMT1_VECTOR_PAIR_REPLAY",
            "atomic replay record is not admitted")
    require(pair["operand_rows"]["zv_frc"]["differing_cells"] == 68,
            "zv_frc operand census moved")
    require(pair["operand_rows"]["ssvmask"]["differing_cells"] == 68,
            "ssvmask operand census moved")
    require(pair["replays"]["pair"]["bit_exact"],
            "atomic pair no longer closes the pre-LBC target")
    require(not pair["replays"]["zv_frc_only"]["bit_exact"],
            "zv_frc half became sufficient")
    require(not pair["replays"]["ssvmask_only"]["bit_exact"],
            "ssvmask half became sufficient")

    comparison = _comparison(base, candidate)
    counts = {}
    for metric in ("rms", "max_abs"):
        census = Counter(
            row["metrics"][metric]["direction"]
            for row in comparison["moved_rows"])
        counts[metric] = {
            name: int(census[name]) for name in ("toward", "away", "equal")}
    if plant == "false-majority":
        counts["rms"] = {"toward": 0, "away": 1, "equal": 0}
    require(sum(counts["rms"].values()) == comparison["moved_row_count"],
            "RMS direction census is incomplete")
    rms_votes = counts["rms"]["toward"] + counts["rms"]["away"]
    majority = counts["rms"]["toward"] * 2 > rms_votes

    first_target = (1, "stage1", "T")
    first_before, first_after = _row(base, first_target), _row(candidate, first_target)
    first_direction = _direction(first_before, first_after, "rms")
    ssh_target = (1, "stage1", "ssh")
    ssh_before, ssh_after = _row(base, ssh_target), _row(candidate, ssh_target)
    ssh_not_worse = float(ssh_after["max_abs"]) <= float(ssh_before["max_abs"])
    eligible = (
        majority and first_direction in ("toward", "equal")
        and not comparison["bit_identical_losses"] and ssh_not_worse)
    require(eligible, "atomic pair is not Decision-96 eligible")
    return {
        "format": "nemo-testcase-l4-orca2-round215-decision96-v1",
        "claim_label": next(k for k in ("independent", "given_nemo_entry") if k in base),
        "operand_proof": {
            "zv_frc_differing_cells": 68, "ssvmask_differing_cells": 68,
            "each_half_refuted": True, "atomic_pair_bit_exact": True,
        },
        "ladder": {
            **comparison,
            "rms_direction_census": counts["rms"],
            "max_direction_census": counts["max_abs"],
            "rms_score_moved_row_count": rms_votes,
            "strict_majority_toward": majority,
            "first_over_bar_rms_direction": first_direction,
            "kt1_stage1_ssh": {
                "rms_before": ssh_before["rms"], "rms_after": ssh_after["rms"],
                "max_before": ssh_before["max_abs"], "max_after": ssh_after["max_abs"],
                "max_not_worse": ssh_not_worse,
            },
        },
        "decision96_eligible": True,
        "status": "PASS_R215_VECTOR_PAIR_DECISION96",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--plant", choices=("none", "pair-closure", "exact-loss", "false-majority"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(*(
            json.loads(path.read_text(encoding="utf-8"))
            for path in (args.base, args.candidate, args.pair)
        ), plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, StopIteration, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
