#!/usr/bin/env python3
"""Gate the round-184 north-fold HPG/depth-average cancelling unit."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round111_ladder_compare as compare_gate,
)


PLANTS = (
    "none",
    "operand-closure",
    "exact-row-loss",
    "false-majority",
    "score-equal-vote",
)


class GateError(RuntimeError):
    """The atomic unit evidence is incomplete or ineligible to land."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _direction_census(comparison: dict, metric: str) -> dict[str, int]:
    counts = Counter(
        row["metrics"][metric]["direction"]
        for row in comparison["moved_rows"]
    )
    return {name: int(counts[name]) for name in ("toward", "away", "equal")}


def _row(document: dict, target: tuple[int, str, str]) -> dict:
    rows, _ = compare_gate._rows(document)
    for row in rows:
        if (row["kt"], row["checkpoint"], row["field"]) == target:
            return row
    raise GateError(f"missing registered ladder row {target}")


def _metric_direction(before: dict, after: dict, metric: str) -> str:
    delta = float(after[metric]) - float(before[metric])
    return "toward" if delta < 0.0 else "away" if delta > 0.0 else "equal"


def _strict_score_moved_majority(
    rms: dict[str, int], *, include_equal: bool = False,
) -> bool:
    denominator = rms["toward"] + rms["away"]
    if include_equal:
        denominator += rms["equal"]
    return rms["toward"] * 2 > denominator


def classify(
    rung0_base: dict,
    rung0_candidate: dict,
    rung7_base: dict,
    rung7_candidate: dict,
    operand_report: dict,
    *,
    plant: str = "none",
) -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    rung0_candidate = copy.deepcopy(rung0_candidate)
    operand_report = copy.deepcopy(operand_report)
    if plant == "operand-closure":
        operand_report["north_only_replay"]["bit_exact"] = False
        operand_report["north_only_replay"]["at_floor"] = False
    elif plant == "exact-row-loss":
        rung0_candidate["rows"][0]["bit_identical"] = False
        rung0_candidate["rows"][0]["unequal"] = 1

    require(operand_report["target"] == {
        "cells": 68, "row": 147, "wet_levels": 1319},
        "round-183 registered unit moved")
    require(operand_report["north_only_replay"]["bit_exact"],
            "north-fold HPG replay no longer closes bit-for-bit")
    require(operand_report["fold_component_to_rhs"]["bit_exact"],
            "recorded HPG component no longer closes the raw RHS")
    require(operand_report["atomic_rows"]["oracle_all"]["bit_exact"],
            "four recorded operands no longer close the depth average")
    expected_operand_counts = {
        "raw_hpg": 1319, "e3v": 27, "vmask": 1319, "r1_hv0": 68}
    actual_operand_counts = {
        name: int(operand_report["atomic_operand_rows"][name]["differing_cells"])
        for name in expected_operand_counts
    }
    require(actual_operand_counts == expected_operand_counts,
            f"four-operand census moved: {actual_operand_counts}")

    require(
        rung0_candidate.get("private_arm", {}).get(
            "hpg_fold_depth_average_unit") is True,
        "rung-0 candidate did not execute the atomic private arm",
    )
    require(
        rung7_candidate.get("candidate_trajectory", {}).get(
            "private_arm", {}).get("hpg_fold_depth_average_unit") is True,
        "rung-7 candidate did not execute the atomic private arm",
    )
    comparisons = {
        "rung0_independent": compare_gate.compare(
            rung0_base, rung0_candidate),
        "rung7_given_nemo_entry": compare_gate.compare(
            rung7_base, rung7_candidate),
    }
    summaries: dict[str, object] = {}
    eligible = True
    for name, comparison in comparisons.items():
        rms = _direction_census(comparison, "rms")
        maximum = _direction_census(comparison, "max_abs")
        if plant == "false-majority" and name == "rung0_independent":
            rms = {"toward": 0, "away": 0, "equal": 0}
        require(sum(rms.values()) == comparison["moved_row_count"],
                f"{name} RMS direction census is incomplete")
        first_target = (1, "stage1", "T")
        before_first = _row(
            rung0_base if name == "rung0_independent" else rung7_base,
            first_target,
        )
        after_first = _row(
            rung0_candidate if name == "rung0_independent" else rung7_candidate,
            first_target,
        )
        first_direction = _metric_direction(before_first, after_first, "rms")
        ssh_target = (10, "stage3", "ssh")
        before_ssh = _row(
            rung0_base if name == "rung0_independent" else rung7_base,
            ssh_target,
        )
        after_ssh = _row(
            rung0_candidate if name == "rung0_independent" else rung7_candidate,
            ssh_target,
        )
        unchanged = comparison["moved_row_count"] == 0
        rms_moved = rms["toward"] + rms["away"]
        require(
            rms_moved + rms["equal"] == comparison["moved_row_count"],
            f"{name} RMS score-moved census is incomplete",
        )
        # Decision 96 counts rows whose score moved.  A row whose bits moved
        # but whose RMS is unchanged is registered, but is not a vote.
        strict_majority = _strict_score_moved_majority(
            rms,
            include_equal=(
                plant == "score-equal-vote"
                and name == "rung0_independent"),
        )
        ladder_eligible = (
            (unchanged or strict_majority)
            and first_direction in ("toward", "equal")
            and not comparison["bit_identical_losses"]
            and float(after_ssh["max_abs"]) <= float(before_ssh["max_abs"])
        )
        eligible = eligible and ladder_eligible
        summaries[name] = {
            **comparison,
            "rms_direction_census": rms,
            "max_direction_census": maximum,
            "first_over_bar_rms_direction": first_direction,
            "kt10_stage3_ssh_max_abs": {
                "before": before_ssh["max_abs"],
                "after": after_ssh["max_abs"],
                "direction": _metric_direction(before_ssh, after_ssh, "max_abs"),
            },
            "byte_unchanged": unchanged,
            "rms_score_moved_row_count": rms_moved,
            "strict_majority_toward": strict_majority,
            "decision96_eligible": ladder_eligible,
        }

    require(eligible, "atomic unit is not Decision-96 eligible")

    result = {
        "format": "nemo-testcase-l4-orca2-round184-atomic-hpg-unit-v1",
        "claim_labels": {
            "rung0": "independent hierarchy rung 0",
            "rung7": "given NEMO's entry",
        },
        "compiled_sources": operand_report["compiled_source"],
        "operand_proof": {
            "target": operand_report["target"],
            "differing_cells_before_substitution": actual_operand_counts,
            "north_fold_hpg_bit_exact_after_substitution": True,
            "raw_hpg_component_bit_exact": True,
            "four_operand_endpoint_bit_exact": True,
        },
        "ladders": summaries,
        "decision96_eligible": eligible,
        "status": (
            "LANDED_ATOMIC_HPG_UNIT" if eligible
            else "HELD_ATOMIC_HPG_UNIT_NOT_NET_IMPROVEMENT"),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung0-base", type=Path, required=True)
    parser.add_argument("--rung0-candidate", type=Path, required=True)
    parser.add_argument("--rung7-base", type=Path, required=True)
    parser.add_argument("--rung7-candidate", type=Path, required=True)
    parser.add_argument("--operand-report", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(*(
            json.loads(path.read_text(encoding="utf-8"))
            for path in (
                args.rung0_base,
                args.rung0_candidate,
                args.rung7_base,
                args.rung7_candidate,
                args.operand_report,
            )
        ), plant=args.plant)
    except (OSError, KeyError, TypeError, ValueError,
            GateError, compare_gate.GateError) as error:
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
