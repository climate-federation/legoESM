#!/usr/bin/env python3
"""Fail-closed ORCA2 trajectory outcome for the round-33 mask landing."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round25_outcome_gate as r25,
)

EXPECTED_KT10 = {"u": 0.33860877536933787, "v": 0.5353439568885162}


def _row(document: dict, field: str) -> dict:
    return next(
        item["rows"][field]
        for item in document["candidate_trajectory"]["checkpoints"]
        if item["kt"] == 10 and item["checkpoint"] == "stage3"
    )


def evaluate(parent_path: Path, arm_path: Path, *, plant: bool = False) -> dict:
    parent = r25._read(parent_path)
    arm = r25._read(arm_path)
    if plant:
        arm = copy.deepcopy(arm)
        arm["candidate_trajectory"]["checkpoints"][0]["rows"]["T"].update(
            bit_identical=False, unequal=1, max_abs=float.fromhex("0x1p-1074"),
            mean_abs_over_unequal=float.fromhex("0x1p-1074"),
            first_unequal_index=[0, 0, 0])
    comparison = r25._compare(parent, arm, "round33_single_mask")
    velocities = {
        name: {"measured": float(_row(arm, name)["max_abs"]),
               "expected": expected}
        for name, expected in EXPECTED_KT10.items()
    }
    expected = (
        comparison["checkpoint_count"] == 40
        and comparison["moved_row_count"] == 175
        and comparison["toward"] == 119
        and comparison["away"] == 56
        and comparison["same_maximum"] == 0
        and comparison["first_moved_row"] == "kt=2:stage1:T"
        and comparison["at_bar_rows_left"] == []
        and comparison["first_non_bit_statement_unchanged"]
        and all(row["measured"] == row["expected"]
                for row in velocities.values())
    )
    return {
        "status": "LANDED" if expected else "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "comparison": comparison,
        "kt10_stage3_velocity_maxima": velocities,
        "plant": plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(args.parent, args.arm, plant=args.plant)
    except (r25.GateError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.plant:
        return 1 if result["status"] != "LANDED" else 3
    return 0 if result["status"] == "LANDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
