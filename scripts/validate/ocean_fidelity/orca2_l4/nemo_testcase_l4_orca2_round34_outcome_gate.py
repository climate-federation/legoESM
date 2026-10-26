#!/usr/bin/env python3
"""Fail-closed ORCA2 trajectory outcome for the round-34 hf_0 landing."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round25_outcome_gate as r25,
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
    comparison = r25._compare(parent, arm, "round34_carried_hf0")
    accepted = (
        comparison["checkpoint_count"] == 40
        and comparison["moved_row_count"] > 0
        and comparison["at_bar_rows_left"] == []
        and comparison["first_non_bit_statement_unchanged"]
    )
    return {
        "status": "LANDED" if accepted else "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "comparison": comparison,
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
