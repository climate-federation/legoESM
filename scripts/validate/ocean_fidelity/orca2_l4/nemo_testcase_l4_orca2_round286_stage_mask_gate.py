#!/usr/bin/env python3
"""Round 286: gate the held resolved WS-RK3 face-mask measurement.

The producer commit is deliberately no longer the branch tip: its same-tree
candidate failed the independent rung-0 trajectory and was reverted.  This
gate preserves and checks the local replay plus that fail-closed trajectory
result; it cannot accidentally promote the held arm back into production.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
)

GateError = r276.GateError
require = r276.require
ARM = "resolved_ws_stage_masks"


def replay_gate(before: dict, after: dict, *, plant: str = "none") -> dict:
    """Require the measured consumer to close while the stale probe stays put."""

    old = before["candidate_raw_vs_nemo"]
    new = after["candidate_raw_vs_nemo"]
    if plant == "consumer-bit":
        new = {**new, "unequal": 1, "bit_exact": False,
               "max_abs": float.fromhex("0x1p-52")}
    require(old["unequal"] == 1319 and not old["bit_exact"],
            "historical consumer control moved")
    require(new["unequal"] == 0 and new["bit_exact"]
            and new["max_abs"] == 0.0, "stage-update consumer did not close")
    require(before["operand_rows"]["vmask"]["unequal"] == 1319
            and after["operand_rows"]["vmask"]["unequal"] == 1319,
            "standalone compact-mask diagnostic unexpectedly moved")
    require(all(after["passivity"].values()), "passive state changed")
    require(after["one_bit_control"]["unequal"] == 1,
            "one-bit known-answer control did not fire")
    return {
        "format": "nemo-testcase-l4-orca2-round286-stage-mask-replay-v1",
        "status": "PASS_R286_STAGE_MASK_REPLAY",
        "consumer_unequal_before": old["unequal"],
        "consumer_unequal_after": new["unequal"],
        "stale_diagnostic_unequal": 1319,
        "plant": plant,
    }


def refusal_gate(control: dict, candidate_log: str, *,
                 candidate_commit: str, plant: str = "none") -> dict:
    """Require the same-tree control to finish and candidate to fail closed."""

    require(control.get("status") == "PASS_R276_SOURCE_LADDER"
            and control.get("arm") == "control", "control is not admitted")
    require(control["rung"] == "rung0" and control["label"] == "independent",
            "refusal control is not independent rung 0")
    require(control["ladder"]["row_count"] == 200,
            "rung-0 control did not complete ten steps")
    require(control["worktree"]["commit"] == candidate_commit,
            "control and candidate are not from one committed tree")
    observed = candidate_log
    if plant == "missing-refusal":
        observed = observed.replace(
            "raw-mesh e3w_int must contain only finite values > 0", "")
    require(all(f"PROGRESS independent kt={kt}" in observed
                for kt in (1, 2, 3)), "candidate did not complete kt=1..3")
    require("PROGRESS independent kt=4" not in observed,
            "candidate unexpectedly completed kt=4")
    require("raw-mesh e3w_int must contain only finite values > 0" in observed,
            "candidate live-thickness refusal is absent")
    require("STATUS REFUSE" in observed, "candidate did not fail closed")
    return {
        "format": "nemo-testcase-l4-orca2-round286-trajectory-refusal-v1",
        "status": "PASS_R286_TRAJECTORY_REFUSAL_GATE",
        "rung": "rung0", "label": "independent",
        "control_rows": 200, "candidate_completed_kt": [1, 2, 3],
        "candidate_refused_before_checkpoint_kt": 4,
        "refusal": "raw-mesh e3w_int must contain only finite values > 0",
        "producer_commit": candidate_commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--replay-gate", nargs=2, type=Path)
    parser.add_argument("--refusal-gate", nargs=2, type=Path)
    parser.add_argument("--candidate-commit")
    parser.add_argument("--plant", choices=("none", "exact-loss", "consumer-bit",
                                             "missing-refusal"),
                        default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify:
            require(args.plant != "consumer-bit",
                    "consumer-bit plant belongs to --replay-gate")
            result = r276.classify(*args.classify, plant=args.plant)
        elif args.replay_gate:
            before = json.loads(args.replay_gate[0].read_text(encoding="utf-8"))
            after = json.loads(args.replay_gate[1].read_text(encoding="utf-8"))
            result = replay_gate(before, after, plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
        else:
            require(args.refusal_gate is not None and args.candidate_commit,
                    "no gate operation selected")
            control = json.loads(args.refusal_gate[0].read_text(encoding="utf-8"))
            log = args.refusal_gate[1].read_text(encoding="utf-8")
            result = refusal_gate(control, log,
                                  candidate_commit=args.candidate_commit,
                                  plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
