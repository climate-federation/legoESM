#!/usr/bin/env python3
"""Classify round 217's complete vector-unit landing attempt."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PLANTS = ("none", "omt-direction", "stage-boundary", "live-thickness")


class GateError(RuntimeError):
    """The complete-unit evidence no longer supports its held verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(independent: dict, given: dict, rung0_log: str,
             plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    independent = copy.deepcopy(independent)
    given = copy.deepcopy(given)
    if plant == "omt-direction":
        independent["ladder"]["rms_direction_census"]["away"] = 1
    elif plant == "stage-boundary":
        rung0_log = rung0_log.replace(
            "PROGRESS kt=8 exposed stages 1-2", "PROGRESS kt=7 exposed stages 1-2")
    elif plant == "live-thickness":
        rung0_log = rung0_log.replace(
            "raw-mesh e3w_int must contain only finite values > 0",
            "different refusal",
        )

    for label, report in (("independent OMT-1", independent),
                          ("given NEMO's entry OMT-1", given)):
        require(report.get("status") == "PASS_R215_VECTOR_PAIR_DECISION96",
                f"{label}: Decision-96 classifier did not pass")
        ladder = report["ladder"]
        require(ladder["moved_row_count"] == 155,
                f"{label}: moved-row census changed")
        require(ladder["rms_direction_census"] == {
            "toward": 155, "away": 0, "equal": 0,
        }, f"{label}: RMS direction census changed")
        require(ladder["first_over_bar_rms_direction"] == "toward",
                f"{label}: first debt no longer moves toward NEMO")
        require(not ladder["bit_identical_losses"],
                f"{label}: an exact row left the bar")
        require(ladder["final_stage3_ssh_max"]["not_worse"],
                f"{label}: final SSH maximum worsened")

    require("PROGRESS kt=8 exposed stages 1-2" in rung0_log,
            "rung 0 did not reach the frozen kt=8 boundary")
    require("PROGRESS kt=8 completed stage 3" not in rung0_log,
            "rung 0 unexpectedly completed kt=8 stage 3")
    require("PROGRESS kt=9" not in rung0_log,
            "rung 0 progressed beyond the registered refusal")
    refusal = "raw-mesh e3w_int must contain only finite values > 0"
    require(refusal in rung0_log and "STATUS REFUSE:" in rung0_log,
            "rung-0 live-thickness refusal moved")

    return {
        "format": "nemo-testcase-l4-orca2-round217-atomic-unit-v1",
        "status": "HELD_R217_RUNG0_LIVE_THICKNESS",
        "omt1": {
            "independent": independent["ladder"],
            "given_nemo_entry": given["ladder"],
        },
        "binding_rung0_refusal": {
            "last_completed_boundary": "kt=8 stages 1-2 exposed",
            "missing_boundary": "kt=8 completed stage 3",
            "message": refusal,
            "verdict": "HELD",
        },
        "first_downstream_boundary": {
            "name": "live W-thickness construction before completed kt=8 stage 3",
            "compiled_consumer": (
                "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:383-402"),
            "statement_verdict": "UNMEASURED_WITH_SPEC",
        },
        "predictions": {
            "R217-P1": "CONFIRMED_DIRECT_TESTS",
            "R217-P2": "CONFIRMED_BOTH_LABELS",
            "R217-P3": "REFUTED_RUNG0_KT8_LIVE_THICKNESS",
            "R217-P4": "UNMEASURED_PREREQUISITE_R217-P3",
            "R217-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
        "open": (
            "Record the complete-unit kt=8 live W thickness and its SSH/r3t "
            "operands passively; name the first non-finite operand before "
            "re-scoring the indivisible unit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--given", type=Path, required=True)
    parser.add_argument("--rung0-log", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(
            json.loads(args.independent.read_text(encoding="utf-8")),
            json.loads(args.given.read_text(encoding="utf-8")),
            args.rung0_log.read_text(encoding="utf-8"),
            args.plant,
        )
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS HELD_R217_RUNG0_LIVE_THICKNESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
