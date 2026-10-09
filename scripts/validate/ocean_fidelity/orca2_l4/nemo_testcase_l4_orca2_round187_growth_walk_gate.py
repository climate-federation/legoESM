#!/usr/bin/env python3
"""Classify ORCA2 rung-0's independent month growth and stage refinement."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


FLOOR = 2.0e-10
GROWTH = 10.0
FIELDS = ("T", "S", "u", "v", "ssh")
COARSE_STEPS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
CHECKPOINTS = ("entry", "stage1", "stage2", "stage3")
EXPECTED_STATEMENT = {
    "statement": "vector-invariant vertical average of the completed 3-D RHS",
    "nemo_source": "ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219",
    "legoesm_discriminator": (
        "round 94 source-associated materialized multiply/add/final-multiply arm"
    ),
    "status": "HELD_BY_GYRE_2ULP_GATE",
}
PLANTS = (
    "none", "terminal-truncation", "earlier-step-header", "growth-selection",
    "stage-order", "operator-closure",
)


class GateError(RuntimeError):
    """An admission, derivation, or source-order prerequisite moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _growth_rows(rows: list[dict], steps: tuple[int, ...]) -> list[dict]:
    by_step = {int(row["step"]): row for row in rows}
    require(tuple(by_step) == steps + (95,), "month checkpoint registry moved")
    previous = FLOOR
    result = []
    for step in steps:
        scores = by_step[step].get("error_rows")
        require(scores is not None and tuple(sorted(scores)) == tuple(sorted(FIELDS)),
                f"step {step}: error field registry moved")
        ranked = sorted(
            ((float(scores[name]["max_abs"]), name,
              scores[name].get("argmax_jik")) for name in FIELDS),
            reverse=True,
        )
        maximum, field, argmax = ranked[0]
        ratio = maximum / max(previous, FLOOR)
        result.append({
            "step": step, "max_abs": maximum, "field": field,
            "argmax_jik": argmax, "previous_max_abs": previous,
            "ratio": ratio, "over_10x": bool(ratio > GROWTH),
            "field_rows": copy.deepcopy(scores),
        })
        previous = maximum
    require(by_step[95].get("error_rows") is None,
            "step 95 unexpectedly has an oracle score")
    return result


def _stage_rows(ladder: dict) -> list[dict]:
    rows = ladder.get("rows", [])
    require(len(rows) == 200, "rung-0 ladder row census moved")
    index = {
        (int(row["kt"]), str(row["checkpoint"]), str(row["field"])): row
        for row in rows
    }
    require(len(index) == 200, "rung-0 ladder keys are not unique")
    previous = FLOOR
    result = []
    for kt in range(1, 11):
        for checkpoint in CHECKPOINTS:
            ranked = sorted(
                ((float(index[(kt, checkpoint, name)]["max_abs"]), name,
                  index[(kt, checkpoint, name)].get("first_unequal_index"))
                 for name in FIELDS),
                reverse=True,
            )
            maximum, field, argmax = ranked[0]
            ratio = maximum / max(previous, FLOOR)
            result.append({
                "kt": kt, "checkpoint": checkpoint, "max_abs": maximum,
                "field": field, "argmax": argmax, "previous_max_abs": previous,
                "ratio": ratio, "over_10x": bool(ratio > GROWTH),
            })
            previous = maximum
    return result


def classify(admission: dict, month: dict, ladder: dict,
             plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    admission = copy.deepcopy(admission)
    month = copy.deepcopy(month)
    ladder = copy.deepcopy(ladder)
    if plant == "terminal-truncation":
        admission["terminal_overwrites"] = []
    elif plant == "earlier-step-header":
        admission["admitted_steps"][0] = 11
    elif plant == "stage-order":
        ladder["rows"][0]["checkpoint"] = "stage1"
    elif plant == "operator-closure":
        ladder["first_non_bit_source_statement"]["statement"] = "PLANTED"

    require(admission.get("status") == "STOP_R187_TERMINAL_RESTART_OVERWRITTEN",
            "terminal-overwrite admission status moved")
    require(tuple(admission.get("admitted_steps", [])) == COARSE_STEPS,
            "admitted checkpoint registry moved")
    require(admission.get("missing_steps") == [95], "missing-step registry moved")
    overwrites = admission.get("terminal_overwrites", [])
    require(len(overwrites) == 4, "terminal overwrite census moved")
    require(all(row.get("kt") == 0.0 and
                not any(row.get("field_sizes", {}).values())
                for row in overwrites), "terminal overwrite payload moved")
    require(month.get("claim_label") == "independent" and
            month.get("initial_mode") == "card_own_state",
            "month population is not independent")
    require(all(month["initial_entry"][name]["bit_exact"] and
                int(month["initial_entry"][name]["unequal"]) == 0
                for name in FIELDS), "independent entry moved")
    require(month.get("missing_oracle_steps") == [95],
            "month missing-step registry moved")
    require(ladder.get("claim_label") == "independent" and
            ladder.get("status") == "PASS_RUNG0_TEN_STEP_LADDER",
            "stage ladder did not pass")
    require(ladder.get("first_non_bit_source_statement") == EXPECTED_STATEMENT,
            "first-statement replay closure moved")

    coarse = _growth_rows(month["growth_table"], COARSE_STEPS)
    first_coarse = next((copy.deepcopy(row) for row in coarse if row["over_10x"]), None)
    stages = _stage_rows(ladder)
    expected_order = [
        (kt, checkpoint) for kt in range(1, 11) for checkpoint in CHECKPOINTS
    ]
    require([(row["kt"], row["checkpoint"]) for row in stages] == expected_order,
            "stage order moved")
    first_stage = next((copy.deepcopy(row) for row in stages if row["over_10x"]), None)
    if plant == "growth-selection" and first_coarse is not None:
        first_coarse["step"] = 20
    derived_first_coarse = next((row for row in coarse if row["over_10x"]), None)
    require(first_coarse == derived_first_coarse, "coarse growth selector moved")

    report = {
        "format": "nemo-testcase-l4-orca2-round187-growth-walk-v1",
        "status": "STOPPED_FOR_RECORD",
        "claim_label": "independent",
        "floor": FLOOR,
        "growth_threshold": GROWTH,
        "admission_status": admission["status"],
        "admitted_steps": list(COARSE_STEPS),
        "missing_steps": [95],
        "coarse_growth": coarse,
        "first_coarse_growth": first_coarse,
        "stage_growth": stages,
        "first_stage_growth": first_stage,
        "first_non_bit_statement": copy.deepcopy(EXPECTED_STATEMENT),
        "month_runtime_refusal": copy.deepcopy(month["runtime_refusal"]),
        "prediction_ledger": {
            "R187-P1": "REFUTED",
            "R187-P2": "CONFIRMED",
            "R187-P3": ("CONFIRMED" if first_coarse and
                         first_coarse["step"] == 10 else "REFUTED"),
            "R187-P4": ("CONFIRMED" if first_stage and
                         (first_stage["kt"], first_stage["checkpoint"]) ==
                         (1, "stage1") and
                         EXPECTED_STATEMENT["statement"] == "HPG" else "REFUTED"),
            "R187-P5": "CONFIRMED",
            "R187-P6": "CONFIRMED",
        },
        "source_citations": {
            "restart_open": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:98-143",
            "restart_cursor": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:190-196",
            "restart_time": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/daymod.f90:407-417",
            "first_statement": EXPECTED_STATEMENT["nemo_source"],
        },
    }
    require(math.isfinite(float(report["first_coarse_growth"]["ratio"])),
            "first coarse growth ratio is not finite")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--month", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(
            json.loads(args.admission.read_text()),
            json.loads(args.month.read_text()),
            json.loads(args.ladder.read_text()),
            args.plant,
        )
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS STOPPED_FOR_RECORD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
