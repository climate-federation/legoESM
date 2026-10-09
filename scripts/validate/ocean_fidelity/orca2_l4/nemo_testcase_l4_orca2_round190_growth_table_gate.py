#!/usr/bin/env python3
"""Classify the complete independent ORCA2 rung-0 month growth table."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round187_growth_walk_gate as prior,
)


STEPS = tuple(range(10, 100, 10)) + (95,)
PLANTS = (
    "none", "record-status", "missing-95", "score-ulp", "growth-selection",
    "stage-order", "operator-closure",
)


class GateError(RuntimeError):
    """The complete record or frozen growth statistic moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _validate_score(score: dict, field: str, step: int) -> None:
    values = tuple(float(score[name]) for name in (
        "max_abs", "rms", "candidate_at_argmax", "oracle_at_argmax"))
    require(all(math.isfinite(value) for value in values),
            f"step {step} {field}: score is non-finite")
    maximum, rms, candidate, oracle = values
    require(maximum == abs(candidate - oracle),
            f"step {step} {field}: maximum does not replay its argmax")
    require(maximum >= 0.0 and rms >= 0.0,
            f"step {step} {field}: negative norm")
    count, unequal = int(score["count"]), int(score["unequal"])
    require(count > 0 and 0 <= unequal <= count,
            f"step {step} {field}: cell census moved")
    require(bool(score["bit_identical"]) == (unequal == 0),
            f"step {step} {field}: exactness flag disagrees with census")
    expected_rank = 2 if field == "ssh" else 3
    require(len(score["argmax_jik"]) == expected_rank,
            f"step {step} {field}: argmax rank moved")


def _growth_rows(month: dict) -> list[dict]:
    rows = month.get("growth_table", [])
    require(tuple(int(row["step"]) for row in rows) == STEPS,
            "month checkpoint registry moved")
    previous = prior.FLOOR
    result = []
    for row in rows:
        step = int(row["step"])
        require(row.get("claim_label") == "independent",
                f"step {step}: claim population moved")
        scores = row.get("error_rows")
        require(scores is not None and set(scores) == set(prior.FIELDS),
                f"step {step}: error field registry moved")
        for field in prior.FIELDS:
            _validate_score(scores[field], field, step)
        maximum, field = max(
            (float(scores[name]["max_abs"]), name) for name in prior.FIELDS)
        ratio = maximum / max(previous, prior.FLOOR)
        result.append({
            "step": step,
            "max_abs": maximum,
            "field": field,
            "argmax_jik": copy.deepcopy(scores[field]["argmax_jik"]),
            "previous_max_abs": previous,
            "ratio": ratio,
            "over_10x": bool(ratio > prior.GROWTH),
            "field_rows": copy.deepcopy(scores),
        })
        previous = maximum
    return result


def classify(admission: dict, month: dict, ladder: dict,
             plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    admission, month, ladder = map(copy.deepcopy, (admission, month, ladder))
    if plant == "record-status":
        admission["status"] = "PLANTED"
    elif plant == "missing-95":
        admission["admitted_steps"] = admission["admitted_steps"][:-1]
    elif plant == "score-ulp":
        score = month["growth_table"][-1]["error_rows"]["v"]
        score["max_abs"] = float(np.nextafter(score["max_abs"], np.inf))
    elif plant == "stage-order":
        ladder["rows"][0]["checkpoint"] = "stage1"
    elif plant == "operator-closure":
        ladder["first_non_bit_source_statement"]["statement"] = "PLANTED"

    require(admission.get("status") == "PASS_R189_GROWTH_RECORD",
            "bounded-list record did not pass")
    require(admission.get("claim_label") == "independent",
            "record claim population moved")
    require(tuple(admission.get("admitted_steps", [])) == STEPS,
            "admitted checkpoint registry moved")
    require(admission.get("missing_steps") == [], "record has missing steps")
    require(int(admission.get("rank_count", 0)) == 2,
            "record rank count moved")
    require(int(admission.get("terminal_sentinel", 0)) == 96,
            "terminal sentinel moved")
    require(admission.get("terminal_overwrites") == [],
            "record contains terminal overwrites")
    comparisons = admission.get("comparisons", [])
    require(len(comparisons) == 2 * len(STEPS),
            "twin-comparison census moved")
    require(all(not any(int(value) for value in row["unequal"].values())
                for row in comparisons), "twin record is not bit-exact")
    sentinel = admission.get("sentinel_comparisons", [])
    require(len(sentinel) == 2 and all(
        int(row["step"]) == 96
        and not any(int(value) for value in row["unequal"].values())
        for row in sentinel), "terminal sentinel is not rank-complete/exact")

    require(month.get("status") == "PASS_R186_MONTH_BOUNDARY",
            "production month boundary gate did not pass")
    require(month.get("claim_label") == "independent"
            and month.get("initial_mode") == "card_own_state",
            "month population is not independent")
    require(month.get("missing_oracle_steps") == [],
            "complete month still has missing oracle steps")
    require(int(month["runtime_refusal"]["step"]) == 96,
            "month terminal step moved")
    require(int(month.get("steps_completed", -1)) == 95,
            "month did not complete step 95")
    require(all(month["initial_entry"][name]["bit_exact"]
                and int(month["initial_entry"][name]["unequal"]) == 0
                for name in prior.FIELDS), "independent entry moved")

    require(ladder.get("claim_label") == "independent"
            and ladder.get("status") == "PASS_RUNG0_TEN_STEP_LADDER",
            "stage ladder did not pass")
    require(ladder.get("first_non_bit_source_statement") == prior.EXPECTED_STATEMENT,
            "first-statement replay closure moved")
    stages = prior._stage_rows(ladder)
    expected_order = [
        (kt, checkpoint) for kt in range(1, 11)
        for checkpoint in prior.CHECKPOINTS
    ]
    require([(row["kt"], row["checkpoint"]) for row in stages] == expected_order,
            "stage order moved")

    coarse = _growth_rows(month)
    first = next((copy.deepcopy(row) for row in coarse if row["over_10x"]), None)
    if plant == "growth-selection" and first is not None:
        first["step"] = 20
    derived = next((row for row in coarse if row["over_10x"]), None)
    require(first == derived, "coarse growth selector moved")
    first_stage = next((copy.deepcopy(row) for row in stages if row["over_10x"]), None)
    require(first is not None and first_stage is not None,
            "growth boundary is absent")

    return {
        "format": "nemo-testcase-l4-orca2-round190-growth-table-v1",
        "status": "HELD_FIRST_STATEMENT_SHARED_GATE",
        "claim_label": "independent",
        "floor": prior.FLOOR,
        "growth_threshold": prior.GROWTH,
        "coarse_growth": coarse,
        "first_coarse_growth": first,
        "first_stage_growth": first_stage,
        "first_non_bit_statement": copy.deepcopy(prior.EXPECTED_STATEMENT),
        "month_runtime_refusal": copy.deepcopy(month["runtime_refusal"]),
        "prediction_ledger": {
            "R190-P1": "CONFIRMED",
            "R190-P2": "CONFIRMED" if first["step"] == 10 else "REFUTED",
            "R190-P3": (
                "CONFIRMED" if coarse[-1]["over_10x"] else "REFUTED"),
            "R190-P4": "CONFIRMED",
            "R190-P5": "CONFIRMED",
            "R190-P6": "CONFIRMED",
        },
        "source_citation": prior.EXPECTED_STATEMENT["nemo_source"],
    }


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
    except (GateError, prior.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
