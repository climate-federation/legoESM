#!/usr/bin/env python3
"""Score round 205's complete OMT-0 fold unit over both ten-step ladders."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round111_ladder_compare as compare_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round184_atomic_hpg_unit_gate as decision96,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)

PLANTS = (
    "none", "missing-arm", "exact-row-loss", "false-majority",
    "score-equal-vote",
)
LABELS = ("independent", "given_nemo_entry")
PRIVATE_ARM = {
    "external_mode_association": True,
    "raw_reference_depth": True,
    "unmasked_v_transport": True,
    "materialize_v_transport": True,
}
BASE_ARM = {name: False for name in PRIVATE_ARM}


class GateError(RuntimeError):
    """The atomic OMT-0 unit census is incomplete or misclassified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _slow_v_is_score_null(report: dict) -> dict[str, object]:
    score = report["slow_v_arm"]["input"]
    require(score["comparison_bit_exact"],
            "round-205 recorded slow-V substitution is not exact")
    association = report["association_arm"]["substep_table"]
    combined = report["slow_v_association_unit"]["substep_table"]
    require(len(association) == len(combined) == 65,
            "round-205 slow-V null proof is incomplete")
    state_fields = ("ssh", "ua_b", "va_b")
    for index, (left, right) in enumerate(zip(association, combined), start=1):
        for field in state_fields:
            require(left[field] == right[field],
                    f"slow-V substitution moved {field} at substep {index}")
    require(len(report["complete_fold_unit"]["substep_table"]) == 65,
            "round-205 complete fold unit is incomplete")
    return {
        "recorded_input_bit_exact": True,
        "association_state_scores_equal_for_65_substeps": True,
    }


def _direction_map(comparison: dict) -> dict[tuple[int, str, str], str]:
    return {
        (row["kt"], row["checkpoint"], row["field"]):
        row["metrics"]["rms"]["direction"]
        for row in comparison["moved_rows"]
    }


def classify(
    baseline: dict,
    candidate: dict,
    substep_report: dict,
    *,
    plant: str = "none",
) -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    baseline = copy.deepcopy(baseline)
    candidate = copy.deepcopy(candidate)
    if plant == "missing-arm":
        candidate["independent"]["private_arm"][
            "materialize_v_transport"] = False
    elif plant == "exact-row-loss":
        candidate["given_nemo_entry"]["rows"][0]["bit_identical"] = False
        candidate["given_nemo_entry"]["rows"][0]["unequal"] = 1

    null_proof = _slow_v_is_score_null(substep_report)
    for label in LABELS:
        require(baseline[label]["private_arm"] == BASE_ARM,
                f"{label}: baseline private arm moved")
        require(candidate[label]["private_arm"] == PRIVATE_ARM,
                f"{label}: candidate is not the four-statement atomic unit")
        require(baseline[label]["row_count"] == 200,
                f"{label}: baseline ladder is incomplete")
        require(candidate[label]["row_count"] == 200,
                f"{label}: candidate ladder is incomplete")

    comparisons = {
        label: compare_gate.compare(baseline[label], candidate[label])
        for label in LABELS
    }
    summaries: dict[str, object] = {}
    eligible = True
    for label, comparison in comparisons.items():
        rms = decision96._direction_census(comparison, "rms")
        maximum = decision96._direction_census(comparison, "max_abs")
        if plant == "false-majority" and label == "independent":
            rms = {"toward": 0, "away": 1, "equal": 0}
            require(False, "false-majority plant removed the toward majority")
        include_equal = plant == "score-equal-vote" and label == "independent"
        if include_equal:
            rms = {"toward": 1, "away": 0, "equal": 2}
            require(
                decision96._strict_score_moved_majority(
                    rms, include_equal=True),
                "score-equal rows were incorrectly counted as votes",
            )
        require(sum(rms.values()) == comparison["moved_row_count"],
                f"{label}: RMS direction census is incomplete")
        first_before = decision96._row(
            baseline[label], (1, "stage1", "T"))
        first_after = decision96._row(
            candidate[label], (1, "stage1", "T"))
        first_direction = decision96._metric_direction(
            first_before, first_after, "rms")
        ssh1_before = decision96._row(
            baseline[label], (1, "stage1", "ssh"))
        ssh1_after = decision96._row(
            candidate[label], (1, "stage1", "ssh"))
        ssh10_before = decision96._row(
            baseline[label], (10, "stage3", "ssh"))
        ssh10_after = decision96._row(
            candidate[label], (10, "stage3", "ssh"))
        strict_majority = decision96._strict_score_moved_majority(rms)
        headline_improved = (
            float(ssh1_after["rms"]) < float(ssh1_before["rms"])
            and float(ssh1_after["max_abs"]) < float(ssh1_before["max_abs"])
            and float(ssh10_after["max_abs"])
            <= float(ssh10_before["max_abs"])
        )
        label_eligible = (
            strict_majority
            and first_direction in ("toward", "equal")
            and not comparison["bit_identical_losses"]
            and headline_improved
        )
        eligible = eligible and label_eligible
        summaries[label] = {
            **comparison,
            "rms_direction_census": rms,
            "max_direction_census": maximum,
            "first_registered_row_rms_direction": first_direction,
            "kt1_stage1_ssh": {
                "before_rms": ssh1_before["rms"],
                "after_rms": ssh1_after["rms"],
                "before_max_abs": ssh1_before["max_abs"],
                "after_max_abs": ssh1_after["max_abs"],
            },
            "kt10_stage3_ssh_max_abs": {
                "before": ssh10_before["max_abs"],
                "after": ssh10_after["max_abs"],
            },
            "strict_majority_toward": strict_majority,
            "headline_ssh_improved": headline_improved,
            "decision96_eligible": label_eligible,
        }

    require(
        _direction_map(comparisons["independent"])
        == _direction_map(comparisons["given_nemo_entry"]),
        "independent and given-entry RMS direction maps disagree",
    )
    return {
        "format": "nemo-testcase-l4-orca2-round206-omt0-fold-census-v1",
        "claim_labels": {
            "independent": "independent OMT-0",
            "given_nemo_entry": "given NEMO's entry OMT-0",
        },
        "slow_v_null_proof": null_proof,
        "ladders": summaries,
        "decision96_eligible": eligible,
        "status": (
            "ELIGIBLE_ATOMIC_OMT0_FOLD_UNIT"
            if eligible else "HELD_ATOMIC_OMT0_FOLD_UNIT"
        ),
    }


def measure(args) -> dict[str, object]:
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower().startswith(
        args.expect_commit.lower()),
        "round-206 measurement requires its clean committed gate")
    common = (
        args.deck_root, args.candidate, args.calibration, args.twin_a,
        args.twin_b, args.month,
    )
    baseline = omt0.run(*common, atomic_fold_unit=False)
    substep_report = json.loads(args.substep_report.read_text(encoding="utf-8"))
    try:
        candidate = omt0.run(*common, atomic_fold_unit=True)
    except omt0.GateError as error:
        return {
            "format": "nemo-testcase-l4-orca2-round206-omt0-fold-census-v1",
            "status": "HELD_ATOMIC_OMT0_FOLD_UNIT_NONFINITE",
            "worktree": stamp,
            "candidate_refusal": str(error),
            "baseline": baseline,
            "slow_v_null_proof": _slow_v_is_score_null(substep_report),
            "decision96_eligible": False,
        }
    result = classify(baseline, candidate, substep_report)
    result.update({"worktree": stamp, "baseline": baseline, "candidate": candidate})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="measure")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--month", type=Path)
    parser.add_argument("--substep-report", type=Path, required=True)
    parser.add_argument("--expect-commit")
    parser.add_argument("--baseline-in", type=Path)
    parser.add_argument("--candidate-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.candidate, args.calibration,
                         args.twin_a, args.twin_b, args.month,
                         args.expect_commit)),
                    "measurement paths and expected commit are required")
            require(args.plant == "none", "measurement does not accept plants")
            result = measure(args)
        else:
            require(args.baseline_in and args.candidate_in,
                    "classification requires baseline and candidate reports")
            result = classify(
                json.loads(args.baseline_in.read_text(encoding="utf-8")),
                json.loads(args.candidate_in.read_text(encoding="utf-8")),
                json.loads(args.substep_report.read_text(encoding="utf-8")),
                plant=args.plant,
            )
    except (GateError, omt0.GateError, compare_gate.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
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
