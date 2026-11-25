#!/usr/bin/env python3
"""Gate the stage-1 V-ratio fold inside the complete OMT-4 unit."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round261_stage2_v_operand_split as r261,
)

LABELS = r249.LABELS
PLANTS = (
    "none", "label-coverage", "passivity", "stage-e3v", "live-reciprocal",
    "barotropic-correction", "stage-transport", "candidate-replay",
    "oracle-replay", "one-bit",
)
PREFIX_PLANTS = ("none", "prefix-labels", "prefix-passivity", "prefix-order",
                 "prefix-one-bit")


class GateError(RuntimeError):
    """A frozen round-262 source-replay predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "passivity":
        independent["passivity"]["T"] = False
    elif plant == "stage-e3v":
        independent["operand_rows"]["e3v"]["unequal"] = 1
    elif plant == "live-reciprocal":
        independent["correction_input_rows"]["r1_hv"]["unequal"] = 1
    elif plant == "barotropic-correction":
        independent["operand_rows"]["zvb"]["unequal"] = 1
    elif plant == "stage-transport":
        independent["zFv"]["unequal"] = 182
    elif plant == "candidate-replay":
        independent["replay"]["candidate"]["unequal"] = 1
    elif plant == "oracle-replay":
        independent["replay"]["oracle"]["unequal"] = 1
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summaries = {}
    reference_records = by_label[LABELS[0]]["records"]
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R261_MEASUREMENT",
                f"{label}: source measurement status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] ==
                "production-d114-resolved-complete-private-unit",
                f"{label}: provenance moved")
        require(all(report["passivity"].values()),
                f"{label}: live trace is not passive")
        require(report["records"] == reference_records,
                f"{label}: record identity moved")
        for name, row in report["operand_rows"].items():
            require(row["unequal"] == 0,
                    f"{label}: stage transport operand {name} is non-bit")
        for name, row in report["correction_input_rows"].items():
            require(row["unequal"] == 0,
                    f"{label}: barotropic correction input {name} is non-bit")
        require(report["zFv"]["unequal"] == 181
                and report["zFv"]["max_abs"] == 0.0,
                f"{label}: registered signed-zero V-transport census moved")
        require(report["replay"]["candidate"]["unequal"] == 0,
                f"{label}: candidate replay is not exact")
        require(report["replay"]["oracle"]["unequal"] == 0,
                f"{label}: oracle replay is not exact")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summaries[label] = {
            "e3v": report["operand_rows"]["e3v"],
            "r1_hv": report["correction_input_rows"]["r1_hv"],
            "zvb": report["operand_rows"]["zvb"],
            "zFv": report["zFv"],
            "passivity": report["passivity"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round262-v-ratio-fold-v1",
        "status": "PASS_R262_V_RATIO_FOLD_D96_RATCHET",
        "labels": summaries,
        "predictions": {
            "R262-P1": "REFUTED_SIGNED_ZERO_ONLY_D96_REGISTERED",
            "R262-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def classify_prefix(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    """Admit the passive kt=1 prefix and report its first valued boundary."""

    require(plant in PREFIX_PLANTS, f"unknown prefix plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "prefix-labels":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both prefix labels are required")
    independent = by_label["independent"]
    if plant == "prefix-passivity":
        independent["passivity"][0]["fields"]["T"] = False
    elif plant == "prefix-order":
        independent["boundaries"] = independent["boundaries"][:-1]

    one = r249._score([1.0], [1.0 + 2.220446049250313e-16])
    if plant == "prefix-one-bit":
        one["unequal"] = 0
    require(one["unequal"] == 1, "prefix one-bit control did not fire")

    summaries = {}
    first_boundaries = {}
    expected_order = [(1, stage) for stage in range(4)]
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R249_STAGE_TABLE"
                and report["unit"] is True
                and report["operand_route"] == "resolved-candidate",
                f"{label}: atomic prefix provenance moved")
        require([(row["kt"], row["stage"]) for row in report["boundaries"]]
                == expected_order, f"{label}: kt=1 boundary coverage moved")
        require(len(report["passivity"]) == 1
                and all(report["passivity"][0]["fields"].values()),
                f"{label}: kt=1 observer moved the completed state")
        first = None
        for row in report["boundaries"]:
            valued = {
                field: row["fields"][field]["fold_band"]
                for field in r249.FIELDS
                if row["fields"][field]["fold_band"]["nonfinite"]
                or row["fields"][field]["fold_band"]["max_abs"]
                > float(r249.FLOOR)
            }
            if valued:
                first = {"kt": row["kt"], "stage": row["stage"],
                         "fields": valued}
                break
        require(first is not None, f"{label}: no valued kt=1 boundary")
        first_boundaries[label] = (first["kt"], first["stage"])
        summaries[label] = {
            "first_valued_boundary": first,
            "kt1_stage2": next(
                row["fields"] for row in report["boundaries"]
                if row["stage"] == 2),
            "kt1_stage3": next(
                row["fields"] for row in report["boundaries"]
                if row["stage"] == 3),
        }
    require(len(set(first_boundaries.values())) == 1,
            "claim labels disagree on the first valued boundary")
    return {
        "format": "nemo-testcase-l4-orca2-round262-atomic-prefix-v1",
        "status": "PASS_R262_ATOMIC_PREFIX_MEASUREMENT",
        "first_valued_boundary": list(first_boundaries.values())[0],
        "labels": summaries,
        "one_bit_control": one,
        "predictions": {
            "R262-P2": "MEASURED",
            "R262-P3": "REFUTED_KT2_LIVE_THICKNESS",
            "R262-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--measure-prefix", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--classify-prefix", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS + PREFIX_PLANTS[1:], default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure_prefix:
            require(args.plant == "none", "prefix measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)),
                    "prefix measurement arguments are incomplete")
            result = r249.measure_stages(
                args.deck_root, args.frame_root, args.label, True,
                "resolved-candidate", args.expect_commit, max_kt=1)
        elif args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.prior_report, args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = r261.measure(
                args.deck_root, args.frame_root, args.record_root,
                args.prior_report, args.label, args.expect_commit,
                associate_stage_v_ratio=True)
        elif args.classify_prefix is not None:
            result = classify_prefix([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify_prefix], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
        else:
            require(args.classify is not None,
                    "classification reports are required")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, r261.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
