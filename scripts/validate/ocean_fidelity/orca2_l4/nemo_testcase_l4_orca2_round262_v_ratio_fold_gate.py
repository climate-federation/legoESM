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
        independent["zFv"]["unequal"] = 1
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
        require(report["zFv"]["unequal"] == 0,
                f"{label}: stage-two V transport is non-bit")
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
        "status": "PASS_R262_V_RATIO_FOLD_SOURCE_REPLAY",
        "labels": summaries,
        "predictions": {
            "R262-P1": "CONFIRMED",
            "R262-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.prior_report, args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = r261.measure(
                args.deck_root, args.frame_root, args.record_root,
                args.prior_report, args.label, args.expect_commit)
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
