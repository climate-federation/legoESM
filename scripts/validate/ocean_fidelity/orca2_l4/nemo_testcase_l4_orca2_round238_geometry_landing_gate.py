#!/usr/bin/env python3
"""Gate Decision 114 on the production ORCA2 stage-geometry path."""

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
    nemo_testcase_l4_orca2_round233_geometry_unit_gate as r233,
)

PLANTS = ("none", "production-route", "face-thickness", "fold-mask")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "production-route":
        report["production_geometry"] = False
    elif plant == "face-thickness":
        report["operand_rows"]["e3v"]["unequal"] = 1
    elif plant == "fold-mask":
        report["operand_rows"]["vmask"]["unequal"] = 1

    label = report.get("label")
    require(label in ("independent", "given_nemo_entry"), "claim label moved")
    require(report.get("production_geometry") is True,
            "measurement did not execute the production geometry route")
    require(report.get("status") == "PASS_R233_GEOMETRY_UNIT_BOUNDARY",
            "inherited geometry boundary did not pass")
    require(all(report["passivity"].values()), "completed-state trace moved")
    require(tuple(report["geometry_registry"])
            == ("fmask", "tmask", "umask", "vmask"),
            "T/U/V/F geometry registry moved")
    require(all(row["unequal"] == 0
                for row in report["geometry_rows"].values()),
            "raw geometry registry is not exact")
    rows = report["operand_rows"]
    require(rows["e3v"]["support"] == 226637
            and rows["e3v"]["unequal"] == 0,
            "production live e3v is not exact")
    require(rows["vmask"]["support"] == 399600
            and rows["vmask"]["unequal"] == 0,
            "production vmask is not exact")
    require(rows["vn_adv"]["support"] == 8589,
            "external-mode support moved")
    return {
        "format": "nemo-testcase-l4-orca2-round238-geometry-landing-v1",
        "status": "PASS_R238_PRODUCTION_GEOMETRY",
        "label": label,
        "live_e3v": rows["e3v"],
        "vmask": rows["vmask"],
        "vn_adv": rows["vn_adv"],
        "zFv": rows["zFv"],
        "fold_band": report["fold_band"],
        "worktree": report["worktree"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            inherited = r233.measure(
                args.deck_root, args.frame_root, args.record_root, args.label,
                args.expect_commit, production_geometry=True)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            inherited = json.loads(args.report_in.read_text(encoding="utf-8"))
        result = classify(inherited, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, r233.GateError,
            GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
