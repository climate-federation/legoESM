#!/usr/bin/env python3
"""Classify round 234's complete external-mode association candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PLANTS = ("none", "association-bit", "label-coverage", "false-owner", "endpoint")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(association: dict, reports: list[dict], *, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    association = copy.deepcopy(association)
    reports = copy.deepcopy(reports)
    if plant == "association-bit":
        association["post_association_rows"]["eta"]["bit_exact"] = False
    elif plant == "label-coverage":
        reports.pop()

    require(association.get("status") == "MEASURED_R146_BOUNDARY_ASSOCIATION",
            "association measurement status moved")
    post = association["post_association_rows"]
    require(tuple(sorted(post)) == (
        "depth_u", "depth_v", "eta", "inverse_u", "inverse_v", "u", "v"),
        "seven-array association registry moved")
    require(all(row["bit_exact"] for row in post.values()),
            "complete association is not bit-exact against NEMO")

    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "claim-label coverage moved")
    if plant == "false-owner":
        by_label["independent"]["operand_rows"]["vn_adv"]["unequal"] = 35
    elif plant == "endpoint":
        by_label["independent"]["fold_band"]["S"]["max_abs"] = 0.0012

    rows = {}
    for label, report in by_label.items():
        require(report.get("status") == "PASS_R233_GEOMETRY_UNIT_BOUNDARY",
                f"{label}: boundary measurement did not pass")
        require(all(report["passivity"].values()),
                f"{label}: passive trace moved state")
        require(all(row["unequal"] == 0
                    for row in report["geometry_rows"].values()),
                f"{label}: private exact geometry moved")
        operand = report["operand_rows"]
        require(operand["vn_adv"]["support"] == 8589,
                f"{label}: vn_adv support moved")
        rows[label] = {
            "vn_adv_unequal": operand["vn_adv"]["unequal"],
            "vn_adv_max_abs": operand["vn_adv"]["max_abs"],
            "zvb_unequal": operand["zvb"]["unequal"],
            "zFv_unequal": operand["zFv"]["unequal"],
            "fold_T_max_abs": report["fold_band"]["T"]["max_abs"],
            "fold_S_max_abs": report["fold_band"]["S"]["max_abs"],
        }

    owner = all(row["vn_adv_unequal"] <= 35 for row in rows.values())
    endpoint = all(
        row["fold_S_max_abs"] <= 0.0012
        and row["fold_T_max_abs"] <= 0.0014
        for row in rows.values())
    if not owner:
        status = "HELD_R234_ASSOCIATION_NULL_EXTERNAL_MODE_DEBT"
        open_item = (
            "Build the registered per-substep operand table; the exact complete "
            "association did not own vn_adv.")
    elif not endpoint:
        status = "HELD_R234_ASSOCIATION_EXPOSES_LATER_DEBT"
        open_item = "Walk the first downstream statement after exact vn_adv."
    else:
        status = "QUALIFIED_R234_COMPLETE_ASSOCIATION"
        open_item = "Run the full Decision-96 landing gates before promotion."

    require(plant == "none", f"{plant} plant fired")
    return {
        "format": "nemo-testcase-l4-orca2-round234-complete-association-v1",
        "status": status,
        "association_rows": post,
        "rows": rows,
        "predictions": {
            "R234-P1": "CONFIRMED",
            "R234-P2": "CONFIRMED",
            "R234-P3": "CONFIRMED" if owner else "REFUTED_VN_ADV_REMAINS_GLOBAL",
            "R234-P4": ("CONFIRMED" if endpoint else
                         "REFUTED_ENDPOINT" if owner else
                         "NOT_ACTIVATED_PREREQUISITE_R234-P3"),
            "R234-P5": ("ACTIVATED" if owner and endpoint else
                         "NOT_ACTIVATED_PREREQUISITE"),
        },
        "open": open_item,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--given", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(
            json.loads(args.association.read_text(encoding="utf-8")),
            [json.loads(args.independent.read_text(encoding="utf-8")),
             json.loads(args.given.read_text(encoding="utf-8"))],
            plant=args.plant,
        )
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
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
