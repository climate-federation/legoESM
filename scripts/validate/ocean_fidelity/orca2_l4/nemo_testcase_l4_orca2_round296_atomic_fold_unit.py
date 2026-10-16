#!/usr/bin/env python3
"""Classify round 296's complete OMT-4 fold unit on its finite prefix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
)

LABELS = ("independent", "given_nemo_entry")
FIELDS = r249.FIELDS
FLOOR = float(r249.FLOOR)
REFUSAL = "raw-mesh e3w_int must contain only finite values > 0"
PLANTS = (
    "none", "label-coverage", "raw-depth-closure", "raw-mask-closure",
    "passivity", "boundary-order", "refusal", "growth", "one-bit",
)


class GateError(RuntimeError):
    """The frozen atomic-unit evidence or a known-answer control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _boundary(report: dict, kt: int, stage: int) -> dict:
    rows = [row for row in report["boundaries"]
            if (row["kt"], row["stage"]) == (kt, stage)]
    require(len(rows) == 1, f"missing unique kt={kt} stage={stage} row")
    return rows[0]


def _first_growth(report: dict) -> dict | None:
    candidates = []
    for region in ("fold_band", "interior"):
        candidates.extend(r249._growth_rows(report["boundaries"], region))
    candidates.sort(key=lambda row: (row["kt"], row["stage"], row["region"]))
    return next((row for row in candidates
                 if row["previous_max_abs"] > FLOOR and row["over_10x"]),
                None)


def classify(prefix_paths: list[Path], refusal_paths: list[Path],
             raw_depth_path: Path, raw_mask_path: Path, *,
             plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    prefixes = [_load(path) for path in prefix_paths]
    refusal_logs = [path.read_text(encoding="utf-8") for path in refusal_paths]
    raw_depth = _load(raw_depth_path)
    raw_mask = _load(raw_mask_path)
    prefixes, refusal_logs = copy.deepcopy((prefixes, refusal_logs))

    if plant == "label-coverage":
        prefixes.pop()
    by_label = {report["label"]: report for report in prefixes}
    require(set(by_label) == set(LABELS), "both OMT-4 labels are required")
    commits = {report["worktree"]["commit"] for report in prefixes}
    require(len(commits) == 1, "finite prefixes came from different trees")

    if plant == "raw-depth-closure":
        raw_depth["direct_arms"]["raw_hv_only"]["face_depth_v_mid"][
            "complete"]["unequal"] = 1
    depth_row = raw_depth["direct_arms"]["raw_hv_only"][
        "face_depth_v_mid"]["complete"]
    require(depth_row["unequal"] == 0 and depth_row["max_abs"] == 0.0,
            "raw hv_0 local closure moved")

    if plant == "raw-mask-closure":
        raw_mask["single_substitution"]["ssvmask"]["complete"][
            "unequal"] = 1
    mask_row = raw_mask["single_substitution"]["ssvmask"]["complete"]
    require(mask_row["unequal"] == 0 and mask_row["max_abs"] == 0.0,
            "raw ssvmask local closure moved")

    expected_order = [(kt, stage) for kt in range(1, 8)
                      for stage in range(4)]
    if plant == "boundary-order":
        by_label["independent"]["boundaries"].pop()
    if plant == "passivity":
        by_label["independent"]["passivity"][0]["fields"]["T"] = False

    summaries = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R249_STAGE_TABLE"
                and report["unit"] is True
                and report["operand_route"] == "resolved-candidate"
                and report["execution"] == "production-jit-cpu-fp64-x64-libm",
                f"{label}: atomic-unit identity moved")
        actual_order = [(row["kt"], row["stage"])
                        for row in report["boundaries"]]
        require(actual_order == expected_order,
                f"{label}: kt=1..7 finite-prefix order moved")
        require(len(report["passivity"]) == 7
                and all(all(row["fields"].values())
                        for row in report["passivity"]),
                f"{label}: passive observer moved completed state")
        for row in report["boundaries"]:
            for field in FIELDS:
                require(row["fields"][field]["complete"]["nonfinite"] == 0,
                        f"{label}: finite prefix contains non-finite {field}")
        first = _first_growth(report)
        if plant == "growth" and label == "independent":
            first = None
        require(first is not None, f"{label}: no post-floor growth boundary")
        summaries[label] = {
            "first_post_floor_over_10x": first,
            "first_boundary_fields": _boundary(
                report, first["kt"], first["stage"])["fields"],
            "kt7_stage3": _boundary(report, 7, 3)["fields"],
        }

    if plant == "refusal":
        refusal_logs[0] = refusal_logs[0].replace(REFUSAL, "planted refusal")
    require(len(refusal_logs) == 2, "both terminal refusal logs are required")
    for label, log in zip(LABELS, refusal_logs, strict=True):
        require("PROGRESS round249 " + label + " unit=1 kt=7" in log,
                f"{label}: terminal log lacks completed kt=7")
        require("PROGRESS round249 " + label + " unit=1 kt=8" not in log,
                f"{label}: terminal log unexpectedly completed kt=8")
        require(REFUSAL in log, f"{label}: live-thickness refusal moved")

    one = np.asarray([1.0], np.float64)
    one_bit = r249._score(one, np.nextafter(one, np.inf))
    if plant == "one-bit":
        one_bit["unequal"] = 0
    require(one_bit["unequal"] == 1, "one-bit control did not fire")

    result = {
        "format": "nemo-testcase-l4-orca2-round296-atomic-unit-v1",
        "status": "HELD_R296_LIVE_THICKNESS_AT_KT8",
        "claim": "OMT-4, both labels",
        "worktree_commit": commits.pop(),
        "local_closures": {
            "raw_hv_0_substep1_depth_v": depth_row,
            "raw_ssvmask_substep1_v_update": mask_row,
        },
        "labels": summaries,
        "terminal": {
            "last_completed_kt": 7,
            "next_kt": 8,
            "message": REFUSAL,
        },
        "one_bit_control": one_bit,
        "predictions": {
            "R296-P1": "CONFIRMED",
            "R296-P2": "CONFIRMED",
            "R296-P3": "REFUTED_LIVE_THICKNESS_AT_KT8",
            "R296-P4": "CONFIRMED_FIRST_GROWTH_SELECTED",
            "R296-P5": "NOT_REACHED",
            "R296-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }
    require(plant == "none", f"{plant} plant stayed green")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", nargs=2, type=Path, required=True)
    parser.add_argument("--refusal-log", nargs=2, type=Path, required=True)
    parser.add_argument("--raw-depth", type=Path, required=True)
    parser.add_argument("--raw-mask", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = classify(
            args.prefix, args.refusal_log, args.raw_depth, args.raw_mask,
            plant=args.plant)
    except (GateError, r249.GateError, OSError, KeyError, TypeError,
            ValueError, IndexError, StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
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
