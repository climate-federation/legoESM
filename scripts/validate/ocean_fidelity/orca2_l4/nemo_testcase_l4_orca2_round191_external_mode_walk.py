#!/usr/bin/env python3
"""Continue the independent rung-0 external-mode walk from exact forcing."""

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
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)


INHERITED_PLANTS = tuple(plant for plant in r178.PLANTS if plant != "none")
PLANTS = ("none", *INHERITED_PLANTS, "exact-arm-order", "exact-arm-selector")


class GateError(RuntimeError):
    """The exact-forcing walk no longer satisfies its frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _row_id(row: dict[str, object]) -> str:
    if "substep" in row:
        return f"{int(row['substep']):03d}:{row['name']}"
    return str(row["name"])


def _first_over_floor(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return next((row for row in rows if not bool(row["at_floor"])), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)

    if plant in INHERITED_PLANTS:
        # Reuse the round-178 controls that own record placement, bit mutation,
        # source order, arm identity, and endpoint sensitivity.
        r178.classify(report, plant)

    arms = report.get("source_rows_by_arm")
    require(isinstance(arms, dict), "round-178 replay did not publish arm rows")
    require(list(arms) == list(r178.ARM_ORDER), "exact replay arm order moved")
    rows = arms["slow_only"]
    history_rows = arms["slow_and_history"]
    require(isinstance(rows, list) and rows, "exact-forcing rows are empty")
    require(isinstance(history_rows, list) and history_rows,
            "exact-forcing/history rows are empty")

    frozen_first = report.get("round191_frozen_first_over_floor")
    if frozen_first is None:
        frozen_first = copy.deepcopy(_first_over_floor(rows))
        report["round191_frozen_first_over_floor"] = frozen_first

    if plant == "exact-arm-order":
        rows[0], rows[1] = rows[1], rows[0]
    elif plant == "exact-arm-selector":
        rows[0]["at_floor"] = False

    expected_order = list(r178.ENTRY_ORDER)
    for substep in range(1, 66):
        expected_order.extend(
            f"{substep:03d}:{name}" for name in r178.SUBSTEP_ORDER)
    expected_order.extend(r178.EXIT_ORDER)
    row_order = [_row_id(row) for row in rows]
    require(row_order == expected_order[:len(row_order)],
            "exact-forcing source order moved")
    require(all(bool(row["at_floor"]) for row in rows[:-1]),
            "walk crossed an earlier exact-forcing debt")

    first = _first_over_floor(rows)
    require(first == frozen_first, "exact-forcing first-debt selector moved")
    require(first is None or first == rows[-1],
            "walk continued past exact-forcing first debt")

    history_null = (
        report["endpoint_scores"]["slow_only"]["trace_digest"]
        == report["endpoint_scores"]["slow_and_history"]["trace_digest"]
    )
    require(history_null == bool(report["history_arm_null"]),
            "history null-arm discriminator moved")
    if history_null:
        require(rows == history_rows,
                "history endpoint is null but its source rows moved")

    report["round191"] = {
        "claim_label": "independent hierarchy rung 0",
        "source_rows": rows,
        "source_order": row_order,
        "first_over_floor": first,
        "history_arm_null": history_null,
        "compiled_source": {
            "midpoint_and_transport": (
                "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:502-570"),
            "continuity": (
                "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:580-591"),
            "pressure_and_velocity": (
                "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:647-728"),
        },
        "prediction_ledger": {
            "R191-P1": "CONFIRMED",
            "R191-P2": (
                "CONFIRMED" if first is not None
                and first.get("name") == "ssh_after"
                and int(first.get("substep", -1)) == 1 else "REFUTED"),
            "R191-P3": "CONFIRMED" if history_null else "REFUTED",
            "R191-P4": (
                "CONFIRMED" if first is not None
                and first.get("name") == "ssh_after" else "REFUTED"),
            "R191-P5": "CONFIRMED_MEASUREMENT_ONLY",
            "R191-P6": "CONFIRMED",
        },
    }
    report["status"] = "HELD_EXACT_FORCING_FIRST_DOWNSTREAM_DEBT"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.expect_commit)), "measurement arguments missing")
            result = r178.measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.expect_commit)
        else:
            require(args.report_in is not None, "--report-in is required")
            result = json.loads(args.report_in.read_text(encoding="utf-8"))
        result = classify(result, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r178.GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
            return 2
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
