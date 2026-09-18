#!/usr/bin/env python3
"""Decision-43 admission gate for magnitude-first GYRE landings.

This gate consumes existing certified artifacts.  It does not run either
model and it does not replace the oracle-relative ladder comparison.  It
applies the temporary Decision-43 policy to that comparison and to the
day-gap reports, then resolves whether the changed production statement is
reachable on the other named cards.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp


FORMAT = "nemo-testcase-l2-gyre-decision43-v1"
COMPARISON_FORMAT = "legoesm-ocean-oracle-relative-move-gate-v3"
DAY_GAP_FORMAT = "gyre-year-owners-day-gap-v1"


class GateError(RuntimeError):
    """A fail-closed artifact or admission error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing artifact {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: top level is not an object")
    return value


def _day30(report: dict, label: str) -> dict:
    require(report.get("format") == DAY_GAP_FORMAT,
            f"{label}: unexpected format {report.get('format')!r}")
    rows = report.get("rows")
    require(isinstance(rows, list), f"{label}: rows are absent")
    matches = [row for row in rows if row.get("day") == 30]
    require(len(matches) == 1, f"{label}: expected exactly one day-30 row")
    value = matches[0].get("rms_T")
    require(isinstance(value, (int, float)),
            f"{label}: day-30 rms_T is not numeric")
    return matches[0]


def _first_over_bar_kt(value: object, label: str) -> int | None:
    if value is None:
        return None
    require(isinstance(value, dict), f"{label}: first-over-bar is not an object")
    kt = value.get("kt")
    require(isinstance(kt, int) and kt >= 1,
            f"{label}: invalid first-over-bar kt {kt!r}")
    return kt


def _card_execution() -> dict:
    """Resolve the exact source condition on each in-scope shipped card."""
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_model_config,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    rows = {}
    for case in ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        config = build_nemo_testcase_card(case).recipe.model_config
        rows[case] = {
            "tracer_time_integrator": config.tracer_time_integrator,
            "gm_redi_configured": config.gm_redi is not None,
            "executes_route": bool(
                config.tracer_time_integrator == "rk3_ws"
                and config.gm_redi is not None),
        }
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        dino = dino_config_for_recipe(recipe)
        grid = dino_lat_lon_grid(dino, n_lon=10)
        config, _ = dino_lat_lon_model_config(grid, dino, physics=True)
        rows[f"DINO:{recipe}"] = {
            "outer_integrator": config.outer_integrator,
            "tracer_time_integrator": config.tracer_time_integrator,
            "gm_redi_configured": config.gm_redi is not None,
            "executes_route": bool(
                config.tracer_time_integrator == "rk3_ws"
                and config.gm_redi is not None),
        }
    return rows


def evaluate(
    comparison: dict,
    before_day_gap: dict,
    after_day_gap: dict,
    *,
    expected_candidate_commit: str,
    plant: str | None = None,
) -> dict:
    comparison = copy.deepcopy(comparison)
    before_day_gap = copy.deepcopy(before_day_gap)
    after_day_gap = copy.deepcopy(after_day_gap)

    require(comparison.get("format") == COMPARISON_FORMAT,
            f"unexpected comparison format {comparison.get('format')!r}")
    require(comparison.get("n_certified_rows_compared") == 70,
            "Decision-43 endpoint admission requires all 70 trajectory rows")
    require(comparison.get("row_filter_applied") is False,
            "filtered comparisons cannot admit a landing")
    require(comparison.get("plant") is None,
            "a planted ladder comparison cannot admit a landing")

    after_stamp = after_day_gap.get("worktree", {})
    comparison_stamp = comparison.get("worktree", {})
    require(after_stamp.get("clean") is True,
            "candidate day-gap worktree is not clean")
    require(comparison_stamp.get("clean") is True,
            "candidate comparison worktree is not clean")
    require(after_stamp.get("commit") == expected_candidate_commit,
            "candidate day-gap commit does not match --expect-candidate-commit")
    require(comparison_stamp.get("commit") == expected_candidate_commit,
            "candidate comparison commit does not match --expect-candidate-commit")

    before30 = _day30(before_day_gap, "before day gap")
    after30 = _day30(after_day_gap, "after day gap")
    if plant == "day30-no-improvement":
        after30["rms_T"] = before30["rms_T"]
    elif plant == "earlier-first-over-bar":
        comparison["first_over_bar_candidate"] = {"kt": 1, "fields": ["T"]}
    elif plant == "kt1-at-bar-loss":
        comparison.setdefault("row_status_changes", []).append({
            "row": "GYRE-zco.kt1.before.T",
            "reference": "AT-BAR",
            "candidate": "DEBT",
        })
    elif plant is not None:
        raise GateError(f"unknown plant {plant!r}")

    before_kt = _first_over_bar_kt(
        comparison.get("first_over_bar_reference"), "reference")
    after_kt = _first_over_bar_kt(
        comparison.get("first_over_bar_candidate"), "candidate")
    first_not_earlier = bool(
        after_kt is None or (before_kt is not None and after_kt >= before_kt))

    kt1_losses = [
        row for row in comparison.get("row_status_changes", [])
        if ".kt1." in str(row.get("row"))
        and row.get("reference") == "AT-BAR"
        and row.get("candidate") == "DEBT"
    ]
    field_moves = comparison.get("field_moves")
    require(isinstance(field_moves, list), "comparison has no field_moves table")
    moved = [
        row for row in field_moves
        if row.get("max_previous_legoesm_field_move") != 0
    ]
    require(all(isinstance(row.get("row"), str) for row in moved),
            "a moved comparison row has no name")
    require(len({row["row"] for row in moved}) == len(moved),
            "moved-row registry contains duplicate names")

    cards = _card_execution()
    require(cards["GYRE-zco"]["executes_route"],
            "GYRE unexpectedly does not execute the candidate statement")
    dino_shared = any(
        row["executes_route"]
        for name, row in cards.items() if name.startswith("DINO:"))

    criteria = {
        "day30_T_rms_decreases": after30["rms_T"] < before30["rms_T"],
        "first_over_bar_not_earlier": first_not_earlier,
        "no_kt1_at_bar_row_leaves": not kt1_losses,
        "all_moved_rows_registered": len(moved) > 0,
        "dino_measurement_required": dino_shared,
        "dino_statement_not_executed": not dino_shared,
    }
    # Decision 43 requires a DINO before/after measurement only when the exact
    # source condition executes.  This route does not: both shipped DINO cards
    # use the Euler tracer lane.  Keep both booleans so the waiver is explicit.
    admissible = bool(
        criteria["day30_T_rms_decreases"]
        and criteria["first_over_bar_not_earlier"]
        and criteria["no_kt1_at_bar_row_leaves"]
        and criteria["all_moved_rows_registered"]
        and criteria["dino_statement_not_executed"])

    return {
        "format": FORMAT,
        "worktree": worktree_stamp(),
        "status": "PASS" if admissible else "FAIL",
        "plant": plant,
        "expected_candidate_commit": expected_candidate_commit,
        "before_day30_T_rms": before30["rms_T"],
        "after_day30_T_rms": after30["rms_T"],
        "improvement_factor": (
            before30["rms_T"] / after30["rms_T"]
            if after30["rms_T"] != 0 else float("inf")),
        "first_over_bar_reference": comparison.get("first_over_bar_reference"),
        "first_over_bar_candidate": comparison.get("first_over_bar_candidate"),
        "kt1_at_bar_losses": kt1_losses,
        "moved_row_count": len(moved),
        "moved_rows": moved,
        "card_execution": cards,
        "criteria": criteria,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--before-day-gap", type=Path, required=True)
    parser.add_argument("--after-day-gap", type=Path, required=True)
    parser.add_argument("--expect-candidate-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=(
        "day30-no-improvement", "earlier-first-over-bar", "kt1-at-bar-loss"))
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            _read(args.comparison),
            _read(args.before_day_gap),
            _read(args.after_day_gap),
            expected_candidate_commit=args.expect_candidate_commit,
            plant=args.plant,
        )
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (GateError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    if args.plant:
        if report["status"] == "FAIL":
            print(f"STATUS PLANT-FIRED: {args.plant}")
            return 1
        print(f"STATUS REFUSE: plant {args.plant} did not make the gate fail")
        return 2
    print(f"STATUS {report['status']}: moved_rows={report['moved_row_count']} "
          f"day30_T={report['before_day30_T_rms']:.17e}->"
          f"{report['after_day30_T_rms']:.17e}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
