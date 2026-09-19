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
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
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


def _card_execution(route: str = "ldf_stage3") -> dict:
    """Resolve one exact source condition from every in-scope recipe."""
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_model_config,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    require(route in {"ldf_stage3", "fct_metric_upstream"},
            f"unknown Decision-43 source route {route!r}")

    def row(config, **extra):
        values = {
            "tracer_time_integrator": config.tracer_time_integrator,
            "tracer_advection": config.tracer_advection,
            "adaptive_implicit_vertadv": bool(
                config.adaptive_implicit_vertadv),
            "gm_redi_configured": config.gm_redi is not None,
        }
        values.update(extra)
        values["executes_route"] = bool(
            (config.tracer_time_integrator == "rk3_ws"
             and config.gm_redi is not None)
            if route == "ldf_stage3" else
            (config.tracer_time_integrator == "rk3_ws"
             and config.tracer_advection == "fct2"
             and not config.adaptive_implicit_vertadv)
        )
        return values

    rows = {}
    for case in ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        config = build_nemo_testcase_card(case).recipe.model_config
        rows[case] = row(config, recipe_source="nemo_testcase_card")
    rows["NEMO-GYRE-recipe"] = row(
        build_nemo_gyre_recipe().model_config,
        recipe_source="build_nemo_gyre_recipe")
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        dino = dino_config_for_recipe(recipe)
        grid = dino_lat_lon_grid(dino, n_lon=10)
        config, _ = dino_lat_lon_model_config(grid, dino, physics=True)
        rows[f"DINO:{recipe}"] = row(
            config, outer_integrator=config.outer_integrator,
            recipe_source="dino_config_for_recipe")
    return rows


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def measure_generic_nemo_gyre(snapshot: Path) -> dict:
    """Run the card's certified three-step loop and retain exact endpoints."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        apply_nemo_gyre_surface_forcing,
        build_nemo_gyre_recipe,
        nemo_gyre_wind_forcing,
    )

    stamp = worktree_stamp()
    require(stamp.get("clean") is True,
            "generic NEMO-GYRE measurement worktree is dirty")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    recipe = build_nemo_gyre_recipe()
    config = recipe.model_config
    require(config.tracer_time_integrator == "rk3_ws",
            "generic NEMO-GYRE no longer uses the WS tracer lane")
    require(config.gm_redi is not None,
            "generic NEMO-GYRE no longer configures GM/Redi")
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, config)
    state = recipe.initial_state
    n_lat, n_lon = state.T.data.shape[:2]
    arrays: dict[str, np.ndarray] = {}
    rows = []
    fields = ("T", "S", "u", "v", "eta")
    for step in range(1, 4):
        time = (step - 1) * _NEMO_GYRE_DT_S
        state = apply_nemo_gyre_surface_forcing(
            state, recipe.z_coord, _NEMO_GYRE_DT_S, t_seconds=time)
        state = model.step(
            state, dt=_NEMO_GYRE_DT_S,
            surface_forcing=nemo_gyre_wind_forcing(
                n_lat, n_lon, t_seconds=time))
        state = jax.device_get(state)
        for field in fields:
            value = np.asarray(getattr(state, field).data)
            key = f"step{step}_{field}"
            arrays[key] = value
            rows.append({
                "row": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(np.all(np.isfinite(value))),
                "max_abs": float(np.max(np.abs(value))),
                "sha256": _array_sha256(value),
            })

    windless = recipe.initial_state
    for step in range(1, 4):
        time = (step - 1) * _NEMO_GYRE_DT_S
        windless = apply_nemo_gyre_surface_forcing(
            windless, recipe.z_coord, _NEMO_GYRE_DT_S, t_seconds=time)
        windless = model.step(windless, dt=_NEMO_GYRE_DT_S)
    windless = jax.device_get(windless)
    thermal_move = float(np.max(np.abs(
        arrays["step3_T"] - np.asarray(recipe.initial_state.T.data))))
    wind_move = float(np.max(np.abs(
        arrays["step3_u"] - np.asarray(windless.u.data))))
    certifications = {
        "all_fields_finite": all(row["finite"] for row in rows),
        "max_abs_u_below_one": float(np.max(np.abs(arrays["step3_u"]))) < 1.0,
        "max_abs_eta_below_one": (
            float(np.max(np.abs(arrays["step3_eta"]))) < 1.0),
        "thermal_forcing_nonvacuous": thermal_move > 1.0e-3,
        "wind_forcing_nonvacuous": wind_move > 1.0e-4,
    }
    np.savez(snapshot, **arrays)
    return {
        "format": "nemo-gyre-generic-card-three-step-v1",
        "status": "PASS" if all(certifications.values()) else "FAIL",
        "worktree": stamp,
        "snapshot": str(snapshot),
        "route_observation": {
            "tracer_time_integrator": config.tracer_time_integrator,
            "gm_redi_configured": config.gm_redi is not None,
            "executes_ldf_stage3_route": bool(
                config.tracer_time_integrator == "rk3_ws"
                and config.gm_redi is not None),
        },
        "rows": rows,
        "certifications": certifications,
        "thermal_move": thermal_move,
        "wind_move": wind_move,
    }


def compare_generic_nemo_gyre(
    before_report: dict,
    before_snapshot: Path,
    after_report: dict,
    after_snapshot: Path,
) -> dict:
    """Register every exact field move between two card measurements."""
    expected_format = "nemo-gyre-generic-card-three-step-v1"
    require(before_report.get("format") == expected_format,
            "generic-card before report has the wrong format")
    require(after_report.get("format") == expected_format,
            "generic-card after report has the wrong format")
    require(before_report.get("status") == "PASS",
            "generic-card before certification failed")
    require(after_report.get("status") == "PASS",
            "generic-card after certification failed")
    require(before_report.get("certifications")
            == after_report.get("certifications"),
            "generic-card certification dispositions changed")
    before = np.load(before_snapshot)
    after = np.load(after_snapshot)
    require(set(before.files) == set(after.files),
            "generic-card snapshot schemas differ")
    rows = []
    for name in sorted(before.files):
        left = np.asarray(before[name])
        right = np.asarray(after[name])
        require(left.shape == right.shape and left.dtype == right.dtype,
                f"generic-card row {name} schema differs")
        changed = left.view(np.uint64) != right.view(np.uint64)
        rows.append({
            "row": name,
            "cells": int(left.size),
            "cells_unequal": int(np.count_nonzero(changed)),
            "max_abs_move": float(np.max(np.abs(right - left))),
            "before_sha256": _array_sha256(left),
            "after_sha256": _array_sha256(right),
        })
    moved = [row for row in rows if row["cells_unequal"]]
    return {
        "format": "nemo-gyre-generic-card-three-step-comparison-v1",
        "status": "PASS" if moved else "FAIL",
        "worktree": worktree_stamp(),
        "before_commit": before_report["worktree"]["commit"],
        "after_commit": after_report["worktree"]["commit"],
        "certifications_unchanged": True,
        "rows": rows,
        "moved_rows": moved,
        "moved_row_count": len(moved),
    }


def _read_moved_row_registry(path: Path) -> tuple[str, ...]:
    require(path.is_file(), f"missing moved-row registry {path}")
    rows = tuple(
        line.split("\t", 1)[0]
        for line in path.read_text().splitlines() if line.strip())
    require(rows, "moved-row registry is empty")
    require(len(set(rows)) == len(rows),
            "moved-row registry contains duplicate names")
    return rows


def evaluate(
    comparison: dict,
    before_day_gap: dict,
    after_day_gap: dict,
    *,
    expected_candidate_commit: str,
    route: str = "ldf_stage3",
    measured_cards: tuple[str, ...] = (),
    registered_rows: tuple[str, ...] = (),
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
    elif plant == "missing-moved-registry":
        registered_rows = registered_rows[1:]
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
    registered_set = set(registered_rows)
    moved_set = {row["row"] for row in moved}
    require(len(registered_set) == len(registered_rows),
            "explicit moved-row registry contains duplicate names")
    missing_registered_rows = sorted(moved_set - registered_set)
    unexpected_registered_rows = sorted(registered_set - moved_set)

    cards = _card_execution(route)
    require(cards["GYRE-zco"]["executes_route"],
            "GYRE unexpectedly does not execute the candidate statement")
    unknown_measurements = sorted(set(measured_cards) - set(cards))
    require(not unknown_measurements,
            f"measurement registry names unknown cards {unknown_measurements}")
    executing_cards = sorted(
        name for name, row in cards.items() if row["executes_route"])
    unmeasured_executing_cards = sorted(
        set(executing_cards) - {"GYRE-zco"} - set(measured_cards))
    dino_shared = any(
        row["executes_route"]
        for name, row in cards.items() if name.startswith("DINO:"))

    criteria = {
        "day30_T_rms_decreases": after30["rms_T"] < before30["rms_T"],
        "first_over_bar_not_earlier": first_not_earlier,
        "no_kt1_at_bar_row_leaves": not kt1_losses,
        "all_moved_rows_registered": bool(
            moved and not missing_registered_rows
            and not unexpected_registered_rows),
        "dino_measurement_required": dino_shared,
        "dino_statement_not_executed": not dino_shared,
        "all_executing_cards_measured": not unmeasured_executing_cards,
    }
    # Decision 43 requires a DINO before/after measurement only when the exact
    # source condition executes.  This route does not: both shipped DINO cards
    # use the Euler tracer lane.  Keep both booleans so the waiver is explicit.
    admissible = bool(
        criteria["day30_T_rms_decreases"]
        and criteria["first_over_bar_not_earlier"]
        and criteria["no_kt1_at_bar_row_leaves"]
        and criteria["all_moved_rows_registered"]
        and criteria["dino_statement_not_executed"]
        and criteria["all_executing_cards_measured"])

    return {
        "format": FORMAT,
        "worktree": worktree_stamp(),
        "status": "PASS" if admissible else "FAIL",
        "plant": plant,
        "route": route,
        "measured_cards": sorted(set(measured_cards)),
        "executing_cards": executing_cards,
        "unmeasured_executing_cards": unmeasured_executing_cards,
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
        "registered_row_count": len(registered_rows),
        "missing_registered_rows": missing_registered_rows,
        "unexpected_registered_rows": unexpected_registered_rows,
        "card_execution": cards,
        "criteria": criteria,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--before-day-gap", type=Path)
    parser.add_argument("--after-day-gap", type=Path)
    parser.add_argument("--expect-candidate-commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generic-measure-snapshot", type=Path)
    parser.add_argument("--generic-before-report", type=Path)
    parser.add_argument("--generic-before-snapshot", type=Path)
    parser.add_argument("--generic-after-report", type=Path)
    parser.add_argument("--generic-after-snapshot", type=Path)
    parser.add_argument("--moved-row-registry", type=Path)
    parser.add_argument(
        "--route", choices=("ldf_stage3", "fct_metric_upstream"),
        default="ldf_stage3")
    parser.add_argument(
        "--measured-card", action="append", default=[],
        help="executing non-primary card discharged by a separate measurement")
    parser.add_argument("--plant", choices=(
        "day30-no-improvement", "earlier-first-over-bar", "kt1-at-bar-loss",
        "missing-moved-registry"))
    args = parser.parse_args(argv)
    try:
        if args.generic_measure_snapshot is not None:
            report = measure_generic_nemo_gyre(args.generic_measure_snapshot)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"STATUS {report['status']}: generic NEMO-GYRE three-step card")
            return 0 if report["status"] == "PASS" else 1
        generic_compare = (
            args.generic_before_report, args.generic_before_snapshot,
            args.generic_after_report, args.generic_after_snapshot)
        if any(value is not None for value in generic_compare):
            require(all(value is not None for value in generic_compare),
                    "generic comparison requires all four report/snapshot paths")
            report = compare_generic_nemo_gyre(
                _read(args.generic_before_report), args.generic_before_snapshot,
                _read(args.generic_after_report), args.generic_after_snapshot)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"STATUS {report['status']}: generic NEMO-GYRE "
                  f"moved_rows={report['moved_row_count']}")
            return 0 if report["status"] == "PASS" else 1
        require(all(value is not None for value in (
            args.comparison, args.before_day_gap, args.after_day_gap,
            args.expect_candidate_commit, args.moved_row_registry)),
            "Decision-43 admission requires comparison, day gaps, commit, "
            "and --moved-row-registry")
        report = evaluate(
            _read(args.comparison),
            _read(args.before_day_gap),
            _read(args.after_day_gap),
            expected_candidate_commit=args.expect_candidate_commit,
            route=args.route,
            measured_cards=tuple(args.measured_card),
            registered_rows=_read_moved_row_registry(
                args.moved_row_registry),
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
