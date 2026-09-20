#!/usr/bin/env python3
"""Fail-closed Round-130 ranking of held GYRE patches by year response.

The gate consumes only already-produced, commit-stamped artifacts.  A small
registry names every single-patch arm, the exact local-proof assertions to
check, and the complete set of moved ladder rows.  It therefore cannot turn a
partial candidate set, an unproved arm, or an abbreviated movement table into
a ranking claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp


FORMAT = "nemo-testcase-l2-gyre-round130-year-patch-ranking-v1"
REGISTRY_FORMAT = "nemo-testcase-l2-gyre-round130-registry-v1"
DAY_GAP_FORMAT = "gyre-year-owners-day-gap-v1"
COMPARISON_FORMAT = "legoesm-ocean-oracle-relative-move-gate-v3"
CARD_COMPARISON_FORMAT = "nemo-gyre-generic-card-three-step-comparison-v1"
YEAR_DAYS = (30, 60, 90, 120, 180, 240, 300, 360)
MONTH_DAYS = tuple(range(1, 31))
CANDIDATES = (
    "r62_coeff",
    "r88_kaa",
    "r89_assign",
    "r97_rhs",
    "r99_wclock",
    "r105_shear",
    "r109_handoff",
    "r112_fct",
)
BASELINE_YEAR_T = {
    30: 6.890484901489568e-5,
    60: 1.9329973681936875e-4,
    90: 1.8645021144913585e-3,
    120: 1.0501256819510476e-3,
    180: 3.580551011866709e-3,
    240: 1.6446741930292448e-2,
    300: 1.3597404177319843e-2,
    360: 1.1223573910167267e-2,
}


def _route_executes(config: object, route: str) -> bool:
    """Return whether a resolved recipe reaches the candidate statement."""
    if route == "rk3_ws_vector_stage1_handoff":
        return bool(
            getattr(config, "tracer_time_integrator", None) == "rk3_ws"
            and getattr(config, "momentum_advection", None)
            == "vector_invariant"
        )
    raise GateError(f"unknown Round-130 execution route {route!r}")


def _card_execution(route: str) -> dict[str, dict[str, Any]]:
    """Derive the complete shipped-card execution census from recipes."""
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_model_config,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    def row(config: object, source: str) -> dict[str, Any]:
        return {
            "recipe_source": source,
            "tracer_time_integrator": getattr(
                config, "tracer_time_integrator", None),
            "momentum_advection": getattr(config, "momentum_advection", None),
            "outer_integrator": getattr(config, "outer_integrator", None),
            "executes_route": _route_executes(config, route),
        }

    rows: dict[str, dict[str, Any]] = {}
    for case in ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        rows[case] = row(
            build_nemo_testcase_card(case).recipe.model_config,
            "build_nemo_testcase_card",
        )
    rows["NEMO-GYRE-recipe"] = row(
        build_nemo_gyre_recipe().model_config, "build_nemo_gyre_recipe")
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        dino = dino_config_for_recipe(recipe)
        grid = dino_lat_lon_grid(dino, n_lon=10)
        config, _ = dino_lat_lon_model_config(grid, dino, physics=True)
        rows[f"DINO:{recipe}"] = row(config, "dino_config_for_recipe")
    return rows


class GateError(RuntimeError):
    """An input is incomplete, stale, or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _read(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: top level is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(base: Path, value: object, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}: path is absent")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _pointer(value: Any, pointer: str, label: str) -> Any:
    require(isinstance(pointer, str) and pointer.startswith("/"),
            f"{label}: invalid JSON pointer {pointer!r}")
    current = value
    for raw in pointer.split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            require(key.isdigit() and int(key) < len(current),
                    f"{label}: list pointer component {key!r} is absent")
            current = current[int(key)]
        else:
            require(isinstance(current, dict) and key in current,
                    f"{label}: object pointer component {key!r} is absent")
            current = current[key]
    return current


def _stamp_commit(report: dict[str, Any], label: str) -> str:
    stamp = report.get("worktree")
    require(isinstance(stamp, dict), f"{label}: worktree stamp absent")
    require(stamp.get("clean") is True, f"{label}: producer tree was dirty")
    commit = stamp.get("commit")
    require(isinstance(commit, str) and len(commit) == 40,
            f"{label}: producer commit absent")
    return commit


def _year_commit(report: dict[str, Any], label: str) -> str:
    admission = report.get("member_admission")
    require(isinstance(admission, dict), f"{label}: member admission absent")
    member = admission.get("record")
    require(isinstance(member, dict), f"{label}: member manifest absent")
    stamp = member.get("worktree")
    require(isinstance(stamp, dict) and stamp.get("clean") is True,
            f"{label}: member producer tree was dirty")
    commit = stamp.get("commit")
    require(isinstance(commit, str) and len(commit) == 40,
            f"{label}: member producer commit absent")
    return commit


def _rows(report: dict[str, Any], days: tuple[int, ...], label: str) -> dict[int, dict]:
    require(report.get("format") == DAY_GAP_FORMAT,
            f"{label}: unexpected format {report.get('format')!r}")
    rows = report.get("rows")
    require(isinstance(rows, list), f"{label}: rows absent")
    require(tuple(row.get("day") for row in rows) == days,
            f"{label}: expected exactly days {list(days)}")
    result = {}
    for row in rows:
        value = row.get("rms_T")
        require(isinstance(value, (int, float)) and np.isfinite(value),
                f"{label}: day {row.get('day')} T rms is not finite")
        result[int(row["day"])] = row
    return result


def _local_proof(entry: dict[str, Any], base: Path, commit: str) -> dict[str, Any]:
    local = entry.get("local_proof")
    require(isinstance(local, dict), f"{entry.get('id')}: local proof absent")
    artifacts_spec = local.get("artifacts")
    require(isinstance(artifacts_spec, dict) and artifacts_spec,
            f"{entry.get('id')}: local artifacts absent")
    artifacts = {}
    artifact_rows = []
    for name, spec in artifacts_spec.items():
        require(isinstance(spec, dict), f"{entry.get('id')}:{name}: bad artifact")
        path = _path(base, spec.get("path"), f"{entry.get('id')}:{name}")
        expected_sha = spec.get("sha256")
        require(isinstance(expected_sha, str) and len(expected_sha) == 64,
                f"{entry.get('id')}:{name}: sha256 absent")
        actual_sha = _sha256(path)
        require(actual_sha == expected_sha,
                f"{entry.get('id')}:{name}: digest moved")
        report = _read(path)
        artifact_commit = spec.get("commit", commit)
        require(isinstance(artifact_commit, str) and len(artifact_commit) == 40,
                f"{entry.get('id')}:{name}: expected commit is invalid")
        if isinstance(report.get("worktree"), dict):
            require(_stamp_commit(report, f"{entry.get('id')}:{name}")
                    == artifact_commit,
                    f"{entry.get('id')}:{name}: commit mismatch")
        artifacts[name] = report
        artifact_rows.append({"name": name, "path": str(path),
                              "sha256": actual_sha,
                              "expected_commit": artifact_commit})

    checks = local.get("checks")
    require(isinstance(checks, list) and checks,
            f"{entry.get('id')}: no local assertions registered")
    checked = []
    for index, check in enumerate(checks):
        label = f"{entry.get('id')}:check[{index}]"
        require(isinstance(check, dict), f"{label}: not an object")
        artifact = check.get("artifact")
        require(artifact in artifacts, f"{label}: unknown artifact {artifact!r}")
        value = _pointer(artifacts[artifact], check.get("pointer"), label)
        kind = check.get("kind", "equals")
        expected = check.get("equals")
        if kind == "equals":
            require(value == expected,
                    f"{label}: {value!r} != {expected!r}")
            count = 1
        elif kind in {"all-field", "filtered-all-field"}:
            require(isinstance(value, list) and value,
                    f"{label}: all-field target is empty or not a list")
            selected = value
            if kind == "filtered-all-field":
                where = check.get("where")
                require(isinstance(where, dict) and where,
                        f"{label}: filtered assertion has no selector")
                selected = [row for row in value if isinstance(row, dict)
                            and all(row.get(key) == wanted
                                    for key, wanted in where.items())]
                require(selected,
                        f"{label}: filtered assertion selected no rows")
            field = check.get("field")
            require(isinstance(field, str) and all(
                isinstance(row, dict) and row.get(field) == expected
                for row in selected),
                f"{label}: not every selected {field!r} equals {expected!r}")
            count = len(selected)
        else:
            raise GateError(f"{label}: unknown check kind {kind!r}")
        checked.append({"artifact": artifact, "pointer": check["pointer"],
                        "kind": kind, "where": check.get("where"),
                        "assertions": count})

    plant = local.get("plant")
    require(isinstance(plant, dict), f"{entry.get('id')}: plant assertion absent")
    artifact = plant.get("artifact")
    require(artifact in artifacts,
            f"{entry.get('id')}: plant names unknown artifact {artifact!r}")
    value = _pointer(artifacts[artifact], plant.get("pointer"),
                     f"{entry.get('id')}:plant")
    accepted = plant.get("accepted")
    require(isinstance(accepted, list) and value in accepted,
            f"{entry.get('id')}: plant status {value!r} not in {accepted!r}")
    return {"status": "PASS", "artifacts": artifact_rows,
            "checks": checked, "plant_value": value}


def _card_measurements(entry: dict[str, Any], base: Path, commit: str,
                       measured: list[str]) -> list[dict[str, Any]]:
    """Verify that every claimed non-primary card has a hashed measurement."""
    specs = entry.get("card_measurements", {})
    require(isinstance(specs, dict),
            f"{entry.get('id')}: card-measurement registry is not an object")
    require(set(specs) == set(measured),
            f"{entry.get('id')}: measured-card names lack exact artifacts")
    rows = []
    for card in measured:
        spec = specs[card]
        require(isinstance(spec, dict),
                f"{entry.get('id')}:{card}: bad card-measurement spec")
        path = _path(base, spec.get("path"),
                     f"{entry.get('id')}:{card}:measurement")
        expected_sha = spec.get("sha256")
        require(isinstance(expected_sha, str) and len(expected_sha) == 64,
                f"{entry.get('id')}:{card}: measurement sha256 absent")
        actual_sha = _sha256(path)
        require(actual_sha == expected_sha,
                f"{entry.get('id')}:{card}: measurement digest moved")
        report = _read(path)
        require(report.get("format") == CARD_COMPARISON_FORMAT,
                f"{entry.get('id')}:{card}: unexpected measurement format")
        require(report.get("status") == "PASS",
                f"{entry.get('id')}:{card}: measurement did not pass")
        require(report.get("after_commit") == commit,
                f"{entry.get('id')}:{card}: after commit mismatch")
        require(report.get("certifications_unchanged") is True,
                f"{entry.get('id')}:{card}: certification moved")
        all_rows = report.get("rows")
        moved_rows = report.get("moved_rows")
        require(isinstance(all_rows, list) and len(all_rows) == 15,
                f"{entry.get('id')}:{card}: certified row table incomplete")
        require(isinstance(moved_rows, list)
                and report.get("moved_row_count") == len(moved_rows),
                f"{entry.get('id')}:{card}: moved row table incomplete")
        rows.append({"card": card, "path": str(path), "sha256": actual_sha,
                     "moved_row_count": len(moved_rows)})
    return rows


def _measurement(entry: dict[str, Any], base: Path, baseline_year: dict[int, dict],
                 baseline_month: dict[int, dict]) -> dict[str, Any]:
    candidate_id = entry.get("id")
    commit = entry.get("commit")
    require(candidate_id in CANDIDATES or str(candidate_id).startswith("pair:"),
            f"unknown candidate {candidate_id!r}")
    require(isinstance(commit, str) and len(commit) == 40,
            f"{candidate_id}: commit absent")
    local = _local_proof(entry, base, commit)

    year_report = _read(_path(base, entry.get("year_gap"), f"{candidate_id}:year"))
    month_report = _read(_path(base, entry.get("month_gap"), f"{candidate_id}:month"))
    comparison = _read(_path(base, entry.get("comparison"),
                             f"{candidate_id}:comparison"))
    require(_year_commit(year_report, f"{candidate_id}:year") == commit,
            f"{candidate_id}: year commit mismatch")
    require(_stamp_commit(month_report, f"{candidate_id}:month") == commit,
            f"{candidate_id}: month commit mismatch")
    require(_stamp_commit(comparison, f"{candidate_id}:comparison") == commit,
            f"{candidate_id}: comparison commit mismatch")
    year = _rows(year_report, YEAR_DAYS, f"{candidate_id}:year")
    month = _rows(month_report, MONTH_DAYS, f"{candidate_id}:month")
    require(year[30]["rms_T"] == month[30]["rms_T"],
            f"{candidate_id}: month/year day-30 mismatch")

    require(comparison.get("format") == COMPARISON_FORMAT,
            f"{candidate_id}: comparison format moved")
    require(comparison.get("n_certified_rows_compared") == 70,
            f"{candidate_id}: comparison is not the full 70-row trajectory")
    require(comparison.get("row_filter_applied") is False,
            f"{candidate_id}: filtered comparison is inadmissible")
    field_moves = comparison.get("field_moves")
    require(isinstance(field_moves, list) and len(field_moves) == 70,
            f"{candidate_id}: field movement table is incomplete")
    moved = [row["row"] for row in field_moves
             if row.get("max_previous_legoesm_field_move") != 0]
    registered = entry.get("moved_rows")
    require(isinstance(registered, list) and len(set(registered)) == len(registered),
            f"{candidate_id}: moved-row registry absent or duplicated")
    require(set(registered) == set(moved),
            f"{candidate_id}: moved-row registry is not exact")

    before_first = comparison.get("first_over_bar_reference")
    after_first = comparison.get("first_over_bar_candidate")
    before_kt = before_first.get("kt") if isinstance(before_first, dict) else None
    after_kt = after_first.get("kt") if isinstance(after_first, dict) else None
    first_not_earlier = bool(
        after_kt is None or (isinstance(before_kt, int) and after_kt >= before_kt))
    kt1_losses = [row for row in comparison.get("row_status_changes", [])
                  if ".kt1." in str(row.get("row"))
                  and row.get("reference") == "AT-BAR"
                  and row.get("candidate") == "DEBT"]
    route = entry.get("execution_route")
    card_execution = None
    if route is not None:
        require(isinstance(route, str),
                f"{candidate_id}: execution route is not a string")
        card_execution = _card_execution(route)
        executing = sorted(
            name for name, row in card_execution.items()
            if row["executes_route"])
        require(entry.get("executing_cards") == executing,
                f"{candidate_id}: executing-card registry disagrees with recipes")
    else:
        executing = entry.get("executing_cards", ["GYRE-zco"])
    measured = entry.get("measured_cards", [])
    require(isinstance(executing, list) and "GYRE-zco" in executing,
            f"{candidate_id}: executing-card registry is invalid")
    require(isinstance(measured, list), f"{candidate_id}: measured cards invalid")
    card_measurements = _card_measurements(entry, base, commit, measured)
    cards_complete = set(executing) <= ({"GYRE-zco"} | set(measured))

    criteria = {
        "day30_decreases": month[30]["rms_T"] < baseline_month[30]["rms_T"],
        "day240_not_worse": year[240]["rms_T"] <= baseline_year[240]["rms_T"],
        "day360_not_worse": year[360]["rms_T"] <= baseline_year[360]["rms_T"],
        "first_over_bar_not_earlier": first_not_earlier,
        "no_kt1_at_bar_loss": not kt1_losses,
        "all_moved_rows_registered": set(registered) == set(moved),
        "all_year_rows_registered": tuple(year) == YEAR_DAYS,
        "all_month_rows_registered": tuple(month) == MONTH_DAYS,
        "all_executing_cards_measured": cards_complete,
    }
    core = all(value for key, value in criteria.items()
               if key != "all_executing_cards_measured")
    return {
        "id": candidate_id,
        "commit": commit,
        "local_proof": local,
        "day30_T_rms": month[30]["rms_T"],
        "day240_T_rms": year[240]["rms_T"],
        "day360_T_rms": year[360]["rms_T"],
        "day240_improvement": baseline_year[240]["rms_T"] - year[240]["rms_T"],
        "year_rows": [{"day": day,
                       "before_T_rms": baseline_year[day]["rms_T"],
                       "after_T_rms": year[day]["rms_T"],
                       "delta_T_rms": year[day]["rms_T"] - baseline_year[day]["rms_T"]}
                      for day in YEAR_DAYS],
        "month_rows": [{"day": day,
                        "before_T_rms": baseline_month[day]["rms_T"],
                        "after_T_rms": month[day]["rms_T"],
                        "delta_T_rms": month[day]["rms_T"] - baseline_month[day]["rms_T"]}
                       for day in MONTH_DAYS],
        "first_over_bar_before": before_first,
        "first_over_bar_after": after_first,
        "kt1_at_bar_losses": kt1_losses,
        "moved_row_count": len(moved),
        "moved_rows": [row for row in field_moves
                       if row.get("max_previous_legoesm_field_move") != 0],
        "executing_cards": executing,
        "card_execution": card_execution,
        "measured_cards": measured,
        "card_measurements": card_measurements,
        "criteria": criteria,
        "core_trajectory_and_year_pass": core,
        "landing_ready": core and cards_complete,
    }


def evaluate(registry: dict[str, Any], registry_path: Path,
             baseline_year_report: dict[str, Any],
             baseline_month_report: dict[str, Any], plant: str | None) -> dict[str, Any]:
    require(registry.get("format") == REGISTRY_FORMAT,
            f"unexpected registry format {registry.get('format')!r}")
    entries = copy.deepcopy(registry.get("candidates"))
    require(isinstance(entries, list), "candidate registry is not a list")
    if plant == "missing-candidate" and entries:
        entries.pop()
    elif plant == "baseline-day240":
        baseline_year_report = copy.deepcopy(baseline_year_report)
        for row in baseline_year_report.get("rows", []):
            if row.get("day") == 240:
                row["rms_T"] = float(np.nextafter(row["rms_T"], np.inf))
    elif plant not in (None, "missing-moved-row"):
        raise GateError(f"unknown plant {plant!r}")
    require(tuple(entry.get("id") for entry in entries) == CANDIDATES,
            f"candidate registry must be exactly {list(CANDIDATES)}")

    baseline_commit = registry.get("baseline_commit")
    require(isinstance(baseline_commit, str) and len(baseline_commit) == 40,
            "baseline commit absent")
    require(_year_commit(baseline_year_report, "baseline year") == baseline_commit,
            "baseline year commit mismatch")
    require(_stamp_commit(baseline_month_report, "baseline month") == baseline_commit,
            "baseline month commit mismatch")
    baseline_year = _rows(baseline_year_report, YEAR_DAYS, "baseline year")
    baseline_month = _rows(baseline_month_report, MONTH_DAYS, "baseline month")
    for day, expected in BASELINE_YEAR_T.items():
        require(baseline_year[day]["rms_T"] == expected,
                f"baseline day {day} moved: {baseline_year[day]['rms_T']!r}")
    require(baseline_year[30]["rms_T"] == baseline_month[30]["rms_T"],
            "baseline month/year day-30 mismatch")

    if plant == "missing-moved-row":
        for entry in entries:
            if entry.get("moved_rows"):
                entry["moved_rows"].pop()
                break
        else:
            raise GateError("missing-moved-row plant found no moved registry")

    base = registry_path.parent
    stamp = worktree_stamp()
    require(stamp.get("clean") is True,
            "Round-130 ranking gate worktree is dirty")
    rows = [_measurement(entry, base, baseline_year, baseline_month)
            for entry in entries]
    order = {name: index for index, name in enumerate(CANDIDATES)}
    ranking = sorted(rows, key=lambda row: (
        -row["day240_improvement"],
        row["day360_T_rms"] - baseline_year[360]["rms_T"],
        row["day30_T_rms"] - baseline_month[30]["rms_T"],
        order[row["id"]],
    ))
    best = ranking[0]
    improving = [row for row in ranking if row["day240_improvement"] > 0]
    card_measurement_required = bool(
        best["core_trajectory_and_year_pass"] and not best["landing_ready"])
    pair_required = bool(
        not best["core_trajectory_and_year_pass"]
        and len(improving) >= 2)
    pair_row = None
    if pair_required and registry.get("pair") is not None:
        pair = registry["pair"]
        require(pair.get("components") == [improving[0]["id"], improving[1]["id"]],
                "pair components are not the top two day-240 improvers")
        pair_row = _measurement(pair, base, baseline_year, baseline_month)
        pair_required = False

    if best["landing_ready"]:
        status = "LANDING-CANDIDATE"
        selected = best["id"]
    elif card_measurement_required:
        status = "CARD-MEASUREMENT-REQUIRED"
        selected = None
    elif pair_row is not None and pair_row["landing_ready"]:
        status = "LANDING-CANDIDATE"
        selected = pair_row["id"]
    elif pair_row is not None and pair_row["core_trajectory_and_year_pass"]:
        status = "CARD-MEASUREMENT-REQUIRED"
        card_measurement_required = True
        selected = None
    elif pair_required:
        status = "PAIR-REQUIRED"
        selected = None
    else:
        status = "HELD"
        selected = None
    return {
        "format": FORMAT,
        "status": status,
        "plant": plant,
        "worktree": stamp,
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "baseline_commit": baseline_commit,
        "baseline_year_T_rms": {str(day): baseline_year[day]["rms_T"]
                                for day in YEAR_DAYS},
        "ranking": ranking,
        "best_single": best["id"],
        "strict_day240_improvers": [row["id"] for row in improving],
        "pair_required": pair_required,
        "card_measurement_required": card_measurement_required,
        "pair": pair_row,
        "selected": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--baseline-year", type=Path, required=True)
    parser.add_argument("--baseline-month", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=(
        "missing-candidate", "baseline-day240", "missing-moved-row"))
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            _read(args.registry), args.registry,
            _read(args.baseline_year), _read(args.baseline_month), args.plant)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (GateError, OSError, ValueError, json.JSONDecodeError) as error:
        if args.plant:
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        print(f"STATUS REFUSE: {error}")
        return 2
    if args.plant:
        print(f"STATUS REFUSE: plant {args.plant} did not fire")
        return 2
    print(f"STATUS {report['status']}: best={report['best_single']} "
          f"day240_improvers={len(report['strict_day240_improvers'])}")
    return 0 if report["status"] in {"HELD", "LANDING-CANDIDATE"} else 1


if __name__ == "__main__":
    sys.exit(main())
