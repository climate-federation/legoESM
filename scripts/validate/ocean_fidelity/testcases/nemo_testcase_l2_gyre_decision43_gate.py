#!/usr/bin/env python3
"""Decision-43/45 admission gate for magnitude-first GYRE landings.

This gate consumes existing certified artifacts.  It does not run either
model and it does not replace the oracle-relative ladder comparison.  It
applies the temporary Decision-43 policy to that comparison and to the
month/year day-gap reports, then resolves whether the changed production
statement is reachable on the other named cards.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.provenance import worktree_stamp


FORMAT = "nemo-testcase-l2-gyre-decision43-v2"
COMPARISON_FORMAT = "legoesm-ocean-oracle-relative-move-gate-v3"
DAY_GAP_FORMAT = "gyre-year-owners-day-gap-v1"
YEAR_MEMBER_FORMAT = "nemo-testcase-l2-gyre-year-fromrest-member-v1"
YEAR_DAYS = (30, 60, 90, 120, 180, 240, 300, 360)
YEAR_TAG = "year"
YEAR_DT_S = 14400.0
YEAR_STEPS = 2160
YEAR_SNAPSHOT_STEP_INTERVAL = 6


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_year_owners():
    path = Path(__file__).with_name("nemo_testcase_l2_gyre_year_owners.py")
    spec = importlib.util.spec_from_file_location(
        "_decision43_year_owners", path)
    require(spec is not None and spec.loader is not None,
            f"cannot import year scorer {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _admit_year_member(
    root: Path,
    *,
    expected_commit: str,
    label: str,
) -> dict:
    """Admit one Decision-45 seed-0 year member and its fp64 snapshots."""
    member = root / f"lego_seed0_{YEAR_TAG}"
    manifest_path = member / "manifest.json"
    manifest = _read(manifest_path)
    require(manifest.get("format") == YEAR_MEMBER_FORMAT,
            f"{label}: unexpected member format {manifest.get('format')!r}")
    expected = {
        "case": "GYRE-zco",
        "seed": 0,
        "tag": YEAR_TAG,
        "days": 360,
        "steps": YEAR_STEPS,
        "dt_s": YEAR_DT_S,
        "snapshot_step_interval": YEAR_SNAPSHOT_STEP_INTERVAL,
        "snapshot_days": list(range(1, 361)),
    }
    for key, value in expected.items():
        require(manifest.get(key) == value,
                f"{label}: manifest {key}={manifest.get(key)!r}, "
                f"expected {value!r}")
    stamp = manifest.get("worktree")
    require(isinstance(stamp, dict), f"{label}: member worktree stamp absent")
    require(stamp.get("clean") is True,
            f"{label}: member producer worktree is dirty")
    require(stamp.get("commit") == expected_commit,
            f"{label}: member commit {stamp.get('commit')!r} does not match "
            f"expected {expected_commit!r}")
    required_fields = {"T", "S", "u", "v", "ssh"}
    snapshot_hashes = {}
    for day in YEAR_DAYS:
        path = member / f"day{day:03d}.npz"
        require(path.is_file(), f"{label}: missing year snapshot {path}")
        with np.load(path) as arrays:
            require(required_fields.issubset(arrays.files),
                    f"{label}: {path} lacks a required state field")
            for field in required_fields:
                require(arrays[field].dtype == np.dtype(np.float64),
                        f"{label}: {path}:{field} is {arrays[field].dtype}, "
                        "expected float64")
                require(np.all(np.isfinite(arrays[field])),
                        f"{label}: {path}:{field} contains non-finite values")
        snapshot_hashes[str(day)] = _sha256(path)
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "record": manifest,
        "snapshot_sha256": snapshot_hashes,
    }


def score_year_root(
    root: Path,
    nemo_root: Path,
    *,
    expected_commit: str,
    label: str,
) -> dict:
    """Score the registered Decision-45 rows with the existing day-gap tool."""
    admission = _admit_year_member(
        root, expected_commit=expected_commit, label=label)
    mesh_path = nemo_root / "nemo_seed0" / "mesh_mask.nc"
    require(mesh_path.is_file(), f"{label}: missing NEMO mesh {mesh_path}")
    report = _load_year_owners().day_gap(
        lego_root=root,
        lego_tag=YEAR_TAG,
        nemo_root=nemo_root,
        seed=0,
        mesh_path=mesh_path,
        days=YEAR_DAYS,
    )
    stamp = report.get("worktree")
    require(isinstance(stamp, dict) and stamp.get("clean") is True,
            f"{label}: year scorer worktree is dirty or unstamped")
    report["member_admission"] = admission
    report["nemo_root"] = str(nemo_root)
    report["mesh_path"] = str(mesh_path)
    report["mesh_sha256"] = _sha256(mesh_path)
    return report


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


def _year_rows(report: dict, label: str, expected_commit: str) -> dict[int, dict]:
    require(report.get("format") == DAY_GAP_FORMAT,
            f"{label}: unexpected format {report.get('format')!r}")
    require(report.get("seed") == 0, f"{label}: expected seed 0")
    require(report.get("plant") is None, f"{label}: planted year report")
    admission = report.get("member_admission")
    require(isinstance(admission, dict), f"{label}: member admission absent")
    manifest = admission.get("record")
    require(isinstance(manifest, dict), f"{label}: member manifest absent")
    stamp = manifest.get("worktree")
    require(isinstance(stamp, dict), f"{label}: producer stamp absent")
    require(stamp.get("clean") is True, f"{label}: producer is dirty")
    require(stamp.get("commit") == expected_commit,
            f"{label}: producer commit does not match expected commit")
    rows = report.get("rows")
    require(isinstance(rows, list), f"{label}: rows are absent")
    require([row.get("day") for row in rows] == list(YEAR_DAYS),
            f"{label}: year days are not exactly {list(YEAR_DAYS)}")
    by_day = {}
    for row in rows:
        day = row["day"]
        value = row.get("rms_T")
        require(isinstance(value, (int, float)) and np.isfinite(value),
                f"{label}: day {day} rms_T is not finite numeric")
        by_day[day] = row
    return by_day


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
        build_orca2_zps_card,
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    require(route in {
        "ldf_stage3", "fct_metric_upstream", "wind_qco",
        "momentum_ldf_live_geometry", "stage_momentum_wzv",
        "tke_shear_step_entry_eta",
    },
            f"unknown Decision-43 source route {route!r}")

    def row(config, **extra):
        vertical_mixing = getattr(
            getattr(config, "physics", None), "vertical_mixing", None)
        tke = getattr(vertical_mixing, "tke", None)
        values = {
            "tracer_time_integrator": config.tracer_time_integrator,
            "momentum_time_integrator": config.momentum_time_integrator,
            "lateral_viscosity_operator": config.lateral_viscosity_operator,
            "lateral_viscosity_e3_weighting": (
                config.lateral_viscosity_e3_weighting),
            "surface_stress_implicit": bool(config.surface_stress_implicit),
            "tracer_advection": config.tracer_advection,
            "adaptive_implicit_vertadv": bool(
                config.adaptive_implicit_vertadv),
            "gm_redi_configured": config.gm_redi is not None,
            "momentum_advection": getattr(
                config, "momentum_advection", "flux_form"),
            "wzv_call2_evaluation": getattr(
                config, "wzv_call2_evaluation", "generic"),
            "vertical_mixing_scheme": getattr(
                vertical_mixing, "scheme", None),
            "tke_prognostic": bool(getattr(tke, "prognostic", False)),
            "tke_shear_evaluation_stage": getattr(
                tke, "tke_shear_evaluation_stage", None),
            "tke_shear_production": getattr(
                tke, "tke_shear_production", None),
            "tke_shear_metric_source": getattr(
                tke, "tke_shear_metric_source", None),
        }
        values.update(extra)
        if route == "ldf_stage3":
            executes = (config.tracer_time_integrator == "rk3_ws"
                        and config.gm_redi is not None)
        elif route == "fct_metric_upstream":
            executes = (config.tracer_time_integrator == "rk3_ws"
                        and config.tracer_advection == "fct2"
                        and not config.adaptive_implicit_vertadv)
        elif route == "wind_qco":
            executes = (config.momentum_time_integrator == "rk3_ws"
                        and config.surface_stress_implicit)
        elif route == "stage_momentum_wzv":
            # Round 160 built it, round 163 LANDED it (Decision 55, note AT).
            # The census imports the model's OWN predicates rather than
            # restating them, so the gate cannot encode a condition the code
            # does not (operator note AR, finding 2).  Two rows, because they
            # answer different questions: whether the card's configuration
            # selects NEMO's two-solve stage program at all -- the blast
            # radius of the statement -- and whether this run actually takes
            # it, which is each card's OWN explicit config choice (both are
            # True after ORCA2 Decision 58) and can disagree with the blast
            # radius unless a test opts out/in explicitly via a hook.
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                nemo_stage_momentum_wzv_executes,
                nemo_stage_momentum_wzv_resolved)

            # ``executes_route`` is what the gate scores: which cards run
            # the CANDIDATE, i.e. with its arm selected.  That is the blast
            # radius of the statement under test, and it is the question the
            # admission gate exists to answer.  ``executes_at_this_tip``
            # answers the other question -- what runs today -- which is the
            # production card choice at this tip.
            values["executes_at_this_tip"] = bool(
                nemo_stage_momentum_wzv_executes(config))
            executes = nemo_stage_momentum_wzv_resolved(config)
        elif route == "tke_shear_step_entry_eta":
            executes = (
                config.momentum_time_integrator == "rk3_ws"
                and getattr(vertical_mixing, "scheme", None) == "tke"
                and bool(getattr(tke, "prognostic", False))
                and getattr(tke, "tke_shear_evaluation_stage", None)
                == "step_entry"
                and getattr(tke, "tke_shear_production", None) in (
                    "nemo_face_native_now2", "nemo_face_native_nbb2")
                and getattr(tke, "tke_shear_metric_source", None)
                == "nemo_qco_live_face")
        else:
            executes = (
                config.momentum_time_integrator == "rk3_ws"
                and config.lateral_viscosity_operator == "nemo_div_curl"
                and config.lateral_viscosity_e3_weighting == "nemo_e3")
        values["executes_route"] = bool(executes)
        return values

    rows = {}
    for case in ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        config = build_nemo_testcase_card(case).recipe.model_config
        rows[case] = row(config, recipe_source="nemo_testcase_card")
    # The ORCA2 card is source-file driven and therefore cannot be represented
    # by the three synthetic-card dispatch calls above.  Build the real card
    # from the campaign's pinned deck so the census covers the shared RK3/QCO
    # stage program instead of inferring ORCA2 from a nominal config.
    orca2_deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    orca2 = build_orca2_zps_card(orca2_deck)
    rows[orca2.case] = row(
        orca2.recipe.model_config,
        recipe_source="build_orca2_zps_card",
        deck_root=str(orca2_deck),
        unmeasured_features=list(orca2.unmeasured_features),
    )
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

    # The same committed instrument is deliberately used against the landing's
    # archived base implementation and the descendant.  Stamp both parties:
    # the model tree is the invocation cwd; the instrument tree owns this file.
    stamp = worktree_stamp(repo=Path.cwd())
    instrument_stamp = worktree_stamp(
        repo=Path(__file__).resolve().parents[4])
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
        "instrument_worktree": instrument_stamp,
        "snapshot": str(snapshot),
        "route_observation": {
            "tracer_time_integrator": config.tracer_time_integrator,
            "momentum_time_integrator": config.momentum_time_integrator,
            "gm_redi_configured": config.gm_redi is not None,
            "executes_ldf_stage3_route": bool(
                config.tracer_time_integrator == "rk3_ws"
                and config.gm_redi is not None),
            "executes_momentum_ldf_live_geometry_route": bool(
                config.momentum_time_integrator == "rk3_ws"
                and config.lateral_viscosity_operator == "nemo_div_curl"
                and config.lateral_viscosity_e3_weighting == "nemo_e3"),
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
        "status": "PASS",
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
    before_year_gap: dict,
    after_year_gap: dict,
    *,
    expected_candidate_commit: str,
    expected_before_year_commit: str,
    route: str = "ldf_stage3",
    measured_cards: tuple[str, ...] = (),
    registered_rows: tuple[str, ...] = (),
    plant: str | None = None,
) -> dict:
    comparison = copy.deepcopy(comparison)
    before_day_gap = copy.deepcopy(before_day_gap)
    after_day_gap = copy.deepcopy(after_day_gap)
    before_year_gap = copy.deepcopy(before_year_gap)
    after_year_gap = copy.deepcopy(after_year_gap)

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
    before_year = _year_rows(
        before_year_gap, "before year", expected_before_year_commit)
    after_year = _year_rows(
        after_year_gap, "after year", expected_candidate_commit)
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
    elif plant == "year-day240-worse":
        after_year[240]["rms_T"] = float(np.nextafter(
            np.float64(before_year[240]["rms_T"]), np.float64(np.inf)))
    elif plant is not None:
        raise GateError(f"unknown plant {plant!r}")

    year_rows = []
    for day in YEAR_DAYS:
        before_value = float(before_year[day]["rms_T"])
        after_value = float(after_year[day]["rms_T"])
        year_rows.append({
            "day": day,
            "before_T_rms": before_value,
            "after_T_rms": after_value,
            "delta_T_rms": after_value - before_value,
            "not_worse": after_value <= before_value,
        })
    year_by_day = {row["day"]: row for row in year_rows}

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
        "month_and_year_day30_agree": bool(
            before30["rms_T"] == before_year[30]["rms_T"]
            and after30["rms_T"] == after_year[30]["rms_T"]),
        "year_day240_T_rms_not_worse": year_by_day[240]["not_worse"],
        "year_day360_T_rms_not_worse": year_by_day[360]["not_worse"],
        "all_year_rows_registered": len(year_rows) == len(YEAR_DAYS),
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
        and criteria["month_and_year_day30_agree"]
        and criteria["year_day240_T_rms_not_worse"]
        and criteria["year_day360_T_rms_not_worse"]
        and criteria["all_year_rows_registered"]
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
        "expected_before_year_commit": expected_before_year_commit,
        "before_day30_T_rms": before30["rms_T"],
        "after_day30_T_rms": after30["rms_T"],
        "year_days": list(YEAR_DAYS),
        "year_rows": year_rows,
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
    parser.add_argument("--before-year-root", type=Path)
    parser.add_argument("--after-year-root", type=Path)
    parser.add_argument("--year-nemo-root", type=Path)
    parser.add_argument("--expect-before-year-commit")
    parser.add_argument("--year-measure-root", type=Path)
    parser.add_argument("--expect-year-commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generic-measure-snapshot", type=Path)
    parser.add_argument("--generic-before-report", type=Path)
    parser.add_argument("--generic-before-snapshot", type=Path)
    parser.add_argument("--generic-after-report", type=Path)
    parser.add_argument("--generic-after-snapshot", type=Path)
    parser.add_argument("--moved-row-registry", type=Path)
    parser.add_argument(
        "--route", choices=(
            "ldf_stage3", "fct_metric_upstream", "wind_qco",
            "momentum_ldf_live_geometry", "stage_momentum_wzv",
            "tke_shear_step_entry_eta"),
        default="ldf_stage3")
    parser.add_argument(
        "--measured-card", action="append", default=[],
        help="executing non-primary card discharged by a separate measurement")
    parser.add_argument("--plant", choices=(
        "day30-no-improvement", "earlier-first-over-bar", "kt1-at-bar-loss",
        "missing-moved-registry", "year-day240-worse"))
    args = parser.parse_args(argv)
    try:
        if args.year_measure_root is not None:
            require(args.year_nemo_root is not None,
                    "year measurement requires --year-nemo-root")
            require(args.expect_year_commit is not None,
                    "year measurement requires --expect-year-commit")
            report = score_year_root(
                args.year_measure_root,
                args.year_nemo_root,
                expected_commit=args.expect_year_commit,
                label="year measurement",
            )
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            print("STATUS PASS: Decision-45 year rows="
                  + ",".join(str(day) for day in YEAR_DAYS))
            return 0
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
            args.expect_candidate_commit, args.moved_row_registry,
            args.before_year_root, args.after_year_root,
            args.year_nemo_root, args.expect_before_year_commit)),
            "Decision-43 admission requires comparison, day gaps, commit, "
            "moved-row registry, and Decision-45 year roots")
        before_year_gap = score_year_root(
            args.before_year_root,
            args.year_nemo_root,
            expected_commit=args.expect_before_year_commit,
            label="before year",
        )
        after_year_gap = score_year_root(
            args.after_year_root,
            args.year_nemo_root,
            expected_commit=args.expect_candidate_commit,
            label="after year",
        )
        report = evaluate(
            _read(args.comparison),
            _read(args.before_day_gap),
            _read(args.after_day_gap),
            before_year_gap,
            after_year_gap,
            expected_candidate_commit=args.expect_candidate_commit,
            expected_before_year_commit=args.expect_before_year_commit,
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
    year_by_day = {row["day"]: row for row in report["year_rows"]}
    print(f"STATUS {report['status']}: moved_rows={report['moved_row_count']} "
          f"day30_T={report['before_day30_T_rms']:.17e}->"
          f"{report['after_day30_T_rms']:.17e} "
          f"day240_T={year_by_day[240]['before_T_rms']:.17e}->"
          f"{year_by_day[240]['after_T_rms']:.17e} "
          f"day360_T={year_by_day[360]['before_T_rms']:.17e}->"
          f"{year_by_day[360]['after_T_rms']:.17e}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
