#!/usr/bin/env python3
"""Locate the first finite growth boundary of round 249's private fold unit."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round248_land_support_gate as r248,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean, read_ordered  # noqa: E402

LABELS = ("independent", "given_nemo_entry")
FIELDS = ("ssh", "u", "v", "T", "S")
POINT_TYPES = {"ssh": "T", "u": "U", "v": "V", "T": "T", "S": "T"}
SIGNS = {"ssh": 1.0, "u": -1.0, "v": -1.0, "T": 1.0, "S": 1.0}
FLOOR = np.float64(2.0e-10)
GROWTH = np.float64(10.0)
PLANTS = (
    "none", "label-coverage", "production-route", "passivity",
    "boundary-order", "finite-coverage", "growth-owner", "fold-sign",
    "nemo-fold", "slow-owner",
)


class GateError(RuntimeError):
    """The passive growth table or a frozen causal predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native(name: str, values) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if name == "u":
        return values[:, 1:]
    if name == "v":
        return values[1:]
    return values


def _state_fields(state) -> dict[str, np.ndarray]:
    return {
        "ssh": np.asarray(state.eta.data, np.float64),
        "u": _native("u", state.u.data),
        "v": _native("v", state.v.data),
        "T": np.asarray(state.T.data, np.float64),
        "S": np.asarray(state.S.data, np.float64),
    }


def _score(left, right) -> dict[str, object]:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    require(left.shape == right.shape, f"shape mismatch {left.shape} != {right.shape}")
    finite = np.isfinite(left) & np.isfinite(right)
    bad = np.argwhere(~finite)
    unequal = left.view(np.uint64) != right.view(np.uint64)
    if np.any(finite):
        delta = np.zeros_like(left)
        np.subtract(left, right, out=delta, where=finite)
        absolute = np.abs(delta)
        flat = int(np.argmax(np.where(finite, absolute, -np.inf)))
        argmax = list(map(int, np.unravel_index(flat, left.shape)))
        maximum = float(absolute[tuple(argmax)])
        rms = float(np.sqrt(np.mean(np.square(delta[finite]), dtype=np.float64)))
    else:
        argmax, maximum, rms = None, None, None
    return {
        "support": int(left.size),
        "unequal": int(np.count_nonzero(unequal)),
        "max_abs": maximum,
        "rms": rms,
        "argmax": argmax,
        "first_nonfinite": None if bad.size == 0 else list(map(int, bad[0])),
        "nonfinite": int(bad.shape[0]),
    }


def _field_row(card, name: str, candidate, oracle) -> dict[str, object]:
    candidate = np.asarray(candidate, np.float64)
    oracle = np.asarray(oracle, np.float64)
    require(candidate.shape == oracle.shape, f"{name}: native shape moved")
    return {
        "complete": _score(candidate, oracle),
        "interior": _score(candidate[:-3], oracle[:-3]),
        "fold_band": _score(candidate[-3:], oracle[-3:]),
        "fold_identity": r228.fold_residual(
            candidate, POINT_TYPES[name], card.recipe.grid.fold, SIGNS[name]),
        "nemo_fold_identity": r228.fold_residual(
            oracle, POINT_TYPES[name], card.recipe.grid.fold, SIGNS[name]),
    }


def _boundary(card, state, oracle: dict[str, np.ndarray], kt: int,
              stage: int) -> dict[str, object]:
    candidate = _state_fields(state)
    return {
        "kt": kt,
        "stage": stage,
        "fields": {
            name: _field_row(card, name, candidate[name], oracle[name])
            for name in FIELDS
        },
    }


def _hooks(card, unit: bool, *, live: bool = False, stage: int = 0,
           substeps: bool = False):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    reference_depth = (
        rung0.ladder.build_reference_depth_override(card) if unit else None)
    return _NEMOWSRK3TestHooks(
        expose_live_stage_operands=live,
        expose_momentum_stage=stage,
        expose_tracer_stage=stage,
        expose_barotropic_substeps=substeps,
        expose_barotropic_boundary_association=substeps,
        barotropic_external_mode_association=unit,
        barotropic_reference_face_depth_override=reference_depth,
        barotropic_unmasked_v_transport=unit,
        barotropic_materialize_v_transport=unit,
        barotropic_atomic_fold_unit=unit,
        preserve_literal_hpg_vmask=(True if unit else None),
    )


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-249 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-249 measurement requires production JIT on CPU")
    return stamp


def measure_stages(deck_root: Path, record_root: Path, label: str, unit: bool,
                   route: str, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    require(route in ("production", "resolved-candidate"), f"bad route {route}")
    require((unit and route == "resolved-candidate")
            or (not unit and route == "production"),
            "unit/route pairing moved")
    stamp = _policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(record_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, unit))
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, unit, live=True))

    boundaries = []
    passivity = []
    arm = (r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
        if unit else contextlib.nullcontext())
    with arm:
        for kt in range(1, 5):
            boundaries.append(_boundary(
                card, state, rung0.assemble_frame(record_root, kt, 0), kt, 0))
            ordinary_after = jax.device_get(ordinary.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
            observed = jax.device_get(traced.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
            require(isinstance(observed, _NEMOWSLiveOperandTrace),
                    "stage observer return type moved")
            equal = passive._ordinary_state_equal(
                observed.state_after, ordinary_after)
            require(all(equal.values()), f"kt={kt}: stage observer moved bits")
            passivity.append({"kt": kt, "fields": equal})
            for stage, values in enumerate(observed.stage_outputs, start=1):
                stage_state = state._replace(
                    u=state.u.replace(data=values[0]),
                    v=state.v.replace(data=values[1]),
                    T=state.T.replace(data=values[2]),
                    S=state.S.replace(data=values[3]),
                    eta=state.eta.replace(data=values[4]),
                )
                boundaries.append(_boundary(
                    card, stage_state,
                    rung0.assemble_frame(record_root, kt, stage), kt, stage))
            state = ordinary_after
            print(f"PROGRESS round249 {label} unit={int(unit)} kt={kt}",
                  file=sys.stderr, flush=True)
    return {
        "format": "nemo-testcase-l4-orca2-round249-stage-growth-v1",
        "status": "PASS_R249_STAGE_TABLE",
        "label": label,
        "unit": unit,
        "operand_route": route,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "passivity": passivity,
        "boundaries": boundaries,
    }


def measure_substeps(deck_root: Path, record_root: Path, twin_a: Path,
                     twin_b: Path, label: str,
                     expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSBoundaryAssociationTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    admission = r235._admit_streams(twin_a, twin_b)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2)
    advmean = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    require(ordered["header"]["nrows"] == 2, "ordered stream coverage moved")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(record_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, True))
    observed_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, True, substeps=True))
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        ordinary_after = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        observed = jax.device_get(observed_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(observed, _NEMOWSBoundaryAssociationTrace),
            "barotropic observer return type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary_after)
    require(all(passivity.values()), "barotropic observer moved completed state")
    rows = r235._source_registry(observed.substeps, substeps, advmean)
    return {
        "format": "nemo-testcase-l4-orca2-round249-substeps-v1",
        "status": "PASS_R249_SUBSTEP_TABLE",
        "label": label,
        "operand_route": "resolved-candidate",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "record_admission": admission,
        "passivity": passivity,
        "source_order": list(r235.SOURCE_ORDER),
        "rows": rows,
        "first_complete": r235._first(rows, "complete"),
        "first_interior": r235._first(rows, "interior"),
        "first_fold_band": r235._first(rows, "fold_band"),
    }


def _boundary_max(boundary: dict, region: str) -> float:
    values = []
    for field in FIELDS:
        row = boundary["fields"][field][region]
        if row["nonfinite"]:
            return math.inf
        require(row["max_abs"] is not None, "finite row lacks maximum")
        values.append(float(row["max_abs"]))
    return max(values)


def _growth_rows(boundaries: list[dict], region: str) -> list[dict]:
    previous = float(FLOOR)
    rows = []
    for boundary in boundaries:
        current = _boundary_max(boundary, region)
        ratio = (math.inf if math.isinf(current) else
                 current / max(previous, float(FLOOR)))
        rows.append({
            "kt": boundary["kt"], "stage": boundary["stage"],
            "region": region, "max_abs": current,
            "previous_max_abs": previous, "ratio": ratio,
            "over_10x": bool(ratio > GROWTH),
        })
        previous = current
    return rows


def _load_by_label(paths: list[Path]) -> dict[str, dict]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return {report["label"]: report for report in reports}


def classify(off: dict[str, dict], unit: dict[str, dict],
             substeps: dict[str, dict], slow: dict[str, dict], *,
             plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    off, unit, substeps = copy.deepcopy((off, unit, substeps))
    if plant == "label-coverage":
        unit.pop("given_nemo_entry")
    elif plant == "production-route":
        unit["independent"]["operand_route"] = "production"
    elif plant == "passivity":
        unit["independent"]["passivity"][0]["fields"]["T"] = False
    elif plant == "boundary-order":
        unit["independent"]["boundaries"][0]["stage"] = 1
    elif plant == "finite-coverage":
        unit["independent"]["boundaries"][1]["fields"]["S"]["complete"]["nonfinite"] = 1
    elif plant == "growth-owner":
        for field in FIELDS:
            unit["independent"]["boundaries"][1]["fields"][field][
                "fold_band"]["max_abs"] = 0.0
    elif plant == "fold-sign":
        unit["independent"]["boundaries"][1]["fields"]["v"]["fold_identity"]["sign"] = 1.0
    elif plant == "nemo-fold":
        unit["independent"]["boundaries"][1]["fields"]["S"]["nemo_fold_identity"]["unequal"] = 1
    elif plant == "slow-owner":
        slow["independent"]["rows"]["post_drag_v"]["bit_exact"] = True

    require(set(off) == set(unit) == set(substeps) == set(slow) == set(LABELS),
            "both claim labels are required in every table")
    expected_order = [(kt, stage) for kt in range(1, 5) for stage in range(4)]
    growth = {}
    first_growth = {}
    fold_hits = {}
    kt1_stage1 = {}
    substep_summary = {}
    slow_summary = {}
    for label in LABELS:
        for report, expected_unit, route in (
            (off[label], False, "production"),
            (unit[label], True, "resolved-candidate"),
        ):
            require(report["status"] == "PASS_R249_STAGE_TABLE",
                    f"{label}: stage report status moved")
            require(report["unit"] is expected_unit
                    and report["operand_route"] == route,
                    f"{label}: production/candidate route moved")
            require([(row["kt"], row["stage"]) for row in report["boundaries"]]
                    == expected_order, f"{label}: boundary order moved")
            require(len(report["passivity"]) == 4
                    and all(all(row["fields"].values())
                            for row in report["passivity"]),
                    f"{label}: observer passivity moved")
            for boundary in report["boundaries"]:
                for name in FIELDS:
                    field = boundary["fields"][name]
                    require(field["fold_identity"]["sign"] == SIGNS[name],
                            f"{label}: {name} fold sign moved")
                    require(field["complete"]["nonfinite"] == 0
                            and field["nemo_fold_identity"]["first_nonfinite"] is None,
                            f"{label}: kt=1..4 finite coverage moved")

        rows = _growth_rows(unit[label]["boundaries"], "fold_band")
        growth[label] = rows
        first = next((copy.deepcopy(row) for row in rows if row["over_10x"]), None)
        require(first is not None, f"{label}: no >10x fold-band boundary")
        first_growth[label] = first
        boundary = next(row for row in unit[label]["boundaries"]
                        if (row["kt"], row["stage"])
                        == (first["kt"], first["stage"]))
        hits = []
        for name in FIELDS:
            field = boundary["fields"][name]
            require(field["nemo_fold_identity"]["unequal"] == 0,
                    f"{label}: NEMO {name} fold identity moved")
            if field["fold_identity"]["unequal"]:
                hits.append({"field": name, **field["fold_identity"]})
        fold_hits[label] = hits
        kt1_stage1[label] = {
            name: {
                "off": off[label]["boundaries"][1]["fields"][name]["fold_band"],
                "unit": unit[label]["boundaries"][1]["fields"][name]["fold_band"],
                "nemo_fold_unequal": unit[label]["boundaries"][1]["fields"][name][
                    "nemo_fold_identity"]["unequal"],
                "unit_fold_unequal": unit[label]["boundaries"][1]["fields"][name][
                    "fold_identity"]["unequal"],
            }
            for name in FIELDS
        }

        trace = substeps[label]
        require(trace["status"] == "PASS_R249_SUBSTEP_TABLE"
                and trace["operand_route"] == "resolved-candidate",
                f"{label}: substep report moved")
        require(all(trace["passivity"].values()),
                f"{label}: barotropic trace moved completed state")
        require(tuple(trace["source_order"]) == r235.SOURCE_ORDER,
                f"{label}: substep source order moved")
        substep_summary[label] = {
            "first_complete": trace["first_complete"],
            "first_interior": trace["first_interior"],
            "first_fold_band": trace["first_fold_band"],
        }

        slow_report = slow[label]
        require(slow_report["status"] == "PASS_R236_SLOW_V_SPLIT"
                and all(slow_report["passivity"].values()),
                f"{label}: slow-forcing replay is not passive/admitted")
        slow_rows = slow_report["rows"]
        require(slow_report["owner_support"]["unequal"] == 35,
                f"{label}: slow-V fold support moved")
        require(slow_rows["depth_mean_v"]["bit_exact"],
                f"{label}: depth mean is not the last exact output")
        require(not slow_rows["post_drag_v"]["bit_exact"],
                f"{label}: bottom-drag correction did not introduce debt")
        require(not slow_rows["post_wind_incoming_v"]["bit_exact"]
                and slow_rows["coriolis_v"]["bit_exact"]
                and slow_rows["ssvmask"]["bit_exact"],
                f"{label}: slow-V downstream source order moved")
        slow_summary[label] = {
            "last_exact_output": "depth_mean_v",
            "retracted_non_source_exact_output": "post_drag_v",
            "post_drag": slow_rows["post_drag_v"],
            "final_slow_v": slow_rows["final_slow_v"],
            "retraction": (
                "round 250 replays the literal drag correction exactly; "
                "this field came from the private slow-forcing override"
            ),
        }

    require((first_growth["independent"]["kt"],
             first_growth["independent"]["stage"])
            == (first_growth["given_nemo_entry"]["kt"],
                first_growth["given_nemo_entry"]["stage"]),
            "labels disagree on first growth boundary")
    kt1_stage1_growth = (
        (first_growth["independent"]["kt"],
         first_growth["independent"]["stage"]) == (1, 1)
        and all(fold_hits[label] for label in LABELS))
    require(kt1_stage1_growth, "kt=1 stage-1 growth boundary moved")
    status = "RETRACTED_R249_DRAG_OWNER_BY_R250"
    return {
        "format": "nemo-testcase-l4-orca2-round249-classification-v1",
        "status": status,
        "floor": float(FLOOR),
        "growth_threshold": float(GROWTH),
        "first_growth": first_growth,
        "fold_identity_hits": fold_hits,
        "kt1_stage1": kt1_stage1,
        "growth": growth,
        "substeps": substep_summary,
        "slow_forcing": slow_summary,
        "predictions": {
            "R249-P1": "CONFIRMED",
            "R249-P2": "REFUTED_TRACERS_EXACT_AT_KT1_STAGE1",
            "R249-P3": "RETRACTED_BY_R250_NON_SOURCE_EXACT_ARM",
            "R249-P4": "CONFIRMED_MEASUREMENT_ONLY",
            "R249-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stages", "substeps", "classify"),
                        required=True)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--unit", action="store_true")
    parser.add_argument("--route", choices=("production", "resolved-candidate"))
    parser.add_argument("--expect-commit")
    parser.add_argument("--off", nargs=2, type=Path)
    parser.add_argument("--on", nargs=2, type=Path)
    parser.add_argument("--substep", nargs=2, type=Path)
    parser.add_argument("--slow", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "stages":
            require(all((args.deck_root, args.record_root, args.label,
                         args.route, args.expect_commit)),
                    "stage measurement arguments are incomplete")
            result = measure_stages(
                args.deck_root, args.record_root, args.label, args.unit,
                args.route, args.expect_commit)
        elif args.mode == "substeps":
            require(all((args.deck_root, args.record_root, args.twin_a,
                         args.twin_b, args.label, args.expect_commit)),
                    "substep measurement arguments are incomplete")
            result = measure_substeps(
                args.deck_root, args.record_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)
        else:
            require(all((args.off, args.on, args.substep, args.slow)),
                    "classification inputs are incomplete")
            result = classify(
                _load_by_label(args.off), _load_by_label(args.on),
                _load_by_label(args.substep), _load_by_label(args.slow),
                plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError,
            r228.GateError, r235.GateError, rung0.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
