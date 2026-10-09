#!/usr/bin/env python3
"""Locate the first corrected-entry growth boundary under the atomic unit."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as r103,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round174_stage_growth_gate as r174,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)

FIELDS = ("T", "S", "u", "v", "ssh")
GROWTH_FIELDS = ("T", "S", "u", "v")
STEPS = tuple(range(1, 11))
STAGES = (1, 2, 3)
FLOOR = np.float64(2.0e-10)
GROWTH = np.float64(10.0)
PLANTS = (
    "none", "entry", "admission", "passivity", "order",
    "growth-selection", "ulp",
)


class GateError(RuntimeError):
    """A record, entry, passivity, or selection prerequisite moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _entry_score(actual, expected, active) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(actual.shape == expected.shape == active.shape,
            "independent-entry score shape moved")
    left, right = actual[active], expected[active]
    require(left.size > 0 and np.isfinite(left).all() and np.isfinite(right).all(),
            "independent-entry score is empty or non-finite")
    return {
        "bit_exact": bool(np.array_equal(left, right)),
        "unequal": int(np.count_nonzero(
            left.view(np.uint64) != right.view(np.uint64))),
        "active_cells": int(left.size),
        "max_abs": float(np.max(np.abs(left - right))),
    }


def _boundary_max(boundary: dict[str, object]) -> float:
    values = []
    for field in GROWTH_FIELDS:
        row = boundary["rows"][field]
        if row["candidate_nonfinite"] or row["oracle_nonfinite"]:
            return math.inf
        require(row["max_abs"] is not None, "finite growth row lacks maximum")
        values.append(float(row["max_abs"]))
    return max(values)


def _growth_rows(boundaries: list[dict[str, object]]) -> list[dict[str, object]]:
    previous = float(FLOOR)
    rows = []
    for boundary in boundaries:
        current = _boundary_max(boundary)
        if math.isinf(current) and math.isinf(previous):
            ratio = 1.0
        elif math.isinf(current):
            ratio = math.inf
        elif math.isinf(previous):
            ratio = 0.0
        else:
            ratio = current / max(previous, float(FLOOR))
        rows.append({
            "kt": boundary["kt"], "stage": boundary["stage"],
            "max_abs": current, "previous_max_abs": previous,
            "ratio": ratio, "over_10x": bool(ratio > GROWTH),
        })
        previous = current
    return rows


def _first_growth(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return next((copy.deepcopy(row) for row in rows if row["over_10x"]), None)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "entry":
        report["entry"]["T"]["bit_exact"] = False
    elif plant == "admission":
        report["admission"]["record_count"] = 79
    elif plant == "passivity":
        report["passivity"]["ordinary_duplicate"]["T"] = False
    elif plant == "order":
        report["boundaries"][0], report["boundaries"][1] = (
            report["boundaries"][1], report["boundaries"][0])
    elif plant == "growth-selection":
        report["first_growth"]["stage"] = 2
    elif plant == "ulp":
        report["one_ulp_control"]["differing_cells"] = 0

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state"
            and report.get("decision52_bridge") is None,
            "growth table is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report["admission"]["record_count"] == 80,
            "frame admission census moved")
    require(tuple(report["field_order"]) == FIELDS,
            "field registry moved")
    require(tuple(report["growth_field_order"]) == GROWTH_FIELDS,
            "growth-field registry moved")
    require(float(report["floor"]) == FLOOR
            and float(report["growth_threshold"]) == GROWTH,
            "growth constants moved")
    require(tuple(report["entry"]) == FIELDS
            and all(report["entry"][field]["bit_exact"]
                    and report["entry"][field]["unequal"] == 0
                    for field in FIELDS),
            "corrected independent entry is not active-domain exact")
    private = report["private_arm"]
    require(private == {
        "slow_depth_source_order": True,
        "external_mode_association": True,
        "raw_reference_depth": True,
        "unmasked_v_transport": True,
        "materialize_v_transport": True,
    }, "complete private arm moved")
    require(report["passivity"]["detached_stage_outputs_not_carried"] is True,
            "stage outputs entered carried state")
    require(all(report["passivity"]["ordinary_duplicate"].values()),
            "same-tree ordinary duplicate moved")

    observed = [(row["kt"], row["stage"]) for row in report["boundaries"]]
    expected = [(kt, stage) for kt in STEPS for stage in STAGES][:len(observed)]
    require(observed == expected and observed,
            "completed-stage order moved or is empty")
    for boundary in report["boundaries"]:
        require(tuple(boundary["rows"]) == FIELDS,
                "boundary field registry moved")
        for field in FIELDS:
            require(boundary["rows"][field]["oracle_nonfinite"] == 0,
                    f"oracle is non-finite at {boundary['kt']}/{boundary['stage']} {field}")
    terminal = report.get("terminal")
    require(terminal is None or terminal["status"] == "REFUSED",
            "terminal boundary is malformed")
    if terminal is None:
        require(len(observed) == len(STEPS) * len(STAGES),
                "growth walk stopped without a terminal")
    else:
        next_boundary = [(kt, stage) for kt in STEPS for stage in STAGES][len(observed)]
        require((terminal["kt"], terminal["stage"]) == next_boundary,
                "terminal is not the first missing boundary")

    growth = _growth_rows(report["boundaries"])
    require(growth == report["growth"], "growth table is not derived")
    first = _first_growth(growth)
    require(first == report["first_growth"], "first-growth selector moved")
    require(report["one_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1,
    }, "one-ULP control did not fire")
    report["prediction_ledger"] = {
        "R193-P1": "CONFIRMED",
        "R193-P2": "CONFIRMED",
        "R193-P3": ("CONFIRMED" if first and
                     (first["kt"], first["stage"]) == (1, 1)
                     else "REFUTED"),
        "R193-P4": "UNMEASURED_PENDING_OFFLINE_REPLAY",
        "R193-P5": "UNMEASURED_PENDING_OFFLINE_REPLAY",
        "R193-P6": "CONFIRMED_MEASUREMENT_ONLY",
    }
    report["status"] = "PASS_R193_STAGE_GROWTH_BOUNDARY"
    return report


def measure(deck_root: Path, record_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-193 measurement requires its clean committed candidate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "growth walk requires production JIT on CPU")

    admission = r103.frames.admit(record_root, r103.EXPECTED_PRODUCER, None)
    require(admission["record_count"] == 80, "frame record census moved")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    entry = rung0.assemble_frame(record_root, 1, 0)
    actual_entry = rung0.candidate_fields(state)
    tmask = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    active = {
        "T": tmask, "S": tmask,
        "u": np.ones_like(actual_entry["u"], dtype=bool),
        "v": np.ones_like(actual_entry["v"], dtype=bool),
        "ssh": np.ones_like(actual_entry["ssh"], dtype=bool),
    }
    entry_rows = {
        field: _entry_score(actual_entry[field], entry[field], active[field])
        for field in FIELDS
    }
    require(all(row["bit_exact"] for row in entry_rows.values()),
            "corrected independent entry moved")

    freshwater, surface = r103._zero_forcing(entry["ssh"].shape)
    stage_models = {
        stage: LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=r166._hooks(card, expose_stage=stage))
        for stage in (1, 2)
    }
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r166._hooks(card))
    duplicate = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r166._hooks(card))

    boundaries = []
    terminal = None
    duplicate_passivity = None
    started = time.time()
    for kt in STEPS:
        candidates = {}
        for stage, model in stage_models.items():
            try:
                candidates[stage] = jax.device_get(jax.block_until_ready(model.step(
                    state, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface)))
            except ValueError as error:
                terminal = {"kt": kt, "stage": stage, "status": "REFUSED",
                            "error": str(error)}
                break
        if terminal is not None:
            break
        try:
            state_after = jax.device_get(jax.block_until_ready(ordinary.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface)))
        except ValueError as error:
            terminal = {"kt": kt, "stage": 3, "status": "REFUSED",
                        "error": str(error)}
            state_after = None
        if kt == 1 and state_after is not None:
            duplicate_after = jax.device_get(jax.block_until_ready(duplicate.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface)))
            first = rung0.candidate_fields(state_after)
            second = rung0.candidate_fields(duplicate_after)
            duplicate_passivity = {
                field: bool(np.array_equal(first[field], second[field]))
                for field in FIELDS
            }
            require(all(duplicate_passivity.values()),
                    "same-tree ordinary duplicate is not bit-identical")
        if state_after is not None:
            candidates[3] = state_after
        for stage in sorted(candidates):
            actual = rung0.candidate_fields(candidates[stage])
            expected = rung0.assemble_frame(record_root, kt, stage)
            boundaries.append({
                "kt": kt, "stage": stage,
                "rows": {field: r174._score(actual[field], expected[field])
                         for field in FIELDS},
            })
        print(f"PROGRESS round193 kt={kt} stages={sorted(candidates)} "
              f"wall_s={time.time() - started:.1f}", file=sys.stderr, flush=True)
        if state_after is None:
            break
        state = state_after

    require(duplicate_passivity is not None,
            "ordinary duplicate passivity was not measured")
    growth = _growth_rows(boundaries)
    zero = np.zeros((2,), dtype=np.float64)
    planted = zero.copy()
    planted[0] = np.nextafter(0.0, np.float64(np.inf))
    ulp = r174._score(planted, zero)
    return classify({
        "format": "nemo-testcase-l4-orca2-round193-stage-growth-v1",
        "claim_label": "independent", "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "admission": {"status": admission["status"],
                      "record_count": admission["record_count"]},
        "entry": entry_rows,
        "private_arm": {
            "slow_depth_source_order": True,
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
        },
        "passivity": {
            "detached_stage_outputs_not_carried": True,
            "ordinary_duplicate": duplicate_passivity,
        },
        "field_order": list(FIELDS),
        "growth_field_order": list(GROWTH_FIELDS),
        "floor": float(FLOOR), "growth_threshold": float(GROWTH),
        "boundaries": boundaries, "growth": growth,
        "first_growth": _first_growth(growth), "terminal": terminal,
        "one_ulp_control": {
            "bit_exact": ulp["bit_exact"],
            "differing_cells": ulp["differing_cells"],
        },
        "compiled_citation": (
            "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/"
            "stprk3.f90:200-233"),
        "wall_seconds": time.time() - started,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in:
            require(not any((args.deck_root, args.record_root, args.expect_commit)),
                    "classification cannot take runtime inputs")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.record_root, args.expect_commit)),
                    "runtime inputs are incomplete")
            result = measure(args.deck_root, args.record_root, args.expect_commit)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r103.GateError, r103.frames.GateError, r166.GateError,
            r174.GateError, rung0.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R193_STAGE_GROWTH_BOUNDARY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
