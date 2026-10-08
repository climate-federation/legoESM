#!/usr/bin/env python3
"""Locate the first >10x independent rung-0 completed-stage error growth."""

from __future__ import annotations

import argparse
import copy
import hashlib
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


FIELDS = ("T", "S", "u", "v")
STEPS = tuple(range(1, 9))
STAGES = (1, 2, 3)
FLOOR = np.float64(2.0e-10)
GROWTH = np.float64(10.0)
CONTROL_SHA256 = "30304b22ae5e5ef86dc7bd95f33f30ec9ce38eddda0d6e96765653b97b4f7190"
EXPECTED_TERMINAL = "raw-mesh e3w_int must contain only finite values > 0"
PLANTS = ("none", "admission", "passivity", "order", "growth-selection")


class GateError(RuntimeError):
    """A record, passive-stage, or frozen-selection prerequisite moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape, "field shape moved")
    candidate_nonfinite = int(np.count_nonzero(~np.isfinite(actual)))
    oracle_nonfinite = int(np.count_nonzero(~np.isfinite(expected)))
    exact = bool(np.array_equal(actual, expected, equal_nan=True))
    differing = int(np.count_nonzero(actual.view(np.uint64) != expected.view(np.uint64)))
    finite = np.isfinite(actual) & np.isfinite(expected)
    if np.any(finite):
        delta = np.abs(actual[finite] - expected[finite])
        max_abs = float(np.max(delta))
        argmax_flat = int(np.flatnonzero(finite)[int(np.argmax(delta))])
        argmax = [int(index) for index in np.unravel_index(argmax_flat, actual.shape)]
    else:
        max_abs = None
        argmax = None
    return {
        "bit_exact": exact,
        "differing_cells": differing,
        "cells": int(actual.size),
        "candidate_nonfinite": candidate_nonfinite,
        "oracle_nonfinite": oracle_nonfinite,
        "max_abs": max_abs,
        "argmax": argmax,
        "candidate_max_abs": (
            float(np.max(np.abs(actual[np.isfinite(actual)])))
            if np.any(np.isfinite(actual)) else None
        ),
        "oracle_max_abs": (
            float(np.max(np.abs(expected[np.isfinite(expected)])))
            if np.any(np.isfinite(expected)) else None
        ),
    }


def _boundary_max(boundary: dict[str, object]) -> float:
    values = []
    for field in FIELDS:
        row = boundary["rows"][field]
        if row["candidate_nonfinite"] or row["oracle_nonfinite"]:
            return math.inf
        require(row["max_abs"] is not None, "finite row lacks a maximum")
        values.append(float(row["max_abs"]))
    return max(values)


def _growth_rows(boundaries: list[dict[str, object]]) -> list[dict[str, object]]:
    previous = float(FLOOR)
    result = []
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
        result.append({
            "kt": boundary["kt"], "stage": boundary["stage"],
            "max_abs": current, "previous_max_abs": previous,
            "ratio": ratio, "over_10x": bool(ratio > GROWTH),
        })
        previous = current
    return result


def _first_growth(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return next((copy.deepcopy(row) for row in rows if row["over_10x"]), None)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "admission":
        report["admission"]["record_count"] = 79
    elif plant == "passivity":
        report["passivity"]["1"]["T"] = False
    elif plant == "order":
        report["boundaries"][0], report["boundaries"][1] = (
            report["boundaries"][1], report["boundaries"][0])
    elif plant == "growth-selection":
        report["first_growth"]["stage"] = 2

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "growth table is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report["admission"]["record_count"] == 80,
            "frame admission census moved")
    require(report["control"]["sha256"] == CONTROL_SHA256,
            "round-166 control pin moved")
    require(tuple(report["field_order"]) == FIELDS, "field order moved")
    require(float(report["floor"]) == FLOOR
            and float(report["growth_threshold"]) == GROWTH,
            "growth selector constants moved")
    require(all(all(fields.values()) for fields in report["passivity"].values()),
            "ordinary complete-arm state moved from round 166")
    expected_order = [
        (kt, stage) for kt in STEPS for stage in STAGES
        if not (kt == 8 and stage == 3)
    ]
    observed_order = [(row["kt"], row["stage"]) for row in report["boundaries"]]
    require(observed_order == expected_order, "completed-stage order moved")
    require(report["terminal"] == {
        "kt": 8, "stage": 3, "status": "REFUSED",
        "error": EXPECTED_TERMINAL,
    }, "kt=8 stage-3 terminal boundary moved")
    for boundary in report["boundaries"]:
        require(tuple(boundary["rows"]) == FIELDS,
                f"kt={boundary['kt']} stage={boundary['stage']} field order moved")
        for field in FIELDS:
            row = boundary["rows"][field]
            require(row["oracle_nonfinite"] == 0,
                    f"oracle is non-finite at kt={boundary['kt']} stage={boundary['stage']} {field}")
            require(row["candidate_nonfinite"] >= 0,
                    "candidate non-finite count is invalid")
    growth = _growth_rows(report["boundaries"])
    require(growth == report["growth"], "growth table is not derived from rows")
    first = _first_growth(growth)
    require(first == report["first_growth"], "first-growth selector moved")
    require(report["one_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1,
    }, "one-ULP control did not fire")
    report["prediction_ledger"] = {
        "R174-P1": "CONFIRMED",
        "R174-P2": "CONFIRMED",
        "R174-P3": ("CONFIRMED" if first and
                    (first["kt"], first["stage"]) == (1, 1) else "REFUTED"),
        "R174-P4": "CONFIRMED",
        "R174-P5": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND174_STAGE_GROWTH_BOUNDARY"
    return report


def measure(deck_root: Path, record_root: Path, control_path: Path,
            expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-174 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "stage-growth walk requires production JIT on CPU")

    require(control_path.is_file() and sha256(control_path) == CONTROL_SHA256,
            "round-166 passivity control content moved")
    control = json.loads(control_path.read_text())
    expected_digests = {
        int(row["kt"]): row["fields"]
        for row in control["completed_checkpoint_digests"]
    }
    require(tuple(expected_digests) == tuple(range(1, 8)),
            "round-166 passivity step registry moved")

    card, state, freshwater, surface = r166._setup(deck_root, record_root)
    admission = r103.frames.admit(record_root, r103.EXPECTED_PRODUCER, None)
    require(admission["record_count"] == 80, "frame record census moved")
    stage_models = {
        stage: LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=r166._hooks(card, expose_stage=stage))
        for stage in (1, 2)
    }
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r166._hooks(card))

    boundaries = []
    passivity: dict[str, dict[str, bool]] = {}
    terminal = None
    started = time.time()
    for kt in STEPS:
        candidates = {}
        for stage, model in stage_models.items():
            candidates[stage] = jax.device_get(model.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
        try:
            state_after = jax.device_get(ordinary.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
        except Exception as error:
            require(kt == 8 and EXPECTED_TERMINAL in str(error),
                    f"ordinary complete arm refused at kt={kt}: {error}")
            terminal = {"kt": 8, "stage": 3, "status": "REFUSED",
                        "error": EXPECTED_TERMINAL}
            state_after = None
        if state_after is not None:
            candidates[3] = state_after
            # Reuse the producer's exact digest serialization.  A locally
            # restated dtype/shape prefix caused the first instrument run to
            # refuse identical arrays before any science was emitted.
            actual_digests = r166._state_digests(state_after)
            passivity[str(kt)] = {
                field: actual_digests[field] == expected_digests[kt][field]
                for field in ("T", "S", "u", "v", "ssh")
            }
            require(all(passivity[str(kt)].values()),
                    f"kt={kt}: complete-arm digest moved")
        for stage in sorted(candidates):
            actual = r103.rung0.candidate_fields(candidates[stage])
            expected = r103.rung0.assemble_frame(record_root, kt, stage)
            boundaries.append({
                "kt": kt, "stage": stage,
                "rows": {field: _score(actual[field], expected[field])
                         for field in FIELDS},
            })
        print(f"PROGRESS round174 kt={kt} stages={sorted(candidates)} "
              f"wall_s={time.time() - started:.1f}", file=sys.stderr, flush=True)
        if state_after is None:
            break
        state = state_after

    require(terminal is not None, "kt=8 stage-3 refusal disappeared")
    growth = _growth_rows(boundaries)
    synthetic = np.zeros((2,), dtype=np.float64)
    planted = synthetic.copy()
    planted[0] = np.nextafter(0.0, np.float64(np.inf))
    control_row = _score(planted, synthetic)
    raw = {
        "format": "nemo-testcase-l4-orca2-round174-stage-growth-v1",
        "claim_label": "independent", "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "admission": {"status": admission["status"],
                      "record_count": admission["record_count"]},
        "control": {"path": str(control_path), "sha256": sha256(control_path)},
        "private_arm": {
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
        },
        "field_order": list(FIELDS), "floor": float(FLOOR),
        "growth_threshold": float(GROWTH), "passivity": passivity,
        "boundaries": boundaries, "growth": growth,
        "first_growth": _first_growth(growth), "terminal": terminal,
        "one_ulp_control": {
            "bit_exact": control_row["bit_exact"],
            "differing_cells": control_row["differing_cells"],
        },
        "compiled_citation": (
            "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/"
            "stprk3.f90:200-233"),
        "wall_seconds": time.time() - started,
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in:
            require(not any((args.deck_root, args.record_root, args.control,
                             args.expect_commit)),
                    "classification cannot take runtime inputs")
            raw = json.loads(args.report_in.read_text())
            result = classify(raw, args.plant)
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.record_root, args.control,
                         args.expect_commit)), "runtime inputs are incomplete")
            result = measure(args.deck_root, args.record_root, args.control,
                             args.expect_commit)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r103.GateError, r103.frames.GateError, r166.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND174_STAGE_GROWTH_BOUNDARY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
