#!/usr/bin/env python3
"""Admit the rung-0 step-36 tracer-ZDF trace and walk its source order."""

from __future__ import annotations

import argparse
import json
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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round131_step16_walk_gate as prior,
)

TARGET = (86, 159, 0)
ROW_ORDER = (
    "pre_zdf_content", "heat_K", "effective_K", "e3t_after", "e3w_now",
    "lower", "diagonal", "upper", "eliminated", "forward", "solved_T",
)
PLANTS = ("none", "source-order", "passivity", "target-finite")


class GateError(RuntimeError):
    """The ZDF observer or source-order census is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits_equal(left, right) -> bool:
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return (left.dtype == right.dtype and left.shape == right.shape
            and np.array_equal(left.view(np.uint8), right.view(np.uint8)))


def _state_equal(left, right) -> dict[str, bool]:
    import jax

    rows = {}
    for name in right._fields:
        left_leaves, left_tree = jax.tree_util.tree_flatten(getattr(left, name))
        right_leaves, right_tree = jax.tree_util.tree_flatten(getattr(right, name))
        rows[name] = bool(
            left_tree == right_tree
            and len(left_leaves) == len(right_leaves)
            and all(_bits_equal(a, b)
                    for a, b in zip(left_leaves, right_leaves, strict=True)))
    return rows


def _summary(values) -> dict[str, object]:
    array = np.asarray(values)
    require(array.dtype == np.dtype(np.float64), "ZDF trace row is not fp64")
    require(array.ndim == 3, f"ZDF trace row is not 3-D: {array.shape}")
    j, i, k = TARGET
    require(j < array.shape[0] and i < array.shape[1] and k < array.shape[2],
            f"target {TARGET} is outside trace row {array.shape}")
    bad = np.argwhere(~np.isfinite(array))
    finite = np.abs(array[np.isfinite(array)])
    value = array[TARGET]
    return {
        "shape": list(array.shape),
        "nonfinite": int(bad.shape[0]),
        "first_nonfinite": (list(map(int, bad[0])) if bad.size else None),
        "finite_max_abs": float(finite.max()) if finite.size else None,
        "target_value": str(value),
        "target_nonfinite": int(not np.isfinite(value)),
    }


def _first_target(rows: dict[str, object]) -> str | None:
    return next((name for name in ROW_ORDER
                 if int(rows[name]["target_nonfinite"])), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "source-order":
        report["first_target_nonfinite_row"] = "lower"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "target-finite":
        report["rows"]["pre_zdf_content"]["target_nonfinite"] = 1

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "ZDF walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "ZDF walk did not start from step 35")
    require(tuple(report.get("row_order", ())) == ROW_ORDER,
            "ZDF source-order registry changed")
    require(all(report.get("observer_state_equal", {}).values()),
            "ZDF side output moved an ordinary state leaf")
    require(report.get("ordinary_repeat_state_equal") == {
                name: True for name in prior.FIELDS},
            "ordinary step-36 repeat changed bits")
    require(report.get("returned_first_nonfinite") == {
                "field": "T", "index": list(TARGET), "value": "nan"},
            "step-36 headline failure did not reproduce")
    for name in ROW_ORDER:
        row = report["rows"][name]
        require((row["first_nonfinite"] is None) == (row["nonfinite"] == 0),
                f"{name}: non-finite census disagrees")
        require(int(row["target_nonfinite"]) in (0, 1),
                f"{name}: target census is not boolean")
    first = _first_target(report["rows"])
    require(first == report.get("first_target_nonfinite_row"),
            "first target-cell non-finite row is not source ordered")

    p3 = first in {
        "heat_K", "effective_K", "e3t_after", "e3w_now", "lower",
        "diagonal", "upper", "eliminated", "forward", "solved_T",
    }
    p4 = first in {"forward", "solved_T"}
    report["prediction_ledger"] = {
        "R136-P3": {"status": "CONFIRMED" if p3 else "REFUTED",
                     "observed": first},
        "R136-P4": {"status": "CONFIRMED" if p4 else "REFUTED",
                     "observed": first},
        "R136-P5": {"status": "CONFIRMED",
                     "observed": "all ordinary state leaves bit-identical"},
    }
    return {**report, "status": "PASS_ROUND136_STEP36_ZDF_WALK"}


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
        _NEMOWSTracerZDFTrace,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-136 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-136 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "ZDF walk requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    freshwater, surface = ladder._zero_forcing(
        tuple(np.asarray(card.recipe.initial_state.eta.data).shape))

    def model(hooks=None):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)

    ordinary = model()
    state = card.recipe.initial_state
    started = time.time()
    for step in range(1, 36):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        require(prior.boundary_summary(prior._state_arrays(state))[
                    "nonfinite_total"] == 0,
                f"trajectory became non-finite before step 36: step={step}")
        if step % 5 == 0:
            print(f"ROUND136_ZDF_PROGRESS step={step}/35 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    state35 = state
    returned = jax.device_get(ordinary.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    returned_repeat = jax.device_get(ordinary.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed = jax.device_get(model(_NEMOWSRK3TestHooks(
        tracer_zdf_trace=True)).step(
            state35, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    require(isinstance(observed, _NEMOWSTracerZDFTrace),
            "tracer ZDF side output has the wrong type")
    solve = observed.solve
    arrays = {
        "pre_zdf_content": observed.content_T,
        "heat_K": solve.heat_K,
        "effective_K": solve.effective_K,
        "e3t_after": solve.e3t_after,
        "e3w_now": solve.e3w_now,
        "lower": solve.lower,
        "diagonal": solve.diagonal,
        "upper": solve.upper,
        "eliminated": solve.eliminated_T,
        "forward": solve.forward_T,
        "solved_T": solve.solved_T,
    }
    rows = {name: _summary(arrays[name]) for name in ROW_ORDER}
    returned_summary = prior.boundary_summary(prior._state_arrays(returned))
    return {
        "format": "nemo-testcase-l4-orca2-round136-step36-zdf-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "unmeasured_features": list(card.unmeasured_features),
        "row_order": list(ROW_ORDER),
        "rows": rows,
        "first_target_nonfinite_row": _first_target(rows),
        "returned_first_nonfinite": returned_summary["first_nonfinite"],
        "ordinary_repeat_state_equal": prior.state_bit_rows(
            returned_repeat, returned),
        "observer_state_equal": _state_equal(observed.state_after, returned),
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "stage_order": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749",
            "matrix": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:180-235",
            "factor": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:268-273",
            "forward": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:283-291",
            "backsolve": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:293-299",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.expect_commit is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root is not None and args.expect_commit,
                    "runtime mode requires deck and commit")
            raw = measure(args.deck_root, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, prior.GateError, rung0.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND136_STEP36_ZDF_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
