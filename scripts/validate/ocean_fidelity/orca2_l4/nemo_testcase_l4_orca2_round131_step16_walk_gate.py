#!/usr/bin/env python3
"""Walk rung-0's independent step 16 to its first non-finite boundary."""

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


BOUNDARIES = (
    "step15_entry",
    "stage1_tracer",
    "stage2_tracer",
    "stage3_advection_content",
    "pre_implicit_content",
    "pre_implicit_concentration",
    "returned_state",
)
FIELDS = ("T", "S", "u", "v", "ssh")
TARGET = (1, 49, 0)
PLANTS = ("none", "stage1-count", "passivity")


class GateError(RuntimeError):
    """The step-16 walk violated a frozen identity or instrument predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first_nonfinite(arrays: dict[str, np.ndarray]) -> dict[str, object] | None:
    for name, values in arrays.items():
        locations = np.argwhere(~np.isfinite(values))
        if locations.size:
            index = tuple(int(value) for value in locations[0])
            return {"field": name, "index": list(index),
                    "value": str(values[index])}
    return None


def boundary_summary(arrays: dict[str, object]) -> dict[str, object]:
    values = {name: np.asarray(value) for name, value in arrays.items()}
    require(values, "empty boundary")
    require(all(value.dtype == np.dtype(np.float64) for value in values.values()),
            "boundary is not fp64")
    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(value)))
        for name, value in values.items()
    }
    finite_max = {
        name: (float(np.max(np.abs(value[np.isfinite(value)])))
               if bool(np.isfinite(value).any()) else None)
        for name, value in values.items()
    }
    target = {}
    for name in ("T", "S"):
        if name in values and values[name].ndim == 3:
            target[name] = str(values[name][TARGET])
    return {
        "fields": list(values),
        "nonfinite": nonfinite,
        "nonfinite_total": sum(nonfinite.values()),
        "first_nonfinite": _first_nonfinite(values),
        "finite_max_abs": finite_max,
        "target_jik": list(TARGET),
        "target_values": target,
    }


def _state_arrays(state) -> dict[str, np.ndarray]:
    return rung0.candidate_fields(state)


def _tracer_arrays(temperature, salinity) -> dict[str, np.ndarray]:
    return {"T": np.asarray(temperature), "S": np.asarray(salinity)}


def _bit_equal(left, right) -> bool:
    a = np.ascontiguousarray(np.asarray(left))
    b = np.ascontiguousarray(np.asarray(right))
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def state_bit_rows(left, right) -> dict[str, bool]:
    a, b = _state_arrays(left), _state_arrays(right)
    return {name: _bit_equal(a[name], b[name]) for name in FIELDS}


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "stage1-count":
        report["boundaries"]["stage1_tracer"]["nonfinite"]["T"] = 1
    elif plant == "passivity":
        report["live_trace_state_equal"]["T"] = False

    require(tuple(report.get("boundary_order", ())) == BOUNDARIES,
            "step-16 boundary order changed")
    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "step-16 walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 15,
            "walk did not start from step 15")
    require(all(report["live_trace_state_equal"].values()),
            "live stage trace changed the ordinary step-16 state")

    for name in BOUNDARIES:
        row = report["boundaries"][name]
        total = sum(int(value) for value in row["nonfinite"].values())
        require(total == int(row["nonfinite_total"]),
                f"{name}: non-finite census disagrees")
        require((row["first_nonfinite"] is None) == (total == 0),
                f"{name}: first-nonfinite/census disagreement")

    first = next((name for name in BOUNDARIES
                  if report["boundaries"][name]["nonfinite_total"]), None)
    require(first == report.get("first_nonfinite_boundary"),
            "first non-finite boundary is not source ordered")
    returned = report["boundaries"]["returned_state"]["first_nonfinite"]
    require(returned == {"field": "T", "index": list(TARGET), "value": "nan"},
            "round-130 step-16 failure did not reproduce")

    predictions = {
        "R131-P1": {
            "status": "CONFIRMED",
            "observed": returned,
        },
        "R131-P2": {
            "status": ("CONFIRMED" if first == "stage3_advection_content"
                       else "REFUTED"),
            "predicted": "stage3_advection_content",
            "observed": first,
        },
        "R131-P3": {
            "status": ("CONFIRMED" if first == "stage3_advection_content"
                       and all(report["boundaries"][name]["nonfinite_total"]
                               for name in BOUNDARIES[3:]) else "REFUTED"),
            "observed": [name for name in BOUNDARIES[3:]
                         if report["boundaries"][name]["nonfinite_total"]],
        },
        "R131-P4": {
            "status": "CONFIRMED",
            "observed": report["live_trace_state_equal"],
        },
        "R131-P5": {
            "status": "CONFIRMED",
            "observed": "measurement-only",
        },
    }
    return {**report, "status": "PASS_ROUND131_STEP16_WALK",
            "prediction_ledger": predictions}


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-131 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-131 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "step-16 walk requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    require(card.unmeasured_features == ("linear_implicit_bottom_drag",),
            "rung-0 unmeasured-feature registry changed")
    freshwater, surface = ladder._zero_forcing(
        tuple(np.asarray(card.recipe.initial_state.eta.data).shape))

    def model(hooks=None):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=(hooks or _NEMOWSRK3TestHooks()))

    ordinary = model()
    state = card.recipe.initial_state
    started = time.time()
    for step in range(1, 16):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        summary = boundary_summary(_state_arrays(state))
        require(summary["nonfinite_total"] == 0,
                f"trajectory became non-finite before step 16: step={step}")
        if step % 5 == 0:
            print(f"STEP16_WALK_PROGRESS step={step}/15 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    state15 = state
    returned = jax.device_get(ordinary.step(
        state15, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    live = model(_NEMOWSRK3TestHooks(expose_live_stage_operands=True))
    live_trace = jax.device_get(live.step(
        state15, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(len(live_trace.stage_outputs) == 3,
            "live stage trace is incomplete")

    def exposed(hooks):
        return jax.device_get(model(hooks).step(
            state15, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))

    adv_content = exposed(_NEMOWSRK3TestHooks(
        expose_stage3_advection_content=True))
    pre_content = exposed(_NEMOWSRK3TestHooks(
        expose_pre_implicit_content=True))
    pre_concentration = exposed(_NEMOWSRK3TestHooks(
        expose_pre_implicit_state=True))

    stage1 = live_trace.stage_outputs[0]
    stage2 = live_trace.stage_outputs[1]
    boundaries = {
        "step15_entry": boundary_summary(_state_arrays(state15)),
        "stage1_tracer": boundary_summary(
            _tracer_arrays(stage1[2], stage1[3])),
        "stage2_tracer": boundary_summary(
            _tracer_arrays(stage2[2], stage2[3])),
        "stage3_advection_content": boundary_summary(
            _tracer_arrays(adv_content.T.data, adv_content.S.data)),
        "pre_implicit_content": boundary_summary(
            _tracer_arrays(pre_content.T.data, pre_content.S.data)),
        "pre_implicit_concentration": boundary_summary(
            _tracer_arrays(pre_concentration.T.data,
                           pre_concentration.S.data)),
        "returned_state": boundary_summary(_state_arrays(returned)),
    }
    first = next((name for name in BOUNDARIES
                  if boundaries[name]["nonfinite_total"]), None)
    return {
        "format": "nemo-testcase-l4-orca2-round131-step16-walk-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 15,
        "unmeasured_features": list(card.unmeasured_features),
        "boundary_order": list(BOUNDARIES),
        "boundaries": boundaries,
        "first_nonfinite_boundary": first,
        "live_trace_state_equal": state_bit_rows(
            live_trace.state_after, returned),
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "stage_order": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227",
            "tracer_accumulator": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:548-604",
            "vertical_solve": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:97-107",
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
                    "runtime mode requires deck root and commit stamp")
            raw = measure(args.deck_root, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND131_STEP16_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
