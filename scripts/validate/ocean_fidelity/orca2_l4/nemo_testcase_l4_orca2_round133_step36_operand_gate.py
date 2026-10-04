#!/usr/bin/env python3
"""Walk rung-0's independent step-36 stage-1 transport operands."""

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


OPERAND_ORDER = (
    "stage1_thickness",
    "stage1_transport_average",
    "stage1_corrected_velocity",
    "stage1_metric_transport",
)
FIELDS = prior.FIELDS
TARGET = (86, 159, 0)
PLANTS = ("none", "earlier-boundary", "passivity")


class GateError(RuntimeError):
    """The step-36 operand walk violated a frozen instrument predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def array_summary(arrays: dict[str, object]) -> dict[str, object]:
    values = {name: np.asarray(value) for name, value in arrays.items()}
    require(values, "empty operand boundary")
    require(all(value.dtype == np.dtype(np.float64) for value in values.values()),
            "operand boundary is not fp64")
    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(value)))
        for name, value in values.items()
    }
    first = prior._first_nonfinite(values)
    return {
        "fields": list(values),
        "shapes": {name: list(value.shape) for name, value in values.items()},
        "nonfinite": nonfinite,
        "nonfinite_total": sum(nonfinite.values()),
        "first_nonfinite": first,
        "finite_max_abs": {
            name: (float(np.max(np.abs(value[np.isfinite(value)])))
                   if bool(np.isfinite(value).any()) else None)
            for name, value in values.items()
        },
    }


def _first_operand(report: dict[str, object]) -> str | None:
    return next((name for name in OPERAND_ORDER
                 if report["operands"][name]["nonfinite_total"]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "earlier-boundary":
        report["operands"]["stage1_thickness"]["nonfinite"]["hu"] = 1
        report["operands"]["stage1_thickness"]["nonfinite_total"] = 1
        report["operands"]["stage1_thickness"]["first_nonfinite"] = {
            "field": "hu", "index": [0, 0, 0], "value": "nan"}
    elif plant == "passivity":
        report["passivity"]["stage1_thickness"]["T"] = False

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "step-36 walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("operand_order", ())) == OPERAND_ORDER,
            "operand source order changed")
    require(report.get("step35_entry", {}).get("nonfinite_total") == 0,
            "step-35 entry is not finite")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in FIELDS},
        "ordinary step-36 repeat changed bits")
    require(report.get("returned_state", {}).get("first_nonfinite") == {
        "field": "T", "index": list(TARGET), "value": "nan"},
        "round-132 step-36 failure did not reproduce")

    for name in OPERAND_ORDER:
        row = report["operands"][name]
        require(sum(int(value) for value in row["nonfinite"].values())
                == int(row["nonfinite_total"]),
                f"{name}: non-finite census disagrees")
        require((row["first_nonfinite"] is None)
                == (row["nonfinite_total"] == 0),
                f"{name}: first-nonfinite/census disagreement")
    first = _first_operand(report)
    require(first == report.get("first_nonfinite_operand"),
            "first non-finite operand is not source ordered")

    expected_passivity = {
        "stage1_thickness": ("T", "S", "ssh"),
        "stage1_transport_average": ("T", "S", "ssh"),
        "stage1_corrected_velocity": ("T", "S", "ssh"),
        "stage1_metric_transport": ("S", "ssh"),
    }
    for name, fields in expected_passivity.items():
        require(tuple(report["passivity"][name]) == fields,
                f"{name}: passivity field registry changed")
        require(all(report["passivity"][name].values()),
                f"{name}: write-only exposure changed an ordinary field")

    corrected_first = first == "stage1_corrected_velocity"
    transport_nonfinite = bool(
        report["operands"]["stage1_metric_transport"]["nonfinite_total"])
    predictions = {
        "R133-P1": {"status": "CONFIRMED", "observed": TARGET},
        "R133-P2": {
            "status": "CONFIRMED" if corrected_first else "REFUTED",
            "predicted": "stage1_corrected_velocity", "observed": first,
        },
        "R133-P3": {
            "status": "CONFIRMED" if transport_nonfinite else "REFUTED",
            "observed_nonfinite": transport_nonfinite,
        },
        "R133-P4": {"status": "CONFIRMED", "observed": "passive"},
        "R133-P5": {"status": "CONFIRMED", "observed": "measurement-only"},
    }
    return {**report, "status": "PASS_ROUND133_STEP36_OPERAND_WALK",
            "prediction_ledger": predictions}


def _unchanged_fields(exposed, ordinary, names: tuple[str, ...]) -> dict[str, bool]:
    left = rung0.candidate_fields(exposed)
    right = rung0.candidate_fields(ordinary)
    return {name: prior._bit_equal(left[name], right[name]) for name in names}


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-133 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-133 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "step-36 walk requires production JIT on CPU")

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

    ordinary_model = model()
    state = card.recipe.initial_state
    started = time.time()
    for step in range(1, 36):
        state = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        summary = prior.boundary_summary(prior._state_arrays(state))
        require(summary["nonfinite_total"] == 0,
                f"trajectory became non-finite before step 36: step={step}")
        if step % 5 == 0:
            print(f"STEP36_OPERAND_PROGRESS step={step}/35 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    state35 = state
    returned = jax.device_get(ordinary_model.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    returned_repeat = jax.device_get(ordinary_model.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    def exposed(hooks):
        return jax.device_get(model(hooks).step(
            state35, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))

    thickness = exposed(_NEMOWSRK3TestHooks(
        expose_stage1_transport_operand="thickness"))
    average = exposed(_NEMOWSRK3TestHooks(
        expose_stage1_transport_operand="transport_average"))
    corrected = exposed(_NEMOWSRK3TestHooks(
        expose_stage1_transport_operand="corrected_velocity"))
    transport = exposed(_NEMOWSRK3TestHooks(
        expose_tracer_transport_stage=1))

    operands = {
        "stage1_thickness": array_summary({
            "hu": thickness.u.data, "hv": thickness.v.data}),
        "stage1_transport_average": array_summary({
            "un_adv": average.u.data, "vn_adv": average.v.data}),
        "stage1_corrected_velocity": array_summary({
            "u_corrected": corrected.u.data,
            "v_corrected": corrected.v.data}),
        "stage1_metric_transport": array_summary({
            "zFu": transport.u.data, "zFv": transport.v.data,
            "zFw": transport.T.data}),
    }
    report = {
        "format": "nemo-testcase-l4-orca2-round133-step36-operands-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "unmeasured_features": list(card.unmeasured_features),
        "operand_order": list(OPERAND_ORDER),
        "step35_entry": prior.boundary_summary(prior._state_arrays(state35)),
        "operands": operands,
        "first_nonfinite_operand": next(
            (name for name in OPERAND_ORDER
             if operands[name]["nonfinite_total"]), None),
        "returned_state": prior.boundary_summary(prior._state_arrays(returned)),
        "ordinary_repeat_state_equal": prior.state_bit_rows(
            returned_repeat, returned),
        "passivity": {
            "stage1_thickness": _unchanged_fields(
                thickness, returned, ("T", "S", "ssh")),
            "stage1_transport_average": _unchanged_fields(
                average, returned, ("T", "S", "ssh")),
            "stage1_corrected_velocity": _unchanged_fields(
                corrected, returned, ("T", "S", "ssh")),
            "stage1_metric_transport": _unchanged_fields(
                transport, returned, ("S", "ssh")),
        },
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "external_before_stage1":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:200-212",
            "correction_and_transport":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:265-284",
        },
    }
    return report


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
    except (GateError, prior.GateError, rung0.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND133_STEP36_OPERAND_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
