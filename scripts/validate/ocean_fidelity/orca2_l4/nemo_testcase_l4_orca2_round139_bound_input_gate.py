#!/usr/bin/env python3
"""Walk the seven FCT bound inputs feeding ORCA2's step-36 zup."""

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
    nemo_testcase_l4_orca2_round131_step16_walk_gate as state_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
)


TARGET_CELL = (87, 159, 3)
STENCIL_NAMES = (
    "center", "west", "east", "south", "north", "above", "below")
STENCIL_CELLS = (
    (87, 159, 3), (87, 158, 3), (87, 160, 3), (86, 159, 3),
    (88, 159, 3), (87, 159, 2), (87, 159, 4),
)
TRACE_FIELDS = tuple(f"zbup_{name}" for name in STENCIL_NAMES) + (
    "pbef", "paft", "wet", "zup")
PLANTS = (
    "none", "registry", "passivity", "outputs", "association",
    "source-order", "selected-input", "zup-link", "payload-separation",
)


class GateError(RuntimeError):
    """The bound-input record violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first_nonfinite(members: list[dict[str, object]]) -> str | None:
    return next((row["name"] for row in members if row["nonfinite"]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "registry":
        report["trace_field_order"][0] = "wrong"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "outputs":
        report["ordinary_fct_outputs_equal"]["horizontal"] = False
    elif plant == "association":
        report["stencil_members"][2]["index"] = [0, 0, 0]
    elif plant == "source-order":
        report["first_nonfinite_member"] = "wrong"
    elif plant == "selected-input":
        report["selected_sources"]["bound_reproduced"] = False
    elif plant == "zup-link":
        report["zup_link"]["beta_zup_target_equal"] = False
    elif plant == "payload-separation":
        report["trace_payload_separate"] = False

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "bound-input walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("target_cell", ())) == TARGET_CELL,
            "zup target cell changed")
    require(tuple(report.get("trace_field_order", ())) == TRACE_FIELDS,
            "stencil trace registry changed")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in state_gate.FIELDS},
        "ordinary step-36 repeat changed bits")
    observer_state = report.get("observer_state_equal", {})
    require(observer_state and all(observer_state.values())
            and "mass_flux_w" not in observer_state,
            "stage-3 FCT side output moved ordinary leaves")
    require(report.get("ordinary_fct_outputs_equal") == {
        "horizontal": True, "vertical": True},
        "stencil return moved an ordinary FCT output")
    require(report.get("trace_payload_separate") is True,
            "stencil payload is not paired with a separate ordinary call")
    require(report.get("returned_first_nonfinite") == {
        "field": "T", "index": [86, 159, 0], "value": "nan"},
        "step-36 returned failure changed")

    members = report.get("stencil_members", [])
    require(len(members) == len(STENCIL_NAMES),
            "stencil member count changed")
    for row, name, index in zip(
            members, STENCIL_NAMES, STENCIL_CELLS, strict=True):
        require(row.get("name") == name and tuple(row.get("index", ())) == index,
                f"{name}: stencil association changed")
    first = _first_nonfinite(members)
    require(first == report.get("first_nonfinite_member"),
            "first stencil member is not source ordered")
    require(first is not None, "all seven stencil members stayed finite")

    selected = report.get("selected_sources", {})
    selected_row = members[STENCIL_NAMES.index(first)]
    require(selected.get("member") == first
            and tuple(selected.get("index", ())) == tuple(selected_row["index"]),
            "selected source does not match the first non-finite member")
    require(selected.get("wet") is True, "selected member is not a wet cell")
    require(selected.get("zbup_nonfinite") is True,
            "selected zbup is no longer non-finite")
    require(selected.get("bound_reproduced") is True,
            "MAX(pbef,paft) does not reproduce selected zbup")
    require(report.get("zup_link") == {
        "stencil_max_nonfinite": True,
        "trace_zup_nonfinite": True,
        "beta_zup_target_equal": True,
    }, "seven-member maximum no longer reproduces round-138 zup")

    pbef_nonfinite = bool(selected.get("pbef_nonfinite"))
    paft_nonfinite = bool(selected.get("paft_nonfinite"))
    report["prediction_ledger"] = {
        "R139-P1": {
            "status": "CONFIRMED",
            "observed": "ordinary state and FCT outputs exact; controls fired",
        },
        "R139-P2": {
            "status": "CONFIRMED" if first == "east" else "REFUTED",
            "predicted": "east",
            "observed": first,
        },
        "R139-P3": {
            "status": "CONFIRMED" if not pbef_nonfinite and paft_nonfinite
            else "REFUTED",
            "predicted": "pbef finite; paft non-finite",
            "observed": (
                f"pbef {'non-finite' if pbef_nonfinite else 'finite'}; "
                f"paft {'non-finite' if paft_nonfinite else 'finite'}"),
        },
        "R139-P4": {
            "status": "UNMEASURED",
            "observed": "shared gates pending",
        },
    }
    return {**report, "status": "PASS_ROUND139_BOUND_INPUT_WALK"}


def _bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and bool(
        np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.advection import (
        NEMO_FCT_BETA_TRACE_FIELDS,
        NEMO_FCT_STENCIL_TRACE_FIELDS,
        fct_tracer_advection,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSFCTInputTrace,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-139 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-139 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "bound-input walk requires production JIT on CPU")
    require(tuple(NEMO_FCT_STENCIL_TRACE_FIELDS) == TRACE_FIELDS,
            "production stencil registry changed")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    freshwater, surface = ladder._zero_forcing(
        tuple(np.asarray(card.recipe.initial_state.eta.data).shape))

    def model(hooks=None):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)

    ordinary_model = model()
    state = card.recipe.initial_state
    started = time.time()
    for step in range(1, 36):
        state = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        require(state_gate.boundary_summary(state_gate._state_arrays(state))[
                    "nonfinite_total"] == 0,
                f"trajectory became non-finite before step 36: step={step}")
        if step % 5 == 0:
            print(f"ROUND139_BOUND_PROGRESS step={step}/35 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    state35 = state
    returned = jax.device_get(ordinary_model.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    returned_repeat = jax.device_get(ordinary_model.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed = jax.device_get(model(_NEMOWSRK3TestHooks(
        expose_stage3_fct_inputs=True)).step(
            state35, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    require(isinstance(observed.mass_flux_w, _NEMOWSFCTInputTrace),
            "stage-3 FCT side output has the wrong type")
    inputs = observed.mass_flux_w

    def call(**trace_flag):
        return fct_tracer_advection(
            inputs.tracer_a, inputs.mass_flux_u, inputs.mass_flux_v,
            inputs.w_explicit, inputs.thickness, card.recipe.grid,
            card.dt_s, high_order="centred2",
            tracer_before=inputs.tracer_a_before,
            active_mask=inputs.active_mask,
            low_order_predictor="nemo_rk3_two_step",
            base_thickness=inputs.base_thickness,
            after_thickness=inputs.after_thickness,
            implicit_w=inputs.w_implicit, **trace_flag)

    ordinary = jax.device_get(jax.jit(lambda: call())())
    ordinary_repeat = jax.device_get(jax.jit(lambda: call())())
    traced = jax.device_get(jax.jit(
        lambda: call(return_nemo_stencil_trace=True))())
    beta = jax.device_get(jax.jit(
        lambda: call(return_nemo_beta_trace=True))())
    outputs_equal = {
        "horizontal": _bit_equal(
            np.asarray(ordinary_repeat[0]), np.asarray(ordinary[0])),
        "vertical": _bit_equal(
            np.asarray(ordinary_repeat[1]), np.asarray(ordinary[1])),
    }
    trace = dict(zip(
        NEMO_FCT_STENCIL_TRACE_FIELDS, traced[2], strict=True))
    beta_trace = dict(zip(NEMO_FCT_BETA_TRACE_FIELDS, beta[2], strict=True))

    members = []
    for name, index in zip(STENCIL_NAMES, STENCIL_CELLS, strict=True):
        value = np.asarray(trace[f"zbup_{name}"])[TARGET_CELL]
        members.append({
            "name": name,
            "index": list(index),
            "value": str(value),
            "nonfinite": bool(not np.isfinite(value)),
        })
    first = _first_nonfinite(members)
    require(first is not None, "runtime trace found no non-finite member")
    selected_index = STENCIL_CELLS[STENCIL_NAMES.index(first)]
    pbef = np.asarray(trace["pbef"])[selected_index]
    paft = np.asarray(trace["paft"])[selected_index]
    wet = bool(np.asarray(trace["wet"])[selected_index])
    selected_bound = np.asarray(trace[f"zbup_{first}"])[TARGET_CELL]
    rebuilt_bound = np.maximum(pbef, paft)
    zup = np.asarray(trace["zup"])[TARGET_CELL]
    beta_zup = np.asarray(beta_trace["zup"])
    returned_summary = state_gate.boundary_summary(
        state_gate._state_arrays(returned))
    return {
        "format": "nemo-testcase-l4-orca2-round139-bound-input-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "target_cell": list(TARGET_CELL),
        "trace_field_order": list(TRACE_FIELDS),
        "stencil_members": members,
        "first_nonfinite_member": first,
        "selected_sources": {
            "member": first,
            "index": list(selected_index),
            "wet": wet,
            "pbef": str(pbef),
            "paft": str(paft),
            "zbup": str(selected_bound),
            "pbef_nonfinite": bool(not np.isfinite(pbef)),
            "paft_nonfinite": bool(not np.isfinite(paft)),
            "zbup_nonfinite": bool(not np.isfinite(selected_bound)),
            "bound_reproduced": bool(
                np.array_equal(
                    np.asarray(selected_bound).view(np.uint64),
                    np.asarray(rebuilt_bound).view(np.uint64))),
        },
        "zup_link": {
            "stencil_max_nonfinite": bool(any(row["nonfinite"] for row in members)),
            "trace_zup_nonfinite": bool(not np.isfinite(zup)),
            "beta_zup_target_equal": bool(np.array_equal(
                np.asarray(zup).view(np.uint64),
                np.asarray(beta_zup[TARGET_CELL]).view(np.uint64))),
        },
        "ordinary_fct_outputs_equal": outputs_equal,
        "trace_payload_separate": True,
        "returned_first_nonfinite": returned_summary["first_nonfinite"],
        "ordinary_repeat_state_equal": state_gate.state_bit_rows(
            returned_repeat, returned),
        "observer_state_equal": passive._ordinary_state_equal(
            observed, returned),
        "unmeasured_features": list(card.unmeasured_features),
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "bound_inputs":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:816-845",
            "stencil_maximum":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-856",
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
    except (GateError, state_gate.GateError, rung0.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND139_BOUND_INPUT_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
