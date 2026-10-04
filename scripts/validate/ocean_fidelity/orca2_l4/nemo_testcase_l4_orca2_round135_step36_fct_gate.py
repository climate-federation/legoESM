#!/usr/bin/env python3
"""Admit a passive stage-3 FCT input side output and walk its arithmetic."""

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
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round134_step36_fct_gate as rejected,
)


TRACE_FIELD_ORDER = rejected.TRACE_FIELD_ORDER
GROUP_ORDER = rejected.GROUP_ORDER
TARGET = rejected.TARGET
FIELDS = prior.FIELDS
PLANTS = ("none", "source-order", "passivity", "support")


class GateError(RuntimeError):
    """The passive FCT input exposure or source-order walk is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _array_bits_equal(left, right) -> bool:
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return (left.dtype == right.dtype and left.shape == right.shape
            and np.array_equal(left.view(np.uint8), right.view(np.uint8)))


def _ordinary_state_equal(observed, ordinary) -> dict[str, bool]:
    """Compare every state slot except the private side-output slot."""

    import jax

    rows: dict[str, bool] = {}
    for name in ordinary._fields:
        if name == "mass_flux_w":
            continue
        left, right = getattr(observed, name), getattr(ordinary, name)
        left_leaves, left_tree = jax.tree_util.tree_flatten(left)
        right_leaves, right_tree = jax.tree_util.tree_flatten(right)
        equal = left_tree == right_tree and len(left_leaves) == len(right_leaves)
        if equal:
            equal = all(_array_bits_equal(a, b)
                        for a, b in zip(left_leaves, right_leaves, strict=True))
        rows[name] = bool(equal)
    return rows


def _first_nonfinite(groups: dict[str, object], key: str) -> str | None:
    return next((name for name in GROUP_ORDER if groups[name][key]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "source-order":
        report["first_nonfinite_group"] = "first_upstream_flux"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "support":
        report["groups"]["antidiffusive_flux"]["active_count"] += 1

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "FCT walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("trace_field_order", ())) == TRACE_FIELD_ORDER,
            "FCT trace field order changed")
    require(tuple(report.get("group_order", ())) == GROUP_ORDER,
            "FCT group order changed")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in FIELDS}, "ordinary step-36 repeat changed bits")
    require(all(report.get("observer_state_equal", {}).values()),
            "post-step FCT side output moved an ordinary state leaf")
    require(report.get("returned_first_nonfinite") == {
        "field": "T", "index": list(TARGET), "value": "nan"},
        "merged step-36 failure did not reproduce")
    require(report.get("side_output_type") == "_NEMOWSFCTInputTrace"
            and report.get("side_output_field") == "mass_flux_w",
            "FCT side output scope changed")

    for group, fields in rejected.TRACE_GROUPS:
        row = report["groups"][group]
        require(tuple(row["fields"]) == fields,
                f"{group}: field registry changed")
        require(int(row["nonfinite_total"]) == sum(
            int(value) for value in row["nonfinite"].values()),
            f"{group}: non-finite census disagrees")
        require(int(row["target_nonfinite_total"]) == sum(
            int(value) for value in row["target_nonfinite"].values()),
            f"{group}: target census disagrees")
        expected_active = sum(int(row["support_count"][name]) for name in fields)
        require(int(row["active_count"]) == expected_active,
                f"{group}: active support census disagrees")

    first = _first_nonfinite(report["groups"], "nonfinite_total")
    first_target = _first_nonfinite(report["groups"], "target_nonfinite_total")
    require(first == report.get("first_nonfinite_group"),
            "first active-support non-finite group is not source ordered")
    require(first_target == report.get("first_target_nonfinite_group"),
            "first target non-finite group is not source ordered")
    require(first is not None and first_target is not None,
            "FCT walk did not reach a non-finite boundary")
    report["prediction_ledger"] = {
        "R135-P3": {"status": "CONFIRMED",
                     "observed": report["returned_first_nonfinite"]},
        "R135-P4": {"status": "CONFIRMED",
                     "observed": "all ordinary state leaves bit-identical"},
        "R135-P5": {
            "status": "CONFIRMED" if first == "antidiffusive_flux" else "REFUTED",
            "predicted": "antidiffusive_flux", "observed": first,
        },
        "R135-P6": {
            "status": "CONFIRMED" if first_target == first else "REFUTED",
            "global": first, "target": first_target,
        },
        "R135-P7": {"status": "UNMEASURED", "observed": "shared gates pending"},
    }
    return {**report, "status": "PASS_ROUND135_STEP36_FCT_WALK"}


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.advection import NEMO_FCT_TRACE_FIELDS, fct_tracer_advection
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSFCTInputTrace,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-135 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-135 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "FCT walk requires production JIT on CPU")
    require(tuple(NEMO_FCT_TRACE_FIELDS) == TRACE_FIELD_ORDER,
            "production FCT trace registry changed")

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
            print(f"ROUND135_FCT_PROGRESS step={step}/35 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    state35 = state
    returned = jax.device_get(ordinary.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    returned_repeat = jax.device_get(ordinary.step(
        state35, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed = jax.device_get(model(_NEMOWSRK3TestHooks(
        expose_stage3_fct_inputs=True)).step(
            state35, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    require(isinstance(observed.mass_flux_w, _NEMOWSFCTInputTrace),
            "stage-3 FCT side output has the wrong type")
    inputs = observed.mass_flux_w

    content_state = jax.device_get(model(_NEMOWSRK3TestHooks(
        expose_stage3_advection_content=True)).step(
            state35, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))

    shared = ("mass_flux_u", "mass_flux_v", "w_explicit", "thickness",
              "active_mask", "base_thickness", "after_thickness", "w_implicit")
    for name in shared:
        require(np.asarray(getattr(inputs, name)).dtype in (
                    np.dtype(np.float64), np.dtype(bool)),
                f"{name}: side output dtype is not fp64/bool")

    @jax.jit
    def isolated_trace(now, before):
        _div_h, _div_w, trace = fct_tracer_advection(
            now, inputs.mass_flux_u, inputs.mass_flux_v,
            inputs.w_explicit, inputs.thickness, card.recipe.grid, card.dt_s,
            high_order="centred2", tracer_before=before,
            active_mask=inputs.active_mask,
            low_order_predictor="nemo_rk3_two_step",
            base_thickness=inputs.base_thickness,
            after_thickness=inputs.after_thickness,
            implicit_w=inputs.w_implicit, return_nemo_trace=True)
        return trace

    traces = {}
    for tracer, now, before in (
        ("T", inputs.tracer_a, inputs.tracer_a_before),
        ("S", inputs.tracer_b, inputs.tracer_b_before),
    ):
        trace = jax.device_get(isolated_trace(now, before))
        traces[tracer] = dict(zip(TRACE_FIELD_ORDER, trace, strict=True))

    supports = rejected._support_masks(
        np.asarray(card.recipe.z_coord.is_active, dtype=bool))
    active = supports["cell"]
    content_arrays = {
        "T": np.asarray(content_state.T.data),
        "S": np.asarray(content_state.S.data),
    }
    content_summary = prior.boundary_summary({
        name: np.where(active, values, 0.0)
        for name, values in content_arrays.items()})
    j, i, k = TARGET
    content_summary["active_count"] = int(np.count_nonzero(active))
    content_summary["target_nonfinite"] = {
        name: int(not np.isfinite(values[j, i, k]))
        for name, values in content_arrays.items()
    }
    groups = rejected._group_summaries(traces, supports, content_summary)
    first = _first_nonfinite(groups, "nonfinite_total")
    first_target = _first_nonfinite(groups, "target_nonfinite_total")
    returned_summary = prior.boundary_summary(prior._state_arrays(returned))
    return {
        "format": "nemo-testcase-l4-orca2-round135-step36-fct-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "unmeasured_features": list(card.unmeasured_features),
        "trace_field_order": list(TRACE_FIELD_ORDER),
        "group_order": list(GROUP_ORDER),
        "groups": groups,
        "first_nonfinite_group": first,
        "first_target_nonfinite_group": first_target,
        "returned_first_nonfinite": returned_summary["first_nonfinite"],
        "ordinary_repeat_state_equal": prior.state_bit_rows(
            returned_repeat, returned),
        "observer_state_equal": _ordinary_state_equal(observed, returned),
        "side_output_type": type(inputs).__name__,
        "side_output_field": "mass_flux_w",
        "stage3_advection_content": content_summary,
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "upstream": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-610",
            "antidiffusive": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:193-200,260-280",
            "limiter": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:314-316,743-938",
            "final_rhs": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:318-330",
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
    except (GateError, rejected.GateError, prior.GateError, rung0.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND135_STEP36_FCT_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
