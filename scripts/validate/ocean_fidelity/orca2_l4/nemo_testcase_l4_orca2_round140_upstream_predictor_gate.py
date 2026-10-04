#!/usr/bin/env python3
"""Walk the source-ordered FCT predictor that makes round-139 paft infinite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round134_step36_fct_gate as registry,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round131_step16_walk_gate as state_gate,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp


TARGET = (87, 159, 4)
RETURNED_TARGET = (86, 159, 0)
SOURCE_FIELDS = (
    "first_u_raw", "first_v_raw", "first_w_raw", "first_div", "midpoint",
    "average_u_raw", "average_v_raw", "average_w_raw", "explicit_ztra",
    "implicit_ztra", "total_ztra", "base_content", "dt_ztra", "numerator",
    "after_thickness", "paft",
)
PLANTS = (
    "none", "registry", "passivity", "source-order", "paft-link",
    "finite-prefix",
)


class GateError(RuntimeError):
    """The upstream-predictor record violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first_nonfinite(rows: list[dict[str, object]]) -> str | None:
    return next((str(row["field"]) for row in rows if row["nonfinite"]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "registry":
        report["source_field_order"][0] = "wrong"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "source-order":
        report["first_target_nonfinite_field"] = "wrong"
    elif plant == "paft-link":
        report["round139_link"]["paft_nonfinite"] = False
    elif plant == "finite-prefix":
        report["target_rows"][0]["nonfinite"] = 1

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "predictor walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("target", ())) == TARGET,
            "predictor target changed")
    require(tuple(report.get("source_field_order", ())) == SOURCE_FIELDS,
            "source field registry changed")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in passive.FIELDS},
        "ordinary step-36 repeat changed bits")
    require(all(report.get("observer_state_equal", {}).values()),
            "passive side output moved ordinary state")
    require(report.get("ordinary_fct_outputs_equal") == {
        "horizontal": True, "vertical": True},
        "raw predictor return moved ordinary FCT output")
    require(report.get("raw_paft_equal_stencil") is True,
            "raw predictor paft disagrees with round-139 stencil path")
    require(report.get("returned_first_nonfinite") == {
        "field": "T", "index": list(RETURNED_TARGET), "value": "nan"},
        "returned step-36 boundary changed")
    require(report.get("round139_link") == {
        "cell": list(TARGET), "pbef_nonfinite": False,
        "paft_nonfinite": True, "paft": "inf"},
        "round-139 paft boundary changed")

    rows = report.get("target_rows", [])
    require([row.get("field") for row in rows] == list(SOURCE_FIELDS),
            "target rows are not source ordered")
    first = _first_nonfinite(rows)
    require(first == report.get("first_target_nonfinite_field"),
            "first non-finite field is not source ordered")
    require(first is not None, "all predictor intermediates stayed finite")
    first_index = SOURCE_FIELDS.index(first)
    require(all(not row["nonfinite"] for row in rows[:first_index]),
            "a source field before the named owner is non-finite")

    averaged = first in ("average_u_raw", "average_v_raw")
    divisor_finite = not next(
        row for row in rows if row["field"] == "after_thickness")["nonfinite"]
    numerator_bad = bool(next(
        row for row in rows if row["field"] == "numerator")["nonfinite"])
    report["prediction_ledger"] = {
        "R140-P1": {"status": "CONFIRMED",
                     "observed": "ordinary state exact; controls fired"},
        "R140-P2": {"status": "CONFIRMED" if averaged else "REFUTED",
                     "predicted": "incident averaged horizontal face flux",
                     "observed": first},
        "R140-P3": {"status": ("CONFIRMED" if divisor_finite and numerator_bad
                                  else "REFUTED"),
                     "observed": (f"numerator_nonfinite={numerator_bad}; "
                                  f"divisor_finite={divisor_finite}")},
        "R140-P4": {"status": "UNMEASURED", "observed": "EVD plant pending"},
        "R140-P5": {"status": "UNMEASURED", "observed": "shared gates pending"},
    }
    return {**report, "status": "PASS_ROUND140_UPSTREAM_PREDICTOR_WALK"}


def measure(deck_root: Path, expect_commit: str,
            round139_json: Path) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.advection import (
        NEMO_FCT_STENCIL_TRACE_FIELDS,
        NEMO_FCT_UP1_TRACE_FIELDS,
        fct_tracer_advection,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSFCTInputTrace,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-140 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-140 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "predictor walk requires production JIT on CPU")
    require(tuple(NEMO_FCT_UP1_TRACE_FIELDS) == SOURCE_FIELDS,
            "raw predictor registry changed")

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
    for step in range(1, 36):
        state = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        require(state_gate.boundary_summary(state_gate._state_arrays(state))[
                    "nonfinite_total"] == 0,
                f"trajectory became non-finite before step 36: step={step}")
        if step % 5 == 0:
            print(f"ROUND140_RAW_PROGRESS step={step}/35", flush=True)

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
            inputs.w_explicit, inputs.thickness, card.recipe.grid, card.dt_s,
            high_order="centred2", tracer_before=inputs.tracer_a_before,
            active_mask=inputs.active_mask,
            low_order_predictor="nemo_rk3_two_step",
            base_thickness=inputs.base_thickness,
            after_thickness=inputs.after_thickness,
            implicit_w=inputs.w_implicit, **trace_flag)

    ordinary = jax.device_get(jax.jit(lambda: call())())
    ordinary_repeat = jax.device_get(jax.jit(lambda: call())())
    raw = jax.device_get(jax.jit(
        lambda: call(return_nemo_up1_trace=True))())
    stencil = jax.device_get(jax.jit(
        lambda: call(return_nemo_stencil_trace=True))())
    trace = dict(zip(NEMO_FCT_UP1_TRACE_FIELDS, raw[2], strict=True))
    stencil_trace = dict(zip(
        NEMO_FCT_STENCIL_TRACE_FIELDS, stencil[2], strict=True))

    rows = []
    j, i, k = TARGET
    for field in SOURCE_FIELDS:
        values = np.asarray(trace[field])
        if field.endswith("u_raw"):
            indices = ((j, i, k), (j, i + 1, k))
        elif field.endswith("v_raw"):
            indices = ((j, i, k), (j + 1, i, k))
        elif field.endswith("w_raw"):
            indices = ((j, i, k), (j, i, k + 1))
        else:
            indices = (TARGET,)
        selected_values = np.asarray([values[index] for index in indices])
        rows.append({
            "field": field,
            "indices": [list(index) for index in indices],
            "values": [str(value) for value in selected_values],
            "nonfinite": int(np.count_nonzero(~np.isfinite(selected_values))),
        })

    previous = json.loads(round139_json.read_text())
    selected = previous["selected_sources"]
    raw_paft = np.asarray(trace["paft"])[TARGET]
    stencil_paft = np.asarray(stencil_trace["paft"])[TARGET]
    report = {
        "format": "nemo-testcase-l4-orca2-round140-upstream-predictor-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "target": list(TARGET),
        "source_field_order": list(SOURCE_FIELDS),
        "target_rows": rows,
        "first_target_nonfinite_field": _first_nonfinite(rows),
        "ordinary_repeat_state_equal": state_gate.state_bit_rows(
            returned_repeat, returned),
        "observer_state_equal": passive._ordinary_state_equal(
            observed, returned),
        "ordinary_fct_outputs_equal": {
            "horizontal": bool(np.array_equal(
                np.asarray(ordinary[0]).view(np.uint64),
                np.asarray(ordinary_repeat[0]).view(np.uint64))),
            "vertical": bool(np.array_equal(
                np.asarray(ordinary[1]).view(np.uint64),
                np.asarray(ordinary_repeat[1]).view(np.uint64))),
        },
        "raw_paft_equal_stencil": bool(np.array_equal(
            np.asarray(raw_paft).view(np.uint64),
            np.asarray(stencil_paft).view(np.uint64))),
        "returned_first_nonfinite": state_gate.boundary_summary(
            state_gate._state_arrays(returned))["first_nonfinite"],
        "round139_link": {
            "cell": selected["index"],
            "pbef_nonfinite": selected["pbef_nonfinite"],
            "paft_nonfinite": selected["paft_nonfinite"],
            "paft": selected["paft"],
        },
        "retraction": {
            "claim": "scaled average_w is the first production non-finite",
            "killed_by": ("raw vertical flux and raw source-ordered predictor "
                          "trace; the earlier infinity was area scaling"),
        },
        "worktree": stamp,
        "unmeasured_features": list(card.unmeasured_features),
        "compiled_citations": {
            "first_fluxes": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:500-526",
            "midpoint": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:528-539",
            "averaged_fluxes": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:562-596",
            "after_update": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:598-610",
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--round139-json", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.expect_commit is None
                    and args.round139_json is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root is not None and args.expect_commit
                    and args.round139_json is not None,
                    "runtime mode requires deck, commit, and round-139 record")
            raw = measure(args.deck_root, args.expect_commit, args.round139_json)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, passive.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND140_UPSTREAM_PREDICTOR_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
