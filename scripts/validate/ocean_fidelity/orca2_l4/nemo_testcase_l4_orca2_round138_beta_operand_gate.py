#!/usr/bin/env python3
"""Walk the adjacent beta operands feeding ORCA2's step-36 V coefficient."""

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


TARGET_CELL = (86, 159, 3)
TARGET_FACE = (87, 159, 3)
ADJACENT_CELLS = ((86, 159, 3), (87, 159, 3))
BETA_CELL_FIELDS = (
    "zup", "zdo", "zpos", "zneg", "zbt",
    "zbetup_literal", "zbetdo_literal", "r_in", "r_out",
)
PLANTS = ("none", "registry", "passivity", "coefficient", "adjacency",
          "source-order", "live-selection")


class GateError(RuntimeError):
    """The adjacent-beta record violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first_nonfinite(rows: dict[str, object]) -> str | None:
    return next((name for name in BETA_CELL_FIELDS
                 if int(rows[name]["nonfinite_count"])), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "registry":
        report["beta_field_order"][0] = "wrong"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "coefficient":
        report["coefficient_nonfinite_masks_equal"]["coef_v"] = False
    elif plant == "adjacency":
        report["adjacent_cells"][1] = [0, 0, 0]
    elif plant == "source-order":
        report["first_nonfinite_beta_operand"] = "wrong"
    elif plant == "live-selection":
        report["face_selection"]["live_selected_nonfinite"] = False

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "beta walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("target_cell", ())) == TARGET_CELL,
            "level-3 target cell changed")
    require(tuple(report.get("target_face", ())) == TARGET_FACE,
            "north incident V face changed")
    require(tuple(map(tuple, report.get("adjacent_cells", ()))) == ADJACENT_CELLS,
            "V-face adjacent-cell association changed")
    require(tuple(report.get("beta_field_order", ())) == BETA_CELL_FIELDS,
            "beta field registry changed")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in state_gate.FIELDS},
        "ordinary step-36 repeat changed bits")
    observer_state = report.get("observer_state_equal", {})
    require(observer_state and all(observer_state.values())
            and "mass_flux_w" not in observer_state,
            "stage-3 FCT input side output moved ordinary leaves: "
            f"{observer_state}")
    require(report.get("coefficient_nonfinite_masks_equal") == {
        name: True for name in ("coef_u", "coef_v", "coef_w")},
        "beta observer changed a coefficient's finite/non-finite support")
    require(report.get("returned_first_nonfinite") == {
        "field": "T", "index": [86, 159, 0], "value": "nan"},
        "step-36 returned failure changed")

    rows = report["beta_rows"]
    for name in BETA_CELL_FIELDS:
        require(tuple(map(tuple, rows[name]["indices"])) == ADJACENT_CELLS,
                f"{name}: adjacent-cell indices changed")
        require(int(rows[name]["nonfinite_count"]) == sum(
            int(value) for value in rows[name]["nonfinite"]),
            f"{name}: non-finite census disagrees")
    first = _first_nonfinite(rows)
    require(first == report.get("first_nonfinite_beta_operand"),
            "first beta operand is not source ordered")
    require(first is not None, "beta trace did not reach a non-finite operand")

    face = report["face_selection"]
    require(face.get("antidiffusive_v_sign") == "positive",
            "north incident V-face sign changed")
    require(face.get("selected_live_pair") == ["r_out_south", "r_in_north"],
            "positive V-face selected the wrong adjacent live ratios")
    require(face.get("live_selected_nonfinite") is True,
            "live adjacent ratios no longer reproduce the non-finite face")
    require(face.get("coefficient_nonfinite") is True,
            "round-137 V coefficient is no longer non-finite")

    report["prediction_ledger"] = {
        "R138-P1": {
            "status": "CONFIRMED",
            "observed": "ordinary state exact; controls fired",
        },
        "R138-P2": {
            "status": "CONFIRMED" if first in ("zpos", "zneg") else "REFUTED",
            "predicted": "zpos or zneg",
            "observed": first,
        },
        "R138-P3": {
            "status": "CONFIRMED",
            "observed": f"{first} -> live ratio pair -> coef_v",
        },
        "R138-P4": {
            "status": "UNMEASURED",
            "observed": "shared gates pending",
        },
    }
    return {**report, "status": "PASS_ROUND138_BETA_OPERAND_WALK"}


def measure(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.advection import (
        NEMO_FCT_BETA_TRACE_FIELDS,
        NEMO_FCT_TRACE_FIELDS,
        fct_tracer_advection,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSFCTInputTrace,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-138 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-138 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "beta walk requires production JIT on CPU")
    require(tuple(NEMO_FCT_BETA_TRACE_FIELDS) == BETA_CELL_FIELDS + (
        "coef_u", "coef_v", "coef_w"), "production beta registry changed")

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
        require(state_gate.boundary_summary(state_gate._state_arrays(state))[
                    "nonfinite_total"] == 0,
                f"trajectory became non-finite before step 36: step={step}")
        if step % 5 == 0:
            print(f"ROUND138_BETA_PROGRESS step={step}/35 "
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

    def trace(expose_beta: bool):
        @jax.jit
        def run(now, before):
            result = fct_tracer_advection(
                now, inputs.mass_flux_u, inputs.mass_flux_v,
                inputs.w_explicit, inputs.thickness, card.recipe.grid,
                card.dt_s, high_order="centred2", tracer_before=before,
                active_mask=inputs.active_mask,
                low_order_predictor="nemo_rk3_two_step",
                base_thickness=inputs.base_thickness,
                after_thickness=inputs.after_thickness,
                implicit_w=inputs.w_implicit,
                return_nemo_trace=not expose_beta,
                return_nemo_beta_trace=expose_beta)
            return result[2]
        return jax.device_get(run(inputs.tracer_a, inputs.tracer_a_before))

    standard_values = trace(False)
    beta_values = trace(True)
    standard = dict(zip(NEMO_FCT_TRACE_FIELDS, standard_values, strict=True))
    beta = dict(zip(NEMO_FCT_BETA_TRACE_FIELDS, beta_values, strict=True))
    coefficient_masks = {
        name: bool(np.array_equal(
            np.isfinite(np.asarray(beta[name])),
            np.isfinite(np.asarray(standard[name]))))
        for name in ("coef_u", "coef_v", "coef_w")
    }

    rows = {}
    for name in BETA_CELL_FIELDS:
        values = np.asarray(beta[name])
        require(values.dtype == np.dtype(np.float64),
                f"{name}: beta trace is not fp64")
        adjacent = [values[index] for index in ADJACENT_CELLS]
        rows[name] = {
            "indices": [list(index) for index in ADJACENT_CELLS],
            "values": [str(value) for value in adjacent],
            "nonfinite": [bool(not np.isfinite(value)) for value in adjacent],
            "nonfinite_count": sum(not np.isfinite(value) for value in adjacent),
        }

    anti_v = np.asarray(standard["anti_pre_v"])[TARGET_FACE]
    coef_v = np.asarray(standard["coef_v"])[TARGET_FACE]
    live_pair = (np.asarray(beta["r_out"])[ADJACENT_CELLS[0]],
                 np.asarray(beta["r_in"])[ADJACENT_CELLS[1]])
    literal_pair = (
        np.asarray(beta["zbetdo_literal"])[ADJACENT_CELLS[0]],
        np.asarray(beta["zbetup_literal"])[ADJACENT_CELLS[1]],
    )
    returned_summary = state_gate.boundary_summary(
        state_gate._state_arrays(returned))
    return {
        "format": "nemo-testcase-l4-orca2-round138-beta-operand-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "target_cell": list(TARGET_CELL),
        "target_face": list(TARGET_FACE),
        "adjacent_cells": [list(index) for index in ADJACENT_CELLS],
        "beta_field_order": list(BETA_CELL_FIELDS),
        "beta_rows": rows,
        "first_nonfinite_beta_operand": _first_nonfinite(rows),
        "face_selection": {
            "antidiffusive_v": str(anti_v),
            "antidiffusive_v_sign": "positive" if anti_v > 0.0 else "negative",
            "selected_live_pair": ["r_out_south", "r_in_north"],
            "live_pair_values": [str(value) for value in live_pair],
            "live_selected_nonfinite": bool(not np.isfinite(np.min(live_pair))),
            "literal_pair_values": [str(value) for value in literal_pair],
            "literal_selected_nonfinite": bool(
                not np.isfinite(np.min((1.0,) + literal_pair))),
            "coefficient": str(coef_v),
            "coefficient_nonfinite": bool(not np.isfinite(coef_v)),
        },
        "coefficient_nonfinite_masks_equal": coefficient_masks,
        "returned_first_nonfinite": returned_summary["first_nonfinite"],
        "ordinary_repeat_state_equal": state_gate.state_bit_rows(
            returned_repeat, returned),
        "observer_state_equal": passive._ordinary_state_equal(
            observed, returned),
        "unmeasured_features": list(card.unmeasured_features),
        "worktree": stamp,
        "wall_seconds": time.time() - started,
        "compiled_citations": {
            "beta_operands":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:849-878",
            "v_face_coefficient":
                "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:910-913",
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
    print("STATUS PASS_ROUND138_BETA_OPERAND_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
