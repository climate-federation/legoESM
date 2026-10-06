#!/usr/bin/env python3
"""Measure NEMO's final external-mode U/V association on ORCA2 rung 0."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung103,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round111_ladder_compare as compare_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round145_initial_growth_gate as growth,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round156_association_growth_gate as exact,
)


ARMS = ("production", "final_only", "substep_only", "pair")
PLANTS = ("none", "passivity", "boundary-bit", "salinity")


class GateError(RuntimeError):
    """The final-association measurement or ladder predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _barotropic_fields(state) -> dict[str, np.ndarray]:
    require(state.uu_b is not None and state.vv_b is not None,
            "rung-0 state has no carried barotropic velocity")
    return {
        "uu_b": np.asarray(state.uu_b.data)[:, 1:],
        "vv_b": np.asarray(state.vv_b.data)[1:, :],
    }


def _state_exact(left, right) -> bool:
    fields = tuple(rung103.rung0.candidate_fields(left))
    left_fields = {**rung103.rung0.candidate_fields(left), **_barotropic_fields(left)}
    right_fields = {**rung103.rung0.candidate_fields(right), **_barotropic_fields(right)}
    require(tuple(rung103.rung0.candidate_fields(right)) == fields,
            "state field registry changed")
    return all(np.array_equal(left_fields[name], right_fields[name])
               for name in (*fields, "uu_b", "vv_b"))


def _models(card):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    def build(*, substep: bool, final: bool):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                barotropic_external_mode_association=substep,
                barotropic_final_mean_association=final,
            ),
        )

    return {
        "production": build(substep=False, final=False),
        "final_only": build(substep=False, final=True),
        "substep_only": build(substep=True, final=False),
        "pair": build(substep=True, final=True),
        "production_repeat": build(substep=False, final=False),
    }


def measure_boundary(deck_root: Path, record_root: Path, admission: Path,
                     expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"], "round-162 boundary measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-162 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-162 measurement requires production JIT on CPU")

    growth.validate_admission(admission)
    record, census = growth.assemble_record(record_root, 1)
    card = growth.shared.rung0.build_rung0_card(deck_root)
    growth.shared.rung0.validate_rung0_card(card)
    require(card.unmeasured_features == ("linear_implicit_bottom_drag",),
            "rung-0 unmeasured-feature registry changed")
    freshwater, surface = rung103._zero_forcing(
        tuple(np.asarray(card.recipe.initial_state.eta.data).shape))
    outputs = {
        name: jax.device_get(model.step(
            card.recipe.initial_state, card.dt_s,
            freshwater=freshwater, surface_forcing=surface,
        ))
        for name, model in _models(card).items()
    }
    passive = _state_exact(outputs["production"], outputs["production_repeat"])
    oracle = {"uu_b": record["uub_after"], "vv_b": record["vvb_after"]}
    rows = {
        arm: {
            field: exact.exact_pair_row(values[field], oracle[field])
            for field in ("uu_b", "vv_b")
        }
        for arm, state in outputs.items() if arm in ARMS
        for values in (_barotropic_fields(state),)
    }
    movement = {
        arm: {
            field: exact.exact_pair_row(
                _barotropic_fields(outputs[arm])[field],
                _barotropic_fields(outputs["production"])[field],
            )
            for field in ("uu_b", "vv_b")
        }
        for arm in ("final_only", "substep_only", "pair")
    }
    return {
        "format": "nemo-testcase-l4-orca2-round162-final-association-v1",
        "status": "PASS_R162_FINAL_ASSOCIATION_BOUNDARY",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": growth.validate_admission(admission),
        "record_census": census,
        "production_repeat_passive": passive,
        "rows_against_nemo": rows,
        "movement_against_production": movement,
        "worktree": stamp,
        "compiled_citations": {
            "substep_association":
                "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779",
            "primary_transport_sum":
                "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:850-868",
            "final_velocity_and_association":
                "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:924-937",
        },
    }


def _unequal_total(block: dict[str, dict[str, object]]) -> int:
    return sum(int(block[field]["differing_cells"]) for field in ("uu_b", "vv_b"))


def classify(boundary: dict, production: dict, substep: dict, pair: dict,
             *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    boundary = copy.deepcopy(boundary)
    pair = copy.deepcopy(pair)
    if plant == "passivity":
        boundary["production_repeat_passive"] = False
    elif plant == "boundary-bit":
        boundary["movement_against_production"]["final_only"]["uu_b"][
            "differing_cells"] = 0
        boundary["movement_against_production"]["final_only"]["vv_b"][
            "differing_cells"] = 0
    elif plant == "salinity":
        for row in pair["rows"]:
            if (row["kt"], row["checkpoint"], row["field"]) == (10, "stage3", "S"):
                row["max_abs"] = 1.0
                break

    require(boundary.get("status") == "PASS_R162_FINAL_ASSOCIATION_BOUNDARY",
            "boundary measurement is not admitted")
    require(boundary.get("claim_label") == "independent",
            "boundary measurement is not independent")
    require(boundary.get("production_repeat_passive") is True,
            "false-default production repeat moved")
    moved = sum(
        int(boundary["movement_against_production"]["final_only"][field][
            "differing_cells"])
        for field in ("uu_b", "vv_b"))
    require(moved > 0, "final-mean association arm is vacuous")
    substep_unequal = _unequal_total(boundary["rows_against_nemo"]["substep_only"])
    pair_unequal = _unequal_total(boundary["rows_against_nemo"]["pair"])
    pair_finite = all(
        boundary["rows_against_nemo"]["pair"][field]["candidate_finite"]
        for field in ("uu_b", "vv_b"))
    comparison = compare_gate.compare(production, pair)
    veto = compare_gate.require_salinity_veto(production, pair)
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round162-classification-v1",
        "status": "PASS_R162_FINAL_ASSOCIATION_CLASSIFICATION",
        "claim_label": "independent",
        "boundary": boundary,
        "ladder_comparison": comparison,
        "salinity_veto": veto,
        "prediction_ledger": {
            "R162-P1": {"status": "CONFIRMED", "moved_cells": moved},
            "R162-P2": {
                "status": ("CONFIRMED" if pair_finite and pair_unequal < substep_unequal
                           else "REFUTED"),
                "substep_only_unequal": substep_unequal,
                "pair_unequal": pair_unequal,
            },
            "R162-P3": {"status": "CONFIRMED"},
            "R162-P4": {"status": "UNMEASURED_PENDING_MONTH"},
            "R162-P5": {"status": "UNMEASURED_PENDING_GYRE"},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--boundary-json", type=Path)
    parser.add_argument("--production-ladder", type=Path)
    parser.add_argument("--substep-ladder", type=Path)
    parser.add_argument("--pair-ladder", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        runtime = all((args.deck_root, args.record_root, args.admission,
                       args.expect_commit))
        classification = all((args.boundary_json, args.production_ladder,
                              args.substep_ladder, args.pair_ladder))
        require(runtime != classification,
                "select exactly one of runtime or classification mode")
        if runtime:
            require(args.plant == "none", "runtime mode does not accept plants")
            result = measure_boundary(
                args.deck_root, args.record_root, args.admission,
                args.expect_commit)
        else:
            documents = [json.loads(path.read_text()) for path in (
                args.boundary_json, args.production_ladder,
                args.substep_ladder, args.pair_ladder)]
            result = classify(*documents, plant=args.plant)
    except (GateError, compare_gate.GateError, growth.GateError,
            growth.shared.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
