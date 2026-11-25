#!/usr/bin/env python3
"""Bracket the corrected ORCA2 external-mode association arm against production."""

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
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung103,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)


FIELDS = ("T", "S", "u", "v", "ssh", "uu_b", "vv_b")
ASSOCIATION_FIELDS = (
    "u", "v", "depth_u", "depth_v", "inverse_u", "inverse_v", "eta",
)
PLANTS = ("none", "comparison-bit", "signed-zero", "nonfinite", "field-selector")


class GateError(RuntimeError):
    """The arm/control comparison is incomplete or its controls do not bind."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def validate_association_field(name: str) -> None:
    require(name in ASSOCIATION_FIELDS, f"unknown association field {name!r}")


def state_fields(state) -> dict[str, np.ndarray]:
    fields = rung0.candidate_fields(state)
    require(state.uu_b is not None and state.vv_b is not None,
            "rung-0 state lacks carried depth-mean velocities")
    return {
        **fields,
        "uu_b": np.asarray(state.uu_b.data),
        "vv_b": np.asarray(state.vv_b.data),
    }


def exact_pair_row(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    require(candidate.shape == reference.shape,
            f"pair shape mismatch: {candidate.shape} versus {reference.shape}")
    candidate_finite = np.isfinite(candidate)
    reference_finite = np.isfinite(reference)
    require(bool(np.all(reference_finite)), "production reference is non-finite")
    nonfinite = np.argwhere(~candidate_finite)
    if nonfinite.size:
        return {
            "bit_exact": False,
            "candidate_finite": False,
            "nonfinite_cells": int(nonfinite.shape[0]),
            "first_nonfinite_index": [int(value) for value in nonfinite[0]],
            "differing_cells": None,
            "maximum_absolute": None,
            "argmax_index": None,
        }
    candidate_bits = np.ascontiguousarray(candidate).view(np.uint64)
    reference_bits = np.ascontiguousarray(reference).view(np.uint64)
    unequal = candidate_bits != reference_bits
    absolute = np.abs(candidate - reference)
    argmax_flat = int(np.argmax(absolute))
    return {
        "bit_exact": not bool(np.any(unequal)),
        "candidate_finite": True,
        "nonfinite_cells": 0,
        "first_nonfinite_index": None,
        "differing_cells": int(np.count_nonzero(unequal)),
        "maximum_absolute": float(absolute.flat[argmax_flat]),
        "argmax_index": [int(value) for value in np.unravel_index(
            argmax_flat, absolute.shape)],
    }


def compare_states(candidate, production, *, kt: int, boundary: str) -> dict[str, object]:
    candidate_fields = state_fields(candidate)
    production_fields = state_fields(production)
    rows = {
        name: exact_pair_row(candidate_fields[name], production_fields[name])
        for name in FIELDS
    }
    return {
        "kt": kt,
        "boundary": boundary,
        "all_finite": all(row["candidate_finite"] for row in rows.values()),
        "bit_exact": all(row["bit_exact"] for row in rows.values()),
        "rows": rows,
    }


def run_plant(plant: str) -> None:
    if plant == "comparison-bit":
        row = exact_pair_row(
            np.nextafter(np.array([1.0]), np.array([np.inf])),
            np.array([1.0]))
        require(row["differing_cells"] == 1 and not row["bit_exact"],
                "comparison-bit plant stayed green")
        raise GateError("comparison-bit plant fired")
    if plant == "signed-zero":
        row = exact_pair_row(np.array([0.0]), np.array([-0.0]))
        require(row["differing_cells"] == 1 and row["maximum_absolute"] == 0.0,
                "signed-zero plant stayed green")
        raise GateError("signed-zero plant fired")
    if plant == "nonfinite":
        row = exact_pair_row(np.array([np.nan]), np.array([0.0]))
        require(not row["candidate_finite"] and row["nonfinite_cells"] == 1,
                "nonfinite plant stayed green")
        raise GateError("nonfinite plant fired")
    if plant == "field-selector":
        validate_association_field("not-a-field")
        raise GateError("field-selector plant stayed green")


def _hooks(reference_depth, *, stage: int = 0, complete: bool = False,
           field: str = "", candidate: bool = False):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    require(not (complete and field), "complete and one-field arms overlap")
    require(candidate or (not complete and not field),
            "association candidate selector is inconsistent")
    if field:
        validate_association_field(field)
    return _NEMOWSRK3TestHooks(
        expose_momentum_stage=stage,
        expose_tracer_stage=stage,
        barotropic_external_mode_association=complete,
        barotropic_external_mode_association_field=field,
        barotropic_reference_face_depth_override=(
            reference_depth if candidate else None),
        barotropic_unmasked_v_transport=candidate,
        barotropic_materialize_v_transport=candidate,
    )


def _models(card, reference_depth, *, complete: bool = False, field: str = "",
            candidate: bool = False):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    def model(stage: int = 0):
        return LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            _nemo_ws_test_hooks=_hooks(
                reference_depth, stage=stage, complete=complete, field=field,
                candidate=candidate),
        )

    return (model(1), model(2), model())


def _step_models(models, state, card, freshwater, surface):
    import jax

    return tuple(jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
    )) for model in models)


def _first_changed(pair_rows: list[dict[str, object]]) -> dict[str, object] | None:
    for checkpoint in pair_rows:
        if not checkpoint["all_finite"]:
            return None
        for name in FIELDS:
            if not checkpoint["rows"][name]["bit_exact"]:
                return {
                    "kt": checkpoint["kt"],
                    "boundary": checkpoint["boundary"],
                    "field": name,
                    **checkpoint["rows"][name],
                }
    return None


def _first_nonfinite(pair_rows: list[dict[str, object]]) -> dict[str, object] | None:
    for checkpoint in pair_rows:
        for name in FIELDS:
            row = checkpoint["rows"][name]
            if not row["candidate_finite"]:
                return {
                    "kt": checkpoint["kt"],
                    "boundary": checkpoint["boundary"],
                    "field": name,
                    **row,
                }
    return None


def _run_to_boundary(card, initial_state, reference_depth, freshwater, surface,
                     *, kt: int, boundary: str, field: str = "",
                     candidate: bool = False):
    require(boundary in ("entry", "stage1", "stage2", "stage3"),
            f"unknown target boundary {boundary}")
    if boundary == "entry":
        return initial_state
    stage = int(boundary[-1])
    models = _models(
        card, reference_depth, field=field, candidate=candidate)
    final_model = models[2]
    target_model = models[stage - 1]
    state = initial_state
    for _ in range(1, kt):
        state = _step_models(
            (final_model,), state, card, freshwater, surface)[0]
    return _step_models(
        (target_model,), state, card, freshwater, surface)[0]


def _setup(deck_root: Path, record_root: Path):
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-156 gate requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    reference_depth = rung0.ladder.build_reference_depth_override(card)
    entry = rung0.assemble_frame(record_root, 1, 0)
    initial_state = rung0.bridge_entry(card, entry)
    freshwater, surface = rung103._zero_forcing(entry["ssh"].shape)
    return card, reference_depth, entry, initial_state, freshwater, surface


def measure_bracket(deck_root: Path, record_root: Path) -> dict[str, object]:
    (card, reference_depth, entry, _initial_state,
     freshwater, surface) = _setup(deck_root, record_root)
    production_state = rung0.bridge_entry(card, entry)
    complete_state = rung0.bridge_entry(card, entry)
    production_models = _models(card, reference_depth)
    complete_models = _models(
        card, reference_depth, complete=True, candidate=True)

    pair_rows: list[dict[str, object]] = []
    production_checkpoints: list[dict[str, object]] = []
    pair_rows.append(compare_states(
        complete_state, production_state, kt=1, boundary="entry"))
    terminal = None
    for kt in range(1, 11):
        if kt > 1:
            if terminal is None:
                pair_rows.append(compare_states(
                    complete_state, production_state, kt=kt, boundary="entry"))
        production_checkpoints.append(rung103._checkpoint(
            kt, "entry", rung0.candidate_fields(production_state),
            rung0.assemble_frame(record_root, kt, 0)))
        production_stages = _step_models(
            production_models, production_state, card, freshwater, surface)
        complete_stages = (None, None, None)
        if terminal is None:
            complete_stages = _step_models(
                complete_models, complete_state, card, freshwater, surface)
        for stage, production in enumerate(production_stages, start=1):
            boundary = f"stage{stage}"
            production_checkpoints.append(rung103._checkpoint(
                kt, boundary, rung0.candidate_fields(production),
                rung0.assemble_frame(record_root, kt, stage)))
            if terminal is None:
                candidate = complete_stages[stage - 1]
                row = compare_states(
                    candidate, production, kt=kt, boundary=boundary)
                pair_rows.append(row)
                if not row["all_finite"]:
                    terminal = (kt, boundary)
        production_state = production_stages[2]
        if terminal is None:
            complete_state = complete_stages[2]

    require(len(production_checkpoints) == 40,
            "production arm did not emit 40 canonical checkpoints")
    production_first = None
    for checkpoint in production_checkpoints:
        if checkpoint["first_non_bit_field"] is not None:
            production_first = {
                "kt": checkpoint["kt"],
                "checkpoint": checkpoint["checkpoint"],
                "field": checkpoint["first_non_bit_field"],
            }
            break
    require(production_first == {
        "kt": 1, "checkpoint": "stage1", "field": "T"},
        f"unchanged production first debt moved: {production_first}")

    first_changed = _first_changed(pair_rows)
    first_nonfinite = _first_nonfinite(pair_rows)
    require(first_changed is not None,
            "complete arm stayed exact until its first non-finite boundary")
    require(first_nonfinite is not None,
            "complete arm did not reproduce its registered terminal boundary")
    return {
        "status": "MEASURED_R156_ASSOCIATION_BRACKET",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "production_ladder": {
            "status": "PASS_RUNG0_TEN_STEP_LADDER",
            "checkpoint_count": len(production_checkpoints),
            "row_count": len(production_checkpoints) * len(rung103.FIELDS),
            "first_non_bit_checkpoint": production_first,
        },
        "complete_arm": {
            "first_finite_departure": first_changed,
            "first_nonfinite": first_nonfinite,
            "checkpoint_count": len(pair_rows),
            "rows": pair_rows,
        },
        "source_order": list(ASSOCIATION_FIELDS),
        "predictions": {
            "R156-P1": "CONFIRMED",
            "R156-P2": "CONFIRMED",
            "R156-P3": (
                "CONFIRMED" if first_changed["field"] in
                ("u", "v", "ssh", "uu_b", "vv_b") else "REFUTED"),
            "R156-P4": "UNMEASURED_PENDING_FIELD_SPLIT",
            "R156-P5": (
                "CONFIRMED" if (
                    first_nonfinite["kt"] == 8
                    and first_nonfinite["boundary"] == "stage1"
                    and first_nonfinite["field"] == "T") else "REFUTED"),
        },
        "choices": {"asked": [], "unasked": []},
    }


def measure_field(
    deck_root: Path,
    record_root: Path,
    bracket: dict[str, object],
    field: str,
) -> dict[str, object]:
    validate_association_field(field)
    (card, reference_depth, _entry, initial_state,
     freshwater, surface) = _setup(deck_root, record_root)
    target = bracket["complete_arm"]["first_finite_departure"]
    kt = int(target["kt"])
    boundary = str(target["boundary"])
    production = _run_to_boundary(
        card, initial_state, reference_depth, freshwater, surface,
        kt=kt, boundary=boundary)
    partial = _run_to_boundary(
        card, initial_state, reference_depth, freshwater, surface,
        kt=kt, boundary=boundary, field=field, candidate=True)
    return {
        "status": "MEASURED_R156_ONE_FIELD_ASSOCIATION",
        "claim_label": "independent",
        "field": field,
        "target": {"kt": kt, "boundary": boundary},
        "comparison": compare_states(
            partial, production, kt=kt, boundary=boundary),
    }


def aggregate(
    bracket: dict[str, object],
    field_reports: list[dict[str, object]],
) -> dict[str, object]:
    by_name = {str(report["field"]): report for report in field_reports}
    require(tuple(by_name) == ASSOCIATION_FIELDS,
            "one-field reports are missing or out of source order")
    moved = [
        field for field in ASSOCIATION_FIELDS
        if not by_name[field]["comparison"]["bit_exact"]
    ]
    require(moved, "all one-field association arms stayed exact")
    result = dict(bracket)
    result["status"] = "MEASURED_R156_ASSOCIATION_GROWTH"
    result["one_field_arms_at_first_departure"] = {
        field: by_name[field]["comparison"] for field in ASSOCIATION_FIELDS
    }
    result["one_field_arms_that_move"] = moved
    result["predictions"] = dict(result["predictions"], **{
        "R156-P4": "CONFIRMED",
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument(
        "--mode", choices=("bracket", "field", "aggregate"),
        default="bracket")
    parser.add_argument("--bracket", type=Path)
    parser.add_argument("--field", choices=ASSOCIATION_FIELDS)
    parser.add_argument("--field-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plant != "none":
            run_plant(args.plant)
        if args.mode == "bracket":
            result = measure_bracket(args.deck_root, args.record_root)
        elif args.mode == "field":
            require(args.bracket is not None and args.field is not None,
                    "field mode requires --bracket and --field")
            result = measure_field(
                args.deck_root, args.record_root,
                json.loads(args.bracket.read_text()), args.field)
        else:
            require(args.bracket is not None, "aggregate mode requires --bracket")
            require(len(args.field_report) == len(ASSOCIATION_FIELDS),
                    "aggregate mode requires seven --field-report paths")
            result = aggregate(
                json.loads(args.bracket.read_text()),
                [json.loads(path.read_text()) for path in args.field_report])
    except (GateError, rung103.GateError, rung0.GateError, OSError, ValueError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
