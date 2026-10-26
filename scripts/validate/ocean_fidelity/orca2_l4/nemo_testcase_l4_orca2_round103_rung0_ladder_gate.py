#!/usr/bin/env python3
"""Score the independent ORCA2 hierarchy rung-0 kt=1..10 ladder."""

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
    nemo_testcase_l4_orca2_round84_rung0_frame_gate as frames,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)


FIELDS = ("T", "S", "u", "v", "ssh")
BOUNDARIES = ("entry", "stage1", "stage2", "stage3")
EXPECTED_PRODUCER = "a62f67376bae4445623149fc44b570c9e2a2012c"
FIRST_SOURCE_STATEMENT = {
    "statement": "vector-invariant vertical average of the completed 3-D RHS",
    "nemo_source": "ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219",
    "legoesm_discriminator": (
        "round 94 source-associated materialized multiply/add/final-multiply arm"
    ),
    "status": "HELD_BY_GYRE_2ULP_GATE",
}
PLANTS = ("none", "entry-bit")


class GateError(RuntimeError):
    """The rung-0 ten-step ladder is incomplete or invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _zero_forcing(shape: tuple[int, int]):
    import jax.numpy as jnp

    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    zero = jnp.zeros(shape, dtype=jnp.float64)
    freshwater = FreshwaterForcing(zero, zero, zero, zero, zero)
    surface = OceanSurfaceForcing(
        sw_down=zero,
        q_net=zero,
        tau_x=zero,
        tau_y=zero,
        freshwater=zero,
        salt_flux=zero,
        taum=zero,
        tau_i_native=zero,
        tau_j_native=zero,
    )
    return freshwater, surface


def _checkpoint(
    kt: int,
    boundary: str,
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> dict[str, object]:
    require(boundary in BOUNDARIES, f"unknown boundary {boundary}")
    comparison = rung0.ladder.compare_fields(actual, expected)
    for name in FIELDS:
        actual_field = np.asarray(actual[name])
        expected_field = np.asarray(expected[name])
        require(bool(np.all(np.isfinite(actual_field))),
                f"kt={kt} {boundary} {name}: candidate is non-finite")
        require(bool(np.all(np.isfinite(expected_field))),
                f"kt={kt} {boundary} {name}: oracle is non-finite")
        comparison["rows"][name]["status"] = (
            "AT_BAR_BIT_EXACT"
            if comparison["rows"][name]["bit_identical"] else "DEBT"
        )
    return {"kt": kt, "checkpoint": boundary, **comparison}


def run(
    deck_root: Path,
    record_root: Path,
    *,
    plant: str = "none",
) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    require(plant in PLANTS, f"unknown plant {plant}")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "rung-0 ladder requires production JIT on CPU")

    admission = frames.admit(record_root, EXPECTED_PRODUCER, None)
    print("PROGRESS admitted 80 rung-0 frame shards", file=sys.stderr, flush=True)
    require(admission["record_count"] == 80,
            "rung-0 record did not admit exactly 80 shards")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    require(card.unmeasured_features == ("linear_implicit_bottom_drag",),
            "rung-0 exact-input debt changed")

    entry = rung0.assemble_frame(record_root, 1, 0)
    state = rung0.bridge_entry(card, entry)
    entry_expected = {name: np.array(value, copy=True)
                      for name, value in entry.items()}
    if plant == "entry-bit":
        entry_expected["T"].flat[0] = np.nextafter(
            entry_expected["T"].flat[0], np.float64(np.inf))
    entry_row = _checkpoint(
        1, "entry", rung0.candidate_fields(state), entry_expected)
    require(entry_row["first_non_bit_field"] is None,
            "kt=1 entry bridge is not bit-exact")

    freshwater, surface = _zero_forcing(entry["ssh"].shape)
    stage_models = tuple(LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_momentum_stage=stage,
            expose_tracer_stage=stage),
    ) for stage in (1, 2))
    final_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    checkpoints: list[dict[str, object]] = []
    first_non_bit: dict[str, object] | None = None
    for kt in range(1, 11):
        oracle_entry = rung0.assemble_frame(record_root, kt, 0)
        entry_score = _checkpoint(
            kt, "entry", rung0.candidate_fields(state), oracle_entry)
        checkpoints.append(entry_score)
        stage_states = tuple(jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        )) for model in stage_models)
        print(f"PROGRESS kt={kt} exposed stages 1-2", file=sys.stderr, flush=True)
        state_after = jax.device_get(final_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        print(f"PROGRESS kt={kt} completed stage 3", file=sys.stderr, flush=True)
        stage_states = (*stage_states, state_after)
        for stage, stage_state in enumerate(stage_states, start=1):
            boundary = f"stage{stage}"
            row = _checkpoint(
                kt,
                boundary,
                rung0.candidate_fields(stage_state),
                rung0.assemble_frame(record_root, kt, stage),
            )
            checkpoints.append(row)
            if first_non_bit is None and row["first_non_bit_field"] is not None:
                first_non_bit = {
                    "kt": kt,
                    "checkpoint": boundary,
                    "field": row["first_non_bit_field"],
                }
        state = state_after

    require(len(checkpoints) == 40, "rung-0 ladder did not emit 40 checkpoints")
    require(first_non_bit is not None,
            "rung-0 ladder unexpectedly stayed bit-exact")
    require(first_non_bit == {"kt": 1, "checkpoint": "stage1", "field": "T"},
            f"first non-bit checkpoint moved: {first_non_bit}")
    rows = [
        {"kt": checkpoint["kt"], "checkpoint": checkpoint["checkpoint"],
         "field": field, **checkpoint["rows"][field]}
        for checkpoint in checkpoints for field in FIELDS
    ]
    require(len(rows) == 200, "rung-0 ladder did not emit 200 field rows")
    return {
        "status": "PASS_RUNG0_TEN_STEP_LADDER",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_admission": {
            "status": admission["status"],
            "record_count": admission["record_count"],
            "producer_commit": admission["producer_commit"],
        },
        "card_case": card.case,
        "unmeasured_features": list(card.unmeasured_features),
        "entry_identity": entry_row,
        "checkpoint_count": len(checkpoints),
        "row_count": len(rows),
        "first_non_bit_checkpoint": first_non_bit,
        "first_non_bit_source_statement": FIRST_SOURCE_STATEMENT,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, frames.GateError, OSError, ValueError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_RUNG0_TEN_STEP_LADDER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
