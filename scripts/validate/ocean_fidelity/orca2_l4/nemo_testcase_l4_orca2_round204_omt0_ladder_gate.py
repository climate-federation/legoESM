#!/usr/bin/env python3
"""Build and score the Decision-103 OMT-0 card against its admitted frames."""

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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round203_omt0_frame_record_gate as record_gate,
)

FIELDS = ("T", "S", "u", "v", "ssh")
PLANTS = ("none", "card-module", "entry-bit")


class GateError(RuntimeError):
    """OMT-0 is not the admitted five-module-off identity."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def build_omt0_card(deck_root: Path, *, plant: str = "none"):
    """Apply exactly Decision 103's five OFF module selections to rung 0."""

    card = rung0.build_rung0_card(deck_root)
    cfg = card.recipe.model_config
    viscosity = cfg.lateral_viscosity._replace(A_h=0.0)
    if plant == "card-module":
        viscosity = viscosity._replace(A_h=cfg.lateral_viscosity.A_h)
    cfg = cfg._replace(
        # ln_dynadv_OFF -> np_LIN_dyn (compiled dynadv.f90:185); NEMO still
        # calls wzv for the tracer transports (stprk3_stg.f90:332-339).
        momentum_advection="flux_form",
        momentum_flux_scheme="none",
        vertical_momentum_scheme="none",
        adaptive_implicit_vertadv=False,
        vorticity_scheme="een_planetary",
        # ln_traadv_OFF -> np_NO_adv (compiled traadv.f90:444).
        tracer_advection="none",
        # ln_dynldf_OFF.  All remaining viscosity coefficients were already
        # exact zero on the rung-0 carrier.
        lateral_viscosity=viscosity,
        # ln_traldf_OFF: the carrier already has K_h=K_bih=0 and gm_redi=None.
        K_h=0.0,
        K_bih=0.0,
        gm_redi=None,
        # ln_drg_OFF.  A zero legacy rate is the exact no-op dispatcher arm.
        bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_r=0.0, bottom_drag_bbl_thickness=0.0),
        zdf_drag_in_matrix=False,
        barotropic_drag_substep=False,
    )
    recipe = card.recipe._replace(model_config=cfg)
    return card._replace(
        case="ORCA2-OMT0-zps",
        recipe=recipe,
        unmeasured_features=(),
    )


def validate_omt0_card(card) -> dict[str, object]:
    cfg = card.recipe.model_config
    lv = cfg.lateral_viscosity
    selectors = {
        "case": card.case == "ORCA2-OMT0-zps",
        "dynadv_off": (
            cfg.momentum_advection == "flux_form"
            and cfg.momentum_flux_scheme == "none"
            and cfg.vertical_momentum_scheme == "none"
            and not cfg.adaptive_implicit_vertadv
            and cfg.vorticity_scheme == "een_planetary"
        ),
        "traadv_off": cfg.tracer_advection == "none",
        "dynldf_off": all(getattr(lv, name) == 0.0 for name in (
            "A_h", "B_h", "C_smag", "C_smag_lap", "C_leith")),
        "traldf_off": cfg.K_h == 0.0 and cfg.K_bih == 0.0 and cfg.gm_redi is None,
        "drag_off": (
            cfg.bottom_drag.bottom_drag_r == 0.0
            and cfg.bottom_drag.bottom_drag_bbl_thickness == 0.0
            and not cfg.zdf_drag_in_matrix
            and not cfg.barotropic_drag_substep
        ),
        "no_unmeasured": tuple(card.unmeasured_features) == (),
    }
    require(all(selectors.values()), f"OMT-0 selector moved: {selectors}")
    return selectors


def _checkpoint(kt: int, boundary: str, actual, expected) -> dict[str, object]:
    result = rung0.ladder.compare_fields(actual, expected)
    for name in FIELDS:
        require(bool(np.all(np.isfinite(actual[name]))),
                f"kt={kt} {boundary} {name}: candidate is non-finite")
        require(bool(np.all(np.isfinite(expected[name]))),
                f"kt={kt} {boundary} {name}: oracle is non-finite")
    return {"kt": kt, "checkpoint": boundary, **result}


def _run_ladder(
    card,
    record_root: Path,
    state,
    label: str,
    *,
    atomic_fold_unit: bool = False,
) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    freshwater, surface = rung0_ladder._zero_forcing((148, 180))
    reference_depth = (
        rung0.ladder.build_reference_depth_override(card)
        if atomic_fold_unit else None
    )

    def hooks(stage: int = 0):
        return _NEMOWSRK3TestHooks(
            expose_momentum_stage=stage,
            expose_tracer_stage=stage,
            barotropic_external_mode_association=atomic_fold_unit,
            barotropic_reference_face_depth_override=reference_depth,
            barotropic_unmasked_v_transport=atomic_fold_unit,
            barotropic_materialize_v_transport=atomic_fold_unit,
            barotropic_atomic_fold_unit=atomic_fold_unit,
        )

    stage_models = tuple(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks(stage),
    ) for stage in (1, 2))
    final_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks())

    checkpoints: list[dict[str, object]] = []
    first_non_bit = None
    for kt in range(1, 11):
        entry = rung0.assemble_frame(record_root, kt, 0)
        checkpoints.append(_checkpoint(
            kt, "entry", rung0.candidate_fields(state), entry))
        stage_states = tuple(jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        )) for model in stage_models)
        state_after = jax.device_get(final_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        for stage, stage_state in enumerate((*stage_states, state_after), start=1):
            boundary = f"stage{stage}"
            row = _checkpoint(
                kt, boundary, rung0.candidate_fields(stage_state),
                rung0.assemble_frame(record_root, kt, stage),
            )
            checkpoints.append(row)
            if first_non_bit is None and row["first_non_bit_field"] is not None:
                first_non_bit = {
                    "kt": kt, "checkpoint": boundary,
                    "field": row["first_non_bit_field"],
                }
        print(f"PROGRESS {label} kt={kt}", file=sys.stderr, flush=True)
        state = state_after

    require(len(checkpoints) == 40, f"{label}: incomplete checkpoint ladder")
    rows = [
        {"kt": cp["kt"], "checkpoint": cp["checkpoint"], "field": field,
         **cp["rows"][field]}
        for cp in checkpoints for field in FIELDS
    ]
    require(len(rows) == 200, f"{label}: incomplete row ladder")
    return {
        "label": label,
        "private_arm": {
            "external_mode_association": atomic_fold_unit,
            "raw_reference_depth": atomic_fold_unit,
            "unmasked_v_transport": atomic_fold_unit,
            "materialize_v_transport": atomic_fold_unit,
        },
        "checkpoint_count": len(checkpoints),
        "row_count": len(rows),
        "first_non_bit_checkpoint": first_non_bit,
        "rows": rows,
    }


def run(deck_root: Path, canonical: Path, calibration: Path, twin_a: Path,
        twin_b: Path, month: Path, *, plant: str = "none",
        atomic_fold_unit: bool = False) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(plant in PLANTS, f"unknown plant {plant}")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "OMT-0 ladder requires production JIT on CPU")

    admission = record_gate.admit(
        canonical, calibration, twin_a, twin_b, month, "none")
    card = build_omt0_card(deck_root, plant=plant)
    selectors = validate_omt0_card(card)
    entry = rung0.assemble_frame(twin_a, 1, 0)
    independent_state = card.recipe.initial_state
    entry_actual = rung0.candidate_fields(independent_state)
    if plant == "entry-bit":
        entry = {name: np.array(value, copy=True) for name, value in entry.items()}
        entry["T"].flat[0] = np.nextafter(entry["T"].flat[0], np.inf)
    entry_identity = rung0.ladder.compare_card_entry_to_masked_record(
        entry_actual, entry)
    require(entry_identity["first_non_bit_field"] is None,
            "OMT-0 independent card entry is not bit-exact after the admitted "
            "dry-temperature signed-zero classification")
    given_state = rung0.bridge_entry(card, rung0.assemble_frame(twin_a, 1, 0))

    independent = _run_ladder(
        card, twin_a, independent_state, "independent",
        atomic_fold_unit=atomic_fold_unit)
    given = _run_ladder(
        card, twin_a, given_state, "given_nemo_entry",
        atomic_fold_unit=atomic_fold_unit)
    return {
        "status": "PASS_R204_OMT0_CARD_AND_LADDERS",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_status": admission["status"],
        "card_case": card.case,
        "selectors": selectors,
        "entry_identity": entry_identity,
        "independent": independent,
        "given_nemo_entry": given,
        "month_boundary": admission["month_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--twin-a", type=Path, required=True)
    parser.add_argument("--twin-b", type=Path, required=True)
    parser.add_argument("--month", type=Path, required=True)
    parser.add_argument("--atomic-fold-unit", action="store_true")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(
            args.deck_root, args.candidate, args.calibration, args.twin_a,
            args.twin_b, args.month, plant=args.plant,
            atomic_fold_unit=args.atomic_fold_unit,
        )
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, record_gate.GateError,
            OSError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
