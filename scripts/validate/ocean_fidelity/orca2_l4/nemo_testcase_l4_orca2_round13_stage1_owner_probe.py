#!/usr/bin/env python3
"""ORCA2 round-13 probe: what owns the kt=1 stage-1 disagreement.

Magnitude-first (Decision 43), the kt=1 stage-1 rows rank sea surface, zonal
velocity, meridional velocity, then the two tracers.  Three downstream
statements already have bit-exact gates against this record and all three are
AT BAR (the stage-1 transport, the divergence/WZV recurrence including the
runoff, and the hydrostatic pressure gradient), so the owner is upstream of
them.  This probe asks TWO questions that a ratio and a variance can answer
without a new NEMO run.

1.  The sea surface.  NEMO's stage sea surfaces are the step's end-of-step
    value interpolated at 1/3, 1/2 and 1 (``stprk3_stg.f90`` HYB), and so are
    legoESM's, so if the stage-1, stage-2 and stage-3 sea-surface
    disagreements stand in the ratio 1 : 1.5 : 3 they carry NO stage-specific
    content: the whole disagreement is inherited from the end-of-step sea
    surface.

    **WEAK BY CONSTRUCTION, AND LABELLED SO.**  Once the step-entry sea
    surface agrees bitwise -- it does at kt=1 -- and both sides interpolate
    the same weights, ``stage_i_candidate - stage_i_oracle`` equals
    ``w_i * (end_candidate - end_oracle)`` ALGEBRAICALLY, whatever caused the
    end-of-step difference.  So a passing ratio says only that legoESM adds no
    per-stage sea-surface source of its own and that the two interpolations
    agree; it says NOTHING about what produced the end-of-step error.  It is
    reported for what it rules out, not as an attribution.

2.  The velocities.  Each stage velocity carries the barotropic correction
    ``un_adv/hu(Kmm) - uu_b(Kmm)`` (``stprk3_stg.f90:270-277``), which is a
    single two-dimensional field added to EVERY level of a column.  If the
    stage-1 velocity disagreement is depth-uniform on the columns where it is
    non-zero, it is that correction -- the same barotropic owner -- and not a
    three-dimensional momentum operator.

The second question is a prediction one named mechanism makes and another does
not.  The first is weaker than it looks and is labelled so where it is
computed: it follows algebraically from the shared interpolation once the
step-entry sea surface agrees, so it rules a stage-specific source OUT rather
than ruling any cause in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
CITATIONS = {
    "stage_sea_surface_interpolation": f"{_PP}/stprk3_stg.f90:137,152-154",
    "stage_velocity_barotropic_correction": f"{_PP}/stprk3_stg.f90:270-277",
    "external_mode_sea_surface": f"{_PP}/stp2d.f90:278-281",
}


class ProbeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def run(deck_root: Path, root: Path, json_out: Path | None, kt: int = 1):
    import jax.numpy as jnp

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )

    _, card = ladder.card_fields(deck_root)
    entry = ladder.assemble_state_fields(root, kt, stage=None)
    state = card.recipe.initial_state
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"], dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(entry["S"], dtype=jnp.float64)),
        eta=state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)),
    )
    surface_fields = ladder.assemble_surface_fields(root, kt)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, kt)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    )
    trace = model.step(state, dt=card.dt_s,
                       freshwater=freshwater, surface_forcing=surface)

    rows: dict[str, object] = {}

    # --- 1. is the sea-surface row ONE quantity? --------------------------
    ssh_max = []
    for stage in (1, 2, 3):
        oracle = ladder.read_state_frame(
            root / f"oracle_stage_kt{kt:08d}_s{stage}.bin",
            kt=kt, stage=stage)
        candidate = ladder._stage_candidate_fields(
            trace.stage_outputs[stage - 1])
        ny, nx = oracle["ssh"].shape
        delta = np.abs(np.asarray(candidate["ssh"])[:ny, :nx] - oracle["ssh"])
        ssh_max.append(float(delta.max()))
    base = ssh_max[0]
    require(base > 0.0, "the stage-1 sea surface agrees; this probe has "
                        "nothing to attribute")
    ratios = [value / base for value in ssh_max]
    predicted = [1.0, 1.5, 3.0]
    worst = max(abs(r - p) / p for r, p in zip(ratios, predicted))
    rows["sea_surface_is_one_end_of_step_quantity"] = {
        "stage_max_abs_m": ssh_max,
        "measured_ratios": ratios,
        "predicted_by_the_HYB_interpolation": predicted,
        "worst_relative_departure": worst,
        "verdict": ("NO_PER_STAGE_SEA_SURFACE_SOURCE" if worst < 1.0e-06
                    else "A_PER_STAGE_SEA_SURFACE_SOURCE_EXISTS"),
        "implied_end_of_step_disagreement_m": ssh_max[2],
        "note": ("WEAK BY CONSTRUCTION: with the step-entry sea surface "
                 "bitwise equal and both sides interpolating the same "
                 "weights, this ratio follows algebraically for ANY "
                 "end-of-step difference.  It rules out a per-stage "
                 "sea-surface source in legoESM; it attributes nothing"),
    }

    # --- 2. is the velocity row the barotropic correction? ----------------
    oracle1 = ladder.read_state_frame(
        root / f"oracle_stage_kt{kt:08d}_s1.bin", kt=kt, stage=1)
    candidate1 = ladder._stage_candidate_fields(trace.stage_outputs[0])
    for name in ("u", "v"):
        ny, nx, nz = oracle1[name].shape
        delta = np.asarray(candidate1[name])[:ny, :nx, :nz] - oracle1[name]
        wet = np.asarray(
            card.recipe.z_coord.is_active, dtype=np.float64)[:ny, :nx, :nz] > 0
        # A depth-uniform offset is the barotropic correction's signature.
        # Compare each column's SPREAD about its own wet mean with the size of
        # that mean: a three-dimensional operator error has no reason to be
        # constant down a column.
        counts = wet.sum(axis=-1)
        column = counts > 1
        totals = np.where(wet, delta, 0.0).sum(axis=-1)
        means = np.zeros_like(totals)
        np.divide(totals, counts, out=means, where=counts > 0)
        spread = np.where(wet, np.abs(delta - means[..., None]), 0.0).max(-1)
        live = column & (np.abs(means) > 0.0)
        require(int(live.sum()) > 0, f"no scoreable {name} column")
        share = spread[live] / np.abs(means[live])
        rows[f"{name}_stage1_is_a_depth_uniform_offset"] = {
            "columns_scored": int(live.sum()),
            "median_spread_over_mean": float(np.median(share)),
            "mean_spread_over_mean": float(share.mean()),
            "columns_where_spread_is_under_1_percent_of_the_mean": int(
                (share < 0.01).sum()),
            "columns_where_spread_exceeds_the_mean": int((share > 1.0).sum()),
            "max_abs_column_mean_m_s": float(np.abs(means[live]).max()),
            "note": ("the stage velocity carries the barotropic correction "
                     "un_adv/hu - uu_b, one 2-D field added to every level; "
                     "a small spread/mean says the disagreement IS that field"),
        }

    result = {
        "probe": "nemo_testcase_l4_orca2_round13_stage1_owner_probe",
        "kt": kt,
        "label": f"given NEMO's entry (kt={kt} recorded state and frames)",
        "citations": CITATIONS,
        "already_at_bar_against_this_record": [
            "nemo_testcase_l4_orca2_stage1_transport_gate (U, V)",
            "nemo_testcase_l4_orca2_wzv_gate (divergence, runoff, wzv, pfw)",
            "nemo_testcase_l4_orca2_hpg_gate (zhpi, zuap, sum)",
        ],
        "provenance": worktree_stamp(),
        "rows": rows,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--kt", type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, args.json_out, args.kt)
    except ProbeError as exc:
        print(f"FAIL: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
