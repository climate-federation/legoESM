#!/usr/bin/env python3
"""Exact round-20 one-step reverse-mode T-gradient proof after round 21."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe  # noqa: E402
from tests.ocean.fidelity.test_nemo_testcase_g9_orca2tadv import CASES  # noqa: E402

RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "gridladder_vmix_rounds/round20/oracle_l3_r20/twin_a")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("G9", "G9+L2"))
    args = parser.parse_args()
    if not RECORD.is_dir():
        raise SystemExit(f"missing admitted round-20 L3 record: {RECORD}")

    # Rebuild the exact round-20 witness, then select its G9 or G9+L2
    # composition.  Keeping the admitted L3 entry state and forcing makes the
    # before/after comparison literal rather than merely shape-equivalent.
    card = recipe.build_nemo_testcase_card("ORCA2L3-eiv", deck_root=RECORD)
    cfg = card.recipe.model_config
    if args.case == "G9":
        cfg = cfg._replace(gm_redi=None)
    else:
        gm = cfg.gm_redi
        gm = recipe._orca2_isoneutral_gm_redi(
            gm.kappa_Redi,
            slope_prd_evaluation=gm.slope_prd_evaluation,
            north_fold=gm.nemo_slope_north_fold,
            redi_flux_reference_thickness_evaluation=(
                gm.redi_flux_reference_thickness_evaluation),
            redi_flux_geometry_stage=gm.redi_flux_geometry_stage,
            slope_uv_source_rounding=gm.slope_uv_source_rounding,
            redi_a33_source_ordering=gm.redi_a33_source_ordering,
            redi_flux_stored_face_ratios=gm.redi_flux_stored_face_ratios,
            eddy_induced_velocity=False,
        )
        if not recipe._orca2l2_isoneutral_selected(gm):
            raise SystemExit("G9+L2 composition does not select certified L2")
        cfg = cfg._replace(gm_redi=gm)
    card = card._replace(
        recipe=card.recipe._replace(model_config=cfg))

    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5

    run = card.recipe
    fresh, surface = g5.nemo_supplied_forcing(card, RECORD, 1)
    model = LatLonCGridOceanModel(run.grid, run.z_coord, run.model_config)
    state = run.initial_state
    model.prime_step_caches(state)
    # Re-run round 20's stamped loss exactly.  That measurement established
    # the all-NaN G9 and G9+L2 baseline which triggered the group-A import.
    wet = jnp.asarray(run.z_coord.is_active, dtype=state.T.data.dtype)

    def loss(initial_t):
        seeded = state._replace(T=state.T.replace(data=initial_t))
        out = model.step(
            seeded, card.dt_s, freshwater=fresh, surface_forcing=surface)
        return jnp.sum(out.T.data * wet) + jnp.sum(out.S.data * wet)

    # The production step owns its JIT boundary.  Round 20 intentionally kept
    # this outer reverse transform eager, so retain that exact boundary too.
    value, gradient = jax.value_and_grad(loss)(state.T.data)
    gradient = np.asarray(gradient)
    nonfinite = int(np.count_nonzero(~np.isfinite(gradient)))
    wet_np = np.asarray(wet) > 0.5
    print(
        f"{args.case} T-loss finite={bool(np.isfinite(value))} "
        f"dT_nonfinite={nonfinite} of={gradient.size} "
        f"wet_nonfinite={int(np.count_nonzero(~np.isfinite(gradient) & wet_np))} "
        f"dT_nonzero={int(np.count_nonzero(gradient))}",
        flush=True,
    )
    return 0 if np.isfinite(value) and nonfinite == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
