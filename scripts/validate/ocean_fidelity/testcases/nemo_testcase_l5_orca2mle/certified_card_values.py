#!/usr/bin/env python3
"""Pin one certified card's two-step ocean-state bits at 353b24265796."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe  # noqa: E402


PHASE3 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
CASES = Path(__file__).resolve().parents[1]
ROOTS = {
    "ORCA2G5-ice": PHASE3 / "gridladder_rounds/round31/oracle_g5_r31/twin_a",
    "ORCA2G6-vec": PHASE3 / "gridladder_rounds/round36/oracle_g6_r36/twin_a",
    "ORCA2G7-drg": PHASE3 / "gridladder_rounds/round41/oracle_g7_r41/twin_a",
    "ORCA2G8-ldf": PHASE3 / "gridladder_rounds/round45/oracle_g8_r45/twin_a",
    "ORCA2G9-tadv": PHASE3 / "gridladder_rounds/round46/oracle_g9_r46/twin_a",
    "ORCA2V1-tke": PHASE3 / "gridladder_vmix_rounds/round1/oracle_v1_r1/twin_a",
    "ORCA2V2-evd": PHASE3 / "gridladder_vmix_rounds/round12/oracle_v2_r12/twin_a",
    "ORCA2V3-ddm": PHASE3 / "gridladder_vmix_rounds/round13/oracle_v3_r13/twin_a",
    "ORCA2V4-iwm": PHASE3 / "gridladder_vmix_rounds/round14/oracle_v4_r14/twin_a",
    "ORCA2V5-havtb": PHASE3 / "gridladder_vmix_rounds/round15/oracle_v5_r15/twin_a",
    "ORCA2L1-bbl": PHASE3 / "gridladder_vmix_rounds/round16/oracle_l1_r16/twin_a",
    "ORCA2L2-iso": PHASE3 / "gridladder_vmix_rounds/round17/oracle_l2_r17/twin_a",
    "ORCA2L3-eiv": PHASE3 / "gridladder_vmix_rounds/round20/oracle_l3_r20/twin_a",
    "ORCA2L5-mle": PHASE3 / "gridladder_vmix_rounds/round21/oracle_l5_r21/twin_a",
}
EXPECTED = {
    "ORCA2G5-ice": "e7ba063de90e8a954bc49a36f23a4f10461bcecbf12ac5582994d4e7f26b84fa",
    "ORCA2G6-vec": "d8e16d346276926c39b347b4aaae93c925498bfb4158e8c9568754afee836251",
    "ORCA2G7-drg": "5e563fc7c293b926a8723b56b41b561b334412a0f488174784abbddfb4158d8f",
    "ORCA2G8-ldf": "e108914a68ef020b65e4f7eb577fd768ab3c744d51ccf8da0df2908eabf51f9a",
    "ORCA2G9-tadv": "d5e168b95f9023c249242e83fe32a4f660649c92dd26ff0b988a193207649ae4",
    "ORCA2V1-tke": "1e4ea66c654336e895905ab6f1619df4fc09617fe29995f8dcdc90cabe03b08e",
    "ORCA2V2-evd": "c50ef938fb975ae101e2d4ae50ed1556a327d736bed92208b1438cfd7bf8cddf",
    "ORCA2V3-ddm": "cba33e2caf64e621c13c14b2dce6bfecbebca275fe61fb088fbdf816bdc5870e",
    "ORCA2V4-iwm": "d536a523b9fe645f89084629de99fbd8098b3235877a55235453fc117f408831",
    "ORCA2V5-havtb": "5806db06ebbe0d47546256a2992fb9557b92cf5d4a7c8fa604ca2e852701b2e8",
    "ORCA2L1-bbl": "271d20a2d9a04a1b1707178383e7f4527ac2c37c9eaba45c658412be4e73d156",
    "ORCA2L2-iso": "c05c3052d540c1beacbfc610b47f863f40332c4be791184ab34795ba280c96f6",
    "ORCA2L3-eiv": "9c83d48f76cfbae1905d82d2c5466ff87f84026fc8e8c8d9fbd0060f05745fea",
    "ORCA2L5-mle": "109b7a9e38d7e824778e47f803ce67678d304378a274bc30c1a246e2c430bb90",
}


def two_step_digest(case: str) -> str:
    """Run one card on its admitted forcing and hash T/S/u/v/SSH after kt 2."""
    root = ROOTS[case]
    if not root.is_dir():
        raise FileNotFoundError(f"missing immutable certified record {root}")
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5

    card = recipe.build_nemo_testcase_card(case, deck_root=root)
    run = card.recipe
    model = LatLonCGridOceanModel(
        run.grid,
        run.z_coord,
        run.model_config,
        iwm_forcing=run.iwm_forcing,
    )
    state = run.initial_state
    for kt in (1, 2):
        fresh, surface = g5.nemo_supplied_forcing(card, root, kt)
        if case.startswith("ORCA2V"):
            surface = surface._replace(
                ice_mean_thickness_hm_i=g5.read_v1_hm_i(root, kt))
        state = model.step(
            state,
            card.dt_s,
            freshwater=fresh,
            surface_forcing=surface,
        )
        jax.block_until_ready(state.T.data)
    digest = hashlib.sha256()
    for name in ("T", "S", "u", "v", "eta"):
        digest.update(
            np.ascontiguousarray(np.asarray(getattr(state, name).data)).tobytes()
        )
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(EXPECTED))
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        raise SystemExit("JAX_ENABLE_X64=1 is required")
    actual = two_step_digest(args.case)
    print(f"{args.case} {actual}", flush=True)
    return 0 if actual == EXPECTED[args.case] else 1


if __name__ == "__main__":
    raise SystemExit(main())
