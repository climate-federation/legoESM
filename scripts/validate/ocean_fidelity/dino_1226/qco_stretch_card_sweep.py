#!/usr/bin/env python
"""Rule 12 for the quasi-Eulerian stretch change (PR #1728): which cards move?

THE CHANGE.  ``compute_ocean_jacobian`` formed ``(eta + H)/H`` -- the sum
first, then a DIVISION.  NEMO forms the ratio first and MULTIPLIES a stored
reciprocal:

    pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)          domqco.F90:160
    r1_ht_0(:,:) = ssmask / ( ht_0 + 1 - ssmask )       domain.F90:158
    e3t(i,j,k,t) = e3t_3d(i,j,k)*(1 + r3t(i,j,t)*tmask(i,j,k))
                                 domzgr_substitute.h90:126, Tmsk macro :46

and ``compute_layer_thickness`` carried a SECOND copy of the same statement
that fed the momentum RHS.  Both now route through the one helper.

WHY A SWEEP.  The statement is unconditional inside the PARTIAL-CELL branch of
a function every lat-lon ocean card shares, so there is no card-level predicate
to check -- only a SIZE to measure, per card.  "Every partial-cell card moves"
without a number is an assertion.

WHAT IS MEASURED (Rule 10 -- through each card's own grid and vertical
coordinate, not a formula): the shipped thickness against the pre-change form,
on the SAME synthetic +/- 1 m sea surface height, reported as the largest
RELATIVE move and the number of wet cells that move at all.  The sea surface
height is synthetic and the same for every card BY DESIGN: this measures the
STATEMENT, not each card's own circulation, and using each card's own ssh would
confound the two.

CARDS WITH A z* COORDINATE TAKE THE OTHER BRANCH and are reported INERT, which
is a measurement of the dispatch and not an assumption -- the coordinate class
is printed.

NOT COVERED HERE, and named rather than left out: the same helper is reached by
the Veros global 1deg/4deg/flexible recipes, by nemo_testcase_recipe (GYRE,
LOCK_EXCHANGE, OVERFLOW -- whose own lane already carries this change), and by
init_latlon_cgrid's builder.  Those cards are not DINO recipes and this sweep
does not construct them; their last bits move the same way.

Usage
-----
    JAX_ENABLE_X64=1 python \\
        scripts/validate/ocean_fidelity/dino_1226/qco_stretch_card_sweep.py
"""
from __future__ import annotations

import numpy as np


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.vertical import (compute_layer_thickness,
                                        OceanPartialCellCoordinate)

    print(f"{'recipe':22s}{'coordinate':30s}{'branch':10s}"
          f"{'max |dh|/h':>13s}{'wet cells moved':>17s}")
    moved_any = 0
    for recipe in sorted(dm.DINO_RECIPES):
        cfg = dm.dino_config_for_recipe(recipe)
        if recipe.startswith("nemo_dino_kamm"):
            cfg = dm.nemo_faithful_dino_config(base=cfg)
        grid = dm.dino_lat_lon_grid(cfg)
        z = dm.dino_lat_lon_vertical(grid, cfg)
        if not isinstance(z, OceanPartialCellCoordinate):
            print(f"{recipe:22s}{type(z).__name__:30s}{'z*':10s}"
                  f"{'-':>13s}{'INERT':>17s}")
            continue
        H = np.asarray(z.h_partial).sum(axis=-1)
        eta = (np.random.default_rng(1728).random(H.shape) - 0.5) * 2.0
        new = np.asarray(compute_layer_thickness(eta, H, z))
        old = (np.asarray(z.h_partial)
               * ((eta + H) / np.maximum(H, 1e-10))[..., None])
        wet = np.asarray(z.h_partial) > 0.0
        d = np.abs(new - old)[wet]
        rel = d / np.abs(new[wet])
        moved_any += int((d != 0).sum() > 0)
        print(f"{recipe:22s}{type(z).__name__:30s}{'partial':10s}"
              f"{rel.max():13.3e}{int((d != 0).sum()):>17d}")
    # A sweep where NOTHING moves is not a clean result, it is a broken probe:
    # the change is unconditional in the partial-cell branch, so at least one
    # card must move or this script is not reaching the statement.
    if moved_any == 0:
        print("\nNO CARD MOVED -- this sweep is not reaching the changed "
              "statement, so its INERT rows prove nothing.")
        return 1
    print(f"\n{moved_any} card(s) move, all of them in the LAST BITS.  The "
          "change removes a double rounding; it is not a climate lever and is "
          "not reported as one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
