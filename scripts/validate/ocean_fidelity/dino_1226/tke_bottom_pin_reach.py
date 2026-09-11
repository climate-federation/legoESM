#!/usr/bin/env python
"""How far does the ``bottom_level == N`` TKE pin reach on the DINO card?

THE DEFECT, as the GYRE lane registered it (and did NOT land -- their receipt
lists "do NOT land the bottom_level == N pin correction this round | UNASKED"):
``_solve_tke_backward_euler`` clips the Dirichlet pin index with
``jnp.clip(bottom_level, 0, N-1)``.  For a column that is wet all the way to
the deepest carried interface, ``bottom_level == N`` and the clip moves the pin
one row UP, onto ``N-1``, which NEMO leaves as an ordinary coupled equation.
NEMO pins ``mbkt+1`` (``zdftke.F90``: ``en(...,mbkt+1) = MAX(zebot,rn_emin) *
ssmask``) and its matrix loop runs ``DO jk = 2, jpkm1``, so row ``jpkm1``
(legoESM's ``N-1``) is never the pin.

THERE IS NO FIX TO CHERRY-PICK.  This probe answers the question that decides
whether writing one is worth a round: on how many DINO columns is the clip
REACHED?  A clip that never binds is a comment; a clip that binds on every
column is the pin being in the wrong row everywhere.

It changes nothing.  It builds the card, prints the resolved switches that
decide whether the pin runs at all (Rule 10: instantiate and print), and
counts.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    a = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.experiments import dino as dm

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe(a.recipe))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    model = LatLonCGridOceanModel(grid, z, mc)

    vmix = mc.physics.vertical_mixing
    print(f"recipe               {a.recipe}")
    print(f"vertical_mixing      {vmix.scheme!r}")
    print(f"tke.bottom_tke_bc    {getattr(vmix.tke, 'bottom_tke_bc', None)!r}")
    print(f"z_coord class        {type(z).__name__}")

    bl = model._tke_bottom_level(z_coord=z, config=mc)
    if bl is None:
        print("\nbottom_level is None on this card -> the clipped branch is "
              "NOT taken at all (the unconditional last-row pin runs "
              "instead). The defect is DEAD here.")
        return 0
    bl = np.asarray(bl)
    # N IS THE SOLVER'S N, not the coordinate's.  The clip is
    # ``clip(bottom_level, 0, N-1)`` with ``N = e_old.shape[-1]``, and the TKE
    # state carries one fewer level than dz_ref -- so reading N off dz_ref
    # overstates the headroom by one and the census answers the wrong
    # question.  An independent review caught that; the number below did not
    # move (it is 0 either way) but the margin did, from two levels to one.
    nlev = int(np.asarray(z.dz_ref).shape[-1])
    N = nlev - 1
    print(f"coordinate levels    {nlev}  (dz_ref)")
    print(f"TKE solve levels N   {N}  (e_old.shape[-1]; the clip is "
          f"clip(bottom_level, 0, N-1) = {N - 1})")
    wet = bl > 0
    binds = (bl > N - 1) & wet            # the clip moves the index iff bl > N-1
    print(f"wet columns          {int(wet.sum())} of {bl.size}")
    print(f"max bottom_level     {int(bl.max())}")
    print(f"bottom_level == N-1  {int(((bl == N - 1) & wet).sum())} columns "
          "(on the last row the clip allows, but NOT clipped)")
    print(f"THE CLIP BINDS ON    {int(binds.sum())} columns "
          f"({100.0 * binds.sum() / max(1, wet.sum()):.2f}% of wet)")
    print("\nREACH: 0 means the defect is dead on this card and the "
          "correction would change nothing; anything else is the number of "
          "columns whose TKE bottom row is pinned one level higher than NEMO "
          "pins it.")
    print("STRUCTURAL, and stronger than this census: NEMO caps mbathy at "
          "jpkm1, and bottom_level = mbathy - 1 <= N - 1, so the clip cannot "
          "bind on ANY card whose bathymetry came from a NEMO mesh_mask -- "
          "not just this one.  The census is what makes that checkable here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
