"""Idealized atmospheric forcings for canonical dycore validation tests.

This subpackage hosts the lightweight, paper-cited forcing packages that
power Hughes (2026) §7 climate-timescale benchmarks and the moist
DCMIP-2016 cases (M3 milestone). Currently delivered (M1.a):

- :mod:`held_suarez_topo` — Held-Suarez (1994) Newtonian relaxation +
  Rayleigh friction over an idealized Gaussian/cosine-bell mountain.

Scheduled for M1.b:

- :mod:`small_planet` — small-Earth grid scaling (Wedi & Smolarkiewicz 2009).
- :mod:`frierson_gray` — Frierson et al. (2006) gray-radiation aquaplanet
  (dry version in M1.b; moist coupling in M3).

Scheduled for M3:

- :mod:`reed_jablonowski` — Reed & Jablonowski (2012) simple physics.
- :mod:`moist_held_suarez` — Thatcher & Jablonowski (2016) moist HS.
- :mod:`klemp_supercell` — Klemp (2015) supercell forcing.

The full Held-Suarez machinery without topography stays in
:mod:`legoesm.atmosphere.forcing.idealized.held_suarez` to preserve backwards compatibility
with the existing matrix-script wiring.
"""

from legoesm.atmosphere.idealized import (  # noqa: F401
    held_suarez_topo,
    small_planet,
    topography,
)
