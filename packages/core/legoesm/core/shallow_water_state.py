"""Shared shallow-water dynamical-core state pytrees.

The FV3 C-D-grid / edge-D-grid shallow-water *state* NamedTuples used by the
atmosphere shallow-water dycore AND by the ocean barotropic solver (the
barotropic ocean is itself a shallow-water problem and reuses the same FV3
shallow-water core).  They live in ``core`` (the shared substrate, importing
nothing above it) so the ocean can build a shallow-water state without importing
the atmosphere component; ``atmosphere.dynamics.shallow_water_fv3_cdgrid``
re-imports them (it is the dycore that evolves these states).

Pure pytrees: four ``jax.Array`` fields each.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class CDGridShallowWaterState(NamedTuple):
    """Shallow water state on the FV3 C-D grid.

    h : (6, n, n) -- height at cell centres
    u_d : (6, n+1, n+1) -- x-velocity at cell corners (D-grid)
    v_d : (6, n+1, n+1) -- y-velocity at cell corners (D-grid)
    h_s : (6, n, n) -- surface topography at cell centres
    """
    h: jax.Array
    u_d: jax.Array
    v_d: jax.Array
    h_s: jax.Array


class FV3EdgeShallowWaterState(NamedTuple):
    """Shallow water state with FV3 edge-midpoint D-grid stagger.

    h   : (6, n, n)   -- height at cell centres
    u_d : (6, n, n+1) -- x-velocity at x-edge midpoints
    v_d : (6, n+1, n) -- y-velocity at y-edge midpoints
    h_s : (6, n, n)   -- surface topography at cell centres
    """
    h: jax.Array
    u_d: jax.Array
    v_d: jax.Array
    h_s: jax.Array
