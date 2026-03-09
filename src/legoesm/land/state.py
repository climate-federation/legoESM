"""Land model state container."""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm.core.field import Field


class LandState(NamedTuple):
    """Slab land state.

    All fields have shape (6, n, n).
    """
    T_soil: Field          # Soil slab temperature [K]
    W_bucket: Field        # Bucket soil moisture [kg/m2]


class MultiLayerLandState(NamedTuple):
    """Multi-layer land state (Task 8).

    2D fields have shape (ncol,).
    3D fields have shape (ncol, n_layers).
    """
    T_soil: jax.Array          # Soil temperature [K], (ncol, n_layers)
    psi_soil: jax.Array        # Soil matric potential [m], (ncol, n_layers)
    theta_soil: jax.Array      # Volumetric water content [m3/m3], (ncol, n_layers)
    runoff_surface: jax.Array  # Surface runoff [kg/m2/s], (ncol,)
    runoff_subsurface: jax.Array  # Subsurface runoff [kg/m2/s], (ncol,)
