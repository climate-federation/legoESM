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
    snow_depth: Field      # Snow water equivalent [kg/m2]
    snow_age: Field        # Time since last snowfall [s]
    runoff: jax.Array | None = None  # Surface runoff [kg/m2/s]


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
    snow_depth: jax.Array      # Snow water equivalent [kg/m2], (ncol,)
    snow_age: jax.Array        # Time since last snowfall [s], (ncol,)
    # Surface ponding depth [m], (ncol,) — a coupled surface cell (ParFlow/CliMA
    # overland store): excess precip ponds, infiltrates on later steps, and
    # overflows to runoff above ``pond_max``.  ``None`` (legacy) is zero ponding.
    surface_water: jax.Array = None
    # Sub-grid elevation-band SWE [kg/m2], (ncol, n_bands) — present iff the
    # elevation-band snow scheme is enabled (``config.elev_bands``); ``None``
    # (legacy) runs the single cell-mean snowpack.  ``snow_depth`` always carries
    # the area-weighted aggregate for diagnostics/restart compatibility.
    snow_bands: jax.Array = None
    # Per-band snow age [s], (ncol, n_bands) — companion to ``snow_bands`` (fresh
    # snow rejuvenates only its own band's albedo while perennial firn keeps aging).
    # ``None`` (legacy) uses the single cell-mean ``snow_age`` clock.
    snow_age_bands: jax.Array = None
    # Sub-grid elevation-band firn/glacier-ice reservoir [kg/m2], (ncol, n_bands) —
    # companion to ``snow_bands`` (gap 4): seasonal snow above the snow cap firnifies
    # into this store, which discharges slowly (delayed glacier outflow) and exposes
    # dark ablation ice where the seasonal snow melts off.  ``None`` (legacy / bands
    # off) has no perennial-ice reservoir.
    ice_bands: jax.Array = None
