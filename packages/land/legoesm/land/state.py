"""Land model state container."""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm.core.field import Field


class LandState(NamedTuple):
    """Slab land state.

    Field shape is grid-dependent: (ncol,) on the column/SCM path,
    (6, n, n) on the cubed-sphere.
    """
    T_soil: Field          # Soil slab temperature [K]
    W_bucket: Field        # Bucket soil moisture [kg/m2]
    snow_depth: Field      # Snow water equivalent [kg/m2]
    snow_age: Field        # Time since last snowfall [s]
    runoff: jax.Array | None = None  # Surface runoff [kg/m2/s]
    # 30-day exponential moving average of near-surface air temperature
    # in [°C] — feeds the Leuning Vcmax growth-temperature acclimation
    # term when the two-leaf canopy surface scheme is active.  ``None``
    # by default; populated by ``init_*_land_state`` and advanced each
    # step.  Independent of any externally prescribed
    # ``CanopyLandParams.TgC`` (which takes precedence when both are set).
    TgC: jax.Array | None = None


class MultiLayerLandState(NamedTuple):
    """Multi-layer land state (Task 8).

    2D fields have shape (ncol,).
    3D fields have shape (ncol, n_soil_layers).
    """
    T_soil: jax.Array          # Soil temperature [K], (ncol, n_soil_layers)
    psi_soil: jax.Array        # Soil matric potential [m], (ncol, n_soil_layers)
    theta_soil: jax.Array      # Volumetric water content [m3/m3], (ncol, n_soil_layers)
    runoff_surface: jax.Array  # Surface runoff [kg/m2/s], (ncol,)
    runoff_subsurface: jax.Array  # Subsurface runoff [kg/m2/s], (ncol,)
    snow_depth: jax.Array      # Snow water equivalent [kg/m2], (ncol,)
    snow_age: jax.Array        # Time since last snowfall [s], (ncol,)
    # 30-day exponential moving average of near-surface air temperature
    # in [°C] — see ``LandState.TgC``.  Optional; ``None`` by default.
    TgC: jax.Array | None = None
    # Surface ponding depth [m], (ncol,) — a coupled surface cell (ParFlow/CliMA
    # overland store): excess precip ponds, infiltrates on later steps, and
    # overflows to runoff above ``pond_max``.  ``None`` (legacy) is zero ponding.
    surface_water: jax.Array = None
