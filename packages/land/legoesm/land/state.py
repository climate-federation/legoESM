"""Land model state container."""

from __future__ import annotations

from typing import Any, NamedTuple

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
    # Prognostic state for the CLM-ML-JAX multilayer canopy scheme.
    # Holds the ``mlcanopy_type`` instance carried forward between steps.
    # ``None`` when the CLM-ML canopy scheme is not active.
    canopy_state: Any | None = None
    # Intercepted canopy water store [kg m-2], (ncol,) — the shared canopy
    # interception scheme (``config.interception``; two-leaf / SimpleSEB path).
    # ``None`` (legacy / interception off) carries no store.  The CLM-ML canopy
    # keeps its own internal ``h2ocan`` store inside ``canopy_state`` and does
    # NOT use this field.
    W_canopy: jax.Array | None = None
    # Prognostic multi-layer snow column (``snow_column.SnowColumnState``:
    # per-layer swe_ice/swe_liq/T/density, shape (ncol, n_snow_layers)) — present
    # iff ``config.snow_scheme == "multilayer"``; ``None`` (default / "single")
    # runs the single cell-mean ``snow_depth`` budget.  The cell-mean ``snow_depth``
    # (= column total SWE) is still carried for diagnostics/restart/albedo when the
    # column is active.  See ``docs/land/phase2b_snow_thermal_plan.md``.
    snow_column: Any | None = None
