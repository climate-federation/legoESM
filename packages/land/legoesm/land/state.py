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
    # Warm-start cache for the two-leaf canopy Newton closure: the last
    # CONVERGED solution per column, ``(ncol, 6)``, NaN where a column has never
    # converged (which the canopy reads as "cold start").  NOT a prognostic
    # variable — it carries no physics, only the seed the iteration starts from
    # (the solve's adjoint returns a zero cotangent for it).  A seed outside the
    # physical box is discarded for a cold start (``canopy_state_admissible``),
    # since a far-off seed can reach a spurious root.  Present iff the surface
    # scheme is the two-leaf canopy; ``None`` otherwise, which restores the
    # cold-start-every-step behaviour exactly.  Appended last (positional-ABI).
    canopy_x: Any | None = None
    # Layered snowpack (``MultiLayerLandConfig.snow_scheme == "layered"``,
    # ``legoesm.land.snow_column``): per-layer ice and liquid mass [kg/m2],
    # temperature [K] and density [kg/m3], ``(ncol, n_snow_layers)``, top layer
    # first.  ``snow_depth`` stays the TOTAL pack water (sum of ice + liquid), so
    # albedo, snow cover and the latent partition read it unchanged.  ``None``
    # for the bulk snowpack.
    snow_ice_layers: jax.Array | None = None
    snow_liq_layers: jax.Array | None = None
    snow_T_layers: jax.Array | None = None
    snow_rho_layers: jax.Array | None = None
    # One-layer snow thermal node temperature [K], (ncol,): the SNOW-SURFACE
    # temperature where the column holds snow, the top-soil temperature where
    # it does not.  Present iff ``config.thermal.snow_insulation``; ``None``
    # (default) keeps the legacy heat-free snow bucket.  Appended last.
    T_snow: jax.Array | None = None
