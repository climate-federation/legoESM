"""Melt-pond tracking (CESM / topo-pond style).

State: per-category pond area fraction ``a_pond`` (of ice area)
and mean pond depth ``h_pond`` [m].  Volume: ``V_pond = a_pond ·
h_pond · a_ice`` (per unit grid-cell area).

Sources / sinks:

- **Melt + rain** on bare ice add water to pond volume (snow-on-ice
  intercepts precipitation; ponds appear only when the snow layer
  is depleted).
- **Drainage** to the ocean is a linear relaxation toward zero
  with timescale ``drainage_timescale``.
- **Refreezing** drains ponds when the air is colder than the
  refreeze threshold — water returns to the ice column.

These kernels are JAX-friendly and operate on arbitrary leading
spatial shape with optional trailing category axis.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ice.config import MeltPondConfig

# Canonical melt-pond defaults (single source of truth callers pass in).
_POND_DEFAULTS = MeltPondConfig()


def step_ponds(
    pond_area: jnp.ndarray,
    pond_depth: jnp.ndarray,
    *,
    melt_water_m: jnp.ndarray,
    rain_water_m: jnp.ndarray,
    ice_mask: jnp.ndarray,
    h_snow: jnp.ndarray,
    T_air: jnp.ndarray,
    dt: float,
    drainage_timescale: float,
    refreeze_threshold: float,
    pond_to_ice_max_area: float,
    depth_to_area_ratio: float,
    snow_block_threshold: float = _POND_DEFAULTS.snow_block_threshold,
    refreeze_width_K: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance pond area + depth one time step.

    Parameters
    ----------
    pond_area, pond_depth : arrays
        Current pond area fraction [0-1] (relative to ice area)
        and mean depth [m].
    melt_water_m : array
        Water input from surface ice / snow melt this step [m of
        liquid water].
    rain_water_m : array
        Water input from precipitation this step [m liquid].
    ice_mask : bool array
        True where ice exists.
    h_snow : array
        Current snow depth — ponds only collect melt once snow is
        gone.
    T_air : array
        Reference air temperature [K] (lowest atmospheric layer).
        Used to gate the refreezing path.
    dt : float
        Time step [s].
    drainage_timescale : float
        Linear drainage e-folding time [s].
    refreeze_threshold : float
        Air-T below which ponds refreeze [K].
    pond_to_ice_max_area : float
        Maximum pond area fraction.
    depth_to_area_ratio : float
        ``h_pond / a_pond`` ratio assumed for converting added
        volume into the (area, depth) split.
    refreeze_width_K : float
        Half-width [K] of the smooth linear ramp over which ponds
        transition from fully liquid to fully refrozen as ``T_air``
        crosses ``refreeze_threshold``.  Keeps the refreeze response
        differentiable; smaller → sharper (CICE-like) cutoff.

    Returns
    -------
    pond_area_new, pond_depth_new : arrays
        Updated pond state.
    drain_to_ocean_m : array
        Pond water lost to drainage this step [m of liquid water
        per unit grid-cell area] — caller routes to ocean
        freshwater channel.
    refreeze_volume_m : array
        Pond water that refroze back into ice [m of liquid water
        per unit grid-cell area] — caller adds to ice column as
        white-ice gain (no FW exchange with ocean).
    """
    # Input water this step.  Snow thicker than the configured
    # threshold blocks pond formation; thinner snow / deposition
    # patches do not.
    snow_mask = h_snow > snow_block_threshold
    input_m = jnp.where(
        ice_mask & (~snow_mask),
        jnp.maximum(melt_water_m + rain_water_m, 0.0),
        0.0,
    )

    # Current pond water volume per unit *ice* area (m³_water / m²_ice).
    V_pond = pond_area * pond_depth

    # Refreezing: when T_air < threshold, drain everything back to
    # ice.  Smooth ramp over ``refreeze_width_K`` for differentiability.
    refreeze_fraction = jnp.clip(
        (refreeze_threshold - T_air) / refreeze_width_K,
        0.0,
        1.0,
    )
    V_refreeze = V_pond * refreeze_fraction
    V_after_refreeze = jnp.maximum(V_pond - V_refreeze, 0.0)

    # Drainage: linear relaxation.
    drainage_fraction = 1.0 - jnp.exp(-dt / jnp.maximum(drainage_timescale, 1.0))
    V_drain = V_after_refreeze * drainage_fraction
    V_after_drain = jnp.maximum(V_after_refreeze - V_drain, 0.0)

    # Add new melt + rain water.
    V_new = V_after_drain + input_m

    # Recover (a_pond, h_pond) under the ratio constraint:
    #   h_pond = sqrt(depth_to_area_ratio · V_new)
    #   a_pond = V_new / max(h_pond, 1e-6)
    # Floor the sqrt argument at a tiny positive (not 0): at an empty pond
    # (V_new = 0 — the common no-melt / cold-season state) sqrt'(0) = inf gives
    # a NaN reverse-mode gradient through h_new.  The forward is unchanged for
    # any real pond (and at V_new=0 the downstream ``max(h_new, 1e-6)`` and
    # ``a_new = V_new/h_safe = 0`` are identical).  Mirrors the sea_ice.py
    # pond_depth fix (iter 2).
    h_new = jnp.sqrt(jnp.maximum(depth_to_area_ratio * V_new, 1e-14))
    h_safe = jnp.maximum(h_new, 1e-6)
    a_new = V_new / h_safe
    a_new = jnp.clip(a_new, 0.0, pond_to_ice_max_area)

    # If we hit the area cap, fold residual volume into depth.
    capped = a_new >= pond_to_ice_max_area
    a_new = jnp.where(capped, pond_to_ice_max_area, a_new)
    h_new = jnp.where(capped, V_new / jnp.maximum(a_new, 1e-6), h_new)

    return a_new, h_new, V_drain, V_refreeze
