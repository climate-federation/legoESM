"""Snow budget: accumulation, melt, and aging.

Shared by both the slab land and multi-layer land models.

Energy-limited melt (default):
    M = min(snow, max(0, Q_net * dt / L_f))
where Q_net is the net energy available for melting [W/m2] and L_f is the
latent heat of fusion.  The melted mass is returned so callers can:
  (a) add it to the liquid-water budget, and
  (b) subtract the latent-heat cost from the surface energy budget.

A degree-day fallback is retained for backwards compatibility when Q_net
is not supplied.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def update_snow(
    snow: jnp.ndarray,
    snow_age: jnp.ndarray,
    T_surface: jnp.ndarray,
    precip_snow: jnp.ndarray,
    dt: float,
    *,
    Q_net: jnp.ndarray | None = None,
    snow_melt_rate: float = 5.0e-6,
    T_snow_melt: float = constants.T_freeze,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update snow depth and age, returning the melt amount.

    Parameters
    ----------
    snow : jnp.ndarray
        Current snow depth [kg/m2].
    snow_age : jnp.ndarray
        Current snow age [s].
    T_surface : jnp.ndarray
        Surface temperature [K].
    precip_snow : jnp.ndarray
        Snowfall rate [kg/m2/s].
    dt : float
        Time step [s].
    Q_net : jnp.ndarray or None
        Net surface energy flux available for melting [W/m2], positive
        meaning energy directed into the surface.  When provided (the
        default path in both land models), energy-limited melt is used::

            melt = min(snow_available, max(0, Q_net * dt / L_f))

        When ``None``, the legacy degree-day approach is used as a
        fallback.
    snow_melt_rate : float
        Degree-day snowmelt rate coefficient [kg/m2/s/K above T_melt].
        Only used when ``Q_net is None``.
    T_snow_melt : float
        Temperature above which snow melts [K].

    Returns
    -------
    (snow_new, snow_age_new, snow_melt)
        Updated snow depth [kg/m2], snow age [s], and amount of snow
        that melted this time-step [kg/m2].  The caller must:
        * add ``snow_melt`` to the liquid-water budget, and
        * subtract ``snow_melt * L_f / (C * d)`` from the surface
          temperature (or equivalent energy sink).
    """
    # Accumulation from snowfall
    snow_accum = precip_snow * dt  # kg/m2
    snow_available = snow + snow_accum

    if Q_net is not None:
        # --- Energy-limited melt (scientific guide, eq. for M) ---
        # Only melt when T_surface >= T_snow_melt AND Q_net > 0
        above_freezing = T_surface >= T_snow_melt
        energy_melt = jnp.maximum(Q_net * dt / constants.L_f, 0.0)  # kg/m2
        snow_melt = jnp.where(above_freezing, energy_melt, 0.0)
        snow_melt = jnp.minimum(snow_melt, snow_available)
    else:
        # --- Legacy degree-day fallback ---
        melt_rate = snow_melt_rate * jnp.maximum(T_surface - T_snow_melt, 0.0)
        snow_melt = jnp.minimum(melt_rate * dt, snow_available)

    snow_new = jnp.maximum(snow_available - snow_melt, 0.0)

    # Snow age: reset when fresh snowfall, otherwise age
    is_snowing = precip_snow > 1e-10
    snow_age_new = jnp.where(is_snowing, 0.0, snow_age + dt)
    # If all snow has melted, reset age to zero
    snow_age_new = jnp.where(snow_new > 0.0, snow_age_new, 0.0)

    return snow_new, snow_age_new, snow_melt
