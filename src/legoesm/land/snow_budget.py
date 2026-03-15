"""Snow budget: accumulation, melt, and aging.

Shared by both the slab land and multi-layer land models.
"""

from __future__ import annotations

import jax.numpy as jnp


def update_snow(
    snow: jnp.ndarray,
    snow_age: jnp.ndarray,
    T_surface: jnp.ndarray,
    precip_snow: jnp.ndarray,
    snow_melt_rate: float,
    T_snow_melt: float,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Update snow depth and age.

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
    snow_melt_rate : float
        Snowmelt rate coefficient [kg/m2/s/K above T_melt].
    T_snow_melt : float
        Temperature above which snow melts [K].
    dt : float
        Time step [s].

    Returns
    -------
    (snow_new, snow_age_new)
        Updated snow depth [kg/m2] and snow age [s].
    """
    # Accumulation from snowfall
    snow_accum = precip_snow * dt  # kg/m2

    # Melt: proportional to T above freezing
    melt_rate = snow_melt_rate * jnp.maximum(T_surface - T_snow_melt, 0.0)
    snow_melt = jnp.minimum(melt_rate * dt, snow + snow_accum)

    snow_new = jnp.maximum(snow + snow_accum - snow_melt, 0.0)

    # Snow age: reset when fresh snowfall, otherwise age
    is_snowing = precip_snow > 1e-10
    snow_age_new = jnp.where(is_snowing, 0.0, snow_age + dt)
    # If all snow has melted, reset age to zero
    snow_age_new = jnp.where(snow_new > 0.0, snow_age_new, 0.0)

    return snow_new, snow_age_new
