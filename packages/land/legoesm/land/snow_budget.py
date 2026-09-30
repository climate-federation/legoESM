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

# Degree-day-style snow melt rate default [kg/m2/s/K] (scheme default).  Used as
# ``rate*dt`` subtracted from SWE in kg/m2, so its unit is kg/m2 per second per kelvin
# above T_melt (NOT m w.e./s/K — the SWE budget is mass, kg/m2).
_SNOW_MELT_RATE_DEFAULT = 5.0e-6


def update_snow(
    snow: jnp.ndarray,
    snow_age: jnp.ndarray,
    T_sfc: jnp.ndarray,
    precip_snow: jnp.ndarray,
    dt: float,
    *,
    Q_net: jnp.ndarray | None = None,
    snow_melt_rate: float = _SNOW_MELT_RATE_DEFAULT,
    T_snow_melt: float = constants.T_freeze,
    snow_age_activation_K: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update snow depth and age, returning the melt amount.

    Parameters
    ----------
    snow : jnp.ndarray
        Current snow depth [kg/m2].
    snow_age : jnp.ndarray
        Current snow age [s].
    T_sfc : jnp.ndarray
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
        # Only melt when T_sfc >= T_snow_melt AND Q_net > 0
        above_freezing = T_sfc >= T_snow_melt
        energy_melt = jnp.maximum(Q_net * dt / constants.L_f, 0.0)  # kg/m2  # latent-ok: melt at T_freeze, L_f(T_freeze) == L_f
        snow_melt = jnp.where(above_freezing, energy_melt, 0.0)
        snow_melt = jnp.minimum(snow_melt, snow_available)
    else:
        # --- Legacy degree-day fallback ---
        melt_rate = snow_melt_rate * jnp.maximum(T_sfc - T_snow_melt, 0.0)
        snow_melt = jnp.minimum(melt_rate * dt, snow_available)

    snow_new = jnp.maximum(snow_available - snow_melt, 0.0)

    snow_age_new = update_snow_age(
        snow_new, snow_age, precip_snow, dt,
        T_snow=T_sfc, age_activation_K=snow_age_activation_K)

    return snow_new, snow_age_new, snow_melt


def metamorphism_rate(T_snow: jnp.ndarray, activation_K: float) -> jnp.ndarray:
    """Temperature scaling of snow grain growth, BATS/CLM form.

    ``r(T) = exp(A * (1/T_freeze - 1/T))``, the Arrhenius-like grain-growth term
    of Dickinson et al. (1993) BATS with activation temperature ``A`` [K]
    (BATS uses 5000 K).  It is 1 at the freezing point and falls steeply as the
    snow cools: 0.08 at 240 K, 0.03 at 230 K, so cold dry polar snow ages ~12-30
    times more slowly than melting snow.  Bounded to (0, 1]: snow never ages
    FASTER than at the melting point, which keeps the effective age monotone in
    real time whatever the surface temperature does.

    ``activation_K = 0`` returns exactly 1.0 (the temperature-independent clock).
    """
    if activation_K == 0.0:
        return jnp.ones_like(T_snow)
    T = jnp.maximum(T_snow, 1.0)                       # guard the 1/T at T -> 0
    return jnp.clip(
        jnp.exp(activation_K * (1.0 / constants.T_freeze - 1.0 / T)), 0.0, 1.0)


def update_snow_age(
    snow_new: jnp.ndarray,
    snow_age: jnp.ndarray,
    precip_snow: jnp.ndarray,
    dt: float,
    *,
    T_snow: jnp.ndarray | None = None,
    age_activation_K: float = 0.0,
) -> jnp.ndarray:
    """Snow-age clock: mass-weighted grain-age mixing of fresh + existing snow.

    Shared by the cell-mean budget (:func:`update_snow`) and the elevation-band
    scheme (``snow_bands``), which keeps a single cell-level age for the albedo
    decay while banding the mass budget.

    Fresh snowfall must NOT hard-reset the albedo age to zero on a trace flurry: the
    old ``precip_snow > 1e-10`` FULL reset drove the albedo to its bright maximum on
    any dusting over a deep aged pack (a large spring warm-side bias).  Instead the
    grain age mixes by MASS (CLM/BATS): existing snow ages by ``dt`` while fresh snow
    enters at age 0, so

        new_age = (snow_age + dt) * swe_old / (swe_old + fresh_swe)

    with ``fresh_swe = precip_snow*dt`` [kg/m2] the fresh snow added this step and
    ``swe_old = snow_new - fresh_swe`` [kg/m2] the pre-existing snow surviving this
    step's melt (melt removes mass but does not change grain age, so ``swe_old`` is
    ``snow_new`` net of the fresh addition).  Trace snow on a deep aged pack barely
    moves the age (``swe_old >> fresh_swe``); fresh snow on bare/thin ground drives
    it to ~0 (``swe_old -> 0``, exact for accumulation from zero SWE).  Age is forced
    to 0 once the pack has fully melted (``snow_new == 0``).
    """
    # Units: fresh_swe, swe_old, denom all [kg/m2]; snow_age, dt [s]; ratio is
    # dimensionless in [0, 1], so new_age stays in [0, snow_age + dt] -- age never
    # grows past the aged clock and never goes negative (monotone, positivity kept).
    fresh_swe = jnp.maximum(precip_snow, 0.0) * dt           # kg/m2 fresh this step
    swe_old = jnp.maximum(snow_new - fresh_swe, 0.0)         # kg/m2 old snow surviving melt
    denom = swe_old + fresh_swe                              # kg/m2 (== snow_new where >=0)
    # The clock accumulates METAMORPHIC EXPOSURE, not calendar time: the
    # increment is dt scaled by the grain-growth rate at the snow temperature
    # (BATS/CLM).  Cold dry snow therefore stays near its fresh albedo for
    # months, while melting snow darkens on the same timescale as before.  The
    # state stays in SECONDS of melting-point-equivalent age, so restarts and
    # every downstream albedo consumer are unchanged in meaning.
    # ``age_activation_K = 0`` (default) gives rate 1 => byte-identical.
    _rate = (1.0 if (T_snow is None or age_activation_K == 0.0)
             else metamorphism_rate(T_snow, age_activation_K))
    aged = snow_age + dt * _rate                             # existing snow ages
    # Guard the divide (denom>0 whenever snow_new>0; the snow_new==0 rows are masked
    # out below, so the where only prevents a 0/0 NaN gradient on those dead rows).
    mixed = aged * swe_old / jnp.where(denom > 0.0, denom, 1.0)
    # If all snow has melted, reset age to zero
    return jnp.where(snow_new > 0.0, mixed, 0.0)
