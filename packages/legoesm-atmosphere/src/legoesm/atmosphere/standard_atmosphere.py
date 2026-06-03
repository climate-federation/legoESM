"""Physically-realistic "standard atmosphere" initial condition.

A constant-lapse-rate troposphere capped by an isothermal stratosphere, with a
prescribed equator-to-pole surface-temperature gradient.  This replaces the
uniform-``T_init`` rest state (``held_suarez_init*``) for AMIP-style runs where
a physically sensible starting climate matters.

Why this exists
---------------
The uniform 300 K rest state makes the *whole column* 300 K, so the saturation
mixing ratio stays large even in the upper troposphere.  Paired with the
``RH_init * q_sat * sigma**2`` moisture init that gives a global-mean column
water vapour of ~80 kg/m^2 — roughly 6x Earth's ~13-25 kg/m^2.  A realistic
lapse rate makes the upper troposphere cold, so ``q_sat`` (hence the column
integral) collapses to Earth-like values without any change to the moisture
init itself.

Grid-agnostic
-------------
:func:`standard_atmosphere_temperature` operates purely on a latitude array
(any shape, radians) and a 1-D ``sigma_full`` coordinate, returning a
temperature field of shape ``(*lat.shape, nlev)``.  It therefore works for
lat-lon ``(n_lat, n_lon)``, cubed-sphere ``(6, n, n)``, MPAS ``(nCells,)`` and
any other horizontal layout.

The vertical profile is the analytic constant-lapse-rate (polytropic) solution
of the hydrostatic balance: for a layer with constant lapse rate ``Gamma``
[K/m], ``T(p) = T_sfc * (p/p_s)**(R_d * Gamma / g)``.  With ``sigma = p/p_s``
this is ``T = T_sfc * sigma**(R_d*Gamma/g)``, floored at the stratospheric
temperature.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants


class StandardAtmosphereConfig(NamedTuple):
    """Parameters of the standard-atmosphere initial condition.

    Defaults are Earth-like (US-standard tropospheric lapse rate, a 40 K
    equator-to-pole surface contrast, and a 216 K isothermal stratosphere).
    These are initial-condition choices, not hot-loop tunables.
    """

    T_sfc_equator_K: float = 300.0       # surface temperature at the equator [K]
    equator_pole_delta_K: float = 40.0   # equator-minus-pole surface T drop [K]
    lapse_rate_K_per_m: float = 6.5e-3   # tropospheric lapse rate [K/m]
    T_strato_K: float = 216.0            # isothermal stratosphere floor [K]
    wind_taper_lat_deg: float = 15.0     # e-folding lat of the deep-tropics
    #                                      taper on the balanced jet [deg]:
    #                                      geostrophy is invalid at the equator,
    #                                      so the jet is tapered to ~0 there.


def standard_atmosphere_temperature(
    lat: jnp.ndarray,
    sigma_full: jnp.ndarray,
    config: StandardAtmosphereConfig = StandardAtmosphereConfig(),
) -> jnp.ndarray:
    """Temperature for a constant-lapse-rate troposphere + isothermal strato.

    Parameters
    ----------
    lat
        Latitude [radians], any shape ``(...)``.
    sigma_full
        Full-level sigma coordinate ``sigma = p / p_s``, shape ``(nlev,)``,
        in (0, 1].
    config
        :class:`StandardAtmosphereConfig`.

    Returns
    -------
    jnp.ndarray
        Temperature [K] of shape ``(*lat.shape, nlev)``, monotonically
        decreasing with height down to the stratospheric floor.
    """
    # Equator-to-pole surface-temperature gradient.  sin^2(lat) -> 0 at the
    # equator, 1 at the poles, so T_sfc spans [T_eq - delta, T_eq].
    T_sfc = config.T_sfc_equator_K - config.equator_pole_delta_K * jnp.sin(lat) ** 2

    # Polytropic (constant-lapse-rate) vertical profile in sigma, floored at the
    # stratospheric temperature.  Exponent R_d*Gamma/g is dimensionless (~0.19
    # for Earth) since Gamma is in K/m, R_d in J/kg/K, g in m/s^2.
    exponent = constants.R_d * config.lapse_rate_K_per_m / constants.g
    T = T_sfc[..., None] * sigma_full ** exponent
    return jnp.maximum(T, config.T_strato_K)


def standard_atmosphere_zonal_wind(
    lat: jnp.ndarray,
    sigma_full: jnp.ndarray,
    radius: float,
    omega: float,
    config: StandardAtmosphereConfig = StandardAtmosphereConfig(),
) -> jnp.ndarray:
    """Thermal-wind-balanced zonal wind for :func:`standard_atmosphere_temperature`.

    Without balancing winds the equator-pole temperature gradient leaves an
    unbalanced meridional pressure-gradient force, so a cold start radiates a
    burst of gravity/geostrophic-adjustment waves.  The geostrophic zonal wind
    that balances that gradient is, for the ``sin^2(lat)`` surface profile,
    analytically regular at the equator: the meridional geopotential gradient
    goes as ``sin(2·lat)`` while ``f`` goes as ``sin(lat)``, so their ratio goes
    as ``cos(lat)`` (finite everywhere).  Integrating the hydrostatic
    geopotential of the constant-lapse-rate profile and dividing by ``a·f``:

        u_g(lat, sigma) = (R_d * dT / (a * kappa * Omega)) * cos(lat)
                          * (1 - sigma_eff**kappa)

    with ``kappa = R_d * Gamma / g`` and ``sigma_eff = max(sigma, sigma_trop)``
    so the jet stops growing above the (latitude-dependent) tropopause where the
    temperature becomes isothermal and the meridional gradient vanishes — this
    keeps the jet at a realistic ~30 m/s instead of the ~130 m/s an
    un-capped integral to the model top would give.

    Parameters
    ----------
    lat
        Latitude [radians], any shape ``(...)``.
    sigma_full
        Full-level sigma coordinate, shape ``(nlev,)``.
    radius, omega
        Planetary radius [m] and rotation rate [s^-1] (``grid.radius`` /
        ``grid.omega``; default to ``constants`` at the call site).
    config
        :class:`StandardAtmosphereConfig` (must match the temperature call).

    Returns
    -------
    jnp.ndarray
        Zonal wind [m/s] of shape ``(*lat.shape, nlev)``, westerly aloft in
        both hemispheres, zero at the surface, zero at the poles.
    """
    kappa = constants.R_d * config.lapse_rate_K_per_m / constants.g
    T_sfc = config.T_sfc_equator_K - config.equator_pole_delta_K * jnp.sin(lat) ** 2

    # Latitude-dependent tropopause sigma where T_sfc*sigma^kappa = T_strato.
    # Clamp the denominator to >= T_strato before the fractional power: a
    # surface colder than the stratosphere (or a non-physical T_sfc <= 0 from a
    # tiny T_init) has no troposphere, so the ratio is 1, sigma_trop = 1 and the
    # wind is identically zero there.  The clamp also keeps the base positive so
    # the fractional power never produces NaN.
    sigma_trop = jnp.clip(
        (config.T_strato_K / jnp.maximum(T_sfc, config.T_strato_K))
        ** (1.0 / kappa),
        1e-6, 1.0,
    )
    sigma_eff = jnp.maximum(sigma_full, sigma_trop[..., None])

    amp = constants.R_d * config.equator_pole_delta_K / (radius * kappa * omega)
    u_geo = amp * jnp.cos(lat)[..., None] * (1.0 - sigma_eff ** kappa)

    # Deep-tropics taper: the bare geostrophic solution ~cos(lat) peaks at the
    # equator, but geostrophic balance is invalid there (and the meridional PGF
    # it balances vanishes as sin(2·lat) anyway).  Taper to ~0 within a few
    # ``wind_taper_lat_deg`` of the equator so the jet sits at mid-latitudes
    # with realistic weak equatorial winds, keeping the mid-latitude balance.
    lat_w = jnp.deg2rad(config.wind_taper_lat_deg)
    taper = 1.0 - jnp.exp(-(lat / lat_w) ** 2)
    return u_geo * taper[..., None]
