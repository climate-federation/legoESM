"""WeatherBench-2 headline-variable diagnostics from model state.

Pure-JAX and differentiable. These map a model column state (T, q, u, v on
sigma levels + surface fields) onto the WB2 headline set on pressure levels
and screen level: Z500, T850, Q700, U/V@level, MSLP, T2m, 10 m wind.

Coordinate convention (stated once, used everywhere below):
  * geopotential Φ increases UPWARD; pressure DECREASES upward.
  * all pressures in Pa, temperatures in K, geopotential height in m.
  * z_s = phis / g is the surface elevation [m] (phis = g·z_s is surface
    geopotential, the SegmentCarry/spectral-state convention).

Reuses (never re-derives): ``legoesm.grids.vertical.compute_geopotential``
(Simmons-Burridge hydrostatic integration) and the MOST stability functions
``psi_m``/``psi_h`` from ``legoesm.core.bulk_flux``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants

__all__ = [
    "interp_to_pressure_level",
    "geopotential_on_levels",
    "geopotential_height_at",
    "mean_sea_level_pressure",
    "screen_level_t2m",
    "wind_10m",
]

# --- standard-atmosphere / measurement constants (fixed, published) ---
_STD_LAPSE_RATE_K_PER_M = 0.0065  # ICAO standard atmosphere tropospheric lapse rate
_Z_REF_2M = 2.0                   # WMO screen-temperature reference height [m]
_Z_REF_10M = 10.0                 # WMO anemometer reference height [m]


def interp_to_pressure_level(field, p_model, p_target_pa):
    """Interpolate ``field`` to ``p_target_pa`` linearly in ln(p).

    Parameters
    ----------
    field : array, shape (..., nlev)
        Field sampled on the model's full levels.
    p_model : array, shape (..., nlev)
        Pressure [Pa] at each model full level (need not be pre-sorted along
        the level axis; monotonic in pressure physically).
    p_target_pa : float
        Target pressure [Pa].

    Returns
    -------
    array, shape (...)
        ``field`` at the target pressure. Extrapolation beyond the column is
        clamped to the nearest level value (``jnp.interp`` edge behaviour) so
        no NaNs are produced. Differentiable in ``field`` (and in the two
        bracketing ``p_model`` levels) for a FIXED level ordering; the internal
        argsort makes the ordering itself piecewise-constant and duplicate
        pressures are resolved arbitrarily, so callers must pass pressures that
        are monotonic along the level axis (model full levels always are).
    """
    x = jnp.log(p_model)
    xt = jnp.log(jnp.asarray(p_target_pa, dtype=x.dtype))
    nlev = field.shape[-1]
    lead = field.shape[:-1]
    xf = x.reshape(-1, nlev)
    ff = field.reshape(-1, nlev)
    # jnp.interp requires an increasing sample axis -> sort each column in ln(p)
    order = jnp.argsort(xf, axis=-1)
    xs = jnp.take_along_axis(xf, order, axis=-1)
    fs = jnp.take_along_axis(ff, order, axis=-1)
    out = jax.vmap(lambda xp, fp: jnp.interp(xt, xp, fp))(xs, fs)
    return out.reshape(lead)


def _virtual_temperature(T, q):
    """T_v = T (1 + (R_v/R_d - 1) q) = T (1 + (1/epsilon - 1) q)."""
    return T * (1.0 + (1.0 / constants.epsilon - 1.0) * q)


def geopotential_on_levels(T, q, p_s, phis, sigma_coord):
    """Full-level geopotential Φ [J/kg] using VIRTUAL temperature.

    Dispatches on coordinate type (pure-sigma vs hybrid sigma-pressure) so the
    hydrostatic integration is correct over topography. Convention: Φ increases
    upward. Shape ``(..., nlev)``.
    """
    from legoesm.grids.vertical import (
        HybridSigmaPressureCoordinate,
        compute_geopotential,
        compute_geopotential_hybrid,
    )

    T_v = _virtual_temperature(T, q)
    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        return compute_geopotential_hybrid(T_v, p_s, sigma_coord, phis)
    return compute_geopotential(T_v, p_s, sigma_coord, phis)


def geopotential_height_at(T, q, p_s, phis, sigma_coord, p_target_pa):
    """Geopotential HEIGHT [m] at ``p_target_pa`` (e.g. Z500 at 50000 Pa).

    Uses virtual temperature in the hydrostatic integration so the result
    matches ERA5 geopotential. Convention: Φ increases upward. Works for both
    pure-sigma and hybrid sigma-pressure coordinates (dispatches on type;
    pressure comes from the coordinate's own ``pressure_at_full``).
    """
    phi = geopotential_on_levels(T, q, p_s, phis, sigma_coord)  # (..., nlev), J/kg
    p_model = sigma_coord.pressure_at_full(p_s)                 # (..., nlev), Pa (polymorphic)
    phi_at = interp_to_pressure_level(phi, p_model, p_target_pa)
    return phi_at / constants.g                                 # geopotential height [m]


def mean_sea_level_pressure(p_s, T_lowest, phis, *, lapse_rate_k_per_m=_STD_LAPSE_RATE_K_PER_M):
    """Mean sea-level pressure [Pa] via the standard hydrostatic reduction.

        p_msl = p_s · (1 + Γ z_s / T0)^(g / (R_d Γ)),   T0 = T_lowest + Γ z_s / 2

    Convention: z_s = phis/g. For z_s > 0 the reduction RAISES pressure
    (p_msl ≥ p_s); at sea level (z_s = 0) it is the identity; for below-sea-level
    cells (z_s < 0) it lowers pressure. Γ = 0 falls back to the isothermal
    hypsometric limit. Temperatures are floored at ``constants.T_min_atmosphere``
    and the power base is kept positive, so no NaNs arise for cold/deep columns.

    ``lapse_rate_k_per_m`` is a STATIC Python float (a fixed measurement
    convention — the ICAO standard-atmosphere lapse rate), branched on at trace
    time; it is not a traced/differentiable argument (MSLP is a diagnostic, not
    a loss term, and the lapse rate is never a gradient target).
    """
    z_s = phis / constants.g
    gamma = float(lapse_rate_k_per_m)                # static (config), not traced
    T_low_safe = jnp.maximum(T_lowest, constants.T_min_atmosphere)
    if abs(gamma) < 1e-8:                            # isothermal hypsometric limit
        return p_s * jnp.exp(constants.g * z_s / (constants.R_d * T_low_safe))
    T0 = jnp.maximum(T_low_safe + 0.5 * gamma * z_s, constants.T_min_atmosphere)
    exponent = constants.g / (constants.R_d * gamma)
    ratio = jnp.maximum(1.0 + gamma * z_s / T0, 1e-6)   # positive base for fractional power
    return p_s * ratio ** exponent


def _log_frac(z_ref, z_low, z0, psi_fn, obukhov_L):
    """MOST interpolation weight between the surface (z0) and the lowest level.

        f = [ln(z_ref/z0) - ψ(z_ref/L)] / [ln(z_low/z0) - ψ(z_low/L)]

    The log-law fraction is only valid for z0 < z_ref < z_low, so z0 is capped
    below z_ref and f is clipped to the physical bracket [0, 1] (matching
    core.bulk_flux.compute_most_fluxes(return_2m=True), which clips the screen
    value to the [surface, lowest-level] interval). Neutral limit (L → ∞):
    ψ → 0, so f is the pure log-law fraction.
    """
    # floor nonpositive roughness (log(.../0)=inf, log(neg)=NaN) then cap below
    # z_ref so the log-law bracket stays valid (matches core.bulk_flux z0 floor)
    z0_safe = jnp.minimum(jnp.maximum(z0, 1e-12), 0.5 * z_ref)
    num = jnp.log(z_ref / z0_safe) - psi_fn(z_ref / obukhov_L)
    den = jnp.log(z_low / z0_safe) - psi_fn(z_low / obukhov_L)
    # sign-preserving guard against a vanishing denominator (never flips sign)
    den_safe = jnp.where(den >= 0, jnp.maximum(den, 1e-6), jnp.minimum(den, -1e-6))
    return jnp.clip(num / den_safe, 0.0, 1.0)


def screen_level_t2m(T_sfc, T_lowest, z_lowest, obukhov_L, z0h, *, z_ref=_Z_REF_2M):
    """2 m temperature [K] via MOST: T2 = T_sfc + (T_low - T_sfc)·f_h.

    Convention: z up; z0h is the thermal roughness length [m].
    """
    from legoesm.core.bulk_flux import psi_h

    f = _log_frac(z_ref, z_lowest, z0h, psi_h, obukhov_L)
    return T_sfc + (T_lowest - T_sfc) * f


def wind_10m(u_lowest, v_lowest, z_lowest, obukhov_L, z0m, *, z_ref=_Z_REF_10M):
    """10 m wind SPEED [m/s] via MOST: |V10| = |V_low|·f_m.

    Convention: z up; z0m is the momentum roughness length [m]. Returns speed
    (the sign/direction is taken from the lowest-level wind by the caller).
    """
    from legoesm.core.bulk_flux import psi_m

    f = _log_frac(z_ref, z_lowest, z0m, psi_m, obukhov_L)
    speed = jnp.sqrt(u_lowest ** 2 + v_lowest ** 2 + 1e-12)  # AD-safe at calm (u=v=0)
    return speed * f
