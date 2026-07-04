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
        no NaNs are produced. Differentiable in ``field`` and ``p_model``.
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


def geopotential_height_at(T, q, p_s, phis, sigma_coord, p_target_pa):
    """Geopotential HEIGHT [m] at ``p_target_pa`` (e.g. Z500 at 50000 Pa).

    Uses virtual temperature in the hydrostatic integration so the result
    matches ERA5 geopotential. Convention: Φ increases upward.
    """
    from legoesm.grids.vertical import compute_geopotential

    T_v = _virtual_temperature(T, q)
    phi = compute_geopotential(T_v, p_s, sigma_coord, phis)      # (..., nlev), J/kg
    p_model = sigma_coord.sigma_full * p_s[..., None]            # (..., nlev), Pa
    phi_at = interp_to_pressure_level(phi, p_model, p_target_pa)
    return phi_at / constants.g                                  # geopotential height [m]


def mean_sea_level_pressure(p_s, T_lowest, phis, *, lapse_rate_k_per_m=_STD_LAPSE_RATE_K_PER_M):
    """Mean sea-level pressure [Pa] via the standard hydrostatic reduction.

        p_msl = p_s · (1 + Γ z_s / T0)^(g / (R_d Γ)),   T0 = T_lowest + Γ z_s / 2

    Convention: z_s = phis/g. For z_s > 0 the reduction RAISES pressure
    (p_msl ≥ p_s); at sea level (z_s = 0) it is the identity.
    """
    z_s = phis / constants.g
    gamma = lapse_rate_k_per_m
    T0 = T_lowest + 0.5 * gamma * z_s                # mean-layer temperature [K]
    exponent = constants.g / (constants.R_d * gamma)
    ratio = 1.0 + gamma * z_s / jnp.maximum(T0, 1.0)
    return p_s * ratio ** exponent


def _log_frac(z_ref, z_low, z0, psi_fn, obukhov_L):
    """MOST interpolation weight between the surface (z0) and the lowest level.

        f = [ln(z_ref/z0) - ψ(z_ref/L)] / [ln(z_low/z0) - ψ(z_low/L)]

    Neutral limit (L → ∞): ψ → 0, so f is the pure log-law fraction in (0, 1).
    """
    num = jnp.log(z_ref / z0) - psi_fn(z_ref / obukhov_L)
    den = jnp.log(z_low / z0) - psi_fn(z_low / obukhov_L)
    return num / jnp.where(jnp.abs(den) < 1e-6, 1e-6, den)


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
    return jnp.sqrt(u_lowest ** 2 + v_lowest ** 2) * f
