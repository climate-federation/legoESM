"""Shared thermodynamic utilities for atmospheric physics.

Provides fundamental thermodynamic functions used across physics
parameterizations (Kessler microphysics, convection, etc.):

1. Saturation mixing ratio (Tetens formula)
2. Temperature-potential temperature conversions
3. Pressure from equation of state
4. Moist adiabatic lapse rate and profiles
5. CAPE computation

All operations are pure JAX and compatible with jit, grad, vmap, scan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


# ==============================================================================
# Basic thermodynamic relations (extracted from kessler.py)
# ==============================================================================

def saturation_mixing_ratio(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio using Tetens formula.

    e_sat = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))
    q_sat = epsilon * e_sat / (p - e_sat)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation mixing ratio [kg/kg].
    """
    T_c = T - constants.T_freeze  # Celsius
    e_sat = 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))
    # Clip to avoid division by zero
    denom = jnp.clip(p - e_sat, 1.0, None)
    return constants.epsilon * e_sat / denom


def temperature_from_theta(
    theta: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Recover temperature from potential temperature and pressure.

    T = theta * (p / p_0)^kappa

    Parameters
    ----------
    theta : jax.Array
        Potential temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Temperature [K].
    """
    return theta * (p / constants.p_ref) ** constants.kappa


def pressure_from_eos(
    rho: jax.Array,
    theta: jax.Array,
) -> jax.Array:
    """Compute pressure from density and potential temperature.

    p = p_0 * (R_d * rho * theta / p_0)^(c_p / c_v)

    Parameters
    ----------
    rho : jax.Array
        Density [kg/m^3].
    theta : jax.Array
        Potential temperature [K].

    Returns
    -------
    jax.Array
        Pressure [Pa].
    """
    R_d = constants.R_d
    c_p = constants.c_pd
    c_v = constants.c_vd
    p_0 = constants.p_ref

    return p_0 * (R_d * rho * theta / p_0) ** (c_p / c_v)


# ==============================================================================
# Moist thermodynamic functions (for convection)
# ==============================================================================

def moist_adiabat_lapse_rate(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute the moist adiabatic lapse rate dT/dp.

    Gamma_m = (R_d * T / (c_pd * p)) *
              (1 + L_v * q_sat / (R_d * T)) /
              (1 + L_v^2 * q_sat / (c_pd * R_v * T^2))

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Moist adiabatic lapse rate dT/dp [K/Pa].
    """
    R_d = constants.R_d
    c_pd = constants.c_pd
    L_v = constants.L_v
    R_v = constants.R_v

    q_sat = saturation_mixing_ratio(T, p)

    numerator = 1.0 + L_v * q_sat / (R_d * T)
    denominator = 1.0 + L_v ** 2 * q_sat / (c_pd * R_v * T ** 2)

    return (R_d * T / (c_pd * p)) * numerator / denominator


def compute_moist_adiabat(
    T_base: jax.Array,
    p_levels: jax.Array,
) -> jax.Array:
    """Compute moist adiabatic temperature profile from surface upward.

    Integrates dT/dp = Gamma_m(T, p) upward from the lowest pressure
    level using trapezoidal predictor-corrector via jax.lax.scan.

    Parameters
    ----------
    T_base : jax.Array
        Temperature at the lowest level (surface) [K], shape (ncol,).
    p_levels : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
        Ordered top-to-bottom (p increasing with index).

    Returns
    -------
    jax.Array
        Moist adiabatic temperature profile [K], shape (ncol, nlev).
    """
    ncol, nlev = p_levels.shape

    # Reverse to scan from surface (bottom) upward (top)
    p_rev = p_levels[:, ::-1]  # (ncol, nlev), surface first

    def scan_step(T_prev, p_k):
        """Trapezoidal predictor-corrector step."""
        # p_prev is the previous pressure (higher, closer to surface)
        # p_k is current pressure (lower, closer to top)
        # We carry (T, p_prev)
        T_prev_val, p_prev_val = T_prev

        dp = p_k - p_prev_val  # negative (going upward)

        # Predictor: Euler step
        gamma_1 = moist_adiabat_lapse_rate(T_prev_val, p_prev_val)
        T_pred = T_prev_val + gamma_1 * dp

        # Corrector: trapezoidal
        gamma_2 = moist_adiabat_lapse_rate(T_pred, p_k)
        T_new = T_prev_val + 0.5 * (gamma_1 + gamma_2) * dp

        # Ensure temperature stays physical
        T_new = jnp.clip(T_new, 150.0, 500.0)

        return (T_new, p_k), T_new

    # Initial state: temperature at surface level
    init = (T_base, p_rev[:, 0])

    # Scan over levels 1..nlev-1 (moving upward from surface)
    # Transpose to (nlev-1, ncol) for scan
    p_scan = jnp.moveaxis(p_rev[:, 1:], 1, 0)  # (nlev-1, ncol)

    _, T_scan = jax.lax.scan(scan_step, init, p_scan)
    # T_scan: (nlev-1, ncol) — levels from surface+1 to top

    T_scan = jnp.moveaxis(T_scan, 0, 1)  # (ncol, nlev-1)

    # Prepend surface temperature
    T_moist_rev = jnp.concatenate([T_base[:, None], T_scan], axis=1)  # (ncol, nlev)

    # Reverse back to top-to-bottom ordering
    return T_moist_rev[:, ::-1]


def saturation_mixing_ratio_ice(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio over ice (Clausius-Clapeyron).

    e_sat_i = 611.2 * exp(L_s/R_v * (1/T_freeze - 1/T))
    q_sat_i = epsilon * e_sat_i / (p - e_sat_i)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Ice saturation mixing ratio [kg/kg].
    """
    e_sat_i = 611.2 * jnp.exp(
        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
    )
    denom = jnp.clip(p - e_sat_i, 1.0, None)
    return constants.epsilon * e_sat_i / denom


def compute_cape(
    T_env: jax.Array,
    T_parcel: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
) -> jax.Array:
    """Compute Convective Available Potential Energy (CAPE).

    CAPE = R_d * sum(max(0, T_parcel - T_env) * dp / p)

    where dp is the layer pressure thickness and the sum is over
    all levels where the parcel is warmer than the environment.

    Parameters
    ----------
    T_env : jax.Array
        Environmental temperature [K], shape (ncol, nlev).
    T_parcel : jax.Array
        Parcel temperature [K], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).

    Returns
    -------
    jax.Array
        CAPE [J/kg], shape (ncol,).
    """
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
    buoyancy = jnp.maximum(0.0, T_parcel - T_env)

    return constants.R_d * jnp.sum(buoyancy * dp / p_full, axis=1)
