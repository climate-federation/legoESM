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

# Numerical guardrails used across atmosphere dynamics/physics bridges.
_THETA_MIN = 50.0       # [K]
_RHO_MIN = 1.0e-9       # [kg m^-3]
_P_MIN = 1.0            # [Pa]
_P_MAX = 2.0e7          # [Pa]


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
    theta_pos = jnp.clip(theta, _THETA_MIN, None)
    p_pos = jnp.clip(p, _P_MIN, _P_MAX)
    return theta_pos * (p_pos / constants.p_ref) ** constants.kappa


def sanitize_theta_rho(
    theta: jax.Array,
    rho: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Clip thermodynamic state to physically positive ranges.

    Returns
    -------
    theta_pos, rho_pos : jax.Array
        Potential temperature [K] and density [kg/m^3] clipped to
        positive finite floors for robust EOS/exner evaluations.
    """
    theta_pos = jnp.clip(theta, _THETA_MIN, None)
    rho_pos = jnp.clip(rho, _RHO_MIN, None)
    return theta_pos, rho_pos


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

    theta_pos, rho_pos = sanitize_theta_rho(theta, rho)
    base = jnp.clip(R_d * rho_pos * theta_pos / p_0, 1.0e-20, 1.0e20)
    p = p_0 * base ** (c_p / c_v)
    return jnp.clip(p, _P_MIN, _P_MAX)


def reconstruct_half_level_pressure_hydrostatic(
    p_full: jax.Array,
    rho_full: jax.Array,
    z_half: jax.Array,
) -> jax.Array:
    """Reconstruct interface pressure from full-level state via hydrostatic balance.

    This is intended for non-hydrostatic column physics bridges where
    full-level pressure comes from the local EOS but interface pressure is
    needed by parameterizations. Using the evolving column state avoids
    relying on a fixed reference half-level pressure profile.

    Parameters
    ----------
    p_full : jax.Array
        Full-level pressure [Pa], shape (..., nlev).
    rho_full : jax.Array
        Full-level density [kg/m^3], shape (..., nlev).
    z_half : jax.Array
        Interface height [m], shape (..., nlev+1), top-to-bottom ordering.

    Returns
    -------
    jax.Array
        Reconstructed half-level pressure [Pa], shape (..., nlev+1).
    """
    # Layer thicknesses are positive with top-to-bottom level indexing.
    dz = jnp.abs(z_half[..., :-1] - z_half[..., 1:])
    rho_pos = jnp.clip(rho_full, 1e-9, None)

    # Hydrostatic increment across each full layer.
    dp = constants.g * rho_pos * dz

    # Top interface: centered estimate from top full level.
    p_top = p_full[..., 0] - 0.5 * dp[..., 0]
    p_top = jnp.clip(p_top, 1.0, None)

    # Downward integration to all interfaces.
    p_interfaces_inner = p_top[..., None] + jnp.cumsum(dp, axis=-1)
    p_half = jnp.concatenate([p_top[..., None], p_interfaces_inner], axis=-1)
    return jnp.clip(p_half, 1.0, None)


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
