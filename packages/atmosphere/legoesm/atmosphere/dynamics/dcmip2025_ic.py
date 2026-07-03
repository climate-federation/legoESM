"""DCMIP-2025 idealized initial-condition primitives (package home).

Single source of truth for the DCMIP-2025 reference-atmosphere profiles, the
per-case parameter dictionaries, and the squall-line sounding used by BOTH the
production spectral-NH initializers (``spectral_nh.dcmip25_tcN_init_spectral``)
and the cubed-sphere test-case builders under
``tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/``.

Previously these primitives lived only in the test tree and the production
``spectral_nh`` initializers imported them via ``from tests...`` — an inverted
package->tests dependency (audit item 9).  They are pure functions of
``legoesm.constants`` and a height ``z`` only, so they belong in the package;
the test modules now re-export from here.

References
----------
- DCMIP-2025: https://sites.google.com/umich.edu/dcmip-2025/
- Klemp et al. (2015): Idealized Global Nonhydrostatic Atmospheric Test Cases
  on a Reduced-Radius Sphere.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import exner_function


# ==============================================================================
# Reference-atmosphere potential-temperature profiles theta_0(z)
# ==============================================================================

def isothermal_theta_ref(T0: float = 300.0):
    """Return ``theta_0(z)`` for an isothermal atmosphere (T = const).

    Hydrostatic pressure ``p(z) = p_ref * exp(-z/H_s)`` with scale height
    ``H_s = R_d*T0/g``; ``theta = T0 * (p_ref/p)^kappa`` increases upward.
    """
    R_d = constants.R_d
    g = constants.g
    p_0 = constants.p_ref

    def theta_fn(z):
        H_s = R_d * T0 / g
        p = p_0 * jnp.exp(-z / H_s)
        return T0 / exner_function(p)  # θ = T / Π,  Π=(p/p_ref)^κ

    return theta_fn


def piecewise_lapse_theta_ref(
    T_s: float = 300.0,
    lapse_tropo: float = -5.0e-3,
    lapse_strato: float = 5.0e-3,
    z_tropopause: float = 20000.0,
):
    """Return ``theta_0(z)`` for a piecewise-linear temperature profile.

    Parameters
    ----------
    T_s : float
        Surface temperature [K].
    lapse_tropo : float
        Tropospheric lapse rate [K/m] (negative for decreasing T).
    lapse_strato : float
        Stratospheric lapse rate [K/m] (positive for increasing T).
    z_tropopause : float
        Tropopause height [m].
    """
    R_d = constants.R_d
    g = constants.g
    p_0 = constants.p_ref

    def theta_fn(z):
        # Temperature profile.
        T_trop = T_s + lapse_tropo * jnp.minimum(z, z_tropopause)
        T_above = T_trop + lapse_strato * jnp.maximum(z - z_tropopause, 0.0)
        T = jnp.where(z <= z_tropopause, T_s + lapse_tropo * z, T_above)

        # Tropospheric pressure: p = p_0*(T/T_s)^(-g/(R_d*gamma)).
        T_ratio = jnp.clip(
            T_s + lapse_tropo * jnp.minimum(z, z_tropopause), 100.0, None
        ) / T_s
        exponent_tropo = -g / (R_d * lapse_tropo)
        p_tropo = p_0 * T_ratio ** exponent_tropo

        # At the tropopause.
        T_at_trop = T_s + lapse_tropo * z_tropopause
        T_ratio_trop = jnp.clip(T_at_trop, 100.0, None) / T_s
        p_at_trop = p_0 * T_ratio_trop ** exponent_tropo

        # Stratospheric pressure (above tropopause).
        dz_above = jnp.maximum(z - z_tropopause, 0.0)
        T_strato = T_at_trop + lapse_strato * dz_above
        safe_lapse = jnp.where(
            jnp.abs(lapse_strato) > 1e-10, lapse_strato, 1e-10,
        )
        exponent_strato = -g / (R_d * safe_lapse)
        T_ratio_strato = (
            jnp.clip(T_strato, 100.0, None)
            / jnp.clip(T_at_trop, 100.0, None)
        )
        p_strato = p_at_trop * T_ratio_strato ** exponent_strato

        p = jnp.where(z <= z_tropopause, p_tropo, p_strato)
        return T / exner_function(p)  # θ = T / Π,  Π=(p/p_ref)^κ

    return theta_fn


# ==============================================================================
# Squall-line (TC3) thermodynamic sounding
# ==============================================================================

def squall_line_sounding(z, params):
    """Squall-line sounding ``(T, theta, p)`` at height ``z``.

    Tropospheric lapse rate transitioning to an isothermal stratosphere; the
    pressure is the hydrostatic integral of that temperature profile.
    """
    T_s = params["T_s"]
    T_tr = params["T_tropopause"]
    z_tr = params["z_tropopause"]
    p_s = params["p_s"]
    g = constants.g
    R_d = constants.R_d

    gamma = (T_s - T_tr) / z_tr

    T_tropo = T_s - gamma * jnp.minimum(z, z_tr)
    T = jnp.where(z <= z_tr, T_tropo, T_tr)

    T_ratio = jnp.clip(T_tropo, 100.0, None) / T_s
    exponent = g / (R_d * gamma)
    p_tropo = p_s * T_ratio ** exponent

    T_at_tr = T_s - gamma * z_tr
    p_at_tr = p_s * (jnp.clip(T_at_tr, 100.0, None) / T_s) ** exponent

    dz_above = jnp.maximum(z - z_tr, 0.0)
    p_strato = p_at_tr * jnp.exp(-g * dz_above / (R_d * T_tr))

    p = jnp.where(z <= z_tr, p_tropo, p_strato)
    theta = T / exner_function(p)  # θ = T / Π,  Π=(p/p_ref)^κ
    return T, theta, p


def squall_line_theta_fn(params):
    """Return ``theta_0(z)`` for the squall-line sounding."""
    def theta_fn(z):
        _, theta, _ = squall_line_sounding(z, params)
        return theta
    return theta_fn


# ==============================================================================
# Per-case parameter dictionaries
# ==============================================================================

TC1_PARAMS = {
    "T_s": 300.0,                    # Surface temperature [K]
    "lapse_tropo": -5.0e-3,          # Tropospheric lapse rate [K/m]
    "lapse_strato": 5.0e-3,          # Stratospheric lapse rate [K/m]
    "z_tropopause": 20000.0,         # Tropopause height [m]
    "H": 40000.0,                    # Model top [m]
    "u0": 20.0,                      # Background zonal wind [m/s]
    "mountain_lat": 20.0 * jnp.pi / 180.0,  # Mountain center latitude [rad]
    "mountain_lon": 0.0,             # Mountain center longitude [rad]
    "mountain_height": 2000.0,       # Mountain peak [m]
    "mountain_halfwidth": 72.0e3,    # Mountain half-width [m]
    "sponge_width": 10000.0,         # Sponge layer width from top [m]
    "sponge_coeff": 0.05,            # Sponge damping rate [1/s]
}


TC2_PARAMS = {
    "small_earth_factor": 20.0,      # Radius reduction factor
    "T0": 250.0,                     # Isothermal temperature [K]
    "u0": 20.0,                      # Background zonal wind [m/s]
    "H": 30000.0,                    # Model top [m]
    "sponge_width": 15000.0,         # Sponge layer width [m]
    "sponge_coeff": 1.0 / (0.1 * 86400.0),  # 0.1-day timescale [1/s]
    # Gap flow (2a) specific.
    "chain_h0": 2000.0,              # Mountain chain height [m]
    "chain_halfwidth_lon": 50.0e3,   # E-W half-width [m] (on small Earth)
    "chain_halfwidth_lat": 500.0e3,  # N-S half-extent [m]
    "gap_halfwidth": 50.0e3,         # Gap half-width [m]
    "chain_lon": jnp.pi,             # Chain center longitude [rad]
    "gap_lat": 0.0,                  # Gap latitude [rad]
    # Vortex shedding (2b) specific.
    "mountain_h0": 2000.0,           # Mountain height [m]
    "mountain_d": 50.0e3,            # Mountain half-width [m]
    "mountain_lat": 10.0 * jnp.pi / 180.0,  # Off-equator for asymmetry [rad]
    "mountain_lon": jnp.pi,
}


TC3_PARAMS = {
    "small_earth_factor": 60.0,
    "H": 20000.0,                     # Model top [m]
    "n_levels": 40,
    # Sounding parameters.
    "T_s": 302.0,                     # Surface temperature [K]
    "T_tropopause": 213.0,            # Tropopause temperature [K]
    "z_tropopause": 12000.0,          # Tropopause height [m]
    "p_s": 1.0e5,                     # Surface pressure [Pa]
    "RH_low": 0.95,                   # Surface relative humidity
    "RH_high": 0.0,                   # Stratospheric relative humidity
    "RH_transition_z": 8000.0,        # RH transition height [m]
    # Wind shear.
    "U_s": 30.0,                      # Shear magnitude [m/s]
    "z_s": 5000.0,                    # Shear layer height [m]
    "U_c": 0.0,                       # Surface velocity [m/s]
    # Bubbles.
    "n_bubbles": 9,
    "bubble_dtheta": 3.0,             # Perturbation amplitude [K]
    "bubble_rh": 5.0e3,               # Horizontal half-width [m]
    "bubble_rz": 1500.0,              # Vertical half-width [m]
    "bubble_zc": 1500.0,              # Bubble center height [m]
    "bubble_spacing": 10.0e3,         # Spacing between bubbles [m]
    "bubble_lon": jnp.pi,             # Bubble line longitude [rad]
    # Sponge.
    "sponge_width": 5000.0,
    "sponge_coeff": 0.05,
}
