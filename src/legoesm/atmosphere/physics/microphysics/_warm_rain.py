"""Shared warm-rain microphysics helpers.

Functions here are used by multiple microphysics backends (Seifert-Beheng,
Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio


def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
    """Compute smooth saturation adjustment (condensation tendency).

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature [K].
    q_v : array (ncol, nlev)
        Water vapor mixing ratio [kg/kg].
    p_full : array (ncol, nlev)
        Pressure [Pa].
    dt : float
        Time step [s].
    sharpness : float
        Sigmoid sharpness for smooth condensation switch.

    Returns
    -------
    condensation : array (ncol, nlev)
        Condensation tendency [kg/kg/s].
    q_sat : array (ncol, nlev)
        Saturation mixing ratio [kg/kg].
    """
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / dt
    return condensation, q_sat


def effective_Nc(N_c, Nc_0):
    """Use config default cloud droplet number where N_c is zero.

    Parameters
    ----------
    N_c : array
        Cloud droplet number concentration [1/kg].
    Nc_0 : float
        Default cloud droplet number.

    Returns
    -------
    array : Effective N_c.
    """
    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))


def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
    """Seifert-Beheng mass-dependent autoconversion.

    Parameters
    ----------
    q_c : array
        Cloud water mixing ratio [kg/kg].
    N_c_eff : array
        Effective cloud droplet number [1/kg].
    rho : array
        Air density [kg/m3].
    k_au : float
        Autoconversion rate constant.
    x_star : float
        Mean droplet mass threshold [kg].
    sharpness : float
        Sigmoid sharpness.
    gamma_norm : float
        Gamma distribution correction (1.0 for SB/Morrison, != 1.0 for Thompson).

    Returns
    -------
    dq_c_au : array
        Cloud water autoconversion rate [kg/kg/s].
    dN_r_au : array
        Rain number formation rate [1/kg/s].
    x_c : array
        Mean cloud droplet mass [kg].
    """
    q_c_pos = jnp.clip(q_c, 0.0)
    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    onset = jax.nn.sigmoid(sharpness * (x_c - x_star))
    dq_c_au = k_au * q_c_pos ** 2 * onset * gamma_norm * rho
    dN_r_au = dq_c_au * rho / (x_star * 20.0)
    return dq_c_au, dN_r_au, x_c


def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
    """Rain collecting cloud water (accretion).

    Parameters
    ----------
    q_c, q_r : array
        Cloud water and rain mixing ratios [kg/kg].
    rho : array
        Air density [kg/m3].
    k_ac : float
        Accretion rate constant.
    gamma_norm : float
        Gamma distribution correction.

    Returns
    -------
    array : Accretion rate [kg/kg/s].
    """
    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm


def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
    """Self-collection and breakup of rain drops.

    Parameters
    ----------
    N_r : array
        Rain drop number concentration [1/kg].
    q_r : array
        Rain mixing ratio [kg/kg].
    rho : array
        Air density [kg/m3].
    k_sc : float
        Self-collection rate constant.
    breakup_sharpness : float
        Sigmoid sharpness for breakup onset.
    D_eq : float
        Equilibrium drop diameter [m].

    Returns
    -------
    dN_r_sc : array
        Self-collection tendency [1/kg/s].
    dN_r_br : array
        Breakup tendency [1/kg/s].
    """
    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
    D_r = jnp.clip(
        (jnp.clip(q_r, 0.0) * rho / jnp.clip(N_r, 1.0) / (jnp.pi / 6.0 * constants.rho_water)),
        0.0,
    ) ** (1.0 / 3.0)
    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
    dN_r_br = -dN_r_sc * breakup_frac
    return dN_r_sc, dN_r_br


def rain_evaporation(q_v, q_r, q_sat, evap_coeff):
    """Compute rain evaporation in subsaturated air.

    Parameters
    ----------
    q_v : array
        Water vapor mixing ratio [kg/kg].
    q_r : array
        Rain mixing ratio [kg/kg].
    q_sat : array
        Saturation mixing ratio [kg/kg].
    evap_coeff : float
        Evaporation rate coefficient.

    Returns
    -------
    array : Evaporation rate [kg/kg/s].
    """
    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    return evap_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525
