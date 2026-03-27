"""Shared helpers for physics parameterization integration bridges.

Functions here are used by multiple physics packages (GWD, microphysics,
turbulence) to convert between prognostic model variables and the
column-physics inputs each scheme expects.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


# ---------------------------------------------------------------------------
# Height / thickness from hydrostatic balance
# ---------------------------------------------------------------------------

def compute_heights_from_sigma(T, p_half):
    """Approximate full- and half-level heights from hydrostatic balance.

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature at full levels [K].
    p_half : array (ncol, nlev+1)
        Pressure at half levels [Pa], TOA-first.

    Returns
    -------
    z_full : array (ncol, nlev)
        Height at full levels [m].
    z_half : array (ncol, nlev+1)
        Height at half levels [m] (surface = 0).
    """
    ncol, nlev = T.shape

    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(
        constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    )

    # Integrate from surface upward
    dz_rev = dz[:, ::-1]
    z_half_cumsum = jnp.cumsum(dz_rev, axis=1)
    z_half_inner = z_half_cumsum[:, ::-1]
    z_half = jnp.concatenate([z_half_inner, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    return z_full, z_half


def compute_layer_dz(T, p_half):
    """Approximate layer thicknesses from hydrostatic balance.

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature at full levels [K].
    p_half : array (ncol, nlev+1)
        Pressure at half levels [Pa], TOA-first.

    Returns
    -------
    dz : array (ncol, nlev)
        Layer thickness [m].
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    return jnp.abs(
        constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    )


# ---------------------------------------------------------------------------
# Density from ideal-gas law
# ---------------------------------------------------------------------------

def compute_rho(T, p_full):
    """Compute air density from the ideal gas law: rho = p / (R_d * T).

    Parameters
    ----------
    T : array
        Temperature [K].
    p_full : array
        Pressure at full levels [Pa].

    Returns
    -------
    array : Density [kg/m^3].
    """
    return p_full / (constants.R_d * jnp.clip(T, 1.0, None))


# ---------------------------------------------------------------------------
# Virtual temperature
# ---------------------------------------------------------------------------

def virtual_temperature(T, q_v):
    """Compute virtual temperature.

    Parameters
    ----------
    T : array
        Temperature [K].
    q_v : array
        Specific humidity [kg/kg].

    Returns
    -------
    array : Virtual temperature [K].
    """
    return T * (1.0 + (constants.R_v / constants.R_d - 1.0) * q_v)
