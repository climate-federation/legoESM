"""Subsurface shortwave penetration heating.

Distributes downwelling shortwave radiation through the water column
using a two-band exponential absorption profile (Paulson & Simpson 1977,
Jerlov water types).  Without this, all SW heating is applied to the
surface layer, producing unrealistically warm SST and shallow mixed layers.

References
----------
Paulson, C. A. & Simpson, J. J. (1977): Irradiance measurements in the
    upper ocean. J. Phys. Oceanogr., 7(6), 952-956.
Jerlov, N. G. (1976): Marine Optics. Elsevier, 231 pp.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# ==============================================================================
# Jerlov water type parameters
# ==============================================================================
# Two-band model: I(z) = Q_sw * [R * exp(z/zeta1) + (1-R) * exp(z/zeta2)]
# where z is negative (depth below surface), zeta1/zeta2 are e-folding depths.

class JerlovParams(NamedTuple):
    """Two-band parameters for a Jerlov water type."""
    R: float        # Fraction in short-wavelength band (red/IR)
    zeta1: float    # e-folding depth of band 1 [m] (short, ~red/IR)
    zeta2: float    # e-folding depth of band 2 [m] (long, ~blue/green)


# Standard Jerlov water types (Paulson & Simpson 1977, Table 1).
JERLOV_TYPES: dict[str, JerlovParams] = {
    "I":   JerlovParams(R=0.58, zeta1=0.35, zeta2=23.0),
    "IA":  JerlovParams(R=0.62, zeta1=0.60, zeta2=20.0),
    "IB":  JerlovParams(R=0.67, zeta1=1.00, zeta2=17.0),
    "II":  JerlovParams(R=0.77, zeta1=1.50, zeta2=14.0),
    "III": JerlovParams(R=0.78, zeta1=1.40, zeta2=7.9),
}


class ShortwavePenetrationConfig(NamedTuple):
    """Configuration for subsurface SW penetration.

    Parameters
    ----------
    water_type : str
        Jerlov water type ("I", "IA", "IB", "II", "III").
        Type I = clearest open ocean, Type III = coastal/turbid.
        Default "II" is a reasonable global average.
    """
    water_type: str = "II"


def shortwave_penetration_tendency(
    sw_down: jnp.ndarray,
    z_coord_dz_ref: jnp.ndarray,
    z_coord_z_half_ref: jnp.ndarray,
    jacobian: jnp.ndarray,
    config: ShortwavePenetrationConfig = ShortwavePenetrationConfig(),
    rho_0: float = 1025.0,  # = eos.rho_0
    c_sw: float = 3994.0,   # = eos.c_sw
) -> jnp.ndarray:
    """Compute 3D temperature tendency from subsurface SW absorption.

    Parameters
    ----------
    sw_down : array, shape (...,)
        Downwelling shortwave at the sea surface [W/m²].
    z_coord_dz_ref : array, shape (nlev,)
        Reference layer thicknesses [m].
    z_coord_z_half_ref : array, shape (nlev+1,)
        Reference interface depths [m] (negative, z_half_ref[0]=0).
    jacobian : array, shape (...,)
        Dynamic z-star Jacobian (eta + H) / H.
    config : ShortwavePenetrationConfig
    rho_0 : float
        Reference seawater density [kg/m³].
    c_sw : float
        Specific heat of seawater [J/(kg·K)].

    Returns
    -------
    array, shape (..., nlev)
        Temperature tendency dT/dt [K/s] from SW absorption.
    """
    params = JERLOV_TYPES[config.water_type]
    R = params.R
    zeta1 = params.zeta1
    zeta2 = params.zeta2

    # Interface depths (negative), shape (nlev+1,)
    z_half = z_coord_z_half_ref

    # SW flux at each interface: I(z) = Q_sw * [R*exp(z/zeta1) + (1-R)*exp(z/zeta2)]
    # z_half[0] = 0 (surface), z_half[-1] = -H_max (bottom)
    I_half = R * jnp.exp(z_half / zeta1) + (1.0 - R) * jnp.exp(z_half / zeta2)
    # Shape: (nlev+1,)

    # Fraction absorbed in each layer = I_half[k] - I_half[k+1]
    frac_absorbed = I_half[:-1] - I_half[1:]  # (nlev,)

    # Actual layer thickness
    dz_actual = z_coord_dz_ref * jacobian[..., jnp.newaxis]  # (..., nlev)

    # Temperature tendency: dT/dt = Q_sw * frac / (rho_0 * c_sw * dz)
    dT_dt = (
        sw_down[..., jnp.newaxis] * frac_absorbed / (rho_0 * c_sw * dz_actual)
    )

    return dT_dt
