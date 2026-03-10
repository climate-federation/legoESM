"""CLUBB-lite — simplified surrogate, NOT a faithful CLUBB implementation.

This module provides the CLUBB-lite interface but internally delegates
entirely to the TKE scheme. It does NOT implement the CLUBB higher-order
closure (prognostic moments, double-Gaussian PDF, etc.).

The purpose is to reserve the CLUBB interface for future development.
Until actual CLUBB equations are added, this is functionally identical
to the TKE scheme.

The reference CLUBB (Golaz et al. 2002) includes ~13 prognostic moment
equations and binormal PDF diagnostics that are not present here.

References
----------
- Golaz, J.-C., Larson, V. E., & Cotton, W. R. (2002). A PDF-based
  model for boundary layer clouds. Part I: Method and model description.
  J. Atmos. Sci., 59, 3540-3551.
"""

from __future__ import annotations

import jax

from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig, TKEConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence


def clubb_lite_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    tke: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: CLUBBLiteConfig,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Compute turbulence tendencies using CLUBB-lite surrogate.

    WARNING: This is NOT a CLUBB implementation. It delegates entirely
    to the TKE scheme. The CLUBB interface is reserved for future
    development of a proper higher-order closure.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    tke : jax.Array
        Turbulent kinetic energy [m^2/s^2], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full : jax.Array
        Height at full levels [m], shape (ncol, nlev).
    z_half : jax.Array
        Height at half levels [m], shape (ncol, nlev+1).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : CLUBBLiteConfig

    Returns
    -------
    TurbulenceOutput
        Turbulence tendencies and diagnostics.
    tke_new : jax.Array
        Updated TKE [m^2/s^2], shape (ncol, nlev).
    """
    tke_config = TKEConfig(surface=config.surface)
    return tke_turbulence(
        u, v, T, q_v, tke,
        p_full, p_half, z_full, z_half,
        T_sfc, q_sfc, rho,
        dt, tke_config,
    )
