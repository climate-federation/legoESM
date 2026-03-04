"""Entraining mass-flux convective plume parameterization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def plume_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    p_hydro: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: PlumeConfig,
) -> OceanConvectionOutput:
    """Apply entraining mass-flux plume convection.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    p_hydro : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : PlumeConfig

    Returns
    -------
    OceanConvectionOutput
    """
    nlev = T.shape[-1]
    shape_3d = T.shape
    dtype = T.dtype
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Detect unstable surface: rho(k=0) > rho(k=1)
    surface_unstable = rho[..., 0] > rho[..., 1]  # (6, n, n)

    # Initialize plume properties at surface
    T_plume_init = T[..., 0] + cfg.T_excess
    S_plume_init = S[..., 0]

    # Descend plume using scan over levels (starting from level 1)
    def scan_fn(carry, k):
        T_plume, S_plume, active = carry
        dz_k = dz_actual[..., k]

        # Entrain environment
        entrain = cfg.epsilon * dz_k
        T_plume = (1.0 - entrain) * T_plume + entrain * T[..., k]
        S_plume = (1.0 - entrain) * S_plume + entrain * S[..., k]

        # Buoyancy check
        rho_plume = wright_eos(T_plume, S_plume, p_hydro[..., k])
        delta_rho = rho_plume - rho[..., k]

        # Plume is active where it's denser than environment (sinking)
        # Using soft mask for differentiability
        active = active * jax.nn.sigmoid(-delta_rho * 1e4)

        # Detrainment tendency at this level
        dT_k = cfg.alpha_plume * cfg.epsilon * (T_plume - T[..., k]) * active
        dS_k = cfg.alpha_plume * cfg.epsilon * (S_plume - S[..., k]) * active

        return (T_plume, S_plume, active), (dT_k, dS_k, active)

    init_active = surface_unstable.astype(dtype)
    (_, _, _), (dT_levels, dS_levels, active_levels) = jax.lax.scan(
        scan_fn,
        (T_plume_init, S_plume_init, init_active),
        jnp.arange(1, nlev),
    )

    # dT_levels shape: (nlev-1, 6, n, n) — move level axis to last
    dT_dt = jnp.zeros(shape_3d, dtype=dtype)
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)
    # Scatter tendencies to levels 1..nlev-1
    dT_levels_t = jnp.moveaxis(dT_levels, 0, -1)  # (6, n, n, nlev-1)
    dS_levels_t = jnp.moveaxis(dS_levels, 0, -1)
    dT_dt = dT_dt.at[..., 1:].set(dT_levels_t)
    dS_dt = dS_dt.at[..., 1:].set(dS_levels_t)

    # Convection flag at interfaces (average of adjacent levels' activity)
    active_t = jnp.moveaxis(active_levels, 0, -1)  # (6, n, n, nlev-1)
    flag = active_t

    return OceanConvectionOutput(
        dT_dt=dT_dt,
        dS_dt=dS_dt,
        convection_flag=flag,
    )
