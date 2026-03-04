"""Enhanced diffusion for convective adjustment.

Applies large vertical diffusivity where N^2 < 0 (statically unstable).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import compute_buoyancy_frequency
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def enhanced_diffusion_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: EnhancedDiffusionConfig,
) -> OceanConvectionOutput:
    """Apply enhanced diffusion where the water column is unstable.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : EnhancedDiffusionConfig

    Returns
    -------
    OceanConvectionOutput
    """
    # N^2 at interfaces
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)

    # Diffusivity: large where N^2 < 0
    if cfg.smooth_transition:
        # Smooth sigmoid transition
        K = cfg.K_bg + (cfg.K_conv - cfg.K_bg) * jax.nn.sigmoid(
            -N2 * cfg.sigmoid_sharpness
        )
    else:
        K = jnp.where(N2 < 0.0, cfg.K_conv, cfg.K_bg)

    # Convection flag
    if cfg.smooth_transition:
        flag = jax.nn.sigmoid(-N2 * cfg.sigmoid_sharpness)
    else:
        flag = jnp.where(N2 < 0.0, 1.0, 0.0)

    # Apply variable-K vertical diffusion to T and S
    tracers = jnp.stack([T, S], axis=0)
    tr_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K),
        in_axes=0, out_axes=0,
    )(tracers)

    return OceanConvectionOutput(
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        convection_flag=flag,
    )
