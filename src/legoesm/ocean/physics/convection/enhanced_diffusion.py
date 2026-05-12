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
    apply_diffusion: bool = True,
    dt: float | None = None,
) -> OceanConvectionOutput:
    """Apply enhanced diffusion where the water column is unstable.

    When ``dt`` is supplied, the explicit-branch CFL cap uses the
    actual physics step instead of ``cfg.cfl_dt_estimate`` — preserves
    stability and correct convection strength when runtime ``dt``
    differs from the config-time estimate.  Codex iter-3 finding #7.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : EnhancedDiffusionConfig
    apply_diffusion : bool
        Run the explicit diffusion branch.  When False, K is returned
        in the output for downstream implicit solvers.
    dt : float, optional
        Physics step [s].  When provided, used in the CFL cap instead
        of ``cfg.cfl_dt_estimate``.

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

    # CFL safety cap on the EXPLICIT branch.  Backward-Euler (implicit)
    # mixing is unconditionally stable; explicit-Euler vertical
    # diffusion requires ``K · dt / dz² ≤ 1/2``.  Without the cap a
    # ``K_conv = 1 m²/s`` over ``dz = 10 m`` would need ``dt ≤ 50 s`` —
    # 70× tighter than typical ocean physics steps.  See
    # ``EnhancedDiffusionConfig`` for ``cfl_dt_estimate`` and
    # ``cfl_safety``.
    #
    # The earlier coarse cap used the reference ``dz_ref`` alone and
    # ignored ``jacobian`` (z-star contracts layers when ``eta + H``
    # shrinks): a column with jacobian = 0.5 has dz_actual half of
    # dz_ref, so K must be 4× tighter.  Delegating the per-interface
    # cap to ``vertical_diffusion_variable_K(..., dt=dt_eff)`` uses the
    # actual layer thicknesses ``dz_ref · jacobian`` and the per-
    # interface dz_min that the leaf already computes.
    if apply_diffusion:
        dt_eff = cfg.cfl_dt_estimate if dt is None else dt
        tracers = jnp.stack([T, S], axis=0)
        tr_tend = jax.vmap(
            lambda q: vertical_diffusion_variable_K(
                q, z_coord, jacobian, K,
                dt=dt_eff, cfl_safety=cfg.cfl_safety,
            ),
            in_axes=0, out_axes=0,
        )(tracers)
        dT_dt = tr_tend[0]
        dS_dt = tr_tend[1]
    else:
        dT_dt = jnp.zeros_like(T)
        dS_dt = jnp.zeros_like(S)

    return OceanConvectionOutput(
        dT_dt=dT_dt,
        dS_dt=dS_dt,
        convection_flag=flag,
        K_v=K,
    )
