"""Biharmonic lateral mixing.

Wraps hyperdiffusion_3d from core/operators_3d.py.
Velocity is masked (no-slip), tracers unmasked.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.operators_3d import hyperdiffusion_3d
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.dynamics.barotropic import fill_land_cells
from legoesm.ocean.physics.lateral_mixing.config import BiharmonicConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput


def biharmonic_lateral_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: BiharmonicConfig,
) -> LateralMixingOutput:
    """Apply biharmonic lateral mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    mask : array (6, n, n)
    grid : CubedSphereGrid
    cfg : BiharmonicConfig

    Returns
    -------
    LateralMixingOutput
    """
    mask_3d = mask[..., jnp.newaxis]
    z = jnp.zeros_like(u)

    # Optional CFL cap on the explicit biharmonic: B·dt/dx⁴ ≤
    # cfl_safety·(1/16).  Uses ``grid.resolution_km·1000`` as the
    # nominal dx; works on a cubed sphere because all cells are within
    # a factor of √2 of this value.  Opt-in via ``cfg.enforce_cfl``.
    if cfg.enforce_cfl:
        dx = grid.resolution_km * 1000.0
        coeff_cap = (
            cfg.cfl_safety * dx ** 4
            / jnp.maximum(cfg.cfl_dt_estimate, 1.0)
        )
        B_mom_eff = jnp.minimum(cfg.B_h_momentum, coeff_cap)
        B_tr_eff = jnp.minimum(cfg.B_h_tracer, coeff_cap)
    else:
        B_mom_eff = cfg.B_h_momentum
        B_tr_eff = cfg.B_h_tracer

    # Stack the (u, v) and (T, S) pairs along a trailing axis and fold
    # that into the level dim so ``hyperdiffusion_3d`` (∇⁴ = ∇²∇², two
    # ``pad_halo_4d`` halos per call) runs ONCE on the thicker
    # ``(6, n, n, nlev*2)`` field instead of being called twice in a
    # row.  Halves the halo MPI cost of biharmonic mixing under
    # multi-GPU sharding.
    du_dt = z
    dv_dt = z
    if cfg.B_h_momentum > 0:
        vel_stack = jnp.stack(
            [u * mask_3d, v * mask_3d], axis=-1,
        )  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = vel_stack.shape
        vel_flat = vel_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        vel_hyper = hyperdiffusion_3d(vel_flat, grid, B_mom_eff)
        vel_hyper = vel_hyper.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        du_dt = vel_hyper[..., 0]
        dv_dt = vel_hyper[..., 1]

    # Tracer biharmonic: Neumann fill at coastlines BEFORE the operator,
    # then mask the output.  Biharmonic ∇⁴ = ∇²(∇²) has a wider stencil
    # than the harmonic Laplacian, so the unfilled-input bias propagates
    # TWO stencil cells into the ocean rather than one — making the
    # no-flux BC even more important.  See harmonic.py for the
    # analogous fix.
    #
    # ``n_passes=3`` matches the biharmonic ∇⁴ stencil reach.
    # ``hyperdiffusion_3d`` composes ``compact_laplacian`` (reach 1,
    # uses ``f[i±1]``) with ``div(grad(·))`` (reach 2, uses ``f[i±2]``).
    # The full ∇²∇² then samples cells up to ``{i±3}``, so land cells
    # THREE steps from ocean must be filled for the operator to see
    # consistent gradients at coastlines.  Bridges land barriers up to
    # 3 cells wide — wider strips no longer leak.  Codex iter-53
    # stop-time review (n_passes=2 from iter-53 under-counted).
    dT_dt = z
    dS_dt = z
    if cfg.B_h_tracer > 0:
        T_filled = fill_land_cells(T, mask, grid, n_passes=3)
        S_filled = fill_land_cells(S, mask, grid, n_passes=3)
        tr_stack = jnp.stack([T_filled, S_filled], axis=-1)  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = tr_stack.shape
        tr_flat = tr_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        tr_hyper = hyperdiffusion_3d(tr_flat, grid, B_tr_eff)
        tr_hyper = tr_hyper.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        dT_dt = tr_hyper[..., 0] * mask_3d
        dS_dt = tr_hyper[..., 1] * mask_3d

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
