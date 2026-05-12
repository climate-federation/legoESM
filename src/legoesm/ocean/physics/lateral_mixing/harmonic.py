"""Harmonic (Laplacian) lateral mixing.

Wraps the existing laplacian_viscosity_3d from mixing.py.
Velocity is masked (no-slip BC), tracers unmasked.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.dynamics.barotropic import fill_land_cells
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d
from legoesm.ocean.physics.lateral_mixing.config import HarmonicConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput


def harmonic_lateral_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: HarmonicConfig,
) -> LateralMixingOutput:
    """Apply Laplacian lateral mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    mask : array (6, n, n) — land mask (1=ocean)
    grid : CubedSphereGrid
    cfg : HarmonicConfig

    Returns
    -------
    LateralMixingOutput
    """
    mask_3d = mask[..., jnp.newaxis]
    z = jnp.zeros_like(u)

    # Optional CFL cap on the explicit Laplacian: A·dt/dx² ≤ ¼·cfl_safety.
    # Uses ``grid.resolution_km·1000`` as the nominal dx; works on a
    # cubed sphere because all cells are within a factor of √2 of this
    # value.  Cap is opt-in via ``cfg.enforce_cfl``.
    if cfg.enforce_cfl:
        dx = grid.resolution_km * 1000.0  # scalar [m]
        coeff_cap = (
            cfg.cfl_safety * dx ** 2
            / jnp.maximum(cfg.cfl_dt_estimate, 1.0)
        )
        A_h_eff = jnp.minimum(cfg.A_h, coeff_cap)
        K_h_eff = jnp.minimum(cfg.K_h, coeff_cap)
    else:
        A_h_eff = cfg.A_h
        K_h_eff = cfg.K_h

    # Stack the (u, v) and (T, S) pairs along a trailing axis and fold
    # that into the level dim so ``laplacian_viscosity_3d`` (which uses
    # ``pad_halo_4d`` + ``divergence_3d``) runs ONCE on the thicker
    # ``(6, n, n, nlev*2)`` field.  The previous ``vmap``-over-the-pair
    # entered the operator twice and emitted two separate halo MPI
    # exchanges per call under multi-GPU sharding.

    # Velocity: masked before Laplacian (no-slip BC)
    du_dt = z
    dv_dt = z
    if cfg.A_h > 0:
        vel_stack = jnp.stack(
            [u * mask_3d, v * mask_3d], axis=-1,
        )  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = vel_stack.shape
        vel_flat = vel_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        vel_lap_flat = laplacian_viscosity_3d(vel_flat, grid, A_h_eff)
        vel_lap = vel_lap_flat.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        du_dt = vel_lap[..., 0]
        dv_dt = vel_lap[..., 1]

    # Tracers: Neumann fill at coastlines BEFORE the Laplacian, then
    # mask the output.  The fill replicates the ocean value at land
    # neighbours so the stencil sees zero gradient across the land-
    # ocean boundary (no-flux BC).  An output mask alone allows the
    # Laplacian INPUT to see whatever land-cell sentinel is in T / S
    # and propagate spurious gradients one stencil-cell into the
    # ocean before being zeroed.  ``fill_land_cells`` is JIT-friendly
    # and uses the cubed-sphere Neumann fill already in production for
    # pressure-anomaly handling.  Codex finding (iter-25) plus
    # deferred no-flux BC item.
    #
    # ``n_passes=2`` matches the stencil reach of the legoESM Laplacian.
    # ``laplacian_viscosity_3d`` uses ``div(grad(f))`` with the centred
    # FD ``(f[i+1] - f[i-1]) / dx`` (dx = distance between i±1).  The
    # resulting wide-stencil Laplacian at cell i samples cells
    # ``{i-2, i, i+2}`` — so land cells TWO steps from ocean must be
    # filled for the stencil to see correct gradients at coastlines.
    # ``n_passes=1`` (the iter-53 setting) under-counted the reach and
    # left a one-cell strip of unfilled land within the operator's view.
    # ``n_passes=2`` bridges land barriers up to 2 cells wide — wider
    # barriers no longer leak.  Codex iter-53 stop-time review.
    dT_dt = z
    dS_dt = z
    if cfg.K_h > 0:
        T_filled = fill_land_cells(T, mask, grid, n_passes=2)
        S_filled = fill_land_cells(S, mask, grid, n_passes=2)
        tr_stack = jnp.stack([T_filled, S_filled], axis=-1)  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = tr_stack.shape
        tr_flat = tr_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        tr_lap_flat = laplacian_viscosity_3d(tr_flat, grid, K_h_eff)
        tr_lap = tr_lap_flat.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        dT_dt = tr_lap[..., 0] * mask_3d
        dS_dt = tr_lap[..., 1] * mask_3d

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
