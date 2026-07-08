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

__physics_contract__ = {
    "summary": (
        "Harmonic (Laplacian) lateral mixing: A_h*grad^2(u,v) with no-slip "
        "coastal masking and K_h*grad^2(T,S) with a Neumann (no-flux) coastal "
        "fill."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "degC", "S": "psu",
        "mask": "1 (1=ocean, 0=land)", "cfg.A_h": "m^2/s", "cfg.K_h": "m^2/s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
    },
    "sign_convention": (
        "A_h, K_h >= 0; down-gradient Laplacian in divergence form (variance-"
        "dissipative); tracer diffusion uses a no-flux Neumann coastal fill so "
        "the area-integrated heat/salt are conserved; momentum diffusion is "
        "divergence-form with no-slip coastal masking (a boundary stress sink, "
        "like drag); horizontal operator only, no vertical flux."
    ),
    # Flux/divergence-form lateral diffusion conserves the area integral of the
    # diffused heat (energy) and salt (no-flux Neumann tracer BC). Momentum is
    # NOT conserved: no-slip coastal masking is a boundary stress sink.
    "conserves": ["energy", "salt"],
    "differentiable": True,
    "reference": (
        "Laplacian lateral friction/diffusion; Griffies (2004) Fundamentals of "
        "Ocean Climate Models"
    ),
    "idealized_test": (
        "a single-mode tracer/velocity anomaly decays (rate ~ A*k^2) toward the "
        "domain mean while the area integral of T, S is conserved (no-flux "
        "coastal fill); a uniform field gives zero tendency."
    ),
}


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

    # Velocity: zeroed on land before the Laplacian (no-slip BC).  Computed
    # UNCONDITIONALLY — ``A_h`` is a registered trainable parameter (tier-2,
    # SPEC_MODULES), so a Python ``if cfg.A_h > 0`` raises
    # TracerBoolConversionError once a traced override is spliced into the
    # config inside the loss during training.  When ``A_h == 0`` the scaled
    # Laplacian is 0, so the result is identical to the old guarded path; the
    # guard only saved a few FLOPs.  ``jnp.where`` (not ``u * mask_3d``) zeroes
    # land cells so a land-cell NaN/sentinel cannot survive as ``NaN * 0`` —
    # identical to the multiply for finite ocean values, differentiable, and
    # makes the ``A_h == 0`` result exactly zero even with NaN land sentinels.
    vel_stack = jnp.stack(
        [jnp.where(mask_3d > 0.5, u, 0.0), jnp.where(mask_3d > 0.5, v, 0.0)],
        axis=-1,
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
    # Computed unconditionally for the same AD reason as the velocity branch
    # above (``K_h`` is also a registered trainable; ``K_h == 0`` gives an
    # identical zero result).  Land cells are zeroed via ``jnp.where`` before
    # ``fill_land_cells`` so a land-cell NaN/sentinel cannot poison the
    # neighbour-average sum as ``NaN * 0``; for finite ocean values the result
    # is unchanged (land values only ever enter the fill weighted by mask = 0,
    # and ocean output cells are returned untouched).
    T_filled = fill_land_cells(jnp.where(mask_3d > 0.5, T, 0.0), mask, grid, n_passes=2)
    S_filled = fill_land_cells(jnp.where(mask_3d > 0.5, S, 0.0), mask, grid, n_passes=2)
    tr_stack = jnp.stack([T_filled, S_filled], axis=-1)  # (6, n, n, nlev, 2)
    n_face, n_i, n_j, nlev_t, n_pair = tr_stack.shape
    tr_flat = tr_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
    tr_lap_flat = laplacian_viscosity_3d(tr_flat, grid, K_h_eff)
    tr_lap = tr_lap_flat.reshape(n_face, n_i, n_j, nlev_t, n_pair)
    dT_dt = tr_lap[..., 0] * mask_3d
    dS_dt = tr_lap[..., 1] * mask_3d

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
