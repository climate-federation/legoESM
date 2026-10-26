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

# --- compact-outer biharmonic stability (forward-Euler) ---
# The compact ∇⁴ = ∇²_compact(∇²_compact) has max eigenvalue ~1024/dx⁴ at the
# 2Δx checkerboard, so its explicit CFL bound is B·dt/dx⁴ ≤ 1/512 — 32× tighter
# than the legacy wide-outer form's ≤ 1/16 (the wide form's 2Δx null pushes its
# spectral peak to ~4Δx).  The enforce_cfl cap is divided by this factor when
# ``compact_outer`` is selected so the same ``cfl_safety`` stays sub-stable.
_COMPACT_OUTER_CFL_TIGHTENING = 32.0
# Coastal Neumann-fill reach: ∇⁴ = compact(compact) samples cells up to {i±2}
# (vs {i±3} for the wide div(grad) outer, which reaches one cell further).
_COMPACT_OUTER_FILL_PASSES = 2
_WIDE_OUTER_FILL_PASSES = 3

__physics_contract__ = {
    "summary": (
        "Biharmonic (grad^4) scale-selective lateral mixing: -B_h*grad^4(u,v) "
        "with no-slip coastal masking and -B_h*grad^4(T,S) with a Neumann "
        "coastal fill, damping grid-scale variance far more strongly than the "
        "large scales."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "degC", "S": "psu",
        "mask": "1 (1=ocean, 0=land)",
        "cfg.B_h_momentum": "m^4/s", "cfg.B_h_tracer": "m^4/s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
    },
    "sign_convention": (
        "B_h >= 0; scale-selective hyperdiffusion (tendency ~ -B*k^4, a "
        "grid-scale sink); grad^4 = grad^2(grad^2) is a double divergence so the "
        "area integral of the diffused field is conserved (tracers via a "
        "no-flux Neumann coastal fill, momentum divergence-form with no-slip "
        "masking); horizontal only, no vertical flux."
    ),
    # Double-divergence hyperdiffusion conserves the area integral of the
    # diffused heat (energy) and salt (no-flux Neumann tracer BC). Momentum is
    # NOT conserved: no-slip coastal masking is a boundary stress sink.
    "conserves": ["energy", "salt"],
    "differentiable": True,
    "reference": (
        "Biharmonic (scale-selective) lateral mixing; Griffies et al. (2000), "
        "Griffies (2004) Fundamentals of Ocean Climate Models"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_biharmonic_compact_outer.py: with compact_outer "
        "the 2Δx checkerboard is damped MAXIMALLY (~1024·B/dx⁴) while large "
        "scales are barely touched; the legacy wide outer has an exact 2Δx null. "
        "A uniform/linear field gives ~zero tendency; the cube area integral of "
        "T, S is conserved to the halo-interp limit (compact_outer no worse than "
        "the wide form)."
    ),
}


def biharmonic_lateral_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: BiharmonicConfig,
    dt: float | None = None,
) -> LateralMixingOutput:
    """Apply biharmonic lateral mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    mask : array (6, n, n)
    grid : CubedSphereGrid
    cfg : BiharmonicConfig
    dt : float, optional
        The run's timestep [s]; required when ``cfg.enforce_cfl``.

    Returns
    -------
    LateralMixingOutput
    """
    mask_3d = mask[..., jnp.newaxis]
    z = jnp.zeros_like(u)

    # Optional CFL cap on the explicit biharmonic: B·dt/dx⁴ ≤
    # cfl_safety·(1/16) for the wide outer stencil, or ·(1/512) for the
    # compact outer stencil (32× tighter — see the module constant).
    # Uses ``grid.resolution_km·1000`` as the nominal dx; works on a
    # cubed sphere because all cells are within a factor of √2 of this
    # value.  ``dt`` is the run's actual timestep.
    if cfg.enforce_cfl:
        if dt is None:
            raise ValueError(
                "BiharmonicConfig.enforce_cfl=True needs the run's timestep: "
                "pass dt to biharmonic_lateral_mixing / the ocean physics_fn "
                "(or set enforce_cfl=False explicitly).")
        dx = grid.resolution_km * 1000.0
        cfl_denom = (
            _COMPACT_OUTER_CFL_TIGHTENING if cfg.compact_outer else 1.0
        )
        coeff_cap = (
            cfg.cfl_safety * dx ** 4
            / (cfl_denom * jnp.maximum(dt, 1.0))
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
        vel_hyper = hyperdiffusion_3d(
            vel_flat, grid, B_mom_eff, compact_outer=cfg.compact_outer,
        )
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
    # Coastal Neumann-fill reach matches the ∇⁴ stencil.  The inner
    # ``compact_laplacian`` is reach-1 (``f[i±1]``); the WIDE outer
    # ``div(grad(·))`` is reach-2 (``f[i±2]``) so the full ∇²∇² samples
    # up to ``{i±3}`` (n_passes=3), while the COMPACT outer is reach-1 so
    # ∇²∇² samples only ``{i±2}`` (n_passes=2).  Land cells within the
    # operator's reach must be filled for consistent coastal gradients.
    # (Wide n_passes=3: Codex iter-53 stop-time review; n_passes=2 from
    # iter-53 under-counted the wide reach.)
    dT_dt = z
    dS_dt = z
    if cfg.B_h_tracer > 0:
        n_passes = (
            _COMPACT_OUTER_FILL_PASSES if cfg.compact_outer
            else _WIDE_OUTER_FILL_PASSES
        )
        T_filled = fill_land_cells(T, mask, grid, n_passes=n_passes)
        S_filled = fill_land_cells(S, mask, grid, n_passes=n_passes)
        tr_stack = jnp.stack([T_filled, S_filled], axis=-1)  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = tr_stack.shape
        tr_flat = tr_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        tr_hyper = hyperdiffusion_3d(
            tr_flat, grid, B_tr_eff, compact_outer=cfg.compact_outer,
        )
        tr_hyper = tr_hyper.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        dT_dt = tr_hyper[..., 0] * mask_3d
        dS_dt = tr_hyper[..., 1] * mask_3d

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
