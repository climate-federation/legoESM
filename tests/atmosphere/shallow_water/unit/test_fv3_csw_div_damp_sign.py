"""Regression test for the iter-57 divergence-damping sign fix in
``core/fv3_sw_core.py:fv3_csw_tendencies``.

The momentum equation with divergence damping is:
    ∂u/∂t = ... + K_d · ∂D/∂x
    ∂v/∂t = ... + K_d · ∂D/∂y
which on taking the divergence gives:
    ∂D/∂t = K_d · ∇²D
so high-k divergence noise must DAMP (decay).

The prior buggy code had the opposite sign, producing
``∂D/∂t = −K_d · ∇²D`` → grid-scale divergence GROWS.

Test strategy: initialize a divergent velocity field, compute
tendencies with and without div_damp, and assert the damping
increment ``delta_uc = duc(damp) - duc(0)`` produces a NEGATIVE
``∫D · div(delta_u, delta_v) dA`` — i.e., the damping term
extracts energy from the divergence field.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import fv3_csw_tendencies
from legoesm.core.operators_cdgrid import cgrid_divergence


def test_div_damp_extracts_energy_from_divergence():
    """The damping increment must drive ∂D/∂t = K_d·∇²D (decay).

    Strategy: take one Euler step at small dt with and without
    div_damp, compute the change in cell-centred divergence
    ``∫D² dA`` (proxy for divergence-mode energy).  With damping
    active, the divergence of the time-stepped state must DECREASE
    relative to the no-damping baseline.  Under the iter-57 bug,
    the sign was flipped → divergence energy GROWS faster.

    Why non-vacuous: the prior buggy code anti-damps; for the same
    dt and K_d this assertion would be reversed.
    """
    n = 12
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    lon = grid.lon  # (6, n, n)
    h = jnp.full((6, n, n), 1000.0)
    h_s = jnp.zeros_like(h)

    # Initialize D-grid winds with a smooth divergent pattern.  Pad
    # the divergent cell-centred pattern out to D-grid edges via
    # boundary-zero (sufficient for sign-of-energy-change check).
    u_d = jnp.zeros((6, n, n + 1))
    v_d = jnp.zeros((6, n + 1, n))
    u_pattern = 10.0 * jnp.sin(lon)
    u_d = u_d.at[..., :-1].set(u_pattern)

    K_d = 1.0e6  # m^4/s — typical FV3 div-damp magnitude
    g = constants.g
    dt = 0.1  # very small dt to isolate the linear damping increment

    # Baseline: no div damping.
    _, du0, dv0 = fv3_csw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=g, div_damp=0.0,
    )
    # With damping
    _, du1, dv1 = fv3_csw_tendencies(
        h, u_d, v_d, h_s, cdgrid, g=g, div_damp=K_d,
    )

    # Time-step the velocity field by one Euler step under each path.
    u_d_no_damp = u_d + dt * du0
    v_d_no_damp = v_d + dt * dv0
    u_d_with_damp = u_d + dt * du1
    v_d_with_damp = v_d + dt * dv1

    # Cell-centre divergence proxy: D ≈ ∂u_cc/∂x + ∂v_cc/∂y.
    # We just need any DIVERGENCE-LIKE measure to compare the two
    # post-step states.  Use the simplest C-grid divergence
    # ``(u[1:] - u[:-1])/dx + (v[:, 1:] - v[:, :-1])/dy`` on the
    # tangent-space grid.
    def _D_cell(u, v):
        # u: (6, n, n+1) → cell-centre divergence x-component
        Dx = (u[..., 1:] - u[..., :-1])
        Dy = (v[..., 1:, :] - v[..., :-1, :])
        return Dx + Dy  # (6, n, n) — proportional to actual ∇·v

    D_no = _D_cell(u_d_no_damp, v_d_no_damp)
    D_da = _D_cell(u_d_with_damp, v_d_with_damp)
    energy_no = float(jnp.sum(D_no ** 2))
    energy_da = float(jnp.sum(D_da ** 2))

    # Damping must REDUCE divergence energy relative to no-damping
    # baseline (within any roundoff).  The buggy anti-damping code
    # would INCREASE it.
    assert energy_da < energy_no, (
        f"div_damp INCREASED divergence energy "
        f"(with_damp = {energy_da:.6e}, no_damp = {energy_no:.6e}). "
        f"The iter-57 bug caused this anti-damping; with the fix "
        f"the inequality must reverse."
    )
