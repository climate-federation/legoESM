"""FV3_3D iter 293: linear-in-(u, v) scaling of Smagorinsky
strain magnitude.

The Smagorinsky form is::

    A_h_smag = c_s * dx² * sqrt(D11² + 2*D12² + D22²)

where each strain component is a linear finite difference of
(u, v).  Under uniform scaling ``(u, v) → (α u, α v)``::

    D11, D22, D12 → α D11, α D22, α D12
    |D|           → α |D|
    A_h_smag      → α A_h_smag

iter-292 pinned linear-in-c_s; iter-293 pins linear-in-(u, v)
on the orthogonal axis.

Tests
-----

1. ``test_strain_magnitude_linear_in_uv`` — scaling (u, v) by
   α produces α-scaled A_h_smag at fixed c_s and grid.
2. ``test_strain_magnitude_zero_when_uv_zero`` — A_h_smag is
   exactly zero at (u=0, v=0) (validates iter-181 sqrt(0) fix
   in helper).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._smagorinsky_visc import compute_smagorinsky_ah_2d
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup():
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=293)
    u = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1)))
    return cdgrid, u, v


@pytest.mark.parametrize("alpha", [0.5, 2.0, 3.7])
def test_strain_magnitude_linear_in_uv(alpha):
    """A_h_smag scales linearly with (u, v) magnitude."""
    cdgrid, u, v = _setup()
    cs = 0.20

    ah1 = compute_smagorinsky_ah_2d(u, v, cdgrid, cs)
    ah_alpha = compute_smagorinsky_ah_2d(alpha * u, alpha * v, cdgrid, cs)

    # At every corner: ah_alpha = |alpha| * ah1.
    np.testing.assert_allclose(
        np.asarray(ah_alpha), abs(alpha) * np.asarray(ah1),
        rtol=1e-12,
        err_msg=(
            f"A_h_smag must scale linearly with (u, v); "
            f"alpha={alpha}, expected ah_alpha = |alpha|*ah1 "
            f"bit-for-bit."
        ),
    )


def test_strain_magnitude_zero_when_uv_zero():
    """A_h_smag = 0 exactly at (u=0, v=0).  Validates iter-181
    sqrt(0) AD-safe pattern in the helper at the rest state."""
    cdgrid, _, _ = _setup()
    n = cdgrid.n
    u_zero = jnp.zeros((6, n + 1, n + 1))
    v_zero = jnp.zeros((6, n + 1, n + 1))

    ah = compute_smagorinsky_ah_2d(u_zero, v_zero, cdgrid, 0.20)
    assert jnp.all(ah == 0.0), (
        "A_h_smag must be exactly 0 at (u=0, v=0) — strain mag "
        "is sqrt(0)=0 by the iter-181 double-where pattern."
    )

    # Gradient of A_h_smag w.r.t. (u, v) at rest must be finite
    # (no NaN from sqrt(0) singularity).
    def loss(uu, vv):
        return jnp.sum(compute_smagorinsky_ah_2d(uu, vv, cdgrid, 0.20))

    g_u = jax.grad(loss, 0)(u_zero, v_zero)
    g_v = jax.grad(loss, 1)(u_zero, v_zero)
    assert jnp.all(jnp.isfinite(g_u))
    assert jnp.all(jnp.isfinite(g_v))
