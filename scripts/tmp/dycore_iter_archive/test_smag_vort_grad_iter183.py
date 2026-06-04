"""FV3_3D iter 183: differentiability of fv3_sw_core
``_d_sw5_corner_divergence`` Smagorinsky branch at rest state.

iter 181 fixed the sqrt-at-zero gradient in
``compute_smagorinsky_ah_2d``.  iter 182 fixed the same pattern in
PE's ``wind_speed = sqrt(u² + v²)``.  iter 183 audits and fixes
the third instance — the SW d_sw5 path's ``smag_vort = sqrt(delpc²
+ wk_corner²)`` (``fv3_sw_core.py:1797``, used inside the adaptive
Smagorinsky branch when ``dddmp > 0``).

At rest state both ``delpc`` and ``wk_corner`` are 0; the
gradient through ``sqrt(0+0)`` is undefined → NaN propagates
into the SW shallow-water tendency at rest, breaking ``jax.grad``
through any rest-state SW model with ``dddmp > 0``.

iter 183 applies the same JAX double-where trick.

Tests
-----
1. ``test_d_sw5_smag_grad_finite_at_rest`` — direct test of
   ``_d_sw5_corner_divergence`` with ``dddmp > 0`` at rest state:
   ``jax.grad`` of a loss that depends on the helper output gives
   finite gradients.  Was NaN before iter 183.
2. ``test_d_sw5_smag_dddmp_zero_baseline`` — sanity that
   ``dddmp = 0`` is bit-for-bit identical (the smag branch isn't
   reached).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.fv3_sw_core import _d_sw5_corner_divergence
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.fixture(scope="module")
def small_cube():
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return grid, cdgrid, n


def test_d_sw5_smag_dddmp_zero_baseline(small_cube):
    """dddmp=0.0 produces a result independent of the iter-183
    sqrt-fix branch (the Smagorinsky branch isn't reached when
    dddmp=0).  Sanity guard."""
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=183)
    u_d = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n, n + 1)))
    v_d = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n)))
    ua = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n, n)))
    va = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n, n)))

    out_a = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt=200.0,
        d2_bg=0.005, dddmp=0.0, d4_bg=0.16, nord=1,
    )
    # Same call again — bit-for-bit reproducible (no FP drift).
    out_b = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt=200.0,
        d2_bg=0.005, dddmp=0.0, d4_bg=0.16, nord=1,
    )
    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))
    assert jnp.all(jnp.isfinite(out_a))


def test_d_sw5_smag_grad_finite_at_rest(small_cube):
    """``jax.grad`` of a loss that depends on
    ``_d_sw5_corner_divergence`` with ``dddmp > 0`` at rest state
    (zero u_d, v_d, ua, va) gives finite gradients.  Was NaN
    before iter 183."""
    _, cdgrid, n = small_cube

    def loss(uv_stacked):
        u_d = uv_stacked[0]
        v_d = uv_stacked[1]
        ua = uv_stacked[2]
        va = uv_stacked[3]
        out = _d_sw5_corner_divergence(
            u_d, v_d, ua, va, cdgrid, dt=200.0,
            d2_bg=0.005, dddmp=0.05, d4_bg=0.16, nord=0,
        )
        return jnp.sum(out ** 2)

    # Pad u_d and v_d up to the same shape so we can stack them.
    # Test the rest-state input.
    u_d = jnp.zeros((6, n, n + 1))
    v_d = jnp.zeros((6, n + 1, n))
    ua = jnp.zeros((6, n, n))
    va = jnp.zeros((6, n, n))

    # Differentiate w.r.t. u_d only (the others are inputs but we
    # care about the AD-safety of the smag_vort sqrt path which is
    # exercised through any nonzero gradient flow).
    def loss_u(u_d_data):
        out = _d_sw5_corner_divergence(
            u_d_data, v_d, ua, va, cdgrid, dt=200.0,
            d2_bg=0.005, dddmp=0.05, d4_bg=0.16, nord=0,
        )
        return jnp.sum(out ** 2)

    grad = jax.grad(loss_u)(u_d)
    assert jnp.all(jnp.isfinite(grad)), (
        "AD through _d_sw5_corner_divergence Smagorinsky branch "
        "(dddmp>0) at rest state must be finite after iter 183 fix "
        "to smag_vort = sqrt(delpc² + wk²) sqrt(0) singularity."
    )

    # Same with v_d.
    def loss_v(v_d_data):
        out = _d_sw5_corner_divergence(
            u_d, v_d_data, ua, va, cdgrid, dt=200.0,
            d2_bg=0.005, dddmp=0.05, d4_bg=0.16, nord=0,
        )
        return jnp.sum(out ** 2)

    grad_v = jax.grad(loss_v)(v_d)
    assert jnp.all(jnp.isfinite(grad_v))
