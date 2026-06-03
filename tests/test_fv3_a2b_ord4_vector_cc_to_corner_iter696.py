"""FV3_3D iter 696: 4th-order vector cc→corner (a2b_ord4 path).

Closes the last documented PE-vs-NH FV3-fidelity asymmetry on the
NH path's cell-centre → D-corner ``u``/``v`` lift.  ``a2b_edge.F90:
a2b_ord4`` (halo=2 + 4-pt PPM-volume / 4-pt Lagrange cascade) is
the Fortran-faithful alternative to the simple 4-pt arithmetic
average used by default.

Tests
-----

1. ``test_a2b_ord4_uniform_field``.
2. ``test_a2b_ord4_linear_field``.
3. ``test_a2b_ord4_constant_winds``.
4. ``test_a2b_ord4_default_vs_ord4_diff``.
5. ``test_a2b_ord4_3d_vs_4d_consistent``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.operators_cdgrid import (
    _a2b_ord4_corner_from_padded,
    center_to_dgrid_vector,
)


def _make_cdgrid(n=12):
    grid = create_cubed_sphere(n)
    return create_cubed_sphere_cdgrid(grid)


def test_a2b_ord4_uniform_field():
    """Uniform field padded → uniform corner output (4-pt avg ≡ value)."""
    n = 12
    f_pad = jnp.full((6, n + 4, n + 4), 3.7)
    out = _a2b_ord4_corner_from_padded(f_pad, n)
    assert out.shape == (6, n + 1, n + 1)
    assert jnp.all(jnp.abs(out - 3.7) < 1e-12)


def test_a2b_ord4_linear_field():
    """Linear ramp in i: cc f(i) = i.  Corner values at half-integer
    indices = i+0.5 exactly for 4th-order on linear input.

    Build a non-cubed-sphere padded patch (single face, ignoring rotation)
    to verify the stencil math standalone.
    """
    n = 12
    i_vals = jnp.arange(n + 4) - 1.5  # padded index: cell -2..n+1 → values -1.5..n+1.5
    f_pad = jnp.broadcast_to(i_vals[None, :, None], (6, n + 4, n + 4))
    out = _a2b_ord4_corner_from_padded(f_pad, n)
    # Corner k in [0..n]; FV3 corner k corresponds to i-axis value k-0.5
    # in cell-coords (cell centres at integer indices, corners halfway
    # between).  Expected: out[:, k, :] = k - 0.5 to high precision.
    expected = (jnp.arange(n + 1).astype(jnp.float64))[None, :, None]
    expected = jnp.broadcast_to(expected, out.shape)
    assert jnp.all(jnp.abs(out - expected) < 1e-10)


def test_a2b_ord4_constant_winds():
    """Constant u=10, v=5 on cubed sphere via ord4 path → same constant
    at corners (after vector rotation).  Locally on each face the rotation
    is identity, so corner values should equal cell-centre values.

    Test interior corners only (skip face-boundary corners where
    cross-face rotation introduces small but nonzero rotation in the
    duogrid path).
    """
    cdgrid = _make_cdgrid(n=12)
    n = 12
    u = jnp.full((6, n, n), 10.0)
    v = jnp.full((6, n, n), 5.0)
    u_d_ord4, v_d_ord4 = center_to_dgrid_vector(
        u, v, cdgrid, use_fv3_a2b_ord4=True,
    )
    # Interior corners (indices 2..n-2 to avoid edge influence)
    u_interior = u_d_ord4[:, 2:n - 1, 2:n - 1]
    v_interior = v_d_ord4[:, 2:n - 1, 2:n - 1]
    # float32 grid precision floor ~1e-6
    assert jnp.all(jnp.abs(u_interior - 10.0) < 1e-5)
    assert jnp.all(jnp.abs(v_interior - 5.0) < 1e-5)


def test_a2b_ord4_default_vs_ord4_diff():
    """Default 2nd-order ≠ ord4 path on a non-linear field — sanity
    check that the flag actually changes the result."""
    cdgrid = _make_cdgrid(n=12)
    n = 12
    rng = np.random.default_rng(seed=696)
    u = jnp.asarray(rng.normal(scale=5.0, size=(6, n, n)))
    v = jnp.asarray(rng.normal(scale=5.0, size=(6, n, n)))
    u_d_default, _ = center_to_dgrid_vector(u, v, cdgrid)
    u_d_ord4, _ = center_to_dgrid_vector(u, v, cdgrid, use_fv3_a2b_ord4=True)
    # Interior should differ
    diff = float(jnp.max(jnp.abs(u_d_default - u_d_ord4)))
    assert diff > 1e-3


def test_a2b_ord4_3d_vs_4d_consistent():
    """4D ord4 path = stack of 3D ord4 along level axis."""
    cdgrid = _make_cdgrid(n=12)
    n = 12
    nlev = 3
    rng = np.random.default_rng(seed=697)
    u_3d = jnp.asarray(rng.normal(scale=3.0, size=(6, n, n)))
    v_3d = jnp.asarray(rng.normal(scale=3.0, size=(6, n, n)))
    u_4d = jnp.broadcast_to(u_3d[..., None], (6, n, n, nlev))
    v_4d = jnp.broadcast_to(v_3d[..., None], (6, n, n, nlev))
    u_d_3d, v_d_3d = center_to_dgrid_vector(
        u_3d, v_3d, cdgrid, use_fv3_a2b_ord4=True,
    )
    u_d_4d, v_d_4d = center_to_dgrid_vector(
        u_4d, v_4d, cdgrid, use_fv3_a2b_ord4=True,
    )
    assert u_d_4d.shape == (6, n + 1, n + 1, nlev)
    # Each level identical to the 3D result
    for k in range(nlev):
        assert jnp.allclose(u_d_4d[..., k], u_d_3d, atol=1e-5)
        assert jnp.allclose(v_d_4d[..., k], v_d_3d, atol=1e-5)
