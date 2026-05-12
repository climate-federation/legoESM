"""FV3_3D iter 708: cs_prof_fv3 port.

Faithful JAX port of FV3 ``cs_prof`` (tools/fv_diagnostics.F90:
4653-4733).  Non-uniform tridiagonal PPM edge reconstruction with
Lin (2004) monotone constraints.

Tests
-----

1. ``test_cs_prof_uniform_field``.
2. ``test_cs_prof_linear_field``.
3. ``test_cs_prof_monotonicity_bounded``.
4. ``test_cs_prof_nonnegative_iv0``.
5. ``test_cs_prof_shapes_3d``.
6. ``test_cs_prof_finite``.
7. ``test_cs_prof_min_km_raises``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import cs_prof_fv3


def test_cs_prof_uniform_field():
    """Uniform q2 → uniform edge values."""
    km = 10
    q2 = jnp.full((km,), 5.0)
    delp = jnp.full((km,), 1000.0)
    q = cs_prof_fv3(q2, delp)
    assert q.shape == (km + 1,)
    assert jnp.all(jnp.abs(q - 5.0) < 1e-10)


def test_cs_prof_linear_field():
    """Linear q2(k) — PPM edges should be near-linear (monotone clip
    may activate at endpoints).  Test that interior edges are bounded
    by neighboring cell values."""
    km = 10
    q2 = jnp.linspace(0.0, 9.0, km)
    delp = jnp.full((km,), 1000.0)
    q = cs_prof_fv3(q2, delp)
    # Interior edges k=2..km-1: between q2[k-1] and q2[k]
    for k in range(2, km - 1):
        lo = float(jnp.minimum(q2[k - 1], q2[k]))
        hi = float(jnp.maximum(q2[k - 1], q2[k]))
        assert lo - 1e-10 <= float(q[k]) <= hi + 1e-10


def test_cs_prof_monotonicity_bounded():
    """Random non-monotone q2 → edge values bounded by neighboring
    cell mins/maxes (monotone PPM constraint)."""
    rng = np.random.default_rng(seed=708)
    km = 30
    q2 = jnp.asarray(rng.uniform(0.0, 100.0, size=(km,)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(km,)))
    q = cs_prof_fv3(q2, delp)
    # Top edge k=1 clipped between q2[0], q2[1]
    lo = float(jnp.minimum(q2[0], q2[1]))
    hi = float(jnp.maximum(q2[0], q2[1]))
    assert lo - 1e-10 <= float(q[1]) <= hi + 1e-10
    # Bottom edge k=km-1 clipped between q2[km-2], q2[km-1]
    lo = float(jnp.minimum(q2[km - 2], q2[km - 1]))
    hi = float(jnp.maximum(q2[km - 2], q2[km - 1]))
    assert lo - 1e-10 <= float(q[km - 1]) <= hi + 1e-10


def test_cs_prof_nonnegative_iv0():
    """iv=0 (mass species) enforces q[k] >= 0 at local mins."""
    km = 20
    # Construct profile with local min that without enforcement could go negative
    q2_vals = [10.0] * 5 + [0.5, 0.0, 0.5] + [10.0] * (km - 8)
    q2 = jnp.asarray(q2_vals[:km])
    delp = jnp.full((km,), 1000.0)
    q = cs_prof_fv3(q2, delp, iv=0)
    assert jnp.all(q >= 0.0)


def test_cs_prof_shapes_3d():
    """3-D input → 3-D output with extra edge."""
    rng = np.random.default_rng(seed=709)
    n_x, n_y, km = 4, 5, 20
    q2 = jnp.asarray(rng.uniform(0.0, 100.0, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    q = cs_prof_fv3(q2, delp)
    assert q.shape == (n_x, n_y, km + 1)


def test_cs_prof_finite():
    """Random input → finite output."""
    rng = np.random.default_rng(seed=710)
    n_x, n_y, km = 4, 4, 30
    q2 = jnp.asarray(rng.normal(scale=50.0, size=(n_x, n_y, km)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    q = cs_prof_fv3(q2, delp)
    assert jnp.all(jnp.isfinite(q))


def test_cs_prof_min_km_raises():
    """km < 4 raises (bottom closure uses k=km-2)."""
    q2 = jnp.zeros((3,))
    delp = jnp.ones((3,))
    with pytest.raises(ValueError):
        cs_prof_fv3(q2, delp)
