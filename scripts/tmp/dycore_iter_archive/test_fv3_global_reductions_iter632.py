"""FV3_3D iter 632: global_qsum / global_mx / global_mx_c ports.

Faithful JAX ports of FV3 serial reduction helpers:
- ``global_qsum``  (fv_grid_utils.F90:2999) — sum w/o area weight
- ``global_mx``    (fv_grid_utils.F90:3020) — min/max at cell centers
- ``global_mx_c``  (fv_grid_utils.F90:3048) — min/max at cell corners

Tests
-----

1. ``test_global_qsum_constant``.
2. ``test_global_qsum_zero``.
3. ``test_global_qsum_shape_agnostic``.
4. ``test_global_mx_extremes``.
5. ``test_global_mx_uniform``.
6. ``test_global_mx_c_same_as_mx``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import global_mx, global_mx_c, global_qsum


def test_global_qsum_constant():
    """Σc = c·N."""
    p = jnp.full((4, 8), 3.0)
    assert abs(float(global_qsum(p)) - 96.0) < 1e-12


def test_global_qsum_zero():
    """Zero field → 0."""
    p = jnp.zeros((6, 5))
    assert abs(float(global_qsum(p))) < 1e-14


def test_global_qsum_shape_agnostic():
    """Works on any shape (full ndim sum)."""
    rng = np.random.default_rng(seed=632)
    for shape in [(4,), (3, 5), (2, 4, 6), (6, 8, 8, 3)]:
        p = jnp.asarray(rng.normal(size=shape))
        assert abs(float(global_qsum(p)) - float(np.sum(np.asarray(p)))) < 1e-10


def test_global_mx_extremes():
    """min/max return extremes."""
    rng = np.random.default_rng(seed=633)
    q = jnp.asarray(rng.uniform(-10, 10, size=(6, 12, 12)))
    qmin, qmax = global_mx(q)
    assert float(qmin) == float(jnp.min(q))
    assert float(qmax) == float(jnp.max(q))


def test_global_mx_uniform():
    """Uniform field → qmin = qmax = c."""
    q = jnp.full((4, 5), 7.5)
    qmin, qmax = global_mx(q)
    assert float(qmin) == 7.5
    assert float(qmax) == 7.5


def test_global_mx_c_same_as_mx():
    """global_mx_c is the same operation as global_mx (different signature
    in FV3 but identical semantics for JAX arrays)."""
    rng = np.random.default_rng(seed=634)
    q = jnp.asarray(rng.normal(size=(6, 8, 8)))
    qmin1, qmax1 = global_mx(q)
    qmin2, qmax2 = global_mx_c(q)
    assert float(qmin1) == float(qmin2)
    assert float(qmax1) == float(qmax2)
