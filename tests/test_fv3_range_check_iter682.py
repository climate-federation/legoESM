"""FV3_3D iter 682: range_check_fv3 port.

Faithful JAX port of FV3 ``range_check_3d`` / ``range_check_2d``
(tools/fv_diagnostics.F90:3948-4078).

Tests
-----

1. ``test_range_check_in_range_no_bad``.
2. ``test_range_check_below_low``.
3. ``test_range_check_above_high``.
4. ``test_range_check_qmin_qmax``.
5. ``test_range_check_2d_3d_same``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import range_check_fv3


def test_range_check_in_range_no_bad():
    """All values in range → bad_range = False."""
    q = jnp.asarray([0.1, 0.5, 0.9])
    bad, qmin, qmax = range_check_fv3(q, q_low=0.0, q_hi=1.0)
    assert not bool(bad)
    assert abs(float(qmin) - 0.1) < 1e-14
    assert abs(float(qmax) - 0.9) < 1e-14


def test_range_check_below_low():
    """Min < q_low → bad_range = True."""
    q = jnp.asarray([-0.1, 0.5, 0.9])
    bad, _, _ = range_check_fv3(q, q_low=0.0, q_hi=1.0)
    assert bool(bad)


def test_range_check_above_high():
    """Max > q_hi → bad_range = True."""
    q = jnp.asarray([0.1, 0.5, 1.5])
    bad, _, _ = range_check_fv3(q, q_low=0.0, q_hi=1.0)
    assert bool(bad)


def test_range_check_qmin_qmax():
    """qmin / qmax are exact."""
    rng = np.random.default_rng(seed=682)
    q = jnp.asarray(rng.uniform(-5, 5, size=(6, 8, 8, 10)))
    _, qmin, qmax = range_check_fv3(q, q_low=-100, q_hi=100)
    assert float(qmin) == float(jnp.min(q))
    assert float(qmax) == float(jnp.max(q))


def test_range_check_2d_3d_same():
    """Works on 2D, 3D, 4D, etc. — any shape."""
    q_2d = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    q_3d = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    _, qmin_2d, qmax_2d = range_check_fv3(q_2d, 0.0, 10.0)
    _, qmin_3d, qmax_3d = range_check_fv3(q_3d, 0.0, 10.0)
    assert float(qmin_2d) == float(qmin_3d) == 1.0
    assert float(qmax_2d) == float(qmax_3d) == 4.0
