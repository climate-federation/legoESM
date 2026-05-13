"""FV3_3D iter 704: prt_maxmin_fv3 port.

Faithful JAX port of FV3 ``prt_maxmin`` (tools/fv_diagnostics.F90:
4080-4116).  Simpler diagnostic than iter-685 ``prt_mxm`` —
returns plain min/max (no area weighting).

Tests
-----

1. ``test_prt_maxmin_known_field``.
2. ``test_prt_maxmin_fac``.
3. ``test_prt_maxmin_shapes_4d``.
4. ``test_prt_maxmin_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import prt_maxmin_fv3


def test_prt_maxmin_known_field():
    """q = linspace(0, 10) → qmin=0, qmax=10."""
    q = jnp.linspace(0.0, 10.0, 100)
    qmin, qmax = prt_maxmin_fv3(q)
    assert abs(float(qmin) - 0.0) < 1e-12
    assert abs(float(qmax) - 10.0) < 1e-12


def test_prt_maxmin_fac():
    """fac=2 doubles output."""
    q = jnp.linspace(-3.0, 7.0, 50)
    qmin, qmax = prt_maxmin_fv3(q, fac=2.0)
    assert abs(float(qmin) - (-6.0)) < 1e-12
    assert abs(float(qmax) - 14.0) < 1e-12


def test_prt_maxmin_shapes_4d():
    """4-D field input → scalar output."""
    rng = np.random.default_rng(seed=704)
    q = jnp.asarray(rng.normal(size=(6, 8, 8, 12)))
    qmin, qmax = prt_maxmin_fv3(q)
    assert qmin.shape == ()
    assert qmax.shape == ()
    assert float(qmin) == float(jnp.min(q))
    assert float(qmax) == float(jnp.max(q))


def test_prt_maxmin_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=705)
    q = jnp.asarray(rng.uniform(-100, 100, size=(6, 24, 24, 30)))
    qmin, qmax = prt_maxmin_fv3(q, fac=0.01)
    assert jnp.isfinite(qmin)
    assert jnp.isfinite(qmax)
