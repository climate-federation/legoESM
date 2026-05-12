"""FV3_3D iter 685: prt_mxm_fv3 port.

Faithful JAX port of FV3 ``prt_mxm`` (tools/fv_diagnostics.F90:
4118-4161).

Tests
-----

1. ``test_prt_mxm_shapes``.
2. ``test_prt_mxm_qmin_qmax``.
3. ``test_prt_mxm_gmean_uniform``.
4. ``test_prt_mxm_fac``.
5. ``test_prt_mxm_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import prt_mxm_fv3


def test_prt_mxm_shapes():
    """All outputs scalar."""
    n_x, n_y, km = 4, 4, 8
    q = jnp.zeros((n_x, n_y, km))
    area = jnp.ones((n_x, n_y))
    qmin, qmax, gmean = prt_mxm_fv3(q, area)
    assert qmin.shape == ()
    assert qmax.shape == ()
    assert gmean.shape == ()


def test_prt_mxm_qmin_qmax():
    """qmin/qmax = min/max of q × fac."""
    rng = np.random.default_rng(seed=685)
    q = jnp.asarray(rng.uniform(-5, 5, size=(4, 4, 6)))
    area = jnp.ones((4, 4))
    qmin, qmax, _ = prt_mxm_fv3(q, area, fac=1.0)
    assert float(qmin) == float(jnp.min(q))
    assert float(qmax) == float(jnp.max(q))


def test_prt_mxm_gmean_uniform():
    """Uniform field → gmean = constant."""
    q = jnp.full((4, 4, 6), 3.7)
    area = jnp.ones((4, 4))
    _, _, gmean = prt_mxm_fv3(q, area)
    assert abs(float(gmean) - 3.7) < 1e-12


def test_prt_mxm_fac():
    """fac=2 doubles output."""
    q = jnp.full((4, 4, 6), 1.0)
    area = jnp.ones((4, 4))
    qmin, qmax, gmean = prt_mxm_fv3(q, area, fac=2.0)
    assert abs(float(qmin) - 2.0) < 1e-12
    assert abs(float(qmax) - 2.0) < 1e-12
    assert abs(float(gmean) - 2.0) < 1e-12


def test_prt_mxm_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=686)
    q = jnp.asarray(rng.normal(size=(6, 8, 8, 12)))
    area = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 8, 8)))
    qmin, qmax, gmean = prt_mxm_fv3(q, area)
    assert jnp.isfinite(qmin)
    assert jnp.isfinite(qmax)
    assert jnp.isfinite(gmean)
