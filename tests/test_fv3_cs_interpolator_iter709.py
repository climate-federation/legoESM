"""FV3_3D iter 709: cs_interpolator_fv3 port.

Faithful JAX port of FV3 ``cs_interpolator`` (tools/
fv_diagnostics.F90:4603-4649).  Height-level interp via PPM
column profile (iter-708 cs_prof) + subcell parabolic
distribution.

Tests
-----

1. ``test_cs_interp_above_top``.
2. ``test_cs_interp_below_bottom``.
3. ``test_cs_interp_uniform_field``.
4. ``test_cs_interp_qmin_clip``.
5. ``test_cs_interp_shapes_3d``.
6. ``test_cs_interp_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import cs_interpolator_fv3


def test_cs_interp_above_top():
    """zout above top of column → qout = qe[0] (top edge value)."""
    km = 20
    qin = jnp.linspace(0.0, 19.0, km)
    wz = jnp.linspace(20000.0, 0.0, km + 1)
    qout = cs_interpolator_fv3(qin, wz, zout=25000.0, qmin=-1.0)
    # Top edge value clamps to qe[0]; for monotone qin slightly close
    # to qin[0]=0 with PPM clip
    assert float(qout) <= float(qin[0]) + 0.01


def test_cs_interp_below_bottom():
    """zout below surface → qout = qe[km] (bottom edge value)."""
    km = 20
    qin = jnp.linspace(0.0, 19.0, km)
    wz = jnp.linspace(20000.0, 0.0, km + 1)
    qout = cs_interpolator_fv3(qin, wz, zout=-100.0, qmin=-100.0)
    # Bottom edge close to qin[km-1] = 19
    assert float(qout) >= float(qin[-1]) - 0.01


def test_cs_interp_uniform_field():
    """Uniform qin = C → qout = C everywhere."""
    km = 10
    qin = jnp.full((km,), 7.0)
    wz = jnp.linspace(10000.0, 0.0, km + 1)
    for z in [9000.0, 5000.0, 1000.0]:
        qout = cs_interpolator_fv3(qin, wz, zout=z, qmin=-100.0)
        assert abs(float(qout) - 7.0) < 1e-10


def test_cs_interp_qmin_clip():
    """Negative result clipped by qmin."""
    km = 10
    qin = jnp.full((km,), -5.0)
    wz = jnp.linspace(10000.0, 0.0, km + 1)
    qout = cs_interpolator_fv3(qin, wz, zout=5000.0, qmin=0.0)
    assert float(qout) >= 0.0


def test_cs_interp_shapes_3d():
    """3-D qin + 3-D wz → 2-D qout."""
    rng = np.random.default_rng(seed=709)
    n_x, n_y, km = 4, 5, 20
    qin = jnp.asarray(rng.uniform(0.0, 100.0, size=(n_x, n_y, km)))
    wz_col = jnp.linspace(20000.0, 0.0, km + 1)
    wz = jnp.broadcast_to(wz_col[None, None, :], (n_x, n_y, km + 1))
    qout = cs_interpolator_fv3(qin, wz, zout=5000.0, qmin=-100.0)
    assert qout.shape == (n_x, n_y)


def test_cs_interp_finite():
    """Random input → finite output."""
    rng = np.random.default_rng(seed=710)
    n_x, n_y, km = 4, 4, 30
    qin = jnp.asarray(rng.normal(scale=50.0, size=(n_x, n_y, km)))
    wz_col = jnp.linspace(20000.0, 0.0, km + 1)
    wz = jnp.broadcast_to(wz_col[None, None, :], (n_x, n_y, km + 1))
    qout = cs_interpolator_fv3(qin, wz, zout=8000.0, qmin=-1e9)
    assert jnp.all(jnp.isfinite(qout))
