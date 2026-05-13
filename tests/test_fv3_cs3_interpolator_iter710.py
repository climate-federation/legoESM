"""FV3_3D iter 710: cs3_interpolator_fv3 port.

Faithful JAX port of FV3 ``cs3_interpolator`` (tools/
fv_diagnostics.F90:4510-4602).  log-p-level vertical interp via
PPM with multiple output levels.

Tests
-----

1. ``test_cs3_interp_uniform_field``.
2. ``test_cs3_interp_above_top``.
3. ``test_cs3_interp_below_bottom``.
4. ``test_cs3_interp_iv0_clip``.
5. ``test_cs3_interp_shapes_3d``.
6. ``test_cs3_interp_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import cs3_interpolator_fv3


def test_cs3_interp_uniform_field():
    """Uniform qin = C → qout = C at all output levels."""
    km = 10
    qin = jnp.full((km,), 3.7)
    pe = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pout = jnp.log(jnp.array([2.0e4, 5.0e4, 8.0e4]))
    qout = cs3_interpolator_fv3(qin, pe, pout)
    assert qout.shape == (3,)
    assert jnp.all(jnp.abs(qout - 3.7) < 1e-10)


def test_cs3_interp_above_top():
    """pout above top of column → qout = qe[0]."""
    km = 10
    qin = jnp.linspace(0.0, 9.0, km)
    pe = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pout = jnp.array([jnp.log(5.0e3)])  # above top
    qout = cs3_interpolator_fv3(qin, pe, pout)
    assert qout.shape == (1,)
    # Close to qin[0]=0 (top edge)
    assert float(qout[0]) <= float(qin[0]) + 0.01


def test_cs3_interp_below_bottom():
    """pout below surface → qout = qe[km] (iv != 1 path)."""
    km = 10
    qin = jnp.linspace(0.0, 9.0, km)
    pe = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pout = jnp.array([jnp.log(1.5e5)])  # below surface
    qout = cs3_interpolator_fv3(qin, pe, pout, iv=0)
    # Close to qin[-1]=9 (bottom edge)
    assert float(qout[0]) >= float(qin[-1]) - 0.01


def test_cs3_interp_iv0_clip():
    """iv=0 enforces qout >= 0."""
    km = 10
    qin = jnp.full((km,), -5.0)
    pe = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pout = jnp.log(jnp.array([5.0e4, 8.0e4]))
    qout = cs3_interpolator_fv3(qin, pe, pout, iv=0)
    assert jnp.all(qout >= 0.0)


def test_cs3_interp_shapes_3d():
    """3-D qin + 3-D pe + 1-D pout → (n_x, n_y, kd)."""
    rng = np.random.default_rng(seed=710)
    n_x, n_y, km, kd = 4, 5, 20, 4
    qin = jnp.asarray(rng.uniform(0.0, 100.0, size=(n_x, n_y, km)))
    pe_col = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pe = jnp.broadcast_to(pe_col[None, None, :], (n_x, n_y, km + 1))
    pout = jnp.log(jnp.linspace(2.0e4, 9.0e4, kd))
    qout = cs3_interpolator_fv3(qin, pe, pout, iv=0)
    assert qout.shape == (n_x, n_y, kd)


def test_cs3_interp_finite():
    """Random input → finite output."""
    rng = np.random.default_rng(seed=711)
    n_x, n_y, km, kd = 4, 4, 30, 5
    qin = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    pe_col = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pe = jnp.broadcast_to(pe_col[None, None, :], (n_x, n_y, km + 1))
    pout = jnp.log(jnp.linspace(2.0e4, 9.0e4, kd))
    qout = cs3_interpolator_fv3(qin, pe, pout, iv=-1)
    assert jnp.all(jnp.isfinite(qout))
