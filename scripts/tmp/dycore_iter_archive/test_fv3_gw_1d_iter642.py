"""FV3_3D iter 642: gw_1d port.

Faithful JAX port of FV3 ``gw_1d`` (tools/fv_eta.F90:2286-2344).
Gravity-wave 1D vertical-coord init.

Tests
-----

1. ``test_gw_1d_shapes``.
2. ``test_gw_1d_ak_bk_endpoints``.
3. ``test_gw_1d_ptop_equals_pe_top``.
4. ``test_gw_1d_pt1_positive``.
5. ``test_gw_1d_isothermal_vs_constant_n2``.
6. ``test_gw_1d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import gw_1d


def test_gw_1d_shapes():
    """ak, bk shapes (km+1,); pt1 shape (km,); ptop scalar."""
    km = 32
    ak, bk, ptop, pt1 = gw_1d(km, p0=1.0e5, ztop=30.0e3)
    assert ak.shape == (km + 1,)
    assert bk.shape == (km + 1,)
    assert pt1.shape == (km,)
    assert ptop.shape == ()


def test_gw_1d_ak_bk_endpoints():
    """ak[0] = pe[0] = ptop; bk[0] = 0; ak[km] = 0; bk[km] = 1."""
    km = 24
    ak, bk, ptop, _ = gw_1d(km, p0=1.0e5, ztop=30.0e3)
    assert abs(float(ak[0]) - float(ptop)) < 1e-10
    assert abs(float(bk[0])) < 1e-14
    assert abs(float(ak[km])) < 1e-14
    assert abs(float(bk[km]) - 1.0) < 1e-14


def test_gw_1d_ptop_equals_pe_top():
    """ptop returned is the top-of-model pressure (positive, < p0)."""
    km = 20
    p0 = 1.0e5
    _, _, ptop, _ = gw_1d(km, p0=p0, ztop=30.0e3)
    assert 0.0 < float(ptop) < p0


def test_gw_1d_pt1_positive():
    """Potential temperature pt1 is positive everywhere."""
    km = 32
    _, _, _, pt1 = gw_1d(km, p0=1.0e5, ztop=30.0e3)
    assert jnp.all(pt1 > 0.0)


def test_gw_1d_isothermal_vs_constant_n2():
    """Isothermal branch (N² = g²/(cp·T0)) differs from constant-N²."""
    km = 24
    _, _, ptop_iso, _ = gw_1d(km, p0=1.0e5, ztop=30.0e3, isothermal=True)
    _, _, ptop_def, _ = gw_1d(km, p0=1.0e5, ztop=30.0e3, isothermal=False)
    # Two branches use different N² → different ptop
    assert abs(float(ptop_iso) - float(ptop_def)) > 1.0


def test_gw_1d_finite():
    """No NaN/Inf in any output."""
    km = 32
    ak, bk, ptop, pt1 = gw_1d(km, p0=1.0e5, ztop=30.0e3)
    assert jnp.all(jnp.isfinite(ak))
    assert jnp.all(jnp.isfinite(bk))
    assert jnp.isfinite(ptop)
    assert jnp.all(jnp.isfinite(pt1))
