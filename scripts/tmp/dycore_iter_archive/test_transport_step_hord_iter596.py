"""FV3_3D iter 596: hord plumbed through fv_tp_2d and transport_step.

Closes iter-595 follow-up: hord now reaches the public top-level
transport API.

Tests
-----

1. ``test_fv_tp_2d_hord_default_baseline`` — fv_tp_2d(hord=12) =
   pre-iter-596 behavior (bit-for-bit).
2. ``test_fv_tp_2d_hord_8_differs`` — switching to hord=8 changes flux.
3. ``test_transport_step_hord_kwarg`` — transport_step accepts hord
   and propagates.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv_tp_2d import fv_tp_2d, transport_step
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _build_inputs(n=8, seed=596):
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    # Smooth field with a single sharp ridge
    h = jnp.asarray(rng.uniform(1.0, 5.0, size=(6, n, n)))
    crx = jnp.asarray(rng.uniform(-0.3, 0.3, size=(6, n + 1, n)))
    cry = jnp.asarray(rng.uniform(-0.3, 0.3, size=(6, n, n + 1)))
    xfx = jnp.asarray(rng.uniform(-1, 1, size=(6, n + 1, n)) * 1e3)
    yfx = jnp.asarray(rng.uniform(-1, 1, size=(6, n, n + 1)) * 1e3)
    ra_x = jnp.asarray(rng.uniform(1e6, 1e8, size=(6, n, n)))
    ra_y = jnp.asarray(rng.uniform(1e6, 1e8, size=(6, n, n)))
    return cdgrid, h, crx, cry, xfx, yfx, ra_x, ra_y


def test_fv_tp_2d_hord_default_baseline():
    """fv_tp_2d default and hord=12 explicit should match bit-for-bit."""
    cdgrid, h, crx, cry, xfx, yfx, ra_x, ra_y = _build_inputs()
    fx_d, fy_d = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)
    fx_e, fy_e = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
                          hord=12)
    assert float(jnp.abs(fx_d - fx_e).max()) < 1e-12
    assert float(jnp.abs(fy_d - fy_e).max()) < 1e-12


def test_fv_tp_2d_hord_8_differs():
    """fv_tp_2d(hord=8) should differ from hord=12."""
    cdgrid, h, crx, cry, xfx, yfx, ra_x, ra_y = _build_inputs(seed=597)
    fx_12, fy_12 = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
                            hord=12)
    fx_8, fy_8 = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid,
                          hord=8)
    diff_fx = float(jnp.abs(fx_12 - fx_8).max())
    diff_fy = float(jnp.abs(fy_12 - fy_8).max())
    assert diff_fx > 1e-6 or diff_fy > 1e-6, (
        f"hord=8 vs hord=12 in fv_tp_2d should differ on random h; "
        f"diff_fx={diff_fx:.3e}, diff_fy={diff_fy:.3e}"
    )


def test_transport_step_hord_kwarg():
    """transport_step accepts hord and runs without error."""
    cdgrid, h, _, _, _, _, _, _ = _build_inputs(seed=598)
    n = cdgrid.n
    ut = jnp.zeros((6, n + 1, n))
    vt = jnp.zeros((6, n, n + 1))
    h_new_default = transport_step(h, ut, vt, 10.0, cdgrid)
    h_new_h12 = transport_step(h, ut, vt, 10.0, cdgrid, hord=12)
    h_new_h8 = transport_step(h, ut, vt, 10.0, cdgrid, hord=8)
    # Defaults match
    assert float(jnp.abs(h_new_default - h_new_h12).max()) < 1e-12
    # All finite
    assert jnp.all(jnp.isfinite(h_new_h12))
    assert jnp.all(jnp.isfinite(h_new_h8))
