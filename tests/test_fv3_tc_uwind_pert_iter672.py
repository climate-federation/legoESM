"""FV3_3D iter 672: dcmip16_tc_uwind_pert port.

Faithful JAX port of FV3 ``DCMIP16_TC_uwind_pert``
(tools/test_cases.F90:7168-7197).  DCMIP16 TC vortex wind
perturbation.

Tests
-----

1. ``test_tc_pert_above_zt_zero``.
2. ``test_tc_pert_at_center_finite``.
3. ``test_tc_pert_finite``.
4. ``test_tc_pert_far_field_small``.
5. ``test_tc_pert_shapes_match``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import dcmip16_tc_uwind_pert


def test_tc_pert_above_zt_zero():
    """Above zt = 15 km, uu = vv = 0."""
    pi = jnp.pi
    uu, vv = dcmip16_tc_uwind_pert(
        jnp.asarray(20000.0), jnp.asarray(1.0e5),
        jnp.asarray(pi), jnp.asarray(pi / 18),
    )
    assert abs(float(uu)) < 1e-14
    assert abs(float(vv)) < 1e-14


def test_tc_pert_at_center_finite():
    """At TC center (r=0, z=0): finite (regularized via max(d, 1e-25))."""
    pi = jnp.pi
    uu, vv = dcmip16_tc_uwind_pert(
        jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(pi), jnp.asarray(pi / 18),
    )
    assert jnp.isfinite(uu)
    assert jnp.isfinite(vv)


def test_tc_pert_finite():
    """No NaN/Inf on random domain."""
    pi = jnp.pi
    rng = np.random.default_rng(seed=672)
    z = jnp.asarray(rng.uniform(0, 14000, size=30))
    r = jnp.asarray(rng.uniform(1.0, 1.0e6, size=30))
    lon = jnp.asarray(rng.uniform(0, 2 * pi, size=30))
    lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=30))
    uu, vv = dcmip16_tc_uwind_pert(z, r, lon, lat)
    assert jnp.all(jnp.isfinite(uu))
    assert jnp.all(jnp.isfinite(vv))


def test_tc_pert_far_field_small():
    """Far from TC center (r >> rp): perturbation small."""
    pi = jnp.pi
    uu, vv = dcmip16_tc_uwind_pert(
        jnp.asarray(5000.0), jnp.asarray(1.0e7),
        jnp.asarray(0.0), jnp.asarray(0.0),
    )
    speed = float(jnp.sqrt(uu ** 2 + vv ** 2))
    # Beyond ~10 rp, vortex tangential wind very small (1/r decay)
    assert speed < 100.0


def test_tc_pert_shapes_match():
    """Vectorized inputs preserve shape."""
    pi = jnp.pi
    z = jnp.linspace(0.0, 14000.0, 10)
    r = jnp.full_like(z, 50000.0)
    lon = jnp.full_like(z, float(pi))
    lat = jnp.full_like(z, float(pi / 18))
    uu, vv = dcmip16_tc_uwind_pert(z, r, lon, lat)
    assert uu.shape == z.shape
    assert vv.shape == z.shape
