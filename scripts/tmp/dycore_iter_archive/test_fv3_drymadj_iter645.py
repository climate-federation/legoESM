"""FV3_3D iter 645: drymadj port.

Faithful JAX port of FV3 ``drymadj`` (tools/init_hydro.F90:
195-275, serial branch).  Dry-mass surface pressure + adjustment.

Tests
-----

1. ``test_drymadj_ps_formula``.
2. ``test_drymadj_psd_equals_ps_no_water``.
3. ``test_drymadj_psd_less_than_ps_with_water``.
4. ``test_drymadj_dpd_zero_when_disabled``.
5. ``test_drymadj_dpd_correct``.
6. ``test_drymadj_batched``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import drymadj


def test_drymadj_ps_formula():
    """ps = ptop + Σ delp."""
    delp = jnp.asarray([[[1000.0, 2000.0, 3000.0]]])      # (1, 1, 3)
    area = jnp.asarray([[1.0]])                            # (1, 1)
    ptop = 100.0
    ps, _, _ = drymadj(delp, q=None, area=area, ptop=ptop, dry_mass=100000.0)
    assert abs(float(ps[0, 0]) - (ptop + 6000.0)) < 1e-10


def test_drymadj_psd_equals_ps_no_water():
    """psd = ps when nwat=0 or q=None."""
    rng = np.random.default_rng(seed=645)
    delp = jnp.asarray(rng.uniform(500.0, 3000.0, size=(6, 4, 4, 20)))
    area = jnp.full((6, 4, 4), 1.0e10)
    ps, psd, _ = drymadj(
        delp, q=None, area=area, ptop=100.0, dry_mass=98000.0,
        nwat=0,
    )
    assert jnp.allclose(ps, psd, atol=1e-10)


def test_drymadj_psd_less_than_ps_with_water():
    """psd < ps when q > 0 (water mass removed from total)."""
    rng = np.random.default_rng(seed=646)
    km = 20
    delp = jnp.asarray(rng.uniform(500.0, 3000.0, size=(2, 2, km)))
    # Water mixing ratio 0.01 across all levels for 1 tracer
    q = jnp.full((2, 2, km, 1), 0.01)
    area = jnp.full((2, 2), 1.0e10)
    ps, psd, _ = drymadj(
        delp, q=q, area=area, ptop=100.0, dry_mass=98000.0,
        nwat=1,
    )
    assert jnp.all(psd < ps)


def test_drymadj_dpd_zero_when_disabled():
    """adjust_dry_mass=False → dpd = 0."""
    delp = jnp.asarray([[[1000.0, 2000.0, 3000.0]]])
    area = jnp.asarray([[1.0]])
    _, _, dpd = drymadj(
        delp, q=None, area=area, ptop=100.0, dry_mass=99000.0,
        adjust_dry_mass=False,
    )
    assert abs(float(dpd)) < 1e-14


def test_drymadj_dpd_correct():
    """dpd = dry_mass - psdry."""
    rng = np.random.default_rng(seed=647)
    km = 10
    delp = jnp.asarray(rng.uniform(500.0, 3000.0, size=(6, 8, 8, km)))
    area = jnp.asarray(rng.uniform(1.0e9, 1.0e10, size=(6, 8, 8)))
    ptop = 100.0
    target = 95000.0
    _, psd, dpd = drymadj(
        delp, q=None, area=area, ptop=ptop, dry_mass=target,
    )
    expected_psdry = float(jnp.sum(psd * area) / jnp.sum(area))
    assert abs(float(dpd) - (target - expected_psdry)) < 1e-6


def test_drymadj_batched():
    """Leading axes (face, lat, lon) preserved."""
    km = 8
    rng = np.random.default_rng(seed=648)
    delp = jnp.asarray(rng.uniform(500.0, 3000.0, size=(6, 4, 4, km)))
    area = jnp.full((6, 4, 4), 1.0e10)
    ps, psd, _ = drymadj(
        delp, q=None, area=area, ptop=100.0, dry_mass=98000.0,
    )
    assert ps.shape == (6, 4, 4)
    assert psd.shape == (6, 4, 4)
    assert jnp.all(jnp.isfinite(ps))
    assert jnp.all(jnp.isfinite(psd))
