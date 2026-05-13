"""FV3_3D iter 688: compute_brn_fv3 port.

Faithful JAX port of FV3 ``compute_brn`` (tools/fv_diagnostics.F90:
5574-5645).  Bulk Richardson Number supercell diagnostic.

Tests
-----

1. ``test_brn_zero_cape``.
2. ``test_brn_uniform_wind_floor``.
3. ``test_brn_known_shear``.
4. ``test_brn_shapes_3d``.
5. ``test_brn_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import compute_brn_fv3


def _uniform_column(km, dz=-500.0, p_top=10000.0, p_s=100000.0):
    delz = jnp.full((km,), dz)
    delp = jnp.full((km,), (p_s - p_top) / km)
    return delp, delz


def test_brn_zero_cape():
    """CAPE = 0 → BRN = 0 regardless of shear."""
    km = 20
    delp, delz = _uniform_column(km)
    ua = jnp.linspace(0.0, 30.0, km)
    va = jnp.zeros((km,))
    cape = jnp.asarray(0.0)
    brn, shear06 = compute_brn_fv3(ua, va, delp, delz, cape)
    assert abs(float(brn)) < 1e-12
    assert float(shear06) > 0.0


def test_brn_uniform_wind_floor():
    """Uniform wind → zero shear → BRN = CAPE / (0.5 · 0.1) = 20·CAPE."""
    km = 20
    delp, delz = _uniform_column(km)
    ua = jnp.full((km,), 10.0)
    va = jnp.full((km,), 5.0)
    cape = jnp.asarray(2000.0)
    brn, shear06 = compute_brn_fv3(ua, va, delp, delz, cape)
    assert abs(float(shear06)) < 1e-10
    expected = 2000.0 / (0.5 * 0.1)
    assert abs(float(brn) - expected) / expected < 1e-10


def test_brn_known_shear():
    """Layered profile: lowest 500 m wind = 0, 0-6 km mean = U.
    shear06 = U.  BRN = CAPE / (0.5 · U²).

    Set up 30-layer column with dz=-500 m: lowest layer k=29 → ht=250 m.
    Top of 6 km region near k=18.  Use ua = 0 in lowest 1 layer (z≤500),
    ua = U above.
    """
    km = 30
    delz = jnp.full((km,), -500.0)
    delp = jnp.ones((km,))
    U = 20.0
    ua = jnp.concatenate([jnp.full((km - 1,), U), jnp.zeros((1,))], axis=-1)
    va = jnp.zeros((km,))
    cape = jnp.asarray(2000.0)
    brn, shear06 = compute_brn_fv3(ua, va, delp, delz, cape)
    # 0-500m window: only k=29 (ht=250) qualifies → u005=0
    # 0-6km window: ht ≤ 6000 → ht[k]=k_from_bottom·500-250
    #               k=29 ht=250, k=28 ht=750, ..., k=18 ht=5750, k=17 ht=6250 (outside)
    # in-window k=18..29 (12 layers); u06 = (11·U + 1·0)/12 = 11U/12
    u06_expected = 11.0 * U / 12.0
    shear_expected = abs(0.0 - u06_expected)
    assert abs(float(shear06) - shear_expected) / shear_expected < 1e-10
    expected_brn = 2000.0 / (0.5 * shear_expected * shear_expected)
    assert abs(float(brn) - expected_brn) / expected_brn < 1e-10


def test_brn_shapes_3d():
    """3-D field input → 2-D output."""
    rng = np.random.default_rng(seed=688)
    n_x, n_y, km = 4, 5, 30
    ua = jnp.asarray(rng.normal(size=(n_x, n_y, km)))
    va = jnp.asarray(rng.normal(size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    delz = jnp.full((n_x, n_y, km), -300.0)
    cape = jnp.asarray(rng.uniform(0.0, 3000.0, size=(n_x, n_y)))
    brn, shear06 = compute_brn_fv3(ua, va, delp, delz, cape)
    assert brn.shape == (n_x, n_y)
    assert shear06.shape == (n_x, n_y)


def test_brn_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=689)
    km = 30
    ua = jnp.asarray(rng.normal(scale=15.0, size=(4, 4, km)))
    va = jnp.asarray(rng.normal(scale=15.0, size=(4, 4, km)))
    delp = jnp.full((4, 4, km), 800.0)
    delz = jnp.full((4, 4, km), -250.0)
    cape = jnp.asarray(rng.uniform(0.0, 4000.0, size=(4, 4)))
    brn, shear06 = compute_brn_fv3(ua, va, delp, delz, cape)
    assert jnp.all(jnp.isfinite(brn))
    assert jnp.all(jnp.isfinite(shear06))
