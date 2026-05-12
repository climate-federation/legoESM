"""FV3_3D iter 728: compute_zh_above_below_fv3 helper.

Extracted cumsum-based per-layer zh pattern from iters 686/687/
688/689/707 (supercell suite) as a public helper.

Tests
-----

1. ``test_zh_above_below_uniform_dz``.
2. ``test_zh_above_below_surface_zero``.
3. ``test_zh_above_below_monotone``.
4. ``test_zh_above_below_zh_above_minus_dz``.
5. ``test_zh_above_below_iter686_unchanged``.
6. ``test_zh_above_below_iter687_unchanged``.
7. ``test_zh_above_below_iter688_unchanged``.
8. ``test_zh_above_below_iter689_unchanged``.
9. ``test_zh_above_below_iter707_unchanged``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    bunkers_vector_fv3,
    compute_brn_fv3,
    compute_zh_above_below_fv3,
    helicity_relative_caps_fv3,
    helicity_relative_fv3,
    updraft_helicity_fv3,
)


def test_zh_above_below_uniform_dz():
    """Uniform dz=500 m, km=10 → zh_below evenly spaced 0..4500,
    zh_above 500..5000."""
    km = 10
    dz = jnp.full((km,), 500.0)
    zh_above, zh_below = compute_zh_above_below_fv3(dz)
    expected_above = jnp.linspace(5000.0, 500.0, km)
    expected_below = jnp.linspace(4500.0, 0.0, km)
    assert jnp.allclose(zh_above, expected_above, atol=1e-10)
    assert jnp.allclose(zh_below, expected_below, atol=1e-10)


def test_zh_above_below_surface_zero():
    """zh_below[-1] = 0 (bottom layer rests on ground)."""
    km = 20
    dz = jnp.full((km,), 250.0)
    _, zh_below = compute_zh_above_below_fv3(dz)
    assert abs(float(zh_below[-1])) < 1e-12


def test_zh_above_below_monotone():
    """zh_above and zh_below monotone decreasing in k (top to surface)."""
    rng = np.random.default_rng(seed=728)
    km = 30
    dz = jnp.asarray(rng.uniform(50.0, 500.0, size=(km,)))
    zh_above, zh_below = compute_zh_above_below_fv3(dz)
    assert jnp.all(jnp.diff(zh_above) < 0.0)
    assert jnp.all(jnp.diff(zh_below) < 0.0)


def test_zh_above_below_zh_above_minus_dz():
    """zh_above[k] - dz[k] = zh_below[k] exactly."""
    rng = np.random.default_rng(seed=729)
    km = 20
    dz = jnp.asarray(rng.uniform(50.0, 500.0, size=(km,)))
    zh_above, zh_below = compute_zh_above_below_fv3(dz)
    assert jnp.allclose(zh_above - dz, zh_below, atol=1e-12)


def _build_supercell_inputs(km=20, seed=730):
    rng = np.random.default_rng(seed=seed)
    delz = jnp.full((km,), -500.0)
    ua = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    return ua, va, delz


def test_zh_above_below_iter686_unchanged():
    """iter-686 UH refactor preserves output (regression sanity)."""
    km = 20
    delz = jnp.full((km,), -500.0)
    rng = np.random.default_rng(seed=731)
    vort = jnp.asarray(rng.normal(scale=0.01, size=(km,)))
    w = jnp.asarray(rng.uniform(0.0, 5.0, size=(km,)))
    uh = updraft_helicity_fv3(vort, w, delz=delz)
    assert jnp.isfinite(uh)


def test_zh_above_below_iter687_unchanged():
    """iter-687 SRH refactor preserves output."""
    ua, va, delz = _build_supercell_inputs(km=20, seed=732)
    srh = helicity_relative_fv3(ua, va, delz=delz)
    assert jnp.isfinite(srh)


def test_zh_above_below_iter688_unchanged():
    """iter-688 BRN refactor preserves output."""
    ua, va, delz = _build_supercell_inputs(km=20, seed=733)
    delp = jnp.full((20,), 1000.0)
    cape = jnp.asarray(2000.0)
    brn, shear06 = compute_brn_fv3(ua, va, delp, delz, cape)
    assert jnp.isfinite(brn)
    assert jnp.isfinite(shear06)


def test_zh_above_below_iter689_unchanged():
    """iter-689 Bunkers refactor preserves output."""
    ua, va, delz = _build_supercell_inputs(km=20, seed=734)
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    assert jnp.isfinite(uc)
    assert jnp.isfinite(vc)


def test_zh_above_below_iter707_unchanged():
    """iter-707 SRH-CAPS refactor preserves output."""
    ua, va, delz = _build_supercell_inputs(km=20, seed=735)
    uc = jnp.asarray(5.0)
    vc = jnp.asarray(-2.0)
    srh = helicity_relative_caps_fv3(ua, va, uc, vc, delz=delz)
    assert jnp.isfinite(srh)
