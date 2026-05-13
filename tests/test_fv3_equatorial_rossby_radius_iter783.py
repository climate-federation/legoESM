"""FV3_3D iter 783: equatorial_rossby_radius_fv3 (L_eq = √(c/(2β))).

Composes iter-780 ``beta_plane_fv3``.  Complements iter-781 (L_R
diverges as f → 0 at equator).

Tests
-----

1. ``test_lreq_kelvin_atmospheric``: c=30, β=2.29e-11 → L_eq ≈ 810 km.
2. ``test_lreq_ocean_baroclinic``: c=2.7, β=2.29e-11 → L_eq ≈ 243 km.
3. ``test_lreq_zero_c``: c=0 → L_eq=0.
4. ``test_lreq_monotonic_c``: ↑c → ↑L_eq.
5. ``test_lreq_monotonic_beta``: ↑β → ↓L_eq.
6. ``test_lreq_composes_iter780``: pipeline test with iter-780.
7. ``test_lreq_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    beta_plane_fv3,
    equatorial_rossby_radius_fv3,
)


def test_lreq_kelvin_atmospheric():
    """Atmospheric Kelvin wave: c=30 m/s, β=2.29e-11 → L_eq ≈ 810 km."""
    c = jnp.array([30.0])
    beta = jnp.array([2.29e-11])
    L = equatorial_rossby_radius_fv3(c, beta)
    expected = jnp.sqrt(30.0 / (2.0 * 2.29e-11))
    np.testing.assert_allclose(np.asarray(L), [expected], rtol=1e-12)
    # Sanity: ~810 km
    assert 7.5e5 < float(L[0]) < 8.5e5


def test_lreq_ocean_baroclinic():
    """Ocean baroclinic mode 1: c=2.7 m/s → L_eq ≈ 243 km."""
    c = jnp.array([2.7])
    beta = jnp.array([2.29e-11])
    L = equatorial_rossby_radius_fv3(c, beta)
    # √(2.7 / (2·2.29e-11)) ≈ 243 km
    assert 2.0e5 < float(L[0]) < 2.6e5


def test_lreq_zero_c():
    """c=0 → L_eq=0."""
    c = jnp.array([0.0, 0.0, 0.0])
    beta = jnp.array([1e-11, 2e-11, 5e-11])
    L = equatorial_rossby_radius_fv3(c, beta)
    np.testing.assert_allclose(np.asarray(L), [0.0, 0.0, 0.0], atol=1e-15)


def test_lreq_monotonic_c():
    """↑c → ↑L_eq at fixed β."""
    beta = jnp.array([2e-11, 2e-11, 2e-11])
    c_lo = jnp.array([10.0, 30.0, 60.0])
    c_hi = c_lo + 10.0
    L_lo = equatorial_rossby_radius_fv3(c_lo, beta)
    L_hi = equatorial_rossby_radius_fv3(c_hi, beta)
    assert jnp.all(L_hi > L_lo)


def test_lreq_monotonic_beta():
    """↑β → ↓L_eq at fixed c."""
    c = jnp.array([30.0, 30.0, 30.0])
    beta_lo = jnp.array([1e-11, 1.5e-11, 2e-11])
    beta_hi = beta_lo * 4.0  # 4× β → halved L
    L_lo = equatorial_rossby_radius_fv3(c, beta_lo)
    L_hi = equatorial_rossby_radius_fv3(c, beta_hi)
    assert jnp.all(L_hi < L_lo)


def test_lreq_composes_iter780():
    """Compose with iter-780 β at equator → atmospheric Kelvin scale."""
    beta = beta_plane_fv3(jnp.array([0.0]))  # equator
    c = jnp.array([30.0])
    L = equatorial_rossby_radius_fv3(c, beta)
    # β at equator ≈ 2.29e-11 → L ≈ 810 km
    assert 7.5e5 < float(L[0]) < 8.5e5


def test_lreq_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=783)
    n_x, n_y, km = 4, 5, 20
    c = jnp.asarray(rng.uniform(1.0, 100.0, size=(n_x, n_y, km)))
    beta = jnp.asarray(rng.uniform(5e-12, 2.5e-11, size=(n_x, n_y, km)))
    L = equatorial_rossby_radius_fv3(c, beta)
    assert L.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(L >= 0.0)
