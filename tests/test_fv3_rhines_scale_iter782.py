"""FV3_3D iter 782: rhines_scale_fv3 (L_β = √(U/β)).

Composes iter-780 ``beta_plane_fv3``.

Tests
-----

1. ``test_rhines_midlat_known``: U=20, β=2.29e-11 → L_β ≈ 935 km.
2. ``test_rhines_zero_u``: U=0 → L_β = 0.
3. ``test_rhines_monotonic_U``: ↑U → ↑L_β.
4. ``test_rhines_monotonic_beta``: ↑β → ↓L_β.
5. ``test_rhines_pole_floored``: β=0 → huge but finite via floor.
6. ``test_rhines_composes_iter780``: pipeline test with iter-780.
7. ``test_rhines_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    beta_plane_fv3,
    rhines_scale_fv3,
)


def test_rhines_midlat_known():
    """U=20 m/s, β=2.29e-11 (equator) → L_β = √(20/2.29e-11) ≈ 935 km."""
    U = jnp.array([20.0])
    beta = jnp.array([2.29e-11])
    L = rhines_scale_fv3(U, beta)
    expected = jnp.sqrt(20.0 / 2.29e-11)
    np.testing.assert_allclose(np.asarray(L), [expected], rtol=1e-12)
    # Sanity: ~935 km
    assert 9.0e5 < float(L[0]) < 1.0e6


def test_rhines_zero_u():
    """U=0 → L_β = 0."""
    U = jnp.array([0.0, 0.0, 0.0])
    beta = jnp.array([1e-11, 2e-11, 5e-11])
    L = rhines_scale_fv3(U, beta)
    np.testing.assert_allclose(np.asarray(L), [0.0, 0.0, 0.0], atol=1e-15)


def test_rhines_monotonic_U():
    """↑U → ↑L_β at fixed β."""
    beta = jnp.array([2e-11, 2e-11, 2e-11])
    U_lo = jnp.array([5.0, 10.0, 20.0])
    U_hi = U_lo + 5.0
    L_lo = rhines_scale_fv3(U_lo, beta)
    L_hi = rhines_scale_fv3(U_hi, beta)
    assert jnp.all(L_hi > L_lo)


def test_rhines_monotonic_beta():
    """↑β → ↓L_β at fixed U."""
    U = jnp.array([20.0, 20.0, 20.0])
    beta_lo = jnp.array([5e-12, 1e-11, 2e-11])
    beta_hi = beta_lo * 4.0  # 4× β → halved L_β
    L_lo = rhines_scale_fv3(U, beta_lo)
    L_hi = rhines_scale_fv3(U, beta_hi)
    assert jnp.all(L_hi < L_lo)


def test_rhines_pole_floored():
    """β=0 → L_β huge but finite via beta_floor."""
    U = jnp.array([20.0])
    beta = jnp.array([0.0])
    L = rhines_scale_fv3(U, beta, beta_floor=1e-15)
    assert jnp.all(jnp.isfinite(L))
    # √(20/1e-15) ≈ 1.4e8 m
    assert float(L[0]) > 1e7


def test_rhines_composes_iter780():
    """Full pipeline: (lat, U) → β → L_β."""
    lat = jnp.array([45.0])  # mid-lat
    beta = beta_plane_fv3(lat, units="deg")
    U = jnp.array([20.0])
    L = rhines_scale_fv3(U, beta)
    # β at 45° ≈ 2.29e-11·cos(45°) ≈ 1.62e-11
    # L_β = √(20/1.62e-11) ≈ 1.1 Mm
    assert 1.0e6 < float(L[0]) < 1.3e6


def test_rhines_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=782)
    n_x, n_y, km = 4, 5, 20
    U = jnp.asarray(rng.uniform(1.0, 50.0, size=(n_x, n_y, km)))
    beta = jnp.asarray(rng.uniform(5e-12, 2.5e-11, size=(n_x, n_y, km)))
    L = rhines_scale_fv3(U, beta)
    assert L.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(L >= 0.0)
