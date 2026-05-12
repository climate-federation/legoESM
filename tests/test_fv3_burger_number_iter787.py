"""FV3_3D iter 787: burger_number_fv3 (B = (L_R/L)²).

Composes iter-781 ``rossby_radius_fv3``.

Tests
-----

1. ``test_b_qg_classical``: L_R=L=1000 km → B=1 (classical QG).
2. ``test_b_stratified``: L=100 km, L_R=1000 km → B=100.
3. ``test_b_barotropic``: L=10000 km, L_R=1000 km → B=0.01.
4. ``test_b_zero_L_floor``: L=0 → huge but finite.
5. ``test_b_monotonic_inputs``: ↑L_R → ↑B; ↑L → ↓B.
6. ``test_b_composes_iter781``: (N,H,f,L) → L_R → B.
7. ``test_b_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    burger_number_fv3,
    coriolis_parameter_fv3,
    rossby_radius_fv3,
)


def test_b_qg_classical():
    """L_R = L = 1000 km → B = 1.0 (classical QG)."""
    L_R = jnp.array([1.0e6])
    L = jnp.array([1.0e6])
    B = burger_number_fv3(L_R, L)
    np.testing.assert_allclose(np.asarray(B), [1.0], rtol=1e-12)


def test_b_stratified():
    """L_R=1000 km, L=100 km → B=(10)² = 100 (fully stratified)."""
    L_R = jnp.array([1.0e6])
    L = jnp.array([1.0e5])
    B = burger_number_fv3(L_R, L)
    np.testing.assert_allclose(np.asarray(B), [100.0], rtol=1e-12)
    assert float(B[0]) > 10.0  # stratified regime


def test_b_barotropic():
    """L_R=1000 km, L=10000 km → B=(0.1)² = 0.01 (barotropic)."""
    L_R = jnp.array([1.0e6])
    L = jnp.array([1.0e7])
    B = burger_number_fv3(L_R, L)
    np.testing.assert_allclose(np.asarray(B), [0.01], rtol=1e-12)
    assert float(B[0]) < 0.1  # barotropic regime


def test_b_zero_L_floor():
    """L=0 → B huge but finite via L_floor."""
    L_R = jnp.array([1.0e6])
    L = jnp.array([0.0])
    B = burger_number_fv3(L_R, L, L_floor=1e-6)
    assert jnp.all(jnp.isfinite(B))
    # (1e6/1e-6)² = 1e24
    assert float(B[0]) > 1e20


def test_b_monotonic_inputs():
    """↑L_R → ↑B; ↑L → ↓B."""
    base_LR = jnp.array([1.0e6, 1.0e6, 1.0e6])
    base_L = jnp.array([1.0e6, 1.0e6, 1.0e6])
    B_base = burger_number_fv3(base_LR, base_L)

    # ↑L_R → ↑B
    B_LR_hi = burger_number_fv3(base_LR * 2.0, base_L)
    assert jnp.all(B_LR_hi > B_base)

    # ↑L → ↓B
    B_L_hi = burger_number_fv3(base_LR, base_L * 2.0)
    assert jnp.all(B_L_hi < B_base)


def test_b_composes_iter781():
    """Pipeline (N, H, f, L) → L_R → B."""
    N = jnp.array([0.01])
    H = jnp.array([10_000.0])
    f = coriolis_parameter_fv3(jnp.array([30.0]), units="deg")  # |f|=Ω
    L = jnp.array([1.0e6])  # 1000 km
    L_R = rossby_radius_fv3(N, f, H)  # ≈ 1.37 Mm
    B = burger_number_fv3(L_R, L)
    # B = (1.37e6/1e6)² ≈ 1.88
    assert 1.0 < float(B[0]) < 3.0  # near-QG regime


def test_b_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=787)
    n_x, n_y, km = 4, 5, 20
    L_R = jnp.asarray(rng.uniform(1.0e4, 5.0e6, size=(n_x, n_y, km)))
    L = jnp.asarray(rng.uniform(1.0e4, 1.0e7, size=(n_x, n_y, km)))
    B = burger_number_fv3(L_R, L)
    assert B.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(B))
    assert jnp.all(B >= 0.0)
