"""FV3_3D iter 724: compute_hybrid_pressure_fv3 port.

Faithful JAX port of FV3's hybrid σ-pressure setup pattern
(test_cases.F90:2941, 5274, 5370, 5479; fv_restart; IC ingestion).

Tests
-----

1. ``test_hybrid_pure_sigma_matches_ps``.
2. ``test_hybrid_pure_pressure_matches_ak``.
3. ``test_hybrid_blend_layer``.
4. ``test_hybrid_top_floor_avoids_log_zero``.
5. ``test_hybrid_delp_sums_to_ps``.
6. ``test_hybrid_shapes_3d``.
7. ``test_hybrid_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import compute_hybrid_pressure_fv3


def test_hybrid_pure_sigma_matches_ps():
    """Pure σ coordinate (ak=0, bk monotone) → pe = bk · ps."""
    km = 10
    ak = jnp.zeros((km + 1,))
    bk = jnp.linspace(0.0, 1.0, km + 1)
    ps = jnp.asarray(1.0e5)
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    expected_pe = bk * 1.0e5
    # Top is floored at 1e-6, others should match
    assert jnp.allclose(pe[1:], expected_pe[1:], atol=1e-8)


def test_hybrid_pure_pressure_matches_ak():
    """Pure pressure coordinate (bk=0, ak monotone) → pe = ak."""
    km = 10
    ak = jnp.linspace(100.0, 1.0e5, km + 1)
    bk = jnp.zeros((km + 1,))
    ps = jnp.asarray(1.2e5)  # arbitrary, doesn't enter
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    assert jnp.allclose(pe, ak, atol=1e-10)


def test_hybrid_blend_layer():
    """Standard FV3 hybrid: ak top, bk·ps bottom, blend in between."""
    km = 5
    ak = jnp.array([1.0, 10.0, 100.0, 500.0, 1000.0, 0.0])
    bk = jnp.array([0.0, 0.0, 0.0, 0.1, 0.5, 1.0])
    ps = jnp.asarray(1.0e5)
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    expected_pe = ak + 1.0e5 * bk
    assert jnp.allclose(pe, expected_pe, atol=1e-8)


def test_hybrid_top_floor_avoids_log_zero():
    """ak[0]=0, bk[0]=0 → pe[0] floored to 1e-6, peln[0] finite."""
    km = 5
    ak = jnp.array([0.0, 10.0, 100.0, 500.0, 1000.0, 0.0])
    bk = jnp.array([0.0, 0.0, 0.0, 0.1, 0.5, 1.0])
    ps = jnp.asarray(1.0e5)
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    assert float(pe[0]) > 0.0
    assert jnp.all(jnp.isfinite(peln))


def test_hybrid_delp_sums_to_ps():
    """Σ delp = ps - pe_top (mass conservation)."""
    km = 10
    ak = jnp.zeros((km + 1,))
    ak = ak.at[0].set(100.0)
    bk = jnp.linspace(0.0, 1.0, km + 1)
    ps_value = 1.0e5
    ps = jnp.asarray(ps_value)
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    total_delp = float(jnp.sum(delp))
    expected = ps_value - 100.0
    assert abs(total_delp - expected) / expected < 1e-10


def test_hybrid_shapes_3d():
    """3-D ps (n_x, n_y) → (n_x, n_y, km) delp, (n_x, n_y, km+1) pe/peln."""
    rng = np.random.default_rng(seed=724)
    n_x, n_y, km = 4, 5, 20
    ak = jnp.linspace(1.0, 0.0, km + 1) * 100.0
    bk = jnp.linspace(0.0, 1.0, km + 1)
    ps = jnp.asarray(rng.uniform(9.0e4, 1.05e5, size=(n_x, n_y)))
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    assert delp.shape == (n_x, n_y, km)
    assert pe.shape == (n_x, n_y, km + 1)
    assert peln.shape == (n_x, n_y, km + 1)


def test_hybrid_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=725)
    km = 30
    ak = jnp.linspace(100.0, 0.0, km + 1)
    bk = jnp.linspace(0.0, 1.0, km + 1)
    ps = jnp.asarray(rng.uniform(8.0e4, 1.05e5, size=(4, 4)))
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    assert jnp.all(jnp.isfinite(delp))
    assert jnp.all(jnp.isfinite(pe))
    assert jnp.all(jnp.isfinite(peln))
