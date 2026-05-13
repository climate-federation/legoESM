"""FV3_3D iter 850: halosteric_sea_level_fv3.

Δη_halo = −Σ_k β_S·ΔS_k·H_k  [m].

Tests
-----

1. ``test_north_atlantic_freshening``: ΔS=−0.5/1000 m → +0.38 m.
2. ``test_tropical_pacific_salinification``: ΔS=+0.3/500 m → −0.114 m.
3. ``test_no_salinity_change_zero``: ΔS=0 → 0.
4. ``test_global_mean_freshwater_conservation``: equal +/- → ~0.
5. ``test_opposite_sign_to_thermosteric``: freshening raises Δη.
6. ``test_slr_triplet_full_budget``: 848+849+850 chain plausible.
7. ``test_beta_s_scaling``: 2× β_S → 2× |Δη|.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    halosteric_sea_level_fv3,
    ice_mass_to_slr_fv3,
    thermosteric_sea_level_fv3,
)


def test_north_atlantic_freshening():
    """ΔS=−0.5 psu over 1000 m → Δη_halo ≈ +0.38 m."""
    n = 10
    ds = jnp.full((n,), -0.5)
    h = jnp.full((n,), 100.0)  # 100 m × 10 = 1000 m
    eta = halosteric_sea_level_fv3(ds, h)
    expected = -7.6e-4 * (-0.5) * 1000.0
    np.testing.assert_allclose(np.asarray(eta), expected, rtol=1e-12)
    assert 0.35 < float(eta) < 0.42


def test_tropical_pacific_salinification():
    """ΔS=+0.3 psu over 500 m → Δη_halo ≈ −0.114 m."""
    n = 5
    ds = jnp.full((n,), 0.3)
    h = jnp.full((n,), 100.0)
    eta = halosteric_sea_level_fv3(ds, h)
    assert -0.13 < float(eta) < -0.10


def test_no_salinity_change_zero():
    """ΔS=0 → Δη=0."""
    ds = jnp.zeros(20)
    h = jnp.full((20,), 200.0)
    eta = halosteric_sea_level_fv3(ds, h)
    np.testing.assert_allclose(np.asarray(eta), 0.0, atol=1e-14)


def test_global_mean_freshwater_conservation():
    """Equal +/− perturbations sum to ≈0 (mass conservation)."""
    n = 10
    ds = jnp.array([0.5, -0.5] * 5)  # zero mean
    h = jnp.full((n,), 100.0)
    eta = halosteric_sea_level_fv3(ds, h)
    np.testing.assert_allclose(np.asarray(eta), 0.0, atol=1e-12)


def test_opposite_sign_to_thermosteric():
    """Same |perturbation|: freshening (ΔS<0) and warming (ΔT>0)
    both raise Δη (positive sign)."""
    n = 10
    h = jnp.full((n,), 100.0)
    ds = jnp.full((n,), -0.1)   # freshening
    dt = jnp.full((n,), 0.1)    # warming
    eta_halo = halosteric_sea_level_fv3(ds, h)
    eta_thermo = thermosteric_sea_level_fv3(dt, h)
    assert float(eta_halo) > 0.0
    assert float(eta_thermo) > 0.0


def test_slr_triplet_full_budget():
    """Combined iter-848 + iter-849 + iter-850 → plausible SLR budget."""
    n = 40
    h = jnp.full((n,), 100.0)  # 4000 m
    dt = jnp.full((n,), 0.005)
    eta_thermo = thermosteric_sea_level_fv3(dt, h)
    eta_cryo = ice_mass_to_slr_fv3(jnp.array(730.0))
    ds = jnp.zeros(n)   # global mean halo ~ 0
    eta_halo = halosteric_sea_level_fv3(ds, h)
    total = float(eta_thermo) + float(eta_cryo) + float(eta_halo)
    total_mm = total * 1000.0
    assert 1.0 < total_mm < 10.0


def test_beta_s_scaling():
    """2× β_S → 2× |Δη|."""
    n = 10
    ds = jnp.full((n,), 0.1)
    h = jnp.full((n,), 100.0)
    eta1 = halosteric_sea_level_fv3(ds, h, beta_s=7.6e-4)
    eta2 = halosteric_sea_level_fv3(ds, h, beta_s=15.2e-4)
    np.testing.assert_allclose(np.asarray(eta2), 2.0 * np.asarray(eta1),
                                rtol=1e-12)


def test_shapes_finite():
    """3-D batched: (lat, lon, depth) → (lat, lon)."""
    rng = np.random.default_rng(seed=850)
    n_lat, n_lon, n_z = 4, 5, 30
    ds = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_lat, n_lon, n_z)))
    h = jnp.asarray(rng.uniform(50.0, 200.0, size=(n_lat, n_lon, n_z)))
    eta = halosteric_sea_level_fv3(ds, h)
    assert eta.shape == (n_lat, n_lon)
    assert jnp.all(jnp.isfinite(eta))
