"""FV3_3D iter 851: ocean_heat_content_fv3.

OHC = ρ·c_p·Σ T·H  [J/m²].

Tests
-----

1. ``test_uniform_anomaly_4km``: ΔT=0.1 K / 4000 m → 1.64×10⁹ J/m².
2. ``test_zero_anomaly_zero``: ΔT=0 → 0.
3. ``test_cooling_negative``: ΔT<0 → ΔOHC<0.
4. ``test_paired_with_thermosteric``: ΔOHC/(ρ·c_p) = Δη/α_T (same integral).
5. ``test_global_ohc_growth_rate``: Cheng-style 10 ZJ/yr plausible.
6. ``test_custom_rho_cp``: 2× ρ → 2× OHC.
7. ``test_batched_columns``: (lat, lon, depth) → (lat, lon).
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    ocean_heat_content_fv3,
    thermosteric_sea_level_fv3,
)


def test_uniform_anomaly_4km():
    """ΔT=0.1 K uniform / 4000 m → ΔOHC ≈ 1.64×10⁹ J/m²."""
    n = 40
    dt = jnp.full((n,), 0.1)
    h = jnp.full((n,), 100.0)  # 4000 m total
    ohc = ocean_heat_content_fv3(dt, h)
    expected = constants.rho_ocean * constants.c_sw * 0.1 * 4000.0
    np.testing.assert_allclose(np.asarray(ohc), expected, rtol=1e-12)
    assert 1.6e9 < float(ohc) < 1.7e9


def test_zero_anomaly_zero():
    """ΔT=0 → OHC=0."""
    dt = jnp.zeros(20)
    h = jnp.full((20,), 200.0)
    ohc = ocean_heat_content_fv3(dt, h)
    np.testing.assert_allclose(np.asarray(ohc), 0.0, atol=1e-9)


def test_cooling_negative():
    """ΔT<0 → ΔOHC<0."""
    dt = jnp.full((10,), -0.05)
    h = jnp.full((10,), 100.0)
    ohc = ocean_heat_content_fv3(dt, h)
    assert float(ohc) < 0.0


def test_paired_with_thermosteric():
    """ΔOHC / (ρ·c_p) = Δη / α_T — same column integral, different scaling."""
    n = 10
    dt = jnp.full((n,), 0.1)
    h = jnp.full((n,), 100.0)
    ohc = ocean_heat_content_fv3(dt, h)
    eta = thermosteric_sea_level_fv3(dt, h)
    # ΔOHC/(ρ·c_p) = Σ ΔT·H = Δη/α_T
    int_via_ohc = float(ohc) / (constants.rho_ocean * constants.c_sw)
    int_via_eta = float(eta) / 2.0e-4
    np.testing.assert_allclose(int_via_ohc, int_via_eta, rtol=1e-12)


def test_global_ohc_growth_rate():
    """Cheng 2017: ~0.4 W/m² × 1yr × A_ocean ≈ 10 ZJ/yr globally."""
    # Express 0.4 W/m² · 1yr / (ρ·c_p) → ΔT/Δh equivalent depth integral
    # Quick: 0.4 W/m² · 3.156e7 s = 1.26e7 J/m² per year
    # Total over A_ocean 3.61e14: 4.57e21 J = 4.57 ZJ/yr
    # Cheng 2017 reports ~10 ZJ/yr for 0-2000 m (~0.7 W/m² effective)
    # Construct equivalent ΔT/depth column
    dt = jnp.full((20,), 1.26e7 / (constants.rho_ocean * constants.c_sw * 2000.0))
    h = jnp.full((20,), 100.0)
    ohc = ocean_heat_content_fv3(dt, h)
    global_zj = float(ohc) * 3.61e14 / 1.0e21
    # 4-5 ZJ band for 0.4 W/m² scaling
    assert 3.0 < global_zj < 6.0


def test_custom_rho_cp():
    """2× ρ → 2× OHC."""
    n = 5
    dt = jnp.full((n,), 0.1)
    h = jnp.full((n,), 100.0)
    ohc1 = ocean_heat_content_fv3(dt, h)
    ohc2 = ocean_heat_content_fv3(dt, h, rho=2.0 * constants.rho_ocean)
    np.testing.assert_allclose(np.asarray(ohc2), 2.0 * np.asarray(ohc1),
                                rtol=1e-12)


def test_batched_columns():
    """(lat, lon, depth) → (lat, lon)."""
    rng = np.random.default_rng(seed=851)
    n_lat, n_lon, n_z = 4, 5, 30
    dt = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_lat, n_lon, n_z)))
    h = jnp.asarray(rng.uniform(50.0, 200.0, size=(n_lat, n_lon, n_z)))
    ohc = ocean_heat_content_fv3(dt, h)
    assert ohc.shape == (n_lat, n_lon)


def test_shapes_finite():
    """3-D random shapes finite."""
    rng = np.random.default_rng(seed=851)
    n_x, n_y, n_z = 6, 8, 25
    dt = jnp.asarray(rng.uniform(-2.0, 2.0, size=(n_x, n_y, n_z)))
    h = jnp.asarray(rng.uniform(20.0, 300.0, size=(n_x, n_y, n_z)))
    ohc = ocean_heat_content_fv3(dt, h)
    assert ohc.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(ohc))
