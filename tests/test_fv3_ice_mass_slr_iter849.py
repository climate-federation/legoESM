"""FV3_3D iter 849: ice_mass_to_slr_fv3.

Δη = ΔM_Gt · 1e12 / (ρ_water · A_ocean)  [m].

Tests
-----

1. ``test_361gt_to_1mm``: 361 Gt → ~1 mm (AR6 conversion factor).
2. ``test_greenland_present_day``: 250 Gt/yr → ~0.69 mm/yr.
3. ``test_antarctica_present_day``: 150 Gt/yr → ~0.41 mm/yr.
4. ``test_greenland_full_melt``: 2.85e6 Gt → ~7.4 m.
5. ``test_zero_mass_zero``: ΔM=0 → Δη=0.
6. ``test_accretion_negative``: ΔM<0 → Δη<0 (mass gain lowers SLR).
7. ``test_chain_with_thermosteric``: combined cryosphere + thermal ≈ AR6.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    ice_mass_to_slr_fv3,
    thermosteric_sea_level_fv3,
)


def test_361gt_to_1mm():
    """361 Gt ice loss → 1 mm SLR (AR6 conversion factor)."""
    m = jnp.array([361.0])
    eta = ice_mass_to_slr_fv3(m)
    # 361e12 / (1000 × 3.61e14) = 361e12 / 3.61e17 = 1e-3 m = 1 mm
    np.testing.assert_allclose(np.asarray(eta), [1e-3], rtol=1e-6)


def test_greenland_present_day():
    """Greenland ~250 Gt/yr → ~0.69 mm/yr SLR (AR6 IMBIE-3)."""
    m = jnp.array([250.0])
    eta = ice_mass_to_slr_fv3(m)
    eta_mm = float(eta[0]) * 1000.0
    assert 0.65 < eta_mm < 0.72


def test_antarctica_present_day():
    """Antarctica ~150 Gt/yr → ~0.41 mm/yr."""
    m = jnp.array([150.0])
    eta = ice_mass_to_slr_fv3(m)
    eta_mm = float(eta[0]) * 1000.0
    assert 0.39 < eta_mm < 0.43


def test_greenland_full_melt():
    """Greenland total 2.85e6 Gt → ~7.4 m (AR6 §9.5.3)."""
    m = jnp.array([2.85e6])
    eta = ice_mass_to_slr_fv3(m)
    assert 7.0 < float(eta[0]) < 8.0


def test_zero_mass_zero():
    """ΔM=0 → Δη=0."""
    m = jnp.array([0.0])
    eta = ice_mass_to_slr_fv3(m)
    np.testing.assert_allclose(np.asarray(eta), [0.0], atol=1e-14)


def test_accretion_negative():
    """ΔM<0 (ice accretion) → Δη<0 (SLR drop)."""
    m = jnp.array([-100.0])
    eta = ice_mass_to_slr_fv3(m)
    assert float(eta[0]) < 0.0


def test_chain_with_thermosteric():
    """Combined: 730 Gt/yr cryosphere + 0.1 K/4km thermal → AR6-ish."""
    m_cryo = jnp.array([730.0])
    eta_cryo = ice_mass_to_slr_fv3(m_cryo)
    # ~2.0 mm
    eta_cryo_mm = float(eta_cryo[0]) * 1000.0

    dt = jnp.full((40,), 0.005)  # small 0.005 K/yr-ish warming
    h = jnp.full((40,), 100.0)
    eta_thermo = thermosteric_sea_level_fv3(dt, h)
    eta_thermo_mm = float(eta_thermo) * 1000.0

    total_mm = eta_cryo_mm + eta_thermo_mm
    # Plausible SLR sum: cryo ~2 mm + thermal contribution
    assert 1.0 < total_mm < 10.0


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=849)
    n_x, n_y = 6, 8
    m = jnp.asarray(rng.uniform(-500.0, 1000.0, size=(n_x, n_y)))
    eta = ice_mass_to_slr_fv3(m)
    assert eta.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(eta))
