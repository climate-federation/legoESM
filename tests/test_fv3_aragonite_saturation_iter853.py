"""FV3_3D iter 853: aragonite_saturation_state_fv3.

Ω_arag = Ω_ref · (pCO₂_ref / pCO₂_new)^γ.

Tests
-----

1. ``test_pre_industrial``: 278 ppm → Ω=3.5 (reference value).
2. ``test_present_day_decline``: 420 ppm → Ω ≈ 2.47.
3. ``test_ssp37_2100``: 850 ppm → Ω ≈ 1.41.
4. ``test_undersaturation_crossing``: 2100+ ppm → Ω < 1.
5. ``test_lgm_increase``: 180 ppm → Ω > Ω_ref (calcification-favorable).
6. ``test_polar_low_omega``: Ω_ref=1.5 (polar) crosses Ω=1 sooner.
7. ``test_chain_with_ph``: same pCO₂ → both ΔpH and Ω consistent.
8. ``test_floor_no_nan``.
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    aragonite_saturation_state_fv3,
    ocean_ph_change_fv3,
)


def test_pre_industrial():
    """278 ppm → Ω = 3.5 (reference)."""
    omega = aragonite_saturation_state_fv3(jnp.array([278.0]))
    np.testing.assert_allclose(np.asarray(omega), [3.5], rtol=1e-12)


def test_present_day_decline():
    """420 ppm → Ω ≈ 2.47 (AR6 surface)."""
    omega = aragonite_saturation_state_fv3(jnp.array([420.0]))
    # 3.5 × (278/420)^0.85 = 3.5 × 0.7060 = 2.471
    assert 2.4 < float(omega[0]) < 2.55


def test_ssp37_2100():
    """SSP3-7.0 ~850 ppm → Ω ≈ 1.41 (still saturated but marginal)."""
    omega = aragonite_saturation_state_fv3(jnp.array([850.0]))
    # 3.5 × (278/850)^0.85 = 3.5 × 0.404 = 1.41
    assert 1.30 < float(omega[0]) < 1.50


def test_undersaturation_crossing():
    """pCO₂ → 2200+ ppm tropical Ω < 1 (calcification fails)."""
    omega = aragonite_saturation_state_fv3(jnp.array([2200.0]))
    assert float(omega[0]) < 1.0


def test_lgm_increase():
    """LGM 180 ppm → Ω > 3.5 (more calcification-favorable)."""
    omega = aragonite_saturation_state_fv3(jnp.array([180.0]))
    assert float(omega[0]) > 3.5


def test_polar_low_omega():
    """Polar Ω_ref=1.5 → undersaturation crosses ~600-700 ppm."""
    omega = aragonite_saturation_state_fv3(
        jnp.array([700.0]),
        omega_arag_ref=jnp.array([1.5]),
    )
    assert float(omega[0]) < 1.0


def test_chain_with_ph():
    """Same pCO₂ → ΔpH (iter-852) and Ω (iter-853) consistent."""
    pco2 = jnp.array([420.0])
    delta_ph = ocean_ph_change_fv3(pco2)
    omega = aragonite_saturation_state_fv3(pco2)
    # Both indicate acidification: ΔpH < 0 AND Ω < Ω_ref
    assert float(delta_ph[0]) < 0.0
    assert float(omega[0]) < 3.5


def test_floor_no_nan():
    """pCO₂=0 → finite via floor."""
    omega = aragonite_saturation_state_fv3(jnp.array([0.0]))
    assert jnp.all(jnp.isfinite(omega))


def test_shapes_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=853)
    n_x, n_y = 6, 8
    pco2 = jnp.asarray(rng.uniform(150.0, 2500.0, size=(n_x, n_y)))
    omega = aragonite_saturation_state_fv3(pco2)
    assert omega.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(omega))
    assert jnp.all(omega > 0.0)
