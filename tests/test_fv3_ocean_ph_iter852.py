"""FV3_3D iter 852: ocean_ph_change_fv3.

ΔpH = − s · log10(pCO₂_new / pCO₂_ref).

Tests
-----

1. ``test_present_day_ar6``: 420 ppm vs 278 → ΔpH ≈ −0.12.
2. ``test_no_change_zero``: pCO₂_new = pCO₂_ref → ΔpH = 0.
3. ``test_paleoclimate_lgm``: 180 ppm vs 278 → ΔpH > 0 (LGM).
4. ``test_petm_extreme``: 1000 ppm vs 278 → ΔpH ≈ −0.37.
5. ``test_ssp37_2100``: 850 ppm → ΔpH ≈ −0.32.
6. ``test_chain_emission_to_ph``: emission → iter-847 → iter-852.
7. ``test_floor_no_nan``: pCO₂=0 → finite.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    airborne_fraction_co2_fv3,
    ocean_ph_change_fv3,
)


def test_present_day_ar6():
    """420 ppm vs 278 → ΔpH ≈ −0.12 (AR6 observed)."""
    delta = ocean_ph_change_fv3(jnp.array([420.0]))
    expected = -0.67 * float(jnp.log10(420.0 / 278.0))
    np.testing.assert_allclose(np.asarray(delta), [expected], rtol=1e-12)
    assert -0.13 < float(delta[0]) < -0.11


def test_no_change_zero():
    """pCO₂_new = pCO₂_ref → ΔpH = 0."""
    delta = ocean_ph_change_fv3(jnp.array([278.0]))
    np.testing.assert_allclose(np.asarray(delta), [0.0], atol=1e-13)


def test_paleoclimate_lgm():
    """LGM 180 ppm vs 278 → ΔpH > 0 (more basic ocean)."""
    delta = ocean_ph_change_fv3(jnp.array([180.0]))
    assert float(delta[0]) > 0.0
    # 0.67 × log10(278/180) ≈ 0.126
    assert 0.10 < float(delta[0]) < 0.15


def test_petm_extreme():
    """PETM 1000 ppm vs 278 → ΔpH ≈ −0.37."""
    delta = ocean_ph_change_fv3(jnp.array([1000.0]))
    # 0.67 × log10(1000/278) = 0.67 × 0.556 = 0.372
    assert -0.40 < float(delta[0]) < -0.34


def test_ssp37_2100():
    """SSP3-7.0 2100 ~850 ppm → ΔpH ≈ −0.32."""
    delta = ocean_ph_change_fv3(jnp.array([850.0]))
    # 0.67 × log10(850/278) = 0.67 × 0.485 = 0.325
    assert -0.35 < float(delta[0]) < -0.29


def test_chain_emission_to_ph():
    """Emission → iter-847 → iter-852: 1750→2020 cumulative ~141 ppm."""
    e_cum = jnp.array([2400.0])
    delta_co2 = airborne_fraction_co2_fv3(e_cum)  # ~141 ppm
    pco2_new = 278.0 + delta_co2                   # ~419 ppm
    delta_ph = ocean_ph_change_fv3(pco2_new)
    # Should match AR6 observed −0.12
    assert -0.13 < float(delta_ph[0]) < -0.11


def test_floor_no_nan():
    """pCO₂=0 → finite via floor (no NaN)."""
    delta = ocean_ph_change_fv3(jnp.array([0.0]))
    assert jnp.all(jnp.isfinite(delta))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=852)
    n_x, n_y = 6, 8
    pco2 = jnp.asarray(rng.uniform(150.0, 1500.0, size=(n_x, n_y)))
    delta = ocean_ph_change_fv3(pco2)
    assert delta.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(delta))
