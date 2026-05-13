"""FV3_3D iter 868: light_response_curve_fv3.

Non-rectangular hyperbola photosynthesis A_n(PPFD).

Tests
-----

1. ``test_dark_zero``: PPFD=0 → A_n = 0.
2. ``test_overcast_band``: PPFD=200 → A_n ~7-10 μmol CO₂/m²/s.
3. ``test_full_sun_near_max``: PPFD=2000 → A_n ≈ 22-24.
4. ``test_asymptotic_a_max``: PPFD=1e5 → A_n ≈ A_max.
5. ``test_low_light_linear``: PPFD=1 → A_n ≈ α·PPFD.
6. ``test_monotone``: ↑PPFD → ↑A_n.
7. ``test_custom_a_max``: 2× A_max → larger A_n at high light.
8. ``test_chain_with_iter867``: R_s → PAR → PPFD → A_n.
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    light_response_curve_fv3,
    par_from_global_radiation_fv3,
)


def test_dark_zero():
    """PPFD=0 → A_n = 0."""
    a_n = light_response_curve_fv3(jnp.array([0.0]))
    np.testing.assert_allclose(np.asarray(a_n), [0.0], atol=1e-12)


def test_overcast_band():
    """PPFD=200 → A_n ≈ 7-10 μmol CO₂/m²/s (overcast forest)."""
    a_n = light_response_curve_fv3(jnp.array([200.0]))
    assert 7.0 < float(a_n[0]) < 11.0


def test_full_sun_near_max():
    """PPFD=2000 (clear noon) → A_n ≈ 22-24 (approaching A_max=25)."""
    a_n = light_response_curve_fv3(jnp.array([2000.0]))
    assert 22.0 < float(a_n[0]) < 25.0


def test_asymptotic_a_max():
    """PPFD → ∞ → A_n → A_max (saturation plateau)."""
    a_n = light_response_curve_fv3(jnp.array([1.0e5]))
    a_max_default = 25.0
    assert abs(float(a_n[0]) - a_max_default) < 0.1


def test_low_light_linear():
    """PPFD→0: A_n ≈ α·PPFD (light-limited linear regime)."""
    a_n = light_response_curve_fv3(jnp.array([1.0]))
    # α=0.06, PPFD=1 → A_n ≈ 0.06; non-rectangular hyperbola gives
    # slightly less due to curvature, but ≈ 0.06
    assert 0.05 < float(a_n[0]) < 0.07


def test_monotone():
    """↑PPFD → ↑A_n."""
    ppfd = jnp.array([10.0, 100.0, 500.0, 1500.0, 5000.0])
    a_n = light_response_curve_fv3(ppfd)
    diffs = jnp.diff(a_n)
    assert jnp.all(diffs > 0.0)


def test_custom_a_max():
    """2× A_max → larger A_n at high PPFD."""
    a1 = light_response_curve_fv3(
        jnp.array([2000.0]), a_max=jnp.array([25.0]),
    )
    a2 = light_response_curve_fv3(
        jnp.array([2000.0]), a_max=jnp.array([50.0]),
    )
    assert float(a2[0]) > float(a1[0])


def test_chain_with_iter867():
    """R_s → PAR → PPFD → A_n (full chain consistency)."""
    r_s = jnp.array([1000.0])             # tropical noon R_s
    par_w = par_from_global_radiation_fv3(r_s)  # 450 W/m²
    ppfd = par_w * 4.57                    # 2057 μmol/m²/s
    a_n = light_response_curve_fv3(ppfd)
    assert float(a_n[0]) > 20.0  # near A_max=25 at full tropical light


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=868)
    n_x, n_y = 6, 8
    ppfd = jnp.asarray(rng.uniform(0.0, 3000.0, size=(n_x, n_y)))
    a_n = light_response_curve_fv3(ppfd)
    assert a_n.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(a_n))
    assert jnp.all(a_n >= 0.0)
