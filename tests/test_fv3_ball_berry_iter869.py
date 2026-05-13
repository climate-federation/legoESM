"""FV3_3D iter 869: ball_berry_conductance_fv3.

g_s = m·(A_n·h_s/C_s) + b   (mol H₂O/m²/s).

Tests
-----

1. ``test_dark_minimum``: A_n=0 → g_s = b_min (cuticular floor).
2. ``test_forest_morning``: A_n=5, h_s=0.7, C_s=400 → g_s ≈ 0.089.
3. ``test_crop_midday``: A_n=20, h_s=0.5, C_s=400 → g_s ≈ 0.235.
4. ``test_tropical_noon``: A_n=25, h_s=0.8, C_s=400 → g_s ≈ 0.46.
5. ``test_low_rh_suppression``: ↓h_s → ↓g_s.
6. ``test_high_co2_suppression``: ↑C_s → ↓g_s.
7. ``test_c4_slope``: m=4.5 (C4) gives lower g_s than C3 m=9.
8. ``test_chain_with_iter868``: PPFD → A_n → g_s.
9. ``test_c_zero_floored``: C_s=0 → finite.
10. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    ball_berry_conductance_fv3,
    light_response_curve_fv3,
)


def test_dark_minimum():
    """A_n=0 → g_s = b_min (cuticular conductance floor)."""
    g_s = ball_berry_conductance_fv3(
        a_n=jnp.array([0.0]),
        h_s=jnp.array([0.6]),
        c_s=jnp.array([400.0]),
    )
    np.testing.assert_allclose(np.asarray(g_s), [0.01], rtol=1e-12)


def test_forest_morning():
    """A_n=5, h_s=0.7, C_s=400, m=9 → g_s = 9·5·0.7/400 + 0.01 = 0.0888."""
    g_s = ball_berry_conductance_fv3(
        a_n=jnp.array([5.0]),
        h_s=jnp.array([0.7]),
        c_s=jnp.array([400.0]),
    )
    expected = 9.0 * 5.0 * 0.7 / 400.0 + 0.01
    np.testing.assert_allclose(np.asarray(g_s), [expected], rtol=1e-12)


def test_crop_midday():
    """A_n=20, h_s=0.5, C_s=400 → g_s = 9·20·0.5/400 + 0.01 = 0.235."""
    g_s = ball_berry_conductance_fv3(
        a_n=jnp.array([20.0]),
        h_s=jnp.array([0.5]),
        c_s=jnp.array([400.0]),
    )
    np.testing.assert_allclose(np.asarray(g_s), [0.235], rtol=1e-12)


def test_tropical_noon():
    """A_n=25, h_s=0.8, C_s=400 → g_s = 9·25·0.8/400 + 0.01 = 0.46."""
    g_s = ball_berry_conductance_fv3(
        a_n=jnp.array([25.0]),
        h_s=jnp.array([0.8]),
        c_s=jnp.array([400.0]),
    )
    np.testing.assert_allclose(np.asarray(g_s), [0.46], rtol=1e-12)


def test_low_rh_suppression():
    """↓h_s → ↓g_s (humidity suppression)."""
    a = jnp.array([20.0])
    c = jnp.array([400.0])
    g_dry = ball_berry_conductance_fv3(a, jnp.array([0.2]), c)
    g_humid = ball_berry_conductance_fv3(a, jnp.array([0.8]), c)
    assert float(g_dry[0]) < float(g_humid[0])


def test_high_co2_suppression():
    """↑C_s → ↓g_s (CO₂ suppression — less aperture needed)."""
    a = jnp.array([20.0])
    h = jnp.array([0.6])
    g_low_co2 = ball_berry_conductance_fv3(a, h, jnp.array([200.0]))
    g_high_co2 = ball_berry_conductance_fv3(a, h, jnp.array([800.0]))
    assert float(g_high_co2[0]) < float(g_low_co2[0])


def test_c4_slope():
    """C4 (m=4.5) gives lower g_s than C3 (m=9.0)."""
    a = jnp.array([20.0])
    h = jnp.array([0.6])
    c = jnp.array([400.0])
    g_c3 = ball_berry_conductance_fv3(a, h, c, m_slope=9.0)
    g_c4 = ball_berry_conductance_fv3(a, h, c, m_slope=4.5)
    assert float(g_c4[0]) < float(g_c3[0])


def test_chain_with_iter868():
    """PPFD → iter-868 A_n → iter-869 g_s."""
    ppfd = jnp.array([2000.0])
    a_n = light_response_curve_fv3(ppfd)
    g_s = ball_berry_conductance_fv3(
        a_n=a_n,
        h_s=jnp.array([0.7]),
        c_s=jnp.array([400.0]),
    )
    assert jnp.all(jnp.isfinite(g_s))
    assert float(g_s[0]) > 0.01  # above floor


def test_c_zero_floored():
    """C_s=0 → finite via floor."""
    g_s = ball_berry_conductance_fv3(
        a_n=jnp.array([20.0]),
        h_s=jnp.array([0.5]),
        c_s=jnp.array([0.0]),
    )
    assert jnp.all(jnp.isfinite(g_s))


def test_shapes_finite():
    """3-D shapes preserved, finite, ≥ b_min."""
    rng = np.random.default_rng(seed=869)
    n_x, n_y = 6, 8
    a = jnp.asarray(rng.uniform(0.0, 30.0, size=(n_x, n_y)))
    h = jnp.asarray(rng.uniform(0.1, 0.95, size=(n_x, n_y)))
    c = jnp.asarray(rng.uniform(200.0, 800.0, size=(n_x, n_y)))
    g_s = ball_berry_conductance_fv3(a, h, c)
    assert g_s.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(g_s))
    assert jnp.all(g_s >= 0.01 - 1e-12)
