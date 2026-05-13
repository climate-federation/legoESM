"""FV3_3D iter 877: clear_sky_longwave_brunt_fv3.

LW_dn = (a + b·√e_a_hPa) · σ · T⁴   [W/m²].

Tests
-----

1. ``test_tropical_humid``: T=303, e_a=2700 Pa → LW_dn ≈ 400 W/m².
2. ``test_polar_dry``: T=243, e_a=30 → low LW_dn.
3. ``test_zero_humidity``: e_a=0 → ε_a=0.605 (intercept only).
4. ``test_monotone_in_t``: ↑T → ↑LW_dn (σT⁴ scaling).
5. ``test_monotone_in_humidity``: ↑e_a → ↑LW_dn.
6. ``test_chain_with_net_radiation``: LW_dn → iter-876 R_n.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    clear_sky_longwave_brunt_fv3,
    net_radiation_fv3,
)


def test_tropical_humid():
    """T=303 K, e_a=2700 Pa → LW_dn ≈ 400 W/m²."""
    lw = clear_sky_longwave_brunt_fv3(
        t=jnp.array([303.15]),
        e_a_pa=jnp.array([2700.0]),
    )
    # ε_a = 0.605 + 0.048·√27 ≈ 0.605 + 0.249 = 0.855
    # LW = 0.855 × σ × 303.15⁴ ≈ 0.855 × 478 ≈ 408
    assert 380.0 < float(lw[0]) < 430.0


def test_polar_dry():
    """T=243 K, e_a=30 Pa (polar) → LW_dn low band."""
    lw = clear_sky_longwave_brunt_fv3(
        t=jnp.array([243.15]),
        e_a_pa=jnp.array([30.0]),
    )
    # ε_a ≈ 0.605 + 0.048·√0.3 ≈ 0.631
    # LW = 0.631 × σ × 243.15⁴ ≈ 0.631 × 198 ≈ 125
    assert 100.0 < float(lw[0]) < 150.0


def test_zero_humidity():
    """e_a=0 → ε_a = a_Brunt = 0.605 (intercept only)."""
    t_k = 288.15
    lw = clear_sky_longwave_brunt_fv3(
        t=jnp.array([t_k]),
        e_a_pa=jnp.array([0.0]),
    )
    expected = 0.605 * constants.sigma_sb * t_k ** 4
    np.testing.assert_allclose(np.asarray(lw), [expected], rtol=1e-12)


def test_monotone_in_t():
    """↑T → ↑LW_dn (T⁴ scaling dominant)."""
    t = jnp.array([253.15, 273.15, 293.15, 313.15])
    e_a = jnp.full((4,), 1000.0)
    lw = clear_sky_longwave_brunt_fv3(t, e_a)
    diffs = jnp.diff(lw)
    assert jnp.all(diffs > 0.0)


def test_monotone_in_humidity():
    """↑e_a → ↑LW_dn (more atmospheric emission)."""
    t = jnp.full((4,), 293.15)
    e_a = jnp.array([100.0, 500.0, 1500.0, 3000.0])
    lw = clear_sky_longwave_brunt_fv3(t, e_a)
    diffs = jnp.diff(lw)
    assert jnp.all(diffs > 0.0)


def test_chain_with_net_radiation():
    """LW_dn from Brunt → iter-876 R_n."""
    lw_dn = clear_sky_longwave_brunt_fv3(
        t=jnp.array([293.15]),
        e_a_pa=jnp.array([1500.0]),
    )
    r_n = net_radiation_fv3(
        sw_down=jnp.array([800.0]),
        albedo=jnp.array([0.23]),
        lw_down=lw_dn,
        lw_up=jnp.array([430.0]),
    )
    assert jnp.all(jnp.isfinite(r_n))


def test_shapes_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=877)
    n_x, n_y = 6, 8
    t = jnp.asarray(rng.uniform(240.0, 320.0, size=(n_x, n_y)))
    e_a = jnp.asarray(rng.uniform(50.0, 3500.0, size=(n_x, n_y)))
    lw = clear_sky_longwave_brunt_fv3(t, e_a)
    assert lw.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(lw))
    assert jnp.all(lw > 0.0)
