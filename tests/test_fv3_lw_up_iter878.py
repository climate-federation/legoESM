"""FV3_3D iter 878: gray_body_lw_emission_fv3.

LW_up = ε·σ·T_s⁴ + (1−ε)·LW_down  [W/m²].

Tests
-----

1. ``test_blackbody_emission``: ε=1, LW_dn=0 → LW_up = σ·T⁴.
2. ``test_full_blackbody_t_288``: T=288 K, ε=1 → σT⁴ ≈ 390 W/m².
3. ``test_reflective_lw``: ε=0 → LW_up = LW_down (full reflection).
4. ``test_mixed_emission_reflection``: ε=0.97 → mostly emission.
5. ``test_monotone_in_t``: ↑T → ↑LW_up (T⁴).
6. ``test_chain_full_radiation``: LW_up → iter-876 R_n end-to-end.
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
    gray_body_lw_emission_fv3,
    net_radiation_fv3,
)


def test_blackbody_emission():
    """ε=1, LW_dn=0 → LW_up = σ·T⁴ pure blackbody."""
    t = 300.0
    lw_up = gray_body_lw_emission_fv3(
        t_surf=jnp.array([t]),
        emissivity=jnp.array([1.0]),
        lw_down=jnp.array([0.0]),
    )
    expected = constants.sigma_sb * t ** 4
    np.testing.assert_allclose(np.asarray(lw_up), [expected], rtol=1e-12)


def test_full_blackbody_t_288():
    """T=288 K (Earth mean), ε=1 → LW_up ≈ 390 W/m² (canonical σT⁴)."""
    lw_up = gray_body_lw_emission_fv3(
        t_surf=jnp.array([288.0]),
        emissivity=jnp.array([1.0]),
    )
    # σ·288⁴ = 5.67e-8 × 6.88e9 ≈ 390 W/m²
    assert 380.0 < float(lw_up[0]) < 400.0


def test_reflective_lw():
    """ε=0 (perfect mirror) → LW_up = LW_down (no emission)."""
    lw_dn = 400.0
    lw_up = gray_body_lw_emission_fv3(
        t_surf=jnp.array([300.0]),
        emissivity=jnp.array([0.0]),
        lw_down=jnp.array([lw_dn]),
    )
    np.testing.assert_allclose(np.asarray(lw_up), [lw_dn], rtol=1e-12)


def test_mixed_emission_reflection():
    """ε=0.97, T=300, LW_dn=400 → mostly emission with small refl."""
    lw_up = gray_body_lw_emission_fv3(
        t_surf=jnp.array([300.0]),
        emissivity=jnp.array([0.97]),
        lw_down=jnp.array([400.0]),
    )
    # ε·σ·T⁴ = 0.97 × 460 = 445; reflected = 0.03·400 = 12; total ≈ 457
    assert 440.0 < float(lw_up[0]) < 470.0


def test_monotone_in_t():
    """↑T → ↑LW_up (σT⁴)."""
    t = jnp.array([253.15, 273.15, 293.15, 313.15])
    lw_up = gray_body_lw_emission_fv3(t)
    diffs = jnp.diff(lw_up)
    assert jnp.all(diffs > 0.0)


def test_chain_full_radiation():
    """Full radiation chain: iter-877 LW_dn → iter-878 LW_up →
    iter-876 R_n.  Closes everything."""
    t_air = jnp.array([293.15])
    e_a = jnp.array([1500.0])
    lw_dn = clear_sky_longwave_brunt_fv3(t_air, e_a)
    lw_up = gray_body_lw_emission_fv3(
        t_surf=jnp.array([295.15]),
        emissivity=jnp.array([0.97]),
        lw_down=lw_dn,
    )
    r_n = net_radiation_fv3(
        sw_down=jnp.array([800.0]),
        albedo=jnp.array([0.23]),
        lw_down=lw_dn,
        lw_up=lw_up,
    )
    assert jnp.all(jnp.isfinite(r_n))
    assert float(r_n[0]) > 0.0  # daytime positive


def test_shapes_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=878)
    n_x, n_y = 6, 8
    t = jnp.asarray(rng.uniform(240.0, 320.0, size=(n_x, n_y)))
    eps = jnp.asarray(rng.uniform(0.85, 0.99, size=(n_x, n_y)))
    lw_dn = jnp.asarray(rng.uniform(150.0, 450.0, size=(n_x, n_y)))
    lw_up = gray_body_lw_emission_fv3(t, eps, lw_dn)
    assert lw_up.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(lw_up))
    assert jnp.all(lw_up > 0.0)
