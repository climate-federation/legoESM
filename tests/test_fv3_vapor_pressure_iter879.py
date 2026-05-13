"""FV3_3D iter 879: vapor_pressure_from_rh_fv3.

e_a = RH · e_sat(T)  [Pa].

Tests
-----

1. ``test_rh_zero_zero``: RH=0 → e_a=0.
2. ``test_rh_one_full_esat``: RH=1 → e_a = e_sat.
3. ``test_tropical_humid``: T=30°C, RH=0.7 → e_a ≈ 2970 Pa.
4. ``test_monotone_in_rh``: ↑RH → ↑e_a.
5. ``test_monotone_in_t``: ↑T (fixed RH) → ↑e_a (CC).
6. ``test_identity_with_vpd``: VPD + e_a = e_sat.
7. ``test_chain_with_brunt_lw``: e_a → iter-877 LW_dn.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import thermo
from legoesm.grids.cubed_sphere import (
    clear_sky_longwave_brunt_fv3,
    vapor_pressure_from_rh_fv3,
    vpd_from_t_rh_fv3,
)


def test_rh_zero_zero():
    """RH=0 → e_a=0."""
    e_a = vapor_pressure_from_rh_fv3(
        t=jnp.array([293.15]),
        rh=jnp.array([0.0]),
    )
    np.testing.assert_allclose(np.asarray(e_a), [0.0], atol=1e-14)


def test_rh_one_full_esat():
    """RH=1 → e_a = e_sat(T)."""
    t = jnp.array([293.15])
    e_a = vapor_pressure_from_rh_fv3(t, rh=jnp.array([1.0]))
    e_sat = thermo.saturation_vapor_pressure(t)
    np.testing.assert_allclose(np.asarray(e_a), np.asarray(e_sat),
                                rtol=1e-12)


def test_tropical_humid():
    """T=30°C, RH=0.7 → e_a ≈ 2970 Pa."""
    e_a = vapor_pressure_from_rh_fv3(
        t=jnp.array([303.15]),
        rh=jnp.array([0.7]),
    )
    # e_sat(30°C) ≈ 4243 Pa, e_a = 0.7·4243 ≈ 2970
    assert 2700.0 < float(e_a[0]) < 3200.0


def test_monotone_in_rh():
    """↑RH → ↑e_a at fixed T."""
    t = jnp.full((4,), 293.15)
    rh = jnp.array([0.1, 0.3, 0.5, 0.9])
    e_a = vapor_pressure_from_rh_fv3(t, rh)
    diffs = jnp.diff(e_a)
    assert jnp.all(diffs > 0.0)


def test_monotone_in_t():
    """↑T at fixed RH → ↑e_a (CC scaling via e_sat)."""
    t = jnp.array([283.15, 293.15, 303.15, 313.15])
    rh = jnp.full((4,), 0.5)
    e_a = vapor_pressure_from_rh_fv3(t, rh)
    diffs = jnp.diff(e_a)
    assert jnp.all(diffs > 0.0)


def test_identity_with_vpd():
    """VPD + e_a = e_sat (definitional identity)."""
    t = jnp.array([293.15])
    rh = jnp.array([0.5])
    e_a = vapor_pressure_from_rh_fv3(t, rh)
    vpd = vpd_from_t_rh_fv3(t, rh)
    e_sat = thermo.saturation_vapor_pressure(t)
    np.testing.assert_allclose(
        np.asarray(e_a + vpd), np.asarray(e_sat), rtol=1e-12,
    )


def test_chain_with_brunt_lw():
    """e_a from iter-879 → iter-877 Brunt LW_dn."""
    t = jnp.array([293.15])
    rh = jnp.array([0.6])
    e_a = vapor_pressure_from_rh_fv3(t, rh)
    lw_dn = clear_sky_longwave_brunt_fv3(t, e_a)
    assert jnp.all(jnp.isfinite(lw_dn))
    assert float(lw_dn[0]) > 200.0  # plausible mid-lat LW_dn


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=879)
    n_x, n_y = 6, 8
    t = jnp.asarray(rng.uniform(240.0, 320.0, size=(n_x, n_y)))
    rh = jnp.asarray(rng.uniform(0.05, 1.0, size=(n_x, n_y)))
    e_a = vapor_pressure_from_rh_fv3(t, rh)
    assert e_a.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(e_a))
    assert jnp.all(e_a >= 0.0)
