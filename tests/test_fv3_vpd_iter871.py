"""FV3_3D iter 871: vpd_from_t_rh_fv3.

VPD = e_sat(T) · (1 − RH)  [Pa].

Tests
-----

1. ``test_saturated_zero``: RH=1.0 → VPD=0.
2. ``test_zero_rh_full_esat``: RH=0 → VPD = e_sat(T).
3. ``test_tropical_humid``: T=30°C, RH=0.7 → VPD ≈ 1273 Pa.
4. ``test_arid_high_vpd``: T=40°C, RH=0.2 → VPD > 5 kPa.
5. ``test_monotone_in_t``: ↑T → ↑VPD (CC scaling).
6. ``test_monotone_in_rh_inverse``: ↑RH → ↓VPD.
7. ``test_chain_with_penman_monteith``: VPD → iter-870 λE.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import (
    penman_monteith_le_fv3,
    vpd_from_t_rh_fv3,
)


def test_saturated_zero():
    """RH=1.0 → VPD = 0."""
    vpd = vpd_from_t_rh_fv3(
        t=jnp.array([288.0]),
        rh=jnp.array([1.0]),
    )
    np.testing.assert_allclose(np.asarray(vpd), [0.0], atol=1e-12)


def test_zero_rh_full_esat():
    """RH=0 → VPD = e_sat(T) (pure saturation pressure)."""
    t = jnp.array([288.0])
    vpd = vpd_from_t_rh_fv3(t=t, rh=jnp.array([0.0]))
    e_sat = thermo.saturation_vapor_pressure(t)
    np.testing.assert_allclose(np.asarray(vpd), np.asarray(e_sat),
                                rtol=1e-12)


def test_tropical_humid():
    """T=30°C (303.15 K), RH=0.7 → VPD ≈ 1273 Pa."""
    t_k = jnp.array([303.15])
    vpd = vpd_from_t_rh_fv3(t=t_k, rh=jnp.array([0.7]))
    # e_sat(30°C) ≈ 4243 Pa (Bolton 1980), VPD = 4243·0.3 ≈ 1273 Pa
    assert 1100.0 < float(vpd[0]) < 1400.0


def test_arid_high_vpd():
    """T=40°C (313.15 K), RH=0.2 → VPD > 5 kPa (extreme arid)."""
    t_k = jnp.array([313.15])
    vpd = vpd_from_t_rh_fv3(t=t_k, rh=jnp.array([0.2]))
    # e_sat(40°C) ≈ 7376 Pa, VPD = 7376·0.8 ≈ 5901 Pa
    assert 5000.0 < float(vpd[0]) < 7000.0


def test_monotone_in_t():
    """↑T at fixed RH → ↑VPD (CC scaling)."""
    t_k = jnp.array([283.15, 293.15, 303.15, 313.15])
    rh = jnp.full((4,), 0.5)
    vpd = vpd_from_t_rh_fv3(t_k, rh)
    diffs = jnp.diff(vpd)
    assert jnp.all(diffs > 0.0)


def test_monotone_in_rh_inverse():
    """↑RH at fixed T → ↓VPD."""
    t_k = jnp.full((4,), 303.15)
    rh = jnp.array([0.2, 0.4, 0.6, 0.9])
    vpd = vpd_from_t_rh_fv3(t_k, rh)
    diffs = jnp.diff(vpd)
    assert jnp.all(diffs < 0.0)


def test_chain_with_penman_monteith():
    """VPD from iter-871 → iter-870 PM λE consistency."""
    t_k = jnp.array([303.15])
    vpd = vpd_from_t_rh_fv3(t_k, rh=jnp.array([0.5]))
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=vpd,
        delta_pa_k=jnp.array([200.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.02]),
        g_a=jnp.array([0.05]),
    )
    assert jnp.all(jnp.isfinite(le))
    assert float(le[0]) > 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=871)
    n_x, n_y = 6, 8
    t = jnp.asarray(rng.uniform(250.0, 320.0, size=(n_x, n_y)))
    rh = jnp.asarray(rng.uniform(0.05, 1.0, size=(n_x, n_y)))
    vpd = vpd_from_t_rh_fv3(t, rh)
    assert vpd.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(vpd))
    assert jnp.all(vpd >= 0.0)
