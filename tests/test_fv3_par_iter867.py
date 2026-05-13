"""FV3_3D iter 867: par_from_global_radiation_fv3.

PAR = 0.45 · R_s  [W/m² or MJ/m²/day, matching input].

Tests
-----

1. ``test_tropical_noon_clear``: R_s=1000 → PAR=450 W/m².
2. ``test_overcast``: R_s=200 → PAR=90.
3. ``test_zero_no_input``: R_s=0 → PAR=0.
4. ``test_par_fraction_custom``: f_PAR=0.50 → PAR=0.5·R_s.
5. ``test_par_photon_conversion``: PAR_W·4.57 → μmol/m²/s.
6. ``test_chain_with_iter866``: lat+DOY+hr → S_TOA → R_s → PAR.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    clear_sky_toa_radiation_fv3,
    par_from_global_radiation_fv3,
    solar_zenith_cos_fv3,
)


def test_tropical_noon_clear():
    """R_s=1000 W/m² tropical noon → PAR = 450 W/m²."""
    par = par_from_global_radiation_fv3(jnp.array([1000.0]))
    np.testing.assert_allclose(np.asarray(par), [450.0], rtol=1e-12)


def test_overcast():
    """R_s=200 W/m² overcast → PAR = 90 W/m² (f_PAR=0.45)."""
    par = par_from_global_radiation_fv3(jnp.array([200.0]))
    np.testing.assert_allclose(np.asarray(par), [90.0], rtol=1e-12)


def test_zero_no_input():
    """R_s=0 → PAR=0."""
    par = par_from_global_radiation_fv3(jnp.array([0.0]))
    np.testing.assert_allclose(np.asarray(par), [0.0], atol=1e-14)


def test_par_fraction_custom():
    """Britton-Dodd 1976 original f_PAR=0.50 → PAR=0.5·R_s."""
    par = par_from_global_radiation_fv3(
        jnp.array([1000.0]),
        par_fraction=0.50,
    )
    np.testing.assert_allclose(np.asarray(par), [500.0], rtol=1e-12)


def test_par_photon_conversion():
    """R_s=1000 → PAR=450 → ×4.57 → μmol/m²/s (Britton-Dodd factor)."""
    par_w = par_from_global_radiation_fv3(jnp.array([1000.0]))  # tropical R_s
    par_umol = par_w * 4.57
    # 450 × 4.57 = 2056.5 μmol/m²/s tropical noon
    assert 2000.0 < float(par_umol[0]) < 2100.0


def test_chain_with_iter866():
    """lat=0, DOY=80, noon → S_TOA → R_s = τ·S_TOA → PAR via iter-867."""
    cos_t = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([0.0]),
        day_of_year=jnp.array([80.0]),
        hour_local=jnp.array([12.0]),
    )
    s_toa = clear_sky_toa_radiation_fv3(cos_t, jnp.array([80.0]))
    tau_atm = 0.7  # clear-sky transmissivity
    r_s = tau_atm * s_toa
    par = par_from_global_radiation_fv3(r_s)
    # S_TOA ≈ 1369, R_s ≈ 958, PAR ≈ 431
    assert 400.0 < float(par[0]) < 470.0


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=867)
    n_x, n_y = 6, 8
    r_s = jnp.asarray(rng.uniform(0.0, 1200.0, size=(n_x, n_y)))
    par = par_from_global_radiation_fv3(r_s)
    assert par.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(par))
    assert jnp.all(par >= 0.0)
