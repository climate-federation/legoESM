"""Analytic tests for WB2 headline diagnostics (Stage 0, Tasks 1-4).

Run on a compute node (JAX x64):
    srun --account=glab --time=0:15:00 env JAX_ENABLE_X64=1 \
        <conda>/python -m pytest tests/evaluations/test_headline_diagnostics.py -v
"""
import numpy as np
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids import create_sigma_coordinate
from evaluations.headline_diagnostics import (
    interp_to_pressure_level,
    geopotential_height_at,
    mean_sea_level_pressure,
    screen_level_t2m,
    wind_10m,
)


# ---- Task 1: log-p interpolation ----

def test_interp_logp_recovers_linear_in_lnp():
    # field exactly linear in ln(p): f = a + b ln(p) -> interp is exact
    p = jnp.array([1000.0, 700.0, 500.0, 300.0, 100.0]) * 100.0  # Pa
    a, b = 250.0, -7.5
    f = a + b * jnp.log(p)
    got = interp_to_pressure_level(f, p, 50000.0)  # 500 hPa
    expected = a + b * np.log(50000.0)
    assert np.isclose(float(got), float(expected), atol=1e-9)


def test_interp_clamps_above_top():
    p = jnp.array([1000.0, 500.0, 100.0]) * 100.0
    f = jnp.array([10.0, 5.0, 1.0])
    got = interp_to_pressure_level(f, p, 5000.0)  # 50 hPa, above the top level
    assert np.isfinite(float(got))
    assert np.isclose(float(got), 1.0, atol=1e-9)  # clamp to top-level value


def test_interp_vectorised_over_columns():
    # two columns, distinct linear-in-lnp profiles
    p = jnp.stack([jnp.array([1000.0, 500.0, 100.0]) * 100.0] * 2, axis=0)
    f = jnp.array([[10.0, 5.0, 1.0], [20.0, 10.0, 2.0]])
    got = interp_to_pressure_level(f, p, 50000.0)  # exactly the middle level
    assert got.shape == (2,)
    assert np.allclose(np.asarray(got), [5.0, 10.0], atol=1e-9)


# ---- Task 2: geopotential height (Z500) ----

def test_z500_isothermal_atmosphere():
    # Isothermal T_v = 250 K, dry, flat surface. Analytic:
    #   z(p) = (R_d T / g) ln(p_s / p);  at 500 hPa with p_s = 1000 hPa.
    nlev = 20
    sigma = create_sigma_coordinate(nlev)
    T = jnp.full((4, 8, nlev), 250.0)
    q = jnp.zeros((4, 8, nlev))
    p_s = jnp.full((4, 8), 100000.0)
    phis = jnp.zeros((4, 8))
    z500 = geopotential_height_at(T, q, p_s, phis, sigma, 50000.0)
    expected = (constants.R_d * 250.0 / constants.g) * np.log(2.0)
    assert z500.shape == (4, 8)
    assert float(jnp.max(jnp.abs(z500 - expected))) < 50.0  # within 50 m (discretisation)


def test_z500_virtual_temperature_raises_thickness():
    # Adding moisture (virtual T > T) increases layer thickness -> higher Z500.
    nlev = 20
    sigma = create_sigma_coordinate(nlev)
    T = jnp.full((2, 2, nlev), 250.0)
    p_s = jnp.full((2, 2), 100000.0)
    phis = jnp.zeros((2, 2))
    z_dry = geopotential_height_at(T, jnp.zeros_like(T), p_s, phis, sigma, 50000.0)
    z_moist = geopotential_height_at(T, jnp.full_like(T, 0.01), p_s, phis, sigma, 50000.0)
    assert float(jnp.min(z_moist - z_dry)) > 0.0


# ---- Task 3: mean sea-level pressure ----

def test_mslp_reduces_upward_and_identity_at_sea_level():
    p_s = jnp.array([90000.0, 100000.0])
    T_low = jnp.array([280.0, 288.0])
    phis = jnp.array([1000.0 * constants.g, 0.0])  # 1000 m, and sea level
    mslp = mean_sea_level_pressure(p_s, T_low, phis)
    assert np.isclose(float(mslp[1]), float(p_s[1]), atol=1e-6)  # sea level -> identity
    assert float(mslp[0]) > float(p_s[0])                        # elevated -> raised
    assert 100000.0 < float(mslp[0]) < 103000.0                  # physical range


# ---- Task 4: screen-level MOST diagnostics ----

def test_t2m_between_surface_and_lowest_level_neutral():
    T_sfc = jnp.array([300.0])
    T_low = jnp.array([295.0])
    z_low = jnp.array([40.0])
    L = jnp.array([1e12])       # neutral
    z0h = jnp.array([0.01])
    t2 = screen_level_t2m(T_sfc, T_low, z_low, L, z0h)
    assert float(T_low[0]) < float(t2[0]) < float(T_sfc[0])


def test_wind10m_neutral_reduces_below_lowest_level():
    u = jnp.array([10.0])
    v = jnp.array([0.0])
    z_low = jnp.array([40.0])
    L = jnp.array([1e12])
    z0m = jnp.array([0.1])
    w10 = wind_10m(u, v, z_low, L, z0m)
    assert 0.0 < float(w10[0]) < 10.0  # 10 m wind weaker than the 40 m level
