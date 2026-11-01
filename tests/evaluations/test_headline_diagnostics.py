"""Analytic tests for WB2 headline diagnostics (Stage 0, Tasks 1-4).

Run on a compute node (JAX x64):
    srun --account=glab --time=0:15:00 env JAX_ENABLE_X64=1 \
        <conda>/python -m pytest tests/evaluations/test_headline_diagnostics.py -v
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids import create_sigma_coordinate, create_hybrid_coordinate
from evaluations.headline_diagnostics import (
    interp_to_pressure_level,
    geopotential_height_at,
    mean_sea_level_pressure_ecmwf,
    surface_temperature_from_lowest_level,
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


def test_geopotential_on_levels_shape_and_isothermal():
    from evaluations.headline_diagnostics import geopotential_on_levels
    nlev = 10
    sig = create_sigma_coordinate(nlev)
    T = jnp.full((2, 2, nlev), 250.0)
    q = jnp.zeros((2, 2, nlev))
    p_s = jnp.full((2, 2), 100000.0)
    phis = jnp.zeros((2, 2))
    phi = geopotential_on_levels(T, q, p_s, phis, sig)
    assert phi.shape == (2, 2, nlev)
    assert bool(jnp.all(phi >= 0.0))                 # phis=0 -> Phi increases upward from 0
    # top level higher than bottom level (Phi increases upward; index 0 = top)
    assert float(jnp.mean(phi[..., 0] - phi[..., -1])) > 0.0


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


# ---- codex-flagged hardening: hybrid coord, MSLP edges, MOST domain, AD safety ----

def test_z500_hybrid_matches_sigma_in_pure_sigma_limit():
    # A_half=0, B_half=linspace(0,1) => p = B*p_s = pure sigma; hybrid path must
    # dispatch to compute_geopotential_hybrid and agree with the sigma path.
    nlev = 20
    B_half = jnp.linspace(0.0, 1.0, nlev + 1)
    A_half = jnp.zeros(nlev + 1)
    hyb = create_hybrid_coordinate(nlev, A_half, B_half)
    sig = create_sigma_coordinate(nlev)
    T = jnp.full((2, 3, nlev), 260.0)
    q = jnp.zeros((2, 3, nlev))
    p_s = jnp.full((2, 3), 100000.0)
    phis = jnp.zeros((2, 3))
    z_hyb = geopotential_height_at(T, q, p_s, phis, hyb, 50000.0)
    z_sig = geopotential_height_at(T, q, p_s, phis, sig, 50000.0)
    assert float(jnp.max(jnp.abs(z_hyb - z_sig))) < 50.0


def test_t2m_rough_surface_stays_bracketed():
    # z0h > z_ref (rough forest/city): must NOT produce T2m warmer than the surface.
    T_sfc = jnp.array([300.0])
    T_low = jnp.array([295.0])
    z_low = jnp.array([40.0])
    L = jnp.array([1e12])
    z0h = jnp.array([5.0])  # > 2 m
    t2 = screen_level_t2m(T_sfc, T_low, z_low, L, z0h)
    assert 295.0 <= float(t2[0]) <= 300.0


def test_most_stable_vs_unstable_differ_and_bracketed():
    # Non-neutral L must actually move the screen value (pins that psi is exercised), stay bracketed + finite.
    T_sfc = jnp.array([300.0, 300.0])
    T_low = jnp.array([295.0, 295.0])
    z_low = jnp.array([40.0, 40.0])
    z0h = jnp.array([0.01, 0.01])
    L = jnp.array([50.0, -50.0])  # stable, unstable
    t2 = screen_level_t2m(T_sfc, T_low, z_low, L, z0h)
    assert bool(jnp.all(jnp.isfinite(t2)))
    assert 295.0 <= float(t2[0]) <= 300.0 and 295.0 <= float(t2[1]) <= 300.0
    assert abs(float(t2[0]) - float(t2[1])) > 1e-6


def test_t2m_nonpositive_roughness_is_finite():
    # z0h <= 0 must be floored internally (no log(0)/log(neg) NaN) and stay bracketed.
    T_sfc = jnp.array([300.0, 300.0])
    T_low = jnp.array([295.0, 295.0])
    z_low = jnp.array([40.0, 40.0])
    L = jnp.array([1e12, 1e12])
    z0h = jnp.array([0.0, -0.5])  # zero and negative roughness
    t2 = screen_level_t2m(T_sfc, T_low, z_low, L, z0h)
    assert bool(jnp.all(jnp.isfinite(t2)))
    assert bool(jnp.all((t2 >= 295.0) & (t2 <= 300.0)))


def test_wind10m_calm_gradient_finite():
    # sqrt(u^2+v^2) has a NaN gradient at (0,0); the AD-safe floor must fix it.
    def speed(u):
        return wind_10m(jnp.array([u]), jnp.array([0.0]), jnp.array([40.0]),
                        jnp.array([1e12]), jnp.array([0.1]))[0]
    g = jax.grad(speed)(0.0)
    assert bool(jnp.isfinite(g))


def test_z500_gradient_wrt_temperature_finite():
    # grad flows through compute_geopotential + argsort-based interp for a fixed level order.
    nlev = 12
    sig = create_sigma_coordinate(nlev)

    def z500(Tval):
        T = jnp.full((1, 1, nlev), Tval)
        q = jnp.zeros((1, 1, nlev))
        p_s = jnp.full((1, 1), 100000.0)
        phis = jnp.zeros((1, 1))
        return geopotential_height_at(T, q, p_s, phis, sig, 50000.0)[0, 0]

    g = jax.grad(z500)(260.0)
    assert bool(jnp.isfinite(g))


# ---- ECMWF/Trenberth MSLP reduction (RULE 5: hard limit, smooth gap, gradient) ----

# (p_s Pa, T* K, z_s m): sea level, mid terrain, cold limit, warm T0 cap,
# hot T* > 290.5 K, below sea level near the warm cap
_MSLP_CASES = [(101000.0, 288.0, 0.0), (90000.0, 280.0, 1000.0),
               (70000.0, 240.0, 3000.0), (80000.0, 285.0, 2000.0),
               (95000.0, 300.0, 500.0), (101500.0, 290.0, -400.0)]
# measured smooth(1 K) - hard [Pa] per case (commit d858bc8d3a)
_MSLP_GAP_W1K = [0.0, 0.3893932426581159, -2.4744338588789105e-05,
                 0.024214830904384144, 0.0033077911066357046, -2.1548123696265975]


def _ecmwf_mslp_reference(p_s, t_s, z_s):
    """Branching transcription of the ECMWF/Trenberth reduction (oracle)."""
    gamma, g, r = 0.0065, float(constants.g), float(constants.R_d)
    t_star = t_s if t_s >= 255.0 else 0.5 * (255.0 + t_s)
    t0 = t_star + gamma * z_s
    if t0 > 290.5:
        if t_star <= 290.5:
            gamma = (290.5 - t_star) / z_s
        else:
            gamma, t_star = 0.0, 0.5 * (290.5 + t_star)
    x = gamma * z_s / t_star
    return p_s * np.exp(g * z_s / (r * t_star) * (1.0 - x / 2.0 + x * x / 3.0))


@pytest.mark.parametrize("case", _MSLP_CASES)
def test_mslp_ecmwf_hard_limit_matches_reference(case):
    p_s, t_s, z_s = case
    got = mean_sea_level_pressure_ecmwf(jnp.array(p_s), jnp.array(t_s),
                                        jnp.array(z_s * float(constants.g)), limit_width_k=0.0)
    assert np.isclose(float(got), _ecmwf_mslp_reference(p_s, t_s, z_s), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("case,gap", list(zip(_MSLP_CASES, _MSLP_GAP_W1K)))
def test_mslp_ecmwf_smooth_default_gap_per_case(case, gap):
    from evaluations.wb_forecast import MSLP_LIMIT_WIDTH_K
    assert MSLP_LIMIT_WIDTH_K == 1.0  # the gaps were measured at 1 K
    p_s, t_s, z_s = case
    args = (jnp.array(p_s), jnp.array(t_s), jnp.array(z_s * float(constants.g)))
    d = float(mean_sea_level_pressure_ecmwf(*args, limit_width_k=MSLP_LIMIT_WIDTH_K)
              - mean_sea_level_pressure_ecmwf(*args, limit_width_k=0.0))
    assert np.isclose(d, gap, rtol=1e-6, atol=1e-12)


def test_mslp_ecmwf_smooth_gradients():
    from jax.test_util import check_grads
    phis = jnp.array([c[2] for c in _MSLP_CASES]) * float(constants.g)
    p_s = jnp.array([c[0] for c in _MSLP_CASES])
    t_s = jnp.array([c[1] for c in _MSLP_CASES])
    f = lambda p, t: mean_sea_level_pressure_ecmwf(p, t, phis, limit_width_k=1.0)
    check_grads(f, (p_s, t_s), order=2, modes=["rev"])
    dt = jax.grad(lambda t: f(p_s, t).sum())(t_s)
    assert np.all(np.isfinite(dt)) and np.all(np.abs(np.asarray(dt)[1:]) > 0.0)


def test_surface_temperature_from_lowest_level():
    # ln(p_s/p_L) > 0 -> warmer at the ground; equal pressures -> identity
    t = surface_temperature_from_lowest_level(jnp.array([280.0, 280.0]),
                                              jnp.array([99000.0, 100000.0]),
                                              jnp.array([100000.0, 100000.0]))
    expect = 280.0 + 0.0065 * float(constants.R_d) / float(constants.g) * 280.0 * np.log(100000.0 / 99000.0)
    assert np.isclose(float(t[0]), expect, rtol=1e-12) and float(t[1]) == 280.0
