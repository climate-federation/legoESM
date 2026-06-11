"""Unit tests for SDM collision kernels and terminal velocities.

Direct numeric oracles (abs-free relative checks), symmetry, non-negativity,
monotonicity, regime continuity, dispatch hardening, vmap/jit/grad.

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/microphysics/unit/test_sdm_kernels.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm import (
    SDMConfig,
    collision_kernel,
    golovin_kernel,
    hall_kernel,
    long_kernel,
    sedimentation_kernel,
    terminal_velocity,
    terminal_velocity_atlas_ulbrich,
    terminal_velocity_cloud_rain_shima,
    terminal_velocity_rogers_yau,
)

_FOUR_THIRDS_PI = 4.0 / 3.0 * np.pi


# --------------------------------------------------------------------------
# Collision kernels
# --------------------------------------------------------------------------
def test_golovin_equal_radii_matches_oracle():
    r = 2.0e-5
    b = 1.5e3
    got = float(golovin_kernel(jnp.asarray(r), jnp.asarray(r), b))
    expected = b * (2.0 * _FOUR_THIRDS_PI * r**3)
    assert got == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_golovin_symmetric_and_additive():
    r1, r2, b = 1.0e-5, 3.0e-5, 1.5e3
    k12 = float(golovin_kernel(jnp.asarray(r1), jnp.asarray(r2), b))
    k21 = float(golovin_kernel(jnp.asarray(r2), jnp.asarray(r1), b))
    assert k12 == pytest.approx(k21, rel=1e-12)
    expected = b * _FOUR_THIRDS_PI * (r1**3 + r2**3)
    assert k12 == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_sedimentation_equal_radii_efficiency():
    """p=1 -> E = 0.5/(2²) = 1/8; K = π(2r)²·(1/8)·|dv|."""
    r, dv = 3.0e-5, 0.1
    got = float(sedimentation_kernel(jnp.asarray(r), jnp.asarray(r), jnp.asarray(dv)))
    expected = np.pi * (2.0 * r) ** 2 * (1.0 / 8.0) * dv
    assert got == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_sedimentation_symmetric_nonneg_and_zero_dv():
    r1, r2 = 1.0e-5, 4.0e-5
    k12 = float(sedimentation_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(0.2)))
    k21 = float(sedimentation_kernel(jnp.asarray(r2), jnp.asarray(r1), jnp.asarray(-0.2)))
    assert k12 == pytest.approx(k21, rel=1e-12)  # symmetric, |dv|
    assert k12 >= 0.0
    assert float(sedimentation_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(0.0))) == 0.0


def test_long_kernel_geometric_above_threshold():
    """Both radii > 50 um -> efficiency 1 -> geometric K = π(r_l+r_s)²·|dv|."""
    r1, r2, dv = 8.0e-5, 6.0e-5, 0.5
    got = float(long_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv)))
    expected = np.pi * (r1 + r2) ** 2 * dv
    assert got == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_long_kernel_cloud_branch_matches_oracle():
    r1, r2, dv = 3.0e-5, 1.0e-5, 0.05   # r_l = 30 um <= 50 um -> cloud polynomial
    r_l, r_s = max(r1, r2), min(r1, r2)
    c = 4.5e8 * r_l**2 * (1.0 - 3.0e-6 / max(3.01e-6, r_l))
    expected = c * np.pi * (r_l + r_s) ** 2 * dv
    got = float(long_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv)))
    assert got == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("r_l_um,ratio,E_expected", [
    # Exact grid node: r_l=100um (col 11), ratio=0.50 (row 10) -> E=1.0000.
    (100.0, 0.50, 1.0000),
    # Bilinear midpoint: r_l=35um (between cols 6,7), ratio=0.225 (rows 4,5):
    # 0.25*(0.0600+0.5000+0.1000+0.6200) = 0.3200.
    (35.0, 0.225, 0.3200),
    # Small-collector branch (r_l<6um, col 0), ratio=0.50 (row 10) -> 0.0400.
    (5.0, 0.50, 0.0400),
    # Large-collector branch (r_l>300um, col 14), ratio=0.95 (row 19):
    # table 2.3 capped at 1.0 (oracle caps ONLY this branch).
    (400.0, 0.95, 1.0000),
    # Interior wake-capture >1 must NOT be capped: r_l=70um (col 10),
    # ratio=1.0 (row 20) -> 4.0.
    (70.0, 1.00, 4.0000),
])
def test_hall_kernel_matches_table_oracle(r_l_um, ratio, E_expected):
    """Hand-computed Hall-table points: grid nodes, bilinear midpoint, both
    1-D edge branches, the large-branch cap, and uncapped interior E>1."""
    r_l = r_l_um * 1.0e-6
    r_s = ratio * r_l
    dv = 0.37
    got = float(hall_kernel(jnp.asarray(r_l), jnp.asarray(r_s), jnp.asarray(dv)))
    expected = E_expected * np.pi * (r_l + r_s) ** 2 * dv
    assert got == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_hall_kernel_symmetric_nonneg_zero_safe():
    r1, r2, dv = 8.0e-5, 3.0e-5, 0.4
    k12 = float(hall_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv)))
    k21 = float(hall_kernel(jnp.asarray(r2), jnp.asarray(r1), jnp.asarray(dv)))
    assert k12 == pytest.approx(k21, rel=1e-12)
    assert k12 > 0.0
    assert float(hall_kernel(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.4))) == 0.0
    # vmap/jit over a radius batch stays finite
    radii = jnp.asarray(np.geomspace(1e-6, 1e-3, 30))
    K = jax.jit(jax.vmap(lambda r: hall_kernel(r, 0.5 * r, jnp.asarray(0.1))))(radii)
    assert bool(jnp.all(jnp.isfinite(K))) and bool(jnp.all(K >= 0.0))


def test_collision_kernel_dispatch_and_unknown():
    cfg = SDMConfig()
    r1, r2, dv = 2.0e-5, 1.0e-5, 0.1
    assert float(collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                                  cfg._replace(collision_kernel="golovin"))) > 0.0
    assert float(collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                                  cfg._replace(collision_kernel="sedimentation"))) > 0.0
    assert float(collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                                  cfg._replace(collision_kernel="long"))) > 0.0
    assert float(collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                                  cfg._replace(collision_kernel="hall"))) > 0.0
    with pytest.raises(ValueError, match="Unknown SDM collision_kernel"):
        collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                         cfg._replace(collision_kernel="brownian"))


# --------------------------------------------------------------------------
# Terminal velocities
# --------------------------------------------------------------------------
def test_rogers_yau_matches_oracle_and_value():
    r = 1.0e-5
    got = float(terminal_velocity_rogers_yau(jnp.asarray(r)))
    assert got == pytest.approx(1.233e8 * r * r, rel=1e-12, abs=0.0)
    # 10 um droplet falls at ~1.2 cm/s
    assert got == pytest.approx(1.233e-2, rel=1e-6)


def test_atlas_ulbrich_matches_oracle():
    r = 0.5e-3   # D = 1 mm
    got = float(terminal_velocity_atlas_ulbrich(jnp.asarray(r)))
    assert got == pytest.approx(3.778 * (1.0) ** 0.67, rel=1e-9)  # = 3.778 m/s


def test_terminal_velocities_positive_and_monotone():
    radii = jnp.asarray(np.geomspace(2.0e-6, 1.5e-3, 60))
    rho = jnp.full_like(radii, 1.0)
    p = jnp.full_like(radii, 9.0e4)
    T = jnp.full_like(radii, 283.0)
    for fn in (lambda rr: terminal_velocity_rogers_yau(rr),
               lambda rr: terminal_velocity_atlas_ulbrich(rr),
               lambda rr: terminal_velocity_cloud_rain_shima(rr, rho, p, T)):
        v = np.asarray(fn(radii))
        assert np.all(v > 0.0)
        assert np.all(np.diff(v) > 0.0)  # strictly increasing with radius


def test_cloud_rain_shima_magnitudes():
    """Order-of-magnitude sanity across the three regimes."""
    rho, p, T = 1.0, 9.0e4, 283.0
    def v(r):
        return float(terminal_velocity_cloud_rain_shima(
            jnp.asarray(r), jnp.asarray(rho), jnp.asarray(p), jnp.asarray(T)))
    assert 1.0e-3 < v(1.0e-5) < 5.0e-2     # 10 um cloud droplet ~ 1 cm/s
    assert 0.1 < v(1.0e-4) < 1.5           # 100 um drizzle ~ 0.3-0.7 m/s
    assert 3.0 < v(1.0e-3) < 9.0           # 1 mm raindrop ~ 4-7 m/s


def _cloud_rain_shima_oracle(r, rho, p, T):
    """Independent scalar (numpy) re-implementation of SCALE-SDM CloudRainShima.

    Uses its OWN copy of the Beard polynomial coefficients evaluated in explicit
    ascending-power form ``sum(c*x**i)`` — so it cross-checks both the
    coefficient values AND the module's Horner ``_poly`` ordering (a reversed
    ``_poly`` would make the jax output disagree). Same physical constants
    (values are policy, not under test) so the comparison is exact algebra.
    """
    VZ_B = (-3.18657, 0.9926960, -0.153193e-2, -0.987059e-3,
            -0.578878e-3, 0.855176e-4, -0.327815e-5)
    VZ_C = (-5.00015, 5.23778, -2.04914, 0.475294, -0.542819e-1, 0.238449e-2)

    def poly(cs, x):
        return sum(c * x**i for i, c in enumerate(cs))

    Tc = T - constants.T_freeze
    visc = ((1.718 + 4.9e-3 * Tc) * 1e-4 if Tc >= 0
            else (1.718 + 4.9e-3 * Tc - 1.2e-5 * Tc * Tc) * 1e-4)
    d = max(2.0 * r * 100.0, 1.0e-20)
    P_hPa = p / 100.0
    rho_mat = constants.rho_water / 1000.0
    rho_air = rho / 1000.0
    gx = (constants.g * 100.0) * (rho_mat - rho_air)
    sd = 6.62e-6 * (visc / 1.818e-4) * (1013.25 / P_hPa) * np.sqrt(T / 293.15)
    csc = 1.0 + 2.510 * (sd / d)
    v1 = gx / (18.0 * visc) * csc * d * d
    nda = rho_air * (4.0 * gx) / (3.0 * visc * visc) * d**3
    nre2 = csc * np.exp(poly(VZ_B, np.log(max(nda, 1e-300))))
    v2 = visc * nre2 / (rho_air * d)
    tau = 1.0 - T / 647.096
    sig = 0.2358 * np.exp(1.256 * np.log(max(tau, 1e-30))) * (1.0 - 0.625 * tau)
    if T < 267.5:
        sig = sig - 2.854e-3 * np.tanh((T - 243.9) / 35.35) + 1.666e-3
    sig *= 1e3
    bond = (4.0 * gx) / (3.0 * sig) * d * d
    npp = (rho_air**2 * sig**3 / (gx * visc**4))
    npp = np.exp(np.log(max(npp, 1e-300)) / 6.0)
    nre3 = npp * np.exp(poly(VZ_C, np.log(max(bond * npp, 1e-300))))
    v3 = visc * nre3 / (rho_air * d)
    vc = v1 if d < 1.9e-3 else (v2 if d < 1.07e-1 else v3)
    return vc / 100.0 if r > 0 else 0.0


@pytest.mark.parametrize("r,rho,p,T", [
    (1.0e-5, 1.0, 9.0e4, 283.0),    # regime 1 (small cloud)
    (1.0e-4, 1.0, 9.0e4, 283.0),    # regime 2 (large cloud / small rain)
    (1.0e-3, 1.05, 9.0e4, 283.0),   # regime 3 (large rain)
    (9.4e-6, 1.0, 9.0e4, 283.0),    # just below d = 1.9e-3 cm boundary
    (9.6e-6, 1.0, 9.0e4, 283.0),    # just above d = 1.9e-3 cm boundary
    (5.3e-4, 1.0, 9.0e4, 283.0),    # just below d = 1.07e-1 cm boundary
    (5.4e-4, 1.0, 9.0e4, 283.0),    # just above d = 1.07e-1 cm boundary
    (8.0e-4, 1.0, 9.0e4, 260.0),    # cold (T<267.5) surface-tension correction
])
def test_cloud_rain_shima_matches_independent_oracle(r, rho, p, T):
    got = float(terminal_velocity_cloud_rain_shima(
        jnp.asarray(r), jnp.asarray(rho), jnp.asarray(p), jnp.asarray(T)))
    expected = _cloud_rain_shima_oracle(r, rho, p, T)
    assert got == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_cloud_rain_shima_zero_radius_is_safe():
    """r <= 0 must give a finite 0 velocity and a finite gradient (no nan from
    the eager 1/diameter branches)."""
    args = (jnp.asarray(1.0), jnp.asarray(9.0e4), jnp.asarray(283.0))
    v0 = terminal_velocity_cloud_rain_shima(jnp.asarray(0.0), *args)
    assert float(v0) == 0.0 and jnp.isfinite(v0)
    g = jax.grad(lambda r: terminal_velocity_cloud_rain_shima(r, *args))(jnp.asarray(0.0))
    assert jnp.isfinite(g)


def test_terminal_velocity_dispatch_and_unknown():
    cfg = SDMConfig()
    r = jnp.asarray(1.0e-5)
    rho, p, T = jnp.asarray(1.0), jnp.asarray(9.0e4), jnp.asarray(283.0)
    for kind in ("rogers_yau", "atlas_ulbrich", "cloud_rain_shima"):
        v = float(terminal_velocity(r, rho, p, T, cfg._replace(terminal_velocity=kind)))
        assert v > 0.0
    with pytest.raises(ValueError, match="Unknown SDM terminal_velocity"):
        terminal_velocity(r, rho, p, T, cfg._replace(terminal_velocity="beard"))


def test_kernels_jit_vmap_grad():
    cfg = SDMConfig(terminal_velocity="cloud_rain_shima")
    radii = jnp.asarray(np.geomspace(5e-6, 1e-3, 16))
    rho = jnp.full_like(radii, 1.0)
    p = jnp.full_like(radii, 9.0e4)
    T = jnp.full_like(radii, 283.0)

    @jax.jit
    def total_fallspeed(rr):
        return jnp.sum(terminal_velocity(rr, rho, p, T, cfg))

    assert jnp.isfinite(total_fallspeed(radii))
    g = jax.grad(total_fallspeed)(radii)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.all(g > 0.0)  # fall speed increases with radius everywhere
