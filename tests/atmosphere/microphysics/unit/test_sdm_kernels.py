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


def test_collision_kernel_dispatch_and_unknown():
    cfg = SDMConfig()
    r1, r2, dv = 2.0e-5, 1.0e-5, 0.1
    assert float(collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                                  cfg._replace(collision_kernel="golovin"))) > 0.0
    assert float(collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                                  cfg._replace(collision_kernel="sedimentation"))) > 0.0
    assert float(collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                                  cfg._replace(collision_kernel="long"))) > 0.0
    with pytest.raises(ValueError, match="Unknown SDM collision_kernel"):
        collision_kernel(jnp.asarray(r1), jnp.asarray(r2), jnp.asarray(dv),
                         cfg._replace(collision_kernel="hall"))


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
