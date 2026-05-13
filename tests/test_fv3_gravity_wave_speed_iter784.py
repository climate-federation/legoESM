"""FV3_3D iter 784: gravity_wave_speed_fv3 (c = N·H).

Composes with iter-781 (L_R = c/|f|) + iter-783 (L_eq = √(c/(2β))).

Tests
-----

1. ``test_c_atmospheric_kelvin``: N=0.01, H=3 km → c=30 m/s.
2. ``test_c_ocean_baroclinic``: N=0.005, H=540 m → c=2.7 m/s.
3. ``test_c_zero_N``: N=0 → c=0.
4. ``test_c_monotonic_NH``: ↑N → ↑c; ↑H → ↑c.
5. ``test_c_composes_iter783``: pipeline (N, H, β) → c → L_eq.
6. ``test_c_composes_iter781``: pipeline (N, H, f) → c → L_R via c/|f|.
7. ``test_c_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    beta_plane_fv3,
    coriolis_parameter_fv3,
    equatorial_rossby_radius_fv3,
    gravity_wave_speed_fv3,
    rossby_radius_fv3,
)


def test_c_atmospheric_kelvin():
    """Tropical: N=0.01, H=3 km → c = 30 m/s (Kelvin wave)."""
    N = jnp.array([0.01])
    H = jnp.array([3000.0])
    c = gravity_wave_speed_fv3(N, H)
    np.testing.assert_allclose(np.asarray(c), [30.0], rtol=1e-12)


def test_c_ocean_baroclinic():
    """Ocean mode 1: N=0.005, H=540 m → c=2.7 m/s."""
    N = jnp.array([0.005])
    H = jnp.array([540.0])
    c = gravity_wave_speed_fv3(N, H)
    np.testing.assert_allclose(np.asarray(c), [2.7], rtol=1e-12)


def test_c_zero_N():
    """N=0 → c=0 (no stratification, no internal wave)."""
    N = jnp.array([0.0, 0.0, 0.0])
    H = jnp.array([1000.0, 5000.0, 10_000.0])
    c = gravity_wave_speed_fv3(N, H)
    np.testing.assert_allclose(np.asarray(c), [0.0, 0.0, 0.0], atol=1e-15)


def test_c_monotonic_NH():
    """↑N → ↑c at fixed H; ↑H → ↑c at fixed N."""
    H = jnp.array([3000.0, 3000.0, 3000.0])
    N_lo = jnp.array([0.005, 0.010, 0.015])
    N_hi = N_lo + 0.005
    assert jnp.all(gravity_wave_speed_fv3(N_hi, H) > gravity_wave_speed_fv3(N_lo, H))

    N = jnp.array([0.01, 0.01, 0.01])
    H_lo = jnp.array([1000.0, 3000.0, 5000.0])
    H_hi = H_lo + 2000.0
    assert jnp.all(gravity_wave_speed_fv3(N, H_hi) > gravity_wave_speed_fv3(N, H_lo))


def test_c_composes_iter783():
    """Pipeline (N, H, β_eq) → c → L_eq matches manual chain."""
    N = jnp.array([0.01])
    H = jnp.array([3000.0])
    beta = beta_plane_fv3(jnp.array([0.0]))  # equator β = 2·Ω/R
    c = gravity_wave_speed_fv3(N, H)
    L_eq = equatorial_rossby_radius_fv3(c, beta)
    # Reference: c=30, β=2.29e-11 → L_eq ≈ 810 km
    assert 7.5e5 < float(L_eq[0]) < 8.5e5


def test_c_composes_iter781():
    """Mid-lat L_R = N·H/|f| can be computed via c = N·H then L_R = c/|f|."""
    N = jnp.array([0.01])
    H = jnp.array([10_000.0])
    f = coriolis_parameter_fv3(jnp.array([30.0]), units="deg")  # |f|=Ω
    c = gravity_wave_speed_fv3(N, H)
    # L_R via direct iter-781
    L_R_direct = rossby_radius_fv3(N, f, H)
    # L_R via c-derivation: L_R = c / |f|
    L_R_via_c = c / jnp.abs(f)
    np.testing.assert_allclose(
        np.asarray(L_R_direct), np.asarray(L_R_via_c), rtol=1e-12
    )


def test_c_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=784)
    n_x, n_y, km = 4, 5, 20
    N = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    H = jnp.asarray(rng.uniform(1_000.0, 15_000.0, size=(n_x, n_y, km)))
    c = gravity_wave_speed_fv3(N, H)
    assert c.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(c))
    assert jnp.all(c >= 0.0)
