"""FV3_3D iter 781: rossby_radius_fv3 (L_R = N·H/|f|).

Composes iter-772 N² (caller sqrt) + iter-778 Coriolis.

Tests
-----

1. ``test_lr_midlat_synoptic``: N=0.01, H=10km, 30°N → L_R ≈ 1370 km.
2. ``test_lr_equator_floored``: f=0 → huge but finite via f_floor.
3. ``test_lr_pole``: lat=π/2 → L_R = N·H/(2·Ω).
4. ``test_lr_monotonic_N``: ↑N → ↑L_R.
5. ``test_lr_monotonic_H``: ↑H → ↑L_R.
6. ``test_lr_composes_iter772_iter778``: pipeline test.
7. ``test_lr_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    brunt_vaisala_squared_fv3,
    coriolis_parameter_fv3,
    rossby_radius_fv3,
)


def test_lr_midlat_synoptic():
    """30°N, N=0.01, H=10 km → L_R = 0.01·10000/Ω ≈ 1.37 Mm."""
    N = jnp.array([0.01])
    f = coriolis_parameter_fv3(jnp.array([30.0]), units="deg")
    H = jnp.array([10_000.0])
    L_R = rossby_radius_fv3(N, f, H)
    expected = 0.01 * 10_000.0 / constants.Omega  # |f|=Ω at 30°N
    np.testing.assert_allclose(np.asarray(L_R), [expected], rtol=1e-12)
    # Sanity: ~1.37 Mm
    assert 1.3e6 < float(L_R[0]) < 1.5e6


def test_lr_equator_floored():
    """Equator: f=0 → L_R clamped huge but finite."""
    N = jnp.array([0.01])
    f = coriolis_parameter_fv3(jnp.array([0.0]))
    H = jnp.array([10_000.0])
    L_R = rossby_radius_fv3(N, f, H, f_floor=1e-12)
    assert jnp.all(jnp.isfinite(L_R))
    # 0.01·10000/1e-12 = 1e14 m
    assert float(L_R[0]) > 1e13


def test_lr_pole():
    """North pole: |f|=2·Ω → L_R = N·H/(2·Ω)."""
    N = jnp.array([0.01])
    f = coriolis_parameter_fv3(jnp.array([jnp.pi / 2.0]))
    H = jnp.array([10_000.0])
    L_R = rossby_radius_fv3(N, f, H)
    expected = 0.01 * 10_000.0 / (2.0 * constants.Omega)
    np.testing.assert_allclose(np.asarray(L_R), [expected], rtol=1e-12)


def test_lr_monotonic_N():
    """Higher N → larger L_R at fixed (f, H)."""
    f = jnp.array([1.0e-4, 1.0e-4, 1.0e-4])
    H = jnp.array([10_000.0, 10_000.0, 10_000.0])
    N_lo = jnp.array([0.005, 0.010, 0.015])
    N_hi = N_lo + 0.005
    L_lo = rossby_radius_fv3(N_lo, f, H)
    L_hi = rossby_radius_fv3(N_hi, f, H)
    assert jnp.all(L_hi > L_lo)


def test_lr_monotonic_H():
    """Higher H → larger L_R at fixed (N, f)."""
    f = jnp.array([1.0e-4, 1.0e-4, 1.0e-4])
    N = jnp.array([0.01, 0.01, 0.01])
    H_lo = jnp.array([5_000.0, 10_000.0, 15_000.0])
    H_hi = H_lo + 5_000.0
    L_lo = rossby_radius_fv3(N, f, H_lo)
    L_hi = rossby_radius_fv3(N, f, H_hi)
    assert jnp.all(L_hi > L_lo)


def test_lr_composes_iter772_iter778():
    """Full pipeline: (θ, q, z, lat, H) → N² → N → f → L_R."""
    # Hydrostatic isothermal column (analytic N² ≈ g²/(c_p·T))
    T = 290.0
    H_scale = constants.R_d * T / constants.g
    z = jnp.linspace(0.0, 10_000.0, 21)
    p = constants.p_ref * jnp.exp(-z / H_scale)
    theta = T * (constants.p_ref / p) ** constants.kappa
    q = jnp.zeros_like(theta)
    n_sq = brunt_vaisala_squared_fv3(theta, q, z)
    N = jnp.sqrt(jnp.maximum(0.0, n_sq))  # shape (20,)
    f_mid = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    # Broadcast f to N shape
    f_arr = jnp.broadcast_to(f_mid, N.shape)
    H_arr = jnp.full_like(N, 10_000.0)
    L_R = rossby_radius_fv3(N, f_arr, H_arr)
    # Sanity: synoptic L_R range
    assert jnp.all(jnp.isfinite(L_R))
    assert jnp.all(L_R > 0.0)
    assert jnp.all(L_R < 1e8)  # all under 100 000 km


def test_lr_shapes_3d_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=781)
    n_x, n_y, km = 4, 5, 20
    N = jnp.asarray(rng.uniform(0.005, 0.020, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(1e-5, 1.5e-4, size=(n_x, n_y, km)))
    H = jnp.asarray(rng.uniform(1_000.0, 15_000.0, size=(n_x, n_y, km)))
    L_R = rossby_radius_fv3(N, f, H)
    assert L_R.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(L_R))
    assert jnp.all(L_R > 0.0)
