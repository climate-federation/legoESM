"""Unit tests for :mod:`legoesm.atmosphere.dynamics.column_large_scale_extract`.

Stage-4 grid-side extractor: ω from continuity + horizontal advective
tendencies on a lat-lon GCM state.  Analytic checks on the pure cores plus
clean uniform-field cases on a real LatLonGrid (uniform flow/field ⇒ zero
divergence/gradient ⇒ zero ω/advection), and the assembled ColumnLargeScaleState.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.column_forcing import (
    ColumnLargeScaleState,
    build_column_scm_forcing,
)
from legoesm.atmosphere.dynamics.column_large_scale_extract import (
    advective_tendency,
    extract_column_forcing_latlon,
    omega_from_divergence,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

jax.config.update("jax_enable_x64", True)


def test_advective_tendency_analytic():
    u = jnp.full((3,), 2.0)
    v = jnp.zeros((3,))
    dphi_dx = jnp.full((3,), 3.0)
    dphi_dy = jnp.full((3,), -1.0)
    tend = advective_tendency(u, v, dphi_dx, dphi_dy)
    # -(2*3 + 0*-1) = -6
    np.testing.assert_allclose(np.asarray(tend), -6.0, rtol=1e-12)


def test_advective_tendency_zero_for_uniform_field():
    u = jnp.full((4,), 5.0)
    v = jnp.full((4,), 3.0)
    tend = advective_tendency(u, v, jnp.zeros((4,)), jnp.zeros((4,)))
    np.testing.assert_array_equal(np.asarray(tend), np.zeros(4))


def test_omega_zero_for_zero_divergence():
    sigma = create_sigma_coordinate(6)
    div = jnp.zeros((2, 3, 6))
    p_s = jnp.full((2, 3), 1.0e5)
    omega = omega_from_divergence(div, p_s, sigma)
    assert omega.shape == (2, 3, 6)
    np.testing.assert_allclose(np.asarray(omega), 0.0, atol=1e-10)


def test_omega_nonzero_finite_for_divergence():
    sigma = create_sigma_coordinate(6)
    # Uniform convergence (div<0) in the lower half, divergence aloft.
    div = jnp.zeros((1, 1, 6)).at[0, 0, 3:].set(-1e-5).at[0, 0, :3].set(1e-5)
    p_s = jnp.full((1, 1), 1.0e5)
    omega = omega_from_divergence(div, p_s, sigma)
    assert bool(jnp.all(jnp.isfinite(omega)))
    # ω at the model top (σ→0) is ~0; non-trivial in the interior.
    assert float(jnp.max(jnp.abs(omega))) > 0.0


def test_omega_matches_manual_continuity_with_sign():
    """Nonzero column-integrated divergence: verify the dp_s/dt closure + sign
    against an explicit compute_sigma_dot_and_total + compute_pressure_velocity."""
    from legoesm.grids.vertical import (
        compute_pressure_velocity,
        compute_sigma_dot_and_total,
    )

    nlev = 6
    sigma = create_sigma_coordinate(nlev)
    D = 2e-5  # uniform divergence at every level (net mass export)
    div = jnp.full((1, 1, nlev), D)
    p_s = jnp.full((1, 1), 1.0e5)

    omega = omega_from_divergence(div, p_s, sigma)

    sigma_dot, d_total = compute_sigma_dot_and_total(div, sigma)
    sigma_top = float(sigma.sigma_half[0])
    dp_s_dt = -p_s * d_total[..., 0] / (1.0 - sigma_top)
    omega_manual = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma)
    # Float32-precision agreement (the sigma coordinate is float32 internally);
    # the point is the dp_s/dt closure + sign, not bit-exactness.
    np.testing.assert_allclose(np.asarray(omega), np.asarray(omega_manual), rtol=1e-6)

    # Net divergence (mass export) ⇒ surface pressure FALLS ⇒ dp_s/dt < 0.
    assert float(dp_s_dt[0, 0]) < 0.0
    # Δσ sums to (1 − σ_top) ⇒ D_total = D·(1 − σ_top); the /(1−σ_top) factor in
    # dp_s/dt then cancels it ⇒ dp_s/dt = −p_s·D exactly.
    assert float(d_total[0, 0, 0]) == pytest.approx(D * (1.0 - sigma_top), rel=1e-5)
    assert float(dp_s_dt[0, 0]) == pytest.approx(-1.0e5 * D, rel=1e-5)


def _latlon_state(n_lat=8, n_lon=16, nlev=6):
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    T = jnp.full(shape, 280.0)
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 10.0)
    v = jnp.full(shape, 0.0)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    return grid, sigma, T, q_v, u, v, p_s


def test_extract_uniform_state_zero_forcing():
    """A horizontally-uniform state ⇒ zero advection AND zero subsidence."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(20.0)), col_index=(4, 8),
    )
    assert isinstance(ls, ColumnLargeScaleState)
    assert ls.T.shape == (6,)
    np.testing.assert_allclose(np.asarray(ls.theta_adv), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ls.qv_adv), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ls.omega), 0.0, atol=1e-8)


def test_extract_nonuniform_gives_finite_forcing():
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    # Impose a zonal temperature gradient (warmer to the east).
    n_lat, n_lon, nlev = T.shape
    lon_ramp = jnp.linspace(0.0, 10.0, n_lon)[None, :, None]
    T = T + lon_ramp
    # Also impose a zonal humidity ramp (drier east).
    q_v = q_v - 1e-4 * jnp.linspace(0.0, 5.0, n_lon)[None, :, None]
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(20.0)), col_index=(4, 8),
    )
    assert bool(jnp.all(jnp.isfinite(ls.theta_adv)))
    assert float(jnp.max(jnp.abs(ls.theta_adv))) > 0.0

    # Wiring/sign check: the extractor's theta_adv at the column equals
    # -(u·∂θ/∂x + v·∂θ/∂y) from the SAME operator gradient (v=0 here).
    from legoesm.atmosphere.dynamics.column_large_scale_extract import (
        _gradient_latlon_3d,
    )
    from legoesm.atmosphere.physics._shared import exner_function

    # Match the extractor's float64 cast of sigma_full for a bit-exact compare.
    sigma_full64 = jnp.asarray(sigma.sigma_full, dtype=T.dtype)
    theta = T / exner_function(p_s[..., None] * sigma_full64)
    th_x, th_y = _gradient_latlon_3d(theta, grid)
    expected = -(u * th_x + v * th_y)[4, 8, :]
    np.testing.assert_allclose(np.asarray(ls.theta_adv), np.asarray(expected), rtol=1e-10)
    # u>0 (eastward) up a warm-east gradient ⇒ cold advection (theta_adv < 0).
    assert bool(jnp.all(ls.theta_adv < 0.0))
    # Drier east ⇒ u>0 brings moister air ⇒ qv_adv > 0.
    assert bool(jnp.all(ls.qv_adv > 0.0))


def test_extract_feeds_build_column_scm_forcing():
    """The extracted state drives the iter-5 SCMForcing assembler end-to-end."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    T = T + jnp.linspace(0.0, 5.0, T.shape[1])[None, :, None]
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(30.0)), col_index=(4, 8),
    )
    forcing = build_column_scm_forcing(ls)  # omega present ⇒ subsidence set
    assert forcing.subsidence_w is not None
    assert forcing.theta_adv is not None
    # Coriolis at 30N.
    from legoesm import constants
    assert forcing.f_c == pytest.approx(2.0 * constants.Omega * 0.5, rel=1e-9)


def test_extract_jit():
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()

    def run(T_in):
        # col_index closed over as a static constant (documented contract).
        ls = extract_column_forcing_latlon(
            T=T_in, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
            lat_rad=0.5, col_index=(4, 8),
        )
        return ls.omega, ls.theta_adv, ls.qv_adv

    omega, th_adv, qv_adv = jax.jit(run)(T)
    for arr in (omega, th_adv, qv_adv):
        assert bool(jnp.all(jnp.isfinite(arr)))
