"""Flux and tendency kernels: zero divergence on uniform fields, dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.wave.config import WavePhysicsConfig
from legoesm.wave.grid import prepare_env
from legoesm.wave.physics.flux import calculate_flux, first_order_flux, quickest_flux
from legoesm.wave.physics.tendencies import (
    k_tendency,
    theta_tendency,
    x_tendency,
    y_tendency,
)


def _uniform_state(config, depth):
    env = prepare_env(config, depth)
    spectra = jnp.ones((1, 6, 5, 4, 4)) * 0.02
    return env, spectra


def test_quickest_uniform_field_has_zero_divergence(tiny_config, tiny_depth):
    env, spectra = _uniform_state(tiny_config, tiny_depth)
    cg = jnp.broadcast_to(env.group_velocity, spectra.shape)
    dx = jnp.broadcast_to(env.dlon * env.x_metric, spectra.shape)
    minus, plus = quickest_flux(spectra, cg, dx, target_dim=1, dt=3600.0)
    divergence = (minus - plus)[:, 1:-1] / dx[:, 1:-1]
    assert np.allclose(np.asarray(divergence), 0.0, atol=1e-9)


def test_first_order_upwind_uses_left_neighbour_for_positive_velocity():
    N = jnp.arange(1, 7, dtype=jnp.float32).reshape(1, 6, 1, 1, 1)
    Cg = jnp.ones_like(N) * 2.0
    mask = jnp.ones((1, 6, 1), dtype=jnp.float32)
    minus, _ = first_order_flux(N, Cg, mask, target_dim=1, mode="replicate")
    # FLN = Cg_b * N_upwind; with constant Cg=2 and left neighbour.
    assert np.allclose(np.asarray(minus) [0, 1:, 0, 0, 0], 2.0 * np.asarray(N)[0, :-1, 0, 0, 0])


def test_unknown_flux_scheme_raises():
    with pytest.raises(ValueError, match="Unknown flux scheme"):
        calculate_flux(jnp.ones((1, 4, 1, 1, 1)), jnp.ones((1, 4, 1, 1, 1)), scheme="nope")


def test_zero_depth_gradient_has_zero_refraction(tiny_config, tiny_depth):
    env, spectra = _uniform_state(tiny_config, tiny_depth)
    zero = jnp.zeros_like(jnp.broadcast_to(env.group_velocity, spectra.shape))[..., :1, :1]
    depth_5d = tiny_depth[..., None, None]
    # `fdg=0` removes the spherical curvature term; dd_dx=dd_dy=0 removes the
    # depth-refraction term regardless of the local DsDd value.
    physics = WavePhysicsConfig(fdg=0.0)
    theta = theta_tendency(spectra, 3600.0, env, physics, depth_5d, zero, zero)
    assert np.allclose(np.asarray(theta), 0.0, atol=1e-12)
    k_tend = k_tendency(spectra, 3600.0, env, physics, depth_5d, zero, zero)
    assert k_tend.shape == spectra.shape
    assert np.all(np.asarray(k_tend) == 0.0)


def test_quickest_nonuniform_profile_matches_waden():
    """Non-uniform 1-D profile; expected faces from WADEN 46aadc7.

    Generated with `waden.quickest_flux(N, Cg, 1.0, target_dim=4, dt=0.2,
    mode="replicate")` on a clean checkout at the pinned commit.  A uniform
    field cannot catch a donor/limiter/branch swap.
    """
    N = jnp.asarray([1.0, 2.0, 4.0, 3.0, 0.5]).reshape(1, 1, 1, 1, 5)
    Cg = jnp.asarray([0.5, 0.8, -0.3, 0.6, 0.2]).reshape(1, 1, 1, 1, 5)
    minus, plus = quickest_flux(
        N, Cg, jnp.asarray(1.0), target_dim=4, dt=0.2, mode="replicate")
    expected_minus = [0.5, 0.6499999762, 0.6959374547, 0.6000000238, 0.8393600583]
    expected_plus = [0.6499999762, 0.6959374547, 0.6000000238, 0.8393600583, 0.8393600583]
    np.testing.assert_allclose(np.asarray(minus).ravel(), expected_minus, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(plus).ravel(), expected_plus, rtol=1e-6)


def test_quickest_negative_transport_matches_waden():
    """Negative face velocities exercise the other donor/limiter branch."""
    N = jnp.asarray([1.0, 2.0, 4.0, 3.0, 0.5]).reshape(1, 1, 1, 1, 5)
    Cg = jnp.asarray([-0.5, -0.8, 0.3, -0.6, -0.2]).reshape(1, 1, 1, 1, 5)
    minus, plus = quickest_flux(
        N, Cg, jnp.asarray(1.0), target_dim=4, dt=0.2, mode="replicate")
    expected_minus = [
        -0.5, -0.9107475877, -1.0, -0.5602163076, -0.2000000030]
    expected_plus = [
        -0.9107475877, -1.0, -0.5602163076, -0.2000000030, -0.2000000030]
    np.testing.assert_allclose(np.asarray(minus).ravel(), expected_minus, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(plus).ravel(), expected_plus, rtol=1e-6)


def test_theta_negative_transport_matches_waden():
    from legoesm.wave.physics.flux import theta_flux

    N = jnp.asarray([1.0, 2.0, 4.0, 3.0, 0.5]).reshape(1, 1, 1, 5, 1)
    Cg = jnp.asarray([-0.5, -0.8, 0.3, -0.6, -0.2]).reshape(1, 1, 1, 5, 1)
    minus, plus = theta_flux(N, Cg, jnp.asarray(1.0), target_dim=3, dt=0.2)
    expected_minus = [-5.0, -10.0, -1.0, -0.5647162199, -0.2000000030]
    expected_plus = [-10.0, -1.0, -0.5647162199, -0.2000000030, -5.0]
    np.testing.assert_allclose(np.asarray(minus).ravel(), expected_minus, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(plus).ravel(), expected_plus, rtol=1e-6)


def test_first_order_upwind_uses_right_neighbour_for_negative_velocity():
    N = jnp.arange(1, 7, dtype=jnp.float32).reshape(1, 6, 1, 1, 1)
    Cg = -jnp.ones_like(N) * 2.0
    mask = jnp.ones((1, 6, 1), dtype=jnp.float32)
    minus, _ = first_order_flux(N, Cg, mask, target_dim=1, mode="replicate")
    # FLN = Cg_b * N_donor; at face i a negative velocity takes the right
    # cell N_i (the left-donor branch would give -2*N[i-1]).
    expected = -2.0 * np.asarray(N)[0, :, 0, 0, 0]
    np.testing.assert_allclose(np.asarray(minus)[0, :, 0, 0, 0], expected)


def test_theta_flux_nonuniform_profile_matches_waden():
    """Non-uniform direction profile; expected faces from WADEN 46aadc7."""
    from legoesm.wave.physics.flux import theta_flux

    N = jnp.asarray([1.0, 2.0, 4.0, 3.0, 0.5]).reshape(1, 1, 1, 5, 1)
    Cg = jnp.asarray([0.5, 0.8, -0.3, 0.6, 0.2]).reshape(1, 1, 1, 5, 1)
    minus, plus = theta_flux(N, Cg, jnp.asarray(1.0), target_dim=3, dt=0.2)
    expected_minus = [
        0.1749999970, 0.8794986606, 0.6959374547, 0.6000000238, 14.9999990463]
    expected_plus = [
        0.8794986606, 0.6959374547, 0.6000000238, 14.9999990463, 0.1749999970]
    np.testing.assert_allclose(np.asarray(minus).ravel(), expected_minus, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(plus).ravel(), expected_plus, rtol=1e-6)


def test_refraction_velocity_follows_the_depth_gradient(tiny_config, tiny_state):
    from legoesm.wave.physics.dispersion import calculate_depth_dispersion_derivative
    from legoesm.wave.physics.tendencies import theta_transport_velocity

    env = prepare_env(tiny_config, tiny_state.depth)
    depth_5d = tiny_state.depth[..., None, None]
    dd_dx = jnp.full((1, 6, 5, 1, 1), 0.1)
    dd_dy = jnp.full((1, 6, 5, 1, 1), 0.07)
    velocity = theta_transport_velocity(
        tiny_state.spectra, env, WavePhysicsConfig(fdg=0.0), depth_5d, dd_dx, dd_dy)
    dsdd = calculate_depth_dispersion_derivative(
        env.group_velocity, env.wavenum, env.freq, depth_5d)
    expected = -dsdd * (env.cos_theta * 0.1 - env.sin_theta * 0.07) / env.wavenum
    np.testing.assert_allclose(
        np.asarray(velocity), np.asarray(expected), rtol=1e-5, atol=1e-8)


def test_spherical_y_tendency_includes_the_latitude_metric(tiny_config, tiny_depth):
    env, spectra = _uniform_state(tiny_config, tiny_depth)
    ty = y_tendency(spectra, 3600.0, env, tiny_config.physics)
    # A uniform sea on a sphere: the cos(lat) metric alone makes the
    # latitude flux diverge.  Pin the exact value from the metric factors:
    # a uniform spectrum leaves the QUICKEST reconstruction at N, so the
    # face flux is Cg_b*N with the cos(lat) factor in both the velocity and
    # the spacing.
    phi = env.group_velocity * env.cos_theta * env.cos_lat
    phi_im1 = jnp.concatenate([phi[:, :, :1], phi[:, :, :-1]], axis=2)
    face_flux = 0.5 * (phi_im1 + phi) * spectra
    face_flux_plus = jnp.concatenate(
        [face_flux[:, :, 1:], face_flux[:, :, -1:]], axis=2)
    dy = env.dlat * env.y_metric * env.cos_lat
    expected = (face_flux - face_flux_plus) / dy
    np.testing.assert_allclose(
        np.asarray(ty), np.asarray(expected), rtol=1e-5, atol=1e-12)
    assert bool(jnp.any(ty != 0.0))
    cart = tiny_config._replace(
        grid=tiny_config.grid._replace(coordinate_type="Cartesian"))
    env_c, spectra_c = _uniform_state(cart, tiny_depth)
    ty_c = y_tendency(spectra_c, 3600.0, env_c, cart.physics)
    assert np.allclose(np.asarray(ty_c), 0.0, atol=1e-7)


def test_nonfinite_spectrum_is_not_silently_zeroed(tiny_config, tiny_depth):
    env, spectra = _uniform_state(tiny_config, tiny_depth)
    mask = jnp.ones((1, 6, 5), dtype=bool)
    bad = spectra.at[0, 2, 2, 0, 0].set(jnp.nan)
    tendency = x_tendency(bad, 3600.0, env, tiny_config.physics, land_mask=mask)
    assert bool(jnp.any(~jnp.isfinite(tendency)))


def test_spatial_tendencies_use_longitude_and_latitude_axes(tiny_config, tiny_depth):
    grid = tiny_config.grid._replace(
        coordinate_type="Cartesian",
        lon_deg=(0.0, 1.0, 2.0, 3.0, 4.0),
        lat_deg=(0.0, 1.0, 2.0, 3.0, 4.0),
    )
    config = tiny_config._replace(grid=grid)
    env = prepare_env(config, jnp.full((1, 5, 5), 50.0))
    lon_profile = jnp.asarray([1.0, 2.0, 4.0, 3.0, 0.5]).reshape(1, 5, 1, 1, 1)
    lat_profile = jnp.asarray([1.0, 2.0, 4.0, 3.0, 0.5]).reshape(1, 1, 5, 1, 1)
    base = jnp.ones((1, 5, 5, 4, 4), dtype=jnp.float32)
    tx = x_tendency(base * lon_profile, 3600.0, env, config.physics)
    ty = y_tendency(base * lon_profile, 3600.0, env, config.physics)
    assert bool(jnp.any(tx != 0.0))
    assert np.allclose(np.asarray(ty), 0.0, atol=1e-7)
    tx2 = x_tendency(base * lat_profile, 3600.0, env, config.physics)
    ty2 = y_tendency(base * lat_profile, 3600.0, env, config.physics)
    assert np.allclose(np.asarray(tx2), 0.0, atol=1e-7)
    assert bool(jnp.any(ty2 != 0.0))


def test_uniform_field_zero_spatial_tendency(tiny_config, tiny_depth):
    config = tiny_config._replace(grid=tiny_config.grid._replace(coordinate_type="Cartesian"))
    env, spectra = _uniform_state(config, tiny_depth)
    mask = jnp.ones((1, 6, 5), dtype=bool)
    tx = x_tendency(spectra, 3600.0, env, tiny_config.physics, land_mask=mask)
    ty = y_tendency(spectra, 3600.0, env, tiny_config.physics, land_mask=mask)
    assert np.allclose(np.asarray(tx), 0.0, atol=1e-6)
    assert np.allclose(np.asarray(ty), 0.0, atol=1e-6)


def test_boundary_mask_accepts_3d_and_5d_shapes():
    N = jnp.asarray([1.0, 2.0, 4.0, 3.0, 0.5, 2.0]).reshape(1, 6, 1, 1, 1)
    Cg = jnp.ones_like(N)
    land = jnp.ones((1, 6, 1), dtype=bool)
    no_mask, _ = quickest_flux(
        N, Cg, jnp.asarray(1.0), target_dim=1, land_mask=land,
        dt=0.2, mode="replicate")
    mask3 = jnp.zeros((1, 6, 1), dtype=bool).at[0, 2, 0].set(True)
    minus3, plus3 = quickest_flux(
        N, Cg, jnp.asarray(1.0), target_dim=1, land_mask=land,
        boundary_mask=mask3, dt=0.2, mode="replicate")
    minus5, plus5 = quickest_flux(
        N, Cg, jnp.asarray(1.0), target_dim=1, land_mask=land,
        boundary_mask=mask3[..., None, None], dt=0.2, mode="replicate")
    np.testing.assert_array_equal(np.asarray(minus3), np.asarray(minus5))
    np.testing.assert_array_equal(np.asarray(plus3), np.asarray(plus5))
    # The prescribed face takes the first-order fallback: 2.0 at longitude 2,
    # versus the unconstrained QUICKEST face 2.64.
    assert np.isclose(float(np.asarray(minus3).ravel()[2]), 2.0, atol=1e-6)
    assert np.isclose(float(np.asarray(no_mask).ravel()[2]), 2.64, atol=1e-6)
    assert bool(jnp.any(minus3 != no_mask))


def test_theta_flux_dispatch_accepts_land_mask():
    spectra = jnp.ones((1,3,3,4,4),dtype=jnp.float32)
    minus,plus = calculate_flux(spectra,jnp.ones_like(spectra),scheme="theta",target_dim=3,
                                Dgrid=jnp.asarray(1.),land_mask=jnp.ones((1,3,3),dtype=bool),dt=.1)
    np.testing.assert_allclose(minus,plus,atol=1e-7)
