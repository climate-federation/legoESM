"""Integrated wave parameters and Stokes drift."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.wave.grid import prepare_env
from legoesm.wave.indices import stokes_drift, wave_parameters


def _single_bin_spectrum(state, direction_index=0, frequency_index=1, energy=1.0):
    spectra = jnp.zeros_like(state.spectra)
    return spectra.at[..., direction_index, frequency_index].set(energy)


def test_significant_wave_height_of_one_bin(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    energy = 2.0
    spectra = _single_bin_spectrum(tiny_state, energy=energy)
    freq = jnp.asarray(tiny_config.grid.freq_hz)
    df0 = float((freq[2] - freq[0])/2)  # midpoint width of frequency bin 1
    dtheta = 2.0 * np.pi / len(tiny_config.grid.theta_rad)
    m0 = energy * df0 * dtheta
    params = wave_parameters(spectra, env)
    assert np.allclose(np.asarray(params["swh"]), 4.0 * np.sqrt(m0), rtol=1e-5)


def test_mean_direction_reports_direction_from(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    spectra = _single_bin_spectrum(tiny_state)
    params = wave_parameters(spectra, env)
    phi = 7.5  # direction the wave travels towards, clockwise from north
    expected_from = (phi + 180.0) % 360.0
    assert np.allclose(np.asarray(params["mwd"]), expected_from, atol=1e-4)


def test_zero_spectrum_gives_zero_swh_and_stokes(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    spectra = jnp.zeros_like(tiny_state.spectra)
    params = wave_parameters(spectra, env)
    stokes = stokes_drift(spectra, env, tiny_state.depth)
    assert stokes["stokes_magnitude"].shape == spectra.shape[:3]
    # A calm sea returns exactly zero height (zero subgradient at m0 == 0).
    assert np.all(np.asarray(params["swh"]) == 0.0)
    assert np.all(np.asarray(stokes["stokes_magnitude"]) == 0.0)


def test_stokes_components_follow_the_propagation_direction(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    spectra = _single_bin_spectrum(tiny_state)
    stokes = stokes_drift(spectra, env, tiny_state.depth)
    phi = np.deg2rad(7.5)
    east = float(np.asarray(stokes["stokes_east"]).reshape(-1)[10])
    north = float(np.asarray(stokes["stokes_north"]).reshape(-1)[10])
    ratio = np.arctan2(east, north)
    assert np.isclose(ratio, phi, atol=1e-3)
    assert float(np.asarray(stokes["stokes_magnitude"]).reshape(-1)[10]) > 0.0


def test_diagnostic_uses_midpoint_widths_including_endpoints(tiny_config, tiny_state):
    """Pin the WADEN midpoint finite-volume integral on a flat spectrum."""
    from legoesm.wave.grid import frequency_cell_widths, prepare_env

    env = prepare_env(tiny_config, tiny_state.depth)
    flat = jnp.ones_like(tiny_state.spectra)
    freq = np.asarray(tiny_config.grid.freq_hz, dtype=float)
    midpoint = np.asarray(frequency_cell_widths(jnp.asarray(freq)))
    expected_m0 = 2.0 * np.pi * midpoint.sum()
    params = wave_parameters(flat, env)
    assert np.allclose(np.asarray(params["m0"]), expected_m0, rtol=1e-6)
    assert np.allclose(np.asarray(params["swh"]), 4.0 * np.sqrt(expected_m0), rtol=1e-6)


def test_calm_sea_has_zero_direction_and_bounded_gradients(tiny_config, tiny_state):
    import jax

    env = prepare_env(tiny_config, tiny_state.depth)
    zero = jnp.zeros_like(tiny_state.spectra)
    params = wave_parameters(zero, env)
    assert np.all(np.asarray(params["mwd"]) == 0.0)
    assert np.all(np.asarray(params["mwp"]) == 0.0)
    for name in ("swh", "mwp"):
        derivative = jax.grad(
            lambda s, key=name: wave_parameters(s, env)[key].sum())(zero)
        assert bool(jnp.all(jnp.isfinite(derivative)))
        assert float(jnp.max(jnp.abs(derivative))) == 0.0


def test_isotropic_sea_has_no_mean_direction(tiny_config, tiny_state):
    """A directionally symmetric sea has no mean direction.

    With equal energy in every direction the resultant is zero up to
    roundoff.  Without the resultant guard, atan2 picks an arbitrary
    direction and its derivative is huge.
    """
    import jax

    env = prepare_env(tiny_config, tiny_state.depth)
    isotropic = jnp.full_like(tiny_state.spectra, 0.01)
    params = wave_parameters(isotropic, env)
    assert np.all(np.asarray(params["mwd"]) == 0.0)
    derivative = jax.grad(
        lambda s: wave_parameters(s, env)["mwd"].sum())(isotropic)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(jnp.max(jnp.abs(derivative))) == 0.0


def test_sub_floor_sea_is_treated_as_calm(tiny_config, tiny_state):
    """Below the m0 floor the period and direction are zero with zero gradient.

    The floor is 1e-15 m^2; this spectrum has m0 ~ 1.4e-21, i.e. SWH far below
    any physical wave.  Without the shared threshold the period derivative
    would be ~1e14.
    """
    import jax

    env = prepare_env(tiny_config, tiny_state.depth)
    tiny = jnp.full_like(tiny_state.spectra, 1e-20)
    params = wave_parameters(tiny, env)
    assert np.all(np.asarray(params["mwp"]) == 0.0)
    assert np.all(np.asarray(params["mwd"]) == 0.0)
    derivative = jax.grad(
        lambda s: wave_parameters(s, env)["mwp"].sum())(tiny)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(jnp.max(jnp.abs(derivative))) == 0.0


def test_stokes_rejects_ambiguous_depth_shapes(tiny_config, tiny_state):
    from legoesm.wave.grid import prepare_env

    env = prepare_env(tiny_config, tiny_state.depth)
    with pytest.raises(ValueError, match="stokes_drift expects depth"):
        stokes_drift(tiny_state.spectra, env, jnp.ones((6, 5, 1)))
    with pytest.raises(ValueError, match="wave_parameters expects spectra"):
        wave_parameters(jnp.ones((1, 6, 5, 4)), env)


def test_land_stokes_and_calm_gradient_are_finite(tiny_config, tiny_state):
    import jax
    depth = tiny_state.depth.at[:, 2, 2].set(0.)
    env = prepare_env(tiny_config, depth)
    spectrum = jnp.zeros_like(tiny_state.spectra)
    stokes = stokes_drift(spectrum, env, depth)
    assert bool(jnp.all(jnp.isfinite(stokes["stokes_magnitude"])))
    derivative = jax.grad(lambda s: stokes_drift(s, env, depth)["stokes_magnitude"].sum())(spectrum)
    assert bool(jnp.all(jnp.isfinite(derivative)))
