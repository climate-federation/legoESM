"""Physical-cell, spherical-budget, boundary and persistent-phase contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.wave.config import (
    WaveConfig,
    WaveGSEConfig,
    WaveGridConfig,
    WaveNetworkConfig,
    WaveTimeConfig,
)
from legoesm.wave.grid import spectral_integral
from legoesm.wave.model import WaveModel, prepare_env
from legoesm.wave.physics.boundary import apply_boundary_condition
from legoesm.wave.physics.integrator import prepare_integration_schedule
from legoesm.wave.physics.tendencies import y_tendency, k_tendency
from legoesm.wave.state import init_wave_state


def test_periodic_physical_edge_columns_are_evolved():
    config = WaveConfig(
        WaveGridConfig(tuple(map(float,np.arange(0., 360., 45.))), (45., 0., -45.),
                       (.05, .055, .0605), (np.pi/2, 3*np.pi/2)),
        time=WaveTimeConfig(dt_global_s=600, dt_x_s=600, dt_y_s=0,
                           dt_th_s=0, dt_k_s=0, dt_stot_s=0, dt_gse_s=0),
        network=WaveNetworkConfig(parameterization="none", use_transport_networks=False),
        gse=WaveGSEConfig(enabled=False),
    )
    state = init_wave_state(jnp.full((1, 8, 3), 2500.), n_dir=2, n_freq=3)
    state = state._replace(spectra=jnp.ones_like(state.spectra))
    env = prepare_env(config, state.depth)
    model = WaveModel(config, key=jax.random.key(0))
    wind = jnp.zeros(state.mask.shape)
    out = model.advance(state, env, wind, wind)
    np.testing.assert_allclose(out.spectra, state.spectra, rtol=2e-6, atol=1e-7)
    assert int(out.spatial_phase) == 1


def test_spherical_meridional_transport_preserves_area_integral(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    spectra = tiny_state.spectra * (1+jnp.arange(5, dtype=jnp.float32)[None, None, :, None, None])
    rate = y_tendency(spectra, 10., env, tiny_config.physics, land_mask=tiny_state.mask,
                      stencil_mode="circular")
    before = spectral_integral(spectra, env, tiny_state.mask)
    after = spectral_integral(spectra+10*rate, env, tiny_state.mask)
    np.testing.assert_allclose(after, before, rtol=2e-6)


def test_zero_prescribed_boundary_is_enforced(tiny_state):
    fixed = jnp.zeros_like(tiny_state.mask).at[:, 2, 2].set(True)
    out = apply_boundary_condition(tiny_state.spectra, jnp.zeros_like(tiny_state.spectra),
                                   tiny_state.mask, fixed)
    assert np.count_nonzero(np.asarray(out[:, 2, 2])) == 0
    np.testing.assert_array_equal(np.asarray(out[:, 1, 1]), np.asarray(tiny_state.spectra[:, 1, 1]))


def test_simultaneous_spatial_events_count_once():
    schedule = prepare_integration_schedule(3600, 1800, 1800, 0, 0, 0, jnp.float32, True)
    np.testing.assert_array_equal(schedule["yfirst_per_step"], [True, False])
    resumed = prepare_integration_schedule(3600, 1800, 1800, 0, 0, 0, jnp.float32, True,
                                           spatial_phase=1)
    np.testing.assert_array_equal(resumed["yfirst_per_step"], [False, True])


def test_stationary_depth_has_no_physical_frequency_shift(tiny_config, tiny_state):
    depth = tiny_state.depth.at[:, 2, 2].set(5.)
    env = prepare_env(tiny_config, depth)
    slope = jnp.ones_like(depth)[..., None, None]
    out = k_tendency(tiny_state.spectra, 600., env, tiny_config.physics,
                     depth[..., None, None], slope, slope)
    assert out.shape == tiny_state.spectra.shape
    np.testing.assert_array_equal(out, jnp.zeros_like(out))


def test_public_model_preserves_prescribed_zero_through_all_phases(tiny_config, tiny_state):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    env = prepare_env(tiny_config, tiny_state.depth)
    fixed = jnp.zeros_like(tiny_state.mask).at[:,2,2].set(True)
    wind = jnp.ones(tiny_state.mask.shape)*8
    result = model.advance(tiny_state, env, wind, wind,
                           jnp.zeros_like(tiny_state.spectra), boundary_mask=fixed)
    np.testing.assert_array_equal(result.spectra[:,2,2], jnp.zeros_like(result.spectra[:,2,2]))
    assert bool(jnp.any(result.spectra[:,1,1] != tiny_state.spectra[:,1,1]))


def test_clipping_audit_detects_planted_sink(tiny_config, tiny_state):
    from legoesm.wave.physics.integrator import model_advance
    config = tiny_config._replace(
        time=WaveTimeConfig(dt_global_s=3600, dt_x_s=0, dt_y_s=0,
                            dt_th_s=0, dt_k_s=0, dt_stot_s=3600,
                            dt_gse_s=0),
        gse=WaveGSEConfig(enabled=False),
    )
    env = prepare_env(config,tiny_state.depth)
    # The integrator is SI: N -> -N after one step, so the clipped mass is
    # exactly the initial integral.
    source = -2*tiny_state.spectra/3600
    spectrum, clipped = model_advance(tiny_state.spectra, source,
        tiny_state.depth[...,None,None], env, config, tiny_state.mask,
        jnp.zeros_like(tiny_state.spectra), audit_clipping=True)
    np.testing.assert_array_equal(spectrum,jnp.zeros_like(spectrum))
    np.testing.assert_allclose(
        clipped, spectral_integral(tiny_state.spectra, env, tiny_state.mask),
        rtol=2e-6)
