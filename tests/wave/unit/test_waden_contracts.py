"""WADEN scientific regressions: dispersion AD, shore faces, frequency budgets.

Cases and tolerances are inherited from WADEN 46aadc7 and the associated
NeuralSWM test/direct/test_contracts.py; see the committed preregistration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.wave.config import WaveConfig, WaveGridConfig, WaveNetworkConfig
from legoesm.wave.grid import prepare_env
from legoesm.wave.physics.dispersion import solve_wavenumber
from legoesm.wave.physics.flux import quickest_flux
from legoesm.wave.physics.tendencies import neural_k_tendency


def test_dispersion_has_physical_shallow_depth_derivative():
    with jax.enable_x64():
        freq = jnp.asarray([0.1], dtype=jnp.float64)
        depth = jnp.full((1, 1, 1, 1, 1), 5.0, dtype=jnp.float64)
        derivative = jax.grad(lambda d: solve_wavenumber(freq, d).sum())(depth)
        assert bool(jnp.all(derivative < 0))
        check_grads(lambda d: solve_wavenumber(freq, d), (depth,), order=2)


@pytest.mark.parametrize("sign", [-1, 1])
def test_thin_island_uses_incident_sea_speed(sign):
    spectra = jnp.array([2., 2., 100., 3., 3.], dtype=jnp.float32).reshape(1, 5, 1, 1, 1)
    wet = jnp.array([True, True, False, True, True]).reshape(spectra.shape)
    speed = sign * jnp.array([1., 1., 0., .3, .3], dtype=jnp.float32).reshape(spectra.shape)
    minus, plus = quickest_flux(spectra, speed, jnp.asarray(1.), target_dim=1,
                               land_mask=wet, dt=.1, mode="replicate")
    expected = [2., 0.] if sign == 1 else [0., -.9]
    np.testing.assert_allclose(np.asarray(minus).ravel()[[2, 3]], expected, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(plus)[:, :-1], np.asarray(minus)[:, 1:])


def _frequency_case(initial, coefficient):
    freq = np.asarray(.03453 * 1.1 ** np.arange(len(initial)), dtype=np.float32)
    config = WaveConfig(
        WaveGridConfig((0., 1., 2.), (2., 1., 0.), tuple(freq), (0., np.pi)),
        network=WaveNetworkConfig(parameterization="none", use_transport_networks=False),
    )
    depth = jnp.full((1, 3, 3), 100., dtype=jnp.float32)
    env = prepare_env(config, depth)
    shape = (1, 3, 3, 2, len(initial))
    spectra = jnp.broadcast_to(jnp.asarray(initial, dtype=jnp.float32), shape)
    full = jnp.ones((1, 3, 3, 1, len(initial)), dtype=jnp.float32)
    env = env._replace(wavenum=full, group_velocity=full,
                       sin_theta=jnp.zeros_like(env.sin_theta),
                       cos_theta=jnp.ones_like(env.cos_theta))
    raw = jnp.arctanh(jnp.asarray(coefficient, dtype=jnp.float32))

    def network(inputs):
        values = jnp.stack((raw, jnp.zeros_like(raw)), axis=-1).ravel()
        return jnp.broadcast_to(values, (inputs.shape[0], values.size))

    zero = jnp.zeros((1, 3, 3, 1, 1), dtype=jnp.float32)
    options = dict(env=env, depth=depth[..., None, None], dd_dx=zero, dd_dy=zero,
                   network_fn=network,
                   dsdd_fn=lambda cg, k, f, d: jnp.zeros_like(k))
    return spectra, options, freq


def test_closed_frequency_endpoints_do_not_wrap():
    """A wrapped high face would leak the last-bin flux; the closed form must not.

    Direct check of the E*df face budget: spectra [0, 1, 0], coefficients
    [0, -0.431, 0.283], unit widths and dt.  The first coefficient is unused
    (face speeds are ``coefficient[..., 1:]``), so the faces are
    [-0.431, 0.283] and the closed tendency is [0.431, -0.714, 0.283]; a wrap
    changes the last entry.
    """
    from legoesm.wave.physics.tendencies import frequency_flux_tendency

    spectra = jnp.asarray([0.0, 1.0, 0.0]).reshape(1, 1, 1, 1, 3)
    coefficient = jnp.asarray([0.0, -0.431, 0.283]).reshape(1, 1, 1, 1, 3)
    widths = jnp.ones((3,))
    tendency = frequency_flux_tendency(spectra, coefficient, 1.0, widths)
    np.testing.assert_allclose(
        np.asarray(tendency).ravel(), [0.431, -0.714, 0.283], rtol=1e-6, atol=1e-7)
    assert np.isclose(float(jnp.sum(tendency * widths)), 0.0, atol=1e-7)


def test_learned_tendency_uses_the_full_component_step():
    """WADEN's neural_dt is the full step, not the analytical substep."""
    spectra, options, _ = _frequency_case([0., 1., 0.], [0., -.431, .283])
    one = neural_k_tendency(spectra, dt=1.0, neural_dt=1.0, **options)
    two = neural_k_tendency(spectra, dt=1.0, neural_dt=2.0, **options)
    np.testing.assert_allclose(np.asarray(two), 0.5 * np.asarray(one), rtol=1e-6)
    assert bool(jnp.any(one != 0.0))


def test_learned_theta_tendency_uses_the_full_component_step():
    """The theta path shares the neural_dt divisor with the k path."""
    from legoesm.wave.physics.tendencies import neural_theta_tendency

    spectra, options, _ = _frequency_case([1., 0., 0.], [.283, -.431, 0.])
    direction_profile = jnp.asarray([1.0, 2.0]).reshape(1, 1, 1, 2, 1)
    varied = spectra * direction_profile
    one = neural_theta_tendency(varied, dt=1.0, neural_dt=1.0, **options)
    two = neural_theta_tendency(varied, dt=1.0, neural_dt=2.0, **options)
    np.testing.assert_allclose(np.asarray(two), 0.5 * np.asarray(one), rtol=1e-6)
    assert bool(jnp.any(one != 0.0))


def test_sampling_with_full_ratio_matches_plain_k_tendency():
    from legoesm.wave.physics.tendencies import sampling_neural_k_tendency

    spectra, options, _ = _frequency_case([0., 1., 0.], [0., -.431, .283])
    plain = neural_k_tendency(spectra, dt=1.0, **options)
    sampled = sampling_neural_k_tendency(
        spectra, dt=1.0, sample_ratio=1.0, rng_key=jax.random.key(0), **options)
    np.testing.assert_allclose(
        np.asarray(sampled), np.asarray(plain), rtol=1e-5, atol=1e-8)


def test_nonuniform_frequency_budget_and_positivity():
    spectra, options, freq = _frequency_case([.2, 1., .6, .4, .1], [.8, -.9, .9, -.8, .7])
    edges = np.r_[freq[0] - (freq[1]-freq[0])/2,
                  (freq[1:] + freq[:-1])/2, freq[-1]+(freq[-1]-freq[-2])/2]
    widths = np.diff(edges)
    before = np.sum(np.asarray(spectra, dtype=float) * widths, axis=-1)
    for _ in range(100):
        tendency = neural_k_tendency(spectra, dt=3600, **options)
        assert tendency.shape == spectra.shape
        spectra = spectra + 1800 * tendency
        assert np.asarray(spectra).min() >= -1e-7
    after = np.sum(np.asarray(spectra, dtype=float) * widths, axis=-1)
    assert np.max(np.abs(after / before - 1)) < 2e-6
