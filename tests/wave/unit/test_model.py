"""End-to-end wave step: shapes, masking, JIT parity and gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from legoesm.wave.config import WaveGSEConfig, WavePhysicsConfig, WaveTimeConfig
from legoesm.wave.model import WaveModel, prepare_env, validate_wave_environment


def _step(model, config, state, u, v):
    env = prepare_env(config, state.depth)
    return model.step(state.spectra, env, state.depth, state.mask, u, v)


def test_step_preserves_shape_and_land_mask(tiny_config, tiny_state):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    u = jnp.full((1, 6, 5), 8.0)
    v = jnp.full((1, 6, 5), 2.0)
    depth = tiny_state.depth.at[0, 0, 0].set(0.0)
    mask = depth > 0
    state = tiny_state._replace(depth=depth, mask=mask)
    out = _step(model, tiny_config, state, u, v)
    assert out.shape == state.spectra.shape
    assert np.all(np.asarray(out[0, 0, 0]) == 0.0)


def test_eager_and_filter_jit_agree(tiny_config, tiny_state):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    u = jnp.full((1, 6, 5), 8.0)
    v = jnp.full((1, 6, 5), 2.0)
    env = prepare_env(tiny_config, tiny_state.depth)
    step = eqx.filter_jit(
        lambda m, x: m.step(x, env, tiny_state.depth, tiny_state.mask, u, v))
    out_eager = model.step(
        tiny_state.spectra, env, tiny_state.depth, tiny_state.mask, u, v)
    out_jit = step(model, tiny_state.spectra)
    assert np.allclose(np.asarray(out_eager), np.asarray(out_jit), atol=1e-6)


def test_float32_state_and_network_survive_global_x64(tiny_config, tiny_state):
    with jax.enable_x64():
        model = WaveModel(tiny_config, key=jax.random.key(0))
        env = prepare_env(tiny_config, tiny_state.depth)
        wind_u = jnp.full((1, 6, 5), 8.0, dtype=jnp.float64)
        wind_v = jnp.full((1, 6, 5), 2.0, dtype=jnp.float64)

        assert model.parameterization.mlp.linear0.weight.dtype == jnp.float32
        assert model.parameterization.mlp.norm0.weight.dtype == jnp.float32
        out = model.step(
            tiny_state.spectra, env, tiny_state.depth, tiny_state.mask,
            wind_u, wind_v)
        assert out.dtype == tiny_state.spectra.dtype


def test_gradient_flows_to_input_and_network(tiny_config, tiny_state):
    model = WaveModel(tiny_config, key=jax.random.key(0))
    u = jnp.full((1, 6, 5), 8.0)
    v = jnp.full((1, 6, 5), 2.0)
    env = prepare_env(tiny_config, tiny_state.depth)

    def loss_spectra(x):
        return jnp.sum(model.step(x, env, tiny_state.depth, tiny_state.mask, u, v))

    grad = jax.grad(loss_spectra)(tiny_state.spectra)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert bool(jnp.any(grad != 0.0))

    def loss_model(m):
        return jnp.sum(m.step(
            tiny_state.spectra, env, tiny_state.depth, tiny_state.mask, u, v))

    model_grad = jax.grad(loss_model)(model)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(model_grad)
              if hasattr(leaf, "shape")]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0.0)) for leaf in leaves)


def test_step_applies_the_si_bridge_to_the_parameterization_output(
        tiny_config, tiny_state):
    """A unit source must move the spectrum by the adapter's 0.01 * dt.

    The parameterization's 0.001 is pinned in its own test; this one pins the
    adapter's 0.01, so doubling that factor fails here.
    """
    class _UnitParameterization(eqx.Module):
        def __call__(self, spectra, *args, **kwargs):
            return jnp.ones_like(spectra)

    config = tiny_config._replace(
        time=WaveTimeConfig(dt_global_s=3600, dt_x_s=0, dt_y_s=0, dt_th_s=0,
                            dt_k_s=0, dt_stot_s=3600, dt_gse_s=0),
        network=tiny_config.network._replace(parameterization="none",
                                             use_transport_networks=False),
        gse=WaveGSEConfig(enabled=False),
    )
    model = WaveModel(config, key=jax.random.key(0))
    model = eqx.tree_at(
        lambda m: m.parameterization, model, _UnitParameterization())
    zeros = jnp.zeros((1, 6, 5))
    out = _step(model, config, tiny_state, zeros, zeros)
    # The released total is 0.001 * 0.01; the 0.001 is pinned in the
    # parameterization test, so the bridge below is 0.01 * 3600 s.
    expected = float(tiny_state.spectra[0, 1, 1, 0, 0]) + 0.01 * 3600.0
    np.testing.assert_allclose(
        np.asarray(out[:, 1:-1, 1:-1]), expected, rtol=1e-6)


def test_model_step_with_gse_is_finite_and_changes_the_field(tiny_config, tiny_state):
    def run(enabled):
        config = tiny_config._replace(
            time=WaveTimeConfig(dt_gse_s=1800),
            gse=WaveGSEConfig(enabled=enabled, alpha_s=1.0, alpha_n=1.0,
                              periodic_x=True, substeps=2),
        )
        config.validate()
        model = WaveModel(config, key=jax.random.key(0))
        env = prepare_env(config, tiny_state.depth)
        if enabled:
            validate_wave_environment(config, env)
        u = jnp.full((1, 6, 5), 8.0)
        v = jnp.full((1, 6, 5), 2.0)
        spectra = tiny_state.spectra.at[0, 2, 2].set(0.5).at[0, 3, 2].set(0.2)
        return model.step(spectra, env, tiny_state.depth, tiny_state.mask, u, v)

    with_gse = run(True)
    without = run(False)
    assert bool(jnp.all(jnp.isfinite(with_gse)))
    # A non-uniform spectrum makes the nine-point average change the field
    # beyond the transport-only step; removing the GSE wiring fails here.
    assert not np.allclose(np.asarray(with_gse), np.asarray(without), atol=1e-9)


def test_physics_only_switch_off_is_identity_in_interior(tiny_config, tiny_state):
    config = tiny_config._replace(
        physics=WavePhysicsConfig(
            fachfa=0.0, facvx=0.0, facvy=0.0, facth=0.0, fdg=0.0),
        network=tiny_config.network._replace(parameterization="none",
                                             use_transport_networks=False),
        gse=WaveGSEConfig(enabled=False),
    )
    model = WaveModel(config, key=jax.random.key(0))
    u = jnp.full((1, 6, 5), 8.0)
    v = jnp.full((1, 6, 5), 2.0)
    out = _step(model, config, tiny_state, u, v)
    assert np.allclose(
        np.asarray(out[:, 1:-1, 1:-1]),
        np.asarray(tiny_state.spectra[:, 1:-1, 1:-1]), atol=1e-6)


def test_pure_physics_step_spectrum_and_depth_gradients_match_differences(tiny_config):
    """Actual full-step derivatives on a positive, nonflat wet branch."""
    from jax.test_util import check_grads
    from legoesm.wave.state import WaveState
    from legoesm.wave.config import WaveTimeConfig

    with jax.enable_x64():
        config = tiny_config._replace(
            time=WaveTimeConfig(dt_global_s=600, dt_x_s=600, dt_y_s=600,
                               dt_th_s=600, dt_k_s=0, dt_stot_s=0,
                               dt_gse_s=0),
            network=tiny_config.network._replace(parameterization="none",
                                                 use_transport_networks=False),
            gse=WaveGSEConfig(enabled=False),
        )
        model = WaveModel(config, key=jax.random.key(0))
        rng = np.random.default_rng(0)
        spectra = jnp.asarray(rng.uniform(.01, .03, (1,6,5,4,4)), dtype=jnp.float64)
        depth = jnp.asarray(rng.uniform(20., 30., (1,6,5)), dtype=jnp.float64)
        weights = jnp.asarray(rng.uniform(-1., 1., spectra.shape), dtype=jnp.float64)
        mask = jnp.ones(depth.shape, dtype=bool)
        wind = jnp.zeros(depth.shape, dtype=depth.dtype)

        def loss(spectrum, bathymetry):
            state = WaveState(spectrum, bathymetry, mask)
            result = model.advance(state, prepare_env(config, bathymetry), wind, wind)
            return jnp.sum(result.spectra*weights)

        derivatives = jax.jit(jax.grad(loss, argnums=(0,1)))(spectra, depth)
        for derivative in derivatives:
            assert bool(jnp.all(jnp.isfinite(derivative)))
            assert bool(jnp.any(derivative != 0))
        check_grads(loss, (spectra, depth), order=1, modes=("fwd","rev"),
                    eps=1e-6, rtol=1e-4, atol=1e-6)
