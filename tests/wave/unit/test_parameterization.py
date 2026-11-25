"""Source-term parameterization: zero path, trained scale and feature order."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import legoesm.wave.networks.parameterization as parameterization_module
from legoesm.wave.grid import prepare_env
from legoesm.wave.networks.parameterization import (
    SourceTermParameterization,
    ZeroParameterization,
)


def _inputs(tiny_config, tiny_state):
    env = prepare_env(tiny_config, tiny_state.depth)
    u = jnp.full(tiny_state.mask.shape, 8.0)
    v = jnp.full(tiny_state.mask.shape, 2.0)
    mask = jnp.ones(tiny_state.mask.shape, dtype=bool)
    return env, tiny_state.spectra, tiny_state.depth, u, v, mask


def test_zero_parameterization_is_exactly_zero(tiny_config, tiny_state):
    env, spectra, depth, u, v, mask = _inputs(tiny_config, tiny_state)
    out = ZeroParameterization()(spectra, env, depth, depth, u, v, mask)
    assert out.shape == spectra.shape
    assert np.all(np.asarray(out) == 0.0)


class _ConstantMLP(eqx.Module):
    def __call__(self, x):
        return jnp.ones((x.shape[0], x.shape[1] // 4), dtype=x.dtype)


def test_source_term_applies_the_trained_scale_exactly_once(
        tiny_config, tiny_state):
    env, spectra, depth, u, v, mask = _inputs(tiny_config, tiny_state)
    param = SourceTermParameterization(tiny_config.network, key=jax.random.key(0))
    param = eqx.tree_at(lambda p: p.mlp, param, _ConstantMLP())
    out = param(spectra, env, depth, depth, u, v, mask)
    assert np.allclose(np.asarray(out), 1e-3, rtol=1e-6)


class _SelectChannelBlock(eqx.Module):
    start: int = eqx.field(static=True)

    def __call__(self, x):
        return jnp.repeat(x[:, self.start:self.start + 1], x.shape[1] // 4, axis=1)


def test_feature_order_is_spectra_sdn_wind_ww(tiny_config, tiny_state, monkeypatch):
    env, spectra, depth, u, v, mask = _inputs(tiny_config, tiny_state)
    spectra = jnp.ones_like(spectra)
    n_points = int(spectra.shape[0] * spectra.shape[1] * spectra.shape[2])
    values = (1.0, 10.0, 100.0, 1000.0)  # spectra, sdn, wind, wwInter

    monkeypatch.setattr(
        parameterization_module, "sdn_features",
        lambda env, bathymetry: jnp.full((n_points, 1, 4, 4), values[1]))
    monkeypatch.setattr(
        parameterization_module, "wind_features",
        lambda env, u, v: jnp.full((n_points, 1, 4, 4), values[2]))
    monkeypatch.setattr(
        parameterization_module, "ww_inter_features",
        lambda env: jnp.full((n_points, 1, 4, 4), values[3]))

    param = SourceTermParameterization(tiny_config.network, key=jax.random.key(0))
    block = 16  # n_dir * n_freq for the tiny grid
    for index, expected in enumerate(values):
        selected = eqx.tree_at(
            lambda p: p.mlp, param, _SelectChannelBlock(start=index * block))
        out = selected(spectra, env, depth, depth, u, v, mask)
        assert np.allclose(np.asarray(out), expected * 1e-3, rtol=1e-6), (
            f"feature block {index} is not in the trained order")
