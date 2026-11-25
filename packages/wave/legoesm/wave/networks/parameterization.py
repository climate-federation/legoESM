"""Learned source/sink parameterization of the wave spectrum.

Port of ``MLPParameterization`` (NeuralSWM
``neuralwave_jax/parameterizations.py``).  The network consumes the
spectrum plus three feature maps (wind projection, depth-gradient
projection, and wavenumber-plus-frequency) and returns a source tendency
in the same spectral shape.  The ``0.001`` output scale is part of the
trained model's units.

The source also implements sampled and U-Net variants; only the MLP and
the zero (physics-only) parameterization are needed by the released
configuration and are ported here.  A sampling variant changes only the
gradient of unsampled columns, not the forward value, so it is not
required for coupling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.wave.config import WaveNetworkConfig
from legoesm.wave.constants import SOURCE_TERM_SCALE
from legoesm.wave.grid import WaveEnv
from legoesm.wave.networks.embedding import (
    sdn_features,
    wind_features,
    ww_inter_features,
)
from legoesm.wave.networks.mlp import SourceTermMLP


def _flatten(feature_map: jax.Array) -> jax.Array:
    return jnp.reshape(feature_map, (feature_map.shape[0], -1))


class ZeroParameterization(eqx.Module):
    """Physics-only model: no neural source term."""

    def __call__(self, spectra, env, depth, bathymetry, wind_u, wind_v, mask, **kwargs):
        del env, depth, bathymetry, wind_u, wind_v, mask
        return jnp.zeros_like(jnp.asarray(spectra))


class SourceTermParameterization(eqx.Module):
    """MLP source-term parameterization of the released weights."""

    mlp: SourceTermMLP
    config: WaveNetworkConfig = eqx.field(static=True)

    def __init__(self, config: WaveNetworkConfig, *, key: jax.Array):
        self.config = config
        out_channels = config.mlp_input_channels // 4
        self.mlp = SourceTermMLP(
            config.mlp_input_channels,
            out_channels,
            hidden_0=config.mlp_hidden_channels_0,
            hidden_1=config.mlp_hidden_channels_1,
            hidden_2=config.mlp_hidden_channels_2,
            layernorm_eps=config.layernorm_eps,
            key=key,
        )

    def __call__(self, spectra, env, depth, bathymetry, wind_u, wind_v, mask, **kwargs):
        del depth, kwargs
        spectra = jnp.asarray(spectra)
        # The released network and default wave state are float32.  A coupled
        # x64 atmosphere may supply float64 winds or bathymetry; keep the
        # source-term evaluation on the wave state's precision policy.
        bathymetry = jnp.asarray(bathymetry, dtype=spectra.dtype)
        wind_u = jnp.asarray(wind_u, dtype=spectra.dtype)
        wind_v = jnp.asarray(wind_v, dtype=spectra.dtype)
        batch, nlon, nlat, n_dir, n_freq = spectra.shape
        n_points = batch * nlon * nlat

        spectra_masked = spectra * mask[..., None, None]
        spectra_points = jnp.reshape(spectra_masked, (n_points, 1, n_dir, n_freq))

        # Feature order matters: the source's ``feature_generators`` dict is
        # rebuilt in SORTED key order by ``nnx.jit`` ("sdn", "wind", "wwInter"),
        # and BOTH training and the validated inference run under that jit.
        # Eager execution keeps insertion order ("wind", "sdn", ...) and is
        # therefore NOT the trained model.  Keep the trained order.
        features = [
            _flatten(spectra_points),
            _flatten(sdn_features(env, bathymetry)),
            _flatten(wind_features(env, wind_u, wind_v)),
            _flatten(ww_inter_features(env)),
        ]
        combined = jnp.concatenate(features, axis=1)
        expected = self.config.mlp_input_channels
        if combined.shape[1] != expected:
            raise ValueError(
                f"wave source network input mismatch: expected {expected}, "
                f"got {combined.shape[1]}")

        stot_flat = self.mlp(combined)
        stot = jnp.reshape(stot_flat, (batch, nlon, nlat, n_dir, n_freq))
        stot = stot * mask[..., None, None]
        return jnp.asarray(stot * SOURCE_TERM_SCALE, dtype=spectra.dtype)


__all__ = ["SourceTermParameterization", "ZeroParameterization"]
