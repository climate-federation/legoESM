"""Assembled wave model: encoder, learned tendencies, integrator.

Port of ``ModelArchitecture`` (NeuralSWM
``neuralwave_jax/architecture.py``) without ``gin`` and without the
PyTorch weight-conversion path.  The forward pass is, in order:

1. build the static grid metrics and the dispersion fields for ``depth``;
2. mask the spectrum on land;
3. evaluate the source-term parameterization (or zeros);
4. integrate one global step with the analytic or learned refraction
   tendencies;
5. mask land again.

The encoder and decoder of the source are identity transforms apart from
the land mask, so they are folded into steps 2 and 5.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.wave.config import WaveConfig
from legoesm.wave.constants import SOURCE_TIME_SCALE
from legoesm.wave.grid import WaveEnv, prepare_env
from legoesm.wave.networks.mlp import MLP
from legoesm.wave.networks.parameterization import (
    SourceTermParameterization,
    ZeroParameterization,
)
from legoesm.wave.physics.boundary import apply_boundary_condition
from legoesm.wave.physics.gse import GSESpatialAveraging
from legoesm.wave.physics.integrator import model_advance, spatial_event_count
from legoesm.wave.state import WaveState
from legoesm.wave.physics.tendencies import (
    k_tendency,
    neural_k_tendency,
    neural_theta_tendency,
    theta_tendency,
)


class WaveModel(eqx.Module):
    """Differentiable spectral wave model."""

    parameterization: eqx.Module
    theta_transport_network: Optional[MLP]
    k_transport_network: Optional[MLP]
    config: WaveConfig = eqx.field(static=True)

    def __init__(self, config: WaveConfig, *, key: jax.Array):
        config.validate()
        self.config = config
        key_param, key_theta, key_k = jax.random.split(key, 3)

        if config.network.parameterization == "mlp":
            self.parameterization = SourceTermParameterization(config.network, key=key_param)
        elif config.network.parameterization == "none":
            self.parameterization = ZeroParameterization()
        else:
            raise ValueError(f"unknown wave parameterization {config.network.parameterization!r}")

        if config.network.use_transport_networks:
            self.theta_transport_network = MLP(
                config.network.transport_input_channels,
                config.network.transport_output_vectors,
                depth=2,
                hidden_dim=config.network.transport_hidden_channels,
                use_layernorm=True,
                layernorm_eps=config.network.layernorm_eps,
                key=key_theta,
            )
            self.k_transport_network = MLP(
                config.network.transport_input_channels,
                config.network.transport_output_vectors,
                depth=2,
                hidden_dim=config.network.transport_hidden_channels,
                use_layernorm=True,
                layernorm_eps=config.network.layernorm_eps,
                key=key_k,
            )
        else:
            self.theta_transport_network = None
            self.k_transport_network = None

    def tendency_functions(self):
        """Return the (theta, k) tendency callables the integrator uses."""
        if self.theta_transport_network is not None:
            th_fn = partial(neural_theta_tendency, network_fn=self.theta_transport_network)
        else:
            th_fn = theta_tendency
        if self.k_transport_network is not None:
            k_fn = partial(neural_k_tendency, network_fn=self.k_transport_network)
        else:
            k_fn = k_tendency
        return th_fn, k_fn

    def step(
        self,
        spectra: jax.Array,
        env: WaveEnv,
        depth: jax.Array,
        mask: jax.Array,
        wind_u: jax.Array,
        wind_v: jax.Array,
        boundary_condition: Optional[jax.Array] = None,
        *, boundary_mask: Optional[jax.Array] = None, spatial_phase=0, audit_clipping=False,
    ) -> jax.Array:
        """Advance the spectrum by ``config.time.dt_global_s`` seconds.

        ``env`` must already carry the dispersion fields for ``depth``
        (use :func:`prepare_env`).  ``boundary_condition`` defaults to
        zeros, which is the absorbing edge of the source model.
        """
        spectra = jnp.asarray(spectra)
        depth_5d = jnp.asarray(depth)
        if depth_5d.ndim == 3:
            depth_5d = depth_5d[..., None, None]
        elif depth_5d.ndim != 5 or depth_5d.shape[-2:] != (1, 1):
            raise ValueError(
                "wave depth must be (B, Nlon, Nlat) or (B, Nlon, Nlat, 1, 1); "
                f"got {depth_5d.shape}")
        if boundary_condition is None:
            boundary_condition = jnp.zeros_like(spectra)

        # The source encoder masks the spectrum before the network and the
        # integrator (BaseEnTransform); the decoder masks again at the end.
        spectra = spectra * mask[..., None, None]

        # The NeuralSWM weights are trained in their own source units; the
        # adapter owns the 0.01 bridge to SI (the integrator is SI, as WADEN).
        stot = self.parameterization(
            spectra, env, depth_5d, depth_5d, wind_u, wind_v, mask,
        ) * SOURCE_TIME_SCALE
        th_fn, k_fn = self.tendency_functions()
        gse = None
        if self.config.gse.enabled:
            gse = GSESpatialAveraging(self.config.gse)

        integrated = model_advance(
            spectra=spectra,
            stot=stot,
            depth=depth_5d,
            env=env,
            config=self.config,
            mask=mask,
            boundary_condition=boundary_condition,
            th_tendency_fn=th_fn,
            k_tendency_fn=k_fn,
            gse_alleviation=gse, boundary_mask=boundary_mask,
            spatial_phase=spatial_phase, audit_clipping=audit_clipping,
        )
        # Source order: decoder masks, then the boundary condition is applied
        # AGAIN.  The re-injection is what keeps an externally supplied edge
        # ring in the output; dropping it (mask-last) silently loses the ring.
        if audit_clipping:
            integrated, clipped = integrated
        output = integrated * mask[..., None, None]
        output = apply_boundary_condition(output, boundary_condition, mask, boundary_mask)
        return (output, clipped) if audit_clipping else output


    def advance(self, state: WaveState, env: WaveEnv, wind_u: jax.Array, wind_v: jax.Array,
                boundary_condition=None, *, boundary_mask=None, audit_clipping=False):
        """Advance spectrum and persistent spatial-order phase together."""
        spectra = self.step(state.spectra, env, state.depth, state.mask, wind_u, wind_v,
                            boundary_condition, boundary_mask=boundary_mask,
                            spatial_phase=state.spatial_phase, audit_clipping=audit_clipping)
        if audit_clipping:
            spectra, clipped = spectra
        phase = (state.spatial_phase+spatial_event_count(self.config.time)) % 2
        result = state._replace(spectra=spectra, spatial_phase=phase)
        return (result, {"clipping_mass": clipped}) if audit_clipping else result


def validate_wave_environment(config: WaveConfig, env: WaveEnv) -> None:
    """Concrete GSE footprint preflight, outside JIT/AD as in WADEN."""
    if config.gse.enabled:
        gse = GSESpatialAveraging(config.gse)
        gse.validate_displacement(env.group_velocity, env, config.time.dt_gse_s)
