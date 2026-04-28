"""ML microphysics emulator using Equinox MLP.

Provides a neural network surrogate for microphysics that is fully
differentiable with jax.grad. The MLP maps per-level thermodynamic
inputs to per-level microphysics tendencies.

The model is initialized lazily on first call using the seed from
MLEmulatorConfig. An untrained model returns near-zero tendencies
(suitable for testing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MLEmulatorConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)


class MicrophysicsEmulator(eqx.Module):
    """Multi-layer perceptron microphysics emulator."""
    layers: list

    def __init__(self, n_input, n_hidden, n_layers, n_output, *, key):
        keys = jax.random.split(key, n_layers)
        dims = [n_input] + [n_hidden] * (n_layers - 1) + [n_output]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(n_layers)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


def ml_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: MLEmulatorConfig,
    model: MicrophysicsEmulator,
) -> MicrophysicsOutput:
    """Compute ML emulator microphysics tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.
    model : MicrophysicsEmulator
        Equinox MLP model.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape

    # Build input features: [T/300, q_v*1e3, q_c*1e3, q_r*1e3,
    #   q_i*1e3, log(p/p_ref), rho/1.2, dz/1000, dt_norm]
    features = jnp.stack([
        T / 300.0,
        q_v * 1e3,
        hydrometeors.q_c * 1e3,
        hydrometeors.q_r * 1e3,
        hydrometeors.q_i * 1e3,
        jnp.log(jnp.clip(p_full, 1.0) / constants.p_ref),
        rho / 1.2,
        dz / 1000.0,
        jnp.full((ncol, nlev), dt / 3600.0, dtype=T.dtype),
    ], axis=-1)  # (ncol, nlev, n_input)

    # Apply MLP per level via double vmap
    predict = jax.vmap(jax.vmap(model))
    y = predict(features)  # (ncol, nlev, n_output)

    # Map outputs
    dT_dt = y[..., 0]
    dq_v_dt = y[..., 1]
    dq_c_dt = y[..., 2]
    dq_r_dt = y[..., 3]
    dq_i_dt = y[..., 4]
    dq_s_dt = y[..., 5]
    precip_raw = y[..., 6]

    # Residual connection: add identity (no change) baseline
    if config.use_residual:
        dT_dt = dT_dt * 0.01  # scale raw output
        dq_v_dt = dq_v_dt * 1e-6
        dq_c_dt = dq_c_dt * 1e-6
        dq_r_dt = dq_r_dt * 1e-6
        dq_i_dt = dq_i_dt * 1e-6
        dq_s_dt = dq_s_dt * 1e-6

    # Surface precipitation from lowest level output
    precipitation = jax.nn.softplus(precip_raw[:, -1]) * 1e-3

    # Pin dtype to the input precision so we never silently promote
    # the unused-species placeholders to f64 under x64 mode.
    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=dq_s_dt,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=precipitation,
    )
