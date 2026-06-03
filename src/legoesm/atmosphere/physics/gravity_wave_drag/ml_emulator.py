"""ML gravity wave drag emulator using Equinox MLP.

Provides a neural network surrogate for GWD that is fully
differentiable with jax.grad. The MLP maps per-level inputs
(u, v, T, p, z, lat) to per-level GWD tendencies.

Same architecture pattern as microphysics/ml_emulator.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import MLEmulatorConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput


class GWDEmulator(eqx.Module):
    """Multi-layer perceptron GWD emulator."""
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


def ml_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: MLEmulatorConfig,
    model: GWDEmulator,
) -> GWDOutput:
    """Compute ML emulator GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature.
    model : GWDEmulator
        Equinox MLP model.

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    lat_3d = jnp.broadcast_to(lat[:, None], (ncol, nlev))
    features = jnp.stack([
        u / config.norm_u,
        v / config.norm_u,
        T / config.norm_T,
        jnp.log(jnp.clip(p_full, 1.0) / constants.p_ref),
        z_full / config.norm_z,
        jnp.sin(lat_3d),
        jnp.cos(lat_3d),
    ], axis=-1)  # (ncol, nlev, n_input)

    # Apply MLP per level via double vmap
    predict = jax.vmap(jax.vmap(model))
    y = predict(features)  # (ncol, nlev, n_output)

    # Map outputs
    du_dt = y[..., 0]
    dv_dt = y[..., 1]
    dT_dt = y[..., 2]

    # Residual scaling: untrained model should produce near-zero
    if config.use_residual:
        du_dt = du_dt * 0.01
        dv_dt = dv_dt * 0.01
        dT_dt = dT_dt * 0.01

    # Column dissipation: KE → heat conversion rate.  Use the same
    # ``-(u·du + v·dv)`` form as Lindzen / McFarlane / Hines so the
    # conservation tie-back ``c_pd · ∫ρ·dT_dt·dz = eps_gwd`` holds.
    # The earlier ``jnp.abs(u·du + v·dv)`` lost the sign of the
    # untrained tendency: a model that accidentally adds energy
    # would still report eps_gwd as positive, hiding the violation.
    # Audit cycle iter-26 finding P1.
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
