"""ML microphysics emulator using Equinox MLP.

Provides a neural network surrogate for microphysics that is fully
differentiable with jax.grad. The MLP maps per-level thermodynamic
inputs to per-level microphysics tendencies.

The model is initialized lazily on first call using the seed from
MicrophysicsMLEmulatorConfig. An untrained model returns near-zero tendencies
(suitable for testing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsMLEmulatorConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)


# Emulator raw-output scaling.
_OUTPUT_SCALE = 0.01
_PRECIP_SCALE = 1e-3

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


__physics_contract__ = {
    "summary": (
        "Learned (Equinox-MLP) microphysics emulator: predicts the column "
        "microphysical tendencies + surface precipitation from the thermo/"
        "hydrometeor state, as a fast differentiable surrogate for a bulk "
        "scheme."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.q_i": "kg/kg",
        "p_full": "Pa", "rho": "kg/m^3", "dz": "m", "dt": "s",
        "model": "trained MicrophysicsEmulator (Equinox module, passed explicitly)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dq_i_dt": "kg/kg/s", "dq_s_dt": "kg/kg/s",
        "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. The tendencies emulate a bulk microphysics "
        "scheme (dT_dt latent heating, precipitation >= 0 via a softplus head). "
        "Being a LEARNED surrogate it enforces NO hard conservation -- no "
        "contract-level conservation is claimed; positivity of precip is "
        "structural, other outputs are unconstrained."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Learned microphysics emulator (Equinox MLP); cf. Rasp, Pritchard & Gentine (2018), PNAS 115, 9684-9689",
    "idealized_test": (
        "tests/unit/test_physics_microphysics.py — the emulator returns finite "
        "MicrophysicsOutput tendencies with non-negative surface precipitation "
        "and is differentiable wrt the input state."
    ),
}


def ml_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: MicrophysicsMLEmulatorConfig,
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

    features = jnp.stack([
        T / config.norm_T,
        q_v * config.norm_q_factor,
        hydrometeors.q_c * config.norm_q_factor,
        hydrometeors.q_r * config.norm_q_factor,
        hydrometeors.q_i * config.norm_q_factor,
        jnp.log(jnp.clip(p_full, 1.0) / constants.p_ref),
        rho / config.norm_rho,
        dz / config.norm_dz,
        jnp.full((ncol, nlev), dt / config.norm_dt, dtype=T.dtype),
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
        dT_dt = dT_dt * _OUTPUT_SCALE  # scale raw output
        dq_v_dt = dq_v_dt * 1e-6
        dq_c_dt = dq_c_dt * 1e-6
        dq_r_dt = dq_r_dt * 1e-6
        dq_i_dt = dq_i_dt * 1e-6
        dq_s_dt = dq_s_dt * 1e-6

    # Surface precipitation from lowest level output
    precipitation = jax.nn.softplus(precip_raw[:, -1]) * _PRECIP_SCALE

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
