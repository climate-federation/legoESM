"""ML gravity wave drag emulator using Equinox MLP.

Provides a neural network surrogate for GWD that is AD-safe
(differentiable with jax.grad; the only kinks are the input
clip/abs). The MLP maps per-level inputs (u, v, T, p, z, lat)
to per-level GWD tendencies.

Same architecture pattern as microphysics/ml_emulator.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import GWDMLEmulatorConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Neural (Equinox MLP) gravity-wave-drag emulator: maps per-level "
        "(u, v, T, log p, z, sin/cos lat) to per-level GWD wind and "
        "temperature tendencies; a learned surrogate, not a physical closure."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
    },
    "sign_convention": (
        "Signs are LEARNED, not enforced: du_dt/dv_dt/dT_dt are direct network "
        "output channels (residual-scaled by _GWD_OUTPUT_SCALE when "
        "use_residual). eps_gwd = -sum(rho*(u*du_dt+v*dv_dt)*dz) is a diagnostic "
        "of the mean-flow KE removal and is NOT constrained to match dT_dt, so "
        "an untrained or mis-trained model can ADD mean-flow KE (eps_gwd<0)."
    ),
    # Learned emulator: no hard conservation guarantee. dT_dt is a SEPARATE
    # unconstrained network output channel (not the KE->heat tie-back), so
    # energy is not conserved by construction; over-claiming any conserved
    # quantity would be wrong.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Learned Equinox-MLP GWD surrogate (legoESM); architecture per "
        "microphysics/ml_emulator.py (no external physical reference)"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py: untrained "
        "residual-scaled model -> bounded tendencies (|du_dt|, |dv_dt| < 1) with "
        "correct shapes; AD-safe (finite jax.grad through the eqx MLP)"
    ),
}
# Raw-NN-output scaling to physical tendency magnitude (emulator default).
_GWD_OUTPUT_SCALE = 0.01



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
    config: GWDMLEmulatorConfig,
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
        du_dt = du_dt * _GWD_OUTPUT_SCALE
        dv_dt = dv_dt * _GWD_OUTPUT_SCALE
        dT_dt = dT_dt * _GWD_OUTPUT_SCALE

    # Column mean-flow KE removal rate = -sum(rho*(u·du + v·dv)*dz).  This is a
    # DIAGNOSTIC of the KE the (learned) wind tendency removes from the mean
    # flow -- it is NOT tied to dT_dt, which is a SEPARATE unconstrained network
    # output channel (sharing the MLP hidden representation, not statistically
    # independent).  So the physical KE->heat tie-back ``c_pd·∫ρ·dT_dt·dz =
    # eps_gwd`` is NOT ENFORCED by this implementation (a training loss could
    # impose it; the architecture does not) -- consistent with the contract
    # sign_convention.  The SIGNED form (not ``jnp.abs``) is deliberate: a model
    # that ADDS mean-flow KE reports eps_gwd < 0, exposing the violation instead
    # of hiding it behind an absolute value (iter-26 P1).
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
