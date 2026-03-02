"""ML turbulence emulator using Equinox MLP.

Provides a neural network surrogate for turbulence that is fully
differentiable with jax.grad. The MLP maps per-level atmospheric
state inputs to per-level turbulence tendencies and diffusivities.

The model is initialized lazily on first call using the seed from
MLTurbulenceEmulatorConfig. An untrained model returns near-zero
tendencies (suitable for testing) due to residual scaling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import MLTurbulenceEmulatorConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)


class TurbulenceEmulator(eqx.Module):
    """Multi-layer perceptron turbulence emulator."""
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


def ml_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: MLTurbulenceEmulatorConfig,
    model: TurbulenceEmulator,
) -> TurbulenceOutput:
    """Compute ML emulator turbulence tendencies.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full : jax.Array
        Height at full levels [m], shape (ncol, nlev).
    z_half : jax.Array
        Height at half levels [m], shape (ncol, nlev+1).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : MLTurbulenceEmulatorConfig
    model : TurbulenceEmulator
        Equinox MLP model.

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # Compute Ri and S for input features
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    dz_half = jnp.clip(dz_half, 1.0, None)

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2 + 1e-10

    theta_v = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa * (
        1.0 + 0.61 * q_v
    )
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2_half = (constants.g / jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz
    Ri_half = N2_half / S2_half

    # Interpolate to full levels
    Ri_smooth = jnp.zeros((ncol, nlev))
    Ri_smooth = Ri_smooth.at[:, 1:-1].set(0.5 * (Ri_half[:, :-1] + Ri_half[:, 1:]))
    Ri_smooth = Ri_smooth.at[:, 0].set(Ri_half[:, 0])
    Ri_smooth = Ri_smooth.at[:, -1].set(Ri_half[:, -1])
    Ri_smooth = jnp.clip(Ri_smooth, -5.0, 5.0)

    S_norm_half = jnp.sqrt(S2_half)
    S_norm = jnp.zeros((ncol, nlev))
    S_norm = S_norm.at[:, 1:-1].set(0.5 * (S_norm_half[:, :-1] + S_norm_half[:, 1:]))
    S_norm = S_norm.at[:, 0].set(S_norm_half[:, 0])
    S_norm = S_norm.at[:, -1].set(S_norm_half[:, -1])

    # Build input features: 8 per level
    features = jnp.stack([
        u / 30.0,
        v / 30.0,
        T / 300.0,
        q_v * 1e3,
        jnp.log(jnp.clip(p_full, 1.0) / constants.p_ref),
        z_full / 30000.0,
        Ri_smooth,
        S_norm / 0.01,  # normalize shear
    ], axis=-1)  # (ncol, nlev, 8)

    # Apply MLP per level via double vmap
    predict = jax.vmap(jax.vmap(model))
    y = predict(features)  # (ncol, nlev, n_output)

    # Map outputs
    du_dt = y[..., 0]
    dv_dt = y[..., 1]
    dT_dt = y[..., 2]
    dq_v_dt = y[..., 3]
    Km_raw = y[..., 4]
    Kh_raw = y[..., 5]
    # Extra outputs (6-8) reserved for future use
    _ = y[..., 6:]

    # Residual scaling for near-zero untrained output
    if config.use_residual:
        du_dt = du_dt * 0.01
        dv_dt = dv_dt * 0.01
        dT_dt = dT_dt * 0.01
        dq_v_dt = dq_v_dt * 1e-6

    # Positive diffusivities via softplus
    Km = jax.nn.softplus(Km_raw) * 0.1
    Kh = jax.nn.softplus(Kh_raw) * 0.1

    # Surface fluxes: physics-based (not learned)
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )

    return TurbulenceOutput(
        du_dt=du_dt,
        dv_dt=dv_dt,
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        Km=Km,
        Kh=Kh,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
    )
