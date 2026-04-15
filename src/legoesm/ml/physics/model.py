"""Joint ML parameterization model for Louis turbulence and mass-flux convection."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from legoesm import constants


class PhysicsParameterizationModel(eqx.Module):
    """Monolithic column MLP predicting ``Km``, ``Kh``, and ``M_eq``."""

    layers: tuple[eqx.nn.Linear, ...]
    head: eqx.nn.Linear
    nlev: int = eqx.field(static=True)
    n_input: int = eqx.field(static=True)
    n_output: int = eqx.field(static=True)

    def __init__(
        self,
        nlev: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        *,
        key: jax.Array,
    ):
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        self.nlev = nlev
        self.n_input = 6 * nlev + 8
        self.n_output = 2 * nlev + 1
        keys = jax.random.split(key, n_layers + 1)
        dims = [self.n_input] + [hidden_dim] * n_layers
        self.layers = tuple(
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(n_layers)
        )
        self.head = eqx.nn.Linear(hidden_dim, self.n_output, key=keys[-1])

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = jax.nn.gelu(layer(x))
        return self.head(x)


def pack_physics_parameterization_features(
    T: jax.Array,
    u: jax.Array,
    v: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    z_full: jax.Array,
    p_s: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    lat: jax.Array,
    cape: jax.Array,
    M_c: jax.Array,
    dt: jax.Array,
) -> jax.Array:
    """Pack one atmospheric column into the joint ML feature vector."""
    return jnp.concatenate(
        [
            T / 300.0,
            u / 30.0,
            v / 30.0,
            q_v * 1e3,
            jnp.log(jnp.clip(p_full, 1.0, None) / constants.p_ref),
            z_full / 30000.0,
            jnp.asarray(
                [
                    p_s / constants.p_ref,
                    T_sfc / 300.0,
                    q_sfc * 1e3,
                    jnp.sin(lat),
                    jnp.cos(lat),
                    cape / 1000.0,
                    M_c / 0.01,
                    dt / 3600.0,
                ],
                dtype=T.dtype,
            ),
        ],
    )


def pack_physics_parameterization_targets(
    Km: jax.Array,
    Kh: jax.Array,
    M_eq: jax.Array,
) -> jax.Array:
    """Pack joint teacher targets into a flat output vector."""
    return jnp.concatenate([Km, Kh, M_eq[..., None]], axis=-1)


def unpack_physics_parameterization_targets(
    targets: jax.Array,
    nlev: int,
) -> dict[str, jax.Array]:
    """Unpack the flat target vector into named outputs."""
    expected = 2 * nlev + 1
    if targets.shape[-1] != expected:
        raise ValueError(f"Expected {expected} outputs, got {targets.shape[-1]}")
    return {
        "Km": targets[..., :nlev],
        "Kh": targets[..., nlev:2 * nlev],
        "M_eq": targets[..., -1],
    }
