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
    microphysics_scheme: str = eqx.field(static=True)

    def __init__(
        self,
        nlev: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        microphysics_scheme: str = "none",
        *,
        key: jax.Array,
    ):
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        self.nlev = nlev
        self.microphysics_scheme = _validate_microphysics_scheme(microphysics_scheme)
        self.n_input = physics_parameterization_input_size(
            nlev,
            microphysics_scheme=self.microphysics_scheme,
        )
        self.n_output = physics_parameterization_output_size(
            nlev,
            microphysics_scheme=self.microphysics_scheme,
        )
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


def _validate_microphysics_scheme(microphysics_scheme: str) -> str:
    """Validate and normalize the supported microphysics scheme label."""
    if microphysics_scheme not in ("none", "kessler", "sundqvist"):
        raise ValueError(
            f"Unsupported microphysics_scheme={microphysics_scheme!r}; "
            "expected 'none', 'kessler', or 'sundqvist'",
        )
    return microphysics_scheme


def _microphysics_feature_levels(microphysics_scheme: str) -> int:
    """Return how many hydrometeor feature profiles are appended per column."""
    scheme = _validate_microphysics_scheme(microphysics_scheme)
    if scheme == "none":
        return 0
    if scheme == "sundqvist":
        return 1
    return 2


def physics_parameterization_input_size(
    nlev: int,
    *,
    microphysics_scheme: str = "none",
) -> int:
    """Return the packed feature size for one atmospheric column."""
    size = 6 * nlev + 8
    size += _microphysics_feature_levels(microphysics_scheme) * nlev
    return size


def physics_parameterization_output_size(
    nlev: int,
    *,
    microphysics_scheme: str = "none",
) -> int:
    """Return the packed target size for one atmospheric column."""
    scheme = _validate_microphysics_scheme(microphysics_scheme)
    size = 2 * nlev + 1
    if scheme == "kessler":
        size += 3 * nlev + 1
    elif scheme == "sundqvist":
        size += 1
    return size


def pack_physics_parameterization_features(
    T: jax.Array,
    u: jax.Array,
    v: jax.Array,
    q_v: jax.Array,
    q_c: jax.Array,
    q_r: jax.Array,
    p_full: jax.Array,
    z_full: jax.Array,
    p_s: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    lat: jax.Array,
    cape: jax.Array,
    M_c: jax.Array,
    dt: jax.Array,
    *,
    microphysics_scheme: str = "none",
) -> jax.Array:
    """Pack one atmospheric column into the joint ML feature vector."""
    scheme = _validate_microphysics_scheme(microphysics_scheme)
    parts = [
        T / 300.0,
        u / 30.0,
        v / 30.0,
        q_v * 1e3,
    ]
    if scheme != "none":
        parts.append(q_c * 1e3)
    if scheme == "kessler":
        parts.append(q_r * 1e3)
    parts.extend(
        [
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
    return jnp.concatenate(parts)


def pack_physics_parameterization_targets(
    Km: jax.Array,
    Kh: jax.Array,
    M_eq: jax.Array,
    *,
    rain_survival_fraction: jax.Array | None = None,
    dq_v_dt_micro: jax.Array | None = None,
    dq_c_dt_micro: jax.Array | None = None,
    dq_r_dt_micro: jax.Array | None = None,
    precip_micro: jax.Array | None = None,
    microphysics_scheme: str = "none",
) -> jax.Array:
    """Pack joint teacher targets into a flat output vector."""
    scheme = _validate_microphysics_scheme(microphysics_scheme)
    targets = [Km, Kh, M_eq[..., None]]
    if scheme == "kessler":
        if (
            dq_v_dt_micro is None
            or dq_c_dt_micro is None
            or dq_r_dt_micro is None
            or precip_micro is None
        ):
            raise ValueError(
                "Direct microphysics targets require "
                "dq_v_dt_micro, dq_c_dt_micro, dq_r_dt_micro, and precip_micro",
            )
        targets.extend(
            [
                dq_v_dt_micro,
                dq_c_dt_micro,
                dq_r_dt_micro,
                precip_micro[..., None],
            ],
        )
    elif scheme == "sundqvist":
        if rain_survival_fraction is None:
            raise ValueError(
                "Sundqvist microphysics targets require rain_survival_fraction labels",
            )
        targets.append(rain_survival_fraction[..., None])
    return jnp.concatenate(targets, axis=-1)


def unpack_physics_parameterization_targets(
    targets: jax.Array,
    nlev: int,
    *,
    microphysics_scheme: str = "none",
) -> dict[str, jax.Array]:
    """Unpack the flat target vector into named outputs."""
    scheme = _validate_microphysics_scheme(microphysics_scheme)
    expected = physics_parameterization_output_size(
        nlev,
        microphysics_scheme=scheme,
    )
    if targets.shape[-1] != expected:
        raise ValueError(f"Expected {expected} outputs, got {targets.shape[-1]}")
    unpacked = {
        "Km": targets[..., :nlev],
        "Kh": targets[..., nlev:2 * nlev],
        "M_eq": targets[..., 2 * nlev],
    }
    if scheme == "kessler":
        offset = 2 * nlev + 1
        unpacked.update(
            {
                "dq_v_dt_micro": targets[..., offset:offset + nlev],
                "dq_c_dt_micro": targets[..., offset + nlev:offset + 2 * nlev],
                "dq_r_dt_micro": targets[..., offset + 2 * nlev:offset + 3 * nlev],
                "precip_micro": targets[..., -1],
            },
        )
    elif scheme == "sundqvist":
        offset = 2 * nlev + 1
        unpacked["rain_survival_fraction"] = targets[..., offset]
    return unpacked
