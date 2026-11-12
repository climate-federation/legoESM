"""Feature maps fed to the source-term network.

Direct port of the three feature generators the released configuration
uses (``WindEmbeddings``, ``SdnEmbeddings2``, ``WwInterEmbeddings`` in
NeuralSWM ``neuralwave_jax/embeddings.py``).  The other generators in
that file are not part of the released model and are not ported.

Each function returns a ``(B*Nlon*Nlat, 1, Ndir, Nfreq)`` image; the
parameterization flattens it to a channel vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.wave.grid import WaveEnv


def data_direction_radians(env: WaveEnv) -> jax.Array:
    """Direction angle used by the feature maps.

    The physics grid stores ``theta = (180 - phi) mod 360`` where ``phi``
    is the direction the wave propagates towards, measured clockwise
    from north.  The feature generators use ``phi`` itself, so this
    converts back exactly as ``_theta_radians`` does in the source.
    """
    return env.data_theta



def _spatial_3d(array: jax.Array) -> jax.Array:
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return jnp.squeeze(array, axis=-1)
    if array.ndim == 5:
        return jnp.squeeze(jnp.squeeze(array, axis=-1), axis=-1)
    raise ValueError(f"Expected 3D/4D/5D spatial array, got shape {array.shape}")


def _reshape_feature_map(array: jax.Array, channels: int, n_dir: int, n_freq: int) -> jax.Array:
    return jnp.reshape(array, (-1, channels, n_dir, n_freq))


def depth_gradients(env: WaveEnv, bathymetry: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Depth gradients in the source's embedding convention.

    The source divides by the raw radian grid spacing without the
    spherical metric; the port preserves that (the network was trained
    on these units).
    """
    depth_3d = _spatial_3d(bathymetry)
    depthx_3d = jnp.pad(depth_3d, ((0, 0), (1, 1), (0, 0)), mode="wrap")
    depthy_3d = jnp.pad(depth_3d, ((0, 0), (0, 0), (1, 1)), mode="edge")

    depthx = depthx_3d[..., None, None]
    depthy = depthy_3d[..., None, None]

    dlon = jnp.asarray(env.dlon)
    dlat = jnp.asarray(env.dlat)

    dddx = 0.5 * (
        (depthx[:, 2:, ...] - depthx[:, 1:-1, ...])
        + (depthx[:, 1:-1, ...] - depthx[:, :-2, ...])) / dlon
    dddy = 0.5 * (
        (depthy[:, :, 2:, ...] - depthy[:, :, 1:-1, ...])
        + (depthy[:, :, 1:-1, ...] - depthy[:, :, :-2, ...])) / dlat
    return dddx, dddy


# The released NeuralSWM weights were trained on the wind and depth-gradient
# features divided by this scale (``neuralwave_jax/architecture.py``).
_FEATURE_SCALE_FACTOR = 20.0


def wind_features(
    env: WaveEnv,
    wind_u: jax.Array,
    wind_v: jax.Array,
    *,
    scale_factor: float = _FEATURE_SCALE_FACTOR,
) -> jax.Array:
    """Wind projected on the wave direction, scaled by the group velocity."""
    theta = data_direction_radians(env)
    u = _spatial_3d(wind_u)[..., None, None]
    v = _spatial_3d(wind_v)[..., None, None]
    u_proj = u * jnp.sin(theta) + v * jnp.cos(theta)
    weights = u_proj * jnp.asarray(env.group_velocity)
    return _reshape_feature_map(weights / scale_factor, 1, env.n_theta, env.n_freq)


def sdn_features(
    env: WaveEnv,
    bathymetry: jax.Array,
    *,
    scale_factor: float = _FEATURE_SCALE_FACTOR,
) -> jax.Array:
    """Depth-gradient projection times the frequency spacing."""
    dddx, dddy = depth_gradients(env, bathymetry)
    theta = data_direction_radians(env)
    bathy_proj = dddx * jnp.sin(theta) + dddy * jnp.cos(theta)
    dfreq = jnp.asarray(env.dfreq).reshape(1, 1, 1, 1, -1)
    weights = bathy_proj * dfreq
    return _reshape_feature_map(weights / scale_factor, 1, env.n_theta, env.n_freq)


def ww_inter_features(env: WaveEnv) -> jax.Array:
    """Wavenumber plus frequency, broadcast over the direction axis."""
    wave_num = jnp.broadcast_to(
        jnp.asarray(env.wavenum),
        (*env.wavenum.shape[:3], env.n_theta, env.wavenum.shape[-1]),
    )
    wave_freq = jnp.broadcast_to(jnp.asarray(env.freq), wave_num.shape)
    weights = wave_num + wave_freq
    return _reshape_feature_map(weights, 1, env.n_theta, env.n_freq)


__all__ = [
    "data_direction_radians",
    "depth_gradients",
    "sdn_features",
    "wind_features",
    "ww_inter_features",
]
