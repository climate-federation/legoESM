"""Static spectral grid and derived metrics for the wave component.

Direct port of ``ModelArchitecture.set_wave_environment`` from
NeuralSWM (``neuralwave_jax/architecture.py``), with the ``gin`` metadata
dictionary replaced by an explicit :class:`WaveEnv`.

Shapes follow the source exactly::

    lon          (1, Nlon, 1, 1, 1)
    lat          (1, 1, Nlat, 1, 1)
    theta        (1, 1, 1, Ndir, 1)
    freq         (1, 1, 1, 1, Nfreq)
    spectra      (B, Nlon, Nlat, Ndir, Nfreq)

Longitude is the first spatial axis and latitude the second, which is the
storage order of the ERA5 demo data (ascending longitude, descending
latitude).  The source docstring says ``(B, H, W, D, F)`` with H=lat and
W=lon, but every indexing site uses axis 1 for longitude; the port keeps
the code's behaviour and documents the corrected order.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.wave.constants import R_EARTH

from legoesm.wave.config import WaveConfig


class WaveEnv(NamedTuple):
    """Static grid metrics plus the state-dependent dispersion fields."""

    lon: jax.Array          # (1, Nlon, 1, 1, 1) [rad]
    lat: jax.Array          # (1, 1, Nlat, 1, 1) [rad]
    theta: jax.Array        # (1, 1, 1, Ndir, 1) [rad]
    freq: jax.Array         # (1, 1, 1, 1, Nfreq) [Hz]
    dth: jax.Array          # scalar [rad]
    dfreq: jax.Array        # (Nfreq,) [Hz]
    dlon: jax.Array         # (1, Nlon, 1, 1, 1) [rad]
    dlat: jax.Array         # (1, 1, Nlat, 1, 1) [rad]
    sin_theta: jax.Array    # (1, 1, 1, Ndir, 1)
    cos_theta: jax.Array    # (1, 1, 1, Ndir, 1)
    cos_lat: jax.Array      # (1, 1, Nlat, 1, 1)
    n_theta: int
    n_freq: int
    wavenum: jax.Array      # (B, Nlon, Nlat, 1, Nfreq)
    group_velocity: jax.Array  # same shape
    quadrature_df: jax.Array  # positive midpoint-cell widths; NN dfreq stays separate
    x_metric: jax.Array       # R*cos(latitude) for spherical, one for Cartesian
    y_metric: jax.Array       # R for spherical, one for Cartesian
    curvature: jax.Array      # latitude_sign*tan(latitude)/R, zero for Cartesian
    theta_axis_sign: jax.Array
    data_theta: jax.Array     # going-to compass radians, independent of axis orientation



def build_wave_env(config: WaveConfig, dtype=jnp.float32) -> WaveEnv:
    """Build the static part of the environment (no dispersion fields)."""
    config.validate()
    lon = jnp.asarray(config.grid.lon_deg, dtype=dtype)
    lat = jnp.asarray(config.grid.lat_deg, dtype=dtype)
    theta = jnp.asarray(config.grid.theta_rad, dtype=dtype)
    freq = jnp.asarray(config.grid.freq_hz, dtype=dtype)

    lon_grid = lon.reshape(1, -1, 1, 1, 1)
    lat_grid = lat.reshape(1, 1, -1, 1, 1)
    theta_grid = theta.reshape(1, 1, 1, -1, 1)
    freq_grid = freq.reshape(1, 1, 1, 1, -1)

    spherical = config.grid.coordinate_type == "Spherical"
    if spherical:
        lon_grid = lon_grid / 180.0 * jnp.pi
        lat_grid = lat_grid / 180.0 * jnp.pi

    n_theta = int(theta.shape[0])
    n_freq = int(freq.shape[0])
    dth = jnp.asarray(2.0 * jnp.pi / n_theta, dtype=dtype)

    freqc = jnp.pad(freq, (0, 1), constant_values=0.0)
    freqc = freqc.at[-1].set(1.0)
    dfreq = jnp.diff(freqc)

    lonc = jnp.concatenate((lon_grid[:, -2:-1], lon_grid, lon_grid[:, 0:1]), axis=1)
    latc = jnp.concatenate((lat_grid[:, :, -2:-1], lat_grid, lat_grid[:, :, 0:1]), axis=2)
    dlon = jnp.abs(0.5 * (jnp.diff(lonc[:, 1:], axis=1) + jnp.diff(lonc[:, :-1], axis=1)))
    dlat = jnp.abs(0.5 * (jnp.diff(latc[:, :, 1:], axis=2) + jnp.diff(latc[:, :, :-1], axis=2)))
    dlon = dlon.at[:, 0].set(dlon[:, 1])
    dlon = dlon.at[:, -1].set(dlon[:, -2])
    dlat = dlat.at[:, :, 0].set(dlat[:, :, 1])
    dlat = dlat.at[:, :, -1].set(dlat[:, :, -2])

    # The central-spacing stencil needs an interior cell. For two centres,
    # the physical spacing is their sole difference on either axis.
    if lon.shape[0] == 2:
        dlon = jnp.broadcast_to(jnp.abs(lon_grid[:, 1:2] - lon_grid[:, :1]), dlon.shape)
    if lat.shape[0] == 2:
        dlat = jnp.broadcast_to(jnp.abs(lat_grid[:, :, 1:2] - lat_grid[:, :, :1]), dlat.shape)

    cos_lat = jnp.cos(lat_grid) if spherical else jnp.ones_like(lat_grid)
    latitude_sign = 1 if config.grid.lat_deg[1] > config.grid.lat_deg[0] else -1
    longitude_sign = 1 if config.grid.lon_deg[1] > config.grid.lon_deg[0] else -1
    data_theta = np.mod(
        np.arctan2(
            longitude_sign * np.sin(np.asarray(config.grid.theta_rad)),
            latitude_sign * np.cos(np.asarray(config.grid.theta_rad))),
        2 * np.pi)
    zeros = jnp.zeros((1, lon.shape[0], lat.shape[0], 1, n_freq), dtype=dtype)
    return WaveEnv(
        lon=lon_grid,
        lat=lat_grid,
        theta=theta_grid,
        freq=freq_grid,
        dth=dth,
        dfreq=dfreq,
        dlon=dlon,
        dlat=dlat,
        sin_theta=jnp.sin(theta_grid),
        cos_theta=jnp.cos(theta_grid),
        cos_lat=cos_lat,
        n_theta=n_theta,
        n_freq=n_freq,
        wavenum=zeros,
        group_velocity=zeros,
        quadrature_df=frequency_cell_widths(freq),
        x_metric=R_EARTH*cos_lat if spherical else jnp.ones_like(cos_lat),
        y_metric=jnp.asarray(R_EARTH if spherical else 1., dtype=dtype),
        curvature=(latitude_sign * jnp.tan(lat_grid) / R_EARTH
                   if spherical else jnp.zeros_like(lat_grid)),
        theta_axis_sign=jnp.sign(jnp.sin(theta[1]-theta[0])),
        data_theta=jnp.asarray(data_theta, dtype=dtype).reshape(1, 1, 1, -1, 1),
    )


def prepare_env(config: WaveConfig, depth: jax.Array) -> WaveEnv:
    """Static grid metrics plus dispersion fields for this depth."""
    return with_dispersion(build_wave_env(config, dtype=depth.dtype), depth)


def with_dispersion(env: WaveEnv, depth: jax.Array) -> WaveEnv:
    """Attach wavenumber and group velocity for the given depth field.

    ``depth`` is ``(B, Nlon, Nlat)`` or ``(B, Nlon, Nlat, 1, 1)`` in metres.
    The source recomputes these every call; the port keeps that behaviour so
    a changing bathymetry (or a land mask that moves) is handled correctly.
    """
    from legoesm.wave.physics.dispersion import (
        compute_group_velocity,
        solve_wavenumber,
    )

    depth_5d = jnp.asarray(depth)
    if depth_5d.ndim == 3:
        depth_5d = depth_5d[..., None, None]
    elif depth_5d.ndim != 5 or depth_5d.shape[-2:] != (1, 1):
        raise ValueError(
            "wave depth must be (B, Nlon, Nlat) or (B, Nlon, Nlat, 1, 1); "
            f"got {depth_5d.shape}")
    expected = (env.lon.shape[1], env.lat.shape[2])
    if depth_5d.shape[1:3] != expected:
        raise ValueError(
            "wave depth spatial shape must match the environment: "
            f"{depth_5d.shape[1:3]} vs {expected}")
    wavenum = solve_wavenumber(env.freq, depth_5d)
    group_velocity = compute_group_velocity(env.freq, wavenum, depth_5d)
    return env._replace(wavenum=wavenum, group_velocity=group_velocity)


def frequency_cell_widths(freq: jax.Array) -> jax.Array:
    """WADEN midpoint finite-band widths, including both endpoint cells."""
    freq = jnp.ravel(freq)
    edges = jnp.concatenate((freq[:1] - .5*(freq[1:2]-freq[:1]),
                             .5*(freq[1:]+freq[:-1]),
                             freq[-1:] + .5*(freq[-1:]-freq[-2:-1])))
    return jnp.diff(edges)


def wave_cell_area(env: WaveEnv) -> jax.Array:
    """Finite-volume spatial area; same metric as the transport divergence."""
    return env.dlon * env.dlat * env.x_metric * env.y_metric


def spectral_integral(spectra: jax.Array, env: WaveEnv, mask: jax.Array) -> jax.Array:
    """Area/frequency/direction integral per batch, with fixed wet support."""
    weight = wave_cell_area(env) * env.quadrature_df * env.dth
    return jnp.sum(jnp.where(mask[..., None, None], spectra, 0)*weight, axis=(1,2,3,4))
