"""Coupling seam between the atmosphere/ocean and the wave component.

This module is the only place that knows both the legoESM grid
conventions and the wave model's conventions:

* legoESM surface fields are ``(..., lat, lon)`` on a regular lat-lon
  grid; the wave model's are ``(..., lon, lat)`` with latitude DESCENDING
  (the oracle's tendency sign convention).
* The conservative remapper works in ``(lat, lon)`` with ascending
  latitude, so the wave field's latitude axis is reversed around every
  remap call.

Nothing here advances the wave model; the driver does that.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.wave.config import WaveConfig

# Under-sqrt floor [Pa^2] for the stress magnitude: keeps sqrt/divide
# derivatives finite in a calm cell without changing a non-zero wind.
_STRESS_FLOOR_PA2 = 1e-12
# Density floor [kg/m^3] for the friction-velocity division.
_MIN_RHO_KG_M3 = 1e-12
# Neutral-wind reference height [m] of the bulk formula.
_WIND_REFERENCE_HEIGHT_M = 10.0
# Roughness bounds [m] for the log profile: below the bounds the log is
# undefined / the profile is not physical.
_MIN_ROUGHNESS_M = 1e-8
_MAX_ROUGHNESS_M = 10.0
# Grid checks [deg]: uniform spacing, pole reach and longitude span.
_GRID_TOL_DEG = 1e-6
# Latitude of the poles [deg]; a global grid's outer cell edges reach it.
_POLE_LAT_DEG = 90.0


class WaveRemapGrid(NamedTuple):
    """The wave grid in the remapper's convention (radians, ascending)."""

    lon: jax.Array      # (n_lon,) radians, ascending, global
    lat: jax.Array      # (n_lat,) radians, ascending cell centres
    lat_v: jax.Array    # (n_lat+1,) radians, ascending edges
    n_lat: int
    n_lon: int


def wave_remap_grid(config: WaveConfig) -> WaveRemapGrid:
    """Build the ascending-latitude view the conservative remapper needs.

    Requires a global regular grid: the remap enforces full source
    coverage of the destination, so a regional wave grid can drive a
    regional ocean but cannot feed a global one.  Both conditions are
    checked here so the failure is a clear message, not a coverage
    shortfall deep inside the overlap generator.
    """
    grid = config.grid
    if (grid.coordinate_type != "Spherical"
            or grid.boundary_mode != "physical"
            or grid.x_stencil != "circular"):
        raise ValueError(
            "global coupled wave requires spherical physical cells and "
            "circular longitude")
    lon_deg = jnp.asarray(grid.lon_deg, dtype=jnp.float64)
    lat_deg = jnp.asarray(grid.lat_deg, dtype=jnp.float64)
    _require_uniform(lon_deg, "longitude")
    _require_uniform(lat_deg, "latitude")

    if jnp.any(jnp.diff(lat_deg) > 0):
        raise ValueError(
            "wave latitude must be stored descending (the oracle's tendency "
            "sign convention); got an ascending array")
    lat_asc = lat_deg[::-1]
    half = 0.5 * (lat_asc[1] - lat_asc[0])
    edges = jnp.concatenate((
        jnp.asarray([lat_asc[0] - half]),
        lat_asc + half,
    ))
    if (abs(float(edges[0]) + _POLE_LAT_DEG) > _GRID_TOL_DEG
            or abs(float(edges[-1]) - _POLE_LAT_DEG) > _GRID_TOL_DEG):
        raise NotImplementedError(
            "the coupled wave grid must be global in latitude "
            f"(edges {float(edges[0]):.3f}..{float(edges[-1]):.3f} deg); "
            "a regional wave grid cannot be a conservative SOURCE for a "
            "global destination")

    span = float(lon_deg[-1] - lon_deg[0] + (lon_deg[1] - lon_deg[0]))
    if abs(span - 360.0) > _GRID_TOL_DEG:
        raise NotImplementedError(
            "the coupled wave grid must span 360 deg of longitude; "
            f"got {span:.3f}")

    return WaveRemapGrid(
        lon=jnp.deg2rad(lon_deg),
        lat=jnp.deg2rad(lat_asc),
        lat_v=jnp.deg2rad(edges),
        n_lat=int(lat_deg.shape[0]),
        n_lon=int(lon_deg.shape[0]),
    )


def _require_uniform(values: jax.Array, name: str) -> None:
    if values.shape[0] < 2:
        raise ValueError(f"wave {name} grid needs at least two cells")
    spacing = jnp.diff(values)
    if jnp.max(jnp.abs(spacing - spacing[0])) > _GRID_TOL_DEG:
        raise ValueError(
            f"wave {name} grid must be uniform; got spacing {spacing}")


def wave_to_remap(field: jax.Array) -> jax.Array:
    """``(..., lon, lat_desc)`` -> ``(..., lat_asc, lon)``."""
    return jnp.swapaxes(field, -1, -2)[..., ::-1, :]


def remap_to_wave(field: jax.Array) -> jax.Array:
    """``(..., lat_asc, lon)`` -> ``(..., lon, lat_desc)``."""
    return jnp.swapaxes(field[..., ::-1, :], -1, -2)


def neutral_wind_10m(
    tau_x: jax.Array,
    tau_y: jax.Array,
    rho: jax.Array,
    z0: jax.Array,
    *,
    kappa: float = constants.kappa,
) -> tuple[jax.Array, jax.Array]:
    """Neutral 10 m wind from the surface stress and the reported roughness.

    ``u* = sqrt(|tau| / rho)`` and ``U10 = u* / kappa * ln(10 / z0)``, with
    the wind direction taken from the stress.  This is the standard neutral
    log-profile diagnostic.

    ``z0`` and ``tau`` are the coupler's BLENDED surface response.  Each
    tile's z0 is a fixed value (the ocean's ``CouplerConfig.ocean_z0``, land
    ~0.1 m, ...), so a pure-ocean cell is exact; a mixed ocean/ice/land/lake
    cell blends all-tile stress and roughness and is an approximation.
    ``coare3``/``large_yeager`` evolve roughness internally instead of
    reporting it, so those closures are approximate even over open water.
    The default coupled bulk scheme is constant.

    ``tau_x``/``tau_y`` are in the ATMOSPHERIC convention (the drag the
    surface exerts on the air), so the wind is ``-tau * U10 / |tau|``.
    """
    stress2 = tau_x**2 + tau_y**2
    speed = jnp.sqrt(stress2 + _STRESS_FLOOR_PA2)
    u_star = jnp.sqrt(speed / jnp.maximum(rho, _MIN_RHO_KG_M3))
    u10 = u_star / kappa * jnp.log(
        _WIND_REFERENCE_HEIGHT_M
        / jnp.clip(z0, _MIN_ROUGHNESS_M, _MAX_ROUGHNESS_M))
    # Compare the SQUARED stress with its squared floor: comparing the
    # square-rooted value against the Pa^2 floor is always true and gives a
    # calm cell a spurious ~0.03 m/s wind.
    scale = jnp.where(stress2 > _STRESS_FLOOR_PA2, u10 / speed, 0.0)
    # SurfaceToAtm carries drag on the atmosphere (opposite the wind).
    return -tau_x * scale, -tau_y * scale


__all__ = [
    "WaveRemapGrid",
    "neutral_wind_10m",
    "remap_to_wave",
    "wave_remap_grid",
    "wave_to_remap",
]
