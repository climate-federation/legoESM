"""Latitude-longitude grid for legoESM.

Provides a regular lat-lon grid as an alternative to the cubed-sphere
and Gaussian grids. Cell centers avoid exact pole points to prevent
cos(lat) = 0 singularities.

Grid conventions:
- Shape: (n_lat, n_lon) for 2D, (n_lat, n_lon, nlev) for 3D
- Latitude runs South-to-North: -pi/2 + dlat/2 to pi/2 - dlat/2
- Longitude runs 0 to 2*pi - dlon
- dx and dy are distances spanning 2 cells (from cell i-1 to i+1),
  matching the cubed-sphere convention for centered differences.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants


class LatLonGrid(NamedTuple):
    """Latitude-longitude horizontal grid.

    All arrays are JAX arrays. Because this is a NamedTuple it is
    automatically a valid JAX pytree.
    """
    n_lat: int                  # number of latitude points
    n_lon: int                  # number of longitude points
    radius: float               # sphere radius [m]
    lat: jax.Array              # (n_lat,) cell-center latitudes [rad], S->N
    lon: jax.Array              # (n_lon,) cell-center longitudes [rad], [0, 2*pi)
    lat2d: jax.Array            # (n_lat, n_lon)
    lon2d: jax.Array            # (n_lat, n_lon)
    cos_lat: jax.Array          # (n_lat,) — clamped to avoid zero at poles
    sin_lat: jax.Array          # (n_lat,)
    f: jax.Array                # (n_lat, n_lon) Coriolis = 2*Omega*sin(lat)
    dx: jax.Array               # (n_lat, n_lon) distance over 2 cells in lon [m]
    dy: float                   # distance over 2 cells in lat [m] (constant)
    area: jax.Array             # (n_lat, n_lon) cell area [m^2]
    total_area: jax.Array       # scalar sum of all areas
    dlon: float                 # longitude spacing [rad]
    dlat: float                 # latitude spacing [rad]

    # ------------------------------------------------------------------
    # GridProtocol properties
    # ------------------------------------------------------------------

    @property
    def grid_lat(self) -> jax.Array:
        return self.lat2d

    @property
    def grid_lon(self) -> jax.Array:
        return self.lon2d

    @property
    def grid_area(self) -> jax.Array:
        return self.area

    @property
    def grid_total_area(self):
        return self.total_area

    @property
    def grid_coriolis(self) -> jax.Array:
        return self.f

    @property
    def grid_radius(self) -> float:
        return self.radius

    @property
    def n(self) -> int:
        """Grid resolution parameter, analogous to cubed-sphere ``n``.

        For lat-lon grids this returns ``n_lat`` so that code expecting
        ``grid.n`` (e.g. baroclinic wave initialisation) works
        transparently.
        """
        return self.n_lat

    @property
    def grid_n_columns(self) -> int:
        return self.n_lat * self.n_lon

    def to_columns(self, field):
        extra = field.shape[2:]
        return field.reshape(self.n_lat * self.n_lon, *extra)

    def from_columns(self, cols):
        extra = cols.shape[1:]
        return cols.reshape(self.n_lat, self.n_lon, *extra)


def create_latlon_grid(
    n_lat: int,
    n_lon: int | None = None,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
) -> LatLonGrid:
    """Create a latitude-longitude grid.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.
    n_lon : int, optional
        Number of longitude points. Defaults to 2 * n_lat.
    radius : float
        Sphere radius [m].
    omega : float
        Rotation rate [rad/s].

    Returns
    -------
    LatLonGrid
    """
    if n_lon is None:
        n_lon = 2 * n_lat

    dlat = jnp.pi / n_lat
    dlon = 2.0 * jnp.pi / n_lon

    # Cell-centered latitudes: avoid exact poles
    lat = jnp.linspace(
        -jnp.pi / 2.0 + dlat / 2.0,
        jnp.pi / 2.0 - dlat / 2.0,
        n_lat,
    )
    lon = jnp.linspace(0.0, 2.0 * jnp.pi - dlon, n_lon)

    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")  # (n_lat, n_lon)

    cos_lat = jnp.maximum(jnp.cos(lat), 1e-10)
    sin_lat = jnp.sin(lat)

    # Coriolis parameter
    f = 2.0 * omega * sin_lat[:, None] * jnp.ones((1, n_lon))

    # Metric terms — dx and dy span 2 cells (for centered differences)
    # Single-cell width in lon = R * dlon * cos(lat)
    # Two-cell span: dx = R * 2*dlon * cos(lat)
    dx = radius * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))
    dy = float(radius * 2.0 * dlat)

    # Cell area
    area = radius**2 * dlat * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))
    total_area = jnp.sum(area)

    return LatLonGrid(
        n_lat=n_lat,
        n_lon=n_lon,
        radius=float(radius),
        lat=lat,
        lon=lon,
        lat2d=lat2d,
        lon2d=lon2d,
        cos_lat=cos_lat,
        sin_lat=sin_lat,
        f=f,
        dx=dx,
        dy=dy,
        area=area,
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat),
    )
