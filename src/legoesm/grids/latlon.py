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

    @property
    def grid_shape_2d(self) -> tuple[int, ...]:
        return (self.n_lat, self.n_lon)

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
    dtype=None,
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

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

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

    _c = lambda a: a.astype(dtype) if hasattr(a, 'astype') else a
    return LatLonGrid(
        n_lat=n_lat,
        n_lon=n_lon,
        radius=float(radius),
        lat=_c(lat),
        lon=_c(lon),
        lat2d=_c(lat2d),
        lon2d=_c(lon2d),
        cos_lat=_c(cos_lat),
        sin_lat=_c(sin_lat),
        f=_c(f),
        dx=_c(dx),
        dy=dy,
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat),
    )


def create_regional_latlon_grid(
    n_lat: int,
    n_lon: int,
    lat_south: float,
    lat_north: float,
    lon_west: float = 0.0,
    lon_east: float = 60.0,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    periodic_x: bool = False,
) -> tuple[LatLonGrid, jax.Array]:
    """Create a regional lat-lon grid covering a limited domain.

    The grid spans the specified lat/lon bounding box with *n_lat* x
    *n_lon* interior cells.  By default, a 1-cell wall (land mask = 0)
    is placed on all four boundaries for closed-basin experiments.

    When ``periodic_x=True``, no east/west wall cells are added and
    longitude spans from *lon_west* to *lon_east* with periodic
    boundary conditions for channel-like experiments (e.g. Eady
    baroclinic instability, ACC channel).  The latitude dimension
    still has 1-cell walls at north and south.

    Operators remain periodic in longitude via ``jnp.roll``.  In the
    closed-basin case (``periodic_x=False``), the wall mask + Neumann
    fill ensures no-normal-flow at the basin edges.

    Parameters
    ----------
    n_lat, n_lon : int
        Number of interior cells (excluding wall cells).
    lat_south, lat_north : float
        Southern and northern boundaries [degrees].
    lon_west, lon_east : float
        Western and eastern boundaries [degrees].
        When ``periodic_x=True``, these set the zonal extent of the
        periodic channel (default 0-360°).
    radius : float
        Sphere radius [m].
    omega : float
        Rotation rate [rad/s].
    periodic_x : bool
        If True, longitude spans 360° with no east/west walls (channel).

    Returns
    -------
    grid : LatLonGrid
        Grid with *n_lat + 2* latitude rows (wall cells at N/S).
        Longitude columns: *n_lon + 2* if closed, *n_lon* if periodic.
    wall_mask : jax.Array
        1 = ocean interior, 0 = wall.
    """
    if lat_south >= lat_north:
        raise ValueError(f"lat_south={lat_south} must be < lat_north={lat_north}")
    if not periodic_x and lon_west >= lon_east:
        raise ValueError(f"lon_west={lon_west} must be < lon_east={lon_east}")

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

    # Latitude: always has wall cells at N/S
    ny = n_lat + 2
    lat_s_rad = jnp.deg2rad(lat_south)
    lat_n_rad = jnp.deg2rad(lat_north)
    dlat = (lat_n_rad - lat_s_rad) / n_lat

    lat = jnp.linspace(
        float(lat_s_rad) - dlat / 2.0,
        float(lat_n_rad) + dlat / 2.0,
        ny,
    )

    if periodic_x:
        # Channel: periodic in x over [lon_west, lon_east), no wall cells
        nx = n_lon
        lon_w_rad = jnp.deg2rad(lon_west)
        lon_e_rad = jnp.deg2rad(lon_east)
        dlon = (lon_e_rad - lon_w_rad) / n_lon
        lon = jnp.linspace(
            float(lon_w_rad), float(lon_e_rad) - float(dlon), n_lon)
    else:
        # Closed basin: wall cells on east/west
        nx = n_lon + 2
        lon_w_rad = jnp.deg2rad(lon_west)
        lon_e_rad = jnp.deg2rad(lon_east)
        dlon = (lon_e_rad - lon_w_rad) / n_lon
        lon = jnp.linspace(
            float(lon_w_rad) - dlon / 2.0,
            float(lon_e_rad) + dlon / 2.0,
            nx,
        )

    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")

    cos_lat = jnp.maximum(jnp.cos(lat), 1e-10)
    sin_lat = jnp.sin(lat)

    f = 2.0 * omega * sin_lat[:, None] * jnp.ones((1, nx))

    dx = radius * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, nx))
    dy = float(radius * 2.0 * dlat)

    area = radius**2 * dlat * dlon * cos_lat[:, None] * jnp.ones((1, nx))
    total_area = jnp.sum(area)

    # Wall mask: walls at N/S always; E/W walls only for closed basin
    wall_mask = jnp.ones((ny, nx), dtype=dtype)
    wall_mask = wall_mask.at[0, :].set(0.0)   # south wall
    wall_mask = wall_mask.at[-1, :].set(0.0)  # north wall
    if not periodic_x:
        wall_mask = wall_mask.at[:, 0].set(0.0)   # west wall
        wall_mask = wall_mask.at[:, -1].set(0.0)  # east wall

    _c = lambda a: a.astype(dtype) if hasattr(a, 'astype') else a
    grid = LatLonGrid(
        n_lat=ny,
        n_lon=nx,
        radius=float(radius),
        lat=_c(lat),
        lon=_c(lon),
        lat2d=_c(lat2d),
        lon2d=_c(lon2d),
        cos_lat=_c(cos_lat),
        sin_lat=_c(sin_lat),
        f=_c(f),
        dx=_c(dx),
        dy=dy,
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat),
    )
    return grid, wall_mask
