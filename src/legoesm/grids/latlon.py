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
- ``dy`` is a 1D array of shape ``(n_lat,)`` so non-uniform-dlat
  grids (Mercator) work. For uniform-dlat grids every entry is the
  same value ``2 * R * dlat``. The single-cell height of row ``j``
  is ``dy[j] / 2.0``.
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
    dy: jax.Array               # (n_lat,) distance over 2 cells in lat [m]
    area: jax.Array             # (n_lat, n_lon) cell area [m^2]
    total_area: jax.Array       # scalar sum of all areas
    dlon: float                 # longitude spacing [rad]
    dlat: float                 # representative latitude spacing [rad]
    # For uniform-dlat grids this is the constant cell-row dlat. For
    # Mercator it is the smallest (equatorial) row's dlat — used only
    # for scalar CFL diagnostics. Operators must use ``dy`` (1D array)
    # for per-row cell heights, not ``dlat``.

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
    # dy is 1D over latitude — uniform here, but the array shape is
    # the same as for a Mercator grid so operators can treat them
    # uniformly.
    dy = radius * 2.0 * dlat * jnp.ones((n_lat,))

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
        dy=_c(dy),
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
    # dy is 1D over latitude (uniform-dlat regional grid → constant
    # values, but stored as an array for API uniformity with Mercator).
    dy = radius * 2.0 * dlat * jnp.ones((ny,))

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
        dy=_c(dy),
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat),
    )
    return grid, wall_mask


def create_mercator_grid(
    n_lon: int,
    lat_max_deg: float,
    lon_west_deg: float = 0.0,
    lon_east_deg: float = 360.0,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
) -> LatLonGrid:
    """Mercator (isotropic) latitude-longitude grid.

    Latitudes are placed via the Mercator projection so that cell
    heights ``dy(j)`` decrease poleward in proportion to ``cos(φ)``,
    matching the zonal spacing ``dx(j) = R cos(φ) · Δλ``. This yields
    approximately isotropic cells (``dx ≈ dy`` per row), the latitudinal
    scaling of the first baroclinic deformation radius (Hallberg 2013),
    and the placement used by NEMO's DINO ocean test case (Kamm et al.
    2025, GMD).

    Placement formula
    -----------------
    With ``Δλ = (lon_east - lon_west) / n_lon`` (radians), latitudes are
    isotropic when::

        sin(φ_f(k)) = tanh(Δλ · k)

    where ``k`` is an integer face index symmetric about the equator.
    The cell-face indices are ``k = -K, …, 0, …, +K`` with
    ``K = floor(arctanh(sin(lat_max_rad)) / Δλ)``, giving ``n_lat = 2K``
    cells and ``n_lat + 1`` faces. Cell centres sit at the half-integer
    indices ``k = -K + ½, …, +K − ½``.

    Sketch::

        ──── face k=+K     (lat ≈ +lat_max_deg)
         · centre k=K-½    (smallest cell, near pole)
        ──── face k=+K-1
        ⋮
        ──── face k=0      (equator)
         · centre k=½      (largest cell, near equator)
        ──── face k=-1
         · centre k=-½
        ──── face k=-K     (lat ≈ -lat_max_deg)

    Parameters
    ----------
    n_lon : int
        Number of zonal cells.
    lat_max_deg : float
        Symmetric N/S truncation latitude [degrees]. The actual
        northernmost cell face may sit slightly equatorward of
        ``lat_max_deg`` because ``K`` is rounded down to the nearest
        integer; the function emits no surprise — the grid is symmetric.
    lon_west_deg, lon_east_deg : float
        Zonal extent in degrees. Default 0–360 (global periodic). The
        zonal spacing ``Δλ`` derived from these sets the meridional
        placement (Mercator is isotropic per construction).
    radius : float, optional
        Sphere radius [m]. Default ``constants.R_earth``.
    omega : float, optional
        Rotation rate [rad/s]. Default ``constants.Omega``.
    dtype : optional
        Storage dtype for the JAX arrays. Defaults to the current
        precision policy.

    Returns
    -------
    LatLonGrid
        Grid with ``n_lat = 2K`` rows and ``n_lon`` columns, ``dy`` a
        1D ``(n_lat,)`` array, and ``dlat`` set to the equatorial
        (smallest, most CFL-stringent) cell-row dlat for backward-compat
        scalar usage.

    Notes
    -----
    - ``Δλ`` and the lat placement only depend on ``n_lon`` and the
      zonal extent — there is no separate ``n_lat`` knob. The number
      of latitudinal cells is determined by ``lat_max_deg``.
    - Operators must use ``grid.dy`` (1D) for per-row cell heights;
      ``grid.dlat`` is a representative scalar for diagnostics only.
    """
    if n_lon <= 0:
        raise ValueError(f"n_lon must be positive, got {n_lon}")
    if not (0.0 < lat_max_deg < 90.0):
        raise ValueError(
            f"lat_max_deg must lie in (0, 90), got {lat_max_deg}"
        )
    if lon_east_deg <= lon_west_deg:
        raise ValueError(
            f"lon_east_deg ({lon_east_deg}) must exceed "
            f"lon_west_deg ({lon_west_deg})"
        )

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

    # Zonal spacing — drives Mercator latitude placement (isotropic
    # cells when dy/dx = 1 ⇒ dφ/dλ = cos(φ)).
    dlon = (lon_east_deg - lon_west_deg) * jnp.pi / 180.0 / n_lon

    # K = max integer face index from the equator. ``n_lat = 2K``.
    lat_max_rad = lat_max_deg * jnp.pi / 180.0
    j_max_cont = float(jnp.arctanh(jnp.sin(lat_max_rad)) / dlon)
    K = int(jnp.floor(j_max_cont))
    if K < 1:
        raise ValueError(
            f"lat_max_deg={lat_max_deg} too small for n_lon={n_lon}: "
            f"Mercator placement yields zero cells. Increase lat_max_deg "
            f"or n_lon."
        )
    n_lat = 2 * K

    # Face and centre indices (k) symmetric about k=0 (equator face).
    # Compute placement in float64 regardless of the storage policy:
    # ``arcsin(tanh(·))`` loses precision in float32 near the poles
    # because ``tanh`` saturates to 1 quickly. The ``_c(...)`` cast at
    # the end downcasts the resulting fields to the policy dtype.
    k_face = jnp.arange(-K, K + 1, dtype=jnp.float64)        # (n_lat+1,)
    k_center = k_face[:-1] + 0.5                              # (n_lat,)

    # Mercator placement: sin(φ) = tanh(Δλ · k).
    lat_face = jnp.arcsin(jnp.tanh(dlon * k_face))            # (n_lat+1,)
    lat = jnp.arcsin(jnp.tanh(dlon * k_center))               # (n_lat,)

    # Longitude (cell centres).
    lon_w_rad = lon_west_deg * jnp.pi / 180.0
    lon = lon_w_rad + dlon * (jnp.arange(n_lon, dtype=jnp.float64) + 0.5)

    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")

    cos_lat = jnp.maximum(jnp.cos(lat), 1e-10)
    sin_lat = jnp.sin(lat)

    f = 2.0 * omega * sin_lat[:, None] * jnp.ones((1, n_lon))

    # dy(j) = 2 · R · (lat_face[j+1] - lat_face[j])  — "2-cell distance".
    # For Mercator this varies per row (decreases poleward).
    dy_cell = radius * (lat_face[1:] - lat_face[:-1])           # (n_lat,)
    dy = 2.0 * dy_cell

    # dx(j, i) = 2 · R · cos(lat_c(j)) · dlon (2-cell convention).
    dx = radius * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))

    # Cell area: R² · dlon · |sin(lat_face[j+1]) - sin(lat_face[j])|.
    # Equivalent to ∫∫ R² cos(φ) dφ dλ — exact for any orthogonal
    # spherical grid, not assuming uniform dlat.
    sin_face = jnp.sin(lat_face)
    area_lat = radius**2 * dlon * jnp.abs(sin_face[1:] - sin_face[:-1])  # (n_lat,)
    area = area_lat[:, None] * jnp.ones((1, n_lon))
    total_area = jnp.sum(area)

    # Representative dlat — smallest cell-row dlat (at the poles, where
    # Mercator cells are narrowest). Used only for scalar CFL diagnostics;
    # operators must use ``grid.dy`` for per-row spacing.
    dlat_repr = float(jnp.min(lat_face[1:] - lat_face[:-1]))

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
        dy=_c(dy),
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat_repr),
    )
