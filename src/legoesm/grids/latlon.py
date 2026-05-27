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


def _exact_uniform_cell_area_lat(
    radius, lat_centers, dlat, dlon, n_lon
):
    """Exact spherical-cap cell area on a uniform-dlat lat-lon grid.

    area[j, :] = R² dλ (sin(φ_face[j+1]) - sin(φ_face[j]))
              = 2 R² dλ sin(dφ/2) cos(φ_center[j])

    Midpoint formula `R² dφ dλ cos(φ)` overestimates total area by
    factor `dφ / (2 sin(dφ/2)) ≈ 1 + dφ²/24` — 79 ppm at n_lat=72,
    317 ppm at n_lat=36 — the dominant source of cross-grid total-
    area inconsistency on uniform lat-lon. Exact form sums to 4πR²
    for global cell-centered grids.
    """
    cos_lat = jnp.maximum(jnp.cos(lat_centers), 1e-10)
    area_lat = 2.0 * radius**2 * dlon * jnp.sin(0.5 * dlat) * cos_lat
    return area_lat[:, None] * jnp.ones((1, n_lon))


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
    # Mercator the cell-row dlat varies with latitude (largest at the
    # equator, smallest near the truncation latitude); this scalar
    # stores the smallest (most CFL-stringent) row's value. Used only
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

    dlat = jnp.pi / n_lat
    dlon = 2.0 * jnp.pi / n_lon

    # Cell-centered latitudes: avoid exact poles
    lat = jnp.linspace(
        -jnp.pi / 2.0 + dlat / 2.0,
        jnp.pi / 2.0 - dlat / 2.0,
        n_lat,
    )
    lon = jnp.linspace(0.0, 2.0 * jnp.pi - dlon, n_lon)

    return _build_uniform_latlon_grid_from_axes(
        lat=lat, lon=lon, dlat=dlat, dlon=dlon,
        radius=radius, omega=omega, dtype=dtype,
    )


def _build_uniform_latlon_grid_from_axes(
    lat: jax.Array,
    lon: jax.Array,
    dlat: float,
    dlon: float,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
) -> LatLonGrid:
    """Assemble a uniform-``dlat`` ``LatLonGrid`` from pre-computed axes.

    Shared body of ``create_latlon_grid`` and the MPI band-extension
    helper :func:`legoesm.parallel.latlon_mpi.build_padded_grid` — the
    latter feeds in a ``lat`` array extrapolated past one or both
    poles by ``dlat`` so vertex areas stay strictly positive in the
    halo region.  Centralising the metric construction here lets the
    MPI path reuse the exact same formulas instead of inlining a
    near-duplicate that could drift from the serial reference.

    Parameters
    ----------
    lat : (n_lat,) array of cell-center latitudes [rad]; need not lie
        in ``[-π/2, π/2]`` (the MPI extension produces values past
        the poles for halo rows).
    lon : (n_lon,) array of cell-center longitudes [rad].
    dlat : scalar latitude spacing [rad].  Uniform across ``lat`` is
        assumed; for Mercator / variable-dlat grids this helper is
        not appropriate.
    dlon : scalar longitude spacing [rad].
    radius, omega : sphere radius [m] and rotation rate [rad/s].
    dtype : optional storage dtype; falls back to the active
        precision policy when ``None``.

    Returns
    -------
    LatLonGrid
    """
    n_lat = lat.shape[0]
    n_lon = lon.shape[0]

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")  # (n_lat, n_lon)

    # ``abs(cos(lat))`` is identical to ``cos(lat)`` for lat in
    # [-π/2, π/2] (cos is non-negative there) but stays positive for
    # latitudes extrapolated past the poles — which is what the
    # MPI band-extension helper feeds in for halo rows
    # (``build_padded_grid`` in legoesm.parallel.latlon_mpi).
    # Without ``abs`` the post-pole cos values went negative,
    # clamped to 1e-10, and produced ``dx ≈ 0`` → infinite zonal
    # gradients in the halo region.
    cos_lat = jnp.maximum(jnp.abs(jnp.cos(lat)), 1e-10)
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

    # Cell area: exact spherical cap (sums to 4πR² for global grid)
    area = _exact_uniform_cell_area_lat(radius, lat, dlat, dlon, n_lon)
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

    area = _exact_uniform_cell_area_lat(radius, lat, dlat, dlon, nx)
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
        1D ``(n_lat,)`` array, and ``dlat`` set to the *smallest* (most
        CFL-stringent, polar) cell-row dlat for backward-compat scalar
        usage. Mercator cell-row dlat is largest at the equator and
        smallest near the truncation latitude.

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

    # Representative dlat — smallest cell-row dlat. Mercator cell-row
    # dlat is largest at the equator (φ = 0, cos φ = 1) and smallest
    # near the truncation latitude (cos φ → 0 ⇒ dφ/dk → 0), so the
    # min() here picks the polar/truncation rows. Used only for scalar
    # CFL diagnostics; operators must use ``grid.dy`` for per-row
    # spacing.
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


# =========================================================================
# Tripolar-ready C-grid geometry
# =========================================================================


def ensure_geometry(
    grid,
    omega: float = constants.Omega,
) -> "LatLonCGridGeometry":
    """Convert a ``LatLonGrid`` to ``LatLonCGridGeometry`` if needed.

    If *grid* is already a ``LatLonCGridGeometry``, it is returned
    unchanged.  If it is a ``LatLonGrid``, the per-cell metric arrays
    are computed analytically (bit-exact with the inline operator path).

    This should be called **once** at model-construction time (e.g. in
    ``LatLonCGridOceanModel.__init__``), not per operator call.

    Parameters
    ----------
    grid : LatLonGrid or LatLonCGridGeometry
    omega : float
        Rotation rate [rad/s].  Only used when converting from
        ``LatLonGrid`` (which does not store omega).

    Returns
    -------
    LatLonCGridGeometry
    """
    if isinstance(grid, LatLonCGridGeometry):
        return grid
    # Duck-type check: if it has dx_u, assume it's geometry-like
    if hasattr(grid, "dx_u") and hasattr(grid, "fold"):
        return grid  # type: ignore[return-value]
    # Convert LatLonGrid -> LatLonCGridGeometry. Pass the input grid's
    # actual 1-D lat/lon arrays so regional / channel grids preserve
    # their bounds — otherwise create_latlon_geometry would silently
    # rebuild a GLOBAL lat-lon grid from ``n_lat`` / ``n_lon`` alone,
    # inflating ``dx`` and ``dy`` by the ratio between global and
    # regional extents (the 2026-05-17 Petersen-channel investigation
    # caught a 600x dx inflation that damped the gravity-current PGF
    # to ~0.07 m/s vs the Veros peer's 0.86 m/s).
    return create_latlon_geometry(
        n_lat=grid.n_lat,
        n_lon=grid.n_lon,
        radius=grid.radius,
        omega=omega,
        lat_1d=getattr(grid, "lat", None),
        lon_1d=getattr(grid, "lon", None),
    )


class FoldDescriptor(NamedTuple):
    """Description of the bipolar fold seam for a tripolar grid.

    On a regular lat-lon grid, ``is_active`` is ``False`` and all other
    fields are unused sentinels.  On a tripolar grid the fold is the
    northern row where cell ``(i, fold_j)`` is identified with cell
    ``(perm_T[i], fold_j)``.

    Attributes
    ----------
    is_active : bool
        True for tripolar grids, False for regular lat-lon.
    fold_j : int
        Row index of the fold seam for T/u points (the northernmost
        interior row).
    cap_j : int
        Southernmost row of the bipolar cap (where grid lines start to
        deviate from regular lat-lon).
    perm_T : jax.Array
        (n_lon,) int32 — i-reversal permutation for T and u stagger
        points.  On ORCA1 with a T-fold: ``perm_T[i] = n_lon - 1 - i``.
    perm_v : jax.Array
        (n_lon,) int32 — i-reversal permutation for v and q (vertex)
        stagger points.
    vector_sign_u : float
        Sign flip for u-component across the fold (typically -1.0).
    vector_sign_v : float
        Sign flip for v-component across the fold (typically -1.0).
    """
    is_active: bool
    fold_j: int
    cap_j: int
    perm_T: jax.Array
    perm_v: jax.Array
    vector_sign_u: float
    vector_sign_v: float


def _inactive_fold(n_lon: int) -> FoldDescriptor:
    """Create a no-op fold descriptor for regular lat-lon grids."""
    dummy = jnp.arange(n_lon, dtype=jnp.int32)
    return FoldDescriptor(
        is_active=False,
        fold_j=0,
        cap_j=0,
        perm_T=dummy,
        perm_v=dummy,
        vector_sign_u=-1.0,
        vector_sign_v=-1.0,
    )


class LatLonCGridGeometry(NamedTuple):
    """Per-cell metric container for orthogonal curvilinear C-grids.

    This NamedTuple stores pre-computed metric arrays at every stagger
    point (T, u, v, q) so that operators never need to compute
    ``cos(lat)`` inline.  On a regular lat-lon grid the metrics are
    analytically derived from ``dlat``/``dlon``; on a tripolar grid they
    come from a grid file (e.g. ORCA1 NetCDF).

    The staggering convention follows the ocean C-grid:
      - T-points (cell centers): shape ``(n_lat, n_lon)``
      - u-points (lon interfaces): shape ``(n_lat, n_lon+1)``
      - v-points (lat interfaces): shape ``(n_lat+1, n_lon)``
      - q-points (vertices/corners): shape ``(n_lat+1, n_lon+1)``

    Attributes
    ----------
    n_lat, n_lon : int
        Number of cells in latitude and longitude.
    radius : float
        Sphere radius [m].

    lat_T, lon_T : jax.Array
        2D geographic coordinates at T-points [rad].

    dx_T, dy_T : jax.Array
        Single-cell width/height at T-points [m].
    area_T : jax.Array
        Cell area at T-points [m^2].
    total_area : jax.Array
        Scalar sum of all T-point areas.

    dx_u, dy_u : jax.Array
        Zonal/meridional spacing at u-points [m].

    dx_v, dy_v : jax.Array
        Zonal/meridional spacing at v-points [m].

    area_q : jax.Array
        Dual-cell area at vertex (q) points [m^2].

    f_T : jax.Array
        Coriolis parameter at T-points.
    f_u : jax.Array
        Coriolis parameter at u-points.
    f_v : jax.Array
        Coriolis parameter at v-points.

    cos_alpha_u, sin_alpha_u : jax.Array
        Rotation from local i-axis to geographic east at u-points.
        Zero on regular lat-lon grids.
    cos_alpha_v, sin_alpha_v : jax.Array
        Rotation from local j-axis to geographic north at v-points.
        Zero on regular lat-lon grids.

    fold : FoldDescriptor
        Describes the bipolar fold seam.  ``fold.is_active`` is False
        for regular lat-lon grids.

    cos_lat, sin_lat : jax.Array
        Legacy 1D arrays (n_lat,) for backward compatibility with
        operators that have not yet been migrated to per-cell metrics.
    lat, lon : jax.Array
        Legacy 1D arrays (n_lat,) and (n_lon,).
    dlon, dlat : float
        Legacy scalar spacings. Zero on tripolar grids (sentinel to
        catch unmigrated code).
    """
    # Shape / scale
    n_lat: int
    n_lon: int
    radius: float

    # 2D coordinates at T-points
    lat_T: jax.Array   # (n_lat, n_lon)
    lon_T: jax.Array   # (n_lat, n_lon)

    # T-point metrics
    dx_T: jax.Array    # (n_lat, n_lon) single-cell zonal width [m]
    dy_T: jax.Array    # (n_lat, n_lon) single-cell meridional height [m]
    area_T: jax.Array  # (n_lat, n_lon) cell area [m^2]
    total_area: jax.Array  # scalar

    # u-point metrics (lon interfaces)
    dx_u: jax.Array    # (n_lat, n_lon+1) zonal spacing [m]
    dy_u: jax.Array    # (n_lat, n_lon+1) meridional extent [m]

    # v-point metrics (lat interfaces)
    dx_v: jax.Array    # (n_lat+1, n_lon) zonal extent [m]
    dy_v: jax.Array    # (n_lat+1, n_lon) meridional spacing [m]

    # Vertex (q-point) area
    area_q: jax.Array  # (n_lat+1, n_lon+1) dual cell area [m^2]

    # Coriolis at all stagger points
    f_T: jax.Array     # (n_lat, n_lon)
    f_u: jax.Array     # (n_lat, n_lon+1)
    f_v: jax.Array     # (n_lat+1, n_lon)

    # Rotation angles (zero outside bipolar cap)
    cos_alpha_u: jax.Array  # (n_lat, n_lon+1)
    sin_alpha_u: jax.Array  # (n_lat, n_lon+1)
    cos_alpha_v: jax.Array  # (n_lat+1, n_lon)
    sin_alpha_v: jax.Array  # (n_lat+1, n_lon)

    # Fold descriptor
    fold: FoldDescriptor

    # Legacy compatibility fields
    cos_lat: jax.Array  # (n_lat,) clamped cos(lat) at cell centers
    sin_lat: jax.Array  # (n_lat,) sin(lat) at cell centers
    lat: jax.Array      # (n_lat,) 1D cell-center latitudes [rad]
    lon: jax.Array      # (n_lon,) 1D cell-center longitudes [rad]
    dlon: float         # scalar longitude spacing (0.0 sentinel for tripole)
    dlat: float         # scalar latitude spacing (0.0 sentinel for tripole)

    # ------------------------------------------------------------------
    # GridProtocol properties
    # ------------------------------------------------------------------

    @property
    def grid_lat(self) -> jax.Array:
        return self.lat_T

    @property
    def grid_lon(self) -> jax.Array:
        return self.lon_T

    @property
    def grid_area(self) -> jax.Array:
        return self.area_T

    @property
    def grid_total_area(self):
        return self.total_area

    @property
    def grid_coriolis(self) -> jax.Array:
        return self.f_T

    @property
    def grid_radius(self) -> float:
        return self.radius

    @property
    def n(self) -> int:
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

    # Convenience aliases so LatLonCGridGeometry can stand in for
    # LatLonGrid in code that accesses ``grid.f``, ``grid.area``,
    # ``grid.lat2d``, ``grid.lon2d``, or ``grid.dx``/``grid.dy``.

    @property
    def f(self) -> jax.Array:
        return self.f_T

    @property
    def area(self) -> jax.Array:
        return self.area_T

    @property
    def lat2d(self) -> jax.Array:
        return self.lat_T

    @property
    def lon2d(self) -> jax.Array:
        return self.lon_T

    @property
    def dx(self) -> jax.Array:
        """Two-cell zonal span at T-points [m], matching LatLonGrid.dx."""
        return 2.0 * self.dx_T

    @property
    def dy(self) -> jax.Array:
        """Two-cell meridional span [m], shape (n_lat,), matching LatLonGrid.dy."""
        # dy_T is (n_lat, n_lon); take column 0 for 1D — on regular grids
        # all columns are identical; on tripolar the representative 1D dy
        # is used only for CFL diagnostics, not operator metrics.
        return 2.0 * self.dy_T[:, 0]


def create_latlon_geometry(
    n_lat: int,
    n_lon: int | None = None,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    *,
    lat_1d: jax.Array | None = None,
    lon_1d: jax.Array | None = None,
) -> LatLonCGridGeometry:
    """Create a regular lat-lon ``LatLonCGridGeometry``.

    This produces per-cell metric arrays that are **bit-exact equivalent**
    to the values that ``latlon_cgrid_operators.py`` currently computes
    inline from ``LatLonGrid.cos_lat``, ``dlon``, ``dlat``, ``radius``.

    The fold descriptor is inactive (regular lat-lon has no fold).

    Parameters
    ----------
    n_lat : int
        Number of latitude cells.
    n_lon : int, optional
        Number of longitude cells.  Defaults to ``2 * n_lat``.
    radius : float
        Sphere radius [m].
    omega : float
        Rotation rate [rad/s].
    dtype : optional
        Storage dtype.  Defaults to the precision policy's storage type.
    lat_1d : jax.Array, optional
        Cell-center latitudes in radians (length ``n_lat``).  When given,
        ``dlat`` is derived from ``lat_1d[1] - lat_1d[0]`` and the global
        defaults (full pole-to-pole span) are bypassed.  Used by
        :func:`ensure_geometry` to preserve the bounds of regional or
        channel grids — previously this function always reconstructed a
        global grid from ``n_lat``/``n_lon`` alone, which silently broke
        every regional latlon ocean run by inflating ``dx`` / ``dy`` by
        the ratio between the global and regional extents.
    lon_1d : jax.Array, optional
        Cell-center longitudes in radians (length ``n_lon``).  Same
        rationale as ``lat_1d``.

    Returns
    -------
    LatLonCGridGeometry
    """
    if n_lon is None:
        n_lon = 2 * n_lat

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

    # ------- 1D coordinate arrays (native precision) -------
    _is_variable_dlat = False
    if lat_1d is None:
        dlat = jnp.pi / n_lat
        lat_1d = jnp.linspace(
            -jnp.pi / 2.0 + dlat / 2.0,
            jnp.pi / 2.0 - dlat / 2.0,
            n_lat,
        )
    else:
        lat_1d = jnp.asarray(lat_1d)
        if lat_1d.shape != (n_lat,):
            raise ValueError(
                f"lat_1d must have shape ({n_lat},), got {lat_1d.shape}"
            )
        dlat = lat_1d[1] - lat_1d[0] if n_lat > 1 else jnp.pi / n_lat
        # Check for variable dlat (Mercator): compare spacing at the
        # boundary vs the middle of the domain.
        if n_lat > 4:
            _dlat_mid = lat_1d[n_lat // 2] - lat_1d[n_lat // 2 - 1]
            _is_variable_dlat = abs(float(_dlat_mid - dlat)) > 1e-6 * abs(float(dlat))
        else:
            _is_variable_dlat = False

    if lon_1d is None:
        dlon = 2.0 * jnp.pi / n_lon
        lon_1d = jnp.linspace(0.0, 2.0 * jnp.pi - dlon, n_lon)
    else:
        lon_1d = jnp.asarray(lon_1d)
        if lon_1d.shape != (n_lon,):
            raise ValueError(
                f"lon_1d must have shape ({n_lon},), got {lon_1d.shape}"
            )
        dlon = lon_1d[1] - lon_1d[0] if n_lon > 1 else 2.0 * jnp.pi / n_lon

    # Legacy fields: compute in native precision, cast at end to match
    # the exact path that create_latlon_grid() uses.
    _c = lambda a: a.astype(dtype) if hasattr(a, 'astype') else a
    cos_lat_1d = jnp.maximum(jnp.cos(lat_1d), 1e-10)
    sin_lat_1d = jnp.sin(lat_1d)

    # 2D meshgrid at T-points
    lat_T, lon_T = jnp.meshgrid(lat_1d, lon_1d, indexing="ij")

    # Legacy fields that match create_latlon_grid bit-exact
    f_legacy = 2.0 * omega * sin_lat_1d[:, None] * jnp.ones((1, n_lon))

    # For variable-dlat grids (Mercator), compute per-row cell heights
    # from lat_face differences (exact spherical area).
    if _is_variable_dlat:
        # Reconstruct face latitudes from cell centers (inverse of
        # Mercator center placement).  Face j sits halfway between
        # center j-1 and center j.
        lat_face_interior = 0.5 * (lat_1d[:-1] + lat_1d[1:])  # (n_lat-1,)
        # Extend to south and north boundaries symmetrically
        lat_face_south = lat_1d[0] - 0.5 * (lat_1d[1] - lat_1d[0])
        lat_face_north = lat_1d[-1] + 0.5 * (lat_1d[-1] - lat_1d[-2])
        lat_face = jnp.concatenate([
            jnp.array([lat_face_south]),
            lat_face_interior,
            jnp.array([lat_face_north]),
        ])  # (n_lat+1,)
        dlat_1d = lat_face[1:] - lat_face[:-1]  # (n_lat,) per-row dlat
        # Exact spherical area: R² * dlon * |sin(φ_face[j+1]) - sin(φ_face[j])|
        sin_face = jnp.sin(lat_face)
        area_lat_1d = radius**2 * dlon * jnp.abs(sin_face[1:] - sin_face[:-1])
        area_legacy = area_lat_1d[:, None] * jnp.ones((1, n_lon))
    else:
        dlat_1d = None  # uniform — use scalar dlat everywhere
        area_legacy = _exact_uniform_cell_area_lat(
            radius, lat_1d, dlat, dlon, n_lon
        )

    total_area = jnp.sum(area_legacy)

    # Cast 1D coordinates + legacy fields to storage dtype.  All
    # subsequent metric computations use the cast values so that
    # operator-inline and geometry-precomputed paths are bit-exact.
    lat_s = _c(lat_1d)
    lon_s = _c(lon_1d)
    cos_lat_s = _c(cos_lat_1d)
    sin_lat_s = _c(sin_lat_1d)

    # ------- T-point metrics -------
    # Single-cell zonal width: R * dlon * cos(lat)
    dx_T = radius * dlon * cos_lat_s[:, jnp.newaxis] * jnp.ones((1, n_lon))
    # Single-cell meridional height
    if dlat_1d is not None:
        # Variable dlat (Mercator): per-row cell height
        dy_T = (_c(radius * dlat_1d))[:, jnp.newaxis] * jnp.ones((1, n_lon))
    else:
        dy_T = jnp.full((n_lat, n_lon), float(radius * dlat), dtype=dtype)
    # Cell area — use exact spherical area for variable-dlat grids
    area_T = _c(area_legacy)

    # ------- u-point metrics (n_lat, n_lon+1) -------
    # dx_u = R * dlon * cos(lat) — same as gradient_x_cgrid uses.
    dx_u = (
        radius * dlon * cos_lat_s[:, jnp.newaxis]
        * jnp.ones((1, n_lon + 1))
    )
    # dy_u = meridional extent of the u-face
    if dlat_1d is not None:
        dy_u = (_c(radius * dlat_1d))[:, jnp.newaxis] * jnp.ones((1, n_lon + 1))
    else:
        dy_u = jnp.full((n_lat, n_lon + 1), float(radius * dlat), dtype=dtype)

    # ------- v-point metrics (n_lat+1, n_lon) -------
    # v-face latitudes: midpoints between cell centers, with poles at
    # ends.  cos(lat_v) at poles is exactly 0 (wall BC in regular
    # lat-lon).  This matches the inline computation in divergence_cgrid.
    lat_v_interior = 0.5 * (lat_s[:-1] + lat_s[1:])
    cos_lat_v_interior = jnp.cos(lat_v_interior)
    cos_lat_v = jnp.pad(cos_lat_v_interior, (1, 1))  # (n_lat+1,)

    # dx_v = R * cos(lat_v) * dlon — zonal extent of the v-face
    dx_v = (
        radius * cos_lat_v[:, jnp.newaxis] * dlon
        * jnp.ones((1, n_lon))
    )
    # dy_v = meridional spacing at v-faces
    if dlat_1d is not None:
        # Variable dlat: v-face spacing = distance between adjacent cell
        # centers.  Interior: 0.5*(dlat[j] + dlat[j+1]).  Boundary: dlat[0]
        # and dlat[-1] for the half-cells at the poles.
        dy_v_interior = 0.5 * radius * (dlat_1d[:-1] + dlat_1d[1:])  # (n_lat-1,)
        dy_v_south = radius * dlat_1d[0]
        dy_v_north = radius * dlat_1d[-1]
        dy_v_1d = jnp.concatenate([
            jnp.array([float(dy_v_south)]),
            _c(dy_v_interior),
            jnp.array([float(dy_v_north)]),
        ])  # (n_lat+1,)
        dy_v = dy_v_1d[:, jnp.newaxis] * jnp.ones((1, n_lon))
    else:
        dy_v = jnp.full((n_lat + 1, n_lon), float(radius * dlat), dtype=dtype)

    # ------- Vertex (q-point) area (n_lat+1, n_lon+1) -------
    # Matches curl_vertex_cgrid: A_q(i) = R^2 * dlon * |sin(lat[i]) - sin(lat[i-1])|
    # Uses sin_lat_s (storage-dtype) so bit-exact with inline operator.
    sin_ext = jnp.pad(sin_lat_s, (1, 1), constant_values=(-1.0, 1.0))
    area_q_1d = radius**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])
    area_q = area_q_1d[:, jnp.newaxis] * jnp.ones((1, n_lon + 1))

    # ------- Coriolis at all stagger points -------
    # f_T uses the cast (storage-dtype) sin_lat for consistency
    f_T = _c(f_legacy)

    # f at u-points: average of cells sharing the lon-face
    f_u_interior = 0.5 * (jnp.roll(f_T, 1, axis=1) + f_T)
    f_u = jnp.concatenate([f_u_interior, f_u_interior[:, 0:1]], axis=1)

    # f at v-points: average of cells sharing the lat-face
    f_v_interior = 0.5 * (f_T[:-1] + f_T[1:])
    f_v = jnp.concatenate([f_T[0:1], f_v_interior, f_T[-1:]], axis=0)

    # ------- Rotation angles (zero for regular lat-lon) -------
    cos_alpha_u = jnp.ones((n_lat, n_lon + 1), dtype=dtype)
    sin_alpha_u = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
    cos_alpha_v = jnp.ones((n_lat + 1, n_lon), dtype=dtype)
    sin_alpha_v = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)

    # ------- Fold descriptor (inactive) -------
    fold = _inactive_fold(n_lon)

    return LatLonCGridGeometry(
        n_lat=n_lat,
        n_lon=n_lon,
        radius=float(radius),
        lat_T=_c(lat_T),
        lon_T=_c(lon_T),
        dx_T=dx_T,
        dy_T=dy_T,
        area_T=area_T,
        total_area=total_area,
        dx_u=dx_u,
        dy_u=dy_u,
        dx_v=dx_v,
        dy_v=dy_v,
        area_q=area_q,
        f_T=f_T,
        f_u=f_u,
        f_v=f_v,
        cos_alpha_u=cos_alpha_u,
        sin_alpha_u=sin_alpha_u,
        cos_alpha_v=cos_alpha_v,
        sin_alpha_v=sin_alpha_v,
        fold=fold,
        cos_lat=_c(cos_lat_1d),
        sin_lat=_c(sin_lat_1d),
        lat=_c(lat_1d),
        lon=_c(lon_1d),
        dlon=float(dlon),
        dlat=float(dlat),
    )
