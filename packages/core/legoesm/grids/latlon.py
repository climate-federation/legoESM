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
import numpy as np

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
    # v-face (lat-interface) coordinates — length n_lat+1.  Stored
    # explicitly so the Stage 3-E polar filter can build a v-face
    # mask whose latitudes match the actual v-face positions on
    # both the global grid AND each MPI band (Codex review round 3
    # caught the prior approach reconstructing v-face lat with ±π/2
    # padding, which mislabels interior bands' endpoints as poles).
    lat_v: jax.Array            # (n_lat+1,) lat at v-face interfaces
    cos_lat_v: jax.Array        # (n_lat+1,) cos(lat_v)
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

    # Planetary rotation rate [rad/s] — the scalar the ``f`` field was
    # built from.  Stored so dynamics needing the VERTEX planetary
    # vorticity (``absolute_vorticity_coriolis``) can rebuild
    # 2*omega*sin(lat_vertex) exactly on ANY subdomain: recovering it
    # from the local ``f`` rows fails on an MPI band that owns only the
    # equator row (sin(lat) = 0), and hardcoding constants.Omega
    # silently kept omega=0 grids rotating (#521, codex 2026-07-03).
    # APPENDED at the NamedTuple end WITH a default so positional
    # constructions keep working (mirrors the ``radius: float``
    # scalar-leaf precedent).  NOTE: this does add one pytree leaf;
    # no repo code pins the LatLonGrid leaf count (checked 2026-07-03).
    omega: float = constants.Omega

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

    @property
    def weights(self) -> jax.Array:
        """Per-latitude area weights for loss area-weighting.

        Proportional to the cell-row area (∝ cos lat for a uniform-dlon
        grid); consumers (``losses._lat_weighted_mean``) normalise by the
        sum so only the relative profile matters.  Mirrors
        ``GaussianGrid.weights`` so ``combined_loss`` / ``carry_mse``
        area-weight lat-lon losses automatically — without it the lat-lon
        loss is a uniform mean that over-weights the poles and distorts
        the bias and radiation-flux global-mean terms (AIMIP codex
        review #3).
        """
        return self.cos_lat


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

    return build_uniform_latlon_grid_from_axes(
        lat=lat, lon=lon, dlat=dlat, dlon=dlon,
        radius=radius, omega=omega, dtype=dtype,
    )


def compute_v_face_coords(
    lat: jax.Array, dlat: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute ``(lat_v, cos_lat_v)`` at the v-face (lat-interface).

    Length ``n_lat+1``.  Interior: ``lat_v[i] = (lat[i-1] + lat[i])/2``
    for i in [1, n_lat).  Boundary: half-cell extrapolation past the
    cell-center endpoints.  For a GLOBAL grid spanning
    ``[-π/2 + dlat/2, π/2 - dlat/2]`` this puts the boundary v-faces
    at ±π/2.  Under MPI, a band's south/north endpoints are interior
    latitudes; the same half-cell extrapolation gives the correct
    v-face there too — do NOT hard-code ±π/2 (Codex review Stage 3-E
    round 3 caught this regression).

    ``cos_lat_v`` is clamped against zero via ``jnp.maximum(...,
    1e-10)`` so operators that divide by it stay finite at the global
    poles.
    """
    lat_v_interior = 0.5 * (lat[:-1] + lat[1:])
    lat_v = jnp.concatenate([
        lat[:1] - 0.5 * dlat,
        lat_v_interior,
        lat[-1:] + 0.5 * dlat,
    ])
    cos_lat_v = jnp.maximum(jnp.abs(jnp.cos(lat_v)), 1e-10)
    return lat_v, cos_lat_v


def build_uniform_latlon_grid_from_axes(
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

    lat_v, cos_lat_v = compute_v_face_coords(lat, dlat)

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
        lat_v=_c(lat_v),
        cos_lat_v=_c(cos_lat_v),
        f=_c(f),
        dx=_c(dx),
        dy=_c(dy),
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat),
        omega=float(omega),
    )


def _regional_lon_axis(
    n_lon: int,
    lon_west: float,
    lon_east: float,
    periodic_x: bool,
):
    """Longitude axis shared by the regional / stretched lat-lon builders.

    Returns ``(lon, dlon, nx)`` with *lon* the cell-centre longitudes
    [rad].  ``periodic_x=True``: ``nx = n_lon`` cells spanning
    ``[lon_west, lon_east)`` (channel, no E/W walls).
    ``periodic_x=False``: ``nx = n_lon + 2`` with one wall column added
    on each side (closed basin).  Factored out of
    :func:`create_regional_latlon_grid` so that
    :func:`create_stretched_latlon_grid` produces bit-identical zonal
    axes for the same inputs.
    """
    lon_w_rad = jnp.deg2rad(lon_west)
    lon_e_rad = jnp.deg2rad(lon_east)
    dlon = (lon_e_rad - lon_w_rad) / n_lon
    if periodic_x:
        nx = n_lon
        lon = jnp.linspace(
            float(lon_w_rad), float(lon_e_rad) - float(dlon), n_lon)
    else:
        nx = n_lon + 2
        lon = jnp.linspace(
            float(lon_w_rad) - dlon / 2.0,
            float(lon_e_rad) + dlon / 2.0,
            nx,
        )
    return lon, dlon, nx


def _regional_wall_mask(ny: int, nx: int, periodic_x: bool, dtype):
    """Wall mask shared by the regional / stretched lat-lon builders.

    1 = ocean interior, 0 = wall.  Walls at N/S always; E/W walls only
    for the closed-basin (``periodic_x=False``) case.
    """
    wall_mask = jnp.ones((ny, nx), dtype=dtype)
    wall_mask = wall_mask.at[0, :].set(0.0)   # south wall
    wall_mask = wall_mask.at[-1, :].set(0.0)  # north wall
    if not periodic_x:
        wall_mask = wall_mask.at[:, 0].set(0.0)   # west wall
        wall_mask = wall_mask.at[:, -1].set(0.0)  # east wall
    return wall_mask


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

    lon, dlon, nx = _regional_lon_axis(n_lon, lon_west, lon_east, periodic_x)

    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")

    cos_lat = jnp.maximum(jnp.cos(lat), 1e-10)
    sin_lat = jnp.sin(lat)
    lat_v, cos_lat_v = compute_v_face_coords(lat, dlat)

    f = 2.0 * omega * sin_lat[:, None] * jnp.ones((1, nx))

    dx = radius * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, nx))
    # dy is 1D over latitude (uniform-dlat regional grid → constant
    # values, but stored as an array for API uniformity with Mercator).
    dy = radius * 2.0 * dlat * jnp.ones((ny,))

    area = _exact_uniform_cell_area_lat(radius, lat, dlat, dlon, nx)
    total_area = jnp.sum(area)

    # Wall mask: walls at N/S always; E/W walls only for closed basin
    wall_mask = _regional_wall_mask(ny, nx, periodic_x, dtype)

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
        lat_v=_c(lat_v),
        cos_lat_v=_c(cos_lat_v),
        f=_c(f),
        dx=_c(dx),
        dy=_c(dy),
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat),
        omega=float(omega),
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
    *,
    equator_on_tpoint: bool = False,
    n_lat: int | None = None,
    metric_convention: str = "exact",
) -> LatLonGrid:
    """Mercator (isotropic) latitude-longitude grid.

    Latitudes are placed via the Mercator projection so that cell
    heights ``dy(j)`` decrease poleward in proportion to ``cos(φ)``,
    matching the zonal spacing ``dx(j) = R cos(φ) · Δλ``. This yields
    approximately isotropic cells (``dx ≈ dy`` per row), the latitudinal
    scaling of the first baroclinic deformation radius (Hallberg 2013),
    and the placement used by NEMO's DINO ocean test case (Kamm et al.
    2025, GMD).

    ``metric_convention`` (#1226) selects how the T/u-face metrics
    ``dy``/``area`` are computed from the (unchanged) cell/face
    *latitude placement*:

    * ``"exact"`` (default, BIT-IDENTICAL to every existing caller) —
      ``dy`` is the true finite difference of face latitudes and
      ``area`` the exact spherical-cap integral (see below).
    * ``"nemo_isotropic"`` — reproduces NEMO's ``usr_def_hgr.F90``
      DINO closed form verbatim: ``pe1t = pe2t = ra * rad *
      cos(rad*phi_T) * rn_e1_deg`` (usr_def_hgr.F90, DINO
      ``vopikamm/DINO@v0.2.0``, the ``ppglam``/isotropic branch) —
      i.e. NEMO sets the meridional cell height EQUAL to the zonal
      cell width at every row (``dy(j) := dx_single(j)``) instead of
      the true ``R·Δφ(j)``, and ``area := dx_single(j)²``.  This is a
      DELIBERATE closed-form approximation NEMO makes, not a more
      exact grid — legoESM's ``"exact"`` default is geometrically
      MORE correct.  Only the cell latitudes/faces (``lat``,
      ``lat_v``, and therefore ``cos_lat_v``/``f``) are shared between
      both conventions; ``"nemo_isotropic"`` does NOT touch them, so
      the #516 v-face metric (``vface_zonal_cos_lat``, derived from
      cell-center latitudes) is unaffected by this flag.

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
    metric_convention : {"exact", "nemo_isotropic"}, optional
        T/u-face metric convention (see above). Default ``"exact"``
        keeps every existing caller BIT-IDENTICAL. Raises
        ``ValueError`` on any other value.

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
    if metric_convention not in ("exact", "nemo_isotropic"):
        raise ValueError(
            f"metric_convention must be 'exact' or 'nemo_isotropic', "
            f"got {metric_convention!r}"
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

    # K = max integer face index from the equator.  ``n_lat_override`` (when
    # given) sets the count directly — used by the NEMO-faithful DINO grid,
    # whose ``jpjglo`` is a config value, not the ``lat_max`` floor.
    lat_max_rad = lat_max_deg * jnp.pi / 180.0
    if n_lat is not None:
        if n_lat < 2:
            raise ValueError(f"n_lat must be >= 2, got {n_lat}")
        if equator_on_tpoint and n_lat % 2 == 0:
            raise ValueError(
                f"equator_on_tpoint places the equator on a T-point, which "
                f"needs an ODD n_lat; got {n_lat}")
        K = n_lat // 2                       # tpoint: n_lat=2K+1 ; face: n_lat=2K
    else:
        j_max_cont = float(jnp.arctanh(jnp.sin(lat_max_rad)) / dlon)
        K = int(jnp.floor(j_max_cont))
    if K < 1:
        raise ValueError(
            f"lat_max_deg={lat_max_deg} too small for n_lon={n_lon}: "
            f"Mercator placement yields zero cells. Increase lat_max_deg "
            f"or n_lon."
        )

    # Placement in float64 regardless of the storage policy: ``arcsin(tanh(·))``
    # loses precision in float32 near the poles.  ``_c(...)`` downcasts at the end.
    if equator_on_tpoint:
        # NEMO usrdef_hgr convention: cell CENTRES (T-points) at INTEGER k, so
        # a T-point sits ON the equator (k=0); faces at half-integer k.
        # n_lat = 2K+1 (odd).  φ_T(j) = asin(tanh(Δλ·(j-K))) — matches NEMO's
        # asin(tanh(rn_e1_deg·rad·(jg-nn_jeq_s))) to roundoff.
        n_lat_out = 2 * K + 1
        k_center = jnp.arange(-K, K + 1, dtype=jnp.float64)          # (2K+1,)
        k_face = jnp.arange(-K, K + 2, dtype=jnp.float64) - 0.5      # (2K+2,)
    else:
        # Default: equator on a FACE (k=0), n_lat=2K even, centres at half-int k.
        n_lat_out = 2 * K
        k_face = jnp.arange(-K, K + 1, dtype=jnp.float64)            # (n_lat+1,)
        k_center = k_face[:-1] + 0.5                                 # (n_lat,)
    n_lat = n_lat_out

    # Mercator placement: sin(φ) = tanh(Δλ · k).
    lat_face = jnp.arcsin(jnp.tanh(dlon * k_face))
    lat = jnp.arcsin(jnp.tanh(dlon * k_center))

    # Longitude (cell centres).
    lon_w_rad = lon_west_deg * jnp.pi / 180.0
    lon = lon_w_rad + dlon * (jnp.arange(n_lon, dtype=jnp.float64) + 0.5)

    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")

    cos_lat = jnp.maximum(jnp.cos(lat), 1e-10)
    sin_lat = jnp.sin(lat)
    # ``lat_face`` is already the v-face axis (length n_lat+1).  Reuse
    # it directly rather than calling ``_compute_v_face_coords`` which
    # would extrapolate from cell centers and give a slightly
    # different placement on non-uniform Mercator.
    lat_v = lat_face
    cos_lat_v = jnp.maximum(jnp.abs(jnp.cos(lat_v)), 1e-10)

    f = 2.0 * omega * sin_lat[:, None] * jnp.ones((1, n_lon))

    # dx(j, i) = 2 · R · cos(lat_c(j)) · dlon (2-cell convention).
    dx = radius * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))

    if metric_convention == "nemo_isotropic":
        # NEMO usr_def_hgr.F90 (DINO, vopikamm/DINO@v0.2.0, the
        # isotropic-Mercator branch): pe1t = pe2t = ra * rad *
        # cos(rad*phi_T) * rn_e1_deg, i.e. the meridional cell height
        # is SET EQUAL to the zonal cell width at every T-row (a
        # deliberate closed-form approximation, not a re-derivation of
        # the true finite-difference dy — #1226). Single-cell dy(j) :=
        # single-cell dx(j); "2-cell" dy/area below follow the same
        # 2-cell-span / area convention as the "exact" branch so the
        # rest of the C-grid stack (which reads dy/area, never lat_v
        # directly for these) is unaffected in shape/units.
        dx_single = radius * dlon * cos_lat                      # (n_lat,)
        dy = 2.0 * dx_single                                     # (n_lat,) "2-cell" span
        area_lat = dx_single**2                                  # (n_lat,)
        area = area_lat[:, None] * jnp.ones((1, n_lon))
    else:
        # dy(j) = 2 · R · (lat_face[j+1] - lat_face[j])  — "2-cell distance".
        # For Mercator this varies per row (decreases poleward).
        dy_cell = radius * (lat_face[1:] - lat_face[:-1])           # (n_lat,)
        dy = 2.0 * dy_cell

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
        lat_v=_c(lat_v),
        cos_lat_v=_c(cos_lat_v),
        f=_c(f),
        dx=_c(dx),
        dy=_c(dy),
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat_repr),
        omega=float(omega),
    )


def create_stretched_latlon_grid(
    dy_deg,
    n_lon: int,
    lat_south: float,
    lon_west: float = 0.0,
    lon_east: float = 360.0,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    periodic_x: bool = False,
) -> tuple[LatLonGrid, jax.Array]:
    """Lat-lon grid with an ARBITRARY per-row meridional spacing array.

    The meridional analogue of :func:`create_regional_latlon_grid` for
    non-uniform latitude spacing (e.g. a Vinokur-stretched ``dyt`` as in
    the Veros ``global_flexible`` setup): the caller supplies the
    interior cell heights ``dy_deg`` directly instead of a uniform
    ``(lat_north - lat_south) / n_lat``.  Zonal spacing stays uniform
    (all target setups have uniform ``dxt``); operators consume the
    per-row 1-D ``grid.dy`` array, which is already variable-dy safe
    throughout the lat-lon C-grid stack (Mercator/DINO precedent).

    Construction (Mercator pattern — faces supplied directly, exact
    sin-face areas, no ``compute_v_face_coords`` extrapolation):

    - Cell FACES from a cumulative sum: the first interior cell's south
      face sits at ``lat_south``; interior faces follow from
      ``cumsum(dy_deg)``.  One wall row is added at each end (the
      ``create_regional_latlon_grid`` convention) whose height is
      edge-extended (``dy_deg[0]`` / ``dy_deg[-1]`` — the analogue of
      Veros's ghost-row ``dyt[:2] = dyt[2]``).
    - Cell CENTRES via the pyOM/Veros ``u_centered_grid`` placement:
      each face bisects its two adjacent centres
      (``face[j] = (centre[j] + centre[j+1]) / 2``), seeded with
      ``centre[0] = face[1] - dy_wall/2``.  An EXACTLY-uniform
      ``dy_deg`` short-circuits to :func:`create_regional_latlon_grid`
      with ``lat_north = lat_south + sum(dy_deg)`` — the result is then
      BIT-identical to the canonical uniform builder (the cumsum +
      recursion path agrees with it only to float64 round-off, and the
      branch is continuous at that level).  For stretched ``dy_deg``
      the interior
      centres land EXACTLY on Veros ``yt[2:-2]``; note centres are then
      *not* the midpoints of their faces.  This is also the placement
      whose faces ``create_latlon_geometry`` reconstructs exactly from
      centre midpoints (its variable-dlat branch).
    - Cell areas use the exact spherical form
      ``R² Δλ |sin(φ_face[j+1]) − sin(φ_face[j])|`` (Mercator formula —
      valid for any orthogonal spherical grid).

    Veros mapping (``calc_grid`` convention, probe-verified for the
    4deg recipe): Veros aligns ``yu[2] = y_origin``, i.e. ``y_origin``
    is the NORTH face of the first interior cell.  Therefore::

        lat_south = y_origin - dy_deg[0]

    and this grid's interior rows ``[1:-1]`` match Veros ``yt[2:-2]``
    / faces ``lat_v[1:-1]`` match Veros ``yu[1:-2]``.

    Parameters
    ----------
    dy_deg : array-like, shape (n_lat,)
        Interior cell heights [DEGREES latitude], south to north.  All
        entries must be positive and finite; the builder rejects arrays
        whose row-to-row variation is so rapid that a u-centred cell
        centre escapes its cell (smooth stretchings — Vinokur, Mercator,
        geometric ≲ 2x jumps — are fine).
    n_lon : int
        Number of interior zonal cells (uniform spacing).
    lat_south : float
        Latitude of the FIRST INTERIOR cell's south face [degrees]
        (Veros: ``y_origin - dy_deg[0]``).
    lon_west, lon_east : float
        Zonal extent [degrees], as in ``create_regional_latlon_grid``.
    radius : float
        Sphere radius [m].
    omega : float
        Rotation rate [rad/s].
    dtype : optional
        Storage dtype; defaults to the active precision policy.
        Placement math always runs in float64 (Mercator precedent).
    periodic_x : bool
        If True, periodic channel in x (no E/W wall columns).

    Returns
    -------
    grid : LatLonGrid
        ``n_lat + 2`` latitude rows (wall rows at N/S).  Longitude
        columns: ``n_lon + 2`` if closed, ``n_lon`` if periodic.
        ``grid.dy`` is the per-row 2-cell span [m]; ``grid.dlat`` is the
        smallest (most CFL-stringent) row's dlat [rad] — diagnostics
        only, as on the Mercator grid.
    wall_mask : jax.Array
        1 = ocean interior, 0 = wall.
    """
    d_int = np.asarray(dy_deg, dtype=np.float64)
    if d_int.ndim != 1 or d_int.size < 2:
        raise ValueError(
            f"dy_deg must be a 1-D array with at least 2 entries, got "
            f"shape {d_int.shape}"
        )
    if not bool(np.all(np.isfinite(d_int))) or bool(np.any(d_int <= 0.0)):
        raise ValueError(
            "dy_deg entries must all be positive and finite, got "
            f"min={np.min(d_int)!r}"
        )
    span = float(np.sum(d_int))
    if lat_south < -90.0:
        raise ValueError(f"lat_south={lat_south} must be >= -90")
    if lat_south + span > 90.0 + 1e-9:
        raise ValueError(
            f"interior span exceeds the north pole: lat_south={lat_south} "
            f"+ sum(dy_deg)={span} > 90"
        )
    if not periodic_x and lon_west >= lon_east:
        raise ValueError(f"lon_west={lon_west} must be < lon_east={lon_east}")

    # Exactly-uniform dy_deg → delegate to the canonical uniform
    # builder: BIT-identical output to create_regional_latlon_grid
    # (linspace centre placement + product-form areas), and downstream
    # consumers see the one well-trodden uniform-grid object.  The
    # stretched path below agrees with it only to float64 round-off
    # (cumsum + u-centred recursion reorder the float ops), so the
    # branch is continuous at the ~ULP level for near-uniform input.
    if bool(np.all(d_int == d_int[0])):
        return create_regional_latlon_grid(
            n_lat=int(d_int.size),
            n_lon=n_lon,
            lat_south=lat_south,
            lat_north=lat_south + span,
            lon_west=lon_west,
            lon_east=lon_east,
            radius=radius,
            omega=omega,
            dtype=dtype,
            periodic_x=periodic_x,
        )

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

    # --- Meridional placement (float64 numpy; setup-time only) ---
    n_int = d_int.size
    ny = n_int + 2
    # Wall rows edge-extend the adjacent interior height (Veros ghost
    # convention dyt[:2] = dyt[2] / dyt[-2:] = dyt[-3]).
    d_full = np.concatenate([d_int[:1], d_int, d_int[-1:]])      # (ny,)
    # Faces from cumsum; face index 1 (south face of the first interior
    # row) pinned at lat_south.
    faces_deg = (lat_south - d_int[0]) + np.concatenate(
        [np.zeros(1), np.cumsum(d_full)])                        # (ny+1,)
    # Centres: pyOM/Veros u_centered_grid recursion
    # centre[j+1] = 2*face[j+1] - centre[j], vectorised with the same
    # alternating-cumsum trick Veros uses (veros/core/numerics.py).
    yt = np.empty(ny, dtype=np.float64)
    yt[0] = faces_deg[1] - 0.5 * d_full[0]
    yt[1:] = 2.0 * faces_deg[1:-1]
    alt = np.ones(ny, dtype=np.float64)
    alt[::2] = -1.0
    centers_deg = alt * np.cumsum(alt * yt)                      # (ny,)
    if not (
        bool(np.all(centers_deg > faces_deg[:-1]))
        and bool(np.all(centers_deg < faces_deg[1:]))
    ):
        raise ValueError(
            "dy_deg varies too rapidly: a u-centred cell centre escaped "
            "its cell (faces no longer interleave centres). Use a "
            "smoother stretching (adjacent dy ratios well below 2)."
        )

    lat = jnp.asarray(np.deg2rad(centers_deg))                   # (ny,)
    lat_face = jnp.asarray(np.deg2rad(faces_deg))                # (ny+1,)

    lon, dlon, nx = _regional_lon_axis(n_lon, lon_west, lon_east, periodic_x)

    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")

    # abs-clamp like the global builder: wall rows of near-pole domains
    # may poke past ±90° (regional convention keeps them; they are land).
    cos_lat = jnp.maximum(jnp.abs(jnp.cos(lat)), 1e-10)
    sin_lat = jnp.sin(lat)
    # Faces supplied directly (Mercator pattern) — do NOT extrapolate
    # via compute_v_face_coords.
    lat_v = lat_face
    cos_lat_v = jnp.maximum(jnp.abs(jnp.cos(lat_v)), 1e-10)

    f = 2.0 * omega * sin_lat[:, None] * jnp.ones((1, nx))

    dx = radius * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, nx))
    # dy(j) = 2 · R · (lat_face[j+1] - lat_face[j]) — 2-cell convention,
    # defined from face differences for exact dy/lat_v consistency
    # (Mercator pattern).
    dy = 2.0 * radius * (lat_face[1:] - lat_face[:-1])           # (ny,)

    # Exact spherical area: R² · Δλ · |sin(φ_face[j+1]) − sin(φ_face[j])|.
    sin_face = jnp.sin(lat_face)
    area_lat = radius**2 * dlon * jnp.abs(sin_face[1:] - sin_face[:-1])
    area = area_lat[:, None] * jnp.ones((1, nx))
    total_area = jnp.sum(area)

    # Representative scalar dlat = smallest (most CFL-stringent) row —
    # the Mercator convention. Diagnostics only; operators use grid.dy.
    dlat_repr = float(np.min(np.deg2rad(d_full)))

    wall_mask = _regional_wall_mask(ny, nx, periodic_x, dtype)

    def _c(a):
        return a.astype(dtype) if hasattr(a, "astype") else a

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
        lat_v=_c(lat_v),
        cos_lat_v=_c(cos_lat_v),
        f=_c(f),
        dx=_c(dx),
        dy=_c(dy),
        area=_c(area),
        total_area=total_area,
        dlon=float(dlon),
        dlat=float(dlat_repr),
        omega=float(omega),
    )
    return grid, wall_mask


# =========================================================================
# Tripolar-ready C-grid geometry
# =========================================================================


def ensure_geometry(
    grid,
    omega: float | None = None,
    *,
    metric_convention: str = "exact",
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
    omega : float, optional
        Rotation-rate override [rad/s]; only consulted when converting
        from ``LatLonGrid``.  Default ``None`` takes the GRID's stored
        omega (falling back to ``constants.Omega`` for grid-like
        objects without the field), so a non-rotating
        ``create_latlon_grid(..., omega=0.0)`` grid stays non-rotating
        through the conversion — the previous ``constants.Omega``
        default silently re-rotated it (#521, codex 2026-07-03
        round-4 HIGH).
    metric_convention : {"exact", "nemo_isotropic"}, optional (#1226)
        Forwarded to :func:`create_latlon_geometry` when converting
        from a ``LatLonGrid`` (a no-op when *grid* is already a
        ``LatLonCGridGeometry`` — the duck-type/isinstance early
        returns above skip conversion entirely, so an already-built
        geometry's convention cannot be changed here). Default
        ``"exact"`` is BIT-IDENTICAL to every existing caller.

    Returns
    -------
    LatLonCGridGeometry
    """
    if isinstance(grid, LatLonCGridGeometry):
        return grid
    # Duck-type check: if it has dx_u, assume it's geometry-like
    if hasattr(grid, "dx_u") and hasattr(grid, "fold"):
        return grid  # type: ignore[return-value]
    if omega is None:
        omega = float(getattr(grid, "omega", constants.Omega))
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
        # Pass the grid's ACTUAL face latitudes so variable-dlat grids
        # (Mercator, stretched) get exact per-row metrics instead of
        # the centre-midpoint face reconstruction (which is exact for
        # the stretched builder's interior but not at its wall rows,
        # and only O(Δφ²)-approximate on Mercator).  Uniform-dlat
        # grids never consult the faces (scalar-dlat branch), so their
        # geometries are bit-unchanged.
        lat_face_1d=getattr(grid, "lat_v", None),
        metric_convention=metric_convention,
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

    # Planetary rotation rate [rad/s] the f_T/f_u/f_v fields were built
    # from — mirrors LatLonGrid.omega (#521) so operators that rebuild
    # f at other staggers (e.g. QG-Leith vertex absolute vorticity) see
    # the true rotation on the operational geometry path too.  APPENDED
    # at the NamedTuple end with a default; band slicers use _replace
    # and inherit it.
    omega: float = constants.Omega

    # Optional partial-periodic seam-wall profile, shape ``(n_lat,)``,
    # ``1.0`` = the periodic-seam zonal (u-) face is WALLED at that
    # latitude row, ``0.0`` = open/periodic.  Default ``None`` = fully
    # periodic in longitude (byte-identical: ``None`` is an empty pytree
    # subtree, so it adds no leaf).  Set only by the NEMO DINO bridge to
    # reproduce the faithful DINO geometry — ALL interior cells wet, but
    # the zonal seam u-face closed outside the ACC channel (NEMO's halo
    # ``tmask`` land columns).  Read by ``compute_face_masks``,
    # ``compute_face_masks_3d``, ``compute_vertex_mask`` and the
    # barotropic diffusion-mask derivation via ``getattr(grid,
    # "seam_wall_rows", None)``.  APPENDED at the NamedTuple end with a
    # default.  As a per-lat-row (n_lat,) array it is a real pytree leaf
    # only when set, and the MPI/SPMD band slicers (``slice_cgrid_geometry
    # _to_band``, ``widen_cgrid_geometry_band``) slice/widen it like the
    # other T-point cell-row fields so it stays aligned with band-local
    # ``n_lat`` (``None`` passes through unchanged).
    seam_wall_rows: jax.Array | None = None

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
    lat_face_1d: jax.Array | None = None,
    metric_convention: str = "exact",
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
    lat_face_1d : jax.Array, optional
        Cell-face latitudes in radians (length ``n_lat + 1``).  Only
        consulted on variable-dlat grids (Mercator / stretched), where
        it replaces the centre-midpoint face reconstruction with the
        grid's exact faces — the reconstruction is exact for the
        u-centred stretched placement's interior but not at its wall
        rows, and only approximate on Mercator.  Ignored on
        uniform-dlat grids (scalar-dlat branch).
    metric_convention : {"exact", "nemo_isotropic"}, optional (#1226)
        T/u-face metric convention for ``dy_T``/``dy_u``/``area_T`` on
        a variable-dlat (Mercator) grid, mirroring
        :func:`create_mercator_grid`'s parameter of the same name.
        Default ``"exact"`` (the true finite-difference ``dy_T = R·Δφ``
        and exact spherical-cap ``area_T``) is BIT-IDENTICAL to every
        existing caller. ``"nemo_isotropic"`` reproduces NEMO's
        ``usr_def_hgr.F90`` DINO closed form (``pe1t = pe2t``, see
        :func:`create_mercator_grid`'s docstring for the full citation)
        for ``dy_T``/``dy_u``/``area_T`` ONLY.  The v-face metrics
        (``dx_v``, ``dy_v``, ``cos_lat_v``) and the vertex area
        (``area_q``) are the #516 single-source v-face invariant
        (:func:`legoesm.ocean.dynamics.latlon_cgrid_operators.
        vface_zonal_cos_lat`, tested by
        ``tests/ocean/unit/test_vface_metric_consistency_mercator.py``)
        and are DELIBERATELY left on the exact finite-difference
        convention regardless of this flag — do not extend
        ``metric_convention`` to touch them without re-reading that
        test file. Raises ``ValueError`` on any other value. Ignored
        on uniform-dlat grids (scalar-dlat branch has no ``dlat_1d``
        to override).

    Returns
    -------
    LatLonCGridGeometry
    """
    if metric_convention not in ("exact", "nemo_isotropic"):
        raise ValueError(
            f"metric_convention must be 'exact' or 'nemo_isotropic', "
            f"got {metric_convention!r}"
        )
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

    if lat_face_1d is not None:
        # Exact faces available: detect variable dlat from ALL rows,
        # not the boundary-vs-middle two-point probe above — an
        # arbitrary spacing array can be uniform at those two probes
        # while stretched elsewhere, which would silently select the
        # uniform-metric branch.
        lat_face_1d = jnp.asarray(lat_face_1d)
        if lat_face_1d.shape != (n_lat + 1,):
            raise ValueError(
                f"lat_face_1d must have shape ({n_lat + 1},), got "
                f"{lat_face_1d.shape}"
            )
        _row_dlat = lat_face_1d[1:] - lat_face_1d[:-1]
        _is_variable_dlat = bool(
            float(jnp.max(jnp.abs(_row_dlat - _row_dlat[0])))
            > 1e-6 * abs(float(_row_dlat[0]))
        )

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
        if lat_face_1d is not None:
            # Exact faces supplied by the grid (Mercator analytic
            # faces / stretched-builder cumsum faces); shape already
            # validated above.
            lat_face = lat_face_1d
        else:
            # Reconstruct face latitudes from cell centers (inverse of
            # the u-centred placement: face j sits halfway between
            # center j-1 and center j; boundary faces by half-cell
            # extrapolation).  Exact for the stretched builder's
            # interior, approximate at its wall rows and on Mercator.
            lat_face_interior = 0.5 * (lat_1d[:-1] + lat_1d[1:])  # (n_lat-1,)
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
    _c(lon_1d)
    cos_lat_s = _c(cos_lat_1d)
    sin_lat_s = _c(sin_lat_1d)

    # ------- T-point metrics -------
    # Single-cell zonal width: R * dlon * cos(lat)
    dx_T = radius * dlon * cos_lat_s[:, jnp.newaxis] * jnp.ones((1, n_lon))

    if metric_convention == "nemo_isotropic":
        # NEMO usr_def_hgr.F90 (DINO): pe1t = pe2t (see
        # create_mercator_grid's docstring for the full citation). This
        # OVERRIDES the T/u-face row height only -- NOT dlat_1d itself,
        # which dy_v below still consumes unmodified, so the #516
        # v-face metric is untouched by this flag (see the parameter
        # docstring above).
        dx_single_T = radius * dlon * cos_lat_s          # (n_lat,)
        dy_row = dx_single_T                             # (n_lat,) := dx row
        dy_T = dy_row[:, jnp.newaxis] * jnp.ones((1, n_lon))
        area_T = (dy_row**2)[:, jnp.newaxis] * jnp.ones((1, n_lon))
        dy_u = dy_row[:, jnp.newaxis] * jnp.ones((1, n_lon + 1))
    else:
        # Single-cell meridional height
        if dlat_1d is not None:
            # Variable dlat (Mercator): per-row cell height
            dy_T = (_c(radius * dlat_1d))[:, jnp.newaxis] * jnp.ones((1, n_lon))
        else:
            dy_T = jnp.full((n_lat, n_lon), float(radius * dlat), dtype=dtype)
        # Cell area — use exact spherical area for variable-dlat grids
        area_T = _c(area_legacy)

        # dy_u = meridional extent of the u-face
        if dlat_1d is not None:
            dy_u = (_c(radius * dlat_1d))[:, jnp.newaxis] * jnp.ones((1, n_lon + 1))
        else:
            dy_u = jnp.full((n_lat, n_lon + 1), float(radius * dlat), dtype=dtype)

    # ------- u-point metrics (n_lat, n_lon+1) -------
    # dx_u = R * dlon * cos(lat) — same as gradient_x_cgrid uses.
    dx_u = (
        radius * dlon * cos_lat_s[:, jnp.newaxis]
        * jnp.ones((1, n_lon + 1))
    )

    if metric_convention == "nemo_isotropic":
        # total_area must track area_T's convention here (area_legacy /
        # the pre-branch `total_area` above is always the exact-convention
        # area). "exact" leaves the pre-branch `total_area` (summed from
        # area_legacy BEFORE the storage-dtype cast) untouched -- summing
        # the already-cast area_T here instead would silently change its
        # accumulation dtype/order and shift bit-identical callers by
        # float32 roundoff (caught by test_latlon_geometry.py::
        # TestLatLonGridParity::test_area_parity).
        total_area = jnp.sum(area_T)

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
        omega=float(omega),
    )


def create_beta_plane_cgrid_geometry(
    n_lat: int,
    n_lon: int,
    *,
    dx_m: float,
    dy_m: float | None = None,
    f0: float,
    beta: float,
    y_origin_m: float = 0.0,
    x_origin_m: float = 0.0,
    radius: float = constants.R_earth,
    cartesian_pseudo_lat: bool = False,
    dtype=None,
) -> LatLonCGridGeometry:
    r"""Cartesian **beta-plane** C-grid geometry (an f/beta-plane closed box).

    A first-class non-spherical option for the lat-lon C-grid ocean model: the
    horizontal metric is uniform Cartesian (constant ``dx_m`` x ``dy_m``, no
    ``cos(lat)`` convergence) and the Coriolis parameter is the beta-plane
    ``f(y) = f0 + beta * y``, evaluated **directly at each stagger point** from
    its own ``y`` — matching MITgcm's ``ini_cori.F`` (``fCori = f0 + beta*y_c`` at
    mass/u points, ``fCoriG = f0 + beta*y_g`` at the v/vorticity interfaces),
    rather than averaging cell-centre ``f`` to the faces. This is what makes a
    legoESM run able to reproduce a Cartesian beta-plane oracle (e.g. MITgcm
    ``tutorial_barotropic_gyre``) to per-step tendency tolerance.

    Coordinates: ``y_c[j] = y_origin_m + (j + 1/2) dy`` (cell centres),
    ``y_g[j] = y_origin_m + j dy`` (v-face / corner interfaces). ``f0`` is the
    Coriolis value at ``y = 0`` (NOT at the domain centre), matching MITgcm; set
    ``y_origin_m`` to the southern edge so the absolute ``y`` matches the oracle.

    Boundary scope: the barotropic solvers wall the northernmost and
    southernmost v-faces (``_zero_polar_lat_ends``), so this geometry is for a
    **meridionally CLOSED** domain (basin / re-entrant-in-x channel). A domain
    periodic in *y* is NOT supported.  Pass ``cartesian_pseudo_lat=True`` to pin
    the pseudo-``lat`` at **0** (a Cartesian tangent plane has ``cos_lat ≡ 1``):
    operators that recompute ``cos(grid.lat)`` for a metric (``divergence_cgrid``
    v-face length, flux-form advection) then stay EXACTLY equal to the uniform
    ``dx_m`` the explicit metrics promise — a
    non-zero ``y_c/radius`` pseudo-lat leaves a ``1−cos(y_c/radius)`` (~1.8% at
    ``|y_c|/radius=0.19``) grad/div metric mismatch that makes the IMPLICIT free
    surface non-conservative and (on the marginally-resolved gyre) flips the WBC
    turbulent.  The meridional position is carried by ``f = f0 + beta·y_c``, not
    the pseudo-lat.

    Parameters
    ----------
    n_lat, n_lon : int
        Cell counts (meridional, zonal).
    dx_m : float
        Zonal cell width [m].
    dy_m : float, optional
        Meridional cell height [m]; defaults to ``dx_m`` (square cells).
    f0 : float
        Coriolis parameter at ``y = 0`` [s^-1].
    beta : float
        Meridional gradient ``df/dy`` [m^-1 s^-1]. ``beta = 0`` gives an f-plane.
    y_origin_m, x_origin_m : float
        Southern / western edge coordinates [m].
    radius : float
        Nominal sphere radius [m]; does NOT enter the (explicit) Cartesian
        metrics — only seeds the legacy pseudo-``lat``/``lon`` and the
        ``dlon``/``dlat`` consistency sentinels (``radius * dlon = dx_m``).
    cartesian_pseudo_lat : bool
        **OBSOLETE for the metric since #514** — the C-grid operators now READ the
        stored uniform ``dx_v`` (== ``dx_m``) instead of recomputing
        ``cos(grid.lat_v)``, so the pseudo-lat no longer enters any v-face metric
        and ``True``/``False`` give identical dynamics.  Historically ``True``
        pinned the pseudo-``lat`` to 0 so the (then-recomputing) operators agreed
        with the ``cos_lat=1`` metric fields, which the energy-conserving IMPLICIT
        free surface relied on (the MITgcm gyre recipe; see the Boundary-scope
        note).  Retained only for the cosmetic ``grid.lat`` value and the
        non-adjointness-critical secondary readers (adaptive-Smag CFL ceiling,
        polar-filter labelling).  Default ``False`` keeps the natural
        ``lat = y_c/radius``.

    Returns
    -------
    LatLonCGridGeometry
        Consumed directly by the ocean model (``ensure_geometry`` passes it
        through). ``cos_lat = 1``, rotation angles zero, fold inactive.
    """
    if dtype is None:
        dtype = jnp.zeros(()).dtype  # float64 if x64 enabled, else float32
    dy_m = float(dx_m) if dy_m is None else float(dy_m)
    dx_m = float(dx_m)

    def _c(a):
        return a.astype(dtype) if hasattr(a, "astype") else a

    # Stagger-point y-coordinates: centres (y_c) and lat-interfaces (y_g).
    # Computed in float64 then cast (via _c) to the storage dtype: y can be
    # O(1e6 m) while dy is O(1e4 m), so f32 coordinate arithmetic would lose
    # ~2 digits in beta*y. f64-then-cast is strictly more accurate than an
    # inline f32 recompute and keeps f bit-stable across the x64 flag.
    j = jnp.arange(n_lat, dtype=jnp.float64)
    y_c = y_origin_m + (j + 0.5) * dy_m            # (n_lat,)
    y_g = y_origin_m + jnp.arange(n_lat + 1, dtype=jnp.float64) * dy_m  # (n_lat+1,)
    x_c = x_origin_m + (jnp.arange(n_lon, dtype=jnp.float64) + 0.5) * dx_m  # (n_lon,)

    # Coriolis directly from each point's y (MITgcm fCori/fCoriG convention).
    f_t_1d = f0 + beta * y_c                        # (n_lat,)
    f_v_1d = f0 + beta * y_g                        # (n_lat+1,)
    ones_t = jnp.ones((n_lat, n_lon), dtype=dtype)
    f_t = _c(f_t_1d[:, None]) * ones_t
    f_u = _c(f_t_1d[:, None]) * jnp.ones((n_lat, n_lon + 1), dtype=dtype)
    f_v = _c(f_v_1d[:, None]) * jnp.ones((n_lat + 1, n_lon), dtype=dtype)

    # Uniform Cartesian metrics (no cos(lat) convergence).
    dx_t = jnp.full((n_lat, n_lon), dx_m, dtype=dtype)
    dy_t = jnp.full((n_lat, n_lon), dy_m, dtype=dtype)
    area_t = jnp.full((n_lat, n_lon), dx_m * dy_m, dtype=dtype)
    total_area = jnp.asarray(n_lat * n_lon * dx_m * dy_m, dtype=dtype)
    dx_u = jnp.full((n_lat, n_lon + 1), dx_m, dtype=dtype)
    dy_u = jnp.full((n_lat, n_lon + 1), dy_m, dtype=dtype)
    dx_v = jnp.full((n_lat + 1, n_lon), dx_m, dtype=dtype)
    dy_v = jnp.full((n_lat + 1, n_lon), dy_m, dtype=dtype)
    area_q = jnp.full((n_lat + 1, n_lon + 1), dx_m * dy_m, dtype=dtype)

    # No grid rotation (i-axis == east), Cartesian -> cos_lat == 1.
    cos_alpha_u = jnp.ones((n_lat, n_lon + 1), dtype=dtype)
    sin_alpha_u = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
    cos_alpha_v = jnp.ones((n_lat + 1, n_lon), dtype=dtype)
    sin_alpha_v = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)

    # Pseudo-coordinates for diagnostics / polar-filter labelling.
    #
    # ``cartesian_pseudo_lat=True`` pins ``lat ≡ 0``.  Two C-grid operators
    # (``divergence_cgrid``'s v-face length ``R·cos(lat_v)·dlon`` and the flux-form
    # momentum advection's ``cos(lat_v)`` transport metric) RECOMPUTE
    # ``cos(grid.lat)`` instead of reading the ``cos_lat=1`` metric field, so a
    # non-zero ``y_c/radius`` pseudo-lat injects a ``1−cos(y_c/radius)`` (≈1.8% at
    # ``|y_c|/radius=0.19``) mismatch between the zonal GRADIENT metric
    # (``dx_u = R·dlon·cos_lat = dx_m``) and the DIVERGENCE v-face metric —
    # breaking the discrete grad/div adjointness the IMPLICIT free-surface
    # projection relies on.  On the marginally-resolved (Munk δ≈1.7-cell)
    # wind-driven gyre that ~1% non-conservative leak flips the western-boundary
    # current from laminar to a turbulent attractor (MITgcm tutorial_barotropic_gyre
    # stays at |u|max≈0.031; with a non-zero pseudo-lat legoESM overshoots to
    # 0.066+).  A Cartesian tangent plane has cos_lat≡1, so lat≡0 is the
    # self-consistent value; the meridional position lives in ``f = f0 + beta·y_c``.
    #
    # DEFAULT False keeps the legacy ``lat = y_c/radius`` for backward
    # compatibility.  (A prior latent issue — the explicit-substep barotropic
    # solver / any vorticity-based operator blew up on a consistent-metric
    # beta-plane because ``curl_vertex_cgrid`` recomputed the vertex area as
    # ``R²·dlon·|Δsin(lat)|``, which collapses to 0 when ``lat≡0`` — was ROOT-CAUSED
    # and FIXED by having the curl read the grid's stored ``area_q``; ``lat=0`` is
    # now safe for every solver.)  Opt in (the MITgcm gyre + front_relax recipes
    # do) for the metric-consistent implicit free-surface fidelity path.
    lat_1d = jnp.zeros_like(y_c) if cartesian_pseudo_lat else (y_c / radius)
    lon_1d = x_c / radius
    lat_t = _c(lat_1d[:, None]) * jnp.ones((n_lat, n_lon), dtype=dtype)
    lon_t = _c(lon_1d[None, :]) * jnp.ones((n_lat, n_lon), dtype=dtype)

    return LatLonCGridGeometry(
        n_lat=n_lat,
        n_lon=n_lon,
        radius=float(radius),
        lat_T=lat_t,
        lon_T=lon_t,
        dx_T=dx_t,
        dy_T=dy_t,
        area_T=area_t,
        total_area=total_area,
        dx_u=dx_u,
        dy_u=dy_u,
        dx_v=dx_v,
        dy_v=dy_v,
        area_q=area_q,
        f_T=f_t,
        f_u=f_u,
        f_v=f_v,
        cos_alpha_u=cos_alpha_u,
        sin_alpha_u=sin_alpha_u,
        cos_alpha_v=cos_alpha_v,
        sin_alpha_v=sin_alpha_v,
        fold=_inactive_fold(n_lon),
        cos_lat=jnp.ones((n_lat,), dtype=dtype),
        sin_lat=jnp.zeros((n_lat,), dtype=dtype),
        lat=_c(lat_1d),
        lon=_c(lon_1d),
        dlon=float(dx_m / radius),
        dlat=float(dy_m / radius),
    )
