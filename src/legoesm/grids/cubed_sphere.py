"""Cubed-sphere grid for legoESM.

Implements a gnomonic equidistant cubed-sphere grid with 6 faces.
Each face is an N x N grid of cells. The grid uses an A-grid (collocated)
staggering for the shallow-water milestone, with all variables at cell centers.

The cubed-sphere maps 6 faces of a cube onto the sphere via gnomonic
(central) projection. This gives quasi-uniform resolution with no polar
singularity, and regular 2D arrays on each face — ideal for JAX.

References
----------
- Ronchi, Iacono, Paolucci (1996): The "Cubed Sphere"
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Nair, Thomas, Loft (2005): A Discontinuous Galerkin Transport Scheme on the Cubed Sphere
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.halo import (
    pad_halo,
    compute_padded_angle,
    compute_padded_half_metrics,
    compute_halo_interp_offsets,
    compute_halo_interp_offsets_h2,
    compute_halo_interp_offsets_h3,
)


class CubedSphereGrid(NamedTuple):
    """Cubed-sphere grid data structure.

    All arrays have shape (6, n, n) where 6 = number of faces,
    except padded arrays which are (6, n+2, n+2).
    Registered as a JAX pytree via NamedTuple.

    Attributes
    ----------
    n : int
        Number of cells per face edge. Total cells = 6 * n * n.
    radius : float
        Sphere radius [m].
    lon : jax.Array
        Longitude at cell centers [rad], shape (6, n, n).
    lat : jax.Array
        Latitude at cell centers [rad], shape (6, n, n).
    area : jax.Array
        Cell areas [m^2], shape (6, n, n).
    dx : jax.Array
        Cell width in x-direction [m], shape (6, n, n).
    dy : jax.Array
        Cell width in y-direction [m], shape (6, n, n).
    f : jax.Array
        Coriolis parameter at cell centers [1/s], shape (6, n, n).
    cos_lat : jax.Array
        Cosine of latitude, shape (6, n, n).
    sin_lat : jax.Array
        Sine of latitude, shape (6, n, n).
    angle : jax.Array
        Grid rotation angle relative to east [rad], shape (6, n, n).
    x_cart : jax.Array
        Cartesian x-coordinate on unit sphere, shape (6, n, n).
    y_cart : jax.Array
        Cartesian y-coordinate on unit sphere, shape (6, n, n).
    z_cart : jax.Array
        Cartesian z-coordinate on unit sphere, shape (6, n, n).
    angle_padded : jax.Array
        Grid rotation angle on extended grid [rad], shape (6, n+2, n+2).
        Used by vector halo exchange to correctly rotate velocity
        components at face boundaries.
    cos_angle : jax.Array
        Cosine of grid angle, shape (6, n, n). Precomputed for
        vector halo exchange.
    sin_angle : jax.Array
        Sine of grid angle, shape (6, n, n). Precomputed for
        vector halo exchange.
    cos_angle_padded : jax.Array
        Cosine of padded grid angle, shape (6, n+2, n+2).
    sin_angle_padded : jax.Array
        Sine of padded grid angle, shape (6, n+2, n+2).
    hx_ext : jax.Array
        Half dx extrapolated to halo, shape (6, n+2, n+2).
        Used in divergence computation.
    hy_ext : jax.Array
        Half dy extrapolated to halo, shape (6, n+2, n+2).
        Used in divergence computation.
    cos_angle_padded_h2 : jax.Array
        Cosine of padded grid angle for halo=2, shape (6, n+4, n+4).
    sin_angle_padded_h2 : jax.Array
        Sine of padded grid angle for halo=2, shape (6, n+4, n+4).
    hx_ext_h2 : jax.Array
        Half dx on halo=2 extended grid, shape (6, n+4, n+4).
    hy_ext_h2 : jax.Array
        Half dy on halo=2 extended grid, shape (6, n+4, n+4).
    halo_interp_offsets_h2 : jax.Array
        Interpolation offsets for halo=2 exchange, shape (6, 4, 2, n).
    halo_interp_offsets_h3 : jax.Array
        Interpolation offsets for halo=3 exchange, shape (6, 4, 3, n).
        Iter-532: precomputed for the iter-496..501 ng=3 halo
        extension that supports the FB-chain stability work
        (review-doc item #2).
    cos_angle_padded_h3 : jax.Array
        Cosine of padded grid angle for halo=3, shape (6, n+6, n+6).
        Iter-595: added for the ng=3 vector halo round-trip.
    sin_angle_padded_h3 : jax.Array
        Sine of padded grid angle for halo=3, shape (6, n+6, n+6).
    hx_ext_h3 : jax.Array
        Half dx on halo=3 extended grid, shape (6, n+6, n+6).
    hy_ext_h3 : jax.Array
        Half dy on halo=3 extended grid, shape (6, n+6, n+6).
    duogrid : DuoGridData or None
        Duo-Grid kinked-to-extended remapping data. When not None,
        pad_halo applies the Duo-Grid remap instead of interp_offsets.
    """
    n: int
    radius: float
    lon: jax.Array
    lat: jax.Array
    area: jax.Array
    dx: jax.Array
    dy: jax.Array
    f: jax.Array
    cos_lat: jax.Array
    sin_lat: jax.Array
    angle: jax.Array
    x_cart: jax.Array
    y_cart: jax.Array
    z_cart: jax.Array
    angle_padded: jax.Array
    cos_angle: jax.Array
    sin_angle: jax.Array
    cos_angle_padded: jax.Array
    sin_angle_padded: jax.Array
    hx_ext: jax.Array
    hy_ext: jax.Array
    halo_interp_offsets: jax.Array
    cos_angle_padded_h2: jax.Array
    sin_angle_padded_h2: jax.Array
    hx_ext_h2: jax.Array
    hy_ext_h2: jax.Array
    halo_interp_offsets_h2: jax.Array
    halo_interp_offsets_h3: jax.Array
    cos_angle_padded_h3: jax.Array
    sin_angle_padded_h3: jax.Array
    hx_ext_h3: jax.Array
    hy_ext_h3: jax.Array
    duogrid: object  # DuoGridData | None — use object to avoid circular import

    @property
    def n_cells(self) -> int:
        return 6 * self.n * self.n

    @property
    def total_area(self) -> jax.Array:
        return jnp.sum(self.area)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (6, self.n, self.n)

    @property
    def resolution_km(self) -> float:
        """Approximate resolution in kilometers.

        Each cube face spans π/2 radians, so the nominal grid spacing
        is (π/2)*R/n.
        """
        return (jnp.pi / 2) * self.radius / (self.n * 1000.0)

    @property
    def bounded_domain(self) -> bool:
        """Iter-865b: Fortran-faithful ``bounded_domain`` flag per
        ``fv_arrays.F90:1512``: ``bounded_domain = (regional .or.
        nested .or. duogrid)``.  In legoESM:
        - duogrid: ``self.duogrid is not None``.
        - regional / nested: a single-face panel (``self.lat.shape[0]
          == 1``; see ``create_cubed_sphere_panel`` and the
          ``data.shape[0] == 1`` branch of ``pad_halo`` which applies
          Neumann wall BCs instead of inter-face halo exchange).

        Operators with legacy edge-handling fallbacks should bypass
        them when ``bounded_domain`` is True so the duogrid /
        regional Fortran-faithful path is used uniformly across the
        codebase.  Iter-865 originally hardcoded the gate to
        ``self.duogrid is None``; iter-865b exposes the proper
        bounded-domain abstraction so future regional/nested support
        gates the same way.
        """
        return (self.duogrid is not None) or (self.lat.shape[0] == 1)

    # ------------------------------------------------------------------
    # GridProtocol properties
    # ------------------------------------------------------------------

    @property
    def grid_lat(self) -> jax.Array:
        return self.lat

    @property
    def grid_lon(self) -> jax.Array:
        return self.lon

    @property
    def grid_area(self) -> jax.Array:
        return self.area

    @property
    def grid_total_area(self):
        return jnp.sum(self.area)

    @property
    def grid_coriolis(self) -> jax.Array:
        return self.f

    @property
    def grid_radius(self) -> float:
        return self.radius

    @property
    def grid_n_columns(self) -> int:
        return 6 * self.n * self.n

    @property
    def grid_shape_2d(self) -> tuple[int, ...]:
        return (6, self.n, self.n)

    def to_columns(self, field):
        extra = field.shape[3:]
        return field.reshape(6 * self.n * self.n, *extra)

    def from_columns(self, cols):
        extra = cols.shape[1:]
        return cols.reshape(6, self.n, self.n, *extra)


def create_cubed_sphere(
    n: int,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    use_duogrid: bool = False,
    k2e_nord: int = 2,
    duogrid_ng: int | None = None,
    stretch_fac: float = 1.0,
    target_lon: float = 0.0,
    target_lat: float = -0.5 * 3.141592653589793,  # -π/2 = no rotation
    do_cube_transform: bool = False,
    shift_fac: float = 0.0,
) -> CubedSphereGrid:
    """Create a cubed-sphere grid.

    Parameters
    ----------
    n : int
        Number of cells per face edge. Common values:
        C48 (~200km), C96 (~100km), C192 (~50km), C384 (~25km).
    radius : float
        Sphere radius in meters. Default: Earth radius.
    omega : float
        Planetary rotation rate [rad/s]. Default: Earth rotation rate.
        Scale for small-Earth experiments.

    Returns
    -------
    CubedSphereGrid
        The grid with all metric terms computed.
    """
    # Compute gnomonic coordinates on each face
    lon, lat = _compute_gnomonic_lonlat(n)

    # FV3_3D iter 586/589: optional Schmidt stretching.
    apply_schmidt = (
        abs(stretch_fac - 1.0) > 1e-5
        or target_lat > -0.5 * jnp.pi + 1e-5
    )
    if apply_schmidt:
        if do_cube_transform:
            # FV3 cube_transform (fv_grid_utils.F90:920-980)
            lon, lat = cube_transform(
                lon, lat, stretch_fac, target_lon, target_lat,
            )
        else:
            # FV3 direct_transform / do_schmidt (fv_grid_utils.F90:870-917)
            lon, lat = schmidt_transform(
                lon, lat, stretch_fac, target_lon, target_lat,
            )

    # FV3_3D iter 591: shift_fac longitude shift (FV3 fv_grid_tools.F90:662-663).
    # Only applied when NOT using Schmidt/cube_transform (gated in FV3).
    # FV3 default shift_fac=18 → west-shift by π/18 = 10° (away from Japan).
    if shift_fac > 1e-4 and not apply_schmidt:
        lon = lon - jnp.pi / shift_fac
        lon = jnp.where(lon < 0.0, lon + 2.0 * jnp.pi, lon)

    # Cartesian coordinates on unit sphere
    cos_lat = jnp.cos(lat)
    sin_lat = jnp.sin(lat)
    cos_lon = jnp.cos(lon)
    sin_lon = jnp.sin(lon)

    x_cart = cos_lat * cos_lon
    y_cart = cos_lat * sin_lon
    z_cart = sin_lat

    # Coriolis parameter
    f = 2.0 * omega * sin_lat

    # Compute padded quantities from gnomonic extension (smooth across
    # the interior-halo boundary).  All metrics and angles derive from a
    # single source — the extended gnomonic grid — so there is no seam
    # discontinuity at cube-face edges.  The O(Δα⁴) accuracy difference
    # vs pad_halo-based interior values is well below the O(Δα²)
    # truncation error of the 2nd-order stencils.
    angle_padded = compute_padded_angle(n)
    hx_ext, hy_ext = compute_padded_half_metrics(n, radius)

    # Extract interior from padded arrays (no override — single source)
    angle = angle_padded[:, 1:-1, 1:-1]
    dx = 2.0 * hx_ext[:, 1:-1, 1:-1]
    dy = 2.0 * hy_ext[:, 1:-1, 1:-1]
    area = _compute_exact_cell_areas(n, radius)

    # Precompute trig of grid angle for vector halo exchange
    cos_angle_val = jnp.cos(angle)
    sin_angle_val = jnp.sin(angle)
    cos_angle_padded_val = jnp.cos(angle_padded)
    sin_angle_padded_val = jnp.sin(angle_padded)

    # Halo interpolation offsets for corrected cross-face exchange
    halo_offsets = compute_halo_interp_offsets(n)

    # halo=2 quantities for higher-order reconstruction (PPM, WENO5)
    angle_padded_h2 = compute_padded_angle(n, halo=2)
    hx_ext_h2, hy_ext_h2 = compute_padded_half_metrics(n, radius, halo=2)
    cos_angle_padded_h2_val = jnp.cos(angle_padded_h2)
    sin_angle_padded_h2_val = jnp.sin(angle_padded_h2)
    halo_offsets_h2 = compute_halo_interp_offsets_h2(n)

    # halo=3 quantities for the iter-496..501 ng=3 halo extension
    # (FB-chain stability prerequisite, review-doc item #2).
    halo_offsets_h3 = compute_halo_interp_offsets_h3(n)
    # Iter-595: add grid-angle + half-metrics at halo=3 so the vector
    # halo round-trip has the padded-angle reference needed to enable
    # `pad_halo_vector(halo=3)` on the non-MPI backend.
    angle_padded_h3 = compute_padded_angle(n, halo=3)
    hx_ext_h3, hy_ext_h3 = compute_padded_half_metrics(n, radius, halo=3)
    cos_angle_padded_h3_val = jnp.cos(angle_padded_h3)
    sin_angle_padded_h3_val = jnp.sin(angle_padded_h3)

    # Optional: Duo-Grid kinked-to-extended remapping data.
    # Duo-Grid requires ng >= 2 so both halo depths used by the FV3
    # d2a2c_vect and PPM transport paths are remapped. Since ng <= n//2,
    # this means n >= 4. For n < 4, silently skip — these toy grids are
    # too small for face-boundary artifacts to be meaningful.
    duogrid = None
    if use_duogrid and n >= 4:
        from legoesm.grids.duogrid import create_duogrid_data
        # Default ng: min(3, n//2) — FV3 uses 3 at production resolutions,
        # but small test grids need a smaller halo to fit the stencil.
        ng = duogrid_ng if duogrid_ng is not None else min(3, n // 2)
        duogrid = create_duogrid_data(n, radius=radius, ng=ng,
                                       k2e_nord=k2e_nord)

    # Grid arrays use the storage dtype from the precision policy.
    # Defaults to float32 for backward compatibility.
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            _dt = get_policy().storage
        except Exception:
            _dt = jnp.float32
    else:
        _dt = dtype
    grid = CubedSphereGrid(
        n=n,
        radius=radius,
        lon=lon.astype(_dt),
        lat=lat.astype(_dt),
        area=area.astype(_dt),
        dx=dx.astype(_dt),
        dy=dy.astype(_dt),
        f=f.astype(_dt),
        cos_lat=cos_lat.astype(_dt),
        sin_lat=sin_lat.astype(_dt),
        angle=angle.astype(_dt),
        x_cart=x_cart.astype(_dt),
        y_cart=y_cart.astype(_dt),
        z_cart=z_cart.astype(_dt),
        angle_padded=angle_padded.astype(_dt),
        cos_angle=cos_angle_val.astype(_dt),
        sin_angle=sin_angle_val.astype(_dt),
        cos_angle_padded=cos_angle_padded_val.astype(_dt),
        sin_angle_padded=sin_angle_padded_val.astype(_dt),
        hx_ext=hx_ext.astype(_dt),
        hy_ext=hy_ext.astype(_dt),
        halo_interp_offsets=halo_offsets.astype(_dt),
        cos_angle_padded_h2=cos_angle_padded_h2_val.astype(_dt),
        sin_angle_padded_h2=sin_angle_padded_h2_val.astype(_dt),
        hx_ext_h2=hx_ext_h2.astype(_dt),
        hy_ext_h2=hy_ext_h2.astype(_dt),
        halo_interp_offsets_h2=halo_offsets_h2.astype(_dt),
        halo_interp_offsets_h3=halo_offsets_h3.astype(_dt),
        cos_angle_padded_h3=cos_angle_padded_h3_val.astype(_dt),
        sin_angle_padded_h3=sin_angle_padded_h3_val.astype(_dt),
        hx_ext_h3=hx_ext_h3.astype(_dt),
        hy_ext_h3=hy_ext_h3.astype(_dt),
        duogrid=duogrid,
    )

    # Eagerly populate the vectorized halo index cache so that the
    # first call to pad_halo (which may happen inside jax.lax.scan)
    # does not trigger a cache write during JAX tracing.
    from legoesm.grids.halo import precompute_halo_tables
    precompute_halo_tables(n)

    return grid


def schmidt_transform(
    lon: jax.Array,
    lat: jax.Array,
    stretch_fac: float,
    target_lon: float,
    target_lat: float,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 586: Schmidt transformation for stretched cubed-sphere.

    Faithful port of FV3 ``direct_transform`` (fv_grid_utils.F90:870-917).
    Applies a conformal stretching that locally enhances resolution at
    ``(target_lon, target_lat)`` by factor ``stretch_fac``.

    Algorithm:
    1. Latitude stretching:
           lat_t = asin( (c²-1 + (c²+1)·sin_lat) / (c²+1 + (c²-1)·sin_lat) )
       where c = stretch_fac.  c > 1 → stretching (high-res near target).
    2. Pole rotation: rotate the stretched-pole frame so the new pole
       lies at (target_lon, target_lat).

    Parameters
    ----------
    lon, lat : jax.Array
        Input gnomonic coordinates (any shape).  Lat in [-π/2, π/2],
        lon in [0, 2π].
    stretch_fac : float
        Stretching factor c.  1.0 = no stretch.  Typical 2-5 for regional
        focus.  When |c-1| < 1e-5 stretching is skipped (only rotation).
    target_lon, target_lat : float
        Center of high-res face in radians.  When target_lat = -π/2
        (equivalent of FV3 default -90°), no rotation applied.

    Returns
    -------
    lon_new, lat_new : jax.Array
        Transformed coordinates.

    Notes
    -----
    Faithful to FV3.  Adds stretched-grid support to ``create_cubed_sphere``
    via the ``stretch_fac``/``target_*`` kwargs.  Closes user audit item
    #1 (stretched grid) partial — nested grids (2-way refinement) deferred.
    """
    c = stretch_fac
    c2p1 = 1.0 + c * c
    c2m1 = 1.0 - c * c

    # Step 1: latitude stretching.
    sin_lat = jnp.sin(lat)
    # When |c²-1| < 1e-7, no stretching → lat_t = lat.
    do_stretch = abs(c2m1) > 1e-7
    if do_stretch:
        lat_t = jnp.arcsin(
            (c2m1 + c2p1 * sin_lat) / (c2p1 + c2m1 * sin_lat)
        )
    else:
        lat_t = lat

    # Step 2: pole rotation to (target_lon, target_lat).
    sin_p = jnp.sin(target_lat)
    cos_p = jnp.cos(target_lat)
    sin_lat_t = jnp.sin(lat_t)
    cos_lat_t = jnp.cos(lat_t)
    cos_lon_old = jnp.cos(lon)
    sin_lon_old = jnp.sin(lon)

    sin_o = -(sin_p * sin_lat_t + cos_p * cos_lat_t * cos_lon_old)
    sin_o = jnp.clip(sin_o, -1.0, 1.0)  # numerical safety for asin

    is_pole = (1.0 - jnp.abs(sin_o)) < 1e-7
    p2 = 0.5 * jnp.pi
    two_pi = 2.0 * jnp.pi

    # Non-pole branch
    lat_rot = jnp.arcsin(sin_o)
    lon_rot = target_lon + jnp.arctan2(
        -cos_lat_t * sin_lon_old,
        -sin_lat_t * cos_p + cos_lat_t * sin_p * cos_lon_old,
    )
    lon_rot = jnp.where(lon_rot < 0.0, lon_rot + two_pi, lon_rot)
    lon_rot = jnp.where(lon_rot >= two_pi, lon_rot - two_pi, lon_rot)

    # Pole branch
    lat_pole = jnp.sign(sin_o) * p2
    lon_pole = jnp.zeros_like(lon_rot)

    lon_new = jnp.where(is_pole, lon_pole, lon_rot)
    lat_new = jnp.where(is_pole, lat_pole, lat_rot)
    return lon_new, lat_new


def cube_transform(
    lon: jax.Array,
    lat: jax.Array,
    stretch_fac: float,
    target_lon: float,
    target_lat: float,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 589: cube_transform (revised Schmidt at north pole).

    Faithful port of FV3 ``cube_transform`` (fv_grid_utils.F90:920-980).
    Same algorithm as ``schmidt_transform`` (iter 586) but with a
    ``lon += π`` shift before the pole rotation to get the final
    orientation correct.  Selected by FV3 namelist via
    ``do_cube_transform=.true.`` (alternative to ``do_schmidt``).

    Algorithm:
    1. Latitude stretching (identical to direct_transform):
       ``lat_t = asin((c²-1 + (c²+1)·sin_lat) / (c²+1 + (c²-1)·sin_lat))``
    2. **Add π to lon** (the only difference from direct_transform).
    3. Pole rotation to (target_lon, target_lat).

    Parameters
    ----------
    lon, lat : jax.Array
        Input gnomonic coordinates.  Lat ∈ [-π/2, π/2], lon ∈ [0, 2π].
    stretch_fac : float
        Stretching factor c.  1.0 = no stretch.
    target_lon, target_lat : float
        Center of high-res face in radians.

    Returns
    -------
    lon_new, lat_new : jax.Array
        Transformed coordinates.

    See Also
    --------
    schmidt_transform : iter 586, ``do_schmidt`` variant (no π shift).
    """
    c = stretch_fac
    c2p1 = 1.0 + c * c
    c2m1 = 1.0 - c * c

    sin_lat = jnp.sin(lat)
    do_stretch = abs(c2m1) > 1e-7
    if do_stretch:
        lat_t = jnp.arcsin(
            (c2m1 + c2p1 * sin_lat) / (c2p1 + c2m1 * sin_lat)
        )
    else:
        lat_t = lat

    sin_p = jnp.sin(target_lat)
    cos_p = jnp.cos(target_lat)
    sin_lat_t = jnp.sin(lat_t)
    cos_lat_t = jnp.cos(lat_t)

    # iter-589: the only difference from schmidt_transform — lon += π
    lon_pi = lon + jnp.pi
    cos_lon_pi = jnp.cos(lon_pi)
    sin_lon_pi = jnp.sin(lon_pi)

    sin_o = -(sin_p * sin_lat_t + cos_p * cos_lat_t * cos_lon_pi)
    sin_o = jnp.clip(sin_o, -1.0, 1.0)

    is_pole = (1.0 - jnp.abs(sin_o)) < 1e-7
    p2 = 0.5 * jnp.pi
    two_pi = 2.0 * jnp.pi

    lat_rot = jnp.arcsin(sin_o)
    lon_rot = target_lon + jnp.arctan2(
        -cos_lat_t * sin_lon_pi,
        -sin_lat_t * cos_p + cos_lat_t * sin_p * cos_lon_pi,
    )
    lon_rot = jnp.where(lon_rot < 0.0, lon_rot + two_pi, lon_rot)
    lon_rot = jnp.where(lon_rot >= two_pi, lon_rot - two_pi, lon_rot)

    lat_pole = jnp.sign(sin_o) * p2
    lon_pole = jnp.zeros_like(lon_rot)

    lon_new = jnp.where(is_pole, lon_pole, lon_rot)
    lat_new = jnp.where(is_pole, lat_pole, lat_rot)
    return lon_new, lat_new


def _compute_gnomonic_lonlat(n: int) -> tuple[jax.Array, jax.Array]:
    """Compute longitude and latitude on the gnomonic cubed-sphere.

    Uses the equidistant gnomonic projection. Each face of the cube
    is mapped to the sphere via central projection.

    Parameters
    ----------
    n : int
        Number of cells per face edge.

    Returns
    -------
    lon, lat : arrays of shape (6, n, n) in radians.
    """
    # Local coordinates on each face: [-pi/4, pi/4]
    # Cell centers at uniform spacing
    alpha = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n, endpoint=False)
    alpha = alpha + (jnp.pi / 4) / n  # Shift to cell centers
    alpha_x, alpha_y = jnp.meshgrid(alpha, alpha, indexing='ij')

    # Gnomonic projection: (alpha_x, alpha_y) -> (x, y, z) on unit sphere
    # For each face, we define local Cartesian coordinates, then
    # project onto the sphere and convert to lon/lat.

    all_lon = []
    all_lat = []

    for face in range(6):
        x, y, z = _face_to_cartesian(face, alpha_x, alpha_y)
        r = jnp.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / r, y / r, z / r

        face_lon = jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi)  # [0, 2π)
        face_lat = jnp.arcsin(jnp.clip(z, -1.0, 1.0))

        all_lon.append(face_lon)
        all_lat.append(face_lat)

    lon = jnp.stack(all_lon, axis=0)  # (6, n, n)
    lat = jnp.stack(all_lat, axis=0)  # (6, n, n)

    return lon, lat


def _face_to_cartesian(
    face: int, alpha_x: jax.Array, alpha_y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map local gnomonic coordinates to 3D Cartesian coordinates.

    Face numbering:
        0: +x (front)   - equatorial, centered at 0 lon
        1: +y (right)    - equatorial, centered at 90E
        2: -x (back)     - equatorial, centered at 180E
        3: -y (left)     - equatorial, centered at 90W
        4: +z (top)      - north pole
        5: -z (bottom)   - south pole
    """
    tan_x = jnp.tan(alpha_x)
    tan_y = jnp.tan(alpha_y)

    if face == 0:    # +x face
        x = jnp.ones_like(tan_x)
        y = tan_x
        z = tan_y
    elif face == 1:  # +y face
        x = -tan_x
        y = jnp.ones_like(tan_x)
        z = tan_y
    elif face == 2:  # -x face
        x = -jnp.ones_like(tan_x)
        y = -tan_x
        z = tan_y
    elif face == 3:  # -y face
        x = tan_x
        y = -jnp.ones_like(tan_x)
        z = tan_y
    elif face == 4:  # +z face (north pole)
        x = -tan_y
        y = tan_x
        z = jnp.ones_like(tan_x)
    elif face == 5:  # -z face (south pole)
        x = tan_y
        y = tan_x
        z = -jnp.ones_like(tan_x)
    else:
        raise ValueError(f"Invalid face index: {face}")

    return x, y, z


def _compute_exact_cell_areas(n: int, radius: float) -> jax.Array:
    """Compute exact spherical cell areas using l'Huilier's theorem.

    Each cell is a spherical quadrilateral defined by its 4 corners on
    the gnomonic grid.  We split each quad into 2 spherical triangles
    and sum their spherical excess (= area on unit sphere).

    Parameters
    ----------
    n : int
        Number of cells per face edge.
    radius : float
        Sphere radius [m].

    Returns
    -------
    area : jax.Array, shape (6, n, n)
        Exact cell areas [m^2].
    """
    # Cell corners: n+1 points along each edge
    alpha_edges = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n + 1)
    ax, ay = jnp.meshgrid(alpha_edges, alpha_edges, indexing='ij')

    all_areas = []
    for face in range(6):
        # Corner Cartesian coordinates on unit sphere
        x, y, z = _face_to_cartesian(face, ax, ay)
        r = jnp.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / r, y / r, z / r  # (n+1, n+1)

        # For each cell (i,j), corners at (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        # SW, SE, NE, NW
        sw_x, sw_y, sw_z = x[:-1, :-1], y[:-1, :-1], z[:-1, :-1]
        se_x, se_y, se_z = x[1:, :-1], y[1:, :-1], z[1:, :-1]
        ne_x, ne_y, ne_z = x[1:, 1:], y[1:, 1:], z[1:, 1:]
        nw_x, nw_y, nw_z = x[:-1, 1:], y[:-1, 1:], z[:-1, 1:]

        def _triangle_excess(x1, y1, z1, x2, y2, z2, x3, y3, z3):
            """Spherical excess of triangle on unit sphere via l'Huilier."""
            # Arc lengths between vertices
            dot12 = jnp.clip(x1*x2 + y1*y2 + z1*z2, -1.0, 1.0)
            dot23 = jnp.clip(x2*x3 + y2*y3 + z2*z3, -1.0, 1.0)
            dot31 = jnp.clip(x3*x1 + y3*y1 + z3*z1, -1.0, 1.0)
            a = jnp.arccos(dot12)
            b = jnp.arccos(dot23)
            c = jnp.arccos(dot31)
            s = 0.5 * (a + b + c)
            # l'Huilier's theorem
            tan_E4_sq = jnp.clip(
                jnp.tan(s / 2) * jnp.tan((s - a) / 2)
                * jnp.tan((s - b) / 2) * jnp.tan((s - c) / 2),
                0.0, None,
            )
            return 4.0 * jnp.arctan(jnp.sqrt(tan_E4_sq))

        # Split quad into 2 triangles: (SW,SE,NE) + (SW,NE,NW)
        e1 = _triangle_excess(sw_x, sw_y, sw_z, se_x, se_y, se_z,
                              ne_x, ne_y, ne_z)
        e2 = _triangle_excess(sw_x, sw_y, sw_z, ne_x, ne_y, ne_z,
                              nw_x, nw_y, nw_z)
        all_areas.append(radius**2 * (e1 + e2))

    return jnp.stack(all_areas, axis=0)



def _compute_grid_spacing(
    lon: jax.Array, lat: jax.Array, n: int, radius: float
) -> tuple[jax.Array, jax.Array]:
    """Compute grid spacing dx, dy as great-circle distances.

    Uses central differences of cell center positions with proper
    inter-face halo exchange (no jnp.roll).
    """
    # Convert to Cartesian for accurate distance computation
    cos_lat = jnp.cos(lat)
    x = cos_lat * jnp.cos(lon)
    y = cos_lat * jnp.sin(lon)
    z = jnp.sin(lat)

    # Pad with halo data from neighboring faces
    x_pad = pad_halo(x)  # (6, n+2, n+2)
    y_pad = pad_halo(y)
    z_pad = pad_halo(z)

    # dx: distance between (i+1,j) and (i-1,j)
    # In padded coords: axis=1 shift +1 = [:, 2:, 1:-1], shift -1 = [:, :-2, 1:-1]
    dx_vec = jnp.sqrt(
        (x_pad[:, 2:, 1:-1] - x_pad[:, :-2, 1:-1])**2 +
        (y_pad[:, 2:, 1:-1] - y_pad[:, :-2, 1:-1])**2 +
        (z_pad[:, 2:, 1:-1] - z_pad[:, :-2, 1:-1])**2
    )
    # Chord to arc length: 2*R*arcsin(chord/(2*R))
    # For unit sphere, chord = dx_vec, arc = 2*arcsin(chord/2)
    dx = radius * 2.0 * jnp.arcsin(jnp.clip(dx_vec / 2.0, 0.0, 1.0))

    # dy: distance between (i,j+1) and (i,j-1)
    dy_vec = jnp.sqrt(
        (x_pad[:, 1:-1, 2:] - x_pad[:, 1:-1, :-2])**2 +
        (y_pad[:, 1:-1, 2:] - y_pad[:, 1:-1, :-2])**2 +
        (z_pad[:, 1:-1, 2:] - z_pad[:, 1:-1, :-2])**2
    )
    dy = radius * 2.0 * jnp.arcsin(jnp.clip(dy_vec / 2.0, 0.0, 1.0))

    return dx, dy


def _compute_grid_angle(lon: jax.Array, lat: jax.Array, n: int) -> jax.Array:
    """Compute the angle between the grid x-axis and geographic east.

    This is needed to rotate wind vectors between geographic (u_east, v_north)
    and grid-aligned (u_grid, v_grid) coordinates.

    Uses proper inter-face halo exchange instead of jnp.roll.
    """
    # Pad lon and lat with neighbor data
    lon_pad = pad_halo(lon)  # (6, n+2, n+2)
    lat_pad = pad_halo(lat)

    # Centered difference of lon/lat along x-axis (axis=1)
    dlon_dx = lon_pad[:, 2:, 1:-1] - lon_pad[:, :-2, 1:-1]
    dlat_dx = lat_pad[:, 2:, 1:-1] - lat_pad[:, :-2, 1:-1]

    # Handle longitude wrapping
    dlon_dx = jnp.where(dlon_dx > jnp.pi, dlon_dx - 2 * jnp.pi, dlon_dx)
    dlon_dx = jnp.where(dlon_dx < -jnp.pi, dlon_dx + 2 * jnp.pi, dlon_dx)

    cos_lat = jnp.cos(lat)
    # Angle of grid x-axis relative to east
    angle = jnp.arctan2(dlat_dx, dlon_dx * cos_lat)

    return angle


def lonlat_to_cartesian(
    lon: jax.Array, lat: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert lon/lat (radians) to unit sphere Cartesian coordinates."""
    cos_lat = jnp.cos(lat)
    return cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), jnp.sin(lat)


def great_circle_distance(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    radius: float = constants.R_earth,
) -> jax.Array:
    """Compute great-circle distance using the Haversine formula."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = jnp.sin(dlat / 2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2)**2
    return 2.0 * radius * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))


def mid_pt_sphere(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 608: great-circle midpoint of two (lon, lat) points.

    Faithful port of FV3 ``mid_pt_sphere`` (fv_grid_utils.F90:1981-1992).
    Algorithm:
        1. (lon, lat) → 3D Cartesian unit vector e
        2. e_mid = (e1 + e2) / 2 (Cartesian midpoint)
        3. Normalize e_mid → unit sphere
        4. Cartesian → (lon, lat)

    The result is the point on the great circle through (p1, p2)
    equidistant from both endpoints.  NOT the same as the (lon, lat)
    average — that gives wrong results across the dateline or poles.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : jax.Array (any shape, broadcastable)
        Two points on the sphere in radians.

    Returns
    -------
    lon_mid, lat_mid : jax.Array
        Midpoint on the great circle (radians).
    """
    # latlon → Cartesian
    cl1, sl1 = jnp.cos(lat1), jnp.sin(lat1)
    cl2, sl2 = jnp.cos(lat2), jnp.sin(lat2)
    x1 = cl1 * jnp.cos(lon1)
    y1 = cl1 * jnp.sin(lon1)
    z1 = sl1
    x2 = cl2 * jnp.cos(lon2)
    y2 = cl2 * jnp.sin(lon2)
    z2 = sl2
    # Cartesian midpoint
    xm = 0.5 * (x1 + x2)
    ym = 0.5 * (y1 + y2)
    zm = 0.5 * (z1 + z2)
    # Normalize to unit sphere
    norm = jnp.sqrt(xm * xm + ym * ym + zm * zm)
    norm = jnp.where(norm > 1e-30, norm, 1.0)
    xm = xm / norm
    ym = ym / norm
    zm = zm / norm
    # Back to (lon, lat)
    lat_mid = jnp.arcsin(jnp.clip(zm, -1.0, 1.0))
    lon_mid = jnp.arctan2(ym, xm)
    # Wrap lon to [0, 2π)
    lon_mid = jnp.where(lon_mid < 0.0, lon_mid + 2.0 * jnp.pi, lon_mid)
    return lon_mid, lat_mid


def latlon2xyz(
    lon: jax.Array, lat: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 611: FV3-named alias for ``lonlat_to_cartesian``.

    Faithful port of FV3 ``latlon2xyz`` (fv_grid_utils.F90:1639-1665).
    Convert (lon, lat) in radians to 3D Cartesian unit-sphere
    coordinates::

        x = cos(lat) cos(lon)
        y = cos(lat) sin(lon)
        z = sin(lat)
    """
    return lonlat_to_cartesian(lon, lat)


def xyz2latlon(
    x: jax.Array, y: jax.Array, z: jax.Array,
    eps: float = 1e-10,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 611: Cartesian → (lon, lat) inverse of ``latlon2xyz``.

    Faithful port of FV3 ``cart_to_latlon`` (fv_grid_utils.F90:1739-1777).
    Normalizes (x, y, z) to the unit sphere first; returns ``lon`` in
    ``[0, 2π)`` and ``lat`` in ``[-π/2, π/2]``.

    Matches FV3's ``esl=1.d-10`` guard near the poles (where
    ``|x|+|y| < esl``, longitude is set to 0).
    """
    dist = jnp.sqrt(x * x + y * y + z * z)
    safe = jnp.where(dist > 0.0, dist, 1.0)
    x_n = x / safe
    y_n = y / safe
    z_n = z / safe
    lat = jnp.arcsin(jnp.clip(z_n, -1.0, 1.0))
    near_pole = (jnp.abs(x_n) + jnp.abs(y_n)) < eps
    lon = jnp.where(near_pole, 0.0, jnp.arctan2(y_n, x_n))
    lon = jnp.where(lon < 0.0, lon + 2.0 * jnp.pi, lon)
    return lon, lat


def inner_prod(
    v1: jax.Array, v2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: Cartesian dot product.

    Faithful port of FV3 ``inner_prod`` (fv_grid_utils.F90:984-998).
    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.  Each ``v1`` and ``v2`` is shape ``(..., 3)``.
    """
    return jnp.sum(v1 * v2, axis=-1)


def vect_cross(
    p1: jax.Array, p2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: Cartesian cross product ``e = p1 × p2``.

    Faithful port of FV3 ``vect_cross`` (fv_grid_utils.F90:1781-1791).
    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.
    """
    return jnp.cross(p1, p2, axis=-1)


def normalize_vect(
    e: jax.Array, eps: float = 1e-30,
) -> jax.Array:
    """FV3_3D iter 611: normalize Cartesian vector to unit length.

    Faithful port of FV3 ``normalize_vect`` (fv_grid_utils.F90:
    1880-1893).  Takes the last axis as the 3-vector component;
    broadcasts over leading axes.  Zero-vector input returns the
    input unchanged (avoiding NaN).
    """
    pdot = jnp.sqrt(jnp.sum(e * e, axis=-1, keepdims=True))
    safe = jnp.where(pdot > eps, pdot, 1.0)
    return e / safe


def mid_pt3_cart(
    p1: jax.Array, p2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: Cartesian-input great-circle midpoint.

    Faithful port of FV3 ``mid_pt3_cart`` (fv_grid_utils.F90:
    1996-2022).  Returns the normalized sum (p1 + p2) / |p1 + p2|.

    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.  Each of ``p1``, ``p2`` should already be on the
    unit sphere (no extra normalization beyond the post-sum step).
    """
    return normalize_vect(p1 + p2)


def mid_pt_cart(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: (lon, lat)-input → Cartesian midpoint vector.

    Faithful port of FV3 ``mid_pt_cart`` (fv_grid_utils.F90:
    2026-2036).  Convenience for code that takes (lon, lat) inputs
    but wants the Cartesian midpoint (e.g., FV3 grid generation).
    Returns shape ``(..., 3)``.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    return mid_pt3_cart(p1, p2)


def get_unit_vect2(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: unit tangent vector at the GC midpoint.

    Faithful port of FV3 ``get_unit_vect2`` (fv_grid_utils.F90:
    1848-1863).  Returns the unit tangent vector to the great
    circle through (e1, e2), evaluated at the midpoint and pointing
    from e1 toward e2.  Used in FV3 ``edge_factors`` /
    ``efactor_a2c_v`` for metric construction.

    Algorithm:
        p1 = latlon2xyz(e1)
        p2 = latlon2xyz(e2)
        pc = mid_pt3_cart(p1, p2)
        p3 = p2 × p1           (great-circle pole)
        uc = pc × p3           (tangent at pc)
        uc / |uc|

    Returns shape ``(..., 3)``.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    pc = mid_pt3_cart(p1, p2)
    p3 = vect_cross(p2, p1)
    uc = vect_cross(pc, p3)
    return normalize_vect(uc)


def mirror_xyz(
    p1: jax.Array, p2: jax.Array, p0: jax.Array,
) -> jax.Array:
    """FV3_3D iter 612: reflect ``p0`` across great-circle plane (p1, p2).

    Faithful port of FV3 ``mirror_xyz`` (fv_grid_utils.F90:1668-1702).
    The mirror plane is the great circle through ``p1`` and ``p2``;
    the plane normal is ``nb = (p1 × p2) / |p1 × p2|``.  Mirror image
    of ``p0`` is::

        p = p0 - 2·(p0·nb)·nb

    Used in FV3 cubed-sphere grid generation (panel reflections
    across face symmetry planes).

    Takes the last axis as the 3-vector component; broadcasts on
    leading axes.  Each of ``p1``, ``p2``, ``p0`` is shape
    ``(..., 3)``; result is ``(..., 3)``.
    """
    nb_raw = vect_cross(p1, p2)
    nb = normalize_vect(nb_raw)
    pdot = jnp.sum(p0 * nb, axis=-1, keepdims=True)
    return p0 - 2.0 * pdot * nb


def mirror_latlon(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon0: jax.Array, lat0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 612: (lon, lat) reflection across great-circle (p1,p2).

    Faithful port of FV3 ``mirror_latlon`` (fv_grid_utils.F90:
    1705-1736).  Converts inputs to Cartesian, calls ``mirror_xyz``,
    converts back.  Returns ``(lon3, lat3)`` of the mirror image.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x0, y0, z0 = latlon2xyz(lon0, lat0)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    p0 = jnp.stack([x0, y0, z0], axis=-1)
    p3 = mirror_xyz(p1, p2, p0)
    return xyz2latlon(p3[..., 0], p3[..., 1], p3[..., 2])


def intp_great_circle(
    beta: jax.Array,
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 612: linear-in-Cartesian great-circle interpolation.

    Faithful port of FV3 ``intp_great_circle`` (fv_grid_utils.F90:
    1896-1925).  At ``beta ∈ [0, 1]`` interpolates from ``p1``
    (β=0) to ``p2`` (β=1) along the great circle::

        s = (1-β)·e1 + β·e2;   e_out = s / |s|

    NOTE: this is the SECANT linear interpolant projected to the
    sphere — NOT slerp.  For β=0.5 it matches ``mid_pt_sphere``.
    For an arc-length-uniform variant use ``slerp``.
    """
    alpha = 1.0 - beta
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    s1 = alpha * x1 + beta * x2
    s2 = alpha * y1 + beta * y2
    s3 = alpha * z1 + beta * z2
    dd = jnp.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
    safe = jnp.where(dd > 0.0, dd, 1.0)
    return xyz2latlon(s1 / safe, s2 / safe, s3 / safe)


def slerp(
    beta: jax.Array,
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    eps_omg: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 612: spherical linear interpolation (Shoemake slerp).

    Faithful port of FV3 ``spherical_linear_interpolation``
    (fv_grid_utils.F90:1927-1979).  Arc-length-uniform
    interpolation along the great circle::

        ω = acos(e1·e2)
        e_b = (sin((1-β)ω)·e1 + sin(βω)·e2) / sin(ω)

    Returns ``(lon_b, lat_b)`` at parameter ``β ∈ [0, 1]``.

    Antipodal-point safety: FV3 raises a fatal error for
    ``|ω| < 1e-5``; here we silently return the secant interpolant
    (well-defined for ω=0 colocated points; near-antipodal points
    still have ambiguous slerp direction so caller should avoid).
    """
    alpha = 1.0 - beta
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    dot = jnp.clip(x1 * x2 + y1 * y2 + z1 * z2, -1.0, 1.0)
    omg = jnp.arccos(dot)
    sin_omg = jnp.sin(omg)
    safe_sin = jnp.where(jnp.abs(sin_omg) > eps_omg, sin_omg, 1.0)
    w1 = jnp.sin(alpha * omg) / safe_sin
    w2 = jnp.sin(beta * omg) / safe_sin
    # Fallback to secant for tiny ω (well-defined colocated case)
    secant = jnp.abs(omg) <= eps_omg
    w1 = jnp.where(secant, alpha, w1)
    w2 = jnp.where(secant, beta, w2)
    xb = w1 * x1 + w2 * x2
    yb = w1 * y1 + w2 * y2
    zb = w1 * z1 + w2 * z2
    return xyz2latlon(xb, yb, zb)


def spherical_angle(
    p1: jax.Array, p2: jax.Array, p3: jax.Array,
) -> jax.Array:
    """FV3_3D iter 613: angle at vertex ``p1`` of spherical triangle (p1, p2, p3).

    Faithful port of FV3 ``spherical_angle`` (fv_grid_utils.F90:
    2838-2895).  Computes::

        P = p1 × p2
        Q = p1 × p3
        cos(angle) = (P·Q) / (|P|·|Q|)

    With FV3's degenerate-input fixups:
        - ``ddd <= 0`` (colinear or coincident points) → angle = 0
        - ``|cos| > 1`` (numerical) → angle = π or 0 by sign

    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.
    """
    p_vec = vect_cross(p1, p2)
    q_vec = vect_cross(p1, p3)
    p_sq = jnp.sum(p_vec * p_vec, axis=-1)
    q_sq = jnp.sum(q_vec * q_vec, axis=-1)
    pq = jnp.sum(p_vec * q_vec, axis=-1)
    ddd = p_sq * q_sq
    safe = jnp.where(ddd > 0.0, ddd, 1.0)
    cos_a = pq / jnp.sqrt(safe)
    cos_a = jnp.clip(cos_a, -1.0, 1.0)
    angle = jnp.arccos(cos_a)
    # Degenerate ddd <= 0 → 0
    return jnp.where(ddd > 0.0, angle, 0.0)


def cell_center3(
    p1: jax.Array, p2: jax.Array, p3: jax.Array, p4: jax.Array,
) -> jax.Array:
    """FV3_3D iter 613: Cartesian cell center from 4 corner points.

    Faithful port of FV3 ``cell_center3`` (fv_grid_utils.F90:
    2728-2745).  Returns normalized sum ``(p1+p2+p3+p4)/|sum|``.

    Each ``pi`` is shape ``(..., 3)`` on the unit sphere; result is
    ``(..., 3)`` on the unit sphere.
    """
    return normalize_vect(p1 + p2 + p3 + p4)


def cell_center2(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 613: (lon, lat) cell center from 4 corner (lon, lat).

    Faithful port of FV3 ``cell_center2`` (fv_grid_utils.F90:
    2700-2725).  Latlon wrapper for ``cell_center3``.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x3, y3, z3 = latlon2xyz(lon3, lat3)
    x4, y4, z4 = latlon2xyz(lon4, lat4)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    p3 = jnp.stack([x3, y3, z3], axis=-1)
    p4 = jnp.stack([x4, y4, z4], axis=-1)
    ec = cell_center3(p1, p2, p3, p4)
    return xyz2latlon(ec[..., 0], ec[..., 1], ec[..., 2])


def dist2side_latlon(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon_p: jax.Array, lat_p: jax.Array,
) -> jax.Array:
    """FV3_3D iter 613: angular distance from point to great-circle arc.

    Faithful port of FV3 ``dist2side_latlon`` (fv_grid_utils.F90:
    2812-2834).  Returns the normalized (angular) distance on the
    unit sphere from point ``p`` to the great-circle arc through
    ``(v1, v2)``::

        dist = asin( sin(side) · sin(angle) )

    where ``side`` is the angular distance v1 → p and ``angle`` is
    the spherical angle ∠(v1 v2; v1 p).

    Returns a scalar (or broadcast result) in radians.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    xp, yp, zp = latlon2xyz(lon_p, lat_p)
    c1 = jnp.stack([x1, y1, z1], axis=-1)
    c2 = jnp.stack([x2, y2, z2], axis=-1)
    cp = jnp.stack([xp, yp, zp], axis=-1)
    angle = spherical_angle(c1, c2, cp)
    # side = great-circle distance v1 → p on UNIT sphere (radius=1)
    side = great_circle_distance(lon1, lat1, lon_p, lat_p, radius=1.0)
    return jnp.arcsin(jnp.clip(jnp.sin(side) * jnp.sin(angle), -1.0, 1.0))


def get_center_vect(
    pp: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 618: cell-center tangent unit vectors (u1, u2).

    Faithful port of FV3 ``get_center_vect`` (fv_grid_utils.F90:
    1795-1845), non-``OLD_VECT`` branch (FV3 default).  Given an
    array of cell corner positions ``pp`` of shape
    ``(..., n+1, n+1, 3)``, returns the two unit tangent vectors
    at each cell center::

        pc = cell_center3(SW, SE, NW, NE)
        # u1 (along i / x-direction):
        p1_w = mid_pt3_cart(SW, NW)   # west edge midpoint
        p2_e = mid_pt3_cart(SE, NE)   # east edge midpoint
        p3   = p2_e × p1_w
        u1   = normalize(pc × p3)
        # u2 (along j / y-direction):
        p1_s = mid_pt3_cart(SW, SE)   # south edge midpoint
        p2_n = mid_pt3_cart(NW, NE)   # north edge midpoint
        p3   = p2_n × p1_s
        u2   = normalize(pc × p3)

    Returns ``(u1, u2)`` each of shape ``(..., n, n, 3)``.

    Used by FV3 vector-halo rotation: edges between faces project
    vector components onto these per-cell tangent vectors.
    """
    sw = pp[..., :-1, :-1, :]
    se = pp[..., 1:, :-1, :]
    nw = pp[..., :-1, 1:, :]
    ne = pp[..., 1:, 1:, :]
    pc = cell_center3(sw, se, nw, ne)
    # u1 along i-direction (east-west edges)
    p1_w = mid_pt3_cart(sw, nw)
    p2_e = mid_pt3_cart(se, ne)
    p3_1 = vect_cross(p2_e, p1_w)
    u1 = normalize_vect(vect_cross(pc, p3_1))
    # u2 along j-direction (north-south edges)
    p1_s = mid_pt3_cart(sw, se)
    p2_n = mid_pt3_cart(nw, ne)
    p3_2 = vect_cross(p2_n, p1_s)
    u2 = normalize_vect(vect_cross(pc, p3_2))
    return u1, u2


def c2l_ord4_fv3(
    u: jax.Array, v: jax.Array,
    dx: jax.Array, dy: jax.Array,
    a11: jax.Array, a12: jax.Array,
    a21: jax.Array, a22: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 628: D-grid → cell-center latlon winds (FV3 ``c2l_ord4``).

    Faithful port of FV3 ``c2l_ord4`` (fv_grid_utils.F90:2407-2546,
    grid_type<4 branch).  Uses 4-point Lagrange interpolation in
    the interior::

        utmp[i,j] = c2·(u[i,j-1] + u[i,j+2]) + c1·(u[i,j] + u[i,j+1])
        vtmp[i,j] = c2·(v[i-1,j] + v[i+2,j]) + c1·(v[i,j] + v[i+1,j])

    where ``c1 = 1.125``, ``c2 = -0.125`` (FV3 lines 2422-2424).
    For grid-edge cells (where the 4-point stencil reaches across
    a cube face boundary), falls back to the iter-627
    ``c2l_ord2_fv3`` 2nd-order vorticity-conserving formula.

    Then applies iter-626 a-matrix:

        ua[i,j] = a11·utmp + a12·vtmp
        va[i,j] = a21·utmp + a22·vtmp

    Inputs ``u`` shape ``(n_x, n_y+1, [nlev])``: needs at least
    a 2-cell halo in j-direction for the Lagrange stencil to be
    valid in the interior (cells j=1 and j=n_y-2 use the 2nd-order
    fallback per FV3 lines 2455-2530).  This port assumes ``u`` has
    halo cells pre-padded — only the interior LD cells get the
    4th-order treatment; boundary cells fall back to c2l_ord2.

    Parameters
    ----------
    u, v, dx, dy, a11..a22 : as in ``c2l_ord2_fv3``.

    Returns
    -------
    ua, va : jax.Array, shape ``(n_x, n_y, [nlev])``
        Cell-center geographic winds (4th order in interior,
        2nd order at boundaries).
    """
    c1 = 1.125
    c2 = -0.125
    has_level = u.ndim == dx.ndim + 1
    # Broadcast scalars to level dim
    if has_level:
        a11_b = a11[..., None]; a12_b = a12[..., None]
        a21_b = a21[..., None]; a22_b = a22[..., None]
    else:
        a11_b, a12_b, a21_b, a22_b = a11, a12, a21, a22

    # Start with c2l_ord2 result everywhere (used for edge cells)
    ua_ord2, va_ord2 = c2l_ord2_fv3(u, v, dx, dy, a11, a12, a21, a22)

    # Interior 4-pt Lagrange: needs j-1, j, j+1, j+2 cells of u (so
    # u has at least n_y+1 cells indexed [0, n_y]; the stencil reads
    # u[:, j-1], u[:, j], u[:, j+1], u[:, j+2] for cell-centers j in
    # [1, n_y-2]).  For v: v[i-1, :], v[i, :], v[i+1, :], v[i+2, :].
    n_y_u = u.shape[-2] if has_level else u.shape[-1]
    n_x_v = v.shape[-3] if has_level else v.shape[-2]
    n_x_cells = ua_ord2.shape[-2] if has_level else ua_ord2.shape[-1]
    # Determine which cells get 4th-order treatment
    # Cell-center j ranges over [0, n_y-1] (n_y_u = n_y+1).  The 4-pt
    # stencil reads u[:, j-1..j+2], so valid for j ∈ [1, n_y-2].
    # Compute interior 4th-order interpolation
    # u: (..., n_x, n_y+1, [nlev]); stencil along axis -2 (3D) or -1 (2D)

    # Construct utmp_4 of shape (..., n_x, n_y-2, [nlev]) for cells j ∈ [1, n_y-2]
    if has_level:
        u_jm1 = u[..., :, 0:-3, :]
        u_j   = u[..., :, 1:-2, :]
        u_jp1 = u[..., :, 2:-1, :]
        u_jp2 = u[..., :, 3:,   :]
    else:
        u_jm1 = u[..., :, 0:-3]
        u_j   = u[..., :, 1:-2]
        u_jp1 = u[..., :, 2:-1]
        u_jp2 = u[..., :, 3:]
    utmp_4 = c2 * (u_jm1 + u_jp2) + c1 * (u_j + u_jp1)

    if has_level:
        v_im1 = v[..., 0:-3, :, :]
        v_i   = v[..., 1:-2, :, :]
        v_ip1 = v[..., 2:-1, :, :]
        v_ip2 = v[..., 3:,   :, :]
    else:
        v_im1 = v[..., 0:-3, :]
        v_i   = v[..., 1:-2, :]
        v_ip1 = v[..., 2:-1, :]
        v_ip2 = v[..., 3:,   :]
    vtmp_4 = c2 * (v_im1 + v_ip2) + c1 * (v_i + v_ip1)

    # Apply a-matrix to interior cells (those in [1, n_y-2] × [1, n_x-2]).
    # But utmp_4 spans only ~n_y-2 cells in j; vtmp_4 spans n_x-2 in i.
    # Interior cell-center region: i ∈ [1, n_x-2], j ∈ [1, n_y-2].
    # Both arrays must be sliced to this rectangle.
    # utmp_4 shape: (..., n_x, n_y-2, [nlev]) — full n_x but j cropped
    # vtmp_4 shape: (..., n_x-2, n_y, [nlev]) — full n_y but i cropped
    # Interior rectangle: i ∈ [1, n_x-2], j ∈ [1, n_y-2]
    # Slice utmp_4: take i ∈ [1, n_x-2]  → axis -2 (3D) or -1 (2D)
    if has_level:
        utmp_int = utmp_4[..., 1:-1, :, :]   # shape (..., n_x-2, n_y-2, nlev)
        vtmp_int = vtmp_4[..., :, 1:-1, :]   # shape (..., n_x-2, n_y-2, nlev)
        a11_int = a11_b[..., 1:-1, 1:-1, :]
        a12_int = a12_b[..., 1:-1, 1:-1, :]
        a21_int = a21_b[..., 1:-1, 1:-1, :]
        a22_int = a22_b[..., 1:-1, 1:-1, :]
    else:
        utmp_int = utmp_4[..., 1:-1, :]
        vtmp_int = vtmp_4[..., :, 1:-1]
        a11_int = a11_b[..., 1:-1, 1:-1]
        a12_int = a12_b[..., 1:-1, 1:-1]
        a21_int = a21_b[..., 1:-1, 1:-1]
        a22_int = a22_b[..., 1:-1, 1:-1]
    ua_int = a11_int * utmp_int + a12_int * vtmp_int
    va_int = a21_int * utmp_int + a22_int * vtmp_int

    # Overlay interior 4th-order onto c2l_ord2 baseline (boundary cells
    # keep c2l_ord2 values, FV3 lines 2455-2530)
    if has_level:
        ua = ua_ord2.at[..., 1:-1, 1:-1, :].set(ua_int)
        va = va_ord2.at[..., 1:-1, 1:-1, :].set(va_int)
    else:
        ua = ua_ord2.at[..., 1:-1, 1:-1].set(ua_int)
        va = va_ord2.at[..., 1:-1, 1:-1].set(va_int)
    return ua, va


def c2l_ord2_fv3(
    u: jax.Array, v: jax.Array,
    dx: jax.Array, dy: jax.Array,
    a11: jax.Array, a12: jax.Array,
    a21: jax.Array, a22: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 627: D-grid → cell-center latlon winds (FV3 ``c2l_ord2``).

    Faithful port of FV3 ``c2l_ord2`` (fv_grid_utils.F90:2547-2628,
    grid_type<4 branch).  Vorticity-conserving 2nd-order
    interpolation from D-grid covariant winds to cell-center
    (east, north) geographic winds.

    Algorithm:
        wu[i,j] = u[i,j] · dx[i,j]
        wv[i,j] = v[i,j] · dy[i,j]
        u1[i,j] = 2·(wu[i,j] + wu[i,j+1]) / (dx[i,j] + dx[i,j+1])
        v1[i,j] = 2·(wv[i,j] + wv[i+1,j]) / (dy[i,j] + dy[i+1,j])
        ua[i,j] = a11·u1 + a12·v1
        va[i,j] = a21·u1 + a22·v1

    Parameters
    ----------
    u : jax.Array, shape ``(..., n_x, n_y+1, [nlev])``
        D-grid u-component on north/south edges (covariant).
    v : jax.Array, shape ``(..., n_x+1, n_y, [nlev])``
        D-grid v-component on east/west edges (covariant).
    dx : jax.Array, shape ``(..., n_x, n_y+1)``
        Cell-edge x-length.
    dy : jax.Array, shape ``(..., n_x+1, n_y)``
        Cell-edge y-length.
    a11, a12, a21, a22 : jax.Array, shape ``(..., n_x, n_y)``
        Rotation matrix entries from ``init_cubed_to_latlon``
        (iter 626).

    Returns
    -------
    ua, va : jax.Array, shape ``(..., n_x, n_y, [nlev])``
        Cell-center (east, north) winds.
    """
    # Broadcast dx, dy, a-matrix to level dim if u, v carry extra level axis
    has_level = u.ndim == dx.ndim + 1
    if has_level:
        dx_b = dx[..., None]
        dy_b = dy[..., None]
        a11_b = a11[..., None]
        a12_b = a12[..., None]
        a21_b = a21[..., None]
        a22_b = a22[..., None]
    else:
        dx_b, dy_b = dx, dy
        a11_b, a12_b, a21_b, a22_b = a11, a12, a21, a22

    wu = u * dx_b           # (..., n_x, n_y+1, [nlev])
    wv = v * dy_b           # (..., n_x+1, n_y, [nlev])
    # Slice axis -2 of (wu, dx) to average over j; axis -3 (or -2 for 2D) of
    # (wv, dy) to average over i.  Use jnp slicing with axis index resolved
    # at runtime via the position of the n_y+1 / n_x+1 dim.
    # For u shape (..., n_x, n_y+1, [nlev]): j axis is at position -2 if has_level
    # else -1.  Equivalently: ax_j = -2 if has_level else -1.
    if has_level:
        wu_a = wu[..., :, :-1, :]
        wu_b = wu[..., :, 1:, :]
        dx_a = dx_b[..., :, :-1, :]
        dx_b_ = dx_b[..., :, 1:, :]
        wv_a = wv[..., :-1, :, :]
        wv_b = wv[..., 1:, :, :]
        dy_a = dy_b[..., :-1, :, :]
        dy_b_ = dy_b[..., 1:, :, :]
    else:
        wu_a = wu[..., :, :-1]
        wu_b = wu[..., :, 1:]
        dx_a = dx_b[..., :, :-1]
        dx_b_ = dx_b[..., :, 1:]
        wv_a = wv[..., :-1, :]
        wv_b = wv[..., 1:, :]
        dy_a = dy_b[..., :-1, :]
        dy_b_ = dy_b[..., 1:, :]
    u1 = 2.0 * (wu_a + wu_b) / (dx_a + dx_b_)
    v1 = 2.0 * (wv_a + wv_b) / (dy_a + dy_b_)
    ua = a11_b * u1 + a12_b * v1
    va = a21_b * u1 + a22_b * v1
    return ua, va


def init_cubed_to_latlon(
    agrid_lon: jax.Array, agrid_lat: jax.Array,
    ec1: jax.Array, ec2: jax.Array,
    sin_sg5: jax.Array,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array,
]:
    """FV3_3D iter 626: D-grid → latlon wind rotation matrices.

    Faithful port of FV3 ``init_cubed_to_latlon``
    (fv_grid_utils.F90:2321-2384), grid_type<4 branch.  Computes
    the 8 rotation-matrix entries used by FV3 ``c2l_ord4`` to
    rotate D-grid winds to (east, north) at cell centers.

    Algorithm:
        1. vlon, vlat = unit_vect_latlon(agrid)  — local frame
        2. z11 = ec1 · vlon                       — inner products
           z12 = ec1 · vlat
           z21 = ec2 · vlon
           z22 = ec2 · vlat
        3. a11 =  0.5·z22 / sin_sg5
           a12 = -0.5·z12 / sin_sg5
           a21 = -0.5·z21 / sin_sg5
           a22 =  0.5·z11 / sin_sg5

    The (a11, a12, a21, a22) matrix gives the D-grid → (u_east,
    v_north) projection at each cell center.

    Parameters
    ----------
    agrid_lon, agrid_lat : jax.Array, shape ``(..., n, n)``
        Cell-center positions (radians).
    ec1, ec2 : jax.Array, shape ``(..., n, n, 3)``
        Cell-edge unit tangent vectors (FV3 ``ec1``, ``ec2``).
    sin_sg5 : jax.Array, shape ``(..., n, n)``
        sin of the cell-area diagonal (FV3 ``sin_sg(:,:,5)``).

    Returns
    -------
    (a11, a12, a21, a22, z11, z12, z21, z22, vlon, vlat) : tuple
        Rotation matrix entries + intermediate z and unit vectors.
    """
    vlon, vlat = unit_vect_latlon(agrid_lon, agrid_lat)
    z11 = inner_prod(ec1, vlon)
    z12 = inner_prod(ec1, vlat)
    z21 = inner_prod(ec2, vlon)
    z22 = inner_prod(ec2, vlat)
    safe_sin = jnp.where(jnp.abs(sin_sg5) > 1e-30, sin_sg5, 1.0)
    a11 = 0.5 * z22 / safe_sin
    a12 = -0.5 * z12 / safe_sin
    a21 = -0.5 * z21 / safe_sin
    a22 = 0.5 * z11 / safe_sin
    return a11, a12, a21, a22, z11, z12, z21, z22, vlon, vlat


def mirror_grid_face1_symmetrize(
    face1_lon: jax.Array, face1_lat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 625: face-1 SIGN-averaging symmetrization.

    Faithful port of FV3 ``mirror_grid`` first loop
    (fv_grid_tools.F90:2774-2807).  Symmetrizes face 1 about both
    the lon=0 meridian (i-axis) and the equator (j-axis) by:

        1. For each symmetric 4-tuple of grid points
           ``(i, j), (npx-i+1, j), (i, npy-j+1), (npx-i+1, npy-j+1)``:
        2. Compute the average of the absolute values, then assign
           ``SIGN(avg, original_value)`` to each of the 4 corners.

    Result: ``|lon|`` and ``|lat|`` are pairwise-equal across the
    mirror, preserving the sign-pattern of the original grid.

    For odd ``npx``, the central column ``i = (npx+1)/2`` is
    forced to ``lon = 0`` (FV3 lines 2799-2804).

    Parameters
    ----------
    face1_lon, face1_lat : jax.Array, shape ``(npx, npy)``
        Face-1 corner positions in radians.

    Returns
    -------
    lon_sym, lat_sym : jax.Array, shape ``(npx, npy)``
        Symmetrized face-1 grid.
    """
    npx = face1_lon.shape[0]
    npy = face1_lon.shape[1]

    # Build mirrors via reverse-indexing
    lon = face1_lon
    lat = face1_lat
    # 4-tuple of absolute lons
    avg_abs_lon = 0.25 * (
        jnp.abs(lon)
        + jnp.abs(lon[::-1, :])
        + jnp.abs(lon[:, ::-1])
        + jnp.abs(lon[::-1, ::-1])
    )
    avg_abs_lat = 0.25 * (
        jnp.abs(lat)
        + jnp.abs(lat[::-1, :])
        + jnp.abs(lat[:, ::-1])
        + jnp.abs(lat[::-1, ::-1])
    )
    # Apply SIGN(avg, original_value)
    lon_sym = jnp.copysign(avg_abs_lon, lon)
    lat_sym = jnp.copysign(avg_abs_lat, lat)

    # Odd-npx central column: lon = 0
    if npx % 2 == 1:
        center_i = (npx - 1) // 2
        lon_sym = lon_sym.at[center_i, :].set(0.0)
    if npy % 2 == 1:
        # FV3 doesn't have a corresponding odd-npy clause for lat=0,
        # but if the grid is symmetric about the equator, lat=0
        # naturally at j-center; SIGN-averaging already enforces this.
        center_j = (npy - 1) // 2
        # lat at center row is already 0 by symmetry; force exactly 0
        lat_sym = lat_sym.at[:, center_j].set(
            jnp.where(jnp.abs(lat_sym[:, center_j]) < 1e-12, 0.0,
                      lat_sym[:, center_j])
        )

    return lon_sym, lat_sym


def mirror_grid_faces(
    face1_lon: jax.Array, face1_lat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 624: build 6-face cubed-sphere from face 1.

    Faithful port of FV3 ``mirror_grid`` faces-2-to-6 construction
    (fv_grid_tools.F90:2809-2897).  Takes face-1 (lon, lat) and
    builds faces 2-6 via FV3's exact rot_3d (iter 623) sequences:

        face 2: rot_z(-90°)
        face 3: rot_z(-90°) → rot_x(+90°)
        face 4: rot_z(-180°) → rot_x(+90°)
        face 5: rot_z(+90°) → rot_y(+90°)
        face 6: rot_y(+90°)  (FV3 also applies rot_z(0°) = identity)

    Parameters
    ----------
    face1_lon, face1_lat : jax.Array, shape ``(n+1, n+1)``
        Face-1 corner positions in radians (e.g., output of
        ``gnomonic_grids``).

    Returns
    -------
    lons, lats : jax.Array, shape ``(6, n+1, n+1)``
        6-face cubed-sphere corner positions in radians.  Face 1
        is the input.

    Note: this port covers the rotation sequence for faces 2-6.
    FV3's first loop (lines 2774-2807, intra-face-1 symmetrization
    via SIGN-of-(|...|) averaging) is NOT included — input is
    assumed already symmetrized (e.g., post-``symm_ed``).
    """
    x1, y1, z1 = latlon2xyz(face1_lon, face1_lat)

    # Face 2: rot_z(-90°)
    x2, y2, z2 = rot_3d(3, x1, y1, z1, jnp.asarray(-90.0), degrees=True)
    lon2, lat2 = xyz2latlon(x2, y2, z2)

    # Face 3: rot_z(-90°) → rot_x(+90°)
    xa, ya, za = rot_3d(3, x1, y1, z1, jnp.asarray(-90.0), degrees=True)
    x3, y3, z3 = rot_3d(1, xa, ya, za, jnp.asarray(90.0), degrees=True)
    lon3, lat3 = xyz2latlon(x3, y3, z3)

    # Face 4: rot_z(-180°) → rot_x(+90°)
    xa, ya, za = rot_3d(3, x1, y1, z1, jnp.asarray(-180.0), degrees=True)
    x4, y4, z4 = rot_3d(1, xa, ya, za, jnp.asarray(90.0), degrees=True)
    lon4, lat4 = xyz2latlon(x4, y4, z4)

    # Face 5: rot_z(+90°) → rot_y(+90°)
    xa, ya, za = rot_3d(3, x1, y1, z1, jnp.asarray(90.0), degrees=True)
    x5, y5, z5 = rot_3d(2, xa, ya, za, jnp.asarray(90.0), degrees=True)
    lon5, lat5 = xyz2latlon(x5, y5, z5)

    # Face 6: rot_y(+90°)  (rot_z(0°) is identity, omitted)
    x6, y6, z6 = rot_3d(2, x1, y1, z1, jnp.asarray(90.0), degrees=True)
    lon6, lat6 = xyz2latlon(x6, y6, z6)

    lons = jnp.stack([face1_lon, lon2, lon3, lon4, lon5, lon6], axis=0)
    lats = jnp.stack([face1_lat, lat2, lat3, lat4, lat5, lat6], axis=0)
    return lons, lats


def rot_3d(
    axis: int,
    x1: jax.Array, y1: jax.Array, z1: jax.Array,
    angle: jax.Array,
    degrees: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 623: 3D rotation about coordinate axis.

    Faithful port of FV3 ``rot_3d`` (fv_grid_tools.F90:2410-2467).
    Rotates Cartesian (x1, y1, z1) by ``angle`` about axis::

        axis = 1: x-axis (y, z rotated)
        axis = 2: y-axis (x, z rotated)
        axis = 3: z-axis (x, y rotated)

    FV3 sign convention (left-handed about each axis as the code
    is written):
        axis 1: y' = c·y + s·z,    z' = -s·y + c·z
        axis 2: x' = c·x - s·z,    z' =  s·x + c·z
        axis 3: x' = c·x + s·y,    y' = -s·x + c·y

    Parameters
    ----------
    axis : int
        Rotation axis (1, 2, or 3).
    x1, y1, z1 : jax.Array
        Input Cartesian coordinates (any shape).
    angle : jax.Array
        Rotation angle (radians unless ``degrees=True``).
    degrees : bool, default False
        If True, ``angle`` is in degrees.
    """
    a = jnp.deg2rad(angle) if degrees else angle
    c = jnp.cos(a)
    s = jnp.sin(a)
    if axis == 1:
        return x1, c * y1 + s * z1, -s * y1 + c * z1
    if axis == 2:
        return c * x1 - s * z1, y1, s * x1 + c * z1
    if axis == 3:
        return c * x1 + s * y1, -s * x1 + c * y1, z1
    raise ValueError(f"Invalid axis: {axis} (must be 1, 2, or 3)")


def fill_ghost(
    q: jax.Array, ng: int, value: float,
) -> jax.Array:
    """FV3_3D iter 633: fill 4 corner-ghost regions with constant.

    Faithful JAX port of FV3 ``fill_ghost_r4`` / ``fill_ghost_r8``
    (fv_grid_utils.F90:3070-3147).  Fills the 4 corner-ghost
    rectangular regions OUTSIDE the face corners with ``value``.
    Used to mask FV3's cube-vertex singularity (no well-defined
    neighbor at the 8 cube corners, propagated to 4 corner-ghost
    blocks per face).

    Input ``q`` has shape ``(..., npx-1+2·ng, npy-1+2·ng)`` where:
        - npx-1 = number of interior cells in x (cell-centered)
        - ng = number of halo cells on each side

    The 4 corner-ghost regions are the rectangles in the halo
    where BOTH i and j are outside the interior range:
        - SW corner ghost: i ∈ [0, ng-1], j ∈ [0, ng-1]
        - SE corner ghost: i ∈ [-ng:], j ∈ [0, ng-1]
        - NE corner ghost: i ∈ [-ng:], j ∈ [-ng:]
        - NW corner ghost: i ∈ [0, ng-1], j ∈ [-ng:]

    Parameters
    ----------
    q : jax.Array, shape ``(..., n_x_halo, n_y_halo)``
        Field with halo.  Two trailing axes interpreted as (i, j).
    ng : int
        Number of halo cells on each side.
    value : float
        Fill value for corner ghost cells.

    Returns
    -------
    q_filled : jax.Array
        Copy of ``q`` with the 4 corner-ghost regions set to ``value``.
    """
    n_x = q.shape[-2]
    n_y = q.shape[-1]
    i_idx = jnp.arange(n_x)[:, None]
    j_idx = jnp.arange(n_y)[None, :]
    # Interior: ng <= i < n_x - ng, ng <= j < n_y - ng
    # Corner ghost: (i < ng AND j < ng) OR (i >= n_x-ng AND j < ng) OR
    #               (i >= n_x-ng AND j >= n_y-ng) OR (i < ng AND j >= n_y-ng)
    i_lo = i_idx < ng
    i_hi = i_idx >= (n_x - ng)
    j_lo = j_idx < ng
    j_hi = j_idx >= (n_y - ng)
    corner_mask = (
        (i_lo & j_lo) | (i_hi & j_lo) | (i_hi & j_hi) | (i_lo & j_hi)
    )
    # Broadcast mask over leading axes
    return jnp.where(corner_mask, value, q)


def global_qsum(p: jax.Array) -> jax.Array:
    """FV3_3D iter 632: quick global sum without area weighting.

    Faithful JAX port of FV3 ``global_qsum`` (fv_grid_utils.F90:
    2999-3018).  Serial (non-MPI) implementation; for distributed
    runs use ``legoesm.distributed.global_sum_mpi``.

    Returns the scalar sum of all elements in ``p`` (no area
    weighting; unlike iter-623 ``g_sum``).
    """
    return jnp.sum(p)


def global_mx(q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 632: global min / max reduction.

    Faithful JAX port of FV3 ``global_mx`` (fv_grid_utils.F90:
    3020-3046).  Serial (non-MPI) implementation; for distributed
    runs use ``legoesm.distributed.global_min_mpi`` /
    ``global_max_mpi``.

    Returns ``(qmin, qmax)`` over all elements of ``q``.
    """
    return jnp.min(q), jnp.max(q)


def global_mx_c(q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 632: global min / max at cell corners.

    Faithful JAX port of FV3 ``global_mx_c`` (fv_grid_utils.F90:
    3048-3067).  Identical to ``global_mx`` but FV3 distinguishes
    cell-center vs corner indexing in the signature; for legoESM
    they're identical operations on the input array.
    """
    return jnp.min(q), jnp.max(q)


def g_sum(
    p: jax.Array, area: jax.Array, mode: int = 0,
) -> jax.Array:
    """FV3_3D iter 623: area-weighted global sum.

    Faithful JAX port of FV3 ``g_sum`` (fv_grid_utils.F90:2946-2996),
    serial branch.  Computes::

        gsum = Σ_ij  p(i,j) · area(i,j)

    If ``mode == 1``, returns ``gsum / global_area`` (area-weighted
    global mean).  Otherwise returns ``gsum`` (area-weighted total).

    Parameters
    ----------
    p : jax.Array
        Field to be summed (any shape; must match ``area`` shape).
    area : jax.Array
        Cell areas (matching shape).
    mode : int, default 0
        If 1, divide by global area (returns area-weighted mean).

    Returns
    -------
    g : jax.Array
        Scalar global sum (or area-weighted mean if mode=1).

    Note: serial (non-MPI) implementation.  MPI reduction is the
    caller's responsibility (legoESM uses ``global_sum_mpi`` for
    distributed runs).
    """
    weighted = p * area
    gsum = jnp.sum(weighted)
    if mode == 1:
        global_area = jnp.sum(area)
        return gsum / global_area
    return gsum


def edge_factor_along_axis_nonortho(
    agrid_outside_lon: jax.Array, agrid_outside_lat: jax.Array,
    agrid_inside_lon: jax.Array, agrid_inside_lat: jax.Array,
    grid_corner_lon: jax.Array, grid_corner_lat: jax.Array,
) -> jax.Array:
    """FV3_3D iter 631: A→B grid interpolation weights at face boundary.

    Faithful port of FV3 ``edge_factors`` non-ortho branch
    (fv_grid_utils.F90:1212-1289).  Single-axis 1D variant: for
    one face boundary, computes per-corner interpolation weights
    ``edge_factor[j] = d2 / (d1 + d2)`` where:

        py[j]     = mid_pt_sphere(agrid_outside[j], agrid_inside[j])
        d1[j+1]   = great_circle_dist(py[j],   grid_corner[j+1])
        d2[j+1]   = great_circle_dist(py[j+1], grid_corner[j+1])

    This is used by FV3's A-grid → B-grid (corner-located)
    interpolation at non-orthogonal cubed-sphere face boundaries::

        q_corner[j+1] = (1 - edge[j+1]) · q_A[j+1] + edge[j+1] · q_A[j]

    Parameters
    ----------
    agrid_outside_lon, agrid_outside_lat : jax.Array, shape ``(n,)``
        A-grid cell-center positions just OUTSIDE the boundary
        (the halo cells across the face edge).
    agrid_inside_lon, agrid_inside_lat : jax.Array, shape ``(n,)``
        A-grid cell-center positions just INSIDE the boundary.
    grid_corner_lon, grid_corner_lat : jax.Array, shape ``(n+1,)``
        B-grid (corner) positions along the boundary.

    Returns
    -------
    edge_factor : jax.Array, shape ``(n+1,)``
        Per-corner interpolation weights.  Corner j+1 (interior)
        gets ``d2/(d1+d2)``; corners 0 and n (the face corners
        themselves) get NaN — FV3 also leaves them as ``big_number``
        (lines 1213-1216) since the edge factor formula degenerates
        there.
    """
    # Midpoints between outside and inside cells at each row
    py_lon, py_lat = mid_pt_sphere(
        agrid_outside_lon, agrid_outside_lat,
        agrid_inside_lon, agrid_inside_lat,
    )  # shape (n,)
    # For each interior corner j ∈ [1, n-1]: d1 = dist(py[j-1], grid[j]);
    # d2 = dist(py[j], grid[j])
    # py[j-1] = py[:-1], py[j] = py[1:]
    # grid corners interior: grid[1:-1] (shape (n-1,))
    d1 = great_circle_distance(
        py_lon[:-1], py_lat[:-1],
        grid_corner_lon[1:-1], grid_corner_lat[1:-1],
        radius=1.0,
    )
    d2 = great_circle_distance(
        py_lon[1:], py_lat[1:],
        grid_corner_lon[1:-1], grid_corner_lat[1:-1],
        radius=1.0,
    )
    safe_sum = jnp.where(d1 + d2 > 0.0, d1 + d2, 1.0)
    interior = d2 / safe_sum
    # Build full (n+1,) array with NaN at endpoints (FV3 big_number)
    edge_factor = jnp.full(grid_corner_lon.shape[0], jnp.nan)
    edge_factor = edge_factor.at[1:-1].set(interior)
    return edge_factor


def make_fv3_native_grid(
    im: int,
    grid_type: int = 0,
    symmetrize_face1: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 629: end-to-end FV3 native cubed-sphere grid builder.

    Integration wrapper for iters 622, 625, 624:

        1. ``gnomonic_grids(im, grid_type)``       — face-1 (iter 622)
        2. ``mirror_grid_face1_symmetrize``        — face-1 sym (iter 625)
        3. ``mirror_grid_faces``                   — faces 2-6 (iter 624)

    Reproduces FV3's full cubed-sphere construction pipeline as
    a single public API.  Returns the 6-face (lon, lat) arrays
    matching the FV3 face numbering convention (1..6 → indices 0..5).

    Parameters
    ----------
    im : int
        Number of cells per face edge.
    grid_type : int, default 0
        Grid type forwarded to ``gnomonic_grids``:
            0 → ``gnomonic_ed``   (FV3 canonical)
            1 → ``gnomonic_dist``
            2 → ``gnomonic_angl``
    symmetrize_face1 : bool, default True
        If True, apply ``mirror_grid_face1_symmetrize`` (FV3
        first-loop SIGN-averaging) before mirroring to 6 faces.

    Returns
    -------
    lons, lats : jax.Array, shape ``(6, im+1, im+1)``
        Cubed-sphere corner positions for all 6 faces in radians.
    """
    lon1, lat1 = gnomonic_grids(im, grid_type=grid_type)
    if symmetrize_face1:
        lon1, lat1 = mirror_grid_face1_symmetrize(lon1, lat1)
    return mirror_grid_faces(lon1, lat1)


def gnomonic_grids(
    im: int, grid_type: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 622: dispatcher for FV3 ``gnomonic_grids``.

    Faithful port of FV3 ``gnomonic_grids`` (fv_grid_utils.F90:
    1290-1311).  Dispatches to one of three grid generators by
    ``grid_type``:

        grid_type = 0 → ``gnomonic_ed``   (canonical, equal-distance edges; FV3 default)
        grid_type = 1 → ``gnomonic_dist`` (linear equi-distance gnomonic)
        grid_type = 2 → ``gnomonic_angl`` (equi-angular gnomonic)

    Post-processing (FV3 lines 1301-1308) for all grid_type < 3:
        1. ``symm_ed`` symmetrizes about i/j midplanes.
        2. Longitude shift by -π to bring grid into FV3's standard
           orientation (face 2 center → 0, not π).

    Parameters
    ----------
    im : int
        Number of cells per face edge.  Grid has shape ``(im+1, im+1)``.
    grid_type : int, default 0
        Grid construction algorithm (0, 1, or 2).

    Returns
    -------
    lon, lat : jax.Array, shape ``(im+1, im+1)``
        Cubed-sphere face corner positions in radians (FV3
        orientation after the -π shift).
    """
    if grid_type == 0:
        lon, lat = gnomonic_ed(im)
    elif grid_type == 1:
        lon, lat = gnomonic_dist(im)
    elif grid_type == 2:
        lon, lat = gnomonic_angl(im)
    else:
        raise ValueError(
            f"Unsupported grid_type: {grid_type} (must be 0, 1, or 2)"
        )
    # grid_type < 3 post-processing (FV3 lines 1301-1308)
    lon, lat = symm_ed(lon, lat)
    lon = lon - jnp.pi
    return lon, lat


def gnomonic_ed(im: int) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 621: equal-distance-edge cubed-sphere grid for face 2.

    Faithful port of FV3 ``gnomonic_ed`` (fv_grid_utils.F90:1313-1407).
    This is FV3's grid of choice for global cloud-resolving runs.

    Properties (FV3 docstring):
        - Defined by intersections of great circles
        - max(dx,dy) / min(dx,dy) = √2 ≈ 1.4142
        - Max aspect ratio = 1.06089
        - N-S coordinate curves are const longitude on the 4 faces
          with the equator

    Algorithm:
        1. East/West edges at constant longitude (0.75π, 1.25π).
        2. North/South edges obtained by ``mirror_latlon`` of W
           edge across the (NW, SE) diagonal.
        3. Interior Cartesian coordinates obtained by projecting
           edge values onto the constant-x = -1/√3 face cube.
        4. Convert back to (lon, lat).

    Parameters
    ----------
    im : int
        Number of cells per face edge.  Grid has shape ``(im+1, im+1)``.

    Returns
    -------
    lon, lat : jax.Array, shape ``(im+1, im+1)``
        Cubed-sphere face-2 corner positions in radians.
    """
    rsq3 = 1.0 / jnp.sqrt(3.0)
    alpha = jnp.arcsin(rsq3)
    pi = jnp.pi
    dely = 2.0 * alpha / im

    n = im + 1

    # Step 1: W and E edges (FV3 lines 1345-1350)
    j_idx = jnp.arange(n, dtype=jnp.float64)
    lon = jnp.zeros((n, n), dtype=jnp.float64)
    lat = jnp.zeros((n, n), dtype=jnp.float64)
    west_theta = -alpha + dely * j_idx
    lon = lon.at[0, :].set(0.75 * pi)
    lon = lon.at[im, :].set(1.25 * pi)
    lat = lat.at[0, :].set(west_theta)
    lat = lat.at[im, :].set(west_theta)

    # Step 2: S and N edges by mirror_latlon of W edge column (FV3 lines 1354-1359)
    # FV3 loop: for i in 2..im:
    #   mirror_latlon( (lon[0,0], lat[0,0]),  (lon[im,im], lat[im,im]),
    #                  (lon[0,i-1], lat[0,i-1]), (lon[i-1, 0], lat[i-1, 0]) )
    # Vectorize over i ∈ [1, im-1] (0-indexed)
    i_idx = jnp.arange(1, im, dtype=jnp.float64)
    # Reference: SW corner (already at lon[0,0], lat[0,0]) and NE corner
    # (already at lon[im,im], lat[im,im]).  But these are not yet set
    # — lon[im,im] = lat[im,im] are from the W/E edge assignments.
    # W/E edges already set lon[0,0]=0.75π, lat[0,0]=-α; lon[im,im]=1.25π,
    # lat[im,im]=alpha.
    lon_sw, lat_sw = lon[0, 0], lat[0, 0]
    lon_ne, lat_ne = lon[im, im], lat[im, im]
    # Source points (W edge column at j=i): vary i in [1, im-1]
    i_int = jnp.arange(1, im)
    lon_src = lon[0, i_int]
    lat_src = lat[0, i_int]
    lon_s_row, lat_s_row = mirror_latlon(
        lon_sw, lat_sw,
        lon_ne, lat_ne,
        lon_src, lat_src,
    )
    # South edge (j=0): row i, S edge → (lamda(i,1), theta(i,1))
    lon = lon.at[i_int, 0].set(lon_s_row)
    lat = lat.at[i_int, 0].set(lat_s_row)
    # North edge (j=im): same lon, theta flipped
    lon = lon.at[i_int, im].set(lon_s_row)
    lat = lat.at[i_int, im].set(-lat_s_row)

    # Step 3: Convert edges to Cartesian, project onto constant-x face
    # (FV3 lines 1370-1382)
    # i=0 column (W edge), j ∈ [1, im-1]
    x_w_full, y_w_full, z_w_full = latlon2xyz(lon[0, :], lat[0, :])
    safe_x_w = jnp.where(jnp.abs(x_w_full) > 1e-30, x_w_full, 1.0)
    pp2_i0 = -y_w_full * rsq3 / safe_x_w  # y' = -y * rsq3 / x
    pp3_i0 = -z_w_full * rsq3 / safe_x_w
    # j=0 row (S edge), i ∈ [1, im-1]
    x_s_full, y_s_full, z_s_full = latlon2xyz(lon[:, 0], lat[:, 0])
    safe_x_s = jnp.where(jnp.abs(x_s_full) > 1e-30, x_s_full, 1.0)
    pp2_j0 = -y_s_full * rsq3 / safe_x_s
    pp3_j0 = -z_s_full * rsq3 / safe_x_s
    # 4 corners: latlon2xyz directly
    x_corners_w = x_w_full  # (im+1,) — W edge i=0 has all j
    y_corners_w = y_w_full
    z_corners_w = z_w_full
    # FV3 uses raw latlon2xyz for corners but the same projection is needed
    # for j=0 and j=im endpoints too.  For interior points, we use the
    # projection.  For the corners, latlon2xyz gives the position on the
    # unit sphere.  But the FV3 algorithm explicitly sets pp(i, 1) and
    # pp(1, j) from the projection then sets corner positions from raw
    # latlon2xyz2.  Final step is pp(2,i,j) = pp(2,i,1) and
    # pp(3,i,j) = pp(3,1,j) for interior (i>1, j>1).
    # This means the interior i=0, j ∈ [1, im-1] uses the projected
    # values; for i=0 and j=0 corners use direct latlon2xyz.
    # FV3 line 1386: pp(1,i,j) = -rsq3 for ALL i, j → constant x face.

    # Step 4: Build full (pp2, pp3) by taking pp2 from j=0 row (i-varying)
    # and pp3 from i=0 column (j-varying).  This gives the cube-face
    # coordinates on the constant-x face.
    pp1 = jnp.full((n, n), -rsq3)
    # pp2[i, j] = pp2_j0[i] (varies with i, constant in j)
    pp2 = jnp.broadcast_to(pp2_j0[:, None], (n, n))
    # pp3[i, j] = pp3_i0[j] (varies with j, constant in i)
    pp3 = jnp.broadcast_to(pp3_i0[None, :], (n, n))
    # At the 4 corners use direct latlon2xyz (FV3 lines 1362-1365)
    # Corner SW (i=0, j=0): use lon[0,0]/lat[0,0] → (x, y, z) directly
    # We need to override the broadcast values at the 4 corners and
    # the i=0/j=0 edges with the exact values from the projection above.
    # i=0 column: pp2[0, j] should be from latlon2xyz directly (W edge);
    # but the projection above already gives the right answer when
    # pp2_j0[0] = pp2_i0[0] = 0 (W-edge has lon=0.75π, so y/x ratio is
    # known).  Verify by ensuring pp2[0, j] doesn't break and pp3[i, 0]
    # likewise.

    # j=0 row: pp3 should be pp3_j0 (S edge i-vary), NOT pp3_i0[0]
    pp3 = pp3.at[:, 0].set(pp3_j0)
    # i=0 col: pp2 should be pp2_i0 (W edge j-vary), NOT pp2_j0[0]
    pp2 = pp2.at[0, :].set(pp2_i0)
    # j=im row: similar, use S edge mirrored to N
    pp3 = pp3.at[:, im].set(-pp3_j0)  # N edge: theta flipped → z flipped
    # i=im col: E edge mirror of W edge: lon=1.25π so y/x ratio flipped
    pp2 = pp2.at[im, :].set(-pp2_i0)  # E edge: y flipped relative to W

    # Step 5: Convert pp back to (lon, lat) (FV3 line 1399)
    lon_out, lat_out = xyz2latlon(pp1, pp2, pp3)
    return lon_out, lat_out


def symm_ed(
    lamda: jax.Array, theta: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 619: enforce ED-grid symmetry about i/j midplanes.

    Faithful port of FV3 ``symm_ed`` (fv_grid_utils.F90:1587-1626).
    Operates on a face-2 ED grid of shape ``(im+1, im+1)`` and
    enforces symmetry in both axes via three passes:

    1. Copy lamda's first column into all interior columns
       (FV3 lines 1595-1599).
    2. Symmetrize about i=im/2+1: pair (i, im+2-i) gets avg/π
       reflection (lines 1601-1611).
    3. Symmetrize about j=im/2+1: pair (j, im+2-j) gets avg in
       theta (with sign flip) and avg in lamda (lines 1614-1624).

    Assumes input is the FV3 face-2 orientation produced by
    ``gnomonic_dist`` — symmetries use the FV3 ``+π/-π``
    convention that's specific to that face orientation.

    Parameters
    ----------
    lamda, theta : jax.Array, shape ``(im+1, im+1)``
        Longitude, latitude in radians (FV3 face-2 layout).

    Returns
    -------
    lamda_sym, theta_sym : jax.Array, shape ``(im+1, im+1)``
        Symmetrized grid.
    """
    n = lamda.shape[0]
    im = n - 1
    pi = jnp.pi

    # Step 1: lamda[i, 1:im+1] = lamda[i, 0] for i ∈ [1, im-1]
    # FV3 lines 1595-1599: only interior columns (j>0) updated;
    # row i=0 and i=im untouched.
    lamda = lamda.at[1:im, 1:im + 1].set(lamda[1:im, 0:1])

    # Step 2: symmetrize about i=im/2+1 (FV3 1601-1611)
    i_half = im // 2
    i_idx = jnp.arange(i_half)
    ip_idx = im - i_idx
    avg_lon = 0.5 * (lamda[i_idx, :] - lamda[ip_idx, :])
    lamda = lamda.at[i_idx, :].set(avg_lon + pi)
    lamda = lamda.at[ip_idx, :].set(pi - avg_lon)
    avg_lat = 0.5 * (theta[i_idx, :] + theta[ip_idx, :])
    theta = theta.at[i_idx, :].set(avg_lat)
    theta = theta.at[ip_idx, :].set(avg_lat)

    # Step 3: symmetrize about j=im/2+1 (FV3 1614-1624)
    # Only i ∈ [1, im-1] (interior columns) are updated
    j_half = im // 2
    j_idx = jnp.arange(j_half)
    jp_idx = im - j_idx
    int_i = slice(1, im)
    avg_lon_j = 0.5 * (lamda[int_i, :][:, j_idx] + lamda[int_i, :][:, jp_idx])
    lamda = lamda.at[int_i, j_idx].set(avg_lon_j)
    lamda = lamda.at[int_i, jp_idx].set(avg_lon_j)
    avg_lat_j = 0.5 * (theta[int_i, :][:, j_idx] - theta[int_i, :][:, jp_idx])
    theta = theta.at[int_i, j_idx].set(avg_lat_j)
    theta = theta.at[int_i, jp_idx].set(-avg_lat_j)

    return lamda, theta


def gnomonic_angl(im: int) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 617: equi-angular gnomonic grid for FV3 face 2.

    Faithful port of FV3 ``gnomonic_angl`` (fv_grid_utils.F90:
    1531-1556).  Builds the canonical FV3 equi-angular cubed-
    sphere grid for face 2 (-x face)::

        dp = π/(2·im)
        p1 = -1/√3                                (constant)
        p2 = -1/√3 · tan(-π/4 + (j-1)·dp)
        p3 =  1/√3 · tan(-π/4 + (k-1)·dp)

    Then ``cart_to_latlon`` to (lon, lat).

    Parameters
    ----------
    im : int
        Number of cells per face edge.  Grid has shape ``(im+1, im+1)``.

    Returns
    -------
    lon, lat : jax.Array, shape ``(im+1, im+1)``
        Cubed-sphere corner positions in radians.
    """
    dp = 0.5 * jnp.pi / im
    rsq3 = 1.0 / jnp.sqrt(3.0)
    idx = jnp.arange(im + 1, dtype=jnp.float64)
    # Match FV3 (j, k) layout: j varies axis 0, k varies axis 1
    j_grid, k_grid = jnp.meshgrid(idx, idx, indexing="ij")
    p1 = jnp.full_like(j_grid, -rsq3)
    p2 = -rsq3 * jnp.tan(-0.25 * jnp.pi + j_grid * dp)
    p3 = rsq3 * jnp.tan(-0.25 * jnp.pi + k_grid * dp)
    return xyz2latlon(p1, p2, p3)


def gnomonic_dist(im: int) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 617: equi-distance gnomonic grid for FV3 face 2.

    Faithful port of FV3 ``gnomonic_dist`` (fv_grid_utils.F90:
    1558-1585).  Builds the equi-distance cubed-sphere grid for
    face 2 (-x face)::

        p1 = -1/√3                                (constant)
        p2 =  1/√3 - (j-1)·2/(im·√3)
        p3 = -1/√3 + (k-1)·2/(im·√3)

    Then ``cart_to_latlon`` to (lon, lat).

    Same return convention as ``gnomonic_angl``.
    """
    rsq3 = 1.0 / jnp.sqrt(3.0)
    xf = -rsq3
    y0 = rsq3
    dy = -2.0 * rsq3 / im
    z0 = -rsq3
    dz = 2.0 * rsq3 / im
    idx = jnp.arange(im + 1, dtype=jnp.float64)
    j_grid, k_grid = jnp.meshgrid(idx, idx, indexing="ij")
    p1 = jnp.full_like(j_grid, xf)
    p2 = y0 + j_grid * dy
    p3 = z0 + k_grid * dz
    return xyz2latlon(p1, p2, p3)


def rotate_winds_sphere_cube(
    u: jax.Array, v: jax.Array,
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
    lon_t: jax.Array, lat_t: jax.Array,
    direction: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 661: rotate winds between sphere and cube frames at point.

    Faithful JAX port of FV3 ``rotate_winds``
    (tools/test_cases.F90:8183-8226).

    Geometry: at central point ``t1=(lon_t, lat_t)``, the i-axis
    of the cube goes from p3 → p1 (projected to tangent plane);
    j-axis goes from p4 → p2.  FV3 lon-shift by π convention is
    applied to (e_lon, e_lat) of the geographic frame at t1.

    Algorithm:

        ee1 = get_unit_vector_fv3(p3, t1, p1)        # cube i-axis
        ee2 = get_unit_vector_fv3(p4, t1, p2)        # cube j-axis
        elon = (-sin(λ-π), cos(λ-π), 0)              # geo east at t1
        elat = (-sin(φ)·cos(λ-π), -sin(φ)·sin(λ-π), cos(φ))
        g_ij = ee_i · e_lonlat_j
        if dir=1 (sphere → cube):
            newu = u·g11 + v·g12
            newv = u·g21 + v·g22
        else (cube → sphere):
            det = g11·g22 - g21·g12
            newu = (u·g22 - v·g12) / det
            newv = (-u·g21 + v·g11) / det

    Parameters
    ----------
    u, v : jax.Array
        Wind components (broadcastable to scalar or matching p1..t1).
    lon1, lat1, ..., lon4, lat4 : jax.Array
        4 neighboring points (p1, p2, p3, p4) in lat/lon.
    lon_t, lat_t : jax.Array
        Central point t1.
    direction : int, default 1
        1 = sphere-to-cube; 2 = cube-to-sphere.

    Returns
    -------
    newu, newv : jax.Array
    """
    ee1 = get_unit_vector_fv3(lon3, lat3, lon_t, lat_t, lon1, lat1)
    ee2 = get_unit_vector_fv3(lon4, lat4, lon_t, lat_t, lon2, lat2)
    # FV3 lon-shift by π convention
    lon_shifted = lon_t - jnp.pi
    sin_lon = jnp.sin(lon_shifted)
    cos_lon = jnp.cos(lon_shifted)
    sin_lat = jnp.sin(lat_t)
    cos_lat = jnp.cos(lat_t)
    elon = jnp.stack([-sin_lon, cos_lon, jnp.zeros_like(sin_lon)], axis=-1)
    elat = jnp.stack(
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1,
    )
    g11 = inner_prod(ee1, elon)
    g12 = inner_prod(ee1, elat)
    g21 = inner_prod(ee2, elon)
    g22 = inner_prod(ee2, elat)
    if direction == 1:
        new_u = u * g11 + v * g12
        new_v = u * g21 + v * g22
    elif direction == 2:
        det = g11 * g22 - g21 * g12
        safe_det = jnp.where(jnp.abs(det) > 1e-30, det, 1.0)
        new_u = (u * g22 - v * g12) / safe_det
        new_v = (-u * g21 + v * g11) / safe_det
    else:
        raise ValueError(f"direction must be 1 or 2, got {direction}")
    return new_u, new_v


def dcmip16_bc_uwind_pert(
    z: jax.Array, lat: jax.Array, lon: jax.Array,
    up: float = 1.0,
    zp: float = 1.5e4,
    Rp: float | None = None,
    center_lon: float | None = None,
    center_lat: float | None = None,
) -> jax.Array:
    """FV3_3D iter 671: DCMIP16 BC localized wind perturbation.

    Faithful JAX port of FV3 ``DCMIP16_BC_uwind_pert``
    (tools/test_cases.F90:6823-6838).  Localized Gaussian-in-x,
    Hermite-cubic-in-z wind perturbation for triggering the
    baroclinic instability in DCMIP16 Test 410.

    Algorithm:
        zrat = z / zp
        ZZ   = max(1 - 3·zrat² + 2·zrat³, 0)        (Hermite vertical taper)
        dst  = great_circle_distance(point, center)
        pert = max(0, up · ZZ · exp(-(dst/Rp)²))

    Default FV3 constants:
        up=1 m/s (peak amplitude)
        zp=15000 m (vertical scale)
        Rp=R_earth/10 (horizontal scale)
        center = (π/9, 2π/9) (FV3 perturbation focal point)

    Parameters
    ----------
    z : jax.Array
        Height (m).
    lat, lon : jax.Array
        Cell-center positions (radians).
    up, zp, Rp, center_lon, center_lat : float, optional
        DCMIP16 BC perturbation parameters.

    Returns
    -------
    pert : jax.Array
        Wind perturbation (m/s).
    """
    pi = jnp.pi
    if Rp is None:
        Rp = constants.R_earth / 10.0
    if center_lon is None:
        center_lon = pi / 9.0
    if center_lat is None:
        center_lat = 2.0 * pi / 9.0
    zrat = z / zp
    ZZ = jnp.maximum(1.0 - 3.0 * zrat * zrat + 2.0 * zrat * zrat * zrat, 0.0)
    dst = great_circle_distance(
        lon, lat,
        jnp.asarray(center_lon), jnp.asarray(center_lat),
        radius=constants.R_earth,
    )
    return jnp.maximum(0.0, up * ZZ * jnp.exp(-((dst / Rp) ** 2)))


def dcmip16_bc_uwind(
    z: jax.Array, T: jax.Array, lat: jax.Array,
    KK: float = 3.0,
    Te: float = 310.0,
    Tp: float = 240.0,
    b: float = 2.0,
) -> jax.Array:
    """FV3_3D iter 668: DCMIP16 BC zonal wind profile.

    Faithful JAX port of FV3 ``DCMIP16_BC_uwind``
    (tools/test_cases.F90:6807-6821).  Baroclinic-wind profile
    derived from T via geostrophic balance + centripetal::

        Tir = z·exp(-(z·g/(b·R_d·T0))²)
        Ti2 = 0.5·(K+2)·(Te-Tp)/(Te·Tp)·Tir
        UU  = g·K/R · Ti2 · (cos(lat)^(K-1) - cos(lat)^(K+1)) · T
        u   = -Ω·R·cos(lat) + sqrt((Ω·R·cos(lat))² + R·cos(lat)·UU)

    Used with iter-667 ``dcmip16_bc_temperature`` to build the
    DCMIP16 Test 410 IC.

    Parameters
    ----------
    z : jax.Array
        Height (m).
    T : jax.Array
        Temperature (K), from ``dcmip16_bc_temperature``.
    lat : jax.Array
        Latitude (radians).
    """
    g = constants.g
    Rdgas = constants.R_d
    radius = constants.R_earth
    omega = constants.Omega
    T0 = 0.5 * (Te + Tp)
    zsc = z * g / (b * Rdgas * T0)
    Tir = z * jnp.exp(-zsc * zsc)
    Ti2 = 0.5 * (KK + 2.0) * (Te - Tp) / (Te * Tp) * Tir
    cos_lat = jnp.cos(lat)
    K_int = int(KK)
    UU = (
        g * KK / radius * Ti2
        * (cos_lat ** (K_int - 1) - cos_lat ** (K_int + 1)) * T
    )
    discriminant = (omega * radius * cos_lat) ** 2 + radius * cos_lat * UU
    safe_disc = jnp.maximum(discriminant, 0.0)
    return -omega * radius * cos_lat + jnp.sqrt(safe_disc)


def dcmip16_bc_sphum(
    p: jax.Array, ps: jax.Array, lat: jax.Array,
    q0: float = 0.018,
    qt: float = 1.0e-12,
    phiW: float | None = None,
    pw: float = 34000.0,
    p0: float = 1.0e5,
    ptrop: float = 1.0e4,
) -> jax.Array:
    """FV3_3D iter 668: DCMIP16 BC specific humidity profile.

    Faithful JAX port of FV3 ``DCMIP16_BC_sphum``
    (tools/test_cases.F90:6840-6852).

    Algorithm:

        eta = p / ps
        if p > ptrop:
            q = q0·exp(-(lat/phiW)⁴)·exp(-((eta-1)·p0/pw)²)
        else:
            q = qt

    Default DCMIP16 BC constants (FV3 lines 6499-6503):
        q0=0.018, qt=1e-12, phiW=2π/9, pw=34000, p0=1e5, ptrop=1e4

    Used in FV3 DCMIP16 Test 410 BC moist IC.
    """
    if phiW is None:
        phiW = 2.0 * jnp.pi / 9.0
    eta = p / ps
    q_moist = (
        q0
        * jnp.exp(-((lat / phiW) ** 4))
        * jnp.exp(-((eta - 1.0) * p0 / pw) ** 2)
    )
    return jnp.where(p > ptrop, q_moist, qt)


def dcmip16_bc_temperature(
    z: jax.Array, lat: jax.Array,
    KK: float = 3.0,
    Te: float = 310.0,
    Tp: float = 240.0,
    b: float = 2.0,
    lapse: float = 0.005,
) -> jax.Array:
    """FV3_3D iter 667: DCMIP16 baroclinic-instability temperature profile.

    Faithful JAX port of FV3 ``DCMIP16_BC_temperature``
    (tools/test_cases.F90:6774-6789).  Jablonowski-Williamson
    BC test temperature::

        IT = cos(lat)^K - K/(K+2) · cos(lat)^(K+2)
        zsc = z·g/(b·R_d·T0)
        Tr = (1 - 2·zsc²) · exp(-zsc²)
        T1 = (1/T0)·exp(lapse·z/T0) + (T0-Tp)/(T0·Tp)·Tr
        T2 = 0.5·(K+2)·(Te-Tp)/(Te·Tp)·Tr
        T  = 1 / (T1 - T2·IT)

    Default DCMIP16 BC constants:
        KK = 3 (zonal wave number)
        Te = 310 K, Tp = 240 K, T0 = (Te+Tp)/2 = 275 K
        b = 2, lapse = 0.005 K/m
    """
    g = constants.g
    Rdgas = constants.R_d
    T0 = 0.5 * (Te + Tp)  # FV3 note: WRONG in document, here = 275
    IT = (
        jnp.cos(lat) ** KK
        - KK / (KK + 2.0) * jnp.cos(lat) ** (KK + 2.0)
    )
    zsc = z * g / (b * Rdgas * T0)
    Tr = (1.0 - 2.0 * zsc * zsc) * jnp.exp(-zsc * zsc)
    T1 = (1.0 / T0) * jnp.exp(lapse * z / T0) + (T0 - Tp) / (T0 * Tp) * Tr
    T2 = 0.5 * (KK + 2.0) * (Te - Tp) / (Te * Tp) * Tr
    return 1.0 / (T1 - T2 * IT)


def dcmip16_bc_pressure(
    z: jax.Array, lat: jax.Array,
    KK: float = 3.0,
    Te: float = 310.0,
    Tp: float = 240.0,
    b: float = 2.0,
    lapse: float = 0.005,
    p0: float = 1.0e5,
) -> jax.Array:
    """FV3_3D iter 667: DCMIP16 BC pressure profile (companion to T).

    Faithful JAX port of FV3 ``DCMIP16_BC_pressure``
    (tools/test_cases.F90:6791-6805):

        IT  = cos(lat)^K - K/(K+2) · cos(lat)^(K+2)
        Tir = z · exp(-(z·g/(b·R_d·T0))²)
        Ti1 = (1/lapse)·(exp(lapse·z/T0) - 1) + Tir·(T0-Tp)/(T0·Tp)
        Ti2 = 0.5·(K+2)·(Te-Tp)/(Te·Tp)·Tir
        p   = p0·exp(-g/R_d · (Ti1 - Ti2·IT))

    Used in FV3 DCMIP16 baroclinic-instability test (Test 410).
    """
    g = constants.g
    Rdgas = constants.R_d
    T0 = 0.5 * (Te + Tp)
    IT = (
        jnp.cos(lat) ** KK
        - KK / (KK + 2.0) * jnp.cos(lat) ** (KK + 2.0)
    )
    zsc = z * g / (b * Rdgas * T0)
    Tir = z * jnp.exp(-zsc * zsc)
    Ti1 = (
        (1.0 / lapse) * (jnp.exp(lapse * z / T0) - 1.0)
        + Tir * (T0 - Tp) / (T0 * Tp)
    )
    Ti2 = 0.5 * (KK + 2.0) * (Te - Tp) / (Te * Tp) * Tir
    return p0 * jnp.exp(-g / Rdgas * (Ti1 - Ti2 * IT))


def get_staggered_grid_fv3(
    pt_b_lon: jax.Array, pt_b_lat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 674: B-grid corners → C/D-grid edge midpoints.

    Faithful JAX port of FV3 ``get_staggered_grid``
    (tools/fv_treat_da_inc.F90:444-475).  Given B-grid corner
    positions, compute C-grid (east/west edge) and D-grid
    (north/south edge) midpoints via iter-608 ``mid_pt_sphere``::

        pt_d[i, j] = mid_pt_sphere(pt_b[i, j], pt_b[i+1, j])  # N/S edge
        pt_c[i, j] = mid_pt_sphere(pt_b[i, j], pt_b[i, j+1])  # E/W edge

    Parameters
    ----------
    pt_b_lon, pt_b_lat : jax.Array, shape (..., n+1, n+1)
        B-grid corner positions (radians).

    Returns
    -------
    pt_c_lon, pt_c_lat : jax.Array, shape (..., n+1, n)
        C-grid east/west edge midpoints.
    pt_d_lon, pt_d_lat : jax.Array, shape (..., n, n+1)
        D-grid north/south edge midpoints.
    """
    # D-grid (N/S edges): midpoint between (i, j) and (i+1, j) (axis -2)
    pt_d_lon, pt_d_lat = mid_pt_sphere(
        pt_b_lon[..., :-1, :], pt_b_lat[..., :-1, :],
        pt_b_lon[..., 1:, :], pt_b_lat[..., 1:, :],
    )
    # C-grid (E/W edges): midpoint between (i, j) and (i, j+1) (axis -1)
    pt_c_lon, pt_c_lat = mid_pt_sphere(
        pt_b_lon[..., :, :-1], pt_b_lat[..., :, :-1],
        pt_b_lon[..., :, 1:], pt_b_lat[..., :, 1:],
    )
    return pt_c_lon, pt_c_lat, pt_d_lon, pt_d_lat


def get_height_given_pressure_fv3(
    wz: jax.Array, peln: jax.Array,
    log_p: jax.Array,
) -> jax.Array:
    """FV3_3D iter 684: height at given log-pressure (inverse of iter-683).

    Faithful JAX port of FV3 ``get_height_given_pressure``
    (tools/fv_diagnostics.F90:4366-4411), mirror-method
    extrapolation for below-surface pressures.

    Algorithm:
        Extend (pn, gz) below surface via mirror reflection:
            pn[km+1+i] = 2·pn[km] - pn[km-i]    for i = 0..k2
            gz[km+1+i] = 2·gz[km] - gz[km-i]
        where k2 = max(12, km/2+1).  This effectively reflects
        the atmospheric column about the surface for smooth
        extrapolation.

        For each target log_p:
            find k where pn[k] <= log_p <= pn[k+1]
            height = gz[k] + (gz[k+1] - gz[k]) ·
                     (log_p - pn[k]) / (pn[k+1] - pn[k])

    Used by FV3 for pressure-level diagnostic height lookups.

    Parameters
    ----------
    wz : jax.Array, shape (..., km+1)
        Layer interface heights (top-down: wz[0]=top, wz[km]=surface).
    peln : jax.Array, shape (..., km+1)
        Log-pressure at layer interfaces.
    log_p : jax.Array, shape (...,)
        Target log-pressure value(s).

    Returns
    -------
    height : jax.Array, shape (...,)
        Heights at log_p (m).
    """
    km = wz.shape[-1] - 1
    k2 = max(12, km // 2 + 1)
    n_total = km + 1 + k2          # length of extended pn, gz
    # Build extended pn, gz via mirror reflection
    # FV3: gz[km+1+i] = 2·gz[km] - gz[km-i]  for i=0..k2-1 (Python indexing)
    # First, the original arrays go to index km (inclusive) = wz.shape[-1] - 1
    # Then we append k2 more entries.
    # km is the surface index (0-indexed); mirror entries:
    # gz_extended[km+i+1] = 2·wz[km] - wz[km-i-1] for i=0..k2-1
    # FV3 1-indexed: gz[k] = 2·gz[km+1] - gz[l] where l = 2·(km+1) - k.
    # k_ext = km+i+1 (0-indexed) → l_0indexed = km - i - 1.
    # Build via jnp.flip on wz[..., :-1], take first k2.
    mirror_gz = 2.0 * wz[..., -1:] - jnp.flip(
        wz[..., :-1], axis=-1,
    )[..., :k2]
    mirror_pn = 2.0 * peln[..., -1:] - jnp.flip(
        peln[..., :-1], axis=-1,
    )[..., :k2]
    pn_ext = jnp.concatenate([peln, mirror_pn], axis=-1)
    gz_ext = jnp.concatenate([wz, mirror_gz], axis=-1)
    # Find k where pn[k] <= log_p <= pn[k+1]
    # pn_ext is monotonically increasing (peln increases top-down, mirror continues).
    leq_count = jnp.sum(
        (pn_ext[..., :-1] <= log_p[..., None]).astype(jnp.int32), axis=-1,
    )
    k_idx = jnp.clip(leq_count - 1, 0, n_total - 2)
    pn_k = jnp.take_along_axis(pn_ext, k_idx[..., None], axis=-1).squeeze(-1)
    pn_kp1 = jnp.take_along_axis(pn_ext, (k_idx + 1)[..., None], axis=-1).squeeze(-1)
    gz_k = jnp.take_along_axis(gz_ext, k_idx[..., None], axis=-1).squeeze(-1)
    gz_kp1 = jnp.take_along_axis(gz_ext, (k_idx + 1)[..., None], axis=-1).squeeze(-1)
    denom = pn_kp1 - pn_k
    safe_denom = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    return gz_k + (gz_kp1 - gz_k) * (log_p - pn_k) / safe_denom


def get_pressure_given_height_fv3(
    wz: jax.Array, peln: jax.Array,
    height: jax.Array, ts: jax.Array,
    fac: float | None = None,
) -> jax.Array:
    """FV3_3D iter 683: pressure at given height (inverse of iter-681).

    Faithful JAX port of FV3 ``get_pressure_given_height``
    (tools/fv_diagnostics.F90:4312-4365).

    Algorithm:
        For each target height h:
            if h >= wz[km] (above surface):
                find k where wz[k+1] <= h < wz[k]
                logp = peln[k] + (peln[k+1] - peln[k]) ·
                       (wz[k] - h) / (wz[k] - wz[k+1])
                p = exp(logp)
            else (below surface, extrapolate):
                tm = (R_d/g) · (ts + 3.25e-3·(wz[km] - h))
                                       # 6.5 K/km half-depth lapse
                p = exp(peln[km] + (wz[km] - h)/tm)

    Used by FV3 to convert z-level diagnostics to pressure-level.

    Parameters
    ----------
    wz : jax.Array, shape (..., km+1)
        Layer interface heights (FV3 wz[0]=top, wz[km]=surface).
    peln : jax.Array, shape (..., km+1)
        Log-pressure at layer interfaces.
    height : jax.Array, shape (...,)
        Target heights at which to evaluate pressure (m).
    ts : jax.Array, shape (...,)
        Surface temperature (K) for below-surface extrapolation.
    fac : float, optional
        Optional multiplicative factor applied to output.

    Returns
    -------
    p : jax.Array, shape (...,)
        Pressure at target height (Pa).
    """
    g = constants.g
    Rdgas = constants.R_d
    # km = wz.shape[-1] - 1
    # 0-indexed convention: wz[..., 0]=top, wz[..., km]=surface.
    # FV3's "k from 1..km" sweeps top to bottom; we find k_python in [0, km-1].
    km = wz.shape[-1] - 1
    # Search for k where wz[k+1] <= height < wz[k]
    # Since wz monotonically decreasing top→bottom: find smallest k such that
    # wz[k+1] <= height; this gives the candidate layer index.
    # Using a different approach: gtmask[k] = (wz[k] > height); count → k.
    gt_mask = wz[..., :-1] > height[..., None]   # (..., km)
    # k = number of True minus 1? Actually first True gives k.
    # We want the first index where wz[k] > height AND wz[k+1] <= height.
    # i.e., the first k where height is "below" wz[k] (since decreasing).
    # If height >= wz[0] (top), we'd want k=0; if height < wz[km] (surface),
    # we go to the extrapolation branch.
    # Compute k as count of wz[k+1] > height (gives the index k where wz[k+1]
    # first crosses below height).
    # Actually easier: use jnp.searchsorted on -wz to find ascending position.
    # Or use: k = argmax of (height >= wz[1:]) (first index where wz[k+1] is
    # at or below height).
    # Use cumulative sum: count of (wz[k+1] > height); k = that count
    # (clipped to [0, km-1]).
    above_count = jnp.sum((wz[..., 1:] > height[..., None]).astype(jnp.int32), axis=-1)
    k = jnp.clip(above_count, 0, km - 1)
    # Take wz[k], wz[k+1], peln[k], peln[k+1]
    wz_k = jnp.take_along_axis(wz, k[..., None], axis=-1).squeeze(-1)
    wz_kp1 = jnp.take_along_axis(wz, (k + 1)[..., None], axis=-1).squeeze(-1)
    peln_k = jnp.take_along_axis(peln, k[..., None], axis=-1).squeeze(-1)
    peln_kp1 = jnp.take_along_axis(peln, (k + 1)[..., None], axis=-1).squeeze(-1)
    denom = wz_k - wz_kp1
    safe_denom = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    logp_band = peln_k + (peln_kp1 - peln_k) * (wz_k - height) / safe_denom
    p_band = jnp.exp(logp_band)
    # Below-surface extrapolation
    wz_surface = wz[..., -1]   # wz[km]
    peln_surface = peln[..., -1]
    tm = (Rdgas / g) * (ts + 3.25e-3 * (wz_surface - height))
    p_extrap = jnp.exp(peln_surface + (wz_surface - height) / tm)
    above_surface = height >= wz_surface
    p = jnp.where(above_surface, p_band, p_extrap)
    if fac is not None:
        p = fac * p
    return p


def compute_brn_fv3(
    ua: jax.Array, va: jax.Array,
    delp: jax.Array, delz: jax.Array,
    cape: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 688: Bulk Richardson Number supercell diagnostic.

    Faithful JAX port of FV3 ``compute_brn``
    (tools/fv_diagnostics.F90:5574-5645).

    Bulk Richardson Number (BRN) is the classic environmental
    supercell diagnostic.  Pairs with iter-686 UH and iter-687 SRH:

        BRN = CAPE / (0.5 · shear06²)

    where shear06 is the magnitude of the mass-weighted wind
    difference between the 0-500 m and 0-6 km layers.

    BRN ranges (per FV3 reference / forecaster usage):
        < 10    : extreme shear, splitting/short-lived
        10-45   : classic supercell range
        > 50    : weak shear, ordinary thunderstorm

    Algorithm (vectorized):

        ht[k] = layer-midpoint height above surface
              = half-thickness of lowest layer at k=km
              + cumulative full thicknesses upward
        mask06[k]  = (ht[k] <= 6000)
        mask005[k] = (ht[k] <=  500)
        u06  = Σ delp · ua · mask06  / Σ delp · mask06
        u005 = Σ delp · ua · mask005 / Σ delp · mask005
        shear06 = sqrt((u005-u06)² + (v005-v06)²)
        BRN = CAPE / (0.5 · max(0.1, shear06²))

    Parameters
    ----------
    ua, va : jax.Array, shape (..., km)
        A-grid wind components.
    delp : jax.Array, shape (..., km)
        Pressure thickness (positive, FV3 convention).
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3 — top-down).
    cape : jax.Array, shape (...,)
        CAPE (J/kg).

    Returns
    -------
    brn : jax.Array, shape (...,)
        Bulk Richardson Number (dimensionless).
    shear06 : jax.Array, shape (...,)
        0-6 km bulk shear magnitude (m/s).
    """
    half_thick = -0.5 * delz
    step = half_thick[..., :-1] + half_thick[..., 1:]
    step_rev = step[..., ::-1]
    cum = jnp.cumsum(step_rev, axis=-1)
    ht_upper = (half_thick[..., -1:] + cum)[..., ::-1]
    ht = jnp.concatenate(
        [ht_upper, half_thick[..., -1:]],
        axis=-1,
    )
    mask06 = ht <= 6000.0
    mask005 = ht <= 500.0
    u06_num = jnp.sum(delp * ua * mask06, axis=-1)
    v06_num = jnp.sum(delp * va * mask06, axis=-1)
    m06 = jnp.sum(delp * mask06, axis=-1)
    u005_num = jnp.sum(delp * ua * mask005, axis=-1)
    v005_num = jnp.sum(delp * va * mask005, axis=-1)
    m005 = jnp.sum(delp * mask005, axis=-1)
    safe_m06 = jnp.where(m06 > 0.0, m06, 1.0)
    safe_m005 = jnp.where(m005 > 0.0, m005, 1.0)
    du = u005_num / safe_m005 - u06_num / safe_m06
    dv = v005_num / safe_m005 - v06_num / safe_m06
    shear06 = jnp.sqrt(du * du + dv * dv)
    brn = cape / (0.5 * jnp.maximum(0.1, shear06 * shear06))
    return brn, shear06


def bunkers_vector_fv3(
    ua: jax.Array, va: jax.Array,
    delz: jax.Array | None = None,
    pt: jax.Array | None = None, q: jax.Array | None = None,
    peln: jax.Array | None = None,
    hydrostatic: bool = False,
    zvir: float | None = None,
    bunkers_d: float = 7.5,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 689: Bunkers right-mover storm motion vector.

    Faithful JAX port of FV3 ``bunkers_vector``
    (tools/fv_diagnostics.F90:4970-5046).

    Bunkers storm motion is the empirical right-mover supercell
    motion predictor.  Pairs with iter-687 SRH (which expects an
    explicit storm motion (uc, vc) as input).

    Algorithm:

        umn, vmn = depth-weighted mean wind in 0-6 km layer
        usfc, vsfc = surface (lowest layer) wind
        u6km, v6km = linearly-interpolated wind at z = 6000 m
        (ushr, vshr) = (u6km - usfc, v6km - vsfc)
        shrmag = ||(ushr, vshr)||
        uc = umn + 7.5 · vshr / shrmag
        vc = vmn - 7.5 · ushr / shrmag

    The 7.5 m/s offset to the right of the shear vector is the
    empirical Bunkers (2000) right-mover constant.

    Parameters
    ----------
    ua, va : jax.Array, shape (..., km)
        A-grid wind components.
    delz : jax.Array, shape (..., km), optional
        Layer thickness (NEGATIVE in FV3).  Required if not hydrostatic.
    pt, q, peln, zvir : optional
        Hydrostatic dz reconstruction; required if hydrostatic.
    hydrostatic : bool, default False.
    bunkers_d : float, default 7.5
        Empirical right-mover offset (m/s).

    Returns
    -------
    uc, vc : jax.Array, shape (...,)
        Bunkers right-mover storm motion components (m/s).
    """
    if hydrostatic:
        if pt is None or q is None or peln is None:
            raise ValueError("hydrostatic=True requires pt, q, peln")
        if zvir is None:
            zvir = constants.R_v / constants.R_d - 1.0
        rdg = constants.R_d / constants.g
        dz = rdg * pt * (1.0 + zvir * q) * (peln[..., 1:] - peln[..., :-1])
    else:
        if delz is None:
            raise ValueError("hydrostatic=False requires delz")
        dz = -delz

    # Layer top/bottom heights above surface (k=0 top, k=-1 surface)
    dz_reversed = dz[..., ::-1]
    cumsum_from_surface = jnp.cumsum(dz_reversed, axis=-1)
    zh_above = cumsum_from_surface[..., ::-1]
    zh_below = zh_above - dz

    # Mass-weighted mean wind in 0-6 km layer
    dz_eff = jnp.maximum(
        0.0,
        jnp.minimum(zh_above, 6000.0) - jnp.maximum(zh_below, 0.0),
    )
    total = jnp.sum(dz_eff, axis=-1)
    safe_total = jnp.where(total > 0.0, total, 1.0)
    umn = jnp.sum(ua * dz_eff, axis=-1) / safe_total
    vmn = jnp.sum(va * dz_eff, axis=-1) / safe_total

    # Surface wind (lowest layer in FV3 = last Python index)
    usfc = ua[..., -1]
    vsfc = va[..., -1]

    # Linear interpolation of wind at z = 6 km across bracket layer
    in_bracket = (zh_below < 6000.0) & (zh_above >= 6000.0)
    ua_in_b = jnp.sum(ua * in_bracket, axis=-1)
    va_in_b = jnp.sum(va * in_bracket, axis=-1)
    zh_bot_b = jnp.sum(zh_below * in_bracket, axis=-1)
    zh_top_b = jnp.sum(zh_above * in_bracket, axis=-1)
    dz_b = zh_top_b - zh_bot_b
    # Layer just below bracket (lower altitude, higher Python index):
    # shift in_bracket so True moves one position higher
    in_below = jnp.concatenate(
        [jnp.zeros_like(in_bracket[..., :1]), in_bracket[..., :-1]],
        axis=-1,
    )
    ua_below = jnp.sum(ua * in_below, axis=-1)
    va_below = jnp.sum(va * in_below, axis=-1)
    safe_dz_b = jnp.where(dz_b > 0.0, dz_b, 1.0)
    frac = (6000.0 - zh_bot_b) / safe_dz_b
    u6km = ua_below + (ua_in_b - ua_below) * frac
    v6km = va_below + (va_in_b - va_below) * frac
    # If column doesn't bracket 6 km, fall back to umn/vmn
    has_bracket = jnp.any(in_bracket, axis=-1)
    u6km = jnp.where(has_bracket, u6km, umn)
    v6km = jnp.where(has_bracket, v6km, vmn)

    ushr = u6km - usfc
    vshr = v6km - vsfc
    shrmag = jnp.sqrt(ushr * ushr + vshr * vshr)
    safe_shrmag = jnp.where(shrmag > 0.0, shrmag, 1.0)
    uc = umn + bunkers_d * vshr / safe_shrmag
    vc = vmn - bunkers_d * ushr / safe_shrmag
    return uc, vc


def helicity_relative_fv3(
    ua: jax.Array, va: jax.Array,
    delz: jax.Array | None = None,
    pt: jax.Array | None = None, q: jax.Array | None = None,
    peln: jax.Array | None = None,
    z_bot: float = 0.0,
    z_top: float = 3000.0,
    hydrostatic: bool = False,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 687: storm-relative helicity (SRH) diagnostic.

    Faithful JAX port of FV3 ``helicity_relative``
    (tools/fv_diagnostics.F90:4811-4895).  Vectorized integration:

        SRH = Σ_k_in_window  (u_k - uc)·dv_dz_k - (v_k - vc)·du_dz_k

    where (uc, vc) is the depth-weighted mean wind in [z_bot, z_top]
    and dv/dz, du/dz are centered differences from neighboring layers.

    SRH thresholds (per NWS):
        150-299: weak tornado possible
        300-449: supercells + strong tornadoes
        > 450: violent tornadoes

    Algorithm (vectorized, similar to iter-686 UH):

        dz from delz or hydrostatic
        zh_above[k], zh_below[k] cumulative from surface
        dz_eff[k] = max(0, min(zh_above, z_top) - max(zh_below, z_bot))
        uc = Σ ua·dz_eff / Σ dz_eff
        vc = Σ va·dz_eff / Σ dz_eff
        du_dz[k] = 0.5·(ua[k-1] - ua[k+1])     (centered diff)
        dv_dz[k] = 0.5·(va[k-1] - va[k+1])
        SRH = Σ_k in_window (ua[k]-uc)·dv_dz[k] - (va[k]-vc)·du_dz[k]

    Parameters
    ----------
    ua, va : jax.Array, shape (..., km)
        A-grid wind components.
    delz, pt, q, peln, zvir : optional
        Vertical grid info (see iter-686).
    z_bot, z_top : float, default 0, 3000 m
        SRH window (NWS standard 0-3 km layer).
    hydrostatic : bool, default False.

    Returns
    -------
    srh : jax.Array, shape (...,)
        Storm-relative helicity (m²/s²).
    """
    if hydrostatic:
        if pt is None or q is None or peln is None:
            raise ValueError("hydrostatic=True requires pt, q, peln")
        if zvir is None:
            zvir = constants.R_v / constants.R_d - 1.0
        rdg = constants.R_d / constants.g
        dz = rdg * pt * (1.0 + zvir * q) * (peln[..., 1:] - peln[..., :-1])
    else:
        if delz is None:
            raise ValueError("hydrostatic=False requires delz")
        dz = -delz

    # Cumulative zh (top of each layer from surface up)
    dz_reversed = dz[..., ::-1]
    cumsum_from_surface = jnp.cumsum(dz_reversed, axis=-1)
    zh_above = cumsum_from_surface[..., ::-1]
    zh_below = zh_above - dz
    dz_eff = jnp.maximum(
        0.0,
        jnp.minimum(zh_above, z_top) - jnp.maximum(zh_below, z_bot),
    )
    # Depth-weighted mean wind in window
    total_dz_eff = jnp.sum(dz_eff, axis=-1)
    safe_total = jnp.where(total_dz_eff > 0.0, total_dz_eff, 1.0)
    uc = jnp.sum(ua * dz_eff, axis=-1) / safe_total
    vc = jnp.sum(va * dz_eff, axis=-1) / safe_total

    # Centered vertical wind shears (interior layers; edges = 0)
    # FV3 indexing: k=1 is top, k=km bottom (top-down).  Python: k=0 top.
    # du_dz[k] = 0.5·(ua[k-1] - ua[k+1])
    du_dz = jnp.zeros_like(ua)
    dv_dz = jnp.zeros_like(va)
    du_dz = du_dz.at[..., 1:-1].set(
        0.5 * (ua[..., :-2] - ua[..., 2:])
    )
    dv_dz = dv_dz.at[..., 1:-1].set(
        0.5 * (va[..., :-2] - va[..., 2:])
    )
    # Mask: only sum over layers with dz_eff > 0
    in_window = dz_eff > 0.0
    srh_k = (ua - uc[..., None]) * dv_dz - (va - vc[..., None]) * du_dz
    return jnp.sum(jnp.where(in_window, srh_k, 0.0), axis=-1)


def updraft_helicity_fv3(
    vort: jax.Array, w: jax.Array,
    delz: jax.Array | None = None,
    pt: jax.Array | None = None, q: jax.Array | None = None,
    peln: jax.Array | None = None,
    z_bot: float = 2000.0,
    z_top: float = 5000.0,
    hydrostatic: bool = False,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 686: updraft helicity (supercell diagnostic).

    Faithful JAX port of FV3 ``updraft_helicity``
    (tools/fv_diagnostics.F90:5048-5108).  Vertical integral of
    ``vort · w · dz`` between ``z_bot`` and ``z_top``.  Used as
    proxy for storm-rotation intensity (UH > 50 m²/s² = supercell
    threshold per NWS guidance).

    Algorithm:
        if hydrostatic:
            dz[k] = (R_d/g) · pt[k]·(1 + zvir·q[k])·(peln[k+1] - peln[k])
        else:
            dz[k] = -delz[k]    (FV3 delz < 0)
        zh_above[k] = sum(dz[k:], bottom-up)
        zh_below[k] = zh_above[k] - dz[k]
        dz_eff[k] = max(0, min(zh_above[k], z_top)
                          - max(zh_below[k], z_bot))
        uh = sum_k(vort[k] · w[k] · dz_eff[k])

    Parameters
    ----------
    vort : jax.Array, shape (..., km)
        3D vorticity (1/s).
    w : jax.Array, shape (..., km)
        Vertical velocity (m/s).
    delz : jax.Array, shape (..., km), optional
        Layer thicknesses (m, FV3 convention negative).  Required
        if hydrostatic=False.
    pt, q, peln : jax.Array, optional
        For hydrostatic: temperature, water vapor, log-p.
    z_bot, z_top : float, default 2000, 5000 m
        Integration bounds (NWS standard 2-5 km layer).
    hydrostatic : bool, default False
    zvir : float, optional
        Virtual-T factor (R_v/R_d - 1).

    Returns
    -------
    uh : jax.Array, shape (...,)
        Updraft helicity (m²/s²).
    """
    if hydrostatic:
        if pt is None or q is None or peln is None:
            raise ValueError("hydrostatic=True requires pt, q, peln")
        if zvir is None:
            zvir = constants.R_v / constants.R_d - 1.0
        rdg = constants.R_d / constants.g
        dz = rdg * pt * (1.0 + zvir * q) * (peln[..., 1:] - peln[..., :-1])
    else:
        if delz is None:
            raise ValueError("hydrostatic=False requires delz")
        dz = -delz

    # zh_above[k] = sum(dz[k:], axis=-1) — height of top of layer k from surface
    dz_reversed = dz[..., ::-1]
    cumsum_from_surface = jnp.cumsum(dz_reversed, axis=-1)
    zh_above = cumsum_from_surface[..., ::-1]   # top-down order
    zh_below = zh_above - dz                    # bottom of each layer
    # Effective dz within [z_bot, z_top]
    dz_eff = jnp.maximum(
        0.0,
        jnp.minimum(zh_above, z_top) - jnp.maximum(zh_below, z_bot),
    )
    return jnp.sum(vort * w * dz_eff, axis=-1)


def prt_mxm_fv3(
    q: jax.Array, area: jax.Array, fac: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 685: max + min + area-weighted global mean diagnostic.

    Faithful JAX port of FV3 ``prt_mxm`` (tools/fv_diagnostics.F90:
    4118-4161).  Computes::

        qmin = min(q)
        qmax = max(q)
        gmean = area-weighted global mean of q's bottom layer
                (FV3 line 4157: ``q(is:ie, js:je, km)``)
        All scaled by ``fac``.

    Parameters
    ----------
    q : jax.Array, shape (..., n_x, n_y, km)
        3D field.
    area : jax.Array, shape (..., n_x, n_y)
        Cell areas.
    fac : float, default 1.0
        Multiplicative factor applied to output (e.g., unit
        conversion).

    Returns
    -------
    qmin, qmax : jax.Array (scalars)
        Min / max of q over all cells & levels (scaled by fac).
    gmean : jax.Array (scalar)
        Area-weighted global mean of q's bottom layer (k=-1).
    """
    qmin = jnp.min(q) * fac
    qmax = jnp.max(q) * fac
    # FV3 bug-fix line: g_sum on q[..., km] (bottom layer)
    q_bot = q[..., -1]
    total_area = jnp.sum(area)
    safe_area = jnp.where(total_area > 0.0, total_area, 1.0)
    gmean = jnp.sum(q_bot * area) / safe_area * fac
    return qmin, qmax, gmean


def range_check_fv3(
    q: jax.Array, q_low: float, q_hi: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 682: check field stays within [q_low, q_hi] range.

    Faithful JAX port of FV3 ``range_check_3d`` /
    ``range_check_2d`` (tools/fv_diagnostics.F90:3948-4078).
    Unified into one function since the 2D/3D variants are
    identical except for array rank.

    Returns:
        bad_range : bool scalar — True if any q[i, j, k] outside [low, hi]
        qmin, qmax : scalars — min/max of q

    Used by FV3 for diagnostic range sanity (e.g., catch numerical
    blowup before NaN propagation).

    Parameters
    ----------
    q : jax.Array
        Field to check (any shape).
    q_low, q_hi : float
        Valid range bounds.

    Returns
    -------
    bad_range : jax.Array (bool scalar)
    qmin : jax.Array (scalar)
    qmax : jax.Array (scalar)
    """
    qmin = jnp.min(q)
    qmax = jnp.max(q)
    bad_range = (qmin < q_low) | (qmax > q_hi)
    return bad_range, qmin, qmax


def get_height_field_fv3(
    pt: jax.Array, q: jax.Array, peln: jax.Array,
    zsurf: jax.Array,
    delz: jax.Array | None = None,
    zvir: float | None = None,
    hydrostatic: bool = True,
) -> jax.Array:
    """FV3_3D iter 681: geopotential heights at layer interfaces.

    Faithful JAX port of FV3 ``get_height_field``
    (tools/fv_diagnostics.F90:3911-3945).

    Hydrostatic branch:
        wz[km]  = zsurf
        wz[k]   = wz[k+1] + (R_d/g) · pt[k]·(1+zvir·q[k])·(peln[k+1] - peln[k])

    Non-hydrostatic branch:
        wz[km]  = zsurf
        wz[k]   = wz[k+1] - delz[k]    (delz < 0 from FV3 convention)

    Builds wz top-down from surface upward (bottom interface at zsurf).

    Parameters
    ----------
    pt : jax.Array, shape (..., km)
        Temperature (K).
    q : jax.Array, shape (..., km)
        Water-vapor mixing ratio.
    peln : jax.Array, shape (..., km+1)
        Log-pressure at layer interfaces.
    zsurf : jax.Array, shape (...,)
        Surface elevation (m).
    delz : jax.Array, shape (..., km), optional
        Layer thickness (m, FV3 convention delz < 0).  Required if
        hydrostatic=False.
    zvir : float, optional
        Virtual-temperature factor (R_v/R_d - 1).  Default
        ``constants.R_v/constants.R_d - 1``.
    hydrostatic : bool, default True
        If True, use hydrostatic R_d/g log-p formula.

    Returns
    -------
    wz : jax.Array, shape (..., km+1)
        Geopotential heights at layer interfaces (m).
    """
    g = constants.g
    Rdgas = constants.R_d
    if zvir is None:
        zvir = constants.R_v / Rdgas - 1.0
    gg = Rdgas / g

    km = pt.shape[-1]
    # Initialize wz at surface (k=km, 0-indexed)
    # Then build top-down: wz[k] = wz[k+1] + dz_k
    if hydrostatic:
        # dz_k = gg · pt[k] · (1 + zvir·q[k]) · (peln[k+1] - peln[k])
        dz = gg * pt * (1.0 + zvir * q) * (peln[..., 1:] - peln[..., :-1])
    else:
        if delz is None:
            raise ValueError("delz required when hydrostatic=False")
        dz = -delz       # FV3 wz[k] = wz[k+1] - delz[k], delz<0 → dz>0

    # Reverse cumulative sum from bottom up
    # wz[km] = zsurf; wz[k] = zsurf + Σ_{j>=k} dz[j]
    dz_reversed = dz[..., ::-1]                  # bottom layer first
    cumsum_rev = jnp.cumsum(dz_reversed, axis=-1)
    cumsum_rev = cumsum_rev[..., ::-1]           # back to top-down
    # wz has km+1 entries; wz[km] = zsurf
    wz_above_surface = zsurf[..., None] + cumsum_rev
    wz_surface = zsurf[..., None]
    return jnp.concatenate([wz_above_surface, wz_surface], axis=-1)


def interpolate_vertical_fv3(
    a3: jax.Array, peln: jax.Array, plev: jax.Array,
) -> jax.Array:
    """FV3_3D iter 679: linear vertical interpolation to a single p-level.

    Faithful JAX port of FV3 ``interpolate_vertical``
    (tools/fv_diagnostics.F90:4735-4774).

    Algorithm:
        pm[k] = 0.5·(peln[k] + peln[k+1])     # mid-layer log-p
        logp = log(plev)
        if logp <= pm[0]:    a2 = a3[0]        # above top
        elif logp >= pm[-1]: a2 = a3[-1]       # below bottom
        else: linear interp on (pm[k], a3[k]) ↔ (pm[k+1], a3[k+1])

    Pairs with iter-678 ``interpolate_z_fv3`` (z-level variant);
    this is the p-level companion using log-pressure coordinate.

    Used by FV3 for pressure-level diagnostic interpolation
    (winds at 850 hPa, T at 500 hPa, etc.).

    Parameters
    ----------
    a3 : jax.Array, shape (..., km)
        3D field at mid-layer levels.
    peln : jax.Array, shape (..., km+1)
        Layer interface log-pressure, monotonically increasing
        with k (top-down).
    plev : float or jax.Array
        Target pressure level (Pa).

    Returns
    -------
    a2 : jax.Array, shape (...,)
        Field interpolated to ``plev``.
    """
    # pm[k] = 0.5·(peln[k] + peln[k+1])
    pm = 0.5 * (peln[..., :-1] + peln[..., 1:])             # (..., km)
    logp = jnp.log(plev)
    # Search: largest k with pm[k] <= logp; clip to [0, km-2]
    km = pm.shape[-1]
    leq_mask = pm <= logp
    leq_count = jnp.sum(leq_mask.astype(jnp.int32), axis=-1)
    k = jnp.clip(leq_count - 1, 0, km - 2)
    pm_k = jnp.take_along_axis(pm, k[..., None], axis=-1).squeeze(-1)
    pm_kp1 = jnp.take_along_axis(pm, (k + 1)[..., None], axis=-1).squeeze(-1)
    a3_k = jnp.take_along_axis(a3, k[..., None], axis=-1).squeeze(-1)
    a3_kp1 = jnp.take_along_axis(a3, (k + 1)[..., None], axis=-1).squeeze(-1)
    denom = pm_kp1 - pm_k
    safe_denom = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    a2_band = a3_k + (a3_kp1 - a3_k) * (logp - pm_k) / safe_denom
    above_top = logp <= pm[..., 0]
    below_bot = logp >= pm[..., -1]
    a2 = jnp.where(
        above_top, a3[..., 0],
        jnp.where(below_bot, a3[..., -1], a2_band),
    )
    return a2


def interpolate_z_fv3(
    a3: jax.Array, hght: jax.Array, zl: jax.Array,
) -> jax.Array:
    """FV3_3D iter 678: linear vertical interpolation to a single z-level.

    Faithful JAX port of FV3 ``interpolate_z`` (tools/fv_diagnostics.F90:
    4776-4810).

    Algorithm:
        zm[k] = 0.5·(hght[k] + hght[k+1])     # mid-layer height
        if zl >= zm[0]:    a2 = a3[0]         # above top
        elif zl <= zm[-1]: a2 = a3[-1]        # below bottom
        else: linear interp on (zm[k], a3[k]) ↔ (zm[k+1], a3[k+1])

    Note: FV3 ``hght(k) > hght(k+1)`` (decreasing with k = top-down).
    Function interpolates onto a single requested level ``zl``.

    Parameters
    ----------
    a3 : jax.Array, shape (..., km)
        3D field at mid-layer heights.
    hght : jax.Array, shape (..., km+1)
        Layer interface heights, decreasing with k (top-down).
    zl : float or jax.Array
        Target z-level.

    Returns
    -------
    a2 : jax.Array, shape (...,)
        Field interpolated to ``zl``.
    """
    # zm[k] = 0.5·(hght[k] + hght[k+1])
    zm = 0.5 * (hght[..., :-1] + hght[..., 1:])             # (..., km)
    # Search: for each column find k where zm[k] >= zl >= zm[k+1]
    # zm is monotonically decreasing (since hght is).
    # Reformulate: find largest k where zm[k] >= zl; clip to [0, km-2].
    above_mask = zm >= zl
    above_count = jnp.sum(above_mask.astype(jnp.int32), axis=-1)
    km = zm.shape[-1]
    # k_idx = index of last True (in [0, km-1]); k = max(0, above_count - 1)
    # but we need the [k, k+1] pair for interpolation
    k = jnp.clip(above_count - 1, 0, km - 2)
    # Gather zm[k], zm[k+1], a3[k], a3[k+1]
    zm_k = jnp.take_along_axis(zm, k[..., None], axis=-1).squeeze(-1)
    zm_kp1 = jnp.take_along_axis(zm, (k + 1)[..., None], axis=-1).squeeze(-1)
    a3_k = jnp.take_along_axis(a3, k[..., None], axis=-1).squeeze(-1)
    a3_kp1 = jnp.take_along_axis(a3, (k + 1)[..., None], axis=-1).squeeze(-1)
    # Linear interp inside band
    denom = zm_k - zm_kp1
    safe_denom = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    a2_band = a3_k + (a3_kp1 - a3_k) * (zm_k - zl) / safe_denom
    # Boundary cases
    above_top = zl >= zm[..., 0]
    below_bot = zl <= zm[..., -1]
    a2 = jnp.where(
        above_top, a3[..., 0],
        jnp.where(below_bot, a3[..., -1], a2_band),
    )
    return a2


def z_sum_fv3(
    delp: jax.Array, q: jax.Array,
) -> jax.Array:
    """FV3_3D iter 677: column mass-weighted vertical sum.

    Faithful JAX port of FV3 ``z_sum`` (tools/fv_diagnostics.F90:
    4265-4285)::

        sum2[i, j] = Σ_k delp[i, j, k] · q[i, j, k]

    Used by FV3 for column-integrated diagnostics (e.g., total
    water, dry mass).

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Layer pressure thickness (Pa).
    q : jax.Array, shape (..., km)
        Tracer or scalar field.

    Returns
    -------
    sum2 : jax.Array, shape (...,)
        Column mass-weighted sum.
    """
    return jnp.sum(delp * q, axis=-1)


def p_sum_fv3(
    delp: jax.Array, area: jax.Array,
) -> jax.Array:
    """FV3_3D iter 677: global mean of column pressure thickness sum.

    Faithful JAX port of FV3 ``p_sum`` (tools/fv_diagnostics.F90:
    4287-4310, serial branch).  Returns::

        sum2[i, j] = Σ_k delp[i, j, k]                 # column sum
        global_mean = Σ_{ij} sum2 · area / Σ_{ij} area  # area-weighted

    Equivalent to ``g_sum(z_sum(delp, ones), area, mode=1)``.
    Used for global mass-mean diagnostic (mean ps - ptop).

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Layer pressure thickness (Pa).
    area : jax.Array, shape (...,)
        Cell areas (m²).

    Returns
    -------
    p_sum : jax.Array (scalar)
        Area-weighted global mean column pressure thickness.
    """
    col_sum = jnp.sum(delp, axis=-1)
    total_area = jnp.sum(area)
    safe_area = jnp.where(total_area > 0.0, total_area, 1.0)
    return jnp.sum(col_sum * area) / safe_area


def wind_max_fv3(
    us: jax.Array, vs: jax.Array,
    half_window: int = 3,
) -> jax.Array:
    """FV3_3D iter 676: max wind speed in (2·hw+1)² neighborhood.

    Faithful JAX port of FV3 ``wind_max`` (tools/fv_diagnostics.F90:
    3843-3874).  Computes wind speed ``ws = sqrt(us² + vs²)`` then
    takes the maximum over a (2·hw+1) × (2·hw+1) neighborhood
    centered at each cell.  FV3 default ``hw = 3`` → 7×7 window.

    Used by FV3 for storm-tracking / TC max-wind diagnostics.

    Boundary cells (within hw of edge) use a smaller effective
    window (clipped to grid bounds).

    Parameters
    ----------
    us, vs : jax.Array, shape (..., n_x, n_y)
        Surface wind components (m/s).
    half_window : int, default 3
        Half-window size (FV3 uses 3 → 7×7 max-pool).

    Returns
    -------
    ws_max : jax.Array, shape (..., n_x, n_y)
        Maximum wind speed in (2·hw+1)² neighborhood.
    """
    ws = jnp.sqrt(us * us + vs * vs)
    hw = half_window
    # Use jax.lax.reduce_window for efficient max-pool
    from jax import lax
    leading = ws.ndim - 2
    # window_dimensions: 1's for leading axes, (2·hw+1) for (i, j)
    window = (1,) * leading + (2 * hw + 1, 2 * hw + 1)
    strides = (1,) * ws.ndim
    # Pad with -inf so edge cells take max over interior only
    padding = ((0, 0),) * leading + ((hw, hw), (hw, hw))
    ws_padded = jnp.pad(ws, padding, mode="constant", constant_values=-jnp.inf)
    return lax.reduce_window(
        ws_padded, -jnp.inf, lax.max,
        window_dimensions=window, window_strides=strides,
        padding="VALID",
    )


def bilinear_interp_apply(
    src_field: jax.Array,
    id1: jax.Array, id2: jax.Array, jc: jax.Array,
    s2c: jax.Array,
) -> jax.Array:
    """FV3_3D iter 675: apply bilinear remap weights to source field.

    Faithful JAX port of FV3 ``apply_inc_on_3d_scalar`` core
    (tools/fv_treat_da_inc.F90:339-360, inner bilinear loop).

    Algorithm:

        target[..., i, j] = s2c[..., i, j, 0] · src[id1[i, j], jc[i, j]    ]
                          + s2c[..., i, j, 1] · src[id2[i, j], jc[i, j]    ]
                          + s2c[..., i, j, 2] · src[id2[i, j], jc[i, j]+1  ]
                          + s2c[..., i, j, 3] · src[id1[i, j], jc[i, j]+1  ]

    Pairs with iter-673 ``remap_coef_fv3`` (produces id1, id2, jc, s2c)
    to provide full lat-lon → cubed-sphere bilinear interpolation.

    Parameters
    ----------
    src_field : jax.Array, shape (im, jm) or (im, jm, km)
        Source field on regular lat-lon grid.  Trailing axes
        broadcast.
    id1, id2 : jax.Array (int), shape (...,)
        Source longitude indices.
    jc : jax.Array (int), shape (...,)
        Source latitude index (jc and jc+1 are used for bilinear).
    s2c : jax.Array, shape (..., 4)
        Bilinear weights (SW, SE, NE, NW).

    Returns
    -------
    target : jax.Array, shape matches id1 (+ trailing dims of src_field)
    """
    # Gather source values at the 4 corners
    f_sw = src_field[id1, jc]                    # (..., [km])
    f_se = src_field[id2, jc]
    f_ne = src_field[id2, jc + 1]
    f_nw = src_field[id1, jc + 1]
    # Combine
    w_sw = s2c[..., 0]
    w_se = s2c[..., 1]
    w_ne = s2c[..., 2]
    w_nw = s2c[..., 3]
    # Add level-axis broadcast if needed
    if f_sw.ndim > id1.ndim:
        extra = f_sw.ndim - id1.ndim
        for _ in range(extra):
            w_sw = w_sw[..., None]
            w_se = w_se[..., None]
            w_ne = w_ne[..., None]
            w_nw = w_nw[..., None]
    return w_sw * f_sw + w_se * f_se + w_ne * f_ne + w_nw * f_nw


def remap_coef_fv3(
    target_lon: jax.Array, target_lat: jax.Array,
    src_lon: jax.Array, src_lat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 673: bilinear remap weights from regular lat-lon to target.

    Faithful JAX port of FV3 ``remap_coef`` (tools/fv_treat_da_inc.F90:
    366-442).  Computes bilinear interpolation weights for mapping
    a regular lat-lon source grid to arbitrary target positions
    (typically cubed-sphere cell centers).

    Algorithm:
        For each target point (target_lon, target_lat):
            i1 = largest src_lon index ≤ target_lon
            i2 = i1 + 1 (wrap around if at end)
            jc = largest src_lat index ≤ target_lat
            a1 = (target_lon - src_lon[i1]) / (src_lon[i2] - src_lon[i1])
            b1 = (target_lat - src_lat[jc]) / (src_lat[jc+1] - src_lat[jc])
            s2c[..., 0] = (1-a1)·(1-b1)    (SW corner weight)
            s2c[..., 1] =    a1 ·(1-b1)    (SE)
            s2c[..., 2] =    a1 ·   b1     (NE)
            s2c[..., 3] = (1-a1)·   b1     (NW)

    Used for FV3 reading regular lat-lon ICs (ERA5, GFS) and
    interpolating to cubed-sphere grid.

    Parameters
    ----------
    target_lon, target_lat : jax.Array
        Target grid positions (radians), shape (..., n_x, n_y) or
        (n_x, n_y).
    src_lon : jax.Array, shape (im,)
        Source longitude (radians, monotonic, periodic).
    src_lat : jax.Array, shape (jm,)
        Source latitude (radians, monotonic ascending).

    Returns
    -------
    id1, id2 : jax.Array (int)
        Source longitude indices (i1, i2).
    jdc : jax.Array (int)
        Source latitude index (jc).
    s2c : jax.Array, shape (..., 4)
        4 bilinear weights (SW, SE, NE, NW).
    """
    pi = jnp.pi
    im = src_lon.shape[0]
    jm = src_lat.shape[0]
    # Longitude lookup with wrap-around (FV3 lines 394-408)
    # First find i1 = max(i where src_lon[i] <= target_lon); if target_lon <
    # src_lon[0], use wrap (i1 = im-1, i2 = 0); if > src_lon[-1], same.
    in_range = (target_lon >= src_lon[0]) & (target_lon <= src_lon[im - 1])
    # Within range: i1 = searchsorted-1
    i1_in_range = jnp.clip(jnp.searchsorted(src_lon, target_lon, side="right") - 1, 0, im - 2)
    # Out of range: i1 = im-1, i2 = 0 (wrap)
    i1 = jnp.where(in_range, i1_in_range, im - 1)
    i2 = jnp.where(in_range, i1 + 1, 0)
    # a1 computation
    # Wrap case 1: target > src_lon[im-1] → a1 = (target - src_lon[im-1]) / (src_lon[0] + 2π - src_lon[im-1])
    # Wrap case 2: target < src_lon[0] → a1 = (target + 2π - src_lon[im-1]) / (src_lon[0] + 2π - src_lon[im-1])
    # Within range: a1 = (target - src_lon[i1]) / (src_lon[i2] - src_lon[i1])
    rdlon_wrap = 1.0 / (src_lon[0] + 2.0 * pi - src_lon[im - 1])
    a1_wrap_high = (target_lon - src_lon[im - 1]) * rdlon_wrap
    a1_wrap_low = (target_lon + 2.0 * pi - src_lon[im - 1]) * rdlon_wrap
    a1_in_range = (
        (target_lon - src_lon[i1_in_range])
        / (src_lon[i1_in_range + 1] - src_lon[i1_in_range])
    )
    a1 = jnp.where(
        in_range, a1_in_range,
        jnp.where(target_lon > src_lon[im - 1], a1_wrap_high, a1_wrap_low),
    )
    # Latitude lookup (FV3 lines 411-426)
    in_lat_range = (target_lat >= src_lat[0]) & (target_lat <= src_lat[jm - 1])
    jc_in_range = jnp.clip(
        jnp.searchsorted(src_lat, target_lat, side="right") - 1, 0, jm - 2,
    )
    jc = jnp.where(
        in_lat_range, jc_in_range,
        jnp.where(target_lat < src_lat[0], 0, jm - 2),
    )
    b1_in_range = (
        (target_lat - src_lat[jc_in_range])
        / (src_lat[jc_in_range + 1] - src_lat[jc_in_range])
    )
    b1 = jnp.where(
        in_lat_range, b1_in_range,
        jnp.where(target_lat < src_lat[0], 0.0, 1.0),
    )
    # 4 bilinear weights
    s2c = jnp.stack([
        (1.0 - a1) * (1.0 - b1),
        a1 * (1.0 - b1),
        a1 * b1,
        (1.0 - a1) * b1,
    ], axis=-1)
    return i1.astype(jnp.int32), i2.astype(jnp.int32), jc.astype(jnp.int32), s2c


def dcmip16_tc_uwind_pert(
    z: jax.Array, r: jax.Array,
    lon: jax.Array, lat: jax.Array,
    Tv0: float | None = None,
    lapse: float = 7.0e-3,
    zt: float = 15000.0,
    rp: float = 282000.0,
    zp: float = 7000.0,
    pb: float = 101500.0,
    dp: float = 1115.0,
    q0: float = 0.021,
    lamp: float = None,
    phip: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 672: DCMIP16 TC vortex wind perturbation.

    Faithful JAX port of FV3 ``DCMIP16_TC_uwind_pert``
    (tools/test_cases.F90:7168-7197).

    Algorithm (z ≤ zt):
        rfac = (r/rp)^1.5
        fr5  = 0.5·fc·r              # fc = 2·Ω·sin(phip)
        Tvrd = (Tv0 - lapse·z)·R_d
        vt = -fr5 + sqrt(fr5² - 1.5·rfac·Tvrd /
                          (1 + 2·Tvrd·z/(g·zp²) - (pb/dp)·exp(rfac + (z/zp)²)))
        d1  = sin(phip)·cos(lat) - cos(phip)·sin(lat)·cos(lon - lamp)
        d2  = cos(phip)·sin(lon - lamp)
        d   = max(1e-25, sqrt(d1² + d2²))
        uu = vt · d1 / d
        vv = vt · d2 / d
    z > zt: uu = vv = 0

    Default FV3 constants:
        lamp = π (TC center longitude)
        phip = π/18 (TC center latitude, ~10°N)
        Tv0  = 302.15·(1+0.608·q0)
        fc   = 2·Ω·sin(phip)

    Used in FV3 DCMIP16 Test 411 (TC).  With iter-666/669/671:
    full TC IC stack available.

    Returns (uu, vv) wind perturbation components.
    """
    pi = jnp.pi
    if Tv0 is None:
        Tv0 = 302.15 * (1.0 + 0.608 * q0)
    if lamp is None:
        lamp = pi
    if phip is None:
        phip = pi / 18.0
    g = constants.g
    Rdgas = constants.R_d
    omega = constants.Omega
    fc = 2.0 * omega * jnp.sin(jnp.asarray(phip))
    rfac = jnp.sqrt(r / rp) ** 3
    fr5 = 0.5 * fc * r
    Tv = Tv0 - lapse * z
    Tvrd = Tv * Rdgas
    denom = (
        1.0
        + 2.0 * Tvrd * z / (g * zp * zp)
        - (pb / dp) * jnp.exp(rfac + (z / zp) ** 2)
    )
    safe_denom = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    radicand = fr5 ** 2 - (1.5 * rfac * Tvrd) / safe_denom
    vt = -fr5 + jnp.sqrt(jnp.maximum(radicand, 0.0))
    d1 = (
        jnp.sin(phip) * jnp.cos(lat)
        - jnp.cos(phip) * jnp.sin(lat) * jnp.cos(lon - lamp)
    )
    d2 = jnp.cos(phip) * jnp.sin(lon - lamp)
    d = jnp.maximum(1.0e-25, jnp.sqrt(d1 * d1 + d2 * d2))
    uu_below = vt * d1 / d
    vv_below = vt * d2 / d
    uu = jnp.where(z > zt, 0.0, uu_below)
    vv = jnp.where(z > zt, 0.0, vv_below)
    return uu, vv


def dcmip16_tc_temperature(
    z: jax.Array, r: jax.Array,
    Tv0: float | None = None,
    lapse: float = 7.0e-3,
    zt: float = 15000.0,
    rp: float = 282000.0,
    zp: float = 7000.0,
    pb: float = 101500.0,
    dp: float = 1115.0,
    q0: float = 0.021,
) -> jax.Array:
    """FV3_3D iter 669: DCMIP16 TC temperature profile.

    Faithful JAX port of FV3 ``DCMIP16_TC_temperature``
    (tools/test_cases.F90:7137-7152).

    Algorithm:
        z > zt:  T = Tvt = Tv0 - lapse·zt
        else:
          Tv    = Tv0 - lapse·z
          term1 = g·zp²·(1 - (pb/dp)·exp((r/rp)^1.5 + (z/zp)²))
          term2 = 2·R_d·Tv·z
          T     = Tv·(1 + (1/(1 + term2/term1) - 1))

    Used in FV3 DCMIP16 Test 411 (TC).

    Defaults from FV3 lines 6880-6897: Tv0 = 302.15·(1+0.608·q0).

    Parameters
    ----------
    z : jax.Array
        Height (m).
    r : jax.Array
        Great-circle distance from TC center (m).
    Tv0 : float, optional
        Sea-level virtual temperature; default 302.15·(1+0.608·q0).
    lapse, zt, rp, zp, pb, dp, q0 : float
        DCMIP16 TC parameters (see defaults).
    """
    g = constants.g
    Rdgas = constants.R_d
    if Tv0 is None:
        Tv0 = 302.15 * (1.0 + 0.608 * q0)
    Tvt = Tv0 - lapse * zt
    Tv = Tv0 - lapse * z
    rfac = jnp.sqrt(r / rp) ** 3
    term1 = g * zp * zp * (1.0 - (pb / dp) * jnp.exp(rfac + (z / zp) ** 2))
    term2 = 2.0 * Rdgas * Tv * z
    # Safe-divide for term2/term1
    safe_term1 = jnp.where(jnp.abs(term1) > 1e-30, term1, 1.0)
    T_below = Tv + Tv * (1.0 / (1.0 + term2 / safe_term1) - 1.0)
    return jnp.where(z > zt, Tvt, T_below)


def dcmip16_tc_pressure(
    z: jax.Array, r: jax.Array,
    Tv0: float | None = None,
    lapse: float = 7.0e-3,
    zt: float = 15000.0,
    rp: float = 282000.0,
    zp: float = 7000.0,
    pb: float = 101500.0,
    dp: float = 1115.0,
    q0: float = 0.021,
) -> jax.Array:
    """FV3_3D iter 669: DCMIP16 TC pressure profile.

    Faithful JAX port of FV3 ``DCMIP16_TC_pressure``
    (tools/test_cases.F90:7155-7167).

    Algorithm:
        z <= zt:
          p = pb·exp(g/(R_d·lapse)·ln((Tv0-lapse·z)/Tv0))
              - dp·exp(-(r/rp)^1.5 - (z/zp)²)·exp(g/(R_d·lapse)·ln(...))
        z > zt:
          p = ptt·exp(g·(zt-z)/(R_d·Tvt))
          where ptt = pb·(Tvt/Tv0)^(g/R_d/lapse)
    """
    g = constants.g
    Rdgas = constants.R_d
    if Tv0 is None:
        Tv0 = 302.15 * (1.0 + 0.608 * q0)
    Tvt = Tv0 - lapse * zt
    ptt = pb * (Tvt / Tv0) ** (g / (Rdgas * lapse))
    # z <= zt branch
    Tv = Tv0 - lapse * z
    ratio = jnp.maximum(Tv / Tv0, 1e-30)
    p_base = pb * jnp.exp(g / (Rdgas * lapse) * jnp.log(ratio))
    rfac = jnp.sqrt(r / rp) ** 3
    p_below = p_base - dp * jnp.exp(-rfac - (z / zp) ** 2) * jnp.exp(
        g / (Rdgas * lapse) * jnp.log(ratio)
    )
    # z > zt branch
    p_above = ptt * jnp.exp(g * (zt - z) / (Rdgas * Tvt))
    return jnp.where(z <= zt, p_below, p_above)


def dcmip16_tc_sphum(
    z: jax.Array,
    q0: float = 0.021,
    qt: float = 1.0e-11,
    zq1: float = 3000.0,
    zq2: float = 8000.0,
    zt: float = 15000.0,
) -> jax.Array:
    """FV3_3D iter 666: DCMIP16 Reed-Jablonowski TC humidity profile.

    Faithful JAX port of FV3 ``DCMIP16_TC_sphum`` (tools/test_cases.F90:
    7198-7208, DCMIP16 TC test).  Specific humidity (kg/kg) as
    function of height:

        if z >= zt: q = qt   (stratospheric background)
        else:       q = q0 · exp(-z/zq1) · exp(-(z/zq2)²)

    Default DCMIP16 TC constants (FV3 lines 6880-6886):
        q0  = 0.021 kg/kg  (surface peak)
        qt  = 1e-11 kg/kg  (stratospheric)
        zq1 = 3000 m       (exponential decay)
        zq2 = 8000 m       (Gaussian truncation)
        zt  = 15000 m      (tropopause)

    Used in FV3 DCMIP16 idealized tropical-cyclone test.

    Parameters
    ----------
    z : jax.Array
        Height(s) in meters.
    q0, qt, zq1, zq2, zt : float
        TC profile constants (see defaults).

    Returns
    -------
    q : jax.Array
        Specific humidity (kg/kg).
    """
    q_below = q0 * jnp.exp(-z / zq1) * jnp.exp(-(z / zq2) ** 2)
    return jnp.where(z < zt, q_below, qt)


def super_k_u_fv3(
    zz: jax.Array,
    zs: float = 5000.0,
    us: float = 30.0,
    uc: float = 15.0,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 665: super-cell vertical wind-shear profile (MPAS branch).

    Faithful JAX port of FV3 ``SuperK_u`` (tools/test_cases.F90:
    6049-6082, MPAS branch without TEST_TANHP).  Piecewise-cubic
    wind profile for super-cell test cases.

    Algorithm:
        if z > zs + 1km:           um = us;             dudz = 0
        elif |z - zs| ≤ 1km:       um = us·(-4/5 + 3z/zs - 5/4·(z/zs)²)
                                   dudz = us/zs · (3 - 5/2·z/zs)
        else (z < zs - 1km):       um = us·z/zs;         dudz = us/zs
        um -= uc                                         (storm offset)

    Default constants: zs=5 km (shear scale), us=30 m/s (peak
    shear), uc=15 m/s (storm offset for near-stationary storm).

    Used for FV3 super-cell idealized test cases.

    Parameters
    ----------
    zz : jax.Array
        Heights (m).
    zs : float, default 5000
        Shear scale height (m).
    us : float, default 30
        Peak shear wind (m/s).
    uc : float, default 15
        Constant offset wind (m/s).

    Returns
    -------
    um : jax.Array
        Mean wind profile (m/s).
    dudz : jax.Array
        Vertical wind shear (s⁻¹).
    """
    ratio = zz / zs
    # Region 1: z > zs + 1km
    upper = zz > zs + 1.0e3
    um_upper = jnp.full_like(zz, us)
    dudz_upper = jnp.zeros_like(zz)
    # Region 2: |z - zs| ≤ 1km (cubic blend)
    blend = jnp.abs(zz - zs) <= 1.0e3
    um_blend = us * (-4.0 / 5.0 + 3.0 * ratio - 5.0 / 4.0 * ratio ** 2)
    dudz_blend = us / zs * (3.0 - 5.0 / 2.0 * ratio)
    # Region 3: z < zs - 1km (linear)
    um_lower = us * ratio
    dudz_lower = jnp.full_like(zz, us / zs)
    # Compose: upper takes precedence over blend over lower
    um = jnp.where(upper, um_upper,
                   jnp.where(blend, um_blend, um_lower))
    dudz = jnp.where(upper, dudz_upper,
                     jnp.where(blend, dudz_blend, dudz_lower))
    um = um - uc
    return um, dudz


def case9_B(
    lon: jax.Array, lat: jax.Array, gh0: float | None = None,
) -> jax.Array:
    """FV3_3D iter 664: Williamson case 9 spatial forcing pattern B(λ, φ).

    Faithful JAX port of FV3 ``get_case9_B`` (tools/test_cases.F90:
    4361-4389).  Returns::

        if sin(φ) > 0:
            yy = (cos(φ) / sin(φ))² = cot²(φ)
            B  = gh0 · yy · exp(1 - yy) · sin(λ)
        else:
            B = 0

    Default gh0 = 720·g (FV3 calibrated for the SW orographic
    forcing test).  The forcing peaks where yy=1 (i.e., lat=π/4)
    with magnitude gh0·sin(λ).

    Parameters
    ----------
    lon, lat : jax.Array
        Cell-center positions (radians).
    gh0 : float, optional
        Forcing peak amplitude (default 720·g).

    Returns
    -------
    B : jax.Array
        Spatial forcing field.
    """
    if gh0 is None:
        gh0 = 720.0 * constants.g
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    # Safe cot²: avoid divide-by-zero at equator and poles
    safe_sin = jnp.where(jnp.abs(sin_lat) > 1e-30, sin_lat, 1.0)
    yy = (cos_lat / safe_sin) ** 2
    myB = gh0 * yy * jnp.exp(1.0 - yy)
    B = myB * jnp.sin(lon)
    # Zero in southern hemisphere (and equator)
    return jnp.where(sin_lat > 0.0, B, 0.0)


def case9_AofT(
    tday: jax.Array,
) -> jax.Array:
    """FV3_3D iter 664: Williamson case 9 amplitude modulation AofT(t).

    Faithful JAX port of FV3 ``case9_forcing1`` amplitude logic
    (tools/test_cases.F90:4391-4424).  Time-varying amplitude:

        tday ≤ 4:           A = 0.5·(1 - cos(π·tday/4))   [ramp up]
        4 < tday ≤ 16:      A = 1                          [peak]
        16 < tday ≤ 20:     A = 0.5·(1 + cos(π·(tday-16)/4)) [ramp down]
        tday > 20:          A = 0.5·(1 - cos(π·(tday-20)/4)) [new cycle]

    Parameters
    ----------
    tday : jax.Array
        Time in days.

    Returns
    -------
    A : jax.Array
        Amplitude ∈ [0, 1].
    """
    pi = jnp.pi
    ramp_up = 0.5 * (1.0 - jnp.cos(0.25 * pi * tday))
    peak = jnp.ones_like(ramp_up)
    ramp_down = 0.5 * (1.0 + jnp.cos(0.25 * pi * (tday - 16.0)))
    new_cycle = 0.5 * (1.0 - jnp.cos(0.25 * pi * (tday - 20.0)))
    A = jnp.where(
        tday <= 4.0, ramp_up,
        jnp.where(
            tday <= 16.0, peak,
            jnp.where(tday <= 20.0, ramp_down, new_cycle),
        ),
    )
    return A


def u_jet_fv3(
    lat: jax.Array,
    umax: float = 80.0,
    ph0: float | None = None,
    ph1: float | None = None,
) -> jax.Array:
    """FV3_3D iter 663: Galewsky-like zonal jet profile.

    Faithful JAX port of FV3 ``u_jet`` (tools/test_cases.F90:
    4349-4360).  Returns jet zonal wind::

        if ph0 < lat < ph1:
            u_jet = (umax / en) · exp(1 / ((lat - ph0)·(lat - ph1)))
        else:
            u_jet = 0
        where en = exp(-4 / (ph1 - ph0)²)

    Default ph0 = π/7, ph1 = π/2 - π/7 (northern hemisphere jet).

    Parameters
    ----------
    lat : jax.Array
        Latitude in radians.
    umax : float, default 80.0
        Peak zonal wind (m/s).
    ph0, ph1 : float, optional
        Jet boundaries (default FV3 values).
    """
    if ph0 is None:
        ph0 = jnp.pi / 7.0
    if ph1 is None:
        ph1 = jnp.pi / 2.0 - jnp.pi / 7.0
    en = jnp.exp(-4.0 / (ph1 - ph0) ** 2)
    # Safe-divide inside jet band; clamp the argument to avoid -inf
    in_band = (lat > ph0) & (lat < ph1)
    denom = (lat - ph0) * (lat - ph1)
    safe_denom = jnp.where(in_band, denom, -1.0)        # nonzero negative
    profile = (umax / en) * jnp.exp(1.0 / safe_denom)
    return jnp.where(in_band, profile, 0.0)


def gh_jet_fv3(
    lat_in: jax.Array,
    npy: int = 64,
    h0: float = 10157.946867,
    umax: float = 80.0,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
) -> jax.Array:
    """FV3_3D iter 663: geopotential height of Galewsky-like jet.

    Faithful JAX port of FV3 ``gh_jet`` (tools/test_cases.F90:
    4297-4348).  Returns geopotential ``g·h`` along the Galewsky
    barotropic-instability test jet by numerical integration of
    geostrophic + centripetal balance::

        gh[0] = g·h0    (at lat=-π/2)
        gh[j] = gh[j-1] - u·(R·f + tan(lat)·u)·dp

    where ``u = u_jet(lat)``, ``f = 2·Ω·sin(lat)``, ``dp = π/(jm-1)``,
    ``jm = 4·npy``.  Then linear-interp at ``lat_in``.

    Used for Galewsky barotropic-instability test (Galewsky,
    Scott & Polvani 2004) on the sphere.

    Parameters
    ----------
    lat_in : jax.Array
        Latitudes at which to evaluate gh (radians).
    npy : int, default 64
        Reference grid resolution (table size = 4·npy).
    h0 : float, default 10157.946867 (FV3 calibrated)
        South-pole reference height (m).
    umax : float, default 80.0
        Peak zonal wind for ``u_jet`` (m/s).
    radius : float, default constants.R_earth
        Sphere radius.
    omega : float, default constants.omega_earth
        Sphere angular velocity.
    """
    g = constants.g
    jm = 4 * npy
    dp = jnp.pi / (jm - 1)
    # Latitudes at midpoints for integration (FV3 lines 4321-4326)
    j_idx = jnp.arange(2, jm + 1)
    lat_mid = -jnp.pi / 2.0 + (j_idx.astype(jnp.float64) - 1.0 - 0.5) * dp
    uu = u_jet_fv3(lat_mid, umax=umax)
    ft = 2.0 * omega * jnp.sin(lat_mid)
    # increment: -uu·(R·f + tan(lat_mid)·uu)·dp
    increment = -uu * (radius * ft + jnp.tan(lat_mid) * uu) * dp
    # Build gh_table: gh[0] = g·h0; gh[j] = gh[j-1] + increment[j-1]
    gh_inits = jnp.asarray([g * h0])
    gh_rest = gh_inits[0] + jnp.cumsum(increment)
    gh_table = jnp.concatenate([gh_inits, gh_rest])
    # Latitudes table (FV3 line 4326): lat[j] = -π/2 + (j-1)·dp
    j_idx_all = jnp.arange(jm, dtype=jnp.float64)
    lats_table = -jnp.pi / 2.0 + j_idx_all * dp
    # Linear interpolation of gh_table at lat_in
    return jnp.interp(lat_in, lats_table, gh_table)


def add_rankine_vortex(
    u: jax.Array, v: jax.Array,
    grid_lon: jax.Array, grid_lat: jax.Array,
    ubar: float, r0: float,
    center_lon: float, center_lat: float,
    radius: float = constants.R_earth,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 662: add Rankine vortex to D-grid winds.

    Faithful JAX port of FV3 ``rankine_vortex``
    (tools/test_cases.F90:4207-4292).  Adds a Rankine-vortex
    tangential wind onto the D-grid u, v fields at face corners,
    projected onto the local cube-grid tangent vectors.

    Tangential wind profile:
        vr = ubar · r/r0    if r < r0  (solid-body core)
        vr = ubar · r0/r    if r ≥ r0  (1/r decay outside)

    where r is great-circle distance from cell-edge midpoint to
    vortex center (radius·acos(cos_p)).

    Parameters
    ----------
    u : jax.Array, shape (..., n_x, n_y+1)
        D-grid u-wind (north/south edges); updated in-place style.
    v : jax.Array, shape (..., n_x+1, n_y)
        D-grid v-wind (east/west edges).
    grid_lon : jax.Array, shape (..., n_x+1, n_y+1)
        Cubed-sphere corner longitudes.
    grid_lat : jax.Array, shape (..., n_x+1, n_y+1)
        Cubed-sphere corner latitudes.
    ubar : float
        Maximum tangential wind (m/s).
    r0 : float
        Radius of maximum wind (m).
    center_lon, center_lat : float
        Vortex center (radians).
    radius : float, default constants.R_earth
        Sphere radius (m).

    Returns
    -------
    u_new, v_new : jax.Array
        D-grid winds with vortex added.
    """
    pi = jnp.pi

    def _tangential_wind_at(p2_lon, p2_lat):
        """Compute vortex contributions (utmp, vtmp) at point p2."""
        # Shift p2_lon by -center_lon
        p2_lon_s = p2_lon - center_lon
        cos_p = (
            jnp.sin(p2_lat) * jnp.sin(center_lat)
            + jnp.cos(p2_lat) * jnp.cos(center_lat) * jnp.cos(p2_lon_s)
        )
        cos_p = jnp.clip(cos_p, -1.0, 1.0)
        r = radius * jnp.arccos(cos_p)
        # Tangential wind magnitude
        vr_inside = ubar * r / r0
        vr_outside = ubar * r0 / jnp.maximum(r, 1e-30)
        vr = jnp.where(r < r0, vr_inside, vr_outside)
        # Direction of vortex motion (in shifted frame)
        x1 = jnp.cos(p2_lat) * jnp.sin(p2_lon_s)
        y1 = (
            jnp.sin(p2_lat) * jnp.cos(center_lat)
            - jnp.cos(p2_lat) * jnp.sin(center_lat) * jnp.cos(p2_lon_s)
        )
        d2 = jnp.maximum(jnp.sqrt(x1 * x1 + y1 * y1), 1.0e-25)
        utmp = -vr * y1 / d2
        vtmp = vr * x1 / d2
        # Return utmp, vtmp + shifted p2 for elon/elat
        return utmp, vtmp, p2_lon_s, p2_lat

    # ---- u-wind on j-edges: average grid[i, j] and grid[i+1, j] in lon
    # u shape (..., n_x, n_y+1); grid shape (..., n_x+1, n_y+1)
    # j-edge midpoint p2[i, j] = mid_pt_sphere(grid[i, j], grid[i+1, j])
    sw_lon = grid_lon[..., :-1, :]   # (..., n_x, n_y+1)
    sw_lat = grid_lat[..., :-1, :]
    se_lon = grid_lon[..., 1:, :]
    se_lat = grid_lat[..., 1:, :]
    p2_lon_u, p2_lat_u = mid_pt_sphere(sw_lon, sw_lat, se_lon, se_lat)
    utmp_u, vtmp_u, p2_lon_s, p2_lat_s = _tangential_wind_at(p2_lon_u, p2_lat_u)
    # Cube tangent e1 at p2 from p3=(grid[i,j]-center, grid[i,j].lat)
    # to p4=(grid[i+1,j]-center, grid[i+1,j].lat)
    e1 = get_unit_vect2(
        sw_lon - center_lon, sw_lat,
        se_lon - center_lon, se_lat,
    )
    elon_u, elat_u = unit_vect_latlon(p2_lon_s, p2_lat_s)
    u_add = utmp_u * inner_prod(e1, elon_u) + vtmp_u * inner_prod(e1, elat_u)
    u_new = u + u_add

    # ---- v-wind on i-edges: average grid[i, j] and grid[i, j+1] in lat
    # v shape (..., n_x+1, n_y); grid shape (..., n_x+1, n_y+1)
    s_lon = grid_lon[..., :, :-1]
    s_lat = grid_lat[..., :, :-1]
    n_lon = grid_lon[..., :, 1:]
    n_lat = grid_lat[..., :, 1:]
    p2_lon_v, p2_lat_v = mid_pt_sphere(s_lon, s_lat, n_lon, n_lat)
    utmp_v, vtmp_v, p2_lon_s2, p2_lat_s2 = _tangential_wind_at(p2_lon_v, p2_lat_v)
    e2 = get_unit_vect2(
        s_lon - center_lon, s_lat,
        n_lon - center_lon, n_lat,
    )
    elon_v, elat_v = unit_vect_latlon(p2_lon_s2, p2_lat_s2)
    v_add = utmp_v * inner_prod(e2, elon_v) + vtmp_v * inner_prod(e2, elat_v)
    v_new = v + v_add
    return u_new, v_new


def project_sphere_v(
    f: jax.Array, e: jax.Array,
) -> jax.Array:
    """FV3_3D iter 659: project vector onto sphere-tangent plane.

    Faithful JAX port of FV3 ``project_sphere_v``
    (fv_grid_utils.F90:3345-3361).  Given a unit-sphere position
    ``e`` and a 3-vector ``f``, returns ``f`` projected onto the
    tangent plane at ``e``::

        ap = f · e
        f_tangent = f - ap·e

    Takes the last axis as the 3-vector component; broadcasts on
    leading axes.
    """
    ap = jnp.sum(f * e, axis=-1, keepdims=True)
    return f - ap * e


def get_unit_vector_fv3(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
) -> jax.Array:
    """FV3_3D iter 659: unit tangent vector at p2 from p1 → p3.

    Faithful JAX port of FV3 ``get_unit_vector``
    (tools/test_cases.F90:8366-8385).  Algorithm:

        xyz1, xyz2, xyz3 = latlon2xyz(...)
        uvect = xyz3 - xyz1                     # chord
        uvect = project_sphere_v(uvect, xyz2)   # tangent at p2
        uvect = normalize(uvect)

    Returns the unit tangent vector at p2 pointing in the
    direction from p1 toward p3 (projected onto the local
    tangent plane).

    Differs from iter-611 ``get_unit_vect2`` (great-circle
    midpoint tangent) and iter-615 ``get_unit_vect3`` (Cartesian
    variant of get_unit_vect2): this one is the chord-based
    projection at an arbitrary third point.

    Returns shape ``(..., 3)``; broadcasts on leading axes.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x3, y3, z3 = latlon2xyz(lon3, lat3)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    uvect_raw = jnp.stack([x3 - x1, y3 - y1, z3 - z1], axis=-1)
    uvect_tangent = project_sphere_v(uvect_raw, p2)
    return normalize_vect(uvect_tangent)


def terminator_tracers(
    lon: jax.Array, lat: jax.Array,
    km: int,
    qcly: float = 4.0e-6,
    k2: float = 1.0,
    lc: float | None = None,
    thc: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 658: DCMIP 2016 terminator chemistry tracer IC.

    Faithful JAX port of FV3 ``terminator_tracers``
    (tools/test_cases.F90:4136-4205).  DCMIP 2016 idealized
    chemistry test (Lauritzen et al.); paired Cl / Cl2 tracers
    that exchange under photolysis at a localized "sun":

        k1   = max(0, sin(lat)·sin(thc) + cos(lat)·cos(thc)·cos(lon - lc))
        r    = k1/k2 · 0.25
        D    = sqrt(r² + 2·r·qcly)
        Cl   = D - r
        Cl2  = 0.5·(qcly - Cl)

    Same pattern at every vertical level.  Assumes DRY mixing
    ratio (FV3 docstring note).

    Default sun position lc=5π/3, thc=π/9 matches FV3.

    Parameters
    ----------
    lon, lat : jax.Array, shape (..., n_x, n_y)
        Cell-center positions in radians.
    km : int
        Number of vertical levels.
    qcly : float, default 4e-6
        Total chlorine family mixing ratio (kg/kg, DRY).
    k2 : float, default 1.0
        Recombination rate constant.
    lc, thc : float, optional
        Sun position (radians); default FV3 values.

    Returns
    -------
    Cl, Cl2 : jax.Array, shape (..., n_x, n_y, km)
        Chemical species mixing ratios.
    """
    if lc is None:
        lc = 5.0 * jnp.pi / 3.0
    if thc is None:
        thc = jnp.pi / 9.0
    sinthc = jnp.sin(thc)
    costhc = jnp.cos(thc)
    cos_phot = (
        jnp.sin(lat) * sinthc
        + jnp.cos(lat) * costhc * jnp.cos(lon - lc)
    )
    k1 = jnp.maximum(0.0, cos_phot)
    r = k1 / k2 * 0.25
    D = jnp.sqrt(r * r + 2.0 * r * qcly)
    Cl_2d = D - r
    Cl2_2d = 0.5 * (qcly - Cl_2d)
    # Broadcast over km
    Cl = jnp.broadcast_to(Cl_2d[..., None], Cl_2d.shape + (km,))
    Cl2 = jnp.broadcast_to(Cl2_2d[..., None], Cl2_2d.shape + (km,))
    return Cl, Cl2


def checker_tracers(
    lon: jax.Array, lat: jax.Array,
    nq: int, km: int,
    nx: float = 9.0, ny: float = 9.0,
    rn: float | None = None,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 657: checkerboard tracer pattern with optional noise.

    Faithful JAX port of FV3 ``checker_tracers`` (tools/test_cases.F90:
    4067-4135).  Builds a checkerboard tracer pattern based on::

        qt[i, j] = 0.01  if sin(nx·lon)·sin(ny·lat) > 0
                   0     otherwise

    Defaults nx=ny=9 give 20°×20° checker boxes (per FV3 docstring).
    Optional ``rn`` adds uniform random perturbation rn·U(0,1).
    Broadcast across vertical levels (km) and tracer count (nq).

    Coded for the HIWPP benchmark by S.-J. Lin (2014).

    Parameters
    ----------
    lon, lat : jax.Array, shape (..., n_x, n_y)
        Cell-center positions in radians.
    nq : int
        Number of tracers.
    km : int
        Number of vertical levels.
    nx, ny : float, default 9.0
        East-west / North-south wave numbers.
    rn : float, optional
        Magnitude of random perturbation (FV3 suggests 0.1).
    rng_key : jax.Array, optional
        JAX PRNG key required if ``rn is not None``.

    Returns
    -------
    q : jax.Array, shape (..., n_x, n_y, km, nq)
        Tracer field.
    """
    qt = jnp.where(
        jnp.sin(nx * lon) * jnp.sin(ny * lat) > 0.0,
        0.01,
        0.0,
    )
    # Broadcast to (..., n_x, n_y, km, nq)
    q = jnp.broadcast_to(qt[..., None, None], qt.shape + (km, nq))
    if rn is not None:
        if rng_key is None:
            raise ValueError("rng_key required when rn is not None")
        noise = rn * jax.random.uniform(rng_key, q.shape)
        q = q + noise
    return q


def get_vorticity_fv3(
    u: jax.Array, v: jax.Array,
    dx: jax.Array, dy: jax.Array,
    rarea: jax.Array,
) -> jax.Array:
    """FV3_3D iter 656: compute vorticity from D-grid winds.

    Faithful JAX port of FV3 ``get_vorticity`` (tools/test_cases.F90:
    4034-4065).  Standard line-integral / cell-area form:

        utmp[i, j] = u[i, j] · dx[i, j]      # u-circulation
        vtmp[i, j] = v[i, j] · dy[i, j]      # v-circulation
        vort[i, j] = rarea[i, j] · (utmp[i, j] - utmp[i, j+1]
                                    - vtmp[i, j] + vtmp[i+1, j])

    Sign convention: positive = counterclockwise (FV3 vorticity).
    Computes the curl of the D-grid covariant velocity field
    integrated around each cell.

    Parameters
    ----------
    u : jax.Array, shape (n_x, n_y+1, [nlev])
        D-grid u (north/south edges, covariant).
    v : jax.Array, shape (n_x+1, n_y, [nlev])
        D-grid v (east/west edges, covariant).
    dx : jax.Array, shape (n_x, n_y+1)
        Edge x-lengths.
    dy : jax.Array, shape (n_x+1, n_y)
        Edge y-lengths.
    rarea : jax.Array, shape (n_x, n_y)
        Reciprocal cell area (1/m²).

    Returns
    -------
    vort : jax.Array, shape (n_x, n_y, [nlev])
        Cell-center vorticity.
    """
    has_level = u.ndim == dx.ndim + 1
    if has_level:
        dx_b = dx[..., None]
        dy_b = dy[..., None]
        rarea_b = rarea[..., None]
    else:
        dx_b, dy_b, rarea_b = dx, dy, rarea

    utmp = u * dx_b      # (n_x, n_y+1, [nlev])
    vtmp = v * dy_b      # (n_x+1, n_y, [nlev])
    # Slicing axes:
    #   utmp[i, j]   = utmp[..., :, :-1] (or :-1 last axis if 2D)
    #   utmp[i, j+1] = utmp[..., :, 1:]
    #   vtmp[i, j]   = vtmp[..., :-1, :]
    #   vtmp[i+1, j] = vtmp[..., 1:, :]
    if has_level:
        u_j = utmp[..., :, :-1, :]
        u_jp1 = utmp[..., :, 1:, :]
        v_i = vtmp[..., :-1, :, :]
        v_ip1 = vtmp[..., 1:, :, :]
    else:
        u_j = utmp[..., :, :-1]
        u_jp1 = utmp[..., :, 1:]
        v_i = vtmp[..., :-1, :]
        v_ip1 = vtmp[..., 1:, :]

    return rarea_b * (u_j - u_jp1 - v_i + v_ip1)


def atod_vort_on(
    uin: jax.Array, vin: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 655: A-grid → D-grid winds (circulation-conserving).

    Analog of FV3 ``atod`` (tools/test_cases.F90:7833-7892) using
    the circulation-conserving formula consistent with iter-652
    ``dtoa_vort_on`` (the inverse mapping).  FV3's source uses
    ``interp_left_edge_1d`` (interpOrder-dependent); the
    circulation-conserving variant is::

        uout[i, j] = (uin[i, j-1]·dya[i, j-1] + uin[i, j]·dya[i, j])
                    / (dya[i, j-1] + dya[i, j])
        vout[i, j] = (vin[i-1, j]·dxa[i-1, j] + vin[i, j]·dxa[i, j])
                    / (dxa[i-1, j] + dxa[i, j])

    D-grid u lives on north/south edges (n_x, n_y+1); D-grid v
    on east/west edges (n_x+1, n_y).  Interior edges only;
    boundary edges (uout[:, 0], uout[:, -1], vout[0, :], vout[-1, :])
    zero-initialized (FV3 fills via halo).

    Pairs with iter-652 ``dtoa_vort_on`` (D→A) for round-trip
    A-grid ↔ D-grid via circulation-conserving averages.

    Parameters
    ----------
    uin, vin : jax.Array, shape (..., n_x, n_y)
        A-grid wind components.
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout : jax.Array, shape (..., n_x, n_y+1)
        D-grid u (north/south edges).
    vout : jax.Array, shape (..., n_x+1, n_y)
        D-grid v (east/west edges).
    """
    n_x = uin.shape[-2]
    n_y = uin.shape[-1]
    leading_shape = uin.shape[:-2]

    # uout: average A-grid uin along j (n_y → n_y-1 interior edges)
    interior_u = (
        (uin[..., :, :-1] * dya[..., :, :-1]
         + uin[..., :, 1:] * dya[..., :, 1:])
        / (dya[..., :, :-1] + dya[..., :, 1:])
    )  # shape (..., n_x, n_y-1)
    uout = jnp.zeros(leading_shape + (n_x, n_y + 1))
    uout = uout.at[..., :, 1:n_y].set(interior_u)

    # vout: average A-grid vin along i (n_x → n_x-1 interior edges)
    interior_v = (
        (vin[..., :-1, :] * dxa[..., :-1, :]
         + vin[..., 1:, :] * dxa[..., 1:, :])
        / (dxa[..., :-1, :] + dxa[..., 1:, :])
    )  # shape (..., n_x-1, n_y)
    vout = jnp.zeros(leading_shape + (n_x + 1, n_y))
    vout = vout.at[..., 1:n_x, :].set(interior_v)
    return uout, vout


def atoc_vort_on(
    uin: jax.Array, vin: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 654: A-grid → C-grid winds (circulation-conserving).

    Faithful JAX port of FV3 ``atoc`` (tools/test_cases.F90:7965-
    8112, ``VORT_ON`` branch, no ``ALT_INTERP``).

    Algorithm (interior C-grid edges only):

        uout[i, j] = (uin[i, j]·dxa[i, j] + uin[i-1, j]·dxa[i-1, j])
                    / (dxa[i, j] + dxa[i-1, j])
        vout[i, j] = (vin[i, j]·dya[i, j] + vin[i, j-1]·dya[i, j-1])
                    / (dya[i, j] + dya[i, j-1])

    Interior edges only (FV3 ``i ∈ [isd+1, ied]``, ``j ∈ [jsd+1, jed]``).
    Boundary edges (uout[0, :] / uout[-1, :] / vout[:, 0] / vout[:, -1])
    are zero-initialized; FV3 sets them via halo communication or
    fill_corners afterward.

    Parameters
    ----------
    uin : jax.Array, shape (..., n_x, n_y)
        A-grid u (cell-center).
    vin : jax.Array, shape (..., n_x, n_y)
        A-grid v (cell-center).
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout : jax.Array, shape (..., n_x+1, n_y)
        C-grid u (east/west edges).  Boundary edges = 0.
    vout : jax.Array, shape (..., n_x, n_y+1)
        C-grid v (north/south edges).  Boundary edges = 0.
    """
    # Build C-grid uout via vectorized average of adjacent A-grid columns.
    # uout[i, j] for i ∈ [1, n_x-1] uses uin[i-1, j] and uin[i, j].
    interior_u = (
        (uin[..., 1:, :] * dxa[..., 1:, :]
         + uin[..., :-1, :] * dxa[..., :-1, :])
        / (dxa[..., 1:, :] + dxa[..., :-1, :])
    )  # shape (..., n_x-1, n_y)
    n_x = uin.shape[-2]
    n_y = uin.shape[-1]
    # Allocate full uout (..., n_x+1, n_y) with zeros and fill interior
    leading_shape = uin.shape[:-2]
    uout = jnp.zeros(leading_shape + (n_x + 1, n_y))
    uout = uout.at[..., 1:n_x, :].set(interior_u)

    interior_v = (
        (vin[..., :, 1:] * dya[..., :, 1:]
         + vin[..., :, :-1] * dya[..., :, :-1])
        / (dya[..., :, 1:] + dya[..., :, :-1])
    )  # shape (..., n_x, n_y-1)
    vout = jnp.zeros(leading_shape + (n_x, n_y + 1))
    vout = vout.at[..., :, 1:n_y].set(interior_v)
    return uout, vout


def ctoa_vort_on(
    uin: jax.Array, vin: jax.Array,
    dx: jax.Array, dy: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 653: C-grid → A-grid winds (circulation-conserving).

    Faithful JAX port of FV3 ``ctoa`` (tools/test_cases.F90:8114-
    8174, simple/circulation-conserving branch — the
    commented-out FV3 lines 8147-8157).  Note C-grid u is on
    east/west edges (shape (n_x+1, n_y), opposite of D-grid u);
    C-grid v is on north/south edges (shape (n_x, n_y+1)).

    Algorithm:

        uout[i, j] = 0.5·(uin[i, j]·dy[i, j] + uin[i+1, j]·dy[i+1, j])
                       / dya[i, j]
        vout[i, j] = 0.5·(vin[i, j]·dx[i, j] + vin[i, j+1]·dx[i, j+1])
                       / dxa[i, j]

    Mirror of iter-652 ``dtoa_vort_on`` with input axes swapped
    (C-grid puts u on east/west edges; D-grid puts u on north/
    south edges).

    Parameters
    ----------
    uin : jax.Array, shape (..., n_x+1, n_y)
        C-grid u (east/west edges).
    vin : jax.Array, shape (..., n_x, n_y+1)
        C-grid v (north/south edges).
    dx, dy : jax.Array, shape (..., n_x, n_y+1) and (..., n_x+1, n_y)
        Edge lengths.
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout, vout : jax.Array, shape (..., n_x, n_y)
        A-grid cell-center wind components (covariant).
    """
    # uout: average uin·dy along i (axis -2 of uin)
    uout = 0.5 * (
        uin[..., :-1, :] * dy[..., :-1, :]
        + uin[..., 1:, :] * dy[..., 1:, :]
    ) / dya
    # vout: average vin·dx along j (axis -1 of vin)
    vout = 0.5 * (
        vin[..., :, :-1] * dx[..., :, :-1]
        + vin[..., :, 1:] * dx[..., :, 1:]
    ) / dxa
    return uout, vout


def dtoa_vort_on(
    uin: jax.Array, vin: jax.Array,
    dx: jax.Array, dy: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 652: D-grid → A-grid winds (circulation-conserving).

    Faithful JAX port of FV3 ``dtoa`` (tools/test_cases.F90:
    7896-7955, ``VORT_ON`` branch).  Circulation- (vorticity-)
    conserving interpolation from D-grid covariant winds to
    A-grid cell-center winds::

        uout[i, j] = 0.5·(uin[i, j]·dx[i, j] + uin[i, j+1]·dx[i, j+1])
                       / dxa[i, j]
        vout[i, j] = 0.5·(vin[i, j]·dy[i, j] + vin[i+1, j]·dy[i+1, j])
                       / dya[i, j]

    Used by FV3 test-case diagnostics + visualizations.  Differs
    from iter-627 ``c2l_ord2_fv3`` (which applies the a-matrix
    rotation); this is the raw covariant→cell-center step.

    Parameters
    ----------
    uin : jax.Array, shape (..., n_x, n_y+1)
        D-grid u (north/south edges, covariant).
    vin : jax.Array, shape (..., n_x+1, n_y)
        D-grid v (east/west edges, covariant).
    dx, dy : jax.Array, shape (..., n_x, n_y+1) and (..., n_x+1, n_y)
        Edge lengths (matching uin, vin shapes).
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout, vout : jax.Array, shape (..., n_x, n_y)
        A-grid cell-center wind components (covariant).
    """
    # uout: average uin·dx along j (axis -1 of uin)
    uout = 0.5 * (
        uin[..., :, :-1] * dx[..., :, :-1]
        + uin[..., :, 1:] * dx[..., :, 1:]
    ) / dxa
    # vout: average vin·dy along i (axis -2 of vin)
    vout = 0.5 * (
        vin[..., :-1, :] * dy[..., :-1, :]
        + vin[..., 1:, :] * dy[..., 1:, :]
    ) / dya
    return uout, vout


def get_pt_on_great_circle(
    lon1: jax.Array, lat1: jax.Array,
    dist: jax.Array, heading: jax.Array,
    radius: float = constants.R_earth,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 651: point on great circle at given distance + heading.

    Faithful JAX port of FV3 ``get_pt_on_great_circle``
    (tools/test_cases.F90:4805-4826).  Given a start point
    (lon1, lat1), great-circle distance ``dist``, and initial
    heading (radians clockwise from north), returns the target
    point (lon3, lat3) on the same great circle.

    Algorithm:
        pha = dist / radius                            # angular dist
        lat3 = asin(cos(heading)·cos(lat1)·sin(pha)
                    + sin(lat1)·cos(pha))
        dp   = atan2(sin(heading)·sin(pha)·cos(lat1),
                     cos(pha) - sin(lat1)·sin(lat3))
        lon3 = ((lon1 - π) - dp + π) mod 2π            # FV3 0-2π
                                                       # wrap

    Used in FV3 for tropical-cyclone test cases (placing vortex
    along a path) and spherical-trajectory computations.

    Parameters
    ----------
    lon1, lat1 : jax.Array
        Start point (radians).  Broadcasting on leading axes
        supported.
    dist : jax.Array
        Great-circle distance from start (meters; same units as
        ``radius``).
    heading : jax.Array
        Initial heading at start (radians; 0 = north, π/2 = east).
    radius : float, default constants.R_earth

    Returns
    -------
    lon3, lat3 : jax.Array
        Target point on great circle (radians).  lon3 wrapped to
        [0, 2π).
    """
    pha = dist / radius
    sin_pha = jnp.sin(pha)
    cos_pha = jnp.cos(pha)
    sin_lat1 = jnp.sin(lat1)
    cos_lat1 = jnp.cos(lat1)
    sin_heading = jnp.sin(heading)
    cos_heading = jnp.cos(heading)
    lat3 = jnp.arcsin(jnp.clip(
        cos_heading * cos_lat1 * sin_pha + sin_lat1 * cos_pha,
        -1.0, 1.0,
    ))
    dp = jnp.arctan2(
        sin_heading * sin_pha * cos_lat1,
        cos_pha - sin_lat1 * jnp.sin(lat3),
    )
    two_pi = 2.0 * jnp.pi
    lon3 = jnp.mod((lon1 - jnp.pi) - dp + jnp.pi, two_pi)
    return lon3, lat3


def grid_area_fv3(
    grid_lon: jax.Array, grid_lat: jax.Array,
    radius: float = constants.R_earth,
) -> jax.Array:
    """FV3_3D iter 649: 2D vectorized cell-area computation via FV3 get_area.

    Faithful JAX port of FV3 ``grid_area`` (tools/fv_grid_tools.F90:
    2512-2620, spherical-excess branch).  Computes cell areas
    over a 2D corner grid using iter-614 ``get_area``:

        For each cell (i, j) in [0, n_x-1] × [0, n_y-1]:
            p_lL = grid[i,   j  ]  # SW corner
            p_uL = grid[i,   j+1]  # NW
            p_lR = grid[i+1, j  ]  # SE
            p_uR = grid[i+1, j+1]  # NE
            area[i, j] = get_area(p_lL, p_uL, p_lR, p_uR, radius)

    Parameters
    ----------
    grid_lon : jax.Array, shape ``(..., n_x+1, n_y+1)``
        Corner-grid longitudes.
    grid_lat : jax.Array, shape ``(..., n_x+1, n_y+1)``
        Corner-grid latitudes.
    radius : float, default constants.R_earth

    Returns
    -------
    area : jax.Array, shape ``(..., n_x, n_y)``
        Cell areas (m²).  Reuses iter-614 ``get_area``'s
        spherical-excess Gauss-Bonnet formula.
    """
    # Slice 4 corners; pass to vectorized iter-614 get_area
    sw_lon = grid_lon[..., :-1, :-1]; sw_lat = grid_lat[..., :-1, :-1]
    se_lon = grid_lon[..., 1:, :-1];  se_lat = grid_lat[..., 1:, :-1]
    ne_lon = grid_lon[..., 1:, 1:];   ne_lat = grid_lat[..., 1:, 1:]
    nw_lon = grid_lon[..., :-1, 1:];  nw_lat = grid_lat[..., :-1, 1:]
    return get_area(
        sw_lon, sw_lat, se_lon, se_lat,
        ne_lon, ne_lat, nw_lon, nw_lat,
        radius=radius,
    )


def cartesian_to_spherical_fv3(
    x: jax.Array, y: jax.Array, z: jax.Array,
    eps: float = 1.0e-10,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 648: Cartesian (x, y, z) → (lon, lat, r) with FV3 pole branch.

    Faithful JAX port of FV3 ``cartesian_to_spherical``
    (tools/fv_grid_tools.F90:2373-2390).  Differs from iter-611
    ``xyz2latlon`` (which is normalized to unit sphere) by
    returning the actual radius ``r = sqrt(x²+y²+z²)`` and not
    normalizing inputs.

    FV3 pole branch: ``|x|+|y| < 1e-10`` → lon=0.

    Returns
    -------
    lon : jax.Array
        Longitude in [-π, π] (FV3 ATAN2 range, NOT wrapped to [0, 2π)).
    lat : jax.Array
        Latitude (RIGHT_HAND branch: asin(z/r)).
    r : jax.Array
        Radius.
    """
    r = jnp.sqrt(x * x + y * y + z * z)
    safe_r = jnp.where(r > 0.0, r, 1.0)
    near_pole = (jnp.abs(x) + jnp.abs(y)) < eps
    lon = jnp.where(near_pole, 0.0, jnp.arctan2(y, x))
    lat = jnp.arcsin(jnp.clip(z / safe_r, -1.0, 1.0))
    return lon, lat, r


def spherical_to_cartesian_fv3(
    lon: jax.Array, lat: jax.Array, r: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 648: (lon, lat, r) → Cartesian (x, y, z), radius-aware.

    Faithful JAX port of FV3 ``spherical_to_cartesian``
    (tools/fv_grid_tools.F90:2391-2402, RIGHT_HAND branch):

        x = r·cos(lon)·cos(lat)
        y = r·sin(lon)·cos(lat)
        z = r·sin(lat)

    Differs from iter-611 ``latlon2xyz`` by:
        - Output scaled by ``r`` (legoESM iter-611 returns unit-sphere).
        - Lon in any range (FV3 doesn't restrict).
    """
    cos_lat = jnp.cos(lat)
    x = r * jnp.cos(lon) * cos_lat
    y = r * jnp.sin(lon) * cos_lat
    z = r * jnp.sin(lat)
    return x, y, z


def intersect_great_circles(
    a1: jax.Array, a2: jax.Array,
    b1: jax.Array, b2: jax.Array,
    radius: float = 1.0,
    eps: float = 1e-30,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 616: intersection of two great circles (Cartesian inputs).

    Faithful port of FV3 ``intersect`` (fv_grid_utils.F90:2096-2194).
    Two great circles are defined by:
        - Circle A: arc through ``a1`` and ``a2``
        - Circle B: arc through ``b1`` and ``b2``

    Returns:
        - ``x_inter``: the intersection point on sphere closest to
          the centroid of (a1, a2, b1, b2) (FV3's ``get_nearest``
          branch); scaled to ``radius``.
        - ``local_a``: ``True`` if ``x_inter`` lies between ``a1``
          and ``a2`` on circle A (chord distance check, F90:2186).
        - ``local_b``: same for circle B.

    Each ``ai``, ``bi`` shape ``(..., 3)``; broadcasts on leading
    axes.  ``local_a`` / ``local_b`` are boolean arrays.

    Matches FV3's exact determinant formulation (lines 2128-2147)
    for bit-equivalence; handles the FV3 degenerate branches
    (``b1_xyz=0`` → x_inter=b1; ``b2_xyz=0`` → x_inter=b2) via
    ``jnp.where``.
    """
    a1x = a1[..., 0]; a1y = a1[..., 1]; a1z = a1[..., 2]
    a2x = a2[..., 0]; a2y = a2[..., 1]; a2z = a2[..., 2]
    b1x = b1[..., 0]; b1y = b1[..., 1]; b1z = b1[..., 2]
    b2x = b2[..., 0]; b2y = b2[..., 1]; b2z = b2[..., 2]

    a2_xy = a2x * a1y - a2y * a1x
    b1_xy = b1x * a1y - b1y * a1x
    b2_xy = b2x * a1y - b2y * a1x

    a2_xz = a2x * a1z - a2z * a1x
    b1_xz = b1x * a1z - b1z * a1x
    b2_xz = b2x * a1z - b2z * a1x

    b1_xyz = b1_xy * a2_xz - b1_xz * a2_xy
    b2_xyz = b2_xy * a2_xz - b2_xz * a2_xy

    # General branch: x_raw = b2 - b1 * (b2_xyz / b1_xyz), normalized to radius
    safe_b1_xyz = jnp.where(jnp.abs(b1_xyz) > eps, b1_xyz, 1.0)
    ratio = b2_xyz / safe_b1_xyz
    x_general = b2 - b1 * ratio[..., None]
    length = jnp.sqrt(jnp.sum(x_general * x_general, axis=-1, keepdims=True))
    safe_len = jnp.where(length > eps, length, 1.0)
    x_general = radius * x_general / safe_len

    # FV3 degenerate branches (F90:2139-2142)
    b1_zero = jnp.abs(b1_xyz) <= eps
    b2_zero = (~b1_zero) & (jnp.abs(b2_xyz) <= eps)
    x_inter = jnp.where(
        b1_zero[..., None], b1,
        jnp.where(b2_zero[..., None], b2, x_general),
    )

    # get_nearest: pick ±x_inter closer to centroid (F90:2157-2169)
    center = 0.25 * (a1 + a2 + b1 + b2)
    dx_pos = x_inter - center
    dx_neg = -x_inter - center
    d_pos = jnp.sum(dx_pos * dx_pos, axis=-1)
    d_neg = jnp.sum(dx_neg * dx_neg, axis=-1)
    x_inter = jnp.where(
        (d_neg < d_pos)[..., None], -x_inter, x_inter,
    )

    # check_local for A and B (F90:2171-2192): chord-distance test
    def _check_local(x1: jax.Array, x2: jax.Array) -> jax.Array:
        dx = x1 - x2
        dist = jnp.sum(dx * dx, axis=-1)
        dx1 = x1 - x_inter
        dx2 = x2 - x_inter
        d1 = jnp.sum(dx1 * dx1, axis=-1)
        d2 = jnp.sum(dx2 * dx2, axis=-1)
        return (d1 <= dist) & (d2 <= dist)

    local_a = _check_local(a1, a2)
    local_b = _check_local(b1, b2)
    return x_inter, local_a, local_b


def unit_vect_latlon(
    lon: jax.Array, lat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 615: east/north unit tangent vectors at (lon, lat).

    Faithful port of FV3 ``unit_vect_latlon`` (fv_grid_utils.F90:
    2286-2309).  Returns the two Cartesian unit vectors of the
    local geographic frame at the input (lon, lat) point::

        elon = (-sin λ,    cos λ,    0    )
        elat = (-sin φ cos λ, -sin φ sin λ, cos φ)

    where ``λ`` = lon, ``φ`` = lat.  Used by FV3 ``c2l_ord4`` to
    rotate D-grid winds to geographic (east, north) frame.

    Returns two arrays of shape ``(..., 3)``; broadcasts on
    leading axes.
    """
    sin_lon = jnp.sin(lon)
    cos_lon = jnp.cos(lon)
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    zero = jnp.zeros_like(sin_lon)
    elon = jnp.stack([-sin_lon, cos_lon, zero], axis=-1)
    elat = jnp.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    return elon, elat


def get_unit_vect3(
    p1: jax.Array, p2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 615: unit tangent vector at GC midpoint (Cartesian variant).

    Faithful port of FV3 ``get_unit_vect3`` (fv_grid_utils.F90:
    1865-1876).  Cartesian-input version of iter-611
    ``get_unit_vect2`` — takes ``p1``, ``p2`` already in Cartesian
    (last axis = 3-vector) and returns the unit tangent vector at
    the great-circle midpoint pointing from p1 → p2.

    Algorithm (FV3 exact):
        pc = mid_pt3_cart(p1, p2)
        p3 = p2 × p1                   (great-circle pole)
        uc = pc × p3                   (tangent at pc)
        uc / |uc|
    """
    pc = mid_pt3_cart(p1, p2)
    p3 = vect_cross(p2, p1)
    uc = vect_cross(pc, p3)
    return normalize_vect(uc)


def great_circle_distance_cart(
    v1: jax.Array, v2: jax.Array,
    radius: float = constants.R_earth,
) -> jax.Array:
    """FV3_3D iter 614: great-circle distance from Cartesian inputs.

    Faithful port of FV3 ``great_circle_dist_cart`` (fv_grid_utils.F90:
    2065-2092)::

        cos(d/R) = (v1·v2) / (|v1|·|v2|)
        d = R · acos(clip(cos, -1, 1))

    Each ``v1``, ``v2`` shape ``(..., 3)`` on (or near) the unit
    sphere; result broadcasts on leading axes.  Result has same
    units as ``radius`` (default: legoESM R_earth in metres).

    Differentiable; safe near antipodal points via clip.
    """
    norm = jnp.sum(v1 * v1, axis=-1) * jnp.sum(v2 * v2, axis=-1)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    dot = jnp.sum(v1 * v2, axis=-1) / jnp.sqrt(safe_norm)
    dot = jnp.clip(dot, -1.0, 1.0)
    return radius * jnp.arccos(dot)


def get_area(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
    radius: float = constants.R_earth,
) -> jax.Array:
    """FV3_3D iter 614: spherical-excess cell area for quadrilateral cell.

    Faithful port of FV3 ``get_area`` (fv_grid_utils.F90:2749-2790).
    Computes the four spherical angles at the cell corners and uses
    the spherical-excess formula::

        Area = (α1 + α2 + α3 + α4 - 2π) · R²

    The corner-order convention matches FV3's signature exactly
    (note the FV3 call uses ``p1, p4, p2, p3``):

        4 ----- 3
        |       |
        |       |
        1 ----- 2

    and the four corner angles are taken at vertices 1, 2, 3, 4 in
    counterclockwise order.

    Result has units of ``radius²`` (default: legoESM R_earth in m²).
    """
    # Build Cartesian corner vectors
    e1 = jnp.stack(list(latlon2xyz(lon1, lat1)), axis=-1)
    e2 = jnp.stack(list(latlon2xyz(lon2, lat2)), axis=-1)
    e3 = jnp.stack(list(latlon2xyz(lon3, lat3)), axis=-1)
    e4 = jnp.stack(list(latlon2xyz(lon4, lat4)), axis=-1)
    # FV3 fv_grid_utils.F90:2757-2782 corner-angle convention:
    #   ang1 = ∠(at p1; p2 → p4)
    #   ang2 = ∠(at p2; p3 → p1)
    #   ang3 = ∠(at p3; p4 → p2)
    #   ang4 = ∠(at p4; p3 → p1)
    ang1 = spherical_angle(e1, e2, e4)
    ang2 = spherical_angle(e2, e3, e1)
    ang3 = spherical_angle(e3, e4, e2)
    ang4 = spherical_angle(e4, e3, e1)
    excess = ang1 + ang2 + ang3 + ang4 - 2.0 * jnp.pi
    return excess * (radius * radius)


def expand_cell(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
    fac: float,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]:
    """FV3_3D iter 613: expand 4-corner cell about its center by factor ``fac``.

    Faithful port of FV3 ``expand_cell`` (fv_grid_utils.F90:
    2631-2697).  Returns 4 new (lon, lat) corners with the cell
    extrapolated (fac > 1) or shrunk (fac < 1) about the
    spherical center.

        fac = 1: returns the input corners unchanged
        fac = 0: all 4 corners collapse to the cell center
        fac > 1: expansion outward (cell grows)

    All output corners are forced to lie on the unit sphere via
    re-normalization, matching FV3 lines 2675-2686.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x3, y3, z3 = latlon2xyz(lon3, lat3)
    x4, y4, z4 = latlon2xyz(lon4, lat4)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    p3 = jnp.stack([x3, y3, z3], axis=-1)
    p4 = jnp.stack([x4, y4, z4], axis=-1)
    ec = cell_center3(p1, p2, p3, p4)
    qq1 = normalize_vect(ec + fac * (p1 - ec))
    qq2 = normalize_vect(ec + fac * (p2 - ec))
    qq3 = normalize_vect(ec + fac * (p3 - ec))
    qq4 = normalize_vect(ec + fac * (p4 - ec))
    return (
        xyz2latlon(qq1[..., 0], qq1[..., 1], qq1[..., 2]),
        xyz2latlon(qq2[..., 0], qq2[..., 1], qq2[..., 2]),
        xyz2latlon(qq3[..., 0], qq3[..., 1], qq3[..., 2]),
        xyz2latlon(qq4[..., 0], qq4[..., 1], qq4[..., 2]),
    )


def rotate_winds_geo_to_grid(
    u_east: jax.Array, v_north: jax.Array, angle: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rotate winds from geographic (east, north) to grid-aligned (x, y)."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north
    return u_grid, v_grid


def apply_small_earth_scaling(
    grid: CubedSphereGrid,
    factor: float,
) -> CubedSphereGrid:
    """Create a small-Earth grid by scaling radius and rotation rate.

    On a small Earth of radius R/X:
    - dx, dy scale as 1/X
    - Cell areas scale as 1/X^2
    - Omega scales as X (to keep Rossby number constant)
    - Coriolis f scales as X

    Parameters
    ----------
    grid : CubedSphereGrid
        Original grid at Earth radius.
    factor : float
        Reduction factor X. Earth radius becomes R_earth/X.

    Returns
    -------
    CubedSphereGrid
        New grid with scaled metrics.
    """
    if factor == 1.0:
        return grid

    from legoesm import constants
    return create_cubed_sphere(
        grid.n,
        radius=constants.R_earth / factor,
        omega=constants.Omega * factor,
    )


def rotate_winds_grid_to_geo(
    u_grid: jax.Array, v_grid: jax.Array, angle: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rotate winds from grid-aligned (x, y) to geographic (east, north)."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_east = cos_a * u_grid - sin_a * v_grid
    v_north = sin_a * u_grid + cos_a * v_grid
    return u_east, v_north


def create_cubed_sphere_panel(
    n: int,
    face_id: int = 0,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    return_cdgrid: bool = False,
) -> "CubedSphereGrid | tuple[CubedSphereGrid, ...]":
    """Create a single-face cubed-sphere panel for regional experiments.

    Extracts one face from a full cubed-sphere grid and returns a
    ``CubedSphereGrid`` with leading dimension 1 instead of 6.

    All existing operators (gradient, divergence, Laplacian) work
    automatically because :func:`~legoesm.grids.halo.pad_halo` detects
    ``data.shape[0] == 1`` and applies Neumann (zero-gradient) wall
    boundary conditions instead of inter-face halo exchange.

    Parameters
    ----------
    n : int
        Grid resolution (cells per face edge).
    face_id : int
        Which cube face to extract (0-5, default 0 = equatorial).
    radius : float
        Sphere radius [m].
    omega : float
        Rotation rate [rad/s].

    Returns
    -------
    CubedSphereGrid or (CubedSphereGrid, CubedSphereCDGrid)
        Grid with all arrays shaped ``(1, n, n)`` or ``(1, n+2, n+2)``
        for the selected face.  Use with a wall land mask (0 on
        boundary, 1 in interior) for closed-basin experiments.

        When ``return_cdgrid=True``, also returns the single-face
        C-D grid needed by the ocean baroclinic solver.
    """
    full = create_cubed_sphere(n, radius=radius, omega=omega, dtype=dtype)

    # Extract single face, keeping leading dimension
    f = face_id
    _s = lambda arr: arr[f:f+1]  # (6,...) → (1,...)
    _p = lambda arr: arr[f:f+1]  # same for padded arrays

    # Padded angle/metric arrays: extract face then re-pad with edge BC
    # (the full grid's padded arrays have inter-face halo data that doesn't
    # apply to a single-face panel).
    def _repad(arr_full_face, halo=1):
        """Re-pad a single face with Neumann BC."""
        interior = arr_full_face  # (1, n, n) or (1, n+2h, n+2h)
        if interior.shape[1] > n:
            interior = interior[:, halo:-halo, halo:-halo]
        return jnp.pad(interior, ((0, 0), (halo, halo), (halo, halo)),
                        mode="edge")

    angle_p = _repad(_s(full.angle_padded), halo=1)

    panel = CubedSphereGrid(
        n=n,
        radius=float(radius),
        lon=_s(full.lon),
        lat=_s(full.lat),
        area=_s(full.area),
        dx=_s(full.dx),
        dy=_s(full.dy),
        f=_s(full.f),
        cos_lat=_s(full.cos_lat),
        sin_lat=_s(full.sin_lat),
        angle=_s(full.angle),
        x_cart=_s(full.x_cart),
        y_cart=_s(full.y_cart),
        z_cart=_s(full.z_cart),
        angle_padded=angle_p,
        cos_angle=_s(full.cos_angle),
        sin_angle=_s(full.sin_angle),
        cos_angle_padded=jnp.cos(angle_p),
        sin_angle_padded=jnp.sin(angle_p),
        hx_ext=_repad(_s(full.hx_ext), halo=1),
        hy_ext=_repad(_s(full.hy_ext), halo=1),
        halo_interp_offsets=None,  # not needed — wall BC
        cos_angle_padded_h2=jnp.cos(_repad(_s(full.angle), halo=2)),
        sin_angle_padded_h2=jnp.sin(_repad(_s(full.angle), halo=2)),
        hx_ext_h2=_repad(_s(full.hx_ext_h2), halo=2),
        hy_ext_h2=_repad(_s(full.hy_ext_h2), halo=2),
        halo_interp_offsets_h2=None,  # not needed — wall BC
        halo_interp_offsets_h3=None,  # not needed — wall BC
        cos_angle_padded_h3=jnp.cos(_repad(_s(full.angle), halo=3)),
        sin_angle_padded_h3=jnp.sin(_repad(_s(full.angle), halo=3)),
        hx_ext_h3=_repad(_s(full.hx_ext_h3), halo=3),
        hy_ext_h3=_repad(_s(full.hy_ext_h3), halo=3),
        duogrid=None,  # regional panel: no cross-face duogrid data
    )

    if not return_cdgrid:
        return panel

    # Also build single-face C-D grid from the full cdgrid.
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    full_cdgrid = create_cubed_sphere_cdgrid(full, omega=omega)

    # Extract face f from every array leaf in the cdgrid, replacing the
    # base grid reference with the panel.
    def _extract_face(leaf):
        if hasattr(leaf, 'shape') and hasattr(leaf, 'ndim'):
            if leaf.ndim >= 3 and leaf.shape[0] == 6:
                return leaf[f:f+1]
        return leaf

    cdgrid_panel = jax.tree.map(_extract_face, full_cdgrid)
    # Replace the base grid with the panel (base is index 0 of the NamedTuple)
    cdgrid_panel = cdgrid_panel._replace(base=panel)

    # Restore the FV3 bounded_domain rsin_u/rsin_v convention for the panel.
    # The full-grid build applies the 1/sin panel-edge override only when
    # bounded_domain is False (fv_arrays.F90:1512).  A single-face panel is
    # a bounded_domain case (regional) and must use 1/sin² everywhere —
    # matching fv_grid_utils.F90:509.  Undo the override that was inherited
    # from the 6-face build.
    import jax.numpy as _jnp
    _EPS = float(_jnp.finfo(_jnp.float32).eps)
    # cosa_u_panel was already extracted; recompute rsin_u = 1/sin² there.
    sina_u_sq_panel = _jnp.maximum(
        1.0 - cdgrid_panel.cosa_u**2, _EPS)
    rsin_u_panel = 1.0 / _jnp.maximum(sina_u_sq_panel, _EPS)
    sina_v_sq_panel = _jnp.maximum(
        1.0 - cdgrid_panel.cosa_v**2, _EPS)
    rsin_v_panel = 1.0 / _jnp.maximum(sina_v_sq_panel, _EPS)
    cdgrid_panel = cdgrid_panel._replace(
        rsin_u=rsin_u_panel.astype(cdgrid_panel.rsin_u.dtype),
        rsin_v=rsin_v_panel.astype(cdgrid_panel.rsin_v.dtype),
    )
    return panel, cdgrid_panel
