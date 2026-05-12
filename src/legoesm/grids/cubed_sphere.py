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
