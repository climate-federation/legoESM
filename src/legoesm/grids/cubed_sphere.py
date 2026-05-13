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


def geopotential_from_T_peln_fv3(
    pt: jax.Array,
    peln: jax.Array,
    phis: jax.Array,
    q: jax.Array | None = None,
    moist: bool = False,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 738: geopotential at interfaces from hydrostatic balance.

    Faithful JAX port of FV3's standard hydrostatic Φ integration
    (used in fv_diagnostics.F90 height-field paths and IC
    ingestion):

        Φ_surface = phis                                  (k = km)
        Φ[k] = phis + Σ_{j>=k} R_d · T_v[j] · Δpeln[j]    (upward sum)

    where ``T_v = pt`` (dry) or ``T_v = pt · (1 + zvir·q)`` (moist).

    Returns interface geopotential (m²/s²); divide by g for height
    (m).  Pairs with iter-727 ``compute_zh_from_delz_fv3`` (heights
    from delz path), iter-684 ``get_height_given_pressure_fv3``.

    Parameters
    ----------
    pt : jax.Array, shape (..., km)
        Air temperature (K).
    peln : jax.Array, shape (..., km+1)
        log(pressure) at interfaces.
    phis : jax.Array, shape (...,)
        Surface geopotential (m²/s²).
    q : jax.Array, shape (..., km), optional
        Specific humidity — required if moist.
    moist : bool, default False.
    zvir : float, optional
        Virtual-T coefficient.  Default ``R_v/R_d − 1``.

    Returns
    -------
    phi : jax.Array, shape (..., km+1)
        Geopotential at interfaces (m²/s²).
    """
    if moist:
        if q is None:
            raise ValueError("moist=True requires q")
        if zvir is None:
            zvir = constants.R_v / constants.R_d - 1.0
        t_v = pt * (1.0 + zvir * q)
    else:
        t_v = pt
    d_peln = peln[..., 1:] - peln[..., :-1]
    # Layer contribution = R_d · T_v · Δpeln  (positive)
    layer_contrib = constants.R_d * t_v * d_peln                 # (..., km)
    # Cumulative sum from surface upward: reverse → cumsum → reverse
    cum_up = jnp.cumsum(layer_contrib[..., ::-1], axis=-1)[..., ::-1]
    # Φ[..., 0..km-1] = phis + cum_up
    # Φ[..., km] = phis
    phi_above = phis[..., None] + cum_up
    return jnp.concatenate(
        [phi_above, phis[..., None]],
        axis=-1,
    )


def temperature_from_theta_fv3(
    theta: jax.Array,
    p: jax.Array,
    p_ref: float = 1.0e5,
    cappa: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 737: temperature from potential temperature.

    Inverse of iter-735 ``theta_dry_fv3``:

        T = θ · Π = θ · (p / p_ref)^κ

    Pairs with iter-735 ``theta_dry_fv3`` and iter-736
    ``exner_fv3``.  Composes:

        T = theta_dry_fv3⁻¹ → temperature_from_theta_fv3
        T = θ · exner_fv3(p)            (using iter-736)

    Parameters
    ----------
    theta : jax.Array
        Potential temperature (K).
    p : jax.Array
        Pressure (Pa).
    p_ref : float, default 1e5.
    cappa : float or jax.Array, optional.  Default ``constants.kappa``.

    Returns
    -------
    T : jax.Array
        Air temperature (K).
    """
    kap = constants.kappa if cappa is None else cappa
    return theta * (p / p_ref) ** kap


def exner_fv3(
    p: jax.Array,
    p_ref: float = 1.0e5,
    cappa: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 736: point-wise Exner function.

    Standard Exner function:

        Π = (p / p_ref)^κ

    Used throughout FV3 dyn_core for the pressure-gradient term
    ``c_p · θ · ∇Π`` and as the inverse of iter-735 ``theta_dry_fv3``:

        θ = T / Π     ⇔     T = θ · Π

    Compared to iter-722 ``compute_pkz_fv3`` which provides a
    LAYER-MEAN Exner (integral mean over the log-p layer), this
    helper is point-wise: Exner at a specific pressure value.

    Parameters
    ----------
    p : jax.Array
        Pressure (Pa).
    p_ref : float, default 1e5
        Reference pressure (Pa).
    cappa : float or jax.Array, optional
        Poisson exponent.  Default ``constants.kappa``.  Use
        layer-varying array for moist atmosphere (iter-723
        cappa_moist_fv3 output).

    Returns
    -------
    pi : jax.Array
        Exner function (dimensionless).
    """
    kap = constants.kappa if cappa is None else cappa
    return (p / p_ref) ** kap


def theta_dry_fv3(
    pt: jax.Array,
    p: jax.Array,
    p_ref: float = 1.0e5,
    cappa: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 735: dry potential temperature.

    Standard Poisson form:

        θ = T · (p_ref / p)^κ

    Used throughout FV3 dyn_core, fv_mapz, and diagnostic paths.
    For moist atmosphere, pass ``cappa = cappa_moist_fv3(q)`` from
    iter-723 to get the variable-exponent θ.

    Pairs with iter-722 ``compute_pkz_fv3`` (which provides
    layer-mean ``pkz`` for the standard ``θ = pt / pkz`` form).

    Parameters
    ----------
    pt : jax.Array
        Air temperature (K).
    p : jax.Array
        Pressure at evaluation point (Pa).
    p_ref : float, default 1e5
        Reference pressure (Pa).
    cappa : float or jax.Array, optional
        Poisson exponent.  Default ``constants.kappa``.  Use
        layer-varying array for moist atmosphere (iter-723
        cappa_moist_fv3 output).

    Returns
    -------
    theta : jax.Array
        Potential temperature (K).
    """
    kap = constants.kappa if cappa is None else cappa
    return pt * (p_ref / p) ** kap


def potential_energy_column_fv3(
    phi_interfaces: jax.Array,
    delp: jax.Array,
) -> jax.Array:
    """FV3_3D iter 747: column-integrated geopotential energy.

        PE_col = Σ_k delp[k] · 0.5·(phi[k] + phi[k+1]) / g

    Layer-mean geopotential weighted by mass.  Standard component
    of FV3 total-energy budget (iter-693 nh_total_energy uses
    ``phi_avg = 0.5·(phiz[k]+phiz[k+1])`` in this form).

    Composes iter-742 column_integral_delp_fv3.  Use with iter-738
    ``geopotential_from_T_peln_fv3`` or iter-727
    ``compute_zh_from_delz_fv3`` (multiplied by g) to produce
    interface geopotentials.

    Companion to iter-744 IE, iter-745 KE, iter-746 LE.

    Parameters
    ----------
    phi_interfaces : jax.Array, shape (..., km+1)
        Geopotential at interfaces (m²/s²).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).

    Returns
    -------
    pe : jax.Array, shape (...,)
        Column potential energy (J/m²).
    """
    phi_avg = 0.5 * (phi_interfaces[..., :-1] + phi_interfaces[..., 1:])
    return column_integral_delp_fv3(phi_avg, delp)


def surface_pressure_from_delp_fv3(
    delp: jax.Array,
    p_top: float = 0.0,
) -> jax.Array:
    """FV3_3D iter 759: surface pressure from column delp.

        ps = p_top + Σ_k delp[k]

    Standard FV3 IC pattern.  Inverse of iter-731
    ``compute_pe_from_delp_fv3`` for the surface interface
    (``pe[km] = ps``).

    Used in IC ingestion paths to recover ps from a delp profile
    (e.g., after reading external pressure-level data and
    inverting the hybrid-coord chain).

    Pairs with iter-731 (full pe profile) and iter-724
    ``compute_hybrid_pressure_fv3`` (ak/bk/ps path).

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    p_top : float, default 0.0.

    Returns
    -------
    ps : jax.Array, shape (...,)
        Surface pressure (Pa).
    """
    return p_top + jnp.sum(delp, axis=-1)


def column_geopotential_thickness_fv3(
    delz: jax.Array,
) -> jax.Array:
    """FV3_3D iter 758: column geopotential thickness (z_top − z_surface).

        thickness = Σ_k (−delz[k]) = total atmospheric depth (m)

    Standard FV3 dimensional check.  Earth troposphere depth at
    50 layers × ~250 m ≈ 12.5 km.

    Independent of surface elevation phis — only depends on delz.

    Parameters
    ----------
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3).

    Returns
    -------
    thickness : jax.Array, shape (...,)
        Column thickness (m, positive).
    """
    return jnp.sum(-delz, axis=-1)


def total_atmosphere_mass_fv3(
    delp: jax.Array,
) -> jax.Array:
    """FV3_3D iter 757: column total atmospheric mass (kg/m²).

        M_col = Σ_k delp / g
              ≈ (p_s − p_top) / g

    Standard mass-conservation diagnostic.  For Earth at p_s=1e5 Pa,
    p_top=0 → M_col ≈ 10197 kg/m².

    Companion to iter-749 ``total_water_column_fv3`` (water mass)
    and iter-752 ``dry_surface_pressure_fv3``:

        M_col_dry = M_col − TWC

    Composes iter-742 column_integral_delp_fv3 with field=1.

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).

    Returns
    -------
    mass_col : jax.Array, shape (...,)
        Column air mass (kg/m²).
    """
    return column_integral_delp_fv3(jnp.ones_like(delp), delp)


def precipitable_water_fv3(
    q_sphum: jax.Array,
    delp: jax.Array,
) -> jax.Array:
    """FV3_3D iter 756: precipitable water (column water vapor).

        PWV = Σ_k delp · q_sphum / g     (kg/m²)
            ≈ PWV / ρ_water · 1000        (mm, numerically same value
                                            since ρ_water = 1000 kg/m³)

    Standard meteorology diagnostic.  Numeric value in kg/m² equals
    column-water-vapor height in mm exactly (1 kg of water per m²
    spread at 1000 kg/m³ density = 1 mm depth).

    Composes iter-742 ``column_integral_delp_fv3``.

    Parameters
    ----------
    q_sphum : jax.Array, shape (..., km)
        Specific humidity (kg/kg).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).

    Returns
    -------
    pwv : jax.Array, shape (...,)
        Precipitable water (kg/m² ≡ mm).
    """
    return column_integral_delp_fv3(q_sphum, delp)


def mse_column_fv3(
    pt: jax.Array,
    z: jax.Array,
    q_sphum: jax.Array,
    delp: jax.Array,
    cp: float | jax.Array | None = None,
    L: float | None = None,
) -> jax.Array:
    """FV3_3D iter 755: column-integrated moist static energy.

        MSE_col = Σ_k delp · MSE / g

    Composes iter-753 ``moist_static_energy_fv3`` + iter-742
    ``column_integral_delp_fv3``.

    Parameters
    ----------
    pt, z, q_sphum, delp : jax.Array
        Layer-mean inputs (see iter-753).
    cp, L : optional
        Overrides (see iter-753).

    Returns
    -------
    mse_col : jax.Array
        Column MSE (J/m²).
    """
    return column_integral_delp_fv3(
        moist_static_energy_fv3(pt, z, q_sphum, cp=cp, L=L),
        delp,
    )


def dse_column_fv3(
    pt: jax.Array,
    z: jax.Array,
    delp: jax.Array,
    cp: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 755: column-integrated dry static energy.

        DSE_col = Σ_k delp · DSE / g

    Composes iter-754 ``dry_static_energy_fv3`` + iter-742
    ``column_integral_delp_fv3``.

    Parameters
    ----------
    pt, z, delp : jax.Array
        Layer-mean inputs.
    cp : optional
        Override (see iter-754).

    Returns
    -------
    dse_col : jax.Array
        Column DSE (J/m²).
    """
    return column_integral_delp_fv3(
        dry_static_energy_fv3(pt, z, cp=cp),
        delp,
    )


def dry_static_energy_fv3(
    pt: jax.Array,
    z: jax.Array,
    cp: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 754: dry static energy per unit mass.

        DSE = c_p · T + g · z

    Approximately conserved for DRY adiabatic motion (vertical
    component of enthalpy + potential energy).  Companion to
    iter-753 ``moist_static_energy_fv3``:

        MSE = DSE + L · q_sphum

    Used in atmospheric energy budget decompositions.

    Parameters
    ----------
    pt : jax.Array
        Temperature (K).
    z : jax.Array
        Height above surface (m).
    cp : float or jax.Array, optional
        Isobaric specific heat (J/kg/K).  Default ``constants.c_pd``.

    Returns
    -------
    dse : jax.Array
        Dry static energy per unit mass (J/kg).
    """
    if cp is None:
        cp = constants.c_pd
    return cp * pt + constants.g * z


def moist_static_energy_fv3(
    pt: jax.Array,
    z: jax.Array,
    q_sphum: jax.Array,
    cp: float | jax.Array | None = None,
    L: float | None = None,
) -> jax.Array:
    """FV3_3D iter 753: moist static energy per unit mass.

        MSE = c_p · T + g · z + L · q_sphum

    Standard atmospheric conservative variable (approximately
    conserved for moist adiabatic motion).  Used in convection
    parameterizations and tropical-meteorology diagnostics.

    Pairs with iter-744 IE (cv·T), iter-746 LE (L·q), iter-747 PE (Φ).
    MSE = h + Φ = (c_p·T + L·q) + g·z (h = moist static enthalpy).

    Parameters
    ----------
    pt : jax.Array
        Temperature (K).
    z : jax.Array
        Height above surface (m).
    q_sphum : jax.Array
        Specific humidity (kg/kg).
    cp : float or jax.Array, optional
        Isobaric specific heat (J/kg/K).  Default ``constants.c_pd``.
        For moist atmosphere use iter-714 ``moist_cp_fv3`` output.
    L : float, optional
        Latent heat (J/kg).  Default ``constants.L_v`` (vaporization).

    Returns
    -------
    mse : jax.Array
        Moist static energy per unit mass (J/kg).
    """
    if cp is None:
        cp = constants.c_pd
    if L is None:
        L = constants.L_v
    return cp * pt + constants.g * z + L * q_sphum


def dry_surface_pressure_fv3(
    ps: jax.Array,
    delp: jax.Array,
    q_sphum: jax.Array | None = None,
    q_liq_wat: jax.Array | None = None,
    q_rainwat: jax.Array | None = None,
    q_ice_wat: jax.Array | None = None,
    q_snowwat: jax.Array | None = None,
    q_graupel: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 752: dry-air surface pressure (per cell).

        ps_dry = ps − g · TWC
               = ps − g · (column water vapor + liq + rain + ice + snow + graupel)

    where TWC is total-water column (kg/m²) from iter-749
    ``total_water_column_fv3``.

    Per-cell version of iter-694 ``prt_mass`` ``dry_ps_mean`` global
    diagnostic.  Used in IC ingestion + mass conservation for dry-air
    budget.

    Parameters
    ----------
    ps : jax.Array, shape (...,)
        Total surface pressure (Pa).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    q_sphum, q_liq_wat, q_rainwat, q_ice_wat, q_snowwat, q_graupel :
        Mixing ratios (kg/kg).  At least one required.

    Returns
    -------
    ps_dry : jax.Array, shape (...,)
        Dry-air surface pressure (Pa).
    """
    twc = total_water_column_fv3(
        delp,
        q_sphum=q_sphum, q_liq_wat=q_liq_wat, q_rainwat=q_rainwat,
        q_ice_wat=q_ice_wat, q_snowwat=q_snowwat, q_graupel=q_graupel,
    )
    return ps - constants.g * twc


def area_weighted_mean_fv3(
    field: jax.Array,
    area: jax.Array,
    mask: jax.Array | None = None,
    empty_band_sentinel: float = -1.0,
) -> jax.Array:
    """FV3_3D iter 750: area-weighted (masked) global mean.

        mean = Σ (field · area · mask) / Σ (area · mask)
             = sentinel  if Σ(area·mask) <= 1

    Standard FV3 pattern used in iter-694 ``prt_mass``, iter-705
    ``prt_gb_nh_sh``, and many other diagnostic global-mean
    computations.

    Parameters
    ----------
    field : jax.Array
        Scalar field at cell centers.
    area : jax.Array
        Cell areas (m²).
    mask : jax.Array, optional
        Boolean / 0-1 mask selecting a subset of cells.  None →
        all cells.
    empty_band_sentinel : float, default -1.0
        FV3 bugfix value returned when total masked area ≤ 1.0
        (e.g., empty NH/SH band on regional domain).

    Returns
    -------
    mean : jax.Array (scalar)
        Area-weighted mean over masked region.
    """
    if mask is not None:
        weight = area * mask
    else:
        weight = area
    total_w = jnp.sum(weight)
    total_field = jnp.sum(field * weight)
    return jnp.where(total_w > 1.0, total_field / total_w, empty_band_sentinel)


def total_water_column_fv3(
    delp: jax.Array,
    q_sphum: jax.Array | None = None,
    q_liq_wat: jax.Array | None = None,
    q_rainwat: jax.Array | None = None,
    q_ice_wat: jax.Array | None = None,
    q_snowwat: jax.Array | None = None,
    q_graupel: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 749: total water column (kg/m²).

        TWC = Σ_species column_integral(q_species)
            = Σ_k delp · (q_v + q_l + q_r + q_i + q_s + q_g) / g

    Sum of column water across all condensate phases.  Used in
    FV3 water-mass-conservation diagnostics (iter-694 prt_mass
    sums these per-tracer means).

    All tracer inputs optional; missing ones contribute zero.
    At least one must be provided.

    Composes iter-742 column_integral_delp_fv3.

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    q_sphum, q_liq_wat, q_rainwat, q_ice_wat, q_snowwat, q_graupel :
        Mixing ratios (kg/kg).

    Returns
    -------
    twc : jax.Array, shape (...,)
        Total water column (kg/m²).
    """
    tracers = [
        q for q in (q_sphum, q_liq_wat, q_rainwat,
                    q_ice_wat, q_snowwat, q_graupel)
        if q is not None
    ]
    if not tracers:
        raise ValueError("total_water_column_fv3 requires at least one tracer")
    q_total = tracers[0]
    for q in tracers[1:]:
        q_total = q_total + q
    return column_integral_delp_fv3(q_total, delp)


def latent_energy_column_fv3(
    q_sphum: jax.Array,
    delp: jax.Array,
    L: float | None = None,
) -> jax.Array:
    """FV3_3D iter 746: column-integrated latent energy.

        LE_col = L_v · Σ_k delp[k] · q_sphum[k] / g
               = L_v · column_water_vapor (kg/m²)

    Standard component of FV3 total-energy budget (iter-693
    ``nh_total_energy_fv3`` uses ``L_v · q_sphum`` in moist branch).
    Composes iter-742 column_integral.

    Companion to iter-744 ``internal_energy_column_fv3`` +
    iter-745 ``kinetic_energy_column_fv3``.

    Parameters
    ----------
    q_sphum : jax.Array, shape (..., km)
        Specific humidity (kg/kg).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    L : float, optional
        Latent heat (J/kg).  Default ``constants.L_v`` (vaporization,
        2.501e6 J/kg).  Use ``constants.L_s`` for ice sublimation.

    Returns
    -------
    le : jax.Array, shape (...,)
        Column latent energy (J/m²).
    """
    if L is None:
        L = constants.L_v
    return L * column_integral_delp_fv3(q_sphum, delp)


def kinetic_energy_column_fv3(
    ua: jax.Array,
    va: jax.Array,
    delp: jax.Array,
    w: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 745: column-integrated kinetic energy.

        KE_col = Σ_k delp[k] · KE[k] / g
                = Σ_k delp[k] · 0.5·(ua²+va²[+w²]) / g

    Companion to iter-744 ``internal_energy_column_fv3``.  Composes
    iter-740 ``kinetic_energy_fv3`` + iter-742
    ``column_integral_delp_fv3``.

    Standard component of FV3 total-energy budget (iter-693
    ``nh_total_energy_fv3`` uses this as the KE piece).

    Parameters
    ----------
    ua, va : jax.Array, shape (..., km)
        Horizontal wind components.
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    w : jax.Array, shape (..., km), optional
        Vertical velocity.  None → 2-component (hydrostatic).

    Returns
    -------
    ke_col : jax.Array, shape (...,)
        Column kinetic energy (J/m²).
    """
    return column_integral_delp_fv3(kinetic_energy_fv3(ua, va, w), delp)


def internal_energy_column_fv3(
    pt: jax.Array,
    delp: jax.Array,
    cv: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 744: column-integrated internal energy.

        IE = Σ_k delp[k] · cv · pt[k] / g

    where cv defaults to dry isochoric heat ``c_pd − R_d``.  Pass
    ``cv = moist_cv_fv3(...)`` from iter-713 for moisture-weighted
    column IE.

    Standard component of FV3 total-energy budget (iter-693
    ``nh_total_energy_fv3`` uses this as the c_v · pt piece).

    Pairs with iter-742 ``column_integral_delp_fv3`` (delegated
    integration) and iter-713 ``moist_cv_fv3`` (moist cv).

    Parameters
    ----------
    pt : jax.Array, shape (..., km)
        Temperature (K).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    cv : float or jax.Array, optional
        Isochoric specific heat (J/kg/K).  Default ``c_pd − R_d``.
        Pass layer-varying array for moist atmosphere.

    Returns
    -------
    ie : jax.Array, shape (...,)
        Column internal energy (J/m²).
    """
    if cv is None:
        cv = constants.c_pd - constants.R_d
    return column_integral_delp_fv3(cv * pt, delp)


def column_integral_delp_fv3(
    field: jax.Array,
    delp: jax.Array,
    divide_by_g: bool = True,
) -> jax.Array:
    """FV3_3D iter 742: delp-weighted column integral.

    Faithful port of FV3's standard column-mass integral pattern
    (used throughout fv_diagnostics.F90, fv_mapz.F90 — e.g. iter-677
    ``z_sum``, iter-693 ``nh_total_energy``, iter-694 ``prt_mass``):

        col = Σ_k delp[k] · field[k]                  (mass-weighted)
        col_kg_per_m2 = col / g                       (divide_by_g=True)

    For tracer-mass column (kg/m²): ``divide_by_g=True``, field is
    mixing ratio (kg/kg).
    For column-mean pressure-weighted average: divide by Σ delp
    externally.

    Standard FV3 convention: cumulative integral in pressure layers.

    Parameters
    ----------
    field : jax.Array, shape (..., km)
        Layer-mean values to integrate.
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa, positive).
    divide_by_g : bool, default True.
        Apply ``/ g`` to convert pressure-weighted integral to
        mass-weighted (kg/m²).

    Returns
    -------
    col : jax.Array, shape (...,)
        Column integral.
    """
    col = jnp.sum(delp * field, axis=-1)
    if divide_by_g:
        col = col / constants.g
    return col


def wind_speed_fv3(
    ua: jax.Array,
    va: jax.Array,
    w: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 741: wind-speed magnitude.

        |V| = sqrt(ua² + va²)       (2-component, horizontal)
        |V| = sqrt(ua² + va² + w²)  (3-component)

    Standard FV3 diagnostic.  Equivalent to ``sqrt(2 · KE)`` via
    iter-740.  Used by iter-676 ``wind_max_fv3`` and many output
    snapshots.

    Parameters
    ----------
    ua, va : jax.Array
        Horizontal wind components (m/s).
    w : jax.Array, optional
        Vertical velocity (m/s).  None → horizontal speed only.

    Returns
    -------
    speed : jax.Array
        Wind-speed magnitude (m/s).
    """
    if w is None:
        return jnp.sqrt(ua * ua + va * va)
    return jnp.sqrt(ua * ua + va * va + w * w)


def wind_direction_fv3(
    ua: jax.Array,
    va: jax.Array,
    convention: str = "from",
) -> jax.Array:
    """FV3_3D iter 777: meteorological wind direction.

    Standard meteorological convention (``convention='from'``):
    direction the wind is *coming from*, measured clockwise from
    north in degrees on [0, 360):

        0°    — N (wind from north → southward flow)
        90°   — E (wind from east  → westward flow)
        180°  — S (wind from south → northward flow)
        270°  — W (wind from west  → eastward flow)

    Mathematical / oceanographic convention (``convention='to'``):
    direction the wind is *going to*.  Differs from 'from' by 180°.

    Used by: wind-rose generation, gust/shift detection, surface-
    flux directional anisotropy, observation matching (METAR
    convention is 'from'), trajectory dispersion runs.

    Calm air (ua = va = 0) returns 0° by convention (``atan2(0,0)``
    is 0 in JAX).

    Parameters
    ----------
    ua, va : jax.Array
        Horizontal wind components (m/s).  ua = east-positive,
        va = north-positive (FV3 cubed-sphere ``ua``/``va``
        post grid-rotation are zonal/meridional).
    convention : {'from', 'to'}
        Meteorological 'from' (default) or mathematical 'to'.

    Returns
    -------
    wdir : jax.Array
        Wind direction in degrees on [0, 360).
    """
    if convention not in ("from", "to"):
        raise ValueError(f"convention must be 'from' or 'to', got {convention!r}")
    # atan2(ua, va) gives the angle of the wind vector measured
    # clockwise from north (since va is the y-axis here).  Convert
    # rad → deg, wrap to [0, 360).
    wdir_to = jnp.degrees(jnp.arctan2(ua, va))
    wdir_to = jnp.mod(wdir_to, 360.0)
    if convention == "from":
        return jnp.mod(wdir_to + 180.0, 360.0)
    return wdir_to


def coriolis_parameter_fv3(
    lat: jax.Array,
    units: str = "rad",
) -> jax.Array:
    """FV3_3D iter 778: Coriolis parameter f = 2·Ω·sin(lat).

    Vertical component of the planetary vorticity vector
    (2·Ω·sin(lat)) acting on horizontal flow.  Used everywhere:
    geostrophic balance, Rossby-wave dispersion, inertial
    oscillations, Ekman pumping, ageostrophic decomposition.

    Centred on Earth: Ω = ``constants.Omega`` = 7.292·10⁻⁵ rad/s
    (sidereal-day rotation rate).

    Sign convention: f > 0 in the Northern Hemisphere, f < 0 in
    the Southern, f = 0 at the equator.

    Parameters
    ----------
    lat : jax.Array
        Latitude.  Interpreted as radians by default; pass
        ``units='deg'`` for degrees input (will be converted to
        radians internally).
    units : {'rad', 'deg'}
        Input units for ``lat``.

    Returns
    -------
    f : jax.Array
        Coriolis parameter (s⁻¹).
    """
    if units not in ("rad", "deg"):
        raise ValueError(f"units must be 'rad' or 'deg', got {units!r}")
    lat_rad = jnp.radians(lat) if units == "deg" else lat
    return 2.0 * constants.Omega * jnp.sin(lat_rad)


def inertial_period_fv3(
    lat: jax.Array,
    units: str = "rad",
    f_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 779: inertial-oscillation period 2π/|f|.

    Period of a free inertial (Coriolis-only) oscillation:

        T_inertial = 2π / |f|,    f = 2·Ω·sin(lat)   (iter-778)

    Returned in **seconds**.  Near-pole values approach 2π/(2·Ω) =
    sidereal half-day ≈ 11.97 h.  Mid-latitude 30°N has |f| = Ω
    → T_inertial = 2π/Ω = one sidereal day ≈ 86164 s.  Equator
    has f = 0 → T = ∞; the helper clamps |f| to ``f_floor`` to
    keep output finite.

    Used by:
      * Ocean mixed-layer near-inertial wave (NIW) decay timescales.
      * Atmospheric inertia-gravity wave dispersion relations.
      * Mesoscale eddy diagnostic Rhines scale & PV inversion.
      * MJO / equatorial-wave critical-latitude analysis.

    Composes with iter-778 ``coriolis_parameter_fv3``.

    Parameters
    ----------
    lat : jax.Array
        Latitude (rad by default; pass ``units='deg'`` for degrees).
    units : {'rad', 'deg'}
    f_floor : float
        Lower bound on |f| (s⁻¹) to avoid div-by-0 at equator.
        Default 1e-12 → equatorial T ≈ 6.3·10¹² s (≈ 2·10⁵ years —
        effectively "no inertial oscillation").

    Returns
    -------
    t_inertial : jax.Array
        Inertial period (s).
    """
    f = coriolis_parameter_fv3(lat, units=units)
    return 2.0 * jnp.pi / jnp.maximum(jnp.abs(f), f_floor)


def beta_plane_fv3(
    lat: jax.Array,
    units: str = "rad",
) -> jax.Array:
    """FV3_3D iter 780: β = df/dy = 2·Ω·cos(lat) / R_earth.

    Meridional gradient of the Coriolis parameter (planetary
    vorticity gradient).  Foundation of: Rossby-wave dispersion
    (ω = −β·k/(k²+l²+R⁻²)), Rhines scale (L_R = √(U/β)), planetary
    geostrophic theory, β-plane approximation.

    Uses Earth's radius from ``constants.R_earth``.  Result is in
    s⁻¹ · m⁻¹ (the canonical SI unit for β).

    Sign: β > 0 in both hemispheres (always positive — cos(lat) ≥ 0
    everywhere |lat| ≤ π/2).  β maximum at equator (≈ 2.29·10⁻¹¹
    s⁻¹m⁻¹), β → 0 at the poles.

    Parameters
    ----------
    lat : jax.Array
        Latitude (rad by default; pass ``units='deg'`` for degrees).
    units : {'rad', 'deg'}

    Returns
    -------
    beta : jax.Array
        Meridional gradient of f (s⁻¹·m⁻¹).
    """
    if units not in ("rad", "deg"):
        raise ValueError(f"units must be 'rad' or 'deg', got {units!r}")
    lat_rad = jnp.radians(lat) if units == "deg" else lat
    return 2.0 * constants.Omega * jnp.cos(lat_rad) / constants.R_earth


def rossby_radius_fv3(
    n_brunt: jax.Array,
    f: jax.Array,
    H: jax.Array,
    f_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 781: barotropic Rossby radius of deformation.

    Natural horizontal length scale of stratified rotating flow:

        L_R = N · H / |f|

    where N = √N² is the Brunt-Väisälä frequency (caller takes
    the square root of iter-772 output), f is Coriolis (iter-778),
    and H is a vertical-scale depth.

    Sets the meridional/zonal scale at which rotational
    (Coriolis) and stratification (buoyancy) effects balance.

    Physical interpretation:
      * Synoptic mid-latitudes (N ~ 0.01, H ~ 10 km, |f| ~ 1e-4):
        L_R ~ 1000 km — the classic synoptic-scale eddy scale.
      * Tropics (|f| → 0): L_R → ∞; equator-bound modes use the
        equatorial Rossby radius √(N·H/(2·β)) instead.
      * Ocean baroclinic (H ~ 1 km, N ~ 0.01): L_R ~ 30-100 km.

    Used by: baroclinic-instability eddy length scale, model
    eddy-permitting/resolving criterion (Δx < L_R/4 ~ "eddy
    resolving"), mesoscale energy spectra, Rhines transition
    scale to zonal jets.

    The ``f_floor`` clamp prevents div-by-0 at the equator;
    default 1e-12 → equatorial L_R ≈ 6.3·10¹⁵ m (effectively
    "no rotational constraint" at the equator).

    Parameters
    ----------
    n_brunt : jax.Array
        Brunt-Väisälä frequency N (s⁻¹).  Caller computes
        ``jnp.sqrt(jnp.maximum(0, n_sq))`` from iter-772 N².
    f : jax.Array
        Coriolis parameter (s⁻¹), from iter-778
        ``coriolis_parameter_fv3``.
    H : jax.Array
        Vertical scale depth (m).
    f_floor : float
        Lower bound on |f| (s⁻¹); default 1e-12.

    Returns
    -------
    L_R : jax.Array
        Rossby radius of deformation (m).
    """
    return n_brunt * H / jnp.maximum(jnp.abs(f), f_floor)


def rhines_scale_fv3(
    u_eddy: jax.Array,
    beta: jax.Array,
    beta_floor: float = 1e-15,
) -> jax.Array:
    """FV3_3D iter 782: Rhines scale L_β = √(U/β).

    Transition length scale (Rhines 1975) at which eddy nonlinear
    advection (∝ U/L) matches β·L Rossby-wave restoring:

        L_β = √(U / β)

    Below L_β: isotropic 2-D turbulence; cascade is dimensional.
    Above L_β: β-effect breaks isotropy → zonal banded jets and
    Rossby waves dominate (the "Rhines β-arrest" of the inverse
    cascade).

    Sets jet-spacing scale on rapidly rotating planets (Jupiter,
    Saturn) and the meridional eddy-mixing length in QG
    turbulence theory.

    Used by: jet-formation criterion in QG turbulence models,
    eddy-permitting ocean parameterization scales, planetary
    rotation effects on cascade, mesoscale energy spectra.

    Composes with iter-780 ``beta_plane_fv3``.

    ``beta_floor`` prevents div-by-0 at the poles (β→0 there).
    Default 1e-15 → polar L_β ≈ √(U/1e-15) ≈ 3.2·10⁷·√U (huge).

    Parameters
    ----------
    u_eddy : jax.Array
        Eddy velocity scale U (m/s).  Typical RMS eddy speed.
    beta : jax.Array
        Meridional Coriolis gradient β = df/dy (s⁻¹·m⁻¹), from
        iter-780 ``beta_plane_fv3``.
    beta_floor : float
        Lower bound on |β| (s⁻¹·m⁻¹).  Default 1e-15.

    Returns
    -------
    L_beta : jax.Array
        Rhines scale (m).
    """
    return jnp.sqrt(jnp.maximum(u_eddy, 0.0) / jnp.maximum(jnp.abs(beta), beta_floor))


def equatorial_rossby_radius_fv3(
    c_wave: jax.Array,
    beta: jax.Array,
    beta_floor: float = 1e-15,
) -> jax.Array:
    """FV3_3D iter 783: equatorial Rossby radius L_eq = √(c/(2·β)).

    Trapping length scale for equatorial waves (Kelvin, equatorial
    Rossby, mixed Rossby-gravity / Yanai, inertia-gravity).  Below
    L_eq from the equator: wave amplitude e-folds Gaussian-like;
    above L_eq: rotational restoring decouples → off-equator modes.

    The mid-latitude radius L_R = N·H/|f| (iter-781) diverges at
    f → 0; the equatorial radius uses β instead of f as the
    rotational scaling:

        L_eq = √(c / (2·β))

    where c = √(g'·H) (shallow-water) or c = N·H/m (baroclinic
    mode) is the gravity-wave speed.

    Typical values:
      * Atmospheric Kelvin wave (c ~ 30 m/s, β_eq ~ 2.29·10⁻¹¹):
        L_eq ≈ √(30/(2·2.29e-11)) ≈ 810 km
      * Tropical baroclinic mode 1 (c ~ 60 m/s): L_eq ≈ 1145 km
      * Ocean baroclinic mode 1 (c ~ 2.7 m/s, equatorial Pacific):
        L_eq ≈ 243 km

    Used by: equatorial wave dispersion analysis (Matsuno 1966
    spectrum), Madden-Julian Oscillation theory, El Niño coupled
    Kelvin-Rossby dynamics, equatorial-trapping diagnosis.

    Composes with iter-780 ``beta_plane_fv3``.  Caller computes c
    from N·H for baroclinic modes, or √(g·H) for external mode.

    ``beta_floor`` clamps |β| > 0 (β never vanishes on Earth except
    in non-rotating limit).

    Parameters
    ----------
    c_wave : jax.Array
        Gravity-wave speed c (m/s).  Caller derives from N·H for
        baroclinic vertical modes, or √(g·H_eff) for shallow-water
        external mode.
    beta : jax.Array
        Meridional Coriolis gradient β = df/dy (s⁻¹·m⁻¹) from
        iter-780 ``beta_plane_fv3``.  Evaluate at the equator
        for canonical trapping scale.
    beta_floor : float
        Lower bound on |β| (s⁻¹·m⁻¹); default 1e-15.

    Returns
    -------
    L_eq : jax.Array
        Equatorial Rossby radius (m).
    """
    return jnp.sqrt(
        jnp.maximum(c_wave, 0.0)
        / (2.0 * jnp.maximum(jnp.abs(beta), beta_floor))
    )


def gravity_wave_speed_fv3(
    n_brunt: jax.Array,
    H: jax.Array,
) -> jax.Array:
    """FV3_3D iter 784: internal gravity-wave phase speed c = N·H.

    Baroclinic mode-1 (equivalent depth) phase speed for a
    stratified column of depth H with mean Brunt-Väisälä
    frequency N:

        c = N · H

    Used as input ``c_wave`` to iter-783 ``equatorial_rossby_radius_fv3``
    and iter-781 ``rossby_radius_fv3`` (the latter via L_R = c/|f|).

    Derivation: for the rigid-lid linear vertical mode m, the
    eigen-phase-speed solves c_m = N·H/(m·π) for hydrostatic
    waves.  m=1 (first baroclinic) → c_1 = N·H/π.  The simpler
    "scale-only" form c = N·H drops the π factor; both
    conventions appear in the literature.  This helper uses the
    π-less form to match the L_R = N·H/|f| convention of
    iter-781.  Callers who want the eigenmode value should
    divide by π.

    Typical values:
      * Atmospheric tropical (N=0.01, H=3 km): c ≈ 30 m/s
        (matches Kelvin wave c).
      * Ocean baroclinic mode 1 (N=0.005, H=540 m): c ≈ 2.7 m/s.
      * Synoptic mid-lat (N=0.01, H=10 km): c ≈ 100 m/s.

    Used by: equatorial-wave dispersion analysis, eddy-resolving
    model design (Δx < L_R/4 where L_R = c/|f|), gravity-wave
    drag spectra, vertical-mode decomposition diagnostics.

    Parameters
    ----------
    n_brunt : jax.Array
        Brunt-Väisälä frequency N (s⁻¹).  Caller computes
        ``jnp.sqrt(jnp.maximum(0, n_sq))`` from iter-772 N².
    H : jax.Array
        Vertical-scale depth (m).

    Returns
    -------
    c : jax.Array
        Internal gravity-wave phase speed (m/s).
    """
    return jnp.maximum(n_brunt, 0.0) * H


def froude_number_fv3(
    u_speed: jax.Array,
    c_wave: jax.Array,
    c_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 785: Froude number Fr = U / c.

    Dimensionless ratio of mean flow speed to gravity-wave speed:

        Fr = U / max(c, c_floor)

    Diagnostic of subcritical vs supercritical regime:

      * Fr < 1   — subcritical: long gravity waves can propagate
                   upstream; smooth-flow regime.
      * Fr = 1   — critical: hydraulic jump / standing wave.
      * Fr > 1   — supercritical: upstream propagation blocked;
                   wave breaking, downslope windstorm, turbulence.

    Compose with iter-784 ``gravity_wave_speed_fv3`` for c = N·H,
    giving Fr = U/(N·H).

    Used by: mountain-wave drag schemes (Lott-Miller 1997 et al.
    parameterize Fr-dependent breaking), downslope-windstorm
    forecasting (Fr > 1 in lee favors windstorms), hydraulic
    flow-regime selection in open-channel and atmospheric
    blocking diagnostics, gravity-wave critical-level analysis.

    ``c_floor`` prevents div-by-0 in unstratified columns (c→0).
    Default 1e-12 → unstratified Fr ≈ U·10¹² (essentially "fully
    supercritical").

    Parameters
    ----------
    u_speed : jax.Array
        Mean flow speed |V| (m/s).  Caller typically passes
        ``wind_speed_fv3(ua, va)``.
    c_wave : jax.Array
        Gravity-wave phase speed c (m/s) from iter-784.
    c_floor : float
        Lower bound on c (m/s); default 1e-12.

    Returns
    -------
    Fr : jax.Array
        Froude number (dimensionless).
    """
    return jnp.maximum(u_speed, 0.0) / jnp.maximum(c_wave, c_floor)


def rossby_number_fv3(
    u_speed: jax.Array,
    L: jax.Array,
    f: jax.Array,
    fL_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 786: Rossby number Ro = U / (|f|·L).

    Dimensionless ratio of inertial to Coriolis acceleration:

        Ro = U / max(|f|·L, fL_floor)

    Diagnostic of rotational regime:

      * Ro ≪ 1   — quasi-geostrophic: Coriolis dominates, geostrophic
                   balance to leading order (synoptic mid-latitudes,
                   ocean mesoscale).
      * Ro ~ 1   — semi-geostrophic: ageostrophic effects O(1) (jet
                   streaks, fronts, atmospheric rivers).
      * Ro ≫ 1   — inertial: rotation negligible (tornadoes, small-
                   scale convective flows, tropical cyclones near
                   eyewall).

    Compose with iter-778 ``coriolis_parameter_fv3`` to derive |f|
    from lat: Ro = U/(2·Ω·sin(lat)·L).

    Used by: regime-selection diagnostics (QG validity check),
    convective-vs-synoptic scale separation, ageostrophic flow
    parameterizations, mesoscale-to-microscale model design.

    ``fL_floor`` clamps |f·L| at the equator and zero-scale limit.

    Note: this is the **inertial** Rossby number Ro = U/(fL).  The
    related but distinct **Burger** number B = (L_R/L)² compares
    Rossby radius to length scale; not implemented here (use
    iter-781 L_R then compute B = (L_R/L)² directly if needed).

    Parameters
    ----------
    u_speed : jax.Array
        Characteristic flow speed (m/s).
    L : jax.Array
        Characteristic horizontal length scale (m).
    f : jax.Array
        Coriolis parameter (s⁻¹) from iter-778.
    fL_floor : float
        Lower bound on |f·L| (s⁻¹·m); default 1e-12.

    Returns
    -------
    Ro : jax.Array
        Rossby number (dimensionless).
    """
    return jnp.maximum(u_speed, 0.0) / jnp.maximum(jnp.abs(f) * L, fL_floor)


def burger_number_fv3(
    L_R: jax.Array,
    L: jax.Array,
    L_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 787: Burger number B = (L_R / L)².

    Dimensionless ratio of Rossby radius to characteristic length
    scale, squared:

        B = (L_R / max(L, L_floor))²

    Diagnostic of QG closure regime:

      * B ≪ 1   — barotropic limit: L ≫ L_R, rotation dominates
                  over stratification, vorticity advection
                  decoupled from interior buoyancy.
      * B ~ 1   — classical QG: comparable scales, geostrophic
                  balance + thermal-wind closure.
      * B ≫ 1   — fully stratified: L ≪ L_R, hydrostatic balance,
                  hydrostatic primitive equations / non-rotating
                  Boussinesq.

    Together with iter-785 ``froude_number_fv3`` and iter-786
    ``rossby_number_fv3``, completes the (Ro, Fr, B) regime-
    selection triplet.  Quasi-geostrophy = Ro ≪ 1 AND B ~ 1.

    Composes with iter-781 ``rossby_radius_fv3`` directly: caller
    computes L_R = N·H/|f| then passes it here.

    Used by: QG validity checks, baroclinic-instability mode
    selection (most unstable at B ~ 1), eddy-resolving model
    design (resolve features with B near unity), mesoscale
    parameterization regime detection.

    Parameters
    ----------
    L_R : jax.Array
        Rossby radius of deformation (m), from iter-781.
    L : jax.Array
        Characteristic horizontal length scale (m).
    L_floor : float
        Lower bound on L (m); default 1e-6.

    Returns
    -------
    B : jax.Array
        Burger number (dimensionless, ≥ 0).
    """
    return (L_R / jnp.maximum(L, L_floor)) ** 2


def ekman_layer_depth_fv3(
    K_v: jax.Array,
    f: jax.Array,
    f_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 788: laminar Ekman boundary-layer depth.

    Penetration depth of the steady, laminar Ekman spiral driven by
    surface stress with vertical eddy viscosity K_v:

        δ_E = √(2·K_v / |f|)

    Ekman spiral within 0 < z < δ_E rotates with depth; below δ_E
    the boundary-layer effect vanishes.  Defined via the e-folding
    decay of the Ekman solution.

    Typical values:
      * Atmospheric PBL (K_v ≈ 10 m²/s, |f|=1e-4): δ_E ≈ 447 m
      * Ocean mixed layer (K_v ≈ 0.01 m²/s, |f|=1e-4): δ_E ≈ 14 m

    Used by: surface-stress / wind-stress curl ocean spinup
    theory, Sverdrup-balance derivations, atmospheric PBL height
    baseline (the unstratified analog of iter-775 PBL height),
    surface drag coefficient calibration.

    Composes iter-778 ``coriolis_parameter_fv3``.

    ``f_floor`` clamps |f| near equator.  At f=0 the steady Ekman
    solution does not exist (depth → ∞); helper returns large but
    finite value.

    Parameters
    ----------
    K_v : jax.Array
        Vertical eddy viscosity (m²/s).
    f : jax.Array
        Coriolis parameter (s⁻¹) from iter-778.
    f_floor : float
        Lower bound on |f| (s⁻¹); default 1e-12.

    Returns
    -------
    delta_E : jax.Array
        Ekman layer depth (m).
    """
    return jnp.sqrt(
        2.0 * jnp.maximum(K_v, 0.0) / jnp.maximum(jnp.abs(f), f_floor)
    )


def ekman_transport_fv3(
    tau_x: jax.Array,
    tau_y: jax.Array,
    f: jax.Array,
    rho: float | jax.Array = 1025.0,
    f_floor: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 789: depth-integrated Ekman volume transport.

    Mass flux of the wind-stress-driven Ekman layer integrated
    vertically (m²/s ≡ m³/(s·m)):

        M_x =   τ_y / (ρ · f)
        M_y = − τ_x / (ρ · f)

    In the Northern Hemisphere (f > 0), Ekman transport is 90°
    to the **right** of the surface stress vector.  In the
    Southern Hemisphere (f < 0), transport is 90° to the **left**.

    Depth-integrated form valid below the Ekman layer δ_E
    (iter-788); independent of K_v.  Foundational for Sverdrup
    balance and wind-stress curl ocean-gyre theory:

        ∇ × (τ / ρf) → vertically averaged geostrophic flow.

    Used by: ocean wind-stress-curl Sverdrup spinup, equatorial
    upwelling diagnosis (divergence of Ekman transport sets
    upwelling), atmospheric surface-momentum-flux budget,
    coastal-upwelling indices (e.g., the Bakun index).

    ``f_floor`` clamps |f| while preserving sign — at f → 0 the
    Ekman theory itself breaks down (transport diverges), so the
    clamp keeps output large-but-finite rather than NaN.

    Parameters
    ----------
    tau_x, tau_y : jax.Array
        Surface wind-stress components (Pa = N/m²).  Zonal /
        meridional convention.
    f : jax.Array
        Coriolis parameter (s⁻¹) from iter-778.
    rho : float or jax.Array
        Reference density (kg/m³).  Default 1025 (ocean
        reference; pass 1.225 for atmosphere).
    f_floor : float
        Lower bound on |f| (s⁻¹); default 1e-12.

    Returns
    -------
    (M_x, M_y) : tuple of jax.Array
        Depth-integrated Ekman volume transport (m²/s).
    """
    f_abs = jnp.maximum(jnp.abs(f), f_floor)
    f_safe = jnp.where(f >= 0, f_abs, -f_abs)
    M_x = tau_y / (rho * f_safe)
    M_y = -tau_x / (rho * f_safe)
    return M_x, M_y


def geostrophic_wind_fv3(
    dphi_dx: jax.Array,
    dphi_dy: jax.Array,
    f: jax.Array,
    f_floor: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 790: geostrophic wind from geopotential gradient.

    Steady, frictionless balance between pressure-gradient and
    Coriolis forces:

        u_g = − (1/f) · ∂Φ/∂y
        v_g = + (1/f) · ∂Φ/∂x

    Equivalent vector form: V_g = (k̂ × ∇Φ) / f.

    In Northern Hemisphere (f > 0), geostrophic flow has high
    pressure (high Φ) on the **right**; in Southern Hemisphere
    (f < 0), high pressure on the **left**.

    Used by: geostrophic adjustment theory, thermal-wind balance
    derivations (∂V_g/∂z from ∂Φ_z/∂x,y), DA observation
    operators (NWP increments are often added to geostrophic
    components), Charney-Eady baroclinic-instability eigenmodes,
    ageostrophic-wind decomposition for jet-streak diagnostics.

    ``f_floor`` clamps |f| with sign preservation — at f = 0 the
    geostrophic approximation itself breaks down (winds diverge);
    helper keeps output large-but-finite rather than NaN.

    Parameters
    ----------
    dphi_dx : jax.Array
        Zonal geopotential gradient ∂Φ/∂x (m·s⁻²).  Caller is
        responsible for differentiating Φ on the cubed-sphere
        cell-center grid (use existing grid operators).
    dphi_dy : jax.Array
        Meridional geopotential gradient ∂Φ/∂y (m·s⁻²).
    f : jax.Array
        Coriolis parameter (s⁻¹) from iter-778.
    f_floor : float
        Lower bound on |f| (s⁻¹); default 1e-12.

    Returns
    -------
    (u_g, v_g) : tuple of jax.Array
        Geostrophic zonal and meridional wind (m/s).
    """
    f_abs = jnp.maximum(jnp.abs(f), f_floor)
    f_safe = jnp.where(f >= 0, f_abs, -f_abs)
    u_g = -dphi_dy / f_safe
    v_g = dphi_dx / f_safe
    return u_g, v_g


def ageostrophic_wind_fv3(
    u: jax.Array,
    v: jax.Array,
    u_g: jax.Array,
    v_g: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 791: ageostrophic wind V_a = V − V_g.

    Decomposes total horizontal wind into geostrophic and
    ageostrophic components:

        u_a = u − u_g
        v_a = v − v_g

    Ageostrophic flow carries the entire dynamical signature of
    departures from geostrophic balance: jet-streak entrance/exit
    quadrant divergence, frontogenetic secondary circulations,
    isallobaric wind (∂p/∂t driven), gravity-wave emission,
    inertial oscillations.

    For QG flow Ro ≪ 1 the ageostrophic wind is O(Ro)·V_g; for
    semi-geostrophic (Ro ~ 1) it becomes comparable.

    Used by: jet-streak quadrant analysis (left-entrance / right-
    exit = divergence aloft → surface lows), Q-vector
    frontogenesis diagnostics, isallobaric-wind plot generation,
    gravity-wave source identification (V_a · ∇V_g term in TKE
    budget), Rossby-wave non-linear cascade decomposition.

    Composes iter-790 ``geostrophic_wind_fv3``.

    Parameters
    ----------
    u, v : jax.Array
        Total horizontal wind components (m/s).
    u_g, v_g : jax.Array
        Geostrophic components (m/s), from iter-790.

    Returns
    -------
    (u_a, v_a) : tuple of jax.Array
        Ageostrophic wind components (m/s).
    """
    return u - u_g, v - v_g


def thermal_wind_fv3(
    dT_dx: jax.Array,
    dT_dy: jax.Array,
    f: jax.Array,
    T_mean: jax.Array,
    f_floor: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 792: thermal wind ∂V_g/∂z from horizontal T gradient.

    Vertical shear of geostrophic wind in z-coordinates with
    hydrostatic balance + ideal-gas EOS:

        ∂u_g/∂z = − (g / (f·T̄)) · ∂T/∂y
        ∂v_g/∂z = + (g / (f·T̄)) · ∂T/∂x

    Equivalent vector form: ∂V_g/∂z = (g/(f·T̄)) · (k̂ × ∇T).

    Physical interpretation: in the Northern Hemisphere with a
    cold pole (∂T/∂y < 0), the geostrophic wind veers eastward
    with height → westerly jet aloft (the classical baroclinic
    mid-latitude jet).  In the SH the sign mirrors.

    Used by: baroclinic jet-stream structure analysis, Eady
    baroclinic-instability eigenmodes, mass-streamfunction
    diagnostics (V_T closes the Stone-Held atmospheric energy
    cycle), thermal-wind balance check (TWB = ∂V_g/∂z vs
    diagnosed shear), front-genesis Q-vector derivation.

    Composes iter-778 ``coriolis_parameter_fv3``.  Caller computes
    horizontal T gradient on the cubed-sphere grid via existing
    operators (e.g. ``divergence_cube_fv3`` or its gradient
    sibling) and passes layer-mean ``T_mean`` between the two
    levels of interest.

    ``f_floor`` clamps |f| **preserving sign** — thermal-wind
    balance breaks down at the equator; helper returns large-but-
    finite output rather than NaN.

    Parameters
    ----------
    dT_dx, dT_dy : jax.Array
        Horizontal temperature gradients on isobaric surface
        (K/m).
    f : jax.Array
        Coriolis parameter (s⁻¹) from iter-778.
    T_mean : jax.Array
        Layer-mean temperature (K) used in the thermal-wind
        denominator.
    f_floor : float
        Lower bound on |f| (s⁻¹); default 1e-12.

    Returns
    -------
    (du_g_dz, dv_g_dz) : tuple of jax.Array
        Vertical shear of geostrophic wind (s⁻¹).
    """
    f_abs = jnp.maximum(jnp.abs(f), f_floor)
    f_safe = jnp.where(f >= 0, f_abs, -f_abs)
    coeff = constants.g / (f_safe * T_mean)
    du_g_dz = -coeff * dT_dy
    dv_g_dz = coeff * dT_dx
    return du_g_dz, dv_g_dz


def eady_growth_rate_fv3(
    n_brunt: jax.Array,
    f: jax.Array,
    du_g_dz: jax.Array,
    dv_g_dz: jax.Array | None = None,
    N_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 793: Eady (1949) maximum baroclinic-instability growth rate.

    Closed-form growth rate of the most-unstable baroclinic mode
    on a flat-bottom uniformly stratified Eady channel:

        σ_Eady = 0.31 · |f| · |∂V_g/∂z| / N

    where:
      * N = √max(0, N²) is the Brunt-Väisälä frequency (iter-772).
      * f is the Coriolis parameter (iter-778).
      * |∂V_g/∂z| = √((∂u_g/∂z)² + (∂v_g/∂z)²) is the magnitude
        of the geostrophic vertical-shear vector (iter-792).

    The 0.31 prefactor is the Eady (1949) eigenvalue 0.3098 (≈
    half the Charney-Stern PV growth-rate envelope), giving the
    cyclogenesis e-folding timescale:

        τ_Eady = 1 / σ_Eady

    Typical mid-lat (N=0.01, |f|=1e-4, |∂V_g/∂z|=3e-3 s⁻¹):
    σ ≈ 9.3·10⁻⁶ s⁻¹ → τ ≈ 1.2 days.

    Used by: baroclinic-storm-track diagnostics (Hoskins-
    Valdes 1990 climatology), cyclogenesis-frequency
    parameterizations, atmospheric blocking-favoring high-σ
    band detection, NAO/AO regime selection (high σ → strong
    eddy-driven jet variability).

    The ``N_floor`` clamp prevents div-by-0 in unstratified
    (neutral) columns where σ → ∞ (the Eady mode formally
    breaks down without stratification).

    Composes iter-772 (caller takes √N²), iter-778 (f), iter-792
    (∂V_g/∂z components).

    Parameters
    ----------
    n_brunt : jax.Array
        Brunt-Väisälä frequency N (s⁻¹) ≥ 0.
    f : jax.Array
        Coriolis parameter (s⁻¹).
    du_g_dz : jax.Array
        Zonal geostrophic vertical shear (s⁻¹).
    dv_g_dz : jax.Array, optional
        Meridional geostrophic vertical shear (s⁻¹).  If None,
        only the zonal component is used (1-D Eady channel).
    N_floor : float
        Lower bound on N (s⁻¹) to avoid div-by-0; default 1e-6.

    Returns
    -------
    sigma : jax.Array
        Eady growth rate (s⁻¹, ≥ 0).
    """
    if dv_g_dz is None:
        shear_mag = jnp.abs(du_g_dz)
    else:
        shear_mag = jnp.sqrt(du_g_dz * du_g_dz + dv_g_dz * dv_g_dz)
    return 0.31 * jnp.abs(f) * shear_mag / jnp.maximum(n_brunt, N_floor)


def absolute_vorticity_fv3(
    zeta_rel: jax.Array,
    f: jax.Array,
) -> jax.Array:
    """FV3_3D iter 794: absolute vorticity η = ζ + f.

    Sum of relative vorticity (vertical component of ∇ × V) and
    planetary vorticity (iter-778 Coriolis parameter):

        η = ζ_rel + f

    Foundation of:
      * Conservation of potential vorticity P = η/h (or η/ρ for
        compressible) on isentropes / Lagrangian parcels (Ertel
        PV theorem).
      * Rossby-wave dispersion ω = U·k − β·k/|k|² + advection of
        relative vorticity by mean flow.
      * Geostrophic adjustment theory (η_g = ∇²Φ/f + f sets the
        balanced PV).
      * Stretching deformation of vortex columns (η/h conserved
        ⇒ stretching → increased ζ).

    For mid-latitude synoptic flow ζ_rel is O(1e-5 to 1e-4 s⁻¹),
    same order as f, so η can change sign locally (anticyclonic
    stratosphere, tropical cyclones).

    Composes iter-778 ``coriolis_parameter_fv3``.

    Parameters
    ----------
    zeta_rel : jax.Array
        Relative vorticity ζ = ∂v/∂x − ∂u/∂y (s⁻¹).  Caller
        derives via existing cubed-sphere differential operators.
    f : jax.Array
        Coriolis parameter (s⁻¹) from iter-778.

    Returns
    -------
    eta : jax.Array
        Absolute vorticity (s⁻¹).
    """
    return zeta_rel + f


def potential_vorticity_ertel_fv3(
    eta_abs: jax.Array,
    rho: jax.Array,
    dtheta_dz: jax.Array,
    rho_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 795: Ertel potential vorticity.

    Conservative quantity on isentropic surfaces under adiabatic,
    frictionless flow (Ertel 1942):

        PV = η · ∂θ/∂z / ρ

    where η = ζ + f (iter-794) is absolute vorticity, ρ is air
    density, and ∂θ/∂z is the vertical gradient of potential
    temperature.

    Standard unit: PVU = 10⁻⁶ m²·K·kg⁻¹·s⁻¹.

    Typical values:
      * Mid-lat troposphere: 0.5–2 PVU
      * Mid-lat lower stratosphere: 4–10 PVU
      * Polar lower stratosphere: > 10 PVU
      * Dynamic tropopause: 2 PVU (Hoskins-McIntyre-Robertson
        1985 convention).

    Used by: stratosphere-troposphere exchange (STE) diagnosis
    (PV > 2 PVU = stratospheric air; PV streamers, cutoffs,
    folds), jet-stream identification (jet core = local PV
    maximum aloft, sharp PV gradient = jet shoulder), atmospheric
    blocking (high-PV-anomaly aloft blocks Rossby-wave train),
    PV-inversion balanced-state diagnostics (extract balanced
    Φ, V from given PV distribution).

    Composes iter-794 ``absolute_vorticity_fv3`` directly.  Caller
    provides ρ (e.g. from iter-734 ``air_density_fv3`` or moist
    EOS density) and ∂θ/∂z (caller computes from θ profile via
    iter-772-style vertical differencing — note: iter-772
    returns N² = (g/θ)·∂θ/∂z, so ∂θ/∂z = N²·θ/g).

    ``rho_floor`` prevents div-by-0 in vacuum / column-top
    extrapolation.

    Parameters
    ----------
    eta_abs : jax.Array
        Absolute vorticity (s⁻¹) from iter-794.
    rho : jax.Array
        Air density (kg/m³).
    dtheta_dz : jax.Array
        Vertical gradient of potential temperature (K/m).
    rho_floor : float
        Lower bound on ρ (kg/m³); default 1e-6.

    Returns
    -------
    pv : jax.Array
        Ertel potential vorticity (m²·K·kg⁻¹·s⁻¹).
    """
    return eta_abs * dtheta_dz / jnp.maximum(rho, rho_floor)


def dynamic_tropopause_fv3(
    pv: jax.Array,
    z: jax.Array,
    pv_thresh: float = 2.0e-6,
) -> jax.Array:
    """FV3_3D iter 796: dynamic tropopause height from PV threshold.

    Hoskins-McIntyre-Robertson (1985) PV-based tropopause: the
    lowest level at which the magnitude of Ertel PV exceeds a
    critical value:

        z_dt = z[k*]   where k* = argmin{ k : |PV(k)| > pv_thresh }

    Canonical ``pv_thresh = 2 PVU = 2·10⁻⁶ m²·K·kg⁻¹·s⁻¹``.
    ``|PV|`` rather than ``PV`` is used so that the Southern-
    Hemisphere stratosphere (PV < −2 PVU) is also captured.

    Vertical axis last; ``z`` and ``pv`` are surface → top
    oriented.  Fallbacks:
      * No crossing in column (deep troposphere): return z[..., −1]
        (top of column).
      * All |PV| > pv_thresh (degenerate; possible near the polar
        vortex with very low tropopause): return z[..., 0].

    Composes iter-795 ``potential_vorticity_ertel_fv3``.

    Used by: stratosphere-troposphere exchange diagnostics
    (depression/lift of z_dt = STE event), upper-tropospheric
    jet diagnostics (z_dt slopes mark jet shoulders), reanalysis
    PV-θ tropopause climatology, ozone-budget tropopause
    crossings.

    JAX-compatible threshold detection via ``jnp.argmax(above,
    axis=-1)`` + ``jnp.any`` + ``jnp.take_along_axis`` — fully
    vmap-compatible.

    Parameters
    ----------
    pv : jax.Array, shape (..., km)
        Ertel PV at each level (m²·K·kg⁻¹·s⁻¹).
    z : jax.Array, shape (..., km)
        Geopotential height above surface at the same levels (m).
    pv_thresh : float
        Critical PV value (m²·K·kg⁻¹·s⁻¹).  Default 2·10⁻⁶.

    Returns
    -------
    z_dt : jax.Array, shape (...,)
        Dynamic-tropopause height (m).
    """
    above = jnp.abs(pv) > pv_thresh
    idx = jnp.argmax(above, axis=-1)
    any_above = jnp.any(above, axis=-1)
    top_idx = pv.shape[-1] - 1
    idx_safe = jnp.where(any_above, idx, top_idx)
    return jnp.take_along_axis(z, idx_safe[..., None], axis=-1)[..., 0]


def lapse_rate_tropopause_fv3(
    t: jax.Array,
    z: jax.Array,
    dT_dz_thresh: float = -2.0e-3,
) -> jax.Array:
    """FV3_3D iter 797: lapse-rate (WMO 1957) thermal tropopause.

    World Meteorological Organization 1957 definition: lowest
    level at which the temperature lapse rate falls to 2 K/km or
    less.  Equivalently: ``dT/dz`` (rising with height) exceeds
    −2·10⁻³ K/m.

    Computes lapse rate at layer midpoints by centered finite
    difference, then finds the first crossing of the threshold:

        z_lrt = z_midpoint[k*]
        where k* = argmin{k : dT/dz(k) > dT_dz_thresh}

    Returns the midpoint z between levels k and k+1.

    Vertical axis last; ``z`` and ``t`` surface→top oriented.
    Fallbacks:
      * No crossing (purely lapse-rate column, no inversion): return
        top midpoint.
      * All above threshold (deep isothermal/inversion column):
        return bottom midpoint.

    Complements iter-796 ``dynamic_tropopause_fv3``: lapse-rate
    tropopause (z_LRT) and dynamic tropopause (z_DT) generally
    differ by 1–3 km in mid-latitudes, with DT preferred in
    upper-troposphere jet diagnostics and LRT preferred in
    radiosonde climatologies.

    JAX-compatible via ``jnp.argmax(above, axis=-1)`` +
    ``jnp.any`` + ``jnp.take_along_axis`` — fully vmap-compatible.
    Same threshold-crossing pattern as iter-775 (PBL height) and
    iter-796 (dynamic tropopause).

    Note: full WMO definition requires the lapse rate to remain
    < 2 K/km for at least 2 km above the candidate level
    ("stability check").  This helper returns the first crossing
    only; caller can apply the 2-km filter externally.

    Parameters
    ----------
    t : jax.Array, shape (..., km)
        Temperature column (K).
    z : jax.Array, shape (..., km)
        Geopotential height column (m).
    dT_dz_thresh : float
        Lapse-rate threshold (K/m).  Default −2·10⁻³ (WMO 1957).

    Returns
    -------
    z_lrt : jax.Array, shape (...,)
        Lapse-rate tropopause height (m, at layer midpoint).
    """
    dT_dz = (t[..., 1:] - t[..., :-1]) / (z[..., 1:] - z[..., :-1])
    above = dT_dz > dT_dz_thresh
    idx = jnp.argmax(above, axis=-1)
    any_above = jnp.any(above, axis=-1)
    top_idx = dT_dz.shape[-1] - 1
    idx_safe = jnp.where(any_above, idx, top_idx)
    z_mid = 0.5 * (z[..., 1:] + z[..., :-1])
    return jnp.take_along_axis(z_mid, idx_safe[..., None], axis=-1)[..., 0]


def cold_point_tropopause_fv3(
    t: jax.Array,
    z: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 798: cold-point tropopause (CPT).

    Level of minimum temperature in the column:

        k_cpt = argmin_k T(k)
        z_cpt = z[k_cpt],  T_cpt = T[k_cpt]

    Tropical-convention tropopause definition.  Sets the
    stratospheric water-vapor entry "cold trap" — the Brewer-
    Dobson circulation lifts air through z_cpt, freeze-drying
    it to T_cpt-saturation values (~3 ppmv at 190 K).

    Used by: tropical tropopause layer (TTL) diagnostics,
    Brewer-Dobson stratospheric water-vapor entry analysis,
    stratospheric ozone-recovery long-term trends (CPT cools
    under increased CO₂ → less H₂O in stratosphere → ozone-
    layer feedback), MJO / cold-point tropopause coupled
    variability.

    Complements:
      * iter-796 ``dynamic_tropopause_fv3``    (PV > 2 PVU)
      * iter-797 ``lapse_rate_tropopause_fv3`` (WMO 1957)

    In the tropics z_CPT > z_LRT > z_DT (often by 1-3 km).  In
    mid-latitudes all three converge.

    Vertical axis last; ``z`` and ``t`` surface → top oriented.
    Returns (z_cpt, T_cpt) tuple for callers that need both the
    height and the cold-point temperature (e.g., saturation-
    mixing-ratio computation for H₂O entry).

    Parameters
    ----------
    t : jax.Array, shape (..., km)
        Temperature column (K).
    z : jax.Array, shape (..., km)
        Geopotential height column (m).

    Returns
    -------
    (z_cpt, T_cpt) : tuple of jax.Array, shape (...,)
        Cold-point height (m) and temperature (K).
    """
    idx = jnp.argmin(t, axis=-1)
    z_cpt = jnp.take_along_axis(z, idx[..., None], axis=-1)[..., 0]
    t_cpt = jnp.take_along_axis(t, idx[..., None], axis=-1)[..., 0]
    return z_cpt, t_cpt


def stratospheric_h2o_entry_fv3(
    t_cpt: jax.Array,
    p_cpt: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 799: stratospheric water-vapor "cold-trap" entry.

    Saturation specific humidity at cold-point tropopause (CPT)
    conditions:

        q_v_strat = q_sat(T_CPT, p_CPT)
        ppmv      = q_v_strat / ε · 10⁶   where ε = R_d / R_v

    Sets the lower bound on stratospheric water vapor through the
    Brewer-Dobson "cold trap" mechanism: tropical tropospheric air
    is freeze-dried at the cold point before entering the
    stratosphere, so q_strat is the saturation-mixing-ratio at the
    CPT.

    Typical tropical CPT (T=190 K, p=100 hPa): q_v ≈ 3 ppmv —
    matches observed stratospheric "background" water-vapor
    concentration (3.5–4 ppmv in lower tropical stratosphere from
    MLS/HALOE).

    Used by: Brewer-Dobson stratospheric H₂O budget, ozone-recovery
    feedback diagnostics (CPT cools under CO₂ increase → ppmv
    drops → less HOₓ → ozone-layer change), CCM/CCMI evaluation
    against ACE-FTS / MLS observations, methane oxidation
    "lower-bound minus 2×CH₄" inversion for entry mixing ratio.

    Composes iter-798 ``cold_point_tropopause_fv3`` (provides
    T_CPT) + canonical ``thermo.saturation_specific_humidity``
    (provides q_sat).

    Returns ``(q_v_strat_kgkg, ppmv)`` tuple — caller picks
    whichever unit fits the diagnostic context.

    Parameters
    ----------
    t_cpt : jax.Array
        Cold-point tropopause temperature (K), from iter-798.
    p_cpt : jax.Array
        Pressure at the cold-point tropopause level (Pa).

    Returns
    -------
    (q_strat_kgkg, q_strat_ppmv) : tuple of jax.Array
        Stratospheric entry mixing ratio (kg/kg and ppmv).
    """
    from legoesm import thermo

    q_sat = thermo.saturation_specific_humidity(t_cpt, p_cpt)
    ppmv = q_sat / constants.epsilon * 1.0e6
    return q_sat, ppmv


def kinetic_energy_fv3(
    ua: jax.Array,
    va: jax.Array,
    w: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 740: kinetic energy per unit mass.

        KE = 0.5 · (ua² + va²)        (2-component, hydrostatic)
        KE = 0.5 · (ua² + va² + w²)   (3-component, non-hydrostatic)

    Standard FV3 pattern used in iter-693 ``nh_total_energy_fv3``
    and many other diagnostics.  ``w=None`` skips the vertical
    component (hydrostatic limit).

    Parameters
    ----------
    ua, va : jax.Array
        Horizontal wind components (m/s).
    w : jax.Array, optional
        Vertical velocity (m/s).  If None, only ua, va contribute.

    Returns
    -------
    ke : jax.Array
        KE per unit mass (m²/s²).
    """
    ke = 0.5 * (ua * ua + va * va)
    if w is not None:
        ke = ke + 0.5 * w * w
    return ke


def specific_volume_fv3(
    delp: jax.Array,
    delz: jax.Array,
) -> jax.Array:
    """FV3_3D iter 739: specific volume α = 1/ρ from delp/delz.

    Companion to iter-734 ``air_density_fv3``:

        α = 1 / ρ = −g · delz / delp

    With FV3 sign convention (delp > 0, delz < 0) → α > 0.

    FV3 fv_mapz.F90:255 uses this directly as "specific volume / g"
    via the line ``delz(i,j,k) = -delz(i,j,k) / delp(i,j,k)``
    (gravity factor absorbed into thermodynamics later).

    Used in pressure-volume work computations and any path that
    needs reciprocal density.

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa, positive).
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3).

    Returns
    -------
    alpha : jax.Array, shape (..., km)
        Specific volume (m³/kg, positive).
    """
    return -constants.g * delz / delp


def air_density_fv3(
    delp: jax.Array,
    delz: jax.Array,
) -> jax.Array:
    """FV3_3D iter 734: air density from hydrostatic balance.

    Faithful port of FV3's standard ρ pattern (used inline at many
    sites in dyn_core.F90, fv_mapz.F90, and iter-725 omega
    diagnostic):

        ρ = −delp / (g · delz)

    With FV3 sign convention (delp > 0, delz < 0), this returns
    a positive density.  Derived from hydrostatic balance
    ``dp/dz = −ρg`` applied to layer thicknesses.

    Used by iter-725 ``omega_diagnostic_fv3`` (ω = −ρgw) and
    by any thermodynamic budget needing column density.

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa, positive).
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3).

    Returns
    -------
    rho : jax.Array, shape (..., km)
        Air density (kg/m³, positive).
    """
    return -delp / (constants.g * delz)


def layer_mean_pressure_fv3(
    delp: jax.Array,
    peln: jax.Array,
) -> jax.Array:
    """FV3_3D iter 732: layer-mean pressure p_f = delp / Δpeln.

    Faithful JAX port of FV3's standard layer-center pressure
    diagnostic (used inline throughout fv_diagnostics.F90,
    fv_mapz.F90, and IC paths):

        p_f[k] = delp[k] / (peln[k+1] − peln[k])

    Integral mean of p over the layer with pressure as the
    independent variable, equivalent to
    ``(p[k+1] − p[k]) / ln(p[k+1]/p[k])``.  This is the
    thermodynamically-consistent "layer-mean pressure" used in
    saturation-mixing-ratio computations, RH, θ_e, etc.

    Pairs with iter-692 ``eqv_pot_fv3``, iter-695 ``rh_calc_fv3``,
    iter-715 ``rh_calc_fv3 do_cmip``, iter-720
    ``saturation_mixing_ratio_blend`` (any diagnostic that needs
    layer-center p from delp + peln).

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    peln : jax.Array, shape (..., km+1)
        log(pressure) at interfaces.

    Returns
    -------
    p_f : jax.Array, shape (..., km)
        Layer-mean (log-p integral-mean) pressure (Pa).
    """
    return delp / (peln[..., 1:] - peln[..., :-1])


def compute_pe_from_delp_fv3(
    delp: jax.Array,
    p_top: float,
) -> jax.Array:
    """FV3_3D iter 731: interface pressure from layer thicknesses.

    Faithful JAX port of FV3's pe accumulation pattern (used at
    multiple sites in dyn_core.F90, fv_mapz.F90, fv_treat_da_inc.F90):

        pe[0] = p_top                          (top of atmosphere)
        pe[k] = pe[k-1] + delp[k-1]            (k = 1..km)

    Vectorized via cumulative sum.  Inverse of the standard
    delp = pe[k+1] − pe[k] computation.

    Used in IC ingestion (when only delp + p_top are known),
    vertical-remap initialization, and any path that builds pe
    incrementally from layer thicknesses.

    Pairs with:
      * iter-724 ``compute_hybrid_pressure_fv3`` (ak/bk/ps → pe)
        — different input
      * iter-722 ``compute_pkz_fv3`` (consumes peln = log(pe))
      * iter-726 ``hydrostatic_delz_fv3`` (consumes pe)

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Layer pressure thickness (Pa, positive).
    p_top : float
        Top-of-atmosphere pressure (Pa).

    Returns
    -------
    pe : jax.Array, shape (..., km+1)
        Interface pressure (Pa, monotone increasing).
    """
    cumsum_delp = jnp.cumsum(delp, axis=-1)
    pe_top = jnp.full_like(delp[..., :1], p_top)
    pe_below = p_top + cumsum_delp
    return jnp.concatenate([pe_top, pe_below], axis=-1)


def dz_from_delz_or_hydrostatic_fv3(
    delz: jax.Array | None = None,
    pt: jax.Array | None = None,
    q: jax.Array | None = None,
    peln: jax.Array | None = None,
    hydrostatic: bool = False,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 730: positive layer thickness dz from delz or hydrostatic.

    Helper extracted from iters 686/687/688/689/707 (supercell suite)
    which all reconstruct positive layer thickness from either FV3
    delz (non-hydrostatic: dz = -delz) or hydrostatic balance:

        Hydrostatic:
            zvir = R_v/R_d − 1  (default)
            dz = (R_d / g) · pt · (1 + zvir · q) · (peln[k+1] − peln[k])

        Non-hydrostatic:
            dz = −delz                          (FV3 delz < 0 ⇒ dz > 0)

    Output is positive thickness suitable for ``compute_zh_above_below_fv3``
    (iter-728) and window-mask computations.

    Parameters
    ----------
    delz : jax.Array, shape (..., km), optional
        Layer thickness (NEGATIVE in FV3) — required if non-hydrostatic.
    pt : jax.Array, shape (..., km), optional
        Temperature (K) — required if hydrostatic.
    q : jax.Array, shape (..., km), optional
        Specific humidity — required if hydrostatic.
    peln : jax.Array, shape (..., km+1), optional
        log(pressure) at interfaces — required if hydrostatic.
    hydrostatic : bool, default False.
    zvir : float, optional
        Virtual-T coefficient.  Default ``R_v/R_d − 1``.

    Returns
    -------
    dz : jax.Array, shape (..., km)
        Positive layer thickness (m).
    """
    if hydrostatic:
        if pt is None or q is None or peln is None:
            raise ValueError("hydrostatic=True requires pt, q, peln")
        if zvir is None:
            zvir = constants.R_v / constants.R_d - 1.0
        rdg = constants.R_d / constants.g
        return rdg * pt * (1.0 + zvir * q) * (peln[..., 1:] - peln[..., :-1])
    else:
        if delz is None:
            raise ValueError("hydrostatic=False requires delz")
        return -delz


def compute_zh_above_below_fv3(
    dz: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 728: cumulative layer-top/bottom heights above surface.

    Helper for window diagnostics (iter-686 UH, iter-687 SRH,
    iter-688 BRN, iter-689 Bunkers, iter-707 SRH-CAPS) that need
    per-layer top + bottom heights measured from the ground.

        zh_above[k] = sum_{j>=k} dz[j]   (top of layer k from surface)
        zh_below[k] = zh_above[k] - dz[k]

    With FV3 dz > 0 (positive layer thickness, top-down): zh_above
    decreases with k; surface (k=km-1) has zh_below = 0.

    Vectorized via cumsum on reversed dz.  No phis input — heights
    measured from a notional ground of z=0.

    Pairs with iter-727 ``compute_zh_from_delz_fv3`` which adds
    surface elevation (phis/g) and uses NEGATIVE delz.

    Parameters
    ----------
    dz : jax.Array, shape (..., km)
        Layer thickness, POSITIVE (= −delz for FV3 sign convention).

    Returns
    -------
    zh_above : jax.Array, shape (..., km)
        Top-of-layer height above surface (m).
    zh_below : jax.Array, shape (..., km)
        Bottom-of-layer height above surface (m).
    """
    dz_reversed = dz[..., ::-1]
    cumsum_from_surface = jnp.cumsum(dz_reversed, axis=-1)
    zh_above = cumsum_from_surface[..., ::-1]
    zh_below = zh_above - dz
    return zh_above, zh_below


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


def lcl_pressure_fv3(
    pt: jax.Array,
    p: jax.Array,
    t_lcl: jax.Array,
    cappa: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 764: pressure at lifting condensation level.

    Dry adiabatic lift from (T, p) to (T_LCL, p_LCL):

        θ conserved  ⇒  T · p^(−κ) = T_LCL · p_LCL^(−κ)
                     ⇒  p_LCL = p · (T_LCL / T)^(1/κ)

    Companion to iter-763 ``lcl_temperature_fv3``.  Together they
    give the full LCL state (T_LCL, p_LCL).

    Parameters
    ----------
    pt : jax.Array
        Parcel temperature (K).
    p : jax.Array
        Parcel pressure (Pa).
    t_lcl : jax.Array
        LCL temperature (K, from iter-763).
    cappa : float or jax.Array, optional.  Default ``constants.kappa``.

    Returns
    -------
    p_lcl : jax.Array
        LCL pressure (Pa, < p since LCL is above parcel).
    """
    kap = constants.kappa if cappa is None else cappa
    return p * (t_lcl / pt) ** (1.0 / kap)


def lcl_height_fv3(
    z_parcel: jax.Array,
    pt: jax.Array,
    t_lcl: jax.Array,
    cp_air: float | jax.Array | None = None,
    g: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 765: geopotential height of LCL.

    On a dry adiabat (θ conserved) under hydrostatic balance with
    ideal-gas EOS:

        dz/dT = (dz/dp) · (dp/dT) = (−RT/(p·g)) · (p/(κT)) = −cp/g

    Integrating from parcel (z, T) up to (z_LCL, T_LCL):

        z_LCL = z + (cp / g) · (T − T_LCL)

    Since T_LCL < T (cooling required), z_LCL > z (LCL above
    parcel).  Companion to iter-763 ``lcl_temperature_fv3`` and
    iter-764 ``lcl_pressure_fv3``; completes the LCL state triad
    (T_LCL, p_LCL, z_LCL) for parcel-lift diagnostics.

    Parameters
    ----------
    z_parcel : jax.Array
        Parcel geopotential height (m).
    pt : jax.Array
        Parcel temperature (K).
    t_lcl : jax.Array
        LCL temperature (K, from iter-763).
    cp_air : float or jax.Array, optional.  Default ``constants.c_pd``.
    g : float or jax.Array, optional.  Default ``constants.g``.

    Returns
    -------
    z_lcl : jax.Array
        LCL geopotential height (m, > z_parcel).
    """
    cp = constants.c_pd if cp_air is None else cp_air
    g_val = constants.g if g is None else g
    return z_parcel + (cp / g_val) * (pt - t_lcl)


def vapor_pressure_from_q_fv3(
    p_mb: jax.Array,
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 766: water-vapor partial pressure from specific humidity.

    Identity from r = q/(1−q) and e/p = r/(ε+r):

        e = p · q / (ε + q·(1−ε))

    where ε = R_d/R_v ≈ 0.622 = ``constants.epsilon``.

    Output units match input pressure units (mb in, mb out).

    Parameters
    ----------
    p_mb : jax.Array
        Pressure (mb).
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    e : jax.Array
        Vapor partial pressure (mb).
    """
    eps = constants.epsilon
    q_safe = jnp.maximum(1e-12, q_sphum)
    return p_mb * q_safe / (eps + q_safe * (1.0 - eps))


def mixing_ratio_fv3(
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 771: mixing ratio from specific humidity.

    Identity:

        r = q / (1 − q)   [kg/kg]

    The mixing ratio is the mass of water vapor per unit mass of
    dry air (r), while the specific humidity is the mass of water
    vapor per unit mass of total moist air (q).  For typical
    atmospheric q ~ 1e-3 to 2e-2 kg/kg, r ≈ q within ~2 %.

    Used by iter-716 ``eqv_pot_bolton_fv3`` (Bolton 1980 θ_e
    formula uses r explicitly), and any future moist-thermo helper
    that needs the dry-air-referenced ratio (e.g. dry static
    energy budgets, condensate mass-mixing).

    Output uses a small positive floor (1e-12 kg/kg) to keep
    downstream ``log`` / ``1/r`` paths finite for vanishing q.

    Parameters
    ----------
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    r : jax.Array
        Mixing ratio (kg/kg, ≥ 1e-12).
    """
    q_safe = jnp.maximum(1e-12, q_sphum)
    return q_safe / (1.0 - q_safe)


def brunt_vaisala_squared_fv3(
    theta: jax.Array,
    q_sphum: jax.Array,
    z: jax.Array,
) -> jax.Array:
    """FV3_3D iter 772: Brunt–Väisälä squared frequency.

    Buoyancy oscillation frequency squared at layer midpoints:

        N² = (g / θ_v) · dθ_v/dz

    where the virtual potential temperature θ_v = θ·(1 + (1/ε − 1)·q)
    is computed via the canonical
    ``legoesm.atmosphere.physics._shared.virtual_temperature``
    helper.

    Sign convention: stable stratification (θ_v increasing with z) →
    N² > 0; unstable (θ_v decreasing with z) → N² < 0; neutral
    → N² = 0.  Used by stability schemes, gravity-wave drag,
    Richardson-number diagnostics.

    Vertical axis is the last dimension.  Output is at layer
    midpoints: shape ``(..., km-1)``.

    Parameters
    ----------
    theta : jax.Array, shape (..., km)
        Potential temperature (K).
    q_sphum : jax.Array, shape (..., km)
        Specific humidity (kg/kg).  Pass ``0.0`` for dry N²
        (then θ_v = θ exactly).
    z : jax.Array, shape (..., km)
        Layer-center geopotential height (m).  Caller is
        responsible for orienting z and θ to match (increasing z
        with array index recommended).

    Returns
    -------
    n_sq : jax.Array, shape (..., km−1)
        N² at layer midpoints (s⁻²).
    """
    from legoesm.atmosphere.physics import _shared

    theta_v = _shared.virtual_temperature(theta, q_sphum)
    dtheta = theta_v[..., 1:] - theta_v[..., :-1]
    dz = z[..., 1:] - z[..., :-1]
    theta_v_mid = 0.5 * (theta_v[..., 1:] + theta_v[..., :-1])
    return constants.g * dtheta / (dz * theta_v_mid)


def static_stability_fv3(
    theta: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """FV3_3D iter 805: pressure-coord static stability parameter.

        S = − ∂θ/∂p     at layer midpoints

    Pressure-coordinate equivalent of iter-772 ``brunt_vaisala_squared_fv3``
    (z-coord): in stable stratification, θ increases with height
    (i.e. as p decreases), so ``−∂θ/∂p > 0`` ⇒ S > 0.

    Computed via centered finite differences across adjacent
    layers; output at layer midpoints, shape ``(..., km−1)``.

    Sign convention:
      * S > 0   — stable (θ increases as p decreases)
      * S = 0   — neutral
      * S < 0   — unstable

    Used by: QG omega equation (Holton 4th ed. eq. 6.30),
    Eady-model eigenvalue derivation (S enters baroclinic-
    instability dispersion relation), pressure-coord
    diagnostics (mass-coord ESM analyses, ERA5 reanalysis
    isobaric-level diagnostics).

    Relation to N² (iter-772): N² = (g²·ρ/θ)·S  via
    hydrostatic ∂p/∂z = −ρg.

    Vertical axis last; ``θ`` and ``p`` surface→top oriented
    (pressure typically decreasing).

    Parameters
    ----------
    theta : jax.Array, shape (..., km)
        Potential temperature (K).
    p : jax.Array, shape (..., km)
        Pressure (Pa) at the same levels.

    Returns
    -------
    S : jax.Array, shape (..., km−1)
        Static stability ``−∂θ/∂p`` (K/Pa).
    """
    dtheta = theta[..., 1:] - theta[..., :-1]
    dp = p[..., 1:] - p[..., :-1]
    return -dtheta / dp


def lapse_rate_moist_fv3(
    t: jax.Array,
    q_sat: jax.Array,
) -> jax.Array:
    """FV3_3D iter 806: moist (saturated) adiabatic lapse rate.

    Bohren-Albrecht (1998) eq. 6.115 / Holton (4th ed.) eq. 2.55:

        Γ_m = g · (1 + L_v · q_sat / (R_d · T))
              ───────────────────────────────────────
              c_p + L_v² · q_sat / (R_v · T²)

    Dry-air limit (q_sat → 0): Γ_m → g / c_p ≈ 9.76 K/km (the
    dry-adiabatic lapse rate Γ_d).  Moist limit: latent-heat
    release damps cooling on ascent → Γ_m < Γ_d.  At T=288 K,
    q_sat ≈ 10 g/kg: Γ_m ≈ 4.5 K/km (textbook tropical moist-
    adiabat).

    Used by: convective parcel-ascent integration (CAPE/CIN
    integrals use Γ_m above LCL), moist-adiabatic CISK / WISHE
    feedback derivations, parcel-buoyancy diagnostics in deep
    convection, mid-tropospheric moist instability indices.

    Caller supplies the saturation specific humidity q_sat
    (typically from canonical
    ``thermo.saturation_specific_humidity`` or the iter-715/720
    CMIP-blended variant).  Helper does **not** invoke the
    saturation curve itself — keeps composition explicit.

    Parameters
    ----------
    t : jax.Array
        Temperature (K).
    q_sat : jax.Array
        Saturation specific humidity (kg/kg).

    Returns
    -------
    gamma_m : jax.Array
        Moist-adiabatic lapse rate (K/m).
    """
    L_v = constants.L_v
    R_d = constants.R_d
    R_v = constants.R_v
    c_p = constants.c_pd
    num = constants.g * (1.0 + L_v * q_sat / (R_d * t))
    den = c_p + L_v * L_v * q_sat / (R_v * t * t)
    return num / den


def parcel_buoyancy_fv3(
    theta_v_parcel: jax.Array,
    theta_v_env: jax.Array,
    theta_v_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 807: parcel buoyancy from virtual-θ contrast.

        b = g · (θ_v_parcel − θ_v_env) / θ_v_env

    Local vertical acceleration of a Lagrangian parcel relative
    to its environment.  Positive b → upward acceleration (warmer
    parcel rises); negative → downward acceleration (cooler
    parcel sinks).

    Used by: CAPE/CIN parcel-ascent integration (∫ b dz from LFC
    to EL = CAPE; from parcel source to LFC = CIN), bulk Richardson
    derivation (Ri_b = g·Δθ_v·Δz/(θ_v·|V|²)), convective trigger
    parameterizations, gravity-wave generation from convection.

    Caller supplies pre-computed θ_v (from
    ``_shared.virtual_temperature``) — buoyancy is canonically
    defined on virtual θ to capture moisture-driven density
    contrast.

    ``theta_v_floor`` prevents div-by-0 in degenerate columns
    (negligible θ_v unphysical).

    Parameters
    ----------
    theta_v_parcel : jax.Array
        Parcel virtual potential temperature (K).
    theta_v_env : jax.Array
        Environment virtual potential temperature (K).
    theta_v_floor : float
        Lower bound on θ_v_env (K); default 1e-12.

    Returns
    -------
    b : jax.Array
        Buoyancy acceleration (m/s²).
    """
    return constants.g * (theta_v_parcel - theta_v_env) / jnp.maximum(
        theta_v_env, theta_v_floor
    )


def cape_column_fv3(
    b: jax.Array,
    z: jax.Array,
) -> jax.Array:
    """FV3_3D iter 808: CAPE column integral from buoyancy profile.

    Column-integrated positive parcel buoyancy:

        CAPE = ∫_LFC^EL max(b, 0) dz

    Approximated as ∑_k max(b_mid(k), 0) · Δz(k) using midpoint
    rule between adjacent levels.  Output unit J/kg.

    Caller supplies the per-level buoyancy ``b`` from iter-807
    ``parcel_buoyancy_fv3(theta_v_parcel, theta_v_env)``.  The
    LFC and EL are implicitly defined as the lower and upper
    boundaries of the positive-buoyancy region(s); negative-
    buoyancy layers contribute zero (clipped at 0).  Caller
    handles CIN separately (negative-buoyancy column-integral).

    Used by: deep-convection forecast diagnostics (CAPE ≥ 1000
    J/kg → moderate, ≥ 2500 → strong, ≥ 5000 → tornado
    environment), GFDL/CAM/IFS convection-scheme triggers,
    SREF/HRRR severe-weather product generation, climate-model
    convective-precipitation diagnostics.

    Vertical axis last; ``b`` and ``z`` are surface→top oriented.
    Output is sum over the column-integral range.  Typical
    tropical CAPE 500–5000 J/kg; mid-latitude severe storms
    1000–4000 J/kg; nocturnal MCS 2000–6000 J/kg.

    Parameters
    ----------
    b : jax.Array, shape (..., km)
        Parcel buoyancy at each level (m/s²) from iter-807.
    z : jax.Array, shape (..., km)
        Geopotential height at the same levels (m).

    Returns
    -------
    cape : jax.Array, shape (...,)
        Convective available potential energy (J/kg).
    """
    b_mid = 0.5 * (b[..., 1:] + b[..., :-1])
    dz = z[..., 1:] - z[..., :-1]
    return jnp.sum(jnp.maximum(b_mid, 0.0) * dz, axis=-1)


def cin_column_fv3(
    b: jax.Array,
    z: jax.Array,
) -> jax.Array:
    """FV3_3D iter 809: convective inhibition column integral.

    Column-integrated magnitude of negative-buoyancy work:

        CIN = − ∫ min(b, 0) dz       (returned as positive J/kg)

    Discretized via midpoint rule:
        CIN = − Σ_k min(b_mid(k), 0) · Δz(k)

    Companion to iter-808 ``cape_column_fv3``.  Negative-
    buoyancy layers (parcel cooler than environment, below LFC
    or in capping inversions) consume parcel kinetic energy on
    rise — caller must supply CIN-worth of KE for the parcel to
    reach the LFC.

    Reported as a positive magnitude per meteorological
    convention (CIN ≥ 0; larger CIN = stronger capping).
    Caller handles CAPE separately via iter-808.

    Used by: convective trigger gates (CIN > 50 J/kg → trigger
    inhibited unless ML kinetic-energy boost), severe-weather
    pre-storm-environment diagnostics (large CIN allows energy
    buildup before "cap break"), MCS / nocturnal-convection
    forecasting, dryline boundary diagnostics.

    Composes iter-807 ``parcel_buoyancy_fv3``.

    Vertical axis last; ``b`` and ``z`` surface→top oriented.

    Parameters
    ----------
    b : jax.Array, shape (..., km)
        Parcel buoyancy at each level (m/s²) from iter-807.
    z : jax.Array, shape (..., km)
        Geopotential height at the same levels (m).

    Returns
    -------
    cin : jax.Array, shape (...,)
        Convective inhibition magnitude (J/kg, ≥ 0).
    """
    b_mid = 0.5 * (b[..., 1:] + b[..., :-1])
    dz = z[..., 1:] - z[..., :-1]
    return -jnp.sum(jnp.minimum(b_mid, 0.0) * dz, axis=-1)


def lifted_index_fv3(
    t_env_500: jax.Array,
    t_parcel_500: jax.Array,
) -> jax.Array:
    """FV3_3D iter 810: Galway (1956) Lifted Index.

        LI = T_env(500 mb) − T_parcel_lifted(500 mb)

    Severe-thunderstorm instability index.  Sign convention:
      * LI > 0   — stable (parcel cooler than env at 500 mb)
      * LI = 0   — neutral
      * LI < 0   — unstable → convection favored
                   * −3 ≥ LI ≥ −5 : moderate instability
                   * −5 ≥ LI ≥ −7 : strong (severe T-storm risk)
                   * LI ≤ −7 : extreme (tornadic / supercell)

    Caller lifts a surface parcel dry-adiabatically to LCL
    (iter-763), then moist-adiabatically (iter-806 Γ_m) to
    500 mb, and passes T_parcel_500 here together with environment
    T at 500 mb.

    Used by: severe-weather watch/warning issuance (NWS SPC
    convective outlooks key off LI thresholds), pre-storm
    environment assessment, complement to CAPE/CIN (LI provides
    single-number stability index that pre-dates CAPE in
    operational use), aviation hazard forecasting (LI < −3
    triggers thunderstorm hazard advisory).

    Composes iter-808 (CAPE) + iter-809 (CIN) to give the
    canonical CAPE-CIN-LI severe-storm diagnostic triplet.

    Parameters
    ----------
    t_env_500 : jax.Array
        Environment temperature at 500 mb (K).
    t_parcel_500 : jax.Array
        Surface-parcel temperature lifted to 500 mb (K).

    Returns
    -------
    li : jax.Array
        Lifted Index (K, negative = unstable).
    """
    return t_env_500 - t_parcel_500


def k_index_fv3(
    t850: jax.Array,
    td850: jax.Array,
    t700: jax.Array,
    td700: jax.Array,
    t500: jax.Array,
) -> jax.Array:
    """FV3_3D iter 811: George (1960) K-index.

        K = (T_850 − T_500) + T_d850 − (T_700 − T_d700)

    All temperatures in K (or °C — output is invariant under K
    ↔ °C shift since only temperature differences enter).

    Severe-thunderstorm probability index combining mid-tropospheric
    lapse rate (T_850 − T_500), low-level moisture content (T_d850),
    and mid-tropospheric dryness (T_700 − T_d700).  High K requires
    steep lapse rate + moist boundary layer + dry middle troposphere
    — the classic Great Plains severe-storm setup.

    NWS / SPC convective-outlook thresholds (US convention):
      * K < 20      — no thunderstorm activity
      * 20 ≤ K < 26 — isolated thunderstorms
      * 26 ≤ K < 31 — widely scattered
      * 31 ≤ K < 36 — scattered
      * K ≥ 36      — numerous

    Caller supplies pre-extracted level temperatures and dew
    points (e.g., via vertical interpolation onto 850/700/500 mb
    surfaces).  Caller computes dew points via iter-767 ``dew_point_fv3``
    from (p, q) at the same levels.

    Used by: NWS / SPC severe-weather diagnostics, ECMWF EPS
    severe-storm probability product, mesoscale model post-
    processing for convective outlooks, climate-model
    severe-storm-frequency studies.

    Complements iter-810 Lifted Index — together (LI, K) span
    the canonical pre-CAPE-era thunderstorm-instability indices
    still in routine operational use.

    Parameters
    ----------
    t850, td850 : jax.Array
        Temperature and dew point at 850 mb (K).
    t700, td700 : jax.Array
        Temperature and dew point at 700 mb (K).
    t500 : jax.Array
        Temperature at 500 mb (K).

    Returns
    -------
    k : jax.Array
        K-index (K; same units as input, but most operational
        products quote °C — caller converts if desired).
    """
    return (t850 - t500) + td850 - (t700 - td700)


def total_totals_fv3(
    t850: jax.Array,
    td850: jax.Array,
    t500: jax.Array,
) -> jax.Array:
    """FV3_3D iter 812: Miller (1972) Total Totals index.

        TT = (T_850 + T_d850) − 2·T_500
           = VT + CT

    Decomposition:
      * VT = T_850 − T_500     (Vertical Totals — mid-trop lapse rate)
      * CT = T_d850 − T_500    (Cross Totals — low-level moisture vs
                               mid-trop temperature)

    Unit-invariant under K ↔ °C shift since both temperatures and
    T_500 enter via differences after expansion:
        TT_K = TT_C + (T_freeze + T_freeze − 2·T_freeze) = TT_C.

    NWS/SPC thresholds:
      * TT < 44      — no thunderstorms
      * 44 ≤ TT < 50 — isolated
      * 50 ≤ TT < 56 — scattered (severe possible)
      * 56 ≤ TT < 60 — numerous severe
      * TT ≥ 60      — very high tornado risk

    Used by: NWS/SPC convective outlooks, ECMWF severe-storm
    products, climatological severe-storm-frequency studies.

    Complements iter-810 LI + iter-811 K — together (LI, K, TT)
    span the classic pre-CAPE-era thunderstorm-instability index
    triplet still in routine operational use.

    Parameters
    ----------
    t850, td850 : jax.Array
        Temperature and dew point at 850 mb.
    t500 : jax.Array
        Temperature at 500 mb.

    All inputs in K (or °C — TT itself is unit-invariant under the
    K/°C shift).

    Returns
    -------
    tt : jax.Array
        Total Totals index.
    """
    return t850 + td850 - 2.0 * t500


def sweat_index_fv3(
    td850_C: jax.Array,
    tt_index: jax.Array,
    u_850_kts: jax.Array,
    u_500_kts: jax.Array,
    dir_850_deg: jax.Array,
    dir_500_deg: jax.Array,
) -> jax.Array:
    """FV3_3D iter 813: Miller (1972) SWEAT severe-weather index.

    Composite severe-weather threat (Severe WEAther Threat):

        SWEAT = 12·Td850 + 20·(TT − 49) + 2·U_850 + U_500
                + 125·(sin(D_500 − D_850) + 0.2)

    Miller's conditional gating per the original paper:
      * 12·Td850 term: 0 if Td850 < 0 °C (no surface dew-point boost).
      * 20·(TT−49) term: 0 if TT < 49 (only above thunderstorm threshold).
      * 125·sin shear term: 0 unless ALL of:
          - 130° ≤ D_850 ≤ 250°   (SE-SW low-level wind)
          - 210° ≤ D_500 ≤ 310°   (S-W mid-level wind)
          - D_500 > D_850          (veering with height)
          - U_850 ≥ 15 kts
          - U_500 ≥ 15 kts

    NWS/SPC operational thresholds:
      * SWEAT < 250  — no severe storms
      * 250-300      — moderately severe T-storm potential
      * 300-400      — strong severe / tornadic potential
      * > 400        — high tornadic threat

    **Units note**: Td850 in °C; U_850/U_500 in **knots**; directions
    in degrees from-north meteorological convention.  Caller must
    convert m/s wind to knots (1 m/s ≈ 1.94 kts).

    Composes iter-812 ``total_totals_fv3``.

    Used by NWS/SPC severe-storm watches/warnings, mesoscale-model
    severe-storm post-processing, climate-model tornadic-environment
    studies.

    Parameters
    ----------
    td850_C : jax.Array
        Dew point at 850 mb (°C).
    tt_index : jax.Array
        Total Totals index (from iter-812).
    u_850_kts, u_500_kts : jax.Array
        Wind speed at 850 and 500 mb (knots).
    dir_850_deg, dir_500_deg : jax.Array
        Wind direction from-north (degrees) at 850 and 500 mb.

    Returns
    -------
    sweat : jax.Array
        SWEAT index (dimensionless).
    """
    term1 = 12.0 * jnp.maximum(td850_C, 0.0)
    term2 = 20.0 * jnp.maximum(tt_index - 49.0, 0.0)
    term3 = 2.0 * u_850_kts
    term4 = u_500_kts
    d_diff_rad = jnp.deg2rad(dir_500_deg - dir_850_deg)
    valid = (
        (dir_850_deg >= 130.0) & (dir_850_deg <= 250.0)
        & (dir_500_deg >= 210.0) & (dir_500_deg <= 310.0)
        & (dir_500_deg > dir_850_deg)
        & (u_850_kts >= 15.0) & (u_500_kts >= 15.0)
    )
    term5 = jnp.where(valid, 125.0 * (jnp.sin(d_diff_rad) + 0.2), 0.0)
    return term1 + term2 + term3 + term4 + term5


def brn_supercell_fv3(
    cape: jax.Array,
    u_shear: jax.Array,
    v_shear: jax.Array,
    ke_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 814: Bulk Richardson Number (supercell BRN).

        BRN = CAPE / (0.5 · |V_shear|²)
            = CAPE / (0.5 · (u_shear² + v_shear²))

    Weisman-Klemp (1982) supercell discriminator.  Composes
    convective potential energy (CAPE, iter-808) with vector
    shear magnitude (typically 0-6 km bulk shear computed as
    mean low-level wind minus mean upper-level wind).

    Thresholds (Weisman-Klemp 1982 / Bunkers et al. 2006):
      * BRN < 10   — too much shear, multicell or splitting cells
      * 10 ≤ BRN ≤ 50 — supercell-favorable regime
      * BRN > 50   — too little shear, ordinary single cells

    Distinct from iter-774 ``bulk_richardson_fv3`` (Vogelezang-
    Holtslag PBL bulk Ri).  Same dimensional form (energy ratio)
    but different physical use — BRN here measures convective
    potential vs shear KE, used in **storm-mode classification**.

    ``ke_floor`` prevents div-by-0 in calm shear environments.

    Used by: SPC supercell-mode discrimination, ensemble
    severe-storm-mode probabilistic forecasts, climatological
    supercell-frequency studies.

    Parameters
    ----------
    cape : jax.Array
        Convective available potential energy (J/kg) from iter-808.
    u_shear, v_shear : jax.Array
        Bulk-shear vector components (m/s) — typically 0-6 km
        mean-wind difference between low and upper layers.
    ke_floor : float
        Lower bound on shear KE (m²/s²); default 1e-6.

    Returns
    -------
    brn : jax.Array
        Bulk Richardson Number for supercell discrimination
        (dimensionless).
    """
    ke = 0.5 * (u_shear * u_shear + v_shear * v_shear)
    return cape / jnp.maximum(ke, ke_floor)


def supercell_composite_fv3(
    cape: jax.Array,
    srh_3km: jax.Array,
    u_shear_6km: jax.Array,
    v_shear_6km: jax.Array,
    bwd_cap: float = 30.0,
) -> jax.Array:
    """FV3_3D iter 815: Thompson Supercell Composite Parameter.

        SCP = (CAPE / 1000) · (SRH_3km / 100) · (BWD_6km / 20)

    where:
      * CAPE  — most-unstable CAPE (J/kg) from iter-808.
      * SRH_3km — 0-3 km storm-relative helicity (m²/s²) from
                  iter-7755 area helicity helpers.
      * BWD_6km — 0-6 km bulk-shear magnitude (m/s):
                  BWD = √(u_shear² + v_shear²), capped at
                  ``bwd_cap`` (default 30 m/s) per Thompson
                  et al. (2003).

    Operational threshold: SCP ≥ 1 → supercell-favorable
    environment.  Higher values indicate stronger supercell
    likelihood; SCP ≥ 5 marks "very high" supercell risk.

    Distinct from iter-814 BRN: SCP is multiplicative (all three
    ingredients needed simultaneously), while BRN is a ratio
    (favors mid-range CAPE/shear balance).  SCP is now preferred
    in SPC mesoanalysis for tornado-day discrimination.

    Composes iter-808 (CAPE) + iter-814 (provides u/v shear in
    same convention) + iter-7755/7825-area helicity helpers.

    Used by: SPC Mesoanalysis Supercell Composite product
    (operational tornado forecasting), HRRR-SREF ensemble
    severe-storm probabilistic forecasts, climatology of
    tornadic environments (Thompson-Edwards 2000 dataset).

    Parameters
    ----------
    cape : jax.Array
        Most-unstable CAPE (J/kg) from iter-808.
    srh_3km : jax.Array
        0-3 km storm-relative helicity (m²/s²).
    u_shear_6km, v_shear_6km : jax.Array
        0-6 km bulk-shear vector components (m/s).
    bwd_cap : float
        Maximum BWD magnitude before capping (m/s); default 30.

    Returns
    -------
    scp : jax.Array
        Supercell Composite Parameter (dimensionless).
    """
    bwd = jnp.sqrt(u_shear_6km * u_shear_6km + v_shear_6km * v_shear_6km)
    bwd_capped = jnp.minimum(bwd, bwd_cap)
    return (cape / 1000.0) * (srh_3km / 100.0) * (bwd_capped / 20.0)


def significant_tornado_parameter_fv3(
    cape: jax.Array,
    srh_1km: jax.Array,
    u_shear_6km: jax.Array,
    v_shear_6km: jax.Array,
    lcl_height: jax.Array,
    bwd_cap: float = 30.0,
) -> jax.Array:
    """FV3_3D iter 816: Thompson Significant Tornado Parameter.

    Fixed-layer STP (Thompson et al. 2003):

        STP = (CAPE / 1500)
            · (SRH_1km / 150)
            · (BWD_6km / 12)
            · max(0, min(1, (2000 − LCL_height) / 1000))

    Composite environment discriminator for **significant**
    tornadoes (EF2+).  Distinct from iter-815 SCP (which targets
    all supercells); STP refines toward strong/violent tornado
    discrimination via:
      * Lower LCL height (→ wider tornadoes, less rear-flank
        evaporative damping)
      * 0-1 km SRH (low-level rotation, not 0-3 km)
      * 0-6 km BWD with /12 (more shear-sensitive than SCP's /20)

    LCL term clamping:
      * LCL > 2000 m → term = 0    (LCL too high for significant)
      * LCL < 1000 m → term = 1    (LCL low enough; saturated)
      * 1000 ≤ LCL ≤ 2000 m → linear

    BWD capped at ``bwd_cap`` = 30 m/s.

    Operational thresholds:
      * STP < 1   — significant-tornado-unfavorable
      * 1 ≤ STP < 3 — moderate significant-tornado risk
      * STP ≥ 3   — high significant-tornado risk
      * STP ≥ 8   — extreme (violent tornado / outbreak)

    Used by: SPC Mesoanalysis STP product, ensemble tornado-
    threat probabilistic forecasts, Thompson-Edwards 2000
    tornado-environment climatology.

    Composes iter-808 (CAPE) + iter-7755-area helicity (SRH_1km)
    + iter-765 (LCL height z_LCL).

    Parameters
    ----------
    cape : jax.Array
        Most-unstable CAPE (J/kg).
    srh_1km : jax.Array
        0-1 km storm-relative helicity (m²/s²).
    u_shear_6km, v_shear_6km : jax.Array
        0-6 km bulk-shear vector components (m/s).
    lcl_height : jax.Array
        Lifting condensation level height (m, AGL).
    bwd_cap : float
        Maximum BWD magnitude (m/s); default 30.

    Returns
    -------
    stp : jax.Array
        Significant Tornado Parameter (dimensionless).
    """
    bwd = jnp.sqrt(u_shear_6km * u_shear_6km + v_shear_6km * v_shear_6km)
    bwd_capped = jnp.minimum(bwd, bwd_cap)
    lcl_term = jnp.clip((2000.0 - lcl_height) / 1000.0, 0.0, 1.0)
    return (
        (cape / 1500.0)
        * (srh_1km / 150.0)
        * (bwd_capped / 12.0)
        * lcl_term
    )


def effective_inflow_layer_fv3(
    cape_profile: jax.Array,
    cin_mag_profile: jax.Array,
    z: jax.Array,
    cape_min: float = 100.0,
    cin_max: float = 250.0,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 817: effective inflow layer (Thompson 2007).

    Range of levels where a lifted parcel can reach the LFC and
    contribute to storm inflow:

        EIL = { k : CAPE(k) ≥ cape_min  AND  |CIN(k)| ≤ cin_max }

    Defaults follow Thompson et al. (2007):
      * ``cape_min = 100 J/kg`` — minimum positive instability.
      * ``cin_max = 250 J/kg`` — maximum convective inhibition
        (in iter-809 positive-magnitude convention).

    Returns ``(z_bot, z_top)`` tuple bounding the lowest and
    highest qualifying levels.  Both ``NaN`` when no level
    satisfies the criteria.

    Used by: SPC Effective-Layer SRH (ESRH) computation
    (replaces fixed-layer 0-3 km SRH), Effective Bulk Wind
    Difference (EBWD = bulk shear over EIL), refined SCP/STP
    composites with effective-layer inputs (operationally
    preferred over fixed-layer in cool-season and elevated-
    convection environments).

    Caller supplies CAPE/|CIN| profiles per level (e.g., from
    multi-source parcel lifting), with z aligned surface→top.

    Parameters
    ----------
    cape_profile : jax.Array, shape (..., km)
        Per-level CAPE (J/kg).
    cin_mag_profile : jax.Array, shape (..., km)
        Per-level CIN **magnitude** (J/kg, ≥ 0).  Pass iter-809
        ``cin_column_fv3`` output convention.
    z : jax.Array, shape (..., km)
        Geopotential height (m), surface→top oriented.
    cape_min : float
        Minimum CAPE threshold (J/kg); default 100.
    cin_max : float
        Maximum |CIN| threshold (J/kg); default 250.

    Returns
    -------
    (z_bot, z_top) : tuple of jax.Array, shape (...,)
        Effective inflow layer bottom and top (m).  NaN where
        no level qualifies.
    """
    valid = (cape_profile >= cape_min) & (cin_mag_profile <= cin_max)
    any_valid = jnp.any(valid, axis=-1)
    idx_bot = jnp.argmax(valid, axis=-1)
    idx_top = valid.shape[-1] - 1 - jnp.argmax(valid[..., ::-1], axis=-1)
    z_bot_raw = jnp.take_along_axis(z, idx_bot[..., None], axis=-1)[..., 0]
    z_top_raw = jnp.take_along_axis(z, idx_top[..., None], axis=-1)[..., 0]
    nan_arr = jnp.full_like(z_bot_raw, jnp.nan)
    z_bot = jnp.where(any_valid, z_bot_raw, nan_arr)
    z_top = jnp.where(any_valid, z_top_raw, nan_arr)
    return z_bot, z_top


def effective_bulk_shear_fv3(
    u: jax.Array,
    v: jax.Array,
    z: jax.Array,
    z_bot: jax.Array,
    z_top: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 818: vector bulk shear over arbitrary layer.

    Linear interpolation of u, v onto ``z_bot`` and ``z_top``,
    then return the vector difference:

        Δu = u(z_top) − u(z_bot)
        Δv = v(z_top) − v(z_bot)

    For use with iter-817 ``effective_inflow_layer_fv3``: caller
    computes effective inflow-layer bounds (z_bot, z_top), then
    passes them here for the EBWD (Effective Bulk Wind Difference)
    that goes into iter-815 SCP / iter-816 STP effective-layer
    variants.

    Generalizes the fixed-layer (0-6 km, 0-3 km, 0-1 km) bulk-
    shear computation to arbitrary [z_bot, z_top] intervals.

    Uses ``jnp.interp`` (clamps extrapolation to edge values).
    ``z`` must be monotone increasing (surface→top oriented).
    NaN ``z_bot`` / ``z_top`` (e.g., from iter-817 no-EIL case)
    propagate to NaN output.

    Caller responsible for vmap-batching across columns; this
    helper is the per-column scalar form.

    Used by: SPC Effective Bulk Wind Difference (EBWD), effective-
    layer SCP / STP composites, customized "MUSAS" effective
    SRH/shear products.

    Parameters
    ----------
    u, v : jax.Array, shape (km,)
        Wind components on column levels (m/s).
    z : jax.Array, shape (km,)
        Geopotential height column (m), monotone increasing.
    z_bot, z_top : jax.Array (scalar or shape () )
        Layer bounds (m).  May be NaN.

    Returns
    -------
    (du, dv) : tuple of jax.Array
        Vector bulk shear over the layer (m/s).
    """
    u_bot = jnp.interp(z_bot, z, u)
    u_top = jnp.interp(z_top, z, u)
    v_bot = jnp.interp(z_bot, z, v)
    v_top = jnp.interp(z_top, z, v)
    return u_top - u_bot, v_top - v_bot


def mean_layer_field_fv3(
    field: jax.Array,
    z: jax.Array,
    z_bot: jax.Array,
    z_top: jax.Array,
    weight_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 821: generic depth-weighted scalar mean over layer.

        ⟨X⟩ = Σ_k X_mid(k) · Δz(k) · mask(z_mid(k)) / Σ_k Δz(k) · mask(z_mid)

    Mask = 1 where ``z_mid`` ∈ [z_bot, z_top], else 0.

    Generic scalar primitive extracted from the iter-819
    ``mean_wind_layer_fv3`` and iter-820 ``mean_layer_temperature_fv3``
    pattern.  Both helpers refactored to delegate to this function.

    Useful for any depth-weighted layer mean:
      * iter-819 mean wind (u, v components via two calls)
      * iter-820 mean T
      * mean specific humidity over a layer
      * mean θ, θ_e, θ_v over a layer
      * mean RH, mean ρ, mean p

    ``weight_floor`` prevents 0/0 when layer falls entirely outside
    the column (output = 0).

    Parameters
    ----------
    field : jax.Array, shape (km,)
        Scalar field on column levels.
    z : jax.Array, shape (km,)
        Geopotential height column (m), monotone increasing.
    z_bot, z_top : jax.Array
        Layer bounds (m).
    weight_floor : float
        Lower bound on Σ weights (m); default 1e-12.

    Returns
    -------
    field_mean : jax.Array
        Depth-weighted mean of ``field`` over the layer.
    """
    z_mid = 0.5 * (z[1:] + z[:-1])
    dz = z[1:] - z[:-1]
    field_mid = 0.5 * (field[1:] + field[:-1])
    inside = (z_mid >= z_bot) & (z_mid <= z_top)
    weights = jnp.where(inside, dz, 0.0)
    total = jnp.maximum(jnp.sum(weights), weight_floor)
    return jnp.sum(field_mid * weights) / total


def mean_wind_layer_fv3(
    u: jax.Array,
    v: jax.Array,
    z: jax.Array,
    z_bot: jax.Array,
    z_top: jax.Array,
    weight_floor: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 819: depth-weighted mean wind over arbitrary layer.

    Midpoint-rule trapezoidal mean over [z_bot, z_top]:

        ū = (∫_{z_bot}^{z_top} u dz) / (z_top − z_bot)

    Discretized as midpoint-weighted sum over the subset of column
    midpoints falling inside the layer bounds:

        ū = Σ_k u_mid(k) · Δz(k) · mask(z_mid(k)) / Σ_k Δz(k) · mask(z_mid(k))

    Mask = 1 where z_mid is in [z_bot, z_top], else 0.  Same for v.

    Used by: Bunkers storm motion (mean 0-6 km wind ± deviation),
    mean-layer wind for parcel deep advection, storm-relative wind
    diagnostics, effective-layer mean wind for inflow trajectory
    composites.

    Generic over arbitrary [z_bot, z_top] — composes with iter-817
    effective inflow layer or any fixed-layer bounds (e.g.,
    0-1 km, 0-6 km).

    ``weight_floor`` prevents 0/0 when the layer falls entirely
    outside the column or when bounds are too tight to enclose any
    midpoint.  In such cases output is essentially 0 (formally
    a small-quotient artifact, but harmless for diagnostic use).

    Parameters
    ----------
    u, v : jax.Array, shape (km,)
        Wind components on column levels (m/s).
    z : jax.Array, shape (km,)
        Geopotential height column (m), monotone increasing.
    z_bot, z_top : jax.Array
        Layer bounds (m).
    weight_floor : float
        Lower bound on Σ weights (m); default 1e-12.

    Returns
    -------
    (u_mean, v_mean) : tuple of jax.Array
        Depth-weighted mean wind components (m/s).
    """
    # iter-821: delegate scalar layer-mean to mean_layer_field_fv3
    u_mean = mean_layer_field_fv3(u, z, z_bot, z_top, weight_floor)
    v_mean = mean_layer_field_fv3(v, z, z_bot, z_top, weight_floor)
    return u_mean, v_mean


def mean_layer_temperature_fv3(
    t: jax.Array,
    z: jax.Array,
    z_bot: jax.Array,
    z_top: jax.Array,
    weight_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 820: depth-weighted mean temperature over arbitrary layer.

    Scalar analog of iter-819 ``mean_wind_layer_fv3``:

        T̄ = Σ_k T_mid(k) · Δz(k) · mask(z_mid(k)) / Σ_k Δz(k) · mask(z_mid)

    Mask = 1 where ``z_mid`` ∈ [z_bot, z_top], else 0.

    Used by: layer-mean stability indices, atmospheric river layer-
    mean T (AR-Cat scale calibration), tropospheric mean temperature
    diagnostics for climate-change attribution (TMT), mean-layer
    saturation lookups.

    ``weight_floor`` prevents 0/0 when layer falls entirely outside
    the column.

    Parameters
    ----------
    t : jax.Array, shape (km,)
        Temperature on column levels (K).
    z : jax.Array, shape (km,)
        Geopotential height column (m), monotone increasing.
    z_bot, z_top : jax.Array
        Layer bounds (m).
    weight_floor : float
        Lower bound on Σ weights (m); default 1e-12.

    Returns
    -------
    t_mean : jax.Array
        Depth-weighted mean temperature (K).
    """
    # iter-821: delegate to mean_layer_field_fv3
    return mean_layer_field_fv3(t, z, z_bot, z_top, weight_floor)


def mixed_layer_height_fv3(
    theta: jax.Array,
    z: jax.Array,
    theta_jump_thresh: float = 0.5,
) -> jax.Array:
    """FV3_3D iter 822: well-mixed layer top via θ-jump detection.

    Find the lowest level where the potential-temperature departure
    from the surface exceeds a threshold:

        h_ML = z[k*]    where k* = argmin{ k : θ(k) − θ(surface) > τ }

    Default τ = 0.5 K (Stull 1988 convention for daytime
    well-mixed PBL).  Variants in the literature use τ ∈ [0.1, 1.5]
    depending on regime.

    Distinct from iter-775 ``pbl_height_fv3`` (Ri-based) — this
    helper uses pure θ-jump detection, complementary in:
      * Convective BL daytime sounding (θ uniform → θ-jump
        well-defined; Ri less reliable).
      * Free-convective PBL where shear is weak.
      * Climate-model output without explicit u, v columns.

    Fallback: if no level exceeds the threshold (deeply mixed
    column extending to the top), return z at the top of the
    column.

    JAX-compatible threshold detection via ``jnp.argmax(above,
    axis=-1)`` + ``jnp.any`` + ``jnp.take_along_axis`` — fully
    vmap-compatible.  Matches the iter-775/796/797 pattern.

    Used by: WRF/HRRR boundary-layer height diagnostic, ARL
    HYSPLIT trajectory model, CAM/GEOS5 dry mixed-layer height
    output, surface-based parcel-source layer estimation for
    CAPE/CIN integrators.

    Parameters
    ----------
    theta : jax.Array, shape (..., km)
        Potential temperature (K), surface→top oriented.
    z : jax.Array, shape (..., km)
        Geopotential height at the same levels (m).
    theta_jump_thresh : float
        θ-departure threshold (K) above surface; default 0.5 K.

    Returns
    -------
    h_ml : jax.Array, shape (...,)
        Mixed-layer top height (m).
    """
    delta_theta = theta - theta[..., 0:1]
    above = delta_theta > theta_jump_thresh
    any_above = jnp.any(above, axis=-1)
    idx = jnp.argmax(above, axis=-1)
    top_idx = theta.shape[-1] - 1
    idx_safe = jnp.where(any_above, idx, top_idx)
    return jnp.take_along_axis(z, idx_safe[..., None], axis=-1)[..., 0]


def eis_fv3(
    theta_700: jax.Array,
    theta_sfc: jax.Array,
    lcl_height: jax.Array,
    gamma_m_850: jax.Array,
    z_700: float = 3000.0,
) -> jax.Array:
    """FV3_3D iter 823: Wood-Bretherton (2006) Estimated Inversion Strength.

        LTS = θ_700 − θ_surf                         (Klein-Hartmann 1993)
        EIS = LTS − Γ_m_850 · (z_700 − LCL)          (Wood-Bretherton 2006)

    Better predictor of stratocumulus cloud fraction than the
    raw LTS, because it removes the moist-adiabatic component of
    the inversion's apparent stability (i.e., a deeper free
    troposphere with strong CC lapse-rate cooling would inflate
    LTS but doesn't really increase the *additional* stability
    above what a moist adiabat from LCL would predict).

    Stratocumulus regimes (Wood-Bretherton 2006, table 1):
      * EIS < 4 K       — trade-Cu / shallow Cu (open ocean)
      * 4 ≤ EIS < 8 K   — transitional / cumulus-under-Sc
      * EIS ≥ 8 K       — well-formed Sc deck (subtropical
                          eastern boundary currents)

    Strongest correlation with low-cloud fraction in the
    Atlantic, Pacific, and Southern Hemisphere Sc decks (r ≈ 0.7
    for monthly-mean Sc fraction vs EIS in CMIP-class GCMs).

    Composes:
      * iter-765 ``lcl_height_fv3``    — provides LCL height
      * iter-806 ``lapse_rate_moist_fv3`` — provides Γ_m at 850 mb

    Caller supplies pre-computed θ_700, θ_surf, LCL, Γ_m_850.

    Parameters
    ----------
    theta_700 : jax.Array
        Potential temperature at 700 mb (K).
    theta_sfc : jax.Array
        Surface potential temperature (K).
    lcl_height : jax.Array
        Lifting condensation level height (m, AGL).
    gamma_m_850 : jax.Array
        Moist-adiabatic lapse rate at 850 mb (K/m).
    z_700 : float
        Reference height of the 700 mb surface (m); default 3000.

    Returns
    -------
    eis : jax.Array
        Estimated Inversion Strength (K).
    """
    lts = theta_700 - theta_sfc
    return lts - gamma_m_850 * (z_700 - lcl_height)


def sc_fraction_eis_fv3(
    eis: jax.Array,
    slope: float = 0.06,
    intercept: float = 0.41,
) -> jax.Array:
    """FV3_3D iter 824: empirical Sc-fraction from EIS.

    Wood-Bretherton (2006) linear fit to ISCCP low-cloud fraction
    vs Estimated Inversion Strength (iter-823):

        f_low = clip(slope · EIS + intercept, 0, 1)

    Default coefficients (slope = 0.06, intercept = 0.41) are the
    Wood-Bretherton 2006 best-fit values across Sc decks (regression
    against 30°S–30°N JJA ISCCP low-cloud-fraction climatology).

    Sc-fraction interpretation:
      * EIS < −7 K  → f_low → 0 (clear sky, deep convective)
      * EIS = 0 K   → f_low ≈ 0.41 (transitional regime)
      * EIS = 8 K   → f_low ≈ 0.89 (Sc deck)
      * EIS > 10 K  → f_low = 1 (saturated; well-formed Sc)

    Empirical fit only — accuracy ±0.15 globally; specific Sc
    decks (California, Peru) may have offsets ±0.1.

    Used by: low-cloud climate-feedback diagnostics (Klein-Hartmann-
    Wood 2017 Annu Rev Earth review), CMIP cloud-fraction
    evaluation, Sc-deck shortwave-feedback decomposition,
    parameterization tuning against ISCCP/MODIS.

    Composes iter-823 ``eis_fv3``.

    Parameters
    ----------
    eis : jax.Array
        Estimated Inversion Strength (K), from iter-823.
    slope : float
        Linear-fit slope (per K); default 0.06.
    intercept : float
        Linear-fit intercept (dimensionless); default 0.41.

    Returns
    -------
    f_low : jax.Array
        Estimated low-cloud fraction (0 ≤ f_low ≤ 1).
    """
    return jnp.clip(slope * eis + intercept, 0.0, 1.0)


def cloud_optical_thickness_fv3(
    lwp: jax.Array,
    r_eff: jax.Array,
    rho_water: float | jax.Array | None = None,
    r_eff_floor: float = 1e-9,
) -> jax.Array:
    """FV3_3D iter 825: Slingo-Mie cloud optical thickness.

    Slingo (1989) bulk Mie approximation:

        τ = 1.5 · LWP / (ρ_water · r_eff)

    where:
      * LWP   — liquid water path (kg/m²; from
                ``precipitable_water_fv3``-style column integral
                of q_l · delp / g)
      * ρ_water — liquid-water density (default
                ``constants.rho_water = 1000`` kg/m³)
      * r_eff — droplet effective radius (m)

    Used by: shortwave-radiation parameterizations (Slingo 1989
    8-stream Mie + delta-Eddington), Sc-deck cloud-optical-
    thickness retrievals (MODIS COT), low-cloud feedback
    decomposition (Stephens 2005).

    Typical Sc deck (LWP = 100 g/m² = 0.1 kg/m², r_eff = 10 μm):
        τ = 1.5 · 0.1 / (1000 · 1e-5) = 15

    Thin cirrus (LWP = 5 g/m², r_eff = 30 μm): τ ≈ 0.25.

    ``r_eff_floor`` prevents div-by-0 for empty cloud cells.

    Parameters
    ----------
    lwp : jax.Array
        Liquid water path (kg/m²).
    r_eff : jax.Array
        Droplet effective radius (m).
    rho_water : float or jax.Array, optional
        Default ``constants.rho_water``.
    r_eff_floor : float
        Lower bound on r_eff (m); default 1e-9.

    Returns
    -------
    tau : jax.Array
        Cloud optical thickness (dimensionless).
    """
    rho_w = constants.rho_water if rho_water is None else rho_water
    r_safe = jnp.maximum(r_eff, r_eff_floor)
    return 1.5 * lwp / (rho_w * r_safe)


def cloud_albedo_two_stream_fv3(
    tau: jax.Array,
    mu_0: jax.Array,
    g: float = 0.85,
    mu_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 826: Coakley-Chylek (1975) two-stream cloud albedo.

    Closed-form two-stream cloud-top SW albedo for non-absorbing
    conservative scattering:

        α = τ · (1 − g) / (2·μ_0 + τ · (1 − g))

    where:
      * τ   — cloud optical thickness (iter-825).
      * μ_0 — cosine of solar zenith angle.
      * g   — Mie asymmetry parameter; default 0.85 for liquid
              water (ice clouds use g ≈ 0.7).

    Limits:
      * τ → 0  → α → 0 (transparent atmosphere).
      * τ → ∞  → α → 1 (totally reflective).
      * μ_0 → 0 → α → 1 (grazing incidence saturates reflection).

    Used by: Sc-deck shortwave-feedback decomposition (Stephens
    2005 sensitivity formulation), McRad / Slingo 1989 radiation
    parameterization, low-cloud climate-feedback Δα/Δτ analysis,
    MODIS broadband albedo retrieval comparison.

    ``mu_floor`` prevents division by zero at the terminator
    (μ_0 → 0).

    Composes iter-825 ``cloud_optical_thickness_fv3``.

    Parameters
    ----------
    tau : jax.Array
        Cloud optical thickness (dimensionless), from iter-825.
    mu_0 : jax.Array
        Cosine of solar zenith angle (0 ≤ μ_0 ≤ 1).
    g : float
        Mie asymmetry parameter; default 0.85 (water clouds).
    mu_floor : float
        Lower bound on μ_0; default 1e-6.

    Returns
    -------
    alpha : jax.Array
        Cloud-top albedo (0 ≤ α ≤ 1).
    """
    factor = tau * (1.0 - g)
    return factor / (2.0 * jnp.maximum(mu_0, mu_floor) + factor)


def shortwave_cloud_forcing_fv3(
    alpha_cloudy: jax.Array,
    alpha_clear: jax.Array,
    s_incident: jax.Array,
) -> jax.Array:
    """FV3_3D iter 827: TOA shortwave cloud-radiative effect (CRE_SW).

        CRE_SW = − S_in · (α_cloudy − α_clear)

    Sign convention: negative ⇒ TOA cooling (typical clouds
    increase albedo, reduce net SW absorption); positive ⇒ TOA
    warming (rare; cloud darker than underlying surface).

    Derivation: TOA net-SW = S_in·(1 − α).  Cloud forcing =
    all-sky − clear-sky = S_in·(α_clear − α_cloudy) = −S_in·Δα.

    Typical magnitudes:
      * Sc deck (α_cloudy ≈ 0.5, α_clear ≈ 0.1, S_in ≈ 200 W/m²
        for stratocumulus latitudes): CRE_SW ≈ −80 W/m².
      * Cirrus (α_cloudy ≈ 0.2, α_clear ≈ 0.1): CRE_SW ≈ −20 W/m².
      * Polar summer with Sc: CRE_SW ≈ −150 W/m² (record values).

    Caller supplies pre-computed α (e.g., iter-826) and incident
    flux S_in (TOA solar × cos(SZA) × eccentricity correction).
    S_in = 0 at night ⇒ CRE_SW = 0.

    Used by: ISCCP/CERES TOA CRE comparison, cloud-feedback
    decomposition (Soden-Held 2006), CMIP CRE bias diagnostics,
    Sc-deck SW radiative-budget closure.

    Composes iter-826 ``cloud_albedo_two_stream_fv3``.

    Parameters
    ----------
    alpha_cloudy : jax.Array
        All-sky cloud albedo (0 ≤ α ≤ 1), from iter-826.
    alpha_clear : jax.Array
        Clear-sky surface albedo (0 ≤ α ≤ 1).
    s_incident : jax.Array
        TOA incident shortwave flux (W/m²).

    Returns
    -------
    cre_sw : jax.Array
        TOA shortwave cloud-radiative effect (W/m²).
    """
    return -s_incident * (alpha_cloudy - alpha_clear)


def longwave_cloud_forcing_fv3(
    t_cloud_top: jax.Array,
    t_sfc: jax.Array,
    emissivity: jax.Array = 1.0,
) -> jax.Array:
    """FV3_3D iter 828: TOA longwave cloud-radiative effect (CRE_LW).

        CRE_LW = ε · σ · (T_sfc⁴ − T_cloud_top⁴)

    Sign convention: positive ⇒ TOA warming (typical clouds trap
    IR that would otherwise escape from the warmer surface);
    zero when cloud is at surface T (e.g. fog) or transparent.

    Derivation: clear-sky outgoing LW ≈ σ·T_sfc⁴ (assumes window
    emission dominates); cloudy outgoing LW = ε·σ·T_cloud_top⁴
    + (1−ε)·σ·T_sfc⁴ (cloud emits its own + transmits surface).
    Difference = ε·σ·(T_sfc⁴ − T_cloud_top⁴).

    Typical magnitudes:
      * Tropical anvil (T_cloud=200 K, T_sfc=300 K, ε=1):
        CRE_LW ≈ 5.67·10⁻⁸·(300⁴−200⁴) ≈ 367 W/m²
      * Mid-lat Sc (T_cloud=280, T_sfc=290, ε=1): CRE_LW ≈ 50 W/m²
      * Thin cirrus (T_cloud=220, T_sfc=300, ε=0.5): CRE_LW ≈ 170 W/m²

    Uses ``constants.sigma_sb`` per CLAUDE.md hygiene.

    Pairs with iter-827 ``shortwave_cloud_forcing_fv3`` to give
    net TOA cloud forcing: CRE_net = CRE_SW + CRE_LW (typically
    near 0 globally — SW cooling balances LW warming, by ~2 W/m²
    net SW dominance in present climate).

    Parameters
    ----------
    t_cloud_top : jax.Array
        Effective cloud-top emission temperature (K).
    t_sfc : jax.Array
        Surface temperature (K).
    emissivity : jax.Array or float
        Cloud LW emissivity (0 ≤ ε ≤ 1); default 1.0 for
        optically thick clouds.  Thin cirrus typically ε ≈ 0.3–0.7.

    Returns
    -------
    cre_lw : jax.Array
        TOA longwave cloud-radiative effect (W/m²).
    """
    return emissivity * constants.sigma_sb * (t_sfc ** 4 - t_cloud_top ** 4)


def effective_radiating_temperature_fv3(
    olr: jax.Array,
    emissivity: jax.Array = 1.0,
    olr_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 829: effective radiating temperature from OLR.

    Inverse of the broadband Stefan-Boltzmann law:

        T_eff = (OLR / (ε · σ))^(1/4)

    Where σ = ``constants.sigma_sb`` (Stefan-Boltzmann).

    Typical values:
      * Earth global mean (OLR ≈ 240 W/m², ε=1): T_eff ≈ 255 K
      * Mars (OLR ≈ 110 W/m²): T_eff ≈ 210 K
      * Venus (OLR ≈ 156 W/m²): T_eff ≈ 227 K (but T_surf=735 K
        due to runaway greenhouse — illustrates greenhouse effect)

    Used by: radiative-budget diagnostics (Earth's T_eff vs T_surf
    gap = greenhouse effect ≈ 33 K), planetary climate comparison,
    inverse-retrieval of effective temperature from broadband
    satellite-OLR products (CERES, ERBE).

    ``olr_floor`` prevents NaN from non-positive OLR.

    Pairs with iter-828 ``longwave_cloud_forcing_fv3``: caller can
    diagnose effective T of cloud top from OLR change.

    Parameters
    ----------
    olr : jax.Array
        Outgoing longwave radiation (W/m²).
    emissivity : jax.Array or float
        Broadband emissivity; default 1.0.
    olr_floor : float
        Lower bound on OLR (W/m²); default 1e-6.

    Returns
    -------
    t_eff : jax.Array
        Effective radiating temperature (K).
    """
    return (
        jnp.maximum(olr, olr_floor)
        / (emissivity * constants.sigma_sb)
    ) ** 0.25


def equilibrium_temperature_fv3(
    s_incident: jax.Array,
    albedo: jax.Array,
    emissivity: jax.Array = 1.0,
    s_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 830: planetary equilibrium temperature.

    Steady-state TOA energy balance:

        absorbed SW    =    emitted LW
        (1 − α)·S_in   =    ε · σ · T_eq⁴

        T_eq = ((1 − α) · S_in / (ε · σ))^(1/4)

    Where:
      * S_in    — mean TOA incident SW (W/m²; for Earth global-
                  mean = S_solar / 4 ≈ 340.25).
      * α       — Bond albedo (planetary).
      * ε       — broadband emissivity (default 1.0).
      * σ       — ``constants.sigma_sb``.

    Typical values (Earth global-mean ε=1):
      * α = 0.30: T_eq ≈ 255 K (canonical "Earth without
        greenhouse" temperature).
      * α = 0.36 (Bond albedo measured by CERES): T_eq ≈ 250 K.
      * Snowball Earth (α=0.6): T_eq ≈ 222 K.

    Climate-sensitivity sketch: ΔT_eq / Δα ≈ −T_eq / (4·(1−α))
    ≈ −1 K per 0.01 reduction in albedo at α=0.3.

    Used by: zero-D climate models, planetary-climate textbook
    derivations, CMIP equilibrium-climate-sensitivity ECS
    decomposition (T_eq baseline), planetary-comparison studies
    (Mars/Venus/exoplanet T_eq tables).

    Inverse of iter-829 ``effective_radiating_temperature_fv3``
    when ε·σ·T⁴ = (1−α)·S_in (planetary equilibrium): the two
    helpers complement.

    ``s_floor`` prevents NaN from non-positive absorbed flux.

    Parameters
    ----------
    s_incident : jax.Array
        Mean TOA incident shortwave (W/m²).
    albedo : jax.Array
        Bond albedo (0 ≤ α ≤ 1).
    emissivity : jax.Array or float
        Broadband emissivity (0 ≤ ε ≤ 1); default 1.0.
    s_floor : float
        Lower bound on absorbed SW (W/m²); default 1e-6.

    Returns
    -------
    t_eq : jax.Array
        Equilibrium temperature (K).
    """
    s_abs = jnp.maximum((1.0 - albedo) * s_incident, s_floor)
    return (s_abs / (emissivity * constants.sigma_sb)) ** 0.25


def planck_feedback_fv3(
    t_eff: jax.Array,
    emissivity: jax.Array = 1.0,
) -> jax.Array:
    """FV3_3D iter 831: Planck climate-feedback parameter.

    Linearized Stefan-Boltzmann response of TOA outgoing LW to a
    uniform vertical surface-air warming:

        d(OLR)/dT = 4 · ε · σ · T_eff³
        λ_Planck  = − d(OLR)/dT   (sign: negative = stabilizing)

    Where:
      * T_eff   — effective radiating temperature (K); see
                  iter-829 ``effective_radiating_temperature_fv3``.
      * ε       — broadband emissivity (default 1.0).
      * σ       — ``constants.sigma_sb``.

    Sign convention: λ_Planck < 0 because warmer surface → more
    outgoing LW → restoring force on TOA energy balance.  Cement
    of the climate-feedback decomposition (Bony et al. 2006,
    Soden-Held 2006).

    Typical values:
      * Earth (T_eff = 255 K, ε = 1): λ_Planck ≈ −3.76 W/m²/K.
      * Cold climate (T_eff = 240 K): λ_Planck ≈ −3.13 W/m²/K.
      * Warm climate (T_eff = 270 K): λ_Planck ≈ −4.46 W/m²/K.

    The Planck response sets the "reference" climate sensitivity:
    no-feedback ΔT for 2×CO₂ forcing (3.7 W/m²) is approximately
    3.7 / |λ_Planck| ≈ 0.98 K.  All other feedbacks (water vapor,
    lapse rate, albedo, cloud) act on top.

    Used by: CMIP feedback-kernel analysis (Soden-Held 2006),
    equilibrium-climate-sensitivity ECS decomposition, emergent-
    constraint diagnostics on Sherwood-Hall-Caldwell (2014)-style
    feedback uncertainty, Bony et al. 2006 cloud-feedback
    framework, AR5/AR6 climate-feedback tables.

    Composes iter-829 ``effective_radiating_temperature_fv3``:
    caller can chain ``λ_Planck(T_eff(OLR))`` from raw OLR.

    Parameters
    ----------
    t_eff : jax.Array
        Effective radiating temperature (K).
    emissivity : jax.Array or float
        Broadband emissivity (0 ≤ ε ≤ 1); default 1.0.

    Returns
    -------
    lam_planck : jax.Array
        Planck feedback parameter (W/m²/K); negative for stable
        Earth-like climates.
    """
    return -4.0 * emissivity * constants.sigma_sb * t_eff ** 3


def clausius_clapeyron_dqdt_fv3(
    t: jax.Array,
    p: jax.Array,
    rh: jax.Array = 1.0,
    t_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 832: Clausius-Clapeyron scaling dq/dT at fixed RH.

    Temperature derivative of specific humidity holding relative
    humidity fixed — canonical water-vapor amplification primitive
    (~7%/K Earth-mean):

        dq/dT |_RH = RH · dq_sat/dT
        dq_sat/dT ≈ q_sat · L_v / (R_v · T²)     (Clausius-Clapeyron)

    Where:
      * q_sat   — canonical saturation mixing ratio via
                  ``thermo.saturation_mixing_ratio(T, p)``.
      * L_v     — ``constants.L_v``.
      * R_v     — ``constants.R_v``.
      * RH      — relative humidity (0 ≤ RH ≤ 1).

    Relative scaling dq/q/dT = L_v / (R_v · T²) ≈ 6.5–7.5 %/K at
    Earth-surface T 273–300 K — the canonical Clausius-Clapeyron
    rate observed in:
      * GCM-mean water-vapor feedback (Held-Soden 2000, IPCC AR4-AR6)
      * Trenberth-Dai 2003 precipitable-water trend (~7%/K)
      * Lenderink-van Meijgaard 2008 super-CC extreme-precip scaling
      * Cloud-resolving model extreme-precip scaling (O'Gorman 2015)
      * Manabe-Wetherald 1967 fixed-RH paradigm

    Composes canonical ``thermo.saturation_mixing_ratio`` per
    CLAUDE.md hygiene — never re-derives Tetens/Magnus/CC.

    Pairs with iter-806 ``lapse_rate_moist_fv3`` (which also uses
    q_sat · L_v / (R_v · T²) inside Γ_m) — both share the CC
    amplification structure.

    Used by: water-vapor-feedback decomposition (Soden-Held 2006
    kernel approach), CMIP precipitable-water-trend diagnostics,
    extreme-precip-scaling-with-T analysis, fixed-RH GCM closure.

    ``t_floor`` prevents NaN at T=0.

    Parameters
    ----------
    t : jax.Array
        Temperature (K).
    p : jax.Array
        Pressure (Pa).
    rh : jax.Array or float
        Relative humidity (0 ≤ RH ≤ 1); default 1.0 (saturated).
    t_floor : float
        Lower bound on T² (K²); default 1e-6.

    Returns
    -------
    dqdt : jax.Array
        dq/dT at fixed RH (kg/kg/K).
    """
    from legoesm import thermo
    q_sat = thermo.saturation_mixing_ratio(t, p)
    t_sq = jnp.maximum(t * t, t_floor)
    return rh * q_sat * constants.L_v / (constants.R_v * t_sq)


def fixed_rh_humidity_change_fv3(
    q_old: jax.Array,
    t_old: jax.Array,
    t_new: jax.Array,
    p: jax.Array,
    q_sat_floor: float = 1e-30,
) -> jax.Array:
    """FV3_3D iter 833: fixed-RH specific humidity projection.

    Under the Manabe-Wetherald 1967 / Held-Soden 2000 fixed-RH
    paradigm, a column warming from T_old → T_new at constant
    pressure preserves relative humidity:

        RH = q / q_sat(T, p)    (held constant)
        ⇒ q_new = q_old · q_sat(T_new, p) / q_sat(T_old, p)

    Where ``q_sat`` is the canonical
    ``thermo.saturation_mixing_ratio(T, p)`` — never re-derives
    Tetens/Magnus/CC per CLAUDE.md hygiene.

    For small ΔT: q_new/q_old ≈ 1 + (L_v / (R_v · T²)) · ΔT,
    matching iter-832 ``clausius_clapeyron_dqdt_fv3``.  This
    helper is the *finite* version — exact across large ΔT (e.g.
    2×CO₂ ~3 K, 4×CO₂ ~6 K), not just first-order CC.

    Used by:
      * Held-Soden 2000 / Soden-Held 2006 fixed-RH water-vapor
        feedback diagnostic (apply uniform ΔT to column, project
        q via this helper, recompute radiation, regress ΔOLR/ΔT).
      * CMIP fixed-SST simulations (uniform-warming column
        projection).
      * AMIP / cfMIP forcing perturbations (q rescaling under +K
        SST anomaly).
      * Idealized 4×CO₂ "Gregory-plot" analysis with fixed-RH
        moisture extrapolation.

    Composes ``thermo.saturation_mixing_ratio`` per CLAUDE.md
    hygiene.

    Pairs with iter-832 ``clausius_clapeyron_dqdt_fv3`` (linear)
    and iter-831 ``planck_feedback_fv3`` (Planck) to form the
    fixed-RH water-vapor-feedback diagnostic triplet.

    ``q_sat_floor`` prevents div-by-0 at very cold T.

    Parameters
    ----------
    q_old : jax.Array
        Initial specific humidity (kg/kg).
    t_old : jax.Array
        Initial temperature (K).
    t_new : jax.Array
        Target temperature (K).
    p : jax.Array
        Pressure (Pa) — held constant across the projection.
    q_sat_floor : float
        Lower bound on q_sat(T_old) (kg/kg); default 1e-30.

    Returns
    -------
    q_new : jax.Array
        Specific humidity (kg/kg) projected onto T_new at fixed RH.
    """
    from legoesm import thermo
    q_sat_old = jnp.maximum(
        thermo.saturation_mixing_ratio(t_old, p), q_sat_floor
    )
    q_sat_new = thermo.saturation_mixing_ratio(t_new, p)
    return q_old * q_sat_new / q_sat_old


def ice_albedo_feedback_fv3(
    s_incident: jax.Array,
    dalpha_dt: jax.Array,
) -> jax.Array:
    """FV3_3D iter 834: surface-albedo (ice-albedo) feedback.

    Linearized TOA-SW response to a uniform surface-warming
    perturbation, mediated by surface-albedo change:

        d(ASR)/dT = − S_in · dα/dT
        λ_α       = − S_in · dα/dT      (W/m²/K)

    Where:
      * S_in      — mean TOA incident SW (W/m²; Earth = 340.25).
      * dα/dT     — planetary Bond-albedo sensitivity to surface
                    warming (1/K).  Almost always negative
                    (warming → ice/snow melt → darker surface).

    Sign convention: positive feedback (λ_α > 0) because dα/dT
    < 0 ⇒ −S_in·(−) = +; warming amplified by albedo loss.  This
    is the canonical Budyko 1969 / Sellers 1969 destabilizing
    feedback — the closure mechanism for the snowball-Earth
    bifurcation in 0-D energy-balance models.

    Typical CMIP values:
      * Global mean (mixed snow/ice/cloud): dα/dT ≈ −0.001 /K
        ⇒ λ_α ≈ +0.34 W/m²/K
      * High-latitude only (NH 60-90°): dα/dT ≈ −0.01 /K
        ⇒ regional λ_α ≈ +3.4 W/m²/K
      * Snowball bifurcation regime: dα/dT ≈ −0.05 /K
        ⇒ λ_α exceeds |λ_Planck| (runaway)

    Closes the **canonical fast-feedback quartet**:
      * iter-831 λ_Planck   = −4·ε·σ·T_eff³     (≈ −3.76)
      * iter-832 λ_WV       (via fixed-RH CC)   (≈ +1.8)
      * (lapse-rate λ_LR)                       (≈ −0.6, future iter)
      * iter-834 λ_α        = −S_in · dα/dT     (≈ +0.34)

    Pairs naturally with iter-826 ``cloud_albedo_two_stream_fv3``
    and iter-827 ``shortwave_cloud_forcing_fv3`` (which share the
    α-sensitivity structure for cloud-feedback decomposition).

    Used by: Budyko 1969 / Sellers 1969 EBM, Held-Soden 2000
    feedback decomposition, CMIP albedo-feedback diagnostic
    (Soden-Held 2006 kernel approach), snowball-Earth bifurcation
    studies (Hoffman-Schrag 2002, Pierrehumbert 2005), Arctic-
    amplification analysis (Hall 2004, Pithan-Mauritsen 2014).

    Parameters
    ----------
    s_incident : jax.Array
        Mean TOA incident shortwave (W/m²).
    dalpha_dt : jax.Array
        Albedo sensitivity to surface warming (1/K); typically
        negative for ice-albedo melt feedback.

    Returns
    -------
    lam_alpha : jax.Array
        Surface-albedo feedback parameter (W/m²/K); positive for
        destabilizing ice/snow melt.
    """
    return -s_incident * dalpha_dt


def lapse_rate_feedback_fv3(
    t_eff: jax.Array,
    dT_atm_mean: jax.Array,
    dT_sfc: jax.Array,
    emissivity: jax.Array = 1.0,
    dT_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 835: lapse-rate climate feedback parameter.

    Soden-Held 2006 lapse-rate feedback — TOA-LW response to the
    *difference* between mid-tropospheric and surface warming
    (i.e. the departure of column warming from uniform Planck):

        λ_LR = − 4·ε·σ·T_eff³ · (ΔT̄_atm − ΔT_sfc) / ΔT_sfc

    Derivation: linearizing OLR = ε·σ·T_eff⁴ at the effective
    radiating level gives dOLR/dT_eff = 4·ε·σ·T_eff³.  The Planck
    feedback assumes ΔT_eff = ΔT_sfc (uniform warming).  Lapse-
    rate feedback corrects for the warming-amplification ratio
    ΔT̄_atm/ΔT_sfc that the actual profile exhibits:

        λ_LR = λ_Planck · (ΔT̄_atm − ΔT_sfc) / ΔT_sfc

    Where iter-831 ``planck_feedback_fv3`` returns λ_Planck.

    Regimes (sign of ΔT̄_atm − ΔT_sfc):
      * Tropics: moist-adiabatic warming amplifies upper-trop
        relative to surface (ΔT̄/ΔT_sfc ≈ 1.4) → λ_LR < 0
        (stabilizing).  Tropical λ_LR ≈ −1.5 W/m²/K.
      * Polar: surface warms faster than column (snow-ice-feedback
        + stable BL trapping); ΔT̄/ΔT_sfc < 1 → λ_LR > 0
        (destabilizing).  Polar λ_LR ≈ +0.5 W/m²/K locally.
      * Global mean (CMIP): tropical-dominated → λ_LR ≈ −0.6 W/m²/K
        (the canonical AR5/AR6 quartet value).

    Closes the **AR5/AR6 fast-feedback quartet**:
      * iter-831 λ_Planck   ≈ −3.76  W/m²/K   (stabilizing)
      * iter-832 λ_WV (CC)  ≈ +1.80  W/m²/K   (destabilizing)
      * iter-834 λ_α        ≈ +0.34  W/m²/K   (destabilizing)
      * iter-835 λ_LR       ≈ −0.60  W/m²/K   (stabilizing)

    Net AR5/AR6: λ_net ≈ −2.2 W/m²/K → ECS ≈ 3.7/|λ_net| ≈ 1.7 K
    before λ_cloud (the major uncertainty source).

    Composes iter-831 ``planck_feedback_fv3``: caller can chain
    ``λ_LR(T_eff, ΔT̄, ΔT_sfc) = λ_Planck(T_eff)·(ΔT̄ − ΔT_sfc)/ΔT_sfc``.

    Used by: Soden-Held 2006 / Held-Soden 2000 kernel decomposition,
    AR5/AR6 climate-feedback tables, Bony et al. 2006 framework,
    polar-amplification analysis (Pithan-Mauritsen 2014 ranking),
    moist-adiabatic atmospheric-warming-pattern studies.

    ``dT_floor`` prevents div-by-0 when ΔT_sfc → 0 (no perturbation).

    Parameters
    ----------
    t_eff : jax.Array
        Effective radiating temperature (K).
    dT_atm_mean : jax.Array
        Mass-weighted column-mean atmospheric warming (K).
    dT_sfc : jax.Array
        Surface warming (K).
    emissivity : jax.Array or float
        Broadband emissivity (0 ≤ ε ≤ 1); default 1.0.
    dT_floor : float
        Lower bound on ΔT_sfc magnitude (K); default 1e-6.

    Returns
    -------
    lam_lr : jax.Array
        Lapse-rate feedback (W/m²/K); negative in tropics,
        positive at poles.
    """
    dT_sfc_safe = jnp.where(
        jnp.abs(dT_sfc) < dT_floor,
        jnp.sign(dT_sfc) * dT_floor + (dT_sfc == 0) * dT_floor,
        dT_sfc,
    )
    lam_planck = -4.0 * emissivity * constants.sigma_sb * t_eff ** 3
    return lam_planck * (dT_atm_mean - dT_sfc) / dT_sfc_safe


def equilibrium_climate_sensitivity_fv3(
    radiative_forcing: jax.Array,
    lam_net: jax.Array,
    lam_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 836: equilibrium climate sensitivity (ECS).

    Linearized energy-balance-model steady-state response of
    global-mean surface temperature to a sustained radiative
    forcing perturbation:

        0 = ΔF + λ_net · ΔT_eq
        ΔT_eq = − ΔF / λ_net = ΔF / |λ_net|

    Where λ_net < 0 for stable Earth-like climate.

    Canonical 2×CO₂ forcing: ΔF ≈ 3.7 W/m² (Myhre 1998 / AR5).
    Canonical CMIP6 |λ_net| ≈ 1.0–1.5 W/m²/K → ECS ≈ 2.5–3.7 K
    (AR6 likely range 2.5–4.0 K).

    Building blocks (sum across the AR5/AR6 quartet + clouds):
      λ_net = λ_Planck + λ_WV + λ_LR + λ_α + λ_cloud
            ≈ (−3.76) + (+1.80) + (−0.60) + (+0.34) + λ_cloud
            ≈ (−2.22) + λ_cloud   [W/m²/K]

      ECS sensitivity to clouds:
        λ_cloud = 0      ⇒ ECS ≈ 1.66 K (pre-cloud)
        λ_cloud = +0.4   ⇒ ECS ≈ 2.03 K (mild positive)
        λ_cloud = +1.0   ⇒ ECS ≈ 3.03 K (strong positive)
        λ_cloud = +1.5   ⇒ ECS ≈ 5.14 K (very high)

    This helper is the *closure* of the feedback decomposition
    chain — composes:
      * iter-831 ``planck_feedback_fv3``
      * iter-832 ``clausius_clapeyron_dqdt_fv3`` (linear WV)
      * iter-833 ``fixed_rh_humidity_change_fv3`` (finite WV)
      * iter-834 ``ice_albedo_feedback_fv3``
      * iter-835 ``lapse_rate_feedback_fv3``
      * (+ user-provided λ_cloud)

    Used by: Gregory plot (Gregory et al. 2004; intercept = ΔF,
    slope = λ_net, x-intercept = ECS), CMIP ECS-uncertainty
    decomposition (Caldwell et al. 2016), Sherwood et al. 2020
    emergent-constraint synthesis, AR5/AR6 climate-feedback
    tables.

    ``lam_floor`` prevents div-by-0 if λ_net → 0 (runaway
    instability — note the runaway condition itself signals
    snowball bifurcation, but the linearized ECS is undefined).

    Parameters
    ----------
    radiative_forcing : jax.Array
        Sustained TOA radiative forcing (W/m²; positive=warming).
    lam_net : jax.Array
        Net feedback parameter (W/m²/K; negative for stable
        Earth-like climate).
    lam_floor : float
        Lower bound on |λ_net|; default 1e-6.

    Returns
    -------
    ecs : jax.Array
        Equilibrium surface-temperature response (K).
    """
    abs_lam = jnp.maximum(jnp.abs(lam_net), lam_floor)
    return radiative_forcing / abs_lam


def transient_climate_response_fv3(
    radiative_forcing: jax.Array,
    lam_net: jax.Array,
    gamma: jax.Array,
    denom_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 837: transient climate response (TCR).

    Two-layer energy-balance-model transient surface response
    (Held et al. 2010, Geoffroy et al. 2013):

        ΔF = (|λ_net| + γ) · ΔT_trans
        TCR = ΔF / (|λ_net| + γ)

    Where γ > 0 is the ocean heat-uptake efficiency (W/m²/K) —
    the rate at which the deep ocean absorbs surface forcing
    proportional to ΔT_surface.  γ acts as an *additional*
    damping term that retards the transient warming below the
    eventual equilibrium value.

    Sign convention: |λ_net| > 0 (we apply absolute value
    internally — λ_net is naturally negative for stable Earth-
    like climate).  γ > 0 always (ocean is a heat sink during
    warming).

    Canonical 2×CO₂ forcing ΔF ≈ 3.7 W/m² (Myhre 1998 / AR5):
      * AR6 |λ_net|=1.4, γ=0.7  ⇒ TCR ≈ 1.76 K
      * Low-sensitivity     1.8, 0.6  ⇒ TCR ≈ 1.54 K
      * High-sensitivity    0.8, 0.9  ⇒ TCR ≈ 2.18 K
      * No ocean uptake γ=0           ⇒ TCR → ECS

    Always TCR < ECS for γ > 0 — ocean uptake delays equilibration.

    The **ratio TCR/ECS = |λ_net| / (|λ_net| + γ)** is the
    "realized warming fraction" — a dimensionless climate-response
    measure used in:
      * Held et al. 2010 fast/slow component decomposition
      * Geoffroy et al. 2013 two-layer model fits
      * CMIP6 transient-vs-equilibrium-sensitivity diagnostics
      * Sherwood et al. 2020 emergent-constraint TCR analysis
      * AR6 likely-range TCR 1.4–2.2 K (matches AR6 quartet+cloud)

    Composes iter-836 ``equilibrium_climate_sensitivity_fv3``:
    in the γ→0 limit they agree.  Caller can compute both from
    the same primitive chain (iter-829 T_eff → quartet → these).

    Used by: CMIP6 1pctCO2 / abrupt-4xCO2 protocol analysis,
    Gregory plot transient regime, AR5/AR6 TCR tables,
    Sherwood et al. 2020 transient-feedback synthesis.

    ``denom_floor`` prevents div-by-0 if |λ_net|+γ → 0
    (instability regime — runaway warming, linear TCR undefined).

    Parameters
    ----------
    radiative_forcing : jax.Array
        Sustained TOA radiative forcing (W/m²; positive=warming).
    lam_net : jax.Array
        Net feedback parameter (W/m²/K; negative for stable
        climate — absolute value taken internally).
    gamma : jax.Array
        Ocean heat-uptake efficiency (W/m²/K; positive).
    denom_floor : float
        Lower bound on (|λ_net|+γ); default 1e-6.

    Returns
    -------
    tcr : jax.Array
        Transient climate response (K).  TCR ≤ ECS for γ ≥ 0.
    """
    denom = jnp.maximum(jnp.abs(lam_net) + gamma, denom_floor)
    return radiative_forcing / denom


def radiative_forcing_co2_fv3(
    co2_ppm: jax.Array,
    co2_ppm_ref: jax.Array = 278.0,
    alpha_myhre: float = 5.35,
    co2_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 838: Myhre 1998 logarithmic CO₂ radiative forcing.

    Empirical IPCC TAR/AR5/AR6 standard for CO₂ radiative forcing:

        ΔF_CO₂ = α_Myhre · ln(C / C_ref)   (W/m²)

    Where:
      * α_Myhre = 5.35 W/m² (Myhre 1998 fit to detailed line-by-line
        RRTM radiative-transfer calculations; AR5 Table 8.SM.1).
      * C_ref   — reference CO₂ (default 278 ppm = AR5 1750-CE
        pre-industrial).

    Canonical values:
      * 2×CO₂ (C=556, C_ref=278): ΔF = 5.35·ln(2) ≈ 3.71 W/m²
        — the AR5 standard forcing entry.
      * 4×CO₂ (C=1112):           ΔF = 5.35·2·ln(2) ≈ 7.42 W/m²
      * Present-day (C=420, C_ref=278): ΔF ≈ 2.21 W/m²
      * 8×CO₂ runaway:            ΔF = 5.35·3·ln(2) ≈ 11.13 W/m²

    Logarithmic dependence reflects line-overlap saturation in
    main 15-μm CO₂ band — additional CO₂ broadens only into the
    line wings (Pierrehumbert 2010 ch. 4, Wilson-Gea-Kiehl 2008).

    Completes the **CO₂ → climate-sensitivity primitive chain**:

        CO₂ ppm                                     (input)
        → iter-838 radiative_forcing_co2_fv3        (ΔF)
        → iter-836 equilibrium_climate_sensitivity  (ΔT_eq)
        → iter-837 transient_climate_response       (ΔT_trans)

    Users can now go from raw ppm to warming entirely in pure JAX.

    Used by: AR5/AR6 forcing tables, simple climate models (FaIR
    v2.0, MAGICC7), AR-WG1 emissions-to-warming pipelines, CMIP6
    forcing-consistent diagnostic computations, integrated-
    assessment models (DICE/PAGE/REMIND).

    Note: ``alpha_myhre`` is the Myhre-1998 empirical fit
    coefficient.  AR6 has minor revisions (Etminan et al. 2016
    nonlinear N₂O-CH₄-CO₂ overlap form) but the simple-log form
    remains the AR5/AR6 default for CO₂-only forcing.

    Parameters
    ----------
    co2_ppm : jax.Array
        Atmospheric CO₂ concentration (ppm).
    co2_ppm_ref : jax.Array or float
        Reference CO₂ (ppm); default 278 (1750-CE pre-industrial).
    alpha_myhre : float
        Myhre-1998 forcing coefficient (W/m²); default 5.35.
    co2_floor : float
        Lower bound on C and C_ref ratios (ppm); default 1e-6
        — prevents ln(0) for hypothetical CO₂-free atmosphere.

    Returns
    -------
    delta_f : jax.Array
        Radiative forcing (W/m²) relative to ``co2_ppm_ref``.
        Positive for C > C_ref (warming).
    """
    c_safe = jnp.maximum(co2_ppm, co2_floor)
    c_ref_safe = jnp.maximum(co2_ppm_ref, co2_floor)
    return alpha_myhre * jnp.log(c_safe / c_ref_safe)


def radiative_forcing_ch4_fv3(
    ch4_ppb: jax.Array,
    ch4_ppb_ref: jax.Array = 722.0,
    alpha_myhre: float = 0.036,
    ch4_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 839: Myhre 1998 √-form CH₄ radiative forcing.

    IPCC TAR/AR5 standard methane radiative-forcing formula (CH₄-
    only branch, neglecting CH₄-N₂O band-overlap correction):

        ΔF_CH₄ = α_Myhre · (√M − √M_ref)    (W/m²)

    Where:
      * α_Myhre  = 0.036 W/m²/√ppb (Myhre 1998 fit to RRTM
        line-by-line spectral calculations; AR5 Table 8.SM.1).
      * M_ref    — reference CH₄ (default 722 ppb = AR5 1750-CE
        pre-industrial).

    Square-root dependence reflects partial band saturation in
    the 7.66-μm CH₄ ν₄ rotation-vibration band — additional CH₄
    broadens line wings sub-logarithmically (weaker than CO₂'s
    full log).

    Canonical values:
      * M_ref = 722 (1750 CE):              ΔF = 0.00 W/m²
      * Present-day M = 1925 (2024):        ΔF ≈ 0.61 W/m²
      * SSP3-7.0 2100: M = 3500:            ΔF ≈ 1.18 W/m²
      * Methane spike M_ref→2·M_ref=1444:   ΔF ≈ 0.40 W/m²

    The CH₄-N₂O overlap correction (Myhre 1998 eqn 2 last term)
    is < 5% in the AR6 likely range — neglected here for clean
    primitive form.  AR6 (Etminan et al. 2016) uses a fitted
    polynomial replacing both Myhre forms; the simple √-form
    remains the AR5 default and provides the cleanest pure-JAX
    composable.

    Composes naturally with the iter-836 ECS / iter-837 TCR
    primitive chain:

        CH₄ ppb (or co-input with CO₂, N₂O)
        → iter-839 ΔF_CH₄    (+ iter-838 ΔF_CO₂)
        → iter-836/837 ΔT_eq / ΔT_trans

    Used by: AR5/AR6 forcing tables (CH₄ row), simple climate
    models (FaIR v2.0, MAGICC7) for non-CO₂ GHG terms, AR-WG1
    multi-gas emissions-to-warming pipelines.

    Note: ``alpha_myhre`` is Myhre-1998 empirical fit; AR6
    Etminan 2016 has minor revisions (typically <3% offset).
    Default 0.036 reproduces AR5 row 8.SM.1 to 2 sig figs.

    Parameters
    ----------
    ch4_ppb : jax.Array
        Atmospheric CH₄ concentration (ppb).
    ch4_ppb_ref : jax.Array or float
        Reference CH₄ (ppb); default 722 (1750-CE pre-industrial).
    alpha_myhre : float
        Myhre-1998 coefficient (W/m²/√ppb); default 0.036.
    ch4_floor : float
        Lower bound on M and M_ref (ppb); default 1e-6 — prevents
        √(negative) for hypothetical CH₄-free atmosphere.

    Returns
    -------
    delta_f : jax.Array
        Radiative forcing (W/m²) relative to ``ch4_ppb_ref``.
        Positive for M > M_ref (warming).
    """
    m_safe = jnp.maximum(ch4_ppb, ch4_floor)
    m_ref_safe = jnp.maximum(ch4_ppb_ref, ch4_floor)
    return alpha_myhre * (jnp.sqrt(m_safe) - jnp.sqrt(m_ref_safe))


def radiative_forcing_n2o_fv3(
    n2o_ppb: jax.Array,
    n2o_ppb_ref: jax.Array = 270.0,
    alpha_myhre: float = 0.12,
    n2o_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 840: Myhre 1998 √-form N₂O radiative forcing.

    IPCC TAR/AR5 standard nitrous-oxide radiative-forcing formula
    (N₂O-only branch, neglecting N₂O-CH₄ band-overlap correction):

        ΔF_N₂O = α_Myhre · (√N − √N_ref)    (W/m²)

    Where:
      * α_Myhre = 0.12 W/m²/√ppb (Myhre 1998 RRTM fit; AR5 8.SM.1).
      * N_ref   — reference N₂O (default 270 ppb = AR5 1750-CE
        pre-industrial).

    Square-root saturation in main 7.78-μm ν₃ band — same form as
    CH₄ (iter-839) but with stronger coefficient (0.12 vs 0.036
    per √ppb) reflecting N₂O's stronger per-molecule absorption.

    Canonical values:
      * N_ref = 270 (1750 CE):              ΔF = 0.00 W/m²
      * Present-day N = 336 (2024):         ΔF ≈ 0.23 W/m²
      * SSP3-7.0 2100: N = 435:             ΔF ≈ 0.53 W/m²

    Completes the **Myhre 1998 trio** for ΔF of all three major
    well-mixed GHG (iter-838 CO₂, iter-839 CH₄, iter-840 N₂O).
    Multi-gas total forcing:

        ΔF_total = ΔF_CO₂ + ΔF_CH₄ + ΔF_N₂O   (+ halocarbons)

    Pairs with iter-836 ECS / iter-837 TCR to enable FaIR/MAGICC-
    class multi-gas climate emulators in pure JAX.

    Note: ``alpha_myhre`` is Myhre 1998 empirical fit; AR6 Etminan
    et al. 2016 polynomial form preferred for accuracy but the
    √-form remains AR5 default.  N₂O-CH₄ overlap < 5% in AR6
    likely range — neglected here.

    Parameters
    ----------
    n2o_ppb : jax.Array
        Atmospheric N₂O concentration (ppb).
    n2o_ppb_ref : jax.Array or float
        Reference N₂O (ppb); default 270 (1750-CE pre-industrial).
    alpha_myhre : float
        Myhre-1998 coefficient (W/m²/√ppb); default 0.12.
    n2o_floor : float
        Lower bound on N and N_ref (ppb); default 1e-6.

    Returns
    -------
    delta_f : jax.Array
        Radiative forcing (W/m²) relative to ``n2o_ppb_ref``.
        Positive for N > N_ref (warming).
    """
    n_safe = jnp.maximum(n2o_ppb, n2o_floor)
    n_ref_safe = jnp.maximum(n2o_ppb_ref, n2o_floor)
    return alpha_myhre * (jnp.sqrt(n_safe) - jnp.sqrt(n_ref_safe))


def solar_forcing_fv3(
    delta_tsi: jax.Array,
    albedo: jax.Array = 0.30,
) -> jax.Array:
    """FV3_3D iter 841: solar (TSI-anomaly) radiative forcing.

    TOA radiative forcing from a perturbation in total solar
    irradiance (TSI), accounting for Earth's disk-to-sphere
    geometric factor and Bond-albedo reflection:

        ΔF_solar = (1 − α) · ΔTSI / 4    (W/m²)

    Where:
      * ΔTSI    — TSI anomaly relative to reference (W/m² at the
                  Earth's orbital radius).
      * α       — planetary Bond albedo (default 0.30).
      * 1/4     — Earth disk-area/surface-area ratio (πR²/4πR²),
                  converting top-of-atmosphere insolation at the
                  sub-solar point to global-mean TOA flux.

    Canonical magnitudes:
      * 11-yr solar cycle (ΔTSI = 1.0 W/m² peak-to-trough):
        ΔF ≈ +0.175 W/m² (at α=0.30, dawn-of-cycle warming).
      * Maunder Minimum (ΔTSI ≈ −1.0 W/m²):
        ΔF ≈ −0.175 W/m² (LIA contribution; modest vs volcanism).
      * Modern minimum (ΔTSI ≈ −0.13 W/m² Solar Cycle 24 minimum):
        ΔF ≈ −0.023 W/m² (negligible vs anthropogenic +3.0).
      * Faint Young Sun 3.8 Ga (ΔTSI ≈ −340 W/m² = −25% S_solar):
        ΔF ≈ −60 W/m² (Sagan-Mullen paradox).

    Composes with iter-836 ECS / iter-837 TCR:

        ΔTSI → iter-841 ΔF_solar
        ΔT_eq  = iter-836 ECS(ΔF_solar, λ_net)
        ΔT_trn = iter-837 TCR(ΔF_solar, λ_net, γ)

    Extends the GHG-forcing trio (iter-838/839/840) to natural-
    forcing agents.  Critical for:
      * Last-millennium solar-variability climate-attribution
        (e.g. PMIP3, Vieira-Solanki 2010 TSI reconstructions).
      * Modern Maunder-Minimum-style "grand-minimum" scenarios.
      * Faint-Young-Sun paradox / Archean climate studies.
      * Detection-and-attribution residual diagnosis (solar
        contribution to historical warming budget).

    Pairs naturally with iter-834 ``ice_albedo_feedback_fv3``
    (both use the same 1−α structure for SW response to surface
    temperature).

    Note: this primitive returns *radiative* forcing (level-of-
    no-perturbation TOA flux change), not *effective* forcing
    (ERF) which would include rapid stratospheric adjustment
    plus a small heating-rate scaling factor.  For ERF use the
    AR6 efficacy factor f_eff ≈ 0.78 multiplicatively.

    Parameters
    ----------
    delta_tsi : jax.Array
        TSI anomaly (W/m²; positive = brighter sun).
    albedo : jax.Array or float
        Planetary Bond albedo (0 ≤ α ≤ 1); default 0.30 (Earth).

    Returns
    -------
    delta_f : jax.Array
        Solar radiative forcing (W/m²); positive = warming for
        ΔTSI > 0 (and α < 1).
    """
    return (1.0 - albedo) * delta_tsi / 4.0


def volcanic_forcing_fv3(
    tau_strat: jax.Array,
    alpha_volc: float = 25.0,
) -> jax.Array:
    """FV3_3D iter 842: stratospheric-volcanic radiative forcing.

    Hansen et al. 2005 / Lacis-Hansen 1992 linear scaling of TOA
    SW radiative forcing with stratospheric aerosol optical depth
    (AOD at 550 nm):

        ΔF_volc = − α_volc · τ_strat    (W/m²)

    Where:
      * τ_strat   — stratospheric aerosol optical depth at 550 nm
                    (dimensionless, typically 0–1).
      * α_volc    — Hansen 2005 forcing-per-AOD coefficient
                    (W/m²); default 25 W/m² (GISS ModelE fit).
                    AR5 uses 21–25; Pinatubo observations support
                    20–28.

    Sign: negative ⇒ stratospheric sulfate aerosols scatter SW
    back to space → planetary cooling.  Symmetric to iter-841
    solar forcing in sign (volcanism is the canonical natural
    *cooling* agent).

    Canonical eruptions:
      | event                  | τ_strat | ΔF (W/m²) |
      |------------------------|---------|-----------|
      | Background (quiescent) | 0.005   | −0.13     |
      | El Chichón 1982        | 0.10    | −2.5      |
      | Pinatubo 1991 peak     | 0.15    | −3.75     |
      | Krakatoa 1883          | 0.20    | −5.0      |
      | Tambora 1815           | 0.50    | −12.5     |
      | Year Without a Summer  | 0.50    | −12.5     |
      | Toba 74 ka (estimate)  | 1.0–3.0 | −25 to −75|

    Composes iter-836 ECS / iter-837 TCR:

        τ_strat → iter-842 ΔF_volc → iter-836/837 ECS/TCR → ΔT

    Closes the **natural-forcing pair** (iter-841 solar + iter-842
    volcanic) — required for CMIP DAMIP attribution experiments
    (Held-Sato-style historical-warming budget decomposition into
    GHG + aerosol + solar + volcanic + internal-variability).

    Used by: CMIP6 historical / hist-volc / hist-nat / DAMIP
    protocols, Hansen-Sato GISS Forcing Reconstruction (Vernier-
    Thomason CALIPSO-OSIRIS AOD products), PMIP last-millennium
    runs (Crowley 2000, Toohey 2017 eVolv2k reconstruction),
    paleoclimate Tambora-Krakatoa attribution (Stoffel et al. 2015),
    volcanic-aerosol-injection geoengineering studies (SAI).

    Note: this primitive returns instantaneous radiative forcing
    at the TOA — not accounting for τ_strat vertical-profile
    variation or wavelength dependence beyond 550 nm.  More
    sophisticated treatments use band-integrated forcing kernels
    (Schmidt et al. 2018) but the linear −25·τ form remains the
    AR5/AR6 standard for first-order multi-gas-forcing pipelines.

    Pairs with iter-826 ``cloud_albedo_two_stream_fv3`` (also a
    SW-scattering primitive) and iter-841 ``solar_forcing_fv3``
    (paired natural-forcing agent).

    Parameters
    ----------
    tau_strat : jax.Array
        Stratospheric aerosol optical depth at 550 nm
        (dimensionless, ≥ 0).
    alpha_volc : float
        Hansen-2005 forcing-per-AOD coefficient (W/m²); default 25.

    Returns
    -------
    delta_f : jax.Array
        Volcanic radiative forcing (W/m²); negative = cooling
        for τ > 0.
    """
    return -alpha_volc * tau_strat


def aerosol_forcing_fv3(
    tau_aero: jax.Array,
    n_cdnc_ratio: jax.Array = 1.0,
    beta_direct: float = 20.0,
    beta_indirect: float = -0.45,
    ratio_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 843: anthropogenic-aerosol direct + indirect forcing.

    Linear-direct + Boucher-Lohmann-1995 log-indirect TOA forcing
    from anthropogenic-aerosol perturbation:

        ΔF_aero_dir   = − β_direct · τ_aero
        ΔF_aero_indir = β_indirect · ln(N_d / N_d_ref)
        ΔF_aero_total = ΔF_aero_dir + ΔF_aero_indir

    Where:
      * τ_aero      — tropospheric anthropogenic AOD (550 nm).
      * N_d/N_d_ref — cloud-droplet-number-concentration ratio
                      (Twomey effect; >1 ⇒ more CCN ⇒ smaller
                      droplets ⇒ brighter clouds).
      * β_direct    — direct-effect coefficient (W/m²); default
                      20 (Charlson-Schwartz 1992, weaker than
                      iter-842 volcanic 25 W/m² because
                      tropospheric AOD has shorter residence time
                      and lower-altitude scattering).
      * β_indirect  — indirect-effect log coefficient (W/m²);
                      default −0.45 (AR5/AR6 ERFaci median).

    Sign convention: both terms typically negative ⇒ aerosol
    cools climate (sulfate dominates anthropogenic burden;
    AR5/AR6 ERFari+aci ≈ −1.1 W/m² central, range −1.7 to −0.4).

    AR6 typical magnitudes:
      | mechanism            | ΔF (W/m²) |
      |----------------------|-----------|
      | ERFari direct        | −0.22     |
      | ERFaci indirect      | −0.84     |
      | ERFari + ERFaci      | −1.06     |
      | range (5–95% CL)     | −2.0 to −0.6 |

    Closes the AR6 anthropogenic-forcing trinity together with:
      * iter-838/839/840 GHG forcing  (CO₂, CH₄, N₂O)  ≈ +3.0 W/m²
      * iter-843 aerosol forcing                       ≈ −1.1 W/m²
      → AR6 anthropogenic net                          ≈ +1.9 W/m²

    Net anthropogenic-only ECS (excluding solar/volcanic):
    1.9 W/m² → iter-836 ECS(λ=−1.4) ≈ 1.36 K (historical-warming
    consistent with AR6).

    Composes iter-836 ECS / iter-837 TCR:

        {τ_aero, N_d ratio} → iter-843 ΔF_aero
        → iter-836/837 ECS/TCR → ΔT

    Pairs with iter-826 ``cloud_albedo_two_stream_fv3`` (shares
    cloud-microphysics SW-scattering structure), iter-841
    ``solar_forcing_fv3``, iter-842 ``volcanic_forcing_fv3``.

    Used by: CMIP6 DAMIP / hist-aer attribution, AeroCom Phase I-III
    intercomparison, AR5/AR6 ERFari and ERFaci tables, Boucher-
    Lohmann 1995 Twomey-effect studies, Bellouin et al. 2020
    aerosol-forcing review.

    ``ratio_floor`` prevents ln(0) for hypothetical zero-CDNC.

    Parameters
    ----------
    tau_aero : jax.Array
        Tropospheric anthropogenic AOD at 550 nm (dimensionless,
        ≥ 0).  Typical present-day global mean: 0.02 (regional
        SE Asia summer ~0.5).
    n_cdnc_ratio : jax.Array or float
        Cloud-droplet-number ratio N_d/N_d_ref (dimensionless).
        Default 1.0 (no indirect effect).  AR6 present-day
        global mean: ~1.2–1.4 anthropogenic perturbation.
    beta_direct : float
        Direct-effect coefficient (W/m²); default 20.
    beta_indirect : float
        Indirect-effect log coefficient (W/m²); default −0.45.
    ratio_floor : float
        Lower bound on N_d/N_d_ref; default 1e-6.

    Returns
    -------
    delta_f : jax.Array
        Total aerosol radiative forcing (W/m²); negative for
        anthropogenic cooling (typical sign).
    """
    ratio_safe = jnp.maximum(n_cdnc_ratio, ratio_floor)
    direct = -beta_direct * tau_aero
    indirect = beta_indirect * jnp.log(ratio_safe)
    return direct + indirect


def gwp_single_decay_fv3(
    rad_eff: jax.Array,
    tau: jax.Array,
    horizon: jax.Array = 100.0,
    agwp_co2: jax.Array = 8.95e-14,
    tau_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 844: single-decay Global Warming Potential.

    IPCC time-integrated GWP for a well-mixed greenhouse gas with
    single-exponential atmospheric decay:

        AGWP_x(H) = A_x · τ_x · (1 − exp(−H/τ_x))      (W/m²·yr per kg)
        GWP_x(H)  = AGWP_x(H) / AGWP_CO2(H)            (dimensionless)

    Where:
      * A_x   — radiative efficiency (W/m²/kg).
      * τ_x   — perturbation lifetime (years).
      * H     — time horizon (years; default 100).
      * AGWP_CO2(H) — reference CO₂ AGWP at horizon H
                      (W/m²·yr per kg).  Bern-CC multi-decay
                      gives 8.95×10⁻¹⁴ at H=100 (AR6 Table 7.SM.7).

    Sign convention: per-kg GWP > 0 for positive-forcing GHG.

    AR6 reference GWP_100 values (this primitive reproduces with
    appropriate (A, τ) pairs):
      | gas    | A (W/m²/kg)    | τ (yr) | GWP_100 |
      |--------|----------------|--------|---------|
      | CH₄    | 3.88×10⁻¹³     | 11.8   | ≈ 27    |
      | N₂O    | 3.03×10⁻¹³     | 109    | ≈ 273   |
      | HFC-23 | 1.91×10⁻¹¹     | 228    | ≈ 14600 |
      | SF₆    | 2.01×10⁻¹¹     | 3200   | ≈ 24300 |

    Single-decay approximation neglects CO₂'s multi-mode Bern-CC
    decay (with permanent ~22% fraction).  For non-CO₂ gases the
    single-decay form is exact (single chemical sink); for CO₂
    use AGWP_CO2 reference value directly — do not call this
    helper with CO₂'s effective lifetime.

    Closes the emissions → equivalent-CO₂ chain for policy
    applications:

        emission flux (kg/yr) × GWP_H → CO₂-eq emission (kg-CO₂-eq/yr)

    Composes naturally with the forcing primitives:
      * iter-838/839/840 give *instantaneous* ΔF from concentrations
      * iter-844 gives *time-integrated* per-emission contribution

    Used by: IPCC AR5/AR6 Tables 7.SM (radiative-efficiency tables),
    UNFCCC national-inventory reporting (GWP_100 used for CO₂-eq),
    integrated-assessment models (DICE/PAGE/REMIND emission
    weighting), GTP (Global Temperature Potential, ratio with
    impulse-response function), GWP* / sustained-emission-GWP
    alternatives (Allen et al. 2016, Lynch et al. 2020).

    ``tau_floor`` prevents div-by-0 in H/τ ratio at τ→0.

    Parameters
    ----------
    rad_eff : jax.Array
        Per-mass radiative efficiency A_x (W/m²/kg).
    tau : jax.Array
        Perturbation lifetime τ_x (years).
    horizon : jax.Array or float
        Time horizon H (years); default 100.
    agwp_co2 : jax.Array or float
        Reference AGWP_CO2 at horizon H (W/m²·yr per kg);
        default 8.95e-14 (AR6 H=100).
    tau_floor : float
        Lower bound on τ (years); default 1e-6.

    Returns
    -------
    gwp : jax.Array
        Dimensionless GWP relative to CO₂ at horizon H.
    """
    tau_safe = jnp.maximum(tau, tau_floor)
    agwp_x = rad_eff * tau_safe * (1.0 - jnp.exp(-horizon / tau_safe))
    return agwp_x / agwp_co2


def gtp_single_decay_fv3(
    rad_eff: jax.Array,
    tau: jax.Array,
    horizon: jax.Array = 100.0,
    agtp_co2: jax.Array = 6.84e-16,
    c_climate: float = 0.631,
    d_climate: float = 8.4,
    tau_floor: float = 1e-6,
    diff_floor: float = 1e-9,
) -> jax.Array:
    """FV3_3D iter 845: single-decay Global Temperature Potential.

    Shine et al. 2005 instantaneous-temperature-response metric
    for a well-mixed GHG with single-exponential atmospheric
    decay convolved with a single-mode climate-response IRF:

        AGTP_x(H) = (A_x · c · τ_x / (τ_x − d)) · (exp(−H/τ_x) − exp(−H/d))
        GTP_x(H)  = AGTP_x(H) / AGTP_CO2(H)            (dimensionless)

    Where:
      * A_x   — radiative efficiency (W/m²/kg).
      * τ_x   — atmospheric perturbation lifetime (years).
      * c     — climate sensitivity (K per W/m²); default 0.631
                (Shine et al. 2005 fast-mode coefficient).
      * d     — climate-response timescale (years); default 8.4
                (Shine et al. 2005 fast mode).
      * H     — time horizon (years; default 100).
      * AGTP_CO2(H) — reference CO₂ AGTP at horizon H (K per kg).
                Default 6.84×10⁻¹⁶ (Shine et al. 2005 / AR5
                Table 8.SM.16 at H=100).

    Derivation: AGTP = ∫₀^H A·R_x(t')·IRF_T(H−t') dt' with
    R_x(t)=exp(−t/τ_x) and IRF_T(s)=(c/d)·exp(−s/d) collapses to
    the closed-form above.

    Sign: positive GTP for warming GHG, like GWP.

    AR6 reference GTP_100 values (this primitive reproduces with
    appropriate (A, τ) pairs):
      | gas    | A (W/m²/kg)    | τ (yr) | GTP_100 (AR6) |
      |--------|----------------|--------|---------------|
      | CH₄    | 3.88×10⁻¹³     | 11.8   | ≈ 4.7         |
      | N₂O    | 3.03×10⁻¹³     | 109    | ≈ 233         |
      | HFC-23 | 1.91×10⁻¹¹     | 228    | ≈ 12 400      |

    **GTP vs GWP**: GTP focuses on temperature response at a
    specific horizon (policy-relevant for end-state warming),
    while iter-844 GWP integrates radiative forcing over the
    horizon.  For short-lived gases (e.g. CH₄) GTP_100 << GWP_100
    because most of the forcing happened decades earlier and the
    temperature signal has substantially decayed by year 100.
    For long-lived gases (CO₂, N₂O) the two metrics converge.

    Composes naturally with iter-844 ``gwp_single_decay_fv3``:
    the two together span the **emissions-metric primitive pair**
    (time-integrated forcing vs end-state temperature).

    Used by: Shine et al. 2005 / 2007 GTP introduction, AR5/AR6
    Tables 7.SM (GTP_50, GTP_100), Tanaka-O'Neill 2018 policy-
    metric review, GTP* variants, EU Commission emissions-pricing
    proposals, IAM long-horizon mitigation analysis.

    Closed-form numerically singular when τ_x = d (resonance);
    ``diff_floor`` guards against this edge case.

    Parameters
    ----------
    rad_eff : jax.Array
        Per-mass radiative efficiency A_x (W/m²/kg).
    tau : jax.Array
        Atmospheric perturbation lifetime τ_x (years).
    horizon : jax.Array or float
        Time horizon H (years); default 100.
    agtp_co2 : jax.Array or float
        Reference AGTP_CO2 at horizon H (K per kg); default
        6.84e-16 (AR5 Table 8.SM.16 H=100).
    c_climate : float
        Climate sensitivity (K per W/m²); default 0.631.
    d_climate : float
        Climate-response timescale (years); default 8.4.
    tau_floor : float
        Lower bound on τ (years); default 1e-6.
    diff_floor : float
        Lower bound on |τ − d| guard against τ=d resonance;
        default 1e-9.

    Returns
    -------
    gtp : jax.Array
        Dimensionless GTP relative to CO₂ at horizon H.
    """
    tau_safe = jnp.maximum(tau, tau_floor)
    diff = tau_safe - d_climate
    diff_safe = jnp.where(
        jnp.abs(diff) < diff_floor,
        jnp.sign(diff) * diff_floor + (diff == 0) * diff_floor,
        diff,
    )
    agtp_x = (
        rad_eff
        * c_climate
        * tau_safe
        / diff_safe
        * (jnp.exp(-horizon / tau_safe) - jnp.exp(-horizon / d_climate))
    )
    return agtp_x / agtp_co2


def tcre_remaining_budget_fv3(
    delta_t_target: jax.Array,
    delta_t_current: jax.Array,
    tcre: jax.Array = 0.45,
    tcre_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 846: TCRE-based remaining-carbon-budget primitive.

    Allen-Stocker-Matthews TCRE (Transient Climate Response to
    cumulative carbon Emissions; Matthews et al. 2009, Allen
    et al. 2009, Allen-Stocker 2014) gives the near-linear
    relationship between cumulative anthropogenic CO₂ emissions
    and global-mean warming:

        ΔT − ΔT_pre-industrial ≈ TCRE · E_cumulative

    Remaining carbon budget for a target warming threshold:

        E_remaining = (ΔT_target − ΔT_current) / TCRE   (Gt-CO₂)

    Where:
      * ΔT_target  — target warming above pre-industrial (K).
      * ΔT_current — current warming (K), typically ~1.1 K (2020 CE).
      * TCRE       — slope (K per 1000 Gt-CO₂); AR6 central 0.45
                     (likely range 0.27–0.63).

    AR6 central remaining-budget benchmarks (ΔT_current = 1.1 K):
      | target ΔT  | E_remaining (Gt-CO₂) | years @ 40 Gt-CO₂/yr |
      |-----------|----------------------|----------------------|
      | 1.5 K     | 889                  | 22                   |
      | 1.7 K     | 1333                 | 33                   |
      | 2.0 K     | 2000                 | 50                   |
      | 2.5 K     | 3111                 | 78                   |

    Negative output ⇒ target already exceeded (overshoot).

    Sign convention: TCRE > 0 (warming per emission); output
    follows sign of (ΔT_target − ΔT_current).

    Closes the **cumulative-emission climate-budget primitive**
    chain.  Composes with iter-836 ECS / iter-837 TCR and the
    GHG-forcing trio (iter-838-840) for end-to-end emission-
    pathway analysis:

        cumulative emission E → iter-846 ΔT (via TCRE)
        or
        target ΔT → iter-846 E_remaining

    Unlike iter-836 ECS (per-forcing) and iter-844 GWP (per-mass
    cumulative-forcing weighting), TCRE is the canonical
    *cumulative-emission* climate-sensitivity metric used in:
      * AR6 SPM and policy briefs (carbon-budget statements).
      * IPCC AR5 WG1 §12.5.4 carbon-budget framework.
      * Allen-Stocker 2014 fairness and equity analysis.
      * 1.5°C / 2°C remaining-budget calculators
        (Friedlingstein et al. 2022 Global Carbon Project).
      * Net-zero target setting (corporate, national, IAM-based).

    TCRE near-linearity comes from the cancellation between
    sub-linear CO₂ radiative forcing (log) and sub-linear
    ocean carbon-uptake (decreasing saturation) — Goodwin et al.
    2015, Williams et al. 2017 thermodynamic derivation.

    ``tcre_floor`` prevents div-by-0 in pathological TCRE→0.

    Parameters
    ----------
    delta_t_target : jax.Array
        Target warming above pre-industrial (K).
    delta_t_current : jax.Array
        Current warming above pre-industrial (K).
    tcre : jax.Array or float
        Transient Climate Response to cumulative Emissions
        (K per 1000 Gt-CO₂); default 0.45 (AR6 central).
    tcre_floor : float
        Lower bound on |TCRE|; default 1e-6.

    Returns
    -------
    e_remaining : jax.Array
        Remaining carbon budget (Gt-CO₂); negative for overshoot.
    """
    tcre_safe = jnp.where(
        jnp.abs(tcre) < tcre_floor,
        tcre_floor,
        tcre,
    )
    return 1000.0 * (delta_t_target - delta_t_current) / tcre_safe


def airborne_fraction_co2_fv3(
    cumulative_emission: jax.Array,
    airborne_fraction: jax.Array = 0.46,
    gtco2_per_ppm: float = 7.81,
) -> jax.Array:
    """FV3_3D iter 847: airborne-fraction emission→atmospheric-CO₂.

    Friedlingstein et al. 2022 AR6 Global Carbon Project closure
    converting cumulative anthropogenic CO₂ emission to airborne
    ppm increase:

        ΔCO₂_atm (ppm) = AF · E_cum (Gt-CO₂) / k_C2ppm

    Where:
      * AF        — airborne fraction (dimensionless); fraction
                    of cumulative CO₂ emissions remaining in the
                    atmosphere after ocean + land carbon-cycle
                    uptake.  AR6/GCB central 0.46 (range 0.40–0.50).
      * k_C2ppm   — mass-to-mixing-ratio conversion 7.81 Gt-CO₂/ppm.
                    Derived from atmospheric mass and CO₂ molar
                    weight: M_atm = 5.148×10¹⁸ kg, M_CO₂ = 44.01,
                    M_dry-air = 28.97 → 1 ppm = 5.148e18·44.01/
                    (28.97·1e6·1e12) ≈ 7.81 Gt-CO₂.

    Sign: positive for E_cum > 0 (typical anthropogenic emissions).
    Land/ocean uptake handled implicitly by AF < 1 (~54% of human
    emissions absorbed by sinks at present-day).

    AR6 canonical:
      * Cumulative anthropogenic 1750→2020 ≈ 2400 Gt-CO₂
        → AF · 2400 / 7.81 ≈ 141 ppm
        (matches observed 419 − 278 = 141 ppm).
      * Future SSP3-7.0 cumulative 2020→2100 ≈ 7000 Gt-CO₂
        → ΔCO₂_atm ≈ 412 ppm (atm reaches ~830 ppm).

    Closes the **emission → atmospheric-concentration primitive
    chain**, completing the full pipeline:

        emission → iter-847 AF → ΔCO₂_atm (ppm)
                  ↓
                  iter-838 ΔF_CO₂ → iter-836/837 ECS/TCR → ΔT
                  (concentration pathway)

        emission → iter-846 TCRE → ΔT (cumulative-emission shortcut)

    The AF-based path lets you separate the carbon-cycle response
    (AF, varies 0.40–0.50 with feedbacks) from the radiative-
    forcing-and-climate-feedback response (Myhre + AR6 quartet).

    Used by: Friedlingstein et al. 2022 GCB carbon-budget tables,
    IPCC AR6 §5 carbon cycle, FaIR/MAGICC simple-climate-model
    emission-driven runs, integrated-assessment-model CO₂-cycle
    closure, Bern-CC impulse-response replacement when only AF is
    available.

    Note: AF here is the *cumulative* airborne fraction (not the
    annual flux ratio).  The two coincide for sustained emissions
    over multi-century horizons; for net-zero pathways diverge —
    AF_cumulative grows toward 1 because remaining airborne CO₂
    has nowhere left to go.

    Parameters
    ----------
    cumulative_emission : jax.Array
        Cumulative anthropogenic CO₂ emission (Gt-CO₂; positive
        for emissions to atmosphere).
    airborne_fraction : jax.Array or float
        Cumulative airborne fraction (dimensionless); default 0.46
        (AR6/GCB central).
    gtco2_per_ppm : float
        Mass-to-mixing-ratio factor (Gt-CO₂/ppm); default 7.81.

    Returns
    -------
    delta_co2_ppm : jax.Array
        Atmospheric CO₂ mixing-ratio change (ppm); positive for
        emission to atmosphere.
    """
    return airborne_fraction * cumulative_emission / gtco2_per_ppm


def thermosteric_sea_level_fv3(
    delta_t_layer: jax.Array,
    thickness: jax.Array,
    alpha_t: jax.Array = 2.0e-4,
) -> jax.Array:
    """FV3_3D iter 848: thermosteric sea-level change (ocean
    thermal-expansion primitive).

    Column-integrated sea-level rise from ocean warming via the
    linearized equation-of-state thermal-expansion coefficient:

        Δη = Σ_k α_T(k) · ΔT(k) · H(k)        (m)

    Where:
      * α_T    — thermal-expansion coefficient (1/K).  Sea-water
                 typical value 2×10⁻⁴ /K at T=15°C, S=35 psu;
                 ranges 5×10⁻⁵ (cold deep water) to 3×10⁻⁴ (warm
                 surface).
      * ΔT(k)  — temperature change in layer k (K).
      * H(k)   — layer thickness (m).

    Sign: positive Δη ⇒ sea-level rise from warming.  Halosteric
    (salinity) contribution NOT included — for that combine with
    a separate β_S·ΔS·H term.

    Canonical magnitudes:
      * Uniform 0.1 K warming over 4000 m: Δη ≈ 8 cm.
      * Pinatubo 1992 transient cooling (−0.05 K over 1000 m
        upper ocean): Δη ≈ −1 cm.
      * 21st-century thermosteric SLR (AR6 SSP3-7.0): ~30 cm
        (matches GCM-mean integrated heat-uptake → expansion).
      * Last Glacial Maximum cooling −3 K over 5000 m: Δη ≈ −3 m
        (substantial fraction of LGM sea-level lowering of 120 m
        — rest from ice-sheet meltwater).

    Layer-by-layer broadcast: caller passes per-layer arrays,
    helper returns per-column total Δη.  Accepts:
      * 1-D layer arrays — single column.
      * (..., n_layers) batched — last axis is depth.

    Composes naturally with the climate-sensitivity chain:

        emission → iter-846 TCRE → ΔT_surf
        ΔT_surf  → ocean-warming profile (caller / model-specific)
        ΔT(z), H, α_T → iter-848 Δη_thermosteric

    Note: per CLAUDE.md ocean-EOS guidance the *full* α_T(T,S,p)
    lives in ``ocean.eos``; this helper takes α_T as scalar/array
    input to remain pure-primitive without re-deriving EOS.  For
    higher-fidelity calculations:

        from legoesm.ocean import eos
        alpha = -eos.compute_ocean_rho_dT(t, s, p) / eos.rho_0

    Used by: AR5/AR6 §9.4 thermosteric-SLR diagnostics, CMIP6 RFMIP-
    SLR analyses (Slangen et al. 2017), reconstructions from ARGO
    OHC (Cheng et al. 2017), paleoclimate sea-level studies
    (Lambeck et al. 2014 LGM lowering).

    Parameters
    ----------
    delta_t_layer : jax.Array
        Per-layer temperature change (K), shape ``(..., n_layers)``.
    thickness : jax.Array
        Per-layer thickness (m), shape ``(..., n_layers)``.
    alpha_t : jax.Array or float
        Thermal-expansion coefficient (1/K); default 2.0e-4.

    Returns
    -------
    delta_eta : jax.Array
        Column-integrated thermosteric SLR (m); positive for
        warming.
    """
    return jnp.sum(alpha_t * delta_t_layer * thickness, axis=-1)


def ice_mass_to_slr_fv3(
    mass_loss_gt: jax.Array,
    ocean_area: float = 3.61e14,
    rho_water: float = None,
) -> jax.Array:
    """FV3_3D iter 849: ice-sheet mass-loss → sea-level-rise primitive.

    Mass-balance closure converting ice-sheet / glacier mass loss
    (Greenland, Antarctica, mountain glaciers) into equivalent
    eustatic SLR via fresh-water-volume distribution over the
    global ocean area:

        Δη = ΔM_ice (kg) / (ρ_water · A_ocean)        (m)

    With mass input in Gt (10¹² kg):

        Δη (m) = ΔM_Gt · 10¹² / (ρ_water · A_ocean)

    Defaults: ρ_water = ``constants.rho_water`` (1000 kg/m³),
    A_ocean = 3.61×10¹⁴ m² (AR6 global ocean area).

    Sign: ΔM_ice > 0 (mass loss to ocean) ⇒ Δη > 0 (SLR).
    Reverse sign for accretion (e.g. Antarctic Ice Sheet gain
    during snowfall episodes; LGM-onset ice-sheet build-up).

    Conversion factor: 1 Gt ice → ~2.77 μm SLR, or equivalently
    361 Gt ice → 1 mm SLR (AR6 reference value).

    AR6 (2010–2019 mass-balance rates, IMBIE-3 / Mauritzen 2020):
      | source                  | Rate (Gt/yr) | SLR (mm/yr) |
      |-------------------------|--------------|-------------|
      | Greenland ice sheet     | ~250         | 0.69        |
      | Antarctic ice sheet     | ~150         | 0.41        |
      | Mountain glaciers       | ~330         | 0.91        |
      | All cryosphere          | ~730         | 2.02        |
      | (vs total SLR ~3.7 mm/yr; ~55% cryospheric)             |

    Worst-case bookkeeping limits:
      * Greenland total volume: 2.85×10⁶ Gt → 7.4 m SLR if fully
        melted (matches AR6 §9.5.3 contribution).
      * West Antarctica (WAIS) marine-based: 2.0×10⁶ Gt → 5.3 m
        SLR (potential bifurcation under MISI / MICI mechanisms).
      * East Antarctica: 51.9×10⁶ Gt → 53 m SLR (geological-time
        upper bound).

    Closes the **AR6 SLR contributor pair** with iter-848:
      * iter-848 thermosteric_sea_level_fv3  — ocean thermal exp
      * iter-849 ice_mass_to_slr_fv3         — cryosphere melt

    Together with halosteric (caller-derived β_S·ΔS·H) these give
    the four canonical SLR budget terms.  Composes with:

        emission → iter-846 TCRE → ΔT_surf
        ΔT_surf  → cryosphere mass-balance model (caller-specific)
        ΔM_ice   → iter-849 Δη_eustatic

    Per CLAUDE.md hygiene: uses ``constants.rho_water`` as default
    (the canonical 1000 kg/m³).  Ocean-area 3.61×10¹⁴ m² is an
    AR6 reference geometric value (not a fundamental physical
    constant — default arg).

    Used by: AR5/AR6 §9.5 cryospheric SLR diagnostics, IMBIE
    Mass-Balance Intercomparison (Shepherd et al. 2018, 2020),
    Mauritzen 2020 SROCC update, GlacierMIP / ISMIP6 protocols,
    paleoclimate ice-sheet retreat / advance studies, sea-level
    Earth-system-model (SLES) coupling.

    Parameters
    ----------
    mass_loss_gt : jax.Array
        Cryosphere mass loss (Gt; positive = ice loss to ocean,
        negative = accretion).
    ocean_area : float
        Global ocean area (m²); default 3.61×10¹⁴ (AR6).
    rho_water : float or None
        Water density (kg/m³); default None → uses
        ``constants.rho_water``.

    Returns
    -------
    delta_eta : jax.Array
        Eustatic SLR contribution (m); positive for ice loss.
    """
    rho = rho_water if rho_water is not None else constants.rho_water
    return mass_loss_gt * 1.0e12 / (rho * ocean_area)


def halosteric_sea_level_fv3(
    delta_s_layer: jax.Array,
    thickness: jax.Array,
    beta_s: jax.Array = 7.6e-4,
) -> jax.Array:
    """FV3_3D iter 850: halosteric sea-level change (salinity primitive).

    Column-integrated sea-level change from ocean salinity
    perturbation via linearized-EOS haline-contraction coefficient
    (mirror of iter-848 thermosteric):

        Δη_halo = − Σ_k β_S(k) · ΔS(k) · H(k)      (m)

    Where:
      * β_S    — haline-contraction coefficient (1/psu, equivalently
                 (g/kg)⁻¹).  Sea-water typical 7.6×10⁻⁴ /psu (T=15°C
                 S=35); range 7×10⁻⁴ (warm) to 8×10⁻⁴ (cold).
      * ΔS     — salinity change in layer (psu).
      * H      — layer thickness (m).

    Sign: ΔS > 0 (salinification) ⇒ Δη_halo < 0 (water contracts);
    ΔS < 0 (freshening) ⇒ Δη_halo > 0 (local SLR).  This is the
    opposite-sign mirror of iter-848 thermosteric.

    Canonical magnitudes:
      * Global mean ΔS ≈ 0 (mass conservation; halosteric ≈ 0 on
        global mean — but locally important).
      * North Atlantic freshening ~−0.5 psu over 1000 m:
        Δη_halo ≈ +0.4 m (regional, contributes to AMOC-slowdown
        SLR pattern).
      * Tropical-Pacific salinification ~+0.3 psu over 500 m:
        Δη_halo ≈ −0.1 m (regional dynamic SLR).
      * AR6 §9.5.1: halosteric SLR globally ≈ 0 but spatially
        explains O(0.1 m) regional pattern variation (Durack-
        Wijffels 2015).

    Layer-by-layer broadcast: same signature as iter-848.

    **Closes the SLR-budget triplet** with iter-848 + iter-849:
      * iter-848 thermosteric_sea_level_fv3  — ocean thermal exp
      * iter-850 halosteric_sea_level_fv3    — ocean salinity exp
      * iter-849 ice_mass_to_slr_fv3         — cryosphere melt

    Together these span the AR6 SLR budget terms (~100% closure
    when combined with land-water-storage residual).

    Per CLAUDE.md ocean-EOS guidance: full β_S(T,S,p) lives in
    ``legoesm.ocean.eos``.  Caller can use:

        from legoesm.ocean import eos
        beta = eos.compute_ocean_rho_dS(t, s, p) / eos.rho_0
        eta_h = halosteric_sea_level_fv3(delta_s, thickness, beta)

    Used by: AR5/AR6 §9.5.1 halosteric SLR diagnostics, Durack-
    Wijffels 2015 ocean-salinity-pattern attribution, CMIP6 RFMIP-
    SLR, ARGO salinity reconstructions (Durack et al. 2014).

    Parameters
    ----------
    delta_s_layer : jax.Array
        Per-layer salinity change (psu); shape ``(..., n_layers)``.
    thickness : jax.Array
        Per-layer thickness (m); same shape.
    beta_s : jax.Array or float
        Haline-contraction coefficient (1/psu); default 7.6e-4.

    Returns
    -------
    delta_eta : jax.Array
        Column-integrated halosteric SLR contribution (m); positive
        for freshening (ΔS<0), negative for salinification.
    """
    return -jnp.sum(beta_s * delta_s_layer * thickness, axis=-1)


def ocean_heat_content_fv3(
    t_layer: jax.Array,
    thickness: jax.Array,
    rho: float = None,
    c_p: float = None,
) -> jax.Array:
    """FV3_3D iter 851: ocean heat content column-integral primitive.

    Column-integrated sensible heat (or heat-content anomaly when
    ``t_layer`` is an anomaly) per unit horizontal area:

        OHC = ρ_0 · c_p · Σ_k T(k) · H(k)        (J/m²)

    Where:
      * ρ_0   — reference seawater density (kg/m³); default
                ``constants.rho_ocean`` = 1025.
      * c_p   — specific heat of seawater at constant pressure
                (J/(kg·K)); default ``constants.c_sw`` = 3994 (Gill 1982).
      * T(k)  — layer temperature OR temperature anomaly (K).
      * H(k)  — layer thickness (m).

    Sign: positive for warm-anomaly column; passing absolute T
    gives total OHC, passing ΔT gives ΔOHC (heat-uptake anomaly).

    Canonical magnitudes (use ΔT for anomalies):
      * 0.1 K uniform / 4000 m: ΔOHC ≈ 1.64×10⁹ J/m²
        → globally (×3.61×10¹⁴ m²) ≈ 590 ZJ.
      * 1991–2020 observed (Cheng 2017 0–2000 m): ~10 ZJ/yr
        → ~0.4 W/m² ocean heat-uptake rate.
      * CMIP6 RCP8.5 2100 (top 2000 m): ~600 ZJ accumulated.

    OHC is the canonical diagnostic for Earth's energy imbalance
    (Hansen 2005, Cheng 2017) — ocean absorbs >90% of climate
    forcing, so ΔOHC growth rate ≈ TOA imbalance × A_ocean.

    Pairs directly with iter-848 ``thermosteric_sea_level_fv3``
    (same column-integral structure):

        Δη_thermo = α_T · Σ ΔT · H        (m;     ~thermal expansion)
        ΔOHC      = ρ·c_p · Σ ΔT · H      (J/m²; heat absorbed)

    so they share the input ΔT·H weighting — caller can compute
    both from same per-layer arrays.

    Per CLAUDE.md hygiene: uses ``constants.rho_ocean`` and
    ``constants.c_sw`` defaults.  For high-fidelity EOS-derived
    ρ(T,S,p) and c_p(T,S,p), caller imports from ``legoesm.ocean.eos``
    and passes explicitly.

    Used by: Hansen 2005 / 2011 planetary-energy-imbalance OHC
    diagnostic, Cheng et al. 2017 ARGO-based OHC reconstruction,
    AR6 §7.2 energy-budget closure, von Schuckmann 2020 GEWEX-EEI
    review, Domingues 2008 XBT-bias-corrected reanalysis.

    Parameters
    ----------
    t_layer : jax.Array
        Per-layer temperature or temperature anomaly (K), shape
        ``(..., n_layers)``.
    thickness : jax.Array
        Per-layer thickness (m), same shape.
    rho : float or None
        Reference density (kg/m³); default None → uses
        ``constants.rho_ocean``.
    c_p : float or None
        Specific heat (J/(kg·K)); default None → uses
        ``constants.c_sw``.

    Returns
    -------
    ohc : jax.Array
        Column-integrated heat content (J/m²); positive for warm
        column or warm anomaly.
    """
    rho_use = rho if rho is not None else constants.rho_ocean
    c_p_use = c_p if c_p is not None else constants.c_sw
    return rho_use * c_p_use * jnp.sum(t_layer * thickness, axis=-1)


def ocean_ph_change_fv3(
    pco2_new: jax.Array,
    pco2_ref: jax.Array = 278.0,
    sensitivity_s: float = 0.67,
    pco2_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 852: ocean-acidification ΔpH primitive.

    Caldeira-Wickett 2003 / Bates 2014 logarithmic-CO₂ closure
    for surface-ocean pH change from atmospheric pCO₂ perturbation:

        ΔpH = − s · log10(pCO₂_new / pCO₂_ref)

    Where:
      * pCO₂_ref     — reference atmospheric pCO₂ (ppm; default
                       278 = 1750-CE pre-industrial).
      * pCO₂_new     — current/scenario atmospheric pCO₂ (ppm).
      * s            — empirical sensitivity (dimensionless);
                       default 0.67 (calibrated to AR6 observed
                       278→420 ppm → ΔpH ≈ −0.12).

    Sign: ΔpH < 0 for pCO₂_new > pCO₂_ref (warming-era acidification).
    pCO₂_new < pCO₂_ref → ΔpH > 0 (paleoclimate / cooling regimes).

    Log-form derivation: surface ocean equilibrium [H₂CO₃*] tracks
    atmospheric pCO₂ via Henry's law.  pH = −log₁₀[H⁺], and
    carbonate-system buffering gives [H⁺] ∝ pCO₂^s with s≈0.67
    at present-day chemistry (varies 0.5–0.8 across pCO₂ range
    300–800 ppm; Revelle factor varies with DIC saturation).

    Canonical AR6 acidification:
      | epoch           | pCO₂   | ΔpH from 1750 |
      |-----------------|--------|---------------|
      | 1750 CE pre-ind | 278    |  0.00         |
      | 2024 present    | 420    | −0.12         |
      | SSP1-2.6 2100   | 450    | −0.14         |
      | SSP3-7.0 2100   | 850    | −0.32         |
      | LGM glacial     | 180    | +0.13         |
      | PETM ~55 Ma     | 1000   | −0.37         |

    Observed surface-ocean pH dropped from ~8.18 (1750) to ~8.06
    (2024) — global mean −0.12.  Below pH 7.7 most aragonite-
    dependent species (corals, pteropods) struggle to calcify.

    Composes with iter-847 ``airborne_fraction_co2_fv3`` (emission
    → ΔCO₂_atm) and iter-838 ``radiative_forcing_co2_fv3`` (CO₂
    → ΔF):

        E_cum → iter-847 AF → ΔCO₂_atm
              ↓
              iter-852 ΔpH               (ocean BGC impact)
              iter-838 ΔF_CO₂ → ECS/TCR  (climate impact)

    Adds first **ocean-biogeochemistry primitive** to the
    emission→impact chain.  Pairs with iter-851 ``ocean_heat_content_fv3``
    (ocean BGC impact + thermal impact = full ocean response to
    anthropogenic forcing).

    Used by: Bates et al. 2014 ESSD ocean-acidification overview,
    AR6 §5.3 ocean acidification, Caldeira-Wickett 2003 / Orr et al.
    2005 GBC OA projections, IPCC SROCC §5.2.2, OA-MIP CMIP6
    protocol (Schwinger et al. 2020), reef-impact studies (Hoegh-
    Guldberg 2007).

    Note: linear pH vs ΔpCO₂ in small-perturbation limit:
    ΔpH ≈ −s/(ln10·pCO₂_ref)·ΔpCO₂ ≈ −1.05×10⁻³/ppm at PI.  Use
    log form for non-linear regimes (>50% ΔpCO₂).

    ``pco2_floor`` prevents log10(0) for zero / negative input.

    Parameters
    ----------
    pco2_new : jax.Array
        Current atmospheric pCO₂ (ppm).
    pco2_ref : jax.Array or float
        Reference atmospheric pCO₂ (ppm); default 278 (1750 CE).
    sensitivity_s : float
        Caldeira-Wickett dimensionless sensitivity; default 0.67.
    pco2_floor : float
        Lower bound on pCO₂ (ppm); default 1e-6.

    Returns
    -------
    delta_ph : jax.Array
        pH change (dimensionless); negative for acidification.
    """
    p_new_safe = jnp.maximum(pco2_new, pco2_floor)
    p_ref_safe = jnp.maximum(pco2_ref, pco2_floor)
    return -sensitivity_s * jnp.log10(p_new_safe / p_ref_safe)


def aragonite_saturation_state_fv3(
    pco2_new: jax.Array,
    omega_arag_ref: jax.Array = 3.5,
    pco2_ref: jax.Array = 278.0,
    gamma_exp: float = 0.85,
    pco2_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 853: aragonite saturation state Ω_arag.

    Orr et al. 2005 / Feely et al. 2009 power-law parameterization
    of aragonite saturation state Ω_arag = [Ca²⁺][CO₃²⁻]/K_sp under
    atmospheric pCO₂ perturbation:

        Ω_arag(t) = Ω_arag_ref · (pCO₂_ref / pCO₂_new)^γ

    Where:
      * Ω_arag_ref  — reference Ω_arag at pCO₂_ref; default 3.5
                      (pre-industrial surface tropics, Orr 2005).
                      Surface mean ~3.0, subtropics ~3.8, polar ~1.5.
      * pCO₂_ref    — reference atmospheric pCO₂ (ppm); default 278.
      * γ           — empirical exponent ~0.85 (Orr 2005; varies
                      0.8–0.9 across saturation regimes via [CO₃²⁻]
                      buffering response to DIC).

    Sign: Ω > 1 ⇒ supersaturated (calcification favorable); Ω < 1
    ⇒ undersaturated (dissolution, aragonite-dependent organisms
    die back).

    Canonical AR6 trajectory (warm tropical surface):
      | epoch           | pCO₂   | Ω_arag |
      |-----------------|--------|--------|
      | 1750 pre-ind    | 278    | 3.50   |
      | 2024 present    | 420    | 2.47   |
      | SSP1-2.6 2100   | 450    | 2.34   |
      | SSP3-7.0 2100   | 850    | 1.41   |
      | Undersaturation | ~2100  | 1.00   |
      | PETM ~55 Ma     | 1000   | 1.23   |

    Below Ω_arag = 1 corals/pteropods/mollusks struggle to calcify
    (Hoegh-Guldberg 2007).  Polar oceans already near Ω_arag=1 at
    present-day (Orr 2005); Southern Ocean projected to cross 1.0
    by ~2030–2040 under SSP scenarios (Hauck-Völker 2015).

    Composes with iter-852 ``ocean_ph_change_fv3`` to give the
    **paired carbonate-chemistry primitives**:
      * iter-852 ΔpH    — protonation state
      * iter-853 Ω_arag — calcite-saturation state

    Together they span the AR6 §5.3 OA-impact metrics.

    Used by: Orr et al. 2005 GBC OA projections, Feely et al. 2009
    Annu Rev shells-and-skeletons synthesis, AR6 §5.3 ocean
    acidification, IPCC SROCC §5.2.2.5, OA-MIP / CMIP6 carbonate-
    chemistry diagnostics, reef-impact studies (Hoegh-Guldberg
    2007), Hauck-Völker 2015 Southern Ocean projections.

    Note: γ varies with saturation regime; for high-CO₂ regimes
    (PETM, paleoclimate) the [CO₃²⁻] response saturates and γ
    decreases.  Linear-in-log-pCO₂ form (this primitive) is most
    accurate in 200–1000 ppm band.

    Parameters
    ----------
    pco2_new : jax.Array
        Current atmospheric pCO₂ (ppm).
    omega_arag_ref : jax.Array or float
        Reference Ω_arag at pco2_ref; default 3.5 (PI tropical).
    pco2_ref : jax.Array or float
        Reference atmospheric pCO₂ (ppm); default 278 (1750 CE).
    gamma_exp : float
        Orr 2005 empirical exponent; default 0.85.
    pco2_floor : float
        Lower bound on pCO₂ (ppm); default 1e-6.

    Returns
    -------
    omega_arag : jax.Array
        Aragonite saturation state (dimensionless); Ω<1 indicates
        dissolution regime.
    """
    p_new_safe = jnp.maximum(pco2_new, pco2_floor)
    p_ref_safe = jnp.maximum(pco2_ref, pco2_floor)
    return omega_arag_ref * (p_ref_safe / p_new_safe) ** gamma_exp


def ocean_oxygen_decline_fv3(
    delta_t_ocean: jax.Array,
    alpha_o2: float = 0.05,
) -> jax.Array:
    """FV3_3D iter 854: ocean deoxygenation fractional-decline primitive.

    Schmidtko et al. 2017 / Keeling et al. 2010 linear closure
    relating ocean warming to dissolved-O₂ fractional decline,
    combining temperature-dependent solubility (Henry's law,
    ~2–3% per K) and warming-induced stratification (reduced
    ventilation, ~2% per K):

        ΔO₂/O₂_ref = − α_O2 · ΔT_ocean         (dimensionless)

    Where:
      * ΔT_ocean — ocean-volume-weighted (or basin-mean) warming (K).
      * α_O2     — empirical sensitivity (per K); default 0.05/K
                   (Schmidtko 2017 + Keeling 2010 / Bopp 2013 CMIP
                   global mean).  Subcomponents:
                   - solubility:  −0.022/K (Garcia-Gordon 1992)
                   - ventilation: −0.028/K (Schmidtko 2017 inferred)

    Sign: ΔT > 0 ⇒ ΔO₂ < 0 (warming-induced deoxygenation).

    Canonical AR6 / Schmidtko 2017:
      | period            | ΔT_ocean | ΔO₂ |
      |-------------------|----------|-----|
      | 1960–2010 observed| ~+0.4 K  | −2% |
      | SSP1-2.6 2100     | +1.0 K   | −5% |
      | SSP3-7.0 2100     | +2.5 K   | −12% |
      | Pre-industrial ref| 0        |  0  |

    Observed −2% (~−4 Pmol O₂) since 1960 (Schmidtko 2017 Nature),
    consistent with α_O2 ≈ 0.05/K at observed ΔT ≈ 0.4 K.

    Deoxygenation has cascading impacts: (1) OMZ (Oxygen Minimum
    Zone) expansion (Stramma 2008, Breitburg 2018); (2) N₂O
    production via denitrification, amplifying GHG (Codispoti 2010);
    (3) fish habitat compression (Pörtner 2008); (4) hypoxic
    'dead zone' coastal expansion (Diaz-Rosenberg 2008).

    **Closes the AR6 §5 ocean-BGC triplet** with iter-852 + iter-853:
      * iter-852 ocean_ph_change_fv3       — acidification ΔpH
      * iter-853 aragonite_saturation_fv3  — calcite-saturation Ω
      * iter-854 ocean_oxygen_decline_fv3  — deoxygenation ΔO₂

    Together these are the AR6 §5.3 OA-and-deoxygenation impact
    pair (technically OA is two: pH + Ω; deoxygenation is third).

    Pairs with iter-851 ``ocean_heat_content_fv3`` as natural
    chain: OHC → ΔT_ocean → ΔO₂.

    Used by: Schmidtko et al. 2017 Nature deoxygenation map,
    Keeling et al. 2010 Annu Rev review, Breitburg et al. 2018
    Science synthesis, Bopp et al. 2013 ESD CMIP5 multi-stressor,
    AR6 §5.3.4, IPCC SROCC §5.2.2.4, OMZ-and-N₂O coupled
    biogeochemistry studies.

    Note: linear primitive; non-linear basin-specific effects
    (Atlantic vs Pacific stratification, NADW vs AABW ventilation
    differences) require 3-D BGC model.  For CMIP-class basin
    decomposition use OMIP-BGC kernels.

    Parameters
    ----------
    delta_t_ocean : jax.Array
        Ocean warming anomaly (K); positive for warming.
    alpha_o2 : float
        Schmidtko-Keeling sensitivity (per K); default 0.05.

    Returns
    -------
    delta_o2_frac : jax.Array
        Fractional O₂ change (dimensionless); negative for
        deoxygenation under warming.
    """
    return -alpha_o2 * delta_t_ocean


def degree_heating_weeks_fv3(
    sst_weekly: jax.Array,
    mmm: jax.Array,
    hotspot_threshold: float = 1.0,
) -> jax.Array:
    """FV3_3D iter 855: NOAA Coral Reef Watch degree-heating-weeks.

    Eakin et al. 2010 / NOAA Coral Reef Watch (CRW) accumulated-
    thermal-stress primitive used in coral-bleaching forecasts:

        HS(t)  = max(SST(t) − MMM − τ_HS, 0)        (°C; hotspot)
        DHW    = Σ_{k=t-12+1}^{t} HS(k)             (°C·weeks)

    Where:
      * SST(t)       — weekly satellite SST (K or °C).
      * MMM          — Maximum Monthly Mean climatology (same
                       units; pre-computed from 1985-2012 or
                       custom baseline).
      * τ_HS         — hotspot threshold (default 1.0 °C; ≥1°C
                       above MMM marks 'hotspot' day under Eakin
                       2010 CRW protocol).
      * Σ over 12-wk — accumulating thermal stress over the
                       rolling 12-week window (NOAA convention).

    This primitive operates on the *final 12-week window* of input
    SST — caller passes ``(..., 12)`` of weekly SST and gets DHW
    summed over all 12 weeks.  For full time-series rolling DHW,
    caller uses ``jax.vmap`` or ``jnp.cumsum`` outside.

    Sign: DHW ≥ 0 always (only positive HS contributes).

    Bleaching thresholds (Eakin 2010, NOAA CRW):
      | DHW (°C-wk) | Outcome                                |
      |-------------|----------------------------------------|
      | 0–4         | thermal stress accumulating            |
      | 4–8         | significant bleaching expected         |
      | 8+          | severe bleaching + widespread mortality|

    Canonical real-world events:
      * 1998 global bleaching:        DHW peak ~12 °C-wk
      * 2016 GBR mass bleaching:      DHW peak ~16 °C-wk
        (29% Great Barrier Reef coral mortality)
      * 2023 Caribbean MHW:           DHW peak ~22 °C-wk
        (record Florida Reef Tract bleaching)

    Composes naturally with iter-851 ``ocean_heat_content_fv3``
    (OHC growth → SST anomaly → DHW exceedance).  Pairs with
    iter-853 ``aragonite_saturation_state_fv3`` (Ω + thermal stress
    are the two main reef-impact pathways).

    Adds **first marine-extreme-event primitive** to the impact
    chain — complements long-term mean diagnostics (iter-852/853/
    854) with the event-scale acute-stress metric.

    Used by: Eakin et al. 2010 J. Climate CRW protocol, Hobday et al.
    2016 marine-heatwave definition (related), Heron et al. 2016
    Sci Reports satellite-DHW reef forecast, AR6 §3.5 marine
    extreme events, IPCC SROCC §5.3.4 reef impacts, Hoegh-Guldberg
    1999/2007 coral-bleaching synthesis.

    Note: NOAA CRW uses additional logic (HS only contributes if
    ≥1 °C, only ≥1°C HS counted toward DHW).  This primitive
    matches the canonical published formula; production NOAA
    pipeline adds night-time SST filtering, cloud-cover gating,
    and 5-km grid resolution.

    Parameters
    ----------
    sst_weekly : jax.Array
        Weekly SST values over last 12 weeks (K or °C); shape
        ``(..., 12)`` (last axis is time).
    mmm : jax.Array
        Maximum Monthly Mean climatology (same units as SST);
        shape ``(...,)`` or broadcastable.
    hotspot_threshold : float
        Hotspot anomaly threshold (°C); default 1.0 (Eakin 2010).

    Returns
    -------
    dhw : jax.Array
        Accumulated degree-heating-weeks (°C·weeks); shape
        ``(...,)`` after summing time axis.
    """
    mmm_b = jnp.expand_dims(mmm, axis=-1) if mmm.ndim < sst_weekly.ndim else mmm
    anomaly = sst_weekly - mmm_b
    hotspot = jnp.maximum(anomaly - hotspot_threshold, 0.0)
    return jnp.sum(hotspot, axis=-1)


def marine_heatwave_category_fv3(
    sst: jax.Array,
    clim: jax.Array,
    threshold_90: jax.Array,
    delta_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 856: Hobday 2016/2018 marine-heatwave category.

    Hobday et al. 2018 (Oceanography) category classification for
    a marine-heatwave event using 90th-percentile threshold
    exceedance multiples:

        Δ_clim = SST − clim                       (raw anomaly, °C)
        Δ_90   = T_90 − clim                      (threshold exceedance, °C)
        x      = Δ_clim / Δ_90                    (multiplicative intensity)

        Category =
          0  (no MHW)            x ≤ 1
          1  (moderate)          1 < x ≤ 2
          2  (strong)            2 < x ≤ 3
          3  (severe)            3 < x ≤ 4
          4  (extreme)           x > 4

    Where:
      * SST          — current sea-surface temperature (K or °C).
      * clim         — daily climatology (same units).
      * T_90         — 90th-percentile threshold (climatology +
                       daily-resolution percentile from baseline
                       1982-2011 typically; same units).

    Output: integer-valued ``jax.Array`` (still float-dtype for
    JAX-pytree compatibility) giving the Hobday category 0–4.

    Real-world events:
      | event                         | peak category |
      |-------------------------------|---------------|
      | Blob NE Pacific 2014–2016     | III           |
      | Tasman Sea 2015–2016          | IV            |
      | Florida Reef Tract 2023       | V (off-scale) |
      | NW Atlantic 2012              | III           |

    Categories I–IV mirror Saffir-Simpson hurricane scale framing
    (Hobday 2018), providing intuitive non-specialist communication
    of MHW severity.

    Pairs with iter-855 ``degree_heating_weeks_fv3`` to close the
    **marine-extreme-event primitive pair**:
      * iter-855 DHW         — accumulated chronic thermal stress
      * iter-856 MHW cat     — instantaneous acute-event category

    DHW measures *cumulative reef-bleaching* exposure over 12 wks;
    MHW category measures *daily intensity* relative to baseline
    variability — both used in NOAA / IMOS / CSIRO operational
    products.

    Composes with iter-851 OHC chain: warming → SST → MHW cat.

    ``delta_floor`` prevents div-by-0 when Δ_90 → 0 (cells where
    no MHW threshold exists, e.g. polar climatologies with no
    distinguishable 90th percentile).

    Used by: Hobday et al. 2016 PIO MHW definition, Hobday et al.
    2018 Oceanography category framework, IMOS Marine Heatwave
    Portal, Smale et al. 2019 Nature Climate Change global MHW
    impact synthesis, AR6 §11.3.5 marine heatwave projections.

    Parameters
    ----------
    sst : jax.Array
        Daily SST (K or °C).
    clim : jax.Array
        Daily climatology (same units).
    threshold_90 : jax.Array
        90th-percentile threshold T_90 (same units).
    delta_floor : float
        Lower bound on |T_90 − clim|; default 1e-6.

    Returns
    -------
    category : jax.Array
        Hobday MHW category (0–4); 0 = no MHW, 4 = extreme.
    """
    delta_clim = sst - clim
    delta_90 = threshold_90 - clim
    delta_90_safe = jnp.where(
        jnp.abs(delta_90) < delta_floor,
        delta_floor,
        delta_90,
    )
    x = delta_clim / delta_90_safe
    # Clip below 1 (no MHW) to 0 explicitly; cat = floor(x) clamped [0, 4]
    cat = jnp.floor(jnp.maximum(x, 0.0))
    cat = jnp.where(x <= 1.0, 0.0, cat)
    return jnp.clip(cat, 0.0, 4.0)


def wet_bulb_temperature_stull_fv3(
    t_c: jax.Array,
    rh_pct: jax.Array,
) -> jax.Array:
    """FV3_3D iter 857: Stull 2011 empirical wet-bulb temperature.

    Stull (2011) J. Appl. Meteor. Climatol. 50: 2267-2269 closed-
    form empirical fit to standard psychrometric wet-bulb T_w at
    pressure ~1013.25 hPa:

        T_w = T · atan(0.151977 · (RH + 8.313659)^0.5)
            + atan(T + RH)
            − atan(RH − 1.676331)
            + 0.00391838 · RH^1.5 · atan(0.023101 · RH)
            − 4.686035

    Where:
      * T   — dry-bulb temperature (°C).
      * RH  — relative humidity (%, 0–100).
      * T_w — wet-bulb temperature (°C).

    Validity: −20 < T < 50 °C, 5 < RH < 99 %, 850 < p < 1080 hPa.
    Max error 0.3 K vs full psychrometric solution (Stull 2011).

    Wet-bulb is the key **heat-stress** metric (more relevant than
    dry-bulb for human / livestock thermoregulation) because
    evaporative cooling is the body's primary heat-shedding
    pathway in hot environments.

    Sherwood-Huber 2010 PNAS survivability thresholds:
      | T_w (°C) | Outcome                                |
      |----------|----------------------------------------|
      | < 28     | mild heat stress; tolerable            |
      | 28–32    | dangerous for vulnerable populations   |
      | 32–35    | extreme heat stress; current peaks     |
      | ≥ 35     | survivability ceiling — evaporative    |
      |          | cooling fails, hyperthermia within 6h  |

    Real-world peaks (Raymond et al. 2020 Sci. Adv.):
      * Persian Gulf (Jacobabad PK 2015): T_w peak 35°C
      * Indus Valley monsoon onset: T_w 34°C routine
      * AR6 SSP3-7.0 2100: 1+ billion people exposed to T_w>35°C

    Adds **first terrestrial heat-stress primitive** to impact
    chain.  Pairs with iter-855 ``degree_heating_weeks_fv3``
    (marine reef bleaching) — both 'survivability-threshold'
    diagnostics for AR6 §3.5.

    Composes with iter-832 ``clausius_clapeyron_dqdt_fv3`` (humidity
    scaling under warming) and iter-833 ``fixed_rh_humidity_change_fv3``
    (RH-preserving warming projection).  Under fixed-RH warming
    T_w rises faster than T (heat-stress amplifies via CC).

    Per CLAUDE.md hygiene: Stull 2011 is the *empirical fit* form,
    distinct from the full psychrometric ``saturation_vapor_pressure``-
    based wet-bulb solution.  Stull's coefficients are not physical
    constants but fit numerics — kept as literal numbers (not
    `constants.py` entries).  For full psychrometric solve use
    iterative Newton on the Bolton-style saturation curve in
    ``legoesm.thermo``.

    Used by: Stull 2011 J. Appl. Meteor., Sherwood-Huber 2010 PNAS
    survivability analysis, Raymond et al. 2020 Sci. Adv. global
    T_w extremes, AR6 §11.3.2 heat extremes, Mora et al. 2017
    Nature Climate Change deadly-heat-day projections, NOAA NWS
    operational heat-stress warnings.

    Parameters
    ----------
    t_c : jax.Array
        Dry-bulb temperature (°C).
    rh_pct : jax.Array
        Relative humidity (%; 0–100).

    Returns
    -------
    t_w : jax.Array
        Wet-bulb temperature (°C).
    """
    arg1 = jnp.sqrt(rh_pct + 8.313659)
    return (
        t_c * jnp.arctan(0.151977 * arg1)
        + jnp.arctan(t_c + rh_pct)
        - jnp.arctan(rh_pct - 1.676331)
        + 0.00391838 * rh_pct ** 1.5 * jnp.arctan(0.023101 * rh_pct)
        - 4.686035
    )


def heat_index_rothfusz_fv3(
    t_c: jax.Array,
    rh_pct: jax.Array,
) -> jax.Array:
    """FV3_3D iter 858: NOAA Rothfusz 1990 apparent-temperature heat index.

    Rothfusz 1990 (NWS Tech. Memo. SR-90) operational NOAA heat-
    index regression converting (T, RH) to perceived "feels-like"
    apparent temperature:

        HI(°F) = c₁ + c₂·T + c₃·R + c₄·T·R + c₅·T² + c₆·R²
              + c₇·T²·R + c₈·T·R² + c₉·T²·R²

    Where T in °F, R in %, and (Rothfusz 1990 9-term fit to
    Steadman 1979 PNAS apparent-temperature lookup table):

        c₁ = −42.379,  c₂ = 2.04901523,
        c₃ = 10.14333127,  c₄ = −0.22475541,
        c₅ = −6.83783e-3, c₆ = −5.481717e-2,
        c₇ = 1.22874e-3,  c₈ = 8.5282e-4,
        c₉ = −1.99e-6

    Input/output in °C for consistency with rest of FV3_3D primitives:
    internally converts to °F for the regression, returns °F → °C.

    Valid for T ≥ 27 °C (≥80 °F) and RH ≥ 40 %.  Outside this band,
    Rothfusz formula is less accurate; NOAA falls back to alternative
    adjustments (e.g. low-RH correction).  This primitive returns
    the raw regression — caller is responsible for gating on validity
    range.

    NOAA HI severity thresholds:
      | HI (°C) | NOAA category   | Outcome                       |
      |---------|-----------------|-------------------------------|
      | 27–32   | Caution         | fatigue with prolonged exposure|
      | 32–39   | Extreme caution | heat cramps, heat exhaustion  |
      | 39–51   | Danger          | heat exhaustion likely; stroke|
      |         |                 | possible                      |
      | ≥ 51    | Extreme danger  | heat stroke imminent          |

    Real-world peak HI events:
      * Phoenix 2024 summer: HI peak ~55 °C
      * Iraq 2015 heat wave: HI peak 70 °C (record)
      * U.S. Midwest 1995:   HI sustained >40 °C, 700+ Chicago deaths

    Composes with iter-857 ``wet_bulb_temperature_stull_fv3`` to
    give the **NOAA-operational terrestrial heat-stress pair**:
      * iter-857 T_w (Stull)     — physiological cooling-failure threshold
      * iter-858 HI (Rothfusz)   — perceived apparent-temperature

    T_w focuses on humid limits to sweat-evaporation cooling;
    HI focuses on warning-issuance thresholds for public health.
    Both are AR6 §11.3.2 heat-extreme diagnostics.

    Composes with iter-832 ``clausius_clapeyron_dqdt_fv3`` and
    iter-833 ``fixed_rh_humidity_change_fv3``: fixed-RH warming
    raises both T (linear) and HI (faster than linear due to
    quadratic T² and RH terms).

    Per CLAUDE.md hygiene: Rothfusz 1990 9-coefficient fit is an
    empirical regression (not physical constants).  Coefficients
    kept as literals.

    Used by: Rothfusz 1990 NWS Tech. Memo. SR-90, Steadman 1979
    PNAS apparent-T lookup, NOAA NWS operational heat warnings,
    AR6 §11.3.2 heat extremes, Vecellio et al. 2022 PNAS critical
    environmental-limit experiments.

    Parameters
    ----------
    t_c : jax.Array
        Dry-bulb temperature (°C).
    rh_pct : jax.Array
        Relative humidity (%, 0–100).

    Returns
    -------
    hi_c : jax.Array
        Apparent-temperature heat index (°C).
    """
    t_f = 9.0 / 5.0 * t_c + 32.0
    r = rh_pct
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6
    hi_f = (
        c1
        + c2 * t_f
        + c3 * r
        + c4 * t_f * r
        + c5 * t_f * t_f
        + c6 * r * r
        + c7 * t_f * t_f * r
        + c8 * t_f * r * r
        + c9 * t_f * t_f * r * r
    )
    return (hi_f - 32.0) * 5.0 / 9.0


def growing_degree_days_fv3(
    t_daily: jax.Array,
    t_base: jax.Array = 10.0,
    t_cap: jax.Array = None,
) -> jax.Array:
    """FV3_3D iter 859: agricultural growing-degree-days primitive.

    Wang 1960 / McMaster-Wilhelm 1997 cumulative heat-unit measure
    of crop development:

        GDD(t) = max(T_daily(t) − T_base, 0)
        GDD_total = Σ_t GDD(t)         (°C·days)

    With optional upper cap (modified GDD method):

        T_eff = min(T_daily, T_cap)
        GDD(t) = max(T_eff(t) − T_base, 0)

    Where:
      * T_daily — daily mean (or (T_max+T_min)/2) temperature (°C).
      * T_base  — crop-specific lower threshold (°C):
                  default 10 (corn / rice / soy);
                  wheat / oats: 0 °C; cotton: 15.5 °C.
      * T_cap   — optional upper cap (°C); commonly 30 °C for corn
                  modifies GDD to prevent over-counting extreme-heat
                  damage.  None = no cap (standard simple GDD).

    Sign: GDD ≥ 0 always (only positive contributions).

    Crop maturity GDD requirements (canonical):
      | crop          | T_base | GDD to maturity |
      |---------------|--------|-----------------|
      | Corn (maize)  | 10 °C  | 2500–2800       |
      | Soybeans      | 10 °C  | 2400–2900       |
      | Spring wheat  | 0 °C   | 1500–1700       |
      | Cotton        | 15.5 °C| 2200–2800       |
      | Rice          | 10 °C  | 2000–2500       |

    Counter-balance to heat-stress primitives (iter-857 T_w,
    iter-858 HI) — represents *positive* effect of warming on
    growing-season length / crop maturation rate.  Under +2 K
    Northern-mid-latitude warming, corn GDD can rise ~15-20%,
    enabling double-cropping in former marginal zones (Hatfield
    et al. 2011 Agron. J.).

    But: T_cap-capped GDD declines under extreme heat as max(T-base,
    0) saturates → crop yield falls (Schlenker-Roberts 2009 PNAS
    "non-linear temperature effect on US crop yields").

    Composes with iter-832 ``clausius_clapeyron_dqdt_fv3`` (humidity
    increase under warming may offset heat-stress impact on crops
    via stomatal-closure suppression).

    **Adds first agricultural-impact primitive** to the impact
    chain.  Pairs with heat-stress duo (iter-857 + iter-858) as
    the **terrestrial-impact triplet** (one positive + two
    negative crop-climate response metrics).

    Used by: Wang 1960 Annu. Rev. Phytopathol., McMaster-Wilhelm
    1997 Agric. For. Meteor., Hatfield et al. 2011 Agron. J. crop
    response synthesis, USDA crop-growth models (CERES-Maize),
    AR6 §5.4 food-system risks, Schlenker-Roberts 2009 PNAS yield-
    temperature nonlinearity.

    Parameters
    ----------
    t_daily : jax.Array
        Daily mean (or (T_max+T_min)/2) temperature (°C); shape
        ``(..., n_days)`` (last axis is time).
    t_base : jax.Array or float
        Lower threshold (°C); default 10 (corn).
    t_cap : jax.Array or float or None
        Optional upper cap (°C); None disables capping (default).

    Returns
    -------
    gdd_total : jax.Array
        Cumulative growing-degree-days (°C·days), summed over time
        axis; shape ``(...,)``.
    """
    t_eff = t_daily if t_cap is None else jnp.minimum(t_daily, t_cap)
    return jnp.sum(jnp.maximum(t_eff - t_base, 0.0), axis=-1)


def spi_z_score_fv3(
    precip: jax.Array,
    mu_p: jax.Array,
    sigma_p: jax.Array,
    sigma_floor: float = 1e-6,
) -> jax.Array:
    """FV3_3D iter 860: Standardized Precipitation Index (z-score form).

    McKee et al. 1993 (8th Conf. Appl. Climatol.) drought-index
    primitive — standardized precipitation anomaly relative to
    long-term climatology:

        SPI = (P − μ_P) / σ_P

    Where:
      * P       — observed precipitation accumulated over a fixed
                  time window (typically 1, 3, 6, 12 months).
      * μ_P     — climatological mean over baseline (e.g. 1981-2010).
      * σ_P     — climatological standard deviation.

    Full McKee formulation fits a gamma distribution to the
    precipitation series and inverse-transforms quantiles to the
    standard normal.  This primitive returns the *z-score form*
    (linear normalization), which is the gamma-fit limit for
    nearly-symmetric distributions and the standard simplified
    formula used in IAM food-system and drought-impact diagnostics
    (Vicente-Serrano et al. 2010 SPEI extension).

    Sign: SPI < 0 ⇒ drier than baseline (drought); SPI > 0 ⇒ wetter.

    Severity thresholds (McKee 1993 / NOAA NDMC):
      | SPI            | Category               |
      |----------------|------------------------|
      |  SPI ≥ 2.0     | Extreme wet            |
      | 1.5 ≤ SPI <2.0 | Severe wet             |
      | 1.0 ≤ SPI <1.5 | Moderate wet           |
      | −1 < SPI < 1   | Near-normal            |
      | −1.5< SPI ≤−1  | Moderate drought       |
      | −2 < SPI ≤−1.5 | Severe drought         |
      |  SPI ≤ −2      | Extreme drought        |

    Adds **first drought-impact primitive** — completes the
    terrestrial-impact quartet:
      * iter-857 wet-bulb T_w        — heat physiological
      * iter-858 NOAA HI             — heat warning
      * iter-859 GDD                 — crop development (positive)
      * iter-860 SPI                 — drought stress

    Closes AR6 §11 + §5 heat-and-water terrestrial-impact set.

    Composes with iter-832 ``clausius_clapeyron_dqdt_fv3`` (more
    water-vapor capacity under warming → potentially wetter or
    drier depending on circulation; SPI captures observed/projected
    P, decoupled from CC theory).

    Used by: McKee et al. 1993 SPI introduction, Hayes et al. 1999
    Bull. Am. Meteor. Soc. drought-monitor protocol, Vicente-Serrano
    et al. 2010 SPEI extension (T-corrected variant), AR6 §11.6
    drought projections, IPCC SREX 2012 drought definition, U.S.
    Drought Monitor operational pipeline, IAM crop-yield modules
    (Lobell-Schlenker 2010).

    Note: gamma-fit form (full McKee SPI) would require fitting
    distribution parameters per cell — not a pure primitive.  This
    z-score variant operates on pre-computed (μ, σ) climatology;
    caller computes those externally.

    ``sigma_floor`` prevents div-by-0 when σ_P → 0 (degenerate
    deterministic-climatology cell).

    Parameters
    ----------
    precip : jax.Array
        Observed precipitation accumulation (mm or kg/m²).
    mu_p : jax.Array
        Climatological mean (same units).
    sigma_p : jax.Array
        Climatological standard deviation (same units).
    sigma_floor : float
        Lower bound on σ_P; default 1e-6.

    Returns
    -------
    spi : jax.Array
        Standardized Precipitation Index (dimensionless z-score);
        negative for drought, positive for wet conditions.
    """
    sigma_safe = jnp.maximum(sigma_p, sigma_floor)
    return (precip - mu_p) / sigma_safe


def shear_squared_fv3(
    u: jax.Array,
    v: jax.Array,
    z: jax.Array,
) -> jax.Array:
    """FV3_3D iter 776: vertical-shear squared at layer midpoints.

        S² = (du/dz)² + (dv/dz)²

    Centered finite differences across adjacent layer pairs.
    Output shape ``(..., km−1)`` (matches iter-772 N² layout, so
    direct Ri = N²/S² composition is index-aligned).

    Used by iter-773 ``richardson_number_fv3`` (now delegates),
    KPP-style PBL mixing schemes, gravity-wave breakdown
    diagnostics, and clear-air-turbulence forecasting that
    inspects S² in isolation (independent of N²).

    Parameters
    ----------
    u, v : jax.Array, shape (..., km)
        Horizontal wind components (m/s).
    z : jax.Array, shape (..., km)
        Layer-center heights (m), matching wind orientation.

    Returns
    -------
    s_sq : jax.Array, shape (..., km−1)
        Vertical-shear squared at midpoints (s⁻²).
    """
    du = u[..., 1:] - u[..., :-1]
    dv = v[..., 1:] - v[..., :-1]
    dz = z[..., 1:] - z[..., :-1]
    return (du / dz) ** 2 + (dv / dz) ** 2


def richardson_number_fv3(
    theta: jax.Array,
    q_sphum: jax.Array,
    u: jax.Array,
    v: jax.Array,
    z: jax.Array,
    shear_floor: float = 1e-12,
) -> jax.Array:
    """FV3_3D iter 773: gradient Richardson number.

    Dimensionless ratio of stratification to vertical shear:

        Ri = N² / S²,    S² = (du/dz)² + (dv/dz)²

    N² is computed via iter-772 ``brunt_vaisala_squared_fv3``;
    shears are centered differences across the same layer
    midpoints.  Output is at layer midpoints with shape
    ``(..., km−1)``.

    Physical interpretation:

      * Ri < 0      — convectively unstable (N² < 0).
      * 0 ≤ Ri < ¼  — turbulent (Kelvin-Helmholtz instability).
      * Ri > 1      — strongly stable, turbulence damped.

    The classical critical value is Ri_c = ¼ (Miles-Howard).

    Used by: PBL turbulence onset/decay, KPP-style mixing schemes,
    clear-air turbulence (CAT) forecasting, gravity-wave breakdown
    detection.

    Parameters
    ----------
    theta : jax.Array, shape (..., km)
        Potential temperature (K).
    q_sphum : jax.Array, shape (..., km)
        Specific humidity (kg/kg); pass 0 for dry Ri.
    u, v : jax.Array, shape (..., km)
        Horizontal wind components (m/s).
    z : jax.Array, shape (..., km)
        Layer-center geopotential height (m).
    shear_floor : float
        Lower bound on |S²| to avoid div-by-0 in calm air.
        Default 1e-12 s⁻².

    Returns
    -------
    ri : jax.Array, shape (..., km−1)
        Gradient Richardson number at layer midpoints.
    """
    n_sq = brunt_vaisala_squared_fv3(theta, q_sphum, z)
    # iter-776: delegate (du/dz)² + (dv/dz)² to shear_squared_fv3
    s_sq = shear_squared_fv3(u, v, z)
    return n_sq / jnp.maximum(s_sq, shear_floor)


def bulk_richardson_fv3(
    theta_v_surf: jax.Array,
    theta_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    z: jax.Array,
    wind_sq_floor: float = 0.01,
) -> jax.Array:
    """FV3_3D iter 774: bulk Richardson number from surface to z.

    Layered (non-gradient) stability metric:

        Ri_b(z) = g · z · (θ_v(z) − θ_v_surf) /
                  (θ_v_surf · (u(z)² + v(z)²))

    Differs from the iter-773 gradient Ri (which uses ``dθ_v/dz``
    and ``du/dz`` at layer midpoints).  Ri_b integrates from the
    surface to a finite height z — used by Vogelezang-Holtslag,
    Troen-Mahrt, Holtslag-Boville PBL schemes to detect the
    boundary-layer top: PBL height = first level where Ri_b
    exceeds a critical value (≈ 0.25 over land, 0.5 over ocean).

    Caller passes virtual potential temperature (computed via
    ``_shared.virtual_temperature``) — Ri_b is canonically
    defined on θ_v, not θ.

    ``wind_sq_floor`` prevents div-by-0 at near-calm surface
    layers.  Default 0.01 m²/s² corresponds to a 0.1 m/s wind.

    Parameters
    ----------
    theta_v_surf : jax.Array
        Surface virtual potential temperature (K).  Typically
        ``theta_v[..., 0]`` for FV3 bottom-up arrays or
        ``theta_v[..., -1]`` for top-down arrays.
    theta_v : jax.Array, shape (..., km)
        Virtual potential temperature column (K).
    u, v : jax.Array, shape (..., km)
        Horizontal wind components (m/s).
    z : jax.Array, shape (..., km)
        Geopotential height above the surface (m).
    wind_sq_floor : float
        Lower bound on wind² (m²/s²); default 0.01 (0.1 m/s).

    Returns
    -------
    ri_b : jax.Array, shape (..., km)
        Bulk Richardson number at each level.
    """
    theta_v_surf_b = theta_v_surf[..., jnp.newaxis]
    wind_sq = u * u + v * v
    return (
        constants.g
        * z
        * (theta_v - theta_v_surf_b)
        / (theta_v_surf_b * jnp.maximum(wind_sq, wind_sq_floor))
    )


def pbl_height_fv3(
    ri_b: jax.Array,
    z: jax.Array,
    ri_crit: float = 0.25,
) -> jax.Array:
    """FV3_3D iter 775: PBL top from Ri_b threshold crossing.

    Holtslag-Boville convention: PBL height = lowest level at which
    the bulk Richardson number exceeds a critical value.

        z_pbl = z[k*]   where k* = argmin{k : Ri_b(k) > ri_crit}

    Standard ``ri_crit`` values:

      * 0.25  — land (Vogelezang-Holtslag 1996, Troen-Mahrt 1986)
      * 0.50  — ocean (Holtslag-Boville 1993)

    Vertical axis last; ``z`` and ``ri_b`` must share orientation
    with index 0 = surface, index km-1 = top.  Fallbacks:

      * No crossing in column (deep mixed or near-neutral): return
        z[..., −1] (top of column).
      * All Ri_b > ri_crit (strongly stable everywhere): return
        z[..., 0] (surface).

    Composes with iter-774 ``bulk_richardson_fv3`` for the
    (θ_v, u, v, z) → z_pbl pipeline.

    Parameters
    ----------
    ri_b : jax.Array, shape (..., km)
        Bulk Richardson number at each level (iter-774).
    z : jax.Array, shape (..., km)
        Geopotential height above surface at the same levels (m).
    ri_crit : float
        Critical Richardson value; default 0.25.

    Returns
    -------
    z_pbl : jax.Array, shape (...,)
        PBL-top height (m).
    """
    above = ri_b > ri_crit
    idx = jnp.argmax(above, axis=-1)
    any_above = jnp.any(above, axis=-1)
    top_idx = ri_b.shape[-1] - 1
    idx_safe = jnp.where(any_above, idx, top_idx)
    return jnp.take_along_axis(z, idx_safe[..., None], axis=-1)[..., 0]


def dew_point_fv3(
    e_mb: jax.Array,
) -> jax.Array:
    """FV3_3D iter 767: dew-point temperature from vapor pressure.

    Bolton (1980) eq. 11 inverse of Magnus saturation curve:

        γ      = ln(e / 6.112)
        T_d_C  = 243.5 · γ / (17.67 − γ)
        T_d_K  = T_d_C + 273.15

    The fitted coefficients 243.5, 17.67, 6.112 are Bolton-paper
    constants (saturation curve fit over 250-313 K).  The reference
    value 6.112 mb is the saturation vapor pressure at 0 °C.

    Composes with iter-766 ``vapor_pressure_from_q_fv3`` to give
    the (p, q) → T_d chain for relative-humidity diagnostics.

    Parameters
    ----------
    e_mb : jax.Array
        Water vapor partial pressure (mb).  Must be positive.

    Returns
    -------
    t_dew : jax.Array
        Dew-point temperature (K).
    """
    e_safe = jnp.maximum(1e-12, e_mb)
    gamma = jnp.log(e_safe / 6.112)
    t_dew_c = 243.5 * gamma / (17.67 - gamma)
    return t_dew_c + constants.T_freeze


def frost_point_temperature_fv3(
    p_pa: jax.Array,
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 803: frost-point temperature (Lawrence 2005 Magnus-ice).

    Inverse of saturation-over-ice curve.  Closed-form Magnus
    approximation (Lawrence 2005, BAMS):

        γ      = ln(e / 6.112)
        T_f_C  = 272.62 · γ / (22.46 − γ)
        T_f_K  = T_f_C + 273.15

    Valid range −80 °C ≤ T_f ≤ 0 °C.  Distinct from iter-767
    ``dew_point_fv3`` (Magnus liquid).  For subfreezing air
    T_frost > T_dew at the same q (e_sat_ice < e_sat_liquid).

    The ice and liquid Magnus curves both pass through
    e = 6.112 mb at T = T_freeze (0 °C), so T_frost(6.112) = T_dew
    (6.112) = T_freeze exactly.

    Used by: aviation icing forecasts (frost / hoar-frost
    deposition needs T < T_frost), polar stratospheric cloud
    (PSC) NAT/STS formation thresholds, cirrus cloud-base height
    estimation (T_frost is where ice first saturates), satellite-
    retrieval calibration (frost-point hygrometer reference).

    Composes iter-766 ``vapor_pressure_from_q_fv3``.

    Parameters
    ----------
    p_pa : jax.Array
        Pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    t_frost : jax.Array
        Frost-point temperature (K).
    """
    p_mb = p_pa / 100.0
    e = vapor_pressure_from_q_fv3(p_mb, q_sphum)
    e_safe = jnp.maximum(1e-12, e)
    gamma = jnp.log(e_safe / 6.112)
    t_f_c = 272.62 * gamma / (22.46 - gamma)
    return t_f_c + constants.T_freeze


def frost_point_depression_fv3(
    t: jax.Array,
    p_pa: jax.Array,
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 804: frost-point depression (T − T_frost).

    Ice analog of iter-770 ``dewpoint_depression_fv3``.  Composes
    iter-803 ``frost_point_temperature_fv3`` to give the canonical
    ice-saturation gap:

        T − T_frost ≥ 0   (zero at ice-saturation, positive in
                          sub-ice-saturated air).

    Used as a:
      * Cirrus/PSC cloud-base proxy (low T − T_frost → ice
        saturation aloft).
      * Aviation icing severity index (hoar-frost rate scales
        with T − T_frost gradient).
      * Polar-night H₂O sink diagnostic (T − T_frost ≈ 0 inside
        the polar vortex during polar-night cooling).

    Below freezing, T − T_frost < T − T_dew at the same q (since
    T_frost > T_dew below 0 °C).  Above freezing the two
    depressions formally coincide (Magnus liquid and ice curves
    meet at T_freeze).

    Parameters
    ----------
    t : jax.Array
        Temperature (K).
    p_pa : jax.Array
        Pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    depression : jax.Array
        T − T_frost (K, non-negative for physical sub-ice-saturated air).
    """
    t_frost = frost_point_temperature_fv3(p_pa, q_sphum)
    return t - t_frost


def relative_humidity_fv3(
    t: jax.Array,
    p_pa: jax.Array,
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 768: relative humidity from (T, p, q).

    RH [%] = 100 · e(p, q) / e_sat(T)

    where:
      * e(p, q) is vapor partial pressure from iter-766
        ``vapor_pressure_from_q_fv3`` (input p in Pa → output e in Pa).
      * e_sat(T) is the canonical legoesm saturation vapor pressure
        from ``thermo.saturation_vapor_pressure`` (Pa).

    Uses the WMO/ICAO standard RH definition (vapor-pressure ratio).
    Composes with iter-766 vapor-pressure and the shared
    ``thermo`` saturation curve to give a unit-consistent diagnostic.

    Parameters
    ----------
    t : jax.Array
        Temperature (K).
    p_pa : jax.Array
        Pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    rh : jax.Array
        Relative humidity (%, can exceed 100 for supersaturated air).
    """
    from legoesm import thermo

    e = vapor_pressure_from_q_fv3(p_pa, q_sphum)
    e_sat = thermo.saturation_vapor_pressure(t)
    return 100.0 * e / e_sat


def relative_humidity_ice_fv3(
    t: jax.Array,
    p_pa: jax.Array,
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 800: relative humidity over ice surface.

    Ratio of water-vapor mixing ratio to saturation mixing ratio
    over **ice**:

        RH_ice [%] = 100 · q / q_sat_ice(T, p)

    where q_sat_ice is the canonical ``thermo.saturation_mixing_ratio_ice``
    (Goff-Gratch / Murphy-Koop fit appropriate for T < 273.15 K).

    Distinct from iter-768 ``relative_humidity_fv3`` which uses
    the **liquid** saturation curve.  Below freezing, RH_liquid <
    RH_ice (since e_sat_ice < e_sat_liquid at same T).  Cloud
    schemes and mixed-phase microphysics need both; ice-
    supersaturation (RH_ice > 100% but RH_liquid < 100%) is the
    canonical pre-cloud cold-cloud regime.

    Used by: cirrus / mixed-phase cloud-formation thresholds
    (RH_ice > 100% triggers homogeneous nucleation), aircraft
    contrail forecasting (Schmidt-Appleman criterion needs
    RH_ice), upper-tropospheric H₂O diagnostics, polar
    stratospheric cloud (PSC) onset (RH_ice > 100% at PSC
    temperatures), satellite-IR moisture retrievals.

    Composes canonical ``thermo.saturation_mixing_ratio_ice`` per
    CLAUDE.md "use existing saturation curve" rule.

    Parameters
    ----------
    t : jax.Array
        Temperature (K).  ``saturation_mixing_ratio_ice`` is
        valid below T_freeze; warm-temperature output is
        formally extrapolative but caller can mask.
    p_pa : jax.Array
        Pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    rh_ice : jax.Array
        Relative humidity over ice (%, can exceed 100 for
        supersaturated air).
    """
    from legoesm import thermo

    q_sat_ice = thermo.saturation_mixing_ratio_ice(t, p_pa)
    return 100.0 * q_sphum / q_sat_ice


def ice_supersaturation_fv3(
    t: jax.Array,
    p_pa: jax.Array,
    q_sphum: jax.Array,
    rh_thresh: float = 140.0,
) -> jax.Array:
    """FV3_3D iter 801: ice-supersaturation mask.

    Boolean mask of points where relative humidity over ice
    exceeds a critical threshold:

        ISS = RH_ice > rh_thresh

    where ``RH_ice = 100·q/q_sat_ice`` from iter-800.

    Standard thresholds:
      * ``rh_thresh = 100``  — onset of homogeneous nucleation
        regime (formally any value > 100% can sustain ice).
      * ``rh_thresh = 140``  — Koop et al. (2000) homogeneous-
        nucleation threshold for aqueous-aerosol droplets (the
        default; cold-cloud-droplet → ice transition begins at
        140 % RH_ice).
      * ``rh_thresh = 160``  — Krämer et al. (2009) MOZAIC
        "cirrus cloud detection" upper bound; rarely sustained
        in observations.

    Used by: cirrus parameterization onset (CAM5-Liu, ECMWF
    IFS, GFDL-AM4 all use Koop-2000 140 % criterion), aircraft
    contrail forecasting (ISS region is a contrail-favorable
    region), MOZAIC/IAGOS upper-tropospheric H₂O climatology,
    polar stratospheric cloud (PSC) onset diagnostics.

    Composes iter-800 ``relative_humidity_ice_fv3``.

    Parameters
    ----------
    t : jax.Array
        Temperature (K).
    p_pa : jax.Array
        Pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).
    rh_thresh : float
        Critical RH_ice threshold (%, > 100).  Default 140
        (Koop-2000 homogeneous-nucleation threshold).

    Returns
    -------
    iss : jax.Array of bool
        True where RH_ice > rh_thresh.
    """
    rh_ice = relative_humidity_ice_fv3(t, p_pa, q_sphum)
    return rh_ice > rh_thresh


def contrail_appleman_fv3(
    t: jax.Array,
    p_pa: jax.Array,
    q_sphum: jax.Array,
    T_SA: float = 233.15,
    rh_ice_thresh: float = 100.0,
) -> jax.Array:
    """FV3_3D iter 802: simplified Schmidt-Appleman contrail criterion.

    Boolean mask of potential persistent-contrail region for
    conventional jet aviation:

        contrail = (T_amb < T_SA) AND (RH_ice ≥ rh_ice_thresh)

    Default ``T_SA = −40 °C = 233.15 K`` is the simplified
    Schmidt-Appleman threshold for jet engines (η ≈ 0.3,
    EI_H2O ≈ 1.25); below this T the engine-exhaust mixing line
    crosses the liquid-saturation curve, ice can homogeneously
    nucleate on cooling exhaust → contrail forms.  Default
    ``rh_ice_thresh = 100`` % distinguishes **persistent**
    contrails (RH_ice > 100% → contrail spreads into cirrus)
    from short-lived contrails (RH_ice < 100% → evaporates).

    The full Schmidt-Appleman criterion has a pressure-dependent
    threshold via the mixing slope G(p) = ε·EI_H2O·p / (LHV·(1−η)),
    with the tangent-to-saturation construction giving T_LC(p).
    This helper uses the constant T_SA simplification, which is
    accurate to ~3 K across jet flight levels.

    Used by: aircraft contrail-formation forecasting (NWS
    contrail charts), aviation-induced cirrus climatology
    (Burkhardt-Kärcher 2011, Schumann 2012 CoCiP framework),
    flight-routing optimization to avoid contrail regions,
    contrail-cirrus radiative-forcing assessment.

    Composes iter-800 ``relative_humidity_ice_fv3``.

    Parameters
    ----------
    t : jax.Array
        Temperature (K).
    p_pa : jax.Array
        Pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).
    T_SA : float
        Schmidt-Appleman temperature threshold (K).  Default
        233.15 K (−40 °C).  Pass a smaller value (e.g., 230 K)
        for more conservative detection, larger (e.g., 240 K)
        for aviation-cirrus-cover upper bound.
    rh_ice_thresh : float
        Persistent-contrail RH_ice threshold (%).  Default 100
        (formal definition of persistent contrail).

    Returns
    -------
    contrail : jax.Array of bool
        True where simplified Schmidt-Appleman criterion satisfied.
    """
    rh_ice = relative_humidity_ice_fv3(t, p_pa, q_sphum)
    return (t < T_SA) & (rh_ice >= rh_ice_thresh)


def lcl_temperature_fv3(
    pt: jax.Array,
    p_mb: jax.Array,
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 763: lifting-condensation-level temperature (Bolton 1980).

    Bolton (1980) eq. 21 LCL temperature formula:

        e     = vapor_pressure_from_q_fv3(p_mb, q)   (mb, iter-766)
        T_LCL = 2840 / (3.5·ln(T) − ln(e) − 4.805) + 55

    Used by iter-716 ``eqv_pot_bolton_fv3`` inline.  Extracted as
    standalone helper for SRH / supercell workflows that need T_LCL
    independently.  Iter-766 refactored e-from-q computation to
    delegate to ``vapor_pressure_from_q_fv3`` and use
    ``constants.epsilon`` instead of the Bolton-paper literal 622.

    Parameters
    ----------
    pt : jax.Array
        Temperature (K).
    p_mb : jax.Array
        Pressure (mb).
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    t_lcl : jax.Array
        LCL temperature (K).
    """
    e = vapor_pressure_from_q_fv3(p_mb, q_sphum)
    return 2840.0 / (3.5 * jnp.log(pt) - jnp.log(e) - 4.805) + 55.0


def lcl_state_fv3(
    pt: jax.Array,
    p_pa: jax.Array,
    q_sphum: jax.Array,
    z_parcel: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 769: full LCL state (T_LCL, p_LCL, z_LCL) in one call.

    Composes:

      * iter-763 ``lcl_temperature_fv3``  — Bolton (1980) eq. 21 T_LCL
      * iter-764 ``lcl_pressure_fv3``     — Poisson p_LCL = p·(T_LCL/T)^(1/κ)
      * iter-765 ``lcl_height_fv3``       — z_LCL = z + (c_pd/g)·(T − T_LCL)

    Returned triad is mutually consistent on the dry adiabat below
    LCL: in particular, since DSE = c_pd·T + g·z is conserved on
    the dry adiabat (by construction of iter-765) and q is
    conserved (no condensation below LCL), the MSE = DSE + L_v·q
    of the parcel and LCL endpoint match identically.  This
    property is verified in the iter-769 regression test.

    Single-call API for parcel-lift diagnostics, CAPE/CIN
    integrators, supercell parcel-source helpers, and convective-
    initiation triggers.

    Parameters
    ----------
    pt : jax.Array
        Parcel temperature (K).
    p_pa : jax.Array
        Parcel pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).
    z_parcel : jax.Array
        Parcel geopotential height (m).

    Returns
    -------
    (t_lcl, p_lcl, z_lcl) : tuple of jax.Array
        LCL temperature (K), pressure (Pa), height (m).
    """
    p_mb = p_pa / 100.0
    t_lcl = lcl_temperature_fv3(pt, p_mb, q_sphum)
    p_lcl = lcl_pressure_fv3(pt, p_pa, t_lcl)
    z_lcl = lcl_height_fv3(z_parcel, pt, t_lcl)
    return t_lcl, p_lcl, z_lcl


def dewpoint_depression_fv3(
    t: jax.Array,
    p_pa: jax.Array,
    q_sphum: jax.Array,
) -> jax.Array:
    """FV3_3D iter 770: dew-point depression (T − T_d).

    Composes iter-766 ``vapor_pressure_from_q_fv3`` and iter-767
    ``dew_point_fv3`` to give the canonical moisture-saturation gap:

        T − T_d ≥ 0   (zero at saturation, positive in subsaturated air).

    Used as a stability / cloud-base proxy in synoptic and severe-
    weather analysis.  Small T − T_d at 850 mb indicates moist
    boundary-layer air; large T − T_d above the cloud base indicates
    dry-air intrusion (entrainment / capping).

    Parameters
    ----------
    t : jax.Array
        Temperature (K).
    p_pa : jax.Array
        Pressure (Pa).
    q_sphum : jax.Array
        Specific humidity (kg/kg).

    Returns
    -------
    depression : jax.Array
        T − T_d (K, non-negative for physical subsaturated air).
    """
    p_mb = p_pa / 100.0
    e = vapor_pressure_from_q_fv3(p_mb, q_sphum)
    t_d = dew_point_fv3(e)
    return t - t_d


def saturation_deficit_column_fv3(
    p_full: jax.Array,
    t: jax.Array,
    qv: jax.Array,
    delp: jax.Array,
    do_cmip: bool = False,
) -> jax.Array:
    """FV3_3D iter 762: column saturation deficit (kg/m²).

        Q_sat_col = column_integral(q_sat, delp)
        SatDef = Q_sat_col − PWV

    Mass-equivalent of (100 − RH) integrated over column.  Indicator
    of how much additional water vapor a column could hold before
    saturating.

    Composes iter-742 column_integral + iter-756 precipitable_water
    + legoesm.thermo saturation helpers (with optional CMIP blend).

    Parameters
    ----------
    p_full : jax.Array, shape (..., km)
        Layer-center pressure (Pa).
    t : jax.Array, shape (..., km)
        Temperature (K).
    qv : jax.Array, shape (..., km)
        Specific humidity (kg/kg).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    do_cmip : bool, default False.
        Use es-over-liq-and-ice blend.

    Returns
    -------
    sat_def : jax.Array, shape (...,)
        Column saturation deficit (kg/m²).  Non-negative for sub-saturated.
    """
    from legoesm import thermo
    if do_cmip:
        qs = thermo.saturation_mixing_ratio_blend(t, p_full)
    else:
        qs = thermo.saturation_mixing_ratio(t, p_full)
    qs_col = column_integral_delp_fv3(qs, delp)
    pwv = column_integral_delp_fv3(qv, delp)
    return qs_col - pwv


def column_mean_field_fv3(
    field: jax.Array,
    delp: jax.Array,
) -> jax.Array:
    """FV3_3D iter 761: mass-weighted column-mean field.

        <field> = Σ_k delp · field / Σ_k delp

    Generic helper for any mass-weighted column average (T, RH, q,
    θ_e, etc.).  Used inline by iter-760 ``column_mean_rh_fv3`` and
    pattern-matched in many FV3 diagnostics.

    Mass cancellation: ratio of two delp integrals is g-independent.

    Parameters
    ----------
    field : jax.Array, shape (..., km)
        Layer-mean values to average.
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).

    Returns
    -------
    mean : jax.Array, shape (...,)
        Mass-weighted column-mean field.
    """
    sum_field = jnp.sum(delp * field, axis=-1)
    sum_delp = jnp.sum(delp, axis=-1)
    safe = jnp.where(sum_delp > 0.0, sum_delp, 1.0)
    return sum_field / safe


def column_mean_rh_fv3(
    p_full: jax.Array,
    t: jax.Array,
    qv: jax.Array,
    delp: jax.Array,
    do_cmip: bool = False,
) -> jax.Array:
    """FV3_3D iter 760: mass-weighted column-mean relative humidity.

        RH_col = Σ_k delp · RH_layer / Σ_k delp

    Mass-weighted (not / g — ratio cancels), so divide_by_g=False in
    the layer-RH integral and then divide by total delp.

    Composes iter-715 ``rh_calc_fv3`` (with optional do_cmip) +
    iter-742 ``column_integral_delp_fv3``.

    Parameters
    ----------
    p_full : jax.Array, shape (..., km)
        Layer-center pressure (Pa).
    t : jax.Array, shape (..., km)
        Temperature (K).
    qv : jax.Array, shape (..., km)
        Specific humidity (kg/kg).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    do_cmip : bool, default False.
        es-over-liq-and-ice blend (CMIP convention).

    Returns
    -------
    rh_col : jax.Array, shape (...,)
        Column-mean relative humidity (percent).
    """
    # iter-761: delegate mass-weighted mean to column_mean_field_fv3
    rh_layer = rh_calc_fv3(p_full, t, qv, do_cmip=do_cmip)
    return column_mean_field_fv3(rh_layer, delp)


def rh_calc_fv3(
    p_full: jax.Array,
    t: jax.Array,
    qv: jax.Array,
    do_cmip: bool = False,
) -> jax.Array:
    """FV3_3D iter 695/715: relative humidity diagnostic (percent).

    Faithful JAX port of FV3 ``rh_calc``
    (tools/fv_diagnostics.F90:5309-5339).

        RH = 100 · qv / qs(T, p_full)

    Two saturation-reference options:
        * ``do_cmip=False`` (iter-695 default): saturation over liquid
          via ``thermo.saturation_mixing_ratio``.
        * ``do_cmip=True`` (iter-715, CMIP convention): saturation
          blended over liquid + ice — pure liquid above T_freeze,
          pure ice below T_freeze − 20 K, linear blend in between.
          Matches FV3's ``compute_qs(... es_over_liq_and_ice=.true.)``
          path.

    Reuses legoesm ``thermo.saturation_mixing_ratio`` and
    ``saturation_mixing_ratio_ice`` (CLAUDE.md mandate: never
    re-derive Tetens / Magnus / Clausius-Clapeyron).

    Parameters
    ----------
    p_full : jax.Array
        Layer-center pressure (Pa).
    t : jax.Array
        Air temperature (K).
    qv : jax.Array
        Specific humidity (kg/kg).
    do_cmip : bool, default False.
        Use es-over-liq-and-ice blend (CMIP convention).

    Returns
    -------
    rh : jax.Array
        Relative humidity (percent).
    """
    from legoesm import thermo
    if do_cmip:
        # iter-720: refactored to reuse thermo.saturation_mixing_ratio_blend
        qs = thermo.saturation_mixing_ratio_blend(t, p_full)
    else:
        qs = thermo.saturation_mixing_ratio(t, p_full)
    return 100.0 * qv / qs


def prt_mass_fv3(
    ps: jax.Array,
    delp: jax.Array,
    q_tracers: dict[str, jax.Array],
    area: jax.Array,
) -> dict[str, float]:
    """FV3_3D iter 694: global mass-budget diagnostic.

    Faithful JAX port of FV3 ``prt_mass``
    (tools/fv_diagnostics.F90:4164-4263).

    Computes the column-integrated mass of each water tracer
    (kg/m²) and the area-weighted global mean.  Used as a
    conservation check at runtime.

    Returns a dict with:

        ``ps_mean``      : global-mean surface pressure (Pa)
        ``dry_ps_mean``  : ps_mean - total water mass · g (Pa)
        ``<tracer>``     : global mean column mass (kg/m²) per
                            tracer name in ``q_tracers``
        ``total_water``  : sum of column water across all tracers (kg/m²)

    Algorithm:

        ps_mean = Σ area · ps / Σ area
        For each tracer:
            column[i,j] = Σ_k delp[i,j,k] · q[i,j,k] / g
            global[name] = Σ area · column / Σ area
        total_water = Σ_tracers global[name]
        dry_ps_mean = ps_mean - g · total_water    (= ps_mean - total_water · g)

    Parameters
    ----------
    ps : jax.Array, shape (n_x, n_y)
        Surface pressure (Pa).
    delp : jax.Array, shape (n_x, n_y, km)
        Layer pressure thickness (Pa).
    q_tracers : dict[str, jax.Array]
        Mapping from tracer name (e.g. ``"sphum"``, ``"liq_wat"``,
        ``"ice_wat"``, ``"rainwat"``, ``"snowwat"``, ``"graupel"``)
        to mixing ratio array of shape (n_x, n_y, km).
    area : jax.Array, shape (n_x, n_y)
        Cell area (m²).

    Returns
    -------
    diag : dict[str, float]
        Mass-budget summary.  All values float.
    """
    # iter-751: delegate area-weighted means to area_weighted_mean_fv3
    g = constants.g
    ps_mean = float(area_weighted_mean_fv3(ps, area))

    diag: dict[str, float] = {"ps_mean": ps_mean}
    total_water_col_mean = 0.0
    for name, q in q_tracers.items():
        col_mass = column_integral_delp_fv3(q, delp)
        col_mean = float(area_weighted_mean_fv3(col_mass, area))
        diag[name] = col_mean
        total_water_col_mean += col_mean
    diag["total_water"] = total_water_col_mean
    diag["dry_ps_mean"] = ps_mean - g * total_water_col_mean
    return diag


def moist_cp_fv3(
    q_sphum: jax.Array | None = None,
    q_liq_wat: jax.Array | None = None,
    q_rainwat: jax.Array | None = None,
    q_ice_wat: jax.Array | None = None,
    q_snowwat: jax.Array | None = None,
    q_graupel: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 714: moisture-weighted isobaric specific heat.

    Faithful JAX port of FV3 ``moist_cp`` (general nwat≥3 branch)
    (model/fv_mapz.F90:3656-3733).  Companion to iter-713
    ``moist_cv_fv3`` — uses c_pd / c_pv / c_pw / c_pi directly
    (no isochoric R subtraction):

        q_l = (liq_wat or 0) + (rainwat or 0)
        q_i = (ice_wat or 0) + (snowwat or 0) + (graupel or 0)
        q_d = q_l + q_i
        cpm = (1 − q_sphum − q_d) · c_pd
              + q_sphum · c_pv
              + q_l · c_pw
              + q_i · c_pi

    Parameters
    ----------
    q_sphum, q_liq_wat, q_rainwat, q_ice_wat, q_snowwat, q_graupel :
        Mixing ratios (kg/kg).  Any subset can be provided.

    Returns
    -------
    cpm : jax.Array
        Moisture-weighted isobaric specific heat (J/(kg·K)).
    q_con : jax.Array
        Total condensate mass fraction (kg/kg).
    """
    qs = next(
        (q for q in (q_sphum, q_liq_wat, q_rainwat, q_ice_wat, q_snowwat,
                     q_graupel) if q is not None),
        None,
    )
    if qs is None:
        raise ValueError("moist_cp_fv3 requires at least one tracer input")
    zero = jnp.zeros_like(qs)

    qv = jnp.maximum(0.0, q_sphum if q_sphum is not None else zero)
    ql = (q_liq_wat if q_liq_wat is not None else zero) + (
        q_rainwat if q_rainwat is not None else zero
    )
    qi = (q_ice_wat if q_ice_wat is not None else zero) + (
        q_snowwat if q_snowwat is not None else zero
    ) + (q_graupel if q_graupel is not None else zero)
    q_con = ql + qi

    cpm = (
        (1.0 - qv - q_con) * constants.c_pd
        + qv * constants.c_pv
        + ql * constants.c_pw
        + qi * constants.c_pi
    )
    return cpm, q_con


def moist_cv_fv3(
    q_sphum: jax.Array | None = None,
    q_liq_wat: jax.Array | None = None,
    q_rainwat: jax.Array | None = None,
    q_ice_wat: jax.Array | None = None,
    q_snowwat: jax.Array | None = None,
    q_graupel: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 713: moisture-weighted isochoric specific heat.

    Faithful JAX port of FV3 ``moist_cv`` (general nwat≥3 branch)
    (model/fv_mapz.F90:3579-3654).

    Returns layer-wise moisture-weighted ``cvm`` and total condensate
    ``q_con = q_liquid + q_ice``:

        cv_air = c_pd − R_d
        cv_vap = c_pv − R_v
        q_l = (liq_wat or 0) + (rainwat or 0)
        q_i = (ice_wat or 0) + (snowwat or 0) + (graupel or 0)
        q_d = q_l + q_i                        (total condensate)
        cvm = (1 − q_sphum − q_d) · cv_air
              + q_sphum · cv_vap
              + q_l · c_pw
              + q_i · c_pi

    All inputs are optional; missing components default to zero.
    Uses legoesm constants: ``c_pd``, ``R_d``, ``c_pv``, ``R_v``,
    ``c_pw``, ``c_pi``.

    Parameters
    ----------
    q_sphum, q_liq_wat, q_rainwat, q_ice_wat, q_snowwat, q_graupel :
        Mixing ratios (kg/kg).  Any subset can be provided.

    Returns
    -------
    cvm : jax.Array
        Moisture-weighted isochoric specific heat (J/(kg·K)).
    q_con : jax.Array
        Total condensate mass fraction (kg/kg).
    """
    cv_air = constants.c_pd - constants.R_d
    cv_vap = constants.c_pv - constants.R_v
    # Reference shape from any non-None input
    qs = next(
        (q for q in (q_sphum, q_liq_wat, q_rainwat, q_ice_wat, q_snowwat,
                     q_graupel) if q is not None),
        None,
    )
    if qs is None:
        raise ValueError("moist_cv_fv3 requires at least one tracer input")
    zero = jnp.zeros_like(qs)

    qv = jnp.maximum(0.0, q_sphum if q_sphum is not None else zero)
    ql = (q_liq_wat if q_liq_wat is not None else zero) + (
        q_rainwat if q_rainwat is not None else zero
    )
    qi = (q_ice_wat if q_ice_wat is not None else zero) + (
        q_snowwat if q_snowwat is not None else zero
    ) + (q_graupel if q_graupel is not None else zero)
    q_con = ql + qi

    cvm = (
        (1.0 - qv - q_con) * cv_air
        + qv * cv_vap
        + ql * constants.c_pw
        + qi * constants.c_pi
    )
    return cvm, q_con


def nh_total_energy_fv3(
    ua: jax.Array, va: jax.Array, w: jax.Array,
    pt: jax.Array, delp: jax.Array, delz: jax.Array,
    hs: jax.Array,
    q_sphum: jax.Array | None = None,
    moist_phys: bool = False,
    use_moist_cv: bool = False,
    q_liq_wat: jax.Array | None = None,
    q_rainwat: jax.Array | None = None,
    q_ice_wat: jax.Array | None = None,
    q_snowwat: jax.Array | None = None,
    q_graupel: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 693: vertically-integrated total energy per column.

    Faithful JAX port of FV3 ``nh_total_energy``
    (tools/fv_diagnostics.F90:5501-5571).

    Per-column total energy (J/m²):

        phiz[km] = hs                               (surface geopot)
        phiz[k]  = phiz[k+1] − g · delz[k]          (cumul. upward)
        TE = (1/g) · Σ_k  delp[k] · ( cv · pt[k]
                                     + L_v · q_sphum[k]            (moist)
                                     + 0.5·(phiz[k] + phiz[k+1])
                                     + 0.5·(ua²+va²+w²) )

    where ``cv = c_pd − R_d`` (dry isochoric specific heat).
    Moist-physics branch (full FV3) uses a moisture-weighted cv via
    ``moist_cv``; here we use dry cv with an explicit L_v·q term
    which is the leading-order approximation correct for
    sphum-only moist energy.

    Parameters
    ----------
    ua, va, w : jax.Array, shape (..., km)
        Wind components (a-grid).
    pt : jax.Array, shape (..., km)
        Temperature (K).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3).
    hs : jax.Array, shape (...,)
        Surface geopotential height·g (m²/s²).
    q_sphum : jax.Array, shape (..., km), optional
        Specific humidity — required if moist_phys=True.
    moist_phys : bool, default False.

    Returns
    -------
    te : jax.Array, shape (...,)
        Column total energy (J/m²).
    """
    # iter-748: delegate to 4 column-energy helpers (744/745/746/747).
    # Build phi_interfaces from delz + hs (cumulative −g·delz from surface up).
    g = constants.g
    minus_g_delz = -g * delz                              # (..., km)
    cum_up = jnp.cumsum(minus_g_delz[..., ::-1], axis=-1)[..., ::-1]
    phi_above = hs[..., None] + cum_up                    # (..., km)
    phi_interfaces = jnp.concatenate(
        [phi_above, hs[..., None]],
        axis=-1,
    )                                                      # (..., km+1)
    # PE column (iter-747)
    pe_col = potential_energy_column_fv3(phi_interfaces, delp)
    # KE column (iter-745)
    ke_col = kinetic_energy_column_fv3(ua, va, delp, w=w)
    if moist_phys:
        if q_sphum is None:
            raise ValueError("moist_phys=True requires q_sphum")
        if use_moist_cv:
            cvm, _ = moist_cv_fv3(
                q_sphum=q_sphum,
                q_liq_wat=q_liq_wat, q_rainwat=q_rainwat,
                q_ice_wat=q_ice_wat, q_snowwat=q_snowwat,
                q_graupel=q_graupel,
            )
            ie_col = internal_energy_column_fv3(pt, delp, cv=cvm)
        else:
            ie_col = internal_energy_column_fv3(pt, delp)
        le_col = latent_energy_column_fv3(q_sphum, delp)
        return ie_col + ke_col + pe_col + le_col
    else:
        ie_col = internal_energy_column_fv3(pt, delp)
        return ie_col + ke_col + pe_col


def eqv_pot_bolton_fv3(
    pt: jax.Array,
    delp: jax.Array,
    q: jax.Array,
    delz: jax.Array | None = None,
    peln: jax.Array | None = None,
    hydrostatic: bool = False,
    moist: bool = True,
) -> jax.Array:
    """FV3_3D iter 716: equivalent potential temperature (Bolton 1980).

    Faithful JAX port of FV3's Xi.Chen Bolton-form ``eqv_pot``
    (tools/fv_diagnostics.F90:5421-5497).  Alternative to iter-692
    simplified S.-J. Lin form — Bolton 1980 thermodynamics gives a
    more accurate θ_e using the lifting-condensation-level temp.

    Algorithm:

        p_mb (mb) = hydrostatic: delp/Δpeln · 0.01
                  | non-hydro  : −R_d/g · delp/delz · pt · (1+zvir·q) · 0.01

        Moist:
          cappa = R_d / (R_d + ((1-q)·cv_air + q·cv_vap)/(1+zvir·q))
          r     = q/(1-q) · 1000              (dry mixing ratio, g/kg)
          e     = p_mb · r / (622 + r)        (vapor pressure, mb)
          T_LCL = 2840 / (3.5·ln(T) − ln(e) − 4.805) + 55   (Bolton eq. 21)
          capa  = cappa · (1 − r · 0.28e-3)
          θ_e   = T · (1000/p_mb)^capa
                  · exp((3.376/T_LCL − 0.00254) · r · (1 + r · 0.81e-3))

        Dry: θ_e = T · (1000/p_mb)^kappa

    Parameters
    ----------
    pt : jax.Array, shape (..., km)
        Air temperature (K).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    q : jax.Array, shape (..., km)
        Specific humidity (kg/kg).
    delz : jax.Array, shape (..., km), optional
        Layer thickness — required if not hydrostatic.
    peln : jax.Array, shape (..., km+1), optional
        log(pressure) at interfaces — required if hydrostatic.
    hydrostatic, moist : bool

    Returns
    -------
    theta_e : jax.Array, shape (..., km)
        Equivalent potential temperature (K).
    """
    cv_air = constants.c_pd - constants.R_d
    cv_vap = constants.c_pv - constants.R_v
    zvir = (constants.R_v / constants.R_d - 1.0) if moist else 0.0
    rdg = -constants.R_d / constants.g

    if hydrostatic:
        if peln is None:
            raise ValueError("hydrostatic=True requires peln")
        # iter-733: delegate hydrostatic layer-center p to layer_mean_pressure
        p_mb = 0.01 * layer_mean_pressure_fv3(delp, peln)
    else:
        if delz is None:
            raise ValueError("hydrostatic=False requires delz")
        p_mb = 0.01 * rdg * delp / delz * pt * (1.0 + zvir * q)

    if moist:
        cappa = cappa_moist_fv3(q, zvir=zvir)
        # iter-763: delegate T_LCL to lcl_temperature_fv3
        t_l = lcl_temperature_fv3(pt, p_mb, q)
        # iter-771: delegate r = q/(1-q) to mixing_ratio_fv3 (×1000 for g/kg)
        r = mixing_ratio_fv3(q) * 1000.0
        capa = cappa * (1.0 - r * 0.28e-3)
        return jnp.exp(
            (3.376 / t_l - 0.00254) * r * (1.0 + r * 0.81e-3)
        ) * pt * (1000.0 / p_mb) ** capa
    else:
        return pt * jnp.exp(constants.kappa * jnp.log(1000.0 / p_mb))


def eqv_pot_fv3(
    pt: jax.Array,
    delp: jax.Array,
    q: jax.Array,
    delz: jax.Array | None = None,
    peln: jax.Array | None = None,
    hydrostatic: bool = False,
    moist: bool = True,
) -> jax.Array:
    """FV3_3D iter 692: equivalent potential temperature θ_e.

    Faithful JAX port of FV3 ``eqv_pot``
    (tools/fv_diagnostics.F90:5341-5419).

    Simplified S.-J. Lin form (RH term ignored).  Reference dry
    pressure ``pd`` from either hydrostatic ``delp / Δpeln`` or
    non-hydrostatic ``ρ R_d T`` form.  Moist branch uses the
    full Bolton-style expression with the c_pv − c_pw isobaric
    heating correction; dry branch reduces to standard Poisson.

    Algorithm:

        rq = q if moist else 0  (clipped to ≥ 0)
        Hydrostatic:
            pd = (1 − rq) · delp / (peln[k+1] − peln[k])
        Non-hydrostatic:
            pd = − R_d · pt · (1 − rq) · delp / (g · delz)

        Moist:
            dc_vap = c_pv − c_pw
            θ_e = pt · exp( rq/(c_pd·pt) · (L_v + dc_vap·(pt − T_freeze))
                           + κ · ln(1e5/pd) )
        Dry:
            θ_e = pt · (1e5/pd)^κ

    Parameters
    ----------
    pt : jax.Array, shape (..., km)
        Air temperature T (K).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    q : jax.Array, shape (..., km)
        Specific humidity (kg/kg).
    delz : jax.Array, shape (..., km), optional
        Layer thickness (NEGATIVE in FV3) — required if not hydrostatic.
    peln : jax.Array, shape (..., km+1), optional
        log(pressure) at interfaces — required if hydrostatic.
    hydrostatic : bool, default False.
    moist : bool, default True.

    Returns
    -------
    theta_e : jax.Array, shape (..., km)
        Equivalent potential temperature (K).
    """
    # iter-729: delegate dry-pressure computation to dry_pressure_fv3
    pd = dry_pressure_fv3(
        delp, q=q, pt=pt, delz=delz, peln=peln,
        hydrostatic=hydrostatic, moist=moist,
    )
    poisson = constants.kappa * jnp.log(1.0e5 / pd)
    if moist:
        dc_vap = constants.c_pv - constants.c_pw
        rq_full = jnp.maximum(0.0, q)  # FV3 re-reads q here w/o wfac mask
        moist_exp = rq_full / (constants.c_pd * pt) * (
            constants.L_v + dc_vap * (pt - constants.T_freeze)
        )
        return pt * jnp.exp(moist_exp + poisson)
    else:
        return pt * jnp.exp(poisson)


def cs3_interpolator_fv3(
    qin: jax.Array,
    pe: jax.Array,
    pout: jax.Array,
    iv: int = 0,
    wz_surface: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 710/717: log-p-level vertical interp via PPM (multi-level).

    Faithful JAX port of FV3 ``cs3_interpolator``
    (tools/fv_diagnostics.F90:4510-4602).  Uses iter-708
    ``cs_prof_fv3`` for PPM edge values + subcell parabolic
    distribution.

    Differs from iter-709 ``cs_interpolator_fv3``:
        * Interpolation coordinate is log-p (``pe``), not height (``wz``).
        * Output is at multiple levels (``pout`` is shape ``(kd,)``).
        * Supports tracer type ``iv``: -1 winds, 0 positive scalar,
          1 temperature.

    iter-717: below-surface temperature (``iv==1``) now uses the
    FV3-faithful ECMWF (Trenberth 1993) extrapolation when
    ``wz_surface`` is provided:

        alpha = 0.0065 · R_d / g
        pbot  = (exp(pe[km]) − exp(pe[km-1])) / (pe[km] − pe[km-1])
        ts    = q2[km-1] + alpha · q2[km-1] · (exp(pe[km])/pbot − 1)
        t0    = ts + 0.0065 · wz_surface
        tmp   = min(t0, 298 K)
        For wz_surface in [2000, 2500]:
            tmp = 0.002·((2500 − wz)·t0 + (wz − 2000)·tmp)
            alpha = (R_d·(tmp − ts) / (wz · g)) if tmp > ts else 0
        qout = ts · exp(alpha · (pout − pe[km]))

    If ``iv==1`` but ``wz_surface`` is None, falls back to
    edge-value clamp (iter-710 behaviour).

    Parameters
    ----------
    qin : jax.Array, shape (..., km)
        Cell-center values.
    pe : jax.Array, shape (..., km+1)
        log-pressure at interfaces.
    pout : jax.Array, shape (kd,)
        Target log-pressure levels (monotonic increasing).
    iv : int, default 0
        Variable type: -1 winds, 0 positive scalar, 1 temperature.
    wz_surface : jax.Array, shape (...,), optional
        Surface elevation (m).  Required for iv==1 ECMWF extrapolation
        (iter-717); ignored otherwise.

    Returns
    -------
    qout : jax.Array, shape (..., kd)
        Interpolated values per output level.
    """
    km = qin.shape[-1]
    dp = pe[..., 1:] - pe[..., :-1]                    # (..., km)
    qe = cs_prof_fv3(qin, dp, iv=1 if iv != 0 else 0)  # (..., km+1)

    a6 = 3.0 * (2.0 * qin - (qe[..., :-1] + qe[..., 1:]))   # (..., km)
    safe_dp = jnp.where(dp > 0.0, dp, 1.0)

    # iter-717 below-surface T extrapolation (Trenberth 1993)
    use_ecmwf_bot = iv == 1 and wz_surface is not None
    if use_ecmwf_bot:
        alpha0 = 0.0065 * constants.R_d / constants.g
        pe_kp1 = pe[..., -1]                # log-p at surface
        pe_k = pe[..., -2]                  # log-p at top of bottom layer
        # pbot = (exp(pe[km]) - exp(pe[km-1])) / (pe[km] - pe[km-1])
        pbot = (jnp.exp(pe_kp1) - jnp.exp(pe_k)) / (pe_kp1 - pe_k)
        q2_km = qin[..., -1]                # bottom-layer T
        ts = q2_km + alpha0 * q2_km * (jnp.exp(pe_kp1) / pbot - 1.0)
        t0 = ts + 0.0065 * wz_surface
        tmp_capped = jnp.minimum(t0, 298.0)
        # High-terrain blend: wz in [2000, 2500]
        wz_in_blend = (wz_surface >= 2000.0) & (wz_surface <= 2500.0)
        wz_ge_2000 = wz_surface >= 2000.0
        tmp_blended = jnp.where(
            wz_in_blend,
            0.002 * ((2500.0 - wz_surface) * t0
                     + (wz_surface - 2000.0) * tmp_capped),
            tmp_capped,
        )
        tmp_final = jnp.where(wz_ge_2000, tmp_blended, tmp_capped)
        # alpha recomputed for high terrain
        safe_wz = jnp.where(wz_surface > 0.0, wz_surface, 1.0)
        alpha_high = jnp.where(
            tmp_final > ts,
            constants.R_d * (tmp_final - ts) / (safe_wz * constants.g),
            0.0,
        )
        alpha = jnp.where(wz_ge_2000, alpha_high, alpha0)

    def _interp_one(p):
        pe_top = pe[..., :-1]
        pe_bot = pe[..., 1:]
        in_layer = (pe_top <= p) & (p < pe_bot)
        s0 = (p - pe_top) / safe_dp
        q_k = qe[..., :-1] + s0 * (
            qe[..., 1:] - qe[..., :-1] + a6 * (1.0 - s0)
        )
        q_in_range = jnp.sum(jnp.where(in_layer, q_k, 0.0), axis=-1)
        above_top = p < pe[..., 0]
        below_bot = p >= pe[..., -1]
        q = jnp.where(above_top, qe[..., 0], q_in_range)
        if use_ecmwf_bot:
            q_below = ts * jnp.exp(alpha * (p - pe[..., -1]))
            q = jnp.where(below_bot, q_below, q)
        else:
            q = jnp.where(below_bot, qe[..., -1], q)
        if iv == 0:
            q = jnp.maximum(0.0, q)
        return q

    qout_per_level = jax.vmap(_interp_one, in_axes=0, out_axes=-1)(pout)
    return qout_per_level


def ppme_fv3(
    p: jax.Array,
    delp: jax.Array,
) -> jax.Array:
    """FV3_3D iter 711: PPM cell-edge values with non-uniform delp.

    Faithful JAX port of FV3 ``ppme``
    (tools/fv_diagnostics.F90:5196-5305).

    Builds km+1 edge values from km cell averages using Lin (1996)
    PPM with non-uniform layer thicknesses.  Interior uses 4th-order
    edge formula with Van-Leer-limited monotone slopes; top edge uses
    3-cell parabolic with discriminant fallback to linear; second cell
    (k=1) uses off-centered area-preserving cubic; bottom 2 edges use
    area-preserving cubic with 2nd-derivative=0 at surface.

    Companion of iter-708 ``cs_prof_fv3`` (tridiagonal PPM edges).
    Used by FV3 internally for vertical remap and edge reconstruction.

    Parameters
    ----------
    p : jax.Array, shape (..., km)
        Cell-mean values.  Minimum km = 4 (top + bottom closures use
        4 cells).
    delp : jax.Array, shape (..., km)
        Layer thickness (positive).

    Returns
    -------
    qe : jax.Array, shape (..., km+1)
        PPM edge values at interfaces.
    """
    km = p.shape[-1]
    if km < 4:
        raise ValueError(f"ppme_fv3 requires km >= 4, got km={km}")

    # FV3 a6[k] = delp[k-1] + delp[k] for k=2..km   (Python a6[k] for k=1..km-1)
    a6 = jnp.zeros_like(delp)
    a6 = a6.at[..., 1:].set(delp[..., :-1] + delp[..., 1:])

    # FV3 delq[k] = p[k+1] - p[k] for k=1..km-1   (Python delq[k] for k=0..km-2)
    delq = jnp.zeros_like(p)
    delq = delq.at[..., :-1].set(p[..., 1:] - p[..., :-1])

    # Limited monotone slope dc[k] for k=2..km-1 (Python k=1..km-2)
    p_km1 = p[..., :-2]                    # p[k-1], k=1..km-2  (Python idx 0..km-3)
    p_k = p[..., 1:-1]                     # p[k]
    p_kp1 = p[..., 2:]                     # p[k+1]
    delp_km1 = delp[..., :-2]
    delp_k = delp[..., 1:-1]
    delp_kp1 = delp[..., 2:]
    a6_kp1 = a6[..., 2:]                   # a6[k+1]   k=1..km-2
    a6_k = a6[..., 1:-1]                   # a6[k]
    delq_k = delq[..., 1:-1]               # delq[k]  k=1..km-2 (Python idx)
    delq_km1 = delq[..., :-2]              # delq[k-1]
    c1 = (delp_km1 + 0.5 * delp_k) / a6_kp1
    c2 = (delp_kp1 + 0.5 * delp_k) / a6_k
    tmp = delp_k * (c1 * delq_k + c2 * delq_km1) / (a6_k + delp_kp1)
    qmax = jnp.maximum(jnp.maximum(p_km1, p_k), p_kp1) - p_k
    qmin = p_k - jnp.minimum(jnp.minimum(p_km1, p_k), p_kp1)
    dc_interior = jnp.sign(tmp) * jnp.minimum(
        jnp.minimum(jnp.abs(tmp), qmax), qmin,
    )
    dc = jnp.zeros_like(p)
    dc = dc.at[..., 1:-1].set(dc_interior)

    # 4th-order interior edge values qe[k] for k=3..km-1 (Python k=2..km-2)
    # Uses delp[k-1], delp[k], a6[k-1], a6[k], a6[k+1], dc[k-1], dc[k]
    delp_km1_i = delp[..., 1:-2]          # delp[k-1]   k=2..km-2 (Python idx 1..km-3)
    delp_k_i = delp[..., 2:-1]            # delp[k]
    a6_km1_i = a6[..., 1:-2]              # a6[k-1]
    a6_k_i = a6[..., 2:-1]
    a6_kp1_i = a6[..., 3:]                # a6[k+1]
    dc_km1_i = dc[..., 1:-2]              # dc[k-1]
    dc_k_i = dc[..., 2:-1]
    delq_km1_i = delq[..., 1:-2]          # delq[k-1]
    p_km1_i = p[..., 1:-2]                # p[k-1]

    c1_i = delq_km1_i * delp_km1_i / a6_k_i
    a1_i = a6_km1_i / (a6_k_i + delp_km1_i)
    a2_i = a6_kp1_i / (a6_k_i + delp_k_i)
    qe_interior = p_km1_i + c1_i + 2.0 / (a6_km1_i + a6_kp1_i) * (
        delp_k_i * (c1_i * (a1_i - a2_i) + a2_i * dc_km1_i)
        - delp_km1_i * a1_i * dc_k_i
    )
    qe = jnp.zeros(p.shape[:-1] + (km + 1,), dtype=p.dtype)
    # qe_interior occupies Python idx 2..km-2  (size km-3)
    qe = qe.at[..., 2:km - 1].set(qe_interior)

    # Top cell k=0: 3-cell parabolic with discriminant check
    s1 = delp[..., 0]
    s2 = delp[..., 1] + s1
    a3_top = (delq[..., 1] - delq[..., 0] * (delp[..., 1] + delp[..., 2]) / s2) / (
        (delp[..., 1] + delp[..., 2]) * ((delp[..., 1] + delp[..., 2]) + s1)
    )
    b2_top = delq[..., 0] / s2 - a3_top * (s1 + s2)
    sc_top = jnp.where(jnp.abs(a3_top) > 1e-14, -b2_top / (3.0 * a3_top + 1e-300), -1.0)
    qe_top_parabolic = p[..., 0] - s1 * (a3_top * s1 + b2_top)
    qe_top_linear = p[..., 0] - delq[..., 0] * s1 / s2
    use_linear = (jnp.abs(a3_top) <= 1e-14) | (sc_top < 0.0) | (sc_top > s1)
    qe_top = jnp.where(use_linear, qe_top_linear, qe_top_parabolic)
    qe = qe.at[..., 0].set(qe_top)
    dc = dc.at[..., 0].set(p[..., 0] - qe_top)

    # k=1 off-centered area-preserving cubic
    s3 = delp[..., 1] + delp[..., 2]
    s4 = s3 + delp[..., 3]
    ss3 = s3 + s1
    s32 = s3 * s3
    s42 = s4 * s4
    s34 = s3 * s4
    dm = delp[..., 0] / (s34 * ss3 * (delp[..., 1] + s3) * (s4 + delp[..., 0]))
    f1 = delp[..., 1] * s34 / (s2 * ss3 * (s4 + delp[..., 0]))
    f2 = (delp[..., 1] + s3) * (
        ss3 * (delp[..., 1] * s3 + s34 + delp[..., 1] * s4)
        + s42 * (delp[..., 1] + s3 + s32 / s2)
    )
    f3 = -delp[..., 1] * (
        ss3 * (s32 * (s3 + s4) / (s4 - delp[..., 1])
               + (delp[..., 1] * s3 + s34 + delp[..., 1] * s4))
        + s42 * (delp[..., 1] + s3)
    )
    f4 = ss3 * delp[..., 1] * s32 * (delp[..., 1] + s3) / (s4 - delp[..., 1])
    qe_k1 = f1 * p[..., 0] + (
        f2 * p[..., 1] + f3 * p[..., 2] + f4 * p[..., 3]
    ) * dm
    qe = qe.at[..., 1].set(qe_k1)

    # Bottom: area-preserving cubic w/ 2nd deriv = 0 at surface.
    # FV3 indices: d1=delp[km], d2=delp[km-1]; reads qe[km-1] (last
    # 4th-order edge), writes qe[km] and qe[km+1].
    # Python:      d1=delp[..., km-1], d2=delp[..., km-2]; reads
    # qe[..., km-2] (last 4th-order), writes qe[..., km-1], qe[..., km].
    d1 = delp[..., -1]
    d2 = delp[..., -2]
    qm = (d2 * p[..., -1] + d1 * p[..., -2]) / (d1 + d2)
    dq = 2.0 * (p[..., -2] - p[..., -1]) / (d1 + d2)
    qe_last_ord4 = qe[..., km - 2]
    c1_bot = (qe_last_ord4 - qm - d2 * dq) / (
        d2 * (2.0 * d2 * d2 + d1 * (d2 + 3.0 * d1))
    )
    c3_bot = dq - 2.0 * c1_bot * (d2 * (5.0 * d1 + d2) - 3.0 * d1 * d1)
    qe_kmm1 = qm - c1_bot * d1 * d2 * (d2 + 3.0 * d1)        # FV3 qe[km]
    qe_km = d1 * (8.0 * c1_bot * d1 * d1 - c3_bot) + qe_kmm1  # FV3 qe[km+1]
    qe = qe.at[..., km - 1].set(qe_kmm1)
    qe = qe.at[..., km].set(qe_km)
    return qe


def cs_interpolator_fv3(
    qin: jax.Array,
    wz: jax.Array,
    zout: float,
    qmin: float = 0.0,
) -> jax.Array:
    """FV3_3D iter 709: height-level interpolation via PPM column profile.

    Faithful JAX port of FV3 ``cs_interpolator``
    (tools/fv_diagnostics.F90:4603-4649).  Uses iter-708
    ``cs_prof_fv3`` for PPM edge values + subcell parabolic
    distribution.

    Algorithm:
        dz[..., k] = wz[..., k] - wz[..., k+1]    (top-down)
        qe = cs_prof_fv3(qin, dz, iv=1)           (km+1 edge values)
        For target zout (scalar):
            above top (zout >= wz[..., 0])      → qe[..., 0]
            below bot (zout <= wz[..., km])     → qe[..., km]
            interior layer k containing zout:
                a6 = 3·(2·qin[k] - (qe[k] + qe[k+1]))
                s0 = (wz[k] - zout) / dz[k]
                qout = qe[k] + s0·(qe[k+1] - qe[k] + a6·(1 - s0))
        Clip qout >= qmin.

    Parameters
    ----------
    qin : jax.Array, shape (..., km)
        Cell-center values.
    wz : jax.Array, shape (..., km+1)
        Layer interface heights (top-down: wz[..., 0] = top,
        wz[..., km] = surface).
    zout : float
        Target height (m).
    qmin : float, default 0.0
        Minimum-allowed output value (FV3 clip floor).

    Returns
    -------
    qout : jax.Array, shape (...,)
        Interpolated value at zout.
    """
    km = qin.shape[-1]
    dz = wz[..., :-1] - wz[..., 1:]           # (..., km), positive
    qe = cs_prof_fv3(qin, dz, iv=1)           # (..., km+1)

    # Locate target layer per column: in_layer[k] = (wz[k] >= zout >= wz[k+1])
    wz_top = wz[..., :-1]                     # (..., km)
    wz_bot = wz[..., 1:]                      # (..., km)
    # Exclusive on lower bound so boundary points pick exactly one layer
    in_layer = (zout <= wz_top) & (zout > wz_bot)    # (..., km), one True per column inside range

    # PPM subcell interp
    a6 = 3.0 * (2.0 * qin - (qe[..., :-1] + qe[..., 1:]))   # (..., km)
    safe_dz = jnp.where(dz > 0.0, dz, 1.0)
    s0 = (wz_top - zout) / safe_dz                          # (..., km)
    qout_k = qe[..., :-1] + s0 * (qe[..., 1:] - qe[..., :-1]
                                  + a6 * (1.0 - s0))         # (..., km)
    # Sum over k with in_layer mask (only one True per column when in range)
    qout_in_range = jnp.sum(jnp.where(in_layer, qout_k, 0.0), axis=-1)

    # Top / bottom clamps
    above_top = zout >= wz[..., 0]
    below_bot = zout <= wz[..., -1]
    qout = jnp.where(above_top, qe[..., 0], qout_in_range)
    qout = jnp.where(below_bot, qe[..., -1], qout)
    return jnp.maximum(qmin, qout)


def cs_prof_fv3(
    q2: jax.Array,
    delp: jax.Array,
    iv: int = 1,
) -> jax.Array:
    """FV3_3D iter 708: PPM column profile edge reconstruction.

    Faithful JAX port of FV3 ``cs_prof``
    (tools/fv_diagnostics.F90:4653-4733).  Non-uniform tridiagonal
    PPM edge values from layer-mean q2 with Lin (2004) monotone
    constraints.

    Algorithm:
        1. Top edge q[0] : explicit closure using delp[1]/delp[0].
        2. Forward sweep k=1..km-1 : Thomas-style with γ coefficient.
        3. Bottom edge q[km] : explicit closure with a_bot.
        4. Back-substitution k=km-1..0 : q[k] -= γ[k]·q[k+1].
        5. Top edge k=1 : large-scale clip between q2[0] and q2[1].
        6. Interior k=2..km-2 : monotone clip based on slope-sign
           pattern of neighbors.
           - same-sign neighbors: clip to [min, max] of q2[k-1], q2[k]
           - local max  : floor at min(q2[k-1], q2[k])
           - local min  : ceiling at max(q2[k-1], q2[k]);
                          if iv==0 (mass species) enforce q[k] ≥ 0.
        7. Bottom k=km-1 : large-scale clip between q2[km-2], q2[km-1].

    Vertical sequence handled via Python loops (JAX traces unroll
    statically for fixed km).

    Parameters
    ----------
    q2 : jax.Array, shape (..., km)
        Layer-mean values.
    delp : jax.Array, shape (..., km)
        Layer pressure thickness (Pa, positive).
    iv : int, default 1
        Variable kind:
            0 = mass species (enforce non-negativity at local mins)
            1 = otherwise

    Returns
    -------
    q : jax.Array, shape (..., km+1)
        PPM edge values at level interfaces.
    """
    km = q2.shape[-1]
    if km < 4:
        raise ValueError(f"cs_prof_fv3 requires km >= 4, got km={km}")

    # Initialize q (km+1 edges) and gam (km layers)
    out_shape = q2.shape[:-1] + (km + 1,)
    q = jnp.zeros(out_shape, dtype=q2.dtype)
    gam = jnp.zeros_like(q2)

    # Top edge (k=0): explicit
    grat = delp[..., 1] / delp[..., 0]
    bet = grat * (grat + 0.5)
    q = q.at[..., 0].set(
        ((grat + grat) * (grat + 1.0) * q2[..., 0] + q2[..., 1]) / bet
    )
    gam = gam.at[..., 0].set((1.0 + grat * (grat + 1.5)) / bet)

    # Forward sweep k=1..km-1 (FV3 k=2..km)
    d4 = None
    for k in range(1, km):
        d4 = delp[..., k - 1] / delp[..., k]
        bet_k = 2.0 + d4 + d4 - gam[..., k - 1]
        q = q.at[..., k].set(
            (3.0 * (q2[..., k - 1] + d4 * q2[..., k]) - q[..., k - 1]) / bet_k
        )
        gam = gam.at[..., k].set(d4 / bet_k)

    # Bottom edge (k=km): explicit closure
    a_bot = 1.0 + d4 * (d4 + 1.5)
    q = q.at[..., km].set(
        (2.0 * d4 * (d4 + 1.0) * q2[..., km - 1] + q2[..., km - 2]
         - a_bot * q[..., km - 1])
        / (d4 * (d4 + 0.5) - a_bot * gam[..., km - 1])
    )

    # Back-substitution k=km-1..0
    for k in range(km - 1, -1, -1):
        q = q.at[..., k].set(q[..., k] - gam[..., k] * q[..., k + 1])

    # Large-scale constraint at top edge (FV3 k=2 → Python k=1)
    q1_min = jnp.minimum(q2[..., 0], q2[..., 1])
    q1_max = jnp.maximum(q2[..., 0], q2[..., 1])
    q = q.at[..., 1].set(jnp.clip(q[..., 1], q1_min, q1_max))

    # Slopes γ[k] = q2[k] - q2[k-1] for k=1..km-1
    gam_slopes = jnp.zeros_like(q2)
    for k in range(1, km):
        gam_slopes = gam_slopes.at[..., k].set(q2[..., k] - q2[..., k - 1])

    # Interior k=2..km-2 (FV3 k=3..km-1)
    for k in range(2, km - 1):
        gleft = gam_slopes[..., k - 1]
        gright = gam_slopes[..., k + 1]
        q_k = q[..., k]
        # Range of q2 in current layer pair
        qk_min = jnp.minimum(q2[..., k - 1], q2[..., k])
        qk_max = jnp.maximum(q2[..., k - 1], q2[..., k])
        # Standard clip (same-sign neighbors)
        q_clip_full = jnp.clip(q_k, qk_min, qk_max)
        # Local max: gleft > 0  (and gright <= 0)
        q_clip_max = jnp.maximum(q_k, qk_min)
        # Local min: gleft <= 0
        q_clip_min = jnp.minimum(q_k, qk_max)
        if iv == 0:
            q_clip_min = jnp.maximum(0.0, q_clip_min)
        # Selectors
        same_sign = gleft * gright > 0.0
        is_local_max = gleft > 0.0
        # Apply
        q_new = jnp.where(
            same_sign,
            q_clip_full,
            jnp.where(is_local_max, q_clip_max, q_clip_min),
        )
        q = q.at[..., k].set(q_new)

    # Bottom layer k=km-1 (FV3 k=km)
    qb_min = jnp.minimum(q2[..., km - 2], q2[..., km - 1])
    qb_max = jnp.maximum(q2[..., km - 2], q2[..., km - 1])
    q = q.at[..., km - 1].set(jnp.clip(q[..., km - 1], qb_min, qb_max))

    return q


def pv_entropy_fv3(
    vort: jax.Array,
    f_d: jax.Array,
    theta: jax.Array,
    delp: jax.Array,
    use_ppme: bool = False,
) -> jax.Array:
    """FV3_3D iter 691/712: Ertel potential vorticity (EPV) diagnostic.

    Faithful JAX port of FV3 ``pv_entropy``
    (tools/fv_diagnostics.F90:5111-5193).

    EPV using the entropy form (S.-J. Lin):

        EPV = - g · (vort + f) / delp · d(theta) / theta

    Edge θ reconstruction:
        ``use_ppme=False`` (default, backward-compat with iter-691):
            second-order linear edge average + endpoint extrapolation.
        ``use_ppme=True`` (iter-712, FV3-faithful):
            iter-711 ``ppme_fv3`` (Van-Leer-limited PPM with non-uniform
            delp, 4th-order interior + parabolic/cubic boundaries).
            This is what FV3's pv_entropy actually calls.

        d_theta[k] = theta_edge[k] - theta_edge[k+1]   (positive in
                       stable column where θ increases upward)

    Parameters
    ----------
    vort : jax.Array, shape (..., km)
        Relative vorticity ζ on cell centers (1/s).
    f_d : jax.Array, shape (...,)
        Coriolis parameter f at cell center (1/s).
    theta : jax.Array, shape (..., km)
        Potential temperature θ = pt / pkz at layer centers (K).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa).
    use_ppme : bool, default False.
        FV3-faithful PPME edge reconstruction (iter-712).  Default
        OFF to preserve iter-691 behavior.

    Returns
    -------
    epv : jax.Array, shape (..., km)
        Ertel potential vorticity (m²·K·s⁻¹·kg⁻¹ = PVU·1e6).
    """
    if use_ppme:
        theta_edges = ppme_fv3(theta, delp)              # (..., km+1)
    else:
        # Default: linear edge average + endpoint extrapolation
        theta_interior = 0.5 * (theta[..., :-1] + theta[..., 1:])
        theta_top = theta[..., :1]
        theta_bot = theta[..., -1:]
        theta_edges = jnp.concatenate(
            [theta_top, theta_interior, theta_bot],
            axis=-1,
        )
    d_theta = theta_edges[..., :-1] - theta_edges[..., 1:]
    abs_vort = vort + f_d[..., None]
    return constants.g * abs_vort * d_theta / (theta * delp)


def bunkers_vector_fv3(
    ua: jax.Array, va: jax.Array,
    delz: jax.Array | None = None,
    pt: jax.Array | None = None, q: jax.Array | None = None,
    peln: jax.Array | None = None,
    hydrostatic: bool = False,
    zvir: float | None = None,
    bunkers_d: float = 7.5,
    right_mover: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 689/719: Bunkers storm motion vector.

    Faithful JAX port of FV3 ``bunkers_vector``
    (tools/fv_diagnostics.F90:4970-5046).

    Empirical supercell storm motion predictor:
        Right-mover (default, FV3 form):
            uc = umn + 7.5 · vshr / shrmag
            vc = vmn − 7.5 · ushr / shrmag
        Left-mover (iter-719, ``right_mover=False``):
            uc = umn − 7.5 · vshr / shrmag
            vc = vmn + 7.5 · ushr / shrmag

    Pairs with iter-687/707 SRH (storm motion input).

    Parameters
    ----------
    ua, va : jax.Array, shape (..., km)
        A-grid wind components.
    delz : jax.Array, shape (..., km), optional
        Layer thickness (NEGATIVE in FV3).
    pt, q, peln, zvir : optional
        Hydrostatic dz reconstruction; required if hydrostatic.
    hydrostatic : bool, default False.
    bunkers_d : float, default 7.5
        Empirical offset magnitude (m/s).
    right_mover : bool, default True.
        iter-719: when False, returns the left-mover storm motion
        (sign flip on shear-perpendicular offset).

    Returns
    -------
    uc, vc : jax.Array, shape (...,)
        Bunkers storm motion components (m/s).
    """
    dz = dz_from_delz_or_hydrostatic_fv3(
        delz=delz, pt=pt, q=q, peln=peln,
        hydrostatic=hydrostatic, zvir=zvir,
    )

    # Layer top/bottom heights above surface (k=0 top, k=-1 surface)
    zh_above, zh_below = compute_zh_above_below_fv3(dz)

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
    sign = 1.0 if right_mover else -1.0
    uc = umn + sign * bunkers_d * vshr / safe_shrmag
    vc = vmn - sign * bunkers_d * ushr / safe_shrmag
    return uc, vc


def helicity_relative_caps_fv3(
    ua: jax.Array, va: jax.Array,
    uc: jax.Array, vc: jax.Array,
    delz: jax.Array | None = None,
    pt: jax.Array | None = None, q: jax.Array | None = None,
    peln: jax.Array | None = None,
    z_bot: float = 0.0,
    z_top: float = 3000.0,
    hydrostatic: bool = False,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 707: storm-relative helicity (SRH) with external (uc, vc).

    Faithful JAX port of FV3 ``helicity_relative_CAPS``
    (tools/fv_diagnostics.F90:4894-4967).

    Variant of iter-687 ``helicity_relative_fv3`` that takes the
    storm motion ``(uc, vc)`` as INPUT rather than computing it as
    the depth-weighted mean wind.  Pairs with iter-689
    ``bunkers_vector_fv3`` which produces the empirical right-mover
    storm motion.

    Algorithm (identical to iter-687 once (uc, vc) given):

        dz_eff[k] = max(0, min(zh_above, z_top) - max(zh_below, z_bot))
        du_dz[k]  = 0.5·(ua[k-1] - ua[k+1])    (interior centered)
        dv_dz[k]  = 0.5·(va[k-1] - va[k+1])
        SRH = Σ_k_in_window (ua-uc)·dv_dz - (va-vc)·du_dz

    Parameters
    ----------
    ua, va : jax.Array, shape (..., km)
        A-grid wind components.
    uc, vc : jax.Array, shape (...,)
        Storm motion components (m/s) — typically from
        ``bunkers_vector_fv3`` (iter-689).
    delz, pt, q, peln, zvir : optional
        Vertical grid info (see iter-687).
    z_bot, z_top : float, default 0, 3000 m.
    hydrostatic : bool, default False.

    Returns
    -------
    srh : jax.Array, shape (...,)
        Storm-relative helicity (m²/s²).
    """
    dz = dz_from_delz_or_hydrostatic_fv3(
        delz=delz, pt=pt, q=q, peln=peln,
        hydrostatic=hydrostatic, zvir=zvir,
    )

    zh_above, zh_below = compute_zh_above_below_fv3(dz)
    dz_eff = jnp.maximum(
        0.0,
        jnp.minimum(zh_above, z_top) - jnp.maximum(zh_below, z_bot),
    )

    du_dz = jnp.zeros_like(ua)
    dv_dz = jnp.zeros_like(va)
    du_dz = du_dz.at[..., 1:-1].set(
        0.5 * (ua[..., :-2] - ua[..., 2:])
    )
    dv_dz = dv_dz.at[..., 1:-1].set(
        0.5 * (va[..., :-2] - va[..., 2:])
    )
    in_window = dz_eff > 0.0
    srh_k = (ua - uc[..., None]) * dv_dz - (va - vc[..., None]) * du_dz
    return jnp.sum(jnp.where(in_window, srh_k, 0.0), axis=-1)


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
    dz = dz_from_delz_or_hydrostatic_fv3(
        delz=delz, pt=pt, q=q, peln=peln,
        hydrostatic=hydrostatic, zvir=zvir,
    )

    # Cumulative zh (top of each layer from surface up)
    zh_above, zh_below = compute_zh_above_below_fv3(dz)
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
    dz = dz_from_delz_or_hydrostatic_fv3(
        delz=delz, pt=pt, q=q, peln=peln,
        hydrostatic=hydrostatic, zvir=zvir,
    )

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


def prt_height_fv3(
    press: float,
    phis: jax.Array,
    delz: jax.Array,
    peln: jax.Array,
    area: jax.Array,
    lat: jax.Array,
) -> dict[str, float]:
    """FV3_3D iter 706: height-of-p-surface diagnostic with lat-band means.

    Faithful JAX port of FV3 ``prt_height``
    (tools/fv_diagnostics.F90:4413-4460).  Composition of iter-684
    ``get_height_given_pressure_fv3`` (mirror-method below-surface
    extrapolation) + iter-705 ``prt_gb_nh_sh_fv3`` (lat-band area-
    weighted means).

    For each cell:
        wz[km]   = phis / g
        wz[k]    = wz[k+1] - delz[k]    (k = km-1..0, building up)
        height(press) = interp at log(press) via mirror method
    Then global / NH / SH / EQ area-weighted means.

    Parameters
    ----------
    press : float
        Target pressure (Pa).
    phis : jax.Array, shape (..., )
        Surface geopotential (m²/s²).
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3).
    peln : jax.Array, shape (..., km+1)
        log(pressure) at interfaces.
    area : jax.Array, shape (...,)
        Cell area (m²).
    lat : jax.Array, shape (...,)
        Cell-center latitude (radians).

    Returns
    -------
    dict[str, float]
        ``gb`` / ``nh`` / ``sh`` / ``eq`` band means of the
        pressure-surface height (m).
    """
    # iter-727: delegate height-from-delz to compute_zh_from_delz_fv3
    wz = compute_zh_from_delz_fv3(phis, delz)
    log_p = jnp.log(press)
    height_cells = get_height_given_pressure_fv3(wz, peln, log_p)
    return prt_gb_nh_sh_fv3(height_cells, area, lat)


def prt_gb_nh_sh_fv3(
    a2: jax.Array,
    area: jax.Array,
    lat: jax.Array,
) -> dict[str, float]:
    """FV3_3D iter 705: lat-band area-weighted mean diagnostic.

    Faithful JAX port of FV3 ``prt_gb_nh_sh``
    (tools/fv_diagnostics.F90:4462-4509).

    Returns area-weighted means over 4 latitude bands matching
    FV3's diagnostic output:

        ``gb``  : global mean (all latitudes)
        ``nh``  : northern hemisphere (20° ≤ lat < 80°)
        ``sh``  : southern hemisphere (−80° < lat ≤ −20°)
        ``eq``  : equatorial (−20° < lat < 20°)

    Used by FV3 as a fast climate-style mean diagnostic at output
    cadence.  Pairs with iter-685 prt_mxm / iter-694 prt_mass /
    iter-704 prt_maxmin.

    Parameters
    ----------
    a2 : jax.Array, shape (...,)
        Field at cell centers (e.g. height of p-surface, T2m, etc.).
    area : jax.Array, shape (...,)
        Cell area (m²).
    lat : jax.Array, shape (...,)
        Cell-center latitude (radians).

    Returns
    -------
    dict[str, float]
        Keys: ``gb``, ``nh``, ``sh``, ``eq``.  Missing bands
        return ``-1.0`` (FV3 bugfix for non-global domains).
    """
    # iter-751: delegate area-weighted means to area_weighted_mean_fv3
    lat_deg = lat * (180.0 / jnp.pi)
    mask_eq = (lat_deg > -20.0) & (lat_deg < 20.0)
    mask_nh = (lat_deg >= 20.0) & (lat_deg < 80.0)
    mask_sh = (lat_deg <= -20.0) & (lat_deg > -80.0)
    return {
        "gb": float(area_weighted_mean_fv3(a2, area)),
        "nh": float(area_weighted_mean_fv3(a2, area, mask=mask_nh)),
        "sh": float(area_weighted_mean_fv3(a2, area, mask=mask_sh)),
        "eq": float(area_weighted_mean_fv3(a2, area, mask=mask_eq)),
    }


def prt_maxmin_fv3(
    q: jax.Array,
    fac: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 704: max/min field diagnostic (no area weighting).

    Faithful JAX port of FV3 ``prt_maxmin``
    (tools/fv_diagnostics.F90:4080-4116).

    Simpler than iter-685 ``prt_mxm_fv3`` which adds area-weighted
    global mean.  This is the lighter-weight diagnostic for sanity
    checks during integration:

        qmin = min(q) · fac
        qmax = max(q) · fac

    Parameters
    ----------
    q : jax.Array
        Field of any shape.
    fac : float, default 1.0
        Multiplicative factor (FV3 convention for unit scaling).

    Returns
    -------
    qmin, qmax : jax.Array
        Scalar min / max of q · fac.
    """
    qmin = jnp.min(q) * fac
    qmax = jnp.max(q) * fac
    return qmin, qmax


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


def compute_zh_from_delz_fv3(
    phis: jax.Array,
    delz: jax.Array,
) -> jax.Array:
    """FV3_3D iter 727: layer interface heights from surface phis + delz.

    Faithful JAX port of FV3's standard z-from-delz pattern (used
    inline in iter-706 ``prt_height_fv3`` and many FV3 diagnostics):

        zh[km] = phis / g                       (surface elevation)
        zh[k]  = zh[k+1] − delz[k]              (cumulative upward)

    With FV3 sign convention (delz < 0, top-down indexing), each
    upward step adds |delz[k]|.  Inverse direction of iter-726
    ``hydrostatic_delz_fv3`` (which goes z → delz).

    Pairs with iter-722 ``compute_pkz_fv3`` (peln→pkz), iter-684
    ``get_height_given_pressure_fv3`` (interpolation), and iter-681
    ``get_height_field_fv3`` (full atmospheric height field).

    Parameters
    ----------
    phis : jax.Array, shape (...,)
        Surface geopotential (m²/s²).
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3 — top-down).

    Returns
    -------
    zh : jax.Array, shape (..., km+1)
        Layer interface heights (m, monotone decreasing in k).
    """
    g = constants.g
    zh_surface = phis / g                          # (...,)
    # Cumulative -delz from surface upward
    minus_delz = -delz                             # positive top-down
    cum_up = jnp.cumsum(minus_delz[..., ::-1], axis=-1)[..., ::-1]
    zh_above = zh_surface[..., None] + cum_up      # (..., km)
    return jnp.concatenate(
        [zh_above, zh_surface[..., None]],
        axis=-1,
    )                                                # (..., km+1)


def hydrostatic_delz_fv3(
    pt: jax.Array,
    pe: jax.Array,
    q: jax.Array | None = None,
    moist: bool = False,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 726: layer thickness from hydrostatic balance.

    Faithful JAX port of FV3's hydrostatic delz initialization pattern
    (model/fv_mapz.F90:3402, 3411 HYDRO_DELZ_REMAP/EXTRAP):

        delz = (R_d / g) · T_v · (pe[k] − pe[k+1])

    where T_v = pt (dry) or T_v = pt·(1 + zvir·q) (moist).

    Sign: FV3 convention has pe[k] < pe[k+1] (top has lower pressure),
    so delz < 0.

    Inverse of iter-722 ``compute_pkz_fv3`` non-hydrostatic branch.
    Used in vertical-remap initialization, IC ingestion, and any
    diagnostic that needs delz from (T, p) under hydrostatic
    assumption.

    Parameters
    ----------
    pt : jax.Array, shape (..., km)
        Air temperature (K).
    pe : jax.Array, shape (..., km+1)
        Interface pressure (Pa, monotone increasing in k).
    q : jax.Array, shape (..., km), optional
        Specific humidity (kg/kg).  Required if moist.
    moist : bool, default False.
        Use T_v = pt·(1 + zvir·q) instead of T.
    zvir : float, optional
        Virtual-T coefficient.  Default ``R_v/R_d − 1``.

    Returns
    -------
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3 convention).
    """
    if moist:
        if q is None:
            raise ValueError("moist=True requires q")
        if zvir is None:
            zvir = constants.R_v / constants.R_d - 1.0
        t_v = pt * (1.0 + zvir * q)
    else:
        t_v = pt
    rdg = constants.R_d / constants.g
    return rdg * t_v * (pe[..., :-1] - pe[..., 1:])


def omega_diagnostic_fv3(
    w: jax.Array,
    delp: jax.Array,
    delz: jax.Array,
) -> jax.Array:
    """FV3_3D iter 725: pressure vertical velocity ω = dp/dt diagnostic.

    Hydrostatic-limit approximation used by FV3 in quasi-static
    atmospheres:

        ρ      = −delp / (g · delz)     (FV3 delz < 0 ⇒ ρ > 0)
        ω      = −ρ · g · w = w · delp / delz

    Sign convention: ω > 0 means descending air (Lagrangian pressure
    increases), ω < 0 means ascending air.

    Exact in the quasi-hydrostatic limit where ∂p/∂t and horizontal
    advection are small.  FV3's true omga (dyn_core.F90:1642) uses
    ``omga = (pe[k+1] − pem[k+1]) · rdt`` which is the full
    Lagrangian derivative including all three terms; this diagnostic
    captures only the dominant w·∂p/∂z piece.

    Parameters
    ----------
    w : jax.Array, shape (..., km)
        Vertical velocity (m/s).
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa, positive).
    delz : jax.Array, shape (..., km)
        Layer thickness (NEGATIVE in FV3).

    Returns
    -------
    omega : jax.Array, shape (..., km)
        Pressure vertical velocity (Pa/s).
    """
    # iter-734: ω = -ρgw, where ρ = -delp/(g·delz).
    # Note original ``w * delp / delz`` is bit-identical to ``-ρ·g·w``.
    return -air_density_fv3(delp, delz) * constants.g * w


def compute_hybrid_pressure_fv3(
    ak: jax.Array,
    bk: jax.Array,
    ps: jax.Array,
    pe_top_floor: float = 1.0e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 724: hybrid σ-pressure setup from ak, bk, ps.

    Faithful JAX port of FV3's hybrid-pressure pattern (used
    throughout tools/test_cases.F90 at lines 2941, 5274, 5370,
    5479; also fv_restart and IC ingestion paths):

        pe[k]   = ak[k] + ps · bk[k]            (interface pressure, k=0..km)
        delp[k] = ak[k+1] − ak[k] + ps · (bk[k+1] − bk[k])
        peln[k] = log(pe[k])

    Top interface (k=0) typically has ak[0]=0, bk[0]=0 → pe[0]=0,
    log(pe[0]) = −∞.  ``pe_top_floor`` (default 1e-6 Pa) clamps to
    avoid log singularity.

    Used by:
      * iter-673 ``remap_coef_fv3`` (DA-increment vertical grid)
      * iter-675 ``bilinear_interp_apply``
      * iter-722 ``compute_pkz_fv3`` (consumes peln/pe)
      * IC ingestion from external pressure-level data

    Parameters
    ----------
    ak : jax.Array, shape (km+1,)
        Hybrid coordinate σ_a offset (Pa).
    bk : jax.Array, shape (km+1,)
        Hybrid coordinate σ_b weight (dimensionless).
    ps : jax.Array, shape (...,)
        Surface pressure (Pa).
    pe_top_floor : float, default 1e-6
        Minimum allowed pe[0] to avoid log(0).

    Returns
    -------
    delp : jax.Array, shape (..., km)
        Layer pressure thickness (Pa).
    pe : jax.Array, shape (..., km+1)
        Interface pressure (Pa).
    peln : jax.Array, shape (..., km+1)
        log(pe).
    """
    # Broadcast ak/bk over (...) shape of ps
    pe = ak + ps[..., None] * bk
    pe = jnp.maximum(pe, pe_top_floor)
    delp = pe[..., 1:] - pe[..., :-1]
    peln = jnp.log(pe)
    return delp, pe, peln


def cappa_moist_fv3(
    q_sphum: jax.Array,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 723: variable Poisson exponent in moist air.

    Faithful JAX port of FV3's moist cappa formula (used by FV3
    Riem_Solver, pv_entropy, eqv_pot, etc.):

        cv_air = c_pd − R_d
        cv_vap = c_pv − R_v
        cappa  = R_d / (R_d + ((1−q)·cv_air + q·cv_vap)/(1+zvir·q))

    In the dry limit (q=0):
        cappa = R_d / (R_d + cv_air) = R_d / c_pd = constants.kappa

    Pairs with iter-722 ``compute_pkz_fv3`` (accepts layer-varying
    cappa array) and iter-716 ``eqv_pot_bolton_fv3`` (uses same
    formula inline for moist θ_e).

    Parameters
    ----------
    q_sphum : jax.Array
        Specific humidity (kg/kg).
    zvir : float, optional
        Virtual-T coefficient.  Default ``R_v/R_d − 1`` from
        legoesm constants.

    Returns
    -------
    cappa : jax.Array
        Layer-varying Poisson exponent (dimensionless).
    """
    if zvir is None:
        zvir = constants.R_v / constants.R_d - 1.0
    cv_air = constants.c_pd - constants.R_d
    cv_vap = constants.c_pv - constants.R_v
    return constants.R_d / (
        constants.R_d
        + ((1.0 - q_sphum) * cv_air + q_sphum * cv_vap) / (1.0 + zvir * q_sphum)
    )


def compute_pkz_fv3(
    delp: jax.Array,
    peln: jax.Array | None = None,
    pt: jax.Array | None = None,
    delz: jax.Array | None = None,
    hydrostatic: bool = True,
    cappa: float | jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 722: layer-mean Exner factor (pkz).

    Faithful JAX port of FV3 pkz computation
    (model/fv_mapz.F90:457 hydrostatic, :481 non-hydrostatic dry,
    :470/:475 non-hydrostatic moist with varying cappa).

    Two branches:

        Hydrostatic (default):
            pkz = (p_top^κ − p_bot^κ) / (κ · (peln_top − peln_bot))
            (line 457; uses peln = ln(p) at interfaces).

        Non-hydrostatic dry (line 481):
            pkz = exp(κ · ln(R_d · delp / (delz · g) · pt))
                = (R_d · delp · pt / (g · delz))^κ

        Non-hydrostatic moist (line 470, varying κ per layer):
            pkz = exp(cappa · ln(R_d · delp / (delz · g) · pt))

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa, positive).
    peln : jax.Array, shape (..., km+1), optional
        log(pressure) at interfaces — required if hydrostatic.
    pt : jax.Array, shape (..., km), optional
        Air temperature (K) — required if not hydrostatic.
    delz : jax.Array, shape (..., km), optional
        Layer thickness (NEGATIVE in FV3 — top-down) — required
        if not hydrostatic.
    hydrostatic : bool, default True.
    cappa : float or jax.Array, optional
        Exponent.  Default ``constants.kappa``.  Pass an array of
        shape ``(..., km)`` for layer-varying moist cappa (FV3
        line 470 path).

    Returns
    -------
    pkz : jax.Array, shape (..., km)
        Layer-mean Exner factor (dimensionless).
    """
    kap = constants.kappa if cappa is None else cappa
    if hydrostatic:
        if peln is None:
            raise ValueError("hydrostatic=True requires peln")
        pe = jnp.exp(peln)
        pk_top = pe[..., :-1] ** kap
        pk_bot = pe[..., 1:] ** kap
        d_peln = peln[..., 1:] - peln[..., :-1]
        return (pk_bot - pk_top) / (kap * d_peln)
    else:
        if pt is None or delz is None:
            raise ValueError("hydrostatic=False requires pt and delz")
        # Non-hydrostatic: pkz = (R_d · delp · pt / (g · |delz|))^κ
        # FV3 uses delz < 0; rrg·delp/delz·pt is negative on a finite
        # column, so use -delz to keep base positive.
        rrg = constants.R_d / constants.g
        base = rrg * delp * pt / (-delz)
        return jnp.exp(kap * jnp.log(base))


def dry_pressure_fv3(
    delp: jax.Array,
    q: jax.Array | None = None,
    pt: jax.Array | None = None,
    delz: jax.Array | None = None,
    peln: jax.Array | None = None,
    hydrostatic: bool = False,
    moist: bool = True,
) -> jax.Array:
    """FV3_3D iter 729: dry partial pressure layer-mean.

    Faithful JAX port of FV3's pd computation pattern (used inline
    in iter-692 ``eqv_pot_fv3`` and iter-716 ``eqv_pot_bolton_fv3``,
    matching FV3 tools/fv_diagnostics.F90:5375-5391):

        rq = max(0, q) if moist else 0
        Hydrostatic:
            pd = (1 − rq) · delp / (peln[k+1] − peln[k])
        Non-hydrostatic:
            pd = −R_d · pt · (1 − rq) · delp / (g · delz)

    Dry partial pressure (excludes water vapor contribution) used
    in the moist Poisson exponent and equivalent potential
    temperature formulas.

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Pressure thickness (Pa, positive).
    q : jax.Array, shape (..., km), optional
        Specific humidity (kg/kg).  Required when moist=True.
    pt : jax.Array, shape (..., km), optional
        Temperature (K).  Required if not hydrostatic.
    delz : jax.Array, shape (..., km), optional
        Layer thickness (NEGATIVE in FV3).  Required if not hydrostatic.
    peln : jax.Array, shape (..., km+1), optional
        log(pressure) at interfaces.  Required if hydrostatic.
    hydrostatic : bool, default False.
    moist : bool, default True.

    Returns
    -------
    pd : jax.Array, shape (..., km)
        Dry partial pressure (Pa).
    """
    if moist:
        if q is None:
            raise ValueError("moist=True requires q")
        rq = jnp.maximum(0.0, q)
    else:
        rq = 0.0
    one_minus_rq = 1.0 - rq
    if hydrostatic:
        if peln is None:
            raise ValueError("hydrostatic=True requires peln")
        # iter-733: delegate hydrostatic p_f to layer_mean_pressure_fv3
        return one_minus_rq * layer_mean_pressure_fv3(delp, peln)
    else:
        if pt is None or delz is None:
            raise ValueError("hydrostatic=False requires pt and delz")
        return -constants.R_d * pt * one_minus_rq * delp / (constants.g * delz)


def virtual_temp_fv3(
    pt: jax.Array,
    q: jax.Array,
    zvir: float | None = None,
) -> jax.Array:
    """FV3_3D iter 721: virtual temperature helper.

    Faithful JAX port of FV3's ``T·(1 + zvir·q)`` pattern used
    throughout dyn_core.F90 and fv_diagnostics.F90 (e.g. lines
    2959, 2990, 3532).

        T_v = T · (1 + zvir · q_sphum)
        zvir = R_v/R_d − 1 (≈ 0.6078 with legoesm constants)

    Used when computing density / pressure under moist air via
    p = ρ·R_d·T_v.

    Parameters
    ----------
    pt : jax.Array
        Air temperature (K).
    q : jax.Array
        Specific humidity (kg/kg).
    zvir : float, optional
        Virtual-T coefficient.  Default ``R_v/R_d − 1`` from
        legoesm constants.

    Returns
    -------
    t_v : jax.Array
        Virtual temperature (K).
    """
    if zvir is None:
        zvir = constants.R_v / constants.R_d - 1.0
    return pt * (1.0 + zvir * q)


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
    # iter-743: delegate to column_integral_delp_fv3 (no /g for raw FV3 sum)
    return column_integral_delp_fv3(q, delp, divide_by_g=False)


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
    # iter-778: delegate Coriolis to coriolis_parameter_fv3
    fc = coriolis_parameter_fv3(jnp.asarray(phip))
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
    # 2·Ω·sin(lat) — uses the function's configurable ``omega`` kwarg
    # (not delegated to ``coriolis_parameter_fv3`` since that fixes
    # Ω = ``constants.Omega``)
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
