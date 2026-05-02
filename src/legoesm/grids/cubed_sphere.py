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
