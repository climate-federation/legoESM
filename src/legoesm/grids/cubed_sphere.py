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

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import pad_halo, extrapolate_to_halo, compute_padded_angle


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


def create_cubed_sphere(n: int, radius: float = 6.371229e6) -> CubedSphereGrid:
    """Create a cubed-sphere grid.

    Parameters
    ----------
    n : int
        Number of cells per face edge. Common values:
        C48 (~200km), C96 (~100km), C192 (~50km), C384 (~25km).
    radius : float
        Sphere radius in meters. Default: Earth radius.

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

    # Cell areas via spherical excess
    area = _compute_cell_areas(lon, lat, n, radius)

    # Grid spacings (great-circle distance between cell centers)
    dx, dy = _compute_grid_spacing(lon, lat, n, radius)

    # Coriolis parameter
    omega = 7.292e-5
    f = 2.0 * omega * sin_lat

    # Grid angle (rotation of local grid axes relative to east-north)
    angle = _compute_grid_angle(lon, lat, n)

    # Padded grid angle for vector halo exchange
    # Halo cells use extended gnomonic coordinates; interior is overwritten
    # with the actual computed angle to ensure exact roundtrip consistency.
    angle_padded = compute_padded_angle(n)
    angle_padded = angle_padded.at[:, 1:-1, 1:-1].set(angle)

    # Precompute trig of grid angle for vector halo exchange
    cos_angle_val = jnp.cos(angle)
    sin_angle_val = jnp.sin(angle)
    cos_angle_padded_val = jnp.cos(angle_padded)
    sin_angle_padded_val = jnp.sin(angle_padded)

    # Precompute extrapolated half-metrics for divergence operator
    hx_ext = extrapolate_to_halo(dx * 0.5)
    hy_ext = extrapolate_to_halo(dy * 0.5)

    # Cast all arrays to float32. The cubed-sphere PE and tracer transport
    # models run in float32, so grid arrays must match to avoid scatter
    # cast warnings when jax_enable_x64 is True (e.g., spectral tests).
    _f32 = jnp.float32
    return CubedSphereGrid(
        n=n,
        radius=radius,
        lon=lon.astype(_f32),
        lat=lat.astype(_f32),
        area=area.astype(_f32),
        dx=dx.astype(_f32),
        dy=dy.astype(_f32),
        f=f.astype(_f32),
        cos_lat=cos_lat.astype(_f32),
        sin_lat=sin_lat.astype(_f32),
        angle=angle.astype(_f32),
        x_cart=x_cart.astype(_f32),
        y_cart=y_cart.astype(_f32),
        z_cart=z_cart.astype(_f32),
        angle_padded=angle_padded.astype(_f32),
        cos_angle=cos_angle_val.astype(_f32),
        sin_angle=sin_angle_val.astype(_f32),
        cos_angle_padded=cos_angle_padded_val.astype(_f32),
        sin_angle_padded=sin_angle_padded_val.astype(_f32),
        hx_ext=hx_ext.astype(_f32),
        hy_ext=hy_ext.astype(_f32),
    )


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

        face_lon = jnp.arctan2(y, x)
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


def _compute_cell_areas(
    lon: jax.Array, lat: jax.Array, n: int, radius: float
) -> jax.Array:
    """Compute cell areas on the cubed-sphere.

    Uses the fact that for a gnomonic equidistant grid, the area of each
    cell can be computed from the solid angle subtended.

    For uniform gnomonic grid, the area element is:
        dA = R^2 * cos(alpha_y) / (1 + tan^2(alpha_x) + tan^2(alpha_y))^(3/2) * dalpha_x * dalpha_y

    We use a simpler approach: compute areas from the 4 corner coordinates
    via the spherical excess formula.
    """
    # Approximate: use the Jacobian of the gnomonic projection
    dalpha = jnp.pi / (2 * n)  # Grid spacing in gnomonic coordinates

    # For each cell center, compute the area element
    alpha = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n, endpoint=False)
    alpha = alpha + (jnp.pi / 4) / n
    ax, ay = jnp.meshgrid(alpha, alpha, indexing='ij')

    # Area element for gnomonic projection
    # dA = R^2 / (1 + tan^2(ax) + tan^2(ay))^2 * sec^2(ax) * sec^2(ay) * dalpha^2
    # Simplified: dA = R^2 * dalpha^2 / cos^3(ax) / cos^3(ay) / (1/cos^2(ax)/cos^2(ay) * D^3)
    # where D = sqrt(1 + tan^2(ax) + tan^2(ay))

    tan_ax = jnp.tan(ax)
    tan_ay = jnp.tan(ay)
    D = jnp.sqrt(1.0 + tan_ax**2 + tan_ay**2)

    # Exact area element for gnomonic equidistant projection
    area_face = radius**2 * dalpha**2 / (jnp.cos(ax)**2 * jnp.cos(ay)**2 * D**3)

    # Broadcast to all 6 faces (same local coordinates, same areas)
    area = jnp.broadcast_to(area_face[None, :, :], (6, n, n))

    return area


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
    radius: float = 6.371229e6,
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


def rotate_winds_grid_to_geo(
    u_grid: jax.Array, v_grid: jax.Array, angle: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rotate winds from grid-aligned (x, y) to geographic (east, north)."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_east = cos_a * u_grid - sin_a * v_grid
    v_north = sin_a * u_grid + cos_a * v_grid
    return u_east, v_north
