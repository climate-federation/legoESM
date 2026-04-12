"""C-grid finite-volume operators on the latitude-longitude grid.

Compact-stencil gradient, divergence, Coriolis, and advection operators
for the Arakawa C-grid staggering:

  - u lives at lon interfaces:  shape (n_lat, n_lon+1, ...)
  - v lives at lat interfaces:  shape (n_lat+1, n_lon, ...)
  - scalars (eta, T, S) at cell centers: shape (n_lat, n_lon, ...)

Key advantage over A-grid: the pressure gradient and divergence use
adjacent-cell differences (compact stencil), eliminating the 2*dx
checkerboard null space that plagues centered A-grid operators.

Grid conventions (LatLonGrid):
- Latitude (axis 0): South-to-North, bounded (wall BCs at poles)
- Longitude (axis 1): periodic

References
----------
- Arakawa & Lamb (1977): Computational Design of the Basic Dynamical
  Processes of the UCLA General Circulation Model
- Griffies (2004): Fundamentals of Ocean Climate Models (MOM6 C-grid)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid


# =============================================================================
# Gradient operators (scalar at cell center -> vector at faces)
# =============================================================================

def gradient_x_cgrid(
    f: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Zonal gradient df/dx at u-points (lon interfaces).

    Compact stencil: (f[i, j+1] - f[i, j]) / dx_u

    Uses periodic wrapping in longitude.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
        Scalar field at cell centers.
    grid : LatLonGrid

    Returns
    -------
    df_dx : array, shape (n_lat, n_lon+1, ...) at u-points.
    """
    R = grid.radius
    dlon = grid.dlon
    cos_lat = grid.cos_lat  # (n_lat,)

    # Face j sits between cell (j-1) mod n_lon (west) and cell j (east),
    # matching the divergence convention (cell j: west=face j, east=face j+1).
    # Gradient at face j: (f[j] - f[(j-1) mod n_lon]) / dx
    f_west = jnp.roll(f, 1, axis=1)   # f[:, (j-1) % n_lon]
    df = f - f_west  # shape (n_lat, n_lon, ...)

    # Wrap: face at j=n_lon equals face at j=0
    df_wrap = df[..., 0:1] if f.ndim == 2 else df[:, 0:1, :]
    if f.ndim == 2:
        df_full = jnp.concatenate([df, df_wrap], axis=1)
    else:
        df_full = jnp.concatenate([df, df_wrap], axis=1)

    # dx at u-point: R * dlon * cos(lat)  (single cell width)
    dx_u = R * dlon * cos_lat
    if f.ndim == 2:
        return df_full / dx_u[:, jnp.newaxis]
    else:
        return df_full / dx_u[:, jnp.newaxis, jnp.newaxis]


def gradient_y_cgrid(
    f: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Meridional gradient df/dy at v-points (lat interfaces).

    Compact stencil: (f[i+1, j] - f[i, j]) / dy_v

    Wall BC: v=0 at poles, so gradient at pole boundaries is zero.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
        Scalar field at cell centers.
    grid : LatLonGrid

    Returns
    -------
    df_dy : array, shape (n_lat+1, n_lon, ...) at v-points.
    """
    R = grid.radius
    dlat = grid.dlat

    # dy at v-point: R * dlat (single cell height)
    dy_v = R * dlat

    # Interior v-faces: i=1..n_lat-1
    df_interior = (f[1:] - f[:-1]) / dy_v  # (n_lat-1, n_lon, ...)

    # Boundary faces at poles: zero gradient (wall BC)
    if f.ndim == 2:
        zero_row = jnp.zeros((1, f.shape[1]), dtype=f.dtype)
    else:
        zero_row = jnp.zeros((1, f.shape[1], f.shape[2]), dtype=f.dtype)

    df_dy = jnp.concatenate([zero_row, df_interior, zero_row], axis=0)
    return df_dy


# =============================================================================
# Divergence operator (face velocities -> cell center)
# =============================================================================

def divergence_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Conservative FV divergence on the C-grid.

    div = (1/A) * [u_e*dy_e - u_w*dy_w + v_n*dx_n - v_s*dx_s]

    where u_e, u_w are zonal velocity at east/west faces and
    v_n, v_s are meridional velocity at north/south faces.

    Parameters
    ----------
    u : array, shape (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
        Zonal velocity at lon interfaces.
    v : array, shape (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
        Meridional velocity at lat interfaces.
    grid : LatLonGrid
    u_mask, v_mask : array, optional
        Face masks. Applied before flux computation.

    Returns
    -------
    div : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    """
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    cos_lat = grid.cos_lat  # (n_lat,)

    u_eff = u
    v_eff = v
    if u_mask is not None:
        um = u_mask[..., jnp.newaxis] if u.ndim == 3 and u_mask.ndim == 2 else u_mask
        u_eff = u * um
    if v_mask is not None:
        vm = v_mask[..., jnp.newaxis] if v.ndim == 3 and v_mask.ndim == 2 else v_mask
        v_eff = v * vm

    # Zonal face length (meridional extent): R * dlat
    face_dy = R * dlat

    # East face flux - west face flux
    # u[:, j+1] is east face of cell j, u[:, j] is west face
    if u.ndim == 2:
        u_east = u_eff[:, 1:]   # shape (n_lat, n_lon)  -- but n_lon+1-1 = n_lon
        u_west = u_eff[:, :-1]  # shape (n_lat, n_lon)
    else:
        u_east = u_eff[:, 1:, :]
        u_west = u_eff[:, :-1, :]

    # But we have n_lon+1 faces. East face of cell j is face j+1,
    # west face of cell j is face j. So for n_lon cells:
    # cell j has east face at j+1, west face at j.
    # u_east[:, j] = u[:, j+1], u_west[:, j] = u[:, j]
    # With periodic wrap, u[:, n_lon] = u[:, 0] (already included).
    net_zonal = (u_east - u_west) * face_dy  # (n_lat, n_lon, ...)

    # Meridional face length (zonal extent) at lat interface:
    # R * cos(lat_face) * dlon
    lat = grid.lat  # (n_lat,)
    # v-point latitudes: at interfaces between cells
    # South pole face at lat = -pi/2
    # Interior faces at midpoints between cell centers
    # North pole face at lat = +pi/2
    lat_south_pole = jnp.array([-jnp.pi / 2], dtype=lat.dtype)
    lat_north_pole = jnp.array([jnp.pi / 2], dtype=lat.dtype)
    lat_interior = 0.5 * (lat[:-1] + lat[1:])  # (n_lat-1,)
    lat_v = jnp.concatenate([lat_south_pole, lat_interior, lat_north_pole])
    cos_lat_v = jnp.maximum(jnp.cos(lat_v), 1e-10)  # (n_lat+1,)

    face_dx = R * cos_lat_v * dlon  # (n_lat+1,)

    # North face flux - south face flux
    # v[i+1, :] is north face of cell i, v[i, :] is south face
    if v.ndim == 2:
        v_north = v_eff[1:]   # (n_lat, n_lon)
        v_south = v_eff[:-1]  # (n_lat, n_lon)
        fd = face_dx
        net_merid = v_north * fd[1:, jnp.newaxis] - v_south * fd[:-1, jnp.newaxis]
    else:
        v_north = v_eff[1:, :, :]
        v_south = v_eff[:-1, :, :]
        fd = face_dx
        net_merid = (v_north * fd[1:, jnp.newaxis, jnp.newaxis]
                     - v_south * fd[:-1, jnp.newaxis, jnp.newaxis])

    # Cell area
    area = grid.area  # (n_lat, n_lon)
    if u.ndim == 3:
        area = area[..., jnp.newaxis]

    div = (net_zonal + net_merid) / area
    return div


# =============================================================================
# Coriolis terms for C-grid
# =============================================================================

def coriolis_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Coriolis acceleration on the C-grid.

    On a C-grid, computing f*v at a u-point requires averaging v to
    the u-point, and vice versa. We use the Sadourny (1975) energy-
    conserving scheme:

    (f*v)_at_u = f_u * avg(v neighbors)
    (-f*u)_at_v = -f_v * avg(u neighbors)

    Parameters
    ----------
    u : array, shape (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
        Zonal velocity at u-points.
    v : array, shape (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
        Meridional velocity at v-points.
    grid : LatLonGrid
    u_mask, v_mask : array, optional
        Face masks.

    Returns
    -------
    cor_u : array, shape like u
        Coriolis contribution to du/dt.
    cor_v : array, shape like v
        Coriolis contribution to dv/dt.
    """
    f_cell = grid.f  # (n_lat, n_lon)

    is_3d = u.ndim == 3
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # --- f at u-points ---
    # u-point at face j is between cell (j-1) mod n_lon and cell j.
    # f_u = 0.5 * (f[j-1 mod n_lon] + f[j])
    f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)  # (n_lat, n_lon)
    # Append periodic wrap
    f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)  # (n_lat, n_lon+1)

    # --- f at v-points ---
    # v-point (i+1/2, j) is between cell (i, j) and cell (i+1, j).
    # f_v = 0.5 * (f[i,j] + f[i+1,j]) for interior
    # At boundaries (poles), use cell value
    f_v_interior = 0.5 * (f_cell[:-1] + f_cell[1:])  # (n_lat-1, n_lon)
    f_v = jnp.concatenate([f_cell[0:1], f_v_interior, f_cell[-1:]], axis=0)
    # (n_lat+1, n_lon)

    # --- Average v to u-points ---
    # u-point at face j is between cell (j-1) and cell j.
    # The 4 neighboring v-points are:
    # v[i, j-1], v[i+1, j-1], v[i, j], v[i+1, j]
    v_west = jnp.roll(v, 1, axis=1)  # v[:, (j-1) mod n_lon]
    v_avg = 0.25 * (v[:-1] + v[1:] + v_west[:-1] + v_west[1:])
    # v_avg shape: (n_lat, n_lon, ...). Need (n_lat, n_lon+1, ...)
    if is_3d:
        v_avg_wrap = v_avg[:, 0:1, :]
    else:
        v_avg_wrap = v_avg[:, 0:1]
    v_at_u = jnp.concatenate([v_avg, v_avg_wrap], axis=1)

    # --- Average u to v-points ---
    # v-point (i+1/2, j) has 4 neighboring u-points:
    # u[i, j], u[i, j+1], u[i+1, j], u[i+1, j+1]
    # Average: u_at_v = 0.25 * (u[i,j] + u[i,j+1] + u[i+1,j] + u[i+1,j+1])
    u_avg_interior = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    # u_avg_interior shape: (n_lat-1, n_lon, ...)
    # Boundary: at poles (j=0 and j=n_lat), u_at_v = 0 (v=0 at poles anyway)
    if is_3d:
        zero_row = jnp.zeros((1, n_lon, u.shape[2]), dtype=u.dtype)
    else:
        zero_row = jnp.zeros((1, n_lon), dtype=u.dtype)
    u_at_v = jnp.concatenate([zero_row, u_avg_interior, zero_row], axis=0)

    # --- Coriolis terms ---
    if is_3d:
        cor_u = f_u[:, :, jnp.newaxis] * v_at_u
        cor_v = -f_v[:, :, jnp.newaxis] * u_at_v
    else:
        cor_u = f_u * v_at_u
        cor_v = -f_v * u_at_v

    # Apply masks
    if u_mask is not None:
        um = u_mask[..., jnp.newaxis] if is_3d and u_mask.ndim == 2 else u_mask
        cor_u = cor_u * um
    if v_mask is not None:
        vm = v_mask[..., jnp.newaxis] if is_3d and v_mask.ndim == 2 else v_mask
        cor_v = cor_v * vm

    return cor_u, cor_v


# =============================================================================
# =============================================================================
# Laplacian for C-grid scalar fields (at cell centers)
# =============================================================================

def laplacian_cgrid(
    f: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Scalar Laplacian on C-grid cell centers using compact stencils.

    Computes lap(f) = div(grad(f)) using gradient_x/y_cgrid and
    divergence_cgrid. This is a 5-point Laplacian, not the 2*dx
    centered-difference Laplacian of the A-grid.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    grid : LatLonGrid
    mask : array, optional

    Returns
    -------
    lap_f : array, same shape as f
    """
    is_3d = f.ndim == 3

    if is_3d:
        # vmap over levels
        f_t = jnp.moveaxis(f, -1, 0)

        def lap_2d(fi):
            return laplacian_cgrid(fi, grid, mask=mask)

        lap_t = jax.vmap(lap_2d)(f_t)
        return jnp.moveaxis(lap_t, 0, -1)

    # 2D case: compact gradient -> divergence
    grad_x = gradient_x_cgrid(f, grid)  # (n_lat, n_lon+1)
    grad_y = gradient_y_cgrid(f, grid)  # (n_lat+1, n_lon)

    if mask is not None:
        # Zero gradient at land-ocean boundaries
        u_mask = mask * jnp.roll(mask, 1, axis=1)
        u_mask = jnp.concatenate([u_mask, u_mask[:, 0:1]], axis=1)
        v_mask_interior = mask[:-1] * mask[1:]
        zero_row = jnp.zeros((1, mask.shape[1]), dtype=mask.dtype)
        v_mask = jnp.concatenate([zero_row, v_mask_interior, zero_row], axis=0)
        grad_x = grad_x * u_mask
        grad_y = grad_y * v_mask

    lap = divergence_cgrid(grad_x, grad_y, grid)

    if mask is not None:
        lap = lap * mask

    return lap


# =============================================================================
# Vector Laplacian: grad(div) - k x grad(curl)
# =============================================================================

def curl_vertex_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Relative vorticity at vertex (corner) points via circulation integral.

    Vertices sit at the intersection of u-face latitudes and v-face
    longitudes: shape (n_lat+1, n_lon+1), with periodic wrap in
    longitude (column n_lon == column 0).

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    zeta : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev)
    """
    is_3d = u.ndim == 3
    if is_3d:
        import jax
        u_t = jnp.moveaxis(u, -1, 0)   # (nlev, n_lat, n_lon+1)
        v_t = jnp.moveaxis(v, -1, 0)   # (nlev, n_lat+1, n_lon)

        def _curl_2d(u_k, v_k):
            return curl_vertex_cgrid(u_k, v_k, grid)

        zeta_t = jax.vmap(_curl_2d)(u_t, v_t)  # (nlev, n_lat+1, n_lon+1)
        return jnp.moveaxis(zeta_t, 0, -1)

    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    lat = grid.lat  # cell-center latitudes (n_lat,)
    cos_lat = grid.cos_lat  # (n_lat,)
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Vertex area: A_v(i) = R^2 * dlon * |sin(lat_cell[i]) - sin(lat_cell[i-1])|
    # with lat_cell[-1] = -pi/2 (south pole), lat_cell[n_lat] = +pi/2 (north pole)
    sin_lat = jnp.sin(lat)
    sin_ext = jnp.concatenate([
        jnp.array([-1.0], dtype=lat.dtype),   # sin(-pi/2)
        sin_lat,
        jnp.array([1.0], dtype=lat.dtype),     # sin(+pi/2)
    ])
    A_vertex = R**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])  # (n_lat+1,)
    A_vertex = jnp.maximum(A_vertex, 1e-30)  # avoid division by zero at poles

    # Circulation around vertex (i, j), CCW:
    #   south edge (eastward): +u[i-1, j] * R*cos(lat[i-1])*dlon
    #   east edge (northward): +v[i, j] * R*dlat
    #   north edge (westward): -u[i, j] * R*cos(lat[i])*dlon
    #   west edge (southward): -v[i, j-1] * R*dlat
    #
    # For vertex row i: south cell row = i-1, north cell row = i
    # For vertex col j: east v-col = j, west v-col = (j-1) mod n_lon

    # Edge lengths
    dx_cell = R * cos_lat * dlon  # (n_lat,) zonal edge at each cell latitude
    dy_edge = R * dlat             # scalar, meridional edge length

    # v contribution: v[i, j]*dy - v[i, (j-1)%n_lon]*dy
    # v shape: (n_lat+1, n_lon). Wrap in longitude.
    v_east = v                                    # (n_lat+1, n_lon)
    v_west = jnp.roll(v, 1, axis=1)              # (n_lat+1, n_lon)
    dv_circ = (v_east - v_west) * dy_edge         # (n_lat+1, n_lon)

    # u contribution: u[i-1, j]*dx[i-1] - u[i, j]*dx[i]
    # u shape: (n_lat, n_lon+1). Vertex row i uses u rows i-1 and i.
    # Pad with zeros at poles (u at pole vertex has no cell row beyond)
    zero_u = jnp.zeros((1, n_lon + 1), dtype=u.dtype)
    u_ext = jnp.concatenate([zero_u, u, zero_u], axis=0)  # (n_lat+2, n_lon+1)
    dx_ext = jnp.concatenate([
        jnp.zeros(1, dtype=lat.dtype),
        dx_cell,
        jnp.zeros(1, dtype=lat.dtype),
    ])  # (n_lat+2,)

    # u_south = u_ext[i] = u[i-1] for vertex row i (0-indexed)
    # u_north = u_ext[i+1] = u[i] for vertex row i
    u_south = u_ext[:-1, :]  # (n_lat+1, n_lon+1)
    u_north = u_ext[1:, :]   # (n_lat+1, n_lon+1)
    dx_south = dx_ext[:-1]    # (n_lat+1,)
    dx_north = dx_ext[1:]     # (n_lat+1,)

    du_circ = (u_south * dx_south[:, jnp.newaxis]
               - u_north * dx_north[:, jnp.newaxis])  # (n_lat+1, n_lon+1)

    # Combine: need dv_circ at (n_lat+1, n_lon+1) — append periodic wrap column
    dv_circ_full = jnp.concatenate(
        [dv_circ, dv_circ[:, 0:1]], axis=1)  # (n_lat+1, n_lon+1)

    circ = du_circ + dv_circ_full
    zeta = circ / A_vertex[:, jnp.newaxis]

    # Zero at poles (degenerate vertex area, undefined vorticity)
    zeta = zeta.at[0, :].set(0.0)
    zeta = zeta.at[-1, :].set(0.0)

    return zeta


def _gradient_curl_to_u(
    zeta: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Meridional gradient of vertex scalar to u-face points.

    The tangential direction at a u-face is south-to-north.
    result(i, j) = (zeta[i+1, j] - zeta[i, j]) / (R * dlat)

    Parameters
    ----------
    zeta : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev)

    Returns
    -------
    grad : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    """
    R = grid.radius
    dlat = grid.dlat
    dy = R * dlat
    return (zeta[1:] - zeta[:-1]) / dy


def _gradient_curl_to_v(
    zeta: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Zonal gradient of vertex scalar to v-face points.

    The tangential direction at a v-face is west-to-east.
    result(i, j) = (zeta[i, j+1] - zeta[i, j]) / (R * cos(lat_v[i]) * dlon)

    Parameters
    ----------
    zeta : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev)

    Returns
    -------
    grad : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    """
    R = grid.radius
    dlon = grid.dlon
    lat = grid.lat

    # v-face latitudes (same as in divergence_cgrid)
    lat_south_pole = jnp.array([-jnp.pi / 2], dtype=lat.dtype)
    lat_north_pole = jnp.array([jnp.pi / 2], dtype=lat.dtype)
    lat_interior = 0.5 * (lat[:-1] + lat[1:])
    lat_v = jnp.concatenate([lat_south_pole, lat_interior, lat_north_pole])
    cos_lat_v = jnp.maximum(jnp.cos(lat_v), 1e-10)

    dx_v = R * cos_lat_v * dlon  # (n_lat+1,)

    # zeta[:, j+1] - zeta[:, j] for j=0..n_lon-1
    # Column n_lon of zeta is the periodic wrap of column 0
    dzeta = zeta[:, 1:] - zeta[:, :-1]  # (n_lat+1, n_lon)

    if zeta.ndim == 2:
        return dzeta / dx_v[:, jnp.newaxis]
    else:
        return dzeta / dx_v[:, jnp.newaxis, jnp.newaxis]


def vector_laplacian_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vector Laplacian on the C-grid: grad(div) - k x grad(curl).

    Operates directly on face velocities without cell-center detour.
    This is the rectangular-grid analog of TRiSK's vector_laplacian_del2.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid
    mask : (n_lat, n_lon) cell-center land mask, optional
    u_mask : (n_lat, n_lon+1) u-face mask, optional
    v_mask : (n_lat+1, n_lon) v-face mask, optional

    Returns
    -------
    vlap_u : same shape as u
    vlap_v : same shape as v
    """
    is_3d = u.ndim == 3

    if is_3d:
        import jax
        u_t = jnp.moveaxis(u, -1, 0)   # (nlev, n_lat, n_lon+1)
        v_t = jnp.moveaxis(v, -1, 0)   # (nlev, n_lat+1, n_lon)

        def _vlap_2d(u_k, v_k):
            return vector_laplacian_cgrid(
                u_k, v_k, grid,
                mask=mask, u_mask=u_mask, v_mask=v_mask)

        lu_t, lv_t = jax.vmap(_vlap_2d)(u_t, v_t)
        return jnp.moveaxis(lu_t, 0, -1), jnp.moveaxis(lv_t, 0, -1)

    # --- 2D case ---

    # Apply face masks before computing div and curl
    u_eff = u
    v_eff = v
    if u_mask is not None:
        u_eff = u * u_mask
    if v_mask is not None:
        v_eff = v * v_mask

    # 1. Divergence at cell centers
    div = divergence_cgrid(u_eff, v_eff, grid)
    if mask is not None:
        div = div * mask

    # 2. grad(div) at faces
    grad_div_u = gradient_x_cgrid(div, grid)  # (n_lat, n_lon+1)
    grad_div_v = gradient_y_cgrid(div, grid)  # (n_lat+1, n_lon)

    # 3. Curl at vertices
    zeta = curl_vertex_cgrid(u_eff, v_eff, grid)  # (n_lat+1, n_lon+1)

    # Mask curl at land-adjacent vertices
    if mask is not None:
        vmask = _compute_vertex_mask(mask)
        zeta = zeta * vmask

    # 4. Tangential gradient of curl at faces
    grad_curl_u = _gradient_curl_to_u(zeta, grid)  # (n_lat, n_lon+1)
    grad_curl_v = _gradient_curl_to_v(zeta, grid)  # (n_lat+1, n_lon)

    # 5. Vector Laplacian = grad(div) - curl(curl)
    # curl(curl F) = k × ∇ζ = (-∂ζ/∂y, +∂ζ/∂x).
    # Signs verified via bump tests (both components must be diffusive):
    #   u-bump: vlap_u = grad_div_u - grad_curl_u → negative ✓
    #   v-bump: vlap_v = grad_div_v + grad_curl_v → negative ✓
    # The sign asymmetry arises because _gradient_curl_to_u computes
    # ∂ζ/∂y (northward) while _gradient_curl_to_v computes ∂ζ/∂x
    # (eastward), and (k × ∇ζ)_x = -∂ζ/∂y but (k × ∇ζ)_y = +∂ζ/∂x.
    vlap_u = grad_div_u - grad_curl_u
    vlap_v = grad_div_v + grad_curl_v

    # Apply face masks to output
    if u_mask is not None:
        vlap_u = vlap_u * u_mask
    if v_mask is not None:
        vlap_v = vlap_v * v_mask

    return vlap_u, vlap_v


def _compute_vertex_mask(land_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute vertex mask: wet only if all four surrounding cells are wet.

    Parameters
    ----------
    land_mask : (n_lat, n_lon)

    Returns
    -------
    vertex_mask : (n_lat+1, n_lon+1)
    """
    m = land_mask
    n_lat, n_lon = m.shape

    # Interior vertices (i, j) for i=1..n_lat-1, j=0..n_lon-1
    # surrounded by cells (i-1, j-1), (i-1, j), (i, j-1), (i, j)
    m_sw = jnp.roll(m, 1, axis=1)  # m[:, j-1]
    interior = m[:-1] * m[1:] * m_sw[:-1] * m_sw[1:]  # (n_lat-1, n_lon)

    # Append periodic wrap column
    interior_full = jnp.concatenate(
        [interior, interior[:, 0:1]], axis=1)  # (n_lat-1, n_lon+1)

    # Pole rows: zero (degenerate vertices)
    zero_row = jnp.zeros((1, n_lon + 1), dtype=m.dtype)

    return jnp.concatenate([zero_row, interior_full, zero_row], axis=0)


# =============================================================================
# Utility: compute face masks from cell mask
# =============================================================================

def compute_face_masks(
    land_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Derive u-face and v-face masks from cell-center land mask.

    A face is wet only if both adjacent cells are wet.

    Parameters
    ----------
    land_mask : array, shape (n_lat, n_lon)
        Cell-center ocean mask (1 = ocean, 0 = land).

    Returns
    -------
    u_mask : array, shape (n_lat, n_lon+1)
        Mask at zonal (lon) interfaces.
    v_mask : array, shape (n_lat+1, n_lon)
        Mask at meridional (lat) interfaces.
    """
    # u-face j is between cell (j-1) mod n_lon and cell j (periodic in lon)
    u_mask_interior = land_mask * jnp.roll(land_mask, 1, axis=1)
    # Append periodic wrap
    u_mask = jnp.concatenate(
        [u_mask_interior, u_mask_interior[:, 0:1]], axis=1,
    )

    # v-face i is between cell i and cell i+1
    v_mask_interior = land_mask[:-1] * land_mask[1:]
    # Pole boundaries: v=0 (always masked)
    zero_row = jnp.zeros((1, land_mask.shape[1]), dtype=land_mask.dtype)
    v_mask = jnp.concatenate([zero_row, v_mask_interior, zero_row], axis=0)

    return u_mask, v_mask
