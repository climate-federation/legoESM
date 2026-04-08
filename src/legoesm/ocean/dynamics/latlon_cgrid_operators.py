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

    # f_east wraps periodically: f_east[:, j] = f[:, (j+1) % n_lon]
    f_east = jnp.roll(f, -1, axis=1)

    # Interior faces: j=0..n_lon-1 have f[:, j+1] - f[:, j]
    # but we need n_lon+1 faces; face j=n_lon is the same as face j=0
    # (periodic). We compute all n_lon interior differences then append
    # the wrap-around face.
    df = f_east - f  # shape (n_lat, n_lon, ...)

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
    # u-point (i, j+1/2) is between cell (i, j) and cell (i, j+1 mod n_lon).
    # f_u = 0.5 * (f[i,j] + f[i, j+1 mod n_lon])
    f_u = 0.5 * (f_cell + jnp.roll(f_cell, -1, axis=1))  # (n_lat, n_lon)
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
    # u-point (i, j+1/2) has 4 neighboring v-points:
    # v[i, j], v[i+1, j], v[i, j+1], v[i+1, j+1]
    # (i.e., the 4 corners of the cell face)
    # Average: v_at_u = 0.25 * (v[i,j] + v[i+1,j] + v[i,j+1] + v[i+1,j+1])
    # with j+1 periodic in longitude
    v_east = jnp.roll(v, -1, axis=1)  # v[:, j+1 mod n_lon]
    v_avg = 0.25 * (v[:-1] + v[1:] + v_east[:-1] + v_east[1:])
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
# Scalar advection (face-normal velocities)
# =============================================================================

def scalar_advection_cgrid(
    q: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Upwind scalar advection on the C-grid: -div(q * v).

    Uses first-order upwind reconstruction at cell faces.

    Parameters
    ----------
    q : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
        Scalar at cell centers.
    u : array, shape (n_lat, n_lon+1, ...) at u-points.
    v : array, shape (n_lat+1, n_lon, ...) at v-points.
    grid : LatLonGrid
    mask, u_mask, v_mask : arrays, optional

    Returns
    -------
    dq_dt : array, shape (n_lat, n_lon, ...)
    """
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat

    is_3d = q.ndim == 3
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Apply face masks to velocities
    u_eff = u
    v_eff = v
    if u_mask is not None:
        um = u_mask[..., jnp.newaxis] if is_3d and u_mask.ndim == 2 else u_mask
        u_eff = u * um
    if v_mask is not None:
        vm = v_mask[..., jnp.newaxis] if is_3d and v_mask.ndim == 2 else v_mask
        v_eff = v * vm

    # --- Zonal advection ---
    # Face j: between cell j-1 and cell j
    # q_face = upwind(u_face, q_left=q[j-1], q_right=q[j])
    # q_left at face j: q[:, j-1]  (periodic)
    # q_right at face j: q[:, j]
    q_left = jnp.roll(q, 1, axis=1)  # q[:, j-1]

    # For faces 0..n_lon-1 and the periodic face at n_lon:
    # face j has left=q[:,j-1], right=q[:,j]
    # face n_lon is same as face 0 (periodic wrap)
    if is_3d:
        q_left_full = jnp.concatenate([q_left, q_left[:, 0:1, :]], axis=1)
        q_right_full = jnp.concatenate([q, q[:, 0:1, :]], axis=1)
    else:
        q_left_full = jnp.concatenate([q_left, q_left[:, 0:1]], axis=1)
        q_right_full = jnp.concatenate([q, q[:, 0:1]], axis=1)

    # Upwind: use q_left if u > 0 (flow from left to right)
    q_u_face = jnp.where(u_eff > 0, q_left_full, q_right_full)

    # Face flux: u * q * face_length
    face_dy = R * dlat
    zonal_flux = u_eff * q_u_face * face_dy  # (n_lat, n_lon+1, ...)

    # Net zonal flux per cell: flux_east - flux_west
    if is_3d:
        net_zonal = zonal_flux[:, 1:, :] - zonal_flux[:, :-1, :]
    else:
        net_zonal = zonal_flux[:, 1:] - zonal_flux[:, :-1]

    # --- Meridional advection ---
    lat = grid.lat
    lat_south_pole = jnp.array([-jnp.pi / 2], dtype=lat.dtype)
    lat_north_pole = jnp.array([jnp.pi / 2], dtype=lat.dtype)
    lat_interior = 0.5 * (lat[:-1] + lat[1:])
    lat_v = jnp.concatenate([lat_south_pole, lat_interior, lat_north_pole])
    cos_lat_v = jnp.maximum(jnp.cos(lat_v), 1e-10)
    face_dx = R * cos_lat_v * dlon  # (n_lat+1,)

    # q at v-faces: upwind
    # face i: between cell i-1 (south) and cell i (north)
    # Interior faces i=1..n_lat-1
    q_south = q[:-1]  # cell i-1
    q_north_cells = q[1:]   # cell i
    v_interior = v_eff[1:-1]  # interior v-faces (exclude poles)

    q_v_interior = jnp.where(v_interior > 0, q_south, q_north_cells)

    # Pole faces: zero flux (v=0 at poles)
    if is_3d:
        zero_row = jnp.zeros((1, n_lon, q.shape[2]), dtype=q.dtype)
    else:
        zero_row = jnp.zeros((1, n_lon), dtype=q.dtype)

    q_v_face = jnp.concatenate([zero_row, q_v_interior, zero_row], axis=0)

    if is_3d:
        merid_flux = v_eff * q_v_face * face_dx[:, jnp.newaxis, jnp.newaxis]
        net_merid = merid_flux[1:, :, :] - merid_flux[:-1, :, :]
    else:
        merid_flux = v_eff * q_v_face * face_dx[:, jnp.newaxis]
        net_merid = merid_flux[1:] - merid_flux[:-1]

    # Divergence
    area = grid.area
    if is_3d:
        area = area[..., jnp.newaxis]

    dq_dt = -(net_zonal + net_merid) / area

    if mask is not None:
        m = mask[..., jnp.newaxis] if is_3d and mask.ndim == 2 else mask
        dq_dt = dq_dt * m

    return dq_dt


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
        import jax
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
        u_mask = mask * jnp.roll(mask, -1, axis=1)
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
    # u-face j is between cell j and cell j+1 (periodic in lon)
    u_mask_interior = land_mask * jnp.roll(land_mask, -1, axis=1)
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
