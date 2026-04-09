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
# Vector Laplacian on C-grid faces
# =============================================================================

def vorticity_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Relative vorticity at cell corners (vertex points).

    Computes ζ = ∂v/∂x - ∂u/∂y on the C-grid using circulation
    around each vertex.  Returns shape (n_lat+1, n_lon+1).

    Corner (i, j) is the SW corner of cell (i, j), surrounded by:
    - u-face (i-1, j) to the south
    - u-face (i, j) to the north
    - v-face (i, j-1) to the west
    - v-face (i, j) to the east
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    n_lat, n_lon = grid.n_lat, grid.n_lon

    # v-face contribution: ∂v/∂x at corners
    # v has shape (n_lat+1, n_lon). Corner (i,j) uses v(i, j) and v(i, j-1).
    v_east = v                                    # (n_lat+1, n_lon) at j
    v_west = jnp.roll(v, 1, axis=1)              # (n_lat+1, n_lon) at j-1
    # Pad to (n_lat+1, n_lon+1) — periodic in longitude
    dv_dx = jnp.concatenate([v_east - v_west, (v_east - v_west)[:, 0:1]], axis=1)

    # u-face contribution: -∂(u*cos_lat)/∂y at corners
    # u has shape (n_lat, n_lon+1). Corner (i,j) uses u(i, j) and u(i-1, j).
    # Latitude of u-face row i = lat[i] (cell-center latitude)
    cos_lat_u = jnp.cos(grid.lat)[:, None]  # (n_lat, 1)
    u_cos = u * cos_lat_u                   # (n_lat, n_lon+1)
    u_north = u_cos                          # at i
    u_south = jnp.zeros((1, n_lon + 1), dtype=u.dtype)
    # Pad: row 0 of corners uses u(0, j) (north) and nothing to south (wall)
    u_north_full = jnp.concatenate([u_cos, jnp.zeros((1, n_lon + 1), dtype=u.dtype)], axis=0)
    u_south_full = jnp.concatenate([jnp.zeros((1, n_lon + 1), dtype=u.dtype), u_cos], axis=0)
    du_cos_dy = u_north_full - u_south_full   # (n_lat+1, n_lon+1)

    # Corner latitude (needed for 1/cos_lat_corner metric)
    lat_corner = jnp.concatenate([
        jnp.array([-jnp.pi / 2]),
        0.5 * (grid.lat[:-1] + grid.lat[1:]),
        jnp.array([jnp.pi / 2]),
    ])  # (n_lat+1,)
    cos_lat_corner = jnp.maximum(jnp.cos(lat_corner), 1e-10)[:, None]

    # ζ = (1 / R cos_lat_corner) * (dv_dx / dlon - du_cos_dy / dlat)
    zeta = (dv_dx / dlon - du_cos_dy / dlat) / (R * cos_lat_corner)
    return zeta


def vector_laplacian_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vector Laplacian: grad(div) - curl(curl) at C-grid face points.

    Computes the proper vector Laplacian directly on face velocities
    without the cell-center interpolation detour.  This avoids the
    extra smoothing from double-interpolation that over-dissipates KE.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid
    mask, u_mask, v_mask : optional land masks

    Returns
    -------
    lap_u : same shape as u
    lap_v : same shape as v
    """
    import jax

    is_3d = u.ndim == 3

    def _vlap_2d(u2d, v2d):
        # --- Term 1: grad(div) ---
        div_uv = divergence_cgrid(u2d, v2d, grid, u_mask=u_mask, v_mask=v_mask)
        if mask is not None:
            div_uv = div_uv * mask
        graddiv_u = gradient_x_cgrid(div_uv, grid)  # (n_lat, n_lon+1)
        graddiv_v = gradient_y_cgrid(div_uv, grid)  # (n_lat+1, n_lon)

        # --- Term 2: curl(curl) = -k × grad(ζ) ---
        # ζ at corners (n_lat+1, n_lon+1)
        zeta = vorticity_cgrid(u2d, v2d, grid)
        if mask is not None:
            # Mask vorticity at corners: corner (i,j) is wet only if
            # all adjacent cells are wet.  Interior corners (1..n_lat-1,
            # 0..n_lon) use 4 cells; pole corners are always masked.
            n_lat, n_lon = mask.shape
            # Interior corner mask (n_lat-1, n_lon)
            m_s = mask[:-1, :]      # cell to south
            m_n = mask[1:, :]       # cell to north
            m_sw = m_s              # same row, same col
            m_nw = m_n
            m_se = jnp.roll(m_s, -1, axis=1)
            m_ne = jnp.roll(m_n, -1, axis=1)
            cm_interior = m_sw * m_nw * m_se * m_ne  # (n_lat-1, n_lon)
            # Pad: poles masked, periodic in lon
            zero_row = jnp.zeros((1, n_lon), dtype=mask.dtype)
            cm = jnp.concatenate([zero_row, cm_interior, zero_row], axis=0)
            cm = jnp.concatenate([cm, cm[:, 0:1]], axis=1)  # (n_lat+1, n_lon+1)
            zeta = zeta * cm

        # grad(ζ): ζ at corners → gradient at faces
        # ∂ζ/∂y at u-points (n_lat, n_lon+1): u-face (i,j) between
        # corners (i,j) [south] and (i+1,j) [north]
        R = grid.radius
        dzetady_at_u = (zeta[1:, :] - zeta[:-1, :]) / (R * grid.dlat)

        # ∂ζ/∂x at v-points (n_lat+1, n_lon): v-face (i,j) between
        # corners (i,j) [west] and (i,j+1) [east]
        cos_lat_v = jnp.concatenate([
            jnp.array([1e-10]),
            jnp.maximum(jnp.cos(0.5 * (grid.lat[:-1] + grid.lat[1:])), 1e-10),
            jnp.array([1e-10]),
        ])[:, None]
        dzetadx_at_v = (zeta[:, 1:] - zeta[:, :-1]) / (R * cos_lat_v * grid.dlon)

        # curl(curl) = (∂ζ/∂y, -∂ζ/∂x)  (k × grad ζ)
        # vector Laplacian = grad(div) - curl(curl)
        lap_u = graddiv_u - dzetady_at_u
        lap_v = graddiv_v + dzetadx_at_v
        return lap_u, lap_v

    if not is_3d:
        return _vlap_2d(u, v)

    # 3D: vmap over levels (u and v have different spatial shapes,
    # so we vmap them as separate arguments, moving level to axis 0).
    u_t = jnp.moveaxis(u, -1, 0)   # (nlev, n_lat, n_lon+1)
    v_t = jnp.moveaxis(v, -1, 0)   # (nlev, n_lat+1, n_lon)
    lu_t, lv_t = jax.vmap(_vlap_2d)(u_t, v_t)
    return jnp.moveaxis(lu_t, 0, -1), jnp.moveaxis(lv_t, 0, -1)


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
