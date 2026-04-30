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
# Cell-center ↔ face interpolation (shared by atmosphere and ocean)
# =============================================================================

def interp_cell_to_uface(f: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a cell-center field to u-face (lon interface) positions.

    Simple average of the two cells sharing each lon face.
    Periodic in longitude: face n_lon wraps to face 0.

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_u : (n_lat, n_lon+1, ...) at u-faces.
    """
    f_u = 0.5 * (jnp.roll(f, 1, axis=1) + f)
    return jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)


def interp_cell_to_vface(f: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a cell-center field to v-face (lat interface) positions.

    Interior faces: average of adjacent cells.
    Pole faces (south=0, north=n_lat): copy the adjacent cell value.
    The pole value is numerically inert since v = 0 at the wall.

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_v : (n_lat+1, n_lon, ...) at v-faces.
    """
    f_v_interior = 0.5 * (f[:-1] + f[1:])  # (n_lat-1, ...)
    return jnp.concatenate([f[0:1], f_v_interior, f[-1:]], axis=0)


def min_cell_to_uface(f: jnp.ndarray) -> jnp.ndarray:
    """Min-rule interpolation of a cell-center thickness to u-faces.

    Use this (NOT ``interp_cell_to_uface``) for layer thickness ``h_k``
    when the model has partial bottom cells.  At a face between cell
    W (partial cell at level k, h_W) and cell E (full cell, h_E),
    the face's effective wet thickness equals the shallower side's
    thickness — the deeper side has rock below the shallower seafloor
    at that level, so the face is closed there and the wet area is
    bounded by ``min(h_W, h_E)``.

    Equivalent to MOM6/MITgcm's ``hFacW = min(hFacC_L, hFacC_R)``
    convention (Adcroft, Hill & Marshall 1997 eq. 11-13).

    For full-cell columns (pure z\\* with same bathymetry on both
    sides), ``min(h_W, h_E) == h_W == h_E`` so this is bit-exact
    backwards-compat.  Differs from arithmetic mean when adjacent
    cells have different layer thicknesses (partial-cell faces, or
    z\\* with horizontally varying eta).

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_u : (n_lat, n_lon+1, ...) at u-faces.
    """
    f_u = jnp.minimum(jnp.roll(f, 1, axis=1), f)
    return jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)


def min_cell_to_vface(f: jnp.ndarray) -> jnp.ndarray:
    """Min-rule interpolation of a cell-center thickness to v-faces.

    Same convention as ``min_cell_to_uface`` for the meridional
    direction.  Pole rows (south=0, north=n_lat) are zero-padded:
    the pole is a wall, no fluid passes through, so the face wet
    thickness there is exactly zero.  This matches the ocean-PE
    convention used everywhere else for thickness at v-faces.

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_v : (n_lat+1, n_lon, ...) at v-faces.
    """
    f_v_interior = jnp.minimum(f[:-1], f[1:])
    pad_axes = ((0, 0),) * (f_v_interior.ndim - 1)
    return jnp.pad(f_v_interior, ((1, 1), *pad_axes))


def cell_to_cgrid_winds(
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert cell-centered winds to C-grid face-staggered winds.

    Works for both 2D (n_lat, n_lon) and 3D (n_lat, n_lon, nlev).
    v = 0 at pole walls (wall boundary condition).

    Parameters
    ----------
    u_cell, v_cell : cell-centered wind components.

    Returns
    -------
    u_face : (..., n_lon+1, ...) at lon interfaces.
    v_face : (n_lat+1, ...) at lat interfaces, zero at poles.
    """
    u_face = interp_cell_to_uface(u_cell)
    # v at poles is zero (wall BC), not the average of adjacent cells.
    v_interior = 0.5 * (v_cell[:-1] + v_cell[1:])
    # ``jnp.pad`` with zero fill is one Pad HLO op; the previous form
    # alloc-zeros + concatenate-of-three is two HLO ops.  Wall-BC at
    # poles is preserved (pole rows are zero).
    pad_axes = ((0, 0),) * (v_interior.ndim - 1)
    v_face = jnp.pad(v_interior, ((1, 1), *pad_axes))
    return u_face, v_face


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

    # Boundary faces at poles: zero gradient (wall BC).
    # Single Pad HLO op replaces alloc-zeros + concatenate-of-three.
    pad_axes = ((0, 0),) * (df_interior.ndim - 1)
    df_dy = jnp.pad(df_interior, ((1, 1), *pad_axes))
    return df_dy


# =============================================================================
# Divergence operator (face velocities -> cell center)
# =============================================================================

def bilaplacian_cgrid(
    f: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Scalar bilaplacian (∇⁴f) on C-grid cell centers.

    Applies ``laplacian_cgrid`` twice: ∇⁴f = ∇²(∇²f).

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    grid : LatLonGrid
    mask : array, optional

    Returns
    -------
    bilap_f : array, same shape as f
    """
    lap_f = laplacian_cgrid(f, grid, mask=mask)
    return laplacian_cgrid(lap_f, grid, mask=mask)


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
    # v-point latitudes: at interfaces between cells.  cos(±π/2) is
    # exactly 0 analytically (and roundoff-level in finite precision),
    # so build cos_lat_v directly via Pad — pole rows are exactly 0
    # regardless of dtype, and this avoids the alloc-2-singleton +
    # concatenate-of-three + cos tower (4 HLO ops → 2 HLO ops).
    lat_interior = 0.5 * (lat[:-1] + lat[1:])  # (n_lat-1,)
    cos_lat_v_interior = jnp.cos(lat_interior)  # (n_lat-1,)
    cos_lat_v = jnp.pad(cos_lat_v_interior, (1, 1))  # (n_lat+1,)

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
    # v_avg shape: (n_lat, n_lon, ...). Append wrap column to (n_lat, n_lon+1, ...);
    # ``v_avg[:, 0:1]`` works for both 2D and 3D (slice along axis 1 only).
    v_at_u = jnp.concatenate([v_avg, v_avg[:, 0:1]], axis=1)

    # --- Average u to v-points ---
    # v-point (i+1/2, j) has 4 neighboring u-points:
    # u[i, j], u[i, j+1], u[i+1, j], u[i+1, j+1]
    # Average: u_at_v = 0.25 * (u[i,j] + u[i,j+1] + u[i+1,j] + u[i+1,j+1])
    u_avg_interior = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    # u_avg_interior shape: (n_lat-1, n_lon, ...)
    # Boundary: at poles, u_at_v = 0 (v=0 at poles anyway).
    # Single Pad HLO op replaces alloc-zeros + concatenate-of-three.
    pad_axes = ((0, 0),) * (u_avg_interior.ndim - 1)
    u_at_v = jnp.pad(u_avg_interior, ((1, 1), *pad_axes))

    # --- Coriolis terms ---
    # Reshape 2D ``f_u``/``f_v`` to broadcast over the trailing level
    # axis when 3D (no-op when 2D).
    f_u_b = f_u[..., jnp.newaxis] if is_3d else f_u
    f_v_b = f_v[..., jnp.newaxis] if is_3d else f_v
    cor_u = f_u_b * v_at_u
    cor_v = -f_v_b * u_at_v

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

    # ``gradient_x_cgrid`` / ``gradient_y_cgrid`` / ``divergence_cgrid``
    # all natively support 3D inputs (they internally branch on ndim
    # for the dx_u / dy_v / cos_lat broadcasting).  Call them directly
    # on 3D — the previous moveaxis + vmap + moveaxis round-trip was
    # redundant.
    grad_x = gradient_x_cgrid(f, grid)  # (n_lat, n_lon+1[, nlev])
    grad_y = gradient_y_cgrid(f, grid)  # (n_lat+1, n_lon[, nlev])

    if mask is not None:
        # Zero gradient at land-ocean boundaries
        u_mask = mask * jnp.roll(mask, 1, axis=1)
        u_mask = jnp.concatenate([u_mask, u_mask[:, 0:1]], axis=1)
        # Pole rows of v_mask are zero (wall BC); single Pad HLO op
        # replaces alloc-zeros + concatenate-of-three.
        v_mask_interior = mask[:-1] * mask[1:]
        v_mask = jnp.pad(v_mask_interior, ((1, 1), (0, 0)))
        if is_3d:
            grad_x = grad_x * u_mask[..., jnp.newaxis]
            grad_y = grad_y * v_mask[..., jnp.newaxis]
        else:
            grad_x = grad_x * u_mask
            grad_y = grad_y * v_mask

    lap = divergence_cgrid(grad_x, grad_y, grid)

    if mask is not None:
        if is_3d:
            lap = lap * mask[..., jnp.newaxis]
        else:
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
    # Native 2D and 3D — broadcast 1D metric over the trailing level
    # axis when needed.  Eliminates the prior moveaxis + vmap +
    # moveaxis round-trip.
    is_3d = u.ndim == 3
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    lat = grid.lat  # cell-center latitudes (n_lat,)
    cos_lat = grid.cos_lat  # (n_lat,)

    # Vertex area: A_v(i) = R^2 * dlon * |sin(lat_cell[i]) - sin(lat_cell[i-1])|
    # with lat_cell[-1] = -pi/2 (south pole), lat_cell[n_lat] = +pi/2 (north pole).
    sin_lat = jnp.sin(lat)
    sin_ext = jnp.pad(sin_lat, (1, 1), constant_values=(-1.0, 1.0))
    A_vertex_all = R**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])  # (n_lat+1,)
    A_vertex_interior = A_vertex_all[1:-1]  # (n_lat-1,)

    # Edge lengths
    dx_cell = R * cos_lat * dlon  # (n_lat,) zonal edge at each cell latitude
    dy_edge = R * dlat             # scalar, meridional edge length

    # v contribution: v[i, j]*dy - v[i, (j-1)%n_lon]*dy.  Works for 2D
    # and 3D directly.
    v_east = v
    v_west = jnp.roll(v, 1, axis=1)
    dv_circ = (v_east - v_west) * dy_edge

    # u contribution: u[i-1, j]*dx[i-1] - u[i, j]*dx[i].
    # Pad with zeros at poles along the lat axis (axis 0).  Extra
    # ``(0, 0)`` pad-tuples for any trailing dims (level axis in 3D).
    pad_extra = ((0, 0),) * (u.ndim - 2)
    u_ext = jnp.pad(u, ((1, 1), (0, 0), *pad_extra))
    dx_ext = jnp.pad(dx_cell, (1, 1))

    u_south = u_ext[:-1]
    u_north = u_ext[1:]
    dx_south = dx_ext[:-1]
    dx_north = dx_ext[1:]
    # Reshape lat metrics to broadcast over (n_lat+1, n_lon+1[, nlev]).
    bcast = (slice(None),) + (jnp.newaxis,) * (u.ndim - 1)
    du_circ = (u_south * dx_south[bcast] - u_north * dx_north[bcast])

    # Append periodic wrap column to dv_circ.
    dv_circ_full = jnp.concatenate([dv_circ, dv_circ[:, 0:1]], axis=1)

    circ = du_circ + dv_circ_full

    # Compute vorticity only on interior rows (pole rows zero by
    # construction; avoids 1/0 division — issue #173).
    circ_interior = circ[1:-1]
    zeta_interior = circ_interior / A_vertex_interior[bcast]
    # Pad pole rows with zero along the lat axis (extra (0, 0) pads
    # for trailing dims if 3D).
    zeta = jnp.pad(zeta_interior, ((1, 1), (0, 0), *pad_extra))

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

    # v-face latitudes — interior only (issue #173: avoid pole division)
    lat_interior = 0.5 * (lat[:-1] + lat[1:])  # (n_lat-1,)
    cos_lat_v_int = jnp.cos(lat_interior)  # nonzero for interior rows
    dx_v_int = R * cos_lat_v_int * dlon  # (n_lat-1,)

    # zeta[:, j+1] - zeta[:, j] for j=0..n_lon-1
    dzeta = zeta[:, 1:] - zeta[:, :-1]  # (n_lat+1, n_lon [, nlev])

    # Compute gradient only on interior rows (1..n_lat-1), pad poles
    # with zero.  Single Pad HLO op replaces alloc-zeros + concatenate-
    # of-three.
    dzeta_int = dzeta[1:-1]  # (n_lat-1, n_lon [, nlev])
    if zeta.ndim == 2:
        grad_int = dzeta_int / dx_v_int[:, jnp.newaxis]
    else:
        grad_int = dzeta_int / dx_v_int[:, jnp.newaxis, jnp.newaxis]
    pad_axes = ((0, 0),) * (grad_int.ndim - 1)
    return jnp.pad(grad_int, ((1, 1), *pad_axes))


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
    # Native 2D and 3D — all underlying operators (``divergence_cgrid``,
    # ``gradient_*_cgrid``, ``curl_vertex_cgrid``, ``_gradient_curl_to_*``)
    # natively support 3D inputs.  Masks remain 2D and broadcast over
    # the trailing level axis when present.
    is_3d = u.ndim == 3

    def _bcast(m, like):
        # Broadcast 2D ``m`` over trailing axes of ``like``.
        return m[..., jnp.newaxis] if is_3d else m

    # Apply face masks before computing div and curl
    u_eff = u if u_mask is None else u * _bcast(u_mask, u)
    v_eff = v if v_mask is None else v * _bcast(v_mask, v)

    # 1. Divergence at cell centers
    div = divergence_cgrid(u_eff, v_eff, grid)
    if mask is not None:
        div = div * _bcast(mask, div)

    # 2. grad(div) at faces
    grad_div_u = gradient_x_cgrid(div, grid)  # (n_lat, n_lon+1[, nlev])
    grad_div_v = gradient_y_cgrid(div, grid)  # (n_lat+1, n_lon[, nlev])

    # 3. Curl at vertices
    zeta = curl_vertex_cgrid(u_eff, v_eff, grid)  # (n_lat+1, n_lon+1[, nlev])

    # Mask curl at land-adjacent vertices
    if mask is not None:
        vmask = _compute_vertex_mask(mask)
        zeta = zeta * _bcast(vmask, zeta)

    # 4. Tangential gradient of curl at faces
    grad_curl_u = _gradient_curl_to_u(zeta, grid)
    grad_curl_v = _gradient_curl_to_v(zeta, grid)

    # 5. Vector Laplacian = grad(div) - curl(curl).  curl(curl F) =
    # k × ∇ζ = (-∂ζ/∂y, +∂ζ/∂x).  Signs verified via bump tests.
    vlap_u = grad_div_u - grad_curl_u
    vlap_v = grad_div_v + grad_curl_v

    # Apply face masks to output
    if u_mask is not None:
        vlap_u = vlap_u * _bcast(u_mask, vlap_u)
    if v_mask is not None:
        vlap_v = vlap_v * _bcast(v_mask, vlap_v)

    return vlap_u, vlap_v


def vector_bilaplacian_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Biharmonic (∇⁴) vector operator on the C-grid.

    Computes ∇²(∇²(u, v)) by applying ``vector_laplacian_cgrid`` twice.

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
    bilap_u : same shape as u
        ∇⁴u component (biharmonic tendency for du/dt).
    bilap_v : same shape as v
        ∇⁴v component (biharmonic tendency for dv/dt).
    """
    vlap_u, vlap_v = vector_laplacian_cgrid(
        u, v, grid, mask=mask, u_mask=u_mask, v_mask=v_mask)
    bilap_u, bilap_v = vector_laplacian_cgrid(
        vlap_u, vlap_v, grid, mask=mask, u_mask=u_mask, v_mask=v_mask)
    return bilap_u, bilap_v


def biharmonic_scaling_factor(grid: LatLonGrid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Grid-dependent scaling for biharmonic viscosity on a lat-lon grid.

    On a latitude-longitude grid the zonal grid spacing shrinks as
    ``cos(lat)`` near the poles.  Because the biharmonic CFL scales
    as ``B_h * dt / dx^4``, a constant ``B_h`` violates CFL near the
    poles while under-diffusing at the equator.

    Following the MOM6 convention (Griffies & Hallberg 2000), the
    biharmonic coefficient should be multiplied by
    ``(dx_local / dx_ref)^4`` where ``dx_ref`` is a reference spacing
    (typically the maximum or equatorial value).  This function returns
    the pre-computed scaling arrays for u-face and v-face points.

    Usage in the tendency function::

        scale_u, scale_v = biharmonic_scaling_factor(grid)
        bilap_u, bilap_v = vector_bilaplacian_cgrid(u, v, grid, ...)
        du_dt -= B_h * scale_u * bilap_u
        dv_dt -= B_h * scale_v * bilap_v

    Parameters
    ----------
    grid : LatLonGrid

    Returns
    -------
    scale_u : (n_lat,)
        Scaling factor at u-face latitudes.  Callers should reshape
        to ``[:, None]`` for 2D fields or ``[:, None, None]`` for 3D.
    scale_v : (n_lat+1,)
        Scaling factor at v-face latitudes.
    """
    cos_lat = grid.cos_lat  # (n_lat,)

    # Reference: equatorial (maximum) spacing
    cos_max = 1.0

    # u-face points sit at cell-center latitudes
    scale_u = (cos_lat / cos_max) ** 4  # (n_lat,)

    # v-face points sit at latitude interfaces between cells;
    # interpolate cos_lat to v-face positions.
    cos_v_interior = 0.5 * (cos_lat[:-1] + cos_lat[1:])  # (n_lat-1,)
    cos_v = jnp.concatenate([
        cos_lat[:1],         # south boundary ≈ cos(lat[0])
        cos_v_interior,
        cos_lat[-1:],        # north boundary ≈ cos(lat[-1])
    ])  # (n_lat+1,)
    scale_v = (cos_v / cos_max) ** 4  # (n_lat+1,)

    return scale_u, scale_v


def strain_rate_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Strain rate components on the C-grid.

    Returns tension D_T at cell centers and shearing strain D_S at vertices.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)

    Returns
    -------
    D_T : (n_lat, n_lon, ...) at h-points — du/dx - dv/dy
    D_S : (n_lat+1, n_lon+1, ...) at q-points — dv/dx + du/dy
    """
    # Native 2D and 3D — broadcast 1D / 2D metrics over the trailing
    # level axis when needed.  Eliminates the prior moveaxis + vmap +
    # moveaxis round-trip.
    is_3d = u.ndim == 3
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    lat = grid.lat
    cos_lat = grid.cos_lat
    n_lon = grid.n_lon

    def _bcast2d(m):
        return m[..., jnp.newaxis] if is_3d else m

    # Reshape lat-only metric to broadcast along the trailing axes.
    lat_bcast = (slice(None),) + (jnp.newaxis,) * (u.ndim - 1)
    pad_extra = ((0, 0),) * (u.ndim - 2)

    u_eff = u if u_mask is None else u * _bcast2d(u_mask)
    v_eff = v if v_mask is None else v * _bcast2d(v_mask)

    # Structural periodicity wrap column.  Works directly for 2D and 3D.
    u_eff = jnp.concatenate([u_eff[:, :n_lon], u_eff[:, 0:1]], axis=1)

    # --- D_T at h-points: du/dx - dv/dy ---
    face_dy = R * dlat
    u_east = u_eff[:, 1:]
    u_west = u_eff[:, :-1]
    du_dx = (u_east - u_west) * face_dy

    lat_interior = 0.5 * (lat[:-1] + lat[1:])
    cos_lat_v = jnp.pad(
        jnp.maximum(jnp.cos(lat_interior), 1e-10),
        (1, 1), constant_values=1e-10,
    )
    face_dx = R * cos_lat_v * dlon  # (n_lat+1,)

    v_north = v_eff[1:]
    v_south = v_eff[:-1]
    dv_dy = (v_north * face_dx[1:][lat_bcast]
             - v_south * face_dx[:-1][lat_bcast])

    area = grid.area  # (n_lat, n_lon) — broadcasts naturally over (...,, nlev)
    D_T = (du_dx - dv_dy) / area[..., jnp.newaxis] if is_3d else \
        (du_dx - dv_dy) / area
    if mask is not None:
        D_T = D_T * _bcast2d(mask)

    # --- D_S at q-points: dv/dx + du/dy ---
    sin_lat = jnp.sin(lat)
    sin_ext = jnp.pad(sin_lat, (1, 1), constant_values=(-1.0, 1.0))
    A_vertex = R**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])
    A_vertex = jnp.maximum(A_vertex, 1e-30)

    dx_cell = R * cos_lat * dlon
    dy_edge = R * dlat

    # dv/dx at vertex: (v_east - v_west) * dy / A_vertex
    v_east = v_eff
    v_west = jnp.roll(v_eff, 1, axis=1)
    dv_circ = (v_east - v_west) * dy_edge
    dv_circ_full = jnp.concatenate([dv_circ, dv_circ[:, 0:1]], axis=1)

    # du/dy at vertex: sign FLIPPED vs curl.
    u_ext = jnp.pad(u_eff, ((1, 1), (0, 0), *pad_extra))
    dx_ext = jnp.pad(dx_cell, (1, 1))
    u_south = u_ext[:-1]
    u_north = u_ext[1:]
    dx_south = dx_ext[:-1]
    dx_north = dx_ext[1:]
    du_circ = (u_north * dx_north[lat_bcast]
               - u_south * dx_south[lat_bcast])

    D_S = (dv_circ_full + du_circ) / A_vertex[lat_bcast]
    # Pole rows zero (wall BC).
    D_S = jnp.pad(D_S[1:-1], ((1, 1), (0, 0), *pad_extra))

    if mask is not None:
        vmask = _compute_vertex_mask(mask)
        D_S = D_S * _bcast2d(vmask)

    return D_T, D_S


def smagorinsky_viscosity_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
    C_smag: float,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Smagorinsky viscosity coefficient at cell centers.

    A_smag = (C_smag * Delta)^2 * |D|

    where |D| = sqrt(D_T^2 + D_S^2) and Delta = sqrt(cell area).

    Returns
    -------
    A_smag : (n_lat, n_lon, ...) at h-points [m^2/s]
    """
    D_T, D_S = strain_rate_cgrid(u, v, grid,
                                  mask=mask, u_mask=u_mask, v_mask=v_mask)

    # Interpolate D_S from vertices to cell centers
    if D_S.ndim == 2:
        D_S_h = 0.25 * (D_S[:-1, :-1] + D_S[1:, :-1]
                         + D_S[:-1, 1:] + D_S[1:, 1:])
    else:
        D_S_h = 0.25 * (D_S[:-1, :-1, :] + D_S[1:, :-1, :]
                         + D_S[:-1, 1:, :] + D_S[1:, 1:, :])

    # Small epsilon prevents NaN gradient of sqrt at zero (masked points).
    deformation = jnp.sqrt(D_T**2 + D_S_h**2 + 1e-30)

    Delta = jnp.sqrt(grid.area)
    if D_T.ndim == 3:
        Delta = Delta[..., jnp.newaxis]

    A_smag = (C_smag * Delta)**2 * deformation

    if mask is not None:
        m = mask[..., jnp.newaxis] if D_T.ndim == 3 else mask
        A_smag = A_smag * m

    return A_smag


def stress_divergence_cgrid(
    stress_h: jnp.ndarray,
    stress_q: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    normalize: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stress divergence on the C-grid (exact discrete adjoint of strain).

    Given pre-formed stresses:
        stress_h = A_h * D_T   at h-points (cell centers)
        stress_q = A_q * D_S   at q-points (vertices)

    Returns the momentum tendency at face points, constructed as the
    EXACT discrete adjoint of ``strain_rate_cgrid``.  The area-weighted
    energy identity

        sum_faces (u · tend_u + v · tend_v) · area_face_dual
            = -sum_h stress_h · D_T · area_h
              - sum_q stress_q · D_S · A_vertex_q

    holds to machine precision (the area_face_dual factors cancel
    with the internal normalization), guaranteeing energy stability
    for any non-negative A_h, A_q.

    The stencils are derived by transposing the exact weights used
    in ``strain_rate_cgrid``:

    For D_T_num[i,j] = (u[i,j+1] - u[i,j]) * dy
                       - (v[i+1,j]*dx_v[i+1] - v[i,j]*dx_v[i]):

        tend_u[i,k] += dy * (sh[i,k] - sh[i,k-1])
        tend_v[m,j] += -(dx_v[m] * sh[m,j] - dx_v[m-1] * sh[m-1,j])

    For D_S_num[m,j] = (v[m,j] - v[m,(j-1)%n]) * dy_edge
                       + u[m,j]*dx_cell[m] - u[m-1,j]*dx_cell[m-1]:

        tend_u[i,k] += dx_cell[i] * (sq[i+1,k] - sq[i,k])
        tend_v[m,j] += dy_edge * (sq[m,j+1] - sq[m,j])

    Because we want the NEGATIVE adjoint (dissipative when added):
    tend = -S^T(stress), all signs above are negated.

    Parameters
    ----------
    stress_h : (n_lat, n_lon) or (n_lat, n_lon, nlev)
        A_h * D_T at cell centers.
    stress_q : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev)
        A_q * D_S at vertices.
    grid : LatLonGrid
    u_mask, v_mask : optional face masks.

    Returns
    -------
    tend_u : (n_lat, n_lon+1, ...) at u-faces
    tend_v : (n_lat+1, n_lon, ...) at v-faces
    """
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    lat = grid.lat
    cos_lat = grid.cos_lat

    dy = R * dlat       # face_dy: meridional edge length
    dy_edge = R * dlat  # same as dy (edge length for vertex circulation)
    dx_cell = R * cos_lat * dlon  # (n_lat,) zonal edge at cell-center latitude

    # v-face latitudes and zonal edge lengths.  cos(±π/2) is roundoff-
    # level; build cos_lat_v directly with a 1e-10 floor at the pole
    # rows via Pad constant_values.
    lat_interior = 0.5 * (lat[:-1] + lat[1:])
    cos_lat_v = jnp.pad(
        jnp.maximum(jnp.cos(lat_interior), 1e-10),
        (1, 1), constant_values=1e-10,
    )
    dx_v = R * cos_lat_v * dlon  # (n_lat+1,) face_dx at v-face latitudes

    # Reshape lat metric to broadcast over (n_lat, n_lon[+1][, nlev]).
    is_3d = stress_h.ndim == 3
    lat_bcast = (slice(None),) + (jnp.newaxis,) * (stress_h.ndim - 1)

    # =====================================================================
    # tend_u: contribution from D_T adjoint
    # =====================================================================
    # u[i,k] in D_T_num[i,j]:  coeff +dy at j=k-1, coeff -dy at j=k
    # Adjoint: dy * (sh[i,k] - sh[i,k-1]) with periodic wrap.
    sh_west = jnp.roll(stress_h, 1, axis=1)
    dsh = stress_h - sh_west
    dsh_full = jnp.concatenate([dsh, dsh[:, 0:1]], axis=1)
    tend_u_DT = dy * dsh_full  # (n_lat, n_lon+1, ...)

    # =====================================================================
    # tend_u: contribution from D_S adjoint
    # =====================================================================
    # Adjoint: dx_cell[i] * (sq[i+1,k] - sq[i,k])
    dsq_meridional = stress_q[1:] - stress_q[:-1]  # (n_lat, n_lon+1, ...)
    tend_u_DS = dx_cell[lat_bcast] * dsq_meridional

    tend_u = tend_u_DT + tend_u_DS

    # =====================================================================
    # tend_v: contribution from D_T adjoint
    # =====================================================================
    # Adjoint: dx_v[m] * (sh[m-1,j] - sh[m,j]).  Interior v-faces only;
    # pole rows zero (wall BC).
    dsh_merid = stress_h[:-1] - stress_h[1:]
    tend_v_DT_interior = dx_v[1:-1][lat_bcast] * dsh_merid
    # Single Pad HLO op replaces alloc-zeros + concatenate-of-three.
    pad_axes = ((0, 0),) * (tend_v_DT_interior.ndim - 1)
    tend_v_DT = jnp.pad(tend_v_DT_interior, ((1, 1), *pad_axes))

    # =====================================================================
    # tend_v: contribution from D_S adjoint
    # =====================================================================
    # v[m,j] in D_S_num[m',j']:
    #   v[m,j] appears at vertex (m, j) as v_east: coeff +dy_edge
    #   v[m,j] appears at vertex (m, (j+1)%n) as v_west: coeff -dy_edge
    #
    # Wait: D_S_num uses v_east = v[i,j] and v_west = v[i,(j-1)%n],
    # so at vertex (m, j'): v_east = v[m,j'], v_west = v[m,(j'-1)%n]
    #
    # v[m,j] appears at vertex (m, j) as v_east: coeff +dy_edge
    # v[m,j] appears at vertex (m, j+1) as v_west (since (j+1-1)%n = j): coeff -dy_edge
    #
    # Adjoint: -[+dy_edge*sq[m,j] + (-dy_edge)*sq[m,j+1]]
    #        = dy_edge * (sq[m,j+1] - sq[m,j])
    dsq_zonal = stress_q[:, 1:] - stress_q[:, :-1]  # (n_lat+1, n_lon, ...)
    # stress_q[:, n_lon] is the periodic wrap = stress_q[:, 0], so
    # dsq_zonal[:, j] = sq[:, j+1] - sq[:, j] for j=0..n_lon-1
    tend_v_DS = dy_edge * dsq_zonal

    tend_v = tend_v_DT + tend_v_DS

    # --- Area normalization ---
    # The raw adjoint gives "sum of edge fluxes" around the face dual cell.
    # Dividing by the dual cell area converts to a proper acceleration
    # (m/s²), consistent with vector_laplacian_cgrid units.
    #
    # u-face dual cell area: dy * dx_cell[i] = R²*dlat*dlon*cos(lat[i])
    # v-face dual cell area: dy_edge * dx_v[m] = R²*dlat*dlon*cos(lat_v[m])
    #
    # These are the products of the SAME edge lengths used in the stencil,
    # ensuring the adjoint identity:
    #   sum u * tend * area_u_dual = sum u * tend_raw = -sum A*D²*area_h - ...
    # holds exactly (area_u_dual cancels in the energy diagnostic).
    area_u_dual = dy * dx_cell  # (n_lat,)
    area_v_dual = dy_edge * dx_v  # (n_lat+1,)
    # Floor to avoid division by zero at poles
    area_u_dual = jnp.maximum(area_u_dual, 1e-30)
    area_v_dual = jnp.maximum(area_v_dual, 1e-30)

    if normalize:
        tend_u = tend_u / area_u_dual[lat_bcast]
        tend_v = tend_v / area_v_dual[lat_bcast]

    if u_mask is not None:
        um = u_mask[..., jnp.newaxis] if is_3d and u_mask.ndim == 2 else u_mask
        tend_u = tend_u * um
    if v_mask is not None:
        vm = v_mask[..., jnp.newaxis] if is_3d and v_mask.ndim == 2 else v_mask
        tend_v = tend_v * vm

    return tend_u, tend_v


def viscous_tendency_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    A_h: jnp.ndarray | float,
    A_q: jnp.ndarray | float,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    normalize: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Viscous tendency via stress-tensor formulation on the C-grid.

    Computes:
        1. Strain: D_T, D_S = strain_rate(u, v)
        2. Stress: stress_h = A_h * D_T,  stress_q = A_q * D_S
        3. Tendency: stress_divergence(stress_h, stress_q)

    This operator differs from ``vector_laplacian_cgrid`` (grad-div minus
    curl-curl) by spherical metric terms.  On the discrete C-grid the
    stress-tensor form is the EXACT adjoint of the strain-rate operator,
    guaranteeing the energy identity:

        sum (u·tend_u·area_u + v·tend_v·area_v)
            = -sum A_h·D_T²·area_h - sum A_q·D_S²·A_vert

    to machine precision. This holds for both uniform and spatially
    varying A_h, A_q, making it the correct choice for Smagorinsky-type
    viscosity where the coefficient varies in space.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid
    A_h : scalar or (n_lat, n_lon, ...) viscosity at h-points
    A_q : scalar or (n_lat+1, n_lon+1, ...) viscosity at q-points
    mask, u_mask, v_mask : optional masks

    Returns
    -------
    tend_u, tend_v : dissipative when ADDED to du/dt, dv/dt
    """
    # Native 2D and 3D — ``strain_rate_cgrid`` and
    # ``stress_divergence_cgrid`` both natively support 3D inputs.
    # The only care needed is broadcasting 2D ``A_h``/``A_q``
    # coefficients over the trailing level axis when the velocity is 3D.
    is_3d = u.ndim == 3

    def _bcast_coef(A, like_shape_ndim):
        # Add a trailing newaxis if A is a 2D array and the field is 3D.
        if (
            is_3d and isinstance(A, jnp.ndarray) and A.ndim == 2
        ):
            return A[..., jnp.newaxis]
        return A

    # Apply face masks to input velocities
    if u_mask is not None:
        u_eff = u * (u_mask[..., jnp.newaxis] if is_3d else u_mask)
    else:
        u_eff = u
    if v_mask is not None:
        v_eff = v * (v_mask[..., jnp.newaxis] if is_3d else v_mask)
    else:
        v_eff = v

    # 1. Strain rate (3D-native)
    D_T, D_S = strain_rate_cgrid(u_eff, v_eff, grid, mask=mask)

    # 2. Form stresses (broadcast 2D coefficients over the level axis)
    stress_h = _bcast_coef(A_h, D_T.ndim) * D_T
    stress_q = _bcast_coef(A_q, D_S.ndim) * D_S

    # 3. Stress divergence (already 3D-native; normalize controls area
    # normalization)
    tend_u, tend_v = stress_divergence_cgrid(
        stress_h, stress_q, grid,
        u_mask=u_mask, v_mask=v_mask, normalize=normalize)

    return tend_u, tend_v


def _vertex_area(grid: LatLonGrid) -> jnp.ndarray:
    """Dual-cell area at vertex (corner) points.

    Returns
    -------
    A_vertex : (n_lat+1,)
        Area of each vertex dual cell.  Pole rows are set to a small
        positive floor (1e-30) to avoid division by zero.
    """
    R = grid.radius
    dlon = grid.dlon
    lat = grid.lat
    sin_lat = jnp.sin(lat)
    # Single Pad HLO op (constant_values=(-1, 1)) replaces alloc-2-
    # singletons + concatenate-of-three.
    sin_ext = jnp.pad(sin_lat, (1, 1), constant_values=(-1.0, 1.0))
    A_v = R**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])
    return jnp.maximum(A_v, 1e-30)


def smagorinsky_viscosity_q_cgrid(
    D_T: jnp.ndarray,
    D_S: jnp.ndarray,
    grid: LatLonGrid,
    C_smag: float,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Smagorinsky viscosity coefficient computed DIRECTLY at q-points.

    Unlike interpolating A_smag from h-points to q-points, this computes
    the coefficient at vertex locations using D_S (native at q-points)
    and D_T interpolated from h-points to q-points.

        A_smag_q = (C_s * Delta_q)^2 * |D|_q

    where Delta_q = sqrt(A_vertex) is the vertex dual cell length scale
    and |D|_q = sqrt(D_T_q^2 + D_S^2).

    Parameters
    ----------
    D_T : (n_lat, n_lon) or (n_lat, n_lon, nlev) at h-points
    D_S : (n_lat+1, n_lon+1) or (n_lat+1, n_lon+1, nlev) at q-points
    grid : LatLonGrid
    C_smag : Smagorinsky coefficient
    mask : cell-center land mask, optional

    Returns
    -------
    A_smag_q : (n_lat+1, n_lon+1, ...) at q-points [m^2/s]
    """
    is_3d = D_T.ndim == 3

    # Interpolate D_T from h-points to q-points (4-point average).
    # ``D_T[:-1]`` works for both 2D and 3D (sliced along axis 0).
    D_T_roll = jnp.roll(D_T, 1, axis=1)
    D_T_q = 0.25 * (D_T[:-1] + D_T[1:] + D_T_roll[:-1] + D_T_roll[1:])

    # D_T_q shape: (n_lat-1, n_lon, ...). Need (n_lat+1, n_lon+1, ...).
    # Pad pole rows with zero (degenerate vertices).
    pad_axes = ((0, 0),) * (D_T_q.ndim - 1)
    D_T_q = jnp.pad(D_T_q, ((1, 1), *pad_axes))
    # Append periodic wrap column
    D_T_q = jnp.concatenate(
        [D_T_q, D_T_q[:, 0:1]], axis=1)  # (n_lat+1, n_lon+1, ...)

    # Small epsilon prevents NaN gradient of sqrt at zero (masked points).
    deformation_q = jnp.sqrt(D_T_q**2 + D_S**2 + 1e-30)

    # Vertex dual cell area; reshape for broadcast over (n_lat+1, n_lon+1[, nlev]).
    A_vert = _vertex_area(grid)  # (n_lat+1,)
    Delta_q = jnp.sqrt(A_vert)
    bcast = (slice(None),) + (jnp.newaxis,) * (D_T.ndim - 1)
    Delta_q = Delta_q[bcast]

    A_smag_q = (C_smag * Delta_q)**2 * deformation_q

    # Zero at pole vertices.
    A_smag_q = A_smag_q.at[0].set(0.0)
    A_smag_q = A_smag_q.at[-1].set(0.0)

    if mask is not None:
        vmask = _compute_vertex_mask(mask)
        if is_3d:
            vmask = vmask[..., jnp.newaxis]
        A_smag_q = A_smag_q * vmask

    return A_smag_q


def smagorinsky_biharmonic_tendency_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
    C_smag: float,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Biharmonic Smagorinsky viscosity via stress-tensor formulation.

    Uses the MOM6-style stress-tensor approach where the strain and
    stress-divergence operators are discrete adjoints on the C-grid.
    This guarantees energy stability by construction:

        dE/dt = -sum_h B_h * D_T[L(u)]^2 * area_h
                -sum_q B_q * D_S[L(u)]^2 * area_q  <= 0

    where L(u) is the (unit-coefficient) vector Laplacian and B is the
    biharmonic Smagorinsky coefficient B = C_s^2 * Delta^4 * |D|.

    Algorithm:
        1. Compute strain D_T, D_S of the input velocity
        2. Compute B_smag at h-points and q-points INDEPENDENTLY
           (no interpolation of coefficient between grids)
        3. First pass: unit-coefficient stress-divergence = vector Laplacian
           (u*, v*) = L_1(u, v)
        4. Second pass: B-weighted stress-divergence of (u*, v*)
           tend = L_B(u*, v*)

    The result is subtracted by the caller:  du/dt -= tend_u.

    Previous implementation used a sandwich form nabla^2(B nabla^2 u)
    which is NOT energy-stable on the discrete C-grid because the
    vector Laplacian (grad-div minus curl-curl) does not satisfy
    discrete integration-by-parts with spatially-varying B.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid
    C_smag : Smagorinsky coefficient (dimensionless)
    mask : (n_lat, n_lon) cell-center land mask, optional
    u_mask : (n_lat, n_lon+1) u-face mask, optional
    v_mask : (n_lat+1, n_lon) v-face mask, optional

    Returns
    -------
    tend_u, tend_v : same shapes as u, v
        Biharmonic dissipative tendencies.  Caller subtracts these:
        du/dt -= tend_u, dv/dt -= tend_v.
    """
    is_3d = u.ndim == 3

    # --- 1. Compute strain of the INPUT velocity ---
    D_T, D_S = strain_rate_cgrid(
        u, v, grid, mask=mask, u_mask=u_mask, v_mask=v_mask)

    # --- 2. Smagorinsky coefficient at h-points and q-points ---
    # A_smag [m^2/s] at h-points (cell centers)
    A_smag_h = smagorinsky_viscosity_cgrid(
        u, v, grid, C_smag,
        mask=mask, u_mask=u_mask, v_mask=v_mask)

    # A_smag at q-points computed DIRECTLY (not interpolated from h)
    A_smag_q = smagorinsky_viscosity_q_cgrid(
        D_T, D_S, grid, C_smag, mask=mask)

    # --- 3. First pass: UNNORMALIZED unit-coefficient stress-divergence ---
    # Returns raw flux sums with units m/s (same as velocity), NOT 1/(ms).
    # MOM6 approach: inner div(strain(u)) is NOT divided by cell area,
    # producing a velocity-like intermediate for the second pass.
    u_star, v_star = viscous_tendency_cgrid(
        u, v, grid, 1.0, 1.0,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=False)

    # --- 4. Second pass: A_smag-weighted NORMALIZED stress-divergence ---
    # Uses A_smag (m²/s). Two passes give biharmonic scaling: A*u/dx⁴.
    # CFL: A_smag × dt / dx² = C_s² × |D| × dt ≈ 0.003. Safe.
    tend_u, tend_v = viscous_tendency_cgrid(
        u_star, v_star, grid, A_smag_h, A_smag_q,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=True)

    return tend_u, tend_v


# =============================================================================
# Leith viscosity (Leith 1996; Fox-Kemper & Menemenlis 2008)
# =============================================================================
# Leith is a flow-adaptive horizontal viscosity whose magnitude scales with
# the gradient of the relative vorticity.  At cell centres
#
#     A_L = (C_L * Δ)³ * |∇ζ|                    (classical Leith)
#     A_L = (C_L * Δ)³ * sqrt(|∇ζ|² + |∇δ|²)      (modified Leith, incl.
#                                                  divergence gradient)
#
# with Δ = sqrt(cell area), ζ = ∂v/∂x − ∂u/∂y the relative vorticity at
# vertices, and δ = ∂u/∂x + ∂v/∂y the horizontal divergence at cell centres.
# The classical form targets quasi-nondivergent flows; the modified form adds
# a divergence-gradient term and is preferred when the simulated flow has
# strong vertical motions (Fox-Kemper & Menemenlis 2008 §2.3).
#
# Units check: (m)³ · (1/(m·s)) = m²/s, matching a harmonic (Laplacian)
# viscosity coefficient.  Feeding A_L into ``viscous_tendency_cgrid`` gives
# a ∇·(A_L ∇u)-type tendency; feeding it into the two-pass
# ``leith_biharmonic_tendency_cgrid`` below gives the Leith-biharmonic
# operator ∇²(A_L ∇²u) with effective coefficient (C_L)³ Δ⁵ |∇ζ|.

def _grad_zeta_mag_h(zeta_q: jnp.ndarray, grid: "LatLonGrid") -> jnp.ndarray:
    """|∇ζ| at cell centres from ζ at vertices.

    Parameters
    ----------
    zeta_q : (n_lat+1, n_lon+1, ...) relative vorticity at vertices.
    grid : LatLonGrid.

    Returns
    -------
    grad_mag : (n_lat, n_lon, ...) with units 1/(m·s).
    """
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    cos_lat = grid.cos_lat

    if zeta_q.ndim == 3:
        cos_lat_b = cos_lat[:, jnp.newaxis, jnp.newaxis]
    else:
        cos_lat_b = cos_lat[:, jnp.newaxis]

    # Cell-centre spacings.  cos_lat evaluated at cell-centre latitudes.
    dx_h = R * cos_lat_b * dlon                           # (n_lat,1[,1])
    dy_h = R * dlat                                       # scalar

    # ∂ζ/∂x at (i,j): average of north/south vertex-pair zonal differences.
    dz_dx = 0.5 * ((zeta_q[:-1, 1:] - zeta_q[:-1, :-1])
                   + (zeta_q[1:, 1:] - zeta_q[1:, :-1])) / dx_h
    # ∂ζ/∂y at (i,j): average of west/east vertex-pair meridional differences.
    dz_dy = 0.5 * ((zeta_q[1:, :-1] - zeta_q[:-1, :-1])
                   + (zeta_q[1:, 1:] - zeta_q[:-1, 1:])) / dy_h

    return jnp.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1e-30)


def _grad_div_mag_h(div_h: jnp.ndarray, grid: "LatLonGrid") -> jnp.ndarray:
    """|∇δ| at cell centres from δ at cell centres (periodic in lon).

    Uses centred differences with periodic wrap in longitude and one-sided
    reflection at the poles (so the magnitude remains non-negative).

    Parameters
    ----------
    div_h : (n_lat, n_lon, ...) horizontal divergence at cell centres.
    grid : LatLonGrid.

    Returns
    -------
    grad_mag : (n_lat, n_lon, ...) with units 1/(m·s).
    """
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    cos_lat = grid.cos_lat

    if div_h.ndim == 3:
        cos_lat_b = cos_lat[:, jnp.newaxis, jnp.newaxis]
    else:
        cos_lat_b = cos_lat[:, jnp.newaxis]

    dx_h = R * cos_lat_b * dlon
    dy_h = R * dlat

    # Zonal gradient: centred difference with periodic wrap (works for any ndim).
    dd_dx = (jnp.roll(div_h, -1, axis=1) - jnp.roll(div_h, 1, axis=1)) / (2.0 * dx_h)

    # Meridional gradient: centred in interior, one-sided at pole rows.
    dd_dy_interior = (div_h[2:] - div_h[:-2]) / (2.0 * dy_h)
    dd_dy_south = (div_h[1:2] - div_h[0:1]) / dy_h
    dd_dy_north = (div_h[-1:] - div_h[-2:-1]) / dy_h
    dd_dy = jnp.concatenate([dd_dy_south, dd_dy_interior, dd_dy_north], axis=0)

    return jnp.sqrt(dd_dx ** 2 + dd_dy ** 2 + 1e-30)


def leith_viscosity_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: "LatLonGrid",
    C_leith: float,
    *,
    modified: bool = False,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Leith viscosity coefficient at cell centres (h-points).

    ``A_L = (C_L * Δ)³ * |∇ζ|`` (classical) or
    ``A_L = (C_L * Δ)³ * sqrt(|∇ζ|² + |∇δ|²)`` (modified).

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev) u-face velocity.
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev) v-face velocity.
    grid : LatLonGrid.
    C_leith : dimensionless Leith coefficient (typical 1.0–2.0).
    modified : if True, include the divergence-gradient term.
    mask : cell-centre land mask, optional.
    u_mask, v_mask : face masks, optional.

    Returns
    -------
    A_leith : (n_lat, n_lon, ...) viscosity [m²/s] at h-points.
    """
    is_3d = u.ndim == 3

    # Apply face masks to input velocities (same convention as Smagorinsky).
    # Expand 2D masks to 3D when velocity is 3D to avoid broadcast mismatch.
    _um = u_mask[..., jnp.newaxis] if (u_mask is not None and is_3d) else u_mask
    _vm = v_mask[..., jnp.newaxis] if (v_mask is not None and is_3d) else v_mask
    u_eff = u if _um is None else u * _um
    v_eff = v if _vm is None else v * _vm

    # 1. Relative vorticity at vertices.
    zeta_q = curl_vertex_cgrid(u_eff, v_eff, grid)
    grad_zeta = _grad_zeta_mag_h(zeta_q, grid)

    total_sq = grad_zeta ** 2
    if modified:
        # Divergence at cell centres.  Includes boundary-safe spherical
        # metric terms already.
        div_h = divergence_cgrid(u_eff, v_eff, grid,
                                 u_mask=u_mask, v_mask=v_mask)
        total_sq = total_sq + _grad_div_mag_h(div_h, grid) ** 2

    # Epsilon guards sqrt-gradient at zero (already added to grad_zeta).
    norm = jnp.sqrt(total_sq + 1e-30)

    Delta = jnp.sqrt(grid.area)
    if is_3d:
        Delta = Delta[..., jnp.newaxis]

    A_leith = (C_leith * Delta) ** 3 * norm

    if mask is not None:
        m = mask[..., jnp.newaxis] if is_3d else mask
        A_leith = A_leith * m

    return A_leith


def leith_viscosity_q_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: "LatLonGrid",
    C_leith: float,
    *,
    modified: bool = False,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Leith viscosity coefficient at vertices (q-points).

    Evaluated from ``|∇ζ|`` on the q-point grid itself: centred differences
    of ζ between adjacent vertices, normalised by the vertex dual-cell
    length scale Δ_q = sqrt(A_vertex).

    Parameters
    ----------
    u, v, grid, C_leith, modified, mask, u_mask, v_mask — see
    ``leith_viscosity_cgrid``.

    Returns
    -------
    A_leith_q : (n_lat+1, n_lon+1, ...) [m²/s] at vertices.
    """
    is_3d = u.ndim == 3

    _um = u_mask[..., jnp.newaxis] if (u_mask is not None and is_3d) else u_mask
    _vm = v_mask[..., jnp.newaxis] if (v_mask is not None and is_3d) else v_mask
    u_eff = u if _um is None else u * _um
    v_eff = v if _vm is None else v * _vm

    zeta_q = curl_vertex_cgrid(u_eff, v_eff, grid)   # (n_lat+1, n_lon+1[, nlev])

    # --- ∇ζ magnitude AT q-points via centred differences of ζ itself ---
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon

    # Cosine of q-point latitudes: cos(±π/2) is roundoff-level, so
    # build cos_lat_q directly via Pad of cos(interior) with 1e-10 floor
    # at the pole rows.  Single Pad HLO op replaces alloc-2-singletons +
    # concatenate-of-three + cos tower.
    lat = grid.lat
    lat_interior = 0.5 * (lat[:-1] + lat[1:])
    cos_lat_q = jnp.pad(
        jnp.maximum(jnp.cos(lat_interior), 1e-10),
        (1, 1), constant_values=1e-10,
    )

    bcast = (slice(None),) + (jnp.newaxis,) * (zeta_q.ndim - 1)
    cos_lat_q_b = cos_lat_q[bcast]

    dx_q = R * cos_lat_q_b * dlon
    dy_q = R * dlat

    # Zonal difference of ζ at q-points.  ``zeta_q`` has shape
    # ``(n_lat+1, n_lon+1)`` with the wrap column ``[:, n_lon] == [:, 0]``;
    # rolling the full array would make column 0's west neighbour be the
    # duplicate, not column ``n_lon-1``.  Do the centred difference on the
    # first ``n_lon`` columns and restore the wrap at the end.
    n_lon = grid.n_lon
    zeta_core = zeta_q[:, :n_lon]
    zeta_e_core = jnp.roll(zeta_core, -1, axis=1)
    zeta_w_core = jnp.roll(zeta_core, 1, axis=1)
    dx_q_core = dx_q[..., :n_lon] if dx_q.ndim > 1 else dx_q
    dz_dx_core = (zeta_e_core - zeta_w_core) / (2.0 * dx_q_core)
    dz_dx_q = jnp.concatenate(
        [dz_dx_core, dz_dx_core[:, 0:1]], axis=1)

    # Meridional difference of ζ at q-points.  Pad poles with their own row
    # so the centred stencil collapses to a one-sided difference there.
    zeta_south = jnp.concatenate([zeta_q[0:1], zeta_q[:-1]], axis=0)
    zeta_north = jnp.concatenate([zeta_q[1:], zeta_q[-1:]], axis=0)
    dz_dy_q = (zeta_north - zeta_south) / (2.0 * dy_q)

    total_sq = dz_dx_q ** 2 + dz_dy_q ** 2
    if modified:
        div_h = divergence_cgrid(u_eff, v_eff, grid,
                                 u_mask=u_mask, v_mask=v_mask)
        grad_div_h = _grad_div_mag_h(div_h, grid)
        # Interpolate h→q with the same 4-point average used for D_T_q,
        # then pad pole rows and the wrap column with zeros.  ``[:-1]``
        # / ``[1:]`` slice along axis 0 work for both 2D and 3D.
        gd_roll = jnp.roll(grad_div_h, 1, axis=1)
        gd_q_int = 0.25 * (
            grad_div_h[:-1] + grad_div_h[1:]
            + gd_roll[:-1] + gd_roll[1:]
        )
        # Pole rows zero; single Pad HLO op replaces alloc-zeros +
        # concatenate-of-three.
        pad_axes = ((0, 0),) * (gd_q_int.ndim - 1)
        grad_div_q = jnp.pad(gd_q_int, ((1, 1), *pad_axes))
        grad_div_q = jnp.concatenate(
            [grad_div_q, grad_div_q[:, 0:1]], axis=1)
        total_sq = total_sq + grad_div_q ** 2

    norm_q = jnp.sqrt(total_sq + 1e-30)

    A_vert = _vertex_area(grid)                          # (n_lat+1,)
    bcast = (slice(None),) + (jnp.newaxis,) * (norm_q.ndim - 1)
    Delta_q = jnp.sqrt(A_vert)[bcast]

    A_leith_q = (C_leith * Delta_q) ** 3 * norm_q

    # Zero at pole vertices, matching smagorinsky_viscosity_q_cgrid.
    # Slice + single Pad HLO op replaces zeros_like-of-slice ×2 +
    # concatenate-of-three.
    pad_axes = ((0, 0),) * (A_leith_q.ndim - 1)
    A_leith_q = jnp.pad(A_leith_q[1:-1], ((1, 1), *pad_axes))

    if mask is not None:
        vmask = _compute_vertex_mask(mask)
        if is_3d:
            vmask = vmask[..., jnp.newaxis]
        A_leith_q = A_leith_q * vmask

    return A_leith_q


def leith_biharmonic_tendency_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: "LatLonGrid",
    C_leith: float,
    *,
    modified: bool = False,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Leith-biharmonic viscosity via the MOM6 two-pass stress formulation.

    Mirrors ``smagorinsky_biharmonic_tendency_cgrid`` but with a
    vorticity-gradient coefficient.  The effective biharmonic coefficient
    is ``(C_L)³ · Δ⁵ · |∇ζ|`` (or with the divergence-gradient term when
    ``modified=True``).  Energy-stable by construction because the first
    pass is unnormalised and the second pass uses the same discrete
    adjoint ``stress_divergence_cgrid`` as the Smagorinsky biharmonic.

    Caller convention: ``du/dt -= tend_u`` (dissipative when SUBTRACTED).

    Returns
    -------
    tend_u, tend_v : same shapes as u, v.
    """
    # 1. Leith coefficients at h- and q-points.
    A_h = leith_viscosity_cgrid(
        u, v, grid, C_leith,
        modified=modified, mask=mask, u_mask=u_mask, v_mask=v_mask)
    A_q = leith_viscosity_q_cgrid(
        u, v, grid, C_leith,
        modified=modified, mask=mask, u_mask=u_mask, v_mask=v_mask)

    # 2. First pass: unit-coefficient, UNNORMALISED stress-divergence.
    u_star, v_star = viscous_tendency_cgrid(
        u, v, grid, 1.0, 1.0,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=False)

    # 3. Second pass: A_leith-weighted, NORMALISED stress-divergence.
    tend_u, tend_v = viscous_tendency_cgrid(
        u_star, v_star, grid, A_h, A_q,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=True)

    return tend_u, tend_v


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

    # Pole rows zero (degenerate vertices); single Pad HLO op replaces
    # alloc-zeros + concatenate-of-three.
    return jnp.pad(interior_full, ((1, 1), (0, 0)))


# =============================================================================
# Utility: compute face masks from cell mask
# =============================================================================

def compute_face_masks_3d(
    is_active_3d: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-level u-face and v-face masks from a 3D cell-activity mask.

    A face is wet at level k only if BOTH adjacent cells are wet at
    that level — partial-cell-aware analogue of ``compute_face_masks``.
    For columns with the same ``bottom_level``, this is identical to
    broadcasting the 2D ``compute_face_masks`` result.  For columns
    with different ``bottom_level`` (the realistic-bathymetry case),
    this correctly zeroes the face below the shallower column's
    seafloor.

    Parameters
    ----------
    is_active_3d : array, shape (n_lat, n_lon, nlev)
        Per-cell activity mask: True/1.0 where the cell has water,
        False/0.0 below the seafloor.  Typically
        ``partial_coord.is_active.astype(...)``.

    Returns
    -------
    u_mask_3d : array, shape (n_lat, n_lon+1, nlev)
        Wet u-face mask at each level (periodic in longitude).
    v_mask_3d : array, shape (n_lat+1, n_lon, nlev)
        Wet v-face mask at each level (pole rows always zero).
    """
    a = is_active_3d.astype(jnp.float32)
    # u-face j is between cell (j-1) mod n_lon (west) and cell j (east).
    u_mask_interior = a * jnp.roll(a, 1, axis=1)
    u_mask = jnp.concatenate(
        [u_mask_interior, u_mask_interior[:, 0:1, :]], axis=1,
    )
    # v-face i is between cell i-1 (south) and cell i (north).  Pole
    # boundaries (i=0 and i=n_lat) are always wall.
    v_mask_interior = a[:-1] * a[1:]
    pad_axes = ((0, 0),) * (v_mask_interior.ndim - 1)
    v_mask = jnp.pad(v_mask_interior, ((1, 1), *pad_axes))
    return u_mask, v_mask


def partial_cell_pgf_correction_x(
    centroid_depth: jnp.ndarray,
    rho_prime: jnp.ndarray,
    grid: LatLonGrid,
    g: float,
) -> jnp.ndarray:
    """Adcroft & Campin (2004) face correction at u-faces (zonal direction).

    Returns an additive correction to ``∂p'/∂x`` that shifts each
    adjacent cell's baroclinic pressure to the face-reference depth
    (the shallower of the two cell centroids).  At u-face j between
    cell west=(j-1) mod n_lon and cell east=j::

        face_ref[k] = min(centroid_east[k], centroid_west[k])
        excess_east[k] = centroid_east[k] - face_ref[k]   ≥ 0
        excess_west[k] = centroid_west[k] - face_ref[k]   ≥ 0
        correction[k] = -g * (rho_prime_east * excess_east
                                - rho_prime_west * excess_west) / dx_u

    Adding this to the standard ``(p_east - p_west) / dx`` is
    mathematically equivalent to comparing ``p_eff = p - rho_prime * g
    * excess`` at the face-reference depth — eliminating the partial-
    cell-vs-full PGF cancellation error that drives spurious flow on
    realistic bathymetry.

    For full-cell columns where centroids align across cells, both
    excess values are zero and the correction is identically zero —
    so the legacy z\\* path is bit-exact unaffected.

    Output shape matches ``gradient_x_cgrid``: ``(n_lat, n_lon+1, nlev)``,
    with face j=n_lon wrapping around to face j=0.
    """
    R = grid.radius
    dlon = grid.dlon
    cos_lat = grid.cos_lat

    centroid_east = centroid_depth                     # (n_lat, n_lon, nlev)
    centroid_west = jnp.roll(centroid_depth, 1, axis=1)
    rho_prime_east = rho_prime
    rho_prime_west = jnp.roll(rho_prime, 1, axis=1)

    face_ref = jnp.minimum(centroid_east, centroid_west)
    excess_east = centroid_east - face_ref
    excess_west = centroid_west - face_ref

    # Per-face correction at faces 0..n_lon-1
    correction = -g * (
        rho_prime_east * excess_east
        - rho_prime_west * excess_west
    )

    # Wrap face j=n_lon to face j=0 (matches gradient_x_cgrid convention)
    correction_full = jnp.concatenate(
        [correction, correction[:, 0:1, :]], axis=1,
    )

    dx_u = R * dlon * cos_lat
    return correction_full / dx_u[:, jnp.newaxis, jnp.newaxis]


def partial_cell_pgf_correction_y(
    centroid_depth: jnp.ndarray,
    rho_prime: jnp.ndarray,
    grid: LatLonGrid,
    g: float,
) -> jnp.ndarray:
    """Adcroft & Campin (2004) face correction at v-faces (meridional).

    Same structure as ``partial_cell_pgf_correction_x`` but for the
    v-faces.  Wall BCs at poles → boundary v-faces have zero
    correction (consistent with v=0 there).

    Output shape: ``(n_lat+1, n_lon, nlev)``.
    """
    R = grid.radius
    dlat = grid.dlat
    dy_v = R * dlat

    # Interior v-faces: between cell i and cell i+1 in latitude
    centroid_north = centroid_depth[1:]                 # (n_lat-1, n_lon, nlev)
    centroid_south = centroid_depth[:-1]                # (n_lat-1, n_lon, nlev)
    rho_prime_north = rho_prime[1:]
    rho_prime_south = rho_prime[:-1]

    face_ref = jnp.minimum(centroid_north, centroid_south)
    excess_north = centroid_north - face_ref
    excess_south = centroid_south - face_ref

    correction_interior = -g * (
        rho_prime_north * excess_north
        - rho_prime_south * excess_south
    )

    # Pad pole faces with zero (wall BC: no v-flux through poles)
    pad_axes = ((0, 0),) * (correction_interior.ndim - 1)
    correction = jnp.pad(correction_interior, ((1, 1), *pad_axes))

    return correction / dy_v


# =============================================================================
# Density-Jacobian PGF (Shchepetkin & McWilliams 2003) — building blocks
# =============================================================================


def reconstruct_harmonic_slopes(
    rho_per_cell: jnp.ndarray,
    z_centroid: jnp.ndarray,
    is_active: jnp.ndarray,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """Per-cell harmonic-mean monotonized density slopes ``σ_k``.

    For Shchepetkin & McWilliams 2003 density-Jacobian PGF.  Within
    each cell ``k`` of a column we represent ``ρ(z) = ρ_k + σ_k · (z −
    z_centroid_k)``.  The slope ``σ_k`` is the harmonic mean of the
    one-sided slopes computed from the cell-centroid finite differences

        Δρ_top_k = (ρ_{k-1} − ρ_k) / (z_{k-1} − z_k)
        Δρ_bot_k = (ρ_k − ρ_{k+1}) / (z_k − z_{k+1})
        σ_k      = 2 · Δρ_top · Δρ_bot / (Δρ_top + Δρ_bot)

    monotonized to zero at extrema (signs differ).  Two key properties:

    1. For linear ρ(z), ``Δρ_top = Δρ_bot = a`` and ``σ_k = a`` exactly
       in every column, regardless of where the centroids sit.  This
       makes adjacent columns reconstruct ρ at intermediate depths
       identically — the property that lets the rest-state PGF vanish
       on partial cells with shifted centroids.
    2. At local extrema the limiter sets ``σ_k = 0`` (flat-top), so the
       reconstruction is monotone (no overshoots).

    Boundary handling:
    - Top cell (no neighbour above): ``σ_0 = Δρ_bot_0`` (one-sided).
    - Bottom-active cell (no active neighbour below — the partial
      seafloor): ``σ_{bot} = Δρ_top_{bot}`` (one-sided).
    - Inactive cells (below seafloor): ``σ = 0``.

    Parameters
    ----------
    rho_per_cell : array, shape (..., nlev)
        Cell-mean density [kg/m³] (often the baroclinic anomaly
        ``ρ'`` from ``iterate_eos_and_pressure_anomaly``).
    z_centroid : array, shape (..., nlev)
        Per-cell centroid depth [m], positive downward.
    is_active : array, shape (..., nlev)
        1.0 for wet cells, 0.0 below the partial seafloor.
    eps : float
        Safety floor for the harmonic-mean denominator.

    Returns
    -------
    sigma : array, shape (..., nlev)
        Per-cell density slope [kg/m⁴] (dρ/dz, positive z downward).

    References
    ----------
    Shchepetkin & McWilliams (2003), JGR Oceans 108(C9), §4.
    """
    rho = rho_per_cell
    z = z_centroid
    active_f = is_active.astype(rho.dtype)

    # Roll along the cell axis to get neighbour values.  Boundary slots
    # (k=0 above, k=nlev-1 below) are filled with the cell's own values
    # so that "Δρ" at the boundary safely evaluates to zero — the
    # boundary mask below selects the correct one-sided fall-back.
    rho_above = jnp.concatenate([rho[..., :1], rho[..., :-1]], axis=-1)
    rho_below = jnp.concatenate([rho[..., 1:], rho[..., -1:]], axis=-1)
    z_above = jnp.concatenate([z[..., :1], z[..., :-1]], axis=-1)
    z_below = jnp.concatenate([z[..., 1:], z[..., -1:]], axis=-1)

    # Has-active-neighbour masks.  The slot at k=0 has no upper
    # neighbour by construction; same for k=nlev-1 below.
    is_active_above = jnp.concatenate(
        [jnp.zeros_like(active_f[..., :1]), active_f[..., :-1]], axis=-1,
    )
    is_active_below = jnp.concatenate(
        [active_f[..., 1:], jnp.zeros_like(active_f[..., -1:])], axis=-1,
    )
    has_top = (active_f * is_active_above) > 0.5
    has_bot = (active_f * is_active_below) > 0.5

    # Safe-divide one-sided slopes.  When there is no active neighbour
    # the denominator can be zero; we substitute 1 to keep gradients
    # finite and zero out the result via ``jnp.where``.
    dz_top = z_above - z
    dz_bot = z - z_below
    safe_dz_top = jnp.where(has_top, dz_top, 1.0)
    safe_dz_bot = jnp.where(has_bot, dz_bot, 1.0)
    delta_top = jnp.where(has_top, (rho_above - rho) / safe_dz_top, 0.0)
    delta_bot = jnp.where(has_bot, (rho - rho_below) / safe_dz_bot, 0.0)

    # Harmonic mean of one-sided slopes (when both signs agree).
    sum_slopes = delta_top + delta_bot
    safe_sum = jnp.where(jnp.abs(sum_slopes) > eps, sum_slopes, eps)
    sigma_harm = 2.0 * delta_top * delta_bot / safe_sum
    same_sign = (delta_top * delta_bot) > 0.0
    sigma_interior = jnp.where(same_sign, sigma_harm, 0.0)

    sigma = jnp.where(
        has_top & has_bot, sigma_interior,
        jnp.where(has_top, delta_top,
                  jnp.where(has_bot, delta_bot, 0.0)),
    )
    return jnp.where(active_f > 0.5, sigma, 0.0)


def compute_pressure_at_target_smc03(
    rho_per_cell: jnp.ndarray,
    h_partial: jnp.ndarray,
    z_centroid: jnp.ndarray,
    sigma: jnp.ndarray,
    z_target: jnp.ndarray,
    g: float,
) -> jnp.ndarray:
    """Per-column pressure at arbitrary target depths, evaluated from
    the harmonic-slope piecewise-linear ρ(z) reconstruction.

    Sign convention: all depths are **positive downward** [m].

    Algorithm:

    1. Cell-top interface depths and pressures by cumulative sum:

       ``z_top_0   = 0,                  P_top_0   = 0``
       ``z_top_k   = z_top_{k-1} + h_{k-1}``
       ``P_top_k   = P_top_{k-1} + g · h_{k-1} · ρ_{k-1}``

       (Cell-mean integral of the linear deviation ``σ_k · (z' − z_c)``
       across a full cell vanishes because ``z_c`` is the geometric
       centroid — so ``P_top_{k+1} − P_top_k = g · h_k · ρ_k`` exactly.)

    2. For each target ``z_t`` find its enclosing cell ``k_t`` such
       that ``z_top_{k_t} ≤ z_t ≤ z_top_{k_t}+h_{k_t}``.  Within that
       cell the analytic linear-deviation integral gives

       ``P(z_t) = P_top_{k_t}
                  + g · (z_t − z_top_{k_t})
                      · [ρ_{k_t}
                         + 0.5 · σ_{k_t}
                              · (z_t + z_top_{k_t} − 2 · z_c_{k_t})]``

    Parameters
    ----------
    rho_per_cell : array, shape (..., nlev)
        Cell-mean density [kg/m³].
    h_partial : array, shape (..., nlev)
        Per-cell layer thickness [m].  Inactive cells (below the
        partial seafloor) have ``h = 0`` and contribute nothing to the
        integral.
    z_centroid : array, shape (..., nlev)
        Per-cell centroid depth [m, positive downward].  Inactive
        cells inherit the seafloor depth from above (``h = 0`` cells
        have ``z_centroid`` at the seafloor; inert).
    sigma : array, shape (..., nlev)
        Per-cell density slope [kg/m⁴] from
        ``reconstruct_harmonic_slopes``.  Inactive cells: 0.
    z_target : array, shape (..., n_targets)
        Target depths [m, positive downward].  Targets outside the
        column ``[0, sum h_partial]`` are clamped — the resulting
        pressure equals zero (above surface) or the seafloor pressure
        (below).  Phase 3 face-mask logic should keep that branch
        from materially affecting answers, but the clamp ensures
        finite output and stable AD.
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    P : array, shape (..., n_targets)
        Hydrostatic pressure [Pa] at each target depth.
    """
    # 1. Cell-top depths and pressures (cumulative).
    z_bot_per_cell = jnp.cumsum(h_partial, axis=-1)
    z_top_per_cell = z_bot_per_cell - h_partial

    cell_dP = g * h_partial * rho_per_cell
    P_bot_per_cell = jnp.cumsum(cell_dP, axis=-1)
    P_top_per_cell = P_bot_per_cell - cell_dP

    # 2. Clamp z_target to the column's valid range.  Targets above the
    # surface saturate to z=0 (P=0); targets below the column-bottom
    # saturate to the seafloor depth (P = column-integrated weight).
    z_seafloor = z_bot_per_cell[..., -1:]                  # (..., 1)
    z_t_clamped = jnp.clip(z_target, min=0.0, max=z_seafloor)

    # 3. Find enclosing cell per target via broadcasting + argmax.
    # in_cell[..., k, t] == True iff z_top_k <= z_t <= z_bot_k.
    z_top_e = z_top_per_cell[..., :, None]                 # (..., nlev, 1)
    z_bot_e = z_bot_per_cell[..., :, None]
    z_t_e = z_t_clamped[..., None, :]                       # (..., 1, n_t)
    in_cell = (z_t_e >= z_top_e) & (z_t_e <= z_bot_e)
    # First-True (argmax of int) handles interface ties deterministically:
    # a target sitting exactly at z_top_k matches both cell k-1 (its bottom)
    # and cell k (its top) — argmax picks k-1, which is a valid cell.
    k_t = jnp.argmax(in_cell.astype(jnp.int32), axis=-2)   # (..., n_t)

    # 4. Gather per-cell quantities at k_t and evaluate the in-cell integral.
    rho_kt = jnp.take_along_axis(rho_per_cell, k_t, axis=-1)
    sigma_kt = jnp.take_along_axis(sigma, k_t, axis=-1)
    z_top_kt = jnp.take_along_axis(z_top_per_cell, k_t, axis=-1)
    z_c_kt = jnp.take_along_axis(z_centroid, k_t, axis=-1)
    P_top_kt = jnp.take_along_axis(P_top_per_cell, k_t, axis=-1)

    dz = z_t_clamped - z_top_kt
    rho_eff = rho_kt + 0.5 * sigma_kt * (z_t_clamped + z_top_kt - 2.0 * z_c_kt)
    return P_top_kt + g * dz * rho_eff


def density_jacobian_pgf_smc03_x(
    rho_per_cell: jnp.ndarray,
    h_partial: jnp.ndarray,
    is_active: jnp.ndarray,
    grid: LatLonGrid,
    g: float,
) -> jnp.ndarray:
    """Density-Jacobian PGF at u-faces (S&M03 §4) — zonal direction.

    Replaces the cumsum ``p'`` + Adcroft-Campin face-correction stack
    with a per-column ``P(z)`` reconstruction from harmonic-mean
    monotonized slopes, evaluated at a face-reference depth and
    differenced horizontally.

    Algorithm (per u-face j between cell W=(j-1) mod n_lon and cell
    E=j; per level k):

    1. Per-column ``z_centroid`` = ``cumsum(h_partial) − 0.5 h``.
       (η=0 reference, consistent with the rest of the baroclinic
       path.)
    2. Per-column ``σ`` from ``reconstruct_harmonic_slopes``.
    3. **Face-adaptive z_target** = ``0.5 · (z_centroid_W + z_centroid_E)``
       (Option B from plan §2.3).  At full-cell faces this reduces to
       the standard reference-cell centroid (both centroids equal
       ``|z_full_ref[k]|``).  At partial-cell faces — where the
       column-independent ``|z_full_ref[k]|`` of Option A can fall
       below one column's seafloor when the partial cell sits in the
       upper half of the reference cell — the per-face midpoint of
       centroids is by construction inside both columns' partial
       cells.  This avoids the clamp pathology that drove the BH
       seamount blowup with Option A.
    4. ``P_at_target`` per column from
       ``compute_pressure_at_target_smc03`` (each column evaluated at
       the face-pair midpoint of *its* face).
    5. Horizontal Jacobian: ``∂P/∂x = (P_E − P_W) / dx_u``, periodic
       in longitude.

    Output shape matches ``gradient_x_cgrid``: ``(n_lat, n_lon+1,
    nlev)``, with face j=n_lon wrapping to face j=0.

    For face-levels at which one column is inactive (``h = 0`` past
    its seafloor), the face mask in the integrating PE step gates
    the result downstream.
    """
    # Per-column geometry and slopes.
    z_centroid = jnp.cumsum(h_partial, axis=-1) - 0.5 * h_partial
    sigma = reconstruct_harmonic_slopes(rho_per_cell, z_centroid, is_active)

    # West-neighbour rolls (column j-1 at u-face j).
    rho_W = jnp.roll(rho_per_cell, 1, axis=1)
    h_W = jnp.roll(h_partial, 1, axis=1)
    z_c_W = jnp.roll(z_centroid, 1, axis=1)
    sigma_W = jnp.roll(sigma, 1, axis=1)

    # Face-adaptive target depth: midpoint of W and E centroids.
    # Shape (n_lat, n_lon, nlev) — value at index j is the target for
    # u-face j.
    z_target_face = 0.5 * (z_c_W + z_centroid)

    # Per-face-pair pressures evaluated at the SAME z_target.
    P_E = compute_pressure_at_target_smc03(
        rho_per_cell, h_partial, z_centroid, sigma, z_target_face, g,
    )
    P_W = compute_pressure_at_target_smc03(
        rho_W, h_W, z_c_W, sigma_W, z_target_face, g,
    )
    diff_interior = P_E - P_W
    diff = jnp.concatenate([diff_interior, diff_interior[:, 0:1, :]], axis=1)

    R = grid.radius
    dlon = grid.dlon
    cos_lat = grid.cos_lat
    dx_u = R * dlon * cos_lat                           # (n_lat,)
    return diff / dx_u[:, jnp.newaxis, jnp.newaxis]


def density_jacobian_pgf_smc03_y(
    rho_per_cell: jnp.ndarray,
    h_partial: jnp.ndarray,
    is_active: jnp.ndarray,
    grid: LatLonGrid,
    g: float,
) -> jnp.ndarray:
    """Density-Jacobian PGF at v-faces (S&M03 §4) — meridional direction.

    Same machinery as ``density_jacobian_pgf_smc03_x``; v-face i is
    between cell S=(i−1) and cell N=i; pole faces (i=0, i=n_lat) are
    walls and pad with zero (consistent with v=0 at the wall).
    Uses the same face-adaptive midpoint-of-centroids target depth
    (Option B from plan §2.3).

    Output shape: ``(n_lat+1, n_lon, nlev)``.
    """
    z_centroid = jnp.cumsum(h_partial, axis=-1) - 0.5 * h_partial
    sigma = reconstruct_harmonic_slopes(rho_per_cell, z_centroid, is_active)

    # North-direction interior pairs (i and i-1).
    rho_N = rho_per_cell[1:]
    rho_S = rho_per_cell[:-1]
    h_N = h_partial[1:]
    h_S = h_partial[:-1]
    z_c_N = z_centroid[1:]
    z_c_S = z_centroid[:-1]
    sigma_N = sigma[1:]
    sigma_S = sigma[:-1]

    z_target_face_int = 0.5 * (z_c_S + z_c_N)
    P_N = compute_pressure_at_target_smc03(
        rho_N, h_N, z_c_N, sigma_N, z_target_face_int, g,
    )
    P_S = compute_pressure_at_target_smc03(
        rho_S, h_S, z_c_S, sigma_S, z_target_face_int, g,
    )
    diff_interior = P_N - P_S

    pad_axes = ((0, 0),) * (diff_interior.ndim - 1)
    diff = jnp.pad(diff_interior, ((1, 1), *pad_axes))

    R = grid.radius
    dlat = grid.dlat
    dy_v = R * dlat
    return diff / dy_v


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

    # v-face i is between cell i and cell i+1.
    # Pole boundaries: v=0 (always masked).  Single Pad HLO op replaces
    # alloc-zeros + concatenate-of-three.
    v_mask_interior = land_mask[:-1] * land_mask[1:]
    v_mask = jnp.pad(v_mask_interior, ((1, 1), (0, 0)))

    return u_mask, v_mask
