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


def _cos_lat_uv(grid: LatLonGrid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """cos(lat) on u- and v-face latitudes for a lat-lon C-grid.

    u-face points sit at cell-centre latitudes (where ``grid.cos_lat``
    is defined directly).  v-face points sit at latitude interfaces
    between cells and are obtained by linear interpolation of
    ``cos_lat``, with the south/north boundary v-faces clamped to the
    nearest cell-centre value.

    Parameters
    ----------
    grid : LatLonGrid

    Returns
    -------
    cos_u : (n_lat,)
        cos(lat) at u-face latitudes.
    cos_v : (n_lat+1,)
        cos(lat) at v-face latitudes.
    """
    cos_u = grid.cos_lat                                  # (n_lat,)
    cos_v_interior = 0.5 * (cos_u[:-1] + cos_u[1:])       # (n_lat-1,)
    cos_v = jnp.concatenate([
        cos_u[:1],                                        # south boundary ≈ cos(lat[0])
        cos_v_interior,
        cos_u[-1:],                                       # north boundary ≈ cos(lat[-1])
    ])                                                    # (n_lat+1,)
    return cos_u, cos_v


def laplacian_scaling_factor(grid: LatLonGrid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Grid-dependent scaling for Laplacian viscosity on a lat-lon grid.

    On a latitude-longitude grid the zonal grid spacing shrinks as
    ``cos(lat)`` near the poles.  Because the Laplacian-viscosity
    timescale ``dx^2 / A_h`` shrinks with ``cos^2(lat)``, a constant
    ``A_h`` becomes effectively very large near the poles — and at the
    high-latitude coastal partial-cell vertices on real ETOPO this
    triggers a viscous-Coriolis amplification that produces a localized
    runaway in η at lat ~82.5° (see ``docs/ocean_experiments/realistic_geometry_phase4_results.md``
    and the D1 diagnostic in ``ah_diagnostics/``).

    Following the standard MITgcm/MOM6/NEMO production convention, the
    Laplacian coefficient is multiplied by ``cos^2(lat)``.  This keeps
    the viscous CFL number ``A_h * dt / dx^2`` latitude-independent and
    matches the cos⁴-scaling that ``biharmonic_scaling_factor`` already
    applies to ``B_h``.

    Usage::

        scale_u, scale_v = laplacian_scaling_factor(grid)
        vlap_u, vlap_v = vector_laplacian_cgrid(u, v, grid, ...)
        du_dt += A_h * scale_u[:, None, None] * vlap_u
        dv_dt += A_h * scale_v[:, None, None] * vlap_v

    Parameters
    ----------
    grid : LatLonGrid

    Returns
    -------
    scale_u : (n_lat,)
        cos²(lat) at u-face latitudes (cell centres).  Reshape to
        ``[:, None]`` for 2D fields or ``[:, None, None]`` for 3D.
    scale_v : (n_lat+1,)
        cos²(lat) at v-face latitudes.
    """
    cos_u, cos_v = _cos_lat_uv(grid)
    return cos_u ** 2, cos_v ** 2


def equatorial_boost_factor(
    grid: LatLonGrid, sigma_deg: float, boost: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Latitude-dependent equatorial enhancement factor for A_h.

    Returns a multiplier ``1 + (boost - 1) * exp(-(lat / sigma_deg)^2)``
    evaluated at u- and v-face latitudes.  Used to boost horizontal
    Laplacian viscosity within ±sigma_deg of the equator, where the
    Coriolis parameter f→0 leaves no rotational stiffness to constrain
    the ocean's response to wind stress.  At coarse resolution (~1°)
    without this boost, the equatorial currents and upwelling become
    unconstrained and produce a runaway cold tongue.  Production
    OGCMs (MOM6 OM4 ``KH_VEL_LAT_RES``, NEMO meridional ``rn_ahm0``
    profiles, POP anisotropic viscosity) all enhance equatorial
    momentum dissipation in some form.

    The factor is applied multiplicatively on top of the cos²(lat)
    CFL scaling from ``laplacian_scaling_factor``.

    Parameters
    ----------
    grid : LatLonGrid
    sigma_deg : float
        Gaussian half-width in degrees.  Typical values 3-7°.
    boost : float
        Multiplier at the exact equator (lat=0).  Values >= 1.0;
        boost = 1.0 disables the enhancement.  Typical values 3-10.

    Returns
    -------
    boost_u : (n_lat,)
        Boost factor at u-face (cell-centre) latitudes.
    boost_v : (n_lat+1,)
        Boost factor at v-face latitudes.
    """
    if boost <= 1.0:
        n_lat = grid.lat.shape[0]
        ones_u = jnp.ones(n_lat, dtype=grid.lat.dtype)
        ones_v = jnp.ones(n_lat + 1, dtype=grid.lat.dtype)
        return ones_u, ones_v

    sigma_rad = jnp.deg2rad(sigma_deg)
    lat_u = grid.lat                                       # (n_lat,)
    lat_v_interior = 0.5 * (lat_u[:-1] + lat_u[1:])         # (n_lat-1,)
    lat_v = jnp.concatenate([lat_u[:1], lat_v_interior, lat_u[-1:]])

    boost_u = 1.0 + (boost - 1.0) * jnp.exp(-(lat_u / sigma_rad) ** 2)
    boost_v = 1.0 + (boost - 1.0) * jnp.exp(-(lat_v / sigma_rad) ** 2)
    return boost_u, boost_v


def biharmonic_scaling_factor(grid: LatLonGrid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Grid-dependent scaling for biharmonic viscosity on a lat-lon grid.

    On a latitude-longitude grid the zonal grid spacing shrinks as
    ``cos(lat)`` near the poles.  Because the biharmonic CFL scales
    as ``B_h * dt / dx^4``, a constant ``B_h`` violates CFL near the
    poles while under-diffusing at the equator.

    Following the MOM6 convention (Griffies & Hallberg 2000), the
    biharmonic coefficient should be multiplied by
    ``(dx_local / dx_ref)^4`` where ``dx_ref`` is the reference
    (equatorial / maximum) spacing.  Here ``dx_ref`` corresponds to
    ``cos_lat = 1`` so the scaling reduces to ``cos^4(lat)``.

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
        cos⁴(lat) at u-face latitudes.  Callers should reshape to
        ``[:, None]`` for 2D fields or ``[:, None, None]`` for 3D.
    scale_v : (n_lat+1,)
        cos⁴(lat) at v-face latitudes.
    """
    cos_u, cos_v = _cos_lat_uv(grid)
    return cos_u ** 4, cos_v ** 4


def slope_foot_enhancement_3d(
    H_bathy: jnp.ndarray,
    mask: jnp.ndarray,
    grid: LatLonGrid,
    n_levels_from_bottom: int = 5,
    alpha: float = 3.0,
    threshold: float = 0.1,
    is_active: jnp.ndarray = None,
    nlev: int = None,
) -> jnp.ndarray:
    """Slope-foot viscosity enhancement factor (MOM6 OM4 KH_BG_2D analog).

    Returns a 3D multiplicative factor (≥1) that boosts viscosity in the
    bottom ``n_levels_from_bottom`` levels over steep bathymetric slopes:

        E(j,i,k) = 1 + α · tanh(|∇H|/H / δ) · vertical_taper(k)

    where ``vertical_taper(k) = 1`` for the bottom-N active levels per
    column, 0 elsewhere. The factor saturates at ``1 + α`` over very
    steep slopes (e.g., the African shelf, Indonesian Throughflow).

    MOM6 OM4 standard: ``α = 3``, ``δ = 0.1``, N = 5.

    Parameters
    ----------
    H_bathy : (n_lat, n_lon) array — column depths [m]
    mask : (n_lat, n_lon) — ocean mask (1=ocean, 0=land)
    grid : LatLonGrid
    is_active : (n_lat, n_lon, nlev) bool, optional — partial-cell per-level
        active mask. When None, treats all levels as active.
    nlev : int, required when is_active is None.

    Returns
    -------
    E_3d : (n_lat, n_lon, nlev) — multiplicative factor, ≥1.
    """
    R = getattr(grid, "radius", 6.371e6)
    n_lat, n_lon = H_bathy.shape

    # ∇H at cell centres via centred differences (periodic in lon, walls in lat)
    cos_lat = jnp.cos(grid.lat * (jnp.pi / 180.0))
    cos_lat = jnp.maximum(cos_lat, 1e-3)
    dlat = jnp.pi / n_lat
    dlon = 2.0 * jnp.pi / n_lon
    dy = R * dlat
    dx = R * cos_lat[:, None] * dlon

    H_e = jnp.roll(H_bathy, -1, axis=1)
    H_w = jnp.roll(H_bathy, 1, axis=1)
    dHdx = (H_e - H_w) / (2.0 * dx)

    # No wrap in lat — use one-sided differences at boundaries
    H_n = jnp.concatenate([H_bathy[1:, :], H_bathy[-1:, :]], axis=0)
    H_s = jnp.concatenate([H_bathy[:1, :], H_bathy[:-1, :]], axis=0)
    dHdy = (H_n - H_s) / (2.0 * dy)

    grad_H_mag = jnp.sqrt(dHdx ** 2 + dHdy ** 2)
    H_safe = jnp.maximum(H_bathy, 1.0)
    slope_metric = grad_H_mag / (H_safe * threshold)
    enhancement_2d = alpha * jnp.tanh(slope_metric) * mask  # (n_lat, n_lon)

    # Vertical taper: bottom N active levels
    if is_active is not None:
        nlev_local = is_active.shape[-1]
        # Per-column deepest active level: count active levels - 1
        n_active = jnp.sum(is_active.astype(jnp.int32), axis=-1)  # (n_lat, n_lon)
        k_bottom = n_active - 1                                    # (n_lat, n_lon)
        k_idx = jnp.arange(nlev_local)[None, None, :]              # (1,1,nlev)
        in_band = (k_idx >= (k_bottom[..., None] - n_levels_from_bottom + 1)) & \
                  (k_idx <= k_bottom[..., None])
        vertical_taper = (in_band & is_active).astype(H_bathy.dtype)
    else:
        if nlev is None:
            raise ValueError("nlev required when is_active is None")
        v = jnp.zeros((nlev,), dtype=H_bathy.dtype)
        v = v.at[nlev - n_levels_from_bottom:].set(1.0)
        vertical_taper = jnp.broadcast_to(
            v[None, None, :], (n_lat, n_lon, nlev)
        )

    return 1.0 + enhancement_2d[..., None] * vertical_taper


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


def neumann_fill_vertex(
    f: jnp.ndarray,
    vtx_mask: jnp.ndarray,
    n_passes: int = 3,
) -> jnp.ndarray:
    """Fill land vertices with nearest ocean-neighbour (Neumann BC).

    Vertex-level analog of ``_neumann_fill_cgrid`` for the
    ``(n_lat+1, n_lon+1)`` vertex grid.  Longitude is periodic
    (column ``n_lon`` duplicates column 0); rows 0 and ``n_lat`` are
    pole vertices with Neumann padding in the meridional direction.

    Public helper because it is reused by:
      - the WENO branch of ``ocean_pe_latlon_cgrid`` (smooth q before
        the smoothness-detector reconstruction)
      - the AL81 ``pv_flux_al81_partial_cell`` helper (smooth q before
        the 12-point triad stencil)

    Parameters
    ----------
    f : (n_lat+1, n_lon+1, nlev) or (n_lat+1, n_lon+1)
        Vertex field to fill.
    vtx_mask : (n_lat+1, n_lon+1)
        1 = ocean vertex, 0 = land vertex.
    n_passes : int
        Number of fill passes.  3 is sufficient to cover the typical
        coastal triad stencil.

    Returns
    -------
    filled : same shape as ``f``
        ``f`` at ocean vertices; nearest-neighbour-averaged value
        at land vertices that have at least one wet neighbour after
        ``n_passes`` iterations; original value (typically zero) at
        fully-isolated land vertices.
    """
    m = vtx_mask
    filled = f

    for _ in range(n_passes):
        # N/S neighbours: Neumann padding at rows 0 and n_lat.
        f_s = jnp.concatenate([filled[0:1], filled[:-1]], axis=0)
        m_s = jnp.concatenate([m[0:1], m[:-1]], axis=0)
        f_n = jnp.concatenate([filled[1:], filled[-1:]], axis=0)
        m_n = jnp.concatenate([m[1:], m[-1:]], axis=0)

        # E/W neighbours: periodic on core columns 0..n_lon-1, then wrap.
        # Column n_lon duplicates column 0, so rolling the full array
        # along axis 1 is correct for the core columns and the wrap
        # column picks up the right neighbour automatically.
        f_w = jnp.roll(filled, 1, axis=1)
        m_w = jnp.roll(m, 1, axis=1)
        f_e = jnp.roll(filled, -1, axis=1)
        m_e = jnp.roll(m, -1, axis=1)

        is_land = m < 0.5

        if f.ndim > 2:
            m_s_e = m_s[..., jnp.newaxis]
            m_n_e = m_n[..., jnp.newaxis]
            m_w_e = m_w[..., jnp.newaxis]
            m_e_e = m_e[..., jnp.newaxis]
            is_land_e = is_land[..., jnp.newaxis]
        else:
            m_s_e = m_s
            m_n_e = m_n
            m_w_e = m_w
            m_e_e = m_e
            is_land_e = is_land

        nbr_sum = f_s * m_s_e + f_n * m_n_e + f_w * m_w_e + f_e * m_e_e
        nbr_count = m_s_e + m_n_e + m_w_e + m_e_e
        nbr_avg = nbr_sum / jnp.maximum(nbr_count, 1.0)

        has_any_nbr = (m_s + m_n + m_w + m_e) > 0.0
        if f.ndim > 2:
            has_any_nbr_e = has_any_nbr[..., jnp.newaxis]
        else:
            has_any_nbr_e = has_any_nbr

        filled = jnp.where(is_land_e & has_any_nbr_e, nbr_avg, filled)
        m = jnp.where(is_land & has_any_nbr, 1.0, m)

    # Re-sync periodic wrap column.
    filled = filled.at[:, -1].set(filled[:, 0])
    return filled


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

    # Face-adaptive target depth: the *shallower* of the two centroids
    # (Adcroft & Campin 2004 face_ref convention; see
    # ``partial_cell_pgf_correction_x`` for the matching choice in
    # the legacy path).  Using ``min`` rather than ``mean`` is
    # essential: at a face between a full-cell column and a partial-
    # bottom column with ``h_partial / dz_ref < 1/3``, the midpoint
    # ``0.5·(z_c_W + z_c_E)`` falls *below* the partial column's
    # seafloor, ``compute_pressure_at_target_smc03`` clamps that
    # column to its seafloor pressure while the deeper column
    # evaluates in-cell, and the asymmetric clamp leaves a residual
    # ``ρ·g·(dz_ref − 3·h_partial)/4`` per face that does not vanish
    # for any ρ — drove the 525 mm/s BH steady state in an earlier
    # iteration.  The shallower centroid is by construction inside
    # both columns (the partial column's centroid sits inside its
    # own partial cell, and a deeper column's full or partial cell
    # at the same level extends at least to that depth).  Reduces
    # to the standard centroid on full-cell faces.
    z_target_face = jnp.minimum(z_c_W, z_centroid)

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

    # Shallower-of-centroids (Adcroft & Campin convention; see x-direction
    # operator for the rationale and the C1 bug it resolves).
    z_target_face_int = jnp.minimum(z_c_S, z_c_N)
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


def pv_flux_al81_partial_cell(
    zeta: jnp.ndarray,
    h_vtx: jnp.ndarray,
    h_v: jnp.ndarray,
    v: jnp.ndarray,
    h_u: jnp.ndarray,
    u: jnp.ndarray,
    u_mask_3d: jnp.ndarray,
    v_mask_3d: jnp.ndarray,
    vtx_mask: jnp.ndarray,
    eps_h: float = 1.0e-10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Arakawa-Lamb 1981 (AL81) energy-and-enstrophy-conserving PV flux.

    Production-grade vector-invariant Coriolis advection on the
    Arakawa C-grid for z-coordinate models with partial cells (NEMO
    ``ln_zps`` regime).  Implements the 12-point triad (4-corner ⊗
    3-vertex) stencil of Arakawa & Lamb (1981) — equivalent to NEMO
    ``dyn_vor_een`` (Le Sommer et al. 2009) and to the Hamiltonian
    discretisation of Salmon (2004) / Stewart & Dellar (2016) when
    the AL81 coefficient set
    {α, β, γ ∈ Appendix A, Stewart-Dellar 2016} is used.

    The simple 2-point Sadourny enstrophy form
    ``q_at_u = ½(q_S + q_N)`` is unstable on real partial-cell
    bathymetry (live-T ETOPO 30-day NaN by day 19).  The AL81 form
    suppresses the grid-scale q-noise mode at step vertices because
    the 12-point triad averages q over **9 neighbouring vertices**
    (not 2) at every face, with weights chosen so that **discrete
    energy AND discrete potential enstrophy are conserved
    simultaneously** in the inviscid, flat-bottom limit.

    On partial cells, ``h_vtx`` (the F-point thickness, MITgcm
    ``hFacZ`` / NEMO ``e3f``) absorbs the geometric dependence: at a
    step vertex with one tall and three short surrounding cells,
    ``h_vtx = min`` is small, so ``q = ζ/h_vtx`` is large there.
    The triad's 1/12 weighting on each q value, combined with mass
    fluxes ``h·v`` and ``h·u`` that vanish at closed faces, gives a
    PV flux that is bounded and consistent with the same
    ``min(h_W, h_E)`` face-thickness convention used in continuity
    (Adcroft, Hill & Marshall 1997 eq. 11; Pacanowski & Gnanadesikan
    1998 §3).

    Index conventions (same as the rest of latlon_cgrid_operators)
    --------------------------------------------------------------
    - cell-centre  ``(j, i)``,           shape ``(n_lat, n_lon, nlev)``
    - u-face       ``u[j, i]``  =  west face of cell ``(j, i)``,
                                  shape ``(n_lat, n_lon+1, nlev)``,
                                  periodic wrap ``u[:, n_lon] = u[:, 0]``.
    - v-face       ``v[j, i]``  =  south face of cell ``(j, i)``,
                                  shape ``(n_lat+1, n_lon, nlev)``,
                                  with pole walls at ``j=0, n_lat``.
    - vertex       ``q[j, i]``  =  SW corner of cell ``(j, i)``,
                                  shape ``(n_lat+1, n_lon+1, nlev)``,
                                  periodic wrap.

    AL81 stencil
    ------------
    For each cell ``(j, i)`` define **four corner triads**, each the
    1/12-weighted sum of the three vertices nearest the named corner
    of that cell (the L-shape of corners excluding the diagonal):

        SW corner triad : (q_SW, q_SE, q_NW)  / 12
        SE corner triad : (q_SE, q_SW, q_NE)  / 12
        NW corner triad : (q_NW, q_SW, q_NE)  / 12
        NE corner triad : (q_NE, q_SE, q_NW)  / 12

    where (using the array convention above) the four corners of
    cell ``(j, i)`` are::

        q_SW = q[j  , i  ]    q_SE = q[j  , i+1]
        q_NW = q[j+1, i  ]    q_NE = q[j+1, i+1]

    For u-face ``u[j, i]`` (between west cell ``(j, i-1)`` and east
    cell ``(j, i)``), the AL81 PV-flux contribution is::

        +F_u[j,i] = + SE_triad(west_cell) * V[j+1, i-1]
                    + SW_triad(east_cell) * V[j+1, i  ]
                    + NE_triad(west_cell) * V[j  , i-1]
                    + NW_triad(east_cell) * V[j  , i  ]

    where ``V = h_v · v`` is the meridional mass flux at v-faces.

    For v-face ``v[j, i]`` (between south cell ``(j-1, i)`` and
    north cell ``(j, i)``)::

        -F_v[j,i] = + NW_triad(south_cell) * U[j-1, i+1]
                    + NE_triad(south_cell) * U[j-1, i  ]
                    + SW_triad(north_cell) * U[j  , i+1]
                    + SE_triad(north_cell) * U[j  , i  ]

    where ``U = h_u · u`` is the zonal mass flux at u-faces.

    On a uniform-h, fully-wet grid this stencil reduces to a 9-point
    average of q (the symmetric AL81 "energy-enstrophy compromise"),
    not the 2-point Sadourny form.  In the smooth limit the truncation
    error is the same O(d²) as Sadourny but the leading-order
    grid-scale dispersion is much smaller — that is the property that
    suppresses the partial-cell q-noise mode.

    Land treatment
    --------------
    PV ``q = ζ/h_vtx`` is evaluated AFTER ``h_vtx`` has the active-cell
    masking applied (``BIG_H`` on dry sides; min over wet cells gives
    the true F-point wet thickness — MITgcm ``hFacZ``).  Where the
    vertex itself is fully dry (all 4 surrounding cells inactive),
    ``h_vtx → BIG_H`` makes ``q → 0`` — and the surrounding mass
    fluxes ``V = h_v·v·v_mask`` and ``U = h_u·u·u_mask`` also vanish
    at the closed faces, so triad contributions through dry vertices
    are exactly zero (no spurious flow at the coast).

    A Neumann fill of ``q`` at land-adjacent vertices (where
    ``vtx_mask == 0`` but at least one neighbour is wet) replaces the
    masked-zero value with the average of wet neighbours.  This
    avoids a discontinuity in q at the coast that would otherwise
    drive a spurious PV gradient even when the mass flux is zero
    (the discontinuity does not affect the dynamics through ``q·F``,
    but it pollutes the coupling to neighbouring faces through the
    triad's 9-vertex stencil).  Same Neumann fill helper as is used
    by the WENO branch.

    Parameters
    ----------
    zeta : (n_lat+1, n_lon+1, nlev)
        Relative vorticity at vertices.
    h_vtx : (n_lat+1, n_lon+1, nlev)
        F-point layer thickness (MITgcm hFacZ-like, min over active
        cells with a ``BIG_H`` sentinel for fully-dry vertices).
    h_v, v : (n_lat+1, n_lon, nlev)
        Layer thickness and meridional velocity at v-faces.
    h_u, u : (n_lat, n_lon+1, nlev)
        Layer thickness and zonal velocity at u-faces.
    u_mask_3d : (n_lat, n_lon+1, nlev) or broadcastable
        u-face active mask (1 at wet faces, 0 at closed/dry faces).
    v_mask_3d : (n_lat+1, n_lon, nlev) or broadcastable
        v-face active mask.
    vtx_mask : (n_lat+1, n_lon+1)
        Vertex mask (1 if all 4 surrounding cells are wet, 0 otherwise);
        used to drive the Neumann fill of q.
    eps_h : float
        Floor on ``h_vtx`` to avoid divide-by-zero at fully-dry verts.
        Already partially handled by ``BIG_H`` sentinel; this is a
        belt-and-braces guard.

    Returns
    -------
    diag_vortcor_u : (n_lat, n_lon+1, nlev)
        ``+ q · F_v`` contribution to ``du/dt`` at u-faces.
    diag_vortcor_v : (n_lat+1, n_lon, nlev)
        ``- q · F_u`` contribution to ``dv/dt`` at v-faces.

    References
    ----------
    - Arakawa, A. and Lamb, V.R. (1981): A potential-enstrophy and
      energy-conserving scheme for the shallow-water equations.
      Mon. Wea. Rev. 109, 18-36.
    - Salmon, R. (2004): Poisson-bracket approach to the construction
      of energy- and potential-enstrophy-conserving algorithms for
      the shallow-water equations.  J. Atmos. Sci. 61, 2016-2036.
    - Stewart, A.L. and Dellar, P.J. (2016): An energy- and
      potential-enstrophy-conserving numerical scheme for the
      multilayer shallow-water equations with the complete Coriolis
      force.  J. Comput. Phys. 313, 99-120.  (Appendix A: AL81
      coefficient set.)
    - Le Sommer, J., Penduff, T., Theetten, S., Madec, G., Barnier, B.
      (2009): How momentum advection schemes influence
      current-topography interactions at eddy-permitting resolution.
      Ocean Modelling 29, 1-14.  (NEMO ``dyn_vor_een``;
      recommendation for ``ln_zps``.)
    - Adcroft, A. and Hallberg, R. (2006): On methods for solving the
      oceanic equations of motion in generalized vertical
      coordinates.  Ocean Modelling 11, 224-233.  (PV consistency on
      partial cells.)
    - Pacanowski, R.C. and Gnanadesikan, A. (1998): Transient response
      in a z-level ocean model that resolves topography with
      partial cells.  Mon. Wea. Rev. 126, 3248-3270.  (min-rule for
      vertex thickness.)
    """
    # --- 1. PV at vertices, ``q = ζ / h_vtx`` ----------------------
    # ``h_vtx`` already carries the BIG_H sentinel at fully-dry
    # vertices (set by the caller) so q ≈ 0 there; eps_h is a guard
    # against floating-point edge cases.
    q = zeta / jnp.maximum(h_vtx, eps_h)

    # Neumann-fill q at land-adjacent vertices so the triad sees a
    # smooth field across coastlines.  The fill is idempotent at
    # interior wet vertices (vtx_mask == 1).  Keeps q in the same
    # 4D shape ``(n_lat+1, n_lon+1, nlev)`` as zeta.
    q = neumann_fill_vertex(q, vtx_mask)

    # --- 2. Mass fluxes at u/v faces -------------------------------
    # ``F_u = h·u`` at u-faces, ``F_v = h·v`` at v-faces.  Multiply
    # by the per-level face mask so closed/dry faces contribute
    # exactly zero — required for q·F to vanish at the coast.
    F_u = h_u * u * u_mask_3d            # (n_lat, n_lon+1, nlev)
    F_v = h_v * v * v_mask_3d            # (n_lat+1, n_lon, nlev)

    # --- 3. Corner triads at every cell ----------------------------
    # Each triad lives at a corner of a cell.  We index triads by the
    # cell ``(j, i)`` they belong to, with shape ``(n_lat, n_lon,
    # nlev)`` and the named corner indicating which 3 of the cell's
    # 4 corner-q values are summed.
    #
    # Cell (j, i) has corners (using array indexing on q[j', i']):
    #   q_SW = q[j  , i  ]    q_SE = q[j  , i+1]
    #   q_NW = q[j+1, i  ]    q_NE = q[j+1, i+1]
    #
    # We need q_SW, q_SE, q_NW, q_NE as ``(n_lat, n_lon, nlev)``
    # arrays.  Because ``q`` has shape ``(n_lat+1, n_lon+1, nlev)``
    # with periodic wrap on the longitude axis (column n_lon == col
    # 0), simple slicing extracts each corner.
    q_SW = q[:-1, :-1, :]                # (n_lat, n_lon, nlev)
    q_SE = q[:-1, 1:, :]
    q_NW = q[1:, :-1, :]
    q_NE = q[1:, 1:, :]

    inv12 = 1.0 / 12.0
    # 4 triads per cell (1/12-weighted sum of 3 corner-q values, the
    # 3 q's nearest the named corner).
    t_SW = inv12 * (q_SW + q_SE + q_NW)
    t_SE = inv12 * (q_SE + q_SW + q_NE)
    t_NW = inv12 * (q_NW + q_SW + q_NE)
    t_NE = inv12 * (q_NE + q_SE + q_NW)

    # --- 4. AL81 PV flux at u-faces --------------------------------
    # u-face u[j, i] is between west cell (j, i-1) and east cell
    # (j, i).  AL81 form (NEMO dyn_vor_een, translated to our index
    # convention):
    #   +F_pv_u[j, i] = + t_SE(west_cell)  * F_v[j+1, i-1]
    #                   + t_SW(east_cell)  * F_v[j+1, i  ]
    #                   + t_NE(west_cell)  * F_v[j  , i-1]
    #                   + t_NW(east_cell)  * F_v[j  , i  ]
    #
    # We need the west-cell triads (cell at (j, i-1)) at u-face index
    # i; this is ``t_*`` rolled +1 in axis 1.  East-cell triads at
    # u-face index i are ``t_*`` itself, but we need to extend along
    # axis 1 from n_lon → n_lon+1 to match u-face shape (the periodic
    # wrap face).  We use ``jnp.concatenate`` with the col-0 wrap.
    #
    # F_v is (n_lat+1, n_lon, nlev); we need F_v at v-face indices
    # (j, i-1), (j, i), (j+1, i-1), (j+1, i).  For u-face (j, i)
    # with i ∈ [0, n_lon], periodic in i.

    # Roll periodic in axis 1 to get west-cell triads aligned with
    # u-face index.  After rolling +1, position i holds cell index
    # (i-1) mod n_lon, which is the west cell of u-face i.
    t_SE_W = jnp.roll(t_SE, 1, axis=1)   # west-cell SE at u-face i
    t_NE_W = jnp.roll(t_NE, 1, axis=1)
    # East-cell triads are at u-face i = cell i.  Also wrap the
    # n_lon-th u-face to col 0 (periodic).
    # t_SW, t_NW have shape (n_lat, n_lon, nlev); pad axis 1 by 1 on
    # the right with the col-0 value to match u-face shape.
    t_SW_E = jnp.concatenate([t_SW, t_SW[:, 0:1, :]], axis=1)
    t_NW_E = jnp.concatenate([t_NW, t_NW[:, 0:1, :]], axis=1)
    # West-cell triads also need the periodic wrap column
    t_SE_W = jnp.concatenate([t_SE_W, t_SE_W[:, 0:1, :]], axis=1)
    t_NE_W = jnp.concatenate([t_NE_W, t_NE_W[:, 0:1, :]], axis=1)

    # F_v at the four offsets, mapped to u-face index.  At u-face
    # (j, i), we need:
    #   F_v_S_W = F_v[j  , i-1, :]   (south-west of u-face)
    #   F_v_S_E = F_v[j  , i  , :]
    #   F_v_N_W = F_v[j+1, i-1, :]
    #   F_v_N_E = F_v[j+1, i  , :]
    # F_v has shape (n_lat+1, n_lon, nlev); the south face of the
    # u-face row j is F_v[j, :, :], the north face is F_v[j+1, :, :].
    F_v_south = F_v[:-1, :, :]           # (n_lat, n_lon, nlev) — south of each u-row
    F_v_north = F_v[1:, :, :]            # (n_lat, n_lon, nlev)
    # West/east neighbour in i, periodic, plus wrap to (n_lat, n_lon+1, nlev).
    F_v_S_E = jnp.concatenate([F_v_south, F_v_south[:, 0:1, :]], axis=1)
    F_v_N_E = jnp.concatenate([F_v_north, F_v_north[:, 0:1, :]], axis=1)
    F_v_S_W = jnp.roll(F_v_S_E, 1, axis=1)
    F_v_N_W = jnp.roll(F_v_N_E, 1, axis=1)

    # AL81 contribution at u-faces.
    diag_vortcor_u = (
        t_SE_W * F_v_N_W       # west-cell SE × NW V
        + t_SW_E * F_v_N_E     # east-cell SW × NE V
        + t_NE_W * F_v_S_W     # west-cell NE × SW V
        + t_NW_E * F_v_S_E     # east-cell NW × SE V
    )

    # --- 5. AL81 PV flux at v-faces --------------------------------
    # v-face v[j, i] is between south cell (j-1, i) and north cell
    # (j, i).  AL81 form:
    #   -F_pv_v[j, i] = + t_NW(south_cell) * F_u[j-1, i+1]
    #                   + t_NE(south_cell) * F_u[j-1, i  ]
    #                   + t_SW(north_cell) * F_u[j  , i+1]
    #                   + t_SE(north_cell) * F_u[j  , i  ]
    # The v-tendency is the negative of this (since q × u with the
    # cross-product sign convention is q × F_u for v).
    #
    # South-cell triads at v-face j are ``t_*`` shifted +1 in axis 0
    # (i.e., t_*[j-1, i] = south-cell of v-face j).  North-cell
    # triads at v-face j are ``t_*`` itself.  v-face has shape
    # (n_lat+1, n_lon, nlev); pole faces (j=0, n_lat) are walls
    # → set the contribution to zero by zero-padding in axis 0.
    #
    # Pad t_* in axis 0 by 1 on south (south-cell of v-face 0 doesn't
    # exist) and 1 on north (north-cell of v-face n_lat doesn't
    # exist).  This produces (n_lat+2, n_lon, nlev) arrays from which
    # the south-cell view is t_pad[:-1, ...] (rows 0..n_lat) and the
    # north-cell view is t_pad[1:, ...] (rows 1..n_lat+1).  At the
    # pole rows the corresponding triad value is 0, so the v-tendency
    # at pole faces vanishes naturally.
    pad0 = ((1, 1), (0, 0), (0, 0))
    t_NW_S = jnp.pad(t_NW, pad0)[:-1, :, :]   # south-cell NW at v-face j
    t_NE_S = jnp.pad(t_NE, pad0)[:-1, :, :]
    t_SW_N = jnp.pad(t_SW, pad0)[1:, :, :]    # north-cell SW at v-face j
    t_SE_N = jnp.pad(t_SE, pad0)[1:, :, :]

    # F_u at the four offsets, mapped to v-face index.  At v-face
    # (j, i), we need:
    #   F_u_S_W = F_u[j-1, i  , :]   (south-west of v-face)
    #   F_u_S_E = F_u[j-1, i+1, :]
    #   F_u_N_W = F_u[j  , i  , :]
    #   F_u_N_E = F_u[j  , i+1, :]
    # F_u has shape (n_lat, n_lon+1, nlev); pad in axis 0 with zeros
    # to align with v-face row index (rows 0..n_lat for v).
    F_u_pad = jnp.pad(F_u, pad0)              # (n_lat+2, n_lon+1, nlev)
    F_u_south = F_u_pad[:-1, :, :]            # (n_lat+1, n_lon+1, nlev)
    F_u_north = F_u_pad[1:, :, :]
    # Convert (n_lon+1) periodic to per-cell-index (n_lon).  At v-face
    # i (cell column i):
    #   F_u_*_W = F_u[*, i, :]      (west u-face of cell i)
    #   F_u_*_E = F_u[*, i+1, :]    (east u-face of cell i)
    F_u_S_W = F_u_south[:, :-1, :]            # (n_lat+1, n_lon, nlev)
    F_u_S_E = F_u_south[:, 1:, :]
    F_u_N_W = F_u_north[:, :-1, :]
    F_u_N_E = F_u_north[:, 1:, :]

    # v-tendency from PV (negative sign per the cross-product
    # convention used by the simple Sadourny call site).
    diag_vortcor_v = -(
        t_NW_S * F_u_S_E       # south-cell NW × SE U
        + t_NE_S * F_u_S_W     # south-cell NE × SW U
        + t_SW_N * F_u_N_E     # north-cell SW × NE U
        + t_SE_N * F_u_N_W     # north-cell SE × NW U
    )

    return diag_vortcor_u, diag_vortcor_v


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
