"""Shared lat-lon C-grid differential operators.

The finite-difference C-grid stencils (gradient / divergence / curl / Laplacian /
vector-Laplacian / cell<->face interpolation) used by BOTH the ocean lat-lon
C-grid dynamics AND the atmosphere lat-lon C-grid dycores.  They live in
``grids`` (the shared substrate) so the two components share one implementation
without an atmosphere->ocean import — they were historically defined in
``ocean.dynamics.latlon_cgrid_operators`` and moved here verbatim.

Pure grid operators: they depend only on a ``LatLonGrid`` (its dx/dy/masks) and
jax; the ocean-specific physics (Smagorinsky/Leith viscosity, SMC03 pressure
gradient, Coriolis, strain/stress) stays in ``ocean.dynamics.latlon_cgrid_operators``,
which imports these operators from here.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants  # noqa: F401 (kept for parity with the source module)
from legoesm.grids.latlon import LatLonGrid  # noqa: F401 — type compat


def pad_ns_zero(interior: jnp.ndarray) -> jnp.ndarray:
    """Zero-pad south and north rows (wall BC), backend-dispatched.

    Local backend
        Pads ``interior`` with zeros at both lat ends (the historical
        single-rank wall BC).  Bit-identical to the previous
        implementation.

    MPI backend (lat-lon band layout)
        Pole-touching ranks pad their pole side with zero (wall BC at
        the actual pole); interior partition cuts MPI-sendrecv with
        the neighbour rank so the gradient / divergence stencil sees
        continuous data across the cut.  Routes through
        :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` which
        in turn delegates to
        :func:`legoesm.parallel.latlon_mpi._pad_with_pole_bc_lat_mpi`.

    Parameters
    ----------
    interior : (..., n_interior, n_lon, ...) — missing the first and last
        rows along the latitude axis (axis 0 for 2D/3D fields).

    Returns
    -------
    padded : (..., n_interior+2, n_lon, ...)
    """
    # Deferred import to avoid cycles — ocean.dynamics is imported by
    # many grid-related modules.
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    return pad_with_pole_bc_lat(
        interior, halo=1, south_value=0.0, north_value=0.0,
    )


def is_tripolar(grid) -> bool:
    """Return True if ``grid`` carries an active tripolar fold descriptor.

    Replaces the legacy ``hasattr(grid, "<metric>") and grid.dlat == 0.0``
    sentinel dispatch.  Tripolar geometries always carry a ``fold`` field
    with ``is_active=True``; regular/Mercator ``LatLonCGridGeometry`` have
    ``is_active=False``; the old ``LatLonGrid`` lacks the field entirely.

    Under MPI latitude-band decomposition, ``is_tripolar`` returns True on
    ALL ranks of a tripolar run (the fold is active on every rank, even
    though the fold seam is physically present only on the northernmost
    rank).  Use :func:`_fold_is_local` to test whether the fold seam is
    locally present for fold-specific computations.
    """
    fold = getattr(grid, "fold", None)
    return fold is not None and bool(fold.is_active)


def _fold_is_local(grid) -> bool:
    """Return True if the tripolar fold seam is physically local to this rank.

    Under MPI decomposition, ``is_tripolar(grid)`` is True on all ranks of
    a tripolar run, but the fold seam (``fold_j >= 0``) exists only on the
    northernmost rank.  Non-northernmost ranks carry a sentinel fold with
    ``fold_j == -1`` and ``cap_j == -1`` to indicate "fold exists but is
    not local."  This function checks that the fold is both active AND
    locally present, which is the correct guard for fold-specific
    computations (fold-partner lookups, fold-permutation padding, etc.).
    """
    fold = getattr(grid, "fold", None)
    return fold is not None and bool(fold.is_active) and fold.fold_j >= 0


def pad_ns_scalar(interior: jnp.ndarray, grid) -> jnp.ndarray:
    """Pad south/north rows for a v-face or vertex scalar quantity.

    On regular lat-lon: zero-pad (wall BC).
    On tripolar (fold.is_active): south = zero, north = fold-reflected.

    Under MPI, all ranks call ``pad_ns_zero`` first (ensuring consistent
    MPI sendrecv call counts), then the northernmost rank replaces the
    north ghost row with fold-permuted data via :func:`_fold_is_local`.

    Parameters
    ----------
    interior : (n_lat-1, n_lon, ...) — interior v-face or vertex rows.
    grid : LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    padded : (n_lat+1, n_lon, ...)
    """
    padded = pad_ns_zero(interior)
    fold = getattr(grid, "fold", None)
    if fold is not None and fold.is_active and fold.fold_j >= 0:
        # Fold: last interior row, i-reversed via perm_T.
        # Handle the wrap column: fields with n_lon+1 columns have a
        # periodic wrap at column n_lon (== column 0).  Fold the first
        # n_lon columns, then append the wrap.
        last_row = interior[-1:]                     # (1, n_cols, ...)
        n_cols = last_row.shape[1]
        n_lon = fold.perm_T.shape[0]
        if n_cols == n_lon:
            north = last_row[:, fold.perm_T]
        else:
            # n_cols == n_lon + 1 (vertex or u-face field with wrap column)
            core = last_row[:, :n_lon][:, fold.perm_T]
            north = jnp.concatenate([core, core[:, 0:1]], axis=1)
        padded = jnp.concatenate([padded[:-1], north], axis=0)
    return padded


def _fold_row(last_row, perm, sign, n_lon):
    """Apply fold permutation with optional sign flip to one row."""
    n_cols = last_row.shape[1]
    if n_cols == n_lon:
        return sign * last_row[:, perm]
    else:
        core = sign * last_row[:, :n_lon][:, perm]
        return jnp.concatenate([core, core[:, 0:1]], axis=1)


def pad_ns_vector_v(interior: jnp.ndarray, grid) -> jnp.ndarray:
    """Pad south/north for a v-component at v-face latitudes.

    On regular lat-lon: zero-pad.
    On tripolar: south = zero, north = fold-reflected with sign flip.

    Under MPI, all ranks call ``pad_ns_zero`` first (consistent MPI call
    counts), then the northernmost rank applies the fold correction.
    """
    padded = pad_ns_zero(interior)
    fold = getattr(grid, "fold", None)
    if fold is not None and fold.is_active and fold.fold_j >= 0:
        n_lon = fold.perm_v.shape[0]
        north = _fold_row(interior[-1:], fold.perm_v, fold.vector_sign_v, n_lon)
        padded = jnp.concatenate([padded[:-1], north], axis=0)
    return padded


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


def interp_cell_to_vface(f: jnp.ndarray, grid=None) -> jnp.ndarray:
    """Interpolate a cell-center field to v-face (lat interface) positions.

    Interior faces: average of adjacent cells.

    Boundary faces depend on ``grid``:

    - ``grid is None`` (legacy default): south/north pole faces copy the
      adjacent cell value.  The pole value is numerically inert since
      ``v = 0`` at the wall.  Bit-exact backwards-compat.
    - ``grid`` provided: use :func:`pad_ns_scalar` — south = 0 and north
      = 0 (wall) on regular lat-lon, or north = fold-reflected on a
      tripolar grid where the north boundary is an active fold rather
      than a wall.  This is the form needed when the v-face value at the
      north fold is physically meaningful (e.g. interpolating a vertical
      velocity for momentum advection on eORCA1).

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.
    grid : optional LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    f_v : (n_lat+1, n_lon, ...) at v-faces.
    """
    f_v_interior = 0.5 * (f[:-1] + f[1:])  # (n_lat-1, ...)
    if grid is not None:
        return pad_ns_scalar(f_v_interior, grid)
    return jnp.concatenate([f[0:1], f_v_interior, f[-1:]], axis=0)


def cell_to_cgrid_winds(
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    grid=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert cell-centered winds to C-grid face-staggered winds.

    Works for both 2D (n_lat, n_lon) and 3D (n_lat, n_lon, nlev).
    v = 0 at pole walls on regular lat-lon; fold-reflected on tripolar.

    Parameters
    ----------
    u_cell, v_cell : cell-centered wind components.
    grid : optional LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    u_face : (..., n_lon+1, ...) at lon interfaces.
    v_face : (n_lat+1, ...) at lat interfaces.
    """
    u_face = interp_cell_to_uface(u_cell)
    v_interior = 0.5 * (v_cell[:-1] + v_cell[1:])
    fold = getattr(grid, "fold", None) if grid is not None else None
    if fold is not None and fold.is_active and fold.fold_j >= 0:
        v_face = pad_ns_vector_v(v_interior, grid)
    else:
        v_face = pad_ns_zero(v_interior)
    return u_face, v_face


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

    # dx at u-point.  On a regular lat-lon grid (dlat > 0), use the
    # legacy 1D path for bit-exact backward compat.  On a tripolar
    # grid (dlat == 0 sentinel), use the pre-computed 2D metric.
    if is_tripolar(grid):
        dx_u = grid.dx_u  # (n_lat, n_lon+1)
        if f.ndim == 2:
            return df_full / dx_u
        else:
            return df_full / dx_u[:, :, jnp.newaxis]
    else:
        dx_u = grid.radius * grid.dlon * grid.cos_lat
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
    # Tripolar grids retain the legacy compact-stencil-then-pad path
    # — the north fold-face gradient uses a per-cell partner lookup
    # that does not fit the pre-pad model.  Regular / Mercator lat-lon
    # grids switch to pre-pad-then-stencil so the compact stencil
    # spans rank-local partition cuts via backend-dispatched halo
    # exchange (the cross-partition correctness fix flagged in the
    # previous commit).
    if is_tripolar(grid):
        dy_v_int = grid.dy_v[1:-1]  # (n_lat-1, n_lon)
        dy_v_interior = (
            dy_v_int if f.ndim == 2 else dy_v_int[:, :, jnp.newaxis]
        )
        f_diff = f[1:] - f[:-1]
        df_interior = f_diff / dy_v_interior
        # Tripolar north-boundary fold partner.
        fold = grid.fold
        f_partner = f[-1:, fold.perm_T]
        dy_fold = grid.dy_v[-1:]
        if f.ndim == 3:
            dy_fold = dy_fold[:, :, jnp.newaxis]
        dy_fold_safe = jnp.maximum(dy_fold, 1.0e-30)
        df_fold = (f_partner - f[-1:]) / dy_fold_safe
        south = jnp.zeros_like(df_interior[:1])
        return jnp.concatenate([south, df_interior, df_fold], axis=0)

    # ---- Regular / Mercator lat-lon: pre-pad then compact stencil ----
    #
    # Why pre-pad: the old "compute interior gradient → pad with
    # zero" pattern hid an architectural mismatch under MPI.  At
    # interior partition cuts the band-end v-face should carry the
    # gradient across the cut (= (this_rank_f[0] - south_neighbour_f[-1])
    # / dy_v), but the old pad helper had no access to the neighbour
    # ``f`` value — only to a sendrecv'd ``df`` endpoint, which is
    # the wrong v-face's gradient.
    #
    # The new path pre-pads ``f`` via ``pad_halo_latlon`` (backend-
    # dispatched: local pole-fold OR MPI sendrecv + boundary pole-
    # fold), runs the compact stencil on the padded ``f``, and then
    # overrides the polar v-faces with zero via ``zero_polar_lat_ends``
    # (also backend-aware so interior cuts are left alone).
    #
    # Serial bit-exactness: the local backend's pole-fold gives a
    # non-zero gradient at v-faces 0 and -1, exactly what the new
    # path computes; ``zero_polar_lat_ends`` then zeros those two
    # ends — equivalent to the historical ``pad_ns_zero(df_interior)``.
    from legoesm.grids.halo_latlon import (
        pad_halo_latlon,
        pad_halo_latlon_3d,
        zero_polar_lat_ends,
    )
    if f.ndim == 2:
        f_padded = pad_halo_latlon(f, halo=1)
        # Strip the lon halo — gradient_y only needs the lat halo.
        f_padded = f_padded[:, 1:-1]
    elif f.ndim == 3:
        f_padded = pad_halo_latlon_3d(f, halo=1)
        f_padded = f_padded[:, 1:-1, :]
    else:
        raise ValueError(
            f"gradient_y_cgrid: f.ndim must be 2 or 3, got {f.ndim}"
        )

    # Compact stencil on padded f — gradient at ALL v-faces of the
    # rank-local band, including the partition cuts.
    f_diff = f_padded[1:] - f_padded[:-1]  # (n_lat_v, n_lon[, nlev])

    # ``dy_v`` at all v-faces.  For uniform-dlat regular lat-lon
    # this is constant (= R * dlat); ``mode='edge'`` pad simply
    # repeats the constant.  For Mercator (variable dlat) edge-pad
    # is the existing convention extended by one row — kept here so
    # the operator's behaviour is unchanged on Mercator grids.
    dy_h = grid.dy * 0.5                           # (n_lat,)
    dy_h_padded = jnp.pad(dy_h, (1, 1), mode="edge")
    dy_v = 0.5 * (dy_h_padded[1:] + dy_h_padded[:-1])  # (n_lat+1,)
    bcast = (slice(None),) + (jnp.newaxis,) * (f_diff.ndim - 1)
    df_dy = f_diff / dy_v[bcast]

    # Wall BC at the pole-touching v-faces.  Under MPI on interior
    # ranks this is a no-op; cross-partition gradients survive.
    return zero_polar_lat_ends(df_dy)


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
    u_eff = u
    v_eff = v
    if u_mask is not None:
        um = u_mask[..., jnp.newaxis] if u.ndim == 3 and u_mask.ndim == 2 else u_mask
        u_eff = u * um
    if v_mask is not None:
        vm = v_mask[..., jnp.newaxis] if v.ndim == 3 and v_mask.ndim == 2 else v_mask
        v_eff = v * vm

    # --- Zonal face length (meridional extent of u-face) ---
    if is_tripolar(grid):
        # Tripolar: use full 2D dy_u.  On a regular lat-lon grid dy_u
        # is constant in longitude, but on the bipolar cap it varies
        # significantly — column-0 extraction is NOT valid.
        face_dy = grid.dy_u  # (n_lat, n_lon+1) — full 2D
    else:
        # Regular or Mercator: variable-dy safe (1D over latitude).
        face_dy = grid.dy * 0.5  # (n_lat,)

    # East face flux - west face flux
    _is_2d_dy = hasattr(face_dy, 'ndim') and face_dy.ndim == 2
    if u.ndim == 2:
        u_east = u_eff[:, 1:]
        u_west = u_eff[:, :-1]
        if _is_2d_dy:
            face_dy_e = face_dy[:, 1:]
            face_dy_w = face_dy[:, :-1]
        elif hasattr(face_dy, 'ndim') and face_dy.ndim == 1:
            face_dy = face_dy[:, jnp.newaxis]
    else:
        u_east = u_eff[:, 1:, :]
        u_west = u_eff[:, :-1, :]
        if _is_2d_dy:
            face_dy_e = face_dy[:, 1:, jnp.newaxis]
            face_dy_w = face_dy[:, :-1, jnp.newaxis]
        elif hasattr(face_dy, 'ndim') and face_dy.ndim == 1:
            face_dy = face_dy[:, jnp.newaxis, jnp.newaxis]

    if _is_2d_dy:
        # Per-face dy: each u-face has its own meridional extent.
        net_zonal = u_east * face_dy_e - u_west * face_dy_w
    else:
        net_zonal = (u_east - u_west) * face_dy  # preserve arithmetic order

    # --- Meridional face length (zonal extent of v-face) ---
    # On a regular lat-lon grid this is the 1D array
    # R*cos(lat_v)*dlon; on a tripolar grid (dlat==0 sentinel) it
    # is the 2D array grid.dx_v.
    if is_tripolar(grid):
        face_dx = grid.dx_v  # (n_lat+1, n_lon) — 2D for tripolar
    else:
        lat = grid.lat
        lat_interior = 0.5 * (lat[:-1] + lat[1:])
        cos_lat_v_interior = jnp.cos(lat_interior)
        # Wall-BC pad: cos at the polar v-faces = 0 (no flux through
        # the pole).  Under MPI on a band-only rank the same call
        # sendrecv's the neighbour's cos_lat_v_interior at interior
        # partition cuts so flux continuity holds across the cut.
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        cos_lat_v = pad_with_pole_bc_lat(
            cos_lat_v_interior, halo=1,
            south_value=0.0, north_value=0.0,
        )
        face_dx = grid.radius * cos_lat_v * grid.dlon  # (n_lat+1,)

    # North face flux - south face flux
    if v.ndim == 2:
        v_north = v_eff[1:]
        v_south = v_eff[:-1]
        fd = face_dx
        if fd.ndim == 1:
            net_merid = v_north * fd[1:, jnp.newaxis] - v_south * fd[:-1, jnp.newaxis]
        else:
            net_merid = v_north * fd[1:] - v_south * fd[:-1]
    else:
        v_north = v_eff[1:, :, :]
        v_south = v_eff[:-1, :, :]
        fd = face_dx
        if fd.ndim == 1:
            net_merid = (v_north * fd[1:, jnp.newaxis, jnp.newaxis]
                         - v_south * fd[:-1, jnp.newaxis, jnp.newaxis])
        else:
            net_merid = (v_north * fd[1:, :, jnp.newaxis]
                         - v_south * fd[:-1, :, jnp.newaxis])

    # Cell area
    area = grid.area  # (n_lat, n_lon)
    if u.ndim == 3:
        area = area[..., jnp.newaxis]

    div = (net_zonal + net_merid) / area
    return div


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
        # Boundary: wall BC on regular lat-lon; fold on tripolar.
        v_mask_interior = mask[:-1] * mask[1:]
        v_mask = pad_ns_scalar(v_mask_interior, grid)
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

    if is_tripolar(grid):
        # Tripolar: use full 2D per-cell metrics.  On the bipolar cap,
        # metrics vary significantly in BOTH lat and lon — column-0
        # extraction is not valid.
        A_vertex_interior = grid.area_q[1:-1]         # (n_lat-1, n_lon+1)
        dx_cell = grid.dx_T                            # (n_lat, n_lon)
        # dy at each v-face edge of the circulation loop: east and west
        # edges have different dy on the distorted cap.
        dy_v_2d = grid.dy_v                            # (n_lat+1, n_lon)
        _tripolar_curl = True
    else:
        R = grid.radius
        dlon = grid.dlon
        dlat = grid.dlat
        lat = grid.lat
        cos_lat = grid.cos_lat
        sin_lat = jnp.sin(lat)
        # Wall-BC pad of sin_lat: sin(south_pole)=-1, sin(north_pole)=+1.
        # Backend-aware so MPI interior ranks sendrecv from neighbour
        # rather than apply pole BC at the wrong location.
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        sin_ext = pad_with_pole_bc_lat(
            sin_lat, halo=1, south_value=-1.0, north_value=1.0,
        )
        A_vertex_all = R**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])
        A_vertex_interior = A_vertex_all[1:-1]
        dx_cell = R * cos_lat * dlon
        dy_edge = R * dlat
        _tripolar_curl = False

    # v contribution: circulation from v-edges (east minus west).
    v_east = v
    v_west = jnp.roll(v, 1, axis=1)
    if _tripolar_curl:
        # Per-face dy: v_east * dy_east - v_west * dy_west
        dy_east = dy_v_2d
        dy_west = jnp.roll(dy_v_2d, 1, axis=1)
        if is_3d:
            dy_east = dy_east[:, :, jnp.newaxis]
            dy_west = dy_west[:, :, jnp.newaxis]
        dv_circ = v_east * dy_east - v_west * dy_west
    else:
        # Meridional edge length at vertex rows: variable-dy safe.
        dy_h = grid.dy * 0.5                              # (n_lat,) cell heights
        dy_edge_interior = 0.5 * (dy_h[1:] + dy_h[:-1])    # (n_lat-1,)
        dy_edge = jnp.pad(dy_edge_interior, (1, 1), mode='edge')  # (n_lat+1,)
        bcast_lat = (slice(None),) + (jnp.newaxis,) * (v.ndim - 1)
        dv_circ = (v_east - v_west) * dy_edge[bcast_lat]

    # u contribution: u[i-1, j]*dx[i-1] - u[i, j]*dx[i].
    # Pad with zeros at poles along the lat axis (axis 0).  Wall BC
    # at the pole; under MPI on an interior rank the same call
    # sendrecv's the neighbour's u row instead (the pole pad fires
    # only at boundary ranks).
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    u_ext = pad_with_pole_bc_lat(
        u, halo=1, south_value=0.0, north_value=0.0,
    )
    if _tripolar_curl:
        # dx_cell is 2D (n_lat, n_lon); pad lat axis, append wrap column
        dx_pad = jnp.pad(dx_cell, ((1, 1), (0, 0)))  # (n_lat+2, n_lon)
        dx_pad = jnp.concatenate([dx_pad, dx_pad[:, 0:1]], axis=1)  # (n_lat+2, n_lon+1)
        dx_south = dx_pad[:-1]  # (n_lat+1, n_lon+1)
        dx_north = dx_pad[1:]   # (n_lat+1, n_lon+1)
        if is_3d:
            dx_south = dx_south[:, :, jnp.newaxis]
            dx_north = dx_north[:, :, jnp.newaxis]
        u_south = u_ext[:-1]
        u_north = u_ext[1:]
        du_circ = u_south * dx_south - u_north * dx_north
    else:
        # Wall-BC pad of dx_cell at poles (zero contribution beyond
        # the pole); under MPI interior ranks pad with neighbour's
        # dx via sendrecv instead.
        dx_ext = pad_with_pole_bc_lat(
            dx_cell, halo=1, south_value=0.0, north_value=0.0,
        )
        u_south = u_ext[:-1]
        u_north = u_ext[1:]
        dx_south = dx_ext[:-1]
        dx_north = dx_ext[1:]
        bcast = (slice(None),) + (jnp.newaxis,) * (u.ndim - 1)
        du_circ = (u_south * dx_south[bcast] - u_north * dx_north[bcast])

    # Append periodic wrap column to dv_circ.
    dv_circ_full = jnp.concatenate([dv_circ, dv_circ[:, 0:1]], axis=1)

    circ = du_circ + dv_circ_full

    # Compute vorticity only on interior rows (pole rows zero by
    # construction; avoids 1/0 division — issue #173).
    circ_interior = circ[1:-1]
    if _tripolar_curl:
        if is_3d:
            zeta_interior = circ_interior / A_vertex_interior[:, :, jnp.newaxis]
        else:
            zeta_interior = circ_interior / A_vertex_interior
    else:
        bcast = (slice(None),) + (jnp.newaxis,) * (u.ndim - 1)
        zeta_interior = circ_interior / A_vertex_interior[bcast]
    # Boundary: wall BC (zero) on regular lat-lon; fold halo on tripolar.
    zeta = pad_ns_scalar(zeta_interior, grid)

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
    if is_tripolar(grid):
        # Tripolar: full 2D dy at u-face stagger.
        dy = grid.dy_u  # (n_lat, n_lon+1)
        if zeta.ndim == 3:
            return (zeta[1:] - zeta[:-1]) / dy[:, :, jnp.newaxis]
        return (zeta[1:] - zeta[:-1]) / dy
    else:
        # Regular or Mercator: variable-dy safe.
        dy = grid.dy * 0.5  # (n_lat,)
        diff = zeta[1:] - zeta[:-1]
        bcast = (slice(None),) + (jnp.newaxis,) * (diff.ndim - 1)
        return diff / dy[bcast]


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
    if is_tripolar(grid):
        dx_v_int = grid.dx_v[1:-1]  # (n_lat-1, n_lon) — full 2D
    else:
        R = grid.radius
        dlon = grid.dlon
        lat = grid.lat
        lat_interior = 0.5 * (lat[:-1] + lat[1:])
        cos_lat_v_int = jnp.cos(lat_interior)
        dx_v_int = R * cos_lat_v_int * dlon

    # zeta[:, j+1] - zeta[:, j] for j=0..n_lon-1
    dzeta = zeta[:, 1:] - zeta[:, :-1]  # (n_lat+1, n_lon [, nlev])

    # Compute gradient only on interior rows (1..n_lat-1), pad poles
    # with zero.
    dzeta_int = dzeta[1:-1]  # (n_lat-1, n_lon [, nlev])
    if dx_v_int.ndim == 2:
        # Tripolar: full 2D dx_v
        if zeta.ndim == 3:
            grad_int = dzeta_int / dx_v_int[:, :, jnp.newaxis]
        else:
            grad_int = dzeta_int / dx_v_int
    else:
        # Regular lat-lon: 1D dx_v
        if zeta.ndim == 2:
            grad_int = dzeta_int / dx_v_int[:, jnp.newaxis]
        else:
            grad_int = dzeta_int / dx_v_int[:, jnp.newaxis, jnp.newaxis]
    return pad_ns_scalar(grad_int, grid)


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
        vmask = _compute_vertex_mask(mask, grid=grid)
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


def _compute_vertex_mask(land_mask: jnp.ndarray, grid=None) -> jnp.ndarray:
    """Compute vertex mask: wet only if all four surrounding cells are wet.

    Parameters
    ----------
    land_mask : (n_lat, n_lon)
    grid : optional LatLonGrid or LatLonCGridGeometry.
        When provided and a tripolar fold is active, the north-boundary
        vertex mask is computed from the fold-partner cells rather than
        being zero (wall BC).

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

    fold = getattr(grid, "fold", None) if grid is not None else None
    if fold is not None and fold.is_active and fold.fold_j >= 0:
        return pad_ns_scalar(interior_full, grid)
    return pad_ns_zero(interior_full)
