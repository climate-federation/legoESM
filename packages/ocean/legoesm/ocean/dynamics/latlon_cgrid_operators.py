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
import numpy as np

from legoesm import constants
from legoesm.grids.latlon import LatLonGrid  # noqa: F401 — kept for type compat
from legoesm.grids.operators_latlon_cgrid import (
    pad_ns_zero,
    pad_ns_zero_multi,
    is_tripolar,
    reads_stored_vface_metric,
    fold_is_local,
    north_fold_mask,
    apply_north_fold,
    lat_ends_are_poles,  # noqa: F401 — re-export for ocean dynamics call sites
    pad_ns_scalar,
    fold_row,
    pad_ns_vector_v,
    interp_cell_to_uface,
    interp_cell_to_vface,
    interp_u_to_vface_4pt,
    cell_to_cgrid_winds,
    gradient_x_cgrid,
    gradient_y_cgrid,
    divergence_cgrid,
    laplacian_cgrid,
    curl_vertex_cgrid,
    gradient_curl_to_u,
    gradient_curl_to_v,
    vector_laplacian_cgrid,
    compute_vertex_mask,
)
# Operators accept LatLonGrid or LatLonCGridGeometry via duck typing.
# When geometry fields (dx_u, dy_v, etc.) are available they are used
# directly; otherwise the legacy inline computation from cos_lat/dlon
# is the fallback.  See ensure_geometry() in latlon.py.


# =============================================================================
# North/south boundary padding helpers (Phase 1B)
# =============================================================================
#
# On a regular lat-lon grid, v-face and vertex quantities are zero-padded
# at the north and south poles (wall BC).  On a tripolar grid, the south
# pole remains a wall, but the north boundary is a fold seam where data
# comes from the opposite side of the fold.
#
# These helpers abstract the padding so operators don't need to know
# whether the grid has a fold.  For regular lat-lon grids (fold inactive),
# they reduce to jnp.pad — bit-exact to the current code.


# ``is_tripolar``, ``fold_is_local``, ``pad_ns_scalar``, ``pad_ns_vector_v``,
# ``cell_to_cgrid_winds`` and ``compute_vertex_mask`` live in
# ``legoesm.grids.operators_latlon_cgrid`` (imported above) and are
# re-exported here for the ocean dynamics call sites.


def fold_vface_row(cell_field: jnp.ndarray, grid) -> jnp.ndarray:
    """Compute the fold ghost row for a v-face quantity.

    On a tripolar grid, the v-face at the fold boundary connects cell
    (fold_j, i) with its fold partner (fold_j, perm_T[i]).  This helper
    returns the fold-partner's cell values, permuted by the fold.  The
    caller is responsible for combining it with the local cell values
    (e.g. ``jnp.minimum(cell[-1:], fold_vface_row(cell, grid))`` for
    face depths).

    On a regular lat-lon grid (no fold), returns a zero row.

    Parameters
    ----------
    cell_field : (n_lat, n_lon, ...) — cell-center quantity.
    grid : LatLonGrid or LatLonCGridGeometry with ``fold`` attribute.

    Returns
    -------
    partner_row : (1, n_lon, ...) — fold partner values at the fold row.
    """
    fold = getattr(grid, "fold", None)
    # ``fold.is_active`` (not ``fold_is_local``): the perms are preserved on
    # every SPMD band (fold_j=-1) so this returns a valid fold-partner row on
    # any band; the CALLER selects it on the north band (data-dependent) or the
    # fold-local rank.  Callers only ever invoke this inside a fold gate, so
    # dropping the ``fold_j >= 0`` check leaves serial/MPI behaviour unchanged.
    if fold is not None and fold.is_active:
        return cell_field[-1:, fold.perm_T]
    return jnp.zeros_like(cell_field[-1:])


def pad_ns_vector_u(interior: jnp.ndarray, grid) -> jnp.ndarray:
    """Pad south/north for a u-component at v-face latitudes.

    On regular lat-lon: zero-pad.
    On tripolar: south = zero, north = fold-reflected with sign flip.

    Under MPI, all ranks call ``pad_ns_zero`` first (consistent MPI call
    counts), then the northernmost rank applies the fold correction.
    """
    padded = pad_ns_zero(interior)
    fold = getattr(grid, "fold", None)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        north = fold_row(interior[-1:], fold.perm_T, fold.vector_sign_u,
                         fold.perm_T.shape[0])
        padded = apply_north_fold(padded, north, grid, north_mask=nmask)
    return padded


# ``interp_u_to_vface_4pt`` is the backend-dispatched CELL-PAD-FIRST core
# operator imported from ``legoesm.grids.operators_latlon_cgrid`` above (line ~44).
# An older interior-average-then-``pad_ns_vector_u`` redefinition used to shadow it
# here; it was SPMD-blind — at a lat-band partition cut it averaged only the
# rank-LOCAL interior v-faces and then refilled the cut row from the neighbour's
# ADJACENT interior face (one row off), so the two bands sharing a v-face disagreed
# (eORCA025 SPMD equivalence: barotropic V_bar diverged ~8e-4 at the cut rows
# whose ``f_v`` is non-zero — the equator cut was masked by ``f_v≈0``).  The core
# version pads ``u`` over latitude FIRST (``pad_with_pole_bc_lat`` -> local jnp.pad
# / MPI-or-SPMD neighbour row) THEN averages, so every band-cut v-face uses the
# true neighbour-band u row.  It is bit-identical to the old interior-then-pad form
# for SERIAL / full-domain + physical-boundary behavior (its docstring proves the
# pole-wall + local tripolar-fold rows match ``pad_ns_vector_u`` exactly); at an
# MPI/SPMD interior band cut it is DELIBERATELY different (the one-row-off stale
# value is the bug being fixed).  De-duplicated to the single canonical operator
# (no shadowing copy) so the barotropic solver + Matsuno Coriolis backward step
# share the SPMD-correct interpolation.


def pad_ns_vector_pair(
    u_interior: jnp.ndarray,
    v_interior: jnp.ndarray,
    grid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad both vector components with fold + rotation at the north boundary.

    On a regular lat-lon grid (fold inactive), this is equivalent to
    calling ``pad_ns_vector_u`` and ``pad_ns_vector_v`` independently.

    On a tripolar grid with rotation angles in the bipolar cap, the
    fold halo exchange must rotate the vector from the source cell's
    local frame to the destination cell's frame.  The combined
    transformation for the north ghost row is:

        u_dest = -cos(Δα) * u_src - sin(Δα) * v_src
        v_dest =  sin(Δα) * u_src - cos(Δα) * v_src

    where Δα = α_dest - α_source is the rotation-angle difference
    between the destination cell and the fold-partner source cell.
    When all rotation angles are zero (regular lat-lon or synthetic
    tripolar), this reduces to ``(u_d, v_d) = (-u_s, -v_s)``.

    Parameters
    ----------
    u_interior : (n_lat-1, n_lon, ...) — u-component at interior v-face rows.
    v_interior : (n_lat-1, n_lon, ...) — v-component at interior v-face rows.
    grid : LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    (u_padded, v_padded) : each (n_lat+1, n_lon, ...)
    """
    fold = getattr(grid, "fold", None)
    if fold is None or not fold.is_active or fold.fold_j < 0:
        return pad_ns_zero_multi(u_interior, v_interior)

    u_padded, v_padded = pad_ns_zero_multi(u_interior, v_interior)

    n_lon = fold.perm_T.shape[0]

    u_src = fold_row(u_interior[-1:], fold.perm_T, 1.0, n_lon)
    v_src = fold_row(v_interior[-1:], fold.perm_v, 1.0, n_lon)

    cos_alpha_v = getattr(grid, "cos_alpha_v", None)
    sin_alpha_v = getattr(grid, "sin_alpha_v", None)

    if cos_alpha_v is not None and sin_alpha_v is not None:
        cos_d = cos_alpha_v[-1:, :]
        sin_d = sin_alpha_v[-1:, :]

        cos_s_row = cos_alpha_v[-2:-1, :]
        sin_s_row = sin_alpha_v[-2:-1, :]
        cos_s = cos_s_row[:, fold.perm_v]
        sin_s = sin_s_row[:, fold.perm_v]

        cos_da = cos_d * cos_s + sin_d * sin_s
        sin_da = sin_d * cos_s - cos_d * sin_s

        if u_interior.ndim == 3:
            cos_da = cos_da[:, :, jnp.newaxis]
            sin_da = sin_da[:, :, jnp.newaxis]

        north_u = -cos_da * u_src - sin_da * v_src
        north_v = sin_da * u_src - cos_da * v_src
    else:
        north_u = -u_src
        north_v = -v_src

    u_padded = jnp.concatenate([u_padded[:-1], north_u], axis=0)
    v_padded = jnp.concatenate([v_padded[:-1], north_v], axis=0)
    return u_padded, v_padded


# =============================================================================
# Cell-center ↔ face interpolation (shared by atmosphere and ocean)
# =============================================================================


def interp_wface_to_center(f: jnp.ndarray) -> jnp.ndarray:
    """Interpolate an interface (w-face) field UP to full (cell-center) levels.

    Inverse-direction vertical companion to ``interp_cell_to_uface`` /
    ``interp_cell_to_vface`` (which move horizontally between cell centers
    and u/v-faces).  Here the staggering is vertical: the input lives at the
    ``nlev-1`` interior interfaces (w-faces, between adjacent cell centers),
    the output at the ``nlev`` cell centers.

    Convention matches the rest of this package: ``k=0`` is the surface,
    ``k`` increases downward, and interface ``k`` (for ``k = 0 .. nlev-2``)
    sits between full levels ``k`` and ``k+1``.  So interior center ``k``
    (``1 <= k <= nlev-2``) is the average of the interfaces above (``k-1``)
    and below (``k``); the top center (``k=0``) and bottom center
    (``k=nlev-1``) have only one adjacent interface and take it one-sided::

        center[0]      = iface[0]
        center[k]      = 0.5 * (iface[k-1] + iface[k])   1 <= k <= nlev-2
        center[nlev-1] = iface[nlev-2]

    Used to lift a depth-resolved interface diffusivity (the 3-D EKE
    ``kappa_GM`` / ``kappa_Redi``, which lives at the W-grid interfaces) to
    cell centers before horizontal interpolation to u/v-faces in the GM/Redi
    tracer-tendency operators — so the interface kappa is placed directly on
    the w-faces for the vertical flux (no lossy round-trip) while the
    horizontal flux still sees a center-then-face interpolation.

    Pure slicing + concatenation: pytree-friendly, vmappable, differentiable.

    Parameters
    ----------
    f : (n_lat, n_lon, nlev-1) at interior interfaces (w-faces).

    Returns
    -------
    f_c : (n_lat, n_lon, nlev) at cell centers.
    """
    interior = 0.5 * (f[:, :, :-1] + f[:, :, 1:])  # (n_lat, n_lon, nlev-2)
    top = f[:, :, 0:1]                              # (n_lat, n_lon, 1)
    bottom = f[:, :, -1:]                           # (n_lat, n_lon, 1)
    return jnp.concatenate([top, interior, bottom], axis=-1)


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


def min_cell_to_vface(f: jnp.ndarray, grid=None) -> jnp.ndarray:
    """Min-rule interpolation of a cell-center thickness to v-faces.

    Same convention as ``min_cell_to_uface`` for the meridional
    direction.  South pole row is zero (wall).  North pole row is
    zero on regular lat-lon (wall) or fold min-rule on tripolar.

    Parameters
    ----------
    f : (n_lat, n_lon, ...) at cell centers.
    grid : optional LatLonGrid or LatLonCGridGeometry.

    Returns
    -------
    f_v : (n_lat+1, n_lon, ...) at v-faces.
    """
    # Cell-pad-first (PR357 Bug-2 pattern): pad the cell field so the v-face
    # min at a partition cut sees the neighbour rank's adjacent cell (MPI
    # halo exchange) rather than a rank-local-only / zeroed boundary row.
    f_padded = pad_ns_zero(f)
    f_v = jnp.minimum(f_padded[:-1], f_padded[1:])
    from legoesm.grids.halo_latlon import zero_polar_lat_ends
    f_v = zero_polar_lat_ends(f_v)
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        f_partner = f[-1:, grid.fold.perm_T]
        north = jnp.minimum(f[-1:], f_partner)
        f_v = apply_north_fold(f_v, north, grid, north_mask=nmask)
    return f_v


# =============================================================================
# Gradient operators (scalar at cell center -> vector at faces)
# =============================================================================


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
    # Shared cell-pad-first 4-pt average (interp_u_to_vface_4pt): the
    # MPI band partition-cut v-face averages the neighbour rank's true
    # u row instead of the old interior-then-pad_ns_vector_u refill
    # (one face row off at a cut).  Wall BC (zero) at physical poles;
    # fold (sign*perm) on tripolar — serial bit-identical.
    u_at_v = interp_u_to_vface_4pt(u, grid)

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


def vertex_coriolis(grid: LatLonGrid) -> jnp.ndarray:
    """Planetary Coriolis ``f`` at C-grid VERTICES (corners), shape
    ``(n_lat+1, n_lon+1)``.

    On a lat-lon grid ``f`` depends only on latitude, and the vertex latitude
    equals the v-face latitude, so the vertex ``f`` is ``grid.f_v`` extended by
    one periodic-wrap column.  This is the single shared ``f`` value that makes
    the C-grid Coriolis energy-conserving on a β-plane (see
    :func:`coriolis_cgrid_energy_conserving`).
    """
    if hasattr(grid, "f_v"):
        f_v = grid.f_v  # (n_lat+1, n_lon)
    else:
        f_cell = grid.f
        f_v_int = 0.5 * (f_cell[:-1] + f_cell[1:])
        f_v = jnp.concatenate([f_cell[0:1], f_v_int, f_cell[-1:]], axis=0)
    return jnp.concatenate([f_v, f_v[:, 0:1]], axis=1)  # (n_lat+1, n_lon+1)


# Back-compat internal alias (promoted to public for the ene_total consumer;
# no private cross-module imports).
_vertex_coriolis = vertex_coriolis


def coriolis_cgrid_energy_conserving(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ENERGY-CONSERVING C-grid Coriolis (Sadourny 1975) using VERTEX ``f``.

    The default :func:`coriolis_cgrid` evaluates ``f`` at u-points (``f_u``) for
    the ``f·v→u`` term and at v-points (``f_v``) for the ``-f·u→v`` term.  On a
    β-plane these are DIFFERENT values (``f`` varies with latitude), so the two
    terms do not cancel in the discrete kinetic-energy budget
    ``Σ u·(f·v) − Σ v·(f·u) ≠ 0`` and the scheme spuriously injects/removes
    energy (measured ``~1e-6·f·KE`` on grid-scale fields; the root cause of the
    MITgcm barotropic-gyre oracle residual — see
    ``docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md``).

    This variant uses the SINGLE ``f`` value at the VERTEX shared by each
    (u-point, v-point) pair, so the paired contributions
    ``0.25·f_vertex·u·v`` appear identically in ``cor_u`` and ``cor_v`` and
    cancel exactly in the energy sum (verified machine-zero power on random
    β-plane fields).  On an f-plane (``f`` constant) it reduces to the same
    answer as :func:`coriolis_cgrid`.

    ``cor_u`` depends only on ``v`` and ``cor_v`` only on ``u`` — so a
    forward-backward caller can compute ``cor_u`` from ``v_old`` and ``cor_v``
    from the updated ``u_pred`` by two calls (or by passing the right field).

    Shapes match :func:`coriolis_cgrid`: ``u`` is ``(n_lat, n_lon+1[, nlev])``,
    ``v`` is ``(n_lat+1, n_lon[, nlev])``.
    """
    if is_tripolar(grid):
        raise NotImplementedError(
            "coriolis_cgrid_energy_conserving is not yet implemented on tripolar "
            "grids (vertex f + fold-seam averaging needs the 2D metric handling "
            "of coriolis_cgrid). Use coriolis_energy_conserving=False there."
        )
    is_3d = u.ndim == 3
    f_q = _vertex_coriolis(grid)                       # (n_lat+1, n_lon+1)
    if is_3d:
        f_q = f_q[..., jnp.newaxis]

    # --- cor_u at u-points (n_lat, n_lon+1): + f·v averaged with vertex f ---
    # South/north vertices of u-row i are vertex rows i and i+1.
    fq_s = f_q[:-1]                                     # (n_lat, n_lon+1[,1])
    fq_n = f_q[1:]
    v_west = jnp.roll(v, 1, axis=1)                     # v(:, j-1)

    def _lon_face(a):  # (rows, n_lon[,nlev]) cell field -> (rows, n_lon+1) face
        return jnp.concatenate([a, a[:, 0:1]], axis=1)

    vs_w = _lon_face(v_west[:-1]); vs_e = _lon_face(v[:-1])   # south (v row i)
    vn_w = _lon_face(v_west[1:]);  vn_e = _lon_face(v[1:])    # north (v row i+1)
    cor_u = 0.25 * (fq_s * (vs_w + vs_e) + fq_n * (vn_w + vn_e))

    # --- cor_v at v-points (n_lat+1, n_lon): - f·u averaged with vertex f ---
    # West/east vertices of v-col j are vertex cols j and j+1.
    fq_w = f_q[:, :-1]                                  # (n_lat+1, n_lon[,1])
    fq_e = f_q[:, 1:]
    u_east = jnp.roll(u, -1, axis=1)                    # u(:, J+1)

    def _lat_sum(a):  # sum u-rows i-1 and i at v-face row i; walls at the ends
        return jnp.concatenate([a[0:1], a[:-1] + a[1:], a[-1:]], axis=0)

    u_w = u[:, :-1]                                     # west-face col J=j
    u_e = u_east[:, :-1]                                # east-face col J=j+1
    cor_v = -0.25 * (fq_w * _lat_sum(u_w) + fq_e * _lat_sum(u_e))

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


# =============================================================================
# Vector Laplacian: grad(div) - k x grad(curl)
# =============================================================================


def recover_velocity_from_streamfunction(
    psi: jnp.ndarray,
    inv_H_u: jnp.ndarray,
    inv_H_v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Barotropic velocity from a vertex streamfunction ``psi`` (rigid lid).

    Under a rigid lid the depth-integrated transport is non-divergent and
    carried by a streamfunction on vertex (corner) points:

        (H·u_bt, H·v_bt) = ∇⊥ψ = (-∂ψ/∂y, +∂ψ/∂x)

    so the barotropic velocity at the C-grid faces is

        u_bt = -(1/H_u)·∂ψ/∂y      at u-faces (n_lat, n_lon+1)
        v_bt = +(1/H_v)·∂ψ/∂x      at v-faces (n_lat+1, n_lon)

    The tangential vertex-gradient stencils ``gradient_curl_to_u`` /
    ``gradient_curl_to_v`` supply ∂ψ/∂y at u-faces and ∂ψ/∂x at v-faces
    (periodic-wrap + pole/wall handling inherited).  Sign convention matches
    ``diagnostics_streamfunction.barotropic_streamfunction`` and Veros
    ``core/external/solve_stream.py`` (u = -1/H ∂ψ/∂y, v = +1/H ∂ψ/∂x).

    Parameters
    ----------
    psi : (n_lat+1, n_lon+1) vertex streamfunction [m^3/s].
    inv_H_u : (n_lat, n_lon+1) reciprocal column depth at u-faces [1/m].
    inv_H_v : (n_lat+1, n_lon) reciprocal column depth at v-faces [1/m].
    grid : LatLonGrid.
    u_mask, v_mask : optional face masks (zero velocity on land faces).

    Returns
    -------
    (u_bt, v_bt) : zonal velocity (n_lat, n_lon+1) and meridional (n_lat+1, n_lon).
    """
    dpsi_dy_u = gradient_curl_to_u(psi, grid)   # (n_lat, n_lon+1) — ∂ψ/∂y at u-faces
    dpsi_dx_v = gradient_curl_to_v(psi, grid)   # (n_lat+1, n_lon) — ∂ψ/∂x at v-faces
    u_bt = -inv_H_u * dpsi_dy_u
    v_bt = inv_H_v * dpsi_dx_v
    if u_mask is not None:
        u_bt = u_bt * u_mask
    if v_mask is not None:
        v_bt = v_bt * v_mask
    return u_bt, v_bt


def streamfunction_vorticity_operator(
    psi: jnp.ndarray,
    inv_H_u: jnp.ndarray,
    inv_H_v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Weighted vertex Laplacian ``L(ψ) = ∇·((1/H)∇ψ)`` on the C-grid.

    This is the rigid-lid elliptic operator (Veros ``poisson_matrix``): the
    barotropic vorticity equation is ``L(∂ψ/∂t) = curl((1/H)∫F dz)``.

    It is exactly the vertex curl of the recovered barotropic velocity::

        L(ψ) = curl_vertex( u_bt(ψ), v_bt(ψ) )

    because ``div_vertex(fx, fy) = curl_vertex(-fy, fx)`` and the velocity
    recovery ``(-(1/H)∂ψ/∂y, +(1/H)∂ψ/∂x)`` is exactly ``(-fy, fx)`` with the
    flux ``f = (1/H)∇ψ``.  Defining the operator as this composition (rather
    than a separately-assembled matrix) guarantees the discrete operator and
    the right-hand side ``curl_vertex((1/H)∫F dz)`` use *identical* stencils,
    so there is no operator/RHS discretisation mismatch.  Self-adjoint and
    negative semi-definite in the continuum; the discrete form reuses the
    audited ``curl_vertex_cgrid`` + ``_gradient_curl_to_*`` stencils.

    The land/Dirichlet handling (ψ = island constant on land, pinned for the
    interior solve) lives in the solver (``rigid_lid_latlon_cgrid``), not here:
    pass ``u_mask``/``v_mask`` to zero land-face velocities.

    Returns
    -------
    L(ψ) : vertex field (n_lat+1, n_lon+1).
    """
    u_bt, v_bt = recover_velocity_from_streamfunction(
        psi, inv_H_u, inv_H_v, grid, u_mask=u_mask, v_mask=v_mask)
    return curl_vertex_cgrid(u_bt, v_bt, grid)


def vertex_area_cgrid(grid: LatLonGrid) -> jnp.ndarray:
    """Dual-cell area at vertex (corner) points, shape (n_lat+1, n_lon+1).

    This is the area by which ``curl_vertex_cgrid`` divides the circulation
    to obtain the vertex vorticity, i.e. ``A_vertex·ζ_vertex`` is the discrete
    circulation around the vertex's dual cell.  Summing ``A_vertex·ζ_vertex``
    over a set of vertices therefore gives (discrete Stokes) the circulation
    around the boundary of the union of their dual cells — the basis for the
    rigid-lid island line integrals (``rigid_lid_islands``).

    Pole/wall rows (i = 0, n_lat) carry zero area (wall BC), matching the
    pole handling in ``curl_vertex_cgrid``.
    """
    if is_tripolar(grid):
        # Full 2D vertex areas; pole rows zero.
        A_int = grid.area_q[1:-1]                       # (n_lat-1, n_lon+1)
        n_lon1 = A_int.shape[1]
        zero_row = jnp.zeros((1, n_lon1), dtype=A_int.dtype)
        return jnp.concatenate([zero_row, A_int, zero_row], axis=0)
    A_lat = _vertex_dual_area_interior(grid.lat, grid.radius, grid.dlon)  # (n_lat+1,)
    # Interior rows carry area; pole rows (wall BC) zero — consistent with
    # curl_vertex_cgrid computing vorticity only on interior rows.
    A_lat = A_lat.at[0].set(0.0).at[-1].set(0.0)
    # Broadcast to (n_lat+1, n_lon+1): regular lat-lon area varies only in lat.
    return jnp.broadcast_to(A_lat[:, jnp.newaxis], (grid.n_lat + 1, grid.n_lon + 1))


def vector_laplacian_dissipation_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    A_h_center: jnp.ndarray,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Positive-definite KE-dissipation density of the lateral vector-Laplacian
    viscosity, at cell centres [m²/s³] — legoESM's analogue of Veros's flux-form
    ``K_diss_h = 0.5·Σ(Δu·flux) = A_h|∇u|²`` (``veros/core/friction.py``
    ``calc_diss_u``/``calc_diss_v``).

    legoESM's lateral viscosity is the VECTOR Laplacian
    ``∇²_vec(u,v) = grad(div) − k×grad(curl)`` (``vector_laplacian_cgrid``), NOT
    the componentwise scalar Laplacian Veros uses.  For the vector Laplacian the
    KE-removal energy identity is the Helmholtz form

        −∫ (u·∇²_vec u + v·∇²_vec v) dA  =  ∫ (|div u|² + |ζ|²) dA  ≥ 0

    (integration by parts on a closed/periodic domain), so the POSITIVE-DEFINITE
    per-cell dissipation density that integrates to the SAME mean KE removed by the
    harmonic lateral viscosity is

        diss = A_h · ( div²  +  <ζ²>_corners )                                  (≥ 0)

    — NOT the scalar-Laplacian energy ``A_h|∇u|²``.  This is ≥ 0 by construction
    (squares × A_h ≥ 0), so it needs NO clamp, and (verified on the spun-up ACC
    state) its domain integral equals the unclamped ``−u·A_h∇²_vec u`` to 0.3 %,
    whereas the clamped dynamical form over-credits by ~11–20 %.

    The divergence (cell centres) and relative vorticity ζ (vertices) are formed
    by the SAME shared operators and with the SAME face/vertex masking that
    ``vector_laplacian_cgrid`` uses internally (``divergence_cgrid``,
    ``curl_vertex_cgrid``, ``compute_vertex_mask``) — so the energy this credits is
    exactly the energy the applied viscous tendency removes (no duplicate or
    inconsistent numerics).  ζ² is averaged from the four surrounding vertices to
    the cell centre (the same 4-corner stagger as ``smagorinsky_viscosity_cgrid``).

    Parameters
    ----------
    u : (n_lat, n_lon+1[, nlev]) zonal velocity at u-faces [m/s].
    v : (n_lat+1, n_lon[, nlev]) meridional velocity at v-faces [m/s].
    grid : LatLonGrid.
    A_h_center : (n_lat,) or (n_lat, n_lon) or broadcastable to the cell-centre
        field — the lateral viscosity coefficient AT CELL CENTRES [m²/s], i.e. the
        same ``A_h × scale`` (cos-power / eq-boost / cap-boost / slope-foot) field
        the caller applies to the u vector-Laplacian tendency.  A 1-D ``(n_lat,)``
        latitudinal profile is broadcast over longitude + levels.
    mask : (n_lat, n_lon) cell-centre ocean mask (1 = ocean), optional.
    u_mask : (n_lat, n_lon+1) u-face mask, optional.
    v_mask : (n_lat+1, n_lon) v-face mask, optional.

    Returns
    -------
    diss : (n_lat, n_lon[, nlev]) — non-negative KE-dissipation density [m²/s³]
        at cell centres.
    """
    is_3d = u.ndim == 3

    def _bcast(m, like):
        return m[..., jnp.newaxis] if (is_3d and m.ndim == like.ndim - 1) else m

    # Mask velocities exactly as vector_laplacian_cgrid does before div/curl.
    u_eff = u if u_mask is None else u * _bcast(u_mask, u)
    v_eff = v if v_mask is None else v * _bcast(v_mask, v)

    # Divergence at cell centres (masked) — same operator + masking as the vlap.
    div = divergence_cgrid(u_eff, v_eff, grid)
    if mask is not None:
        div = div * _bcast(mask, div)

    # Relative vorticity at vertices (masked at land-adjacent vertices) — same.
    zeta = curl_vertex_cgrid(u_eff, v_eff, grid)
    if mask is not None:
        vmask = compute_vertex_mask(mask, grid=grid)
        zeta = zeta * _bcast(vmask, zeta)

    # ζ² averaged from the four surrounding vertices to the cell centre (the
    # 4-corner stagger used by smagorinsky_viscosity_cgrid / strain interp).
    z2 = zeta ** 2
    if is_3d:
        z2_center = 0.25 * (z2[:-1, :-1, :] + z2[1:, :-1, :]
                            + z2[:-1, 1:, :] + z2[1:, 1:, :])
    else:
        z2_center = 0.25 * (z2[:-1, :-1] + z2[1:, :-1]
                            + z2[:-1, 1:] + z2[1:, 1:])

    # A_h at cell centres (1-D lat profile broadcasts over lon + levels).
    A_c = jnp.asarray(A_h_center)
    if A_c.ndim == 1:                       # (n_lat,) latitudinal profile
        A_c = A_c[:, jnp.newaxis, jnp.newaxis] if is_3d else A_c[:, jnp.newaxis]
    elif is_3d and A_c.ndim == 2:           # (n_lat, n_lon) -> add level axis
        A_c = A_c[:, :, jnp.newaxis]

    diss = A_c * (div ** 2 + z2_center)
    if mask is not None:
        diss = diss * _bcast(mask, diss)
    return diss


def flux_divergence_viscosity_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    A_h: float,
    *,
    cos_power: int = 0,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    want_dissipation: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Component-wise FLUX-DIVERGENCE lateral viscosity ``∇·(A_h∇u)``, ``∇·(A_h∇v)``.

    This is the Veros ``harmonic_friction`` operator (``veros/core/friction.py``):
    the standard C-grid harmonic friction applied SEPARATELY to each velocity
    component, with NO inter-component metric/curvature coupling.  It is the
    selectable ``lateral_viscosity_operator="flux_divergence"`` alternative to
    legoESM's default VECTOR Laplacian ``grad(div) − k×grad(curl)``
    (:func:`vector_laplacian_cgrid`).  On a sphere the two DIFFER for a vector
    field: the vector Laplacian carries curvature coupling between ``u`` and
    ``v`` (and the ``-2Ω`` / ``tanφ`` metric terms) that the component-wise form
    omits — so this is a genuine numerics choice (not a convention), hence a
    config option rather than a bridge transform.

    Discretisation (regular lat-lon; matches ``harmonic_friction`` term-for-term):

    *Zonal momentum* ``u`` at u-faces ``(n_lat, n_lon+1)``::

        flux_x[cell j]   = A_h·cosᵖ(φ) · (u[j]−u[j−1]) / dx_cell        (cell centre)
        flux_y[vtx i]    = A_h·cosᵖ(φ_v)·cos(φ_v) · (u[i]−u[i−1]) / dy_v (vertex)
        ∂ₜu[face j]     += (flux_x[j]−flux_x[j−1])/dx_u
                          + (flux_y[i+1]−flux_y[i])/(cos(φ)·dy_cell)

    *Meridional momentum* ``v`` at v-faces ``(n_lat+1, n_lon)`` — same structure with
    the staggering swapped (``flux_x`` at vertices, ``flux_y`` at cell centres).

    The extra ``cos(φ_v)`` in the meridional ``u``-flux and the ``1/cos(φ)`` in the
    divergence reproduce the spherical metric ``(1/(a²cosφ)) ∂φ(cosφ ∂φ)`` — exactly
    Veros's ``cosu``-weighted ``flux_north`` ÷ ``cost·dyt``.  The ``cosᵖ`` factor is
    Veros's ``enable_hor_friction_cos_scaling`` with ``hor_friction_cosPower``
    (legoESM ``A_h_lat_scaling`` / ``A_h_cos_power``): ``cos_power=0`` ⇒ no scaling
    (the un-scaled friction), ``cos_power=1`` ⇒ the ACC recipe's cos¹ Munk scaling.

    Reuses the SAME regular-grid metric expressions as :func:`divergence_cgrid` /
    :func:`gradient_x_cgrid` / :func:`gradient_y_cgrid` (``R·cosφ·dlon`` for zonal
    lengths, ``R·dlat`` for meridional, ``grid.area`` for the cell, ``_cos_lat_uv``
    for the face cosines) — NO duplicate numerics.  Masks are applied to the
    gradient stencils as ``maskᵢ·maskᵢ₋₁`` (free-slip: a land neighbour zeroes the
    flux, never a no-slip wall stress; Veros's optional ``enable_noslip_lateral``
    is OFF in ACC and not ported).

    Momentum conservation: the per-component flux divergence telescopes, and the
    fluxes vanish through land/walls (``mask`` zeroing + ``v=0`` poles), so the
    AREA-weighted domain integral of each component is zero to machine eps on a
    closed/periodic domain — the truth-tier conservation gate.

    Energy-consistent dissipation (``want_dissipation=True``).  Returns the
    POSITIVE-DEFINITE component-wise KE-removal density [m²/s³] at cell centres,
    ``K_diss_h = A_h·|∇u|² = 0.5·Σ(Δu·flux)`` — Veros ``calc_diss_u``/``calc_diss_v``.
    Built from the SAME ``flux_x``/``flux_y`` the tendency uses (``0.5·Δu·flux/Δx``
    averaged over the two faces straddling each cell), so the energy credited to the
    EKE source is exactly the energy the applied friction removes (paired with
    ``EKEConfig.kdiss_h_flux_form`` via the ``lateral_viscosity_operator`` dispatch).

    Parameters
    ----------
    u : (n_lat, n_lon+1[, nlev]) zonal velocity at u-faces [m/s].
    v : (n_lat+1, n_lon[, nlev]) meridional velocity at v-faces [m/s].
    grid : LatLonGrid (regular lat-lon; tripolar not yet supported — see below).
    A_h : float — base lateral viscosity [m²/s].
    cos_power : int — exponent on cos(lat) (Veros ``hor_friction_cosPower``).
        0 = no cos scaling; 1 = the ACC recipe's cos¹ Munk scaling.
    mask : (n_lat, n_lon) cell-centre ocean mask (1 = ocean), optional.
    u_mask : (n_lat, n_lon+1) u-face mask, optional.
    v_mask : (n_lat+1, n_lon) v-face mask, optional.
    want_dissipation : bool — also return the component-wise ``K_diss_h`` density.

    Returns
    -------
    visc_u : (n_lat, n_lon+1[, nlev]) — ``∇·(A_h∇u)`` [m/s²] at u-faces (face-masked).
    visc_v : (n_lat+1, n_lon[, nlev]) — ``∇·(A_h∇v)`` [m/s²] at v-faces (face-masked).
    kdiss_h_cell : (n_lat, n_lon[, nlev]) non-negative KE-dissipation density
        [m²/s³] at cell centres, or ``None`` when ``want_dissipation=False``.
    """
    if is_tripolar(grid):
        raise ValueError(
            "lateral_viscosity_operator='flux_divergence' is not yet implemented "
            "on tripolar grids (the vertex/face metric handling needs the 2D "
            "dx_v/dy_u fields, like the flux-form momentum advection). Use "
            "'vector_laplacian' on tripolar, or extend this operator."
        )
    is_3d = u.ndim == 3

    def _bcast_mask(m, like):
        return m[..., jnp.newaxis] if (is_3d and m.ndim == like.ndim - 1) else m

    # --- regular-grid metrics (identical expressions to divergence_cgrid) ---
    R = grid.radius
    dlon = grid.dlon
    cos_u, cos_v = _cos_lat_uv(grid)                # (n_lat,), (n_lat+1,)
    dx_cell = R * cos_u * dlon                       # (n_lat,) zonal cell width at u-lat
    dx_u = dx_cell                                   # u-faces at cell-centre latitudes
    dy_cell = grid.dy * 0.5                          # (n_lat,) meridional cell height
    # v-face meridional spacing (distance between adjacent cell centres in lat).
    dy_v_int = 0.5 * (dy_cell[1:] + dy_cell[:-1])    # (n_lat-1,)

    # cos-power weights at u-lat (cell centres) and v-lat (interfaces).
    fxa_u = cos_u ** cos_power                        # (n_lat,)
    fxa_v = cos_v ** cos_power                        # (n_lat+1,)

    # face masks default to all-ocean (1.0); promote a 2-D face mask to the
    # velocity rank so the gradient-stencil products broadcast over the level axis
    # (the real call site passes 2-D ``u_mask``/``v_mask`` with 3-D velocities).
    if u_mask is None:
        um = jnp.ones_like(u)
    elif is_3d and u_mask.ndim == 2:
        um = u_mask[:, :, jnp.newaxis]
    else:
        um = u_mask
    if v_mask is None:
        vm = jnp.ones_like(v)
    elif is_3d and v_mask.ndim == 2:
        vm = v_mask[:, :, jnp.newaxis]
    else:
        vm = v_mask

    # ============================ ZONAL momentum (u) ============================
    # flux_x at CELL CENTRES: A_h·cosᵖ·∂u/∂x. Cell j sits between u-faces j (west)
    # and j+1 (east), so the cell-centred ∂u/∂x is the consecutive u-face difference
    # along the lon axis — matching Veros's flux_east between adjacent u-points.
    # u: (n_lat, n_lon+1). Adjacent-u-face difference -> (n_lat, n_lon) at cell centres.
    du_dx_cell = (u[:, 1:, ...] - u[:, :-1, ...])
    # divide by dx_cell (per lat); mask with the two u-faces straddling the cell.
    mu_cell = um[:, 1:, ...] * um[:, :-1, ...]                       # (n_lat, n_lon[, nlev])
    flux_x_u = (
        A_h
        * (fxa_u[:, None, None] if is_3d else fxa_u[:, None])
        * du_dx_cell
        / (dx_cell[:, None, None] if is_3d else dx_cell[:, None])
        * mu_cell
    )                                                                # (n_lat, n_lon[, nlev])

    # flux_y at VERTICES: A_h·cosᵖ(φ_v)·cos(φ_v)·∂u/∂y. u-rows i and i+1 straddle
    # vertex row i+1 (interior). Build interior vertex flux, pad poles with zero.
    du_dy_int = (u[1:, ...] - u[:-1, ...])                           # (n_lat-1, n_lon+1[, nlev])
    mu_vtx_int = um[1:, ...] * um[:-1, ...]
    fyw_int = fxa_v[1:-1] * cos_v[1:-1]                              # (n_lat-1,) cosᵖ⁺¹ weight
    flux_y_u_int = (
        A_h
        * (fyw_int[:, None, None] if is_3d else fyw_int[:, None])
        * du_dy_int
        / (dy_v_int[:, None, None] if is_3d else dy_v_int[:, None])
        * mu_vtx_int
    )                                                                # (n_lat-1, n_lon+1[, nlev])
    zrow_u = jnp.zeros_like(u[:1, ...])
    flux_y_u = jnp.concatenate([zrow_u, flux_y_u_int, zrow_u], axis=0)  # (n_lat+1, n_lon+1[, nlev])

    # divergence to u-faces. Zonal: (flux_x[cell j] − flux_x[cell j−1])/dx_u, periodic
    # in lon (face j is between cell j−1 (west) and cell j (east); face 0 wraps).
    net_x_u_core = flux_x_u - jnp.roll(flux_x_u, 1, axis=1)          # (n_lat, n_lon[, nlev])
    net_x_u = jnp.concatenate([net_x_u_core, net_x_u_core[:, :1, ...]], axis=1)
    net_x_u = net_x_u / (dx_u[:, None, None] if is_3d else dx_u[:, None])
    # Meridional: (flux_y[vtx i+1] − flux_y[vtx i]) / (cos(φ)·dy_cell) at u-face i.
    net_y_u = (flux_y_u[1:, ...] - flux_y_u[:-1, ...])              # (n_lat, n_lon+1[, nlev])
    cdy_u = cos_u * dy_cell                                          # (n_lat,)
    net_y_u = net_y_u / (cdy_u[:, None, None] if is_3d else cdy_u[:, None])
    visc_u = (net_x_u + net_y_u)
    if u_mask is not None:
        visc_u = visc_u * _bcast_mask(u_mask, visc_u)

    # ========================= MERIDIONAL momentum (v) =========================
    # flux_x at VERTICES: A_h·cosᵖ(φ_v)·∂v/∂x. v-cols j−1 and j straddle vertex col j
    # (periodic in lon). Build at lon-faces (vertices) via roll.
    v_west = jnp.roll(v, 1, axis=1)
    dv_dx_vtx = (v - v_west)                                         # (n_lat+1, n_lon[, nlev]) at vertices
    mv_vtx = vm * jnp.roll(vm, 1, axis=1)
    # zonal spacing at v-lat between adjacent v-cell centres: R·cos(φ_v)·dlon.
    dx_v = R * cos_v * dlon                                          # (n_lat+1,)
    flux_x_v = (
        A_h
        * (fxa_v[:, None, None] if is_3d else fxa_v[:, None])
        * dv_dx_vtx
        / (dx_v[:, None, None] if is_3d else dx_v[:, None])
        * mv_vtx
    )                                                                # (n_lat+1, n_lon[, nlev]) at lon-faces

    # flux_y at CELL CENTRES: A_h·cosᵖ(φ)·cos(φ)·∂v/∂y. v-rows i (south) and i+1
    # (north) straddle cell i. Build at cell centres (n_lat rows).
    dv_dy_cell = (v[1:, ...] - v[:-1, ...])                          # (n_lat, n_lon[, nlev]) at cell centres
    mv_cell = vm[1:, ...] * vm[:-1, ...]
    fyw_cell = fxa_u * cos_u                                          # (n_lat,) cosᵖ⁺¹ at cell lat
    flux_y_v = (
        A_h
        * (fyw_cell[:, None, None] if is_3d else fyw_cell[:, None])
        * dv_dy_cell
        / (dy_cell[:, None, None] if is_3d else dy_cell[:, None])
        * mv_cell
    )                                                                # (n_lat, n_lon[, nlev])

    # divergence to v-faces. Zonal: (flux_x[vtx j+1] − flux_x[vtx j]) / (cos(φ_v)·dx_v')
    # at v-face (cell centre j): E face = vertex j+1, W face = vertex j (periodic).
    net_x_v = (jnp.roll(flux_x_v, -1, axis=1) - flux_x_v)          # (n_lat+1, n_lon[, nlev])
    cdx_v = cos_v * R * dlon                                         # (n_lat+1,) = cos(φ_v)·R·dlon
    net_x_v = net_x_v / (cdx_v[:, None, None] if is_3d else cdx_v[:, None])
    # Meridional: (flux_y[cell i] − flux_y[cell i−1]) / (cos(φ_v)·dy_v) at v-face i
    # (interior); poles are walls -> zero tendency.  The 1/cos(φ_v) is the spherical
    # metric prefactor of the meridional Laplacian (Veros divides ``dv_mix`` by
    # ``cosu·dyu`` — friction.py:614-615), evaluated at the INTERIOR v-face latitudes
    # ``cos_v[1:-1]`` where the v-tendency lives.  (Mirrors the u-meridional ÷(cos_u·
    # dy_cell) above; without it the v-meridional friction is too weak by cos(φ) and
    # breaks both the Veros term-match and momentum conservation on the sphere.)
    cdy_v_int = dy_v_int * cos_v[1:-1]                              # (n_lat-1,)
    net_y_v_int = (flux_y_v[1:, ...] - flux_y_v[:-1, ...])         # (n_lat-1, n_lon[, nlev])
    net_y_v_int = net_y_v_int / (cdy_v_int[:, None, None] if is_3d else cdy_v_int[:, None])
    zlon_v = jnp.zeros_like(v[:1, ...])
    net_y_v = jnp.concatenate([zlon_v, net_y_v_int, zlon_v], axis=0)
    # zero the zonal part at the pole v-faces too (walls).
    net_x_v = net_x_v.at[0, ...].set(0.0).at[-1, ...].set(0.0)
    visc_v = (net_x_v + net_y_v)
    if v_mask is not None:
        visc_v = visc_v * _bcast_mask(v_mask, visc_v)

    if not want_dissipation:
        return visc_u, visc_v, None

    # ===================== component-wise K_diss_h (≥ 0) =====================
    # Veros calc_diss_u/v: the KE removed by the friction = the FLUX-FORM energy
    # ``∫ A_h·cosᵖ·|∇u|² dA = Σ_faces flux·Δu·(face length)`` (positive-definite by
    # construction).  We return it as a t-cell DENSITY [m²/s³] such that
    # ``density·cell_area`` equals the energy attributed to that cell — so the
    # downstream EKE source (``harmonic_lateral_kediss_eke_source``, which integrates
    # the density over the cell volume) credits exactly the KE the friction removed.
    #
    # At each FLUX location the energy rate × area is ``flux·Δu·L_perp`` (m⁴/s³),
    # where ``L_perp`` is the face length the flux crosses.  Each flux location is
    # shared by two t-cells, so attribute HALF to each, then divide by the t-cell
    # area to form the density.  (Energy-exact: verified ``Σ density·area`` equals
    # ``−Σ u·visc·area_u − v·visc·area_v`` to machine eps on a closed domain.)
    area_c = grid.area                                               # (n_lat, n_lon)
    area_b = area_c[:, :, jnp.newaxis] if is_3d else area_c
    inv_area = jnp.where(area_b > 0.0, 1.0 / jnp.maximum(area_b, 1.0e-30), 0.0)

    # u-ZONAL: flux_x_u at scalar centre [i,j] crosses the u-face of meridional
    # length L = cos(φ)·dy_cell.  Energy E_zx[i,j] = flux_x_u·du_dx_cell·L sits at
    # cell centre [i,j] (it IS the t-cell), so it maps wholly to t-cell [i,j].
    Lzx = cos_u * dy_cell                                            # (n_lat,)
    E_zx = (
        flux_x_u * du_dx_cell
        * (Lzx[:, None, None] if is_3d else Lzx[:, None])
    )                                                                # (n_lat, n_lon[, nlev]) at t-cells
    # u-MERID: flux_y_u_int at interior vertex (between t-cells i and i+1) crosses
    # the vertex face of zonal length L = cos(φ_v)·R·dlon = dx_v.  Energy at the
    # vertex; split half to t-cell i (south) and half to i+1 (north).  The vertex
    # also sits between u-faces j and j+1 in lon, i.e. between t-cells j-? — no: the
    # u-merid flux is at u-face longitudes (n_lon+1), straddling t-cell lon-columns
    # j-1 and j.  So also split half in lon to the two adjacent t-cell columns.
    Lyu = (cos_v[1:-1] * R * dlon)                                   # (n_lat-1,) vertex zonal face length
    E_yu_vtx = (
        flux_y_u_int * du_dy_int
        * (Lyu[:, None, None] if is_3d else Lyu[:, None])
    )                                                                # (n_lat-1, n_lon+1[, nlev]) at interior vertices
    # lon: vertex at u-face column k straddles t-cell columns k-1 and k (periodic).
    # Distribute 0.5 to each lon-neighbour: t-cell column m gets 0.5*(E[:,m] + E[:,m+1]).
    E_yu_loncell = 0.5 * (E_yu_vtx[:, :-1, ...] + E_yu_vtx[:, 1:, ...])  # (n_lat-1, n_lon[, nlev])
    # lat: interior-vertex row r (=between t-rows r and r+1) splits 0.5 to each.
    zr = jnp.zeros_like(E_yu_loncell[:1])
    E_yu_padlat = jnp.concatenate([zr, E_yu_loncell, zr], axis=0)    # (n_lat+1, n_lon[, nlev])
    E_yu_cell = 0.5 * (E_yu_padlat[:-1, ...] + E_yu_padlat[1:, ...]) # (n_lat, n_lon[, nlev]) at t-cells
    diss_u_cell = (E_zx + E_yu_cell) * inv_area

    # v-MERID: flux_y_v at scalar centre [i,j] (it IS the t-cell) crosses the
    # cell's lat-face of zonal length L = cos(φ)·R·dlon = dx_cell.  Maps wholly to
    # t-cell [i,j].
    Lyv = dx_cell                                                    # (n_lat,) = cos_u·R·dlon
    E_yv = (
        flux_y_v * dv_dy_cell
        * (Lyv[:, None, None] if is_3d else Lyv[:, None])
    )                                                                # (n_lat, n_lon[, nlev]) at t-cells
    # v-ZONAL: flux_x_v at vertex row i, lon-face column j (between t-cells j-1, j in
    # lon, and it is at a v-face latitude i, between t-rows i-1 and i).  Crosses the
    # vertex meridional face of length L = cos(φ_v)·dy_v.  Split 0.5 in lat AND 0.5
    # in lon to the four surrounding t-cells (here combined: 0.5 lat × full lon then
    # 0.5 lon).  Vertex is at v-cell longitudes (n_lon) and v-face latitudes (n_lat+1).
    dy_v_full = jnp.concatenate([dy_cell[:1], dy_v_int, dy_cell[-1:]])  # (n_lat+1,)
    Lxv = cos_v * dy_v_full                                          # (n_lat+1,)
    E_xv_vtx = (
        flux_x_v * dv_dx_vtx
        * (Lxv[:, None, None] if is_3d else Lxv[:, None])
    )                                                                # (n_lat+1, n_lon[, nlev]) at vertices
    # lon: vertex at lon-face column j straddles t-cell columns j-1 and j (periodic).
    E_xv_loncell = 0.5 * (E_xv_vtx + jnp.roll(E_xv_vtx, -1, axis=1)) # (n_lat+1, n_lon[, nlev])
    # lat: vertex rows i and i+1 straddle t-cell i; split 0.5 each.
    E_xv_cell = 0.5 * (E_xv_loncell[:-1, ...] + E_xv_loncell[1:, ...])  # (n_lat, n_lon[, nlev])
    diss_v_cell = (E_yv + E_xv_cell) * inv_area

    kdiss_h_cell = diss_u_cell + diss_v_cell
    if mask is not None:
        kdiss_h_cell = kdiss_h_cell * _bcast_mask(mask, kdiss_h_cell)
    # ≥ 0 by construction (Δu·flux/Δx = A_h·cosᵖ·(∂u)² ≥ 0); guard tiny negatives
    # from float round-off only (NOT the clamp the dynamical form needs).
    kdiss_h_cell = jnp.maximum(kdiss_h_cell, 0.0)
    return visc_u, visc_v, kdiss_h_cell


def no_slip_sidedrag_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
    A_h: float,
    *,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    vertex_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""MITgcm no-slip lateral side-drag (``mom_u_sidedrag`` / ``mom_v_sidedrag``).

    The free-slip lateral-viscosity operators (``vector_laplacian_cgrid`` /
    ``flux_divergence_viscosity_cgrid``) zero the viscous flux across a wall face,
    i.e. impose ``∂(tangential u)/∂n = 0`` (free-slip). MITgcm's default
    ``no_slip_sides=.TRUE.`` instead imposes zero tangential velocity at the wall,
    which adds a drag body force from the wall stress (``mom_u_sidedrag.F``)::

        G^u_drag = -(2/Δy) A_h u   at u-cells touching a meridional (N/S) wall
        G^v_drag = -(2/Δx) A_h v   at v-cells touching a zonal     (E/W) wall

    Discretely (``sideDragFactor=2``), per CLOSED side
    ``closed = hFacW − hFacZ = face_open·(1 − vertex_open)``::

        du_drag[j,i] = -(closedS + closedN) · 2 A_h u[j,i] / dy_u[j,i]²
        dv_drag[j,i] = -(closedW + closedE) · 2 A_h v[j,i] / dx_v[j,i]²

    with the wall corners from :func:`compute_vertex_mask` (vertex ``(J,I)``
    touches cells ``(J-1,I-1),(J-1,I),(J,I-1),(J,I)``): u-point ``(j,i)`` has south
    vertex ``vmask[j,i]`` / north ``vmask[j+1,i]``; v-point ``(j,i)`` has west
    ``vmask[j,i]`` / east ``vmask[j,i+1]``. This is the wall-tangential viscous
    stress ``A_h·(u−0)/(Δy/2)`` distributed over the cell width ``Δy``. **This is
    ADDED to** (not a replacement for) the free-slip flux operator, exactly as
    MITgcm adds ``MOM_U_SIDEDRAG`` on top of ``MOM_U_DEL2U``.

    Exact for uniform-Cartesian grids (the beta-plane oracle regime, where
    ``dy_u``/``dx_v`` are constant and ``rAw = dx·dy``). Returns ``(du_drag,
    dv_drag)`` at the u-/v-faces, masked.

    Boundary note: :func:`compute_vertex_mask` zeros the north/south polar vertex
    rows (the standard pole wall BC), so a meridional domain boundary is treated
    as a wall and its edge u-row receives the side-drag even with no explicit
    land — correct for a bounded/closed basin (the gyre), but a y-periodic domain
    should keep ``lateral_side_bc="free_slip"``.
    """
    if not (hasattr(grid, "dy_u") and hasattr(grid, "dx_v")):
        raise ValueError(
            "no_slip_sidedrag_cgrid requires a LatLonCGridGeometry with dy_u/dx_v "
            "metric fields (call ensure_geometry on the grid first)."
        )
    if mask is None and vertex_mask is None:
        raise ValueError("no_slip_sidedrag_cgrid needs either mask or vertex_mask")
    vmask = vertex_mask if vertex_mask is not None else compute_vertex_mask(mask, grid=grid)
    vmask = vmask.astype(u.dtype)

    is_3d = u.ndim == 3

    def _b(a, like):
        return a[..., jnp.newaxis] if (is_3d and a.ndim == like.ndim - 1) else a

    # --- u side-drag: closed N/S sides (a meridional wall above/below) ---
    v_south = vmask[:-1, :]      # (n_lat, n_lon+1): south vertex of u-point (j,i)
    v_north = vmask[1:, :]       # (n_lat, n_lon+1): north vertex
    closed_s = u_mask * (1.0 - v_south)
    closed_n = u_mask * (1.0 - v_north)
    # Guard 1/Δ² against zero-length faces (e.g. the polar boundary v-faces on a
    # spherical grid where dx_v = R·cos(lat_v)·dlon → 0): those faces are masked
    # out (closed/u_mask = 0) so the drag is zero there, but an unguarded 1/0=inf
    # times the 0 mask is NaN.  On uniform-metric grids (beta-plane) Δ>0 so this
    # is bit-identical.
    _dy_u = grid.dy_u.astype(u.dtype)
    inv_dy2_u = jnp.where(_dy_u > 0.0, 1.0 / (_dy_u ** 2), 0.0)
    du_drag = -(2.0 * A_h) * _b(closed_s + closed_n, u) * u * _b(inv_dy2_u, u)
    du_drag = du_drag * _b(u_mask, du_drag)

    # --- v side-drag: closed E/W sides (a zonal wall to left/right) ---
    v_west = vmask[:, :-1]       # (n_lat+1, n_lon): west vertex of v-point (j,i)
    v_east = vmask[:, 1:]        # (n_lat+1, n_lon): east vertex
    closed_w = v_mask * (1.0 - v_west)
    closed_e = v_mask * (1.0 - v_east)
    # Guard against zero-length v-faces (polar boundary: dx_v → 0); see the u note.
    _dx_v = grid.dx_v.astype(v.dtype)
    inv_dx2_v = jnp.where(_dx_v > 0.0, 1.0 / (_dx_v ** 2), 0.0)
    dv_drag = -(2.0 * A_h) * _b(closed_w + closed_e, v) * v * _b(inv_dx2_v, v)
    dv_drag = dv_drag * _b(v_mask, dv_drag)

    return du_drag, dv_drag


def vector_bilaplacian_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    vertex_mask: jnp.ndarray | None = None,
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
        u, v, grid, mask=mask, u_mask=u_mask, v_mask=v_mask,
        vertex_mask=vertex_mask)
    bilap_u, bilap_v = vector_laplacian_cgrid(
        vlap_u, vlap_v, grid, mask=mask, u_mask=u_mask, v_mask=v_mask,
        vertex_mask=vertex_mask)
    return bilap_u, bilap_v


def nemo_lateral_viscosity_coefficients(
    grid: LatLonGrid, half_UM: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""NEMO ``ldf_c2d`` viscosity coefficients ``ahmt``/``ahmf`` (nn_ahm_ijk_t=20).

    Faithful transcription of NEMO 5.0.2 ``ldf_c2d`` (``src/OCE/LDF/ldfc1d_c2d.F90``
    L138-139) for the laplacian ``nn_ahm_ijk_t=20`` case (``ldf_dyn_init``
    ``zUfac = ½·rn_Uv``, ``inn=1``)::

        ahmt(T) = ½·rn_Uv · MAX(e1t, e2t)
        ahmf(F) = ½·rn_Uv · MAX(e1f, e2f)

    with ``rn_Uv`` the lateral viscous velocity [m/s] and ``e1``/``e2`` the zonal /
    meridional grid scales.  ``half_UM = ½·rn_Uv``.  Unlike a single
    ``A_h·cos(φ)^p`` scalar applied OUTSIDE the vector Laplacian, this coefficient
    is defined at the T- and F-points so it can be embedded INSIDE the div/curl
    (see :func:`nemo_ldf_lap_viscosity_cgrid`), matching NEMO's
    ``grad(ahmt·div) − curl(ahmf·curl)``.

    On a uniform-Δφ lat-lon grid ``e2 = R·Δφ`` is latitude-independent, so
    ``MAX(e1,e2) = e2`` at high latitude and ``ahmt`` does NOT shrink with cos(φ)
    (unlike ``A_h·cos φ``).  On a conformal Mercator grid ``e1 ≈ e2`` at every row
    (isotropic cells in the continuum), so ``MAX(e1,e2) ≈ e1 = R·Δλ·cos φ`` and the
    two forms agree in magnitude to ``O(Δλ²)`` — the placement (embedded vs outside)
    is then the dominant difference.  NB the DISCRETE metrics differ slightly:
    ``e2t = R·(φ_face[j+1]−φ_face[j])`` (a face difference) vs
    ``e1t = R·Δλ·cos φ_c`` (a centre cosine), so ``MAX`` picks ``e2`` on a fair
    fraction of rows and the agreement is ``~2e-5`` worst-case on the DINO grid
    (machine-level only exactly at the equator), not machine-zero everywhere.

    Parameters
    ----------
    grid : LatLonGrid or LatLonCGridGeometry
        Must expose the C-grid metric fields ``dx_u``/``dy_u`` (e1u/e2u at the
        u-face = cell-centre latitude → e1t/e2t) and ``dx_v``/``dy_v`` (e1v/e2v at
        the v-face latitude → e1f/e2f).
    half_UM : float
        ``½·rn_Uv`` [m/s].

    Returns
    -------
    ahmt : (n_lat,)      viscosity at T-points [m²/s].
    ahmf : (n_lat+1,)    viscosity at F-points [m²/s].
    """
    if not (hasattr(grid, "dx_u") and hasattr(grid, "dy_u")
            and hasattr(grid, "dx_v") and hasattr(grid, "dy_v")):
        raise ValueError(
            "nemo_lateral_viscosity_coefficients requires a LatLonCGridGeometry "
            "with dx_u/dy_u/dx_v/dy_v metric fields (call ensure_geometry first)."
        )
    # e1t/e2t at the u-face (cell-centre) latitude; e1f/e2f at the v-face latitude.
    # dx_* / dy_* are lon-uniform on a (non-tripolar) lat-lon grid, so column 0
    # carries the full latitudinal metric.
    e1t = grid.dx_u[:, 0]
    e2t = grid.dy_u[:, 0]
    e1f = grid.dx_v[:, 0]
    e2f = grid.dy_v[:, 0]
    ahmt = half_UM * jnp.maximum(e1t, e2t)
    ahmf = half_UM * jnp.maximum(e1f, e2f)
    return ahmt, ahmf


def nemo_ldf_lap_viscosity_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    ahmt: jnp.ndarray,
    ahmf: jnp.ndarray,
    *,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    vertex_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""NEMO ``dyn_ldf_lev_lap`` Laplacian viscosity with the coefficient EMBEDDED
    inside the div/curl (``grad_h(ahmt·div_h U) − curl_h(ahmf·curl_z U)``).

    Faithful transcription of NEMO 5.0.2 ``dynldf_lev.F90::dynldf_lev_lap`` +
    ``dynldf_lev_rot_scheme.h90`` (the ``np_typ_rot`` vorticity-divergence
    operator).  NEMO forms::

        zdiv(T) = ahmt · div_h(U)                      (coeff on the T-point divergence)
        zcur(F) = ahmf · curl_z(U)                     (coeff on the F-point vorticity)
        pu += + ∂_x(zdiv) − ∂_y(zcur)/…               (grad of div  −  curl of curl)
        pv += + ∂_y(zdiv) + ∂_x(zcur)/…

    i.e. ``grad(ahmt·div) − k×grad(ahmf·curl)``.  This DIFFERS from applying a
    single latitude scalar OUTSIDE the whole vector Laplacian
    (``A_h(φ)·[grad(div) − k×grad(curl)]``) by the coefficient-gradient cross terms
    ``∇(ahmt)·div`` and ``∇(ahmf)×curl`` — the node-14 placement fix.

    Reuses the SAME shared C-grid operators (``divergence_cgrid``,
    ``gradient_x/y_cgrid``, ``curl_vertex_cgrid``, ``gradient_curl_to_u/v``) and the
    SAME face/vertex masking as :func:`vector_laplacian_cgrid`, so with a CONSTANT
    ``ahmt = ahmf = A_h`` this returns exactly ``A_h · vector_laplacian_cgrid`` (a
    truth-tier reduction test).  The coefficient is already embedded, so the caller
    adds the returned tendency directly (NEMO's ``+`` sign; no outer ``A_h``).

    NOTE (documented deviation, consistent with :func:`vector_laplacian_cgrid`):
    NEMO weights the div/curl by the layer thickness ``e3`` (``e3t``/``e3f``); the
    shared 2-D ``divergence_cgrid``/``curl_vertex_cgrid`` used here do not.  This is
    the SAME thickness treatment the verified legoESM vector Laplacian uses (DINO
    wiring node 14: div-curl structure + magnitude already certified), so the ONLY
    change vs the current path is the embedded latitude-varying coefficient.

    Parameters
    ----------
    u, v : face velocities (2-D or 3-D).
    grid : LatLonGrid.
    ahmt : (n_lat,)    T-point viscosity coefficient [m²/s].
    ahmf : (n_lat+1,)  F-point viscosity coefficient [m²/s].
    mask, u_mask, v_mask, vertex_mask : the usual C-grid masks.

    Returns
    -------
    visc_u, visc_v : the viscous momentum tendency (coefficient embedded).
    """
    is_3d = u.ndim == 3

    def _bm(m):
        # broadcast a 2-D face/cell/vertex mask over the trailing level axis
        return m[..., jnp.newaxis] if is_3d else m

    def _bc(c):
        # broadcast a (n_lat,) or (n_lat+1,) latitude coefficient over lon [, lev]
        return c[:, None, None] if is_3d else c[:, None]

    u_eff = u if u_mask is None else u * _bm(u_mask)
    v_eff = v if v_mask is None else v * _bm(v_mask)

    # 1. Divergence at T-points, scale by ahmt (NEMO zdiv = ahmt·div)
    div = divergence_cgrid(u_eff, v_eff, grid)
    if mask is not None:
        div = div * _bm(mask)
    div_scaled = div * _bc(ahmt)
    grad_div_u = gradient_x_cgrid(div_scaled, grid)
    grad_div_v = gradient_y_cgrid(div_scaled, grid)

    # 2. Relative vorticity at F-points (vertices), scale by ahmf (NEMO zcur = ahmf·curl)
    zeta = curl_vertex_cgrid(u_eff, v_eff, grid)          # (n_lat+1, n_lon+1[, nlev])
    if mask is not None:
        vmask = (vertex_mask if vertex_mask is not None
                 else compute_vertex_mask(mask, grid=grid))
        zeta = zeta * _bm(vmask)
    zeta_scaled = zeta * _bc(ahmf)
    grad_curl_u = gradient_curl_to_u(zeta_scaled, grid)
    grad_curl_v = gradient_curl_to_v(zeta_scaled, grid)

    # 3. grad(ahmt·div) − k×grad(ahmf·curl); same signs as vector_laplacian_cgrid.
    visc_u = grad_div_u - grad_curl_u
    visc_v = grad_div_v + grad_curl_v

    if u_mask is not None:
        visc_u = visc_u * _bm(u_mask)
    if v_mask is not None:
        visc_v = visc_v * _bm(v_mask)
    return visc_u, visc_v


def flux_divergence_bilaplacian_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    cos_power: int = 0,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""COMPONENT-WISE biharmonic ``∇⁴(u, v)`` = ``∇²(∇²(u, v))`` per component.

    Applies the component harmonic friction :func:`flux_divergence_viscosity_cgrid`
    (with unit coefficient) TWICE — the scale-selective analogue of the harmonic
    ``flux_divergence`` Laplacian, and the FAITHFUL form of MITgcm's ``viscA4``
    biharmonic (``useStrainTensionVisc=.FALSE.`` ⇒ a per-component ``del4``, NOT
    the vector ``grad(div) − curl(curl)`` biharmonic of
    :func:`vector_bilaplacian_cgrid`).  The biharmonic tendency is
    ``∂ₜu = −B_h·∇⁴u`` (same sign convention as ``vector_bilaplacian_cgrid``).

    WHY a separate operator: the vector biharmonic's ``grad(div)``/``curl(curl)``
    composition is ill-scaled on a uniform-Cartesian / near-degenerate (1-column)
    C-grid — it returns a value ~``dx⁴`` too large (an unphysical ``O(u)`` instead
    of ``O(u/dx⁴)``) and blows the integration up within a few steps, whereas this
    component form telescopes a clean 5-point ``∇²`` twice and stays at the correct
    ``u/dx⁴`` scale (front_relax baroclinic oracle).

    Reuses ``flux_divergence_viscosity_cgrid`` verbatim (its metric, masking, and
    cos-scaling), so momentum conservation + free-slip wall handling are inherited.
    """
    lap_u, lap_v, _ = flux_divergence_viscosity_cgrid(
        u, v, grid, 1.0, cos_power=cos_power,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
    )
    bilap_u, bilap_v, _ = flux_divergence_viscosity_cgrid(
        lap_u, lap_v, grid, 1.0, cos_power=cos_power,
        mask=mask, u_mask=u_mask, v_mask=v_mask,
    )
    return bilap_u, bilap_v


def _vertex_dual_area_interior(lat, radius, dlon):
    """Raw vertex dual-cell area ``R**2 * dlon * |Δsin(lat)|`` of shape
    ``(n_lat+1,)`` (#515 consolidation).

    The single source for the interior of the regular (non-tripolar) lat-lon
    vertex/q-cell area, previously recomputed verbatim in ``vertex_area_cgrid``,
    ``strain_rate_cgrid`` and the Smagorinsky ``A_vertex`` floor.  Callers keep
    their OWN pole handling (zero pole rows vs a ``1e-30`` floor) and pass an
    appropriately-typed ``lat`` (stored dtype, or ``result_type(float)`` for the
    #516 working-precision path), so each extraction is byte-identical to the
    former inline recompute.
    """
    sin_lat = jnp.sin(lat)
    sin_ext = jnp.pad(sin_lat, (1, 1), constant_values=(-1.0, 1.0))
    return radius**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])


def vface_zonal_cos_lat(grid: LatLonGrid) -> jnp.ndarray:
    """Canonical ``cos(lat_v)`` at the ``n_lat+1`` v-face latitudes (#516).

    The zonal length of a v-face is ``dx_v = R·cos(lat_v)·dlon``.  This
    is the ONE source for that ``cos(lat_v)`` across every regular
    (non-tripolar) lat-lon C-grid operator that recomputes a v-face
    zonal metric — the strain/stress adjoint pair, the cell-divergence,
    the velocity-divergence split (w-divergence) and the flux-form
    momentum advection — so they share an identical discretisation and
    the discrete adjointness / mass-consistency invariants hold to
    machine precision on a non-uniform-dlat (Mercator/stretched) grid.

    Construction (bit-matches the canonical core ``divergence_cgrid``):

    * interior faces ``j = 1 … n_lat-1`` →
      ``cos(0.5·(lat[j-1] + lat[j]))``  (cos-OF-interface, NOT the
      mean-of-cos ``0.5·(cos lat[j-1] + cos lat[j])`` — they differ at
      O(dlat²) on a stretched grid because ``cos(½(a+b)) ≠
      ½(cos a + cos b)``);
    * the two polar walls ``j = 0`` and ``j = n_lat`` → EXACTLY ``0``
      (transport metric: no meridional flux through the pole wall).

    The construction reuses the SAME backend-dispatched lat padding
    (:func:`pad_with_pole_bc_lat`) and pole-zeroing
    (:func:`zero_polar_lat_ends`) the core divergence uses, so under MPI
    an interior partition cut keeps the neighbour-sendrecv'd face metric
    (only a rank that OWNS a pole zeros that end) and the result is
    bit-identical to ``divergence_cgrid``'s v-face metric.

    NOTE — pole convention.  This metric is exactly ``0`` at the poles.
    Operators that DIVIDE by a v-face-related dual area at the poles
    (``strain_rate_cgrid`` / ``stress_divergence_cgrid`` divide by the
    vertex dual area) must keep their own ``1e-10``-clamped denominator
    to stay finite — they use this helper ONLY for the adjoint-pair-
    critical v-face zonal *length* (a pure multiplicative weight, never a
    divisor), and v ≡ 0 at the polar walls regardless.

    Parameters
    ----------
    grid : LatLonGrid
        Regular or Mercator lat-lon grid (the tripolar branch reads the
        stored 2-D ``grid.dx_v`` instead and never calls this helper).

    Returns
    -------
    cos_lat_v : (n_lat+1,)
        cos(lat) at the v-face latitudes; interior cos-of-interface, the
        two polar walls exactly ``0``.
    """
    from legoesm.grids.halo_latlon import (
        pad_with_pole_bc_lat,
        zero_polar_lat_ends,
    )
    if reads_stored_vface_metric(grid):
        # Rich geometry with an explicit stored v-face metric (#514): READ it
        # instead of recomputing cos(grid.lat_v).  On a Cartesian beta-plane the
        # recompute is WRONG (the pseudo-lat is a nonzero y_c/radius while the
        # stored dx_v is the uniform dx_m); on a spherical rich geometry it is
        # merely redundant.  dx_v is lon-uniform for every geometry that reaches
        # this helper (tripolar reads the 2D dx_v directly and never calls here),
        # so column 0 carries the full metric.  cos(lat_v) = dx_v / (R*dlon):
        # beta-plane -> exactly 1 (R*dlon == dx_m sentinel).  zero_polar_lat_ends
        # enforces this helper's poles==0 CONTRACT (the beta-plane stored dx_v is
        # a uniform dx_m at every row, NOT pole-zeroed; spherical stored dx_v is
        # already 0 there so the clamp is a no-op) — stress/viscosity divide by
        # this at the walls and rely on the zero.
        cos_lat_v = grid.dx_v[:, 0] / (grid.radius * grid.dlon)
        return zero_polar_lat_ends(
            jnp.asarray(cos_lat_v, dtype=jnp.result_type(float))
        )
    # Compute the interface latitudes in the working float precision
    # (float64 under JAX_ENABLE_X64, else float32).  ``grid.lat`` is
    # stored float32, but ``cos(½(lat[j-1]+lat[j]))`` is the
    # adjoint-/mass-critical metric — evaluating it at the model's
    # working precision keeps the four sites that share this helper
    # consistent to machine precision (the single-precision cos-of-
    # interface only agrees to ~1e-7, breaking the x64 invariants).
    lat = jnp.asarray(grid.lat, dtype=jnp.result_type(float))
    lat_pad = pad_with_pole_bc_lat(
        lat, halo=1, south_value=0.0, north_value=0.0,
    )
    lat_v = 0.5 * (lat_pad[:-1] + lat_pad[1:])          # (n_lat+1,)
    return zero_polar_lat_ends(jnp.cos(lat_v))          # (n_lat+1,)


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
        cos(lat) at v-face latitudes.  The INTERIOR faces are the
        canonical single-source :func:`vface_zonal_cos_lat` (cos-of-
        interface), NOT the legacy mean-of-cos — see #516: routing every
        recompute through one helper restores strain↔stress adjointness
        and div↔advection mass-consistency on non-uniform-dlat grids.
        The two POLAR-WALL faces keep the historical non-zero
        cell-centre value ``cos_u[0]`` / ``cos_u[-1]`` because this
        ``cos_v`` is also used as a DIVISOR by
        :func:`flux_divergence_viscosity_cgrid` (``1/(cos_v·R·dlon)``
        spherical metric prefactor); the canonical helper's exact-zero
        pole would make that ``0/0 → NaN``.  The pole rows of those
        viscous tendencies are walls (zeroed) regardless, so the
        boundary value is a finite safety placeholder, not a physical
        flux — and the gate test compares only the INTERIOR faces.
    """
    cos_u = grid.cos_lat                                  # (n_lat,)
    cos_v_canon = vface_zonal_cos_lat(grid)              # (n_lat+1,), poles 0
    # Replace the exact-zero polar walls with the historical non-zero
    # cell-centre cosine so divisor uses stay finite (interior unchanged).
    cos_v = jnp.concatenate([
        cos_u[:1],
        cos_v_canon[1:-1],
        cos_u[-1:],
    ])
    return cos_u, cos_v


def laplacian_scaling_factor(
    grid: LatLonGrid,
    power: int = 1,
    floor: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Grid-dependent scaling for Laplacian viscosity on a lat-lon grid.

    On a latitude-longitude grid the zonal grid spacing shrinks as
    ``cos(lat)`` near the poles.  Two conventions exist:

    ``power=1`` (recommended): ``A_h_eff = A_h × cos(lat)``.
        Keeps the grid Reynolds number ``U·dx / A_h_eff`` latitude-
        independent.  The viscous CFL grows as ``1/cos(lat)`` toward
        the poles but stays well below the stability limit at
        realistic ``dt`` and ``A_h`` (CFL < 0.03 at 85° for
        ``A_h=1e5``, ``dt=300s``).

    ``power=2`` (legacy): ``A_h_eff = A_h × cos²(lat)``.
        Keeps the viscous CFL latitude-independent but makes the grid
        Reynolds number grow as ``1/cos(lat)`` → ∞ at the poles,
        causing blowup at high latitudes on 1° grids.

    Usage::

        scale_u, scale_v = laplacian_scaling_factor(grid, power=1)
        vlap_u, vlap_v = vector_laplacian_cgrid(u, v, grid, ...)
        du_dt += A_h * scale_u[:, None, None] * vlap_u
        dv_dt += A_h * scale_v[:, None, None] * vlap_v

    Parameters
    ----------
    grid : LatLonGrid
    power : {1, 2}
        Exponent on cos(lat).  1 = constant Re_grid (recommended).
        2 = constant viscous CFL (legacy).
    floor : float
        Minimum value for the scaling factor (dimensionless).
        When the caller supplies ``floor = A_h_floor / A_h``, this
        enforces a minimum effective viscosity of ``A_h_floor`` m²/s.

    Returns
    -------
    scale_u : (n_lat,)
        cos^power(lat) at u-face latitudes (cell centres).  Reshape to
        ``[:, None]`` for 2D fields or ``[:, None, None]`` for 3D.
    scale_v : (n_lat+1,)
        cos^power(lat) at v-face latitudes.
    """
    cos_u, cos_v = _cos_lat_uv(grid)
    scale_u = cos_u ** power
    scale_v = cos_v ** power
    if floor > 0:
        scale_u = jnp.maximum(scale_u, floor)
        scale_v = jnp.maximum(scale_v, floor)
    return scale_u, scale_v


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


def polar_cap_boost_factor(
    grid: LatLonGrid,
    cap_lat_deg: float,
    boost: float,
    width_deg: float = 5.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Latitude-dependent polar-cap enhancement factor for A_h.

    Returns a multiplier ``1 + (boost - 1) · S(|lat| − cap_lat_deg)``
    where ``S`` is a smooth tanh ramp of width ``width_deg``.  Active
    only at high latitudes; transparent (factor 1) below the cap.

    Designed to damp the bipolar-cap cascade on tripolar grids where
    the ``cos(lat)`` viscosity scaling drops to zero at the fold
    boundary even though the deformed cells require stronger lateral
    dissipation than the ``A_h_floor`` alone provides.  At a typical
    ORCA1-like cap (start at 75°N), a 5–20× boost over ±5° width
    matches MOM6 ORCA1 ``KH_VEL_LAT_RES`` profile semantics.

    Multiplicative on top of the ``laplacian_scaling_factor`` cos(lat)
    scaling (or the cos²(lat) biharmonic scaling).  On a regular
    lat-lon grid with no fold this still applies — the boost is a
    function of latitude only — so callers should leave it off
    (``boost=1.0``) unless they want extra polar diffusion.

    Parameters
    ----------
    grid : LatLonGrid
    cap_lat_deg : float
        Latitude (degrees, North) at which the boost ramp begins.
        Typical values 70–80°.  For tripolar grids, set close to the
        ``fold_lat`` of the ``FoldDescriptor``.
    boost : float
        Multiplier in the asymptotic limit (deep inside the cap).
        Values ≥ 1.0; ``boost = 1.0`` disables the enhancement.
        Typical values 5–20.
    width_deg : float
        Half-width of the tanh transition [°].  Default 5°.

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

    cap_rad = jnp.deg2rad(cap_lat_deg)
    width_rad = jnp.deg2rad(jnp.maximum(width_deg, 1e-6))

    lat_u = grid.lat
    lat_v_interior = 0.5 * (lat_u[:-1] + lat_u[1:])
    lat_v = jnp.concatenate([lat_u[:1], lat_v_interior, lat_u[-1:]])

    ramp_u = 0.5 * (1.0 + jnp.tanh((jnp.abs(lat_u) - cap_rad) / width_rad))
    ramp_v = 0.5 * (1.0 + jnp.tanh((jnp.abs(lat_v) - cap_rad) / width_rad))
    boost_u = 1.0 + (boost - 1.0) * ramp_u
    boost_v = 1.0 + (boost - 1.0) * ramp_v
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
    R = getattr(grid, "radius", constants.R_earth)
    n_lat, n_lon = H_bathy.shape

    # ∇H at cell centres via centred differences (periodic in lon, walls in lat)
    cos_lat = jnp.maximum(grid.cos_lat, 1e-3)
    dlon = 2.0 * jnp.pi / n_lon
    dx = R * cos_lat[:, None] * dlon

    H_e = jnp.roll(H_bathy, -1, axis=1)
    H_w = jnp.roll(H_bathy, 1, axis=1)
    dHdx = (H_e - H_w) / (2.0 * dx)

    # No wrap in lat — use one-sided differences at boundaries.
    # 2-cell distance per row from grid.dy (Mercator-safe).
    H_n = jnp.concatenate([H_bathy[1:, :], H_bathy[-1:, :]], axis=0)
    H_s = jnp.concatenate([H_bathy[:1, :], H_bathy[:-1, :]], axis=0)
    dHdy = (H_n - H_s) / grid.dy[:, None]

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
    # Extract metrics — legacy scalar/1D path for regular lat-lon,
    # per-cell arrays for tripolar (via the same field names, but the
    # operators below use R/dlon/dlat/cos_lat so we keep those aliases).
    R = grid.radius
    dlon = grid.dlon
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
    if is_tripolar(grid):
        # Tripolar: full 2D metrics (same logic as divergence_cgrid).
        face_dy = grid.dy_u  # (n_lat, n_lon+1)
        u_east = u_eff[:, 1:]
        u_west = u_eff[:, :-1]
        du_dx = u_east * _bcast2d(face_dy[:, 1:]) - u_west * _bcast2d(face_dy[:, :-1])

        face_dx = grid.dx_v  # (n_lat+1, n_lon)
        v_north = v_eff[1:]
        v_south = v_eff[:-1]
        dv_dy = v_north * _bcast2d(face_dx[1:]) - v_south * _bcast2d(face_dx[:-1])
    else:
        # #516: evaluate the regular-grid metrics in the model's working
        # float precision (float64 under JAX_ENABLE_X64).  ``grid.dy`` /
        # ``grid.cos_lat`` / ``grid.area`` are stored float32, which
        # injects ~1e-7 noise that breaks the strain↔stress adjoint
        # identity at x64 tolerance — promote so strain and its exact
        # transpose ``stress_divergence_cgrid`` share bit-identical
        # metrics.
        _fdtype = jnp.result_type(float)
        face_dy = jnp.asarray(grid.dy, dtype=_fdtype) * 0.5  # (n_lat,)
        u_east = u_eff[:, 1:]
        u_west = u_eff[:, :-1]
        du_dx = (u_east - u_west) * face_dy[lat_bcast]

        # #516: single-source v-face zonal LENGTH (canonical cos-of-
        # interface, poles 0).  This is a pure multiplicative weight on
        # v (never a divisor) so the exact-0 polar value is fine and v≡0
        # at the walls anyway; the vertex dual area below keeps its own
        # 1e-10 clamp (it DIVIDES).  Sharing this exact array with
        # stress_divergence_cgrid is what makes the adjoint pair hold.
        face_dx = R * vface_zonal_cos_lat(grid) * dlon  # (n_lat+1,)
        v_north = v_eff[1:]
        v_south = v_eff[:-1]
        dv_dy = (v_north * face_dx[1:][lat_bcast]
                 - v_south * face_dx[:-1][lat_bcast])

    if is_tripolar(grid):
        area = grid.area  # (n_lat, n_lon)
    else:
        area = jnp.asarray(grid.area, dtype=jnp.result_type(float))
    D_T = (du_dx - dv_dy) / _bcast2d(area)
    if mask is not None:
        D_T = D_T * _bcast2d(mask)

    # --- D_S at q-points: dv/dx + du/dy ---
    if is_tripolar(grid):
        # Tripolar: full 2D vertex area and edge lengths
        # (same logic as curl_vertex_cgrid).
        A_vertex = grid.area_q  # (n_lat+1, n_lon+1)
        A_vertex = jnp.maximum(A_vertex, 1e-30)
        dx_cell = grid.dx_T  # (n_lat, n_lon)
        dy_v_2d = grid.dy_v  # (n_lat+1, n_lon)

        v_east = v_eff
        v_west = jnp.roll(v_eff, 1, axis=1)
        dy_east = _bcast2d(dy_v_2d)
        dy_west = _bcast2d(jnp.roll(dy_v_2d, 1, axis=1))
        dv_circ = v_east * dy_east - v_west * dy_west
        dv_circ_full = jnp.concatenate([dv_circ, dv_circ[:, 0:1]], axis=1)

        dx_pad = jnp.pad(dx_cell, ((1, 1), (0, 0)))
        dx_pad = jnp.concatenate([dx_pad, dx_pad[:, 0:1]], axis=1)
        u_ext = jnp.pad(u_eff, ((1, 1), (0, 0), *pad_extra))
        u_south = u_ext[:-1]
        u_north = u_ext[1:]
        dx_south = _bcast2d(dx_pad[:-1])
        dx_north = _bcast2d(dx_pad[1:])
        # Sign FLIPPED vs curl: du/dy = u_north*dx_north - u_south*dx_south
        du_circ = u_north * dx_north - u_south * dx_south

        D_S = (dv_circ_full + du_circ) / _bcast2d(A_vertex)
    else:
        # #516: working-precision metrics (see the D_T branch).
        _fdtype = jnp.result_type(float)
        lat_f = jnp.asarray(lat, dtype=_fdtype)
        cos_lat_f = jnp.asarray(cos_lat, dtype=_fdtype)
        A_vertex = jnp.maximum(
            _vertex_dual_area_interior(lat_f, R, dlon), 1e-30)

        dx_cell = R * cos_lat_f * dlon
        dy_h = jnp.asarray(grid.dy, dtype=_fdtype) * 0.5
        dy_edge_interior = 0.5 * (dy_h[1:] + dy_h[:-1])
        dy_edge = jnp.pad(dy_edge_interior, (1, 1), mode='edge')

        v_east = v_eff
        v_west = jnp.roll(v_eff, 1, axis=1)
        dv_circ = (v_east - v_west) * dy_edge[lat_bcast]
        dv_circ_full = jnp.concatenate([dv_circ, dv_circ[:, 0:1]], axis=1)

        u_ext = jnp.pad(u_eff, ((1, 1), (0, 0), *pad_extra))
        dx_ext = jnp.pad(dx_cell, (1, 1))
        u_south = u_ext[:-1]
        u_north = u_ext[1:]
        dx_south = dx_ext[:-1]
        dx_north = dx_ext[1:]
        du_circ = (u_north * dx_north[lat_bcast]
                   - u_south * dx_south[lat_bcast])

        D_S = (dv_circ_full + du_circ) / A_vertex[lat_bcast]
    # Boundary: wall BC on regular lat-lon; fold on tripolar.
    D_S = pad_ns_scalar(D_S[1:-1], grid)

    if mask is not None:
        vmask = compute_vertex_mask(mask, grid=grid)
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
    is_3d = stress_h.ndim == 3

    def _bcast(m):
        return m[..., jnp.newaxis] if is_3d else m

    if not normalize:
        # #516 exact-adjoint mode.  ``normalize=False`` must return the
        # BIT-EXACT discrete matrix transpose of ``strain_rate_cgrid``
        # (so ``<strain(u,v), T> = <(u,v), stress_div(T)>`` to machine
        # precision — the #516 adjoint gate).  Rather than re-derive the
        # transpose stencils by hand (a subtle sign/metric exercise that
        # only holds in the area-WEIGHTED energy product, not the plain
        # one the gate uses), obtain the exact transpose from JAX's
        # linear transpose of the (linear) strain operator.  This is the
        # true matrix transpose by construction, AD-safe, and shares
        # strain's promoted float64 metrics and unified v-face length, so
        # it is exact on Mercator.  ``normalize=True`` (the production
        # energy-stable viscous path) keeps its hand-written, energy-
        # normalised stencils below — UNCHANGED.
        u_like = jnp.zeros(
            stress_h.shape[:1] + (stress_h.shape[1] + 1,) + stress_h.shape[2:],
            dtype=jnp.result_type(stress_h, float),
        )
        v_like = jnp.zeros(
            (stress_q.shape[0],) + (stress_q.shape[1] - 1,) + stress_q.shape[2:],
            dtype=jnp.result_type(stress_q, float),
        )

        def _strain_lin(u_in, v_in):
            return strain_rate_cgrid(
                u_in, v_in, grid, u_mask=u_mask, v_mask=v_mask,
            )

        # The strain operator applies the face masks to its INPUTS, so
        # the linear transpose already emits the matching output-side
        # masking — no extra mask multiply is needed (and would be
        # redundant since the masks are idempotent 0/1 fields).
        transpose_fn = jax.linear_transpose(_strain_lin, u_like, v_like)
        tend_u, tend_v = transpose_fn((stress_h, stress_q))
        return tend_u, tend_v

    if is_tripolar(grid):
        # Tripolar: full 2D metrics — must match strain_rate_cgrid's
        # 2D path so the adjoint identity holds.
        dy_u = grid.dy_u           # (n_lat, n_lon+1)
        dx_T = grid.dx_T           # (n_lat, n_lon)
        dx_v_2d = grid.dx_v        # (n_lat+1, n_lon)
        dy_v_2d = grid.dy_v        # (n_lat+1, n_lon)

        # tend_u from D_T: dy_u[:, k] * (sh[:, k] - sh[:, k-1])
        sh_west = jnp.roll(stress_h, 1, axis=1)
        dsh = stress_h - sh_west
        dsh_full = jnp.concatenate([dsh, dsh[:, 0:1]], axis=1)
        tend_u_DT = _bcast(dy_u) * dsh_full

        # tend_u from D_S: dx_T with wrap → (n_lat, n_lon+1)
        dx_T_wrap = jnp.concatenate([dx_T, dx_T[:, 0:1]], axis=1)
        dsq_meridional = stress_q[1:] - stress_q[:-1]
        tend_u_DS = _bcast(dx_T_wrap) * dsq_meridional

        tend_u = tend_u_DT + tend_u_DS

        # tend_v from D_T: dx_v * (sh[m-1] - sh[m])
        dsh_merid = stress_h[:-1] - stress_h[1:]
        tend_v_DT_interior = _bcast(dx_v_2d[1:-1]) * dsh_merid
        tend_v_DT = pad_ns_scalar(tend_v_DT_interior, grid)

        # tend_v from D_S: per-face dy_v * (sq[:, j+1] - sq[:, j])
        dsq_zonal = stress_q[:, 1:] - stress_q[:, :-1]
        # Per-face: east edge dy and west edge dy
        jnp.roll(dy_v_2d, 1, axis=1)
        # Adjoint of (v_east*dy_east - v_west*dy_west): same dy weighting
        tend_v_DS = _bcast(dy_v_2d) * dsq_zonal

        tend_v = tend_v_DT + tend_v_DS

        # Area normalization: u-face dual = dy_u * dx_T_wrap
        area_u_dual = dy_u * dx_T_wrap
        # v-face dual = dy_v * dx_v
        area_v_dual = dy_v_2d * dx_v_2d
        area_u_dual = jnp.maximum(area_u_dual, 1e-30)
        area_v_dual = jnp.maximum(area_v_dual, 1e-30)
    else:
        R = grid.radius
        dlon = grid.dlon
        # #516: working-precision metrics (float64 under x64) — MUST
        # bit-match ``strain_rate_cgrid``'s promoted metrics for the
        # discrete adjoint pair to hold at x64 tolerance; the stored
        # float32 grid fields inject ~1e-7 noise.
        _fdtype = jnp.result_type(float)
        lat = jnp.asarray(grid.lat, dtype=_fdtype)
        cos_lat = jnp.asarray(grid.cos_lat, dtype=_fdtype)
        lat_bcast = (slice(None),) + (jnp.newaxis,) * (stress_h.ndim - 1)

        dy_h = jnp.asarray(grid.dy, dtype=_fdtype) * 0.5
        dy_edge_interior = 0.5 * (dy_h[1:] + dy_h[:-1])
        dy_edge = jnp.pad(dy_edge_interior, (1, 1), mode='edge')
        dx_cell = R * cos_lat * dlon
        # #516: single-source v-face zonal LENGTH (canonical cos-of-
        # interface, poles 0) — MUST be the SAME array strain_rate_cgrid
        # uses in dv_dy for the discrete adjoint pair to hold to machine
        # precision.  It enters ``tend_v_DT`` only at the interior faces
        # ``dx_v[1:-1]`` (a pure multiplicative weight), so the exact-0
        # polar value is fine here.  The vertex dual-area DENOMINATOR
        # below keeps the legacy 1e-10 clamp so the normalised tendency
        # stays finite at the (v≡0) polar walls.
        dx_v = R * vface_zonal_cos_lat(grid) * dlon
        cos_lat_v_clamped = jnp.maximum(vface_zonal_cos_lat(grid), 1e-10)
        dx_v_area = R * cos_lat_v_clamped * dlon

        sh_west = jnp.roll(stress_h, 1, axis=1)
        dsh = stress_h - sh_west
        dsh_full = jnp.concatenate([dsh, dsh[:, 0:1]], axis=1)
        tend_u_DT = dy_h[lat_bcast] * dsh_full

        dsq_meridional = stress_q[1:] - stress_q[:-1]
        tend_u_DS = dx_cell[lat_bcast] * dsq_meridional

        tend_u = tend_u_DT + tend_u_DS

        dsh_merid = stress_h[:-1] - stress_h[1:]
        tend_v_DT_interior = dx_v[1:-1][lat_bcast] * dsh_merid
        tend_v_DT = pad_ns_scalar(tend_v_DT_interior, grid)

        dsq_zonal = stress_q[:, 1:] - stress_q[:, :-1]
        tend_v_DS = dy_edge[lat_bcast] * dsq_zonal

        tend_v = tend_v_DT + tend_v_DS

        area_u_dual = dy_h * dx_cell
        area_v_dual = dy_edge * dx_v_area
        area_u_dual = jnp.maximum(area_u_dual, 1e-30)
        area_v_dual = jnp.maximum(area_v_dual, 1e-30)

    if normalize:
        # _bcast appends a level axis only when stress_h is 3D. For
        # the non-tripolar branch area_u_dual / area_v_dual are 1D
        # (n_lat,) — broadcasting (n_lat, n_lon+1, n_lev) /
        # (n_lat, 1) is illegal; we need (n_lat, 1, 1). lat_bcast
        # below appends the right number of trailing newaxes.
        _lat_bcast = (slice(None),) + (jnp.newaxis,) * (tend_u.ndim - 1)
        if jnp.asarray(area_u_dual).ndim == 1:
            tend_u = tend_u / area_u_dual[_lat_bcast]
        else:
            tend_u = tend_u / _bcast(area_u_dual)
        _lat_bcast_v = (slice(None),) + (jnp.newaxis,) * (tend_v.ndim - 1)
        if jnp.asarray(area_v_dual).ndim == 1:
            tend_v = tend_v / area_v_dual[_lat_bcast_v]
        else:
            tend_v = tend_v / _bcast(area_v_dual)

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

    def _bcast_coef(A):
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
    stress_h = _bcast_coef(A_h) * D_T
    stress_q = _bcast_coef(A_q) * D_S

    # 3. Stress divergence.
    #
    # #516.  ``stress_divergence_cgrid(normalize=False)`` now returns the
    # EXACT discrete transpose of the (area-normalised) PUBLIC
    # ``strain_rate_cgrid`` — i.e. ``Sᵀ_norm = L_rawᵀ ∘ diag(1/area)`` —
    # so the #516 adjoint gate ``<strain(u),T> = <u, stress_div(T)>``
    # holds to machine precision.
    #
    # The biharmonic / backscatter first pass instead needs the
    # historical UN-normalised RAW stress-divergence ``-L_rawᵀ(stress)``
    # (the MOM6 velocity-like intermediate, units m/s — NOT the extra
    # ``1/area`` the normalised transpose carries).  Recover it exactly
    # from the normalised transpose using ``L_rawᵀ(s) = Sᵀ_norm(s·area)``
    # (the strain normalises D_T by the cell area and D_S by the vertex
    # dual area, so pre-multiplying the stresses by those SAME areas
    # cancels the ``1/area`` baked into ``Sᵀ_norm``), then negate for the
    # dissipative sign the callers subtract.  ``normalize=True`` (the
    # energy-stable viscous tendency, the default) is UNCHANGED.
    if normalize:
        tend_u, tend_v = stress_divergence_cgrid(
            stress_h, stress_q, grid,
            u_mask=u_mask, v_mask=v_mask, normalize=True)
        return tend_u, tend_v

    # normalize=False — RAW (un-normalised) negative adjoint.
    _fdtype = jnp.result_type(float)
    area_cell = jnp.asarray(grid.area, dtype=_fdtype)        # (n_lat, n_lon)
    A_vertex = vertex_area_1d(grid).astype(_fdtype)          # (n_lat+1,)
    lat_bcast = (slice(None),) + (jnp.newaxis,) * (stress_q.ndim - 1)
    stress_h_raw = stress_h * (
        area_cell[..., jnp.newaxis] if is_3d else area_cell)
    stress_q_raw = stress_q * A_vertex[lat_bcast]
    tend_u, tend_v = stress_divergence_cgrid(
        stress_h_raw, stress_q_raw, grid,
        u_mask=u_mask, v_mask=v_mask, normalize=False)
    return -tend_u, -tend_v


def vertex_area_1d(grid: LatLonGrid) -> jnp.ndarray:
    """Dual-cell area at vertex (corner) points.

    Returns
    -------
    A_vertex : (n_lat+1,)
        Area of each vertex dual cell.  Pole rows are set to a small
        positive floor (1e-30) to avoid division by zero.
    """
    A_v = _vertex_dual_area_interior(grid.lat, grid.radius, grid.dlon)
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
    D_T_q = pad_ns_scalar(D_T_q, grid)
    # Append periodic wrap column
    D_T_q = jnp.concatenate(
        [D_T_q, D_T_q[:, 0:1]], axis=1)  # (n_lat+1, n_lon+1, ...)

    # Small epsilon prevents NaN gradient of sqrt at zero (masked points).
    deformation_q = jnp.sqrt(D_T_q**2 + D_S**2 + 1e-30)

    # Vertex dual cell area; reshape for broadcast over (n_lat+1, n_lon+1[, nlev]).
    A_vert = vertex_area_1d(grid)  # (n_lat+1,)
    Delta_q = jnp.sqrt(A_vert)
    bcast = (slice(None),) + (jnp.newaxis,) * (D_T.ndim - 1)
    Delta_q = Delta_q[bcast]

    A_smag_q = (C_smag * Delta_q)**2 * deformation_q

    # Zero at pole vertices.
    A_smag_q = A_smag_q.at[0].set(0.0)
    A_smag_q = A_smag_q.at[-1].set(0.0)

    if mask is not None:
        vmask = compute_vertex_mask(mask, grid=grid)
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


def om4p25_lateral_friction_tendency_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
    *,
    C2: float = 0.15,
    Cu2: float = 0.01,
    C4: float = 0.06,
    Cu4: float = 0.01,
    deformation_radius: jnp.ndarray | float = 6.75e3,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """OM4p25 lateral-friction closure (GFDL OM4.0, Adcroft et al. 2019) — the
    Silvestri et al. 2024 "SM2" comparison case. A Laplacian + biharmonic
    combination where each viscosity is the MAX of a flow-adaptive Smagorinsky
    term and a static grid-scale term (paper Appendix A, Eqs A5-A8):

        nu2 = max(C2·Δ²·|D|, Cu2·Δ)·F          (Laplacian   [m²/s])
        nu4 = max(C4·Δ⁴·|D|, Cu4·Δ³)            (biharmonic  [m⁴/s])
        F   = 1 / (1 + 0.25·(L_d/Δ)⁴)            (deformation-radius taper)

    with |D| = sqrt(D_T² + D_S²) the strain-rate magnitude (D_T = ∂ₓu−∂ᵧv,
    D_S = ∂ₓv+∂ᵧu), Δ = sqrt(cell area), and L_d the first-baroclinic
    deformation radius. F reduces the LAPLACIAN where the deformation radius is
    well resolved (large L_d/Δ); it does not taper the biharmonic.

    Both viscosities are applied through the energy-stable stress-tensor
    operator (``viscous_tendency_cgrid``, the exact discrete adjoint of the
    strain rate), so dissipation is guaranteed for the non-negative spatially-
    varying coefficients. The biharmonic uses the two-pass form
    ``L_c(L_1(u))`` whose effective coefficient is ``c·Δ²``; to realise
    ``nu4`` the second-pass coefficient is ``c4 = nu4/Δ² = max(C4·Δ²·|D|, Cu4·Δ)``.

    Coefficient defaults are the OM4p25 values: C2=0.15, Cu2=0.01, C4=0.06,
    Cu4=0.01. ``deformation_radius`` is a scalar (or h-point field) in metres;
    for the idealised baroclinic jet it is ~uniform (~6.75 km). A spatially-
    varying L_d from the local N² is a faithfulness refinement.

    Returns ``(tend_u, tend_v)`` already combining Laplacian (added) and
    biharmonic (subtracted); the caller ADDS these to du/dt.
    """
    # Δ²·|D| at h- and q-points via the Smagorinsky helpers with unit
    # coefficient (smagorinsky_viscosity_cgrid returns (C·Δ)²·|D| = Δ²·|D| at C=1).
    S_h = smagorinsky_viscosity_cgrid(
        u, v, grid, 1.0, mask=mask, u_mask=u_mask, v_mask=v_mask)   # Δ_h²·|D|_h
    D_T, D_S = strain_rate_cgrid(
        u, v, grid, mask=mask, u_mask=u_mask, v_mask=v_mask)
    S_q = smagorinsky_viscosity_q_cgrid(D_T, D_S, grid, 1.0, mask=mask)  # Δ_q²·|D|_q

    Delta_h = jnp.sqrt(grid.area)
    if S_h.ndim == 3:
        Delta_h = Delta_h[..., jnp.newaxis]
    Delta_q = jnp.sqrt(vertex_area_1d(grid))
    bcast = (slice(None),) + (jnp.newaxis,) * (S_q.ndim - 1)
    Delta_q = Delta_q[bcast]

    def _taper(Delta):
        # F = 1/(1 + 0.25·(L_d/Δ)⁴): small where L_d >> Δ (resolved eddies).
        Rh = deformation_radius / jnp.maximum(Delta, 1.0e-12)
        return 1.0 / (1.0 + 0.25 * Rh ** 4)

    nu2_h = jnp.maximum(C2 * S_h, Cu2 * Delta_h) * _taper(Delta_h)
    nu2_q = jnp.maximum(C2 * S_q, Cu2 * Delta_q) * _taper(Delta_q)
    c4_h = jnp.maximum(C4 * S_h, Cu4 * Delta_h)
    c4_q = jnp.maximum(C4 * S_q, Cu4 * Delta_q)

    # Laplacian (added directly — energy-dissipative for nu2 >= 0).
    lap_u, lap_v = viscous_tendency_cgrid(
        u, v, grid, nu2_h, nu2_q, mask=mask, u_mask=u_mask, v_mask=v_mask)
    # Biharmonic two-pass (subtracted): u_star = unit-coeff stress-divergence,
    # then c4-weighted stress-divergence → effective nu4·∇⁴u.
    u_star, v_star = viscous_tendency_cgrid(
        u, v, grid, 1.0, 1.0, mask=mask, u_mask=u_mask, v_mask=v_mask,
        normalize=False)
    bih_u, bih_v = viscous_tendency_cgrid(
        u_star, v_star, grid, c4_h, c4_q,
        mask=mask, u_mask=u_mask, v_mask=v_mask)
    return lap_u - bih_u, lap_v - bih_v


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

def _grad_vertex_vec_h(zeta_q: jnp.ndarray, grid: "LatLonGrid") -> tuple:
    """∇ of a VERTEX field, as the (∂ₓ, ∂ᵧ) vector at cell centres [1/(m·s)].

    ``zeta_q`` : (n_lat+1, n_lon+1, ...) at vertices → (gx, gy) each
    (n_lat, n_lon, ...) at cell centres.
    """
    R = grid.radius
    dlon = grid.dlon
    cos_lat = grid.cos_lat
    if zeta_q.ndim == 3:
        cos_lat_b = cos_lat[:, jnp.newaxis, jnp.newaxis]
        dy_h_b = (grid.dy * 0.5)[:, jnp.newaxis, jnp.newaxis]
    else:
        cos_lat_b = cos_lat[:, jnp.newaxis]
        dy_h_b = (grid.dy * 0.5)[:, jnp.newaxis]
    dx_h = R * cos_lat_b * dlon
    dz_dx = 0.5 * ((zeta_q[:-1, 1:] - zeta_q[:-1, :-1])
                   + (zeta_q[1:, 1:] - zeta_q[1:, :-1])) / dx_h
    dz_dy = 0.5 * ((zeta_q[1:, :-1] - zeta_q[:-1, :-1])
                   + (zeta_q[1:, 1:] - zeta_q[:-1, 1:])) / dy_h_b
    return dz_dx, dz_dy


def _grad_zeta_mag_h(zeta_q: jnp.ndarray, grid: "LatLonGrid") -> jnp.ndarray:
    """|∇ζ| at cell centres from ζ at vertices (magnitude of ``_grad_vertex_vec_h``)."""
    gx, gy = _grad_vertex_vec_h(zeta_q, grid)
    return jnp.sqrt(gx ** 2 + gy ** 2 + 1e-30)


def _grad_cell_vec_h(div_h: jnp.ndarray, grid: "LatLonGrid") -> tuple:
    """∇ of a CELL-CENTRE field, as the (∂ₓ, ∂ᵧ) vector at cell centres.

    Centred differences with periodic wrap in longitude and one-sided diffs at
    the pole rows. ``div_h`` : (n_lat, n_lon, ...) → (gx, gy) same shape.
    """
    R = grid.radius
    dlon = grid.dlon
    cos_lat = grid.cos_lat

    if div_h.ndim == 3:
        cos_lat_b = cos_lat[:, jnp.newaxis, jnp.newaxis]
        dy_h = (grid.dy * 0.5)[:, jnp.newaxis, jnp.newaxis]
    else:
        cos_lat_b = cos_lat[:, jnp.newaxis]
        dy_h = (grid.dy * 0.5)[:, jnp.newaxis]

    dx_h = R * cos_lat_b * dlon

    # Zonal gradient: centred difference with periodic wrap (works for any ndim).
    dd_dx = (jnp.roll(div_h, -1, axis=1) - jnp.roll(div_h, 1, axis=1)) / (2.0 * dx_h)

    # Meridional gradient: centred in interior, one-sided at pole rows.
    # Interior centred diff spans rows i-1..i+1; the correct denominator
    # is the cell-centre-to-cell-centre distance from row i-1 to row i+1,
    # computed directly from ``grid.lat`` (Mercator-safe). For uniform
    # dlat this equals ``2 * dy_h[i]`` exactly.
    lat = grid.lat
    if div_h.ndim == 3:
        d_2cell_interior = (R * (lat[2:] - lat[:-2]))[:, jnp.newaxis, jnp.newaxis]
    else:
        d_2cell_interior = (R * (lat[2:] - lat[:-2]))[:, jnp.newaxis]
    dd_dy_interior = (div_h[2:] - div_h[:-2]) / d_2cell_interior
    # One-sided diffs at pole rows: distance from cell-centre row 0 to
    # row 1 (south) / from row n_lat-2 to n_lat-1 (north) — same as
    # ``dy_v(½)``, i.e. ½(dy_h[0]+dy_h[1]).
    dy_v_south = 0.5 * (dy_h[0:1] + dy_h[1:2])
    dy_v_north = 0.5 * (dy_h[-1:] + dy_h[-2:-1])
    dd_dy_south = (div_h[1:2] - div_h[0:1]) / dy_v_south
    dd_dy_north = (div_h[-1:] - div_h[-2:-1]) / dy_v_north
    dd_dy = jnp.concatenate([dd_dy_south, dd_dy_interior, dd_dy_north], axis=0)

    return dd_dx, dd_dy


def _grad_div_mag_h(div_h: jnp.ndarray, grid: "LatLonGrid") -> jnp.ndarray:
    """|∇δ| at cell centres (magnitude of ``_grad_cell_vec_h``)."""
    gx, gy = _grad_cell_vec_h(div_h, grid)
    return jnp.sqrt(gx ** 2 + gy ** 2 + 1e-30)


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

    # dy_q at q-point (vertex row): cell-center-to-cell-center distance.
    # Interior vertices; pole rows padded (their ζ contributions are 0).
    dy_h_arr = grid.dy * 0.5
    dy_q_interior = 0.5 * (dy_h_arr[1:] + dy_h_arr[:-1])               # (n_lat-1,)
    dy_q_1d = jnp.pad(dy_q_interior, (1, 1), mode='edge')              # (n_lat+1,)
    dx_q = R * cos_lat_q_b * dlon
    dy_q = dy_q_1d[bcast]

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
        grad_div_q = pad_ns_zero(gd_q_int)
        grad_div_q = jnp.concatenate(
            [grad_div_q, grad_div_q[:, 0:1]], axis=1)
        total_sq = total_sq + grad_div_q ** 2

    norm_q = jnp.sqrt(total_sq + 1e-30)

    A_vert = vertex_area_1d(grid)                          # (n_lat+1,)
    bcast = (slice(None),) + (jnp.newaxis,) * (norm_q.ndim - 1)
    Delta_q = jnp.sqrt(A_vert)[bcast]

    A_leith_q = (C_leith * Delta_q) ** 3 * norm_q

    # Zero at pole vertices, matching smagorinsky_viscosity_q_cgrid.
    # Slice + single Pad HLO op replaces zeros_like-of-slice ×2 +
    A_leith_q = pad_ns_scalar(A_leith_q[1:-1], grid)

    if mask is not None:
        vmask = compute_vertex_mask(mask, grid=grid)
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


def qg_leith_viscosity_tendency_cgrid(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: "LatLonGrid",
    *,
    C_qgleith: float = 2.0,
    buoyancy: jnp.ndarray | None = None,
    h_k: jnp.ndarray | None = None,
    deformation_radius: jnp.ndarray | float | None = None,
    velocity_scale: float = 1.0,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """QG-Leith eddy viscosity (Silvestri et al. 2024 "QG2" / Bachman et al.
    2017). With ``buoyancy``+``h_k`` supplied it is the FULL QG2 (baroclinic
    stretching + Bu/Ro bound, see below); without them it is the BAROTROPIC
    approximation (∇Q = ∇(ζ+f), bounds inert — label "QG-Leith (barotropic)").
    A HARMONIC (Laplacian) viscosity scaling with the potential-vorticity gradient:

        nu = (C·Δ/π)³ · sqrt(|∇Q|² + |∇δ|²)

    with Q = ζ + f the absolute (barotropic) potential vorticity (ζ = relative
    vorticity at vertices, f = planetary vorticity), δ = ∇·u the horizontal
    divergence, Δ = sqrt(cell area), and C the dimensionless coefficient
    (paper QG2: C=2). Applied through the energy-stable stress-tensor operator
    ``viscous_tendency_cgrid`` (exact strain adjoint → guaranteed dissipation).

    This is the paper's QG counterpart to the 2D Leith closure (Appendix A1/A2,
    ν=(CΔ/π)³|∇q|): it differs from legoESM's existing ``C_leith`` operator in
    three faithful ways — (1) HARMONIC not biharmonic, (2) ABSOLUTE-vorticity
    PV gradient ∇(ζ+f) not relative ∇ζ, (3) the /π³ paper normalisation.

    FULL QG2 (B5b): when ``buoyancy`` (b at cell centres) + ``h_k`` (layer
    thicknesses) are supplied (3D), the baroclinic stretching term is added —
    ∇q₁ = ∇(ζ+f) + ∂_z(f/N²·∇b) (Eq A3) — and the Bachman grid-Burger /
    grid-Rossby min-bound is applied: |∇Q| = min(|∇q₁|, |∇q|(1+1/Bu),
    |∇q|(1+1/Ro²)) with Bu=Δ²/L_d², Ro=V/(|f|Δ). Without buoyancy the operator
    falls back to the BAROTROPIC ∇(ζ+f) (the bounds are inert — see
    ``bound_qg_pv_gradient``); that path must be labeled "QG-Leith (barotropic)".
    Returns (tend_u, tend_v) to be ADDED to du/dt.
    """
    is_3d = u.ndim == 3
    _um = u_mask[..., jnp.newaxis] if (u_mask is not None and is_3d) else u_mask
    _vm = v_mask[..., jnp.newaxis] if (v_mask is not None and is_3d) else v_mask
    u_eff = u if _um is None else u * _um
    v_eff = v if _vm is None else v * _vm

    # Absolute vorticity Q = ζ + f at vertices.  Use the GRID's stored staggered
    # Coriolis (``_vertex_coriolis`` → ``grid.f_v`` at the corners): correct on
    # β-plane / f-plane geometries (f = f0 + β·y), and — since ``f_v`` is built
    # from the grid's own rotation scalar — still exactly non-rotating for an
    # omega=0 grid (#521).  A minimal duck without stored ``f_v``/``f`` falls
    # back to the spherical reconstruction from the grid's rotation scalar.
    zeta_q = curl_vertex_cgrid(u_eff, v_eff, grid)              # (n_lat+1,n_lon+1,...)
    if hasattr(grid, "f_v") or hasattr(grid, "f"):
        f_q = _vertex_coriolis(grid)                           # (n_lat+1,n_lon+1)
    else:
        lat = grid.lat
        lat_v = jnp.concatenate([lat[:1], 0.5 * (lat[:-1] + lat[1:]), lat[-1:]])
        f_q = (2.0 * getattr(grid, "omega", constants.Omega)
               * jnp.sin(lat_v))[:, jnp.newaxis]               # (n_lat+1,1)
    f_q = f_q if zeta_q.ndim == 2 else f_q[..., jnp.newaxis]
    absvort_q = zeta_q + f_q

    qx, qy = _grad_vertex_vec_h(absvort_q, grid)               # vector ∇(ζ+f)
    grad_Q = jnp.sqrt(qx ** 2 + qy ** 2 + 1e-30)               # |∇(ζ+f)|_h
    div_h = divergence_cgrid(u_eff, v_eff, grid, u_mask=u_mask, v_mask=v_mask)
    grad_div = _grad_div_mag_h(div_h, grid)                     # |∇δ|_h

    # FULL QG2 (B5b): add the baroclinic stretching ∇q₁ = ∇(ζ+f) + ∂_z(f/N²∇b)
    # and apply the Bachman grid-Burger / grid-Rossby min-bound. Active only when
    # the buoyancy field is supplied (else the barotropic ∇(ζ+f) is used and the
    # bounds are inert — see `bound_qg_pv_gradient`).
    if buoyancy is not None and h_k is not None and is_3d:
        if hasattr(grid, "f_T"):
            f_h = grid.f_T[..., jnp.newaxis]                    # (n_lat,n_lon,1)
        else:
            f_h = (2.0 * getattr(grid, "omega", constants.Omega)
                   * jnp.sin(grid.lat))[:, jnp.newaxis, jnp.newaxis]
        sx, sy = qg_pv_stretching_vec(buoyancy, h_k, f_h, grid)
        grad_q1 = jnp.sqrt((qx + sx) ** 2 + (qy + sy) ** 2 + 1e-30)
        Delta_bu = jnp.sqrt(grid.area)[..., jnp.newaxis]
        Ld = (deformation_radius if deformation_radius is not None else 6.75e3)
        Bu = (Delta_bu / jnp.maximum(jnp.asarray(Ld), 1e-30)) ** 2     # Δ²/L_d²
        f_abs = jnp.maximum(jnp.abs(f_h), 1e-12)
        Ro = velocity_scale / (f_abs * Delta_bu)
        grad_Q = bound_qg_pv_gradient(grad_Q, grad_q1, Bu, Ro)

    norm = jnp.sqrt(grad_Q ** 2 + grad_div ** 2 + 1e-30)

    Delta = jnp.sqrt(grid.area)
    if is_3d:
        Delta = Delta[..., jnp.newaxis]
    nu_h = (C_qgleith * Delta / jnp.pi) ** 3 * norm            # [m²/s] at h-points
    if mask is not None:
        m = mask[..., jnp.newaxis] if is_3d else mask
        nu_h = nu_h * m

    # q-point viscosity: 4-point average of nu_h to vertices (energy-stable for
    # any nu_q >= 0; direct-at-q is a refinement).
    nu_roll = jnp.roll(nu_h, 1, axis=1)
    nu_q_int = 0.25 * (nu_h[:-1] + nu_h[1:] + nu_roll[:-1] + nu_roll[1:])
    nu_q = pad_ns_scalar(nu_q_int, grid)
    nu_q = jnp.concatenate([nu_q, nu_q[:, 0:1]], axis=1)
    nu_q = jnp.maximum(nu_q, 0.0)

    return viscous_tendency_cgrid(
        u, v, grid, nu_h, nu_q, mask=mask, u_mask=u_mask, v_mask=v_mask)


def bound_qg_pv_gradient(grad_q, grad_q_stretch, Bu, Ro):
    """Bachman et al. (2017) QG-Leith PV-gradient bound (Silvestri Eq A2-A3):

        |∇Q| = min( |∇q + stretch|, |∇q|·(1+1/Bu), |∇q|·(1+1/Ro²) )

    where ``grad_q`` = |∇(ζ+f)|, ``grad_q_stretch`` = |∇q + ∂_z(f/N²∇b)| (the
    full QGPV gradient magnitude incl. baroclinic stretching), ``Bu`` = grid
    Burger number Δ²/L_d², ``Ro`` = grid Rossby number V/(|f|Δ). The grid-Burger
    bound caps the gradient where the deformation radius is under-resolved; the
    grid-Rossby bound caps it where the flow is strongly ageostrophic; the
    closure reverts to 2D Leith where QG does not hold. Used by the full-QG2
    path of ``qg_leith_viscosity_tendency_cgrid`` when buoyancy is supplied."""
    gq2 = grad_q * (1.0 + 1.0 / jnp.maximum(Bu, 1e-30))
    gq3 = grad_q * (1.0 + 1.0 / jnp.maximum(Ro ** 2, 1e-30))
    return jnp.minimum(jnp.minimum(grad_q_stretch, gq2), gq3)


def _ddz_centre(X: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """∂X/∂z at cell centres (z increases UPWARD; level index increases DOWNWARD),
    centred in the interior + one-sided at the surface/bottom. ``X``, ``h`` are
    (..., nlev). The vertical centre-to-centre distance uses the layer thicknesses.
    """
    # Interior k=1..nlev-2: distance centre[k-1]→centre[k+1] = ½h[k-1]+h[k]+½h[k+1].
    dz_int = 0.5 * h[..., :-2] + h[..., 1:-1] + 0.5 * h[..., 2:]
    ddz_int = (X[..., :-2] - X[..., 2:]) / jnp.maximum(dz_int, 1e-12)
    dz_top = jnp.maximum(0.5 * (h[..., 0] + h[..., 1]), 1e-12)
    ddz_top = ((X[..., 0] - X[..., 1]) / dz_top)[..., jnp.newaxis]
    dz_bot = jnp.maximum(0.5 * (h[..., -2] + h[..., -1]), 1e-12)
    ddz_bot = ((X[..., -2] - X[..., -1]) / dz_bot)[..., jnp.newaxis]
    return jnp.concatenate([ddz_top, ddz_int, ddz_bot], axis=-1)


def qg_pv_stretching_vec(buoyancy: jnp.ndarray, h_k: jnp.ndarray,
                         f_h: jnp.ndarray, grid: "LatLonGrid",
                         n2_min: float = 1e-9) -> tuple:
    """Baroclinic QGPV stretching vector ∂_z(f/N²·∇b) at cell centres (Bachman
    et al. 2017 / Silvestri Eq A3 ∇q₁ stretching term).

    ``buoyancy`` b and ``h_k`` (layer thicknesses) are (n_lat, n_lon, nlev) at
    cell centres; ``f_h`` is the Coriolis parameter (n_lat, n_lon) or (n_lat, 1).
    Returns (sx, sy) each (n_lat, n_lon, nlev). N² = ∂b/∂z.

    Where the column is statically UNSTABLE or near-neutral (N² ≤ n2_min), the
    QG stretching is undefined; the contribution is set to ZERO there rather than
    dividing by a tiny floor (which would inflate f/N²·∇b by orders of magnitude
    and spuriously spike the viscosity). n2_min is a physical floor (~1e-9 s⁻²).
    """
    bx, by = _grad_cell_vec_h(buoyancy, grid)            # horizontal ∇b
    N2 = _ddz_centre(buoyancy, h_k)
    f = f_h[..., jnp.newaxis] if f_h.ndim == 2 else f_h
    # f/N² only where stably stratified; 0 elsewhere (no spurious floored spike).
    inv = jnp.where(N2 > n2_min, f / jnp.where(N2 > n2_min, N2, 1.0), 0.0)
    return _ddz_centre(inv * bx, h_k), _ddz_centre(inv * by, h_k)


def neumann_fill_vertex(
    f: jnp.ndarray,
    vtx_mask: jnp.ndarray,
    n_passes: int = 3,
) -> jnp.ndarray:
    """Fill land vertices with nearest ocean-neighbour (Neumann BC).

    Vertex-level analog of ``neumann_fill_cgrid`` for the
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
    filled = f
    # Cast the mask to the FIELD dtype so the fused (field, mask) lat
    # halo packs as ONE message per cut, not two: a float64 derived
    # vertex field + a float32 mask are distinct dtype groups and the
    # multi-pad issues one sendrecv pair per group (halo census
    # 8474554: 2 groups × 3 passes = 6 exchanges/site).  Cross-node
    # halo is latency-bound, so 2→1 message/pass halves the count.
    # For the supported BINARY vertex mask (compute_vertex_mask is a
    # product of the 0/1 cell mask; 0/1 exact in f32 and f64) the
    # up-cast is lossless and the fill arithmetic is bit-identical; a
    # fractional out-of-contract mask would differ at round-off as the
    # f32 path already did.
    m = vtx_mask.astype(f.dtype)

    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
    south_is_pole, north_is_pole = lat_ends_are_poles()
    # Under SPMD the static (south/north)_is_pole are (True, True) on every band,
    # so the per-pass Neumann edge-clamp would fire at every band's INTERIOR cut
    # (SPMD-blind) — select it DATA-dependently per band via ``axis_index``.
    from legoesm.parallel.latlon_spmd import spmd_pole_end_masks
    _spmd_pm = spmd_pole_end_masks()

    for _ in range(n_passes):
        # N/S neighbours.  Vertex rows are DUPLICATED at an MPI band
        # cut (both ranks own the shared row j0), so the neighbour row
        # beyond a band end is the neighbour rank's SECOND row — pad
        # the de-duplicated row block ``[:-1]`` (n_lat_local rows,
        # cell-like partitioning) by TWO rows through the
        # backend-dispatched pad: ``pad[k]`` holds global vertex row
        # ``j0-2+k``, delivering both the ``j0-1`` (south) and
        # ``j1+2`` (north) ghosts in a single pad call per array
        # (halo=2 → one sendrecv pair per cut side).
        # The historical Neumann edge clamp (rows 0 / n_lat copy
        # themselves) is restored at PHYSICAL band ends only — pole or
        # fold seam, where it is the serial convention — so the local
        # backend stays bit-identical while interior cuts see the true
        # neighbour rows (codex round-4 MAJOR: a land-vertex fill at a
        # cut row otherwise clamps at the band edge and diverges from
        # serial wherever land touches the cut).
        # Fused: one sendrecv pair per cut for field + mask per pass
        # (audit lever O4; ``m`` is updated each pass, so it cannot be
        # hoisted out of the loop).
        f_pad, m_pad = pad_with_pole_bc_lat_multi(
            (filled[:-1], m[:-1]), halo=2,
        )  # f_pad[k] = global vertex row j0-2+k; f_pad[1] = j0-1,
        #    f_pad[-1] = j1+2 (true neighbour rows at interior cuts).
        # Interior entries keep the EXACT historical local shifts;
        # only the two end entries are spliced from the pad ghosts.
        f_s = jnp.concatenate([f_pad[1:2], filled[:-1]], axis=0)
        m_s = jnp.concatenate([m_pad[1:2], m[:-1]], axis=0)
        f_n = jnp.concatenate([filled[1:], f_pad[-1:]], axis=0)
        m_n = jnp.concatenate([m[1:], m_pad[-1:]], axis=0)
        if _spmd_pm is not None:
            south_mask, north_mask = _spmd_pm
            f_s = jnp.where(south_mask,
                            jnp.concatenate([filled[0:1], f_s[1:]], axis=0), f_s)
            m_s = jnp.where(south_mask,
                            jnp.concatenate([m[0:1], m_s[1:]], axis=0), m_s)
            f_n = jnp.where(north_mask,
                            jnp.concatenate([f_n[:-1], filled[-1:]], axis=0), f_n)
            m_n = jnp.where(north_mask,
                            jnp.concatenate([m_n[:-1], m[-1:]], axis=0), m_n)
        else:
            if south_is_pole:
                f_s = jnp.concatenate([filled[0:1], f_s[1:]], axis=0)
                m_s = jnp.concatenate([m[0:1], m_s[1:]], axis=0)
            if north_is_pole:
                f_n = jnp.concatenate([f_n[:-1], filled[-1:]], axis=0)
                m_n = jnp.concatenate([m_n[:-1], m[-1:]], axis=0)

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


# =============================================================================
# Utility: compute face masks from cell mask
# =============================================================================

def compute_face_masks_3d(
    is_active_3d: jnp.ndarray,
    grid=None,
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
    grid : optional LatLonGrid or LatLonCGridGeometry.
        Fold face kept as wall (zero) -- see ``compute_face_masks``
        comment.  Consulted for an optional ``seam_wall_rows`` attribute
        (partial-periodic seam wall, NEMO DINO): when present, the seam
        u-face (cols 0 and n_lon) is closed on walled rows at every level.

    Returns
    -------
    u_mask_3d : array, shape (n_lat, n_lon+1, nlev)
        Wet u-face mask at each level (periodic in longitude).
    v_mask_3d : array, shape (n_lat+1, n_lon, nlev)
        Wet v-face mask at each level.
    """
    a = is_active_3d.astype(jnp.float32)
    # u-face j is between cell (j-1) mod n_lon (west) and cell j (east).
    u_mask_interior = a * jnp.roll(a, 1, axis=1)
    u_mask = jnp.concatenate(
        [u_mask_interior, u_mask_interior[:, 0:1, :]], axis=1,
    )
    # Partial-periodic seam wall (NEMO DINO): close the seam u-face at
    # walled latitude rows (all levels).  None → fully periodic.
    u_mask = _apply_seam_wall_u(u_mask, getattr(grid, "seam_wall_rows", None))
    # v-face i is between cell i-1 (south) and cell i (north).
    # Fold face kept as wall (zero) -- see compute_face_masks comment.
    v_mask_interior = a[:-1] * a[1:]
    # Meridionally-periodic (y-re-entrant) mode: boundary v-faces wrap (wet iff
    # the wrap-adjacent cells are wet at that level), not walls. Default OFF =
    # bit-identical. See compute_face_masks.
    from legoesm.grids.halo_latlon import (
        get_meridionally_flat, get_meridionally_periodic)
    if get_meridionally_periodic() or get_meridionally_flat():
        wrap = a[-1:] * a[0:1]
        v_mask = jnp.concatenate([wrap, v_mask_interior, wrap], axis=0)
        return u_mask, v_mask
    south = jnp.zeros_like(a[:1])
    north = jnp.zeros_like(south)
    v_mask = jnp.concatenate([south, v_mask_interior, north], axis=0)
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

    if is_tripolar(grid):
        return correction_full / grid.dx_u[:, :, jnp.newaxis]
    else:
        dx_u = grid.radius * grid.dlon * grid.cos_lat
        return correction_full / dx_u[:, jnp.newaxis, jnp.newaxis]


def _dy_v_full_cell_pad_first(grid) -> jnp.ndarray:
    """v-face meridional spacing at ALL ``n_lat+1`` faces, cell-pad-first.

    Interior faces: ``0.5*(dy_h[i-1] + dy_h[i])`` — the identical
    arithmetic of the historical edge-padded form.  Partition-cut faces
    (codex review 2026-06-10 MAJOR): the ghost half-height comes from
    the neighbour rank via the backend-dispatched pad, so a cut v-face
    divides by the TRUE serial metric on variable-dy (Mercator) grids —
    the previous rank-local ``mode='edge'`` clamp was wrong there at
    np>=2.  Pole faces get an INERT nonzero value (``dy_h[0]`` /
    ``dy_h[-1]`` — NOT the historical edge-midpoint; both call sites
    zero the pole-row numerators upstream, so only nonzero-ness
    matters; a future caller with nonzero pole numerators must handle
    the pole denominator itself).  ``grid.dy`` is a concrete 1-D
    metric, so under MPI the pad constant-folds at trace time
    (static-metric pad) — zero per-step comm.  Shared by
    ``partial_cell_pgf_correction_y`` and
    ``density_jacobian_pgf_smc03_y`` (one numeric source).
    """
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    if isinstance(grid.dy, jax.core.Tracer):
        # Differentiable/dynamic metric (not the current invariant —
        # grid metrics are concrete): fall back to the historical
        # rank-local edge-midpoint form rather than crashing on
        # ``float()`` below.  Wrong-at-cuts only for a traced
        # variable-dy metric, which no current configuration builds.
        dy_h = grid.dy * 0.5
        dy_v_interior = 0.5 * (dy_h[1:] + dy_h[:-1])
        return jnp.pad(dy_v_interior, (1, 1), mode="edge")
    # Pad the CONCRETE grid.dy (a jnp op like ``grid.dy * 0.5`` would
    # stage to a Tracer inside jit — defeating the static fold AND
    # crashing the float() boundary constants); scale afterwards.
    dy_np = np.asarray(grid.dy)
    dy_pad = pad_with_pole_bc_lat(
        grid.dy, halo=1,
        south_value=float(dy_np[0]), north_value=float(dy_np[-1]),
    )
    dy_h_pad = dy_pad * 0.5                               # (n_lat+2,)
    return 0.5 * (dy_h_pad[:-1] + dy_h_pad[1:])           # (n_lat+1,)


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
    # Cell-pad-first (PR357 Bug-2 pattern; see ``interp_to_v_points``):
    # pad the CELL fields so the v-face at a partition cut is built from the
    # neighbour rank's adjacent cell column (MPI halo exchange) rather than
    # from halo-padding an already-computed interior face.  ``pad_ns_zero``
    # halo-exchanges at interior cuts and zero-pads at the physical pole on
    # every rank (consistent MPI call count); ``zero_polar_lat_ends`` then
    # restores the wall BC at the physical pole only.  The fold seam is
    # overwritten on the rank that owns it (``fold_is_local``).
    # Fused: one sendrecv pair per cut for both fields (audit lever O4).
    cd_p, rp_p = pad_ns_zero_multi(centroid_depth, rho_prime)
    centroid_south, centroid_north = cd_p[:-1], cd_p[1:]
    rho_prime_south, rho_prime_north = rp_p[:-1], rp_p[1:]

    face_ref = jnp.minimum(centroid_north, centroid_south)
    excess_north = centroid_north - face_ref
    excess_south = centroid_south - face_ref

    correction = -g * (
        rho_prime_north * excess_north
        - rho_prime_south * excess_south
    )

    from legoesm.grids.halo_latlon import zero_polar_lat_ends
    correction = zero_polar_lat_ends(correction)

    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        centroid_partner = centroid_depth[-1:, grid.fold.perm_T, :]
        rho_partner = rho_prime[-1:, grid.fold.perm_T, :]
        face_ref_fold = jnp.minimum(centroid_depth[-1:], centroid_partner)
        excess_local = centroid_depth[-1:] - face_ref_fold
        excess_partner = centroid_partner - face_ref_fold
        correction_fold = -g * (
            rho_partner * excess_partner - rho_prime[-1:] * excess_local
        )
        correction = apply_north_fold(
            correction, correction_fold, grid, north_mask=nmask)

    # Divide by dy_v after the v-face correction is fully formed (both paths).
    if is_tripolar(grid):
        return correction / grid.dy_v[:, :, jnp.newaxis]
    else:
        # Regular or Mercator: variable-dy safe.  Pole rows are zero from
        # zero_polar_lat_ends, so dividing them by the edge-padded dy is inert.
        bcast = (slice(None),) + (jnp.newaxis,) * (correction.ndim - 1)
        return correction / _dy_v_full_cell_pad_first(grid)[bcast]


# =============================================================================
# Density-Jacobian PGF (Shchepetkin & McWilliams 2003) — lat-lon wrappers
# =============================================================================
# The two grid-neutral building blocks (``reconstruct_harmonic_slopes``,
# ``compute_pressure_at_target_smc03``) live in
# ``legoesm.ocean.dynamics.pgf_smc03`` so the MPAS Voronoi wrapper can
# import them without cross-importing a ``latlon_*`` module.  They are
# re-exported here for back-compat with existing call sites and tests.

from legoesm.ocean.dynamics.pgf_smc03 import (  # noqa: E402
    compute_pressure_at_target_smc03,
    reconstruct_harmonic_slopes,
)


def density_jacobian_pgf_smc03_x(
    rho_per_cell: jnp.ndarray,
    h_partial: jnp.ndarray,
    is_active: jnp.ndarray,
    grid: LatLonGrid,
    g: float,
    bottom_slope_2nd_order: bool = False,
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
    3. **Face-adaptive z_target** = ``min(z_centroid_W, z_centroid_E)`` — the
       *shallower* of the two cell centroids (see the code below, which uses
       ``jnp.minimum``).  At full-cell faces this reduces to the standard
       reference-cell centroid (both centroids equal).  The ``min`` (NOT the
       midpoint ``0.5·(z_c_W+z_c_E)`` once tried as "Option B") is what
       guarantees the target lies inside BOTH columns: the midpoint can fall
       *below* the shallower column's seafloor when its partial cell is thin
       (``h < dz/3``), producing an asymmetric seafloor clamp and a spurious
       ~10⁶ Pa/face pressure gradient — the C1 bug that drove the BH-seamount
       blowup (see ``docs/ocean/experiments/pgf_smc03_code_review.md``).
       ``min`` matches the Adcroft & Campin 2004 /
       ``partial_cell_pgf_correction_x`` convention
       (``face_ref = jnp.minimum(centroid_east, centroid_west)``).
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
    sigma = reconstruct_harmonic_slopes(
        rho_per_cell, z_centroid, is_active,
        bottom_slope_2nd_order=bottom_slope_2nd_order,
    )

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

    if is_tripolar(grid):
        return diff / grid.dx_u[:, :, jnp.newaxis]
    else:
        dx_u = grid.radius * grid.dlon * grid.cos_lat
        return diff / dx_u[:, jnp.newaxis, jnp.newaxis]


def density_jacobian_pgf_smc03_y(
    rho_per_cell: jnp.ndarray,
    h_partial: jnp.ndarray,
    is_active: jnp.ndarray,
    grid: LatLonGrid,
    g: float,
    bottom_slope_2nd_order: bool = False,
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
    sigma = reconstruct_harmonic_slopes(
        rho_per_cell, z_centroid, is_active,
        bottom_slope_2nd_order=bottom_slope_2nd_order,
    )

    # Cell-pad-first (PR357 Bug-2 pattern; see ``interp_to_v_points``): pad
    # the CELL columns so the v-face PGF at a partition cut is built from the
    # neighbour rank's adjacent column (MPI halo exchange) rather than from
    # halo-padding an already-computed interior face.  ``pad_ns_zero``
    # halo-exchanges at interior cuts and zero-pads at the physical pole on
    # every rank (consistent MPI call count); ``zero_polar_lat_ends`` then
    # restores the wall BC at the physical pole.  ``z_centroid``/``sigma`` are
    # per-column quantities, so the halo-exchanged neighbour column is exact.
    # Fused: one sendrecv pair per cut for all four fields (audit lever O4).
    rho_p, h_p, zc_p, sig_p = pad_ns_zero_multi(
        rho_per_cell, h_partial, z_centroid, sigma,
    )
    rho_S, rho_N = rho_p[:-1], rho_p[1:]
    h_S, h_N = h_p[:-1], h_p[1:]
    z_c_S, z_c_N = zc_p[:-1], zc_p[1:]
    sigma_S, sigma_N = sig_p[:-1], sig_p[1:]

    # Shallower-of-centroids (Adcroft & Campin convention; see x-direction
    # operator for the rationale and the C1 bug it resolves).
    z_target_face = jnp.minimum(z_c_S, z_c_N)
    P_N = compute_pressure_at_target_smc03(
        rho_N, h_N, z_c_N, sigma_N, z_target_face, g,
    )
    P_S = compute_pressure_at_target_smc03(
        rho_S, h_S, z_c_S, sigma_S, z_target_face, g,
    )
    diff = P_N - P_S  # (n_lat+1, n_lon, nlev)

    from legoesm.grids.halo_latlon import zero_polar_lat_ends
    diff = zero_polar_lat_ends(diff)

    # North fold seam: the rank / SPMD north band that owns it overwrites the
    # north row (data-dependent under SPMD via north_fold_mask).
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        fold = grid.fold
        rho_F = rho_per_cell[-1:, fold.perm_T, :]
        h_F = h_partial[-1:, fold.perm_T, :]
        z_c_F = z_centroid[-1:, fold.perm_T, :]
        sigma_F = sigma[-1:, fold.perm_T, :]
        z_c_L = z_centroid[-1:]
        z_target_fold = jnp.minimum(z_c_L, z_c_F)
        P_fold = compute_pressure_at_target_smc03(
            rho_F, h_F, z_c_F, sigma_F, z_target_fold, g,
        )
        P_local = compute_pressure_at_target_smc03(
            rho_per_cell[-1:], h_partial[-1:], z_c_L,
            sigma[-1:], z_target_fold, g,
        )
        diff_fold = P_fold - P_local
        diff = apply_north_fold(diff, diff_fold, grid, north_mask=nmask)

    if is_tripolar(grid):
        # Tripolar: divide by full 2D dy_v.
        dy_v = grid.dy_v  # full 2D
        return diff / dy_v[:, :, jnp.newaxis]
    else:
        # Regular or Mercator: variable-dy safe.  diff has shape
        # (n_lat+1, n_lon, nlev); pole rows are zeroed by
        # zero_polar_lat_ends above, so their denominator is inert.
        bcast = (slice(None),) + (jnp.newaxis,) * (diff.ndim - 1)
        return diff / _dy_v_full_cell_pad_first(grid)[bcast]


def pv_flux_ene(
    zeta: jnp.ndarray,
    h_vtx: jnp.ndarray,
    h_v: jnp.ndarray,
    v: jnp.ndarray,
    h_u: jnp.ndarray,
    u: jnp.ndarray,
    u_mask_3d: jnp.ndarray,
    v_mask_3d: jnp.ndarray,
    vtx_mask: jnp.ndarray,
    f_vtx: jnp.ndarray | None = None,
    eps_h: float = 1.0e-10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NEMO ``vor_ene`` — Sadourny (1975) ENERGY-conserving 2-point PV flux.

    Transcription of NEMO ``dynvor.F90::vor_ene`` (the GYRE default
    ``ln_dynvor_ene``) into the ``latlon_cgrid_operators`` index convention
    (see :func:`pv_flux_al81_partial_cell` for the vertex/face indexing).
    Faithful to NEMO's stencil pairing/signs, but NOT term-for-term on a
    varying-``cos(φ)`` grid: this module folds the metric into ``ζ`` and carries
    bare ``h·u``/``h·v`` mass fluxes, dropping NEMO's explicit ``e1v/e1u`` =
    ``cos(φ_v)/cos(φ_u)`` half-cell ratio (an ``O(dφ·tan φ)`` difference —
    sub-1% across GYRE's 24-44°N, larger at high latitude). This matches the
    AL81 sibling's convention and conserves this module's discrete energy exactly.
    NEMO forms the potential vorticity ``q = (f + ζ)/e3f`` at F-points and
    applies the Sadourny 2-point averaging::

        voru = 1/e1u · mj-1[ q · mi(e1v·e3v·v) ]       (NEMO comment)
        vorv = 1/e2v · mi-1[ q · mj(e2u·e3u·u) ]

    which in this module's metric-in-ζ convention (``ζ`` from
    :func:`curl_vertex_cgrid` already carries the ``1/e1e2f`` area factor, and
    the ``e1v/e1u`` horizontal metric cancels on a regular grid) reduces to::

        du/dt[j,i] = +¼ ( q[j  ,i]·(F_v[j  ,i-1]+F_v[j  ,i])
                        + q[j+1,i]·(F_v[j+1,i-1]+F_v[j+1,i]) )
        dv/dt[j,i] = -¼ ( q[j,i  ]·(F_u[j-1,i  ]+F_u[j,i  ])
                        + q[j,i+1]·(F_u[j-1,i+1]+F_u[j,i+1]) )

    with ``F_v = h·v`` (v-face mass flux), ``F_u = h·u`` (u-face mass flux).
    Each F-point ``q`` multiplies ONLY the two mass fluxes on its own side
    (south/north for u, west/east for v) — the property that makes the scheme
    exactly energy-conserving (Sadourny 1975; the paired ``¼·q·u·v`` terms
    appear identically in ``du`` and ``dv`` and cancel in ``Σ u·du+v·dv``).

    This is the ENE sibling of :func:`pv_flux_al81_partial_cell` (EEN/AL81,
    the 12-point triad).  On a uniform, fully-wet grid ENE is the plain
    2-point Sadourny form, AL81 the 9-vertex energy-AND-enstrophy compromise;
    they differ at the grid scale.

    Parameters
    ----------
    f_vtx : (n_lat+1, n_lon+1) or None
        Planetary Coriolis at F-points (vertices).  When given, the FULL ENE
        operator ``q = (f+ζ)/h`` (NEMO ``np_CRV``) — replaces BOTH the
        planetary Coriolis and the relative-vorticity flux.  When ``None``,
        the relative-only form ``q = ζ/h`` (NEMO ``np_RVO``), for isolating
        the ``rvo`` trend.
    Other parameters : identical to :func:`pv_flux_al81_partial_cell`.

    Returns
    -------
    (diag_vortcor_u, diag_vortcor_v) : the ``+q·F_v`` / ``-q·F_u`` momentum
    tendency contributions, same shapes as the AL81 sibling.
    """
    # --- 1. PV at vertices.  q = (f + ζ)/h_vtx (CRV) or ζ/h_vtx (RVO) ---
    # ``h_vtx`` carries the BIG_H sentinel at dry vertices so q ≈ 0 there.
    # f is added BEFORE the /h division (matching NEMO: zwz = ff_f + ζ,
    # then zwz /= e3f) so the planetary term also gets the F-point
    # thickness weighting — the two are one operator, not two.
    total_vort = zeta if f_vtx is None else (zeta + f_vtx[..., jnp.newaxis])
    q = total_vort / jnp.maximum(h_vtx, eps_h)
    # Neumann-fill only the RELATIVE part's discontinuity at the coast; the
    # planetary f is smooth everywhere, so fill the whole q (idempotent at
    # interior wet vertices).
    q = neumann_fill_vertex(q, vtx_mask)

    # --- 2. Mass fluxes at u/v faces (h·u, h·v), closed faces → 0 ---
    F_u = h_u * u * u_mask_3d            # (n_lat, n_lon+1, nlev)
    F_v = h_v * v * v_mask_3d            # (n_lat+1, n_lon, nlev)

    # --- 3. u-face flux: ¼ ( q_S·(F_v_SW+F_v_SE) + q_N·(F_v_NW+F_v_NE) ) ---
    # q already has shape (n_lat+1, n_lon+1, nlev) with the periodic wrap
    # column, so q[:-1]/q[1:] are the south/north vertices of each u-face.
    q_S_u = q[:-1, :, :]                 # (n_lat, n_lon+1, nlev)
    q_N_u = q[1:, :, :]
    # F_v at the four u-face corners — identical construction to AL81.
    F_v_south = F_v[:-1, :, :]           # south v-face of each u-row
    F_v_north = F_v[1:, :, :]
    F_v_S_E = jnp.concatenate([F_v_south, F_v_south[:, 0:1, :]], axis=1)
    F_v_N_E = jnp.concatenate([F_v_north, F_v_north[:, 0:1, :]], axis=1)
    F_v_S_W = jnp.roll(F_v_S_E, 1, axis=1)
    F_v_N_W = jnp.roll(F_v_N_E, 1, axis=1)
    diag_vortcor_u = 0.25 * (
        q_S_u * (F_v_S_W + F_v_S_E) + q_N_u * (F_v_N_W + F_v_N_E)
    )

    # --- 4. v-face flux: -¼ ( q_W·(F_u_SW+F_u_NW) + q_E·(F_u_SE+F_u_NE) ) ---
    # q[:, :-1]/q[:, 1:] are the west/east vertices of each v-face.
    q_W_v = q[:, :-1, :]                 # (n_lat+1, n_lon, nlev)
    q_E_v = q[:, 1:, :]
    # F_u at the four v-face corners — MPI/pole handling identical to AL81
    # (zero-pad at physical poles, halo sendrecv at interior band cuts).
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
    (F_u_pad,) = pad_with_pole_bc_lat_multi((F_u,), halo=1)  # (n_lat+2, n_lon+1, nlev)
    F_u_south = F_u_pad[:-1, :, :]       # (n_lat+1, n_lon+1, nlev)
    F_u_north = F_u_pad[1:, :, :]
    F_u_S_W = F_u_south[:, :-1, :]       # (n_lat+1, n_lon, nlev)
    F_u_S_E = F_u_south[:, 1:, :]
    F_u_N_W = F_u_north[:, :-1, :]
    F_u_N_E = F_u_north[:, 1:, :]
    diag_vortcor_v = -0.25 * (
        q_W_v * (F_u_S_W + F_u_N_W) + q_E_v * (F_u_S_E + F_u_N_E)
    )

    return diag_vortcor_u, diag_vortcor_v


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
    f_vtx: jnp.ndarray | None = None,
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
    # --- 1. PV at vertices ----------------------------------------
    # ``q = (f + ζ)/h_vtx`` (ABSOLUTE vorticity, NEMO ``np_CRV`` EEN) when
    # ``f_vtx`` is given, else ``q = ζ/h_vtx`` (relative-only, ``np_RVO``).
    # f is added BEFORE the /h division (matching NEMO ``vor_een``:
    # ``zwz = ff_f + ζ`` then ``zwz /= e3f``) so the planetary term rides
    # the SAME enstrophy-conserving 12-point triad as the relative
    # vorticity — the two are ONE operator, not two.  This is the
    # difference between the split ``al81`` (ζ-only EEN + a separate
    # non-enstrophy-conserving 4-pt Coriolis) and the faithful ``een_total``.
    # ``h_vtx`` already carries the BIG_H sentinel at fully-dry vertices
    # (set by the caller) so q ≈ 0 there; eps_h guards floating-point edges.
    zeta_abs = zeta if f_vtx is None else zeta + f_vtx[..., jnp.newaxis]
    q = zeta_abs / jnp.maximum(h_vtx, eps_h)

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
    #
    # Backend-dispatched pad (np>=2 parity): the triads are CELL-row
    # quantities, so at an interior MPI band cut the ghost row must be
    # the neighbour rank's edge-cell triads (AD-safe sendrecv) — a
    # plain jnp.pad zero treated the cut like a pole wall and silently
    # dropped the south/north-cell half of the 12-point stencil at the
    # shared v-face.  The corruption is invisible while u = 0 (the
    # triads multiply F_u = h·u), which is why a cold-start first step
    # stays bit-exact and the error appears from step 2.  Physical
    # poles still get exactly 0 (bit-identical to the old jnp.pad on
    # the local backend).  All four triads ride in ONE pad
    # (concatenated along the level axis — pure relabeling, no
    # arithmetic) so the per-RHS collective count grows by exactly two
    # sendrecv pairs (this pad + the F_u pad below), uniformly on
    # every rank.
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
    t_stack = jnp.concatenate([t_NW, t_NE, t_SW, t_SE], axis=-1)
    # Fused with the F_u pad below: ONE sendrecv pair per cut for both
    # (audit lever O4); t_stack and F_u are independent inputs here.
    t_stack_pad, F_u_pad = pad_with_pole_bc_lat_multi(
        (t_stack, F_u), halo=1,
    )  # (n_lat+2, n_lon, 4*nlev), (n_lat+2, n_lon+1, nlev)
    t_NW_pad, t_NE_pad, t_SW_pad, t_SE_pad = jnp.split(t_stack_pad, 4, axis=-1)
    t_NW_S = t_NW_pad[:-1, :, :]   # south-cell NW at v-face j
    t_NE_S = t_NE_pad[:-1, :, :]
    t_SW_N = t_SW_pad[1:, :, :]    # north-cell SW at v-face j
    t_SE_N = t_SE_pad[1:, :, :]

    # F_u at the four offsets, mapped to v-face index.  At v-face
    # (j, i), we need:
    #   F_u_S_W = F_u[j-1, i  , :]   (south-west of v-face)
    #   F_u_S_E = F_u[j-1, i+1, :]
    #   F_u_N_W = F_u[j  , i  , :]
    #   F_u_N_E = F_u[j  , i+1, :]
    # F_u has shape (n_lat, n_lon+1, nlev); pad in axis 0 to align
    # with v-face row index (rows 0..n_lat for v).  Backend-dispatched
    # for the same reason as the triad pad above: at an MPI band cut
    # the v-face needs the neighbour rank's edge F_u row (zero at
    # physical poles — bit-identical to the old jnp.pad locally).
    # F_u_pad computed in the fused exchange with t_stack above.
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


def _apply_seam_wall_u(u_mask: jnp.ndarray, seam_wall_rows) -> jnp.ndarray:
    """Close the periodic-seam u-faces on walled latitude rows.

    The periodic wrap is stored redundantly at BOTH u-face column 0 and
    the appended column ``n_lon`` (the same physical seam face), so both
    are zeroed on a walled row.  ``seam_wall_rows`` is ``(n_lat,)`` with
    ``1.0`` = walled.  ``None`` returns ``u_mask`` unchanged (byte-
    identical).  Works for 2-D ``(n_lat, n_lon+1)`` and 3-D
    ``(n_lat, n_lon+1, nlev)`` masks (broadcast over levels).
    """
    if seam_wall_rows is None:
        return u_mask
    open_rows = (1.0 - jnp.asarray(seam_wall_rows)).astype(u_mask.dtype)
    gate = open_rows[:, None] if u_mask.ndim == 3 else open_rows  # (n_lat[,1])
    u_mask = u_mask.at[:, 0].multiply(gate)
    u_mask = u_mask.at[:, -1].multiply(gate)
    return u_mask


def compute_face_masks(
    land_mask: jnp.ndarray,
    grid=None,
    seam_wall_rows=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Derive u-face and v-face masks from cell-center land mask.

    A face is wet only if both adjacent cells are wet.

    Parameters
    ----------
    land_mask : array, shape (n_lat, n_lon)
        Cell-center ocean mask (1 = ocean, 0 = land).
    grid : optional LatLonGrid or LatLonCGridGeometry.
        Fold face kept as wall (zero) until a proper halo exchange
        architecture (Option B) is implemented.  Consulted for an
        optional ``seam_wall_rows`` attribute (partial-periodic seam
        wall) when the explicit ``seam_wall_rows`` argument is ``None``.
    seam_wall_rows : optional array, shape (n_lat,)
        ``1.0`` = the periodic-seam u-face is walled at that latitude
        row, ``0.0`` = open.  ``None`` (default) → fully periodic (byte-
        identical).  Falls back to ``grid.seam_wall_rows`` when unset.

    Returns
    -------
    u_mask : array, shape (n_lat, n_lon+1)
        Mask at zonal (lon) interfaces.
    v_mask : array, shape (n_lat+1, n_lon)
        Mask at meridional (lat) interfaces.
    """
    if seam_wall_rows is None and grid is not None:
        seam_wall_rows = getattr(grid, "seam_wall_rows", None)
    # u-face j is between cell (j-1) mod n_lon and cell j (periodic in lon)
    u_mask_interior = land_mask * jnp.roll(land_mask, 1, axis=1)
    # Append periodic wrap
    u_mask = jnp.concatenate(
        [u_mask_interior, u_mask_interior[:, 0:1]], axis=1,
    )
    u_mask = _apply_seam_wall_u(u_mask, seam_wall_rows)

    # v-face i is between cell i and cell i+1.
    v_mask_interior = land_mask[:-1] * land_mask[1:]
    # Meridionally-periodic (y-re-entrant channel) mode: the south boundary
    # v-face (between cell N-1 and cell 0, wrapping) and the identical north
    # boundary v-face are WET when both wrap-adjacent cells are wet -- NOT walls.
    # Default OFF -> the historical hard-walled N/S v-faces (bit-identical).
    from legoesm.grids.halo_latlon import (
        get_meridionally_flat, get_meridionally_periodic)
    if get_meridionally_periodic() or get_meridionally_flat():
        wrap = (land_mask[-1:] * land_mask[0:1]).astype(land_mask.dtype)
        v_mask = jnp.concatenate([wrap, v_mask_interior, wrap], axis=0)
        return u_mask, v_mask
    south = jnp.zeros((1, land_mask.shape[1]), dtype=land_mask.dtype)
    # The fold face is kept as a wall (zero) until a proper halo
    # exchange architecture (Option B) is implemented.  Opening the
    # fold face without consistent halo exchange creates fold-asymmetry
    # that drives an instability over ~40 steps.
    north = jnp.zeros_like(south)
    v_mask = jnp.concatenate([south, v_mask_interior, north], axis=0)

    return u_mask, v_mask
