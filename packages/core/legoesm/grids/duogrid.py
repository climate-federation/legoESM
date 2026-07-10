"""Duo-Grid kinked-to-extended remapping for cubed-sphere halo exchange.

Implements the Duo-Grid approach of Mouallem, Harris & Chen (2023) for
eliminating face-boundary artifacts on the cubed sphere. Two coordinate
systems are defined:

  - **Extended grid**: smooth analytic continuation of each face's own
    gnomonic projection into the halo region.
  - **Kinked grid**: where neighbor-face data sits after a standard copy
    (creating a coordinate kink at the face edge).

Precomputed Lagrange interpolation coefficients remap halo data from
kinked to extended positions, so the dynamical core's stencils operate
on a single consistent coordinate system.

Reference Fortran: Zenodo 8327578 (atmos_cubed_sphere-symmetryclean).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

# Maximum interpolation order (fixed array dimension for JAX pytree compat)
MAX_K2E_NORD = 4


# ============================================================================
# Data structure
# ============================================================================

class DuoGridData(NamedTuple):
    """Precomputed Duo-Grid kinked-to-extended remapping data.

    All arrays are static (computed once at grid creation). Integer index
    arrays use int32. Floating-point arrays use float64 for coefficient
    precision. All shapes are fixed for a given (n, ng, k2e_nord).
    """
    n: int
    ng: int
    k2e_nord: int

    # --- Edge remapping coefficients ---
    # k2e_coef[face, edge, depth, j, :MAX_K2E_NORD]: Lagrange weights
    # k2e_lo[face, edge, depth, j]: stencil start index (0-based)
    # Per-face per-edge because equidistant gnomonic breaks face symmetry.
    k2e_coef: jax.Array   # (6, 4, ng, n, MAX_K2E_NORD) float64
    k2e_lo: jax.Array     # (6, 4, ng, n) int32

    # --- Extended grid lon/lat ---
    # A-grid positions on the extended grid. Shape (6, n+2*ng, n+2*ng).
    # Used for corner fill coefficient computation and diagnostics.
    ext_lon: jax.Array   # float64
    ext_lat: jax.Array   # float64

    # --- Lagrange corner interpolation coefficients ---
    # Precomputed 4th-order Lagrange polynomials for fill_corner_region.
    # Following FV3 fv_duogrid.F90 compute_lagrange_coeff.
    # xp: coefficients for X+ direction (east edge → NE/SE corners)
    # xm: coefficients for X- direction (west edge → NW/SW corners)
    # yp: coefficients for Y+ direction (north edge → NE/NW corners)
    # ym: coefficients for Y- direction (south edge → SE/SW corners)
    # Each: (6, interp_order+1, n_target, n_source_range) or None
    corner_xp: jax.Array | None  # (6, 4, ng, n) or None
    corner_xm: jax.Array | None
    corner_yp: jax.Array | None
    corner_ym: jax.Array | None

    # --- Extended-grid unit vectors for ext_vector (a2stag_metrics) ---
    # vlon_ext: unit vector in longitude direction at A-grid positions
    # vlat_ext: unit vector in latitude direction at A-grid positions
    # Shape: (6, n+2*ng, n+2*ng, 3) — 3D Cartesian
    vlon_ext: jax.Array | None
    vlat_ext: jax.Array | None
    # ew_ext: east-west edge-normal vectors (2 variants per FV3 a2stag_metrics)
    # Shape: (6, n+2*ng+1, n+2*ng, 3, 2) — last dim: variant 0=A-grid, 1=B-grid
    # es_ext: south-north edge-normal vectors
    # Shape: (6, n+2*ng, n+2*ng+1, 3, 2)
    # cubed_a2d_halo uses: ud <- es[:,:,:,0], vd <- ew[:,:,:,1]
    ew_ext: jax.Array | None
    es_ext: jax.Array | None


# ============================================================================
# Precompute: supergrid, kinked grid, coordinates, coefficients
# ============================================================================

def _cart2lonlat(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Cartesian (x, y, z) on unit sphere → (lon, lat) in radians."""
    lon = np.mod(np.arctan2(y, x), 2.0 * np.pi)  # [0, 2π)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    return lon, lat


def _face_to_cartesian_np(face: int, tan_x, tan_y):
    """Local gnomonic tangent coords → 3D Cartesian (matches cubed_sphere.py)."""
    ones = np.ones_like(tan_x)
    if face == 0:    return ones, tan_x, tan_y
    elif face == 1:  return -tan_x, ones, tan_y
    elif face == 2:  return -ones, -tan_x, tan_y
    elif face == 3:  return tan_x, -ones, tan_y
    elif face == 4:  return -tan_y, tan_x, ones
    elif face == 5:  return tan_y, tan_x, -ones
    else: raise ValueError(face)


def _build_extended_grid(n: int, ng: int):
    """Build extended-grid lon/lat at A-grid centers for all 6 faces.

    The extended grid continues each face's equidistant gnomonic projection
    smoothly into the halo region (ng cells on each side).

    Returns
    -------
    ext_lon, ext_lat : np.ndarray, shape (6, n + 2*ng, n + 2*ng)
    """
    n_ext = n + 2 * ng
    dalpha = np.pi / (2.0 * n)

    # A-grid cell centers: alpha[k] for k = 0..n_ext-1
    # Interior cells (k = ng..ng+n-1) have alpha in [-pi/4+dalpha/2, pi/4-dalpha/2]
    # Halo cells extend beyond ±pi/4.
    alpha = np.array([
        -np.pi / 4.0 + (k - ng + 0.5) * dalpha for k in range(n_ext)
    ])
    tan_ax, tan_ay = np.meshgrid(np.tan(alpha), np.tan(alpha), indexing='ij')

    ext_lon = np.zeros((6, n_ext, n_ext))
    ext_lat = np.zeros((6, n_ext, n_ext))

    for face in range(6):
        x, y, z = _face_to_cartesian_np(face, tan_ax, tan_ay)
        r = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / r, y / r, z / r
        ext_lon[face], ext_lat[face] = _cart2lonlat(x, y, z)

    return ext_lon, ext_lat


def _build_kinked_grid(n: int, ng: int, ext_lon, ext_lat):
    """Build kinked-grid lon/lat by copying neighbor face data into halos.

    Interior cells match the extended grid exactly. Halo cells are filled
    from the neighbor face's interior via the CONNECTIVITY table.

    Returns
    -------
    kik_lon, kik_lat : np.ndarray, shape (6, n + 2*ng, n + 2*ng)
    """
    n + 2 * ng
    kik_lon = ext_lon.copy()
    kik_lat = ext_lat.copy()

    for face in range(6):
        for edge in [WEST, EAST, SOUTH, NORTH]:
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for d in range(ng):
                for j in range(n):
                    k = (n - 1 - j) if is_reversed else j

                    # Source: neighbor's interior cell
                    if nbr_edge == WEST:
                        si, sj = ng + d, ng + k
                    elif nbr_edge == EAST:
                        si, sj = ng + n - 1 - d, ng + k
                    elif nbr_edge == SOUTH:
                        si, sj = ng + k, ng + d
                    else:  # NORTH
                        si, sj = ng + k, ng + n - 1 - d

                    # Destination: this face's halo cell
                    if edge == WEST:
                        di, dj = ng - 1 - d, ng + j
                    elif edge == EAST:
                        di, dj = ng + n + d, ng + j
                    elif edge == SOUTH:
                        di, dj = ng + j, ng - 1 - d
                    else:  # NORTH
                        di, dj = ng + j, ng + n + d

                    kik_lon[face, di, dj] = ext_lon[nbr_face, si, sj]
                    kik_lat[face, di, dj] = ext_lat[nbr_face, si, sj]

    return kik_lon, kik_lat


def _xyz_to_gnomonic_alpha(face: int, x, y, z):
    """Project (x,y,z) onto face's gnomonic plane, return (alpha_x, alpha_y).

    Matches the inverse of _face_to_cartesian_np.
    """
    if face == 0:    return np.arctan2(y, x), np.arctan2(z, x)
    elif face == 1:  return np.arctan2(-x, y), np.arctan2(z, y)
    elif face == 2:  return np.arctan2(-y, -x), np.arctan2(z, -x)
    elif face == 3:  return np.arctan2(x, -y), np.arctan2(z, -y)
    elif face == 4:  return np.arctan2(y, z), np.arctan2(-x, z)
    elif face == 5:  return np.arctan2(-y, -z), np.arctan2(-x, -z)
    else: raise ValueError(face)


def _extract_1d_coords(n: int, ng: int, ext_lon, ext_lat, kik_lon, kik_lat):
    """Extract 1D coordinates along each edge strip at each halo depth.

    Uses **gnomonic alpha** (the face's own projection coordinate) as
    the 1D coordinate instead of latitude. This ensures monotonicity
    for all edges including cross-axis connections (e.g., equatorial
    face SOUTH to polar face NORTH), where latitude is non-monotone.

    For WEST/EAST edges the strip varies along axis 2 (j), so we use
    alpha_y as the 1D coordinate. For SOUTH/NORTH edges the strip
    varies along axis 1 (i), so we use alpha_x.

    The extended grid's alpha is uniform by construction. The kinked
    data's alpha is computed by projecting its (lon,lat) back onto the
    face's gnomonic plane.

    Returns
    -------
    ext_1d : np.ndarray, shape (6, 4, ng, n)
    kik_1d : np.ndarray, shape (6, 4, ng, n)
    """
    ext_1d = np.zeros((6, 4, ng, n))
    kik_1d = np.zeros((6, 4, ng, n))

    for face in range(6):
        for edge in [WEST, EAST, SOUTH, NORTH]:
            for d in range(ng):
                for j in range(n):
                    if edge == WEST:
                        i_pos, j_pos = ng - 1 - d, ng + j
                    elif edge == EAST:
                        i_pos, j_pos = ng + n + d, ng + j
                    elif edge == SOUTH:
                        i_pos, j_pos = ng + j, ng - 1 - d
                    else:
                        i_pos, j_pos = ng + j, ng + n + d

                    # Extended grid: use gnomonic alpha directly
                    elon = ext_lon[face, i_pos, j_pos]
                    elat = ext_lat[face, i_pos, j_pos]
                    ex = np.cos(elat) * np.cos(elon)
                    ey = np.cos(elat) * np.sin(elon)
                    ez = np.sin(elat)
                    eax, eay = _xyz_to_gnomonic_alpha(face, ex, ey, ez)

                    # Kinked grid: project back onto this face
                    klon = kik_lon[face, i_pos, j_pos]
                    klat = kik_lat[face, i_pos, j_pos]
                    kx = np.cos(klat) * np.cos(klon)
                    ky = np.cos(klat) * np.sin(klon)
                    kz = np.sin(klat)
                    kax, kay = _xyz_to_gnomonic_alpha(face, kx, ky, kz)

                    # Use the coordinate along the strip direction
                    if edge in (WEST, EAST):
                        ext_1d[face, edge, d, j] = eay
                        kik_1d[face, edge, d, j] = kay
                    else:
                        ext_1d[face, edge, d, j] = eax
                        kik_1d[face, edge, d, j] = kax

    # Unwrap kinked alpha to handle ±π boundary crossings at deep halos.
    # The extended grid is always smooth (no unwrapping needed).
    for face in range(6):
        for edge in range(4):
            for d in range(ng):
                kik_1d[face, edge, d, :] = np.unwrap(
                    kik_1d[face, edge, d, :])

    return ext_1d, kik_1d


def _lagrange_coef(target: float, sources: np.ndarray) -> np.ndarray:
    """Compute standard Lagrange interpolation weights."""
    k = len(sources)
    weights = np.ones(k)
    for i in range(k):
        for j in range(k):
            if i != j:
                denom = sources[i] - sources[j]
                if abs(denom) < 1e-30:
                    weights[i] = 0.0
                    break
                weights[i] *= (target - sources[j]) / denom
    return weights


def _compute_k2e_coefficients(n: int, ng: int, k2e_nord: int,
                               ext_1d, kik_1d):
    """Compute kinked-to-extended Lagrange interpolation coefficients.

    Per-face per-edge coefficients because the equidistant gnomonic
    projection breaks the equal-edge face/edge symmetry.

    Returns
    -------
    k2e_coef : np.ndarray, shape (6, 4, ng, n, MAX_K2E_NORD)
    k2e_lo : np.ndarray, shape (6, 4, ng, n) int32
    """
    k2e_coef = np.zeros((6, 4, ng, n, MAX_K2E_NORD))
    k2e_lo = np.zeros((6, 4, ng, n), dtype=np.int32)
    np_pad = max(k2e_nord // 2 - 1, 0)

    for face in range(6):
        for edge in range(4):
            for d in range(ng):
                kik_strip = kik_1d[face, edge, d, :]
                ext_strip = ext_1d[face, edge, d, :]

                diffs = np.diff(kik_strip)
                ascending = True
                if n > 2:
                    nonzero = diffs[np.abs(diffs) > 1e-15]
                    if len(nonzero) > 0:
                        ascending = np.all(nonzero > 0)
                        if not ascending and not np.all(nonzero < 0):
                            raise ValueError(
                                f"kik_1d not monotone: face={face} "
                                f"edge={edge} depth={d}"
                            )

                for j in range(n):
                    target = ext_strip[j]
                    if ascending:
                        idx = int(np.searchsorted(kik_strip, target))
                    else:
                        idx = n - int(np.searchsorted(
                            kik_strip[::-1], target))

                    lo = max(idx - 1, 0)
                    lo = max(lo, np_pad)
                    lo = min(lo, n - np_pad - 2)

                    stencil_start = lo - np_pad
                    stencil_end = stencil_start + k2e_nord
                    stencil_start = max(stencil_start, 0)
                    stencil_end = min(stencil_end, n)
                    stencil_pos = kik_strip[stencil_start:stencil_end]

                    if len(stencil_pos) >= 2:
                        coef = _lagrange_coef(target, stencil_pos)
                        k2e_coef[face, edge, d, j, :len(coef)] = coef
                        k2e_lo[face, edge, d, j] = stencil_start
                    else:
                        k2e_coef[face, edge, d, j, 0] = 1.0
                        k2e_lo[face, edge, d, j] = min(
                            max(idx, 0), n - 1)

    # Verify partition of unity
    coef_sums = k2e_coef.sum(axis=-1)
    max_err = np.max(np.abs(coef_sums - 1.0))
    if max_err > 1e-12:
        raise ValueError(
            f"k2e_coef partition of unity violated: "
            f"max |sum - 1| = {max_err:.2e}"
        )

    return k2e_coef, k2e_lo


def _great_circle_dist(lon1, lat1, lon2, lat2):
    """Great-circle angular distance between two points."""
    dlon = lon2 - lon1
    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)
    tmp1 = (cos_lat2 * np.sin(dlon))**2 + (
        cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * np.cos(dlon))**2
    tmp2 = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * np.cos(dlon)
    return np.arctan2(np.sqrt(np.maximum(tmp1, 0.0)), tmp2)


def _compute_corner_lagrange_coeff(n: int, ng: int, ext_lon, ext_lat):
    """Compute 4th-order Lagrange corner interpolation coefficients.

    Following FV3 fv_duogrid.F90 compute_lagrange_coeff and
    lagrange_poly_interp_2d. Uses great-circle distance as the
    interpolation metric (matching FV3).

    For each corner cell (i, j) in the ng×ng corner region, precompute
    Lagrange polynomial weights for X+, X-, Y+, Y- directions using the
    last ``interporder+1`` interior cells along the corresponding edge.

    Returns
    -------
    xp, xm, yp, ym : np.ndarray, shape (6, 4, ng_target, n_total)
        Lagrange weights. For a given target halo cell and source point,
        xp[face, src_idx, target_idx, :] gives the weight.
        Actually stored as (6, interporder+1, ng_ext, n_padded) where
        ng_ext covers the target positions and n_padded = n_source.

    For simplicity and JAX compatibility, we store per-face coefficients as:
        corner_xp[face, k, i_target] = Lagrange weight for source k at target i
    Shape: (6, interp_order+1, ng) per edge-parallel strip position.
    """
    interp_order = 3  # FV3 default: interporder=3 → 4 stencil points
    n_stencil = interp_order + 1  # 4

    # 4-point Lagrange stencil requires n >= 4; fall back to averaging
    if n < n_stencil:
        return None, None, None, None

    n_ext = n + 2 * ng

    # For each face, compute weights for each direction and target position.
    # FV3 uses great_circle_dist for the interpolation abscissae.
    #
    # X+ direction: interpolate from the last `n_stencil` interior cells
    # along the east (high-i) edge into the halo at i > ng+n-1.
    # X- direction: from the first cells along the west (low-i) edge.
    # Y+ direction: from the last cells along the north (high-j) edge.
    # Y- direction: from the first cells along the south (low-j) edge.

    # Shapes: (6, n_stencil, ng, n_ext) — one weight per target position
    # in the halo, for each source cell in the stencil, for each j along
    # the edge, for each face.
    # But since corner fill only needs ng×ng corner cells, and the j-index
    # within the corner is either in the edge-halo or interior, we compute
    # for all j positions that fill_corner_region will need.

    # Simplified storage: for each face and direction, store (n_stencil, n_ext, n_ext)
    # where only corner cells are populated. For JAX efficiency, store
    # compact arrays: xp[face, stencil_k, target_j] for each target i in halo.
    # FV3 stores xp(interporder+1, ie-interporder:ied+1, jsd:jed+1, 4_stagger)
    # For istag=0, jstag=0 (A-grid): stagger index = 0*2 + 0 + 1 = 1

    # We'll compute for A-grid stagger (istag=0, jstag=0) which is the
    # primary case used by fill_corner_region in ext_scalar.

    # For each direction, the source cells are the last/first n_stencil
    # cells of the interior.  The target is each halo cell.

    xp = np.zeros((6, n_stencil, n_ext, n_ext))
    xm = np.zeros((6, n_stencil, n_ext, n_ext))
    yp = np.zeros((6, n_stencil, n_ext, n_ext))
    ym = np.zeros((6, n_stencil, n_ext, n_ext))

    for face in range(6):
        # Interior bounds in extended-grid coords
        i_lo = ng           # first interior i
        i_hi = ng + n - 1   # last interior i
        j_lo = ng
        j_hi = ng + n - 1

        # ---- X+ direction (east side → NE/SE corners) ----
        # Source cells: i = i_hi - interp_order .. i_hi (= ng+n-4 .. ng+n-1)
        src_i_xp = list(range(i_hi - interp_order, i_hi + 1))
        for j_tgt in range(n_ext):
            # Reference point for distance: the target cell
            for i_tgt in range(i_hi + 1, n_ext):
                tgt_lon = ext_lon[face, i_tgt, j_tgt]
                tgt_lat = ext_lat[face, i_tgt, j_tgt]
                # Compute signed distances from each source to target
                dists = np.zeros(n_stencil)
                for k, si in enumerate(src_i_xp):
                    dists[k] = _great_circle_dist(
                        ext_lon[face, si, j_tgt], ext_lat[face, si, j_tgt],
                        tgt_lon, tgt_lat)
                    # Sign: positive if si < i_tgt (source is to the left)
                    if si > i_tgt:
                        dists[k] = -dists[k]
                # Lagrange weights with signed distances
                weights = _lagrange_coef(0.0, -dists)  # target at dist=0
                xp[face, :, i_tgt, j_tgt] = weights

        # ---- X- direction (west side → NW/SW corners) ----
        src_i_xm = list(range(i_lo, i_lo + n_stencil))  # ng..ng+3
        for j_tgt in range(n_ext):
            for i_tgt in range(0, i_lo):
                tgt_lon = ext_lon[face, i_tgt, j_tgt]
                tgt_lat = ext_lat[face, i_tgt, j_tgt]
                dists = np.zeros(n_stencil)
                for k, si in enumerate(src_i_xm):
                    dists[k] = _great_circle_dist(
                        ext_lon[face, si, j_tgt], ext_lat[face, si, j_tgt],
                        tgt_lon, tgt_lat)
                    if si < i_tgt:
                        dists[k] = -dists[k]
                weights = _lagrange_coef(0.0, -dists)
                xm[face, :, i_tgt, j_tgt] = weights

        # ---- Y+ direction (north side → NE/NW corners) ----
        src_j_yp = list(range(j_hi - interp_order, j_hi + 1))
        for i_tgt in range(n_ext):
            for j_tgt in range(j_hi + 1, n_ext):
                tgt_lon = ext_lon[face, i_tgt, j_tgt]
                tgt_lat = ext_lat[face, i_tgt, j_tgt]
                dists = np.zeros(n_stencil)
                for k, sj in enumerate(src_j_yp):
                    dists[k] = _great_circle_dist(
                        ext_lon[face, i_tgt, sj], ext_lat[face, i_tgt, sj],
                        tgt_lon, tgt_lat)
                    if sj > j_tgt:
                        dists[k] = -dists[k]
                weights = _lagrange_coef(0.0, -dists)
                yp[face, :, i_tgt, j_tgt] = weights

        # ---- Y- direction (south side → SE/SW corners) ----
        src_j_ym = list(range(j_lo, j_lo + n_stencil))
        for i_tgt in range(n_ext):
            for j_tgt in range(0, j_lo):
                tgt_lon = ext_lon[face, i_tgt, j_tgt]
                tgt_lat = ext_lat[face, i_tgt, j_tgt]
                dists = np.zeros(n_stencil)
                for k, sj in enumerate(src_j_ym):
                    dists[k] = _great_circle_dist(
                        ext_lon[face, i_tgt, sj], ext_lat[face, i_tgt, sj],
                        tgt_lon, tgt_lat)
                    if sj < j_tgt:
                        dists[k] = -dists[k]
                weights = _lagrange_coef(0.0, -dists)
                ym[face, :, i_tgt, j_tgt] = weights

    return xp, xm, yp, ym


# ============================================================================
# Factory
# ============================================================================

def create_duogrid_data(
    n: int,
    radius: float = constants.R_earth,
    ng: int = 3,
    k2e_nord: int = 2,
) -> DuoGridData:
    """Create Duo-Grid remapping data for a cubed-sphere grid.

    All computation is done in NumPy at grid creation time. Results are
    converted to JAX arrays for runtime use.

    Parameters
    ----------
    n : int
        Grid resolution (cells per face edge).
    radius : float
        Sphere radius (unused for coefficient computation, kept for API).
    ng : int
        Halo width (default 3).
    k2e_nord : int
        Lagrange interpolation order for edge remapping (2 or 4).
    """
    if k2e_nord > MAX_K2E_NORD:
        raise ValueError(f"k2e_nord={k2e_nord} exceeds MAX_K2E_NORD={MAX_K2E_NORD}")
    if ng not in (1, 2, 3, 4):
        raise ValueError(f"ng={ng} not supported, must be 1, 2, 3, or 4")
    if ng > n // 2:
        raise ValueError(
            f"ng={ng} too large for n={n}: need ng <= n//2 for valid "
            f"Lagrange stencils (got n//2={n//2})")

    # Step P1: Build extended grid
    ext_lon, ext_lat = _build_extended_grid(n, ng)

    # Step P2: Build kinked grid
    kik_lon, kik_lat = _build_kinked_grid(n, ng, ext_lon, ext_lat)

    # Step P3: Extract 1D coordinates (gnomonic alpha, not latitude)
    ext_1d, kik_1d = _extract_1d_coords(n, ng, ext_lon, ext_lat, kik_lon, kik_lat)

    # Step P4: Compute k2e coefficients
    k2e_coef, k2e_lo = _compute_k2e_coefficients(n, ng, k2e_nord, ext_1d, kik_1d)

    # Step P5: Compute Lagrange corner coefficients
    xp, xm, yp, ym = _compute_corner_lagrange_coeff(n, ng, ext_lon, ext_lat)

    # Step P6: Compute extended-grid unit vectors (FV3 a2stag_metrics)
    vlon_ext, vlat_ext, ew_ext, es_ext = _compute_ext_vectors(
        n, ng, ext_lon, ext_lat)

    def _maybe_jnp(arr):
        return jnp.array(arr, dtype=jnp.float64) if arr is not None else None

    return DuoGridData(
        n=n,
        ng=ng,
        k2e_nord=k2e_nord,
        k2e_coef=jnp.array(k2e_coef, dtype=jnp.float64),
        k2e_lo=jnp.array(k2e_lo, dtype=jnp.int32),
        ext_lon=jnp.array(ext_lon, dtype=jnp.float64),
        ext_lat=jnp.array(ext_lat, dtype=jnp.float64),
        corner_xp=_maybe_jnp(xp),
        corner_xm=_maybe_jnp(xm),
        corner_yp=_maybe_jnp(yp),
        corner_ym=_maybe_jnp(ym),
        vlon_ext=jnp.array(vlon_ext, dtype=jnp.float64),
        vlat_ext=jnp.array(vlat_ext, dtype=jnp.float64),
        ew_ext=jnp.array(ew_ext, dtype=jnp.float64),
        es_ext=jnp.array(es_ext, dtype=jnp.float64),
    )


def _compute_ext_vectors(n: int, ng: int, ext_lon, ext_lat):
    """Compute extended-grid unit vectors for ext_vector (a2stag_metrics).

    Following FV3 fv_duogrid.F90 unit_vect_latlon_ext (lines 2846-2868)
    and a2stag_metrics (lines 2871-2992).

    Returns
    -------
    vlon_ext : (6, n_ext, n_ext, 3) — longitude unit vector at A-grid
    vlat_ext : (6, n_ext, n_ext, 3) — latitude unit vector at A-grid
    ew_ext : (6, n_ext+1, n_ext, 3) — east-west edge vector
    es_ext : (6, n_ext, n_ext+1, 3) — north-south edge vector
    """
    n_ext = n + 2 * ng

    # --- vlon_ext, vlat_ext: unit vectors in lon/lat directions ---
    # FV3 unit_vect_latlon_ext:
    # elon = [-sin(lon), cos(lon), 0]
    # elat = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
    vlon_ext = np.zeros((6, n_ext, n_ext, 3))
    vlat_ext = np.zeros((6, n_ext, n_ext, 3))

    for face in range(6):
        lon = ext_lon[face]
        lat = ext_lat[face]
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)

        vlon_ext[face, :, :, 0] = -sin_lon
        vlon_ext[face, :, :, 1] = cos_lon
        vlon_ext[face, :, :, 2] = 0.0

        vlat_ext[face, :, :, 0] = -sin_lat * cos_lon
        vlat_ext[face, :, :, 1] = -sin_lat * sin_lon
        vlat_ext[face, :, :, 2] = cos_lat

    # --- ew_ext, es_ext: edge vectors ---
    # Following FV3 a2stag_metrics (fv_duogrid.F90:2925-2951).
    # These are edge-NORMAL vectors (perpendicular to the line connecting
    # adjacent A-grid centers, lying on the sphere surface).
    #
    # FV3 algorithm (double cross product):
    # For ew at (i, j) (i-edge between cells i-1 and i):
    #   pp = midpoint of B-grid (i,j) and (i,j+1) [on sphere]
    #   p1 = A-grid (i, j) in Cartesian
    #   p3 = A-grid (i-1, j) in Cartesian
    #   p2 = cross(p3, p1) — normal to great circle connecting A-grids
    #   ew = normalize(cross(p2, pp)) — tangent at pp, perpendicular to p2

    # Convert A-grid positions to Cartesian
    cart = np.zeros((6, n_ext, n_ext, 3))
    for face in range(6):
        lon = ext_lon[face]
        lat = ext_lat[face]
        cart[face, :, :, 0] = np.cos(lat) * np.cos(lon)
        cart[face, :, :, 1] = np.cos(lat) * np.sin(lon)
        cart[face, :, :, 2] = np.sin(lat)

    def _cross(a, b):
        """Vectorized cross product of (..., 3) arrays."""
        return np.stack([
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ], axis=-1)

    def _normalize(v):
        """Normalize (..., 3) vectors."""
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norms, 1e-30)

    # Approximate B-grid Cartesian positions from A-grid midpoints.
    # FV3 has actual B-grid from the supergrid; we approximate as the
    # midpoint of 4 surrounding A-cells, normalized to the sphere.
    # B-grid at (i, j) is between A-cells (i-1,j-1), (i,j-1), (i-1,j), (i,j).
    # Shape: (6, n_ext+1, n_ext+1, 3)
    bgrid = np.zeros((6, n_ext + 1, n_ext + 1, 3))
    for face in range(6):
        # Interior B-grid points (i=1..n_ext-1, j=1..n_ext-1)
        bgrid[face, 1:-1, 1:-1, :] = _normalize(
            0.25 * (cart[face, :-1, :-1, :] + cart[face, 1:, :-1, :]
                    + cart[face, :-1, 1:, :] + cart[face, 1:, 1:, :]))
        # Boundary: extrapolate from interior
        bgrid[face, 0, :, :] = bgrid[face, 1, :, :]
        bgrid[face, -1, :, :] = bgrid[face, -2, :, :]
        bgrid[face, :, 0, :] = bgrid[face, :, 1, :]
        bgrid[face, :, -1, :] = bgrid[face, :, -2, :]

    # FV3 computes TWO variants for each edge vector:
    # ew variant 1: cross(cross(A(i-1,j), A(i,j)), midpoint(B(i,j), B(i,j+1)))
    # ew variant 2: cross(cross(B(i,j), B(i,j+1)), midpoint(B(i,j), B(i,j+1)))
    # cubed_a2d_halo uses: ud <- es(:,:,1), vd <- ew(:,:,2)
    # cubed_a2c_halo uses: uc <- ew(:,:,1), vc <- es(:,:,2)

    # ew_ext: shape (6, n_ext+1, n_ext, 3, 2) — last dim = variant
    # FV3 loops i=isd+1:ied → n_ext-1 interior i-edges. We store n_ext+1
    # total with boundary copies.
    ew_ext = np.zeros((6, n_ext + 1, n_ext, 3, 2))
    for face in range(6):
        # Interior i-edges (i=1..n_ext-1): between A(i-1,j) and A(i,j)
        # Shapes: cart[1:,:,:] → (n_ext-1, n_ext, 3)
        p1_a = cart[face, 1:, :, :]     # A(i, j) — (n_ext-1, n_ext, 3)
        p3_a = cart[face, :-1, :, :]    # A(i-1, j)
        # pp = midpoint of B(i,j) and B(i,j+1) — B has (n_ext+1) points
        # B at i ranges 0..n_ext, j ranges 0..n_ext
        # For i-edge at i (1..n_ext-1): use B(i, 0..n_ext-1) and B(i, 1..n_ext)
        pp = _normalize(0.5 * (bgrid[face, 1:-1, :-1, :] + bgrid[face, 1:-1, 1:, :]))
        # pp: (n_ext-1, n_ext, 3) — matches p1_a shape

        # Variant 1: from A-grid normals
        p2 = _cross(p3_a, p1_a)
        ew_ext[face, 1:-1, :, :, 0] = _normalize(_cross(p2, pp))

        # Variant 2: from B-grid normals
        b1 = bgrid[face, 1:-1, :-1, :]  # B(i, j)
        b2 = bgrid[face, 1:-1, 1:, :]   # B(i, j+1)
        p1_b = _cross(b1, b2)
        ew_ext[face, 1:-1, :, :, 1] = _normalize(_cross(p1_b, pp))

        for v in range(2):
            ew_ext[face, 0, :, :, v] = ew_ext[face, 1, :, :, v]
            ew_ext[face, -1, :, :, v] = ew_ext[face, -2, :, :, v]

    # es_ext: shape (6, n_ext, n_ext+1, 3, 2) — last dim = variant
    es_ext = np.zeros((6, n_ext, n_ext + 1, 3, 2))
    for face in range(6):
        # Interior j-edges (j=1..n_ext-1): between A(i,j-1) and A(i,j)
        p1_a = cart[face, :, 1:, :]     # A(i, j) — (n_ext, n_ext-1, 3)
        p3_a = cart[face, :, :-1, :]    # A(i, j-1)
        # pp = midpoint of B(i,j) and B(i+1,j)
        pp = _normalize(0.5 * (bgrid[face, :-1, 1:-1, :] + bgrid[face, 1:, 1:-1, :]))
        # pp: (n_ext, n_ext-1, 3)

        # Variant 1: from B-grid normals
        b1 = bgrid[face, :-1, 1:-1, :]  # B(i, j)
        b2 = bgrid[face, 1:, 1:-1, :]   # B(i+1, j)
        p3_b = _cross(b1, b2)
        es_ext[face, :, 1:-1, :, 0] = _normalize(_cross(p3_b, pp))

        # Variant 2: from A-grid normals
        p2 = _cross(p3_a, p1_a)
        es_ext[face, :, 1:-1, :, 1] = _normalize(_cross(p2, pp))

        for v in range(2):
            es_ext[face, :, 0, :, v] = es_ext[face, :, 1, :, v]
            es_ext[face, :, -1, :, v] = es_ext[face, :, -2, :, v]

    return vlon_ext, vlat_ext, ew_ext, es_ext


# ============================================================================
# Runtime kernels
# ============================================================================

def cube_rmp_vectorized(
    padded: jax.Array,
    duogrid: DuoGridData,
    halo: int,
) -> jax.Array:
    """Remap kinked halo values to extended grid positions.

    Applies 1D Lagrange interpolation along each face edge strip at
    each halo depth. Coefficients are face/edge-independent due to
    cubed-sphere symmetry.

    Vectorized across faces using batched indexing.
    """
    n = duogrid.n
    h = halo
    k2e_lo = duogrid.k2e_lo     # (6, 4, ng, n)
    # Cast coefficients to field dtype to avoid mixed-precision scatter warnings
    k2e_coef = duogrid.k2e_coef.astype(padded.dtype)  # (6, 4, ng, n, MAX_K2E_NORD)
    dst_idx = h + jnp.arange(n)
    max_idx = n + 2 * h - 1  # maximum valid padded index

    for d in range(min(halo, duogrid.ng)):
        # South edge (edge=2): interpolate along axis 1 at fixed axis-2 row
        j_rc = h - 1 - d
        lo_s = k2e_lo[:, SOUTH, d, :]       # (6, n)
        coef_s = k2e_coef[:, SOUTH, d, :, :]  # (6, n, MAX)
        src_s = jnp.clip(h + (
            lo_s[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :]),
            0, max_idx)  # (6, n, MAX)
        for f in range(6):
            vals = padded[f, src_s[f], j_rc]  # (n, MAX)
            remapped = jnp.sum(vals * coef_s[f], axis=-1)
            padded = padded.at[f, dst_idx, j_rc].set(remapped)

        # North edge (edge=3)
        j_rc = h + n + d
        lo_n = k2e_lo[:, NORTH, d, :]
        coef_n = k2e_coef[:, NORTH, d, :, :]
        src_n = jnp.clip(h + (
            lo_n[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :]),
            0, max_idx)
        for f in range(6):
            vals = padded[f, src_n[f], j_rc]
            remapped = jnp.sum(vals * coef_n[f], axis=-1)
            padded = padded.at[f, dst_idx, j_rc].set(remapped)

        # West edge (edge=0): interpolate along axis 2 at fixed axis-1 col
        i_rc = h - 1 - d
        lo_w = k2e_lo[:, WEST, d, :]
        coef_w = k2e_coef[:, WEST, d, :, :]
        src_w = jnp.clip(h + (
            lo_w[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :]),
            0, max_idx)
        for f in range(6):
            vals = padded[f, i_rc, src_w[f]]
            remapped = jnp.sum(vals * coef_w[f], axis=-1)
            padded = padded.at[f, i_rc, dst_idx].set(remapped)

        # East edge (edge=1)
        i_rc = h + n + d
        lo_e = k2e_lo[:, EAST, d, :]
        coef_e = k2e_coef[:, EAST, d, :, :]
        src_e = jnp.clip(h + (
            lo_e[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :]),
            0, max_idx)
        for f in range(6):
            vals = padded[f, i_rc, src_e[f]]
            remapped = jnp.sum(vals * coef_e[f], axis=-1)
            padded = padded.at[f, i_rc, dst_idx].set(remapped)

    return padded


def _lagrange_interp_x_plus(padded, xp_coefs, i_e, j_e, j_p, n, h):
    """Lagrange interpolation in the X+ direction (from east interior).

    Source stencil: last 4 interior cells along i in padded coordinates.
    Coefficient lookup uses extended-grid indices (i_e, j_e).
    Padded array access uses padded index j_p.
    """
    i_hi_pad = h + n - 1  # last interior index in PADDED coords
    src_i = jnp.arange(i_hi_pad - 3, i_hi_pad + 1)  # 4 source cells
    vals = padded[:, src_i, j_p]  # (6, 4)
    weights = xp_coefs[:, :, i_e, j_e]  # (6, 4) — lookup in extended coords
    return jnp.sum(vals * weights, axis=1)  # (6,)


def _lagrange_interp_x_minus(padded, xm_coefs, i_e, j_e, j_p, n, h):
    """Lagrange interpolation in the X- direction (from west interior)."""
    i_lo_pad = h  # first interior index in PADDED coords
    src_i = jnp.arange(i_lo_pad, i_lo_pad + 4)
    vals = padded[:, src_i, j_p]  # (6, 4)
    weights = xm_coefs[:, :, i_e, j_e]
    return jnp.sum(vals * weights, axis=1)


def _lagrange_interp_y_plus(padded, yp_coefs, i_e, j_e, i_p, n, h):
    """Lagrange interpolation in the Y+ direction (from north interior)."""
    j_hi_pad = h + n - 1
    src_j = jnp.arange(j_hi_pad - 3, j_hi_pad + 1)
    vals = padded[:, i_p, src_j]  # (6, 4)
    weights = yp_coefs[:, :, i_e, j_e]
    return jnp.sum(vals * weights, axis=1)


def _lagrange_interp_y_minus(padded, ym_coefs, i_e, j_e, i_p, n, h):
    """Lagrange interpolation in the Y- direction (from south interior)."""
    j_lo_pad = h
    src_j = jnp.arange(j_lo_pad, j_lo_pad + 4)
    vals = padded[:, i_p, src_j]  # (6, 4)
    weights = ym_coefs[:, :, i_e, j_e]
    return jnp.sum(vals * weights, axis=1)


def fill_corner_region(
    padded: jax.Array,
    duogrid: DuoGridData,
    halo: int,
    monotone_clip: bool = False,
    monotone_clip_slack: float = 0.0,
) -> jax.Array:
    """Fill corner blocks using FV3 Lagrange polynomial interpolation.

    Following FV3 fv_duogrid.F90 fill_corner_region_2d:
    - Non-diagonal corner cells are filled first via single-direction
      Lagrange interpolation (X+ or Y+ or X- or Y-)
    - Diagonal corner cells are filled as the average of X-direction and
      Y-direction Lagrange interpolations

    Falls back to simple averaging when Lagrange coefficients are not
    available (corner_xp is None).

    Parameters
    ----------
    padded : jax.Array, shape (6, n+2h, n+2h)
    duogrid : DuoGridData
    halo : int
    monotone_clip : bool, default False
        Iter-802: when True, clip each Lagrange-extrapolated cube-corner
        cell to `[min, max]` of the adjacent interior + edge-halo cells.
        Iter-801 measured a 144 m overshoot of interior max at polar
        cube corners on smooth W2 h, which iter-800 showed drives a
        1172× dh/dt blowup via PPM on the padded h-field.  The clip is
        a pragmatic monotonicity constraint that preserves Lagrange
        values when they fall within the physical range.  Default False
        preserves FV3-faithful behaviour (Fortran `lagrange_poly_interp_2d`
        does not clip either, but Fortran's FB time-splitting structure
        mitigates the overshoot downstream in a way that our A-L+RK3
        path does not).  Enable for DUOGRID mass-transport stability.
    """
    if monotone_clip:
        warnings.warn(
            "fill_corner_region(monotone_clip=True) clips cube-corner halo "
            "cells to the neighbour min/max — a NON-CONSERVATIVE limiter: it "
            "perturbs the Lagrange-interpolated values, so the corner fill no "
            "longer preserves the global tracer/mass integral. Enable only for "
            "duogrid mass-transport stability; leave it False (the default) for "
            "conservation-critical runs.",
            stacklevel=2,
        )
    n = duogrid.n
    ng = duogrid.ng
    h = halo

    if h == 0:
        return padded

    # Fall back to averaging if no corner coefficients or if halo > ng
    # (offset = ng - h would go negative, making Lagrange lookup invalid)
    if getattr(duogrid, 'corner_xp', None) is None or h > ng:
        return _fill_corner_region_averaging(padded, duogrid, halo)

    xp = duogrid.corner_xp.astype(padded.dtype)
    xm = duogrid.corner_xm.astype(padded.dtype)
    yp = duogrid.corner_yp.astype(padded.dtype)
    ym = duogrid.corner_ym.astype(padded.dtype)

    # Map from padded coordinates to extended-grid coordinates.
    # padded has shape (6, n+2*halo, n+2*halo).
    # If halo == ng, padded indices map 1:1 to extended grid indices.
    # If halo < ng, we need to offset. The halo cells in padded correspond
    # to the inner ng-halo..ng-1 halo cells of the extended grid.
    offset = ng - h  # extended_idx = padded_idx + offset (always >= 0)

    # Interior bounds in padded coords
    ie = h + n - 1  # last interior i in padded
    je = h + n - 1  # last interior j in padded

    # Following FV3 fill_corner_region_2d (fv_duogrid.F90:1719-1903).
    # FV3 fill order: (1) non-diagonal cells first, (2) then diagonal
    # cells as average of X and Y interpolations on a COPY of padded.

    def _maybe_clip(val, padded, i_p, j_p):
        """Iter-802: optionally clip val to min/max of the 4 neighbours
        (within the padded array) at (i_p±1, j_p), (i_p, j_p±1).  The
        neighbours are either already-filled halo cells or interior
        cells, all within the physical range."""
        if not monotone_clip:
            return val
        # Wrap at array boundaries using clip-to-last-valid-index so that
        # at the extreme corner cell [0,0] or [-1,-1] we still have 2 valid
        # neighbours.
        n_i = padded.shape[1]
        n_j = padded.shape[2]
        im = max(i_p - 1, 0)
        ip = min(i_p + 1, n_i - 1)
        jm = max(j_p - 1, 0)
        jp = min(j_p + 1, n_j - 1)
        neighbours = jnp.stack([
            padded[:, im, j_p], padded[:, ip, j_p],
            padded[:, i_p, jm], padded[:, i_p, jp],
        ], axis=-1)  # (6, 4)
        lo = jnp.min(neighbours, axis=-1)
        hi = jnp.max(neighbours, axis=-1)
        # iter-501: optional slack — expand clip range by
        # ``slack * (hi - lo)`` on each side.  slack=0 →
        # strict iter-802 clip; slack > 0 → softer monotonic
        # constraint (allows mild overshoot, less aggressive
        # than strict clip that iter-499 found over-corrects).
        if monotone_clip_slack > 0.0:
            band = monotone_clip_slack * (hi - lo)
            lo = lo - band
            hi = hi + band
        return jnp.clip(val, lo, hi)

    def _fill_one_corner(padded, x_interp, y_interp, x_coefs, y_coefs,
                         get_ip, get_jp):
        """Fill one h×h corner block with FV3 ordering.

        Iter-803: adds Fortran-faithful veltemp/veltempp snapshot
        semantics for pass-2 diagonal cells.  Fortran
        `fill_corner_region_2d` (fv_duogrid.F90:1759-1779) captures
        `veltemp = vel` and `veltempp = vel` ONCE after pass-1 and
        BEFORE any pass-2 diagonal writes, so every pass-2 diagonal
        reads from a SNAPSHOT unaffected by earlier pass-2 writes.
        Python's previous pass-2 loop updated `padded` in place, so
        a later pass-2 diagonal could read an earlier pass-2 diagonal
        value — a subtle compounding that is NOT in Fortran.
        """
        # Pass 1: non-diagonal cells (d1 != d2)
        for d1 in range(1, h + 1):
            for d2 in range(1, h + 1):
                if d1 == d2:
                    continue
                i_p = get_ip(d1)
                j_p = get_jp(d2)
                i_e = i_p + offset
                j_e = j_p + offset
                if d2 > d1:
                    val = x_interp(padded, x_coefs, i_e, j_e, j_p, n, h)
                else:
                    val = y_interp(padded, y_coefs, i_e, j_e, i_p, n, h)
                val = _maybe_clip(val, padded, i_p, j_p)
                padded = padded.at[:, i_p, j_p].set(val)

        # Iter-803: Fortran-faithful snapshot.  Fortran pattern:
        #   veltemp = vel   ! snapshot after pass-1
        #   veltempp = vel
        #   do each diagonal cell (i_p, j_p):
        #       lagrange_poly_interp(veltemp, i_p, j_p, 'X+')
        #       lagrange_poly_interp(veltempp, i_p, j_p, 'Y+')
        #       vel(i_p, j_p) = 0.5 * (veltemp(i_p, j_p) + veltempp(i_p, j_p))
        # Key: each lagrange_poly_interp reads from the SNAPSHOT.  We
        # capture the padded state AFTER pass-1 and BEFORE writing any
        # pass-2 diagonal, then read from that snapshot.
        padded_snapshot = padded

        for d in range(1, h + 1):
            i_p = get_ip(d)
            j_p = get_jp(d)
            i_e = i_p + offset
            j_e = j_p + offset
            val_x = x_interp(padded_snapshot, x_coefs, i_e, j_e, j_p, n, h)
            val_y = y_interp(padded_snapshot, y_coefs, i_e, j_e, i_p, n, h)
            val = 0.5 * (val_x + val_y)
            val = _maybe_clip(val, padded, i_p, j_p)
            padded = padded.at[:, i_p, j_p].set(val)

        return padded

    # NE corner
    padded = _fill_one_corner(
        padded, _lagrange_interp_x_plus, _lagrange_interp_y_plus, xp, yp,
        lambda d: ie + d, lambda d: je + d)

    # NW corner
    padded = _fill_one_corner(
        padded, _lagrange_interp_x_minus, _lagrange_interp_y_plus, xm, yp,
        lambda d: h - d, lambda d: je + d)

    # SE corner
    padded = _fill_one_corner(
        padded, _lagrange_interp_x_plus, _lagrange_interp_y_minus, xp, ym,
        lambda d: ie + d, lambda d: h - d)

    # SW corner
    padded = _fill_one_corner(
        padded, _lagrange_interp_x_minus, _lagrange_interp_y_minus, xm, ym,
        lambda d: h - d, lambda d: h - d)

    return padded


def apply_duogrid_4d(padded, duogrid, halo):
    """Apply the duogrid kinked-to-extended remap + corner fill to a 4D
    padded field ``(6, n+2h, n+2h, nlev)``, level-by-level via ``jax.vmap``.

    Mirrors the post-processing loop inside :func:`legoesm.grids.halo.pad_halo_4d`;
    shared by the MPI (``parallel.halo_exchange``) and SPMD
    (``parallel.cubesphere_exchange``) 4D halo paths so the two cannot drift.
    """
    def _remap_level(level_slice):
        level_slice = cube_rmp_vectorized(level_slice, duogrid, halo)
        level_slice = fill_corner_region(level_slice, duogrid, halo)
        return level_slice

    padded_t = jnp.transpose(padded, (3, 0, 1, 2))  # (nlev, 6, ...)
    padded_t = jax.vmap(_remap_level)(padded_t)
    return jnp.transpose(padded_t, (1, 2, 3, 0))


def _fill_corner_region_averaging(
    padded: jax.Array,
    duogrid: DuoGridData,
    halo: int,
) -> jax.Array:
    """Fill corner blocks by averaging adjacent edge halo values (fallback)."""
    n = duogrid.n
    h = halo
    n_p = n + 2 * h

    for s in range(2 * (h - 1), -1, -1):
        for ci in range(h):
            cj = s - ci
            if cj < 0 or cj >= h:
                continue
            ni_sw = min(ci + 1, h)
            nj_sw = min(cj + 1, h)
            padded = padded.at[:, ci, cj].set(
                0.5 * (padded[:, ni_sw, cj] + padded[:, ci, nj_sw]))
            i_se = n_p - 1 - ci
            ni_se = n_p - 1 - min(ci + 1, h)
            nj_se = min(cj + 1, h)
            padded = padded.at[:, i_se, cj].set(
                0.5 * (padded[:, ni_se, cj] + padded[:, i_se, nj_se]))
            j_nw = n_p - 1 - cj
            ni_nw = min(ci + 1, h)
            nj_nw = n_p - 1 - min(cj + 1, h)
            padded = padded.at[:, ci, j_nw].set(
                0.5 * (padded[:, ni_nw, j_nw] + padded[:, ci, nj_nw]))
            i_ne = n_p - 1 - ci
            j_ne = n_p - 1 - cj
            ni_ne = n_p - 1 - min(ci + 1, h)
            nj_ne = n_p - 1 - min(cj + 1, h)
            padded = padded.at[:, i_ne, j_ne].set(
                0.5 * (padded[:, ni_ne, j_ne] + padded[:, i_ne, nj_ne]))

    return padded


# ============================================================================
# D-grid staggered halo exchange
# ============================================================================

def pad_halo_dgrid(
    u_d: jax.Array,
    v_d: jax.Array,
    cos_angle_edge_x: jax.Array,
    sin_angle_edge_x: jax.Array,
    cos_angle_edge_y: jax.Array,
    sin_angle_edge_y: jax.Array,
    duogrid: 'DuoGridData | None' = None,
    halo: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Pad D-grid staggered fields with halo from neighbor faces.

    FV3 exchanges D-grid winds via mpp_update_domains with DGRID_NE
    gridtype before d2a2c_vect. This provides the boundary D-grid
    values needed for 4th-order D→A averaging at face edges.

    Uses exact D-grid edge angles from CubedSphereCDGrid (not averaged
    A-grid angles) for precise geographic rotation at stagger positions.

    Parameters
    ----------
    u_d : (6, n, n+1) D-grid x-velocity at j-edges
    v_d : (6, n+1, n) D-grid y-velocity at i-edges
    cos_angle_edge_x, sin_angle_edge_x : (6, n, n+1) exact edge angles
    cos_angle_edge_y, sin_angle_edge_y : (6, n+1, n) exact edge angles
    halo : int, default 1
        Halo width.  FV3 uses ``ng=3`` but ``halo=2`` gives enough
        additional stencil support for 4th-order A→C recomputation
        directly on D-grid without a separate scalar exchange.

    Returns
    -------
    u_d_pad : (6, n, n + 2*halo + 1) D-grid u padded in j direction
    v_d_pad : (6, n + 2*halo + 1, n) D-grid v padded in i direction
    """
    from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

    if halo not in (1, 2):
        raise NotImplementedError(f"Only halo=1 and halo=2 supported, got {halo}")

    n = u_d.shape[1]
    h = halo

    # Convert D-grid to geographic using EXACT edge angles.
    ca_j = cos_angle_edge_x   # (6, n, n+1)
    sa_j = sin_angle_edge_x
    u_east_ud = ca_j * u_d
    v_north_ud = sa_j * u_d

    ca_i = cos_angle_edge_y   # (6, n+1, n)
    sa_i = sin_angle_edge_y
    u_east_vd = -sa_i * v_d
    v_north_vd = ca_i * v_d

    # Pad u_d along j (axis 2): halo on each side → (6, n, n + 2h + 1)
    u_d_pad = jnp.zeros((6, n, n + 2 * h + 1), dtype=u_d.dtype)
    u_d_pad = u_d_pad.at[:, :, h:-h].set(u_d)

    # Pad v_d along i (axis 1): halo on each side → (6, n + 2h + 1, n)
    v_d_pad = jnp.zeros((6, n + 2 * h + 1, n), dtype=v_d.dtype)
    v_d_pad = v_d_pad.at[:, h:-h, :].set(v_d)

    def _select_src(arr_ud_e, arr_ud_n, arr_vd_e, arr_vd_n,
                    nbr_f, nbr_e, rev, depth):
        """Pick neighbor's edge strip at the given halo depth.

        ``depth`` = 1 → edge adjacent to neighbor's boundary (index -2 / 1)
        ``depth`` = 2 → edge 2 in from neighbor's boundary (index -3 / 2)
        """
        if nbr_e == NORTH:
            src   = arr_ud_e[nbr_f, :, -(depth + 1)]
            src_v = arr_ud_n[nbr_f, :, -(depth + 1)]
        elif nbr_e == SOUTH:
            src   = arr_ud_e[nbr_f, :, depth]
            src_v = arr_ud_n[nbr_f, :, depth]
        elif nbr_e == EAST:
            src   = arr_vd_e[nbr_f, -(depth + 1), :]
            src_v = arr_vd_n[nbr_f, -(depth + 1), :]
        else:  # WEST
            src   = arr_vd_e[nbr_f, depth, :]
            src_v = arr_vd_n[nbr_f, depth, :]
        if rev:
            src = src[::-1]
            src_v = src_v[::-1]
        return src, src_v

    # Fill halo from neighbor faces using CONNECTIVITY.
    # For each face and each of the 4 edges, we fill ``halo`` layers
    # deep using the neighbor's edges at increasing depth from their
    # boundary (depth=1 closest, depth=h furthest).
    for face in range(6):
        # --- u_d j-halo ---
        nbr_f_s, nbr_e_s, rev_s = CONNECTIVITY[face][SOUTH]
        nbr_f_n, nbr_e_n, rev_n = CONNECTIVITY[face][NORTH]
        # Boundary angles for rotation-back to local frame
        ca_bdy_s = ca_j[face, :, 0]
        sa_bdy_s = sa_j[face, :, 0]
        ca_bdy_n = ca_j[face, :, -1]
        sa_bdy_n = sa_j[face, :, -1]
        for d in range(1, h + 1):
            # South halo layer at padded j = h - d (d=1 → h-1, d=h → 0)
            src, src_v = _select_src(u_east_ud, v_north_ud,
                                      u_east_vd, v_north_vd,
                                      nbr_f_s, nbr_e_s, rev_s, d)
            u_local = ca_bdy_s * src + sa_bdy_s * src_v
            u_d_pad = u_d_pad.at[face, :, h - d].set(u_local)

            # North halo layer at padded j = h + n + (d-1) + 1 = h + n + d
            src, src_v = _select_src(u_east_ud, v_north_ud,
                                      u_east_vd, v_north_vd,
                                      nbr_f_n, nbr_e_n, rev_n, d)
            u_local = ca_bdy_n * src + sa_bdy_n * src_v
            u_d_pad = u_d_pad.at[face, :, h + n + d].set(u_local)

        # --- v_d i-halo ---
        nbr_f_w, nbr_e_w, rev_w = CONNECTIVITY[face][WEST]
        nbr_f_e, nbr_e_e, rev_e = CONNECTIVITY[face][EAST]
        ca_bdy_w = ca_i[face, 0, :]
        sa_bdy_w = sa_i[face, 0, :]
        ca_bdy_e = ca_i[face, -1, :]
        sa_bdy_e = sa_i[face, -1, :]
        for d in range(1, h + 1):
            # West halo layer at padded i = h - d
            src, src_v = _select_src(u_east_ud, v_north_ud,
                                      u_east_vd, v_north_vd,
                                      nbr_f_w, nbr_e_w, rev_w, d)
            v_local = -sa_bdy_w * src + ca_bdy_w * src_v
            v_d_pad = v_d_pad.at[face, h - d, :].set(v_local)

            # East halo layer at padded i = h + n + d
            src, src_v = _select_src(u_east_ud, v_north_ud,
                                      u_east_vd, v_north_vd,
                                      nbr_f_e, nbr_e_e, rev_e, d)
            v_local = -sa_bdy_e * src + ca_bdy_e * src_v
            v_d_pad = v_d_pad.at[face, h + n + d, :].set(v_local)

    return u_d_pad, v_d_pad


# ============================================================================
# ext_vector: FV3-faithful vector halo exchange
# ============================================================================

def ext_vector_dgrid(
    utmp: jax.Array,
    vtmp: jax.Array,
    duogrid: 'DuoGridData',
    cos_angle: jax.Array,
    sin_angle: jax.Array,
    cosa_s: jax.Array,
    rsin2: jax.Array,
    halo: int = 2,
    basis: str = "covariant",
) -> tuple[jax.Array, jax.Array]:
    """FV3-faithful D-grid vector halo exchange via lat/lon intermediary.

    Following FV3 ext_vector for DGRID case (fv_duogrid.F90:741-826).
    The flow is:
    1. Convert A-grid (utmp, vtmp) to lat/lon
    2. Halo-exchange the lat/lon winds as scalars (with cube_rmp + corner fill)
    3. Convert A-grid lat/lon back to D-grid at the padded stagger points
    4. Return padded D-grid winds

    Two wind-component conventions are supported via ``basis``:

    - ``"covariant"`` (default, production ``_d2a2c_vect_duogrid`` path):
      inputs are covariant projections onto the (non-orthogonal) grid-line
      tangents; step 1 uses the exact cosa_s inversion (≡ FV3 c2l_ord2
      a11..a22) and step 3 uses ``cubed_a2d_halo`` (FV3 es/ew edge-tangent
      projections).
    - ``"orthogonal"`` (FB SW chain): this model's D winds are
      ``u_d = V·x̂`` (x-line tangent at x-edges, ``angle_edge_x``) and
      ``v_d = V·x̂⊥`` (rot-90 of the x-tangent at y-edges,
      ``angle_edge_y`` is the i-tangent angle — see
      ``cubed_sphere_cdgrid.py`` "i-tangent at y-edge midpoint").  The
      D→A averages (utmp, vtmp) then form a locally ORTHOGONAL pair, so
      step 1 is a pure rotation by the A-grid angle, and step 3 must
      project vd onto rot-90(x-tangent) — NOT FV3's y-line tangent
      ``ew_ext[...,1]``.  Using the covariant machinery on these inputs
      puts an O(cosa_s·|V|) convention error in the halo (up to ~15 m/s
      at C36 panel corners, sign-flipping across the seam) that is a
      ~100% error in the small v_d projection — the FB SW panel-edge
      instability root cause (2026-07-10).

    Parameters
    ----------
    utmp, vtmp : (6, n, n) — A-grid winds (from D→A averaging)
    duogrid : DuoGridData
    cos_angle, sin_angle : (6, n, n) — grid rotation angle at A-grid
    cosa_s : (6, n, n) — cos(angle) between grid axes (non-orthogonality;
        unused for ``basis="orthogonal"``)
    rsin2 : (6, n, n) — 1/sin²(angle) (kept for signature stability; unused)
    halo : int — halo width (default 2)
    basis : str — ``"covariant"`` or ``"orthogonal"`` (see above)

    Returns
    -------
    ud_pad : (6, n+2h, n+2h-1) — D-grid u in padded domain
    vd_pad : (6, n+2h-1, n+2h) — D-grid v in padded domain
    """
    from legoesm.grids.halo import pad_halo

    if basis not in ("covariant", "orthogonal"):
        raise ValueError(
            f"ext_vector_dgrid: unknown basis {basis!r}; "
            f"expected 'covariant' or 'orthogonal'.")

    h = halo

    if basis == "covariant":
        # Step 1: Convert covariant → geographic (lat/lon).
        # iter147 FIX (running-FV3 halo audit): the previous code raised the
        # index to the CONTRAVARIANT coefficients (ua,va) and then applied a
        # SINGLE (orthogonal) grid-angle rotation — valid only if the grid
        # axes were perpendicular.  On the cubed sphere the tangents are
        # NON-orthogonal (e1·e2 = cosa_s ≠ 0) at face edges/corners, so the
        # va·e2 projection dropped the O(cosa_s) non-orthogonality term,
        # giving a GROSS, resolution-NON-convergent halo error (~tens of m/s)
        # at the cube edges.  Use the exact non-orthogonal
        # covariant→geographic conversion (identical to the production
        # `pad_halo_vector`, halo.py): utmp/vtmp are the covariant
        # projections V·x̂, V·ŷ; cosa_s = cos(θ_between_axes); st = sin θ.
        # Reduces to the orthogonal form when cosa_s→0 (interior).
        _EPS_NO = float(jnp.finfo(jnp.float32).eps)
        st = jnp.maximum(
            jnp.sqrt(jnp.maximum(1.0 - cosa_s ** 2, 0.0)), _EPS_NO)
        u_east = cos_angle * utmp + sin_angle * (utmp * cosa_s - vtmp) / st
        v_north = sin_angle * utmp + cos_angle * (vtmp - utmp * cosa_s) / st
    else:
        # Step 1 (orthogonal basis): (utmp, vtmp) ≈ (V·x̂, V·x̂⊥) at cell
        # centres — geographic conversion is a pure rotation by the A-grid
        # angle.  Applying the covariant cosa_s inversion here would
        # re-introduce the O(cosa_s·|V|) convention error.
        u_east = cos_angle * utmp - sin_angle * vtmp
        v_north = sin_angle * utmp + cos_angle * vtmp

    # Step 2: Halo-exchange lat/lon winds as SCALARS with Duo-Grid remap.
    # This applies cube_rmp (kinked→extended) + fill_corner_region.
    # Geographic components are frame-invariant, so the neighbour-copied
    # halo ring is exact up to the step-1 conversion residual.
    u_east_pad = pad_halo(u_east, halo=h, duogrid=duogrid)   # (6, n+2h, n+2h)
    v_north_pad = pad_halo(v_north, halo=h, duogrid=duogrid)  # (6, n+2h, n+2h)

    # Step 3: Convert A-grid lat/lon back to D-grid via 3D Cartesian.
    if basis == "covariant":
        # FV3's cubed_a2d_halo (fv_duogrid.F90:2676-2763).
        ud_pad, vd_pad = cubed_a2d_halo(u_east_pad, v_north_pad, duogrid, h)
    else:
        ud_pad, vd_pad = cubed_a2d_halo_orthogonal(
            u_east_pad, v_north_pad, duogrid, h)

    return ud_pad, vd_pad


def cubed_a2d_halo(
    ull: jax.Array,
    vll: jax.Array,
    duogrid: DuoGridData,
    halo: int,
) -> tuple[jax.Array, jax.Array]:
    """Convert A-grid lat/lon winds to D-grid using 3D Cartesian projection.

    Following FV3 fv_duogrid.F90 cubed_a2d_halo (lines 2676-2763).
    FV3 loops j=jsd+1:jed for ud and i=isd+1:ied for vd, so the output
    covers interior stagger edges only (not the outermost boundary edge).

    Parameters
    ----------
    ull, vll : (6, n_p, n_p) — lat/lon wind components on padded A-grid
        where n_p = n + 2*halo
    duogrid : DuoGridData with vlon_ext, vlat_ext, ew_ext, es_ext
    halo : int — halo width of the padded arrays

    Returns
    -------
    ud : (6, n_p, n_p-1) — D-grid u at j-edges between adjacent A-cells
        (n_p-1 edges at j+1/2 for j=0..n_p-2)
    vd : (6, n_p-1, n_p) — D-grid v at i-edges between adjacent A-cells
        (n_p-1 edges at i+1/2 for i=0..n_p-2)
    """
    n = duogrid.n
    h = halo
    n_p = n + 2 * h
    offset = duogrid.ng - h  # extended-grid index = padded index + offset

    # Get extended-grid vectors, sliced to the padded domain
    vlon = duogrid.vlon_ext  # (6, n_ext, n_ext, 3)
    vlat = duogrid.vlat_ext
    ew = duogrid.ew_ext      # (6, n_ext+1, n_ext, 3, 2)
    es = duogrid.es_ext      # (6, n_ext, n_ext+1, 3, 2)

    # Cast to field dtype
    dtype = ull.dtype
    vlon = vlon.astype(dtype)
    vlat = vlat.astype(dtype)
    ew = ew.astype(dtype)
    es = es.astype(dtype)

    # Step 1: Convert lat/lon to 3D Cartesian on A-grid
    # v3 = u_ll * vlon + v_ll * vlat  — shape (6, n_p, n_p, 3)
    # Index into the extended-grid vectors at the padded positions
    i_slice = slice(offset, offset + n_p)
    j_slice = slice(offset, offset + n_p)
    vlon_p = vlon[:, i_slice, j_slice, :]  # (6, n_p, n_p, 3)
    vlat_p = vlat[:, i_slice, j_slice, :]

    v3 = ull[..., None] * vlon_p + vll[..., None] * vlat_p  # (6, n_p, n_p, 3)

    # Step 2: Interpolate to D-grid edges (simple 2-point average)
    # ud at (i, j+1/2): average of v3(i, j-1) and v3(i, j)
    ue = 0.5 * (v3[:, :, :-1, :] + v3[:, :, 1:, :])  # (6, n_p, n_p-1, 3)
    # vd at (i+1/2, j): average of v3(i-1, j) and v3(i, j)
    ve = 0.5 * (v3[:, :-1, :, :] + v3[:, 1:, :, :])  # (6, n_p-1, n_p, 3)

    # Step 3: Project onto edge vectors
    # FV3 cubed_a2d_halo (fv_duogrid.F90:2746-2757):
    #   ud(i,j,k) = ue . es(i,j,1)  — es variant 0 (B-grid normals)
    #   vd(i,j,k) = ve . ew(i,j,2)  — ew variant 1 (B-grid normals)
    # NOTE: FV3's es(:,:,1) is stored in our index 0, ew(:,:,2) in index 1
    es_slice = es[:, offset:offset + n_p, offset + 1:offset + n_p, :, 0]  # (6, n_p, n_p-1, 3) — B-grid variant
    ud = jnp.sum(ue * es_slice, axis=-1)  # (6, n_p, n_p-1)

    # vd = ve . ew_ext (B-grid normal variant)
    ew_slice = ew[:, offset + 1:offset + n_p, offset:offset + n_p, :, 1]  # (6, n_p-1, n_p, 3)
    vd = jnp.sum(ve * ew_slice, axis=-1)  # (6, n_p-1, n_p)

    return ud, vd


def cubed_a2d_halo_orthogonal(
    ull: jax.Array,
    vll: jax.Array,
    duogrid: DuoGridData,
    halo: int,
) -> tuple[jax.Array, jax.Array]:
    """A-grid lat/lon → D-grid in THIS MODEL's orthogonal wind convention.

    Same 2-point Cartesian edge averaging as :func:`cubed_a2d_halo`, but
    the projection bases match ``CubedSphereCDGrid``'s D-wind convention
    (see ``ext_vector_dgrid`` basis="orthogonal"):

    - ``ud = ue · es_ext[...,0]`` — the x-edge tangent (corner-to-corner
      chord along +i), identical to the model's ``angle_edge_x``
      i-tangent (and to :func:`cubed_a2d_halo`; conventions coincide
      for u).
    - ``vd = ve · rot90(x-tangent)`` — the model's ``v_d = V·x̂⊥``,
      where the x-tangent at v-points is ``ew_ext[...,0]`` (the
      A-neighbour great-circle tangent) and rot-90 in the local tangent
      plane is ``cross(radial, x̂)`` (east→north positive).  FV3's
      ``ew_ext[...,1]`` (y-line tangent) is the covariant convention
      and differs by O(cosa_s·|V|) near panel corners.
    """
    n = duogrid.n
    h = halo
    n_p = n + 2 * h
    offset = duogrid.ng - h

    vlon = duogrid.vlon_ext
    vlat = duogrid.vlat_ext
    ew = duogrid.ew_ext
    es = duogrid.es_ext

    dtype = ull.dtype
    vlon = vlon.astype(dtype)
    vlat = vlat.astype(dtype)
    ew = ew.astype(dtype)
    es = es.astype(dtype)

    i_slice = slice(offset, offset + n_p)
    vlon_p = vlon[:, i_slice, i_slice, :]
    vlat_p = vlat[:, i_slice, i_slice, :]
    v3 = ull[..., None] * vlon_p + vll[..., None] * vlat_p  # (6, n_p, n_p, 3)

    ue = 0.5 * (v3[:, :, :-1, :] + v3[:, :, 1:, :])  # (6, n_p, n_p-1, 3)
    ve = 0.5 * (v3[:, :-1, :, :] + v3[:, 1:, :, :])  # (6, n_p-1, n_p, 3)

    # ud: identical to cubed_a2d_halo (x-edge tangent).
    es_slice = es[:, offset:offset + n_p, offset + 1:offset + n_p, :, 0]
    ud = jnp.sum(ue * es_slice, axis=-1)  # (6, n_p, n_p-1)

    # vd: project onto rot-90 of the x-tangent at the v-point.
    ew0 = ew[:, offset + 1:offset + n_p, offset:offset + n_p, :, 0]
    lon_p = duogrid.ext_lon.astype(dtype)[:, i_slice, i_slice]
    lat_p = duogrid.ext_lat.astype(dtype)[:, i_slice, i_slice]
    clat = jnp.cos(lat_p)
    P = jnp.stack([clat * jnp.cos(lon_p), clat * jnp.sin(lon_p),
                   jnp.sin(lat_p)], axis=-1)  # (6, n_p, n_p, 3)
    P_v = 0.5 * (P[:, :-1, :, :] + P[:, 1:, :, :])  # radial at v-points
    P_v = P_v / jnp.maximum(
        jnp.linalg.norm(P_v, axis=-1, keepdims=True), 1e-30)
    yhat = jnp.cross(P_v, ew0)  # +90° rotation of x̂ (east→north)
    yhat = yhat / jnp.maximum(
        jnp.linalg.norm(yhat, axis=-1, keepdims=True), 1e-30)
    vd = jnp.sum(ve * yhat, axis=-1)  # (6, n_p-1, n_p)

    return ud, vd
