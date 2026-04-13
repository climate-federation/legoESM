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

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

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


# ============================================================================
# Precompute: supergrid, kinked grid, coordinates, coefficients
# ============================================================================

def _cart2lonlat(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Cartesian (x, y, z) on unit sphere → (lon, lat) in radians."""
    lon = np.arctan2(y, x)
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
    n_ext = n + 2 * ng
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
    radius: float = 6.371e6,
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

    return DuoGridData(
        n=n,
        ng=ng,
        k2e_nord=k2e_nord,
        k2e_coef=jnp.array(k2e_coef, dtype=jnp.float64),
        k2e_lo=jnp.array(k2e_lo, dtype=jnp.int32),
        ext_lon=jnp.array(ext_lon, dtype=jnp.float64),
        ext_lat=jnp.array(ext_lat, dtype=jnp.float64),
        corner_xp=jnp.array(xp, dtype=jnp.float64),
        corner_xm=jnp.array(xm, dtype=jnp.float64),
        corner_yp=jnp.array(yp, dtype=jnp.float64),
        corner_ym=jnp.array(ym, dtype=jnp.float64),
    )


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
) -> jax.Array:
    """Fill corner blocks using FV3 Lagrange polynomial interpolation.

    Following FV3 fv_duogrid.F90 fill_corner_region_2d:
    - Non-diagonal corner cells are filled first via single-direction
      Lagrange interpolation (X+ or Y+ or X- or Y-)
    - Diagonal corner cells are filled as the average of X-direction and
      Y-direction Lagrange interpolations

    Falls back to simple averaging when Lagrange coefficients are not
    available (corner_xp is None).
    """
    n = duogrid.n
    ng = duogrid.ng
    h = halo

    if h == 0:
        return padded

    # Fall back to averaging if no corner coefficients
    if getattr(duogrid, 'corner_xp', None) is None:
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
    offset = ng - h  # extended_idx = padded_idx + offset

    # Interior bounds in padded coords
    ie = h + n - 1  # last interior i in padded
    je = h + n - 1  # last interior j in padded

    # Following FV3 fill_corner_region_2d pattern for each corner.
    # NE corner: cells at (ie+1..ie+h, je+1..je+h)
    for d1 in range(1, h + 1):
        for d2 in range(1, h + 1):
            i_p = ie + d1  # padded i
            j_p = je + d2  # padded j
            i_e = i_p + offset  # extended grid i
            j_e = j_p + offset

            if d1 == d2:
                # Diagonal: average of X+ and Y+
                val_x = _lagrange_interp_x_plus(padded, xp, i_e, j_e, j_p, n, h)
                val_y = _lagrange_interp_y_plus(padded, yp, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(0.5 * (val_x + val_y))
            elif d2 > d1:
                # Above diagonal: use X+ (interpolate from east edge)
                val = _lagrange_interp_x_plus(padded, xp, i_e, j_e, j_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)
            else:
                # Below diagonal: use Y+ (interpolate from north edge)
                val = _lagrange_interp_y_plus(padded, yp, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)

    # NW corner: cells at (0..h-1, je+1..je+h)
    for d1 in range(1, h + 1):
        for d2 in range(1, h + 1):
            i_p = h - d1  # padded i (from west edge)
            j_p = je + d2
            i_e = i_p + offset
            j_e = j_p + offset

            if d1 == d2:
                val_x = _lagrange_interp_x_minus(padded, xm, i_e, j_e, j_p, n, h)
                val_y = _lagrange_interp_y_plus(padded, yp, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(0.5 * (val_x + val_y))
            elif d2 > d1:
                val = _lagrange_interp_x_minus(padded, xm, i_e, j_e, j_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)
            else:
                val = _lagrange_interp_y_plus(padded, yp, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)

    # SE corner: cells at (ie+1..ie+h, 0..h-1)
    for d1 in range(1, h + 1):
        for d2 in range(1, h + 1):
            i_p = ie + d1
            j_p = h - d2  # from south edge
            i_e = i_p + offset
            j_e = j_p + offset

            if d1 == d2:
                val_x = _lagrange_interp_x_plus(padded, xp, i_e, j_e, j_p, n, h)
                val_y = _lagrange_interp_y_minus(padded, ym, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(0.5 * (val_x + val_y))
            elif d2 > d1:
                val = _lagrange_interp_x_plus(padded, xp, i_e, j_e, j_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)
            else:
                val = _lagrange_interp_y_minus(padded, ym, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)

    # SW corner: cells at (0..h-1, 0..h-1)
    for d1 in range(1, h + 1):
        for d2 in range(1, h + 1):
            i_p = h - d1
            j_p = h - d2
            i_e = i_p + offset
            j_e = j_p + offset

            if d1 == d2:
                val_x = _lagrange_interp_x_minus(padded, xm, i_e, j_e, j_p, n, h)
                val_y = _lagrange_interp_y_minus(padded, ym, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(0.5 * (val_x + val_y))
            elif d2 > d1:
                val = _lagrange_interp_x_minus(padded, xm, i_e, j_e, j_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)
            else:
                val = _lagrange_interp_y_minus(padded, ym, i_e, j_e, i_p, n, h)
                padded = padded.at[:, i_p, j_p].set(val)

    return padded


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
