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
    if ng not in (1, 2, 3):
        raise ValueError(f"ng={ng} not supported, must be 1, 2, or 3")

    # Step P1: Build extended grid
    ext_lon, ext_lat = _build_extended_grid(n, ng)

    # Step P2: Build kinked grid
    kik_lon, kik_lat = _build_kinked_grid(n, ng, ext_lon, ext_lat)

    # Step P3: Extract 1D coordinates (gnomonic alpha, not latitude)
    ext_1d, kik_1d = _extract_1d_coords(n, ng, ext_lon, ext_lat, kik_lon, kik_lat)

    # Step P4: Compute k2e coefficients
    k2e_coef, k2e_lo = _compute_k2e_coefficients(n, ng, k2e_nord, ext_1d, kik_1d)

    return DuoGridData(
        n=n,
        ng=ng,
        k2e_nord=k2e_nord,
        k2e_coef=jnp.array(k2e_coef, dtype=jnp.float64),
        k2e_lo=jnp.array(k2e_lo, dtype=jnp.int32),
        ext_lon=jnp.array(ext_lon, dtype=jnp.float64),
        ext_lat=jnp.array(ext_lat, dtype=jnp.float64),
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

    for d in range(min(halo, duogrid.ng)):
        # South edge (edge=2): interpolate along axis 1 at fixed axis-2 row
        j_rc = h - 1 - d
        lo_s = k2e_lo[:, SOUTH, d, :]       # (6, n)
        coef_s = k2e_coef[:, SOUTH, d, :, :]  # (6, n, MAX)
        src_s = h + jnp.clip(
            lo_s[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :],
            0, n - 1)  # (6, n, MAX)
        for f in range(6):
            vals = padded[f, src_s[f], j_rc]  # (n, MAX)
            remapped = jnp.sum(vals * coef_s[f], axis=-1)
            padded = padded.at[f, dst_idx, j_rc].set(remapped)

        # North edge (edge=3)
        j_rc = h + n + d
        lo_n = k2e_lo[:, NORTH, d, :]
        coef_n = k2e_coef[:, NORTH, d, :, :]
        src_n = h + jnp.clip(
            lo_n[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :],
            0, n - 1)
        for f in range(6):
            vals = padded[f, src_n[f], j_rc]
            remapped = jnp.sum(vals * coef_n[f], axis=-1)
            padded = padded.at[f, dst_idx, j_rc].set(remapped)

        # West edge (edge=0): interpolate along axis 2 at fixed axis-1 col
        i_rc = h - 1 - d
        lo_w = k2e_lo[:, WEST, d, :]
        coef_w = k2e_coef[:, WEST, d, :, :]
        src_w = h + jnp.clip(
            lo_w[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :],
            0, n - 1)
        for f in range(6):
            vals = padded[f, i_rc, src_w[f]]
            remapped = jnp.sum(vals * coef_w[f], axis=-1)
            padded = padded.at[f, i_rc, dst_idx].set(remapped)

        # East edge (edge=1)
        i_rc = h + n + d
        lo_e = k2e_lo[:, EAST, d, :]
        coef_e = k2e_coef[:, EAST, d, :, :]
        src_e = h + jnp.clip(
            lo_e[:, :, None] + jnp.arange(MAX_K2E_NORD)[None, None, :],
            0, n - 1)
        for f in range(6):
            vals = padded[f, i_rc, src_e[f]]
            remapped = jnp.sum(vals * coef_e[f], axis=-1)
            padded = padded.at[f, i_rc, dst_idx].set(remapped)

    return padded


def fill_corner_region(
    padded: jax.Array,
    duogrid: DuoGridData,
    halo: int,
) -> jax.Array:
    """Fill h×h corner blocks by averaging adjacent edge halo values.

    Each corner cell is the average of its two neighbors one step closer
    to the interior (one from the i-direction edge strip, one from the
    j-direction edge strip or a previously filled corner cell).

    Fills inside-out along anti-diagonals: cells with the largest sum of
    relative coordinates (closest to interior) are filled first, ensuring
    each cell's dependencies are satisfied before it is computed.

    This matches the existing _fill_corners_h1/_fill_corners_h2 behavior
    and generalizes to arbitrary halo width.
    """
    n = duogrid.n
    h = halo
    n_p = n + 2 * h

    if h == 0:
        return padded

    # Process anti-diagonals from interior outward.
    # For relative corner coordinates (ci, cj) ∈ [0, h-1]², the inner
    # diagonal has ci + cj = 2*(h-1), the outer corner has ci + cj = 0.
    for s in range(2 * (h - 1), -1, -1):
        for ci in range(h):
            cj = s - ci
            if cj < 0 or cj >= h:
                continue

            # --- SW corner (low-i, low-j) ---
            # Neighbor toward interior: (ci+1, cj) and (ci, cj+1)
            # When ci+1 == h or cj+1 == h, the neighbor is an edge cell.
            ni_sw = min(ci + 1, h)
            nj_sw = min(cj + 1, h)
            padded = padded.at[:, ci, cj].set(
                0.5 * (padded[:, ni_sw, cj] + padded[:, ci, nj_sw])
            )

            # --- SE corner (high-i, low-j) ---
            i_se = n_p - 1 - ci
            ni_se = n_p - 1 - min(ci + 1, h)  # toward interior = i - 1
            nj_se = min(cj + 1, h)
            padded = padded.at[:, i_se, cj].set(
                0.5 * (padded[:, ni_se, cj] + padded[:, i_se, nj_se])
            )

            # --- NW corner (low-i, high-j) ---
            j_nw = n_p - 1 - cj
            ni_nw = min(ci + 1, h)
            nj_nw = n_p - 1 - min(cj + 1, h)  # toward interior = j - 1
            padded = padded.at[:, ci, j_nw].set(
                0.5 * (padded[:, ni_nw, j_nw] + padded[:, ci, nj_nw])
            )

            # --- NE corner (high-i, high-j) ---
            i_ne = n_p - 1 - ci
            j_ne = n_p - 1 - cj
            ni_ne = n_p - 1 - min(ci + 1, h)
            nj_ne = n_p - 1 - min(cj + 1, h)
            padded = padded.at[:, i_ne, j_ne].set(
                0.5 * (padded[:, ni_ne, j_ne] + padded[:, i_ne, nj_ne])
            )

    return padded
