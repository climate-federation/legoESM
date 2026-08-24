"""Exact FV3 cubed-sphere metric construction (phase 2 of the FV3-native path).

Python replication of the GFDL FV3 ``fv_grid_tools.F90`` ``init_grid`` +
``grid_area`` metric assembly for the GENERATED (non-stretched, non-bounded)
cubed sphere, validated field-by-field against the verbatim Fortran
extraction (``scripts/validate/fv3_native/fv3_gnomonic_oracle.f90``, fixture
``tests/grids/fixtures/fv3_gnomonic_ed_oracle.npz``) at C1/C8/C36 in
``tests/grids/test_fv3_native_metrics_phase2.py``.

Constructed fields (FV3 face numbering, one array per face stacked on axis 0):

- ``area``   (6, n, n)     spherical-excess cell areas (``get_area`` quads)
- ``dx``     (6, n, n+1)   corner-to-corner edge lengths along x
- ``dy``     (6, n+1, n)   corner-to-corner edge lengths along y
- ``dxa``    (6, n, n)     through-cell distances between edge mid-points
- ``dya``    (6, n, n)
- ``dxc``    (6, n+1, n)   centre-to-centre distances; panel edges are
                           ``2 x (edge-midpoint -> centre)`` (the FV3 ×2
                           convention — NOT copied interior values)
- ``dyc``    (6, n, n+1)
- ``area_c`` (6, n+1, n+1) dual-cell areas: interior agrid quads, then the
                           FV3 init_grid ×2 half-dual edge overwrites applied
                           in W,E,S,N code order (the cube corners therefore
                           carry the S/N edge-formula values — the FINAL FV3
                           state; the grid_area corner triangles are
                           intermediate, replicated for order fidelity)

Cross-face halo grid/agrid lines (mpp_update_domains upstream) are
reconstructed geometrically: shared cube-edge nodes are bit-equal between
faces by the mirror construction, so side pairing and orientation follow
from exact node matching.

All computation is float64 numpy at grid-construction time (init-time only,
never traced); storage casts happen at the integration boundary in
``create_cubed_sphere_cdgrid``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_fv3_native_metrics", "compute_fv3_native_angles"]


# --------------------------------------------------------------------------
# Spherical primitives — value-equivalent ports of fv_grid_utils.F90.
# (Right-handed xyz is used throughout; angles, distances and areas are
# invariant under the z-flip of FV3's default converters.)
# --------------------------------------------------------------------------
def _latlon2xyz(p: np.ndarray) -> np.ndarray:
    """(..., 2) lon/lat [rad] -> (..., 3) unit-sphere cartesian."""
    lon, lat = p[..., 0], p[..., 1]
    return np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=-1)


def _great_circle_dist(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Haversine great-circle distance on the unit sphere (fv_grid_utils)."""
    dlat = 0.5 * (q1[..., 1] - q2[..., 1])
    dlon = 0.5 * (q1[..., 0] - q2[..., 0])
    s = np.sqrt(np.sin(dlat) ** 2
                + np.cos(q1[..., 1]) * np.cos(q2[..., 1]) * np.sin(dlon) ** 2)
    return 2.0 * np.arcsin(np.clip(s, 0.0, 1.0))


def _mid_pt_sphere(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Great-circle midpoint via the normalized cartesian sum (mid_pt3_cart)."""
    e = _latlon2xyz(p1) + _latlon2xyz(p2)
    e = e / np.linalg.norm(e, axis=-1, keepdims=True)
    lon = np.where(np.abs(e[..., 0]) + np.abs(e[..., 1]) < 1e-10,
                   0.0, np.arctan2(e[..., 1], e[..., 0]))
    lat = np.arcsin(np.clip(e[..., 2], -1.0, 1.0))
    return np.stack([lon, lat], axis=-1)


def _spherical_angle(e1: np.ndarray, e2: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """Interior angle at e1 between great circles to e2 and e3.

    Computed in extended precision (np.longdouble — 80-bit on x86), mirroring
    FV3's quad-precision ``f_p`` intermediates: the spherical-excess area is
    a tiny difference of angles near pi/2, and plain double precision leaves
    ~1e-12 relative cancellation noise in C36 cell areas (measured 7.6e-13
    against the quad-precision Fortran oracle; longdouble restores <1e-13).
    """
    e1 = np.asarray(e1, dtype=np.longdouble)
    e2 = np.asarray(e2, dtype=np.longdouble)
    e3 = np.asarray(e3, dtype=np.longdouble)
    p = np.cross(e1, e2)
    q = np.cross(e1, e3)
    ddd = np.einsum("...i,...i", p, p) * np.einsum("...i,...i", q, q)
    dot = np.einsum("...i,...i", p, q)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(ddd > 0.0, dot / np.sqrt(np.where(ddd > 0, ddd, 1.0)),
                     np.longdouble(1.0))
    # FV3 computes the angle in f_p (quad) but STORES it into an R_GRID
    # (double) variable before the excess sum — mirror that rounding.
    return np.arccos(np.clip(c, -1.0, 1.0)).astype(np.float64)


def _get_area_quad(p1, p2, p3, p4) -> np.ndarray:
    """Spherical-excess area of the quad, arg roles as fv_grid_utils get_area:

    ``get_area(p1, p4, p2, p3)`` upstream computes the four interior angles
    at (p1: SW between p2,p4), (p2: SE between p3,p1), (p3: NE between
    p4,p2), (p4: NW between p3,p1).  Here the arguments are passed in the
    GEOMETRIC roles (p1=SW/lL, p2=SE/lR, p3=NE/uR, p4=NW/uL) and the angle
    pattern below reproduces the upstream sum exactly.
    """
    e1, e2, e3, e4 = (
        _latlon2xyz(np.asarray(p, dtype=np.longdouble)) for p in (p1, p2, p3, p4)
    )
    ang1 = _spherical_angle(e1, e2, e4)
    ang2 = _spherical_angle(e2, e3, e1)
    ang3 = _spherical_angle(e3, e4, e2)
    ang4 = _spherical_angle(e4, e3, e1)
    # double-precision sum of double-rounded angles, as upstream
    return ang1 + ang2 + ang3 + ang4 - 2.0 * np.pi


def _get_area_tri(p1, p2, p3) -> float:
    """Spherical-excess area of a triangle (get_area_tri, lat/lon branch)."""
    e1, e2, e3 = (_latlon2xyz(np.asarray(p)) for p in (p1, p2, p3))
    a = _spherical_angle(e2, e1, e3)
    b = _spherical_angle(e3, e2, e1)
    c = _spherical_angle(e1, e3, e2)
    return float(a + b + c - np.pi)


def _cell_center2(q1, q2, q3, q4) -> np.ndarray:
    """FV3 agrid centre: normalized xyz sum of the 4 corners (order-free)."""
    e = (_latlon2xyz(q1) + _latlon2xyz(q2) + _latlon2xyz(q3) + _latlon2xyz(q4))
    e = e / np.linalg.norm(e, axis=-1, keepdims=True)
    lon = np.where(np.abs(e[..., 0]) + np.abs(e[..., 1]) < 1e-10,
                   0.0, np.arctan2(e[..., 1], e[..., 0]))
    lon = np.where(lon < 0.0, lon + 2.0 * np.pi, lon)
    lat = np.arcsin(np.clip(e[..., 2], -1.0, 1.0))
    return np.stack([lon, lat], axis=-1)


# --------------------------------------------------------------------------
# Cross-face halo reconstruction (geometric side matching)
# --------------------------------------------------------------------------
_SIDES = ("S", "N", "W", "E")


def _side_nodes(grid_f: np.ndarray, side: str) -> np.ndarray:
    if side == "S":
        return grid_f[:, 0]
    if side == "N":
        return grid_f[:, -1]
    if side == "W":
        return grid_f[0, :]
    return grid_f[-1, :]


def _side_inner_nodes(grid_f: np.ndarray, side: str) -> np.ndarray:
    if side == "S":
        return grid_f[:, 1]
    if side == "N":
        return grid_f[:, -2]
    if side == "W":
        return grid_f[1, :]
    return grid_f[-2, :]


def _side_inner_cells(agrid_f: np.ndarray, side: str) -> np.ndarray:
    if side == "S":
        return agrid_f[:, 0]
    if side == "N":
        return agrid_f[:, -1]
    if side == "W":
        return agrid_f[0, :]
    return agrid_f[-1, :]


def _match_sides(grid6: np.ndarray) -> dict:
    """For each (face, side) find (neighbour face, side, reversed).

    Matching key = the cube-edge midpoint (normalized xyz sum of the side's
    end nodes), unique per cube edge; orientation from the first node.
    """
    def side_mid(f, s):
        nodes = _side_nodes(grid6[f], s)
        e = _latlon2xyz(nodes[0]) + _latlon2xyz(nodes[-1])
        return e / np.linalg.norm(e)

    out = {}
    for f in range(6):
        for s in _SIDES:
            m = side_mid(f, s)
            found = None
            for g in range(6):
                if g == f:
                    continue
                for sp in _SIDES:
                    if np.sum((side_mid(g, sp) - m) ** 2) < 1e-20:
                        found = (g, sp)
                        break
                if found:
                    break
            if found is None:  # pragma: no cover - construction invariant
                raise RuntimeError(f"no neighbour for face {f} side {s}")
            g, sp = found
            a = _latlon2xyz(_side_nodes(grid6[f], s)[0])
            b = _latlon2xyz(_side_nodes(grid6[g], sp)[0])
            rev = bool(np.sum((a - b) ** 2) > 1e-20)
            out[(f, s)] = (g, sp, rev)
    return out


def _fill_halos(grid6: np.ndarray, agrid6: np.ndarray):
    """1-deep cross-face halo lines for grid nodes and agrid centres.

    Returns ``gridh`` of shape (6, npx+2, npx+2, 2) and ``agridh`` of shape
    (6, n+2, n+2, 2) with interiors at [1:-1, 1:-1].  Diagonal halo entries
    hold a poison value: the vectorized area_c step-1 quad sweep DOES read
    them at the four outermost B-nodes, but every one of those nodes is
    overwritten by the corner-triangle and then the W/E/S/N edge blocks
    before the field is returned, so poison never reaches an output (same
    garbage-then-overwrite pattern as FV3's own fill_ghost big_number use).
    """
    npx = grid6.shape[1]
    n = npx - 1
    gridh = np.full((6, npx + 2, npx + 2, 2), -1e25)
    agridh = np.full((6, n + 2, n + 2, 2), -1e25)
    gridh[:, 1:-1, 1:-1] = grid6
    agridh[:, 1:-1, 1:-1] = agrid6

    matches = _match_sides(grid6)
    for f in range(6):
        for s in _SIDES:
            g, sp, rev = matches[(f, s)]
            nodes = _side_inner_nodes(grid6[g], sp)
            cells = _side_inner_cells(agrid6[g], sp)
            if rev:
                nodes = nodes[::-1]
                cells = cells[::-1]
            if s == "S":
                gridh[f, 1:-1, 0] = nodes
                agridh[f, 1:-1, 0] = cells
            elif s == "N":
                gridh[f, 1:-1, -1] = nodes
                agridh[f, 1:-1, -1] = cells
            elif s == "W":
                gridh[f, 0, 1:-1] = nodes
                agridh[f, 0, 1:-1] = cells
            else:
                gridh[f, -1, 1:-1] = nodes
                agridh[f, -1, 1:-1] = cells
    return gridh, agridh


# --------------------------------------------------------------------------
# Metric assembly (fv_grid_tools init_grid + grid_area, generated-grid path)
# --------------------------------------------------------------------------
def compute_fv3_native_metrics(
    lon6: np.ndarray, lat6: np.ndarray, radius: float,
) -> dict:
    """Exact FV3 metrics from 6-face corner grids in FV3 numbering.

    Parameters
    ----------
    lon6, lat6 : (6, n+1, n+1) float64
        Cubed-sphere corner positions (``make_fv3_native_grid`` output —
        FV3 face numbering; any consistent 6-face layout works since every
        construction is face-local plus geometrically matched halos).
    radius : float
        Sphere radius [m].

    Returns
    -------
    dict with keys area, dx, dy, dxa, dya, dxc, dyc, area_c, agrid_lon,
    agrid_lat (shapes documented in the module docstring), float64.
    """
    lon6 = np.asarray(lon6, dtype=np.float64)
    lat6 = np.asarray(lat6, dtype=np.float64)
    nf, npx, _ = lon6.shape
    if nf != 6:  # pragma: no cover - guard
        raise ValueError(f"expected 6 faces, got {nf}")
    n = npx - 1
    grid6 = np.stack([lon6, lat6], axis=-1)          # (6, npx, npx, 2)

    # agrid via cell_center2 of the 4 cell corners
    agrid6 = _cell_center2(grid6[:, :-1, :-1], grid6[:, 1:, :-1],
                           grid6[:, :-1, 1:], grid6[:, 1:, 1:])  # (6, n, n, 2)

    gridh, agridh = _fill_halos(grid6, agrid6)

    area = np.zeros((6, n, n))
    dx = np.zeros((6, n, npx))
    dy = np.zeros((6, npx, n))
    dxa = np.zeros((6, n, n))
    dya = np.zeros((6, n, n))
    dxc = np.zeros((6, npx, n))
    dyc = np.zeros((6, n, npx))
    area_c = np.zeros((6, npx, npx))

    for f in range(6):
        G = gridh[f]      # (npx+2, npx+2, 2), interior at [1:-1]
        A = agridh[f]     # (n+2,  n+2,  2)
        gi = G[1:-1, 1:-1]   # (npx, npx, 2) interior corner grid
        ai = A[1:-1, 1:-1]   # (n, n, 2) interior agrid

        # dx / dy: corner-to-corner edge lengths
        dx[f] = _great_circle_dist(gi[1:, :], gi[:-1, :])          # (n, npx)
        dy[f] = _great_circle_dist(gi[:, 1:], gi[:, :-1])          # (npx, n)

        # dxa / dya: distances between opposing edge mid-points
        mid_w = _mid_pt_sphere(gi[:-1, :-1], gi[:-1, 1:])   # west edge mids
        mid_e = _mid_pt_sphere(gi[1:, :-1], gi[1:, 1:])     # east edge mids
        dxa[f] = _great_circle_dist(mid_e, mid_w)
        mid_s = _mid_pt_sphere(gi[:-1, :-1], gi[1:, :-1])   # south edge mids
        mid_n = _mid_pt_sphere(gi[:-1, 1:], gi[1:, 1:])     # north edge mids
        dya[f] = _great_circle_dist(mid_n, mid_s)

        # dxc: interior agrid distances; panel edges = 2*(mid -> centre)
        dxc[f, 1:n, :] = _great_circle_dist(ai[1:, :], ai[:-1, :])
        dxc[f, 0, :] = 2.0 * _great_circle_dist(mid_w[0, :], ai[0, :])
        dxc[f, n, :] = 2.0 * _great_circle_dist(ai[-1, :], mid_e[-1, :])
        # dyc analog
        dyc[f, :, 1:n] = _great_circle_dist(ai[:, 1:], ai[:, :-1])
        dyc[f, :, 0] = 2.0 * _great_circle_dist(mid_s[:, 0], ai[:, 0])
        dyc[f, :, n] = 2.0 * _great_circle_dist(ai[:, -1], mid_n[:, -1])

        # area: spherical-excess corner quads (SW, SE, NE, NW roles)
        area[f] = _get_area_quad(gi[:-1, :-1], gi[1:, :-1],
                                 gi[1:, 1:], gi[:-1, 1:])

        # area_c step 1 (grid_area): agrid quads everywhere, halo agrid at
        # panel borders.  Node (i,j) [0-based over npx] uses centres
        # (i-1..i, j-1..j) in halo-array coordinates A[i..i+1, j..j+1].
        area_c[f] = _get_area_quad(A[:-1, :-1], A[1:, :-1],
                                   A[1:, 1:], A[:-1, 1:])

        # area_c step 2 (grid_area cube corners): triangles of the 3
        # surrounding centres (intermediate — kept for order fidelity)
        area_c[f, 0, 0] = _get_area_tri(A[0, 1], A[1, 1], A[1, 0])
        area_c[f, n, 0] = _get_area_tri(A[n, 1], A[n, 0], A[n + 1, 1])
        area_c[f, n, n] = _get_area_tri(A[n, n], A[n + 1, n], A[n, n + 1])
        area_c[f, 0, n] = _get_area_tri(A[1, n], A[1, n + 1], A[0, n])

        # area_c step 3 (init_grid): ×2 half-dual overwrites, W,E,S,N order.
        # Upstream call get_area(p1, p4, p2, p3) with the block-specific
        # point sets; expressed here in the geometric (SW, SE, NE, NW) roles
        # of _get_area_quad.
        # West (i=0), all j: quad(mid(j-1,j), agrid(0,j-1), agrid(0,j),
        #                         mid(j,j+1))
        gw = G[1]                       # west corner column w/ halo, (npx+2, 2)
        mw = _mid_pt_sphere(gw[:-1], gw[1:])   # mids (j-1/2), (npx+1, 2)
        area_c[f, 0, :] = 2.0 * _get_area_quad(
            mw[:-1], A[1, :-1], A[1, 1:], mw[1:])[: npx]
        # East (i=n): quad(agrid(n-1,j-1), mid(j-1,j), mid(j,j+1),
        #                  agrid(n-1,j))
        ge = G[-2]
        me = _mid_pt_sphere(ge[:-1], ge[1:])
        area_c[f, n, :] = 2.0 * _get_area_quad(
            A[n, :-1], me[:-1], me[1:], A[n, 1:])[: npx]
        # South (j=0): quad(mid(i-1,i), mid(i,i+1), agrid(i,0), agrid(i-1,0))
        gs = G[:, 1]
        ms = _mid_pt_sphere(gs[:-1], gs[1:])
        area_c[f, :, 0] = 2.0 * _get_area_quad(
            ms[:-1], ms[1:], A[1:, 1], A[:-1, 1])[: npx]
        # North (j=n): quad(agrid(i-1,n-1), agrid(i,n-1), mid(i,i+1),
        #                   mid(i-1,i))
        gn = G[:, -2]
        mn = _mid_pt_sphere(gn[:-1], gn[1:])
        area_c[f, :, n] = 2.0 * _get_area_quad(
            A[:-1, n], A[1:, n], mn[1:], mn[:-1])[: npx]

    return {
        "area": area * radius**2,
        "dx": dx * radius,
        "dy": dy * radius,
        "dxa": dxa * radius,
        "dya": dya * radius,
        "dxc": dxc * radius,
        "dyc": dyc * radius,
        "area_c": area_c * radius**2,
        "agrid_lon": agrid6[..., 0],
        "agrid_lat": agrid6[..., 1],
    }


# --------------------------------------------------------------------------
# Exact staggered angle/tangent fields (grid_utils_init,
# fv_grid_utils.F90:240-561) — phase 2B
# --------------------------------------------------------------------------
# Public aliases for the certified geometric primitives (cross-module
# consumers: fv3_native_gridstruct).  Defined after the private bodies below.
def latlon2xyz(p: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``latlon2xyz`` port."""
    return _latlon2xyz(p)


def great_circle_dist(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``great_circle_dist`` port."""
    return _great_circle_dist(q1, q2)


def mid_pt_sphere(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``mid_pt_sphere`` port."""
    return _mid_pt_sphere(p1, p2)


def cell_center2(q1, q2, q3, q4) -> np.ndarray:
    """Public wrapper over the certified ``cell_center2`` port."""
    return _cell_center2(q1, q2, q3, q4)


def get_area_quad(p1, p2, p3, p4) -> np.ndarray:
    """Public wrapper over the certified spherical-excess quad area port."""
    return _get_area_quad(p1, p2, p3, p4)


def mid_pt3(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``mid_pt3_cart`` port."""
    return _mid_pt3(e1, e2)


def _cos_angle_ld(e1: np.ndarray, e2: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """cos of the angle at e1 between great circles to e2 and e3 (cos_angle).

    Longdouble intermediates mirror the upstream quad ``f_p``; the result is
    double-rounded like the R_GRID return upstream.
    """
    e1 = np.asarray(e1, dtype=np.longdouble)
    e2 = np.asarray(e2, dtype=np.longdouble)
    e3 = np.asarray(e3, dtype=np.longdouble)
    p = np.cross(e1, e2)
    q = np.cross(e1, e3)
    ddd = np.sqrt(np.einsum("...i,...i", p, p)
                  * np.einsum("...i,...i", q, q))
    dot = np.einsum("...i,...i", p, q)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(ddd > 0.0, dot / np.where(ddd > 0, ddd, 1.0),
                     np.longdouble(1.0))
    return np.asarray(c, dtype=np.float64)


def _mid_pt3(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """mid_pt3_cart: normalized cartesian midpoint (xyz in, xyz out)."""
    e = e1 + e2
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# Vector primitives the DCMIP16_BC wind assembly needs
# (fv_grid_utils.F90: vect_cross :1781, get_unit_vect2 :1848,
#  normalize_vect :1880, inner_prod :984, get_latlon_vector :3326)
#
# WHERE UPSTREAM USES QUAD, THIS USES longdouble, matching the convention
# already established by _cos_angle_ld above. `f_p = selected_real_kind(20)`
# whenever NO_QUAD_PRECISION is undefined (fv_grid_utils.F90:43-49), which is
# the case for both the Zenodo duo build and the reference build here.
# x86 longdouble is 80-bit, NOT the 128-bit quad gfortran uses, so this
# NARROWS the gap without closing it -- the residual is the parity floor
# (~1e-16 relative in the unit vectors, propagating linearly into u/v).
# --------------------------------------------------------------------------

def _vect_cross(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """vect_cross (fv_grid_utils.F90:1781-1791) -- e = P1 x P2.

    Plain R_GRID (f64) upstream: this is the ONE primitive in this group with
    no f_p promotion, so it stays float64 deliberately.
    """
    return np.cross(np.asarray(p1, dtype=np.float64),
                    np.asarray(p2, dtype=np.float64))


def _normalize_vect(e: np.ndarray) -> np.ndarray:
    """normalize_vect (:1880-1893). ``pdot`` is f_p upstream."""
    e_ld = np.asarray(e, dtype=np.longdouble)
    pdot = np.sqrt(np.einsum("...i,...i", e_ld, e_ld))
    return np.asarray(e_ld / pdot[..., None], dtype=np.float64)


def _inner_prod(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """inner_prod (:984-996).

    Upstream copies BOTH operands into f_p, forms the sum of products in
    f_p, and assigns back to a default real. Reproduced with longdouble.
    """
    a = np.asarray(v1, dtype=np.longdouble)
    b = np.asarray(v2, dtype=np.longdouble)
    return np.asarray(np.einsum("...i,...i", a, b), dtype=np.float64)


def _get_latlon_vector(pp: np.ndarray) -> tuple:
    """get_latlon_vector (:3326-3340) -> (elon, elat), the local east/north
    unit vectors at the lon/lat point ``pp``.

    RIGHT_HAND system: ``elat(3) = +cos(lat)``. The left-hand variant is
    present upstream but COMMENTED OUT (:3338); using it flips the meridional
    component's sign, so the choice is load-bearing rather than cosmetic.
    """
    pp = np.asarray(pp, dtype=np.float64)
    lon, lat = pp[..., 0], pp[..., 1]
    zero = np.zeros_like(lon)
    elon = np.stack([-np.sin(lon), np.cos(lon), zero], axis=-1)
    elat = np.stack([-np.sin(lat) * np.cos(lon),
                     -np.sin(lat) * np.sin(lon),
                     np.cos(lat)], axis=-1)
    return elon, elat


def _get_unit_vect2(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """get_unit_vect2 (:1848-1863) -- unit tangent pointing e1 --> e2.

    Note the ORDER of the first cross product: ``vect_cross(p3, p2, p1)`` is
    p2 x p1, not p1 x p2. Swapping it reverses the tangent, which would flip
    the sign of every wind projected onto it -- and a sign-flipped wind field
    still looks entirely plausible.
    """
    p1 = _latlon2xyz(np.asarray(e1, dtype=np.float64))
    p2 = _latlon2xyz(np.asarray(e2, dtype=np.float64))
    pc = _mid_pt3(p1, p2)              # mid_pt3_cart
    p3 = _vect_cross(p2, p1)           # :1859  p2 x p1  (this order)
    uc = _vect_cross(pc, p3)           # :1860
    return _normalize_vect(uc)         # :1861


def vect_cross(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``vect_cross`` port."""
    return _vect_cross(p1, p2)


def normalize_vect(e: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``normalize_vect`` port."""
    return _normalize_vect(e)


def inner_prod(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``inner_prod`` port."""
    return _inner_prod(v1, v2)


def get_latlon_vector(pp: np.ndarray) -> tuple:
    """Public wrapper over the certified ``get_latlon_vector`` port."""
    return _get_latlon_vector(pp)


def get_unit_vect2(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Public wrapper over the certified ``get_unit_vect2`` port."""
    return _get_unit_vect2(e1, e2)


def sg_window_fields(X: np.ndarray, ctr: np.ndarray):
    """cos_sg/sin_sg over a cell window (grid_utils_init "No averaging").

    Parameters
    ----------
    X : (M+1, M+1, 3) float64
        Corner-node cartesian positions (any window; halo-inclusive OK).
    ctr : (M, M, 3) float64
        Cell-centre cartesian positions for the same cells.

    Returns
    -------
    csg, ssg : (M, M, 9) float64 with FV3 positions 1..9 at index 0..8
        (0=W, 1=S, 2=E, 3=N mid-edges; 4=centre; 5=SW, 6=SE, 7=NE, 8=NW).
        ``ssg = min(1, sqrt(max(0, 1 - csg**2)))`` exactly as upstream.

    Shared by ``compute_fv3_native_angles`` (phase 2B) and the phase-4
    single-tile gridstruct builder — one construction, never re-derived.
    """
    A = X[:-1, :-1]
    B = X[1:, :-1]
    C = X[:-1, 1:]
    D = X[1:, 1:]

    m0, m1 = ctr.shape[0], ctr.shape[1]
    csg = np.empty((m0, m1, 9))
    # corners (upstream sign pattern; FV3 6,7,8,9 -> idx 5,6,7,8)
    csg[..., 5] = _cos_angle_ld(A, B, C)
    csg[..., 6] = -_cos_angle_ld(B, A, D)
    csg[..., 7] = _cos_angle_ld(D, B, C)
    csg[..., 8] = -_cos_angle_ld(C, A, D)
    # edge mid-points (FV3 1..4 -> idx 0..3): W, S, E, N
    csg[..., 0] = _cos_angle_ld(_mid_pt3(A, C), ctr, C)
    csg[..., 1] = _cos_angle_ld(_mid_pt3(A, B), B, ctr)
    csg[..., 2] = _cos_angle_ld(_mid_pt3(B, D), ctr, B)
    csg[..., 3] = _cos_angle_ld(_mid_pt3(C, D), C, ctr)
    # centre (FV3 5 -> idx 4): inner_prod(ec1, ec2), get_center_vect
    pc = A + B + C + D
    pc = pc / np.linalg.norm(pc, axis=-1, keepdims=True)  # cell_center3
    p3 = np.cross(_mid_pt3(B, D), _mid_pt3(A, C))
    ec1 = np.cross(pc, p3)
    ec1 = ec1 / np.linalg.norm(ec1, axis=-1, keepdims=True)
    p3 = np.cross(_mid_pt3(C, D), _mid_pt3(A, B))
    ec2 = np.cross(pc, p3)
    ec2 = ec2 / np.linalg.norm(ec2, axis=-1, keepdims=True)
    csg[..., 4] = np.einsum(
        "...i,...i", ec1.astype(np.longdouble),
        ec2.astype(np.longdouble)).astype(np.float64)

    ssg = np.minimum(1.0, np.sqrt(np.maximum(0.0, 1.0 - csg**2)))
    return csg, ssg


def compute_fv3_native_angles(lon6: np.ndarray, lat6: np.ndarray) -> dict:
    """Exact FV3 staggered angle/tangent fields from 6-face corner grids.

    Replicates ``grid_utils_init``'s exact ("No averaging") construction:

    - ``cos_sg``/``sin_sg`` (6, n, n, 9), positions 0-indexed as legoESM's
      convention (0=W,1=S,2=E,3=N mid-edges; 4=centre; 5=SW,6=SE,7=NE,8=NW
      corners — FV3's 1..9 shifted by one): corners via ``cos_angle`` at the
      cell corner nodes with the upstream sign pattern; edge mid-points via
      ``mid_pt3_cart`` + the agrid centre; centre via
      ``inner_prod(ec1, ec2)`` from ``get_center_vect``'s cell-local vectors.
    - ``cosa_u``/``sina_u`` (6, n+1, n) and ``cosa_v``/``sina_v`` (6, n, n+1):
      the 0.5*(sg_E + sg_W) staggered averages INCLUDING the cross-face
      cells at panel edges (via the geometric halo reconstruction) — the
      upstream mpp-halo semantics.  The legacy single-sided edge shortcut is
      wrong across a cube seam: the two faces' coordinate lines kink there,
      so their sg values differ.
    - ``cosa_b``/``sina_b`` (6, n+1, n+1): B-node averages
      0.5*(sg_NE[i-1,j-1] + sg_SW[i,j]); the four cube-vertex nodes are NaN
      (upstream computes them from fill_corners XDir ghost geometry and
      poisons ``rsina`` there — that convention bundle is phase-4).

    Reciprocal fields (rsin_u/v/rsin2/rsina) are left to the caller: their
    edge conventions are solver-facing policy (see the cdgrid factory).

    Layout-agnostic: any consistent 6-face corner layout works (constructions
    are face-local plus geometrically matched halos).  float64 in/out.
    """
    lon6 = np.asarray(lon6, dtype=np.float64)
    lat6 = np.asarray(lat6, dtype=np.float64)
    nf, npx, _ = lon6.shape
    if nf != 6:  # pragma: no cover - guard
        raise ValueError(f"expected 6 faces, got {nf}")
    n = npx - 1
    grid6 = np.stack([lon6, lat6], axis=-1)
    agrid6 = _cell_center2(grid6[:, :-1, :-1], grid6[:, 1:, :-1],
                           grid6[:, :-1, 1:], grid6[:, 1:, 1:])
    gridh, agridh = _fill_halos(grid6, agrid6)

    cos_sg = np.zeros((6, n, n, 9))
    sin_sg = np.zeros((6, n, n, 9))
    cosa_u = np.zeros((6, npx, n))
    sina_u = np.zeros((6, npx, n))
    cosa_v = np.zeros((6, n, npx))
    sina_v = np.zeros((6, n, npx))
    cosa_b = np.full((6, npx, npx), np.nan)
    sina_b = np.full((6, npx, npx), np.nan)

    for f in range(6):
        X = _latlon2xyz(gridh[f])          # (npx+2, npx+2, 3) nodes w/ halo
        ctr = _latlon2xyz(agridh[f])       # (n+2, n+2, 3) centres w/ halo
        # sg over ALL (n+2)x(n+2) halo cells (diagonal halo cells produce
        # garbage from poison nodes; they are never read — only side-halo
        # cells feed the staggered averages)
        csg, ssg = sg_window_fields(X, ctr)

        cos_sg[f] = csg[1:-1, 1:-1]
        sin_sg[f] = ssg[1:-1, 1:-1]

        # staggered averages over the halo cell window: u-face k (0..n)
        # sits between window-cells k and k+1 (halo coords)
        cosa_u[f] = 0.5 * (csg[:-1, 1:-1, 2] + csg[1:, 1:-1, 0])
        sina_u[f] = 0.5 * (ssg[:-1, 1:-1, 2] + ssg[1:, 1:-1, 0])
        cosa_v[f] = 0.5 * (csg[1:-1, :-1, 3] + csg[1:-1, 1:, 1])
        sina_v[f] = 0.5 * (ssg[1:-1, :-1, 3] + ssg[1:-1, 1:, 1])
        # B-nodes: 0.5*(sg_NE of SW cell + sg_SW of NE cell); the four
        # cube-vertex nodes would read diagonal halo cells -> stay NaN
        cb = 0.5 * (csg[:-1, :-1, 7] + csg[1:, 1:, 5])
        sb = 0.5 * (ssg[:-1, :-1, 7] + ssg[1:, 1:, 5])
        cb[0, 0] = cb[0, -1] = cb[-1, 0] = cb[-1, -1] = np.nan
        sb[0, 0] = sb[0, -1] = sb[-1, 0] = sb[-1, -1] = np.nan
        cosa_b[f] = cb
        sina_b[f] = sb

    return {
        "cos_sg": cos_sg, "sin_sg": sin_sg,
        "cosa_u": cosa_u, "sina_u": sina_u,
        "cosa_v": cosa_v, "sina_v": sina_v,
        "cosa_b": cosa_b, "sina_b": sina_b,
    }


# --------------------------------------------------------------------------
# Wind unit vectors for PHYSICS COUPLING (grid_utils_init,
# fv_grid_utils.F90:262-317 and :2362) — the four quantities
# ``update_dwinds_phys`` (:3363-3547) needs to turn an A-grid wind
# TENDENCY into D-grid increments.  Nothing in the dycore consumes them:
# it works in grid-relative components and never forms a lat-lon A-grid
# wind, so physics coupling is their first consumer.
# --------------------------------------------------------------------------

def compute_fv3_native_wind_vectors(grid_lon: np.ndarray,
                                    grid_lat: np.ndarray,
                                    agrid_lon: np.ndarray,
                                    agrid_lat: np.ndarray) -> dict:
    """``vlon``/``vlat``/``es[..., 1]``/``ew[..., 2]`` for ONE face.

    Parameters are one face's padded corner lon/lat ``(m + 1, m + 1)`` and
    cell-centre lon/lat ``(m, m)``, in radians, on the data domain — the
    same arrays ``fv3_native_gridstruct`` publishes as ``grid_lon`` /
    ``agrid_lon``.

    Returns ``{"vlon", "vlat", "es1", "ew2"}``.  ``vlon``/``vlat`` are
    ``(m, m, 3)``, the local east/north unit vectors at cell centres
    (``unit_vect_latlon``, :2286-2309, invoked at :2362).  ``es1`` and
    ``ew2`` are ``(m, m + 1, 3)`` and ``(m + 1, m, 3)``: the unit vectors
    the D-grid ``u`` and ``v`` components lie along, at north/south and
    east/west cell edges respectively.

    THE DUO LANE IS WHY THIS IS SHORT.  ``bounded_domain`` is forced true
    for a duo grid (``fv_arrays.F90:1512``: ``regional .or. nested .or.
    duogrid``), so every ``.not. bounded_domain`` branch in the upstream
    construction — the zeroed cube-vertex diagonals at :266/:294 and the
    four panel-edge special cases at :270-277 and :297-304 — is dead
    here, and one formula covers the whole face.  On an UNBOUNDED cubed
    sphere those branches are live and this routine would be wrong; that
    is why it refuses to pretend otherwise (see ``bounded_domain`` in the
    caller's context).

    ponytail: only the two slots ``update_dwinds_phys`` reads are built.
    Upstream also fills ``es[..., 2]`` and ``ew[..., 1]``, but their only
    consumers are the old ``d2a2c_vect`` routines behind
    ``USE_NORM_VECT`` (:654-687), which is not defined in this build and
    whose blocks are ``.not. bounded_domain`` guarded anyway.  Add them
    beside these when something actually reads them.

    PRECISION, stated rather than assumed: ``mid_pt3_cart`` is ``f_p``
    (extended) upstream and the port's ``_mid_pt3`` is float64.  That is
    a pre-existing property of the shared primitive, reused here on
    purpose so these vectors sit at the same floor as everything else
    built from it, not silently re-derived at a different one.

    CELLS UPSTREAM NEVER WRITES COME BACK ``NaN``, not zero.  The
    Fortran loops start at ``isd+1`` / ``jsd+1``, so the first column of
    ``ew`` and the first row of ``es`` are whatever ``allocate`` left
    there.  A zero would be a finite, plausible unit vector that silently
    projects a wind to nothing; ``NaN`` makes a consumer that reaches
    outside the written window fail where it reads.
    """
    grid_lon = np.asarray(grid_lon, dtype=np.float64)
    grid_lat = np.asarray(grid_lat, dtype=np.float64)
    agrid_lon = np.asarray(agrid_lon, dtype=np.float64)
    agrid_lat = np.asarray(agrid_lat, dtype=np.float64)
    if grid_lon.shape != grid_lat.shape or grid_lon.ndim != 2:
        raise ValueError(
            f"grid_lon/grid_lat must be one face's 2-D corner arrays of "
            f"equal shape, got {grid_lon.shape} and {grid_lat.shape}")
    mb = grid_lon.shape[0]
    if grid_lon.shape[1] != mb:
        raise ValueError(f"corner array must be square, got {grid_lon.shape}")
    m = mb - 1
    if agrid_lon.shape != (m, m) or agrid_lat.shape != (m, m):
        raise ValueError(
            f"agrid_lon/agrid_lat must be ({m}, {m}) for a ({mb}, {mb}) "
            f"corner array, got {agrid_lon.shape} and {agrid_lat.shape}")

    # :2362  vlon, vlat = unit_vect_latlon(agrid) -- the same routine the
    # port already carries as get_latlon_vector.
    vlon, vlat = _get_latlon_vector(
        np.stack([agrid_lon, agrid_lat], axis=-1))

    grid_ll = np.stack([grid_lon, grid_lat], axis=-1)

    # es(:, i, j, 1) at :311-313 is
    #     pp = mid_pt_cart(grid(i,j), grid(i+1,j))
    #     p3 = grid3(i,j) x grid3(i+1,j)
    #     es = normalize(p3 x pp)
    # and get_unit_vect2(e1, e2) (:1848-1863, already ported here) is
    #     pc = mid_pt3(p1, p2);  p3 = p2 x p1;  uc = normalize(pc x p3)
    # with pc == pp.  Since pc x (p2 x p1) = (p1 x p2) x pc term by term,
    # the two are the SAME expression -- so this reuses the certified
    # primitive instead of transcribing the formula a second time.
    # Neither ``agrid`` nor the panel-edge branches enter: on a bounded
    # domain the edge tangent is the edge's own great circle.
    #
    # Fortran: j = jsd+1 .. jed, i = isd .. ied -- so the FIRST row (jsd)
    # and the LAST row (jed+1) are never written, even though both are
    # inside the allocation.
    es1 = np.full((m, m + 1, 3), np.nan, dtype=np.float64)
    es1[:, 1:-1, :] = _get_unit_vect2(grid_ll[:-1, 1:-1, :],
                                      grid_ll[1:, 1:-1, :])

    # ew(:, i, j, 2), :283-285.  Same construction on the WEST edge, where
    # the D-grid v lives.  Fortran: j = jsd .. jed, i = isd+1 .. ied, so
    # the LAST column (ied+1) is unwritten too.
    ew2 = np.full((m + 1, m, 3), np.nan, dtype=np.float64)
    ew2[1:-1, :, :] = _get_unit_vect2(grid_ll[1:-1, :-1, :],
                                      grid_ll[1:-1, 1:, :])

    return {"vlon": vlon, "vlat": vlat, "es1": es1, "ew2": ew2}
