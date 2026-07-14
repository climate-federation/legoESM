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

__all__ = ["compute_fv3_native_metrics"]


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
