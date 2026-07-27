"""Faithful six-face port of the authoritative duo-grid ext machinery.

Source of truth: Zenodo 8327578 (Mouallem 2023, "novel Duo-Grid in
GFDL's FV3"), ``atmos_cubed_sphere-symmetryclean/tools/fv_duogrid.F90``
plus ``model/fv_grid_utils.F90`` (c2l_ord2, get_center_vect).  This
module supplies the stepper-side (numpy, Fortran-indexed) analogs of

- ``ext_scalar(var, dg, bd, domain, istag, jstag)`` for A (0,0) and
  B (1,1) staggerings   [fv_duogrid.F90:456-623],
- ``ext_vector(u, v, …, 0,1,1,0)`` (D winds) and ``(…,1,0,0,1)``
  (C winds)             [fv_duogrid.F90:626-975],

replacing the interim index-copy exchanges + the rejected per-stagger
position-only vector remap.  The authoritative vector flow is a
LAT-LON INTERMEDIARY at A-grid points (upstream deliberately does NOT
remap C/D components at their own staggered positions — fv_duogrid
:678-681 records that doing so "produced more noise"):

1. plain mpp vector halo exchange (DGRID_NE / CGRID_NE),
2. ``c2l_ord2`` / ``c2l_ord2_cgrid``: covariant staggered components →
   geographic (east/north) winds at cell centers over the compute
   domain + one ring (``do_halo``), using the vorticity-conserving
   covariant average and the exact ``a11..a22`` center transform
   [fv_grid_utils.F90:2360-2380, 2547-2628; duo c2l_ord2_cgrid
   fv_duogrid.F90:2765-2844],
3. scalar exchange of both geographic components on the ng=4 lattice
   (``domain_for_duo``; set_bd_ext_duo pins ``dg%bd%ng = 4``),
4. ``cube_rmp`` with the A-grid k2e tables (rings 1..4) + Lagrange
   corner-region fill (``fill_corner_region``, laginter default),
5. ``cubed_a2d_halo`` / ``cubed_a2c_halo``: geographic → 3-D Cartesian
   at A points → 2-point edge average → projection onto the EXTENDED
   lattice edge bases ``es_ext/ew_ext`` [fv_duogrid.F90:2590-2763],
6. halo side strips written back into the staggered wind arrays
   (rmp_s/n/w/e; all four fire on one-rank-per-face),
7. ``fill_corner_region`` per component at its OWN staggering.

Scalars have no basis, so steps 2/5 drop out and the remap runs at the
field's own staggered positions with its own table family (A or B).

All arrays follow the stepper's data-domain layout: Fortran index
``1-ng`` at numpy 0 on both axes; ``ng`` is the stepper halo width (3).
The internal geographic lattice uses ``_NG_P1 = 4`` rings exactly like
``domain_for_duo``.
"""

from __future__ import annotations

import os

import numpy as np
from legoesm.grids.fv3_native_gridstruct import (
    exchange_agrid_scalar_halos,
    exchange_bgrid_scalar_halos,
    exchange_cgrid_vector_halos,
    exchange_dgrid_vector_halos,
    k2e_remap_halo_rings,
)

# Vertex-instability diagnostic mode (codex vertex-kill C2), frozen at
# import: the hot _CornerLagrange.fill must not re-read the environment
# per call, and a mid-run env change must not alter a running
# experiment (codex screens-r1 F6).  Default OFF = faithful.
_CORNER_NEAREST = os.environ.get("LEGOESM_DUO_CORNER_MODE", "") == "nearest"

# upstream constants (fv_duogrid.F90)
_NG_P1 = 4          # set_bd_ext_duo: dg%bd%ng = 4        (line 146)
_INTERP_ORDER = 3   # interporder — Lagrange order is +1   (line 80)


# ---------------------------------------------------------------------------
# geometry primitives (fv_grid_utils.F90 conventions)
# ---------------------------------------------------------------------------

def _xyz(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _unit_vect_latlon(lon, lat):
    vlon = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)],
                    axis=-1)
    vlat = np.stack([-np.sin(lat) * np.cos(lon),
                     -np.sin(lat) * np.sin(lon),
                     np.cos(lat)], axis=-1)
    return vlon, vlat


def center_a_matrix(gs: dict) -> tuple:
    """``a11, a12, a21, a22`` on the full (mpp-state) data domain.

    fv_grid_utils.F90 init_grid_utils:2360-2380: ``z`` = center
    covariant unit basis (``get_center_vect`` ec1/ec2) dotted with the
    geographic unit vectors at agrid; ``a`` = the ±0.5·z / sin_sg(5)
    pairing.  ``sin_sg(5)`` is recomputed from ``cos_sg(5) =
    inner(ec1, ec2)`` rather than read from ``gs["sin_sg"]``, whose
    corner-diagonal slots carry the c_sw transport-patch/tiny-floor
    conventions (1e-8) and would poison the a-matrix inside c2l's
    do_halo ring.  NOT bit-identical to upstream at the four
    corner-DIAGONAL cells (codex ext r1 P2-4): upstream
    ``get_center_vect`` zeroes those ec vectors and ``fill_ghost``
    applies the tiny floor, so upstream's a-matrix there is
    floor-poisoned garbage of a different flavor (~1e-8-scale sin
    difference).  Both variants are DEAD: the geographic corner
    Lagrange overwrites every value derived from them before
    projection.  Non-corner cells agree with upstream to machine
    precision.  Valid wherever the mpp-state ``grid``/``agrid`` halos
    are real — the whole data domain.
    """
    p = _xyz(gs["grid_lon"], gs["grid_lat"])            # corners (m_b, m_b, 3)
    # cell_center3: normalized 4-corner sum
    pc = _norm(p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])
    # get_center_vect (non-OLD_VECT): ec1 = norm(pc x (midE x midW)),
    # ec2 = norm(pc x (midN x midS)); mid_pt3_cart = normalized 2-sum.
    mid_w = _norm(p[:-1, :-1] + p[:-1, 1:])
    mid_e = _norm(p[1:, :-1] + p[1:, 1:])
    ec1 = _norm(np.cross(pc, np.cross(mid_e, mid_w)))
    mid_s = _norm(p[:-1, :-1] + p[1:, :-1])
    mid_n = _norm(p[:-1, 1:] + p[1:, 1:])
    ec2 = _norm(np.cross(pc, np.cross(mid_n, mid_s)))

    vlon, vlat = _unit_vect_latlon(gs["agrid_lon"], gs["agrid_lat"])
    z11 = (ec1 * vlon).sum(-1)
    z12 = (ec1 * vlat).sum(-1)
    z21 = (ec2 * vlon).sum(-1)
    z22 = (ec2 * vlat).sum(-1)
    cos5 = (ec1 * ec2).sum(-1)
    sin5 = np.sqrt(np.maximum(1.0 - cos5 * cos5, 1.0e-30))
    a11 = 0.5 * z22 / sin5
    a12 = -0.5 * z12 / sin5
    a21 = -0.5 * z21 / sin5
    a22 = 0.5 * z11 / sin5
    return a11, a12, a21, a22


# ---------------------------------------------------------------------------
# c2l_ord2 (D) / c2l_ord2_cgrid (C)  — covariant staggered -> geographic
# ---------------------------------------------------------------------------

def c2l_ord2_face(u: np.ndarray, v: np.ndarray, dx: np.ndarray,
                  dy: np.ndarray, amat: tuple,
                  n: int, ng: int) -> tuple:
    """fv_grid_utils.F90:2547-2628 with ``do_halo=.true.`` (one ring).

    ``u`` (m_a, m_b): D-grid x-wind on y-faces; ``v`` (m_b, m_a).
    ``dx``/``dy``: the KINKED mpp-state metrics (upstream c2l reads the
    model gridstruct, never the ext lattice).  Returns geographic
    ``ua, va`` (m_a, m_a) valid on Fortran ``is-1..ie+1`` (rest NaN).
    """
    a11, a12, a21, a22 = amat
    lo = 1 - ng
    s, e = (1 - 1) - lo, (n + 1) - lo               # numpy is-1 .. ie+1
    ua = np.full((n + 2 * ng, n + 2 * ng), np.nan)
    va = np.full_like(ua, np.nan)
    isl = slice(s, e + 1)
    # wu = u*dx on the two bounding y-faces; covariant average
    wu_lo = u[isl, s:e + 1] * dx[isl, s:e + 1]
    wu_hi = u[isl, s + 1:e + 2] * dx[isl, s + 1:e + 2]
    u1 = 2.0 * (wu_lo + wu_hi) / (dx[isl, s:e + 1] + dx[isl, s + 1:e + 2])
    wv_lo = v[s:e + 1, isl] * dy[s:e + 1, isl]
    wv_hi = v[s + 1:e + 2, isl] * dy[s + 1:e + 2, isl]
    v1 = 2.0 * (wv_lo + wv_hi) / (dy[s:e + 1, isl] + dy[s + 1:e + 2, isl])
    ua[isl, isl] = a11[isl, isl] * u1 + a12[isl, isl] * v1
    va[isl, isl] = a21[isl, isl] * u1 + a22[isl, isl] * v1
    return ua, va


def c2l_ord2_cgrid_face(uc: np.ndarray, vc: np.ndarray, dx: np.ndarray,
                        dy: np.ndarray, amat: tuple,
                        n: int, ng: int) -> tuple:
    """fv_duogrid.F90:2765-2844 (duo companion), ``do_halo=.true.``.

    ``uc`` (m_b, m_a): C-grid x-wind on x-faces; ``vc`` (m_a, m_b).
    """
    a11, a12, a21, a22 = amat
    lo = 1 - ng
    s, e = (1 - 1) - lo, (n + 1) - lo
    ua = np.full((n + 2 * ng, n + 2 * ng), np.nan)
    va = np.full_like(ua, np.nan)
    isl = slice(s, e + 1)
    # wu = u*dy on the two bounding x-faces (note the metric swap)
    wu_lo = uc[s:e + 1, isl] * dy[s:e + 1, isl]
    wu_hi = uc[s + 1:e + 2, isl] * dy[s + 1:e + 2, isl]
    u1 = 2.0 * (wu_lo + wu_hi) / (dy[s:e + 1, isl] + dy[s + 1:e + 2, isl])
    wv_lo = vc[isl, s:e + 1] * dx[isl, s:e + 1]
    wv_hi = vc[isl, s + 1:e + 2] * dx[isl, s + 1:e + 2]
    v1 = 2.0 * (wv_lo + wv_hi) / (dx[isl, s:e + 1] + dx[isl, s + 1:e + 2])
    ua[isl, isl] = a11[isl, isl] * u1 + a12[isl, isl] * v1
    va[isl, isl] = a21[isl, isl] * u1 + a22[isl, isl] * v1
    return ua, va


# ---------------------------------------------------------------------------
# corner-region Lagrange fill (fill_corner_region / compute_lagrange_coeff)
# ---------------------------------------------------------------------------

def ext_parity_lonlat_ref(n: int, ng: int, parity: str):
    """ED EXTENDED lon/lat at a supergrid parity, REFERENCE (FV3 tile)
    face numbering — the stepper's convention (build_kinked/
    build_extended_corner_lonlat use raw ``_ED_CARTS``).

    ``_ed_ext_agrid_lonlat``/``_ed_ext_stagger_lonlat`` in
    fv3_native_halos return the CREATE face layout (perm/rot applied)
    for the jax pad_halo stack — using those here puts every basis on
    the wrong face.  Same lattice values, no remap.

    parity "A": (2i, 2j) nodes, (6, n+2ng, n+2ng);
    parity "B": (2i-1, 2j-1) nodes, (6, n+2ng+1, n+2ng+1).
    """
    from legoesm.grids.fv3_native_halos import _ED_CARTS, _ed_line

    line = _ed_line(n, 2 * (ng + 2))
    if parity == "A":
        idx = np.arange(1 - ng, n + ng + 1)
        vals = np.array([line[2 * i] for i in idx])
    elif parity == "B":
        idx = np.arange(1 - ng, n + ng + 2)
        vals = np.array([line[2 * i - 1] for i in idx])
    else:  # pragma: no cover - guard
        raise ValueError(parity)
    xg, yg = np.meshgrid(vals, vals, indexing="ij")
    m = len(idx)
    lon6 = np.zeros((6, m, m))
    lat6 = np.zeros((6, m, m))
    for t in range(6):
        cx, cy, cz = _ED_CARTS[t](xg, yg)
        r = np.sqrt(cx * cx + cy * cy + cz * cz)
        lon6[t] = np.mod(np.arctan2(cy, cx), 2.0 * np.pi)
        lat6[t] = np.arcsin(cz / r)
    return lon6, lat6


def _row_arc_coords(lon_row, lat_row):
    """Signed arc-length coordinate along one lattice line.

    The ED extension lines are gnomonic coordinate lines = great
    circles, so pairwise great-circle distances equal arc-coordinate
    differences; upstream's it-signed distance-ratio products
    (compute_lagrange_coeff) are then exactly the 1-D Lagrange weights
    on these abscissae.
    """
    p = _xyz(lon_row, lat_row)
    d = np.arccos(np.clip((p[:-1] * p[1:]).sum(-1), -1.0, 1.0))
    x = np.zeros(len(lon_row))
    x[1:] = np.cumsum(d)
    return x


def _lagrange_w(xt: float, xs: np.ndarray) -> np.ndarray:
    from legoesm.grids.fv3_native_halos import _lagrange_coef

    return _lagrange_coef(xt, xs)


class _CornerLagrange:
    """Per-stagger corner-region Lagrange operator on one face lattice.

    Mirrors fill_corner_region_2d [fv_duogrid.F90:1719-1903]: per
    corner, six one-directional extrapolations + three diagonal slots
    averaged between the X- and Y-direction results; abscissae from the
    A-point lattice with the upstream stagger index shift
    (compute_lagrange_coeff uses ``a_pt`` rows for every stagger).
    """

    def __init__(self, a_lon, a_lat, n: int, ng: int,
                 istag: int, jstag: int):
        self.n, self.ng = n, ng
        self.istag, self.jstag = istag, jstag
        # abscissa lattice: upstream reads dg%a_pt, which lives on the
        # ng=4 duo bounds — staggered corner targets reach A index
        # n+ng+1, one PAST the field's own ng-ring lattice.  a_lon/a_lat
        # may therefore be built with MORE rings than the field has;
        # infer the lattice origin from its shape.
        m_lat = a_lon.shape[0]
        lat_ng = (m_lat - n) // 2
        if lat_ng < max(istag, jstag) + _INTERP_ORDER:
            raise ValueError(
                f"corner-Lagrange abscissa lattice too small: {m_lat} "
                f"rows for n={n}, stagger ({istag},{jstag}) — staggered "
                f"targets read A abscissae to n+stag+{_INTERP_ORDER}")
        self.glo = 1 - lat_ng                      # lattice Fortran origin
        self.flo = 1 - ng                          # field Fortran origin
        self.xrow = np.array([_row_arc_coords(a_lon[i, :], a_lat[i, :])
                              for i in range(m_lat)])   # along j, per row i
        self.xcol = np.array([_row_arc_coords(a_lon[:, j], a_lat[:, j])
                              for j in range(m_lat)])   # along i, per col j

    def _w_x(self, i_t: int, j_t: int, plus: bool) -> tuple:
        """Weights + source Fortran i-list for an X± fill at (i_t, j_t)."""
        n = self.n
        glo = self.glo
        order = _INTERP_ORDER
        if plus:
            src = list(range(n - order, n + 1))          # ie-3..ie
            src_f = [s + self.istag for s in src]        # field cols read
        else:
            src = list(range(1, 1 + order + 1))          # is..is+3
            src_f = src
        # upstream abscissae: a_pt row j (or j-jstag), target index
        # i - istag (X+) / i (X-)  [fv_duogrid.F90:2181-2220]
        j_row = j_t if j_t > n - 1 else j_t - self.jstag
        row = self.xcol[j_row - glo]
        it = (i_t - self.istag) if plus else i_t
        xs = np.array([row[s - glo] for s in src])
        w = _lagrange_w(row[it - glo], xs)
        return w, src_f

    def _w_y(self, i_t: int, j_t: int, plus: bool) -> tuple:
        n = self.n
        glo = self.glo
        order = _INTERP_ORDER
        if plus:
            src = list(range(n - order, n + 1))
            src_f = [s + self.jstag for s in src]
        else:
            src = list(range(1, 1 + order + 1))
            src_f = src
        i_row = i_t if i_t > n - 1 else i_t - self.istag
        col = self.xrow[i_row - glo]
        jt = (j_t - self.jstag) if plus else j_t
        xs = np.array([col[s - glo] for s in src])
        w = _lagrange_w(col[jt - glo], xs)
        return w, src_f

    def _apply_dir(self, f: np.ndarray, i_t: int, j_t: int, direction: str):
        lo = self.flo
        if direction == "X+":
            w, src = self._w_x(i_t, j_t, True)
            vals = np.array([f[s - lo, j_t - lo] for s in src])
        elif direction == "X-":
            w, src = self._w_x(i_t, j_t, False)
            vals = np.array([f[s - lo, j_t - lo] for s in src])
        elif direction == "Y+":
            w, src = self._w_y(i_t, j_t, True)
            vals = np.array([f[i_t - lo, s - lo] for s in src])
        else:
            w, src = self._w_y(i_t, j_t, False)
            vals = np.array([f[i_t - lo, s - lo] for s in src])
        f[i_t - lo, j_t - lo] = float((w * vals).sum())

    def fill(self, f: np.ndarray):
        """The nine-slot per-corner sequence [fv_duogrid.F90:1743-1901]."""
        import os

        n = self.n
        lo = self.flo
        ie = n + self.istag                       # last compute slot
        je = n + self.jstag
        is_, js_ = 1, 1

        corner_nearest = _CORNER_NEAREST

        def diag(i_t, j_t, d1, d2):
            fa, fb = f.copy(), f.copy()
            self._apply_dir(fa, i_t, j_t, d1)
            self._apply_dir(fb, i_t, j_t, d2)
            f[i_t - lo, j_t - lo] = 0.5 * (fa[i_t - lo, j_t - lo]
                                           + fb[i_t - lo, j_t - lo])

        # NE
        for (i_t, j_t, d) in ((ie + 1, je + 2, "X+"), (ie + 1, je + 3, "X+"),
                              (ie + 2, je + 3, "X+"), (ie + 2, je + 1, "Y+"),
                              (ie + 3, je + 1, "Y+"), (ie + 3, je + 2, "Y+")):
            self._apply_dir(f, i_t, j_t, d)
        diag(ie + 1, je + 1, "X+", "Y+")
        diag(ie + 3, je + 3, "X+", "Y+")
        diag(ie + 2, je + 2, "X+", "Y+")
        # NW
        for (i_t, j_t, d) in ((is_ - 1, je + 2, "X-"), (is_ - 1, je + 3, "X-"),
                              (is_ - 2, je + 3, "X-"), (is_ - 2, je + 1, "Y+"),
                              (is_ - 3, je + 1, "Y+"), (is_ - 3, je + 2, "Y+")):
            self._apply_dir(f, i_t, j_t, d)
        diag(is_ - 1, je + 1, "X-", "Y+")
        diag(is_ - 3, je + 3, "X-", "Y+")
        diag(is_ - 2, je + 2, "X-", "Y+")
        # SE
        for (i_t, j_t, d) in ((ie + 1, js_ - 2, "X+"), (ie + 1, js_ - 3, "X+"),
                              (ie + 2, js_ - 3, "X+"), (ie + 2, js_ - 1, "Y-"),
                              (ie + 3, js_ - 1, "Y-"), (ie + 3, js_ - 2, "Y-")):
            self._apply_dir(f, i_t, j_t, d)
        diag(ie + 1, js_ - 1, "X+", "Y-")
        diag(ie + 3, js_ - 3, "X+", "Y-")
        diag(ie + 2, js_ - 2, "X+", "Y-")
        # SW
        for (i_t, j_t, d) in ((is_ - 1, js_ - 2, "X-"), (is_ - 1, js_ - 3, "X-"),
                              (is_ - 2, js_ - 3, "X-"), (is_ - 2, js_ - 1, "Y-"),
                              (is_ - 3, js_ - 1, "Y-"), (is_ - 3, js_ - 2, "Y-")):
            self._apply_dir(f, i_t, j_t, d)
        diag(is_ - 1, js_ - 1, "X-", "Y-")
        diag(is_ - 3, js_ - 3, "X-", "Y-")
        diag(is_ - 2, js_ - 2, "X-", "Y-")

        if corner_nearest:
            # DIAGNOSTIC (codex vertex-kill C2 screen, NON-FAITHFUL):
            # after the standard sequence, overwrite ONLY the 3x3
            # diagonal wedges with the nearest compute-corner value
            # (upstream fv_duogrid.F90:1715 warns Lagrange
            # extrapolation is "not highly recommended"; weights reach
            # ~35 -> overshoot on sharp fields).  Directional strips
            # keep their standard fills.
            for (ci, cj, si, sj) in ((ie + 1, je + 1, ie, je),
                                     (is_ - 1, je + 1, is_, je),
                                     (ie + 1, js_ - 1, ie, js_),
                                     (is_ - 1, js_ - 1, is_, js_)):
                di = 1 if ci > ie else -1
                dj = 1 if cj > je else -1
                for a in range(3):
                    for b in range(3):
                        f[ci + di * a - lo, cj + dj * b - lo] = \
                            f[si - lo, sj - lo]


# ---------------------------------------------------------------------------
# context: per-resolution precomputed tables/bases
# ---------------------------------------------------------------------------

def build_ext_context(n: int, ng: int, gs6: list, *,
                      vector_corner: str = "lagrange") -> dict:
    """Precompute everything the ext exchanges need at resolution n.

    ``gs6`` MUST be the KINKED (pre-``extend_gridstruct``) mpp-state
    gridstructs — c2l and the a-matrices read the model's kinked
    metrics, exactly like upstream (the ext lattice only enters via the
    a2stag bases and the k2e remap targets).

    - center a-matrices + kinked dx/dy snapshots per face,
    - ng=4 A-lattice geographic exchange machinery: k2e A tables to
      ring 4 (upstream ``dg%bd%ng = 4``) + A corner Lagrange operator,
    - EXT-lattice a2stag bases (vlon/vlat/ew/es) on the ng=4 lattice,
    - per-stagger corner Lagrange operators on the stepper lattice.
    """
    from legoesm.grids.fv3_native_halos import (
        _compute_ext_vectors_native,
    )

    amat6 = [center_a_matrix(gs) for gs in gs6]
    dx6 = [np.array(gs["dx"], copy=True) for gs in gs6]
    dy6 = [np.array(gs["dy"], copy=True) for gs in gs6]

    ngp = _NG_P1
    a_lon4, a_lat4 = ext_parity_lonlat_ref(n, ngp, "A")
    b_lon4, b_lat4 = ext_parity_lonlat_ref(n, ngp, "B")
    vlon4, vlat4, ew4, es4 = _compute_ext_vectors_native(
        a_lon4, a_lat4, b_lon4, b_lat4)

    # A tables at the ext lattice positions (per-face identical by ED
    # symmetry; stored tile-1) — rings 1..4 for the geographic lattice,
    # rings 1..ng for the stepper-lattice scalar exchanges.
    # every operator reads its abscissae from the ng=4 A lattice — the
    # analog of dg%a_pt living on the duo (ng=4) bounds, which staggered
    # corner targets index one past the field's own ng-ring lattice
    corner_a4 = [_CornerLagrange(a_lon4[t], a_lat4[t], n, ngp, 0, 0)
                 for t in range(6)]
    corner_a3 = [_CornerLagrange(a_lon4[t], a_lat4[t], n, ng, 0, 0)
                 for t in range(6)]
    corner_b3 = [_CornerLagrange(a_lon4[t], a_lat4[t], n, ng, 1, 1)
                 for t in range(6)]
    corner_du3 = [_CornerLagrange(a_lon4[t], a_lat4[t], n, ng, 0, 1)
                  for t in range(6)]
    corner_dv3 = [_CornerLagrange(a_lon4[t], a_lat4[t], n, ng, 1, 0)
                  for t in range(6)]

    if vector_corner not in ("lagrange", "a2d"):
        raise ValueError(f"vector_corner={vector_corner!r} "
                         "(expected 'lagrange' or 'a2d')")
    return {
        "n": n, "ng": ng, "amat6": amat6, "dx6": dx6, "dy6": dy6,
        "vlon4": vlon4, "vlat4": vlat4, "ew4": ew4, "es4": es4,
        "corner_a4": corner_a4, "corner_a3": corner_a3,
        "corner_b3": corner_b3,
        "corner_du3": corner_du3, "corner_dv3": corner_dv3,
        "vector_corner": vector_corner,
    }


# ---------------------------------------------------------------------------
# ext_scalar — A (0,0) and B (1,1)
# ---------------------------------------------------------------------------

def ext_scalar_sixface(f6: list, stag: str, ectx: dict):
    """ext_scalar analog: mpp exchange + cube_rmp rings + Lagrange
    corner regions, at the field's own staggering (A or B)."""
    n, ng = ectx["n"], ectx["ng"]
    if stag == "A":
        for t in range(1, 7):
            exchange_agrid_scalar_halos(f6, t, n, ng)
        k2e_remap_halo_rings(f6, "A", n, ng)
        for t in range(6):
            ectx["corner_a3"][t].fill(f6[t])
    elif stag == "B":
        for t in range(1, 7):
            exchange_bgrid_scalar_halos(f6, t, n, ng)
        k2e_remap_halo_rings(f6, "B", n, ng)
        for t in range(6):
            ectx["corner_b3"][t].fill(f6[t])
    else:
        raise ValueError(
            f"ext_scalar_sixface: stagger {stag!r} not implemented "
            "(upstream ext_scalar supports (0,0) and (1,1) only)")


# ---------------------------------------------------------------------------
# geographic A-lattice halo treatment on the ng=4 lattice
# ---------------------------------------------------------------------------

def _geo_lattice_exchange(g6: list, ectx: dict, dump=None,
                          names: tuple = ("S4_ullp1", "S5_ullp1")):
    """Steps 3-4 of the vector flow on the _NG_P1 lattice: neighbour
    exchange, k2e ring remap (rings 1..4), Lagrange corner regions.

    Corner-wedge state (codex ext r1 P2-6 / r2): rings 1..3 of each
    wedge end Lagrange-filled; the ring-4 wedge slots keep the AGRID
    index-copy values (kinked-position neighbours) — unlike upstream,
    where the mpp corner exchange leaves them undefined at cube
    vertices.  Non-propagating ON THE FAITHFUL (lagrange) PATH ONLY:
    the a2d/a2c consumers that touch ring-4 wedge rows feed only
    wedge-column strip slots, which the final per-component corner
    fill overwrites.  The vector_corner="a2d" variant SKIPS that
    overwrite and therefore DOES carry the ring-4 contamination into
    its wedge values (measured ~3.9 m/s at C12) — a2d is a
    measurement variant, not a production mode."""
    n = ectx["n"]
    ngp = _NG_P1
    for t in range(1, 7):
        exchange_agrid_scalar_halos(g6, t, n, ngp)
    k2e_remap_halo_rings(g6, "A", n, ngp)
    if dump:
        st, nm = names[0].split("_")     # post-k2e, pre-corner (cube_rmp)
        for t in range(6):
            dump(st, t, nm, g6[t])
    for t in range(6):
        ectx["corner_a4"][t].fill(g6[t])
    if dump:
        st, nm = names[1].split("_")     # post corner Lagrange (in-a2d fill)
        for t in range(6):
            dump(st, t, nm, g6[t])


def _pack_p1(ua: np.ndarray, n: int, ng: int) -> np.ndarray:
    """Embed the (m_a)^2 stepper-lattice array into the ng=4 lattice."""
    ngp = _NG_P1
    out = np.full((n + 2 * ngp, n + 2 * ngp), np.nan)
    d = ngp - ng
    out[d:d + n + 2 * ng, d:d + n + 2 * ng] = ua
    return out


# ---------------------------------------------------------------------------
# cubed_a2d_halo / cubed_a2c_halo on the ng=4 lattice
# ---------------------------------------------------------------------------

def _a2d_project(ug: np.ndarray, vg: np.ndarray, t: int, ectx: dict):
    """cubed_a2d_halo [fv_duogrid.F90:2676-2763]: geographic A → 3-D
    Cartesian → 2-point edge average → project onto es/ew ext bases.
    Returns ud (m4, m4-1), vd (m4-1, m4) at inter-center edge slots."""
    vlon, vlat = ectx["vlon4"][t], ectx["vlat4"][t]
    ew, es = ectx["ew4"][t], ectx["es4"][t]
    v3 = ug[..., None] * vlon + vg[..., None] * vlat
    ue = 0.5 * (v3[:, :-1] + v3[:, 1:])          # D-u slots (i, j-1/2)
    ve = 0.5 * (v3[:-1, :] + v3[1:, :])          # D-v slots (i-1/2, j)
    ud = (ue * es[:, 1:-1, :, 0]).sum(-1)
    vd = (ve * ew[1:-1, :, :, 1]).sum(-1)
    return ud, vd


def _a2c_project(ug: np.ndarray, vg: np.ndarray, t: int, ectx: dict):
    """cubed_a2c_halo [fv_duogrid.F90:2590-2674]."""
    vlon, vlat = ectx["vlon4"][t], ectx["vlat4"][t]
    ew, es = ectx["ew4"][t], ectx["es4"][t]
    v3 = ug[..., None] * vlon + vg[..., None] * vlat
    ue = 0.5 * (v3[:-1, :] + v3[1:, :])          # C-u slots (i-1/2, j)
    ve = 0.5 * (v3[:, :-1] + v3[:, 1:])          # C-v slots (i, j-1/2)
    uc = (ue * ew[1:-1, :, :, 0]).sum(-1)
    vc = (ve * es[:, 1:-1, :, 1]).sum(-1)
    return uc, vc


# ---------------------------------------------------------------------------
# ext_vector — D and C staggerings
# ---------------------------------------------------------------------------

def _write_d_strips(u: np.ndarray, v: np.ndarray, ud4: np.ndarray,
                    vd4: np.ndarray, n: int, ng: int):
    """Copy halo side strips from the ng=4 D projection lattice into
    the stepper wind arrays (ext_vector rmp_s/n/w/e blocks,
    fv_duogrid.F90:890-955: S/N strips span the full i data range, then
    W/E overwrite; corner wedges are re-filled by fill_corner_region).

    Index maps (Fortran → numpy): stepper D-u (cell i, node j) is numpy
    ``(i-1+ng, j-1+ng)``; ud4 rows are centers (``i-1+ngp``) and its
    columns are the inter-center nodes — node j (between centers j-1
    and j) sits at column ``(j-1) - (1-ngp) = j-2+ngp``.
    """
    ngp = _NG_P1
    npx = n + 1

    def u4(i_f, j_f):                 # Fortran (cell, node) -> ud4 value
        return ud4[i_f - 1 + ngp, j_f - 2 + ngp]

    def v4(i_f, j_f):                 # Fortran (node, cell) -> vd4 value
        return vd4[i_f - 2 + ngp, j_f - 1 + ngp]

    lo = 1 - ng
    # south / north (full i span)
    for j_f in list(range(1 - ng, 0 + 1)):
        for i_f in range(1 - ng, n + ng + 1):
            u[i_f - lo, j_f - lo] = u4(i_f, j_f)
    for j_f in list(range(npx + 1, npx + ng + 1)):
        for i_f in range(1 - ng, n + ng + 1):
            u[i_f - lo, j_f - lo] = u4(i_f, j_f)
    for j_f in list(range(1 - ng, 0 + 1)) + list(range(n + 1, n + ng + 1)):
        for i_f in range(1 - ng, npx + ng + 1):
            v[i_f - lo, j_f - lo] = v4(i_f, j_f)
    # west / east (full j span)
    for i_f in list(range(1 - ng, 0 + 1)) + list(range(n + 1, n + ng + 1)):
        for j_f in range(1 - ng, npx + ng + 1):
            u[i_f - lo, j_f - lo] = u4(i_f, j_f)
    for i_f in list(range(1 - ng, 0 + 1)) + list(range(npx + 1, npx + ng + 1)):
        for j_f in range(1 - ng, n + ng + 1):
            v[i_f - lo, j_f - lo] = v4(i_f, j_f)


def ext_vector_dgrid_sixface(u6: list, v6: list, ectx: dict):
    """ext_vector(u, v, …, 0,1,1,0): D-grid covariant winds.

    ``ectx["vector_corner"]`` selects the wedge treatment:
    "lagrange" (upstream-faithful DEFAULT: fill_corner_region
    re-extrapolates each covariant component; wedge accuracy ~0.08 m/s
    at C12 — codex r2 probe) or "a2d" (MEASUREMENT VARIANT ONLY: keep
    the strip-written cubed_a2d values; their ring-3 wedge slots
    consume the ring-4 index-copy geographic values and carry ~3.9 m/s
    contamination — see _geo_lattice_exchange; kept solely for the
    vertex-attribution A/B).
    """
    n, ng = ectx["n"], ectx["ng"]
    # optional oracle stage-dump hook: callable(stage, tile0, name, arr)
    # invoked at the same pipeline points instrumented in the Zenodo
    # model's ext_vector (fv3_recon ext_vector operand-diff harness);
    # None (default) = byte-identical production behavior.
    dump = ectx.get("stage_dump")
    for t in range(1, 7):
        exchange_dgrid_vector_halos(u6, v6, t, n, ng)
    if dump:
        for t in range(6):
            dump("S1", t, "uin", u6[t])
            dump("S1", t, "vin", v6[t])
    ug6, vg6 = [], []
    for t in range(6):
        ua, va = c2l_ord2_face(u6[t], v6[t], ectx["dx6"][t],
                               ectx["dy6"][t], ectx["amat6"][t],
                               n, ng)
        if dump:
            dump("S2", t, "ull", ua)
            dump("S2", t, "vll", va)
        ug6.append(_pack_p1(ua, n, ng))
        vg6.append(_pack_p1(va, n, ng))
    if dump:
        ngp = _NG_P1
        for g6, nm in ((ug6, "ullp1"), (vg6, "vllp1")):
            g6x = [np.array(g, copy=True) for g in g6]
            for t in range(1, 7):
                exchange_agrid_scalar_halos(g6x, t, n, ngp)
            for t in range(6):
                dump("S3", t, nm, g6x[t])
    _geo_lattice_exchange(ug6, ectx, dump=dump)
    _geo_lattice_exchange(vg6, ectx, dump=dump,
                          names=("S4_vllp1", "S5_vllp1"))
    for t in range(6):
        ud4, vd4 = _a2d_project(ug6[t], vg6[t], t, ectx)
        if dump:
            dump("S5", t, "up1", ud4)
            dump("S5", t, "vp1", vd4)
        _write_d_strips(u6[t], v6[t], ud4, vd4, n, ng)
        if ectx.get("vector_corner", "lagrange") == "lagrange":
            ectx["corner_du3"][t].fill(u6[t])
            ectx["corner_dv3"][t].fill(v6[t])
    if dump:
        for t in range(6):
            dump("S6", t, "uin", u6[t])
            dump("S6", t, "vin", v6[t])


def _write_c_strips(uc: np.ndarray, vc: np.ndarray, uc4: np.ndarray,
                    vc4: np.ndarray, n: int, ng: int):
    """C-stagger mirror of :func:`_write_d_strips` (u on x-faces)."""
    ngp = _NG_P1
    npx = n + 1
    lo = 1 - ng

    def u4(i_f, j_f):                 # Fortran (node, cell) -> uc4 value
        return uc4[i_f - 2 + ngp, j_f - 1 + ngp]

    def v4(i_f, j_f):                 # Fortran (cell, node) -> vc4 value
        return vc4[i_f - 1 + ngp, j_f - 2 + ngp]

    for j_f in list(range(1 - ng, 0 + 1)) + list(range(n + 1, n + ng + 1)):
        for i_f in range(1 - ng, npx + ng + 1):
            uc[i_f - lo, j_f - lo] = u4(i_f, j_f)
    for j_f in list(range(1 - ng, 0 + 1)) + list(range(npx + 1, npx + ng + 1)):
        for i_f in range(1 - ng, n + ng + 1):
            vc[i_f - lo, j_f - lo] = v4(i_f, j_f)
    for i_f in list(range(1 - ng, 0 + 1)) + list(range(npx + 1, npx + ng + 1)):
        for j_f in range(1 - ng, n + ng + 1):
            uc[i_f - lo, j_f - lo] = u4(i_f, j_f)
    for i_f in list(range(1 - ng, 0 + 1)) + list(range(n + 1, n + ng + 1)):
        for j_f in range(1 - ng, npx + ng + 1):
            vc[i_f - lo, j_f - lo] = v4(i_f, j_f)


def ext_vector_cgrid_sixface(uc6: list, vc6: list, ectx: dict):
    """ext_vector(uc, vc, …, 1,0,0,1): C-grid covariant winds."""
    n, ng = ectx["n"], ectx["ng"]
    for t in range(1, 7):
        exchange_cgrid_vector_halos(uc6, vc6, t, n, ng)
    ug6, vg6 = [], []
    for t in range(6):
        ua, va = c2l_ord2_cgrid_face(uc6[t], vc6[t], ectx["dx6"][t],
                                     ectx["dy6"][t], ectx["amat6"][t],
                                     n, ng)
        ug6.append(_pack_p1(ua, n, ng))
        vg6.append(_pack_p1(va, n, ng))
    _geo_lattice_exchange(ug6, ectx)
    _geo_lattice_exchange(vg6, ectx)
    for t in range(6):
        uc4, vc4 = _a2c_project(ug6[t], vg6[t], t, ectx)
        _write_c_strips(uc6[t], vc6[t], uc4, vc4, n, ng)
        if ectx.get("vector_corner", "lagrange") == "lagrange":
            ectx["corner_dv3"][t].fill(uc6[t])     # C-u stagger = (1,0)
            ectx["corner_du3"][t].fill(vc6[t])     # C-v stagger = (0,1)
