"""Exact FV3 duo-grid halo remap tables (phase 3 of the FV3-native path).

Python replication of the duo-grid variant's ``global_grid_mod`` k2e
(kinked-to-extended) construction — the machinery behind Mouallem, Harris &
Chen (2023)'s cube-edge halo interpolation — validated record-by-record
against the VERBATIM Fortran oracle
(``scripts/validate/fv3_native/gen_duogrid_oracle.sh``, reference mirror
``luanfs/FV3_container`` @ 7d06431e, fixture
``tests/grids/fixtures/fv3_duogrid_oracle.npz``) at C12/C24 in
``tests/grids/test_fv3_native_halos_phase3.py``.

Pipeline (all init-time float64 numpy; Fortran 1-based index bookkeeping is
preserved internally so oracle records compare key-for-key):

1. the ED ("equal_edge") supergrid 1-D ``line`` with its gnomonic ghost
   continuation and mirror symmetry (``global_grid_gen_lonlat_equal_edge``);
2. the six gnomonic cube-face embeddings -> ``pt_ext`` latitudes;
3. ``pt_kik``: neighbour-tile values via the odd/even tile orientation index
   maps (``global_grid_update_kik`` / ``get_neighbor_*``);
4. per-halo-layer 1-D remap coordinates as latitude offsets along meridians
   (``global_grid_gen_coords`` — "take advantage of meridians");
5. per-stagger (A, B, CX, CY, DX, DY) bisection + Lagrange coefficient
   tables (``global_grid_gen_k2e`` ``get_loc_*`` bodies, incl. the
   ``[is+np, ie-np-1]`` window clamps).

These tables drive the FV3-native cross-face halo interpolation
(``ext_scalar``/``ext_vector``/``cube_rmp`` consumption); the legacy
equiangular duogrid in :mod:`legoesm.grids.duogrid` is untouched.
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_fv3_native_k2e", "create_fv3_native_duogrid_data"]

_STAGGERS = ("A", "B", "CX", "CY", "DX", "DY")


def _lagrange_coef(x: float, vx: np.ndarray) -> np.ndarray:
    """lib_interp_lag_get_coef: standard Lagrange weights at x over nodes vx."""
    n = len(vx)
    coef = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                coef[j] *= (x - vx[i]) / (vx[j] - vx[i])
    return coef


class _F:
    """1-based (Fortran-index) view over a numpy array with negative bounds."""

    def __init__(self, lo: int, hi: int, extra_dims: tuple = ()):
        self.lo = lo
        self.a = np.full((hi - lo + 1,) + extra_dims, -999.0)

    def __getitem__(self, f):
        return self.a[f - self.lo]

    def __setitem__(self, f, v):
        self.a[f - self.lo] = v


def _ed_line(n: int, sg_ng: int) -> "_F":
    """gen_lonlat_equal_edge 1-D supergrid line (grid_type == 0), ghosted.

    Returns the ED gnomonic tangent coordinates on supergrid indices
    ``1-sg_ng .. 2n+1+sg_ng`` (Fortran numbering via ``_F``): interior
    equal-angle tangents, lower ghosts by the reference's own
    ``tan(-pi/2 - atan(...))`` continuation, upper half mirrored.  One
    construction shared by every extended/kinked lattice builder here.
    """
    sg_is, sg_ie = 1, 2 * n + 1
    sg_nc = 1 + n
    sg_isd, sg_ied = sg_is - sg_ng, sg_ie + sg_ng

    line = _F(sg_isd, sg_ied)
    rsq3 = 1.0 / np.sqrt(3.0)
    alpha = np.arcsin(rsq3)
    dela = 2.0 * alpha / (sg_ie - sg_is)
    line[sg_is] = -1.0
    line[sg_nc] = 0.0
    for j in range(sg_is + 1, sg_nc):
        line[j] = np.tan((j - 1) * dela - alpha) * np.sqrt(2.0)
    # ghost continuation below the panel
    for j in range(sg_isd, sg_is):
        jj = 2 * sg_is - j
        line[j] = np.tan(-0.5 * np.pi - np.arctan(line[jj]))
    # mirror to the upper half (incl. upper ghosts)
    for j in range(sg_isd, sg_nc):
        line[sg_ie - j + 1] = -line[j]
    return line


# The six reference-numbering cube-face embeddings of the [-1, 1]^2 tangent
# plane (global_grid's tile order).
_ED_CARTS = (
    lambda x, y: (np.ones_like(x), x, y),        # tile 1
    lambda x, y: (-x, np.ones_like(x), y),       # tile 2
    lambda x, y: (-x, -y, np.ones_like(x)),      # tile 3
    lambda x, y: (-np.ones_like(x), -y, -x),     # tile 4
    lambda x, y: (y, -np.ones_like(x), -x),      # tile 5
    lambda x, y: (y, x, -np.ones_like(x)),       # tile 6
)


def ed_supergrid_lonlat_ref(n: int):
    """Compute-domain ED supergrid lon/lat, REFERENCE face numbering.

    Returns ``lon6, lat6`` of shape (6, 2n+1, 2n+1) over supergrid nodes
    1..2n+1 (index 0 == supergrid 1) — every stagger's physical positions
    (corners at odd-odd, centres at even-even, face mid-points mixed).
    Used by the phase-4 single-tile gridstruct builder to sample KINKED
    (neighbour-face) halo coordinates, i.e. exactly what an mpp halo
    exchange delivers.
    """
    line = _ed_line(n, 0)
    vals = np.array([line[j] for j in range(1, 2 * n + 2)])
    X, Y = np.meshgrid(vals, vals, indexing="ij")
    m = len(vals)
    lon6 = np.zeros((6, m, m))
    lat6 = np.zeros((6, m, m))
    for t in range(6):
        cx, cy, cz = _ED_CARTS[t](X, Y)
        r = np.sqrt(cx * cx + cy * cy + cz * cz)
        lon6[t] = np.mod(np.arctan2(cy, cx), 2.0 * np.pi)
        lat6[t] = np.arcsin(cz / r)
    return lon6, lat6


# Public aliases for the certified neighbour maps (cross-module consumers:
# fv3_native_gridstruct; private names retained for the oracle-pinned k2e
# internals below).
def neighbor_tiles(n: int) -> tuple[int, int, int, int]:
    """Public wrapper over the certified ``get_neighbor_tile_num`` port."""
    return _neighbor_tiles(n)


def neighbor_index(i: int, j: int, n: int, n_src: int, npx: int, npy: int):
    """Public wrapper over the certified ``get_neighbor_index`` port."""
    return _neighbor_index(i, j, n, n_src, npx, npy)


def _neighbor_tiles(n: int) -> tuple[int, int, int, int]:
    """get_neighbor_tile_num (tiles 1..6) -> (nw, ne, ns, nn)."""
    if n % 2 == 0:
        nn = (n + 0) % 6 + 1
        ne = (n + 1) % 6 + 1
        ns = (n + 3) % 6 + 1
        nw = (n + 4) % 6 + 1
    else:
        ne = (n + 0) % 6 + 1
        nn = (n + 1) % 6 + 1
        nw = (n + 3) % 6 + 1
        ns = (n + 4) % 6 + 1
    return nw, ne, ns, nn


def _neighbor_index(i: int, j: int, n: int, n_src: int, npx: int, npy: int):
    """get_neighbor_index: (i, j) on tile n -> (ii, jj) on neighbour n_src."""
    isc, jsc, iec, jec = 1, 1, npx, npy
    nw, ne, ns, nn = _neighbor_tiles(n)
    if n % 2 != 0:
        if n_src == nw:
            return iec - (j - jsc), jec + (i - isc)
        if n_src == ne:
            return isc + (i - iec), jsc + (j - jsc)
        if n_src == ns:
            return isc + (i - isc), jec + (j - jsc)
        if n_src == nn:
            return isc + (j - jec), jec - (i - isc)
    else:
        if n_src == nw:
            return iec + (i - isc), jsc + (j - jsc)
        if n_src == ne:
            return iec - (j - jsc), jsc + (i - iec)
        if n_src == ns:
            return iec + (j - jsc), jec - (i - isc)
        if n_src == nn:
            return isc + (i - isc), jsc + (j - jec)
    raise ValueError("n_src is not a neighbour of n")  # pragma: no cover


def _neighbor_bounds(n: int, n_src: int, npx: int, npy: int, ng: int):
    """get_neighbor_bounds: halo strip of tile n filled from n_src."""
    isc, jsc, iec, jec = 1, 1, npx, npy
    nw, ne, ns, nn = _neighbor_tiles(n)
    if n_src == nw:
        return isc - ng, isc - 1, jsc, jec
    if n_src == ne:
        return iec + 1, iec + ng, jsc, jec
    if n_src == ns:
        return isc, iec, jsc - ng, jsc - 1
    if n_src == nn:
        return isc, iec, jec + 1, jec + ng
    raise ValueError("n_src is not a neighbour of n")  # pragma: no cover


def compute_fv3_native_k2e(res: int, remap_ng: int = 3,
                           k2e_nord: int = 4) -> dict:
    """Exact duo-grid k2e remap tables for the ED cubed sphere.

    Parameters
    ----------
    res : int
        Cells per face edge (C``res``).
    remap_ng : int
        Number of halo remap rings (upstream ``gg%ng - 2``; driver uses 3).
    k2e_nord : int
        Lagrange stencil width (upstream struct default 4).

    Returns
    -------
    dict with, per stagger in ``("A","B","CX","CY","DX","DY")``:
        ``<S>_ij``   (m, 2) int 1-based Fortran (i, j) record keys,
        ``<S>_loc``  (m,)   int 1-based ``klo`` window anchors,
        ``<S>_coef`` (m, k2e_nord) Lagrange weights,
    sorted by (i, j) — the exact record set the Fortran oracle emits
    (tables are tile- and side-symmetric; tile 1 stored).
    """
    if k2e_nord not in (2, 4):
        raise NotImplementedError(
            f"k2e_nord={k2e_nord}: supported orders are 2 (the "
            "AUTHORITATIVE live default — Zenodo fv_arrays.F90:150/"
            "global_grid_data.F90:57, no nml override) and 4 (the "
            "luanfs-mirror monolith order the historical fixtures "
            "pinned; 2026-07-27 root cause: its corner-adjacent "
            "extrapolation lobes are the vertex amplifier)")
    gg_ng = remap_ng + 2

    # ---- supergrid 1-D line (gen_lonlat_equal_edge, grid_type == 0) ----
    sg_is, sg_ie = 1, 2 * res + 1
    sg_nc = 1 + res          # panel-centre index (gen_coords reuses it)
    sg_ng = 2 * gg_ng
    sg_isd, sg_ied = sg_is - sg_ng, sg_ie + sg_ng

    line = _ed_line(res, sg_ng)

    # ---- six cube-face embeddings -> pt_ext latitudes ----
    idx = np.arange(sg_isd, sg_ied + 1)
    lv = np.array([line[j] for j in idx])
    X, Y = np.meshgrid(lv, lv, indexing="ij")
    carts = _ED_CARTS
    npts = sg_ied - sg_isd + 1
    lat_ext = np.full((6, npts, npts), np.nan)
    for t in range(6):
        cx, cy, cz = carts[t](X, Y)
        r = np.sqrt(cx * cx + cy * cy + cz * cz)
        lat_ext[t] = np.arcsin(cz / r)

    def ext_lat(t, i, j):
        return lat_ext[t - 1, i - sg_isd, j - sg_isd]

    # ---- pt_kik latitudes (update_kik neighbour maps, supergrid space) ----
    # Default -999.0 exactly as upstream (pt_kik(:,:,:,:) = -999.): a handful
    # of extreme corner-window records legitimately reach an UNFILLED
    # position (e.g. supergrid (26, -5) at C12), where upstream's mirror
    # turns -999 into +999 and the Lagrange window carries that node with an
    # ~1e-11 weight.  Replicating the sentinel — not NaN — reproduces the
    # oracle's exact coefficients for those records.
    kik = np.full((6, npts, npts), -999.0)
    ss = sg_ng  # halo width used by update_kik (its `ng` arg)
    # inner copy: is-2 .. ie+2
    for t in range(1, 7):
        sl = slice(sg_is - 2 - sg_isd, sg_ie + 2 - sg_isd + 1)
        kik[t - 1][sl, sl] = lat_ext[t - 1][sl, sl]
    for t in range(1, 7):
        nw, ne, ns, nn = _neighbor_tiles(t)
        for n_src in (nw, ne, ns, nn):
            i0, i1, j0, j1 = _neighbor_bounds(t, n_src, sg_ie, sg_ie, ss)
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    ii, jj = _neighbor_index(i, j, t, n_src, sg_ie, sg_ie)
                    kik[t - 1][i - sg_isd, j - sg_isd] = \
                        lat_ext[n_src - 1][ii - sg_isd, jj - sg_isd]

    def kik_lat(t, i, j):
        return kik[t - 1, i - sg_isd, j - sg_isd]

    # ---- gen_coords: per-layer 1-D remap coordinates (tile 1) ----
    # ext_x/kik_x supergrid tables, filled exactly where gen_coords fills
    # them (rows/cols at layer offsets); everything else stays -999.
    ext_x = _F(sg_isd, sg_ied, (npts,))   # ext_x[i][j-index]
    kik_x = _F(sg_isd, sg_ied, (npts,))

    def _set(table, a, b, v):
        table[a][b - sg_isd] = v

    def _get(table, a, b):
        return table[a][b - sg_isd]

    # gen_coords uses its OWN ng = 2*gg%ng (supergrid halo width) for the
    # layer loop — layers run to is-2*gg_ng (verified against the raw
    # kik/ext dumps: layers -5, -6, ... are filled).
    for k in range(0, sg_ng + 1):
        # kik layer line
        kl = _F(sg_isd, sg_ied)
        kl[sg_nc] = 0.0
        for j in range(sg_is - 1, sg_nc + 1):
            kl[j] = kik_lat(1, sg_is - k, j) - kik_lat(1, sg_is - k, sg_nc)
            kl[sg_ie - j + 1] = -kl[j]
        for j in range(sg_is - 1, sg_ie + 2):
            for i_layer in (sg_is - k, sg_ie + k):
                _set(kik_x, j, i_layer, kl[j])   # kik_x(j, i) = line(j)
        # ext layer line
        el = _F(sg_isd, sg_ied)
        el[sg_nc] = 0.0
        for j in range(sg_isd, sg_nc + 1):
            el[j] = ext_lat(1, sg_is - k, j) - ext_lat(1, sg_is - k, sg_nc)
            el[sg_ie - j + 1] = -el[j]
        for j in range(sg_isd, sg_ied + 1):
            for i_layer in (sg_is - k, sg_ie + k):
                _set(ext_x, j, i_layer, el[j])

    # ---- gen_k2e per stagger ----
    is_, ie_ = 1, res
    ng = remap_ng
    npd = k2e_nord // 2 - 1
    isd_, ied_ = is_ - ng, ie_ + ng

    # parity samplers: (ii, jj) supergrid indices for stagger point (i, j)
    par = {
        "A": lambda i, j: (2 * i, 2 * j),
        "B": lambda i, j: (2 * i - 1, 2 * j - 1),
        "C": lambda i, j: (2 * i - 1, 2 * j),
        "D": lambda i, j: (2 * i, 2 * j - 1),
    }

    def sample_x(table, stag, i, j):
        """gg%{kik,ext}_x(ii, jj): position ii on layer jj."""
        ii, jj = par[stag](i, j)
        return _get(table, ii, jj)

    def sample_y(table, stag, i, j):
        """gg%{kik,ext}_y(ii, jj) == _x(jj, ii): position jj on layer ii
        (gen_coords stores the y tables layer-first — the transpose)."""
        ii, jj = par[stag](i, j)
        return _get(table, jj, ii)

    out = {}

    def _run(stag_key, parity, x_hi_off, y_hi_off, row_calls, col_calls):
        """One stagger's get_loc_x/get_loc_y pair.

        x_hi_off: +1 when the x (row-direction) 1-D problem has ie+1 sources
        (B, CX, DX); y_hi_off likewise for the column problem (B, CY, DY).
        row_calls/col_calls: the halo row/col target lines per upstream loop.
        """
        recs = {}
        xe = ie_ + x_hi_off
        ye = ie_ + y_hi_off
        for iiw in range(1, ng + 1):
            for j in row_calls(iiw):
                x = np.array([sample_x(kik_x, parity, i, j)
                              for i in range(is_, xe + 1)])
                y = {i: sample_x(ext_x, parity, i, j)
                     for i in range(isd_, ied_ + x_hi_off + 1)}
                for i in range(is_ - iiw + 1, xe + iiw - 1 + 1):
                    yy = y[i]
                    klo, khi = is_, xe
                    while khi - klo > 1:
                        k = (khi + klo) // 2
                        if x[k - is_] > yy:
                            khi = k
                        else:
                            klo = k
                    klo = max(klo, is_ + npd)
                    klo = min(klo, xe - npd - 1)
                    khi = klo + 1
                    coef = _lagrange_coef(
                        yy, x[klo - npd - is_: khi + npd - is_ + 1])
                    recs[(i, j)] = (klo, coef)
            for i in col_calls(iiw):
                x = np.array([sample_y(kik_x, parity, i, j)
                              for j in range(is_, ye + 1)])
                y = {j: sample_y(ext_x, parity, i, j)
                     for j in range(isd_, ied_ + y_hi_off + 1)}
                for j in range(is_ - iiw + 1, ye + iiw - 1 + 1):
                    yy = y[j]
                    klo, khi = is_, ye
                    while khi - klo > 1:
                        k = (khi + klo) // 2
                        if x[k - is_] > yy:
                            khi = k
                        else:
                            klo = k
                    klo = max(klo, is_ + npd)
                    klo = min(klo, ye - npd - 1)
                    khi = klo + 1
                    coef = _lagrange_coef(
                        yy, x[klo - npd - is_: khi + npd - is_ + 1])
                    recs[(i, j)] = (klo, coef)
        keys = sorted(recs)
        out[f"{stag_key}_ij"] = np.array(keys, dtype=np.int64)
        out[f"{stag_key}_loc"] = np.array(
            [recs[k][0] for k in keys], dtype=np.int64)
        out[f"{stag_key}_coef"] = np.stack([recs[k][1] for k in keys])

    # A: unstaggered rows/cols; halo rows j = 1-iiw and ie+iiw
    _run("A", "A", 0, 0,
         lambda w: (is_ - w, ie_ + w),
         lambda w: (is_ - w, ie_ + w))
    # B: both directions have ie+1 sources; halo lines at 1-iiw / ie+1+iiw
    _run("B", "B", 1, 1,
         lambda w: (is_ - w, ie_ + 1 + w),
         lambda w: (is_ - w, ie_ + 1 + w))
    # CX: x problem has ie+1 sources (rows at 1-iiw / ie+iiw);
    #     y problem plain (cols at 1-iiw / ie+1+iiw)
    _run("CX", "C", 1, 0,
         lambda w: (is_ - w, ie_ + w),
         lambda w: (is_ - w, ie_ + 1 + w))
    # CY: x plain (rows at 1-iiw / ie+1+iiw); y has ie+1 sources
    _run("CY", "C", 0, 1,
         lambda w: (is_ - w, ie_ + 1 + w),
         lambda w: (is_ - w, ie_ + w))
    # D staggers mirror C with the D parity
    _run("DX", "D", 1, 0,
         lambda w: (is_ - w, ie_ + w),
         lambda w: (is_ - w, ie_ + 1 + w))
    _run("DY", "D", 0, 1,
         lambda w: (is_ - w, ie_ + 1 + w),
         lambda w: (is_ - w, ie_ + w))

    out["k2e_nord"] = k2e_nord
    out["remap_ng"] = remap_ng
    return out


# --------------------------------------------------------------------------
# ED-native DuoGridData (phase 3b): the certified k2e tables + ED extended
# grids, packaged for legoESM's existing duogrid consumption machinery
# (cube_rmp_vectorized, corner Lagrange fill, a2stag vectors).
# --------------------------------------------------------------------------
def _ed_ext_agrid_lonlat(n: int, ng: int):
    """ED extended A-grid lon/lat (create layout), shape (6, n+2ng, n+2ng).

    A-grid extended positions are the EVEN supergrid nodes of the duo-grid
    line construction (own-face gnomonic extension) — the exact ``pt_ext``
    A-points the reference uses, remapped to create's face layout.
    """
    from legoesm.grids.cubed_sphere import (
        GNOMONIC_ED_FACE_PERM as _GNOMONIC_ED_FACE_PERM,
        GNOMONIC_ED_FACE_ROT as _GNOMONIC_ED_FACE_ROT,
    )

    # supergrid ghost width: need A-points to i = n+ng -> supergrid 2(n+ng),
    # i.e. sg_ie + 2*ng - 1; build with the reference's own sg_ng = 2*(ng+2)
    # ghost layers, which covers every ng <= 3 use.
    line = _ed_line(n, 2 * (ng + 2))

    # A-point 1-D values: supergrid even indices 2i for i = 1-ng .. n+ng
    ai = np.arange(1 - ng, n + ng + 1)
    av = np.array([line[2 * i] for i in ai])
    X, Y = np.meshgrid(av, av, indexing="ij")
    carts = _ED_CARTS
    m = len(ai)
    lon6 = np.zeros((6, m, m))
    lat6 = np.zeros((6, m, m))
    for t in range(6):
        cx, cy, cz = carts[t](X, Y)
        r = np.sqrt(cx * cx + cy * cy + cz * cz)
        lon6[t] = np.mod(np.arctan2(cy, cx), 2.0 * np.pi)
        lat6[t] = np.arcsin(cz / r)
    # remap FV3 face numbering -> create layout
    lon_c = np.stack([
        np.rot90(lon6[_GNOMONIC_ED_FACE_PERM[F]], _GNOMONIC_ED_FACE_ROT[F])
        for F in range(6)
    ])
    lat_c = np.stack([
        np.rot90(lat6[_GNOMONIC_ED_FACE_PERM[F]], _GNOMONIC_ED_FACE_ROT[F])
        for F in range(6)
    ])
    return lon_c, lat_c


def _ed_ext_stagger_lonlat(n: int, ng: int, parity: str):
    """ED extended lon/lat at a supergrid parity, create layout.

    parity "A": even nodes (2i, 2j), shape (6, n+2ng, n+2ng);
    parity "B": odd nodes (2i-1, 2j-1), shape (6, n+2ng+1, n+2ng+1).
    """
    from legoesm.grids.cubed_sphere import (
        GNOMONIC_ED_FACE_PERM as _GNOMONIC_ED_FACE_PERM,
        GNOMONIC_ED_FACE_ROT as _GNOMONIC_ED_FACE_ROT,
    )

    line = _ed_line(n, 2 * (ng + 2))

    if parity == "A":
        idx = np.arange(1 - ng, n + ng + 1)
        vals = np.array([line[2 * i] for i in idx])
    elif parity == "B":
        idx = np.arange(1 - ng, n + ng + 2)
        vals = np.array([line[2 * i - 1] for i in idx])
    else:  # pragma: no cover - guard
        raise ValueError(parity)
    X, Y = np.meshgrid(vals, vals, indexing="ij")
    carts = _ED_CARTS
    m = len(idx)
    lon6 = np.zeros((6, m, m))
    lat6 = np.zeros((6, m, m))
    for t in range(6):
        cx, cy, cz = carts[t](X, Y)
        r = np.sqrt(cx * cx + cy * cy + cz * cz)
        lon6[t] = np.mod(np.arctan2(cy, cx), 2.0 * np.pi)
        lat6[t] = np.arcsin(cz / r)
    lon_c = np.stack([
        np.rot90(lon6[_GNOMONIC_ED_FACE_PERM[F]], _GNOMONIC_ED_FACE_ROT[F])
        for F in range(6)
    ])
    lat_c = np.stack([
        np.rot90(lat6[_GNOMONIC_ED_FACE_PERM[F]], _GNOMONIC_ED_FACE_ROT[F])
        for F in range(6)
    ])
    return lon_c, lat_c


def _compute_ext_vectors_native(a_lon, a_lat, b_lon, b_lat):
    """a2stag_metrics with REAL staggered B points (fv_duogrid:2871-2992).

    The legacy builder synthesizes B-grid points from A-point averages;
    upstream uses the actual odd-supergrid ``dg%b_pt`` values (codex p3 P1).
    Shapes follow legoESM's DuoGridData conventions:
    ew (6, m+1, m, 3, 2), es (6, m, m+1, 3, 2), vlon/vlat (6, m, m, 3)
    where m = n + 2*ng.  Boundary lines outside the upstream loop ranges
    (ew i=0, es j=0) replicate their inner neighbour (upstream leaves them
    undefined-and-unused).
    """
    def xyz(lon, lat):
        return np.stack([np.cos(lat) * np.cos(lon),
                         np.cos(lat) * np.sin(lon),
                         np.sin(lat)], axis=-1)

    def norm(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    A = xyz(a_lon, a_lat)                     # (6, m, m, 3)
    B = xyz(b_lon, b_lat)                     # (6, m+1, m+1, 3)
    nf, m, _, _ = A.shape

    # ew at u-positions (i between A(i-1) and A(i)), i = 1..m-1 upstream
    ppw = norm(B[:, 1:-1, :-1] + B[:, 1:-1, 1:])   # mid(B(i,j), B(i,j+1)), (6, m-1, m)
    p2 = np.cross(A[:, :-1], A[:, 1:])             # cross(A(i-1,j), A(i,j)), (6, m-1, m)
    ew1 = norm(np.cross(p2, ppw))
    gb = np.cross(B[:, 1:-1, :-1], B[:, 1:-1, 1:])  # cross(B(i,j), B(i,j+1))
    ew2 = norm(np.cross(gb, ppw))
    ew = np.zeros((nf, m + 1, m, 3, 2))
    ew[:, 1:m, :, :, 0] = ew1
    ew[:, 1:m, :, :, 1] = ew2
    ew[:, 0] = ew[:, 1]
    ew[:, m] = ew[:, m - 1]

    # es at v-positions (j between A(j-1) and A(j)), j = 1..m-1 upstream
    pps = norm(B[:, :-1, 1:-1] + B[:, 1:, 1:-1])   # mid(B(i,j), B(i+1,j))
    p2s = np.cross(A[:, :, :-1], A[:, :, 1:])      # cross(A(i,j-1), A(i,j))
    es2 = norm(np.cross(p2s, pps))
    gbs = np.cross(B[:, :-1, 1:-1], B[:, 1:, 1:-1])
    es1 = norm(np.cross(gbs, pps))
    es = np.zeros((nf, m, m + 1, 3, 2))
    es[:, :, 1:m, :, 0] = es1
    es[:, :, 1:m, :, 1] = es2
    es[:, :, 0] = es[:, :, 1]
    es[:, :, m] = es[:, :, m - 1]

    # vlon/vlat: geographic unit vectors at A points (unit_vect_latlon)
    sin_lon, cos_lon = np.sin(a_lon), np.cos(a_lon)
    sin_lat, cos_lat = np.sin(a_lat), np.cos(a_lat)
    vlon = np.stack([-sin_lon, cos_lon, np.zeros_like(a_lon)], axis=-1)
    vlat = np.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                    axis=-1)
    return vlon, vlat, ew, es


def create_fv3_native_duogrid_data(n: int, ng: int = 3, k2e_nord: int = 4,
                                   corner_lagrange: bool = False):
    """ED-native DuoGridData: certified duo-grid k2e tables + ED extension.

    Drop-in replacement for :func:`legoesm.grids.duogrid.create_duogrid_data`
    on ED-provenance grids: the edge remap coefficients come from the
    oracle-pinned :func:`compute_fv3_native_k2e` A-grid tables (the legacy
    builder's equiangular extension lines are the WRONG interpolants on ED —
    measured ~4e-2 coefficient error), and the extended-grid lon/lat (and
    everything derived from them: corner Lagrange fill, a2stag vectors) use
    the ED gnomonic extension.  Fails loudly on unsupported widths — never
    approximates.
    """
    from legoesm.grids.duogrid import (
        DuoGridData,
        MAX_K2E_NORD,
        compute_corner_lagrange_coeff as _compute_corner_lagrange_coeff,
    )
    from legoesm.grids.halo import EAST, NORTH, SOUTH, WEST
    import jax.numpy as jnp

    if k2e_nord not in (2, 4):
        raise NotImplementedError(
            f"fv3-native duogrid: k2e_nord={k2e_nord}; supported are 2 "
            "(authoritative live default) and 4 (historical fixture "
            "order) — see compute_fv3_native_k2e")
    if ng not in (1, 2, 3):
        raise NotImplementedError(
            f"fv3-native duogrid: ng={ng} unsupported (oracle-pinned remap "
            "rings are 1..3); refusing to approximate")
    if ng > n // 2:
        raise ValueError(f"ng={ng} too large for n={n} (need ng <= n//2)")

    tab = compute_fv3_native_k2e(n, remap_ng=3, k2e_nord=k2e_nord)
    ij = tab["A_ij"]
    loc = tab["A_loc"]
    coef = tab["A_coef"]
    rec = {tuple(k): (int(l), c) for k, l, c in zip(ij, loc, coef)}

    npd = k2e_nord // 2 - 1
    k2e_coef = np.zeros((6, 4, ng, n, MAX_K2E_NORD))
    k2e_lo = np.zeros((6, 4, ng, n), dtype=np.int32)
    for d in range(ng):
        for t in range(1, n + 1):
            # rows: south j = -d, north j = n+1+d; cols: west i = -d,
            # east i = n+1+d (1-based record keys; targets 1..n)
            for edge, key in ((SOUTH, (t, -d)), (NORTH, (t, n + 1 + d)),
                              (WEST, (-d, t)), (EAST, (n + 1 + d, t))):
                l, c = rec[key]
                # 0-based stencil start into the interior strip:
                # window klo-np .. klo+1+np -> start = klo - np - 1
                k2e_coef[:, edge, d, t - 1, :k2e_nord] = c
                k2e_lo[:, edge, d, t - 1] = l - npd - 1

    s = k2e_coef.sum(axis=-1)
    if np.abs(s - 1.0).max() > 1e-10:
        raise AssertionError(
            "fv3-native duogrid: partition of unity violated "
            f"({np.abs(s - 1.0).max():.2e})")

    ext_lon, ext_lat = _ed_ext_agrid_lonlat(n, ng)
    if corner_lagrange:
        # EXPERIMENTAL (codex p3 r2): the corner Lagrange APPLICATION is
        # order-ported (iter-803) and functionally convergent, but its
        # VALUES are not yet pinned to a dg-level Fortran oracle (staged
        # with the phase-4 c_sw harness).  Until then the DEFAULT path
        # uses the pre-existing averaging corner fill (corner_xp=None ->
        # _fill_corner_region_averaging), whose behavior is legacy-
        # validated; opt in here for the Lagrange corners.
        xp, xm, yp, ym = _compute_corner_lagrange_coeff(
            n, ng, ext_lon, ext_lat)
    else:
        xp = xm = yp = ym = None
    # codex p3 P1: a2stag vectors from REAL odd-supergrid B points (the
    # legacy builder synthesizes B from A-averages; upstream uses dg%b_pt).
    b_lon, b_lat = _ed_ext_stagger_lonlat(n, ng, "B")
    vlon_ext, vlat_ext, ew_ext, es_ext = _compute_ext_vectors_native(
        ext_lon, ext_lat, b_lon, b_lat)

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


# Public promotions (CLAUDE.md cross-module private-import ratchet):
# these symbols are imported by sibling modules; expose a public alias
# so importers use the sanctioned public name (definitions keep the
# original underscore name for in-module callers).
ED_CARTS = _ED_CARTS
ed_line = _ed_line
lagrange_coef = _lagrange_coef
compute_ext_vectors_native = _compute_ext_vectors_native
