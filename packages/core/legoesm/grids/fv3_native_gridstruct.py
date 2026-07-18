"""Single-tile FV3 gridstruct with mpp-equivalent (kinked) halos — phase 4.

Builds every ``gridstruct`` field the FV3 ``c_sw``/``d_sw`` call tree
consumes, on ONE whole-face tile (reference numbering, tile 1) with
``ng``-deep halos filled the way FV3's own init + mpp exchanges fill them:

- **side-strip halos** hold the NEIGHBOUR face's values.  On the cubed
  sphere a neighbour's stored metric at a shared physical location equals
  the same geometric formula evaluated on the *kinked* lattice (the
  neighbour's own grid nodes continued into this tile's halo), because an
  ``mpp_update_domains`` SCALAR_PAIR/vector exchange is an index copy of
  the neighbour's storage.  The kinked node positions come from the
  certified phase-3 neighbour maps (``neighbor_tiles``/``neighbor_index``)
  applied to the exact ED supergrid.
- **corner-diagonal regions** have no neighbour; FV3 leaves them at
  ``fill_ghost`` garbage (``big_number``/``tiny_number``) except for the
  explicitly patched ``sin_sg``/``cos_sg`` transport slots
  (fv_grid_utils.F90's "For transport operation" blocks) — replicated
  verbatim here, second (post-``fill_ghost``) version last, exactly as
  upstream's call order leaves them.
- **compute domains** are taken from the certified phase-1/2 builders
  (``compute_fv3_native_metrics``/``compute_fv3_native_angles`` formula
  set) and the halo constructions are self-verified against them on the
  overlap.

The corner-diagonal regions of the metric fields carry FV3's own
``fill_corners`` ghost values (ported r8 index maps below), the grid and
agrid corner regions carry the upstream mirrored geometry, and the sg /
cosa pipelines therefore see upstream-exact inputs everywhere (codex r1
P1-2).  Remaining known-junk slots: the ``cosa_u`` isd/ied+1 columns and
``rsina`` borders match upstream's uninitialized choices exactly; the
``divg_u``/``divg_v`` outermost rows/cols hold big_number-derived junk
that upstream instead overwrites via its final mpp exchange — a
DOCUMENTED, NOT-UPSTREAM-EQUAL region.  Safe because d_sw's divergence
damping loops run ``nt <= nord <= ng-1`` deep and never reach the
outermost row/col (codex r2 finding 5; tripwire-tested).

Fortran index convention: arrays are plain numpy with row 0 == Fortran
``isd = 1-ng`` (cell/lower-node axes) — helper ``fort`` views give
Fortran-indexed access for the verbatim patch loops.
"""

from __future__ import annotations

import numpy as np
from legoesm.grids.fv3_native_halos import (
    ed_supergrid_lonlat_ref,
    neighbor_index,
    neighbor_tiles,
)
from legoesm.grids.fv3_native_metrics import (
    cell_center2,
    compute_fv3_native_metrics,
    get_area_quad,
    great_circle_dist,
    latlon2xyz,
    mid_pt_sphere,
    sg_window_fields,
)

from legoesm import constants

# --- upstream fill/sentinel constants (fv_grid_utils.F90) ---
BIG_NUMBER = 1.0e8       # fv_grid_utils big_number
TINY_NUMBER = 1.0e-8     # fv_grid_utils tiny_number (rsin floors + sin_sg ghost)

# --- FV3/FMS physical constants for oracle pinning (FMS constants_mod,
#     GFDL flavour); legoESM production paths use legoesm.constants ---
# const-ok: FMS RADIUS differs from legoESM R_earth; oracle pins upstream's
FV3_RADIUS_M = 6371.0e3
FV3_OMEGA = constants.Omega  # FMS OMEGA equals legoESM's rotation rate


class fort:
    """Fortran-indexed 2-D/3-D view over a numpy array (lo bounds given)."""

    def __init__(self, a: np.ndarray, ilo: int, jlo: int):
        self.a = a
        self.ilo = ilo
        self.jlo = jlo

    def __getitem__(self, idx):
        i, j, *k = idx
        return self.a[(i - self.ilo, j - self.jlo, *k)]

    def __setitem__(self, idx, v):
        i, j, *k = idx
        self.a[(i - self.ilo, j - self.jlo, *k)] = v


def rsin_border_override(s: np.ndarray) -> np.ndarray:
    """1/SIGN(max(tiny,|s|), s) — the fv_grid_utils panel-border rsin rule.

    np.copysign is Fortran SIGN exactly, including BOTH signed zeros
    (gfortran default -fsign-zero makes SIGN(x, -0.0) negative; np.sign
    and ``s >= 0`` selections each get one of the zeros wrong).
    """
    return 1.0 / np.copysign(np.maximum(TINY_NUMBER, np.abs(s)), s)


def build_tile1_kinked_corner_lonlat(n: int, ng: int = 3,
                                     sentinel: float = BIG_NUMBER):
    """Tile-1 convenience wrapper over ``build_kinked_corner_lonlat``."""
    return build_kinked_corner_lonlat(n, ng, tile=1, sentinel=sentinel)


def build_kinked_corner_lonlat(n: int, ng: int = 3, *, tile: int = 1,
                               sentinel: float = BIG_NUMBER):
    """Kinked (mpp-equivalent) corner-node lon/lat for one reference tile.

    Returns ``lon, lat`` of shape (n+2ng+1, n+2ng+1) over Fortran node
    indices ``1-ng .. n+1+ng``: interior from the tile's own ED supergrid,
    side strips sampled from the neighbour faces via the certified
    neighbour maps (== what the mpp corner-position exchange delivers),
    corner-diagonal regions at ``sentinel``.
    """
    if ng < 1 or 2 * ng >= n:
        raise ValueError(f"ng={ng} unsupported for n={n}")
    lon6, lat6 = ed_supergrid_lonlat_ref(n)   # (6, 2n+1, 2n+1), sg index 1..2n+1
    sg_npx = 2 * n + 1
    m = n + 2 * ng + 1
    lon = np.full((m, m), sentinel, dtype=np.float64)
    lat = np.full((m, m), sentinel, dtype=np.float64)

    def put(i_node: int, j_node: int, lo, la):
        # Fortran node index -> np row: i_node - (1 - ng)
        lon[i_node - 1 + ng, j_node - 1 + ng] = lo
        lat[i_node - 1 + ng, j_node - 1 + ng] = la

    # interior (the tile's own supergrid odd-odd nodes)
    for i_node in range(1, n + 2):
        for j_node in range(1, n + 2):
            si, sj = 2 * i_node - 1, 2 * j_node - 1
            put(i_node, j_node, lon6[tile - 1][si - 1, sj - 1],
                lat6[tile - 1][si - 1, sj - 1])

    # side strips from neighbours (supergrid index space)
    nw, ne, ns, nn = neighbor_tiles(tile)
    strips = (
        (nw, range(1 - 2 * ng, 0 + 1), range(1, sg_npx + 1)),
        (ne, range(sg_npx + 1, sg_npx + 2 * ng + 1), range(1, sg_npx + 1)),
        (ns, range(1, sg_npx + 1), range(1 - 2 * ng, 0 + 1)),
        (nn, range(1, sg_npx + 1), range(sg_npx + 1, sg_npx + 2 * ng + 1)),
    )
    for n_src, si_range, sj_range in strips:
        for si in si_range:
            if si % 2 == 0:
                continue
            for sj in sj_range:
                if sj % 2 == 0:
                    continue
                ii_s, jj_s = neighbor_index(si, sj, tile, n_src,
                                            sg_npx, sg_npx)
                put((si + 1) // 2, (sj + 1) // 2,
                    lon6[n_src - 1][ii_s - 1, jj_s - 1],
                    lat6[n_src - 1][ii_s - 1, jj_s - 1])
    return lon, lat


def build_extended_corner_lonlat(n: int, ng: int = 3, *, tile: int = 1):
    """EXTENDED-lattice corner-node lon/lat for one reference tile.

    The duo grid continues each face's own gnomonic coordinate lines
    beyond the panel edge (fv_duogrid's pt_ext construction): every
    node — side halos AND corner wedges — is the face's own smooth
    extension, no neighbour copies, no sentinels.  Shape and index
    convention match :func:`build_kinked_corner_lonlat`
    ((n+2ng+1, n+2ng+1) over Fortran nodes 1-ng..n+1+ng); the interior
    nodes (1..n+1) are BITWISE identical to the kinked builder's (same
    ED line, same carts).
    """
    from legoesm.grids.fv3_native_halos import _ED_CARTS, _ed_line

    line = _ed_line(n, 2 * (ng + 2))
    idx = np.arange(1 - ng, n + 1 + ng + 1)
    vals = np.array([line[2 * i - 1] for i in idx])
    X, Y = np.meshgrid(vals, vals, indexing="ij")
    cx, cy, cz = _ED_CARTS[tile - 1](X, Y)
    r = np.sqrt(cx * cx + cy * cy + cz * cz)
    lon = np.mod(np.arctan2(cy, cx), 2.0 * np.pi)
    lat = np.arcsin(cz / r)
    return lon, lat


def extend_gridstruct(gs: dict, n: int, ng: int, *, tile: int = 1,
                      radius: float = FV3_RADIUS_M) -> dict:
    """EXTENDED-lattice gridstruct for the duo consistency bundle.

    Runs the certified metric + angle engines on the face's own
    gnomonic EXTENSION treated as one big face (size n+2ng == the data
    domain, so every engine output maps 1:1 onto the gridstruct
    layouts), then RESTORES the kinked builder's interior — so the
    compute domain keeps the certified upstream conventions (including
    the divg/del6 seam-row specials) BITWISE, while every halo and
    corner-wedge cell carries real smooth extended-grid metrics (the
    duo semantics: no neighbour copies, no fill_ghost poison).

    KNOWN JUNK REGION (documented): the OUTERMOST ext ring — the
    engines apply their face-edge special-casing at the big-face
    boundary, which is not a real panel edge here.  Consumers must stay
    >=1 ring inside (the ng-deep stencils of the stepper do).
    """
    from legoesm.grids.fv3_native_metrics import (
        compute_fv3_native_metrics,
    )

    n_big = n + 2 * ng
    lon, lat = build_extended_corner_lonlat(n, ng, tile=tile)
    lon6 = np.stack([lon] * 6)
    lat6 = np.stack([lat] * 6)
    mets = compute_fv3_native_metrics(lon6, lat6, radius)

    out = dict(gs)

    def take(key):
        return np.array(np.asarray(mets[key])[0], dtype=np.float64)

    ext = {
        "area": take("area"), "dx": take("dx"), "dy": take("dy"),
        "dxa": take("dxa"), "dya": take("dya"),
        "dxc": take("dxc"), "dyc": take("dyc"),
        "area_c": take("area_c"),
        "agrid_lon": take("agrid_lon"), "agrid_lat": take("agrid_lat"),
    }
    ext["rarea"] = 1.0 / ext["area"]
    ext["rarea_c"] = 1.0 / ext["area_c"]
    ext["rdx"] = 1.0 / ext["dx"]
    ext["rdy"] = 1.0 / ext["dy"]
    ext["rdxa"] = 1.0 / ext["dxa"]
    ext["rdya"] = 1.0 / ext["dya"]
    ext["rdxc"] = 1.0 / ext["dxc"]
    ext["rdyc"] = 1.0 / ext["dyc"]

    # Angle families from LOCAL finite-difference tangents on the
    # extended lattice (self-contained differential geometry; the
    # angles engine's cross-face halo reconstruction is only valid
    # with real cube neighbours, which the big-face treatment does not
    # provide — its outer rings were NaN/garbage exactly where the
    # halos live).  2nd-order central tangents; interior is restored
    # from the certified kinked builder below regardless.
    def _xyz(lo, la):
        return np.stack([np.cos(la) * np.cos(lo),
                         np.cos(la) * np.sin(lo),
                         np.sin(la)], axis=-1)

    pb = _xyz(lon, lat)                       # corner nodes (m_b, m_b, 3)

    def _unit_tangents(p):
        e1 = np.empty_like(p)
        e2 = np.empty_like(p)
        e1[1:-1] = p[2:] - p[:-2]
        e1[0] = p[1] - p[0]
        e1[-1] = p[-1] - p[-2]
        e2[:, 1:-1] = p[:, 2:] - p[:, :-2]
        e2[:, 0] = p[:, 1] - p[:, 0]
        e2[:, -1] = p[:, -1] - p[:, -2]
        for e in (e1, e2):
            e -= (e * p).sum(-1, keepdims=True) * p
            e /= np.linalg.norm(e, axis=-1, keepdims=True)
        return e1, e2

    def _cos_sin(p):
        e1, e2 = _unit_tangents(p)
        c = (e1 * e2).sum(-1)
        s = np.sqrt(np.maximum(TINY_NUMBER ** 2, 1.0 - c * c))
        return c, s

    cb, sb = _cos_sin(pb)                     # B nodes
    ext["cosa"] = cb
    ext["sina"] = sb
    ext["rsina"] = 1.0 / np.maximum(TINY_NUMBER, sb * sb)

    # A centres and edge midpoints from the corner lattice
    pa = _xyz(ext["agrid_lon"], ext["agrid_lat"])
    ca, sa = _cos_sin(pa)
    ext["cosa_s"] = ca
    ext["rsin2"] = 1.0 / np.maximum(TINY_NUMBER, sa * sa)

    mid_u = pb[:, :-1] + pb[:, 1:]            # x-face midpoints (m_b, m_a)
    mid_u /= np.linalg.norm(mid_u, axis=-1, keepdims=True)
    cu, su = _cos_sin(mid_u)
    ext["cosa_u"] = cu
    ext["sina_u"] = su
    ext["rsin_u"] = 1.0 / np.maximum(TINY_NUMBER, su * su)

    mid_v = pb[:-1, :] + pb[1:, :]            # y-face midpoints (m_a, m_b)
    mid_v /= np.linalg.norm(mid_v, axis=-1, keepdims=True)
    cv, sv = _cos_sin(mid_v)
    ext["cosa_v"] = cv
    ext["sina_v"] = sv
    ext["rsin_v"] = 1.0 / np.maximum(TINY_NUMBER, sv * sv)

    # sin/cos_sg 9-slot family: centre slot from the A tangents; the
    # W/S/E/N mid-edge slots from the face-midpoint tangents; corner
    # slots from the B tangents (halo consumers on the duo lane read
    # the centre + mid-edge slots; interior restored below).
    m_a_loc = n_big
    ssg = np.empty((m_a_loc, m_a_loc, 9))
    csg = np.empty((m_a_loc, m_a_loc, 9))
    csg[..., 4] = ca
    ssg[..., 4] = sa
    csg[..., 0] = cu[:-1, :]
    ssg[..., 0] = su[:-1, :]
    csg[..., 2] = cu[1:, :]
    ssg[..., 2] = su[1:, :]
    csg[..., 1] = cv[:, :-1]
    ssg[..., 1] = sv[:, :-1]
    csg[..., 3] = cv[:, 1:]
    ssg[..., 3] = sv[:, 1:]
    csg[..., 5] = cb[:-1, :-1]
    ssg[..., 5] = sb[:-1, :-1]
    csg[..., 6] = cb[1:, :-1]
    ssg[..., 6] = sb[1:, :-1]
    csg[..., 7] = cb[1:, 1:]
    ssg[..., 7] = sb[1:, 1:]
    csg[..., 8] = cb[:-1, 1:]
    ssg[..., 8] = sb[:-1, 1:]
    ext["sin_sg"] = ssg
    ext["cos_sg"] = csg

    # Coriolis at centres/B nodes from the extended geometry
    ext["f0"] = 2.0 * FV3_OMEGA * np.sin(ext["agrid_lat"])
    blat = np.array(lat, dtype=np.float64)
    ext["fC"] = 2.0 * FV3_OMEGA * np.sin(blat)

    # divg/del6: the PLAIN formulas on the extended lattice (the duo
    # grid has real angles everywhere; the plain-path seam specials are
    # restored with the interior below)
    ext["divg_u"] = ext["sina_v"] * ext["dyc"] / ext["dx"]
    ext["del6_u"] = ext["sina_v"] * ext["dx"] / ext["dyc"]
    ext["divg_v"] = ext["sina_u"] * ext["dxc"] / ext["dy"]
    ext["del6_v"] = ext["sina_u"] * ext["dy"] / ext["dxc"]

    ci = slice(ng, ng + n)          # interior cell rows/cols
    bi = slice(ng, ng + n + 1)      # interior node rows/cols
    axmap = {"c": ci, "b": bi}

    def restore(key, axes):
        a = ext[key]
        k0 = gs[key]
        slc = tuple(axmap[ax] for ax in axes)
        a[slc] = k0[slc]
        out[key] = a

    for key, axes in (
        ("area", "cc"), ("rarea", "cc"), ("dxa", "cc"), ("dya", "cc"),
        ("rdxa", "cc"), ("rdya", "cc"), ("agrid_lon", "cc"),
        ("agrid_lat", "cc"), ("cosa_s", "cc"), ("rsin2", "cc"),
        ("f0", "cc"),
        ("dx", "cb"), ("rdx", "cb"), ("dyc", "cb"), ("rdyc", "cb"),
        ("cosa_v", "cb"), ("sina_v", "cb"), ("rsin_v", "cb"),
        ("divg_u", "cb"), ("del6_u", "cb"),
        ("dy", "bc"), ("rdy", "bc"), ("dxc", "bc"), ("rdxc", "bc"),
        ("cosa_u", "bc"), ("sina_u", "bc"), ("rsin_u", "bc"),
        ("divg_v", "bc"), ("del6_v", "bc"),
        ("area_c", "bb"), ("rarea_c", "bb"), ("fC", "bb"),
        ("cosa", "bb"), ("sina", "bb"), ("rsina", "bb"),
    ):
        restore(key, axes)
    a = ext["sin_sg"]
    a[ci, ci, :] = gs["sin_sg"][ci, ci, :]
    out["sin_sg"] = a
    a = ext["cos_sg"]
    a[ci, ci, :] = gs["cos_sg"][ci, ci, :]
    out["cos_sg"] = a

    # grid corner-node lon/lat: the extended lattice itself
    out["grid_lon"] = np.array(lon, dtype=np.float64)
    out["grid_lat"] = np.array(lat, dtype=np.float64)
    gl = out["grid_lon"]
    gt = out["grid_lat"]
    gl[bi, bi] = gs["grid_lon"][bi, bi]
    gt[bi, bi] = gs["grid_lat"][bi, bi]
    return out


# ---------------------------------------------------------------------------
# fill_corners ports (tools/fv_mp_mod.F90 r8 bodies, verbatim index maps).
# These fill the four corner-diagonal ng x ng regions from side-strip values
# exactly as FV3 does after each mpp metric exchange.  All metric-pair calls
# use mySign = +1 (VECTOR fills use -1).
# ---------------------------------------------------------------------------
def _fill_corners_bgrid_x(Q: fort, npx: int, ng: int) -> None:
    npy = npx
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            Q[1 - i, 1 - j] = Q[1 - j, i + 1]
            Q[1 - i, npy + j] = Q[1 - j, npy - i]
            Q[npx + i, 1 - j] = Q[npx + j, i + 1]
            Q[npx + i, npy + j] = Q[npx + j, npy - i]


def _fill_corners_agrid_x(Q: fort, npx: int, ng: int) -> None:
    npy = npx
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            Q[1 - i, 1 - j] = Q[1 - j, i]
            Q[1 - i, npy - 1 + j] = Q[1 - j, npy - i]
            Q[npx - 1 + i, 1 - j] = Q[npx - 1 + j, i]
            Q[npx - 1 + i, npy - 1 + j] = Q[npx - 1 + j, npy - i]


def _fill_corners_agrid_y(Q: fort, npx: int, ng: int) -> None:
    npy = npx
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            Q[1 - j, 1 - i] = Q[i, 1 - j]
            Q[1 - j, npy - 1 + i] = Q[i, npy - 1 + j]
            Q[npx - 1 + j, 1 - i] = Q[npx - i, 1 - j]
            Q[npx - 1 + j, npy - 1 + i] = Q[npx - i, npy - 1 + j]


def _fill_corners_dgrid(X: fort, Y: fort, npx: int, ng: int,
                        sign: float = 1.0) -> None:
    npy = npx
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            X[1 - i, 1 - j] = sign * Y[1 - j, i]
            X[1 - i, npy + j] = Y[1 - j, npy - i]
            X[npx - 1 + i, 1 - j] = Y[npx + j, i]
            X[npx - 1 + i, npy + j] = sign * Y[npx + j, npy - i]
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            Y[1 - i, 1 - j] = sign * X[j, 1 - i]
            Y[1 - i, npy - 1 + j] = X[j, npy + i]
            Y[npx + i, 1 - j] = X[npx - j, 1 - i]
            Y[npx + i, npy - 1 + j] = sign * X[npx - j, npy + i]


def _fill_corners_cgrid(X: fort, Y: fort, npx: int, ng: int,
                        sign: float = 1.0) -> None:
    npy = npx
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            X[1 - i, 1 - j] = Y[j, 1 - i]
            X[1 - i, npy - 1 + j] = sign * Y[j, npy + i]
            X[npx + i, 1 - j] = sign * Y[npx - j, 1 - i]
            X[npx + i, npy - 1 + j] = Y[npx - j, npy + i]
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            Y[1 - i, 1 - j] = X[1 - j, i]
            Y[1 - i, npy + j] = sign * X[1 - j, npy - i]
            Y[npx - 1 + i, 1 - j] = sign * X[npx + j, i]
            Y[npx - 1 + i, npy + j] = X[npx + j, npy - i]


def _fill_corners_agrid_pair(X: fort, Y: fort, npx: int, ng: int,
                             sign: float = 1.0) -> None:
    npy = npx
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            X[1 - i, 1 - j] = sign * Y[1 - j, i]
            X[1 - i, npy - 1 + j] = Y[1 - j, npy - i]
            X[npx - 1 + i, 1 - j] = Y[npx - 1 + j, i]
            X[npx - 1 + i, npy - 1 + j] = sign * Y[npx - 1 + j, npy - i]
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            Y[1 - j, 1 - i] = sign * X[i, 1 - j]
            Y[1 - j, npy - 1 + i] = X[i, npy - 1 + j]
            Y[npx - 1 + j, 1 - i] = X[npx - i, 1 - j]
            Y[npx - 1 + j, npy - 1 + i] = sign * X[npx - i, npy - 1 + j]


def _node_real_mask(n: int, ng: int) -> np.ndarray:
    """True where a corner node exists (interior or side strip)."""
    idx = np.arange(1 - ng, n + 1 + ng + 1)   # Fortran node index per row
    in_face = (idx >= 1) & (idx <= n + 1)
    return in_face[:, None] | in_face[None, :]


def _cell_real_mask(n: int, ng: int) -> np.ndarray:
    """True where a cell exists (interior or side strip)."""
    idx = np.arange(1 - ng, n + ng + 1)
    in_face = (idx >= 1) & (idx <= n)
    return in_face[:, None] | in_face[None, :]


def build_fv3_native_gridstruct(n: int, ng: int = 3, *, tile: int = 1,
                                radius: float = constants.R_earth,
                                omega: float = constants.Omega,
                                rotation_alpha: float = 0.0) -> dict:
    """All c_sw-consumed gridstruct fields, halo-complete, single tile.

    Returns a dict of float64 arrays in Fortran shapes (see module
    docstring for index origin), plus ``n``/``ng``/``npx`` metadata.  Every
    side strip and the whole compute domain hold exact upstream-faithful
    values; corner-diagonal regions replicate ``fill_ghost`` sentinels
    (with the verbatim post-fill sg transport patches applied last).
    """
    npx = n + 1
    m_a = n + 2 * ng          # cell axis length
    m_b = n + 2 * ng + 1      # node axis length
    clo = 1 - ng              # Fortran lower bound, cell axes
    g_lon, g_lat = build_tile1_kinked_corner_lonlat(n, ng)
    node_ok = _node_real_mask(n, ng)
    cell_ok = _cell_real_mask(n, ng)

    # grid corner regions: fill_corners(grid(:,:,1..2), XDir, BGRID)
    # (fv_grid_tools.F90:727-729) — replaces the construction sentinels
    # with FV3's mirrored ghost geometry (codex r1 P1-2).  Built one ring
    # WIDER than requested: the outermost dxc/dyc rows are mpp-filled
    # upstream and need centres one cell beyond the data domain.
    ngw = ng + 1
    g_lon_w, g_lat_w = build_kinked_corner_lonlat(n, ngw, tile=tile)
    wlo = 1 - ngw
    _fill_corners_bgrid_x(fort(g_lon_w, wlo, wlo), npx, ngw)
    _fill_corners_bgrid_x(fort(g_lat_w, wlo, wlo), npx, ngw)
    grid_w = np.stack([g_lon_w, g_lat_w], axis=-1)

    # agrid: local cell_center2, then corner regions from
    # fill_corners(agrid lon XDir / lat YDir, AGRID)  (fv_grid_tools:811-814)
    agrid_w = cell_center2(grid_w[:-1, :-1], grid_w[1:, :-1],
                           grid_w[:-1, 1:], grid_w[1:, 1:])
    aw_lon = np.ascontiguousarray(agrid_w[..., 0])
    aw_lat = np.ascontiguousarray(agrid_w[..., 1])
    _fill_corners_agrid_x(fort(aw_lon, wlo, wlo), npx, ngw)
    _fill_corners_agrid_y(fort(aw_lat, wlo, wlo), npx, ngw)
    agrid_w = np.stack([aw_lon, aw_lat], axis=-1)

    g_lon = g_lon_w[1:-1, 1:-1]
    g_lat = g_lat_w[1:-1, 1:-1]
    grid_ll = grid_w[1:-1, 1:-1]
    agrid_ll = agrid_w[1:-1, 1:-1]

    # ---- certified compute-domain metrics (6-face, reference layout) ----
    lon6, lat6 = ed_supergrid_lonlat_ref(n)
    sgc = 2 * np.arange(1, n + 2) - 1
    corner6_lon = lon6[:, sgc - 1][:, :, sgc - 1]
    corner6_lat = lat6[:, sgc - 1][:, :, sgc - 1]
    cert = compute_fv3_native_metrics(corner6_lon, corner6_lat, radius)

    # ---- halo-complete length/area fields (post-mpp state) ----
    def gcd_scaled(p, q):
        return great_circle_dist(p, q) * radius

    # dx/dy: strips are the mpp copies (kinked formula); corner regions
    # from fill_corners(dx, dy, DGRID)  (fv_grid_tools:781-784)
    dx = gcd_scaled(grid_ll[:-1, :], grid_ll[1:, :])
    dy = gcd_scaled(grid_ll[:, :-1], grid_ll[:, 1:])
    _fill_corners_dgrid(fort(dx, clo, clo), fort(dy, clo, clo), npx, ng)

    # per-cell mid-points of the four edges (from the FILLED grid — this is
    # upstream's own local data-domain computation for dxa/dya)
    mid_w = mid_pt_sphere(grid_ll[:-1, :-1], grid_ll[:-1, 1:])   # (m_a, m_a, 2)
    mid_e = mid_pt_sphere(grid_ll[1:, :-1], grid_ll[1:, 1:])
    mid_s = mid_pt_sphere(grid_ll[:-1, :-1], grid_ll[1:, :-1])
    mid_n = mid_pt_sphere(grid_ll[:-1, 1:], grid_ll[1:, 1:])

    # dxa/dya: full-data-domain formulas (the upstream mpp update is
    # commented out — local compute IS the semantics), then
    # fill_corners(dxa, dya, AGRID)  (fv_grid_tools:816-828)
    dxa = gcd_scaled(mid_e, mid_w)
    dya = gcd_scaled(mid_n, mid_s)
    _fill_corners_agrid_pair(fort(dxa, clo, clo), fort(dya, clo, clo),
                             npx, ng)

    # area: strips = mpp copies (kinked quads); corner regions =
    # fill_ghost(area, -big_number); rarea = 1/area over the FULL domain
    # so the corner regions carry upstream's -1e-8  (fv_grid_tools:978-1009)
    area = get_area_quad(grid_ll[:-1, :-1], grid_ll[1:, :-1],
                         grid_ll[1:, 1:], grid_ll[:-1, 1:]) * radius**2
    area[~cell_ok] = -BIG_NUMBER
    rarea = 1.0 / area

    # dxc: agrid spacing in x with the upstream panel-border special at
    # Fortran i == 1 and i == npx.  The specials are computed upstream on
    # the compute rows only, but the mpp exchange delivers the neighbours'
    # own (seam-aligned, axis-swapped) specials at the same Fortran rows in
    # the strips — one uniform rule.  Corner regions from
    # fill_corners(dxc, dyc, CGRID)  (fv_grid_tools:939-943; the pre-mpp
    # isd/ied+1 replication is dead state on the cubed sphere).
    i1 = ng          # np row of Fortran node/face index 1
    inpx = npx + ng - 1
    dxc = np.empty((m_b, m_a))
    dxc[1:-1, :] = gcd_scaled(agrid_ll[:-1, :], agrid_ll[1:, :])
    # outermost rows: the mpp copy needs the centre one beyond the domain
    dxc[0, :] = gcd_scaled(agrid_w[0, 1:-1], agrid_w[1, 1:-1])
    dxc[-1, :] = gcd_scaled(agrid_w[-2, 1:-1], agrid_w[-1, 1:-1])
    dxc[i1, :] = 2.0 * gcd_scaled(mid_w[i1, :], agrid_ll[i1, :])
    dxc[inpx, :] = 2.0 * gcd_scaled(agrid_ll[inpx - 1, :],
                                    mid_e[inpx - 1, :])
    dyc = np.empty((m_a, m_b))
    dyc[:, 1:-1] = gcd_scaled(agrid_ll[:, :-1], agrid_ll[:, 1:])
    dyc[:, 0] = gcd_scaled(agrid_w[1:-1, 0], agrid_w[1:-1, 1])
    dyc[:, -1] = gcd_scaled(agrid_w[1:-1, -2], agrid_w[1:-1, -1])
    dyc[:, i1] = 2.0 * gcd_scaled(mid_s[:, i1], agrid_ll[:, i1])
    dyc[:, inpx] = 2.0 * gcd_scaled(agrid_ll[:, inpx - 1],
                                    mid_n[:, inpx - 1])
    _fill_corners_cgrid(fort(dxc, clo, clo), fort(dyc, clo, clo), npx, ng)
    rdxc = 1.0 / dxc
    rdyc = 1.0 / dyc

    # ---- self-verification: halo constructions == certified builders on
    #      the compute domain (same formulas, same inputs -> byte-equal) ----
    sl_a = slice(ng, ng + n)       # Fortran cells 1..n
    sl_b = slice(ng, ng + n + 1)   # Fortran nodes/faces 1..n+1
    checks = (
        ("dx", dx[sl_a, sl_b], cert["dx"][tile - 1]),
        ("dy", dy[sl_b, sl_a], cert["dy"][tile - 1]),
        ("dxa", dxa[sl_a, sl_a], cert["dxa"][tile - 1]),
        ("dya", dya[sl_a, sl_a], cert["dya"][tile - 1]),
        ("area", area[sl_a, sl_a], cert["area"][tile - 1]),
        ("dxc", dxc[sl_b, sl_a], cert["dxc"][tile - 1]),
        ("dyc", dyc[sl_a, sl_b], cert["dyc"][tile - 1]),
    )
    for name, got, want in checks:
        if not np.array_equal(got, want):  # pragma: no cover - tripwire
            raise AssertionError(
                f"kinked-lattice {name} construction diverged from the "
                "certified compute-domain builder")

    # ---- sg pipeline (grid_utils_init order) ----
    X = latlon2xyz(grid_ll)                     # (m_b, m_b, 3)
    ctr = latlon2xyz(agrid_ll)                  # (m_a, m_a, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        csg, ssg = sg_window_fields(X, ctr)     # (m_a, m_a, 9)
    #   corner-region cells produce NaN from the sentinel nodes and are
    #   fill_ghost'ed below; sanitize so every derived field stays finite
    #   and array-comparable (upstream holds finite fill_corners junk here)
    csg = np.nan_to_num(csg, nan=BIG_NUMBER)
    ssg = np.nan_to_num(ssg, nan=BIG_NUMBER)
    CSG = fort(csg, clo, clo)
    SSG = fort(ssg, clo, clo)

    # v1 patches (sin only; fv_grid_utils.F90:373-399) — sequenced before
    # the cosa/rsin derivations exactly as upstream
    for i in range(-2, 0 + 1):                  # sw
        SSG[0, i, 3 - 1] = SSG[i, 1, 2 - 1]
        SSG[i, 0, 4 - 1] = SSG[1, i, 1 - 1]
    for i in range(npx, npx + 2 + 1):           # nw (npy == npx)
        SSG[0, i, 3 - 1] = SSG[npx - i, npx - 1, 4 - 1]
    for i in range(-2, 0 + 1):
        SSG[i, npx, 2 - 1] = SSG[1, npx + i, 1 - 1]
    for j in range(-2, 0 + 1):                  # se
        SSG[npx, j, 1 - 1] = SSG[npx - j, 1, 2 - 1]
    for i in range(npx, npx + 2 + 1):
        SSG[i, 0, 4 - 1] = SSG[npx - 1, npx - i, 3 - 1]
    for i in range(npx, npx + 2 + 1):           # ne
        SSG[npx, i, 1 - 1] = SSG[i, npx - 1, 4 - 1]
        SSG[i, npx, 2 - 1] = SSG[npx - 1, i, 3 - 1]

    # cosa_u/sina_u/rsin_u (upstream loops run i = isd+1..ied; the isd and
    # ied+1 u-columns keep their big_number initialisation)
    cosa_u = np.full((m_b, m_a), BIG_NUMBER)
    sina_u = np.full((m_b, m_a), BIG_NUMBER)
    rsin_u = np.full((m_b, m_a), BIG_NUMBER)
    cosa_u[1:-1, :] = 0.5 * (csg[:-1, :, 2] + csg[1:, :, 0])
    sina_u[1:-1, :] = 0.5 * (ssg[:-1, :, 2] + ssg[1:, :, 0])
    rsin_u[1:-1, :] = 1.0 / np.maximum(TINY_NUMBER, sina_u[1:-1, :] ** 2)

    cosa_v = np.full((m_a, m_b), BIG_NUMBER)
    sina_v = np.full((m_a, m_b), BIG_NUMBER)
    rsin_v = np.full((m_a, m_b), BIG_NUMBER)
    cosa_v[:, 1:-1] = 0.5 * (csg[:, :-1, 3] + csg[:, 1:, 1])
    sina_v[:, 1:-1] = 0.5 * (ssg[:, :-1, 3] + ssg[:, 1:, 1])
    rsin_v[:, 1:-1] = 1.0 / np.maximum(TINY_NUMBER, sina_v[:, 1:-1] ** 2)

    cosa_s = csg[..., 4].copy()
    rsin2 = 1.0 / np.maximum(TINY_NUMBER, ssg[..., 4] ** 2)

    # fill_ghost(cosa_s, big_number)
    ghost = ~cell_ok
    cosa_s[ghost] = BIG_NUMBER

    # rsin_u/v panel-border overrides (rsin_border_override: Fortran
    # SIGN semantics incl BOTH signed zeros — codex r2 P2-4 / r3 P2-b)
    for irow in (i1, inpx):
        rsin_u[irow, :] = rsin_border_override(sina_u[irow, :])
    for jcol in (i1, inpx):
        rsin_v[:, jcol] = rsin_border_override(sina_v[:, jcol])

    # fill_ghost on sin/cos_sg (tiny/big), then v2 patches (sin AND cos;
    # note the nw x-strip source differs from v1: npx-i, not npx+i)
    ssg[ghost] = TINY_NUMBER
    csg[ghost] = BIG_NUMBER
    for i in range(0, -2 - 1, -1):              # sw
        SSG[0, i, 3 - 1] = SSG[i, 1, 2 - 1]
        SSG[i, 0, 4 - 1] = SSG[1, i, 1 - 1]
        CSG[0, i, 3 - 1] = CSG[i, 1, 2 - 1]
        CSG[i, 0, 4 - 1] = CSG[1, i, 1 - 1]
    for i in range(npx, npx + 2 + 1):           # nw
        SSG[0, i, 3 - 1] = SSG[npx - i, npx - 1, 4 - 1]
        CSG[0, i, 3 - 1] = CSG[npx - i, npx - 1, 4 - 1]
    for i in range(0, -2 - 1, -1):
        SSG[i, npx, 2 - 1] = SSG[1, npx - i, 1 - 1]
        CSG[i, npx, 2 - 1] = CSG[1, npx - i, 1 - 1]
    for j in range(0, -2 - 1, -1):              # se
        SSG[npx, j, 1 - 1] = SSG[npx - j, 1, 2 - 1]
        CSG[npx, j, 1 - 1] = CSG[npx - j, 1, 2 - 1]
    for i in range(npx, npx + 2 + 1):
        SSG[i, 0, 4 - 1] = SSG[npx - 1, npx - i, 3 - 1]
        CSG[i, 0, 4 - 1] = CSG[npx - 1, npx - i, 3 - 1]
    for i in range(0, 2 + 1):                   # ne
        SSG[npx, npx + i, 1 - 1] = SSG[npx + i, npx - 1, 4 - 1]
        SSG[npx + i, npx, 2 - 1] = SSG[npx - 1, npx + i, 3 - 1]
        CSG[npx, npx + i, 1 - 1] = CSG[npx + i, npx - 1, 4 - 1]
        CSG[npx + i, npx, 2 - 1] = CSG[npx - 1, npx + i, 3 - 1]

    # ---- Coriolis at B nodes (formula over the FULL filled lattice; the
    # test_cases fill_corners(fC,...,XDir) call passes no AGRID/BGRID flag
    # and is an upstream no-op) ----
    fC = 2.0 * omega * (
        -np.cos(g_lon) * np.cos(g_lat) * np.sin(rotation_alpha)
        + np.sin(g_lat) * np.cos(rotation_alpha))

    # area_c: certified compute B + strip B-quads on kinked centres with
    # the seam x2 half-dual specials, corner regions from
    # fill_corners(area_c, XDir, BGRID), then rarea_c over the FULL node
    # domain (fv_grid_tools:974-981, 1011-1014).
    area_c = np.empty((m_b, m_b))
    area_c[1:-1, 1:-1] = get_area_quad(
        agrid_ll[:-1, :-1], agrid_ll[1:, :-1],
        agrid_ll[1:, 1:], agrid_ll[:-1, 1:]) * radius**2
    area_c[0, 1:-1] = get_area_quad(
        agrid_w[0, 1:-2], agrid_w[1, 1:-2],
        agrid_w[1, 2:-1], agrid_w[0, 2:-1]) * radius**2
    area_c[-1, 1:-1] = get_area_quad(
        agrid_w[-2, 1:-2], agrid_w[-1, 1:-2],
        agrid_w[-1, 2:-1], agrid_w[-2, 2:-1]) * radius**2
    area_c[:, 0] = np.concatenate((
        [area_c[1, 1]],   # placeholder; corner slots overwritten below
        get_area_quad(agrid_w[1:-2, 0], agrid_w[2:-1, 0],
                      agrid_w[2:-1, 1], agrid_w[1:-2, 1]) * radius**2,
        [area_c[-2, 1]]))
    area_c[:, -1] = np.concatenate((
        [area_c[1, -2]],
        get_area_quad(agrid_w[1:-2, -2], agrid_w[2:-1, -2],
                      agrid_w[2:-1, -1], agrid_w[1:-2, -1]) * radius**2,
        [area_c[-2, -2]]))
    # seam x2 specials along the full Fortran i==1/npx columns and
    # j==1/npy rows (strips carry the neighbours' own border specials);
    # per-side point orders mirror the certified step-3 constructions.
    # Evaluated on the WIDE window so the outermost B slots (which need a
    # centre/node one beyond the cropped domain) also carry the special —
    # codex r2 finding 1.
    iw = i1 + 1              # wide np index of Fortran node/cell 1
    inpxw = inpx + 1
    gm_y_w = mid_pt_sphere(grid_w[:, :-1], grid_w[:, 1:])   # (m_b+2, m_a+2, 2)
    gm_x_w = mid_pt_sphere(grid_w[:-1, :], grid_w[1:, :])   # (m_a+2, m_b+2, 2)
    pm = gm_y_w[iw]                       # mids indexed by wide cell
    cw = agrid_w[iw]
    area_c[i1, :] = 2.0 * get_area_quad(
        pm[:m_b], cw[:m_b], cw[1:m_b + 1], pm[1:m_b + 1]) * radius**2  # west
    pm = gm_y_w[inpxw]
    cw = agrid_w[inpxw - 1]
    area_c[inpx, :] = 2.0 * get_area_quad(
        cw[:m_b], pm[:m_b], pm[1:m_b + 1], cw[1:m_b + 1]) * radius**2  # east
    pm = gm_x_w[:, iw]
    cw = agrid_w[:, iw]
    area_c[:, i1] = 2.0 * get_area_quad(
        pm[:m_b], pm[1:m_b + 1], cw[1:m_b + 1], cw[:m_b]) * radius**2  # south
    pm = gm_x_w[:, inpxw]
    cw = agrid_w[:, inpxw - 1]
    area_c[:, inpx] = 2.0 * get_area_quad(
        cw[:m_b], cw[1:m_b + 1], pm[1:m_b + 1], pm[:m_b]) * radius**2  # north
    # certified compute-domain values (incl. the x2 borders and x3 cube
    # vertices) take precedence over the generic constructions above
    area_c[sl_b, sl_b] = cert["area_c"][tile - 1]
    _fill_corners_bgrid_x(fort(area_c, clo, clo), npx, ng)
    rarea_c = 1.0 / area_c

    # ---- d_sw additions (phase 4b) ----
    rdx, rdy = 1.0 / dx, 1.0 / dy
    rdxa, rdya = 1.0 / dxa, 1.0 / dya

    # f0: Coriolis at cell centres over the full filled lattice
    # (test_cases init; its no-flag fill_corners call is a no-op)
    f0 = 2.0 * omega * (
        -np.cos(agrid_ll[..., 0]) * np.cos(agrid_ll[..., 1])
        * np.sin(rotation_alpha)
        + np.sin(agrid_ll[..., 1]) * np.cos(rotation_alpha))

    # B-node cosa/sina (grid_utils_init loop js..je+1 — compute B only) and
    # rsina with the panel-border big_number rule (the (npx,npy) branch is
    # an upstream no-op: that node keeps the big_number initialisation)
    cosa_b = np.full((m_b, m_b), BIG_NUMBER)
    sina_b = np.full((m_b, m_b), BIG_NUMBER)
    rsina = np.full((m_b, m_b), BIG_NUMBER)
    cosa_b[sl_b, sl_b] = 0.5 * (csg[ng - 1:ng + n, ng - 1:ng + n, 7]
                                + csg[ng:ng + n + 1, ng:ng + n + 1, 5])
    sina_b[sl_b, sl_b] = 0.5 * (ssg[ng - 1:ng + n, ng - 1:ng + n, 7]
                                + ssg[ng:ng + n + 1, ng:ng + n + 1, 5])
    inner = slice(ng + 1, ng + n)      # Fortran B 2..npx-1
    rsina[inner, inner] = 1.0 / np.maximum(TINY_NUMBER,
                                           sina_b[inner, inner] ** 2)

    # divg_u/del6_u (u-position: cell-i x node-j) over the FULL data domain
    # with the seam-row special at Fortran j==1/npy — the same rows carry
    # the neighbours' own border specials in the halo strips (seam-aligned,
    # axis-swapped CGRID pair), so one uniform rule reproduces the
    # post-mpp state.
    divg_u = np.full((m_a, m_b), BIG_NUMBER)
    del6_u = np.full((m_a, m_b), BIG_NUMBER)
    with np.errstate(invalid="ignore", divide="ignore"):
        plain_u = sina_v * dyc / np.where(dx != BIG_NUMBER, dx, np.nan)
        plain6_u = sina_v * dx / np.where(dyc != BIG_NUMBER, dyc, np.nan)
    ok_u = (sina_v != BIG_NUMBER) & (dyc != BIG_NUMBER) & (dx != BIG_NUMBER)
    divg_u[ok_u] = plain_u[ok_u]
    del6_u[ok_u] = plain6_u[ok_u]
    SSGf = fort(ssg, clo, clo)
    for jrow in (1, npx):              # Fortran node j == 1, npy
        jn = jrow + ng - 1
        for icell in range(1 - ng, n + ng + 1):
            ic = icell + ng - 1
            if not (cell_ok[ic, jn - 1] if jn - 1 >= 0 else False) \
                    or not (cell_ok[ic, jn] if jn < m_a else False):
                continue
            s = 0.5 * (SSGf[icell, jrow, 2 - 1]
                       + SSGf[icell, jrow - 1, 4 - 1])
            divg_u[ic, jn] = s * dyc[ic, jn] / dx[ic, jn]
            del6_u[ic, jn] = s * dx[ic, jn] / dyc[ic, jn]

    divg_v = np.full((m_b, m_a), BIG_NUMBER)
    del6_v = np.full((m_b, m_a), BIG_NUMBER)
    ok_v = (sina_u != BIG_NUMBER) & (dxc != BIG_NUMBER) & (dy != BIG_NUMBER)
    with np.errstate(invalid="ignore", divide="ignore"):
        plain_v = sina_u * dxc / np.where(dy != BIG_NUMBER, dy, np.nan)
        plain6_v = sina_u * dy / np.where(dxc != BIG_NUMBER, dxc, np.nan)
    divg_v[ok_v] = plain_v[ok_v]
    del6_v[ok_v] = plain6_v[ok_v]
    for irow in (1, npx):              # Fortran face i == 1, npx
        ic = irow + ng - 1
        for jcell in range(1 - ng, n + ng + 1):
            jn = jcell + ng - 1
            if not (cell_ok[ic - 1, jn] if ic - 1 >= 0 else False) \
                    or not (cell_ok[ic, jn] if ic < m_a else False):
                continue
            s = 0.5 * (SSGf[irow, jcell, 1 - 1]
                       + SSGf[irow - 1, jcell, 3 - 1])
            divg_v[ic, jn] = s * dxc[ic, jn] / dy[ic, jn]
            del6_v[ic, jn] = s * dy[ic, jn] / dxc[ic, jn]

    # A->B edge interpolation factors (edge_factors, non_ortho route)
    edge_w = np.full(n + 1, BIG_NUMBER)
    edge_e = np.full(n + 1, BIG_NUMBER)
    edge_s = np.full(n + 1, BIG_NUMBER)
    edge_n = np.full(n + 1, BIG_NUMBER)
    for j in range(2, n + 1):          # Fortran j = 2..npy-1
        jn = j + ng - 1
        # west/east: py(j) = mid(agrid(i-1, j), agrid(i, j)) at Fortran
        # i = 1 / npx — the i-index pair straddles the border into the
        # halo strip on the east side (cell npx == first halo cell)
        for arr, irow in ((edge_w, ng), (edge_e, npx + ng - 1)):
            py0 = mid_pt_sphere(agrid_ll[irow - 1, jn - 1],
                                agrid_ll[irow, jn - 1])
            py1 = mid_pt_sphere(agrid_ll[irow - 1, jn],
                                agrid_ll[irow, jn])
            gpt = grid_ll[irow, jn]
            d1 = great_circle_dist(py0, gpt)
            d2 = great_circle_dist(py1, gpt)
            arr[j - 1] = d2 / (d1 + d2)
        for arr, jrow in ((edge_s, ng), (edge_n, npx + ng - 1)):
            px0 = mid_pt_sphere(agrid_ll[jn - 1, jrow - 1],
                                agrid_ll[jn - 1, jrow])
            px1 = mid_pt_sphere(agrid_ll[jn, jrow - 1],
                                agrid_ll[jn, jrow])
            gpt = grid_ll[jn, jrow]
            d1 = great_circle_dist(px0, gpt)
            d2 = great_circle_dist(px1, gpt)
            arr[j - 1] = d2 / (d1 + d2)

    da_min = float(cert["area"].min())
    da_max = float(cert["area"].max())
    da_min_c = float(cert["area_c"][:, :n, :n].min())
    da_max_c = float(cert["area_c"][:, :n, :n].max())

    return {
        "rdx": rdx, "rdy": rdy, "rdxa": rdxa, "rdya": rdya,
        "f0": f0, "cosa": cosa_b, "sina": sina_b, "rsina": rsina,
        "divg_u": divg_u, "divg_v": divg_v,
        "del6_u": del6_u, "del6_v": del6_v,
        "edge_s": edge_s, "edge_n": edge_n,
        "edge_w": edge_w, "edge_e": edge_e,
        "area_c": area_c,
        "da_min": da_min, "da_max": da_max,
        "da_min_c": da_min_c, "da_max_c": da_max_c,
        "n": n, "ng": ng, "npx": npx,
        "grid_lon": g_lon, "grid_lat": g_lat,
        "agrid_lon": agrid_ll[..., 0], "agrid_lat": agrid_ll[..., 1],
        "dx": dx, "dy": dy, "dxa": dxa, "dya": dya,
        "dxc": dxc, "dyc": dyc, "rdxc": rdxc, "rdyc": rdyc,
        "area": area, "rarea": rarea, "rarea_c": rarea_c,
        "cosa_u": cosa_u, "sina_u": sina_u, "rsin_u": rsin_u,
        "cosa_v": cosa_v, "sina_v": sina_v, "rsin_v": rsin_v,
        "cosa_s": cosa_s, "rsin2": rsin2,
        "sin_sg": ssg, "cos_sg": csg,
        "fC": fC,
        "cell_ok": cell_ok, "node_ok": node_ok,
    }


def exchange_bgrid_scalar_halos(field6: list, tile: int, n: int, ng: int):
    """mpp_update_domains(field, position=CORNER) for one tile, B-grid scalar.

    Fills the side-strip halos of ``field6[tile-1]`` (numpy, shape
    (n+2ng+1, n+2ng+1), Fortran B-node index ``1-ng..n+1+ng``) from the six
    faces' stored B-node arrays.  A CORNER scalar exchange is a plain index
    copy of the neighbour's value at the shared B node (no component
    rotation) — the exact map is the certified phase-3 neighbour index on
    the odd-odd supergrid.  dyn_core starts this exchange on ``divgd`` right
    after c_sw and completes it before d_sw (dyn_core.F90:451/577); d_sw
    (nord=1) reads the immediate B-node halo in the divergence-damping
    n-loop.  Corner-diagonal regions are left untouched.  Mutates in place.
    """
    sg_npx = 2 * n + 1
    npx = n + 1
    lo = 1 - ng
    fld = field6[tile - 1]
    nw, ne, ns, nn = neighbor_tiles(tile)
    strips = (
        (nw, range(1 - ng, 0 + 1), range(1, npx + 1)),
        (ne, range(npx + 1, npx + ng + 1), range(1, npx + 1)),
        (ns, range(1, npx + 1), range(1 - ng, 0 + 1)),
        (nn, range(1, npx + 1), range(npx + 1, npx + ng + 1)),
    )
    for n_src, fi_range, fj_range in strips:
        src = field6[n_src - 1]
        for fi in fi_range:
            for fj in fj_range:
                si, sj = 2 * fi - 1, 2 * fj - 1     # B node -> supergrid
                ii, jj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
                bi, bj = (ii + 1) // 2, (jj + 1) // 2
                if 1 <= bi <= npx and 1 <= bj <= npx:
                    fld[fi - lo, fj - lo] = src[bi - lo, bj - lo]


def exchange_cgrid_vector_halos(uc6: list, vc6: list, tile: int,
                                n: int, ng: int):
    """mpp_update_domains(uc, vc, gridtype=CGRID_NE) for one tile.

    Fills the side-strip halos of ``uc6[tile-1]``/``vc6[tile-1]`` (numpy,
    Fortran shapes: uc (n+2ng+1, n+2ng), vc (n+2ng, n+2ng+1)) from the six
    faces' stored arrays.  Component selection and orientation sign follow
    the discrete rotation of the certified neighbour index map: components
    transform like basis vectors, so halo_x picks the source component
    whose supergrid axis maps onto the local +i direction, with the sign
    of that map derivative (mpp's NE-vector convention).  Corner-diagonal
    regions get the FV3 VECTOR corner fill (mySign=-1) afterwards — the
    plain-mpp treatment (upstream duo Lagrange-fills them via
    ext_vector; interim).

    Returns nothing; mutates the tile's arrays in place.
    """
    sg_npx = 2 * n + 1
    npx = n + 1
    lo = 1 - ng
    uc = uc6[tile - 1]
    vc = vc6[tile - 1]
    nw, ne, ns, nn = neighbor_tiles(tile)
    # halo strips in Fortran FACE indices per stagger:
    #   uc x-faces (i in isd..ied+1, j in jsd..jed), supergrid (2i-1, 2j)
    #   vc y-faces (i in isd..ied,   j in jsd..jed+1), supergrid (2i, 2j-1)
    strips = (
        (nw, range(1 - ng, 0 + 1), "i"), (ne, range(npx + 1, npx + ng + 1), "i"),
        (ns, range(1 - ng, 0 + 1), "j"), (nn, range(npx + 1, npx + ng + 1), "j"),
    )

    def src_value(si: int, sj: int, n_src: int) -> float:
        """Source component + sign for a local x-face supergrid slot."""
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        # map derivative along the local +i supergrid direction
        sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src, sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        src_uc = uc6[n_src - 1]
        src_vc = vc6[n_src - 1]
        if dii != 0:                      # axes aligned: uc <- uc
            sgn = 1.0 if dii > 0 else -1.0
            fi, fj = (sii + 1) // 2, sjj // 2
            return sgn * src_uc[fi - lo, fj - lo]
        sgn = 1.0 if djj > 0 else -1.0    # axes swapped: uc <- vc
        fi, fj = sii // 2, (sjj + 1) // 2
        return sgn * src_vc[fi - lo, fj - lo]

    def src_value_y(si: int, sj: int, n_src: int) -> float:
        """Source component + sign for a local y-face supergrid slot."""
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src, sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        src_uc = uc6[n_src - 1]
        src_vc = vc6[n_src - 1]
        if djj != 0:                      # axes aligned: vc <- vc
            sgn = 1.0 if djj > 0 else -1.0
            fi, fj = sii // 2, (sjj + 1) // 2
            return sgn * src_vc[fi - lo, fj - lo]
        sgn = 1.0 if dii > 0 else -1.0    # axes swapped: vc <- uc
        fi, fj = (sii + 1) // 2, sjj // 2
        return sgn * src_uc[fi - lo, fj - lo]

    for n_src, rng, axis in strips:
        if axis == "i":
            # uc: halo x-face columns; vc: halo cell columns
            for fi in rng:
                for fj in range(1, n + 1):        # compute rows only
                    uc[fi - lo, fj - lo] = src_value(2 * fi - 1, 2 * fj,
                                                     n_src)
            vc_cols = (range(1 - ng, 0 + 1) if rng.start < 1
                       else range(npx, npx + ng))     # cells beyond face
            for fi in vc_cols:
                for fj in range(1, n + 2):        # y-faces 1..npx
                    vc[fi - lo, fj - lo] = src_value_y(2 * fi, 2 * fj - 1,
                                                       n_src)
        else:
            for fj in rng:
                for fi in range(1, n + 1):
                    vc[fi - lo, fj - lo] = src_value_y(2 * fi, 2 * fj - 1,
                                                       n_src)
            uc_rows = (range(1 - ng, 0 + 1) if rng.start < 1
                       else range(npx, npx + ng))
            for fj in uc_rows:
                for fi in range(1, n + 2):
                    uc[fi - lo, fj - lo] = src_value(2 * fi - 1, 2 * fj,
                                                     n_src)
    # corner-diagonal regions: FV3 VECTOR corner fill (mySign=-1), the
    # plain-mpp treatment — upstream duo Lagrange-fills these via
    # ext_vector; interim so corner-region uc/vc never carry stale
    # ghosts into downstream consumers (c_sw delpc ring, p_grad_c wk).
    _fill_corners_cgrid(fort(uc, lo, lo), fort(vc, lo, lo),
                        npx, ng, -1.0)


def exchange_agrid_scalar_halos(f6: list, tile: int, n: int, ng: int):
    """mpp_update_domains A-grid scalar (cell centers, supergrid
    (2i, 2j)) for one tile — side-strip index copy of the neighbour's
    coincident cells, then the FV3 AGRID corner fill on the
    corner-diagonal regions.  INTERIM stand-in for the duo
    ext_scalar(…,0,0) k2e machinery (which K2E-interpolates and fills
    corners upstream)."""
    sg_npx = 2 * n + 1
    lo = 1 - ng
    fld = f6[tile - 1]
    nw, ne, ns, nn = neighbor_tiles(tile)
    strips = (
        (nw, range(1 - ng, 0 + 1), range(1, n + 1)),
        (ne, range(n + 1, n + ng + 1), range(1, n + 1)),
        (ns, range(1, n + 1), range(1 - ng, 0 + 1)),
        (nn, range(1, n + 1), range(n + 1, n + ng + 1)),
    )
    for n_src, fi_range, fj_range in strips:
        src = f6[n_src - 1]
        for fi in fi_range:
            for fj in fj_range:
                si, sj = 2 * fi, 2 * fj
                ii, jj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
                ci, cj = ii // 2, jj // 2
                if 1 <= ci <= n and 1 <= cj <= n:
                    fld[fi - lo, fj - lo] = src[ci - lo, cj - lo]
    # corner-diagonal regions: the FV3 AGRID corner fill (index map
    # ported from fv_mp_mod) — the duo ext machinery Lagrange-fills
    # these upstream; this is the mpp-consistent interim so consumers
    # reading within +/-ng of a cube corner (duo a2b at corner B-nodes)
    # never see stale ghost values.
    _fill_corners_agrid_x(fort(fld, lo, lo), n + 1, ng)


def exchange_dgrid_vector_halos(u6: list, v6: list, tile: int,
                                n: int, ng: int):
    """mpp_update_domains(u, v, gridtype=DGRID_NE) for one tile.

    D-grid staggering: u is the x-component on y-faces (supergrid
    (2i, 2j-1); Fortran u(isd:ied, jsd:jed+1)), v the y-component on
    x-faces ((2i-1, 2j); v(isd:ied+1, jsd:jed)).  Same
    component/orientation-sign machinery as the certified CGRID
    exchange, with the slot parities swapped.  INTERIM stand-in for
    ext_vector(u, v, dg, …, 0,1,1,0).  Mutates in place.
    """
    sg_npx = 2 * n + 1
    npx = n + 1
    lo = 1 - ng
    u = u6[tile - 1]
    v = v6[tile - 1]
    nw, ne, ns, nn = neighbor_tiles(tile)

    def src_u(si, sj, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src, sg_npx,
                                    sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        if dii != 0:                      # aligned: u <- u
            sgn = 1.0 if dii > 0 else -1.0
            fi, fj = sii // 2, (sjj + 1) // 2
            return sgn * u6[n_src - 1][fi - lo, fj - lo]
        sgn = 1.0 if djj > 0 else -1.0    # swapped: u <- v
        fi, fj = (sii + 1) // 2, sjj // 2
        return sgn * v6[n_src - 1][fi - lo, fj - lo]

    def src_v(si, sj, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src, sg_npx,
                                    sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        if djj != 0:                      # aligned: v <- v
            sgn = 1.0 if djj > 0 else -1.0
            fi, fj = (sii + 1) // 2, sjj // 2
            return sgn * v6[n_src - 1][fi - lo, fj - lo]
        sgn = 1.0 if dii > 0 else -1.0    # swapped: v <- u
        fi, fj = sii // 2, (sjj + 1) // 2
        return sgn * u6[n_src - 1][fi - lo, fj - lo]

    strips = (
        (nw, range(1 - ng, 0 + 1), "i"),
        (ne, range(n + 1, n + ng + 1), "i"),
        (ns, range(1 - ng, 0 + 1), "j"),
        (nn, range(npx + 1, npx + ng + 1), "j"),
    )
    for n_src, rng, axis in strips:
        if axis == "i":
            for fi in rng:
                for fj in range(1, npx + 1):      # u y-faces 1..npx
                    u[fi - lo, fj - lo] = src_u(2 * fi, 2 * fj - 1, n_src)
            v_cols = (range(1 - ng, 0 + 1) if rng.start < 1
                      else range(npx + 1, npx + ng + 1))
            for fi in v_cols:
                for fj in range(1, n + 1):
                    v[fi - lo, fj - lo] = src_v(2 * fi - 1, 2 * fj, n_src)
        else:
            for fj in rng:
                for fi in range(1, n + 1):
                    u[fi - lo, fj - lo] = src_u(2 * fi, 2 * fj - 1, n_src)
            vj = (range(1 - ng, 0 + 1) if rng.start <= 0
                  else range(n + 1, n + ng + 1))
            for fj in vj:
                for fi in range(1, npx + 1):
                    v[fi - lo, fj - lo] = src_v(2 * fi - 1, 2 * fj, n_src)
    _fill_corners_dgrid(fort(u, lo, lo), fort(v, lo, lo), npx, ng, -1.0)


_K2E_TAB_CACHE: dict = {}


def _k2e_tables(n: int, remap_ng: int = 3):
    key = (n, remap_ng)
    if key not in _K2E_TAB_CACHE:
        from legoesm.grids.fv3_native_halos import compute_fv3_native_k2e

        _K2E_TAB_CACHE[key] = compute_fv3_native_k2e(n, remap_ng=remap_ng,
                                                     k2e_nord=4)
    return _K2E_TAB_CACHE[key]


def k2e_remap_halo_rings(f6: list, stag: str, n: int, ng: int):
    """Kinked-to-extended along-edge Lagrange remap of the halo rings
    (cube_rmp semantics) for one stagger family, applied AFTER the
    index-copy exchange: each halo-ring value becomes the certified
    k2e interpolation of the copied (neighbour-line) ring at the
    EXTENDED-lattice position.

    ``stag``: one of A, B, CX, CY, DX, DY — the oracle-pinned phase-3a
    table families.  ``f6``: per-face data-domain numpy arrays whose
    layout matches the stagger.  Record keys are 1-based Fortran; the
    arrays start at Fortran ``1-ng`` on both axes.  A record with the
    i-key outside ``[1, n+1]`` is a W/E ring (along-edge coordinate =
    j); otherwise it is a S/N ring (along-edge = i).  Mutates in
    place; corner-diagonal cells are untouched (the vector corner
    fills / AGRID fill own them).
    """
    if ng > 4:
        raise NotImplementedError(
            "k2e remap rings pinned for ng<=3 (stepper) / ng==4 (the "
            "ext_vector geographic lattice, upstream dg%bd%ng)")
    tab = _k2e_tables(n, remap_ng=ng)
    ij = tab[f"{stag}_ij"]
    loc = tab[f"{stag}_loc"]
    coef = tab[f"{stag}_coef"]
    npd = coef.shape[1] // 2 - 1
    lo_off = 1 - ng
    a0 = 1 - lo_off                      # array index of Fortran 1
    # per-stagger interior extents (cells 1..n, nodes 1..n+1)
    i_hi = n + 1 if stag in ("B", "CX", "DX") else n
    j_hi = n + 1 if stag in ("B", "CY", "DY") else n

    for t6 in range(len(f6)):
        f = f6[t6]
        src = f.copy()
        for (fi, fj), lv, cw in zip(ij, loc, coef):
            start = a0 + (int(lv) - npd - 1)
            i_ring = fi < 1 or fi > i_hi
            j_ring = fj < 1 or fj > j_hi
            if i_ring == j_ring:
                continue                 # corner-diagonal record classes
            if i_ring:                   # W/E ring: along-edge = j
                col = fi - lo_off
                row = fj - lo_off
                if not (0 <= col < f.shape[0]
                        and 0 <= row < f.shape[1]):
                    continue
                if not (0 <= start and start + len(cw) <= f.shape[1]):
                    continue
                f[col, row] = float(
                    (cw * src[col, start:start + len(cw)]).sum())
            else:                        # S/N ring: along-edge = i
                col = fi - lo_off
                row = fj - lo_off
                if not (0 <= col < f.shape[0]
                        and 0 <= row < f.shape[1]):
                    continue
                if not (0 <= start and start + len(cw) <= f.shape[0]):
                    continue
                f[col, row] = float(
                    (cw * src[start:start + len(cw), row]).sum())


def _cgrid_edge_partner(fx6: list, fy6: list, tile: int, si: int, sj: int,
                        along: str, n: int, ng: int, n_src: int) -> float:
    """Neighbour's COINCIDENT C-edge flux value for a shared-edge slot.

    ``(si, sj)`` is the local supergrid slot of the flux point (x-face
    (2i-1, 2j); y-face (2i, 2j-1)); ``along`` is the local component
    axis ('i' for fx, 'j' for fy).  Component selection + orientation
    sign follow the discrete-rotation map derivative, exactly the
    certified ``exchange_cgrid_vector_halos`` convention (CGRID_NE).
    """
    sg_npx = 2 * n + 1
    sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
    if along == "i":
        sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src, sg_npx, sg_npx)
    else:
        sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src, sg_npx, sg_npx)
    dii, djj = sii2 - sii, sjj2 - sjj
    if (sii % 2 == 1) and (sjj % 2 == 0):        # lands on an x-face slot
        sgn = 1.0 if (dii if dii != 0 else djj) > 0 else -1.0
        fi, fj = (sii + 1) // 2, sjj // 2
        return sgn * fx6[n_src - 1][fi - 1, fj - 1]
    sgn = 1.0 if (djj if djj != 0 else dii) > 0 else -1.0
    fi, fj = sii // 2, (sjj + 1) // 2
    return sgn * fy6[n_src - 1][fi - 1, fj - 1]


def average_shared_edge_cgrid(fx6: list, fy6: list, n: int, ng: int):
    """dyn_core.F90:853-900 analog — mpp_get_boundary(CGRID_NE) + 0.5
    blend of the C-ring fluxes at the four face edges, all six faces.

    ``fx6``/``fy6``: per-face COMPUTE-ring flux slabs on the allflux
    layout — fx (n+1, n) x-faces (is:ie+1, js:je) at origin (1, 1),
    fy (n, n+1) y-faces (is:ie, js:je+1).  Blends exactly the dyn_core
    slots: fx columns i=1, npx over j=1..n; fy rows j=1, npx over
    i=1..n.

    Gather-then-apply (two phases) so every partner read sees the
    PRE-blend state, exactly like mpp_get_boundary buffering.  The
    0.5*(own + mapped-neighbour) blend leaves both faces' coincident
    slots equal up to the component sign map.  Mutates in place.
    """
    npx = n + 1
    updates = []
    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        fx = fx6[tile - 1]
        fy = fy6[tile - 1]
        for fj in range(1, n + 1):
            for fi, n_src in ((1, nw), (npx, ne)):
                part = _cgrid_edge_partner(fx6, fy6, tile, 2 * fi - 1,
                                           2 * fj, "i", n, ng, n_src)
                updates.append((fx, fi - 1, fj - 1,
                                0.5 * (fx[fi - 1, fj - 1] + part)))
        for fi in range(1, n + 1):
            for fj, n_src in ((1, ns), (npx, nn)):
                part = _cgrid_edge_partner(fx6, fy6, tile, 2 * fi,
                                           2 * fj - 1, "j", n, ng, n_src)
                updates.append((fy, fi - 1, fj - 1,
                                0.5 * (fy[fi - 1, fj - 1] + part)))
    for arr, i, j, val in updates:
        arr[i, j] = val


def average_allflux_shared_edges(afx6: list, afy6: list, nq: int,
                                 n: int, ng: int):
    """dyn_core.F90:853-900 slot selection over the allflux stacks.

    ``afx6``/``afy6``: per-face allflux slabs (n+1, n, 4+nq) /
    (n, n+1, 4+nq).  Fortran averages ONLY iq=1 (delp), iq=4 (temp)
    and iq>4 (tracers); slots 2 (w) and 3 (q_con) are NOT averaged —
    this wrapper applies :func:`average_shared_edge_cgrid` per
    selected slot and leaves 2/3 byte-untouched.
    """
    for iq in range(1, 4 + nq + 1):
        if iq == 1 or iq >= 4:
            fx6 = [a[:, :, iq - 1] for a in afx6]
            fy6 = [a[:, :, iq - 1] for a in afy6]
            average_shared_edge_cgrid(fx6, fy6, n, ng)


def _bgrid_edge_partner(xb6: list, yb6: list, tile: int, si: int, sj: int,
                        along: str, n: int, ng: int, n_src: int) -> float:
    """Neighbour's coincident B-node vector-component value (BGRID_NE)."""
    sg_npx = 2 * n + 1
    sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
    if along == "i":
        sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src, sg_npx, sg_npx)
    else:
        sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src, sg_npx, sg_npx)
    dii, djj = sii2 - sii, sjj2 - sjj
    bi, bj = (sii + 1) // 2, (sjj + 1) // 2       # B node (odd, odd)
    if along == "i":
        aligned = dii != 0
    else:
        aligned = djj != 0
    if along == "i":
        src = xb6[n_src - 1] if aligned else yb6[n_src - 1]
        d = dii if aligned else djj
    else:
        src = yb6[n_src - 1] if aligned else xb6[n_src - 1]
        d = djj if aligned else dii
    sgn = 1.0 if d > 0 else -1.0
    return sgn * src[bi - 1, bj - 1]


def average_shared_edge_bgrid(xb6: list, yb6: list, n: int, ng: int):
    """dyn_core.F90:968-1020 analog — mpp_get_boundary(BGRID_NE) + 0.5
    blend of the B-grid KE ingredients (ubb = x-like, vbbtemp = y-like)
    at the four face edges, all six faces.

    ``xb6``/``yb6``: per-face COMPUTE-ring B arrays, shape
    (n+1, n+1) at origin (is=1, js=1) — the d_sw3 output layout.
    Blends exactly the dyn_core slots: yb rows j=1, npx over
    i=1..npx; xb columns i=1, npx over j=1..npx (corner B-nodes
    touched once per array, matching the Fortran loop split).
    Gather-then-apply; mutates in place.
    """
    npx = n + 1
    updates = []
    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        xb = xb6[tile - 1]
        yb = yb6[tile - 1]
        for fi in range(1, npx + 1):
            for fj, n_src in ((1, ns), (npx, nn)):
                part = _bgrid_edge_partner(xb6, yb6, tile, 2 * fi - 1,
                                           2 * fj - 1, "j", n, ng, n_src)
                updates.append((yb, fi - 1, fj - 1,
                                0.5 * (yb[fi - 1, fj - 1] + part)))
        for fj in range(1, npx + 1):
            for fi, n_src in ((1, nw), (npx, ne)):
                part = _bgrid_edge_partner(xb6, yb6, tile, 2 * fi - 1,
                                           2 * fj - 1, "i", n, ng, n_src)
                updates.append((xb, fi - 1, fj - 1,
                                0.5 * (xb[fi - 1, fj - 1] + part)))
    for arr, i, j, val in updates:
        arr[i, j] = val


def _get_unit_vect2(ll1: np.ndarray, ll2: np.ndarray) -> np.ndarray:
    """get_unit_vect2: unit tangent at the arc midpoint, oriented p1->p2."""
    e1 = latlon2xyz(ll1)
    e2 = latlon2xyz(ll2)
    with np.errstate(invalid="ignore", divide="ignore"):
        pc = e1 + e2
        pc = pc / np.linalg.norm(pc, axis=-1, keepdims=True)   # mid_pt3_cart
        p3 = np.cross(e1, e2)
        uv = np.cross(p3, pc)
        return uv / np.linalg.norm(uv, axis=-1, keepdims=True)


def _latlon_vectors(ll: np.ndarray):
    """get_latlon_vector: geographic east/north unit vectors at ll."""
    lon = ll[..., 0]
    lat = ll[..., 1]
    ex = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    ey = np.stack([-np.sin(lat) * np.cos(lon),
                   -np.sin(lat) * np.sin(lon),
                   np.cos(lat)], axis=-1)
    return ex, ey


def analytic_swcore_state(gs: dict, *, u0: float = 40.0,
                          alpha: float = np.pi / 4.0,
                          delp0: float = 3.0e4, ddelp: float = 1.0e4,
                          pt0: float = 300.0, dpt: float = 10.0) -> dict:
    """Smooth analytic (delp, pt, u, v) on the kinked single-tile lattice.

    Solid-body wind rotated by ``alpha`` (nontrivial at every face seam),
    D-grid covariant components via the upstream ``test_cases`` recipe
    (edge-midpoint unit tangent inner products); delp/pt are smooth
    positive scalars of position.  Halo values are evaluated directly at
    the kinked-node geometry, which equals what the mpp exchanges deliver
    (same physical locations; the neighbour's storage orientation is the
    kinked lattice's own axis ordering).  Corner-diagonal slots get
    ``BIG_NUMBER`` — FV3 fills scalar corners itself (``fill2_4corners``)
    and the d2a2c corner fixes overwrite the vector ones before any
    consumed read.
    """
    g_lon, g_lat = gs["grid_lon"], gs["grid_lat"]
    node_ok, cell_ok = gs["node_ok"], gs["cell_ok"]
    grid_ll = np.stack([g_lon, g_lat], axis=-1)

    def wind(ll):
        lon = ll[..., 0]
        lat = ll[..., 1]
        uu = u0 * (np.cos(lat) * np.cos(alpha)
                   + np.sin(lat) * np.cos(lon) * np.sin(alpha))
        vv = -u0 * np.sin(lon) * np.sin(alpha)
        return uu, vv

    def scalars(ll):
        lon = ll[..., 0]
        lat = ll[..., 1]
        s = (-np.cos(lon) * np.cos(lat) * np.sin(alpha)
             + np.sin(lat) * np.cos(alpha))
        delp = delp0 - ddelp * s**2
        pt = pt0 + dpt * np.cos(2.0 * lon) * np.cos(lat) ** 3
        return delp, pt

    a_ll = np.stack([gs["agrid_lon"], gs["agrid_lat"]], axis=-1)
    delp_all, pt_all = scalars(a_ll)
    delp = np.full(delp_all.shape, BIG_NUMBER)
    pt = np.full(pt_all.shape, BIG_NUMBER)
    delp[cell_ok] = delp_all[cell_ok]
    pt[cell_ok] = pt_all[cell_ok]

    # u on x-edges (nodes (i,j)-(i+1,j)); v on y-edges (nodes (i,j)-(i,j+1))
    def covariant(ll_a, ll_b, ok):
        e = _get_unit_vect2(ll_a, ll_b)
        mid = mid_pt_sphere(ll_a, ll_b)
        ex, ey = _latlon_vectors(mid)
        uu, vv = wind(mid)
        comp = (uu * np.einsum("...i,...i", e, ex)
                + vv * np.einsum("...i,...i", e, ey))
        out = np.full(comp.shape, BIG_NUMBER)
        out[ok] = comp[ok]
        return out

    u = covariant(grid_ll[:-1, :], grid_ll[1:, :],
                  node_ok[:-1, :] & node_ok[1:, :])
    v = covariant(grid_ll[:, :-1], grid_ll[:, 1:],
                  node_ok[:, :-1] & node_ok[:, 1:])
    # D-wind corner regions: upstream initialization runs
    # fill_corners(u, v, VECTOR=.true., DGRID=.true.) after the mpp
    # exchange (test_cases), and d_sw's PPM halo-row stencils DO read
    # them (c_sw never did) — mirror the fill with the vector sign.
    n = gs["n"]
    ng = gs["ng"]
    clo = 1 - ng
    _fill_corners_dgrid(fort(u, clo, clo), fort(v, clo, clo),
                        n + 1, ng, sign=-1.0)
    return {"delp": delp, "pt": pt, "u": u, "v": v}
