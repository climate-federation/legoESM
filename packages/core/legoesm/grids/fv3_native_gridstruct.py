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

Known, documented divergence from upstream: inside the four corner-diagonal
regions the *derived* angle fields (``cosa_u``/``cosa_v``/``cosa_s``/
``rsin*``) are computed from this module's sentinel-lattice ``sg`` garbage,
while upstream derives them from ``fill_corners``-extended grid geometry.
Both are deterministic garbage; FV3's ``c_sw`` consumes those slots only in
halo-ring outputs (never in compute-domain results), which the phase-4
oracle test pins by comparing full arrays produced from *identical* inputs.

Fortran index convention: arrays are plain numpy with row 0 == Fortran
``isd = 1-ng`` (cell/lower-node axes) — helper ``fort`` views give
Fortran-indexed access for the verbatim patch loops.
"""

from __future__ import annotations

import numpy as np

from legoesm import constants
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

# --- upstream fill/sentinel constants (fv_grid_utils.F90) ---
BIG_NUMBER = 1.0e8       # fv_grid_utils big_number
TINY_NUMBER = 1.0e-8     # fv_grid_utils tiny_number (rsin floors + sin_sg ghost)

# --- FV3/FMS physical constants for oracle pinning (FMS constants_mod,
#     GFDL flavour); legoESM production paths use legoesm.constants ---
FV3_RADIUS_M = 6371.0e3  # const-ok: FMS RADIUS differs from legoESM R_earth; the oracle must pin upstream's value
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


def build_tile1_kinked_corner_lonlat(n: int, ng: int = 3,
                                     sentinel: float = BIG_NUMBER):
    """Kinked (mpp-equivalent) corner-node lon/lat for tile 1.

    Returns ``lon, lat`` of shape (n+2ng+1, n+2ng+1) over Fortran node
    indices ``1-ng .. n+1+ng``: interior from tile 1's own ED supergrid,
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

    # interior (tile 1 own supergrid odd-odd nodes)
    for i_node in range(1, n + 2):
        for j_node in range(1, n + 2):
            si, sj = 2 * i_node - 1, 2 * j_node - 1
            put(i_node, j_node, lon6[0][si - 1, sj - 1], lat6[0][si - 1, sj - 1])

    # side strips from neighbours (supergrid index space, tile 1)
    nw, ne, ns, nn = neighbor_tiles(1)
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
                ii_s, jj_s = neighbor_index(si, sj, 1, n_src, sg_npx, sg_npx)
                put((si + 1) // 2, (sj + 1) // 2,
                    lon6[n_src - 1][ii_s - 1, jj_s - 1],
                    lat6[n_src - 1][ii_s - 1, jj_s - 1])
    return lon, lat


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


def build_fv3_native_gridstruct(n: int, ng: int = 3, *,
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
    grid_ll = np.stack([g_lon, g_lat], axis=-1)
    agrid_ll = cell_center2(grid_ll[:-1, :-1], grid_ll[1:, :-1],
                            grid_ll[:-1, 1:], grid_ll[1:, 1:])   # (m_a, m_a, 2)

    # ---- certified compute-domain metrics (6-face, reference layout) ----
    lon6, lat6 = ed_supergrid_lonlat_ref(n)
    sgc = 2 * np.arange(1, n + 2) - 1
    corner6_lon = lon6[:, sgc - 1][:, :, sgc - 1]
    corner6_lat = lat6[:, sgc - 1][:, :, sgc - 1]
    cert = compute_fv3_native_metrics(corner6_lon, corner6_lat, radius)

    # ---- halo-complete length/area fields from the kinked lattice ----
    def gcd_scaled(p, q):
        return great_circle_dist(p, q) * radius

    dx = np.full((m_a, m_b), BIG_NUMBER)
    ok = node_ok[:-1, :] & node_ok[1:, :]
    dx[ok] = gcd_scaled(grid_ll[:-1, :], grid_ll[1:, :])[ok]

    dy = np.full((m_b, m_a), BIG_NUMBER)
    ok = node_ok[:, :-1] & node_ok[:, 1:]
    dy[ok] = gcd_scaled(grid_ll[:, :-1], grid_ll[:, 1:])[ok]

    # per-cell mid-points of the four edges
    mid_w = mid_pt_sphere(grid_ll[:-1, :-1], grid_ll[:-1, 1:])   # (m_a, m_a, 2)
    mid_e = mid_pt_sphere(grid_ll[1:, :-1], grid_ll[1:, 1:])
    mid_s = mid_pt_sphere(grid_ll[:-1, :-1], grid_ll[1:, :-1])
    mid_n = mid_pt_sphere(grid_ll[:-1, 1:], grid_ll[1:, 1:])

    dxa = np.full((m_a, m_a), BIG_NUMBER)
    dxa[cell_ok] = gcd_scaled(mid_e, mid_w)[cell_ok]
    dya = np.full((m_a, m_a), BIG_NUMBER)
    dya[cell_ok] = gcd_scaled(mid_n, mid_s)[cell_ok]

    area = np.full((m_a, m_a), BIG_NUMBER)
    area_all = get_area_quad(grid_ll[:-1, :-1], grid_ll[1:, :-1],
                             grid_ll[1:, 1:], grid_ll[:-1, 1:]) * radius**2
    area[cell_ok] = area_all[cell_ok]
    rarea = np.full((m_a, m_a), BIG_NUMBER)
    rarea[cell_ok] = 1.0 / area[cell_ok]

    # dxc: agrid spacing in x with the upstream panel-border special at
    # Fortran i == 1 and i == npx (the special holds in the halo rows too —
    # there it is the neighbour's own border special, seam-aligned).
    i1 = ng          # np row of Fortran node/face index 1
    inpx = npx + ng - 1
    dxc = np.full((m_b, m_a), BIG_NUMBER)
    okc = cell_ok[:-1, :] & cell_ok[1:, :]
    dxc[1:-1, :][okc] = gcd_scaled(agrid_ll[:-1, :], agrid_ll[1:, :])[okc]
    ok_row = cell_ok[i1, :]
    dxc[i1, ok_row] = 2.0 * gcd_scaled(mid_w[i1, :], agrid_ll[i1, :])[ok_row]
    ok_row = cell_ok[inpx - 1, :]
    dxc[inpx, ok_row] = 2.0 * gcd_scaled(agrid_ll[inpx - 1, :],
                                         mid_e[inpx - 1, :])[ok_row]
    rdxc = np.where(dxc != BIG_NUMBER, 1.0 / dxc, BIG_NUMBER)

    dyc = np.full((m_a, m_b), BIG_NUMBER)
    okc = cell_ok[:, :-1] & cell_ok[:, 1:]
    dyc[:, 1:-1][okc] = gcd_scaled(agrid_ll[:, :-1], agrid_ll[:, 1:])[okc]
    ok_col = cell_ok[:, i1]
    dyc[ok_col, i1] = 2.0 * gcd_scaled(mid_s[:, i1], agrid_ll[:, i1])[ok_col]
    ok_col = cell_ok[:, inpx - 1]
    dyc[ok_col, inpx] = 2.0 * gcd_scaled(agrid_ll[:, inpx - 1],
                                         mid_n[:, inpx - 1])[ok_col]
    rdyc = np.where(dyc != BIG_NUMBER, 1.0 / dyc, BIG_NUMBER)

    # ---- self-verification: halo constructions == certified builders on
    #      the compute domain (same formulas, same inputs -> byte-equal) ----
    sl_a = slice(ng, ng + n)       # Fortran cells 1..n
    sl_b = slice(ng, ng + n + 1)   # Fortran nodes/faces 1..n+1
    checks = (
        ("dx", dx[sl_a, sl_b], cert["dx"][0]),
        ("dy", dy[sl_b, sl_a], cert["dy"][0]),
        ("dxa", dxa[sl_a, sl_a], cert["dxa"][0]),
        ("dya", dya[sl_a, sl_a], cert["dya"][0]),
        ("area", area[sl_a, sl_a], cert["area"][0]),
        ("dxc", dxc[sl_b, sl_a], cert["dxc"][0]),
        ("dyc", dyc[sl_a, sl_b], cert["dyc"][0]),
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

    # rsin_u/v panel-border overrides: signed 1/sina at i==1 / i==npx
    for irow in (i1, inpx):
        s = sina_u[irow, :]
        rsin_u[irow, :] = 1.0 / (np.sign(s)
                                 * np.maximum(TINY_NUMBER, np.abs(s)))
    for jcol in (i1, inpx):
        s = sina_v[:, jcol]
        rsin_v[:, jcol] = 1.0 / (np.sign(s)
                                 * np.maximum(TINY_NUMBER, np.abs(s)))

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

    # ---- Coriolis at B nodes + rarea_c (compute domain certified) ----
    fC = np.full((m_b, m_b), BIG_NUMBER)
    fC[node_ok] = 2.0 * omega * (
        -np.cos(g_lon[node_ok]) * np.cos(g_lat[node_ok])
        * np.sin(rotation_alpha)
        + np.sin(g_lat[node_ok]) * np.cos(rotation_alpha))

    rarea_c = np.full((m_b, m_b), BIG_NUMBER)
    rarea_c[sl_b, sl_b] = 1.0 / cert["area_c"][0]
    area_c = np.full((m_b, m_b), BIG_NUMBER)
    area_c[sl_b, sl_b] = cert["area_c"][0]

    # ---- d_sw additions (phase 4b) ----
    def recip(x):
        return np.where(x != BIG_NUMBER, 1.0 / x, BIG_NUMBER)

    rdx, rdy = recip(dx), recip(dy)
    rdxa, rdya = recip(dxa), recip(dya)

    # f0: Coriolis at cell centres (test_cases init formula on agrid)
    f0 = np.full((m_a, m_a), BIG_NUMBER)
    f0[cell_ok] = 2.0 * omega * (
        -np.cos(agrid_ll[..., 0][cell_ok]) * np.cos(agrid_ll[..., 1][cell_ok])
        * np.sin(rotation_alpha)
        + np.sin(agrid_ll[..., 1][cell_ok]) * np.cos(rotation_alpha))

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
    okj = cell_ok[:, :-1] & cell_ok[:, 1:]         # dyc-style validity
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
    return {"delp": delp, "pt": pt, "u": u, "v": v}
