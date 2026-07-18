"""Phase-4c — extended-lattice gridstruct (duo consistency bundle entry).

``extend_gridstruct`` runs the certified metric engine on the face's
own gnomonic EXTENSION as one big face (size n+2ng == the data domain,
1:1 layouts) with self-contained finite-difference tangent angles for
the halo rings, then RESTORES the certified kinked interior bitwise.

Gates:
- interior BITWISE identical to the kinked builder for every field
  (upstream conventions — including the divg/del6 seam specials and
  the border rsina override — preserved exactly);
- every HALO cell inside the outermost ring is finite and unpoisoned
  for all consumed metric/angle families (the extended lattice has
  real geometry everywhere: side strips AND corner wedges);
- the outermost ext ring is the DOCUMENTED junk region (engine
  face-edge special-casing at a non-edge) — excluded, and consumers
  stay >= 1 ring inside.
"""

import numpy as np
import pytest
from legoesm.grids.fv3_native_gridstruct import (
    BIG_NUMBER,
    build_extended_corner_lonlat,
    build_fv3_native_gridstruct,
    build_kinked_corner_lonlat,
    extend_gridstruct,
)

N = 12
NG = 3


@pytest.fixture(scope="module", params=[1, 3, 6])
def pair(request):
    t = request.param
    gs = build_fv3_native_gridstruct(N, NG, tile=t)
    return t, gs, extend_gridstruct(gs, N, NG, tile=t)


def test_ext_lattice_interior_bitwise():
    for t in (1, 3, 6):
        le, be = build_extended_corner_lonlat(N, NG, tile=t)
        lk, bk = build_kinked_corner_lonlat(N, NG, tile=t)
        sl = slice(NG, NG + N + 1)
        assert np.array_equal(le[sl, sl], lk[sl, sl]), t
        assert np.array_equal(be[sl, sl], bk[sl, sl]), t
        assert np.isfinite(le).all() and np.isfinite(be).all(), t


_CI = slice(NG, NG + N)
_BI = slice(NG, NG + N + 1)
_FIELDS = (
    ("dx", (_CI, _BI)), ("dy", (_BI, _CI)), ("area", (_CI, _CI)),
    ("rarea", (_CI, _CI)), ("dxa", (_CI, _CI)), ("dya", (_CI, _CI)),
    ("dxc", (_BI, _CI)), ("dyc", (_CI, _BI)), ("area_c", (_BI, _BI)),
    ("rarea_c", (_BI, _BI)), ("cosa_u", (_BI, _CI)),
    ("sina_u", (_BI, _CI)), ("rsin_u", (_BI, _CI)),
    ("cosa_v", (_CI, _BI)), ("sina_v", (_CI, _BI)),
    ("rsin_v", (_CI, _BI)), ("cosa", (_BI, _BI)), ("sina", (_BI, _BI)),
    ("rsina", (_BI, _BI)), ("cosa_s", (_CI, _CI)), ("rsin2", (_CI, _CI)),
    ("divg_u", (_CI, _BI)), ("del6_u", (_CI, _BI)),
    ("divg_v", (_BI, _CI)), ("del6_v", (_BI, _CI)),
    ("f0", (_CI, _CI)), ("fC", (_BI, _BI)),
)


def test_interior_bitwise(pair):
    t, gs, ge = pair
    for key, sl in _FIELDS:
        assert np.array_equal(ge[key][sl], gs[key][sl]), (t, key)
    assert np.array_equal(ge["sin_sg"][_CI, _CI, :],
                          gs["sin_sg"][_CI, _CI, :]), t
    assert np.array_equal(ge["cos_sg"][_CI, _CI, :],
                          gs["cos_sg"][_CI, _CI, :]), t


def test_halo_clean_inside_outer_ring(pair):
    t, _gs, ge = pair

    def halo_mask(shape, isl, jsl):
        m = np.ones(shape[:2], dtype=bool)
        m[isl, jsl] = False
        m[0, :] = m[-1, :] = False
        m[:, 0] = m[:, -1] = False
        return m

    for key, (isl, jsl) in _FIELDS:
        a = ge[key]
        v = a[halo_mask(a.shape, isl, jsl)]
        assert np.isfinite(v).all(), (t, key)
        assert (np.abs(v) != BIG_NUMBER).all(), (t, key)
    for key in ("area", "area_c", "rarea", "rarea_c"):
        a = ge[key]
        isl, jsl = dict(_FIELDS)[key]
        v = a[halo_mask(a.shape, isl, jsl)]
        assert (v > 0).all(), (t, key)
    a = ge["sin_sg"]
    m = halo_mask(a.shape[:2], _CI, _CI)
    assert np.isfinite(a[m]).all(), t


def _analytic(lon, lat):
    return np.sin(lon) * np.cos(2 * lat)


def test_k2e_remap_halo_rings_a_stagger():
    """The generic staggered k2e remap moves the index-copied (kinked)
    halo rings onto the EXTENDED positions: for a smooth analytic
    field, |halo - f(ext position)| drops from the kink offset (~0.42
    at C12 ring 3) to Lagrange accuracy (<5e-3) — >100x. Interior
    untouched."""
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
        k2e_remap_halo_rings,
    )

    gs6 = [build_fv3_native_gridstruct(N, NG, tile=t) for t in range(1, 7)]
    ge6 = [extend_gridstruct(gs6[t], N, NG, tile=t + 1) for t in range(6)]
    f6 = [_analytic(g["agrid_lon"], g["agrid_lat"]) for g in gs6]
    for t in range(1, 7):
        exchange_agrid_scalar_halos(f6, t, N, NG)
    copied = [x.copy() for x in f6]
    k2e_remap_halo_rings(f6, "A", N, NG)
    strips = ((slice(0, NG), slice(NG, NG + N)),
              (slice(NG + N, None), slice(NG, NG + N)),
              (slice(NG, NG + N), slice(0, NG)),
              (slice(NG, NG + N), slice(NG + N, None)))
    wc = wr = 0.0
    for t in range(6):
        truth = _analytic(ge6[t]["agrid_lon"], ge6[t]["agrid_lat"])
        for sl in strips:
            wc = max(wc, float(np.abs(copied[t][sl] - truth[sl]).max()))
            wr = max(wr, float(np.abs(f6[t][sl] - truth[sl]).max()))
        assert np.array_equal(f6[t][NG:NG + N, NG:NG + N],
                              copied[t][NG:NG + N, NG:NG + N]), t
    assert wc > 0.3, wc                 # the kink offset is real
    assert wr < 5e-3, wr                # remap hits ext positions
    assert wr < wc / 100.0, (wc, wr)


def test_k2e_remap_halo_rings_b_stagger():
    """Same validation on the B stagger (node extents 1..n+1 — the
    classification path that differs from A; C/D families share the
    identical code with mixed extents)."""
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_bgrid_scalar_halos,
        k2e_remap_halo_rings,
    )

    f6 = []
    truths = []
    for t in range(1, 7):
        lon, lat = build_extended_corner_lonlat(N, NG, tile=t)
        lonk, latk = build_kinked_corner_lonlat(N, NG, tile=t)
        fk = _analytic(lonk, latk)
        fk[~np.isfinite(fk)] = 0.0
        f6.append(fk)
        truths.append(_analytic(lon, lat))
    for t in range(1, 7):
        exchange_bgrid_scalar_halos(f6, t, N, NG)
    copied = [x.copy() for x in f6]
    k2e_remap_halo_rings(f6, "B", N, NG)
    strips = ((slice(0, NG), slice(NG, NG + N + 1)),
              (slice(NG + N + 1, None), slice(NG, NG + N + 1)),
              (slice(NG, NG + N + 1), slice(0, NG)),
              (slice(NG, NG + N + 1), slice(NG + N + 1, None)))
    wc = wr = 0.0
    for t in range(6):
        for sl in strips:
            wc = max(wc, float(np.abs(copied[t][sl] - truths[t][sl]).max()))
            wr = max(wr, float(np.abs(f6[t][sl] - truths[t][sl]).max()))
    assert wr < wc / 20.0, (wc, wr)
    assert wr < 2e-2, wr
