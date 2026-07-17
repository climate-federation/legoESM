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
