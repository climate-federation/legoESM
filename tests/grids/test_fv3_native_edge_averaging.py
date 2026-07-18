"""Phase-4c — six-face shared-edge flux averaging analogs.

``average_shared_edge_cgrid`` / ``average_shared_edge_bgrid`` are the
legoESM analogs of the two duo inter-panel averaging sites the
single-face translation certificates exclude:

- dyn_core.F90:853-900 — mpp_get_boundary(CGRID_NE) + 0.5 blend of the
  d_sw1 allflux C-ring fluxes (slots 1 and 4 + tracers);
- dyn_core.F90:968-1020 — mpp_get_boundary(BGRID_NE) + 0.5 blend of the
  d_sw3 ubb / vbbtemp B arrays before the kee assembly.

No Fortran mpp oracle is extractable (FMS domain machinery), so these
certify by TRUTH TIERS instead of bit-exactness against upstream:

T1a location-soundness — every blended slot's mapped partner is the
    COINCIDENT physical point on the neighbour face (lon/lat equality
    on the certified ED supergrid; a wrong/off-by-one partner map
    cannot pass);
T1b idempotency — a second application is a bitwise no-op;
T1c consistency — after one application both faces hold the same
    physical edge value (own == sign-mapped partner, bitwise);
T1d midpoint tripwire — perturbing one face's edge slot moves both
    faces to the exact midpoint of the two originals.

The integrated W2 run against the Zenodo duo reference remains the
external end-to-end check (tier 3).
"""

import numpy as np
import pytest
from legoesm.grids.fv3_native_gridstruct import (
    _bgrid_edge_partner,
    _cgrid_edge_partner,
    average_shared_edge_bgrid,
    average_shared_edge_cgrid,
)
from legoesm.grids.fv3_native_halos import (
    ed_supergrid_lonlat_ref,
    neighbor_index,
    neighbor_tiles,
)

N = 12
NG = 3
NPX = N + 1
SG_NPX = 2 * N + 1


def _rand6(shape, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape) for _ in range(6)]


def _cgrid_edge_slots():
    """(tile, si, sj, along, n_src) for every blended C slot."""
    out = []
    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        for fj in range(1, N + 1):
            out.append((tile, 1, 2 * fj, "i", nw))
            out.append((tile, 2 * NPX - 1, 2 * fj, "i", ne))
        for fi in range(1, N + 1):
            out.append((tile, 2 * fi, 1, "j", ns))
            out.append((tile, 2 * fi, 2 * NPX - 1, "j", nn))
    return out


def _bgrid_edge_slots():
    out = []
    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        for fi in range(1, NPX + 1):
            out.append((tile, 2 * fi - 1, 1, "j", ns))
            out.append((tile, 2 * fi - 1, 2 * NPX - 1, "j", nn))
        for fj in range(1, NPX + 1):
            out.append((tile, 1, 2 * fj - 1, "i", nw))
            out.append((tile, 2 * NPX - 1, 2 * fj - 1, "i", ne))
    return out


@pytest.mark.parametrize("slots", [_cgrid_edge_slots(), _bgrid_edge_slots()],
                         ids=["cgrid", "bgrid"])
def test_partner_map_is_coincident(slots):
    """T1a: the mapped partner slot is the SAME physical point (lon/lat
    equality on the certified ED supergrid)."""
    lon6, lat6 = ed_supergrid_lonlat_ref(N)
    for tile, si, sj, _along, n_src in slots:
        sii, sjj = neighbor_index(si, sj, tile, n_src, SG_NPX, SG_NPX)
        assert 1 <= sii <= SG_NPX and 1 <= sjj <= SG_NPX, (tile, si, sj)
        dlon = abs(lon6[tile - 1][si - 1, sj - 1]
                   - lon6[n_src - 1][sii - 1, sjj - 1])
        dlon = min(dlon, 2 * np.pi - dlon)
        dlat = abs(lat6[tile - 1][si - 1, sj - 1]
                   - lat6[n_src - 1][sii - 1, sjj - 1])
        assert dlon < 1e-12 and dlat < 1e-12, (tile, si, sj, n_src)


def test_cgrid_idempotent_and_consistent():
    """T1b + T1c for the C-ring averaging."""
    fx6 = _rand6((NPX, N), 7)
    fy6 = _rand6((N, NPX), 8)
    average_shared_edge_cgrid(fx6, fy6, N, NG)
    snap_x = [a.copy() for a in fx6]
    snap_y = [a.copy() for a in fy6]
    average_shared_edge_cgrid(fx6, fy6, N, NG)
    for a, b in zip(fx6, snap_x):
        assert np.array_equal(a, b)
    for a, b in zip(fy6, snap_y):
        assert np.array_equal(a, b)
    # consistency: own slot equals its sign-mapped partner value
    for tile, si, sj, along, n_src in _cgrid_edge_slots():
        part = _cgrid_edge_partner(fx6, fy6, tile, si, sj, along, N, NG,
                                   n_src)
        if along == "i":
            own = fx6[tile - 1][(si + 1) // 2 - 1, sj // 2 - 1]
        else:
            own = fy6[tile - 1][si // 2 - 1, (sj + 1) // 2 - 1]
        assert own == part, (tile, si, sj)


def test_bgrid_idempotent_and_consistent():
    """T1b + T1c for the B-grid averaging."""
    xb6 = _rand6((NPX, NPX), 9)
    yb6 = _rand6((NPX, NPX), 10)
    average_shared_edge_bgrid(xb6, yb6, N, NG)
    snap_x = [a.copy() for a in xb6]
    snap_y = [a.copy() for a in yb6]
    average_shared_edge_bgrid(xb6, yb6, N, NG)
    for a, b in zip(xb6, snap_x):
        assert np.array_equal(a, b)
    for a, b in zip(yb6, snap_y):
        assert np.array_equal(a, b)
    for tile, si, sj, along, n_src in _bgrid_edge_slots():
        part = _bgrid_edge_partner(xb6, yb6, tile, si, sj, along, N, NG,
                                   n_src)
        bi, bj = (si + 1) // 2, (sj + 1) // 2
        if along == "i":
            own = xb6[tile - 1][bi - 1, bj - 1]
        else:
            own = yb6[tile - 1][bi - 1, bj - 1]
        assert own == part, (tile, si, sj)


def test_cgrid_midpoint_tripwire():
    """T1d: perturb one west-edge fx slot; both faces land on the exact
    midpoint of the two pre-blend coincident values."""
    fx6 = _rand6((NPX, N), 11)
    fy6 = _rand6((N, NPX), 12)
    tile = 1
    nw = neighbor_tiles(tile)[0]
    fj = 5
    si, sj = 1, 2 * fj
    own_before = fx6[tile - 1][0, fj - 1]
    part_before = _cgrid_edge_partner(fx6, fy6, tile, si, sj, "i", N, NG,
                                      nw)
    average_shared_edge_cgrid(fx6, fy6, N, NG)
    assert fx6[tile - 1][0, fj - 1] == 0.5 * (own_before + part_before)


# ---------------------------------------------------------------------------
# INDEPENDENT contact table (codex avg-r1 P1): derived from pure supergrid
# GEOMETRY — coincident-edge coordinate matching + tangent-vector dot
# products (scripts recorded in the campaign log), NOT from the
# neighbor_index machinery under test.  Per directed edge:
#   (tile, edge) -> (partner tile, partner edge, param order, partner
#                    component for the local NORMAL, sign)
# order +1: edge params run the same direction; -1: reversed.  All
# normal-component signs on this tiling are +1 (the -1s live on
# tangential components, outside these boundary loops).
# ---------------------------------------------------------------------------
CONTACT = {
    (1, "E"): (2, "W", +1, "i", +1),
    (1, "N"): (3, "W", -1, "i", +1),
    (1, "S"): (6, "N", +1, "j", +1),
    (1, "W"): (5, "N", -1, "j", +1),
    (2, "E"): (4, "S", -1, "j", +1),
    (2, "N"): (3, "S", +1, "j", +1),
    (2, "S"): (6, "E", -1, "i", +1),
    (2, "W"): (1, "E", +1, "i", +1),
    (3, "E"): (4, "W", +1, "i", +1),
    (3, "N"): (5, "W", -1, "i", +1),
    (3, "S"): (2, "N", +1, "j", +1),
    (3, "W"): (1, "N", -1, "j", +1),
    (4, "E"): (6, "S", -1, "j", +1),
    (4, "N"): (5, "S", +1, "j", +1),
    (4, "S"): (2, "E", -1, "i", +1),
    (4, "W"): (3, "E", +1, "i", +1),
    (5, "E"): (6, "W", +1, "i", +1),
    (5, "N"): (1, "W", -1, "i", +1),
    (5, "S"): (4, "N", +1, "j", +1),
    (5, "W"): (3, "N", -1, "j", +1),
    (6, "E"): (2, "S", -1, "j", +1),
    (6, "N"): (1, "S", +1, "j", +1),
    (6, "S"): (4, "E", -1, "i", +1),
    (6, "W"): (5, "E", +1, "i", +1),
}


def _cgrid_partner_slot(tile, edge, k):
    """Expected partner (tile, array, i, j) for own C edge param k
    (cell index 1..N), from CONTACT alone."""
    t2, e2, order, comp, sgn = CONTACT[(tile, edge)]
    kp = k if order == 1 else N + 1 - k
    if e2 == "W":
        return t2, "fx", 1, kp, sgn
    if e2 == "E":
        return t2, "fx", NPX, kp, sgn
    if e2 == "S":
        return t2, "fy", kp, 1, sgn
    return t2, "fy", kp, NPX, sgn


def _own_cgrid_slot(tile, edge, k):
    if edge == "W":
        return "fx", 1, k
    if edge == "E":
        return "fx", NPX, k
    if edge == "S":
        return "fy", k, 1
    return "fy", k, NPX


def test_cgrid_impulse_vs_contact_table():
    """Independent sign/order pinning (codex avg-r1 P1): put a unit
    impulse at the partner slot named by the GEOMETRIC contact table;
    the own slot must blend to exactly +0.5 (sign +1 table-wide) at the
    table's index mapping — sweeping every directed edge and both an
    interior and an end param (order-reversal discrimination)."""
    for (tile, edge), _v in CONTACT.items():
        for k in (2, N - 1):
            fx6 = [np.zeros((NPX, N)) for _ in range(6)]
            fy6 = [np.zeros((N, NPX)) for _ in range(6)]
            t2, arr, pi, pj, sgn = _cgrid_partner_slot(tile, edge, k)
            (fx6 if arr == "fx" else fy6)[t2 - 1][pi - 1, pj - 1] = 1.0
            average_shared_edge_cgrid(fx6, fy6, N, NG)
            oarr, oi, oj = _own_cgrid_slot(tile, edge, k)
            own = (fx6 if oarr == "fx" else fy6)[tile - 1][oi - 1, oj - 1]
            assert own == 0.5 * sgn, (tile, edge, k, own)


def _bgrid_partner_slot(tile, edge, k):
    """Expected partner B slot for own edge param k (B index 1..NPX)."""
    t2, e2, order, comp, sgn = CONTACT[(tile, edge)]
    kp = k if order == 1 else NPX + 1 - k
    if e2 == "W":
        return t2, 1, kp, comp, sgn
    if e2 == "E":
        return t2, NPX, kp, comp, sgn
    if e2 == "S":
        return t2, kp, 1, comp, sgn
    return t2, kp, NPX, comp, sgn


def test_bgrid_impulse_vs_contact_table():
    """Same independent pinning for the B-grid blend, INCLUDING the
    corner B-nodes (k=1 and k=NPX) the Fortran loops touch."""
    for (tile, edge), (_t2, _e2, _order, comp, _sgn) in CONTACT.items():
        for k in (1, 3, NPX):
            xb6 = [np.zeros((NPX, NPX)) for _ in range(6)]
            yb6 = [np.zeros((NPX, NPX)) for _ in range(6)]
            t2, pi, pj, comp, sgn = _bgrid_partner_slot(tile, edge, k)
            # own normal component: x-like on W/E (xb), y-like on S/N (yb);
            # partner stores it in comp per the table
            (xb6 if comp == "i" else yb6)[t2 - 1][pi - 1, pj - 1] = 1.0
            average_shared_edge_bgrid(xb6, yb6, N, NG)
            if edge == "W":
                own = xb6[tile - 1][0, k - 1]
            elif edge == "E":
                own = xb6[tile - 1][NPX - 1, k - 1]
            elif edge == "S":
                own = yb6[tile - 1][k - 1, 0]
            else:
                own = yb6[tile - 1][k - 1, NPX - 1]
            assert own == 0.5 * sgn, (tile, edge, k, own)


def test_allflux_wrapper_slot_selection():
    """dyn_core iq selection (codex avg-r1 P1): slots 1 and 4 (+tracer
    slot 5) blend; slots 2 and 3 stay byte-untouched."""
    from legoesm.grids.fv3_native_gridstruct import (
        average_allflux_shared_edges,
    )

    nq = 1
    rng = np.random.default_rng(21)
    afx6 = [rng.standard_normal((NPX, N, 4 + nq)) for _ in range(6)]
    afy6 = [rng.standard_normal((N, NPX, 4 + nq)) for _ in range(6)]
    keep_x = [a.copy() for a in afx6]
    keep_y = [a.copy() for a in afy6]
    average_allflux_shared_edges(afx6, afy6, nq, N, NG)
    for t in range(6):
        for iq0 in (1, 2):          # slots 2,3 (0-based 1,2) untouched
            assert np.array_equal(afx6[t][:, :, iq0], keep_x[t][:, :, iq0])
            assert np.array_equal(afy6[t][:, :, iq0], keep_y[t][:, :, iq0])
        for iq0 in (0, 3, 4):       # slots 1,4,5 blended at the edges
            assert not np.array_equal(afx6[t][:, :, iq0],
                                      keep_x[t][:, :, iq0])
            assert not np.array_equal(afy6[t][:, :, iq0],
                                      keep_y[t][:, :, iq0])
