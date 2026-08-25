"""SPMD machinery for the duo six-face exchange (M3 ladder, PR #1656 line).

This module holds the pieces that let the certified duo halo exchange run
face-sharded with O(halo) communication.  First piece: the CROSS-FACE DEPTH
CENSUS -- the design's load-bearing number.

Measured (jobs 9493433-9493438, decode verified against
``_FlatLayout.idx`` by adversarial review): every cross-face READ in the
duo halo machinery lives in the strip scatters, at source depth-from-edge
<= 7 (the ng=4 geo lattice; A/B/C/D strips 5-6).  The k2e stencils, all
five corner-Lagrange modes, ``fill_corners``, the write-backs and the
edge blends are face-local once strips have landed.  A ring exchange that
gathers width-w full borders (w > census) therefore feeds the certified
flat tables verbatim.

The census is a BUILD-TIME GATE, not a constant: the depth depends on
``(n, ng, ngp)`` (codex MAJOR -- ``ngp`` is contextual), so the SPMD
exchange must call :func:`assert_ring_width_covers` on the actual tables
it will run with, and refuse a ring that does not cover them.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "census_cross_face_depth",
    "assert_ring_width_covers",
]


# table attr -> layout attr, from the exchange fns' own pairings
# (fv3_duo_halos.py:1452-1520 and the builder at :1278-1430).
_TABLE_LAYOUTS = {
    "ex_a_strip": "lay_a", "ex_a_corner": "lay_a",
    "ex_a4_strip": "lay_a4", "ex_a4_corner": "lay_a4",
    "ex_b_strip": "lay_b",
    "ex_c_strip": "lay_c", "ex_c_corner": "lay_c",
    "ex_d_strip": "lay_d", "ex_d_corner": "lay_d",
    "k2e_a": "lay_a", "k2e_a4": "lay_a4", "k2e_b": "lay_b",
    "avg_c": "lay_fx", "avg_b": "lay_bb",
    "wr_d": "lay_wd", "wr_c": "lay_wc",
    "fc_bgrid_x": "lay_b", "fc_agrid_x": "lay_a", "fc_agrid_y": "lay_a",
    "fc_dgrid": "lay_d", "fc_cgrid": "lay_c", "fc_agrid_pair": "lay_ap",
}
_CORNER_LAYOUTS = {"a4": "lay_a4", "a3": "lay_a", "b3": "lay_b",
                   "du3": "lay_d", "dv3": "lay_d"}


def _decode(lay, flat):
    """flat index -> (face, depth-from-edge), inverting ``_FlatLayout.idx``
    (``base + (face*m0 + i0)*m1 + j0``).  Shape-preserving."""
    shp = np.asarray(flat).shape
    flat = np.asarray(flat).ravel()
    bases = np.asarray(list(lay.bases) + [lay.total])
    k = np.searchsorted(bases, flat, side="right") - 1
    face = np.empty_like(flat)
    depth = np.empty_like(flat)
    for kk, (m0, m1) in enumerate(lay.shapes):
        m = k == kk
        if not m.any():
            continue
        rem = flat[m] - lay.bases[kk]
        face[m] = rem // (m0 * m1)
        r2 = rem % (m0 * m1)
        i, j = r2 // m1, r2 % m1
        depth[m] = np.minimum(np.minimum(i, m0 - 1 - i),
                              np.minimum(j, m1 - 1 - j))
    return face.reshape(shp), depth.reshape(shp)


def _self_test(lay):
    """The decode must invert ``lay.idx`` exactly -- a wrong decode would
    bless a broken ring width, the worst failure mode (codex gate)."""
    rng = np.random.default_rng(0)
    for _ in range(64):
        k = int(rng.integers(len(lay.shapes)))
        m0, m1 = lay.shapes[k]
        face = int(rng.integers(6))
        i0, j0 = int(rng.integers(m0)), int(rng.integers(m1))
        f, d = _decode(lay, np.array([lay.idx(k, face, i0, j0)]))
        want_d = min(i0, m0 - 1 - i0, j0, m1 - 1 - j0)
        if int(f[0]) != face or int(d[0]) != want_d:
            raise AssertionError(
                f"duo SPMD census decode failed its self-test: "
                f"idx(k={k}, face={face}, i={i0}, j={j0}) decoded to "
                f"face={int(f[0])}, depth={int(d[0])} (want {face}, "
                f"{want_d})")


def _max_cross_depth(op, lay) -> int:
    """Max source depth over entries whose source face differs from the
    destination's.  -1 if the op never reads across faces."""
    if op is None:
        return -1
    if isinstance(op, (list, tuple)):
        return max((_max_cross_depth(o, lay) for o in op), default=-1)
    dst = getattr(op, "dst", None)
    if dst is None:
        raise TypeError(
            f"duo SPMD census: table op {type(op).__name__} has no 'dst' "
            f"-- a new op family the census does not understand; extend "
            f"it before sharding (fail closed, never guess)")
    dface, _ = _decode(lay, dst)
    worst = -1
    for a in ("src", "srcx", "srcy"):
        s = getattr(op, a, None)
        if s is None:
            continue
        sface, sdepth = _decode(lay, s)
        df = dface
        while df.ndim < sface.ndim:
            df = df[..., None]
        cross = sface != np.broadcast_to(df, sface.shape)
        if cross.any():
            worst = max(worst, int(sdepth[cross].max()))
    return worst


def census_cross_face_depth(tab) -> int:
    """Max depth-from-edge of any cross-face SOURCE cell over EVERY table
    in ``tab`` (a ``DuoHaloTables``).  Runs the decode self-test first.

    An unknown table attribute is a hard error, not a skip: a new op
    family added to the tables without extending this census could read
    deeper than the ring and silently consume zeros.
    """
    lays = {name: getattr(tab, name)
            for name in ("lay_a", "lay_a4", "lay_b", "lay_c", "lay_d",
                         "lay_fx", "lay_bb", "lay_wd", "lay_wc", "lay_ap")}
    for lay in lays.values():
        _self_test(lay)
    worst = -1
    for attr, layname in _TABLE_LAYOUTS.items():
        worst = max(worst,
                    _max_cross_depth(getattr(tab, attr), lays[layname]))
    for mode, layname in _CORNER_LAYOUTS.items():
        worst = max(worst,
                    _max_cross_depth(tab.corner[mode], lays[layname]))
    unknown = set(tab.corner) - set(_CORNER_LAYOUTS)
    if unknown:
        raise ValueError(
            f"duo SPMD census: corner modes {sorted(unknown)} have no "
            f"layout mapping -- extend _CORNER_LAYOUTS (fail closed)")
    return worst


def assert_ring_width_covers(tab, ring_width: int) -> int:
    """The build-time gate: refuse a ring narrower than the tables read.

    Returns the measured census so callers can log it.  ``ring_width``
    must EXCEED the census (>=, not >, would leave zero margin for the
    one-cell effective-footprint slack the design carries -- GLM's
    compounding-depth hole is closed by measurement at the composed
    level, but the raw gate keeps a +1 floor).
    """
    depth = census_cross_face_depth(tab)
    if ring_width <= depth:
        raise ValueError(
            f"duo SPMD ring width {ring_width} does not cover the "
            f"measured cross-face table depth {depth} for this "
            f"(n={tab.n}, ng={tab.ng}, ngp={tab.ngp}) -- the exchange "
            f"would silently read zeros. Widen the ring.")
    return depth
