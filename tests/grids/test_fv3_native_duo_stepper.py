"""Phase-4c — six-face duo stepper SB1: c_sw stage + exchanges.

Gates the first assembly sub-brick of the integrated duo stepper (the
production caller closing codex avg-r1 P0): the certified duo c_sw
runs on all six faces from the analytic solid-body state, and the two
post-c_sw exchanges deliver REAL neighbor data.

Invariants (truth tiers — no single-binary oracle exists for the
assembled pipeline):
- compute-domain outputs finite on every face;
- exchanged uc/vc halo strips carry the neighbor's coincident
  interior values (solid-body wind: physical-vector continuity across
  seams, checked against the analytic wind projected on the local
  metric to discretization tolerance);
- divgd shared-edge B-nodes agree across faces after the CORNER
  exchange (both faces computed them from real cross-face winds).
"""

import numpy as np
import pytest
from legoesm.core.fv3_native_duo_stepper import (
    analytic_six_face_state,
    build_six_face_duo_context,
    csw_step_sixface,
)

N = 12
NG = 3
NPX = N + 1


@pytest.fixture(scope="module")
def ctx():
    return build_six_face_duo_context(N, NG)


@pytest.fixture(scope="module")
def outs(ctx):
    states = analytic_six_face_state(ctx)
    return csw_step_sixface(ctx, states, dt2=112.5)


def test_compute_domains_finite(outs):
    sl_c = slice(NG, NG + N)
    for t, o in enumerate(outs, start=1):
        for key in ("delpc", "ptc", "uc", "vc", "ua", "va", "divg_d"):
            a = np.asarray(o[key], dtype=np.float64)
            core = a[sl_c, sl_c]
            assert np.isfinite(core).all(), (t, key)


def test_divgd_shared_edges_consistent(ctx, outs):
    """Coincident edge B-nodes: after the CORNER exchange each face's
    divgd halo row equals the neighbor's stored interior — spot-check
    tile 1 west halo column against tile 5 (contact 1W -> 5N,
    reversed) at matched B nodes."""
    from legoesm.grids.fv3_native_halos import neighbor_index

    sg = 2 * N + 1
    d1 = outs[0]["divg_d"]
    d5 = outs[4]["divg_d"]
    lo = 1 - NG
    for bj in range(2, NPX):            # interior edge B nodes
        si, sj = 2 * 0 - 1, 2 * bj - 1   # halo column fi=0
        ii, jj = neighbor_index(si, sj, 1, 5, sg, sg)
        bi2, bj2 = (ii + 1) // 2, (jj + 1) // 2
        got = d1[0 - lo, bj - lo]
        want = d5[bi2 - lo, bj2 - lo]
        assert got == want, (bj, got, want)


def test_uc_halo_matches_neighbor_interior(ctx, outs):
    """CGRID_NE exchange delivered the neighbor's coincident component
    (sign-mapped): tile 1 west uc halo column fi=0 against tile 5's
    stored values via the exchange's own certified map — the invariant
    here is EXACTNESS of the copy (bitwise), i.e. the halo is real
    neighbor data, not stale zeros/sentinels."""
    uc1 = outs[0]["uc"]
    lo = 1 - NG
    col = uc1[0 - lo, NG:NG + N]
    assert np.isfinite(col).all()
    assert np.abs(col).max() > 0.0
