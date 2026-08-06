"""Unit 4: D-grid transport with the two duo barriers, km-general.

The barrier is the thing a monolithic d_sw cannot express, so these tests
target the barrier semantics specifically: that it actually couples faces,
that it runs per level with all six faces present, that it leaves the
w/q_con slots untouched, and that km=1 through the 3-D path reproduces the
certified km=1 sequence exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_cgrid_phase_3d import csw_phase_3d
from legoesm.core.fv3_native_dsw_phase_3d import dsw_transport_phase_3d
from legoesm.core.fv3_native_state_3d import build_state_3d

N, NG = 12, 3
KM = 3
DT = 30.0


@pytest.fixture(scope="module")
def ctx():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    return build_six_face_duo_context(N, NG)


def _state(km, seed=0):
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, km)
    for t, face in enumerate(st):
        for name in face:
            a = face[name]
            for k in range(km):
                a[:, :, k] = (0.01 * t + 0.1 * k
                              + 1e-3 * rng.standard_normal(a.shape[:2]))
        face["delp"][:] = np.abs(face["delp"]) + 1.0
        face["pt"][:] = np.abs(face["pt"]) + 280.0
    return st


def _finite_delta(a, b):
    """max |a-b| over entries finite in BOTH.

    allflux carries NaN in the corner-diagonal sentinel regions, and
    `np.abs(a-b).max()` is then NaN -- which makes `> 0` evaluate FALSE and
    silently reports "no change". That is the nanmax hazard, and it made
    this file's coupling test report a working barrier as broken.
    Returns (delta, n_compared) so a caller can reject a vacuous 0-element
    comparison.
    """
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 0.0, 0
    return float(np.abs(a[m] - b[m]).max()), int(m.sum())


def _run(ctx, st, km, **kw):
    csw = csw_phase_3d(ctx, st, dt2=0.5 * DT, km=km)
    return dsw_transport_phase_3d(ctx, st, csw, dt=DT, km=km, **kw)


def test_shapes_and_levels(ctx):
    out = _run(ctx, _state(KM), KM)
    assert len(out) == 6
    m_a = N + 2 * NG
    for acc in out:
        assert acc["delp"].shape == (m_a, m_a, KM)
        assert acc["pt"].shape == (m_a, m_a, KM)
        assert acc["allflux_x"].shape[2] == KM
        assert acc["allflux_y"].shape[2] == KM
        # "levels" carries the per-level d_sw1 stage dicts forward for
        # units 5-6; it is not an array.
        assert isinstance(acc["levels"], list) and len(acc["levels"]) == KM
        for nm, a in acc.items():
            if nm == "levels":
                continue
            assert a.dtype == np.float64, f"{nm} is {a.dtype}"


def test_km1_matches_the_certified_km1_sequence(ctx):
    """Stage-3 check: the 3-D assembler must add cadence, not change math.
    Reproduces d_sw1 -> average_allflux_shared_edges -> d_sw2 by hand and
    requires bit equality."""
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo, d_sw2_duo
    from legoesm.core.fv3_native_dsw_phase_3d import DUO_DECK_CFG
    from legoesm.grids.fv3_native_gridstruct import (
        average_allflux_shared_edges,
    )

    st = _state(1, seed=4)
    got = _run(ctx, st, 1)

    csw = csw_phase_3d(ctx, st, dt2=0.5 * DT, km=1)
    bd, npx, m_a = ctx["bd"], N + 1, N + 2 * NG
    c = DUO_DECK_CFG
    s1 = [d_sw1_duo(
        st[t]["delp"][:, :, 0], st[t]["pt"][:, :, 0], st[t]["w"][:, :, 0],
        csw[t]["uc"][:, :, 0], csw[t]["vc"][:, :, 0],
        np.zeros((npx, m_a)), np.zeros((m_a, npx)),
        np.zeros((npx, m_a)), np.zeros((m_a, npx)),
        ctx["gs6"][t], bd, npx, npx, dt=DT,
        hord_tr=c["hord_tr"], hord_vt=c["hord_vt"], hord_tm=c["hord_tm"],
        hord_dp=c["hord_dp"], nord_v=c["nord_v"], nord_t=0,
        damp_v=c["damp_v"], damp_t=0.0, workspace_sentinel=0.0)
        for t in range(6)]
    average_allflux_shared_edges([o["allflux_x"] for o in s1],
                                 [o["allflux_y"] for o in s1], 1, N, NG)
    for t in range(6):
        ref = d_sw2_duo(s1[t]["delp"], s1[t]["pt"],
                        s1[t]["allflux_x"], s1[t]["allflux_y"],
                        ctx["gs6"][t], bd)
        for name in ("delp", "pt"):
            np.testing.assert_array_equal(
                got[t][name][:, :, 0], ref[name],
                err_msg=f"face {t + 1} {name}: 3-D path diverged from the "
                        f"certified km=1 d_sw1/barrier/d_sw2 sequence")


def test_the_barrier_actually_couples_faces(ctx):
    """Non-vacuity for the whole unit. Perturb face 1 only; a neighbour
    sharing an edge MUST change, because the seam flux is averaged. If the
    barrier were a no-op this passes silently in the km=1 lane and fails
    here."""
    base = _run(ctx, _state(KM, seed=7), KM)
    st2 = _state(KM, seed=7)
    st2[0]["delp"][:, :, 0] += 0.25
    pert = _run(ctx, st2, KM)
    moved = []
    for t in range(1, 6):
        dx, nx = _finite_delta(pert[t]["allflux_x"], base[t]["allflux_x"])
        dy, ny = _finite_delta(pert[t]["allflux_y"], base[t]["allflux_y"])
        assert nx > 0 and ny > 0, (
            f"face {t + 1}: nothing finite to compare -- the assertion "
            f"below would be vacuous")
        if dx > 0 or dy > 0:
            moved.append(t)
    assert moved, ("no neighbouring face responded to a face-1 perturbation "
                   "-- the inter-panel barrier is not coupling anything")


def test_barrier_leaves_the_w_and_qcon_slots_untouched(ctx):
    """dyn_core averages iq==1, iq==4 and iq>4 only; slots 2 (w) and 3
    (q_con) must stay byte-identical."""
    from legoesm.grids.fv3_native_gridstruct import (
        average_allflux_shared_edges,
    )
    rng = np.random.default_rng(11)
    afx6 = [rng.standard_normal((N + 1, N, 5)) for _ in range(6)]
    afy6 = [rng.standard_normal((N, N + 1, 5)) for _ in range(6)]
    bx = [a.copy() for a in afx6]
    by = [a.copy() for a in afy6]
    average_allflux_shared_edges(afx6, afy6, 1, N, NG)
    for t in range(6):
        for slot in (1, 2):                       # iq = 2 and 3, 0-based
            np.testing.assert_array_equal(afx6[t][:, :, slot],
                                          bx[t][:, :, slot])
            np.testing.assert_array_equal(afy6[t][:, :, slot],
                                          by[t][:, :, slot])
        # ...and the averaged slots MUST have moved, else this is vacuous
        assert not np.array_equal(afx6[t][:, :, 0], bx[t][:, :, 0])


def test_each_level_is_barriered_against_its_own_level(ctx):
    """Perturb one level of one face; other LEVELS must not move, even
    though other FACES may. Catches a k/slot axis confusion in the barrier
    lift -- the allflux slot axis sits exactly where the oracle's k axis
    is, which is the easy way to get this wrong."""
    base = _run(ctx, _state(KM, seed=9), KM)
    st2 = _state(KM, seed=9)
    st2[0]["delp"][:, :, 2] += 0.25            # top level only
    pert = _run(ctx, st2, KM)
    d2, n2 = _finite_delta(pert[0]["delp"][:, :, 2], base[0]["delp"][:, :, 2])
    d0, n0 = _finite_delta(pert[0]["delp"][:, :, 0], base[0]["delp"][:, :, 0])
    d1, n1 = _finite_delta(pert[0]["delp"][:, :, 1], base[0]["delp"][:, :, 1])
    assert n0 and n1 and n2, "nothing finite to compare"
    assert d2 > 0.0, "the perturbed level must respond"
    assert d0 == 0.0 and d1 == 0.0, "other levels must be untouched"


def test_km_above_the_remap_window_is_refused(ctx):
    st = _state(1)
    with pytest.raises(ValueError, match="fv_mapz"):
        dsw_transport_phase_3d(ctx, st, [], dt=DT, km=9)
