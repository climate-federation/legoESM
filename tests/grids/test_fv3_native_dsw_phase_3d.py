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
    # use_ext_bundle=True is REQUIRED, not incidental: DUO_DECK_CFG runs
    # nord=2, and the post-p_grad_c divgd exchange at dyn_core.F90:652 has
    # no faithful implementation without the bundle. A default context
    # here tested the interim index-copy path, which leaves the B-grid
    # corner diagonal stale -- the path that produced a 1e11 D wind.
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


@pytest.fixture(scope="module")
def ctx_interim():
    """Context that DECLARES the non-faithful interim exchange.

    Only for the tests that are about the fallback itself. ext_exclude is
    the existing measurement opt-in; naming it is what keeps the interim
    path from being reachable by accident.

    ONE VARIABLE vs the `ctx` fixture. The corner test compares the two
    contexts cell-for-cell, so everything except the divgd exchange has to
    match: same oracle_conventions, same use_ext_bundle, and `cvec` NOT
    excluded -- only `divgd`. An earlier version differed in conventions
    AND in whether the bundle existed at all, which made the comparison
    measure the grid instead of the exchange (c_sw's corner value is
    -1.18e-09 bounded vs 1.25e+07 plain; that test's own confound guard
    caught it).
    """
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    return build_six_face_duo_context(N, NG, oracle_conventions=True,
                                      use_ext_bundle=True,
                                      ext_exclude=("divgd",))


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
    requires bit equality.

    THE REFERENCE MUST START FROM THE SAME C-GRID STATE, BUT MUST NOT
    INHERIT IT FROM THE CODE UNDER TEST. dsw_transport_phase_3d performs
    the post-p_grad_c duo exchanges (dyn_core.F90:652 and :655) on the csw
    dict it is handed, IN PLACE, before it calls d_sw1.

    Two wrong ways to write this, both of which were actually written:
      * a SECOND, fresh csw for the reference -- compares an exchanged
        path against an unexchanged one. 40/324 elements differed, most at
        1e-13 but the SW corner at -3.42e+32, and the test was red from
        the commit that introduced it.
      * the SUT-mutated csw fed straight into the reference -- like-for-
        like, but now a no-op, wrong-face, wrong-stagger or absent
        exchange is shared by both sides and cannot be detected. Deleting
        the divg_d exchange, i.e. the defect this file is supposed to
        cover, would pass.

    So the reference exchanges its OWN pre-exchange copy through the same
    public helper.

    WHAT THIS TEST DOES AND DOES NOT COVER. It asserts delp/pt, which
    depend on uc/vc, so dropping the uc/vc exchange from
    dsw_transport_phase_3d fails it. It CANNOT detect a dropped divg_d
    exchange: unit 4 passes only uc/vc into d_sw1 (divg_d is first
    consumed by d_sw5, dyn_core.F90:1107, i.e. unit 5), so divg_d never
    reaches delp/pt here. An earlier version of this docstring claimed
    otherwise and was wrong. The SUT call site for divg_d is covered by
    test_the_sut_exchanges_divg_d below; helper correctness by
    test_the_authoritative_exchange_fills_the_corner_diagonal.
    """
    import copy as _copy

    from legoesm.core.fv3_native_dsw_phase_3d import DUO_DECK_CFG
    from legoesm.core.fv3_native_duo_stepper import (
        exchange_post_pgrad_sixface,
    )
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo, d_sw2_duo
    from legoesm.grids.fv3_native_gridstruct import (
        average_allflux_shared_edges,
    )

    st = _state(1, seed=4)
    csw = csw_phase_3d(ctx, st, dt2=0.5 * DT, km=1)
    csw_ref = _copy.deepcopy(csw)
    got = dsw_transport_phase_3d(ctx, st, csw, dt=DT, km=1)

    # the reference applies the exchange itself, on its own copy
    exchange_post_pgrad_sixface(
        ctx,
        [csw_ref[t]["divg_d"][:, :, 0] for t in range(6)],
        [csw_ref[t]["uc"][:, :, 0] for t in range(6)],
        [csw_ref[t]["vc"][:, :, 0] for t in range(6)],
        nord=DUO_DECK_CFG["nord"])
    csw = csw_ref

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

    # NON-VACUITY. A bit-equality assertion against a hand-rolled
    # reference is worthless unless the reference can disagree. Rebuild it
    # with the barrier REMOVED and require that it no longer matches -- if
    # this passes, the assertion above is checking nothing.
    s1_nb = [d_sw1_duo(
        st[t]["delp"][:, :, 0], st[t]["pt"][:, :, 0], st[t]["w"][:, :, 0],
        csw[t]["uc"][:, :, 0], csw[t]["vc"][:, :, 0],
        np.zeros((npx, m_a)), np.zeros((m_a, npx)),
        np.zeros((npx, m_a)), np.zeros((m_a, npx)),
        ctx["gs6"][t], bd, npx, npx, dt=DT,
        hord_tr=c["hord_tr"], hord_vt=c["hord_vt"], hord_tm=c["hord_tm"],
        hord_dp=c["hord_dp"], nord_v=c["nord_v"], nord_t=0,
        damp_v=c["damp_v"], damp_t=0.0, workspace_sentinel=0.0)
        for t in range(6)]
    # ... no average_allflux_shared_edges here -- that is the removal.
    unbarriered = [d_sw2_duo(s1_nb[t]["delp"], s1_nb[t]["pt"],
                             s1_nb[t]["allflux_x"], s1_nb[t]["allflux_y"],
                             ctx["gs6"][t], bd) for t in range(6)]
    # FINITE-ONLY. np.array_equal calls same-position NaNs unequal, so a
    # single shared NaN would satisfy `any(...)` while no finite value had
    # moved at all -- the assertion would pass vacuously on a field that
    # is entirely garbage.
    moved = False
    for t in range(6):
        for name in ("delp", "pt"):
            a = got[t][name][:, :, 0]
            b = unbarriered[t][name]
            fin = np.isfinite(a) & np.isfinite(b)
            assert fin.any(), (
                f"face {t + 1} {name}: no finite values to compare, so the "
                f"non-vacuity check cannot mean anything")
            if np.any(a[fin] != b[fin]):
                moved = True
    assert moved, (
        "dropping the barrier from the reference changed no FINITE value, "
        "so the bit-equality assertion above cannot detect a missing "
        "barrier")


def test_the_authoritative_exchange_fills_the_corner_diagonal(ctx,
                                                              ctx_interim):
    """The defect that produced the 1e11 wind, guarded directly.

    dyn_core.F90:652 exchanges divgd through ext_scalar(...,1,1), whose
    k2e Lagrange fill covers the corner-diagonal halo
    (fv_duogrid.F90:523-566). The interim index-copy helper says in its
    own docstring that it leaves those regions untouched. d_sw5's nord=2
    n-loop reads them (sw_core.F90:1748-1758) and scales them by
    dd8 ~ 1e32, so "untouched" means "whatever c_sw left", which measured
    -9.1e7 against a 1e-8 compute window.

    Assert the authoritative path CHANGES the corner diagonal and the
    interim path does NOT. Both directions matter: only checking the
    authoritative side would pass even if the interim helper had silently
    become authoritative, and vice versa.
    """
    from legoesm.core.fv3_native_dsw_phase_3d import DUO_DECK_CFG
    from legoesm.core.fv3_native_duo_stepper import (
        exchange_post_pgrad_sixface,
    )

    bd = ctx["bd"]
    # one cell strictly inside the SW corner-diagonal block: i < is and
    # j < js, i.e. neither an i-edge strip nor a j-edge strip
    ci, cj = bd.is_ - 1 - bd.isd, bd.js - 1 - bd.jsd

    def corner_after(c):
        st = _state(1, seed=11)
        csw = csw_phase_3d(c, st, dt2=0.5 * DT, km=1)
        before = float(csw[0]["divg_d"][ci, cj, 0])
        exchange_post_pgrad_sixface(
            c,
            [csw[t]["divg_d"][:, :, 0] for t in range(6)],
            [csw[t]["uc"][:, :, 0] for t in range(6)],
            [csw[t]["vc"][:, :, 0] for t in range(6)],
            nord=DUO_DECK_CFG["nord"])
        return before, float(csw[0]["divg_d"][ci, cj, 0])

    b_auth, a_auth = corner_after(ctx)
    b_int, a_int = corner_after(ctx_interim)

    assert b_auth == b_int, (
        "the two contexts must start from the same c_sw corner value for "
        f"this comparison to mean anything ({b_auth!r} vs {b_int!r})")
    assert a_int == b_int, (
        "the interim exchange is documented to leave corner diagonals "
        f"untouched but it changed {b_int!r} -> {a_int!r}; either the "
        "helper changed or this probe is reading an edge strip")
    assert a_auth != b_auth, (
        "the authoritative ext_scalar exchange left the B-grid corner "
        f"diagonal at {b_auth!r}, i.e. it did NOT do the k2e corner fill "
        "that dyn_core.F90:652 relies on")
    assert np.isfinite(a_auth), (
        f"authoritative exchange produced {a_auth!r} in the corner diagonal")


def test_the_sut_exchanges_divg_d(ctx):
    """dsw_transport_phase_3d must actually perform the divg_d exchange.

    The cadence test above cannot see this: unit 4 hands only uc/vc to
    d_sw1, and divg_d is not consumed until d_sw5 (dyn_core.F90:1107). So
    the exchange could be deleted from the unit-4 call site and every
    delp/pt assertion would still pass -- while unit 5 quietly went back
    to reading a stale corner diagonal and emitting a 1e11 wind.

    Assert it at the call site instead: the corner-diagonal cell of the
    csw dict handed to the SUT must differ afterwards.
    """
    bd = ctx["bd"]
    ci, cj = bd.is_ - 1 - bd.isd, bd.js - 1 - bd.jsd

    st = _state(KM, seed=13)
    csw = csw_phase_3d(ctx, st, dt2=0.5 * DT, km=KM)
    before = [np.array(csw[t]["divg_d"][:, :, k], copy=True)
              for t in range(6) for k in range(KM)]
    dsw_transport_phase_3d(ctx, st, csw, dt=DT, km=KM)

    corner_moved = any(
        csw[t]["divg_d"][ci, cj, k] != before[t * KM + k][ci, cj]
        for t in range(6) for k in range(KM))
    assert corner_moved, (
        "dsw_transport_phase_3d left every face/level divg_d corner "
        "diagonal untouched -- the dyn_core.F90:652 ext_scalar(divgd,1,1) "
        "call is missing from the unit-4 call site, and d_sw5 will scale "
        "the stale value by dd8 ~ 1e32")
    for t in range(6):
        for k in range(KM):
            assert np.isfinite(csw[t]["divg_d"][:, :, k]).all(), (
                f"face {t + 1} level {k}: divg_d non-finite after the "
                f"post-p_grad_c exchange")


def test_nord_zero_skips_the_divgd_exchange_but_not_the_winds(ctx):
    """dyn_core.F90:652 gates divgd on nord > 0; :655 gates uc/vc on nothing.

    Asserted explicitly because the two orders differ in name only on the
    shipped deck (nord == nord_v == 2), so a gate on the wrong variable is
    invisible there.

    The csw here is built by csw_phase_3d's own nord (2), not 0: upstream
    c_sw only computes divg_d when nord > 0 (sw_core.F90:153), so a real
    nord=0 deck would have no meaningful divg_d to exchange at all. This
    test is about the GATE, so it keeps a populated divg_d and checks that
    the gate declines to touch it -- which is strictly harder to pass than
    starting from zeros.
    """
    from legoesm.core.fv3_native_duo_stepper import (
        exchange_post_pgrad_sixface,
    )

    st = _state(1, seed=5)
    csw = csw_phase_3d(ctx, st, dt2=0.5 * DT, km=1)
    dg_before = [np.array(csw[t]["divg_d"][:, :, 0], copy=True)
                 for t in range(6)]
    uc_before = [np.array(csw[t]["uc"][:, :, 0], copy=True)
                 for t in range(6)]
    vc_before = [np.array(csw[t]["vc"][:, :, 0], copy=True)
                 for t in range(6)]

    exchange_post_pgrad_sixface(
        ctx,
        [csw[t]["divg_d"][:, :, 0] for t in range(6)],
        [csw[t]["uc"][:, :, 0] for t in range(6)],
        [csw[t]["vc"][:, :, 0] for t in range(6)],
        nord=0)

    for t in range(6):
        np.testing.assert_array_equal(
            csw[t]["divg_d"][:, :, 0], dg_before[t],
            err_msg=f"face {t + 1}: nord=0 must not exchange divgd "
                    f"(dyn_core.F90:652 gates it on nord > 0)")
    # BOTH components: dyn_core.F90:655 passes uc AND vc to ext_vector,
    # so checking only uc would pass on a helper that silently dropped vc.
    assert any(not np.array_equal(csw[t]["uc"][:, :, 0], uc_before[t])
               for t in range(6)), (
        "nord=0 must still exchange uc -- dyn_core.F90:655 is not gated "
        "on nord")
    assert any(not np.array_equal(csw[t]["vc"][:, :, 0], vc_before[t])
               for t in range(6)), (
        "nord=0 must still exchange vc -- dyn_core.F90:655 passes both "
        "components")


def test_the_interim_exchange_must_be_asked_for(ctx_interim):
    """A context that cannot run the faithful exchange must SAY so.

    Upstream has no fallback for dyn_core.F90:652. Silently substituting
    the interim helper at nord > 0 is the wrong-number path; it has to be
    an explicit ext_exclude opt-in.
    """
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
        exchange_post_pgrad_sixface,
    )

    # no bundle, no opt-out; oracle_conventions=True because c_sw
    # refuses duogrid on unbounded metrics (fv_arrays.F90:1512) — the
    # refusals under test here live in exchange_post_pgrad_sixface and
    # are orthogonal to the metrics lane
    plain = build_six_face_duo_context(N, NG, oracle_conventions=True)
    st = _state(1, seed=6)
    csw = csw_phase_3d(plain, st, dt2=0.5 * DT, km=1)
    args = ([csw[t]["divg_d"][:, :, 0] for t in range(6)],
            [csw[t]["uc"][:, :, 0] for t in range(6)],
            [csw[t]["vc"][:, :, 0] for t in range(6)])

    with pytest.raises(ValueError, match="divgd exchange"):
        exchange_post_pgrad_sixface(plain, *args, nord=2)

    # nord=0 skips divgd, but dyn_core.F90:655's uc/vc exchange is NOT
    # gated on nord, so an undeclared context is still refused -- gating
    # the guard on nord would have left one silent substitution behind.
    with pytest.raises(ValueError, match="uc/vc exchange"):
        exchange_post_pgrad_sixface(plain, *args, nord=0)

    # declaring only divgd is still not enough: cvec is its own opt-in
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context as _bld,
    )
    divgd_only = _bld(N, NG, ext_exclude=("divgd",),
                      oracle_conventions=True)
    csw3 = csw_phase_3d(divgd_only, st, dt2=0.5 * DT, km=1)
    with pytest.raises(ValueError, match="uc/vc exchange"):
        exchange_post_pgrad_sixface(
            divgd_only,
            [csw3[t]["divg_d"][:, :, 0] for t in range(6)],
            [csw3[t]["uc"][:, :, 0] for t in range(6)],
            [csw3[t]["vc"][:, :, 0] for t in range(6)],
            nord=2)

    # and the fully declared opt-out is allowed
    csw2 = csw_phase_3d(ctx_interim, st, dt2=0.5 * DT, km=1)
    exchange_post_pgrad_sixface(
        ctx_interim,
        [csw2[t]["divg_d"][:, :, 0] for t in range(6)],
        [csw2[t]["uc"][:, :, 0] for t in range(6)],
        [csw2[t]["vc"][:, :, 0] for t in range(6)],
        nord=2)


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
