"""Units 5+6: the D-grid tail and the acoustic sub-step driver.

The end-to-end km=1 equivalence against ``full_acoustic_step_sixface`` is
NOT available here and must not be faked: that stepper drives the SW
adapters (``sw_dynamics=True, ptop=0, akap=1, cp_air=1``), which is the
shallow-water degenerate case, while this path runs the real hydrostatic
column. Per-phase km=1 equivalence is covered in units 3 and 4 where the
comparison is like-for-like.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_acoustic_3d import (
    acoustic_loop_3d,
    acoustic_substep_3d,
)
from legoesm.core.fv3_native_state_3d import build_state_3d, state_signature

N, NG = 12, 3
KM = 3
PTOP, AKAP, CP = 100.0, 0.2857142857142857, 1004.6
P_SFC = 1.0e5      # a real surface pressure; see _state
DT = 30.0


@pytest.fixture(scope="module")
def ctx():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    # use_ext_bundle=True is the FAITHFUL duo path (ext_scalar/ext_vector
    # k2e machinery). Its default is False -- the interim index-copy
    # exchanges -- and an unpassed default has already cost this campaign
    # one measurement, so it is explicit here.
    # oracle_conventions=True is the ONLY lane the oracle can produce:
    # fv_arrays.F90:1512 sets bounded_domain = regional .or. nested .or.
    # duogrid, so duogrid FORCES bounded_domain=.true., and
    # fv_grid_utils.F90:224 then leaves all four corner flags .false.
    # With bounded_domain=False the port runs blocks upstream skips --
    # d_sw4's corner-KE fix (sw_core.F90:1441) and d_sw1's west/east
    # edge blocks (sw_core.F90:656), both guarded on .not.bounded --
    # and reads 1e8 cube-vertex cosa/rsina where the bounded builder
    # has the real 120-degree kink values (cosa=-1/2, rsina=4/3).
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


def _state(km, seed=0):
    """A quiescent, hydrostatically SANE column.

    The layer masses must sum to a real surface pressure. An earlier
    version used delp = 1000 Pa per level, giving ps ~ 3100 Pa -- about 3%
    of an atmosphere. The pressure-gradient and geopotential terms are then
    grossly out of balance and the solver answers with |u| ~ 1e11, which is
    the CORRECT response to a nonsense state, not a dycore defect. Each
    layer therefore carries (p_sfc - ptop)/km.
    """
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, km)
    for t, face in enumerate(st):
        ii = np.arange(face["delp"].shape[0])[:, None]
        jj = np.arange(face["delp"].shape[1])[None, :]
        dp0 = (P_SFC - PTOP) / km          # ~33 000 Pa at km=3
        for k in range(km):
            # Horizontal STRUCTURE is required: with a horizontally uniform
            # delp/pt the halo exchange is a genuine no-op and the
            # first_substep gate test cannot detect anything. The
            # perturbation is ~0.01% of the layer mass -- enough to make the
            # exchange observable, small enough to stay hydrostatic.
            face["delp"][:, :, k] = (dp0
                                     + 2.0 * np.sin(0.3 * ii + 0.2 * jj)
                                     + 0.5 * t)
            face["pt"][:, :, k] = (280.0 + 2.0 * k
                                   + 0.3 * np.cos(0.25 * ii - 0.15 * jj))
            face["u"][:, :, k] = 1e-2 * rng.standard_normal(
                face["u"].shape[:2])
            face["v"][:, :, k] = 1e-2 * rng.standard_normal(
                face["v"].shape[:2])
    return st


def _finite(a):
    a = np.asarray(a)
    return a[np.isfinite(a)]


def test_one_substep_runs_and_changes_the_prognostic_state(ctx):
    st = _state(KM)
    before = state_signature(st)
    acoustic_substep_3d(ctx, st, DT, KM, first_substep=True,
                        ptop=PTOP, akap=AKAP, cp_air=CP)
    assert state_signature(st) != before, "the sub-step was a no-op"


def test_state_stays_finite_in_the_compute_window(ctx):
    st = _state(KM)
    acoustic_substep_3d(ctx, st, DT, KM, first_substep=True,
                        ptop=PTOP, akap=AKAP, cp_air=CP)
    bd = ctx["bd"]
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    ni, nj = bd.ie - bd.is_ + 1, bd.je - bd.js + 1
    for t in range(6):
        for name in ("delp", "pt"):
            win = st[t][name][i0:i0 + ni, j0:j0 + nj, :]
            assert np.all(np.isfinite(win)), (
                f"face {t + 1} {name} non-finite in the compute window")


def test_layer_masses_stay_positive(ctx):
    """delp <= 0 is the failure that makes log-pressure and the hydrostatic
    chain ill-conditioned; catch it here rather than as a NaN later."""
    st = _state(KM)
    acoustic_substep_3d(ctx, st, DT, KM, first_substep=True,
                        ptop=PTOP, akap=AKAP, cp_air=CP)
    bd = ctx["bd"]
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    ni, nj = bd.ie - bd.is_ + 1, bd.je - bd.js + 1
    for t in range(6):
        win = st[t]["delp"][i0:i0 + ni, j0:j0 + nj, :]
        assert win.min() > 0.0, f"face {t + 1}: non-positive layer mass"


def test_the_first_substep_scalar_exchange_gate_is_real(ctx):
    """dyn_core.F90:465 gates the delp/pt exchange on it == 1 while the
    winds exchange every sub-step. If first_substep were ignored, these two
    would agree -- so a difference proves the gate is wired."""
    a = _state(KM, seed=2)
    b = _state(KM, seed=2)
    acoustic_substep_3d(ctx, a, DT, KM, first_substep=True,
                        ptop=PTOP, akap=AKAP, cp_air=CP)
    acoustic_substep_3d(ctx, b, DT, KM, first_substep=False,
                        ptop=PTOP, akap=AKAP, cp_air=CP)
    diffs = [np.abs(_finite(a[t]["delp"]) - _finite(b[t]["delp"])).max()
             for t in range(6)]
    assert max(diffs) > 0.0, (
        "first_substep=True/False gave identical results -- the it==1 "
        "scalar-exchange gate is not wired")


def test_acoustic_loop_runs_multiple_substeps(ctx):
    st = _state(KM, seed=3)
    acoustic_loop_3d(ctx, st, dt_atmos=60.0, km=KM, n_split=2,
                     ptop=PTOP, akap=AKAP, cp_air=CP)
    bd = ctx["bd"]
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    ni, nj = bd.ie - bd.is_ + 1, bd.je - bd.js + 1
    for t in range(6):
        win = st[t]["delp"][i0:i0 + ni, j0:j0 + nj, :]
        assert np.all(np.isfinite(win)) and win.min() > 0.0


def test_n_split_changes_the_answer(ctx):
    """Non-vacuity for the loop: dt = dt_atmos/n_split, so n_split=1 and
    n_split=2 over the same dt_atmos must differ."""
    a = _state(KM, seed=5)
    b = _state(KM, seed=5)
    acoustic_loop_3d(ctx, a, dt_atmos=60.0, km=KM, n_split=1,
                     ptop=PTOP, akap=AKAP, cp_air=CP)
    acoustic_loop_3d(ctx, b, dt_atmos=60.0, km=KM, n_split=2,
                     ptop=PTOP, akap=AKAP, cp_air=CP)
    assert state_signature(a) != state_signature(b)


def test_loop_rejects_a_bad_n_split(ctx):
    with pytest.raises(ValueError, match="n_split"):
        acoustic_loop_3d(ctx, _state(KM), dt_atmos=60.0, km=KM, n_split=0,
                         ptop=PTOP, akap=AKAP, cp_air=CP)


def test_km_above_the_remap_window_is_refused(ctx):
    with pytest.raises(ValueError, match="fv_mapz"):
        acoustic_loop_3d(ctx, _state(4), dt_atmos=60.0, km=9, n_split=1,
                         ptop=PTOP, akap=AKAP, cp_air=CP)


# ---------------------------------------------------------------------------
# NH lane (units 7-9 wiring): update_dz_c/Riem_Solver_c/NH p_grad_c on the
# C stage, the d_sw w arms, update_dz_d/Riem_Solver3/pk3_halo/nh_p_grad on
# the D stage.  These are INTEGRATION smoke certificates (finiteness,
# positivity, sane w, cadence non-vacuity) -- the numerics certificates
# live per-kernel in test_fv3_native_nh_core / test_fv3_native_pgrad, and
# the end gate is the full-step oracle parity run.
# ---------------------------------------------------------------------------


def _nh_state(km, seed=0):
    """The quiescent column of ``_state``, converted to the NH acoustic
    lane's own units, plus ``make_nh``'s ``delz`` and ``w = 0``.

    TWO unit facts, both measured the expensive way (a raw-T fixture
    made SIM1 answer with |w| = 391 m/s -- the CORRECT response to a
    pressure field ~p^kappa out of balance, not a solver defect):

    * dyn_core's ``pt`` is THETA_V in the p0=1 convention
      (``fv_dynamics.F90:396-408``: pt = pt/pkz), so the EOS
      ``(-dm/dz * rgas * pt)^gama`` lands back on the full pressure;
    * ``delz`` must be the make_nh hydrostatic form
      ``-(rgas/grav) * T * dpeln`` (``init_hydro.F90:147-158``), the
      exact discrete balance the Riemann solve inverts.
    """
    from legoesm.core.fv3_native_state_3d import field_shape
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_GRAV,
        FV3_KAPPA,
        FV3_RDGAS,
    )

    st = _state(km, seed=seed)
    for face in st:
        face["delz"] = np.zeros(field_shape("delz", N, NG, km))
        m_a = face["delp"].shape[0]
        pe = np.full((m_a, m_a), PTOP)
        for k in range(km):
            dp = face["delp"][:, :, k]
            tt = np.array(face["pt"][:, :, k], copy=True)   # T [K]
            pe_top = pe
            pe_bot = pe + dp
            dpeln = np.log(pe_bot) - np.log(pe_top)
            pkz = ((np.exp(FV3_KAPPA * np.log(pe_bot))
                    - np.exp(FV3_KAPPA * np.log(pe_top)))
                   / (FV3_KAPPA * dpeln))
            face["pt"][:, :, k] = tt / pkz                  # theta_v, p0=1
            win = slice(NG, NG + N)
            face["delz"][:, :, k] = (
                -(FV3_RDGAS / FV3_GRAV) * tt[win, win]
                * dpeln[win, win])
            pe = pe_bot
    return st


def _nh_kwargs(ctx, km):
    from legoesm.core.fv3_native_acoustic_3d import build_nh_carry

    m_a = N + 2 * NG
    hs6 = [np.zeros((m_a, m_a)) for _ in range(6)]
    return dict(hydrostatic=False,
                nh=build_nh_carry(ctx, km, hs6),
                dp0=np.full(km, (P_SFC - PTOP) / km))


def test_nh_substep_runs_and_stays_sane(ctx):
    st = _nh_state(KM)
    before = state_signature(st)
    acoustic_substep_3d(ctx, st, DT, KM, first_substep=True,
                        ptop=PTOP, akap=AKAP, cp_air=CP,
                        **_nh_kwargs(ctx, KM))
    assert state_signature(st) != before, "the NH sub-step was a no-op"
    bd = ctx["bd"]
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    for t in range(6):
        for name in ("delp", "pt", "u", "v", "w"):
            win = st[t][name][i0:i0 + N, j0:j0 + N, :]
            assert np.all(np.isfinite(win)), (t, name)
        assert st[t]["delp"][i0:i0 + N, j0:j0 + N, :].min() > 0.0, t
        assert np.all(np.isfinite(st[t]["delz"])), t
        assert st[t]["delz"].max() < 0.0, (
            f"face {t + 1}: delz must stay strictly negative")
        # A near-balanced rest column answers with a bounded acoustic
        # transient, not tens of m/s.
        wmax = np.abs(st[t]["w"][i0:i0 + N, j0:j0 + N, :]).max()
        assert wmax < 50.0, (t, wmax)


def test_nh_loop_two_substeps_and_carry_cadence(ctx):
    st = _nh_state(KM, seed=3)
    kw = _nh_kwargs(ctx, KM)
    acoustic_loop_3d(ctx, st, dt_atmos=2 * DT, km=KM, n_split=2,
                     ptop=PTOP, akap=AKAP, cp_air=CP, **kw)
    bd = ctx["bd"]
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    for t in range(6):
        win = st[t]["delp"][i0:i0 + N, j0:j0 + N, :]
        assert np.all(np.isfinite(win)) and win.min() > 0.0, t
        assert np.all(np.isfinite(st[t]["delz"])), t
    # The carry did its job across substeps: zh evolved away from the
    # first-substep gz seed and holds finite heights.
    zh = kw["nh"]["zh6"][0]
    assert np.all(np.isfinite(zh[i0:i0 + N, j0:j0 + N, :]))
    assert np.abs(zh[i0:i0 + N, j0:j0 + N, 0]).max() > 1.0e3, (
        "zh top should sit kilometres above a zs=0 surface")


def test_nh_substep_requires_carry_and_dp0(ctx):
    st = _nh_state(KM)
    with pytest.raises(ValueError, match="nh carry"):
        acoustic_substep_3d(ctx, st, DT, KM, first_substep=True,
                            ptop=PTOP, akap=AKAP, cp_air=CP,
                            hydrostatic=False)


def test_nh_state_builder_carries_delz():
    from legoesm.core.fv3_native_state_3d import validate_state_3d

    st = build_state_3d(N, NG, KM, hydrostatic=False)
    assert all("delz" in face for face in st)
    validate_state_3d(st, N, NG, KM)
    hy = build_state_3d(N, NG, KM)
    assert all("delz" not in face for face in hy)


def test_substeps_out_captures_one_independent_snapshot_per_substep(ctx):
    """``substeps_out`` is the per-sub-step twin of ``press_out``.

    Three things have to hold before a per-sub-step parity probe can use
    it, and each has its own way of silently passing: the list must have
    one entry per sub-step; the entries must be COPIES (appending the
    live dict six times gives n_split identical references that pass a
    length check); and consecutive entries must DIFFER (a capture taken
    before the sub-step instead of after would repeat the input state).
    """
    st = _nh_state(KM, seed=5)
    kw = _nh_kwargs(ctx, KM)
    subs: list = []
    acoustic_loop_3d(ctx, st, dt_atmos=3 * DT, km=KM, n_split=3,
                     ptop=PTOP, akap=AKAP, cp_air=CP,
                     substeps_out=subs, **kw)
    assert len(subs) == 3
    assert all(len(snap) == 6 for snap in subs)

    # COPIES, not references: the loop keeps mutating `st` after each
    # append, so a captured snapshot must not have followed it.
    for snap in subs[:-1]:
        for t in range(6):
            assert snap[t]["w"] is not st[t]["w"]
    moved = max(float(np.abs(subs[0][t]["w"] - st[t]["w"]).max())
                for t in range(6))
    assert moved > 0.0, (
        "sub-step 1's snapshot equals the final state -- it is a live "
        "reference, not a copy")

    # DISTINCT sub-steps: consecutive snapshots must differ.
    for i in range(2):
        d = max(float(np.abs(subs[i][t]["w"] - subs[i + 1][t]["w"]).max())
                for t in range(6))
        assert d > 0.0, f"sub-step {i + 1} and {i + 2} captured the same w"

    # And nothing is captured unless asked.
    st2 = _nh_state(KM, seed=5)
    acoustic_loop_3d(ctx, st2, dt_atmos=3 * DT, km=KM, n_split=3,
                     ptop=PTOP, akap=AKAP, cp_air=CP,
                     **_nh_kwargs(ctx, KM))
