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
    return build_six_face_duo_context(N, NG, use_ext_bundle=True)


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
