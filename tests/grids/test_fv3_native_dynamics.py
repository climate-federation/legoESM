"""Unit 7: the ``fv_dynamics`` outer step that joins dyn_core to fv_mapz.

The end-to-end oracle comparison lives in
``scripts/validate/fv3_native/full_step_oracle_parity.py`` (it needs the
C48 reference restart). What is testable here without that data is the
part the driver ITSELF owns: the ``p_var`` initialisation, the
``T -> theta_v`` round trip, the lane guards, and the one invariant that
proves the remap actually ran -- ``delp`` back on the reference
coordinate.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_dynamics import (
    PTOP_MIN,
    fv_dynamics_step,
    p_var_hydrostatic,
    pt_to_theta_v,
    require_uniform_damping_lane,
)
from legoesm.core.fv3_native_eta import set_eta_analytic
from legoesm.core.fv3_native_state_3d import build_state_3d, field_shape

N, NG = 12, 3
KM = 5
AKAP = 0.2857142857142857
CP = 1004.6
P_SFC = 1.0e5


@pytest.fixture(scope="module")
def eta():
    ak, bk, ptop, ks = set_eta_analytic(KM)
    return ak, bk, float(ptop), ks


@pytest.fixture(scope="module")
def ctx():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


def _state(ak, bk, ptop, km=KM, seed=0):
    """A hydrostatically sane column ON the reference coordinate.

    Starting exactly on the Eulerian coordinate is what makes the
    post-remap ``delp`` check meaningful: the acoustic loop deforms it and
    the remap must put it back.
    """
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, km, remap_follows=True)
    for face in st:
        for k in range(km):
            dp0 = (ak[k + 1] - ak[k]) + (bk[k + 1] - bk[k]) * P_SFC
            face["delp"][:, :, k] = dp0
            face["pt"][:, :, k] = 280.0 + 2.0 * rng.standard_normal(
                face["pt"].shape[:2])
            face["u"][:, :, k] = 1.0 * rng.standard_normal(
                face["u"].shape[:2])
            face["v"][:, :, k] = 1.0 * rng.standard_normal(
                face["v"].shape[:2])
    return st


def _press(state, ptop, km=KM):
    return [p_var_hydrostatic(f["delp"], ptop=ptop, akap=AKAP,
                              n=N, ng=NG, km=km) for f in state]


def _tracers(km=KM):
    # nr = 2 on the pinned deck (ncnst=3, dnats=1): the remap makes two
    # passes through fv_mapz.F90:330-342.
    return [[np.zeros(field_shape("delp", N, NG, km), dtype=np.float64)
             for _ in range(2)] for _ in range(6)]


# --------------------------------------------------------------------
# p_var
# --------------------------------------------------------------------

def test_p_var_reproduces_the_column_sum_and_surface_pressure(eta):
    ak, bk, ptop, _ = eta
    st = _state(ak, bk, ptop)
    g = p_var_hydrostatic(st[0]["delp"], ptop=ptop, akap=AKAP,
                          n=N, ng=NG, km=KM)
    win = st[0]["delp"][NG:NG + N, NG:NG + N, :]
    want = ptop + np.cumsum(win, axis=2)
    # pe is (i, k, j) with origin (is-1, 1, js-1)
    got = g["pe"][1:N + 1, 1:, 1:N + 1].transpose(0, 2, 1)
    assert np.array_equal(got, want)
    assert np.array_equal(g["ps"][NG:NG + N, NG:NG + N], want[:, :, -1])
    assert np.allclose(g["peln"][:, 0, :], np.log(ptop), rtol=0, atol=0)


def test_p_var_pkz_matches_geopks_independent_transcription(eta):
    """Two separate ports of the same formula must agree.

    ``init_hydro.F90:127-133`` and ``dyn_core.F90:2780-2786`` write the
    same ``pkz`` expression in different files; the port transcribes them
    separately (``p_var_hydrostatic`` here, ``geopk`` in
    ``fv3_native_pgrad``). Agreement is therefore evidence about the
    transcription, not a tautology.
    """
    from legoesm.core.fv3_native_pgrad import geopk
    from legoesm.core.fv3_native_sw_core import Bounds

    ak, bk, ptop, _ = eta
    st = _state(ak, bk, ptop)
    g = p_var_hydrostatic(st[0]["delp"], ptop=ptop, akap=AKAP,
                          n=N, ng=NG, km=KM)
    bd = Bounds.single_tile(N, NG)
    hs = np.zeros((N + 2 * NG, N + 2 * NG), dtype=np.float64)
    gk = geopk(st[0]["delp"], st[0]["pt"], hs, bd, km=KM, ptop=ptop,
               akap=AKAP, cp_air=CP, cg=False, duogrid=True,
               computehalo=False, npx=N + 1, npy=N + 1, a2b_ord=4,
               bounded_domain=False, sw_dynamics=False)
    assert np.allclose(g["pkz"], gk["pkz"], rtol=1e-14, atol=0.0)
    assert np.allclose(g["peln"], gk["peln"], rtol=1e-14, atol=0.0)


def test_p_var_rejects_a_bad_delp_shape_and_a_nonpositive_ptop(eta):
    ak, bk, ptop, _ = eta
    st = _state(ak, bk, ptop)
    with pytest.raises(ValueError, match="delp must be"):
        p_var_hydrostatic(st[0]["delp"][..., :-1], ptop=ptop, akap=AKAP,
                          n=N, ng=NG, km=KM)
    with pytest.raises(ValueError, match="ptop must be"):
        p_var_hydrostatic(st[0]["delp"], ptop=0.0, akap=AKAP,
                          n=N, ng=NG, km=KM)


def test_ptop_min_is_the_oracles_parameter():
    # fv_grid_utils.F90:56 -- real, parameter:: ptop_min = 1.d-8
    assert PTOP_MIN == 1.0e-8


# --------------------------------------------------------------------
# the theta_v conversion
# --------------------------------------------------------------------

def test_pt_conversion_touches_the_compute_window_only(eta):
    """fv_dynamics.F90:398-400 loops i=is,ie / j=js,je.

    The halo rows keep TEMPERATURE until the it==1 exchange
    (dyn_core.F90:470) overwrites them. A padded-array conversion would
    also convert the corner-diagonal sentinels, which nothing overwrites.
    """
    ak, bk, ptop, _ = eta
    st = _state(ak, bk, ptop)
    g = p_var_hydrostatic(st[0]["delp"], ptop=ptop, akap=AKAP,
                          n=N, ng=NG, km=KM)
    before = np.array(st[0]["pt"], copy=True)
    pt_to_theta_v(st[0]["pt"], g["pkz"], n=N, ng=NG)

    win = (slice(NG, NG + N), slice(NG, NG + N))
    assert np.allclose(st[0]["pt"][win], before[win] / g["pkz"],
                       rtol=0, atol=0)
    halo = np.ones(before.shape[:2], dtype=bool)
    halo[win] = False
    assert np.array_equal(st[0]["pt"][halo, :], before[halo, :])
    # ...and the conversion is not a no-op, or the check above is vacuous.
    assert not np.allclose(st[0]["pt"][win], before[win])


def test_pt_conversion_with_dp1_is_not_the_adiabatic_branch(eta):
    ak, bk, ptop, _ = eta
    st = _state(ak, bk, ptop)
    g = p_var_hydrostatic(st[0]["delp"], ptop=ptop, akap=AKAP,
                          n=N, ng=NG, km=KM)
    a = np.array(st[0]["pt"], copy=True)
    b = np.array(st[0]["pt"], copy=True)
    dp1 = np.full((N, N, KM), 0.01, dtype=np.float64)
    pt_to_theta_v(a, g["pkz"], n=N, ng=NG)
    pt_to_theta_v(b, g["pkz"], n=N, ng=NG, dp1=dp1)
    win = (slice(NG, NG + N), slice(NG, NG + N))
    assert np.allclose(b[win], a[win] * 1.01, rtol=1e-14, atol=0)


# --------------------------------------------------------------------
# lane guards
# --------------------------------------------------------------------

def test_sponge_deck_is_refused():
    # dyn_core.F90:775-804 gives k=1,2,3 their own nord_v/damp_vt.
    require_uniform_damping_lane(n_sponge=-1, tau=-1.0, npz=KM)
    require_uniform_damping_lane(n_sponge=2, tau=-1.0, npz=1)   # npz==1 arm
    with pytest.raises(NotImplementedError, match="n_sponge"):
        require_uniform_damping_lane(n_sponge=0, tau=-1.0, npz=KM)


def test_rayleigh_deck_is_refused():
    with pytest.raises(NotImplementedError, match="tau"):
        require_uniform_damping_lane(n_sponge=-1, tau=10.0, npz=KM)


def test_moist_validation_and_the_energy_fixer_refusal(ctx, eta):
    """zvir is SUPPORTED now; what this pins is that it is VALIDATED.

    This test asserted ``NotImplementedError`` on ``zvir != 0`` and was
    left stale by the commit that enabled moist coupling in this lane --
    the refusal it named had already become a validation error.
    """
    ak, bk, ptop, _ = eta
    st = _state(ak, bk, ptop)
    pr = _press(st, ptop)
    common = dict(bdt=60.0, km=KM, k_split=1, n_split=1, ptop=ptop,
                  ak=ak, bk=bk, akap=AKAP, cp_air=CP,
                  kord_mt=9, kord_tm=-9, kord_tr=9, q=_tracers())
    # no sphum_index -> a guessed index would couple the wrong species
    with pytest.raises(ValueError, match="sphum_index"):
        fv_dynamics_step(ctx, st, pr, zvir=0.61, **common)
    # POSITIVE consv_te runs now (the fixer is ported and certified);
    # what raises is the prescribed-flux branch at a negative value.
    with pytest.raises(NotImplementedError, match="consv_te"):
        fv_dynamics_step(ctx, st, pr, consv_te=-1.0, **common)


def test_face_count_is_checked(ctx, eta):
    ak, bk, ptop, _ = eta
    st = _state(ak, bk, ptop)
    pr = _press(st, ptop)
    with pytest.raises(ValueError, match="6 faces"):
        fv_dynamics_step(ctx, st[:5], pr, bdt=60.0, km=KM, k_split=1,
                         n_split=1, ptop=ptop, ak=ak, bk=bk, akap=AKAP,
                         cp_air=CP, kord_mt=9, kord_tm=-9, kord_tr=9,
                         q=_tracers())


# --------------------------------------------------------------------
# the step itself
# --------------------------------------------------------------------

@pytest.mark.slow
def test_the_remap_puts_delp_back_on_the_reference_coordinate(ctx, eta):
    """The one invariant that proves the remap ran, not just returned.

    ``fv_mapz.F90:267-285`` builds the target interfaces as
    ``pe2(k) = ak(k) + bk(k)*ps`` and writes ``delp = pe2(k+1) - pe2(k)``.
    So after a full step every layer mass must equal the hybrid
    thickness at that column's OWN new surface pressure -- a property the
    acoustic loop alone cannot produce, since it advects delp freely.
    """
    from legoesm.core.fv3_native_acoustic_3d import acoustic_loop_3d
    from legoesm.core.fv3_native_dynamics import pt_to_theta_v

    ak, bk, ptop, _ = eta
    w = (slice(NG, NG + N), slice(NG, NG + N))

    # NON-VACUITY, PART 1. The assertion below holds for the INITIAL delp
    # by construction (_state builds it on the reference coordinate), so
    # a full step that did nothing at all would pass it. Run the acoustic
    # loop ALONE on an identical state first and require that it has
    # actually taken delp OFF the coordinate -- otherwise there is nothing
    # for the remap to undo and the test proves nothing.
    st0 = _state(ak, bk, ptop, seed=3)
    pr0 = _press(st0, ptop)
    for t in range(6):
        pt_to_theta_v(st0[t]["pt"], pr0[t]["pkz"], n=N, ng=NG)
    acoustic_loop_3d(ctx, st0, 60.0, KM, n_split=2, ptop=ptop, akap=AKAP,
                     cp_air=CP, remap_follows=True)
    off = 0.0
    for t in range(6):
        for k in range(KM):
            want_k = ((ak[k + 1] - ak[k])
                      + (bk[k + 1] - bk[k]) * pr0[t]["ps"][w])
            off = max(off, float(np.abs(st0[t]["delp"][w][:, :, k]
                                        - want_k).max()))
    assert off > 1e-6, (
        f"the acoustic loop left delp within {off:g} Pa of the reference "
        f"coordinate, so the remap has nothing to undo and the check below "
        f"cannot fail")

    st = _state(ak, bk, ptop, seed=3)
    pr = _press(st, ptop)
    ic_pt = [np.array(f["pt"][w], copy=True) for f in st]
    out = fv_dynamics_step(ctx, st, pr, bdt=60.0, km=KM, k_split=1,
                           n_split=2, ptop=ptop, ak=ak, bk=bk, akap=AKAP,
                           cp_air=CP, kord_mt=9, kord_tm=-9, kord_tr=9,
                           q=_tracers())
    assert out["pt_units"] == "K"
    # NON-VACUITY, PART 2. pt must have MOVED. A driver that skipped both
    # the loop and the remap would return the IC with a "K" label.
    moved = max(float(np.abs(st[t]["pt"][w] - ic_pt[t]).max())
                for t in range(6))
    assert moved > 1e-6, (
        f"pt is within {moved:g} K of the initial condition -- the step did "
        f"not run, and every assertion below is vacuous")
    for t in range(6):
        ps = pr[t]["ps"][w]
        delp = st[t]["delp"][w]
        for k in range(KM):
            want = (ak[k + 1] - ak[k]) + (bk[k + 1] - bk[k]) * ps
            assert np.allclose(delp[:, :, k], want, rtol=1e-13, atol=0), \
                f"face {t + 1} level {k} is off the reference coordinate"
        assert np.all(np.isfinite(st[t]["pt"][w]))
        # pt came back to a temperature, not theta_v (~1.4x larger here).
        assert 150.0 < st[t]["pt"][w].min() < st[t]["pt"][w].max() < 400.0


@pytest.mark.slow
def test_km_below_the_remap_gate_leaves_pt_in_theta_v(eta):
    """fv_dynamics.F90:568 gates the remap on npz > 4.

    Below it the ORACLE does not remap either, so the driver must not --
    and must say that pt is still theta_v rather than claim a temperature.
    """
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    km = 3
    ak = np.array([500e2 * (1 - k / km) for k in range(km + 1)])
    bk = np.array([k / km for k in range(km + 1)])
    ptop = 500e2
    c = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                   oracle_conventions=True)
    st = _state(ak, bk, ptop, km=km, seed=5)
    pr = [p_var_hydrostatic(f["delp"], ptop=ptop, akap=AKAP,
                            n=N, ng=NG, km=km) for f in st]
    out = fv_dynamics_step(c, st, pr, bdt=60.0, km=km, k_split=1,
                           n_split=1, ptop=ptop, ak=ak, bk=bk, akap=AKAP,
                           cp_air=CP, kord_mt=9, kord_tm=-9, kord_tr=9,
                           q=_tracers(km))
    assert out["pt_units"] == "theta_v"
