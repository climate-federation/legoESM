"""tracer_2d_1L six-face port: dispatch hardening, mass consistency,
conservation, capacitor threading, and the DCMIP16_BC sphum IC.

The seam subtlety these tests respect: the oracle's flux capacitors carry
the UN-averaged d_sw1 fluxes (sw_core.F90:910 accumulates before
dyn_core's allflux averaging), while the tracer fluxes get their own
flux_adj seam blend.  Constant preservation and exact conservation
therefore hold only when the mfx/mfy capacitors are single-valued at the
seams -- so the synthetic capacitors here are passed through
``average_shared_edge_cgrid`` once first, which makes the flux_adj blend
an exact identity on them (own == sign-mapped partner after one pass).
That is a property of the TEST INPUTS, chosen so the invariants are
exact; the production capacitors are not seam-consistent and the parity
gate scores that lane against the oracle instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_tracer2d import (
    alloc_flux_capacitors,
    require_tracer_2d_1l_lane,
    tracer_2d_1l_sixface,
)
from legoesm.grids.fv3_native_gridstruct import average_shared_edge_cgrid

N, NG = 12, 3
KM = 2
HORD_TR = 6
DT = 1920.0
DP0 = 1.0e4


@pytest.fixture(scope="module")
def ctx():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    # The faithful duo lane: ext bundle for the q ext_scalar halos,
    # oracle conventions for the bounded gridstructs the tracer lane
    # requires (same reasons as test_fv3_native_dsw_phase_3d).
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


def _m_a():
    return N + 2 * NG


def _dp1_6():
    return [np.full((_m_a(), _m_a(), KM), DP0, dtype=np.float64)
            for _ in range(6)]


def _capacitors(ctx, courant_scale: float, seed: int = 0) -> list:
    """Synthetic capacitors with SEAM-CONSISTENT mfx/mfy (see module
    docstring). cx/cy magnitudes ~courant_scale; mfx/mfy sized so the
    divergence perturbs dp by O(1 Pa) against DP0 = 1e4."""
    rng = np.random.default_rng(seed)
    cap = alloc_flux_capacitors(N, NG, KM)
    area_mean = float(np.mean(ctx["gs6"][0]["area"]))
    for t in range(6):
        for k in range(KM):
            cap[t]["cx"][:, :, k] = courant_scale * rng.uniform(
                -1.0, 1.0, cap[t]["cx"].shape[:2])
            cap[t]["cy"][:, :, k] = courant_scale * rng.uniform(
                -1.0, 1.0, cap[t]["cy"].shape[:2])
            cap[t]["mfx"][:, 0:N, k] = area_mean * rng.uniform(
                -0.5, 0.5, (N + 1, N))
            cap[t]["mfy"][0:N, :, k] = area_mean * rng.uniform(
                -0.5, 0.5, (N, N + 1))
    for k in range(KM):
        # One blend pass makes each seam slot equal its sign-mapped
        # partner exactly, so flux_adj on c*mfx is then an identity.
        average_shared_edge_cgrid(
            [cap[t]["mfx"][:, 0:N, k] for t in range(6)],
            [cap[t]["mfy"][0:N, :, k] for t in range(6)], N, NG)
    return cap


def _nsplt_of(ctx, cap) -> list:
    """The per-level nsplt the routine will compute -- reproduced here so
    a test can PROVE which branch it exercised instead of assuming."""
    ns = []
    for k in range(KM):
        cm = 0.0
        for t in range(6):
            ssg5 = ctx["gs6"][t]["sin_sg"][NG:NG + N, NG:NG + N, 4]
            for j in range(N):
                for i in range(N):
                    cm = max(cm,
                             max(abs(cap[t]["cx"][i, NG + j, k]),
                                 abs(cap[t]["cy"][NG + i, j, k]))
                             + 1.0 - ssg5[i, j])
        ns.append(int(1.0 + cm))
    return ns


def _q6_constant(c: float) -> list:
    return [[np.full((_m_a(), _m_a(), KM), c, dtype=np.float64)]
            for _ in range(6)]


# ----------------------------------------------------------------------
# dispatch hardening
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kw,frag", [
    (dict(z_tracer=False), "tracer_2d"),
    (dict(q_split=3), "q_split"),
    (dict(nord_tr=2), "nord_tr"),
    (dict(trdm=0.2), "trdm"),
    (dict(inline_q=True), "inline_q"),
])
def test_unported_arms_raise(kw, frag):
    base = dict(z_tracer=True, q_split=0, nord_tr=0, trdm=0.0,
                inline_q=False)
    base.update(kw)
    with pytest.raises(NotImplementedError, match=frag):
        require_tracer_2d_1l_lane(**base)


def test_resolved_deck_lane_accepted():
    require_tracer_2d_1l_lane(z_tracer=True, q_split=0, nord_tr=0,
                              trdm=0.0, inline_q=False)


def test_shape_validation(ctx):
    cap = alloc_flux_capacitors(N, NG, KM)
    bad = [[np.zeros((3, 3, KM))] for _ in range(6)]
    with pytest.raises(ValueError, match="shape"):
        tracer_2d_1l_sixface(ctx, bad, _dp1_6(), cap, km=KM, nq=1,
                             hord_tr=HORD_TR, dt=DT)


# ----------------------------------------------------------------------
# mass consistency: a constant tracer is preserved exactly
# ----------------------------------------------------------------------

def test_constant_tracer_preserved_nsplt1(ctx):
    cap = _capacitors(ctx, courant_scale=0.05)
    ns = _nsplt_of(ctx, cap)
    assert all(n_ == 1 for n_ in ns), \
        f"control failed: wanted the nsplt==1 branch, got nsplt={ns}"
    q6 = _q6_constant(3.0e-3)
    tracer_2d_1l_sixface(ctx, q6, _dp1_6(), cap, km=KM, nq=1,
                         hord_tr=HORD_TR, dt=DT)
    w = slice(NG, NG + N)
    worst = max(float(np.abs(q6[t][0][w, w, :] - 3.0e-3).max())
                for t in range(6))
    # fp64 roundoff of (c*dp1 + c*div)/dp2 only; the seam-consistent
    # capacitors make this exact up to rounding.
    assert worst < 3.0e-3 * 1e-13, f"constant tracer moved by {worst}"


def test_constant_tracer_preserved_subcycled(ctx):
    cap = _capacitors(ctx, courant_scale=1.5, seed=1)
    ns = _nsplt_of(ctx, cap)
    assert all(n_ >= 2 for n_ in ns), \
        f"control failed: wanted the subcycled branch, got nsplt={ns}"
    dp1_6 = _dp1_6()
    dp1_before = [d.copy() for d in dp1_6]
    q6 = _q6_constant(1.7e-2)
    tracer_2d_1l_sixface(ctx, q6, dp1_6, cap, km=KM, nq=1,
                         hord_tr=HORD_TR, dt=DT)
    # The subcycled branch writes dp1 = dp2 between subcycles
    # (fv_tracer2d.F90:361-365) -- prove the branch actually ran.
    moved = max(float(np.abs(dp1_6[t] - dp1_before[t]).max())
                for t in range(6))
    assert moved > 0.0, "dp1 unchanged -- the subcycled branch did not run"
    w = slice(NG, NG + N)
    worst = max(float(np.abs(q6[t][0][w, w, :] - 1.7e-2).max())
                for t in range(6))
    assert worst < 1.7e-2 * 1e-12, f"constant tracer moved by {worst}"


# ----------------------------------------------------------------------
# conservation: cube-total tracer mass is invariant
# ----------------------------------------------------------------------

def test_tracer_mass_conserved(ctx):
    cap = _capacitors(ctx, courant_scale=0.05, seed=2)
    assert all(n_ == 1 for n_ in _nsplt_of(ctx, cap))
    rng = np.random.default_rng(3)
    m_a = _m_a()
    w = slice(NG, NG + N)
    q6 = []
    for t in range(6):
        a = np.zeros((m_a, m_a, KM))
        ii = np.arange(m_a)[:, None, None]
        jj = np.arange(m_a)[None, :, None]
        kk = np.arange(KM)[None, None, :]
        a[:] = (1.0e-3 * (2.0 + np.sin(0.3 * ii + 0.2 * jj + 0.5 * t))
                * (1.0 + 0.1 * kk))
        a += 1e-6 * rng.standard_normal(a.shape)
        q6.append([a])
    dp1_6 = _dp1_6()

    def cube_mass(q6_, dp_per_face):
        tot = 0.0
        for t in range(6):
            area = ctx["gs6"][t]["area"][w, w]
            for k in range(KM):
                tot += float(np.sum(q6_[t][0][w, w, k]
                                    * dp_per_face[t][:, :, k]
                                    * area))
        return tot

    dp1_win = [dp1_6[t][w, w, :].copy() for t in range(6)]
    m0 = cube_mass(q6, dp1_win)

    # dp2 exactly as the routine computes it, from the same capacitors.
    dp2_win = []
    for t in range(6):
        rarea = ctx["gs6"][t]["rarea"][w, w]
        d = np.empty((N, N, KM))
        for k in range(KM):
            mfx = cap[t]["mfx"][:, 0:N, k]
            mfy = cap[t]["mfy"][0:N, :, k]
            d[:, :, k] = dp1_win[t][:, :, k] + (
                mfx[:-1, :] - mfx[1:, :]
                + mfy[:, :-1] - mfy[:, 1:]) * rarea
        dp2_win.append(d)

    tracer_2d_1l_sixface(ctx, q6, dp1_6, cap, km=KM, nq=1,
                         hord_tr=HORD_TR, dt=DT)
    m1 = cube_mass(q6, dp2_win)
    assert abs(m1 - m0) / abs(m0) < 1e-12, \
        f"tracer mass drifted: {m0} -> {m1} (rel {abs(m1-m0)/abs(m0):.3e})"


# ----------------------------------------------------------------------
# capacitor threading through the transport phase
# ----------------------------------------------------------------------

def test_transport_phase_accumulates_unaveraged_fluxes(ctx):
    """dsw_transport_phase_3d fills the capacitors with d_sw1's OWN
    fluxes -- equal to allflux slot 1 away from the seams and DIFFERENT
    from it on the seam columns, because barrier 1 averages allflux but
    not the capacitor (sw_core.F90:910 vs dyn_core.F90:853-900).
    """
    from legoesm.core.fv3_native_cgrid_phase_3d import csw_phase_3d
    from legoesm.core.fv3_native_dsw_phase_3d import dsw_transport_phase_3d
    from legoesm.core.fv3_native_state_3d import build_state_3d

    rng = np.random.default_rng(4)
    km = 1
    st = build_state_3d(N, NG, km)
    for t, face in enumerate(st):
        for name in face:
            a = face[name]
            a[:, :, 0] = 0.01 * t + 1e-3 * rng.standard_normal(a.shape[:2])
        face["delp"][:] = np.abs(face["delp"]) + 1.0e4
        face["pt"][:] = np.abs(face["pt"]) + 280.0

    cap = alloc_flux_capacitors(N, NG, km)
    csw = csw_phase_3d(ctx, st, dt2=15.0, km=km)
    out = dsw_transport_phase_3d(ctx, st, csw, dt=30.0, km=km,
                                 flux_cap=cap)

    interior = np.s_[2:N - 1, 2:N - 2]     # x-faces clear of the seams
    seam_col = 0                            # i = is: averaged by barrier 1
    hit_interior = hit_seam = 0
    for t in range(6):
        mfx = cap[t]["mfx"][:, 0:N, 0]
        afx = out[t]["allflux_x"][:, :, 0, 0]   # slot 1 (delp), averaged
        assert np.allclose(mfx[interior], afx[interior], rtol=0, atol=0), \
            f"face {t + 1}: interior capacitor != d_sw1 flux"
        hit_interior += 1
        d = float(np.abs(mfx[seam_col, :] - afx[seam_col, :]).max())
        if d > 0.0:
            hit_seam += 1
    assert hit_interior == 6
    # The seam difference is the 0.5*(own - partner) blend; on a
    # random state it is nonzero on essentially every face. Require
    # most faces to show it so the unaveraged-capacitor semantics is
    # actually OBSERVED, not just not-contradicted.
    assert hit_seam >= 5, (
        f"only {hit_seam}/6 faces show the seam capacitor differing from "
        f"the averaged allflux -- either the state is degenerate or the "
        f"capacitor is being averaged (it must NOT be)")


def test_transport_phase_capacitors_accumulate(ctx):
    """Two transport-phase calls on the same inputs double the
    capacitors -- the += accumulation across acoustic sub-steps
    (sw_core.F90:903-920), not a per-call overwrite."""
    from legoesm.core.fv3_native_cgrid_phase_3d import csw_phase_3d
    from legoesm.core.fv3_native_dsw_phase_3d import dsw_transport_phase_3d
    from legoesm.core.fv3_native_state_3d import build_state_3d

    rng = np.random.default_rng(5)
    km = 1
    st = build_state_3d(N, NG, km)
    for t, face in enumerate(st):
        for name in face:
            a = face[name]
            a[:, :, 0] = 0.01 * t + 1e-3 * rng.standard_normal(a.shape[:2])
        face["delp"][:] = np.abs(face["delp"]) + 1.0e4
        face["pt"][:] = np.abs(face["pt"]) + 280.0
    csw = csw_phase_3d(ctx, st, dt2=15.0, km=km)

    cap1 = alloc_flux_capacitors(N, NG, km)
    dsw_transport_phase_3d(ctx, st, csw, dt=30.0, km=km, flux_cap=cap1)
    once = [{k: v.copy() for k, v in cap1[t].items()} for t in range(6)]
    # Second sub-step accumulation on the SAME (unchanged) inputs:
    # d_sw1 adds its fluxes on top, so the capacitor doubles.
    dsw_transport_phase_3d(ctx, st, csw, dt=30.0, km=km, flux_cap=cap1)
    for t in range(6):
        for key in ("mfx", "mfy", "cx", "cy"):
            assert np.allclose(cap1[t][key], 2.0 * once[t][key],
                               rtol=1e-13, atol=1e-300), \
                f"face {t + 1} {key}: capacitor did not accumulate"
    peak = max(float(np.abs(once[t]["mfx"]).max()) for t in range(6))
    assert peak > 0.0, "control: d_sw1 produced identically zero fluxes"


# ----------------------------------------------------------------------
# the DCMIP16_BC sphum IC
# ----------------------------------------------------------------------

def test_dcmip16_bc_sphum_matches_the_fortran_formula():
    from legoesm.core.fv3_native_dcmip16_ic import (
        P0_PA, PHIW_BC_RAD, PTROP_BC_PA, PW_BC_PA, Q0_BC, QT_BC,
        dcmip16_bc_sphum,
    )
    from legoesm.core.fv3_native_eta import set_eta_analytic

    km = 5
    ak, bk, _ptop, _ks = set_eta_analytic(km)
    lat = np.array([[0.0, 0.5], [1.0, 1.4]])
    out = dcmip16_bc_sphum(ak, bk, lat, km)
    assert out.shape == (2, 2, km)

    ak = np.asarray(ak, float)
    bk = np.asarray(bk, float)
    pe = np.array([ak[0]] + [ak[k] + P0_PA * bk[k]
                             for k in range(1, km + 1)])
    peln = np.log(pe)
    for k in range(km):
        delp_k = (ak[k + 1] - ak[k]) + P0_PA * (bk[k + 1] - bk[k])
        p = delp_k / (peln[k + 1] - peln[k])
        if p > PTROP_BC_PA:
            want = (Q0_BC * np.exp(-((lat / PHIW_BC_RAD) ** 4))
                    * np.exp(-(((p / P0_PA - 1.0) * P0_PA / PW_BC_PA)
                               ** 2)))
            assert np.array_equal(out[..., k], want)
        else:
            assert np.all(out[..., k] == QT_BC)

    # Both branches must actually be exercised by the km=5 column, or
    # this test proves half the formula.
    ptot = [( (ak[k + 1] - ak[k]) + P0_PA * (bk[k + 1] - bk[k]))
            / (peln[k + 1] - peln[k]) for k in range(km)]
    assert any(p <= PTROP_BC_PA for p in ptot), "no stratospheric level"
    assert any(p > PTROP_BC_PA for p in ptot), "no tropospheric level"
    # Meridional structure: moisture decays away from the equator.
    tropo = [k for k in range(km) if ptot[k] > PTROP_BC_PA][0]
    col = out[:, :, tropo].ravel()
    lats = np.abs(lat.ravel())
    order = np.argsort(lats)
    assert np.all(np.diff(col[order]) <= 0.0)


def test_alloc_flux_capacitors_shapes():
    cap = alloc_flux_capacitors(N, NG, KM)
    m_a = _m_a()
    assert len(cap) == 6
    for t in range(6):
        assert cap[t]["mfx"].shape == (N + 1, m_a, KM)
        assert cap[t]["mfy"].shape == (m_a, N + 1, KM)
        assert cap[t]["cx"].shape == (N + 1, m_a, KM)
        assert cap[t]["cy"].shape == (m_a, N + 1, KM)
        for v in cap[t].values():
            assert not v.any()
