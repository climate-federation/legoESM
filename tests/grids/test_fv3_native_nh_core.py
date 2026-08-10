"""Property certificates for the NH core routines (unit 3, stage A).

These are TRUTH-TIER checks (independent algebra, exact invariants), not
the oracle fixture certification — the bit-exact Fortran-extract fixtures
(``fv3_nh_cstage_extract.F90``, port-plan unit 3) are the follow-up gate
and nothing here claims to replace them.  What each test establishes:

* the two tridiagonal systems inside ``sim1_solver`` are solved EXACTLY
  (residual vs an independent dense solve of the same matrices, built
  from the inputs, not from the solver's internals);
* ``update_dz_c`` reduces to identities on states where the flux-form
  update collapses (zero wind; spatially uniform gz), and its ``ws``
  diagnosis and ``dz_min`` limiter act exactly as written;
* ``riem_solver_c`` reproduces its boundary contracts (pef top == ptop,
  gz bottom == hs, gz strictly decreasing-with-height rebuild) and
  returns near-zero w for a hydrostatically balanced, w=0 column.

Grid/thermo constants are the oracle's GFS flavour throughout.
"""

import numpy as np
import pytest
from legoesm.core.fv3_native_nh_core import (
    DZ_MIN,
    riem_solver_c,
    sim1_solver,
    update_dz_c,
)
from legoesm.grids.fv3_native_gridstruct import (
    FV3_GRAV,
    FV3_KAPPA,
    FV3_RDGAS,
)

KM = 5              # the pinned npz
NI = 7              # arbitrary i-window width


class _BD:
    """Minimal bounds shim (is/ie/js/je/ng) for the padded-array API."""

    def __init__(self, n, ng):
        self.is_ = 1
        self.ie = n
        self.js = 1
        self.je = n
        self.ng = ng


def _balanced_column(rng, ni=NI, km=KM):
    """A hydrostatically consistent (dm, pm2, pem, dz2, pt2) set.

    Build pem from ptop + cumsum(dm), pm2 as the oracle's log-mean, pt2
    smooth, and dz2 from the EXACT inverse of the solver's own equation
    of state at pe=0 perturbation:
        dz = -dm*rgas*pt * pm^(kappa-1)   [dm here in mass units/grav]
    so the initial NH perturbation pe(=p_full - pm) is ~0 and a w=0
    column should stay nearly at rest through the implicit solve.
    """
    ptop = 100.0
    dm_pa = 10000.0 + 500.0 * rng.standard_normal((ni, km))   # delp [Pa]
    dm_pa = np.abs(dm_pa)
    pem = np.zeros((ni, km + 1))
    pem[:, 0] = ptop
    for k in range(km):
        pem[:, k + 1] = pem[:, k] + dm_pa[:, k]
    pm2 = np.empty((ni, km))
    for k in range(km):
        pm2[:, k] = dm_pa[:, k] / np.log(pem[:, k + 1] / pem[:, k])
    pt2 = 280.0 + 20.0 * rng.standard_normal((ni, km))
    dm = dm_pa / FV3_GRAV
    gama = 1.0 / (1.0 - FV3_KAPPA)
    # Exact inverse of pe-formula at zero perturbation:
    #   (-dm/dz * rgas * pt)^gama == pm  =>  dz = -dm*rgas*pt / pm^(1/gama)
    dz2 = -dm * FV3_RDGAS * pt2 / np.exp(np.log(pm2) / gama)
    return ptop, dm_pa, dm, pm2, pem, pt2, dz2


def test_sim1_tridiagonal_systems_solved_exactly():
    """Rebuild BOTH tridiagonal systems from the INPUTS and verify the
    solver's pp and w against a dense numpy.linalg.solve."""
    rng = np.random.default_rng(7)
    ptop, dm_pa, dm, pm2, pem, pt2, dz0 = _balanced_column(rng)
    dt = 100.0
    p_fac = 0.05
    gama = 1.0 / (1.0 - FV3_KAPPA)

    # Perturb dz so the NH pressure perturbation is nonzero (the system
    # must be exercised, not trivially zero).
    dz2 = np.array(dz0) * (1.0 + 0.05 * np.sin(np.arange(KM)))
    w2 = 0.3 * np.random.default_rng(8).standard_normal((NI, KM))
    w_in = np.array(w2)
    dz_in = np.array(dz2)
    # ws NONZERO: with ws=0 the -p1*ws bottom term is invisible and a
    # sign mutant passes (codex NH r1 #2: ws=0.7 separates the two signs
    # by 1.3 in w2 bottom).
    ws = 0.7 * np.ones(NI)
    pe = np.zeros((NI, KM + 1))

    sim1_solver(dt, 0, NI - 1, KM, FV3_RDGAS, gama, FV3_KAPPA, pe, dm,
                pm2, pem, w2, dz2, pt2, ws, p_fac)

    # --- independent rebuild of system 1 (the pp tridiagonal) ---
    pe0 = np.exp(gama * np.log(-dm / dz_in * FV3_RDGAS * pt2)) - pm2
    g_rat = dm[:, :-1] / dm[:, 1:]
    t1g = gama * 2.0 * dt * dt
    for i in range(NI):
        A = np.zeros((KM, KM))
        b = np.zeros(KM)
        for k in range(KM - 1):
            # row k: pp[k] (lower) missing for k=0 handled by structure:
            # bet recurrence == Thomas on tri(lower=g_rat[k-1], diag=bb, upper=1)
            A[k, k] = 2.0 * (1.0 + g_rat[i, k])
            b[k] = 3.0 * (pe0[i, k] + g_rat[i, k] * pe0[i, k + 1])
        A[KM - 1, KM - 1] = 2.0
        b[KM - 1] = 3.0 * pe0[i, KM - 1]
        # Row k couples pp(k) [lower, coeff 1], pp(k+1) [diag] and
        # pp(k+2) [upper, coeff g_rat] -- the executable lines were
        # right all along; an earlier comment described them swapped.
        for k in range(1, KM):
            A[k, k - 1] = 1.0
        for k in range(KM - 1):
            A[k, k + 1] = g_rat[i, k]
        x = np.linalg.solve(A, b)        # x[k] == pp(k+1), Fortran pp(2..km+1)
        # solver's pp is not returned; verify through the w system instead
        # by reconstructing pp from pe increments:
        # pe(k+1) = pe(k) + dm*(w2-w1)/dt was applied AFTER, so recover pp
        # directly from the dense solve and check the w tridiagonal.
        pp = np.concatenate([[0.0], x])

        aa = np.zeros(KM)
        for k in range(1, KM):
            aa[k] = t1g / (dz_in[i, k - 1] + dz_in[i, k]) * (pem[i, k] + pp[k])
        p1 = t1g / dz_in[i, KM - 1] * (pem[i, KM] + pp[KM])
        W = np.zeros((KM, KM))
        rhs = np.zeros(KM)
        W[0, 0] = dm[i, 0] - aa[1]
        W[0, 1] = aa[1]
        rhs[0] = dm[i, 0] * w_in[i, 0] + dt * pp[1]
        for k in range(1, KM - 1):
            W[k, k - 1] = aa[k]
            W[k, k] = dm[i, k] - (aa[k] + aa[k + 1])
            W[k, k + 1] = aa[k + 1]
            rhs[k] = dm[i, k] * w_in[i, k] + dt * (pp[k + 1] - pp[k])
        W[KM - 1, KM - 2] = aa[KM - 1]
        W[KM - 1, KM - 1] = dm[i, KM - 1] - (aa[KM - 1] + p1)
        rhs[KM - 1] = (dm[i, KM - 1] * w_in[i, KM - 1]
                       + dt * (pp[KM] - pp[KM - 1]) - p1 * ws[i])
        w_dense = np.linalg.solve(W, rhs)
        r = np.abs(w2[i] - w_dense).max() / max(np.abs(w_dense).max(), 1e-30)
        assert r < 1e-11, (i, r)

        # Independent PE-integral assert (codex NH r1 #2: aliasing w1=w2
        # instead of copying leaves the tested w2 identical but zeroes
        # pe's last column): pe(k+1) = pe(k) + dm*(w2-w1)/dt from the
        # INPUT w, not the solver's internals.
        pe_ref = np.zeros(KM + 1)
        for k in range(KM):
            pe_ref[k + 1] = pe_ref[k] + dm[i, k] * (
                w_dense[k] - w_in[i, k]) / dt
        assert np.abs(pe[i] - pe_ref).max() < 1e-9 * max(
            np.abs(pe_ref).max(), 1.0), i

        # Independent dz2 back-out (bottom-up recurrence rebuilt from
        # pe_ref and the dense pp system's own bb/g_rat):
        capa1 = FV3_KAPPA - 1.0
        p1r = (pe_ref[KM - 1] + 2.0 * pe_ref[KM]) / 3.0
        dz_ref = np.empty(KM)
        dz_ref[KM - 1] = -dm[i, KM - 1] * FV3_RDGAS * pt2[i, KM - 1] * np.exp(
            capa1 * np.log(max(0.05 * pm2[i, KM - 1],
                               p1r + pm2[i, KM - 1])))
        for k in range(KM - 2, -1, -1):
            bbk = 2.0 * (1.0 + g_rat[i, k])
            p1r = ((pe_ref[k] + bbk * pe_ref[k + 1]
                    + g_rat[i, k] * pe_ref[k + 2]) / 3.0
                   - g_rat[i, k] * p1r)
            dz_ref[k] = -dm[i, k] * FV3_RDGAS * pt2[i, k] * np.exp(
                capa1 * np.log(max(0.05 * pm2[i, k], p1r + pm2[i, k])))
        rd = np.abs(dz2[i] - dz_ref).max() / np.abs(dz_ref).max()
        assert rd < 1e-9, (i, rd)
    # Non-vacuity: the solve moved w, and pe/dz2 are nonzero.
    assert np.abs(w2 - w_in).max() > 1e-3
    assert np.abs(pe[:, KM]).max() > 1e-3
    assert np.abs(dz2 - dz_in).max() > 1e-2


def test_sim1_balanced_column_stays_at_rest():
    """w=0 + exact hydrostatic dz => the implicit solve returns w ~ 0
    and dz2 nearly unchanged (only the pe-integral rounding moves it)."""
    rng = np.random.default_rng(11)
    ptop, dm_pa, dm, pm2, pem, pt2, dz2 = _balanced_column(rng)
    gama = 1.0 / (1.0 - FV3_KAPPA)
    w2 = np.zeros((NI, KM))
    dz_in = np.array(dz2)
    pe = np.zeros((NI, KM + 1))
    sim1_solver(1920.0, 0, NI - 1, KM, FV3_RDGAS, gama, FV3_KAPPA, pe, dm,
                pm2, pem, w2, dz2, pt2, np.zeros(NI), 0.05)
    assert np.abs(w2).max() < 1e-7, np.abs(w2).max()
    assert (np.abs(dz2 - dz_in) / np.abs(dz_in)).max() < 1e-9


def test_update_dz_c_zero_wind_is_identity_plus_limiter():
    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    rng = np.random.default_rng(3)
    ut = np.zeros((full, full, KM))
    vt = np.zeros((full, full, KM))
    area = np.ones((full, full)) * 5.0e8
    # gz interfaces decreasing with k (k=0 is top), well separated
    gz = np.zeros((full, full, KM + 1))
    for k in range(KM + 1):
        gz[:, :, k] = (KM - k) * 2000.0 + 30.0 * rng.standard_normal(
            (full, full))
    gz = np.sort(gz, axis=2)[:, :, ::-1].copy()   # enforce monotone input
    zs = np.array(gz[:, :, KM], copy=True)
    ws = np.zeros((full, full))
    gz_in = np.array(gz, copy=True)
    dp0 = np.full(KM, 10000.0)

    update_dz_c(bd, KM, 100.0, dp0, zs, area, ut, vt, gz, ws,
                n + 1, n + 1, sw_corner=True, se_corner=True,
                ne_corner=True, nw_corner=True)

    # Zero wind: the flux-form update reduces to (gz2*area)/area ON THE
    # COMPUTE INTERIOR -- an identity in real arithmetic, a one-ULP
    # multiply-divide ROUND TRIP in floating point, so the comparison is
    # ULP-tolerant, not bitwise.  (Two instrument bugs died here: the
    # first version asserted over the is-1..ie+1 ring, where
    # fill_4corners legitimately rewrites the ring corners exactly as in
    # the oracle, nh_utils.F90:142,154 -> :166-171; the second asserted
    # array_equal on the interior and failed on the round-trip ULP.)
    sl = slice(ng, ng + n)             # is..ie (compute interior)
    rel = (np.abs(gz[sl, sl, :] - gz_in[sl, sl, :])
           / np.maximum(np.abs(gz_in[sl, sl, :]), 1.0)).max()
    assert rel < 1e-14, rel
    # ws = (zs - gz_bottom)/dt; gz_bottom moved by at most the same ULP.
    assert np.abs(ws[sl, sl]).max() < 1e-11 * np.abs(zs[sl, sl]).max()


def test_update_dz_c_uniform_gz_is_transport_invariant():
    """gz spatially uniform per level: flux form gives gz_new == gz for
    ANY wind field (numerator == gz * denominator)."""
    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    rng = np.random.default_rng(5)
    ut = 30.0 * rng.standard_normal((full, full, KM)) * 1.0e6
    vt = 30.0 * rng.standard_normal((full, full, KM)) * 1.0e6
    area = np.ones((full, full)) * 5.0e8
    levels = np.array([(KM - k) * 3000.0 for k in range(KM + 1)])
    gz = np.broadcast_to(levels, (full, full, KM + 1)).copy()
    zs = np.array(gz[:, :, KM], copy=True)
    ws = np.zeros((full, full))
    dp0 = np.full(KM, 10000.0)

    update_dz_c(bd, KM, 100.0, dp0, zs, area, ut, vt, gz, ws,
                n + 1, n + 1, sw_corner=True, se_corner=True,
                ne_corner=True, nw_corner=True)

    sl = slice(ng - 1, ng + n + 1)
    for k in range(KM + 1):
        d = np.abs(gz[sl, sl, k] - levels[k]).max()
        assert d < 1e-9 * max(abs(levels[k]), 1.0), (k, d)


def test_update_dz_c_limiter_and_ws_sign():
    """A column squeezed below dz_min is floored at exactly gz(k+1)+dz_min,
    and a bottom interface displaced below zs gives POSITIVE ws."""
    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    ut = np.zeros((full, full, KM))
    vt = np.zeros((full, full, KM))
    area = np.ones((full, full)) * 5.0e8
    gz = np.zeros((full, full, KM + 1))
    for k in range(KM + 1):
        gz[:, :, k] = (KM - k) * 1.0     # 1 m spacing << dz_min
    zs = np.array(gz[:, :, KM] + 5.0)    # zs ABOVE the bottom interface
    ws = np.zeros((full, full))
    dt = 100.0
    dp0 = np.full(KM, 10000.0)

    update_dz_c(bd, KM, dt, dp0, zs, area, ut, vt, gz, ws,
                n + 1, n + 1, sw_corner=True, se_corner=True,
                ne_corner=True, nw_corner=True)

    sl = slice(ng - 1, ng + n + 1)
    for k in range(KM - 1, -1, -1):
        expect = gz[sl, sl, k + 1] + DZ_MIN
        assert np.array_equal(gz[sl, sl, k], expect), k
    # ws sign: zs - gz_bottom = +5 m over dt=100 s -> +0.05 m/s exactly.
    assert np.allclose(ws[sl, sl], 5.0 / dt, rtol=0, atol=0)


def test_riem_solver_c_contracts_and_balanced_rest():
    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    rng = np.random.default_rng(13)
    ptop = 100.0
    delp = np.abs(10000.0 + 300.0 * rng.standard_normal((full, full, KM)))
    pt = 280.0 + 15.0 * rng.standard_normal((full, full, KM))
    w3 = np.zeros((full, full, KM))
    ws = np.zeros((full, full))
    hs = 50.0 * rng.standard_normal((full, full)) * FV3_GRAV / FV3_GRAV
    gama = 1.0 / (1.0 - FV3_KAPPA)

    # Build gz from the exact hydrostatic dz of each column (as the
    # balanced-column helper does), so w stays at rest.
    pem = np.zeros((full, full, KM + 1))
    pem[:, :, 0] = ptop
    for k in range(KM):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
    pm = np.empty((full, full, KM))
    for k in range(KM):
        pm[:, :, k] = delp[:, :, k] / np.log(pem[:, :, k + 1]
                                             / pem[:, :, k])
    dzh = -(delp / FV3_GRAV) * FV3_RDGAS * pt / np.exp(np.log(pm) / gama)
    gz = np.zeros((full, full, KM + 1))
    gz[:, :, KM] = hs
    for k in range(KM - 1, -1, -1):
        gz[:, :, k] = gz[:, :, k + 1] - dzh[:, :, k]

    pef = np.zeros((full, full, KM + 1))
    riem_solver_c(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, ptop, hs, w3, pt,
                  delp, gz, pef, ws, 0.05, 1.0)

    sl = slice(ng - 1, ng + n + 1)
    # Contracts straight from the source:
    assert np.all(pef[sl, sl, 0] == ptop)                 # :354-356
    assert np.array_equal(gz[sl, sl, KM], hs[sl, sl])     # :410-412
    # gz strictly decreasing with k after the rebuild (dz < 0).
    assert np.all(np.diff(gz[sl, sl, :], axis=2) < 0.0)
    # Full pressure close to hydrostatic pem for the rest state.
    for k in range(KM + 1):
        rel = (np.abs(pef[sl, sl, k] - pem[sl, sl, k])
               / np.maximum(pem[sl, sl, k], 1.0)).max()
        assert rel < 1e-9, (k, rel)


def test_sim1_rejects_float32():
    """codex NH r1 #4: a float32 column truncates dz2 by ~0.35 silently;
    the guard must be loud."""
    rng = np.random.default_rng(7)
    _, _, dm, pm2, pem, pt2, dz2 = _balanced_column(rng)
    gama = 1.0 / (1.0 - FV3_KAPPA)
    with pytest.raises(TypeError, match="float64"):
        sim1_solver(100.0, 0, NI - 1, KM, FV3_RDGAS, gama, FV3_KAPPA,
                    np.zeros((NI, KM + 1)), dm.astype(np.float32), pm2,
                    pem, np.zeros((NI, KM)), dz2, pt2, np.zeros(NI), 0.05)


def test_edge_profile_solves_its_tridiagonal_exactly():
    """Rebuild the nonuniform-branch tridiagonal from nh_utils.F90's
    own coefficients (:1583-1618) and verify against a dense solve:
      row 1:      b= g0(g0+.5),          c= 1+g0(g0+1.5),  d= xt1 q1+q2
      rows 2..km: a= 1, b= 2+2 gk,       c= gk,            d= 3(q_{k-1}+gk q_k)
      row km+1:   a= a_bot, b= gk(gk+.5),                  d= xt1 q_km+q_{km-1}
    """
    from legoesm.core.fv3_native_nh_core import edge_profile

    rng = np.random.default_rng(19)
    ni, km = 4, KM
    q1 = 10.0 + rng.standard_normal((ni, km))
    q2 = -3.0 + rng.standard_normal((ni, km))
    dp0 = np.abs(1.0e4 + 2.0e3 * rng.standard_normal(km))

    qe1, qe2 = edge_profile(q1, q2, 0, km, dp0, False, 0)

    g = dp0[:-1] / dp0[1:]      # gk for rows 2..km (0-based g[k-1])
    g0 = dp0[1] / dp0[0]
    gk_last = g[-1]             # the Fortran reuses the LAST loop gk
    a_bot = 1.0 + gk_last * (gk_last + 1.5)
    for i in range(ni):
        for q, qe in ((q1, qe1), (q2, qe2)):
            A = np.zeros((km + 1, km + 1))
            d = np.zeros(km + 1)
            A[0, 0] = g0 * (g0 + 0.5)
            A[0, 1] = 1.0 + g0 * (g0 + 1.5)
            d[0] = 2.0 * g0 * (g0 + 1.0) * q[i, 0] + q[i, 1]
            for k in range(1, km):
                gk = g[k - 1]
                A[k, k - 1] = 1.0
                A[k, k] = 2.0 + 2.0 * gk
                A[k, k + 1] = gk
                d[k] = 3.0 * (q[i, k - 1] + gk * q[i, k])
            A[km, km - 1] = a_bot
            A[km, km] = gk_last * (gk_last + 0.5)
            d[km] = (2.0 * gk_last * (gk_last + 1.0) * q[i, km - 1]
                     + q[i, km - 2])
            want = np.linalg.solve(A, d)
            got = qe[i]
            r = np.abs(got - want).max() / np.abs(want).max()
            assert r < 1e-12, (i, r)
    # Non-vacuity + basic sanity: a CONSTANT profile is reproduced
    # exactly at every edge.
    qc = np.full((ni, km), 7.5)
    e1, _ = edge_profile(qc, qc, 0, km, dp0, False, 0)
    assert np.abs(e1 - 7.5).max() < 1e-12


def test_update_dz_d_uniform_zh_is_transport_invariant():
    """zh spatially uniform per level is invariant under ANY winds on
    BOTH branches: the fv_tp_2d flux of a constant collapses the flux
    form, and del6 of a constant is zero -- so the damped levels
    (damp > 1e-5) certify the del6 wiring too."""
    from legoesm.core.fv3_native_nh_core import update_dz_d
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        build_fv3_native_gridstruct,
    )

    n, ng = 12, 3
    bd = Bounds.single_tile(n, ng)
    full = n + 2 * ng
    km = KM
    gs = dict(build_fv3_native_gridstruct(n, ng, tile=1,
                                          radius=FV3_RADIUS_M,
                                          omega=FV3_OMEGA))
    gs.update(bounded_domain=False, grid_type=0, sw_corner=True,
              se_corner=True, nw_corner=True, ne_corner=True)
    # ★ SENTINEL-FREE AREA (measured, job 9355001): the single-tile
    # gridstruct leaves BIG_NUMBER sentinels in the 36 corner-diagonal
    # halo cells of `area`; fv_tp_2d's y-intermediates read them and the
    # constant-field flux then deviates from xfx*C by 35%.  With a clean
    # area the deviation is 1.5e-05 of 1.2e11 -- pure rounding.  The
    # ORACLE runs update_dz_d with real mpp-exchanged corner areas, so
    # the sentinels are a CONTEXT gap (owned by the six-face NH
    # integration, unit 7), and this unit test certifies the routine's
    # algebra on the oracle's precondition: real areas everywhere.
    area = np.abs(4.0e11 * (1.0 + 0.05 * np.random.default_rng(29)
                            .standard_normal((full, full))))
    rarea = 1.0 / area
    gs["area"] = area
    gs["rarea"] = rarea

    rng = np.random.default_rng(23)
    crx = 0.2 * rng.standard_normal((n + 1, full, km))
    xfx = 1.0e6 * rng.standard_normal((n + 1, full, km))
    cry = 0.2 * rng.standard_normal((full, n + 1, km))
    yfx = 1.0e6 * rng.standard_normal((full, n + 1, km))

    levels = np.array([(km - k) * 3000.0 + 5000.0
                       for k in range(km + 1)])
    zh = np.broadcast_to(levels, (full, full, km + 1)).copy()
    zs = np.array(zh[:, :, km], copy=True)
    ws = np.zeros((n, n))
    # damp: exercise BOTH branches -- k even damped, k odd not.
    damp = np.array([1.0e6 if k % 2 == 0 else 0.0
                     for k in range(km + 1)])
    ndif = np.array([1 if k % 2 == 0 else 0 for k in range(km + 1)])

    update_dz_d(ndif, damp, 6, bd, km, n + 1, n + 1, area, rarea,
                np.full(km, 1.0e4), zs, zh, crx, cry, xfx, yfx, ws,
                1.0 / 100.0, gs, lim_fac=1.0)

    sl = slice(ng, ng + n)
    # NOTE the k=km row is NOT vacuous here (an earlier revision set the
    # bottom level to 0.0, which made its row pass trivially): shift all
    # levels by +5000 so every level is nonzero.
    for k in range(km + 1):
        d = np.abs(zh[sl, sl, k] - levels[k]).max()
        assert d < 1e-8 * max(abs(levels[k]), 1.0), (k, d)
        assert abs(levels[k]) > 1.0, k          # guard the guard
    assert np.abs(ws).max() < 1e-8


def test_update_dz_c_nonuniform_vs_vectorised_reference():
    """codex NH r1 #3: zero-wind/uniform-gz fixtures cannot see a wrong
    top/bottom/interior ratio.  Nonuniform dp0 + random gz + signed
    winds, checked against an INDEPENDENT vectorised transcription of
    nh_utils.F90:73-171 (same fill_4corners dependency, all other code
    paths distinct from the port's loop form)."""
    from legoesm.core.fv3_native_sw_core import fill_4corners
    from legoesm.grids.fv3_native_gridstruct import fort

    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    km = KM
    rng = np.random.default_rng(31)
    dp0 = np.abs(1.0e4 + 3.0e3 * rng.standard_normal(km))
    ut = 1.0e6 * rng.standard_normal((full, full, km))
    vt = 1.0e6 * rng.standard_normal((full, full, km))
    area = np.abs(4.0e11 * (1.0 + 0.1 * rng.standard_normal((full, full))))
    gz = np.cumsum(
        np.abs(500.0 + 100.0 * rng.standard_normal((full, full, km + 1))),
        axis=2)[:, :, ::-1].copy() * 3.0
    zs = np.array(gz[:, :, km], copy=True)
    ws = np.zeros((full, full))
    gz_port = np.array(gz, copy=True)
    ws_port = np.array(ws, copy=True)
    dt = 100.0

    update_dz_c(bd, km, dt, dp0, zs, area, ut, vt, gz_port, ws_port,
                n + 1, n + 1, sw_corner=True, se_corner=True,
                ne_corner=True, nw_corner=True)

    # ---------------- vectorised reference ----------------
    # 0-based windows: is1..ie1 ring = ng-1 .. ng+n, x extends +1 col.
    a0 = ng - 1                    # is-1 in 0-based storage
    nx = n + 2                     # is-1..ie+1 count
    gz_ref = np.array(gz, copy=True)
    for k1 in range(1, km + 2):
        k = k1 - 1
        if k1 == 1:
            tr = dp0[0] / (dp0[0] + dp0[1])
            xful = ut[:, :, 0] + (ut[:, :, 0] - ut[:, :, 1]) * tr
            yful = vt[:, :, 0] + (vt[:, :, 0] - vt[:, :, 1]) * tr
        elif k1 == km + 1:
            br = dp0[km - 1] / (dp0[km - 2] + dp0[km - 1])
            xful = ut[:, :, km - 1] + (ut[:, :, km - 1]
                                       - ut[:, :, km - 2]) * br
            yful = vt[:, :, km - 1] + (vt[:, :, km - 1]
                                       - vt[:, :, km - 2]) * br
        else:
            ir = 1.0 / (dp0[k - 1] + dp0[k])
            xful = (dp0[k] * ut[:, :, k - 1] + dp0[k - 1] * ut[:, :, k]) * ir
            yful = (dp0[k] * vt[:, :, k - 1] + dp0[k - 1] * vt[:, :, k]) * ir
        # windows: xfx (is-1..ie+2, js-1..je+1); yfx (is-1..ie+1, js-1..je+2)
        xw = xful[a0:a0 + nx + 1, a0:a0 + nx]
        yw = yful[a0:a0 + nx, a0:a0 + nx + 1]

        g2 = np.array(gz[:, :, k], copy=True)
        g2f = fort(g2, 1 - ng, 1 - ng)
        fill_4corners(g2f, 1, n + 1, n + 1)
        fxv = xw * np.where(xw > 0.0,
                            g2[a0 - 1:a0 + nx, a0:a0 + nx],
                            g2[a0:a0 + nx + 1, a0:a0 + nx])
        fill_4corners(g2f, 2, n + 1, n + 1)
        fyv = yw * np.where(yw > 0.0,
                            g2[a0:a0 + nx, a0 - 1:a0 + nx],
                            g2[a0:a0 + nx, a0:a0 + nx + 1])
        aw = area[a0:a0 + nx, a0:a0 + nx]
        num = (g2[a0:a0 + nx, a0:a0 + nx] * aw
               + fxv[:-1, :] - fxv[1:, :] + fyv[:, :-1] - fyv[:, 1:])
        den = (aw + xw[:-1, :] - xw[1:, :] + yw[:, :-1] - yw[:, 1:])
        gz_ref[a0:a0 + nx, a0:a0 + nx, k] = num / den
    ws_ref = np.zeros_like(ws)
    ws_ref[a0:a0 + nx, a0:a0 + nx] = (
        zs[a0:a0 + nx, a0:a0 + nx]
        - gz_ref[a0:a0 + nx, a0:a0 + nx, km]) / dt
    for k in range(km - 1, -1, -1):
        gz_ref[a0:a0 + nx, a0:a0 + nx, k] = np.maximum(
            gz_ref[a0:a0 + nx, a0:a0 + nx, k],
            gz_ref[a0:a0 + nx, a0:a0 + nx, k + 1] + DZ_MIN)

    slw = slice(a0, a0 + nx)
    scale = np.abs(gz_ref[slw, slw, :]).max()
    d = np.abs(gz_port[slw, slw, :] - gz_ref[slw, slw, :]).max()
    assert d < 1e-12 * scale, d
    dws = np.abs(ws_port[slw, slw] - ws_ref[slw, slw]).max()
    assert dws < 1e-12 * max(np.abs(ws_ref).max(), 1.0), dws
    # Non-vacuity: transport moved gz and the winds are signed both ways.
    assert np.abs(gz_port[slw, slw, :] - gz[slw, slw, :]).max() > 1.0


def test_riem_solver_c_unbalanced_column_matches_direct_sim1():
    """codex NH r1 #3: the balanced fixture cannot distinguish
    pef = pe2 + pem from pef = pem.  Squeeze dz by 5% and give w3
    structure, then rebuild riem_solver_c's own column plumbing
    (:347-385) independently and require pef == pe2_direct + pem."""
    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    rng = np.random.default_rng(41)
    ptop = 100.0
    delp = np.abs(10000.0 + 300.0 * rng.standard_normal((full, full, KM)))
    pt = 280.0 + 15.0 * rng.standard_normal((full, full, KM))
    w3 = 0.5 * rng.standard_normal((full, full, KM))
    ws = np.zeros((full, full))
    hs = 50.0 * rng.standard_normal((full, full))
    gama = 1.0 / (1.0 - FV3_KAPPA)

    pem3 = np.zeros((full, full, KM + 1))
    pem3[:, :, 0] = ptop
    for k in range(KM):
        pem3[:, :, k + 1] = pem3[:, :, k] + delp[:, :, k]
    pm3 = np.empty((full, full, KM))
    for k in range(KM):
        pm3[:, :, k] = delp[:, :, k] / np.log(pem3[:, :, k + 1]
                                              / pem3[:, :, k])
    dzh = -(delp / FV3_GRAV) * FV3_RDGAS * pt / np.exp(np.log(pm3) / gama)
    dzh = dzh * 1.05                    # UNBALANCED on purpose
    gz = np.zeros((full, full, KM + 1))
    gz[:, :, KM] = hs
    for k in range(KM - 1, -1, -1):
        gz[:, :, k] = gz[:, :, k + 1] - dzh[:, :, k]
    gz_in = np.array(gz, copy=True)

    pef = np.zeros((full, full, KM + 1))
    riem_solver_c(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, ptop, hs, w3, pt,
                  delp, gz, pef, ws, 0.05, 1.0)

    # Independent column rebuild at a mid-domain j (no ring effects).
    j = ng + 4
    ni = n + 2
    o = ng - 1
    dm = np.array(delp[o:o + ni, j, :], dtype=np.float64)
    pem = np.zeros((ni, KM + 1)); pem[:, 0] = ptop
    for k in range(1, KM + 1):
        pem[:, k] = pem[:, k - 1] + dm[:, k - 1]
    pm2 = np.empty((ni, KM)); dz2 = np.empty((ni, KM)); w2 = np.empty((ni, KM))
    for k in range(KM):
        dz2[:, k] = gz_in[o:o + ni, j, k + 1] - gz_in[o:o + ni, j, k]
        pm2[:, k] = dm[:, k] / np.log(pem[:, k + 1] / pem[:, k])
        dm[:, k] = dm[:, k] / FV3_GRAV
        w2[:, k] = w3[o:o + ni, j, k]
    pe2 = np.zeros((ni, KM + 1))
    sim1_solver(100.0, 0, ni - 1, KM, FV3_RDGAS, gama, FV3_KAPPA, pe2,
                dm, pm2, pem, w2,
                dz2, np.array(pt[o:o + ni, j, :], dtype=np.float64),
                np.array(ws[o:o + ni, j], dtype=np.float64), 0.05)
    want = pe2 + pem
    got = pef[o:o + ni, j, :]
    assert np.array_equal(got[:, 0], np.full(ni, ptop))
    rel = np.abs(got - want).max() / np.abs(want).max()
    assert rel < 1e-13, rel
    # Non-vacuity: the perturbation is far from zero, so pef == pem alone
    # would fail loudly.
    assert np.abs(pe2).max() > 1.0
    # And the gz rebuild is nontrivial: it must equal hs at the bottom and
    # differ from the input gz above it.
    assert np.array_equal(gz[o:o + ni, j, KM], hs[o:o + ni, j])
    assert np.abs(gz[o:o + ni, j, :KM] - gz_in[o:o + ni, j, :KM]).max() > 1.0


def test_riem_solver_c_dead_arm_raises():
    bd = _BD(12, 3)
    with pytest.raises(NotImplementedError):
        riem_solver_c(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, 100.0,
                      np.zeros((18, 18)), np.zeros((18, 18, KM)),
                      np.zeros((18, 18, KM)), np.zeros((18, 18, KM)),
                      np.zeros((18, 18, KM + 1)),
                      np.zeros((18, 18, KM + 1)), np.zeros((18, 18)),
                      0.05, 0.4)
