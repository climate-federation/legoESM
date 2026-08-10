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
    ws = np.zeros(NI)
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
        for k in range(1, KM):
            A[k, k - 1] = 1.0            # pp(k) coefficient in row k
        for k in range(KM - 1):
            A[k, k + 1] = g_rat[i, k]    # pp(k+2) coefficient in row k
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
    # Non-vacuity: the solve moved w.
    assert np.abs(w2 - w_in).max() > 1e-3


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

    # Zero wind: the flux-form update is an exact identity ON THE
    # COMPUTE INTERIOR.  It is NOT an identity at the (is-1/ie+1,
    # js-1/je+1) ring corners: fill_4corners overwrites those cells of
    # the gz2 work copy BEFORE the update writes them back, in the
    # oracle exactly as here (nh_utils.F90:142,154 -> :166-171).  The
    # first version of this test asserted identity over the full ring
    # window and failed on precisely those filled cells.
    sl = slice(ng, ng + n)             # is..ie (compute interior)
    assert np.array_equal(gz[sl, sl, :], gz_in[sl, sl, :])
    # ws = (zs - gz_bottom)/dt with gz_bottom unchanged == 0 there.
    assert np.abs(ws[sl, sl]).max() == 0.0


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


def test_riem_solver_c_dead_arm_raises():
    bd = _BD(12, 3)
    with pytest.raises(NotImplementedError):
        riem_solver_c(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, 100.0,
                      np.zeros((18, 18)), np.zeros((18, 18, KM)),
                      np.zeros((18, 18, KM)), np.zeros((18, 18, KM)),
                      np.zeros((18, 18, KM + 1)),
                      np.zeros((18, 18, KM + 1)), np.zeros((18, 18)),
                      0.05, 0.4)
