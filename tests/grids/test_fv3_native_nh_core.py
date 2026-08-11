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


class _BDR:
    """Rectangular bounds with FREE origins (is != js, ie-is != je-js).

    codex NH r2 #6: every earlier fixture hard-coded is == js == 1, so
    the r1 origin bug (jlo derived from the i origin) had no committed
    regression.  This shim is that regression's carrier."""

    def __init__(self, is_, ie, js, je, ng):
        self.is_ = is_
        self.ie = ie
        self.js = js
        self.je = je
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


def _run_native_sim1(dt, rgas, gama, kappa, dm, pm2, pem, w2, dz2, pt2,
                     ws, p_fac):
    """Functional adapter over the in-place native solver.

    Shared driver signature for the impl-parameterized certificates below
    (the JAX twin in test_fv3_nh_core.py plugs its own adapter in here):
    returns (pe, w2, dz2) without mutating the caller's arrays.
    """
    ni, km = dm.shape
    pe = np.zeros((ni, km + 1))
    w2 = np.array(w2)
    dz2 = np.array(dz2)
    sim1_solver(dt, 0, ni - 1, km, rgas, gama, kappa, pe, dm, pm2, pem,
                w2, dz2, pt2, ws, p_fac)
    return pe, w2, dz2


def sim1_dense_certificate(run_sim1):
    """Rebuild BOTH tridiagonal systems from the INPUTS and verify the
    solver's pp and w against a dense numpy.linalg.solve.

    ``run_sim1`` is a functional driver with ``_run_native_sim1``'s
    signature — the certificate is implementation-agnostic so the JAX
    lane certifies against the SAME algebra (no duplicated numerics)."""
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

    pe, w2, dz2 = run_sim1(dt, FV3_RDGAS, gama, FV3_KAPPA, dm, pm2, pem,
                           w2, dz2, pt2, ws, p_fac)

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


def test_sim1_tridiagonal_systems_solved_exactly():
    sim1_dense_certificate(_run_native_sim1)


def sim1_balanced_rest_certificate(run_sim1):
    """w=0 + exact hydrostatic dz => the implicit solve returns w ~ 0
    and dz2 nearly unchanged (only the pe-integral rounding moves it).
    Implementation-agnostic (same driver signature as the dense cert)."""
    rng = np.random.default_rng(11)
    ptop, dm_pa, dm, pm2, pem, pt2, dz2 = _balanced_column(rng)
    gama = 1.0 / (1.0 - FV3_KAPPA)
    w2 = np.zeros((NI, KM))
    dz_in = np.array(dz2)
    pe, w2, dz2 = run_sim1(1920.0, FV3_RDGAS, gama, FV3_KAPPA, dm, pm2,
                           pem, w2, dz2, pt2, np.zeros(NI), 0.05)
    assert np.abs(w2).max() < 1e-7, np.abs(w2).max()
    assert (np.abs(dz2 - dz_in) / np.abs(dz_in)).max() < 1e-9


def test_sim1_balanced_column_stays_at_rest():
    sim1_balanced_rest_certificate(_run_native_sim1)


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


def _run_native_riem_c(dt, bd, km, akap, ptop, hs, w3, pt, delp, gz, pef,
                       ws, p_fac):
    """Functional adapter over the in-place native riem_solver_c.

    Shared driver signature for the impl-parameterized certificates
    below (the JAX twin in test_fv3_nh_core.py plugs its own adapter in
    here): returns (gz, pef) without mutating the caller's arrays."""
    gz = np.array(gz)
    pef = np.array(pef)
    riem_solver_c(1, dt, bd, km, akap, 1004.6, ptop, hs, w3, pt, delp,
                  gz, pef, ws, p_fac, 1.0)
    return gz, pef


def _riem_c_fixture(seed, n=12, ng=3, km=KM, w_amp=0.0, dz_scale=1.0):
    """(ptop, hs, w3, pt, delp, gz, ws): a per-column hydrostatic gz
    build (dz from the exact inverse of the solver's EOS at zero
    perturbation), optionally squeezed by ``dz_scale`` and given w3
    structure — shared by the contracts/balanced and unbalanced
    certificates AND the JAX-lane equivalence fixtures."""
    full = n + 2 * ng
    rng = np.random.default_rng(seed)
    ptop = 100.0
    delp = np.abs(10000.0 + 300.0 * rng.standard_normal((full, full, km)))
    pt = 280.0 + 15.0 * rng.standard_normal((full, full, km))
    w3 = w_amp * rng.standard_normal((full, full, km))
    ws = np.zeros((full, full))
    hs = 50.0 * rng.standard_normal((full, full))
    gama = 1.0 / (1.0 - FV3_KAPPA)
    pem = np.zeros((full, full, km + 1))
    pem[:, :, 0] = ptop
    for k in range(km):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
    pm = np.empty((full, full, km))
    for k in range(km):
        pm[:, :, k] = delp[:, :, k] / np.log(pem[:, :, k + 1]
                                             / pem[:, :, k])
    dzh = -(delp / FV3_GRAV) * FV3_RDGAS * pt / np.exp(np.log(pm) / gama)
    dzh = dzh * dz_scale
    gz = np.zeros((full, full, km + 1))
    gz[:, :, km] = hs
    for k in range(km - 1, -1, -1):
        gz[:, :, k] = gz[:, :, k + 1] - dzh[:, :, k]
    return ptop, hs, w3, pt, delp, gz, ws


def _riem_c_halo_mask(n, ng):
    """True outside riem_solver_c's write window (is-1..ie+1 square)."""
    full = n + 2 * ng
    halo = np.ones((full, full), dtype=bool)
    halo[ng - 1:ng + n + 1, ng - 1:ng + n + 1] = False
    return halo


def riem_c_contracts_balanced_certificate(run_riem_c):
    """Boundary contracts + balanced-rest, impl-parameterized.

    NOTE: this fixture is INSENSITIVE to the g-scaling of hs — the
    rest state passes with or without a FV3_GRAV factor here, so it does
    NOT pin the height-vs-geopotential convention that the module header
    states ("gz enters riem_solver_c as height*grav-like geopotential").
    Whatever the caller settles on, one assertion tying a known
    hydrostatic layer to its dz2 belongs here so a later mismatch fails
    loudly."""
    n, ng = 12, 3
    bd = _BD(n, ng)
    ptop, hs, w3, pt, delp, gz, ws = _riem_c_fixture(13)

    pem = np.zeros(gz.shape)
    pem[:, :, 0] = ptop
    for k in range(KM):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]

    # Footprint guard (codex riem r1 #3): sentinel gz/pef OUTSIDE the
    # (is-1..ie+1, js-1..je+1) window; the solver must carry them
    # through bitwise (gz halo is never read on this fixture's window).
    halo = _riem_c_halo_mask(n, ng)
    gz[halo, :] = _SENT3
    pef = np.zeros(gz.shape)
    pef[halo, :] = _SENT3
    gz, pef = run_riem_c(100.0, bd, KM, FV3_KAPPA, ptop, hs, w3, pt,
                         delp, gz, pef, ws, 0.05)
    assert np.all(gz[halo, :] == _SENT3)
    assert np.all(pef[halo, :] == _SENT3)

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


def test_riem_solver_c_contracts_and_balanced_rest():
    riem_c_contracts_balanced_certificate(_run_native_riem_c)


@pytest.mark.parametrize("bad", ["pe", "dm2", "pm2", "pem", "w2", "dz2",
                                 "pt2", "ws"])
def test_sim1_rejects_float32_every_operand(bad):
    """codex NH r1 #4 + r2 #1: a float32 column truncates dz2 by ~0.35
    silently, and ``ws`` (read by the bottom solve, nh_utils.F90:1287)
    was the operand the first guard missed.  Every operand, one at a
    time."""
    rng = np.random.default_rng(7)
    _, _, dm, pm2, pem, pt2, dz2 = _balanced_column(rng)
    gama = 1.0 / (1.0 - FV3_KAPPA)
    args = {"pe": np.zeros((NI, KM + 1)), "dm2": dm, "pm2": pm2,
            "pem": pem, "w2": np.zeros((NI, KM)), "dz2": dz2,
            "pt2": pt2, "ws": np.zeros(NI)}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        sim1_solver(100.0, 0, NI - 1, KM, FV3_RDGAS, gama, FV3_KAPPA,
                    args["pe"], args["dm2"], args["pm2"], args["pem"],
                    args["w2"], args["dz2"], args["pt2"], args["ws"],
                    0.05)


@pytest.mark.parametrize("bad", ["hs", "w3", "pt", "delp", "gz", "pef",
                                 "ws"])
def test_riem_solver_c_rejects_float32_every_operand(bad):
    """codex NH r2 #1: the wrapper's f64 work copies laundered float32
    inputs past the sim1 guard (measured: gz off 0.208, pef off 2.2e-3
    at km=5), and float32 OUTPUTS (gz, pef, ws) truncate on assignment.
    The entry guard must catch each operand before any cast."""
    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    args = {"hs": np.zeros((full, full)),
            "w3": np.zeros((full, full, KM)),
            "pt": np.full((full, full, KM), 280.0),
            "delp": np.full((full, full, KM), 1.0e4),
            "gz": np.zeros((full, full, KM + 1)),
            "pef": np.zeros((full, full, KM + 1)),
            "ws": np.zeros((full, full))}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        riem_solver_c(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, 100.0,
                      args["hs"], args["w3"], args["pt"], args["delp"],
                      args["gz"], args["pef"], args["ws"], 0.05, 1.0)


@pytest.mark.parametrize("bad", ["zs", "w", "delz", "pt", "delp", "zh",
                                 "pe", "ppe", "pk3", "pk", "peln", "ws"])
def test_riem_solver3_rejects_float32_every_operand(bad):
    from legoesm.core.fv3_native_nh_core import riem_solver3

    bd = _BDR(4, 10, 9, 13, 3)
    ni, nj, ng = 7, 5, 3
    fi, fj = ni + 2 * ng, nj + 2 * ng
    args = {"zs": np.zeros((fi, fj)),
            "w": np.zeros((fi, fj, KM)),
            "delz": np.zeros((ni, nj, KM)),
            "pt": np.full((fi, fj, KM), 280.0),
            "delp": np.full((fi, fj, KM), 1.0e4),
            "zh": np.zeros((fi, fj, KM + 1)),
            "pe": np.zeros((ni + 2, KM + 1, nj + 2)),
            "ppe": np.zeros((fi, fj, KM + 1)),
            "pk3": np.zeros((fi, fj, KM + 1)),
            "pk": np.zeros((ni, nj, KM + 1)),
            "peln": np.zeros((ni, KM + 1, nj)),
            "ws": np.zeros((ni, nj))}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        riem_solver3(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, 100.0,
                     args["zs"], args["w"], args["delz"], args["pt"],
                     args["delp"], args["zh"], args["pe"], args["ppe"],
                     args["pk3"], args["pk"], args["peln"], args["ws"],
                     0.05, 1.0)


@pytest.mark.parametrize("bad", ["dp0", "zs", "area", "ut", "vt", "gz",
                                 "ws"])
def test_update_dz_c_rejects_float32_every_operand(bad):
    n, ng = 12, 3
    bd = _BD(n, ng)
    full = n + 2 * ng
    args = {"dp0": np.full(KM, 1.0e4),
            "zs": np.zeros((full, full)),
            "area": np.full((full, full), 5.0e8),
            "ut": np.zeros((full, full, KM)),
            "vt": np.zeros((full, full, KM)),
            "gz": np.zeros((full, full, KM + 1)),
            "ws": np.zeros((full, full))}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        update_dz_c(bd, KM, 100.0, args["dp0"], args["zs"], args["area"],
                    args["ut"], args["vt"], args["gz"], args["ws"],
                    n + 1, n + 1, sw_corner=True, se_corner=True,
                    ne_corner=True, nw_corner=True)


@pytest.mark.parametrize("bad", ["damp", "dp0", "zs", "zh", "crx", "cry",
                                 "xfx", "yfx", "ws", "area", "rarea"])
def test_update_dz_d_rejects_float32_every_operand(bad):
    from legoesm.core.fv3_native_nh_core import update_dz_d
    from legoesm.core.fv3_native_sw_core import Bounds

    n, ng = 12, 3
    bd = Bounds.single_tile(n, ng)
    full = n + 2 * ng
    area = np.full((full, full), 4.0e11)
    args = {"damp": np.zeros(KM + 1),
            "dp0": np.full(KM, 1.0e4),
            "zs": np.zeros((full, full)),
            "zh": np.zeros((full, full, KM + 1)),
            "crx": np.zeros((n + 1, full, KM)),
            "cry": np.zeros((full, n + 1, KM)),
            "xfx": np.zeros((n + 1, full, KM)),
            "yfx": np.zeros((full, n + 1, KM)),
            "ws": np.zeros((n, n)),
            "area": area,
            "rarea": 1.0 / area}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        update_dz_d(np.zeros(KM + 1, dtype=np.int64), args["damp"], 6,
                    bd, KM, n + 1, n + 1, args["area"], args["rarea"],
                    args["dp0"], args["zs"], args["zh"], args["crx"],
                    args["cry"], args["xfx"], args["yfx"], args["ws"],
                    1.0e-2, {"area": args["area"],
                             "rarea": args["rarea"]}, lim_fac=1.0)


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


def riem_c_unbalanced_certificate(run_riem_c):
    """codex NH r1 #3: the balanced fixture cannot distinguish
    pef = pe2 + pem from pef = pem.  Squeeze dz by 5% and give w3
    structure, then rebuild riem_solver_c's own column plumbing
    (:347-385) independently and require pef == pe2_direct + pem.
    Impl-parameterized (the re-plumb reference stays the NATIVE
    sim1_solver, which carries its own dense certificate)."""
    n, ng = 12, 3
    bd = _BD(n, ng)
    ptop, hs, w3, pt, delp, gz, ws = _riem_c_fixture(
        41, w_amp=0.5, dz_scale=1.05)   # UNBALANCED on purpose
    # Footprint guard (codex riem r1 #3): sentinel outside the window.
    halo = _riem_c_halo_mask(n, ng)
    gz[halo, :] = _SENT3
    gz_in = np.array(gz, copy=True)

    pef = np.zeros(gz.shape)
    pef[halo, :] = _SENT3
    gz, pef = run_riem_c(100.0, bd, KM, FV3_KAPPA, ptop, hs, w3, pt,
                         delp, gz, pef, ws, 0.05)
    assert np.all(gz[halo, :] == _SENT3)
    assert np.all(pef[halo, :] == _SENT3)

    # Independent column rebuild at a mid-domain j (no ring effects).
    gama = 1.0 / (1.0 - FV3_KAPPA)
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


def test_riem_solver_c_unbalanced_column_matches_direct_sim1():
    riem_c_unbalanced_certificate(_run_native_riem_c)


def test_riem_solver_c_dead_arm_raises():
    bd = _BD(12, 3)
    with pytest.raises(NotImplementedError):
        riem_solver_c(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, 100.0,
                      np.zeros((18, 18)), np.zeros((18, 18, KM)),
                      np.zeros((18, 18, KM)), np.zeros((18, 18, KM)),
                      np.zeros((18, 18, KM + 1)),
                      np.zeros((18, 18, KM + 1)), np.zeros((18, 18)),
                      0.05, 0.4)


# --------------------------------------------------------------------------
# codex NH r2 #6: rectangular, non-equal-origin regression.  The r1 bug
# derived jlo from the i origin; with is=4..7, js=10..12 that reads rows
# ng+9 cells too high and IndexErrors (numpy raises on a too-large index;
# it only WRAPS on negative ones, which is why a square origin could not
# see this).  Equivariance form: origins are LABELS -- identical padded
# storage under two different origins must produce bitwise-identical
# results.  grid_type=3 skips fill_4corners for update_dz_c because its
# corner indices are face-absolute (anchored at Fortran 0/npx), which is
# a different concern from origin plumbing.
# --------------------------------------------------------------------------

def _rect_fields(ni, nj, ng, km, seed):
    rng = np.random.default_rng(seed)
    fi, fj = ni + 2 * ng, nj + 2 * ng
    # 1e10 (~2.5% of area 4e11): a 1e6 flux moves gz by only ~1e-2 m,
    # under the >1.0 non-vacuity floor on a small rectangle.
    ut = 1.0e10 * rng.standard_normal((fi, fj, km))
    vt = 1.0e10 * rng.standard_normal((fi, fj, km))
    area = np.abs(4.0e11 * (1.0 + 0.1 * rng.standard_normal((fi, fj))))
    gz = np.cumsum(
        np.abs(500.0 + 100.0 * rng.standard_normal((fi, fj, km + 1))),
        axis=2)[:, :, ::-1].copy() * 3.0
    zs = np.array(gz[:, :, km], copy=True)
    dp0 = np.abs(1.0e4 + 3.0e3 * rng.standard_normal(km))
    return ut, vt, area, gz, zs, dp0


def test_update_dz_c_origin_is_a_pure_relabel():
    ni, nj, ng = 4, 3, 3
    ut, vt, area, gz, zs, dp0 = _rect_fields(ni, nj, ng, KM, 61)
    fi, fj = ni + 2 * ng, nj + 2 * ng

    out = {}
    for tag, bd in (("o11", _BDR(1, ni, 1, nj, ng)),
                    ("o4_10", _BDR(4, 3 + ni, 10, 9 + nj, ng))):
        g = np.array(gz, copy=True)
        w = np.zeros((fi, fj))
        update_dz_c(bd, KM, 100.0, dp0, np.array(zs), np.array(area),
                    np.array(ut), np.array(vt), g, w,
                    ni + 1, nj + 1, sw_corner=True, se_corner=True,
                    ne_corner=True, nw_corner=True, grid_type=3)
        out[tag] = (g, w)

    assert np.array_equal(out["o11"][0], out["o4_10"][0])
    assert np.array_equal(out["o11"][1], out["o4_10"][1])
    # Non-vacuity: the update moved gz somewhere.
    assert np.abs(out["o11"][0] - gz).max() > 1.0


def riem_c_origin_relabel_certificate(run_riem_c):
    """Origins are LABELS: identical padded storage under two different
    (is, js) origins must produce bitwise-identical results (same impl,
    deterministic — holds for the JAX lane too)."""
    ni, nj, ng = 4, 3, 3
    fi, fj = ni + 2 * ng, nj + 2 * ng
    rng = np.random.default_rng(67)
    delp = np.abs(10000.0 + 300.0 * rng.standard_normal((fi, fj, KM)))
    pt = 280.0 + 15.0 * rng.standard_normal((fi, fj, KM))
    w3 = 0.5 * rng.standard_normal((fi, fj, KM))
    hs = 50.0 * rng.standard_normal((fi, fj))
    gama = 1.0 / (1.0 - FV3_KAPPA)
    ptop = 100.0
    pem = np.zeros((fi, fj, KM + 1))
    pem[:, :, 0] = ptop
    for k in range(KM):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
    pm = delp / np.log(pem[:, :, 1:] / pem[:, :, :-1])
    dzh = -(delp / FV3_GRAV) * FV3_RDGAS * pt / np.exp(np.log(pm) / gama)
    dzh *= 1.05
    gz0 = np.zeros((fi, fj, KM + 1))
    gz0[:, :, KM] = hs
    for k in range(KM - 1, -1, -1):
        gz0[:, :, k] = gz0[:, :, k + 1] - dzh[:, :, k]
    # Footprint guard (codex riem r1 #3): sentinel outside the window,
    # so two impls with the SAME stray halo write cannot both pass.
    halo = np.ones((fi, fj), dtype=bool)
    halo[ng - 1:ng + ni + 1, ng - 1:ng + nj + 1] = False
    gz0[halo, :] = _SENT3

    out = {}
    for tag, bd in (("o11", _BDR(1, ni, 1, nj, ng)),
                    ("o4_10", _BDR(4, 3 + ni, 10, 9 + nj, ng))):
        pef0 = np.zeros((fi, fj, KM + 1))
        pef0[halo, :] = _SENT3
        g, pef = run_riem_c(100.0, bd, KM, FV3_KAPPA, ptop,
                            np.array(hs), np.array(w3), np.array(pt),
                            np.array(delp), np.array(gz0, copy=True),
                            pef0, np.zeros((fi, fj)), 0.05)
        assert np.all(g[halo, :] == _SENT3), tag
        assert np.all(pef[halo, :] == _SENT3), tag
        out[tag] = (g, pef)

    assert np.array_equal(out["o11"][0], out["o4_10"][0])
    assert np.array_equal(out["o11"][1], out["o4_10"][1])
    assert np.abs(out["o11"][1][~halo, :]).max() > 0.0


def test_riem_solver_c_origin_is_a_pure_relabel():
    riem_c_origin_relabel_certificate(_run_native_riem_c)


# --------------------------------------------------------------------------
# codex NH r2 #5: the earlier reference shared the port's fill_4corners,
# so a shared corner bug was invisible (a no-op fill_4corners moved this
# seeded case by 2091.27 while port and reference still agreed).  Here
# the corner treatment is transcribed INDEPENDENTLY from the oracle
# (sw_core.F90:3856-3915, verified against the pinned tree), in raw
# storage indices, with a rectangular domain and MIXED corner flags.
# --------------------------------------------------------------------------

def test_update_dz_c_rect_mixed_flags_vs_independent_corner_reference():
    ni, nj, ng = 12, 9, 3
    km = KM
    npx, npy = ni + 1, nj + 1
    flags = dict(sw=True, se=False, ne=True, nw=False)
    bd = _BDR(1, ni, 1, nj, ng)
    ut, vt, area, gz, zs, dp0 = _rect_fields(ni, nj, ng, km, 71)
    fi, fj = ni + 2 * ng, nj + 2 * ng

    gz_port = np.array(gz, copy=True)
    ws_port = np.zeros((fi, fj))
    dt = 100.0
    update_dz_c(bd, km, dt, dp0, np.array(zs), np.array(area),
                np.array(ut), np.array(vt), gz_port, ws_port,
                npx, npy, sw_corner=flags["sw"], se_corner=flags["se"],
                ne_corner=flags["ne"], nw_corner=flags["nw"])

    # ---- independent reference (storage indices; s(f) = f + ng - 1) ----
    def s(f):
        return f + ng - 1

    def corner_fill_ref(q, direction):
        # sw_core.F90:3876-3893 (XDir) / :3895-3913 (YDir), one
        # assignment per oracle line, honoring each flag separately.
        if direction == 1:
            if flags["sw"]:
                q[s(-1), s(0)] = q[s(0), s(2)]
                q[s(0), s(0)] = q[s(0), s(1)]
            if flags["se"]:
                q[s(npx + 1), s(0)] = q[s(npx), s(2)]
                q[s(npx), s(0)] = q[s(npx), s(1)]
            if flags["nw"]:
                q[s(0), s(npy)] = q[s(0), s(npy - 1)]
                q[s(-1), s(npy)] = q[s(0), s(npy - 2)]
            if flags["ne"]:
                q[s(npx), s(npy)] = q[s(npx), s(npy - 1)]
                q[s(npx + 1), s(npy)] = q[s(npx), s(npy - 2)]
        else:
            if flags["sw"]:
                q[s(0), s(0)] = q[s(1), s(0)]
                q[s(0), s(-1)] = q[s(2), s(0)]
            if flags["se"]:
                q[s(npx), s(0)] = q[s(npx - 1), s(0)]
                q[s(npx), s(-1)] = q[s(npx - 2), s(0)]
            if flags["nw"]:
                q[s(0), s(npy)] = q[s(1), s(npy)]
                q[s(0), s(npy + 1)] = q[s(2), s(npy)]
            if flags["ne"]:
                q[s(npx), s(npy)] = q[s(npx - 1), s(npy)]
                q[s(npx), s(npy + 1)] = q[s(npx - 2), s(npy)]

    a0 = ng - 1                     # storage row of is-1 / col of js-1
    nxi = ni + 2                    # is-1..ie+1
    nxj = nj + 2                    # js-1..je+1
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
        xw = xful[a0:a0 + nxi + 1, a0:a0 + nxj]
        yw = yful[a0:a0 + nxi, a0:a0 + nxj + 1]

        g2 = np.array(gz[:, :, k], copy=True)
        corner_fill_ref(g2, 1)
        fxv = xw * np.where(xw > 0.0,
                            g2[a0 - 1:a0 + nxi, a0:a0 + nxj],
                            g2[a0:a0 + nxi + 1, a0:a0 + nxj])
        corner_fill_ref(g2, 2)
        fyv = yw * np.where(yw > 0.0,
                            g2[a0:a0 + nxi, a0 - 1:a0 + nxj],
                            g2[a0:a0 + nxi, a0:a0 + nxj + 1])
        aw = area[a0:a0 + nxi, a0:a0 + nxj]
        num = (g2[a0:a0 + nxi, a0:a0 + nxj] * aw
               + fxv[:-1, :] - fxv[1:, :] + fyv[:, :-1] - fyv[:, 1:])
        den = (aw + xw[:-1, :] - xw[1:, :] + yw[:, :-1] - yw[:, 1:])
        gz_ref[a0:a0 + nxi, a0:a0 + nxj, k] = num / den
    ws_ref = np.zeros((fi, fj))
    # * (1/dt), not / dt: the port multiplies by rdt and x*(1/dt) is not
    # x/dt in the last ulp -- this comparison is array_equal-exact.
    ws_ref[a0:a0 + nxi, a0:a0 + nxj] = (
        zs[a0:a0 + nxi, a0:a0 + nxj]
        - gz_ref[a0:a0 + nxi, a0:a0 + nxj, km]) * (1.0 / dt)
    for k in range(km - 1, -1, -1):
        gz_ref[a0:a0 + nxi, a0:a0 + nxj, k] = np.maximum(
            gz_ref[a0:a0 + nxi, a0:a0 + nxj, k],
            gz_ref[a0:a0 + nxi, a0:a0 + nxj, k + 1] + DZ_MIN)

    assert np.array_equal(gz_port, gz_ref)
    assert np.array_equal(ws_port, ws_ref)
    # Non-vacuity.
    assert np.abs(gz_port - gz).max() > 1.0
    # A DIFFERENT flag set must change the answer (the flags are live,
    # so a fill that ignored them -- or a no-op fill -- cannot pass both
    # this and the reference equality above).
    gz_alt = np.array(gz, copy=True)
    ws_alt = np.zeros((fi, fj))
    update_dz_c(bd, km, dt, dp0, np.array(zs), np.array(area),
                np.array(ut), np.array(vt), gz_alt, ws_alt,
                npx, npy, sw_corner=False, se_corner=True,
                ne_corner=False, nw_corner=True)
    assert np.abs(gz_alt - gz_port).max() > 0.0


# --------------------------------------------------------------------------
# codex NH r2 #2 (HIGH): riem_solver3 had no test instrument at all -- a
# ppe = pe2 + pem mutant (top 100 Pa instead of the oracle-required
# perturbation 0 Pa, nh_core.F90:173-185) passed the suite.  This is the
# direct certificate: sentinel layout, non-square non-equal-origin
# bounds, every flag branch, every written window, and an independent
# per-column re-plumb of :87-202 feeding the SAME sim1_solver (which has
# its own dense-solve certificate above).
# --------------------------------------------------------------------------

_SENT3 = 7.7e30


def _riem3_fixture(bd, km=KM, seed=47):
    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng
    ni, nj = ie - is_ + 1, je - js + 1
    fi, fj = ni + 2 * ng, nj + 2 * ng
    rng = np.random.default_rng(seed)
    delp = np.abs(10000.0 + 300.0 * rng.standard_normal((fi, fj, km)))
    pt = 280.0 + 15.0 * rng.standard_normal((fi, fj, km))
    w = 0.5 * rng.standard_normal((fi, fj, km))
    ws = 0.1 * rng.standard_normal((ni, nj))
    zs = 50.0 * np.abs(rng.standard_normal((fi, fj)))
    dz = -(300.0 + 100.0 * np.abs(rng.standard_normal((fi, fj, km))))
    zh = np.empty((fi, fj, km + 1))
    zh[:, :, km] = zs
    for k in range(km - 1, -1, -1):
        zh[:, :, k] = zh[:, :, k + 1] - dz[:, :, k]
    # Sentinel the halos of the padded in/out arrays so any stray
    # read/write is loud (finite sentinel: a stray READ corrupts the
    # certificate numbers, a stray WRITE overwrites the sentinel).
    o = ng
    halo = np.ones((fi, fj), dtype=bool)
    halo[o:o + ni, o:o + nj] = False
    for a in (w, zh):
        a[halo, :] = _SENT3
    ppe = np.full((fi, fj, km + 1), _SENT3)
    pk3 = np.full((fi, fj, km + 1), _SENT3)
    pe = np.full((ni + 2, km + 1, nj + 2), _SENT3)
    peln = np.full((ni, km + 1, nj), _SENT3)
    pk = np.full((ni, nj, km + 1), _SENT3)
    delz = np.full((ni, nj, km), _SENT3)
    return dict(delp=delp, pt=pt, w=w, ws=ws, zs=zs, zh=zh, ppe=ppe,
                pk3=pk3, pe=pe, peln=peln, pk=pk, delz=delz,
                ni=ni, nj=nj, o=o, halo=halo)


def _riem3_balanced_fixture(bd, km=KM, seed=53):
    """The sentinel fixture with zh rebuilt HYDROSTATIC (dz from the
    exact inverse of the solver's EOS at zero perturbation, using the
    D-stage's peln2-difference pm2) and w = ws = 0, so the implicit
    solve returns w ~ 0 and ppe ~ roundoff.  Shared by the JAX-lane
    balanced equivalence gate."""
    fx = _riem3_fixture(bd, km=km, seed=seed)
    ni, nj, o = fx["ni"], fx["nj"], fx["o"]
    ptop = 100.0
    gama = 1.0 / (1.0 - FV3_KAPPA)
    delp = fx["delp"]
    pt = fx["pt"]
    fi, fj = delp.shape[0], delp.shape[1]
    pem = np.zeros((fi, fj, km + 1))
    pem[:, :, 0] = ptop
    for k in range(km):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
    peln2 = np.log(pem)
    pm = delp / (peln2[:, :, 1:] - peln2[:, :, :-1])
    dzh = -(delp / FV3_GRAV) * FV3_RDGAS * pt / np.exp(np.log(pm) / gama)
    zh = np.empty((fi, fj, km + 1))
    zh[:, :, km] = fx["zs"]
    for k in range(km - 1, -1, -1):
        zh[:, :, k] = zh[:, :, k + 1] - dzh[:, :, k]
    zh[fx["halo"], :] = _SENT3
    w = np.zeros((fi, fj, km))
    w[fx["halo"], :] = _SENT3
    fx["zh"] = zh
    fx["w"] = w
    fx["ws"] = np.zeros((ni, nj))
    return fx


def _run_native_riem3(dt, bd, km, akap, ptop, fx, p_fac, *, use_logp,
                      last_call, fp_out):
    """Functional adapter over the in-place native riem_solver3.

    Shared driver signature for the impl-parameterized certificate (the
    JAX twin in test_fv3_nh_core.py plugs its own adapter in here):
    copies the 8 in/out arrays out of the fixture dict, runs the native
    solver, and returns them as a dict without mutating ``fx``."""
    from legoesm.core.fv3_native_nh_core import riem_solver3

    out = {k: np.array(fx[k], copy=True)
           for k in ("w", "delz", "zh", "pe", "ppe", "pk3", "pk", "peln")}
    riem_solver3(1, dt, bd, km, akap, 1004.6, ptop, fx["zs"], out["w"],
                 out["delz"], fx["pt"], fx["delp"], out["zh"], out["pe"],
                 out["ppe"], out["pk3"], out["pk"], out["peln"],
                 fx["ws"], p_fac, 1.0, use_logp=use_logp,
                 last_call=last_call, fp_out=fp_out)
    return out


def _match(got, want, tol, ctx):
    """Bitwise when tol == 0 (the native lane IS the replumb's op
    sequence); own-scale relative bound otherwise (the JAX lane differs
    from the NumPy replumb by XLA exp/log/FMA ULPs)."""
    if tol == 0.0:
        assert np.array_equal(got, want), ctx
    else:
        d = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
        assert d <= tol, (ctx, d)


def riem3_flags_footprint_certificate(run_riem3, last_call, fp_out,
                                      use_logp, replumb_tol=0.0):
    """Every flag branch, every written window, sentinel footprint, and
    an independent per-column re-plumb of nh_core.F90:87-202 feeding the
    NATIVE sim1_solver (which carries its own dense certificate).
    Impl-parameterized; ``replumb_tol`` keeps the native lane bitwise
    and gives the JAX lane a measured ULP-scale bound against the SAME
    NumPy replumb."""
    bd = _BDR(4, 10, 9, 13, 3)         # ni=7 != nj=5, is != js
    km = KM
    fx = _riem3_fixture(bd)
    ni, nj, o = fx["ni"], fx["nj"], fx["o"]
    ptop = 100.0
    dt = 100.0
    akap = FV3_KAPPA
    gama = 1.0 / (1.0 - akap)

    w_in = np.array(fx["w"], copy=True)
    zh_in = np.array(fx["zh"], copy=True)

    fx = dict(fx)
    out = run_riem3(dt, bd, km, akap, ptop, fx, 0.05,
                    use_logp=use_logp, last_call=last_call,
                    fp_out=fp_out)
    for k in ("w", "delz", "zh", "pe", "ppe", "pk3", "pk", "peln"):
        fx[k] = out[k]

    # ---- independent per-column re-plumb (nh_core.F90:87-202) ----
    peln1 = float(np.log(ptop))
    ptk = float(np.exp(akap * peln1))
    rgrav = 1.0 / FV3_GRAV
    for jc in range(nj):
        jj = o + jc
        dm = np.array(fx["delp"][o:o + ni, jj, :km])
        pem = np.zeros((ni, km + 1))
        peln2 = np.zeros((ni, km + 1))
        pem[:, 0] = ptop
        peln2[:, 0] = peln1
        for k in range(1, km + 1):
            pem[:, k] = pem[:, k - 1] + dm[:, k - 1]
            peln2[:, k] = np.log(pem[:, k])
        pm2 = np.empty((ni, km))
        dz2 = np.empty((ni, km))
        w2 = np.empty((ni, km))
        for k in range(km):
            pm2[:, k] = dm[:, k] / (peln2[:, k + 1] - peln2[:, k])
            dm[:, k] = dm[:, k] * rgrav
            dz2[:, k] = zh_in[o:o + ni, jj, k + 1] - zh_in[o:o + ni, jj, k]
            w2[:, k] = w_in[o:o + ni, jj, k]
        pe2 = np.zeros((ni, km + 1))
        sim1_solver(dt, 0, ni - 1, km, FV3_RDGAS, gama, akap, pe2, dm,
                    pm2, pem, w2, dz2,
                    np.array(fx["pt"][o:o + ni, jj, :km]),
                    np.array(fx["ws"][:, jc]), 0.05)

        # w and delz out
        _match(fx["w"][o:o + ni, jj, :km], w2, replumb_tol, ("w", jc))
        _match(fx["delz"][:, jc, :], dz2, replumb_tol, ("delz", jc))
        # ppe: PERTURBATION unless fp_out (the r2 mutant's kill line --
        # top must be exactly 0, not ptop)
        want_ppe = pe2 + pem if fp_out else pe2
        _match(fx["ppe"][o:o + ni, jj, :], want_ppe, replumb_tol,
               ("ppe", jc))
        if not fp_out:
            # Exact in BOTH lanes: pe2's top row is a hard zero.
            assert np.all(fx["ppe"][o:o + ni, jj, 0] == 0.0)
        # pk3: k=0 always ptk; interior exp(akap*peln2), overwritten to
        # peln2 for k>=1 when use_logp
        assert np.all(fx["pk3"][o:o + ni, jj, 0] == ptk)
        for k in range(1, km + 1):
            want = peln2[:, k] if use_logp else np.exp(akap * peln2[:, k])
            _match(fx["pk3"][o:o + ni, jj, k], want, replumb_tol,
                   ("pk3", jc, k))
        # zh rebuilt from zs upward WITHOUT grav (bottom row is an exact
        # copy of zs in both lanes)
        assert np.array_equal(fx["zh"][o:o + ni, jj, km],
                              fx["zs"][o:o + ni, jj]), jc
        for k in range(km - 1, -1, -1):
            _match(fx["zh"][o:o + ni, jj, k],
                   fx["zh"][o:o + ni, jj, k + 1] - dz2[:, k],
                   replumb_tol, ("zh", jc, k))
        # last_call windows.  pk copies pk3 BEFORE any use_logp
        # overwrite (:164-172 precedes :187-193), so pk always holds the
        # EXP form -- asserted independently, not against pk3.
        if last_call:
            _match(fx["peln"][:, :, jc], peln2, replumb_tol,
                   ("peln", jc))
            want_pk = np.empty((ni, km + 1))
            want_pk[:, 0] = ptk
            for k in range(1, km + 1):
                want_pk[:, k] = np.exp(akap * peln2[:, k])
            _match(fx["pk"][:, jc, :], want_pk, replumb_tol, ("pk", jc))
            _match(fx["pe"][1:1 + ni, :, jc + 1], pem, replumb_tol,
                   ("pe", jc))
        # Non-vacuity per column: the solve moved w and pe2 is nonzero.
        assert np.abs(pe2).max() > 1.0, jc

    # ---- footprint: halos and unwritten windows keep their sentinel ----
    halo = fx["halo"]
    assert np.all(fx["w"][halo, :] == _SENT3)
    assert np.all(fx["zh"][halo, :] == _SENT3)
    assert np.all(fx["ppe"][halo, :] == _SENT3)
    assert np.all(fx["pk3"][halo, :] == _SENT3)
    if last_call:
        # pe ring rows/slots are NEVER written (upstream pe_halo owns them)
        assert np.all(fx["pe"][0, :, :] == _SENT3)
        assert np.all(fx["pe"][ni + 1, :, :] == _SENT3)
        assert np.all(fx["pe"][:, :, 0] == _SENT3)
        assert np.all(fx["pe"][:, :, nj + 1] == _SENT3)
    else:
        assert np.all(fx["pe"] == _SENT3)
        assert np.all(fx["peln"] == _SENT3)
        assert np.all(fx["pk"] == _SENT3)
    # ppe/pk3/delz compute windows were all written (sentinel gone)
    assert np.all(fx["ppe"][o:o + ni, o:o + nj, :] != _SENT3)
    assert np.all(fx["pk3"][o:o + ni, o:o + nj, :] != _SENT3)
    assert np.all(fx["delz"] != _SENT3)


@pytest.mark.parametrize("last_call,fp_out,use_logp",
                         [(True, False, False),   # the pinned deck
                          (False, True, True)])   # every flag flipped
def test_riem_solver3_certificate_flags_and_footprint(last_call, fp_out,
                                                      use_logp):
    riem3_flags_footprint_certificate(_run_native_riem3, last_call,
                                      fp_out, use_logp)


def test_riem_solver3_dead_arm_raises():
    from legoesm.core.fv3_native_nh_core import riem_solver3

    bd = _BDR(4, 10, 9, 13, 3)
    fx = _riem3_fixture(bd)
    with pytest.raises(NotImplementedError, match="a_imp"):
        riem_solver3(1, 100.0, bd, KM, FV3_KAPPA, 1004.6, 100.0,
                     fx["zs"], fx["w"], fx["delz"], fx["pt"], fx["delp"],
                     fx["zh"], fx["pe"], fx["ppe"], fx["pk3"], fx["pk"],
                     fx["peln"], fx["ws"], 0.05, 0.5)


# --------------------------------------------------------------------------
# codex NH r2 #3 (HIGH): the uniform-zh invariance test also passes for a
# routine that RETURNS IMMEDIATELY.  This is the strong companion:
# nonuniform zh, asymmetric per-row fluxes, an independent re-plumb of
# nh_utils.F90:236-309 (sharing only the separately-certified primitives
# fv_tp_2d / del6_vt_flux / edge_profile -- the same trust structure as
# the update_dz_c reference), full-array equality including ghost cells,
# and explicit damp/ndif km-slot mutation checks.
# --------------------------------------------------------------------------

def test_update_dz_d_nonuniform_vs_replumbed_reference():
    from legoesm.core.fv3_native_d_sw import del6_vt_flux, fv_tp_2d
    from legoesm.core.fv3_native_nh_core import edge_profile, update_dz_d
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        build_fv3_native_gridstruct,
        fort,
    )

    n, ng = 12, 3
    bd = Bounds.single_tile(n, ng)
    full = n + 2 * ng
    km = KM
    hord = 6
    gs = dict(build_fv3_native_gridstruct(n, ng, tile=1,
                                          radius=FV3_RADIUS_M,
                                          omega=FV3_OMEGA))
    gs.update(bounded_domain=False, grid_type=0, sw_corner=True,
              se_corner=True, nw_corner=True, ne_corner=True)
    rng = np.random.default_rng(83)
    area = np.abs(4.0e11 * (1.0 + 0.05 * rng.standard_normal((full, full))))
    rarea = 1.0 / area
    gs["area"] = area
    gs["rarea"] = rarea

    crx = 0.2 * rng.standard_normal((n + 1, full, km))
    # 1e10 (~2.5% of area): at 1e6 the flux-form update moves zh by only
    # ~1e-2 m and the no-op-killer floor below would not bind.
    xfx = 1.0e10 * rng.standard_normal((n + 1, full, km))
    cry = 0.2 * rng.standard_normal((full, n + 1, km))
    yfx = 1.0e10 * rng.standard_normal((full, n + 1, km))

    levels = np.array([(km - k) * 3000.0 + 5000.0 for k in range(km + 1)])
    zh0 = (np.broadcast_to(levels, (full, full, km + 1)).copy()
           + 150.0 * rng.standard_normal((full, full, km + 1)))
    zs = np.array(zh0[:, :, km], copy=True)
    damp0 = np.array([1.0e6, 0.0, 1.0e6, 0.0, 1.0e6, 999.0])
    ndif0 = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 999.0])
    rdt = 1.0 / 100.0

    zh_port = np.array(zh0, copy=True)
    ws_port = np.zeros((n, n))
    damp_port = np.array(damp0, copy=True)
    ndif_port = np.array(ndif0, copy=True)
    update_dz_d(ndif_port, damp_port, hord, bd, km, n + 1, n + 1, area,
                rarea, np.full(km, 1.0e4), zs, zh_port, crx, cry, xfx,
                yfx, ws_port, rdt, gs, lim_fac=1.0)

    # damp/ndif km-slot mutation (:231-232)
    assert damp_port[km] == damp0[km - 1]
    assert ndif_port[km] == ndif0[km - 1]
    assert np.array_equal(damp_port[:km], damp0[:km])

    # ------------------- independent re-plumb -------------------
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = is_ - ng, js - ng
    ied, jed = ie + ng, je + ng
    njd = jed - jsd + 1
    damp = np.array(damp0, copy=True)
    ndif = np.array(ndif0, copy=True)
    damp[km] = damp[km - 1]
    ndif[km] = ndif[km - 1]
    dp0 = np.full(km, 1.0e4)

    ni_x = ie + 1 - is_ + 1
    nj_y = je + 1 - js + 1
    nid = ied - isd + 1
    crx_adv = np.zeros((ni_x, njd, km + 1))
    xfx_adv = np.zeros((ni_x, njd, km + 1))
    cry_adv = np.zeros((nid, nj_y, km + 1))
    yfx_adv = np.zeros((nid, nj_y, km + 1))
    for jj in range(njd):
        crx_adv[:, jj, :], xfx_adv[:, jj, :] = edge_profile(
            crx[:, jj, :], xfx[:, jj, :], jsd + jj, km, dp0, False, 0)
        j_f = jsd + jj
        if js <= j_f <= je + 1:
            j2 = j_f - js
            cry_adv[:, j2, :], yfx_adv[:, j2, :] = edge_profile(
                cry[:, j2, :], yfx[:, j2, :], j_f, km, dp0, False, 0)

    gsf = dict(gs)
    for key in ("area", "rarea", "dxa", "dya", "del6_u", "del6_v"):
        if key in gsf and not isinstance(gsf[key], fort):
            gsf[key] = fort(np.asarray(gsf[key]), isd, jsd)

    zh_ref = np.array(zh0, copy=True)
    afort = fort(area, isd, jsd)
    rfort = fort(rarea, isd, jsd)
    zhreff = fort(zh_ref, isd, jsd)
    for k in range(km + 1):
        ra_x = np.zeros((ie - is_ + 1, njd))
        rax = fort(ra_x, is_, jsd)
        for jj in range(njd):
            j = jsd + jj
            for i in range(is_, ie + 1):
                rax[i, j] = (afort[i, j] + xfx_adv[i - is_, jj, k]
                             - xfx_adv[i - is_ + 1, jj, k])
        ra_y = np.zeros((nid, je - js + 1))
        ray = fort(ra_y, isd, js)
        for j in range(js, je + 1):
            for i in range(isd, ied + 1):
                ray[i, j] = (afort[i, j] + yfx_adv[i - isd, j - js, k]
                             - yfx_adv[i - isd, j - js + 1, k])
        crx_k = fort(np.ascontiguousarray(crx_adv[:, :, k]), is_, jsd)
        xfx_k = fort(np.ascontiguousarray(xfx_adv[:, :, k]), is_, jsd)
        cry_k = fort(np.ascontiguousarray(cry_adv[:, :, k]), isd, js)
        yfx_k = fort(np.ascontiguousarray(yfx_adv[:, :, k]), isd, js)
        fxk = fort(np.zeros((ni_x, je - js + 1)), is_, js)
        fyk = fort(np.zeros((ie - is_ + 1, nj_y)), is_, js)
        if damp[k] > 1.0e-5:
            z2 = np.array(zh_ref[:, :, k], copy=True)
            z2f = fort(z2, isd, jsd)
            fv_tp_2d(z2f, crx_k, cry_k, n + 1, n + 1, hord, fxk, fyk,
                     xfx_k, yfx_k, gsf, bd, rax, ray, 1.0)
            wk2 = fort(np.zeros((nid, njd)), isd, jsd)
            fx2 = fort(np.zeros((nid + 1, njd)), isd, jsd)
            fy2 = fort(np.zeros((nid, njd + 1)), isd, jsd)
            del6_vt_flux(int(ndif[k]), n + 1, n + 1, float(damp[k]), z2f,
                         wk2, fx2, fy2, gsf, bd)
            for j in range(js, je + 1):
                for i in range(is_, ie + 1):
                    zhreff[i, j, k] = (
                        (z2f[i, j] * afort[i, j] + fxk[i, j] - fxk[i + 1, j]
                         + fyk[i, j] - fyk[i, j + 1])
                        / (rax[i, j] + ray[i, j] - afort[i, j])
                        + (fx2[i, j] - fx2[i + 1, j] + fy2[i, j]
                           - fy2[i, j + 1]) * rfort[i, j])
        else:
            z2f = fort(zh_ref[:, :, k], isd, jsd)
            fv_tp_2d(z2f, crx_k, cry_k, n + 1, n + 1, hord, fxk, fyk,
                     xfx_k, yfx_k, gsf, bd, rax, ray, 1.0)
            for j in range(js, je + 1):
                for i in range(is_, ie + 1):
                    zhreff[i, j, k] = (
                        (z2f[i, j] * afort[i, j] + fxk[i, j] - fxk[i + 1, j]
                         + fyk[i, j] - fyk[i, j + 1])
                        / (rax[i, j] + ray[i, j] - afort[i, j]))
    ws_ref = np.zeros((n, n))
    wsreff = fort(ws_ref, is_, js)
    zsf = fort(zs, isd, jsd)
    for j in range(js, je + 1):
        for i in range(is_, ie + 1):
            wsreff[i, j] = (zsf[i, j] - zhreff[i, j, km]) * rdt
        for k in range(km - 1, -1, -1):
            for i in range(is_, ie + 1):
                zhreff[i, j, k] = max(zhreff[i, j, k],
                                      zhreff[i, j, k + 1] + DZ_MIN)

    # FULL-array equality: compute window AND ghost cells (the undamped
    # branch aliases zh into fv_tp_2d, whose copy_corners mutates ghost
    # corner blocks in place -- the reference reproduces exactly that).
    assert np.array_equal(zh_port, zh_ref)
    assert np.array_equal(ws_port, ws_ref)
    # No-op mutant killer: the transport moved the interior.
    sl = slice(ng, ng + n)
    assert np.abs(zh_port[sl, sl, :] - zh0[sl, sl, :]).max() > 1.0
    assert np.abs(ws_port).max() > 0.0
    # Damped levels work on a COPY: their ghost cells are untouched.
    halo = np.ones((full, full), dtype=bool)
    halo[sl, sl] = False
    for k in range(km + 1):
        if damp[k] > 1.0e-5:
            assert np.array_equal(zh_port[halo, k], zh0[halo, k]), k


# --------------------------------------------------------------------------
# codex NH r2 #7: edge_profile's uniform and limiter branches were
# untested (and its bottom-ratio comment was reversed -- fixed in the
# source).  Dense-solve certificate for the uniform branch; exact
# zero-crossing semantics for the limiter.
# --------------------------------------------------------------------------

def test_edge_profile_uniform_branch_dense_solve():
    """nh_utils.F90:1552-1581.  The recurrence is Thomas on
      row 1:      3 qe(1) + 7 qe(2)              = 4 q(1) + 2 q(2)
      rows 2..km: qe(k-1) + 4 qe(k) + qe(k+1)    = 3 (q(k-1) + q(k))
      row km+1:   3.5 qe(km) + 1.5 qe(km+1)      = 4 q(km) + q(km-1)
    (row-1/row-km+1 coefficients read off the eliminated forms:
    gak(1) = 7/3 with rhs (4/3) q1 + (2/3) q2, and the bottom's
    bet = 1/(1.5 - 3.5 gak(km)))."""
    from legoesm.core.fv3_native_nh_core import edge_profile

    rng = np.random.default_rng(97)
    ni, km = 4, KM
    q1 = 10.0 + rng.standard_normal((ni, km))
    q2 = -3.0 + rng.standard_normal((ni, km))
    dp0 = np.full(km, 4.0e3)     # ignored by the uniform branch

    qe1, qe2 = edge_profile(q1, q2, 0, km, dp0, True, 0)

    for i in range(ni):
        for q, qe in ((q1, qe1), (q2, qe2)):
            A = np.zeros((km + 1, km + 1))
            d = np.zeros(km + 1)
            A[0, 0] = 3.0
            A[0, 1] = 7.0
            d[0] = 4.0 * q[i, 0] + 2.0 * q[i, 1]
            for k in range(1, km):
                A[k, k - 1] = 1.0
                A[k, k] = 4.0
                A[k, k + 1] = 1.0
                d[k] = 3.0 * (q[i, k - 1] + q[i, k])
            A[km, km - 1] = 3.5
            A[km, km] = 1.5
            d[km] = 4.0 * q[i, km - 1] + q[i, km - 2]
            want = np.linalg.solve(A, d)
            r = np.abs(qe[i] - want).max() / np.abs(want).max()
            assert r < 1e-12, (i, r)
    # MEASURED ORACLE QUIRK, not a port bug: the uniform branch's TOP row
    # (1.5 qe1 + 3.5 qe2 = 2 q1 + q2) does NOT preserve constants, while
    # the nonuniform branch at g0=1 has rhs 4 q1 + q2 and does.  A
    # constant input therefore comes back distorted at the top edge
    # (7.5 -> -19.2 on this km).  The uniform branch is dead on the
    # pinned deck (update_dz_d hardcodes uniform_grid=.false.); this
    # certificate pins the literal transcription, quirk included.
    qc = np.full((ni, km), 7.5)
    e1, _ = edge_profile(qc, qc, 0, km, dp0, True, 0)
    assert np.abs(e1[:, 0] - 7.5).max() > 1.0        # top row distorts
    en, _ = edge_profile(qc, qc, 0, km, dp0, False, 0)
    assert np.abs(en - 7.5).max() < 1e-12            # nonuniform doesn't


def test_edge_profile_limiter_clamps_zero_crossings():
    """nh_utils.F90:1623-1633: limiter != 0 zeroes an edge value whose
    sign OPPOSES the adjacent interior value, at the top and bottom
    edges only; interior edges are untouched."""
    from legoesm.core.fv3_native_nh_core import edge_profile

    rng = np.random.default_rng(101)
    ni, km = 4, KM
    dp0 = np.abs(1.0e4 + 2.0e3 * rng.standard_normal(km))
    # The edge value's sign vs the adjacent cell's is set by the WHOLE
    # column (back-substitution), so a sign-opposed top edge cannot be
    # dialled in analytically -- search a small deterministic ladder of
    # top-cell values and fail loudly if none opposes (q1 = -5 with a
    # +10 interior gives qe1 = -29.8: SAME sign, no clamp -- measured).
    q2 = 10.0 + np.abs(rng.standard_normal((ni, km)))     # never clamps
    for q_top in (-0.01, -0.1, -1.0, 0.01, 0.1, 1.0):
        q1 = 10.0 + rng.standard_normal((ni, km))
        q1[:, 0] = q_top
        base1, base2 = edge_profile(q1, q2, 0, km, dp0, False, 0)
        if (q1[:, 0] * base1[:, 0] < 0.0).any():
            break
    else:
        pytest.fail("no ladder value produced a sign-opposed top edge")
    lim1, lim2 = edge_profile(q1, q2, 0, km, dp0, False, 1)

    # Interior edges identical.
    assert np.array_equal(lim1[:, 1:km], base1[:, 1:km])
    assert np.array_equal(lim2[:, 1:km], base2[:, 1:km])
    # Exact where-rule at the two boundary edges.
    want_top1 = np.where(q1[:, 0] * base1[:, 0] < 0.0, 0.0, base1[:, 0])
    want_bot1 = np.where(q1[:, km - 1] * base1[:, km] < 0.0, 0.0,
                         base1[:, km])
    assert np.array_equal(lim1[:, 0], want_top1)
    assert np.array_equal(lim1[:, km], want_bot1)
    assert np.array_equal(lim2[:, 0], base2[:, 0])
    assert np.array_equal(lim2[:, km], base2[:, km])
    # Non-vacuity: at least one column actually clamped.
    assert np.any(lim1[:, 0] != base1[:, 0])
