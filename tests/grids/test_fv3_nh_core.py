"""Certification of the JAX SIM1_solver against the NumPy fp64 lane.

Four gates (the NH JAX-mirror pattern-setter):

1. equivalence vs ``fv3_native_nh_core.sim1_solver`` on the SAME
   fixtures, <= 1e-15 relative (bitwise status reported in the assert
   message);
2. the impl-parameterized truth-tier certificates from
   ``test_fv3_native_nh_core`` (dense tridiagonal solve, balanced-rest)
   rerun against the JAX driver — no duplicated numerics;
3. jit-vs-eager parity + no-retrace-across-calls (trace counter);
4. ``jax.test_util.check_grads(order=2)`` away from the p_fac floor
   (the ``jnp.maximum`` floor is C^0 at its boundary — documented in the
   module, not force-tested there).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm.core.fv3_nh_core import (  # noqa: E402
    make_riem_solver3_jit,
    make_riem_solver_c_jit,
    make_sim1_solver_jit,
    riem_solver3 as riem3_jax,
    riem_solver3_jit,
    riem_solver_c as riem_c_jax,
    riem_solver_c_jit,
    sim1_solver as sim1_jax,
    sim1_solver_jit,
)
from legoesm.core.fv3_native_nh_core import (  # noqa: E402
    sim1_solver as sim1_np,
)
from legoesm.grids.fv3_native_gridstruct import (  # noqa: E402
    FV3_GRAV,
    FV3_KAPPA,
    FV3_RDGAS,
)

from tests.grids.test_fv3_native_nh_core import (  # noqa: E402
    KM,
    NI,
    _balanced_column,
    _BD,
    _BDR,
    _riem3_balanced_fixture,
    _riem3_fixture,
    _riem_c_fixture,
    _run_native_riem3,
    _run_native_riem_c,
    _run_native_sim1,
    riem3_flags_footprint_certificate,
    riem_c_contracts_balanced_certificate,
    riem_c_origin_relabel_certificate,
    riem_c_unbalanced_certificate,
    sim1_balanced_rest_certificate,
    sim1_dense_certificate,
)

GAMA = 1.0 / (1.0 - FV3_KAPPA)


def _run_jax_sim1(dt, rgas, gama, kappa, dm, pm2, pem, w2, dz2, pt2,
                  ws, p_fac, jit=False):
    """Functional driver with the shared certificate signature."""
    fn = sim1_solver_jit if jit else sim1_jax
    pe, w2o, dz2o = fn(dt, rgas, gama, kappa,
                       jnp.asarray(dm), jnp.asarray(pm2), jnp.asarray(pem),
                       jnp.asarray(w2), jnp.asarray(dz2), jnp.asarray(pt2),
                       jnp.asarray(ws), p_fac)
    return np.asarray(pe), np.asarray(w2o), np.asarray(dz2o)


def _perturbed_inputs(seed=7):
    """The dense-certificate fixture (same construction, same seeds)."""
    rng = np.random.default_rng(seed)
    _, _, dm, pm2, pem, pt2, dz0 = _balanced_column(rng)
    dz2 = np.array(dz0) * (1.0 + 0.05 * np.sin(np.arange(KM)))
    w2 = 0.3 * np.random.default_rng(8).standard_normal((NI, KM))
    ws = 0.7 * np.ones(NI)
    return dm, pm2, pem, w2, dz2, pt2, ws


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


# ---------------------------------------------------------------- gate 1
# Tolerances PINNED TO MEASUREMENT (probe job 9369392, x64 CPU, this
# fixture): the two lanes run the identical operation sequence but XLA
# CPU's mul+add contraction (FMA) yields few-ULP divergence, so bitwise
# equality is NOT achievable and the honest bound is ULP-of-own-scale:
#   perturbed:  pe rel 3.21e-15, w2 rel 3.27e-15, dz2 rel 1.92e-17
#   balanced:   outputs are ~1e-13 roundoff noise (own_max pe 1.08e-13,
#               w2 9.41e-14); absdiffs pe 9.2e-15, w2 1.0e-14;
#               dz2 rel 1.54e-16
# Bounds below = measured x (3..10) margin.  dz2 (the smooth, full-scale
# quantity) meets the 1e-15 target; pe/w2 are cancellation-amplified
# perturbation quantities and sit at ~3e-15 of their own (small) scale.
@pytest.mark.parametrize("case", ["perturbed", "balanced"])
def test_sim1_jax_matches_numpy_lane(case):
    if case == "perturbed":
        dm, pm2, pem, w2, dz2, pt2, ws = _perturbed_inputs()
        dt = 100.0
    else:
        rng = np.random.default_rng(11)
        _, _, dm, pm2, pem, pt2, dz2 = _balanced_column(rng)
        w2 = np.zeros((NI, KM))
        ws = np.zeros(NI)
        dt = 1920.0
    p_fac = 0.05

    pe_n, w_n, dz_n = _run_native_sim1(
        dt, FV3_RDGAS, GAMA, FV3_KAPPA, dm, pm2, pem, w2, dz2, pt2, ws,
        p_fac)
    pe_j, w_j, dz_j = _run_jax_sim1(
        dt, FV3_RDGAS, GAMA, FV3_KAPPA, dm, pm2, pem, w2, dz2, pt2, ws,
        p_fac)

    assert _rel(dz_j, dz_n) <= 1e-15, _rel(dz_j, dz_n)
    if case == "perturbed":
        assert _rel(pe_j, pe_n) <= 1e-14, _rel(pe_j, pe_n)
        assert _rel(w_j, w_n) <= 1e-14, _rel(w_j, w_n)
        # Non-vacuity: the solve moved state (both lanes).
        assert np.abs(w_n - w2).max() > 1e-3
    else:
        # Balanced case: pe/w are roundoff-level; relative-to-own-scale
        # is noise/noise, so bound the ABSOLUTE roundoff instead.
        assert np.abs(pe_j - pe_n).max() <= 1e-13, np.abs(pe_j - pe_n).max()
        assert np.abs(w_j - w_n).max() <= 1e-13, np.abs(w_j - w_n).max()


# ---------------------------------------------------------------- gate 2
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sim1_jax_dense_certificate(jit):
    sim1_dense_certificate(
        lambda *a: _run_jax_sim1(*a, jit=jit))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sim1_jax_balanced_rest_certificate(jit):
    sim1_balanced_rest_certificate(
        lambda *a: _run_jax_sim1(*a, jit=jit))


# ---------------------------------------------------------------- gate 3
def test_sim1_jax_jit_eager_parity_and_no_retrace():
    dm, pm2, pem, w2, dz2, pt2, ws = _perturbed_inputs()
    args = (100.0, FV3_RDGAS, GAMA, FV3_KAPPA, dm, pm2, pem, w2, dz2,
            pt2, ws, 0.05)
    eager = _run_jax_sim1(*args)

    traces = {"n": 0}

    def _counted(*a, **kw):
        traces["n"] += 1
        return sim1_jax(*a, **kw)

    # Built through the PRODUCTION jit factory (codex SIM1 r1 #2), so
    # this measures the same static-argnums policy sim1_solver_jit uses.
    fn = make_sim1_solver_jit(_counted)

    def _call(w2_, dz2_):
        return fn(100.0, FV3_RDGAS, GAMA, FV3_KAPPA,
                  jnp.asarray(dm), jnp.asarray(pm2), jnp.asarray(pem),
                  jnp.asarray(w2_), jnp.asarray(dz2_), jnp.asarray(pt2),
                  jnp.asarray(ws), 0.05)

    jit1 = _call(w2, dz2)
    # Different VALUES, same shapes/dtypes -> must NOT retrace.
    jit2 = _call(w2 * 1.01, dz2 * 0.99)
    assert traces["n"] == 1, traces["n"]
    del jit2

    for name, e, j in zip(("pe", "w2", "dz2"), eager, jit1):
        r = _rel(np.asarray(j), e)
        assert r <= 1e-15, (name, r,
                            f"bitwise={np.array_equal(np.asarray(j), e)}")


# ---------------------------------------------------------------- gate 4
def test_sim1_jax_check_grads_order2_away_from_floor():
    """Order-2 fwd+rev grads on a reduced problem, p_fac floor INACTIVE
    (verified below): the kernel is smooth there (exp/log/div + scans).
    At the floor boundary the jnp.maximum makes it C^0 only — that case
    is documented in fv3_nh_core.py, not force-tested."""
    ni, km = 3, 4
    rng = np.random.default_rng(3)
    _, _, dm, pm2, pem, pt2, dz0 = _balanced_column(rng, ni=ni, km=km)
    dz2 = np.array(dz0) * (1.0 + 0.02 * np.sin(np.arange(km)))
    w2 = 0.1 * rng.standard_normal((ni, km))
    ws = 0.3 * np.ones(ni)
    p_fac = 0.05

    # Floor-inactivity control THROUGH THE JAX LANE on the exact
    # gradient fixture (codex SIM1 r1 #3): if any jnp.maximum had picked
    # the floor branch, halving p_fac would change dz2; bitwise-identical
    # dz2 proves the floor is inactive where the grads are taken.
    _, _, dz_a = _run_jax_sim1(100.0, FV3_RDGAS, GAMA, FV3_KAPPA, dm,
                               pm2, pem, w2, dz2, pt2, ws, p_fac)
    _, _, dz_b = _run_jax_sim1(100.0, FV3_RDGAS, GAMA, FV3_KAPPA, dm,
                               pm2, pem, w2, dz2, pt2, ws, p_fac / 2)
    assert np.array_equal(dz_a, dz_b), "p_fac floor ACTIVE on fixture"
    # Margin check (codex SIM1 r2 #1: the halved-p_fac control alone can
    # false-pass on an exact tie at the maximum's kink).  Invert the
    # output equation dz = -dm*rgas*pt * P^(kappa-1) for the SELECTED
    # pressure P and require it strictly above the floor with an
    # FD-safe margin at every level:
    p_sel = np.exp(np.log(-dz_a / (dm * FV3_RDGAS * pt2))
                   / (FV3_KAPPA - 1.0))
    assert (p_sel > 1.01 * p_fac * pm2).all(), \
        "floor margin < 1% somewhere on the gradient fixture"

    def _loss(pe, w2o, dz2o):
        return (jnp.sum(pe * pe) + jnp.sum(w2o * w2o)
                + jnp.sum(dz2o * dz2o) / 1e6)

    def f(w2_, dz2_, pt2_):
        return _loss(*sim1_jax(
            100.0, FV3_RDGAS, GAMA, FV3_KAPPA, jnp.asarray(dm),
            jnp.asarray(pm2), jnp.asarray(pem), w2_, dz2_, pt2_,
            jnp.asarray(ws), p_fac))

    check_grads(f, (jnp.asarray(w2), jnp.asarray(dz2), jnp.asarray(pt2)),
                order=2, modes=("fwd", "rev"))

    # Remaining dynamic operands (codex SIM1 r1 #4): dm2/pm2/pem/ws —
    # ws uniquely exercises the bottom-row sensitivity.
    def g(dm_, pm_, pem_, ws_):
        return _loss(*sim1_jax(
            100.0, FV3_RDGAS, GAMA, FV3_KAPPA, dm_, pm_, pem_,
            jnp.asarray(w2), jnp.asarray(dz2), jnp.asarray(pt2),
            ws_, p_fac))

    check_grads(g, (jnp.asarray(dm), jnp.asarray(pm2), jnp.asarray(pem),
                    jnp.asarray(ws)), order=2, modes=("fwd", "rev"))


# ------------------------------------------------------------ guards
def test_sim1_jax_rejects_float32():
    dm, pm2, pem, w2, dz2, pt2, ws = _perturbed_inputs()
    bad = jnp.asarray(w2, dtype=jnp.float32)
    with pytest.raises(TypeError, match="float64"):
        sim1_jax(100.0, FV3_RDGAS, GAMA, FV3_KAPPA, jnp.asarray(dm),
                 jnp.asarray(pm2), jnp.asarray(pem), bad,
                 jnp.asarray(dz2), jnp.asarray(pt2), jnp.asarray(ws),
                 0.05)


def test_sim1_jax_km_guard():
    """km=1 rejected (the oracle recurrence divides by bb[:,0]=0)."""
    ni, km = 2, 1
    a = jnp.ones((ni, km), jnp.float64)
    with pytest.raises(ValueError, match="km=1"):
        sim1_jax(100.0, FV3_RDGAS, GAMA, FV3_KAPPA, a, a,
                 jnp.ones((ni, km + 1), jnp.float64), a, -a, a,
                 jnp.ones(ni, jnp.float64), 0.05)


def test_sim1_jax_km2_matches_numpy_lane():
    """km=2 IS valid (codex SIM1 r1 #1): the w forward sweep is empty
    (zero-length scan) and the bottom row reads w2[:, km-2] == w2[:, 0].
    Parity vs the native lane on a perturbed km=2 column."""
    ni, km = 5, 2
    rng = np.random.default_rng(19)
    _, _, dm, pm2, pem, pt2, dz0 = _balanced_column(rng, ni=ni, km=km)
    dz2 = np.array(dz0) * (1.0 + 0.05 * np.sin(np.arange(km)))
    w2 = 0.3 * rng.standard_normal((ni, km))
    ws = 0.7 * np.ones(ni)
    args = (100.0, FV3_RDGAS, GAMA, FV3_KAPPA, dm, pm2, pem, w2, dz2,
            pt2, ws, 0.05)
    pe_n, w_n, dz_n = _run_native_sim1(*args)
    pe_j, w_j, dz_j = _run_jax_sim1(*args)
    assert np.isfinite(pe_n).all() and np.isfinite(w_n).all()
    assert _rel(dz_j, dz_n) <= 1e-15, _rel(dz_j, dz_n)
    assert _rel(pe_j, pe_n) <= 1e-14, _rel(pe_j, pe_n)
    assert _rel(w_j, w_n) <= 1e-14, _rel(w_j, w_n)
    # Non-vacuity: the solve moved state.
    assert np.abs(w_n - w2).max() > 1e-4


# ======================================================================
# Riem_Solver_c (C stage) — JAX twin certification
# ======================================================================

def _bounds(bd):
    """The JAX lane takes a STATIC (is_, ie, js, je, ng) int tuple in
    place of the NumPy lane's bd shim (hashable by value)."""
    return (bd.is_, bd.ie, bd.js, bd.je, bd.ng)


def _run_jax_riem_c(dt, bd, km, akap, ptop, hs, w3, pt, delp, gz, pef,
                    ws, p_fac, jit=False):
    """Functional driver with the shared riem_c certificate signature."""
    fn = riem_solver_c_jit if jit else riem_c_jax
    gz_o, pef_o = fn(1, dt, _bounds(bd), km, akap, 1004.6, ptop,
                     jnp.asarray(hs), jnp.asarray(w3), jnp.asarray(pt),
                     jnp.asarray(delp), jnp.asarray(gz),
                     jnp.asarray(pef), jnp.asarray(ws), p_fac, 1.0)
    return np.asarray(gz_o), np.asarray(pef_o)


# ---------------------------------------------------------------- gate 1
# Tolerances PINNED TO MEASUREMENT (probe job 9369505, x64 CPU, these
# fixtures): both lanes run the same op sequence but XLA's exp/log
# kernels and mul+add (FMA) contraction give few-ULP divergence:
#   unbalanced: gz rel 7.155e-16, pef rel 1.443e-16
#   balanced:   gz rel 6.115e-16, pef rel 1.400e-16
# Bounds below = measured x (~7) margin.
@pytest.mark.parametrize("case", ["unbalanced", "balanced"])
def test_riem_c_jax_matches_numpy_lane(case):
    if case == "unbalanced":
        ptop, hs, w3, pt, delp, gz, ws = _riem_c_fixture(
            41, w_amp=0.5, dz_scale=1.05)
    else:
        ptop, hs, w3, pt, delp, gz, ws = _riem_c_fixture(13)
    bd = _BD(12, 3)
    pef0 = np.zeros(gz.shape)
    args = (100.0, bd, KM, FV3_KAPPA, ptop, hs, w3, pt, delp, gz, pef0,
            ws, 0.05)
    gz_n, pef_n = _run_native_riem_c(*args)
    gz_j, pef_j = _run_jax_riem_c(*args)
    assert _rel(gz_j, gz_n) <= 5e-15, _rel(gz_j, gz_n)
    assert _rel(pef_j, pef_n) <= 1e-15, _rel(pef_j, pef_n)
    if case == "unbalanced":
        # Non-vacuity: the solve moved gz and pef holds a real
        # perturbation (pef == pem alone would differ by >1 Pa).
        assert np.abs(gz_n - gz).max() > 1.0


# ---------------------------------------------------------------- gate 2
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_riem_c_jax_contracts_balanced_certificate(jit):
    riem_c_contracts_balanced_certificate(
        lambda *a: _run_jax_riem_c(*a, jit=jit))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_riem_c_jax_unbalanced_certificate(jit):
    riem_c_unbalanced_certificate(
        lambda *a: _run_jax_riem_c(*a, jit=jit))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_riem_c_jax_origin_relabel_certificate(jit):
    riem_c_origin_relabel_certificate(
        lambda *a: _run_jax_riem_c(*a, jit=jit))


# ---------------------------------------------------------------- gate 3
def test_riem_c_jax_jit_eager_parity_and_no_retrace():
    ptop, hs, w3, pt, delp, gz, ws = _riem_c_fixture(
        41, w_amp=0.5, dz_scale=1.05)
    bd = _BD(12, 3)
    pef0 = np.zeros(gz.shape)
    eager = _run_jax_riem_c(100.0, bd, KM, FV3_KAPPA, ptop, hs, w3, pt,
                            delp, gz, pef0, ws, 0.05)

    traces = {"n": 0}

    def _counted(*a, **kw):
        traces["n"] += 1
        return riem_c_jax(*a, **kw)

    fn = make_riem_solver_c_jit(_counted)   # the PRODUCTION jit policy

    def _call(w3_, gz_):
        return fn(1, 100.0, _bounds(bd), KM, FV3_KAPPA, 1004.6, ptop,
                  jnp.asarray(hs), jnp.asarray(w3_), jnp.asarray(pt),
                  jnp.asarray(delp), jnp.asarray(gz_),
                  jnp.asarray(pef0), jnp.asarray(ws), 0.05, 1.0)

    jit1 = _call(w3, gz)
    # Different VALUES, same shapes/dtypes -> must NOT retrace (the
    # bounds tuple hashes by value, so an equal fresh tuple shares the
    # cache entry).
    jit2 = _call(w3 * 1.01, gz * 0.999)
    assert traces["n"] == 1, traces["n"]
    del jit2

    for name, e, j in zip(("gz", "pef"), eager, jit1):
        r = _rel(np.asarray(j), e)
        assert r <= 1e-15, (name, r,
                            f"bitwise={np.array_equal(np.asarray(j), e)}")


# ---------------------------------------------------------------- gate 4
def test_riem_c_jax_check_grads_order2_away_from_floor():
    """Order-2 fwd+rev grads over ALL seven dynamic operands on a small
    (n=3, ng=1, km=4) domain, with the p_fac floor proven INACTIVE on
    the fixture (halved-p_fac bitwise control + margin inversion through
    the OUTPUT gz, same two-step proof as the SIM1 gate)."""
    n, ng, km = 3, 1, 4
    ptop, hs, w3, pt, delp, gz, ws = _riem_c_fixture(
        5, n=n, ng=ng, km=km, w_amp=0.1, dz_scale=1.02)
    ws = 0.3 * np.ones_like(ws)   # nonzero: exercises the bottom -p1*ws
    bd = _BD(n, ng)
    pef0 = np.zeros(gz.shape)
    p_fac = 0.05

    gz_a, pef_a = _run_jax_riem_c(100.0, bd, km, FV3_KAPPA, ptop, hs,
                                  w3, pt, delp, gz, pef0, ws, p_fac)
    gz_b, pef_b = _run_jax_riem_c(100.0, bd, km, FV3_KAPPA, ptop, hs,
                                  w3, pt, delp, gz, pef0, ws, p_fac / 2)
    assert np.array_equal(gz_a, gz_b) and np.array_equal(pef_a, pef_b), \
        "p_fac floor ACTIVE on the riem_c gradient fixture"
    # Margin inversion: the rebuilt gz encodes the SELECTED dz
    # (dz = (gz[k+1]-gz[k])/grav); invert dz = -dm*rgas*pt*P^(kappa-1)
    # for P and require it strictly above the floor with an FD-safe 1%
    # margin (the halved-p_fac control alone can false-pass on a tie).
    dz_sel = (gz_a[:, :, 1:] - gz_a[:, :, :-1]) / FV3_GRAV
    dm = delp / FV3_GRAV
    pem = np.zeros(gz.shape)
    pem[:, :, 0] = ptop
    for k in range(km):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
    pm2 = delp / np.log(pem[:, :, 1:] / pem[:, :, :-1])
    p_sel = np.exp(np.log(-dz_sel / (dm * FV3_RDGAS * pt))
                   / (FV3_KAPPA - 1.0))
    assert (p_sel > 1.01 * p_fac * pm2).all(), \
        "floor margin < 1% somewhere on the riem_c gradient fixture"

    bounds = _bounds(bd)

    def _loss(gz_o, pef_o):
        return jnp.sum(gz_o * gz_o) / 1e6 + jnp.sum(pef_o * pef_o) / 1e8

    def f(w3_, gz_, pt_):
        return _loss(*riem_c_jax(
            1, 100.0, bounds, km, FV3_KAPPA, 1004.6, ptop,
            jnp.asarray(hs), w3_, pt_, jnp.asarray(delp), gz_,
            jnp.asarray(pef0), jnp.asarray(ws), p_fac, 1.0))

    check_grads(f, (jnp.asarray(w3), jnp.asarray(gz), jnp.asarray(pt)),
                order=2, modes=("fwd", "rev"))

    def g(hs_, delp_, ws_, pef_):
        return _loss(*riem_c_jax(
            1, 100.0, bounds, km, FV3_KAPPA, 1004.6, ptop, hs_,
            jnp.asarray(w3), jnp.asarray(pt), delp_, jnp.asarray(gz),
            pef_, ws_, p_fac, 1.0))

    check_grads(g, (jnp.asarray(hs), jnp.asarray(delp),
                    jnp.asarray(ws), jnp.asarray(pef0)),
                order=2, modes=("fwd", "rev"))


# ------------------------------------------------------------ guards
@pytest.mark.parametrize("bad", ["hs", "w3", "pt", "delp", "gz", "pef",
                                 "ws"])
def test_riem_c_jax_rejects_float32_every_operand(bad):
    n, ng = 4, 2
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
        riem_c_jax(1, 100.0, (1, n, 1, n, ng), KM, FV3_KAPPA, 1004.6,
                   100.0, jnp.asarray(args["hs"]), jnp.asarray(args["w3"]),
                   jnp.asarray(args["pt"]), jnp.asarray(args["delp"]),
                   jnp.asarray(args["gz"]), jnp.asarray(args["pef"]),
                   jnp.asarray(args["ws"]), 0.05, 1.0)


def test_riem_c_jax_dead_arm_raises():
    a2 = jnp.zeros((8, 8), jnp.float64)
    a3 = jnp.zeros((8, 8, KM), jnp.float64)
    a3p = jnp.zeros((8, 8, KM + 1), jnp.float64)
    with pytest.raises(NotImplementedError, match="a_imp"):
        riem_c_jax(1, 100.0, (1, 4, 1, 4, 2), KM, FV3_KAPPA, 1004.6,
                   100.0, a2, a3, a3, a3, a3p, a3p, a2, 0.05, 0.4)


def test_riem_c_jax_km1_raises():
    """km=1 is invalid in the oracle recurrence (sim1's guard fires);
    the NumPy lane silently produces non-finite output there — the JAX
    lane raises instead, matching the SIM1 precedent."""
    n, ng, km = 4, 2, 1
    full = n + 2 * ng
    a2 = jnp.zeros((full, full), jnp.float64)
    a3 = jnp.ones((full, full, km), jnp.float64)
    a3p = jnp.ones((full, full, km + 1), jnp.float64)
    with pytest.raises(ValueError, match="km=1"):
        riem_c_jax(1, 100.0, (1, n, 1, n, ng), km, FV3_KAPPA, 1004.6,
                   100.0, a2, a3, 280.0 * a3, 1e4 * a3, -300.0 * a3p,
                   0.0 * a3p, a2, 0.05, 1.0)


# ======================================================================
# Riem_Solver3 (D stage) — JAX twin certification
# ======================================================================

_R3_KEYS = ("w", "delz", "zh", "pe", "ppe", "pk3", "pk", "peln")


def _run_jax_riem3(dt, bd, km, akap, ptop, fx, p_fac, *, use_logp,
                   last_call, fp_out, jit=False):
    """Functional driver with the shared riem3 certificate signature."""
    fn = riem_solver3_jit if jit else riem3_jax
    outs = fn(1, dt, _bounds(bd), km, akap, 1004.6, ptop,
              jnp.asarray(fx["zs"]), jnp.asarray(fx["w"]),
              jnp.asarray(fx["delz"]), jnp.asarray(fx["pt"]),
              jnp.asarray(fx["delp"]), jnp.asarray(fx["zh"]),
              jnp.asarray(fx["pe"]), jnp.asarray(fx["ppe"]),
              jnp.asarray(fx["pk3"]), jnp.asarray(fx["pk"]),
              jnp.asarray(fx["peln"]), jnp.asarray(fx["ws"]),
              p_fac, 1.0, use_logp=use_logp, last_call=last_call,
              fp_out=fp_out)
    return {k: np.asarray(v) for k, v in zip(_R3_KEYS, outs)}


def _cmp_field(name, got, want, rel_tol, abs_tol=0.0):
    """Sentinel-aware lane comparison: sentinel-magnitude entries (halo
    passthrough) must be BITWISE equal; the finite content is bounded
    rel-to-own-scale (or absolutely for roundoff-level quantities)."""
    sent = np.abs(want) >= 1e29
    assert np.array_equal(got[sent], want[sent]), (name, "halo passthru")
    if not (~sent).any():
        return
    d = np.abs(got - want)[~sent].max()
    scale = np.abs(want[~sent]).max()
    assert d <= max(rel_tol * scale, abs_tol), (name, d, scale)


# ---------------------------------------------------------------- gate 1
# Tolerances PINNED TO MEASUREMENT (probe job 9369505, x64 CPU, these
# fixtures):
#   perturbed: max rel over the 8 outputs 1.33e-15 (ppe; w 9.96e-16,
#              delz 1.07e-15, zh 5.7e-16, pk3/pk 1.6e-16, pe/peln 0.0)
#   balanced:  smooth fields rel <= 1.94e-16; the roundoff-level
#              quantities have absdiff w 7.6e-14 (own_max 8.95e-13) and
#              ppe 1.67e-12 (own_max 1.15e-11) — relative-to-own-scale
#              is noise/noise there, so the ABSOLUTE roundoff is bounded.
# Bounds = measured x (~5..10) margin.
@pytest.mark.parametrize("case", ["perturbed", "balanced"])
def test_riem3_jax_matches_numpy_lane(case):
    bd = _BDR(4, 10, 9, 13, 3)
    if case == "perturbed":
        fx = _riem3_fixture(bd)
    else:
        fx = _riem3_balanced_fixture(bd)
    args = (100.0, bd, KM, FV3_KAPPA, 100.0, fx, 0.05)
    kw = dict(use_logp=False, last_call=True, fp_out=False)  # pinned deck
    out_n = _run_native_riem3(*args, **kw)
    out_j = _run_jax_riem3(*args, **kw)
    if case == "perturbed":
        for k in _R3_KEYS:
            _cmp_field(k, out_j[k], out_n[k], 1e-14)
        # Non-vacuity: the solve moved w.
        assert np.abs(out_n["w"][~fx["halo"], :]
                      - fx["w"][~fx["halo"], :]).max() > 1e-3
    else:
        # Balanced: w and ppe are roundoff-level quantities — bound the
        # ABSOLUTE lane difference; the smooth fields keep the rel bound.
        for k in ("delz", "zh", "pe", "pk3", "pk", "peln"):
            _cmp_field(k, out_j[k], out_n[k], 1e-15)
        for k in ("w", "ppe"):
            _cmp_field(k, out_j[k], out_n[k], 0.0, abs_tol=1e-11)
            assert np.abs(out_n[k][np.abs(out_n[k]) < 1e29]).max() < 1e-6


# ---------------------------------------------------------------- gate 2
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("last_call,fp_out,use_logp",
                         [(True, False, False),   # the pinned deck
                          (False, True, True)])   # every flag flipped
def test_riem3_jax_certificate_flags_and_footprint(last_call, fp_out,
                                                   use_logp, jit):
    # replumb_tol pinned to measurement (probe job 9369505: the JAX lane
    # sits <= 1.33e-15 rel of the NumPy replumb; 1e-14 = ~7x margin).
    riem3_flags_footprint_certificate(
        lambda *a, **kw: _run_jax_riem3(*a, jit=jit, **kw),
        last_call, fp_out, use_logp, replumb_tol=1e-14)


# ---------------------------------------------------------------- gate 3
def test_riem3_jax_jit_eager_parity_and_no_retrace():
    bd = _BDR(4, 10, 9, 13, 3)
    fx = _riem3_fixture(bd)
    kw = dict(use_logp=False, last_call=True, fp_out=False)
    eager = _run_jax_riem3(100.0, bd, KM, FV3_KAPPA, 100.0, fx, 0.05,
                           **kw)

    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return riem3_jax(*a, **k)

    fn = make_riem_solver3_jit(_counted)   # the PRODUCTION jit policy

    def _call(w_, zh_):
        return fn(1, 100.0, _bounds(bd), KM, FV3_KAPPA, 1004.6, 100.0,
                  jnp.asarray(fx["zs"]), jnp.asarray(w_),
                  jnp.asarray(fx["delz"]), jnp.asarray(fx["pt"]),
                  jnp.asarray(fx["delp"]), jnp.asarray(zh_),
                  jnp.asarray(fx["pe"]), jnp.asarray(fx["ppe"]),
                  jnp.asarray(fx["pk3"]), jnp.asarray(fx["pk"]),
                  jnp.asarray(fx["peln"]), jnp.asarray(fx["ws"]),
                  0.05, 1.0, **kw)

    jit1 = _call(fx["w"], fx["zh"])
    jit2 = _call(fx["w"] * 1.01, fx["zh"] * 0.999)
    assert traces["n"] == 1, traces["n"]
    del jit2

    for name, j in zip(_R3_KEYS, jit1):
        r = _rel(np.asarray(j), eager[name])
        assert r <= 1e-15, (name, r)


# ---------------------------------------------------------------- gate 4
def test_riem3_jax_check_grads_order2_away_from_floor():
    """Order-2 fwd+rev grads over all twelve dynamic operands (six real
    + six window/halo passthroughs) on a small (3x2, ng=1, km=4)
    sentinel-free fixture, floor proven inactive (bitwise halved-p_fac
    control + margin inversion through the OUTPUT delz)."""
    ni, nj, ng, km = 3, 2, 1, 4
    bd = _BDR(1, ni, 1, nj, ng)
    fi, fj = ni + 2 * ng, nj + 2 * ng
    rng = np.random.default_rng(9)
    ptop = 100.0
    p_fac = 0.05
    gama = 1.0 / (1.0 - FV3_KAPPA)
    delp = np.abs(10000.0 + 300.0 * rng.standard_normal((fi, fj, km)))
    pt = 280.0 + 15.0 * rng.standard_normal((fi, fj, km))
    w = 0.1 * rng.standard_normal((fi, fj, km))
    ws = 0.1 * rng.standard_normal((ni, nj))
    zs = 50.0 * np.abs(rng.standard_normal((fi, fj)))
    pem = np.zeros((fi, fj, km + 1))
    pem[:, :, 0] = ptop
    for k in range(km):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
    peln2 = np.log(pem)
    pm = delp / (peln2[:, :, 1:] - peln2[:, :, :-1])
    dzh = -(delp / FV3_GRAV) * FV3_RDGAS * pt / np.exp(np.log(pm) / gama)
    dzh = dzh * 1.02                    # slightly off-balance, off-floor
    zh = np.empty((fi, fj, km + 1))
    zh[:, :, km] = zs
    for k in range(km - 1, -1, -1):
        zh[:, :, k] = zh[:, :, k + 1] - dzh[:, :, k]
    fx = dict(zs=zs, w=w, delz=np.zeros((ni, nj, km)), pt=pt, delp=delp,
              zh=zh, pe=np.zeros((ni + 2, km + 1, nj + 2)),
              ppe=np.zeros((fi, fj, km + 1)),
              pk3=np.zeros((fi, fj, km + 1)),
              pk=np.zeros((ni, nj, km + 1)),
              peln=np.zeros((ni, km + 1, nj)), ws=ws)
    kw = dict(use_logp=False, last_call=True, fp_out=False)

    out_a = _run_jax_riem3(100.0, bd, km, FV3_KAPPA, ptop, fx, p_fac,
                           **kw)
    out_b = _run_jax_riem3(100.0, bd, km, FV3_KAPPA, ptop, fx,
                           p_fac / 2, **kw)
    for k in _R3_KEYS:
        assert np.array_equal(out_a[k], out_b[k]), \
            (k, "p_fac floor ACTIVE on the riem3 gradient fixture")
    # Margin inversion through the OUTPUT delz:
    o = ng
    dm = delp[o:o + ni, o:o + nj, :] / FV3_GRAV
    pm2w = pm[o:o + ni, o:o + nj, :]
    p_sel = np.exp(np.log(-out_a["delz"]
                          / (dm * FV3_RDGAS * pt[o:o + ni, o:o + nj, :]))
                   / (FV3_KAPPA - 1.0))
    assert (p_sel > 1.01 * p_fac * pm2w).all(), \
        "floor margin < 1% somewhere on the riem3 gradient fixture"

    bounds = _bounds(bd)

    def _loss(outs):
        w_o, delz_o, zh_o, pe_o, ppe_o, pk3_o, pk_o, peln_o = outs
        return (jnp.sum(w_o * w_o) + jnp.sum(delz_o * delz_o) / 1e4
                + jnp.sum(zh_o * zh_o) / 1e6 + jnp.sum(pe_o * pe_o) / 1e8
                + jnp.sum(ppe_o * ppe_o) / 1e8
                + jnp.sum(pk3_o * pk3_o) + jnp.sum(pk_o * pk_o)
                + jnp.sum(peln_o * peln_o))

    def f(zs_, w_, pt_, delp_, zh_, ws_):
        return _loss(riem3_jax(
            1, 100.0, bounds, km, FV3_KAPPA, 1004.6, ptop, zs_, w_,
            jnp.asarray(fx["delz"]), pt_, delp_, zh_,
            jnp.asarray(fx["pe"]), jnp.asarray(fx["ppe"]),
            jnp.asarray(fx["pk3"]), jnp.asarray(fx["pk"]),
            jnp.asarray(fx["peln"]), ws_, p_fac, 1.0, **kw))

    check_grads(f, (jnp.asarray(zs), jnp.asarray(w), jnp.asarray(pt),
                    jnp.asarray(delp), jnp.asarray(zh),
                    jnp.asarray(ws)), order=2, modes=("fwd", "rev"))

    def g(delz_, pe_, ppe_, pk3_, pk_, peln_):
        return _loss(riem3_jax(
            1, 100.0, bounds, km, FV3_KAPPA, 1004.6, ptop,
            jnp.asarray(zs), jnp.asarray(w), delz_, jnp.asarray(pt),
            jnp.asarray(delp), jnp.asarray(zh), pe_, ppe_, pk3_, pk_,
            peln_, jnp.asarray(ws), p_fac, 1.0, **kw))

    check_grads(g, tuple(jnp.asarray(fx[k]) for k in
                         ("delz", "pe", "ppe", "pk3", "pk", "peln")),
                order=2, modes=("fwd", "rev"))


# ------------------------------------------------------------ guards
@pytest.mark.parametrize("bad", ["zs", "w", "delz", "pt", "delp", "zh",
                                 "pe", "ppe", "pk3", "pk", "peln", "ws"])
def test_riem3_jax_rejects_float32_every_operand(bad):
    ni, nj, ng = 3, 2, 2
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
        riem3_jax(1, 100.0, (1, ni, 1, nj, ng), KM, FV3_KAPPA, 1004.6,
                  100.0, *(jnp.asarray(args[k]) for k in
                           ("zs", "w", "delz", "pt", "delp", "zh", "pe",
                            "ppe", "pk3", "pk", "peln", "ws")),
                  0.05, 1.0)


def test_riem3_jax_dead_arm_and_ws_shape_raise():
    ni, nj, ng = 3, 2, 2
    fi, fj = ni + 2 * ng, nj + 2 * ng
    z2 = jnp.zeros((fi, fj), jnp.float64)
    z3 = jnp.zeros((fi, fj, KM), jnp.float64)
    z3p = jnp.zeros((fi, fj, KM + 1), jnp.float64)
    delz = jnp.zeros((ni, nj, KM), jnp.float64)
    pe = jnp.zeros((ni + 2, KM + 1, nj + 2), jnp.float64)
    pk = jnp.zeros((ni, nj, KM + 1), jnp.float64)
    peln = jnp.zeros((ni, KM + 1, nj), jnp.float64)
    ws = jnp.zeros((ni, nj), jnp.float64)
    with pytest.raises(NotImplementedError, match="a_imp"):
        riem3_jax(1, 100.0, (1, ni, 1, nj, ng), KM, FV3_KAPPA, 1004.6,
                  100.0, z2, z3, delz, z3, z3, z3p, pe, z3p, z3p, pk,
                  peln, ws, 0.05, 0.5)
    with pytest.raises(ValueError, match="ws must be"):
        riem3_jax(1, 100.0, (1, ni, 1, nj, ng), KM, FV3_KAPPA, 1004.6,
                  100.0, z2, z3, delz, z3, z3, z3p, pe, z3p, z3p, pk,
                  peln, jnp.zeros((ni, nj + 1), jnp.float64), 0.05, 1.0)
