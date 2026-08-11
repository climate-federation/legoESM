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
    edge_profile as edge_jax,
    edge_profile_jit,
    make_edge_profile_jit,
    make_riem_solver3_jit,
    make_riem_solver_c_jit,
    make_sim1_solver_jit,
    make_update_dz_c_jit,
    riem_solver3 as riem3_jax,
    riem_solver3_jit,
    riem_solver_c as riem_c_jax,
    riem_solver_c_jit,
    sim1_solver as sim1_jax,
    sim1_solver_jit,
    update_dz_c as udzc_jax,
    update_dz_c_jit,
)
from legoesm.core.fv3_native_nh_core import (  # noqa: E402
    DZ_MIN,
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
    _rect_fields,
    _riem3_balanced_fixture,
    _riem3_fixture,
    _riem_c_fixture,
    _run_native_edge_profile,
    _run_native_riem3,
    _run_native_riem_c,
    _run_native_sim1,
    _run_native_update_dz_c,
    edge_profile_limiter_certificate,
    edge_profile_nonuniform_dense_certificate,
    edge_profile_uniform_dense_certificate,
    riem3_flags_footprint_certificate,
    riem_c_contracts_balanced_certificate,
    riem_c_origin_relabel_certificate,
    riem_c_unbalanced_certificate,
    sim1_balanced_rest_certificate,
    sim1_dense_certificate,
    update_dz_c_limiter_ws_certificate,
    update_dz_c_nonuniform_reference_certificate,
    update_dz_c_origin_relabel_certificate,
    update_dz_c_rect_mixed_flags_certificate,
    update_dz_c_uniform_gz_certificate,
    update_dz_c_zero_wind_certificate,
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
    (n=3, ng=2, km=4) domain, with the p_fac floor proven INACTIVE on
    the fixture (halved-p_fac bitwise control + margin inversion through
    the OUTPUT gz, same two-step proof as the SIM1 gate).

    ng=2 leaves a real halo ring outside the write window, and gz/pef
    carry NONZERO halo values, so the passthrough gradients of gz and
    pef are nonzero and actually checked (codex riem r1 #1: with ng=1
    the window covered the whole array and d(loss)/d(pef) was
    identically zero)."""
    n, ng, km = 3, 2, 4
    ptop, hs, w3, pt, delp, gz, ws = _riem_c_fixture(
        5, n=n, ng=ng, km=km, w_amp=0.1, dz_scale=1.02)
    ws = 0.3 * np.ones_like(ws)   # nonzero: exercises the bottom -p1*ws
    bd = _BD(n, ng)
    rng = np.random.default_rng(21)
    pef0 = 100.0 * rng.standard_normal(gz.shape)   # nonzero passthrough
    p_fac = 0.05
    # The write window (is-1..ie+1 square) and its halo ring:
    win = slice(ng - 1, ng + n + 1)
    halo2 = np.ones(gz.shape[:2], dtype=bool)
    halo2[win, win] = False
    assert halo2.any()   # the ring exists (the r1 #1 vacuity is gone)

    gz_a, pef_a = _run_jax_riem_c(100.0, bd, km, FV3_KAPPA, ptop, hs,
                                  w3, pt, delp, gz, pef0, ws, p_fac)
    gz_b, pef_b = _run_jax_riem_c(100.0, bd, km, FV3_KAPPA, ptop, hs,
                                  w3, pt, delp, gz, pef0, ws, p_fac / 2)
    assert np.array_equal(gz_a, gz_b) and np.array_equal(pef_a, pef_b), \
        "p_fac floor ACTIVE on the riem_c gradient fixture"
    # Margin inversion ON THE WRITE WINDOW: the rebuilt gz encodes the
    # SELECTED dz (dz = (gz[k+1]-gz[k])/grav); invert
    # dz = -dm*rgas*pt*P^(kappa-1) for P and require it strictly above
    # the floor with an FD-safe 1% margin (the halved-p_fac control
    # alone can false-pass on a tie).
    dz_sel = (gz_a[win, win, 1:] - gz_a[win, win, :-1]) / FV3_GRAV
    dm = delp[win, win, :] / FV3_GRAV
    pem = np.zeros(gz.shape)
    pem[:, :, 0] = ptop
    for k in range(km):
        pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
    pm2 = delp[win, win, :] / np.log(pem[win, win, 1:]
                                     / pem[win, win, :-1])
    p_sel = np.exp(np.log(-dz_sel / (dm * FV3_RDGAS * pt[win, win, :]))
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

    # Non-vacuity (codex riem r1 #1): the pef and gz passthrough
    # gradients are NONZERO on the halo ring (2*value/scale from the
    # squared loss), so a broken passthrough VJP cannot hide.
    g_pef = jax.grad(lambda pef_: g(jnp.asarray(hs), jnp.asarray(delp),
                                    jnp.asarray(ws), pef_))(
        jnp.asarray(pef0))
    assert np.abs(np.asarray(g_pef)[halo2, :]).max() > 0.0
    g_gz = jax.grad(lambda gz_: f(jnp.asarray(w3), gz_,
                                  jnp.asarray(pt)))(jnp.asarray(gz))
    assert np.abs(np.asarray(g_gz)[halo2, :]).max() > 0.0


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
    # NONZERO in/out operands (codex riem r1 #2: zero seeds made every
    # passthrough gradient identically zero, so the g-group check was
    # vacuous).  With these seeds the pe ring and the ppe/pk3 halos
    # contribute 2*value/scale to the loss gradient.
    fx = dict(zs=zs, w=w, delz=rng.standard_normal((ni, nj, km)),
              pt=pt, delp=delp, zh=zh,
              pe=rng.standard_normal((ni + 2, km + 1, nj + 2)),
              ppe=rng.standard_normal((fi, fj, km + 1)),
              pk3=rng.standard_normal((fi, fj, km + 1)),
              pk=rng.standard_normal((ni, nj, km + 1)),
              peln=rng.standard_normal((ni, km + 1, nj)), ws=ws)
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

    def g(delz_, pe_, ppe_, pk3_, pk_, peln_, flags=kw):
        return _loss(riem3_jax(
            1, 100.0, bounds, km, FV3_KAPPA, 1004.6, ptop,
            jnp.asarray(zs), jnp.asarray(w), delz_, jnp.asarray(pt),
            jnp.asarray(delp), jnp.asarray(zh), pe_, ppe_, pk3_, pk_,
            peln_, jnp.asarray(ws), p_fac, 1.0, **flags))

    g_args = tuple(jnp.asarray(fx[k]) for k in
                   ("delz", "pe", "ppe", "pk3", "pk", "peln"))
    check_grads(g, g_args, order=2, modes=("fwd", "rev"))

    # Non-vacuity (codex riem r1 #2): under the deck flags the pe ring
    # and the ppe/pk3 halos DO reach the loss — their gradients must be
    # nonzero.  delz/pk/peln are fully overwritten under last_call=True,
    # so their true gradient is zero there; the flipped arm below gives
    # pk/peln (and pe wholly) a real identity-passthrough gradient.
    grads = jax.grad(g, argnums=(1, 2, 3))(*g_args)
    for name, ga in zip(("pe", "ppe", "pk3"), grads):
        assert np.abs(np.asarray(ga)).max() > 0.0, (name, "zero grad")

    kw_flip = dict(use_logp=False, last_call=False, fp_out=False)
    check_grads(lambda pe_, pk_, peln_: g(g_args[0], pe_, g_args[2],
                                          g_args[3], pk_, peln_,
                                          flags=kw_flip),
                (g_args[1], g_args[4], g_args[5]), order=2,
                modes=("fwd", "rev"))
    grads_flip = jax.grad(
        lambda pe_, pk_, peln_: g(g_args[0], pe_, g_args[2], g_args[3],
                                  pk_, peln_, flags=kw_flip),
        argnums=(0, 1, 2))(g_args[1], g_args[4], g_args[5])
    for name, ga in zip(("pe", "pk", "peln"), grads_flip):
        assert np.abs(np.asarray(ga)).max() > 0.0, (name, "zero grad")


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


# ======================================================================
# edge_profile — JAX twin certification
# ======================================================================

def _run_jax_edge(q1, q2, j_lo, km, dp0, uniform_grid, limiter,
                  jit=False):
    """Functional driver with the shared edge_profile certificate
    signature."""
    fn = edge_profile_jit if jit else edge_jax
    e1, e2 = fn(jnp.asarray(q1), jnp.asarray(q2), j_lo, km,
                jnp.asarray(dp0), uniform_grid, limiter)
    return np.asarray(e1), np.asarray(e2)


def _edge_fixture(seed=19, ni=NI, km=KM):
    rng = np.random.default_rng(seed)
    q1 = 10.0 + rng.standard_normal((ni, km))
    q2 = -3.0 + rng.standard_normal((ni, km))
    dp0 = np.abs(1.0e4 + 2.0e3 * rng.standard_normal(km))
    return q1, q2, dp0


def _edge_grad_fixture(ni=NI, km=KM):
    """DETERMINISTIC gradient fixture with the limiter switch provably off.

    ``_edge_fixture``'s ``q2 = -3 + N(0,1)`` straddles zero, so the
    ``limiter=1`` clamp (``jnp.where(q * qe < 0)``, C^0 exactly at the
    zero crossing) can sit on its switch -- which makes the finite
    differences inside ``check_grads`` meaningless there.  Here q1 is
    strictly positive and q2 strictly negative, each bounded away from
    zero by >= 9.9 with only a 1 % ripple, and the profile is still
    NON-constant (the ripple varies in both i and k) so the
    interpolation, and therefore the gradient, is nontrivial.

    A same-sign INPUT does NOT by itself guarantee a same-sign edge
    value: ``test_fv3_native_nh_core.py:1614`` records the oracle quirk
    where the uniform branch turns a constant 7.5 into -19.2 at the top
    edge.  So the off-switch property is not argued here -- the call
    site MEASURES all four end products and asserts the margin.  The
    limiter-ACTIVE certificate lives in the ``"limited"`` arm of
    ``test_edge_profile_jax_matches_numpy_lane``, which drives the clamp
    on purpose.
    """
    i = np.arange(ni, dtype=np.float64)[:, None]
    k = np.arange(km, dtype=np.float64)[None, :]
    q1 = 10.0 + 0.1 * np.cos(0.7 * i + 0.3 * k)
    q2 = -10.0 + 0.1 * np.cos(1.1 * i + 0.5 * k + 1.0)
    dp0 = np.linspace(1.0e4, 1.2e4, km)
    return q1, q2, dp0


# ---------------------------------------------------------------- gate 1
# Tolerances PINNED TO MEASUREMENT (probe job 9371449, x64 CPU, these
# fixtures): both lanes run the identical scan/recurrence sequence; XLA
# FMA contraction leaves few-ULP divergence:
#   nonuniform: qe1 rel 1.094e-15, qe2 rel 1.022e-15
#   uniform:    qe1 rel 1.003e-15, qe2 rel 5.809e-16
#   limited:    qe1 rel 7.004e-16, qe2 rel 1.022e-15
# Bound below = measured x (~5) margin.
@pytest.mark.parametrize("case", ["nonuniform", "uniform", "limited"])
def test_edge_profile_jax_matches_numpy_lane(case):
    q1, q2, dp0 = _edge_fixture()
    uniform, limiter = {"nonuniform": (False, 0),
                        "uniform": (True, 0),
                        "limited": (False, 1)}[case]
    if case == "limited":
        # A top cell that yields a sign-opposed top edge (same ladder as
        # the limiter certificate) so the clamp is EXERCISED, not vacuous.
        for q_top in (-0.01, -0.1, -1.0, 0.01, 0.1, 1.0):
            q1t = np.array(q1)
            q1t[:, 0] = q_top
            b1, _ = _run_native_edge_profile(q1t, q2, 0, KM, dp0, False, 0)
            if (q1t[:, 0] * b1[:, 0] < 0.0).any():
                q1 = q1t
                break
        else:
            pytest.fail("no ladder value produced a sign-opposed edge")
    e1_n, e2_n = _run_native_edge_profile(q1, q2, 0, KM, dp0, uniform,
                                          limiter)
    e1_j, e2_j = _run_jax_edge(q1, q2, 0, KM, dp0, uniform, limiter)
    assert _rel(e1_j, e1_n) <= 5e-15, _rel(e1_j, e1_n)
    assert _rel(e2_j, e2_n) <= 5e-15, _rel(e2_j, e2_n)
    if case == "limited":
        # Non-vacuity: at least one edge actually clamped to 0.
        b1, _ = _run_native_edge_profile(q1, q2, 0, KM, dp0, False, 0)
        assert np.any(e1_n[:, 0] != b1[:, 0])


def test_edge_profile_jax_km2_matches_numpy_lane():
    """km=2: the forward scan has ONE row and the bottom reads
    q[:, km-2] == q[:, 0]; parity on a perturbed pair.  Measured (probe
    job 9371449): qe1 rel 2.577e-15, qe2 rel 1.578e-15; bound = ~x4."""
    q1, q2, dp0 = _edge_fixture(seed=23, ni=5, km=2)
    e1_n, e2_n = _run_native_edge_profile(q1, q2, 0, 2, dp0, False, 0)
    e1_j, e2_j = _run_jax_edge(q1, q2, 0, 2, dp0, False, 0)
    assert np.isfinite(e1_n).all()
    assert _rel(e1_j, e1_n) <= 1e-14, _rel(e1_j, e1_n)
    assert _rel(e2_j, e2_n) <= 1e-14, _rel(e2_j, e2_n)


# ---------------------------------------------------------------- gate 2
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_edge_profile_jax_nonuniform_dense_certificate(jit):
    edge_profile_nonuniform_dense_certificate(
        lambda *a: _run_jax_edge(*a, jit=jit))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_edge_profile_jax_uniform_dense_certificate(jit):
    edge_profile_uniform_dense_certificate(
        lambda *a: _run_jax_edge(*a, jit=jit))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_edge_profile_jax_limiter_certificate(jit):
    edge_profile_limiter_certificate(
        lambda *a: _run_jax_edge(*a, jit=jit))


# ---------------------------------------------------------------- gate 3
def test_edge_profile_jax_jit_eager_parity_and_no_retrace():
    q1, q2, dp0 = _edge_fixture()
    eager = _run_jax_edge(q1, q2, 0, KM, dp0, False, 0)

    traces = {"n": 0}

    def _counted(*a, **kw):
        traces["n"] += 1
        return edge_jax(*a, **kw)

    fn = make_edge_profile_jit(_counted)   # the PRODUCTION jit policy

    def _call(q1_, q2_):
        return fn(jnp.asarray(q1_), jnp.asarray(q2_), 0, KM,
                  jnp.asarray(dp0), False, 0)

    jit1 = _call(q1, q2)
    jit2 = _call(q1 * 1.01, q2 * 0.99)   # same shapes -> no retrace
    assert traces["n"] == 1, traces["n"]
    del jit2

    # Bounds PINNED TO MEASUREMENT (probe job 9371848, x64 CPU, this
    # fixture): qe1 1.036e-15, qe2 1.022e-15.  Both sit just ABOVE the
    # 1e-15 inherited from the SIM1 gate, so the old expectation was the
    # wrong one; bound = measured x ~10.  NB qe2 was never REPORTED as a
    # failure only because the loop asserts qe1 first.
    #
    # A raised budget could in principle hide a jit-ONLY regression, so
    # that is measured, not argued (job 9371947): against the NumPy fp64
    # lane -- the certification authority -- jit is no further away than
    # eager, to every digit printed:
    #   qe1  eager-vs-numpy 1.093817e-15   jit-vs-numpy 1.093817e-15
    #   qe2  eager-vs-numpy 1.022090e-15   jit-vs-numpy 1.022090e-15
    # A jit-only numerical regression would push jit-vs-numpy above
    # eager-vs-numpy; it does not move at all.  The divergence is
    # therefore reassociation-class.  Attributing it specifically to XLA
    # FMA contraction remains PLAUSIBLE, not isolated -- that would need
    # a non-contracted lowering (HLO/LLVM inspection) to confirm.
    for name, e, j in zip(("qe1", "qe2"), eager, jit1):
        r = _rel(np.asarray(j), e)
        assert r <= 1.1e-14, (name, r)


# ---------------------------------------------------------------- gate 4
def test_edge_profile_jax_check_grads_order2():
    """Order-2 fwd+rev grads over all dynamic operands (q1, q2, dp0) on
    both grid branches, plus the limiter=1 arm on a fixture PROVEN off
    the zero-crossing switch (the jnp.where is C^0 exactly at
    q*qe == 0; margin > 1 in product units here, far beyond any FD
    step)."""
    ni, km = 3, 4
    q1, q2, dp0 = _edge_grad_fixture(ni=ni, km=km)

    def f(q1_, q2_, dp0_):
        e1, e2 = edge_jax(q1_, q2_, 0, km, dp0_, False, 0)
        return jnp.sum(e1 * e1) + jnp.sum(e2 * e2)

    check_grads(f, (jnp.asarray(q1), jnp.asarray(q2), jnp.asarray(dp0)),
                order=2, modes=("fwd", "rev"))

    # Uniform branch (dp0 is a dead read there -- q1/q2 only).
    def fu(q1_, q2_):
        e1, e2 = edge_jax(q1_, q2_, 0, km, jnp.asarray(dp0), True, 0)
        return jnp.sum(e1 * e1) + jnp.sum(e2 * e2)

    check_grads(fu, (jnp.asarray(q1), jnp.asarray(q2)), order=2,
                modes=("fwd", "rev"))

    # limiter=1: prove the clamp INACTIVE and off-switch on this fixture
    # (all four q*qe products strictly positive with margin > 1).
    e1, e2 = _run_jax_edge(q1, q2, 0, km, dp0, False, 0)
    prods = (q1[:, 0] * e1[:, 0], q2[:, 0] * e2[:, 0],
             q1[:, km - 1] * e1[:, km], q2[:, km - 1] * e2[:, km])
    for p in prods:
        assert (p > 1.0).all(), "limiter switch margin < 1 on fixture"
    # Record the worst-case margin so a future fixture edit that quietly
    # walks the products toward the switch fails here, not silently.
    assert min(float(p.min()) for p in prods) > 50.0

    def fl(q1_, q2_, dp0_):
        e1_, e2_ = edge_jax(q1_, q2_, 0, km, dp0_, False, 1)
        return jnp.sum(e1_ * e1_) + jnp.sum(e2_ * e2_)

    check_grads(fl, (jnp.asarray(q1), jnp.asarray(q2),
                     jnp.asarray(dp0)), order=2, modes=("fwd", "rev"))


# ------------------------------------------------------------ guards
@pytest.mark.parametrize("bad", ["q1", "q2", "dp0"])
def test_edge_profile_jax_rejects_float32_every_operand(bad):
    q1, q2, dp0 = _edge_fixture()
    args = {"q1": q1, "q2": q2, "dp0": dp0}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        edge_jax(jnp.asarray(args["q1"]), jnp.asarray(args["q2"]), 0,
                 KM, jnp.asarray(args["dp0"]), False, 0)


def test_edge_profile_jax_km1_raises():
    """km=1 rejected loudly (the NumPy lane raises IndexError on
    q[:, 1]; the bottom row would silently WRAP on q[:, km-2])."""
    a = jnp.ones((3, 1), jnp.float64)
    with pytest.raises(ValueError, match="km=1"):
        edge_jax(a, a, 0, 1, jnp.ones(1, jnp.float64), False, 0)


# ======================================================================
# update_dz_c — JAX twin certification
# ======================================================================

def _run_jax_udzc(bd, km, dt, dp0, zs, area, ut, vt, gz, ws, npx, npy,
                  jit=False, **flags):
    """Functional driver with the shared update_dz_c certificate
    signature (bd shim -> static bounds tuple)."""
    fn = update_dz_c_jit if jit else udzc_jax
    gz_o, ws_o = fn(_bounds(bd), km, dt, jnp.asarray(dp0),
                    jnp.asarray(zs), jnp.asarray(area), jnp.asarray(ut),
                    jnp.asarray(vt), jnp.asarray(gz), jnp.asarray(ws),
                    npx, npy, **flags)
    return np.asarray(gz_o), np.asarray(ws_o)


# ---------------------------------------------------------------- gate 1
def test_udzc_jax_matches_numpy_lane_rect_mixed_flags():
    """Strongest equivalence fixture: rectangular domain, nonuniform
    dp0, signed winds, MIXED corner flags (exercises top/bottom/interior
    ratios, both upwind branches, the corner fill and the ws diagnosis).
    Measured (probe job 9371449, x64 CPU): gz rel 0.0, ws absdiff 0.0 —
    BITWISE on this backend; the tolerant bound stays for backends whose
    FMA contraction differs."""
    ni, nj, ng = 12, 9, 3
    npx, npy = ni + 1, nj + 1
    bd = _BDR(1, ni, 1, nj, ng)
    ut, vt, area, gz, zs, dp0 = _rect_fields(ni, nj, ng, KM, 71)
    fi, fj = ni + 2 * ng, nj + 2 * ng
    flags = dict(sw_corner=True, se_corner=False, ne_corner=True,
                 nw_corner=False)
    args = (bd, KM, 100.0, dp0, zs, area, ut, vt, gz,
            np.zeros((fi, fj)), npx, npy)
    gz_n, ws_n = _run_native_update_dz_c(*args, **flags)
    gz_j, ws_j = _run_jax_udzc(*args, **flags)
    assert _rel(gz_j, gz_n) <= 5e-15, _rel(gz_j, gz_n)
    assert np.abs(ws_j - ws_n).max() <= 5e-15 * max(
        np.abs(ws_n).max(), 1.0), np.abs(ws_j - ws_n).max()
    # Non-vacuity: transport moved gz.
    assert np.abs(gz_n - gz).max() > 1.0
    # Halo passthrough is bitwise (both lanes carry the input).
    halo = np.ones((fi, fj), dtype=bool)
    halo[ng - 1:ng + ni + 1, ng - 1:ng + nj + 1] = False
    assert np.array_equal(gz_j[halo, :], gz[halo, :])


def test_udzc_jax_km2_matches_numpy_lane():
    """km=2: the interior interpolation has ONE level and bot_ratio
    reads dp0[0]; parity on the rect fixture.  Measured (probe job
    9371449): gz rel 0.0, ws absdiff 0.0 (bitwise on x64 CPU)."""
    ni, nj, ng, km = 6, 5, 3, 2
    npx, npy = ni + 1, nj + 1
    bd = _BDR(1, ni, 1, nj, ng)
    ut, vt, area, gz, zs, dp0 = _rect_fields(ni, nj, ng, km, 77)
    fi, fj = ni + 2 * ng, nj + 2 * ng
    flags = dict(sw_corner=True, se_corner=True, ne_corner=True,
                 nw_corner=True)
    args = (bd, km, 100.0, dp0, zs, area, ut, vt, gz,
            np.zeros((fi, fj)), npx, npy)
    gz_n, ws_n = _run_native_update_dz_c(*args, **flags)
    gz_j, ws_j = _run_jax_udzc(*args, **flags)
    assert np.isfinite(gz_n).all()
    assert _rel(gz_j, gz_n) <= 5e-15, _rel(gz_j, gz_n)
    assert np.abs(ws_j - ws_n).max() <= 5e-15 * max(
        np.abs(ws_n).max(), 1.0)


# ---------------------------------------------------------------- gate 2
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_udzc_jax_zero_wind_certificate(jit):
    update_dz_c_zero_wind_certificate(
        lambda *a, **kw: _run_jax_udzc(*a, jit=jit, **kw))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_udzc_jax_uniform_gz_certificate(jit):
    update_dz_c_uniform_gz_certificate(
        lambda *a, **kw: _run_jax_udzc(*a, jit=jit, **kw))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_udzc_jax_limiter_ws_certificate(jit):
    update_dz_c_limiter_ws_certificate(
        lambda *a, **kw: _run_jax_udzc(*a, jit=jit, **kw))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_udzc_jax_nonuniform_reference_certificate(jit):
    update_dz_c_nonuniform_reference_certificate(
        lambda *a, **kw: _run_jax_udzc(*a, jit=jit, **kw))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_udzc_jax_origin_relabel_certificate(jit):
    update_dz_c_origin_relabel_certificate(
        lambda *a, **kw: _run_jax_udzc(*a, jit=jit, **kw))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_udzc_jax_rect_mixed_flags_certificate(jit):
    # Measured (probe job 9371449): the JAX lane is BITWISE equal to the
    # native lane on this fixture (x64 CPU), so it inherits the native
    # lane's bitwise match to the independent corner reference; the
    # tolerant bound stays for FMA-differing backends.
    update_dz_c_rect_mixed_flags_certificate(
        lambda *a, **kw: _run_jax_udzc(*a, jit=jit, **kw), tol=5e-15)


# ---------------------------------------------------------------- gate 3
def test_udzc_jax_jit_eager_parity_and_no_retrace():
    ni, nj, ng = 12, 9, 3
    npx, npy = ni + 1, nj + 1
    bd = _BDR(1, ni, 1, nj, ng)
    ut, vt, area, gz, zs, dp0 = _rect_fields(ni, nj, ng, KM, 71)
    fi, fj = ni + 2 * ng, nj + 2 * ng
    ws0 = np.zeros((fi, fj))
    flags = dict(sw_corner=True, se_corner=False, ne_corner=True,
                 nw_corner=False)
    eager = _run_jax_udzc(bd, KM, 100.0, dp0, zs, area, ut, vt, gz, ws0,
                          npx, npy, **flags)

    traces = {"n": 0}

    def _counted(*a, **kw):
        traces["n"] += 1
        return udzc_jax(*a, **kw)

    fn = make_update_dz_c_jit(_counted)   # the PRODUCTION jit policy

    def _call(ut_, gz_):
        return fn(_bounds(bd), KM, 100.0, jnp.asarray(dp0),
                  jnp.asarray(zs), jnp.asarray(area), jnp.asarray(ut_),
                  jnp.asarray(vt), jnp.asarray(gz_), jnp.asarray(ws0),
                  npx, npy, **flags)

    jit1 = _call(ut, gz)
    jit2 = _call(ut * 1.01, gz * 0.999)   # same shapes -> no retrace
    assert traces["n"] == 1, traces["n"]
    del jit2

    # Bounds PINNED TO MEASUREMENT (probe job 9371848, x64 CPU, this
    # fixture), PER FIELD because the two normalisations differ by four
    # orders of magnitude:
    #   gz 3.241e-16  (|gz|max 1.12e4 -- a large, well-scaled field)
    #   ws 1.852e-15  (|ws|max 3.69   -- diagnosed from differences of
    #                  gz-scale quantities, so cancellation-amplified
    #                  relative to its own max)
    # Bound = measured x ~10, kept separate so a gz regression cannot
    # hide behind ws's looser budget.  The same jit-only-regression
    # check as the edge_profile gate applies (job 9371947): against the
    # NumPy lane, jit sits no further away than eager, so the raised
    # budget is not covering a jit defect.  The FMA-contraction
    # attribution is PLAUSIBLE, not isolated.
    bounds = {"gz": 3.3e-15, "ws": 1.9e-14}
    for name, e, j in zip(("gz", "ws"), eager, jit1):
        r = np.abs(np.asarray(j) - e).max() / max(np.abs(e).max(), 1e-30)
        assert r <= bounds[name], (name, r)


# ---------------------------------------------------------------- gate 4
def test_udzc_jax_check_grads_order2_away_from_switches():
    """Order-2 fwd+rev grads over all seven dynamic operands on a small
    (n=4, ng=2, km=4) FD-friendly fixture, with BOTH C^0 switches proven
    off with margin:

    * the upwind flux ``jnp.where(xw > 0)`` — every advective interface
      wind is bounded away from 0 (asserted below on an independent
      3-line rebuild of the level interpolation);
    * the ``dz_min`` bottom-up ``jnp.maximum`` floor — every level of
      the OUTPUT clears the floor strictly (a fired floor would sit at
      exactly gz[k+1] + DZ_MIN).
    """
    n, ng, km = 4, 2, 4
    full = n + 2 * ng
    npx = npy = n + 1
    bd = _BD(n, ng)
    bounds = _bounds(bd)
    flags = dict(sw_corner=True, se_corner=True, ne_corner=True,
                 nw_corner=True)
    dt = 100.0

    # FD-friendly magnitudes (area ~5e2, winds ~1e1, gz ~1e3): the
    # certificate fixtures' 1e10 winds would make check_grads' fixed-eps
    # numerical derivative meaningless.
    #
    # The winds are built DETERMINISTICALLY, not drawn until a seed
    # happens to work: sign is a function of (i, j) ONLY (constant down
    # the column), so every k-interpolation averages same-signed
    # neighbours and cannot land on the upwind switch.  The per-column
    # sinusoidal jitter puts |w| in ~[7.6, 12.6] (NOT [8, 12] -- the
    # +-5 % multiplier widens both ends), so the guarantee taken from
    # the algebra alone is only |mid| >= 7.6 and |top|, |bot| > 0; the
    # numbers that matter are MEASURED and asserted below.  On this dp0
    # (seed 55) they are: top 6.66, bottom 12.13, interior 8.54.
    # The (i + j) checkerboard, with v in antiphase to u, makes BOTH
    # upwind branches fire (asserted below -- a one-sided fixture would
    # leave half the jnp.where untested).  Fixing the sign down each
    # column excludes only switch-CROSSING inputs; both upwind arms
    # still fire at the top, interior and bottom interfaces.
    rng = np.random.default_rng(55)
    dp0 = np.abs(1.0e4 + 2.0e3 * rng.standard_normal(km))
    ii, jj = np.meshgrid(np.arange(full), np.arange(full), indexing="ij")
    sgn_u = np.where((ii + jj) % 2 == 0, 1.0, -1.0)[:, :, None]
    mag = ((8.0 + 4.0 * (np.arange(km) / max(km - 1, 1)))[None, None, :]
           * (1.0 + 0.05 * np.sin(ii + 2.0 * jj))[:, :, None])
    ut = sgn_u * mag
    vt = -sgn_u * mag

    # Off-switch control 1: rebuild the interface winds (the same 3-line
    # dp0 interpolation, nh_utils.F90:94-133) and demand a margin far
    # beyond any FD step at |wind| ~ 1e1, plus both-branches coverage.
    tr = dp0[0] / (dp0[0] + dp0[1])
    br = dp0[km - 1] / (dp0[km - 2] + dp0[km - 1])
    assert 0.0 < tr < 1.0 and 0.0 < br < 1.0
    for w in (ut, vt):
        top = w[:, :, 0] + (w[:, :, 0] - w[:, :, 1]) * tr
        bot = (w[:, :, km - 1]
               + (w[:, :, km - 1] - w[:, :, km - 2]) * br)
        mid = ((dp0[1:] * w[:, :, :-1] + dp0[:-1] * w[:, :, 1:])
               / (dp0[:-1] + dp0[1:]))
        assert min(np.abs(top).min(), np.abs(bot).min(),
                   np.abs(mid).min()) > 4.0
        n_pos = int((top > 0).sum() + (bot > 0).sum() + (mid > 0).sum())
        n_neg = int((top < 0).sum() + (bot < 0).sum() + (mid < 0).sum())
        assert n_pos > 0 and n_neg > 0, (n_pos, n_neg)

    area = np.abs(5.0e2 * (1.0 + 0.1 * rng.standard_normal((full,
                                                            full))))
    gz = np.cumsum(
        np.abs(500.0 + 100.0 * rng.standard_normal(
            (full, full, km + 1))), axis=2)[:, :, ::-1].copy() * 3.0
    zs = np.array(gz[:, :, km], copy=True)
    ws0 = rng.standard_normal((full, full))   # nonzero halo passthrough

    gz_o, ws_o = _run_jax_udzc(bd, km, dt, dp0, zs, area, ut, vt, gz,
                               ws0, npx, npy, **flags)
    # Off-switch control 2: the dz_min floor never fired (output strictly
    # above gz[k+1] + DZ_MIN everywhere in the write window).
    w = slice(ng - 1, ng + n + 1)
    gap = (gz_o[w, w, :-1] - (gz_o[w, w, 1:] + DZ_MIN))
    assert gap.min() > 1.0, f"dz_min floor margin {gap.min()} too small"

    def _loss(gz_out, ws_out):
        return jnp.sum(gz_out * gz_out) / 1e6 + jnp.sum(ws_out * ws_out)

    def f(dp0_, ut_, vt_, gz_):
        return _loss(*udzc_jax(
            bounds, km, dt, dp0_, jnp.asarray(zs), jnp.asarray(area),
            ut_, vt_, gz_, jnp.asarray(ws0), npx, npy, **flags))

    check_grads(f, (jnp.asarray(dp0), jnp.asarray(ut), jnp.asarray(vt),
                    jnp.asarray(gz)), order=2, modes=("fwd", "rev"))

    def g(zs_, area_, ws_):
        return _loss(*udzc_jax(
            bounds, km, dt, jnp.asarray(dp0), zs_, area_,
            jnp.asarray(ut), jnp.asarray(vt), jnp.asarray(gz), ws_,
            npx, npy, **flags))

    check_grads(g, (jnp.asarray(zs), jnp.asarray(area),
                    jnp.asarray(ws0)), order=2, modes=("fwd", "rev"))

    # Non-vacuity, and the LIMIT of what the ws operand certifies: the
    # ws INPUT only survives where update_dz_c does not overwrite it, so
    # this asserts a HALO PASSTHROUGH gradient (the write window covers
    # is-1..ie+1; with ng=2 a one-cell ring remains) -- nothing more.
    # It does NOT certify the ws DIAGNOSIS; that is exercised through
    # the gz, wind, area, zs and dp0 operands above, which do flow
    # through the computed ws in the loss.
    halo = np.ones((full, full), dtype=bool)
    halo[w, w] = False
    assert halo.any()
    g_ws = jax.grad(lambda ws_: g(jnp.asarray(zs), jnp.asarray(area),
                                  ws_))(jnp.asarray(ws0))
    assert np.abs(np.asarray(g_ws)[halo]).max() > 0.0


# ------------------------------------------------------------ guards
@pytest.mark.parametrize("bad", ["dp0", "zs", "area", "ut", "vt", "gz",
                                 "ws"])
def test_udzc_jax_rejects_float32_every_operand(bad):
    n, ng = 4, 2
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
        udzc_jax((1, n, 1, n, ng), KM, 100.0,
                 jnp.asarray(args["dp0"]), jnp.asarray(args["zs"]),
                 jnp.asarray(args["area"]), jnp.asarray(args["ut"]),
                 jnp.asarray(args["vt"]), jnp.asarray(args["gz"]),
                 jnp.asarray(args["ws"]), n + 1, n + 1,
                 sw_corner=True, se_corner=True, ne_corner=True,
                 nw_corner=True)


def test_udzc_jax_km1_and_ng1_raise():
    """km=1 (the NumPy lane raises IndexError on dp0[1]) and ng=1 (the
    NumPy lane's fort views silently WRAP on the is-2 upwind read) are
    both loud errors here."""
    n = 4
    for km, ng, match in ((1, 2, "km=1"), (2, 1, "ng=1")):
        full = n + 2 * ng
        a2 = jnp.zeros((full, full), jnp.float64)
        a3 = jnp.zeros((full, full, km), jnp.float64)
        a3p = jnp.zeros((full, full, km + 1), jnp.float64)
        with pytest.raises(ValueError, match=match):
            udzc_jax((1, n, 1, n, ng), km, 100.0,
                     jnp.full((km,), 1.0e4, jnp.float64), a2,
                     jnp.full((full, full), 5.0e8, jnp.float64), a3, a3,
                     a3p, a2, n + 1, n + 1, sw_corner=True,
                     se_corner=True, ne_corner=True, nw_corner=True)


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
