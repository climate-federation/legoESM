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
from legoesm.core.fv3_native_nh_core import DZ_MIN  # noqa: E402
from legoesm.core.fv3_nh_core import (  # noqa: E402
    edge_profile as edge_jax,
)
from legoesm.core.fv3_nh_core import (
    edge_profile_jit,
    make_edge_profile_jit,
    make_riem_solver3_jit,
    make_riem_solver_c_jit,
    make_sim1_solver_jit,
    make_update_dz_c_jit,
    make_update_dz_d_jit,
    riem_solver3_jit,
    riem_solver_c_jit,
    sim1_solver_jit,
    update_dz_c_jit,
    update_dz_d_jit,
)
from legoesm.core.fv3_nh_core import (
    riem_solver3 as riem3_jax,
)
from legoesm.core.fv3_nh_core import (
    riem_solver_c as riem_c_jax,
)
from legoesm.core.fv3_nh_core import (
    sim1_solver as sim1_jax,
)
from legoesm.core.fv3_nh_core import (
    update_dz_c as udzc_jax,
)
from legoesm.core.fv3_nh_core import (
    update_dz_d as udzd_jax,
)
from legoesm.grids.fv3_native_gridstruct import (  # noqa: E402
    FV3_GRAV,
    FV3_KAPPA,
    FV3_RDGAS,
)

from tests.grids.test_fv3_native_nh_core import (  # noqa: E402
    _BD,
    _BDR,
    KM,
    NI,
    _balanced_column,
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
    # A raised jit-vs-eager budget could hide a jit-ONLY regression, so
    # the jit lane is ALSO bound directly against the NumPy fp64 lane --
    # the certification authority -- at the same tolerance the eager
    # lane is held to.  Measured (job 9371947): eager-vs-numpy and
    # jit-vs-numpy are 1.093817e-15 (qe1) and 1.022090e-15 (qe2), equal
    # to every printed digit.
    #
    # What that establishes, precisely: under this max-norm on this
    # fixture, jit is no further from the authority than eager.  It does
    # NOT exclude a jit change that happens to cancel against eager's own
    # lane error, a change confined to non-maximal elements, or a
    # common-mode eager+jit regression -- and it does not identify the
    # mechanism.  XLA FMA contraction remains a PLAUSIBLE explanation,
    # not an isolated one; that needs a non-contracted lowering
    # (HLO/LLVM inspection) to confirm.
    native = _run_native_edge_profile(q1, q2, 0, KM, dp0, False, 0)
    for name, e, j, n in zip(("qe1", "qe2"), eager, jit1, native):
        r = _rel(np.asarray(j), e)
        assert r <= 1.1e-14, (name, r)
        r_en, r_jn = _rel(np.asarray(e), n), _rel(np.asarray(j), n)
        assert r_jn <= 1.1e-14, (name, "jit-vs-numpy", r_jn)
        # The jit lane may not drift away from the authority relative to
        # eager by more than the eager-jit budget itself.
        assert r_jn <= r_en + 1.1e-14, (name, r_jn, r_en)


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
    # hide behind ws's looser budget.  As in the edge_profile gate, the
    # jit lane is ALSO bound directly against the NumPy fp64 lane so a
    # jit-only regression cannot hide inside the raised jit-vs-eager
    # budget; see that gate for exactly what this does and does not
    # establish (it does not exclude cancellation against eager's own
    # lane error, non-maximal-element changes, or a common-mode
    # regression, and it identifies no mechanism).
    bounds = {"gz": 3.3e-15, "ws": 1.9e-14}
    native = _run_native_update_dz_c(bd, KM, 100.0, dp0, zs, area, ut,
                                     vt, gz, ws0, npx, npy, **flags)
    for name, e, j, n in zip(("gz", "ws"), eager, jit1, native):
        scale = max(np.abs(e).max(), 1e-30)
        r = np.abs(np.asarray(j) - e).max() / scale
        assert r <= bounds[name], (name, r)
        n_scale = max(np.abs(n).max(), 1e-30)
        r_en = np.abs(np.asarray(e) - n).max() / n_scale
        r_jn = np.abs(np.asarray(j) - n).max() / n_scale
        assert r_jn <= bounds[name], (name, "jit-vs-numpy", r_jn)
        assert r_jn <= r_en + bounds[name], (name, r_jn, r_en)


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


# ======================================================================
# update_dz_d — JAX twin certification
#
# Four gates, same shape as the update_dz_c block above:
#   1. equivalence vs ``fv3_native_nh_core.update_dz_d`` on the SAME
#      fixture the NumPy lane's own reference test uses (n=12, ng=3,
#      seed 83, hord=6, mixed damp/ndif), compared over the FULL array
#      including ghost cells;
#   2. jit-vs-eager parity + no-retrace-across-calls (trace counter),
#      asserted, not commented;
#   3. guards: every float64 operand rejects float32, the transport
#      order selector rejects an unported value, and the km/ng/shape/
#      ndif preconditions raise;
#   4. ``check_grads(order=2)`` at hord=2 (the perfectly-linear PPM arm),
#      with both remaining C^0 sites proven off with margin.
# ======================================================================

UDZD_HORD = 6            # DUO_TAIL_CFG['hord_tm'] on the shipped deck
UDZD_RDT = 1.0 / 100.0

# PRE-MEASUREMENT NOTE, scoped exactly.  Every numeric bound below is
# marked TOL-PENDING and carries a provisional 1e-12; none has been
# measured under JAX.  What HAS been run (login-node-legal, no JAX) is a
# NumPy-backed shim of jnp/lax over the SAME module source: on the gate-1
# fixture it reproduced the NumPy lane EXACTLY (max|dzh| = 0.0 over the
# full array INCLUDING ghost cells, ws 0.0) for hord 1-13 and for
# nord = 1 and 2.  That is evidence the index algebra and the operator
# sequence are right; it is NOT a JAX measurement and says nothing about
# jit lowering, dtype promotion, lax.scan association or gradients.
# The orchestrator's measurement job replaces every bound below.


def _udzd_fixture(n=12, ng=3, km=KM, seed=83):
    """The NumPy lane's nonuniform update_dz_d fixture, rebuilt with the
    SAME n/ng/km, the same seed and the same draw ORDER as
    ``test_fv3_native_nh_core.test_update_dz_d_nonuniform_vs_replumbed_reference``
    (:1406-1455), so this gate inherits that fixture's demonstrated
    non-vacuity (it moves zh by >1 m and exercises both the damped and
    the undamped branch).  Both lanes are handed the SAME arrays, so the
    seed only fixes what is exercised, never what is compared.

    ``area``/``rarea`` are overridden with SENTINEL-FREE values: the
    single-tile gridstruct leaves BIG_NUMBER in the corner-diagonal halo
    cells and ``fv_tp_2d``'s y-intermediates read them.  That is the
    NumPy lane's documented precondition (its probe job 9355001), not a
    JAX-lane concern, and it is restated in ``update_dz_d``'s docstring.
    """
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        build_fv3_native_gridstruct,
    )

    full = n + 2 * ng
    bd = Bounds.single_tile(n, ng)
    gs = dict(build_fv3_native_gridstruct(n, ng, tile=1,
                                          radius=FV3_RADIUS_M,
                                          omega=FV3_OMEGA))
    gs.update(bounded_domain=False, grid_type=0, sw_corner=True,
              se_corner=True, nw_corner=True, ne_corner=True)
    rng = np.random.default_rng(seed)
    area = np.abs(4.0e11 * (1.0 + 0.05 * rng.standard_normal((full, full))))
    rarea = 1.0 / area
    gs["area"] = area
    gs["rarea"] = rarea

    crx = 0.2 * rng.standard_normal((n + 1, full, km))
    xfx = 1.0e10 * rng.standard_normal((n + 1, full, km))
    cry = 0.2 * rng.standard_normal((full, n + 1, km))
    yfx = 1.0e10 * rng.standard_normal((full, n + 1, km))

    levels = np.array([(km - k) * 3000.0 + 5000.0 for k in range(km + 1)])
    zh0 = (np.broadcast_to(levels, (full, full, km + 1)).copy()
           + 150.0 * rng.standard_normal((full, full, km + 1)))
    zs = np.array(zh0[:, :, km], copy=True)
    return {"bd": bd, "gs": gs, "n": n, "ng": ng, "km": km, "full": full,
            "area": area, "rarea": rarea, "crx": crx, "cry": cry,
            "xfx": xfx, "yfx": yfx, "zh0": zh0, "zs": zs,
            "dp0": np.full(km, 1.0e4)}


def _run_np_udzd(fxt, ndif, damp, zh, ws, *, hord=UDZD_HORD,
                 rdt=UDZD_RDT, lim_fac=1.0):
    """NumPy lane driver: copies every mutated operand so the caller's
    arrays are never disturbed between the two lanes."""
    from legoesm.core.fv3_native_nh_core import update_dz_d as udzd_np

    zh_w = np.array(zh, dtype=np.float64, copy=True)
    ws_w = np.array(ws, dtype=np.float64, copy=True)
    ndif_w = np.array(ndif, dtype=np.float64, copy=True)
    damp_w = np.array(damp, dtype=np.float64, copy=True)
    udzd_np(ndif_w, damp_w, hord, fxt["bd"], fxt["km"], fxt["n"] + 1,
            fxt["n"] + 1, fxt["area"], fxt["rarea"], fxt["dp0"],
            fxt["zs"], zh_w, fxt["crx"], fxt["cry"], fxt["xfx"],
            fxt["yfx"], ws_w, rdt, dict(fxt["gs"]), lim_fac=lim_fac)
    return zh_w, ws_w, damp_w, ndif_w


def _udzd_jax_args(fxt, ndif, damp, zh, ws, *, hord=UDZD_HORD,
                   rdt=UDZD_RDT):
    """Positional argument tuple for the JAX twin (static leaders as
    hashable tuples so the jit cache keys by value)."""
    gs = fxt["gs"]
    return (tuple(int(x) for x in ndif), tuple(float(x) for x in damp),
            hord, _bounds(fxt["bd"]), fxt["km"], fxt["n"] + 1,
            fxt["n"] + 1,
            jnp.asarray(fxt["area"]), jnp.asarray(fxt["rarea"]),
            jnp.asarray(fxt["dp0"]), jnp.asarray(fxt["zs"]),
            jnp.asarray(zh), jnp.asarray(fxt["crx"]),
            jnp.asarray(fxt["cry"]), jnp.asarray(fxt["xfx"]),
            jnp.asarray(fxt["yfx"]), jnp.asarray(ws), rdt,
            jnp.asarray(gs["dxa"]), jnp.asarray(gs["dya"]),
            jnp.asarray(gs["del6_u"]), jnp.asarray(gs["del6_v"]))


UDZD_FLAGS = dict(lim_fac=1.0, bounded_domain=False, grid_type=0,
                  sw_corner=True, se_corner=True, nw_corner=True,
                  ne_corner=True, duogrid=False)


def _run_jax_udzd(fxt, ndif, damp, zh, ws, *, jit=False, hord=UDZD_HORD,
                  rdt=UDZD_RDT, fn=None):
    if fn is None:
        fn = update_dz_d_jit if jit else udzd_jax
    zh_o, ws_o = fn(*_udzd_jax_args(fxt, ndif, damp, zh, ws, hord=hord,
                                    rdt=rdt), **UDZD_FLAGS)
    return np.asarray(zh_o), np.asarray(ws_o)


def _corner_masks(full, ng):
    """(corner-block mask, non-corner-halo mask) in STORAGE indices.

    ``copy_corners`` writes exactly the four ng x ng blocks at the array
    corners (Fortran i, j in [1-ng, 0] and [npx, npx+ng-1], which map to
    storage [0, ng-1] and [full-ng, full-1]); everything else outside
    the compute window must be carried through untouched."""
    corner = np.zeros((full, full), dtype=bool)
    for si in (slice(0, ng), slice(full - ng, full)):
        for sj in (slice(0, ng), slice(full - ng, full)):
            corner[si, sj] = True
    halo = np.ones((full, full), dtype=bool)
    halo[ng:full - ng, ng:full - ng] = False
    return corner, halo & ~corner


# ---------------------------------------------------------------- gate 1
def test_udzd_jax_matches_numpy_lane():
    """Equivalence on the NumPy lane's own nonuniform fixture: hord=6
    (the deck's hord_tm), damp alternating so BOTH the del-nord branch
    and the plain-transport branch run, ndif alternating 1/0 so both a
    del-4 and a del-2 del6 chain run, and the km+1 slot left at 999 so
    the :231-232 copy is exercised."""
    fxt = _udzd_fixture()
    n, ng, km, full = fxt["n"], fxt["ng"], fxt["km"], fxt["full"]
    damp0 = [1.0e6, 0.0, 1.0e6, 0.0, 1.0e6, 999.0]
    ndif0 = [1, 0, 1, 0, 1, 999]
    assert len(damp0) == km + 1 and len(ndif0) == km + 1
    ws0 = np.zeros((n, n))

    zh_n, ws_n, damp_m, ndif_m = _run_np_udzd(fxt, ndif0, damp0,
                                              fxt["zh0"], ws0)
    zh_j, ws_j = _run_jax_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0)

    # Instrument check BEFORE the comparison: a NaN on both sides would
    # make array_equal-style agreement meaningless and _rel produce nan.
    assert np.isfinite(zh_n).all() and np.isfinite(ws_n).all()
    assert np.isfinite(zh_j).all() and np.isfinite(ws_j).all()
    # The NumPy lane's :231-232 mutation actually happened.
    assert damp_m[km] == damp0[km - 1] and ndif_m[km] == ndif0[km - 1]

    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert _rel(zh_j, zh_n) <= 1e-12, _rel(zh_j, zh_n)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert np.abs(ws_j - ws_n).max() <= 1e-12 * max(
        np.abs(ws_n).max(), 1.0), np.abs(ws_j - ws_n).max()

    # No-op killers: the transport moved the interior, and ws is nonzero.
    sl = slice(ng, ng + n)
    assert np.abs(zh_n[sl, sl, :] - fxt["zh0"][sl, sl, :]).max() > 1.0
    assert np.abs(ws_n).max() > 0.0

    # Ghost-cell contract, per level (this is what makes the full-array
    # comparison above non-trivial):
    #   damped  -> fv_tp_2d works on a COPY, ghosts untouched;
    #   undamped-> fv_tp_2d ALIASES zh, so copy_corners' four corner
    #              blocks are rewritten and MUST have changed.
    corner, halo_rest = _corner_masks(full, ng)
    for k in range(km + 1):
        damped = damp0[k if k < km else km - 1] > 1.0e-5
        assert np.array_equal(zh_j[halo_rest, k], fxt["zh0"][halo_rest, k]), k
        if damped:
            assert np.array_equal(zh_j[corner, k],
                                  fxt["zh0"][corner, k]), k
        else:
            assert not np.array_equal(zh_j[corner, k],
                                      fxt["zh0"][corner, k]), k


def test_udzd_jax_matches_numpy_lane_all_damped_nord2():
    """Second point in the branch space: EVERY level damped with
    nord = 2 (the deck's nord_v), which is the only setting that runs
    del6_vt_flux's two-iteration high-order chain and its four extra
    copy_corners calls.  Gate 1's alternating fixture never reaches
    nord = 2."""
    fxt = _udzd_fixture(seed=97)
    n, km = fxt["n"], fxt["km"]
    damp0 = [0.12] * (km + 1)
    ndif0 = [2] * (km + 1)
    ws0 = np.zeros((n, n))
    zh_n, ws_n, _, _ = _run_np_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0)
    zh_j, ws_j = _run_jax_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0)
    assert np.isfinite(zh_n).all() and np.isfinite(zh_j).all()
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert _rel(zh_j, zh_n) <= 1e-12, _rel(zh_j, zh_n)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert np.abs(ws_j - ws_n).max() <= 1e-12 * max(
        np.abs(ws_n).max(), 1.0), np.abs(ws_j - ws_n).max()
    sl = slice(fxt["ng"], fxt["ng"] + n)
    assert np.abs(zh_n[sl, sl, :] - fxt["zh0"][sl, sl, :]).max() > 1.0


@pytest.mark.parametrize("hord", [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13])
def test_udzd_jax_matches_numpy_lane_every_hord(hord):
    """Every ported transport order against the NumPy lane on one
    fixture.  hord 6 is covered by gate 1; hord 10 additionally splits
    ord_in=8 / ord_ou=10 inside fv_tp_2d, and 9/13 are the only orders
    that reach ``pert_ppm``'s positive-definite arm."""
    fxt = _udzd_fixture(seed=31)
    n = fxt["n"]
    damp0 = [1.0e6, 0.0, 1.0e6, 0.0, 1.0e6, 999.0]
    ndif0 = [1, 0, 1, 0, 1, 999]
    ws0 = np.zeros((n, n))
    zh_n, ws_n, _, _ = _run_np_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0,
                                    hord=hord)
    zh_j, ws_j = _run_jax_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0,
                               hord=hord)
    assert np.isfinite(zh_n).all() and np.isfinite(zh_j).all()
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert _rel(zh_j, zh_n) <= 1e-12, (hord, _rel(zh_j, zh_n))
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert np.abs(ws_j - ws_n).max() <= 1e-12 * max(
        np.abs(ws_n).max(), 1.0), (hord, np.abs(ws_j - ws_n).max())
    sl = slice(fxt["ng"], fxt["ng"] + n)
    assert np.abs(zh_n[sl, sl, :] - fxt["zh0"][sl, sl, :]).max() > 1.0


# ---------------------------------------------------------------- gate 2
def test_udzd_jax_jit_eager_parity_and_no_retrace():
    """jit vs eager as an ASSERTION (not a comment), plus a trace counter
    on the PRODUCTION jit policy: two calls that differ only in data must
    compile once.  Both lanes are ALSO bound directly against the NumPy
    fp64 lane so a jit-only regression cannot hide inside the
    jit-vs-eager budget."""
    fxt = _udzd_fixture()
    n = fxt["n"]
    damp0 = [1.0e6, 0.0, 1.0e6, 0.0, 1.0e6, 999.0]
    ndif0 = [1, 0, 1, 0, 1, 999]
    ws0 = np.zeros((n, n))

    eager = _run_jax_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0)

    traces = {"n": 0}

    def _counted(*a, **kw):
        traces["n"] += 1
        return udzd_jax(*a, **kw)

    fn = make_update_dz_d_jit(_counted)      # the PRODUCTION jit policy
    jit1 = _run_jax_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0, fn=fn)
    jit2 = _run_jax_udzd(fxt, ndif0, damp0, fxt["zh0"] * 1.001, ws0,
                         fn=fn)
    assert traces["n"] == 1, traces["n"]
    assert not np.array_equal(jit1[0], jit2[0])   # the 2nd call ran

    native = _run_np_udzd(fxt, ndif0, damp0, fxt["zh0"], ws0)[:2]
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    bounds_ = {"zh": 1e-12, "ws": 1e-12}
    for name, e, j, nat in zip(("zh", "ws"), eager, jit1, native):
        scale = max(np.abs(e).max(), 1e-30)
        r = np.abs(np.asarray(j) - e).max() / scale
        assert r <= bounds_[name], (name, "jit-vs-eager", r)
        n_scale = max(np.abs(nat).max(), 1e-30)
        r_en = np.abs(np.asarray(e) - nat).max() / n_scale
        r_jn = np.abs(np.asarray(j) - nat).max() / n_scale
        assert r_jn <= bounds_[name], (name, "jit-vs-numpy", r_jn)
        assert r_jn <= r_en + bounds_[name], (name, r_jn, r_en)


# ------------------------------------------------------------ gate 3
@pytest.mark.parametrize("bad", ["area", "rarea", "dp0", "zs", "zh",
                                 "crx", "cry", "xfx", "yfx", "ws",
                                 "dxa", "dya", "del6_u", "del6_v"])
def test_udzd_jax_rejects_float32_every_operand(bad):
    """Every float64 operand, one at a time (the oracle build is
    -fdefault-real-8; a float32 operand silently degrades the solve)."""
    n, ng, km = 4, 3, 3
    full = n + 2 * ng
    args = {
        "area": np.full((full, full), 5.0e8),
        "rarea": np.full((full, full), 1.0 / 5.0e8),
        "dp0": np.full(km, 1.0e4),
        "zs": np.zeros((full, full)),
        "zh": np.zeros((full, full, km + 1)),
        "crx": np.zeros((n + 1, full, km)),
        "cry": np.zeros((full, n + 1, km)),
        "xfx": np.zeros((n + 1, full, km)),
        "yfx": np.zeros((full, n + 1, km)),
        "ws": np.zeros((n, n)),
        "dxa": np.full((full, full), 1.0e5),
        "dya": np.full((full, full), 1.0e5),
        "del6_u": np.ones((full, full + 1)),
        "del6_v": np.ones((full + 1, full)),
    }
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        udzd_jax((0,) * (km + 1), (0.0,) * (km + 1), UDZD_HORD,
                 (1, n, 1, n, ng), km, n + 1, n + 1,
                 jnp.asarray(args["area"]), jnp.asarray(args["rarea"]),
                 jnp.asarray(args["dp0"]), jnp.asarray(args["zs"]),
                 jnp.asarray(args["zh"]), jnp.asarray(args["crx"]),
                 jnp.asarray(args["cry"]), jnp.asarray(args["xfx"]),
                 jnp.asarray(args["yfx"]), jnp.asarray(args["ws"]),
                 UDZD_RDT, jnp.asarray(args["dxa"]),
                 jnp.asarray(args["dya"]), jnp.asarray(args["del6_u"]),
                 jnp.asarray(args["del6_v"]), **UDZD_FLAGS)


def _udzd_min_args(n=4, ng=3, km=3, **over):
    """Smallest well-formed operand set, for the guard tests."""
    full = n + 2 * ng
    a = {
        "ndif": (0,) * (km + 1), "damp": (0.0,) * (km + 1),
        "hord": UDZD_HORD, "bounds": (1, n, 1, n, ng), "km": km,
        "npx": n + 1, "npy": n + 1,
        "area": jnp.full((full, full), 5.0e8, jnp.float64),
        "rarea": jnp.full((full, full), 1.0 / 5.0e8, jnp.float64),
        "dp0": jnp.full((km,), 1.0e4, jnp.float64),
        "zs": jnp.zeros((full, full), jnp.float64),
        "zh": jnp.zeros((full, full, km + 1), jnp.float64),
        "crx": jnp.zeros((n + 1, full, km), jnp.float64),
        "cry": jnp.zeros((full, n + 1, km), jnp.float64),
        "xfx": jnp.zeros((n + 1, full, km), jnp.float64),
        "yfx": jnp.zeros((full, n + 1, km), jnp.float64),
        "ws": jnp.zeros((n, n), jnp.float64),
        "rdt": UDZD_RDT,
        "dxa": jnp.full((full, full), 1.0e5, jnp.float64),
        "dya": jnp.full((full, full), 1.0e5, jnp.float64),
        "del6_u": jnp.ones((full, full + 1), jnp.float64),
        "del6_v": jnp.ones((full + 1, full), jnp.float64),
    }
    a.update(over)
    return [a[k] for k in ("ndif", "damp", "hord", "bounds", "km", "npx",
                           "npy", "area", "rarea", "dp0", "zs", "zh",
                           "crx", "cry", "xfx", "yfx", "ws", "rdt",
                           "dxa", "dya", "del6_u", "del6_v")]


@pytest.mark.parametrize("hord", [0, 14, -7, -8, 20])
def test_udzd_jax_unported_hord_raises(hord):
    """DISPATCH HARDENING, reached THROUGH update_dz_d.

    The oracle silently routes iord <= -7 into the linear mord-5/6 arm
    and iord >= 14 into the monotonic ``bl = al - q`` arm; a typo there
    would run different physics without a word.  The guard lives in
    ``fv3_tp_core._validate_ord`` (against its ``_PPM_ORDS`` = -6..-1,
    1..13) and fires via ``fv_tp_2d``; this asserts update_dz_d actually
    routes into it rather than swallowing the value."""
    with pytest.raises(ValueError, match="not a supported scheme"):
        udzd_jax(*_udzd_min_args(hord=hord), **UDZD_FLAGS)


def test_udzd_jax_precondition_guards():
    """km, ng, damp/ndif length, ndif range and the ws shape all raise
    with a message that names the failing quantity."""
    n, ng, km = 4, 3, 3
    with pytest.raises(ValueError, match="km=1"):
        udzd_jax(*_udzd_min_args(km=1, ndif=(0, 0), damp=(0.0, 0.0),
                                 dp0=jnp.full((1,), 1.0e4, jnp.float64),
                                 zh=jnp.zeros((n + 2 * ng, n + 2 * ng, 2),
                                              jnp.float64),
                                 crx=jnp.zeros((n + 1, n + 2 * ng, 1),
                                               jnp.float64),
                                 xfx=jnp.zeros((n + 1, n + 2 * ng, 1),
                                               jnp.float64),
                                 cry=jnp.zeros((n + 2 * ng, n + 1, 1),
                                               jnp.float64),
                                 yfx=jnp.zeros((n + 2 * ng, n + 1, 1),
                                               jnp.float64)),
                 **UDZD_FLAGS)
    with pytest.raises(ValueError, match="ng=2"):
        n2, ng2, km2 = 4, 2, 3
        f2 = n2 + 2 * ng2
        udzd_jax((0,) * (km2 + 1), (0.0,) * (km2 + 1), UDZD_HORD,
                 (1, n2, 1, n2, ng2), km2, n2 + 1, n2 + 1,
                 jnp.full((f2, f2), 5.0e8, jnp.float64),
                 jnp.full((f2, f2), 2.0e-9, jnp.float64),
                 jnp.full((km2,), 1.0e4, jnp.float64),
                 jnp.zeros((f2, f2), jnp.float64),
                 jnp.zeros((f2, f2, km2 + 1), jnp.float64),
                 jnp.zeros((n2 + 1, f2, km2), jnp.float64),
                 jnp.zeros((f2, n2 + 1, km2), jnp.float64),
                 jnp.zeros((n2 + 1, f2, km2), jnp.float64),
                 jnp.zeros((f2, n2 + 1, km2), jnp.float64),
                 jnp.zeros((n2, n2), jnp.float64), UDZD_RDT,
                 jnp.full((f2, f2), 1.0e5, jnp.float64),
                 jnp.full((f2, f2), 1.0e5, jnp.float64),
                 jnp.ones((f2, f2 + 1), jnp.float64),
                 jnp.ones((f2 + 1, f2), jnp.float64), **UDZD_FLAGS)
    with pytest.raises(ValueError, match="km\\+1"):
        udzd_jax(*_udzd_min_args(ndif=(0,) * km), **UDZD_FLAGS)
    with pytest.raises(ValueError, match="out of range"):
        # nord = ng is one past the widest stencil del6_vt_flux can read.
        udzd_jax(*_udzd_min_args(ndif=(ng,) * (km + 1),
                                 damp=(1.0,) * (km + 1)), **UDZD_FLAGS)
    with pytest.raises(ValueError, match="ws must be"):
        udzd_jax(*_udzd_min_args(
            ws=jnp.zeros((n, n + 1), jnp.float64)), **UDZD_FLAGS)
    with pytest.raises(ValueError, match="del6_u must be"):
        udzd_jax(*_udzd_min_args(
            del6_u=jnp.ones((n + 2 * ng, n + 2 * ng), jnp.float64)),
            **UDZD_FLAGS)


# ---------------------------------------------------------------- gate 4
# Three thirds, per the campaign strategy section 7:
#   4a  smooth-state order-2 check_grads (below);
#   4b/4c  ONE-SIDED directional derivatives approaching each switching
#          surface from BOTH sides, against the documented branch
#          derivative on each side -- the DZ_MIN floor (4b, where the
#          floored-side derivative is EXACTLY zero), the upwind zero
#          (4c), and one representative PPM limiter surface per hord
#          family (4d);
#   4e  adjoint consistency <Jv, w> == <v, J^T w>, which uses NO finite
#       differences, so its power does not depend on the FD step or on
#       the array's dynamic range.
# 4a alone proves differentiability of the LINEAR transport arm only.


def _udzd_grad_fixture(n=6, ng=3, km=4, seed=404):
    """The shared differentiability fixture for every gate-4 test.

    FD-friendly and self-consistent (area ~5e2, metric lengths ~1e1,
    del6 ~1, heights ~1e3): the gate-1 fixture's 4e11 areas would make a
    fixed-eps numerical derivative meaningless.  This is a
    DIFFERENTIABILITY fixture, not a physical one — value parity against
    the NumPy lane is gate 1's job.

    ``crx``/``cry`` are CONSTANT DOWN EACH COLUMN with a sign that
    depends on (i, j) only.  ``edge_profile`` is linear in q, so a
    k-constant column maps to ``c(i,j) * u(k)`` for one profile ``u``
    shared by every column — which is what lets gate 4c locate the
    upwind surface by measurement rather than by assumption.  The
    (i + j) checkerboard, with ``cry`` in antiphase, makes BOTH upwind
    branches fire.

    ``n = 6`` (npx = 7) is the smallest square face on which the
    MONOTONIC branch's interior window ``[is1, ie1] = [3, 4]`` is
    non-empty; at n = 4 it is empty and the iord >= 7 limiters would
    never run, so a PPM-limiter gate there would be vacuous.

    ``dp0`` is uniform, so ``edge_profile`` runs at ``g0 = 1``.
    """
    full = n + 2 * ng
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(full), np.arange(full), indexing="ij")
    sgn = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    mag = 0.2 * (1.0 + 0.1 * np.sin(ii + 2.0 * jj))
    kprof = (1.0 + 0.05 * np.arange(km))[None, None, :]
    area = np.abs(5.0e2 * (1.0 + 0.1 * rng.standard_normal((full, full))))
    zh = np.cumsum(np.abs(300.0 + 40.0 * rng.standard_normal(
        (full, full, km + 1))), axis=2)[:, :, ::-1].copy()
    return {
        "n": n, "ng": ng, "km": km, "full": full,
        "npx": n + 1, "npy": n + 1, "bnds": _bounds(_BD(n, ng)),
        "crx": np.broadcast_to((sgn * mag)[:n + 1, :, None],
                               (n + 1, full, km)).copy(),
        "cry": np.broadcast_to((-sgn * mag)[:, :n + 1, None],
                               (full, n + 1, km)).copy(),
        "xfx": (1.0e1 * (1.0 + 0.1 * rng.standard_normal(
            (n + 1, full, 1))) * kprof),
        "yfx": (1.0e1 * (1.0 + 0.1 * rng.standard_normal(
            (full, n + 1, 1))) * kprof),
        "dp0": np.full(km, 1.0e4),
        "area": area, "rarea": 1.0 / area,
        "dxa": np.abs(1.0e1 * (1.0 + 0.1 * rng.standard_normal(
            (full, full)))),
        "dya": np.abs(1.0e1 * (1.0 + 0.1 * rng.standard_normal(
            (full, full)))),
        "del6_u": 1.0 + 0.1 * rng.standard_normal((full, full + 1)),
        "del6_v": 1.0 + 0.1 * rng.standard_normal((full + 1, full)),
        "zh": zh, "zs": np.array(zh[:, :, km], copy=True),
        "ws0": rng.standard_normal((n, n)),
        "damp": (1.0e3, 0.0, 1.0e3, 0.0, 0.0),   # km+1 = 5 entries
        "ndif": (1, 0, 1, 0, 0),
    }


def _udzd_grad_runner(fxt, hord, *, damp=None, ndif=None):
    """``(zh, crx) -> (zh_out, ws_out)`` on the gate-4 fixture.

    Only the two operands the switching-surface gates move are exposed;
    everything else is closed over, so a directional derivative here is
    a derivative along a straight line in exactly those two."""
    damp = fxt["damp"] if damp is None else damp
    ndif = fxt["ndif"] if ndif is None else ndif
    args0 = (tuple(ndif), tuple(damp), hord, fxt["bnds"], fxt["km"],
             fxt["npx"], fxt["npy"])

    def run(zh_, crx_):
        return udzd_jax(*args0, jnp.asarray(fxt["area"]),
                        jnp.asarray(fxt["rarea"]), jnp.asarray(fxt["dp0"]),
                        jnp.asarray(fxt["zs"]), zh_, crx_,
                        jnp.asarray(fxt["cry"]), jnp.asarray(fxt["xfx"]),
                        jnp.asarray(fxt["yfx"]), jnp.asarray(fxt["ws0"]),
                        UDZD_RDT, jnp.asarray(fxt["dxa"]),
                        jnp.asarray(fxt["dya"]), jnp.asarray(fxt["del6_u"]),
                        jnp.asarray(fxt["del6_v"]), **UDZD_FLAGS)
    return run


def udzd_call_and_operands(fxt):
    """``(_call, ja)`` for the gate-4 fixture -- the operand mapping once.

    PUBLIC (no leading underscore) because the standing triage probe
    ``scripts/tmp/fv3_gradient_gate_triage.py`` builds the SAME gradient
    groups to decide whether this gate's 2.1e-5 check_grads gap is
    finite-difference truncation or a wrong Jacobian.  A probe that
    retyped the fourteen-operand order would be measuring its own
    transcription, and a private cross-module import is banned by a CI
    ratchet -- so the mapping lives here, once, and both callers take it
    from this function.
    """
    km, npx, npy, bnds = fxt["km"], fxt["npx"], fxt["npy"], fxt["bnds"]
    statics = dict(UDZD_FLAGS)

    def _call(nd, dm, dp0_, crx_, cry_, xfx_, yfx_, zh_, zs_, area_,
              rarea_, dxa_, dya_, du_, dv_, ws_):
        """(ndif, damp) first so the no-damp control reuses the SAME
        argument mapping — one place for the operand order."""
        return udzd_jax(nd, dm, 2, bnds, km, npx, npy, area_, rarea_,
                        dp0_, zs_, zh_, crx_, cry_, xfx_, yfx_, ws_,
                        UDZD_RDT, dxa_, dya_, du_, dv_, **statics)

    ja = [jnp.asarray(fxt[k]) for k in
          ("dp0", "crx", "cry", "xfx", "yfx", "zh", "zs", "area",
           "rarea", "dxa", "dya", "del6_u", "del6_v", "ws0")]
    return _call, ja


def udzd_loss(zh_out, ws_out):
    """The scalar the gate differentiates -- shared with the probe."""
    return jnp.sum(zh_out * zh_out) / 1e6 + jnp.sum(ws_out * ws_out)


def _one_sided(phi, s, h):
    """One-sided difference quotient whose TWO stencil points lie on the
    SAME side of ``s`` (sign of ``h`` picks the side), so it never
    straddles a switching surface."""
    return (phi(s + h) - phi(s)) / h


def test_udzd_jax_check_grads_order2_away_from_switches():
    """Order-2 fwd+rev gradients over every dynamic operand except ``ws``
    (whose gradient is identically zero and is asserted so below).

    ``hord = 2`` — the oracle's PERFECTLY LINEAR PPM arm (tp_core.F90
    xppm/yppm ``mord == 2``).  That arm contains no ``smt5``/``smt6``
    selector, no ``copysign``/``min``/``max`` limiter and no
    ``pert_ppm``, so the ONLY non-smooth sites left in the whole call
    tree are:

      * the upwind selection ``jnp.where(c > 0)`` at every flux point,
        where ``c`` is ``crx_adv``/``cry_adv``.  Control 1 below rebuilds
        both with the ALREADY-CERTIFIED ``edge_profile`` twin (no
        re-derived numerics) and asserts a margin far beyond any FD step
        at |c| ~ 1e-1, plus that BOTH branches fire;
      * the ``dz_min`` bottom-up ``jnp.maximum`` floor.  Control 2
        asserts every output level clears it strictly.

    The PPM limiter switches at hord 1/3/4/5/6 and the ``pert_ppm``
    branches at hord 7/9/12/13 are C^0 by construction and are NOT
    differentiated here; they are named in ``update_dz_d``'s docstring.

    Magnitudes are FD-friendly and self-consistent (area ~5e2, metric
    lengths ~1e1, del6 ~1, heights ~1e3): the gate-1 fixture's 4e11 areas
    would make check_grads' fixed-eps numerical derivative meaningless.
    This is a DIFFERENTIABILITY fixture, not a physical one — the value
    comparison against the NumPy lane is gate 1's job.
    """
    fxt = _udzd_grad_fixture()
    # Only the names the CONTROLS below read are unpacked; every
    # operand the gradients take now comes from
    # `udzd_call_and_operands`, which owns the order.
    n, ng, km = fxt["n"], fxt["ng"], fxt["km"]
    crx, cry = fxt["crx"], fxt["cry"]
    dp0, ws0 = fxt["dp0"], fxt["ws0"]
    damp, ndif = fxt["damp"], fxt["ndif"]
    assert len(damp) == km + 1 and len(ndif) == km + 1
    assert any(d > 1.0e-5 for d in damp), "del6 branch never runs"
    assert any(d <= 1.0e-5 for d in damp), "plain branch never runs"

    # --- control 1: the upwind switch is off, with margin, both ways ---
    for name, cc in (("crx", crx), ("cry", cry)):
        adv, _ = edge_jax(jnp.asarray(cc.reshape(-1, km)),
                          jnp.asarray(cc.reshape(-1, km)),
                          0, km, jnp.asarray(dp0), False, 0)
        adv = np.asarray(adv)
        assert np.abs(adv).min() > 1.0e-2, (name, np.abs(adv).min())
        assert (adv > 0).any() and (adv < 0).any(), name

    _call, ja = udzd_call_and_operands(fxt)
    zh_o, ws_o = _call(ndif, damp, *ja)
    zh_o = np.asarray(zh_o)
    assert np.isfinite(zh_o).all() and np.isfinite(np.asarray(ws_o)).all()

    # --- control 2: the dz_min floor never fired ---
    w = slice(ng, ng + n)
    gap = zh_o[w, w, :-1] - (zh_o[w, w, 1:] + DZ_MIN)
    assert gap.min() > 1.0, f"dz_min floor margin {gap.min()} too small"

    # --- control 3: the del-nord term is not a no-op at this damp ---
    # A gradient path that contributes ~0 to the VALUE is not tested by
    # check_grads, so the del6 term must MOVE the answer before its
    # gradient is claimed.  The reference keeps the SAME per-level branch
    # structure (damp just above the 1e-5 gate) so the difference is the
    # del6 TERM alone, not the damped-vs-undamped operator swap; and it
    # is measured inside the COMPUTE WINDOW, because a damp=0 reference
    # would also differ in the ghost corners (the undamped branch
    # aliases zh into copy_corners) and that would flatter the control.
    tiny = tuple(2.0e-5 if d > 1.0e-5 else 0.0 for d in damp)
    zh_ref, _ = _call(ndif, tiny, *ja)
    d_del6 = np.abs(zh_o[w, w, :] - np.asarray(zh_ref)[w, w, :]).max()
    assert d_del6 > 1.0, f"del-nord term moved zh by only {d_del6} m"

    _loss = udzd_loss

    def f(dp0_, crx_, cry_, xfx_, yfx_, zh_):
        return _loss(*_call(ndif, damp, dp0_, crx_, cry_, xfx_, yfx_,
                            zh_, *ja[6:]))

    check_grads(f, tuple(ja[:6]), order=2, modes=("fwd", "rev"))

    def g(zs_, area_, rarea_, dxa_, dya_, du_, dv_):
        return _loss(*_call(ndif, damp, *ja[:6], zs_, area_, rarea_,
                            dxa_, dya_, du_, dv_, ja[13]))

    check_grads(g, tuple(ja[6:13]), order=2, modes=("fwd", "rev"))

    # ``ws`` is intent(out) in the oracle and every compute-window cell
    # is written, so its INPUT cannot influence anything.  Asserted here
    # rather than claimed in prose: a nonzero ws0 with a zero gradient.
    assert np.abs(ws0).max() > 0.0
    g_ws = jax.grad(
        lambda ws_: _loss(*_call(ndif, damp, *ja[:13], ws_)))(ja[13])
    assert np.array_equal(np.asarray(g_ws), np.zeros((n, n)))


# ------------------------------------------------------- gate 4b (floor)
def test_udzd_jax_one_sided_at_dz_min_floor():
    """DZ_MIN floor, approached from BOTH sides, against the DOCUMENTED
    branch derivative on each side.

    Knob: a scalar ``s`` added to the SINGLE input cell
    ``zh[i0, j0, k0]``.  Functional: the SINGLE output cell
    ``zh_out[i0, j0, k0]``.

    Why the two branch derivatives are known in closed form.  The k loop
    is level-independent in its flux phase, so input level k0 changes the
    flux value at level k0 only; the bottom-up limiter
    ``zh[k] = max(zh[k], zh[k+1] + DZ_MIN)`` then propagates UPWARD, so
    ``zh_out[i0, j0, k0+1]`` cannot depend on ``s`` at all.  Hence

      * FLOORED side: ``zh_out[k0] = zh_out[k0+1] + DZ_MIN``, an
        s-independent quantity, so ``dL/ds`` is EXACTLY 0 — asserted
        with ``== 0.0``, no tolerance;
      * UNFLOORED side: ``zh_out[k0]`` is the flux-form value, whose
        leading term is ``q * area / den``, so ``dL/ds`` is O(1) and
        bounded away from 0.

    Locating the surface needs no bisection: at ``hord = 2`` the whole
    transport is LINEAR in q, so ``g(s) = zh_out[k0] - zh_out[k0+1] -
    DZ_MIN`` is the positive part of an affine function of ``s``.  Two
    samples on the unfloored side give the line, hence the crossing
    exactly; the located crossing is then re-verified by evaluating ``g``
    on each side of it.
    """
    fxt = _udzd_grad_fixture()
    ng, km = fxt["ng"], fxt["km"]
    i0, j0, k0 = ng + 2, ng + 3, 1        # interior cell, interior level
    assert 0 <= k0 < km
    run = _udzd_grad_runner(fxt, 2)
    zh0 = jnp.asarray(fxt["zh"])
    crx0 = jnp.asarray(fxt["crx"])
    bump = jnp.zeros_like(zh0).at[i0, j0, k0].set(1.0)

    def out(s):
        return run(zh0 + s * bump, crx0)[0]

    def gap(s):
        z = out(s)
        return float(z[i0, j0, k0] - z[i0, j0, k0 + 1] - DZ_MIN)

    def loss(s):
        return out(s)[i0, j0, k0]

    # --- locate the surface from the affine unfloored branch ---
    # Measured (NumPy shim, this fixture): g(1e3) = 1336, g(2e3) = 2337,
    # slope = 1.0011 -- i.e. d(flux)/ds is the predicted area/den ~ 1 --
    # and s* = -334.757.  At s* -/+ 1e-2 the gap is exactly 0.0 and
    # 1.0011e-2, and the one-sided quotients are -0.0 and 1.00113.
    span = 1.0e3                       # >> the ~300 m level separation
    g1, g2 = gap(span), gap(2.0 * span)
    assert g1 > 0.0 and g2 > g1, (g1, g2)      # both on the raised side
    slope = (g2 - g1) / span
    assert slope > 0.1, slope                  # d(flux)/ds ~ area/den ~ 1
    s_star = span - g1 / slope

    # --- the located crossing is REAL: floored below it, not above ---
    d = 1.0e-2                          # >> any FD step, << the 300 m gap
    assert gap(s_star - d) == 0.0, gap(s_star - d)
    assert gap(s_star + d) > 0.5 * slope * d, gap(s_star + d)

    # --- side A (floored): the branch derivative is EXACTLY zero ---
    g_lo = float(jax.grad(loss)(s_star - d))
    assert g_lo == 0.0, g_lo
    fd_lo = float(_one_sided(loss, s_star - d, -1.0e-4))
    assert fd_lo == 0.0, fd_lo

    # --- side B (unfloored): JVP == one-sided FD, and it is O(1) ---
    g_hi = float(jax.grad(loss)(s_star + d))
    fd_hi = float(_one_sided(loss, s_star + d, +1.0e-4))
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert abs(g_hi - fd_hi) <= 1e-6 * max(abs(g_hi), 1.0), (g_hi, fd_hi)
    assert abs(g_hi) > 0.1, g_hi

    # --- the kink is real: the two branch derivatives DIFFER ---
    assert abs(g_hi - g_lo) > 0.1, (g_lo, g_hi)


# ------------------------------------------------------ gate 4c (upwind)
def test_udzd_jax_one_sided_at_upwind_zero():
    """Upwind selection ``jnp.where(c > 0)``, approached from BOTH sides.

    Knob: a scalar ``s`` added to ``crx[i0, j0, :]`` (one column, every
    level).  ``edge_profile`` is linear in q and elementwise over
    columns, so this moves the Courant number of exactly ONE column and
    leaves every other column where the fixture put it (|c| >= 0.18).

    The surface is LOCATED BY MEASUREMENT, not assumed: the certified
    ``edge_profile`` twin is evaluated at ``s = 0`` and ``s = 1`` to get
    the affine coefficients ``a_k``, ``b_k`` of ``c_k(s) = a_k + s
    b_k``; the per-level crossings ``-a_k / b_k`` then bracket the
    surface, and both sides are re-confirmed by recomputing ``c`` there.

    The documented branch values at the crossing are the oracle's own
    (tp_core.F90 xppm ``mord == 2``): for ``c > 0`` the flux is built
    from ``q(i-1)`` and ``al(i-1), al(i)``, for ``c <= 0`` from ``q(i)``
    and ``al(i), al(i+1)``.  Both give ``al(i)`` at ``c = 0`` — the
    operator is CONTINUOUS there — but their c-derivatives differ, which
    is exactly the kink this gate asserts.
    """
    fxt = _udzd_grad_fixture()
    ng, km = fxt["ng"], fxt["km"]
    i0, j0 = 3, ng + 2                       # a crx column, interior j
    run = _udzd_grad_runner(fxt, 2)
    zh0 = jnp.asarray(fxt["zh"])
    crx0 = jnp.asarray(fxt["crx"])
    bump = jnp.zeros_like(crx0).at[i0, j0, :].set(1.0)

    def c_of(s):
        """The advected Courant column, via the CERTIFIED edge twin."""
        c3 = np.asarray(crx0 + s * bump)
        adv, _ = edge_jax(jnp.asarray(c3.reshape(-1, km)),
                          jnp.asarray(c3.reshape(-1, km)), 0, km,
                          jnp.asarray(fxt["dp0"]), False, 0)
        return np.asarray(adv).reshape(c3.shape[0], c3.shape[1], km + 1)

    a = c_of(0.0)[i0, j0, :]
    b = c_of(1.0)[i0, j0, :] - a
    assert b.min() > 0.5, b                  # c increases with s, all k
    cross = -a / b
    # Stand-off from the surface.  Measured (NumPy shim): the per-level
    # crossings agree to 1.1e-16 and b == 1.000000, so the spread term
    # is negligible and 1e-2 is what actually separates the two sides --
    # still 4 orders inside the nearest other column's |c| = 0.18.
    d = 1.0e-2 + float(cross.max() - cross.min())

    # Both sides CONFIRMED by recomputation, not by assumption; and the
    # other columns must not have crossed with us.
    c_pos = c_of(float(cross.max()) + d)
    c_neg = c_of(float(cross.min()) - d)
    assert c_pos[i0, j0, :].min() > 0.0, c_pos[i0, j0, :]
    assert c_neg[i0, j0, :].max() < 0.0, c_neg[i0, j0, :]
    other = np.ones(c_pos.shape[:2], bool)
    other[i0, j0] = False
    assert np.abs(c_pos[other, :]).min() > 1.0e-2
    assert np.abs(c_neg[other, :]).min() > 1.0e-2

    rng = np.random.default_rng(717)
    wt = jnp.asarray(rng.standard_normal(zh0.shape))

    def phi(s):
        return jnp.sum(wt * run(zh0, crx0 + s * bump)[0])

    s_pos = float(cross.max()) + d
    s_neg = float(cross.min()) - d
    h = 1.0e-6
    out = {}
    for tag, s, hh in (("pos", s_pos, +h), ("neg", s_neg, -h)):
        g = float(jax.grad(phi)(s))
        fd = float(_one_sided(phi, s, hh))
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        assert abs(g - fd) <= 1e-5 * max(abs(g), 1.0), (tag, g, fd)
        out[tag] = g
    # The kink is real (a control that perturbs a zero is not a control).
    assert abs(out["pos"] - out["neg"]) > 1e-3 * max(
        abs(out["pos"]), abs(out["neg"]), 1.0), out


# ------------------------------------------------ gate 4d (PPM limiters)
# Every supported hord, so no PPM limiter family is left undifferentiated.
# hord 2 is the CONTROL, not a case: mord 2 is the perfectly linear arm,
# so with the DZ_MIN floor inactive the whole map is AFFINE in zh and the
# scanned derivative must be CONSTANT.  A detector that fires there is
# broken; a detector that fires nowhere else is vacuous.
@pytest.mark.parametrize("hord", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                  13])
def test_udzd_jax_one_sided_across_ppm_limiter(hord):
    """One representative PPM limiter surface per hord family.

    Knob: a scalar ``s`` on the SINGLE input cell ``zh[i0, j0, k0]``.
    Functional: the SINGLE output cell ``zh_out[i0+1, j0, k0]``, the
    spike's neighbour, whose PPM stencil sees the spike asymmetrically.
    LOCALITY IS THE DESIGN, and it was forced by measurement: a global
    functional over the whole field crosses hundreds of limiter surfaces
    at once, and the largest single jump then stands only ~5x above the
    typical adjacent difference (measured under the NumPy shim), which
    is far too weak to call a kink.  With one cell driving one cell the
    same ratio is 3e10-1e298.

    The surfaces are LOCATED, not constructed: the scalar derivative
    ``phi'`` is sampled on a grid and the largest adjacent jump is taken
    as the bracket.

    At the two BRACKETING grid points — strictly on opposite sides — the
    analytic derivative is asserted against a one-sided FD whose stencil
    steps AWAY from the bracket, so it never straddles the surface.  The
    two derivatives must then differ by much more than the smooth
    variation between neighbouring grid points, which is what makes the
    located point a kink rather than a slope change.

    NOTE, and it matters for how the lane is used: several of these
    limiters make the operator DISCONTINUOUS, not merely non-smooth.
    ``smt5``/``smt6`` select between ``flux`` and ``flux + fx1`` with
    ``fx1`` nonzero at the switch (mord 1/3/4/5/6), and mord 10's
    NEAR_ZERO / clamp arms do the same.  One-sided derivatives on each
    side are still well defined and are what this gate checks; a
    gradient evaluated ON a surface is not.
    """
    fxt = _udzd_grad_fixture()
    ng, n, km = fxt["ng"], fxt["n"], fxt["km"]
    i0, j0, k0 = ng + 2, ng + 3, 1
    assert 0 <= k0 < km
    run = _udzd_grad_runner(fxt, hord)
    zh0 = jnp.asarray(fxt["zh"])
    crx0 = jnp.asarray(fxt["crx"])
    bump = jnp.zeros_like(zh0).at[i0, j0, k0].set(1.0)
    w = slice(ng, ng + n)

    def state(s):
        return run(zh0 + s * bump, crx0)[0]

    def phi(s):
        return state(s)[i0 + 1, j0, k0]

    def floor_gap(s):
        z = np.asarray(state(s))
        assert np.isfinite(z).all(), s
        return float((z[w, w, :-1] - (z[w, w, 1:] + DZ_MIN)).min())

    dphi = jax.jit(jax.grad(phi))
    # Range and resolution PINNED TO MEASUREMENT (NumPy shim, values via
    # finite differences -- the shim has no autodiff; the JAX gate below
    # computes the same quantity with jax.grad).  Over s in [-200, 200]
    # at 201 points every hord except 2 shows one dominant jump of
    # 3.8e-3..6.7e-1 against a median adjacent difference at rounding
    # (1.1e-13, or exactly 0 where the branch is piecewise linear).
    s_grid = np.linspace(-200.0, 200.0, 201)
    step = float(s_grid[1] - s_grid[0])
    d1 = np.array([float(dphi(float(s))) for s in s_grid])

    # The DZ_MIN floor must stay inactive across the scan, or the kink
    # found would be gate 4b's floor and not a PPM limiter at all.
    # Measured min gap over this scan: 97..223 m for every hord.
    for s in (float(s_grid[0]), float(s_grid[len(s_grid) // 2]),
              float(s_grid[-1])):
        assert floor_gap(s) > 1.0, (hord, s, floor_gap(s))

    jumps = np.abs(np.diff(d1))
    med = float(np.median(jumps))

    if hord == 2:
        # CONTROL, not a case: mord 2 is linear in q and the floor is
        # off, so the whole map is AFFINE in zh and phi' is CONSTANT.
        # A detector that fires here is broken.  Measured spread
        # 5.7e-13 against a derivative of 1.07e-2 (relative 5e-11).
        spread = float(np.max(d1) - np.min(d1))
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        assert spread <= 1e-9 * max(abs(float(np.mean(d1))), 1.0), spread
        return

    i = int(np.argmax(jumps))
    # A KINK, not smooth variation and not rounding.  Two floors, both
    # needed: the jump must tower over the typical adjacent difference
    # (which can be exactly 0 when the branch is piecewise linear, hence
    # the second floor), and it must be far above the rounding level of
    # the derivative itself.  Measured margin: >= 3.5 orders on both.
    floor_abs = 1e-6 * max(float(np.abs(d1).max()), 1.0)
    assert jumps[i] > max(50.0 * med, floor_abs), (hord, jumps[i], med,
                                                   floor_abs)

    # One-sided FD at the two BRACKETING grid points, stepping AWAY from
    # the bracket so neither stencil straddles the surface.
    h = step / 100.0
    got = []
    for s, hh in ((float(s_grid[i]), -h), (float(s_grid[i + 1]), +h)):
        assert floor_gap(s) > 1.0, (hord, "floor fired at bracket", s)
        g = float(dphi(s))
        fd = float(_one_sided(phi, s, hh))
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        assert abs(g - fd) <= 1e-4 * max(abs(g), 1.0), (hord, s, g, fd)
        got.append(g)
    assert abs(got[1] - got[0]) > max(50.0 * med, floor_abs), (hord, got)


# --------------------------------------------------- gate 4e (adjoint)
# Deliberately includes the DECK order (6) and both fv_tp_2d ord splits
# (10 -> ord_in 8 / ord_ou 10), because this identity is the only
# gradient check whose power does NOT depend on an FD step or on the
# array's dynamic range.
@pytest.mark.parametrize("hord", [2, 6, 8, 10, 13])
def test_udzd_jax_adjoint_consistency(hord):
    """``<J v, w> == <v, J^T w>`` with ``J v`` from ``jax.jvp`` and
    ``J^T w`` from ``jax.vjp``.

    No finite differences anywhere, so a single element many orders
    above the array's typical magnitude cannot defeat it the way it
    defeats an order-2 FD check.

    WHAT THIS DOES AND DOES NOT ESTABLISH.  It proves reverse mode is the
    exact transpose of forward mode — it catches a wrong VJP rule, a
    mis-transposed scatter, a dropped ``.at[].set()`` cotangent.  It does
    NOT catch a linearisation that is wrong the SAME way in both modes;
    only the FD gates above do that.  The two are complementary, and
    neither replaces the other.
    """
    fxt = _udzd_grad_fixture()
    args0 = (tuple(fxt["ndif"]), tuple(fxt["damp"]), hord, fxt["bnds"],
             fxt["km"], fxt["npx"], fxt["npy"])
    names = ("area", "rarea", "dp0", "zs", "zh", "crx", "cry", "xfx",
             "yfx")
    tail = ("dxa", "dya", "del6_u", "del6_v")

    def f(*ops):
        head, rest = ops[:9], ops[9:]
        return udzd_jax(*args0, *head, jnp.asarray(fxt["ws0"]),
                        UDZD_RDT, *rest, **UDZD_FLAGS)

    primals = tuple(jnp.asarray(fxt[k]) for k in names + tail)
    rng = np.random.default_rng(919)
    tangents = tuple(jnp.asarray(rng.standard_normal(p.shape))
                     for p in primals)
    out, jv = jax.jvp(f, primals, tangents)
    out2, vjp_fn = jax.vjp(f, *primals)
    cot = tuple(jnp.asarray(rng.standard_normal(o.shape)) for o in out)
    jtw = vjp_fn(cot)

    assert all(np.isfinite(np.asarray(o)).all() for o in out)
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(out, out2))
    lhs = float(sum(jnp.sum(a * b) for a, b in zip(jv, cot)))
    rhs = float(sum(jnp.sum(a * b) for a, b in zip(tangents, jtw)))
    # Non-vacuity: a zero pairing would satisfy the identity trivially.
    assert abs(lhs) > 1.0e-6 * float(
        sum(jnp.sum(jnp.abs(a)) for a in jv)), (hord, lhs)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs)), (
        hord, lhs, rhs)
