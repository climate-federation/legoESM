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
    make_sim1_solver_jit,
    sim1_solver as sim1_jax,
    sim1_solver_jit,
)
from legoesm.core.fv3_native_nh_core import (  # noqa: E402
    sim1_solver as sim1_np,
)
from legoesm.grids.fv3_native_gridstruct import (  # noqa: E402
    FV3_KAPPA,
    FV3_RDGAS,
)

from tests.grids.test_fv3_native_nh_core import (  # noqa: E402
    KM,
    NI,
    _balanced_column,
    _run_native_sim1,
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
