"""Certification of the JAX ``fv_mapz`` lane against the NumPy fp64 lane.

Four gates per public routine (the NH JAX-mirror pattern), plus two the
``fv_mapz`` node needs and the NH kernels did not:

1. **equivalence** vs ``fv3_native_mapz`` on the SAME fixtures, over
   EVERY ``kord`` the NumPy lane supports (8, 9, 10, 11, 12, 13, 14, 15,
   16 and >16) and every ``iv`` arm;
2. **jit vs eager**, as an ASSERTION on every routine, not a comment;
3. **guards**: float32 raises ``TypeError``, ``kord <= 7`` raises
   ``ValueError``, the unreachable ``iv=-3`` and every unported lane
   raise exactly as the NumPy lane does -- each shown NON-VACUOUS;
4. **``check_grads(order=2)``** at states PROVEN away from the limiter
   switches by measured margins, with every non-smooth site named.
5. **CONSERVATION** (tolerance-independent), the five invariants of the
   review: global dry mass, per-tracer mass, pressure-thickness closure,
   constant-field preservation, and the ``w_limiter``'s weighted
   momentum.  See ``_conservation_note`` below for the halo / area-weight
   statement.
6. **interface-tie fixtures**: exact, one-ULP-below and one-ULP-above a
   source interface.  A one-ULP move there can select a DIFFERENT source
   layer and a DIFFERENT formula, so a spread between the three is
   expected and is NOT rounding-scale; what must hold is that BOTH lanes
   make the SAME selection, which is what the assertions compare.
7. **adjoint consistency** ``<J v, w> == <v, J^T w>`` -- forward mode
   against reverse mode, with NO finite-difference step, so it has no
   step size and no dynamic-range sensitivity and it runs even ON a
   switching surface.  Added after job 9400424, where eight of nine
   failures were ``check_grads(order=2)`` calls and three causes were
   confounded.
8. **one-sided derivatives AT the switching surface**, approached from
   both sides, each checked against a same-side finite difference AND
   against the closed-form branch derivative for that side.

WHAT JOB 9400424 CHANGED HERE: nine failures, all EXPECTATION, none
code.  Six were one arbitrary factor (a 0.5 where the switch is at 1.0
and the worst measured ratio was 0.70); two were fixtures built smooth
in the LAYER INDEX rather than in the COORDINATE the reconstruction
lives in; one was a match string on a refusal Python itself performs.
The margins are now relative, calibrated to that measurement, and every
one of them REPORTS what it measured.

TOLERANCE STATUS: every numeric agreement bound in this file is
MEASURED (job 9425294, the LEGOESM_FV3_TOL_MEASURE sweep) and set to
measured x 10 (exact-0.0 measurements get a 1e-15 eps guard).  The worst
lane-parity figure is pkz at 8.002e-15; the adjoint identities peak at
3.825e-14 (profile kord 9/13); the loosest bound in the file is the
kord-9 A6-grouping value agreement at 2.427e-13 measured.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from tests.grids.fv3_gate_helpers import gate_scalar  # noqa: E402

from tests.grids.fv3_gate_helpers import check_adjoint  # noqa: E402

from legoesm.core.fv3_mapz import (  # noqa: E402
    LagrangianToEulerianOut,
    cs_limiters as cs_limiters_j,
    cs_limiters_jit,
    cs_profile as cs_profile_j,
    cs_profile_jit,
    lagrangian_to_eulerian as l2e_j,
    lagrangian_to_eulerian_jit,
    make_cs_limiters_jit,
    make_cs_profile_jit,
    make_lagrangian_to_eulerian_jit,
    make_map1_ppm_jit,
    make_map1_q2_jit,
    make_map_scalar_jit,
    make_scalar_profile_jit,
    map1_ppm as map1_ppm_j,
    map1_ppm_jit,
    map1_q2 as map1_q2_j,
    map1_q2_jit,
    map_scalar as map_scalar_j,
    map_scalar_jit,
    pad1 as pad1_j,
    pad1_jit,
    scalar_profile as scalar_profile_j,
    scalar_profile_jit,
    unpad1 as unpad1_j,
    unpad1_jit,
)
from legoesm.core.fv3_native_mapz import (  # noqa: E402
    T_MIN,
    W_MAX_MAPZ,
    cs_limiters as cs_limiters_n,
    cs_profile as cs_profile_n,
    lagrangian_to_eulerian as l2e_n,
    map1_ppm as map1_ppm_n,
    map1_q2 as map1_q2_n,
    map_scalar as map_scalar_n,
    pad1 as pad1_n,
    scalar_profile as scalar_profile_n,
    unpad1 as unpad1_n,
)

from tests.grids.test_fv3_native_mapz import (  # noqa: E402
    KM,
    _a4,
    _column,
    _face,
    _lagrangian_edges,
    _nh_face,
)

# Every ``kord`` the NumPy lane admits.  8 takes the Huynh arm
# (abs(kord) < 9), 17 takes the ``abs(kord) > 16`` early return
# (fv_mapz.F90:1932/:2365), 11 is the trailing arm, and 10/12/13/14/15/16
# each have their own; everything above 9 also reads ext5/ext6, which
# :2003 builds only there.  kord <= 7 is REFUSED (ppm_profile), below.
ALL_KORD = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17)

_conservation_note = """
HALO AND AREA WEIGHTS, stated because both change the answer:

* ``fv_mapz`` is COLUMN-LOCAL -- there is no horizontal operator
  anywhere in it -- so every invariant below is asserted PER COLUMN, not
  as a global sum.  A per-column identity is strictly stronger than any
  horizontally aggregated one, which could hide two columns whose errors
  cancel.
* Because the invariants are per-column, **no area weights are involved
  and none are used**: the cubed-sphere metric never enters a vertical
  remap.  (A global mass total would need ``area``; this diagnostic
  deliberately does not compute one.)
* The diagnostics run on the COMPUTE WINDOW only (i, j = is..ie,
  js..je).  Halo cells are asserted separately to be carried through
  BITWISE unchanged, which is the correct halo statement for a routine
  that never writes them.
"""


def _rel(a, b):
    """Max abs difference relative to the reference's own scale."""
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


def _delp_col(im, km, seed=5):
    """A 1-based (im, km+1) thickness array with real variation."""
    rng = np.random.default_rng(seed)
    dp = np.zeros((im, km + 1), dtype=np.float64)
    dp[:, 1:] = 0.8 + 0.5 * rng.random((im, km))
    return dp


def _np_profile(kind, q1, delp, km, iv, kord, qmin=T_MIN, qs=None):
    """Run the NumPy lane's in-place profile and return the new a4."""
    a4 = _a4(q1, km)
    if kind == "scalar":
        scalar_profile_n(a4, delp, km, iv, kord, qmin, qs=qs)
    else:
        cs_profile_n(a4, delp, km, iv, kord, qs=qs)
    return a4


def _jax_profile(kind, q1, delp, km, iv, kord, qmin=T_MIN, qs=None,
                 jit=False):
    a4 = jnp.asarray(_a4(q1, km))
    if kind == "scalar":
        fn = scalar_profile_jit if jit else scalar_profile_j
        return np.asarray(fn(a4, jnp.asarray(delp), km, iv, kord, qmin,
                             qs=None if qs is None else jnp.asarray(qs)))
    fn = cs_profile_jit if jit else cs_profile_j
    return np.asarray(fn(a4, jnp.asarray(delp), km, iv, kord,
                         qs=None if qs is None else jnp.asarray(qs)))


# ====================================================================== #
# cs_limiters
# ====================================================================== #

def _limiter_cases():
    """(a4, extm) pairs that reach every arm of all three iv branches."""
    rng = np.random.default_rng(3)
    im = 24
    a4 = np.zeros((5, im), dtype=np.float64)
    a4[1] = np.concatenate([
        np.array([10.0, -2.0, 0.0, 5.0, 5.0, 5.0]),      # iv=0 arms
        4.0 + rng.standard_normal(im - 6)])
    a4[2] = a4[1] + np.concatenate([
        np.array([1.0, 1.0, 1.0, -4.0, 4.0, 0.01]),
        rng.standard_normal(im - 6)])
    a4[3] = a4[1] + np.concatenate([
        np.array([2.0, -1.0, 1.0, 4.0, -4.0, -0.01]),
        rng.standard_normal(im - 6)])
    # A6 spanning both clamp thresholds and the flat-parabola a6 == 0.
    a4[4] = np.concatenate([
        np.array([-9.0, -9.0, 0.0, -30.0, 30.0, 0.0]),
        6.0 * rng.standard_normal(im - 6)])
    extm = np.zeros(im, dtype=bool)
    extm[::3] = True
    return a4, extm


@pytest.mark.parametrize("iv", [0, 1, 2, 3])
def test_cs_limiters_jax_matches_numpy_lane(iv):
    """iv=3 is included ON PURPOSE: the oracle's ``else`` at :2747 is a
    REACHED branch (iv=2 and anything not 0/1), so the JAX twin must
    route it the same way rather than hardening it into a raise."""
    a4, extm = _limiter_cases()
    ref = np.array(a4, copy=True)
    cs_limiters_n(ref, extm, iv)
    got = np.asarray(cs_limiters_j(jnp.asarray(a4), jnp.asarray(extm), iv))
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), all iv; bound = 1e-15
    # eps guard (measured exactly 0.0).
    gate_scalar(f"cs_limiters iv={iv}", _rel(got[1:5], ref[1:5]), 1e-15)
    # Slots 0 and 1 are carried through untouched.
    assert np.array_equal(got[0], a4[0]) and np.array_equal(got[1], a4[1])
    # NON-VACUITY (not a tolerance): the limiter actually fired somewhere.
    assert not np.array_equal(ref[2:5], a4[2:5]), "limiter was a no-op"


@pytest.mark.parametrize("iv", [0, 1, 2])
def test_cs_limiters_jax_jit_matches_eager(iv):
    a4, extm = _limiter_cases()
    eager = np.asarray(cs_limiters_j(jnp.asarray(a4), jnp.asarray(extm), iv))
    jitted = np.asarray(cs_limiters_jit(jnp.asarray(a4), jnp.asarray(extm),
                                        iv))
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), all iv; bound = 1e-15
    # eps guard (measured exactly 0.0).
    gate_scalar(f"cs_limiters jit iv={iv}", _rel(jitted, eager), 1e-15)


def test_cs_limiters_jax_no_retrace_and_production_policy():
    """The trace counter rides the PRODUCTION jit factory, so a policy
    drift between ``cs_limiters_jit`` and this wrapper cannot pass."""
    a4, extm = _limiter_cases()
    traces = {"n": 0}

    def _counted(*a, **kw):
        traces["n"] += 1
        return cs_limiters_j(*a, **kw)

    fn = make_cs_limiters_jit(_counted)
    fn(jnp.asarray(a4), jnp.asarray(extm), 1)
    fn(jnp.asarray(a4 * 1.01), jnp.asarray(extm), 1)   # same shapes/dtypes
    assert traces["n"] == 1, traces["n"]


def test_cs_limiters_jax_rejects_float32():
    a4, extm = _limiter_cases()
    with pytest.raises(TypeError, match="float64"):
        cs_limiters_j(jnp.asarray(a4, dtype=jnp.float32),
                      jnp.asarray(extm), 1)
    # NON-VACUITY: the f64 operand does NOT raise.
    cs_limiters_j(jnp.asarray(a4), jnp.asarray(extm), 1)


def test_cs_limiters_jax_iv0_zero_a6_is_inert():
    """The iv=0 divisor sanitisation must be VALUE-NEUTRAL.

    ``a6 == 0`` makes ``0.25*(AR-AL)**2/a6`` infinite in the NumPy lane
    (it runs under ``errstate(divide=ignore)``) and finite in the JAX
    lane (the divisor is guarded).  Both are inert because ``neg`` also
    requires ``abs(AR-AL) < -a6``, i.e. ``x < 0`` for a non-negative x,
    which is False at a6 == 0.  This pins that reasoning to a test.
    """
    a4 = np.zeros((5, 3), dtype=np.float64)
    a4[1] = np.array([5.0, 5.0, -1.0])
    a4[2] = np.array([4.0, 6.0, 2.0])
    a4[3] = np.array([6.0, 4.0, 2.0])
    a4[4] = np.zeros(3)                       # the sanitised case
    ref = np.array(a4, copy=True)
    cs_limiters_n(ref, np.zeros(3, dtype=bool), 0)
    got = np.asarray(cs_limiters_j(jnp.asarray(a4),
                                   jnp.zeros(3, dtype=bool), 0))
    assert np.array_equal(got[1:5], ref[1:5]), (got[1:5], ref[1:5])
    assert np.isfinite(got).all()


def test_cs_limiters_jax_check_grads_order2_off_switch():
    """Order-2 fwd+rev grads with EVERY selection proven off its switch.

    NON-SMOOTH SITES here (named, not silently avoided): the iv=1
    extremum test ``(qbar-AL)*(qbar-AR) >= 0`` and the two clamp tests
    ``a6*(AR-AL) <> +-(AR-AL)^2``.  The fixture below keeps the extremum
    product strictly negative (AL and AR straddle qbar) and ``|a6*da1|``
    strictly under ``da1^2``, both with a margin far beyond any finite-
    difference step, so the function is locally smooth.
    """
    im = 6
    q = 10.0 + np.arange(im, dtype=np.float64)
    a4 = np.zeros((5, im), dtype=np.float64)
    a4[1] = q
    a4[2] = q - 1.0                      # AL < qbar < AR: product = -1
    a4[3] = q + 1.0
    a4[4] = 0.2                          # |a6*da1| = 0.4 << da1^2 = 4
    da1 = a4[3] - a4[2]
    assert ((a4[1] - a4[2]) * (a4[1] - a4[3]) < -0.5).all()
    assert (np.abs(a4[4] * da1) < 0.5 * da1 ** 2).all()

    def f(x):
        out = cs_limiters_j(x, jnp.zeros(im, dtype=bool), 1)
        return jnp.sum(out[2] ** 2 + out[3] ** 2 + out[4] ** 2)

    check_grads(f, (jnp.asarray(a4),), order=2, modes=("fwd", "rev"))

    # The iv=0 arm on a strictly positive column, floor inactive.
    assert (a4[1] > 1.0).all()

    def f0(x):
        out = cs_limiters_j(x, jnp.zeros(im, dtype=bool), 0)
        return jnp.sum(out[2] ** 2 + out[3] ** 2 + out[4] ** 2)

    check_grads(f0, (jnp.asarray(a4),), order=2, modes=("fwd", "rev"))


# ====================================================================== #
# scalar_profile / cs_profile -- EVERY kord, EVERY iv
# ====================================================================== #

KMP = 10          # interior loop k = 3..8: the vectorised block is real


_Q_BASE, _Q_SPAN, _Q_CURV = 250.0, 90.0, 0.3


def _quadratic_on(x_edge):
    """``(f, cell_means)`` for a gentle monotone quadratic ON ``x_edge``.

    ``x_edge`` is 0-based ``(im, nk+1)``.  ``f(x)`` evaluates the
    function; ``cell_means`` is its EXACT mean over each cell, i.e. what
    a finite-volume column of that function is.

    WHY EXACT CELL MEANS OF A SMOOTH FUNCTION OF THE COORDINATE, and not
    a polynomial in the LAYER INDEX -- which is what the first version of
    this fixture was, and is why six gradient gates failed in job
    9400424.  PPM reconstructs a sub-grid profile in the COORDINATE.  A
    column that is smooth in the INDEX but sampled on non-uniform
    thicknesses is not smooth in the coordinate: its interface values
    legitimately overshoot the neighbouring cell means, the large-scale
    min/max constraints at fv_mapz.F90:1949-1983 clamp them back, and
    the fixture then sits exactly ON a switching surface -- the one
    place a finite-difference gradient check means nothing.  Building
    the column from the coordinate removes the cause instead of loosening
    the assertion.

    The curvature is set as a FRACTION of the linear term, which makes
    ``|A6| / |AR-AL| ~ _Q_CURV * h / span`` -- i.e. every ``|A6|``-vs-
    ``|AR-AL|`` switch (the +-da2 clamps, the kord-9 ``fix`` test,
    ``ext5``, ``ext6``) is off by construction, and
    :func:`_assert_profile_off_switch` still MEASURES it rather than
    trusting this comment.
    """
    x_edge = np.asarray(x_edge, dtype=np.float64)
    x0 = x_edge[..., :1]
    u_max = x_edge[..., -1:] - x0
    b = _Q_SPAN / u_max
    c = _Q_CURV * b / u_max

    def f(x):
        u = np.asarray(x, dtype=np.float64) - x0[..., 0]
        return _Q_BASE + b[..., 0] * u + c[..., 0] * u ** 2

    ul = x_edge[..., :-1] - x0
    ur = x_edge[..., 1:] - x0
    means = (_Q_BASE + b * (ul + ur) / 2.0
             + c * (ul ** 2 + ul * ur + ur ** 2) / 3.0)
    return f, means


def _edges_from_delp(delp):
    """1-based ``(im, km+1)`` thicknesses -> 0-based ``(im, km+1)`` edges."""
    return np.concatenate(
        [np.zeros((delp.shape[0], 1)), np.cumsum(delp[:, 1:], axis=1)],
        axis=1)


def _smooth_column(im=4, km=KMP, delp=None, edges=None):
    """1-based ``(im, km+1)`` cell means of :func:`_quadratic_on`.

    Give it EITHER the thicknesses the profile will be built on (the
    pure-profile tests) or the edge array itself (the map tests, where
    the coordinate is ``pe1``).  Passing neither is a bug, not a default.
    """
    if edges is None:
        if delp is None:
            raise ValueError("_smooth_column needs delp or edges: the "
                             "column must be smooth in the COORDINATE "
                             "the profile is built on")
        edges = _edges_from_delp(delp)
    _, means = _quadratic_on(edges)
    q = np.zeros((im, km + 1), dtype=np.float64)
    q[:, 1:] = means
    return q


def _smooth_column_bottom_value(delp=None, edges=None):
    """``f`` at the BOTTOM interface -- the consistent iv=-2 ``qs`` BC.

    ``qs = q(km) + 9`` (the first version) is NOT a value the column's
    own reconstruction would take there, so it forced an overshoot the
    large-scale constraints then clamped -- the fixture defect behind
    ``test_profile_jax_check_grads_order2_iv_minus2_bc``.
    """
    if edges is None:
        edges = _edges_from_delp(delp)
    f, _ = _quadratic_on(edges)
    return f(np.asarray(edges)[:, -1])


def _cold_column(im=3, km=KMP):
    """A column that dips BELOW ``T_MIN`` -- the qmin clause only exists
    in ``scalar_profile`` (fv_mapz.F90:2081/:2183/:2225)."""
    q = np.zeros((im, km + 1), dtype=np.float64)
    base = np.array([300.0, 250.0, 170.0, 160.0, 210.0, 280.0, 300.0,
                     150.0, 240.0, 260.0])
    for i in range(im):
        q[i, 1:] = base[:km] + 3.0 * i
    return q


# --- switch-margin factors, calibrated against job 9400424 -------------
# MEASURED there: the worst |A6*da1| / da1^2 was 0.70, at the TOP layer
# where the one-sided cubic extrapolation is steepest.  The clamp switch
# is at 1.0, so that state was genuinely OFF the switch and the original
# 0.5 factor was an arbitrary over-tightening -- six gradient gates
# failed on the FACTOR, not on the physics.  0.95 leaves 5 % headroom,
# which is ~5e4 times the ~1e-6 relative step check_grads takes.
_CLAMP_MARGIN = 0.95
# The flat arms set AL = AR = qbar BITWISE, so proving "no flat arm
# fired" only has to clear float noise.
_FLAT_EPS = 1e-9
# Margins on predicates a finite-difference step must not cross.  Kept
# RELATIVE: an ABSOLUTE 0.25 K was the second reason the first version
# failed -- |AL - qbar| legitimately ranges from 0.3 to 9 K across the
# columns of one fixture.
_PRED_EPS = 1e-4


def _assert_profile_off_switch(a4, q1, km):
    """MEASURE that every limiter switch is off, on the OUTPUT a4.

    Every clause reports the margin it measured, so a future failure
    says HOW CLOSE the fixture was rather than only that it failed.

    Each clause is decisive because of how the oracle's branches land:

    (a) a flattened layer has ``AL = AR = qbar`` EXACTLY, so a margin on
        those two differences proves no ``flat`` arm fired (that covers
        ``extm``-driven flattening, the iv=1 extremum test and the qmin
        clauses).  ``A6 == 0`` is NOT asserted: a nearly linear column
        legitimately has a small A6, so that clause would fail on
        correct code while adding nothing -- ``AL != qbar`` already
        excludes every ``flat`` arm;
    (b) after a ``+-da2`` clamp fires, ``A6*(AR-AL) == -(AR-AL)^2``
        EXACTLY -- so the RATIO ``|A6*da1| / da1^2`` IS the distance to
        that switch, and staying under ``_CLAMP_MARGIN`` proves neither
        clamp fired.  The same ratio bounds ``|A6|`` against ``|AR-AL|``,
        which also puts the kord-9 ``fix`` test, ``ext5``
        (``|x0| > x1``) and ``ext6`` (``|A6| > x1``) off their switches;
    (c) the INPUT has no interior extremum, with margin -> ``extm`` is
        False in the interior and the ``smooth`` predicate of the
        large-scale constraints is strictly satisfied;
    (d) each interface value lies STRICTLY inside its two neighbouring
        cell means -> the ``min``/``max`` clamps at :1949-1983 are
        inactive with margin;
    (e) ``(AL-qbar)*(AR-qbar) < 0`` with margin -> the k=1/k=km ``extm``
        edge test and the iv=1 flat test are off their switch.
    """
    a1, a2, a3 = a4[1][:, 1:km + 1], a4[2][:, 1:km + 1], a4[3][:, 1:km + 1]
    a6 = a4[4][:, 1:km + 1]
    da1 = a3 - a2
    scale = np.abs(a1).max()

    # (a) the flat arms set AL = AR = qbar BITWISE, so this only has to
    #     clear float noise.
    m_al = np.abs(a2 - a1).min() / scale
    m_ar = np.abs(a3 - a1).min() / scale
    assert m_al > _FLAT_EPS, f"a layer flattened (AL); margin {m_al:.3e}"
    assert m_ar > _FLAT_EPS, f"a layer flattened (AR); margin {m_ar:.3e}"
    assert (np.abs(da1) / scale > _FLAT_EPS).all()
    # (b) a fired clamp leaves |A6*da1| == da1**2 EXACTLY, so the ratio
    #     IS the distance to that switch -- and to the kord-9 `fix` test,
    #     ext5 and ext6, which are all |A6| vs |AR-AL| in disguise.
    ratio = float(np.max(np.abs(a6 * da1) / da1 ** 2))
    assert ratio < _CLAMP_MARGIN, \
        f"clamp/ext switch margin: |A6*da1|/da1^2 = {ratio:.3f} (1.0 is ON)"
    # (c) the INPUT is strictly monotone -> extm off in the interior and
    #     the `smooth` predicate strictly satisfied.
    g = np.diff(q1[:, 1:km + 1], axis=1)
    assert (g[:, :-1] * g[:, 1:] > 0.0).all(), "input has an extremum"
    m_g = np.abs(g).min() / np.abs(g).max()
    assert m_g > _PRED_EPS, f"input increment margin {m_g:.3e}"
    # (d) each interface strictly inside its two neighbouring cell means
    #     -> the :1949-1983 min/max clamps inactive with margin.
    lo = np.minimum(a1[:, :-1], a1[:, 1:])
    hi = np.maximum(a1[:, :-1], a1[:, 1:])
    room = np.minimum(a2[:, 1:] - lo, hi - a2[:, 1:]) / (hi - lo)
    assert room.min() > _PRED_EPS, \
        f"interface clamp margin {room.min():.3e} of the local increment"
    # (e) the k=1/k=km edge extm test and the iv=1 flat test.
    prod = (a2 - a1) * (a3 - a1)
    assert (prod < 0.0).all(), "edge extm switch: AL/AR do not straddle qbar"
    m_e = np.minimum(np.abs(a2 - a1), np.abs(a3 - a1)).min() / scale
    assert m_e > _PRED_EPS, f"edge extm margin {m_e:.3e}"
    # (f) ext5, ext6 and the kord-9 `fix` test, REPORTED SEPARATELY.
    #     All three are ``|A6| vs |AR-AL|`` in disguise -- ext6 and `fix`
    #     ARE clause (b)'s ratio, and ext5 is that ratio over 3 -- so (b)
    #     already subsumes them; naming them anyway means a failure says
    #     WHICH surface instead of leaving "some other switch" as a live
    #     hypothesis (it was one of the three candidates after job
    #     9401521).  Valid on the OUTPUT edges precisely because (a),
    #     (b), (d) and (e) certify that no limiter modified them.
    x0 = 2.0 * a1 - (a2 + a3)
    x1 = np.abs(a2 - a3)
    r_ext5 = float(np.max(np.abs(x0) / x1))
    r_ext6 = float(np.max(np.abs(a6) / x1))
    assert r_ext5 < _CLAMP_MARGIN, f"ext5 margin {r_ext5:.3f} (1.0 is ON)"
    assert r_ext6 < _CLAMP_MARGIN, \
        f"ext6 / kord-9 fix margin {r_ext6:.3f} (1.0 is ON)"


@pytest.mark.parametrize("kord", ALL_KORD)
@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_matches_numpy_lane_every_kord(kind, kord):
    """EVERY kord arm of BOTH routines, on a column that reaches the
    limiters (not the smooth one -- a fixture where no branch fires
    would make the kord sweep vacuous)."""
    im, km = 4, KMP
    q1 = _column(im=im, km=km, amp=8.0)
    delp = _delp_col(im, km)
    ref = _np_profile(kind, q1, delp, km, 1, kord)
    got = _jax_profile(kind, q1, delp, km, 1, kord)
    # MEASURED (job 9425294 sweep): worst 1.940e-15 (both kinds, kord=17);
    # bound = measured x 10 = 2.0e-14.
    gate_scalar(f"profile {kind} kord={kord}", _rel(got[1:5], ref[1:5]),
                2.0e-14)
    # NON-VACUITY: the profile is not the trivial AL = AR = qbar.
    assert np.abs(ref[2] - ref[1]).max() > 1e-6


@pytest.mark.parametrize("kord", ALL_KORD)
@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_matches_numpy_lane_cold_column(kind, kord):
    """The COLD column: below ``T_MIN`` the qmin clauses (kord 9, 15 and
    the trailing 11 arm) exist ONLY in ``scalar_profile``, so this is
    the fixture that separates the two routines."""
    im, km = 3, KMP
    q1 = _cold_column(im=im, km=km)
    delp = _delp_col(im, km, seed=11)
    ref = _np_profile(kind, q1, delp, km, 1, kord)
    got = _jax_profile(kind, q1, delp, km, 1, kord)
    # MEASURED (job 9425294 sweep): worst 5.235e-15 (both kinds, most kords);
    # bound = measured x 10 = 5.3e-14.
    gate_scalar(f"profile cold {kind} kord={kord}",
                _rel(got[1:5], ref[1:5]), 5.3e-14)


@pytest.mark.parametrize("kord", (9, 15, 11))
def test_profile_jax_keeps_the_scalar_vs_cs_qmin_delta(kord):
    """The two routines must NOT agree on a sub-qmin column.

    Non-vacuity for the sweep above: if the JAX twin had shared one body
    without the qmin switch, every parity test would still pass while
    ``scalar_profile`` silently became ``cs_profile``.
    """
    im, km = 3, KMP
    q1 = _cold_column(im=im, km=km)
    delp = _delp_col(im, km, seed=11)
    a_s = _jax_profile("scalar", q1, delp, km, 1, kord)
    a_c = _jax_profile("cs", q1, delp, km, 1, kord)
    assert not np.allclose(a_s[1:5], a_c[1:5]), \
        f"kord={kord}: the JAX scalar/cs bodies did not separate"
    # ...and each one still tracks ITS OWN NumPy twin (so the difference
    # is the oracle's, not a defect in one of them).
    for kind, got in (("scalar", a_s), ("cs", a_c)):
        ref = _np_profile(kind, q1, delp, km, 1, kord)
        # MEASURED (job 9425294 sweep): worst 5.235e-15 (both kinds, every kord);
        # bound = measured x 10 = 5.3e-14.
        gate_scalar(f"profile qmin-delta {kind} kord={kord}",
                    _rel(got[1:5], ref[1:5]), 5.3e-14)


def test_profile_jax_kord15_nesting_difference():
    """cs_profile nests kord=15 differently from scalar_profile.

    ``scalar_profile`` (:2176-2195) is a FLAT elseif chain, so
    ``elseif (ext6)`` is reachable when ext5(k) is set and no neighbour
    is; ``cs_profile`` (:2607-2623) wraps the ext5 tests INSIDE
    ``if (ext5(k))``, which SWALLOWS the ext6 arm.  Column from the
    NumPy lane's own regression test -- it is the one that separates the
    two nestings.
    """
    delp = np.zeros((1, KM + 1), dtype=np.float64)
    delp[0, 1:] = [0.24887241281639058, 0.1673993544216956,
                   1.7437252316008056, 0.16335454668312105,
                   2.384564998820821]
    q1 = np.zeros((1, KM + 1), dtype=np.float64)
    q1[0, 1:] = [-4.566530047187453, -0.061160100798463395,
                 4.344253573583179, 1.950142859220142,
                 -0.7382462074722012]
    a_s = _jax_profile("scalar", q1, delp, KM, 1, 15, qmin=-1e30)
    a_c = _jax_profile("cs", q1, delp, KM, 1, 15)
    assert not np.allclose(a_s[2:5, 0, 3], a_c[2:5, 0, 3]), \
        "the two kord=15 nestings gave the same answer -- one is wrong"
    # cs leaves k=3 untouched (ext5(3) set, neither neighbour set).
    assert not np.allclose(a_c[2, 0, 3], a_c[1, 0, 3])
    assert abs(a_c[4, 0, 3]) > 1.0
    for kind, got, qm in (("scalar", a_s, -1e30), ("cs", a_c, None)):
        ref = _np_profile(kind, q1, delp, KM, 1, 15,
                          qmin=(qm if kind == "scalar" else T_MIN))
        # MEASURED (job 9425294 sweep): worst 5.785e-16 (scalar); bound = measured
        # x 10 = 5.8e-15.
        gate_scalar(f"profile kord15-nesting {kind}",
                    _rel(got[1:5], ref[1:5]), 5.8e-15)


def test_profile_jax_kord12_uses_one_a6_grouping_and_kord9_two():
    """``6.*q - 3.*(AL+AR)`` at kord 12 in BOTH routines (:2141/:2571),
    and the two DIFFERENT groupings at kord 9 (:2077 vs :2509).

    Checked BITWISE: the groupings are algebraically identical, so an
    ``allclose`` would pass with the wrong one.
    """
    im, km = 4, KMP
    # The SMOOTH column: with no layer flattened (``flat`` sets A6 to a
    # literal 0 in BOTH routines) the two groupings are actually
    # exercised, so the bitwise assertions below are not vacuous.
    delp = _delp_col(im, km)
    q1 = _smooth_column(im=im, km=km, delp=delp)
    a_s12 = _jax_profile("scalar", q1, delp, km, 1, 12, qmin=-1e30)
    a_c12 = _jax_profile("cs", q1, delp, km, 1, 12)
    assert a_s12[4].tobytes() == a_c12[4].tobytes(), \
        "kord=12 A6 differs between the routines -- one used 3.*(2.*q-...)"
    a_s9 = _jax_profile("scalar", q1, delp, km, 1, 9, qmin=-1e30)
    a_c9 = _jax_profile("cs", q1, delp, km, 1, 9)
    assert a_s9[4].tobytes() != a_c9[4].tobytes(), \
        "kord=9 A6 is bitwise equal in both routines -- one grouping was lost"
    # MEASURED (job 9425294 sweep): 2.427e-13 (the two groupings really do
    # round differently); bound = measured x 10 = 2.5e-12.
    gate_scalar("profile kord9 a6-groupings", _rel(a_s9[4], a_c9[4]),
                2.5e-12)   # ...but equal in value


@pytest.mark.parametrize("iv", [-1, 0, 1, 2])
@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_matches_numpy_lane_every_iv(kind, iv):
    """Every ``iv`` the default edge solve admits.  iv=0's positivity
    arm needs a column that can undershoot, so this one straddles 0."""
    im, km = 4, KMP
    q1 = _column(im=im, km=km, amp=8.0) - 250.0
    delp = _delp_col(im, km, seed=23)
    ref = _np_profile(kind, q1, delp, km, iv, 9, qmin=0.0)
    got = _jax_profile(kind, q1, delp, km, iv, 9, qmin=0.0)
    # MEASURED (job 9425294 sweep): worst 3.553e-17; bound = measured x 10 =
    # 3.6e-16.
    gate_scalar(f"profile {kind} iv={iv}", _rel(got[1:5], ref[1:5]),
                3.6e-16)


@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_iv_minus2_bottom_bc(kind):
    """iv=-2 selects the OTHER tridiagonal (:1877-1900), whose bottom
    row is the BC ``qs`` -- the w lane's edge solve."""
    im, km = 4, KMP
    q1 = _column(im=im, km=km, amp=6.0) - 248.0
    delp = _delp_col(im, km, seed=31)
    qs = np.array([0.5, -0.25, 1.5, 0.0])
    ref = _np_profile(kind, q1, delp, km, -2, 9, qs=qs)
    got = _jax_profile(kind, q1, delp, km, -2, 9, qs=qs)
    # MEASURED (job 9425294 sweep): worst 1.433e-17 (cs); bound = measured x
    # 10 = 1.5e-16.
    gate_scalar(f"profile {kind} iv=-2 qs-bc", _rel(got[1:5], ref[1:5]),
                1.5e-16)
    # NON-VACUITY: the BC actually moved the answer.
    other = _jax_profile(kind, q1, delp, km, -2, 9, qs=qs + 3.0)
    assert np.abs(other[1:5] - got[1:5]).max() > 1e-6


@pytest.mark.parametrize("kord", ALL_KORD)
@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_jit_matches_eager(kind, kord):
    im, km = 4, KMP
    q1 = _column(im=im, km=km, amp=8.0)
    delp = _delp_col(im, km)
    eager = _jax_profile(kind, q1, delp, km, 1, kord)
    jitted = _jax_profile(kind, q1, delp, km, 1, kord, jit=True)
    # MEASURED (job 9425294 sweep): 3.233e-15 (every kind/kord); bound =
    # measured x 10 = 3.3e-14.
    gate_scalar(f"profile jit {kind} kord={kord}", _rel(jitted, eager),
                3.3e-14)


@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_no_retrace(kind):
    im, km = 4, KMP
    q1 = _column(im=im, km=km, amp=8.0)
    delp = _delp_col(im, km)
    traces = {"n": 0}
    base = scalar_profile_j if kind == "scalar" else cs_profile_j

    def _counted(*a, **kw):
        traces["n"] += 1
        return base(*a, **kw)

    fn = (make_scalar_profile_jit(_counted) if kind == "scalar"
          else make_cs_profile_jit(_counted))
    for scale in (1.0, 1.01):
        a4 = jnp.asarray(_a4(q1 * scale, km))
        if kind == "scalar":
            fn(a4, jnp.asarray(delp), km, 1, 9, T_MIN, qs=None)
        else:
            fn(a4, jnp.asarray(delp), km, 1, 9, qs=None)
    assert traces["n"] == 1, traces["n"]


# ------------------------------------------------------------- guards
@pytest.mark.parametrize("bad", ["a4", "delp", "qs"])
def test_profile_jax_rejects_float32_every_operand(bad):
    im, km = 4, KMP
    args = {"a4": _a4(_column(im=im, km=km), km),
            "delp": _delp_col(im, km),
            "qs": np.zeros(im)}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        cs_profile_j(jnp.asarray(args["a4"]), jnp.asarray(args["delp"]),
                     km, -2, 9, qs=jnp.asarray(args["qs"]))
    # NON-VACUITY: all-f64 does not raise.
    cs_profile_j(jnp.asarray(args["a4"].astype(np.float64)),
                 jnp.asarray(args["delp"].astype(np.float64)), km, -2, 9,
                 qs=jnp.asarray(args["qs"].astype(np.float64)))


@pytest.mark.parametrize("kord", [7, 4, 0, -9])
def test_profile_jax_low_kord_is_refused_not_rerouted(kord):
    """``kord <= 7`` selects ``ppm_profile`` (:1406/:1500/:1708), which
    neither lane ports.  A negative kord is refused too -- the guard
    tests the SIGNED value, exactly as the NumPy lane does."""
    im, km = 4, KMP
    q1, delp = _column(im=im, km=km), _delp_col(im, km)
    with pytest.raises(ValueError, match="ppm_profile"):
        _jax_profile("cs", q1, delp, km, 1, kord)
    with pytest.raises(ValueError, match="ppm_profile"):
        _jax_profile("scalar", q1, delp, km, 1, kord)
    # NON-VACUITY: the pinned lane's kord passes.
    _jax_profile("cs", q1, delp, km, 1, 9)


def test_profile_jax_iv_minus3_is_refused_rather_than_invented():
    """``fv_mapz.F90:2320-2339`` leaves ``gam(i,km)`` UNASSIGNED and
    :2358 reads it.  ``cs_profile`` must refuse; ``scalar_profile`` has
    no -3 arm at all (:1877 tests only -2) and must fall through to the
    default solve."""
    im, km = 3, KMP
    q1, delp = _column(im=im, km=km), _delp_col(im, km)
    with pytest.raises(NotImplementedError, match="iv=-3"):
        _jax_profile("cs", q1, delp, km, -3, 9, qs=np.zeros(im))
    ref = _jax_profile("scalar", q1, delp, km, 1, 9)
    got = _jax_profile("scalar", q1, delp, km, -3, 9)   # must NOT raise
    assert np.array_equal(ref[1:5], got[1:5])


def test_profile_jax_refuses_a_km_its_layer_blocks_would_overlap():
    with pytest.raises(ValueError, match="km=3"):
        _jax_profile("scalar", np.ones((2, 4)), np.ones((2, 4)), 3, 1, 9)
    # NON-VACUITY: km=4 is admitted (the interior loop is simply empty).
    _jax_profile("scalar", _column(im=2, km=4), _delp_col(2, 4), 4, 1, 9)


def test_profile_jax_iv_minus2_without_qs_raises():
    im, km = 3, KMP
    with pytest.raises(ValueError, match="qs"):
        _jax_profile("cs", _column(im=im, km=km), _delp_col(im, km), km,
                     -2, 9)


def test_scalar_profile_jax_requires_qmin():
    im, km = 3, KMP
    with pytest.raises(ValueError, match="qmin"):
        scalar_profile_j(jnp.asarray(_a4(_column(im=im, km=km), km)),
                         jnp.asarray(_delp_col(im, km)), km, 1, 9, None)


# ------------------------------------------------------------- gate 4
@pytest.mark.parametrize("kord", [9, 13])
@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_check_grads_order2_off_switch(kind, kord):
    """Order-2 fwd+rev grads with EVERY limiter switch MEASURED off.

    NON-SMOOTH SITES (named): the ``extm``/``ext5``/``ext6`` selections,
    the ``+-da2`` clamps, the kord-9 ``fix`` test, Huynh's nested
    min/max, and the interface min/max clamps of the large-scale
    constraints.  :func:`_assert_profile_off_switch` measures a margin
    on each from the OUTPUT before the gradient is taken; kord 13 is
    included so an ``ext5``/``ext6`` arm is in the trace.
    """
    im, km = 3, KMP
    delp = _delp_col(im, km, seed=41)
    q1 = _smooth_column(im=im, km=km, delp=delp)
    _assert_profile_off_switch(_jax_profile(kind, q1, delp, km, 1, kord),
                               q1, km)

    def f(a1_, delp_):
        a4 = jnp.zeros((5, im, km + 1), jnp.float64).at[1].set(a1_)
        if kind == "scalar":
            out = scalar_profile_j(a4, delp_, km, 1, kord, T_MIN)
        else:
            out = cs_profile_j(a4, delp_, km, 1, kord)
        return jnp.sum(out[2] ** 2) + jnp.sum(out[3] ** 2) \
            + jnp.sum(out[4] ** 2)

    # ARM (i) -- w.r.t. q1 ALONE, at the DEFAULT step.  At a state where
    # no limiter fires, a4(2:4) are LINEAR in q1 for fixed delp (the
    # tridiagonal RHS is linear in q1, every clamp is the identity, and
    # A6 = 3*(2q - AL - AR) is linear), so this loss is exactly QUADRATIC
    # in q1: every derivative above the second vanishes and the central
    # difference is EXACT up to roundoff.  This arm therefore needs no
    # step tuning, and a failure HERE is a real gradient defect rather
    # than a finite-difference artifact.  That is the permanent
    # separation between cause (a) and cause (c).
    check_grads(lambda a1_: f(a1_, jnp.asarray(delp)),
                (jnp.asarray(q1),), order=2, modes=("fwd", "rev"))

    # ARM (ii) -- both operands, at a SMALLER step.  delp enters the
    # tridiagonal coefficients RATIONALLY, so f''' != 0 along a delp
    # perturbation and jax's central difference carries a truncation
    # error ~ (eps^2/6)*f'''.  MEASURED (job 9401521, at the default
    # EPS = 1e-4, public_test_util.py:33): the JVP tangent is -105.452185
    # against a numerical -105.453734, i.e. 1.469e-5 RELATIVE, which
    # misses the 1e-5 rtol by 1.47x.
    #
    # NOTE, because it is counter-intuitive: normalising this loss to
    # O(1) does NOT fix that.  The failing comparison is RELATIVE, and
    # scaling the loss scales the AD tangent and the finite difference
    # identically, leaving 1.469e-5 exactly where it was.  The levers are
    # the step size and the function's own nonlinearity, not its scale.
    #
    # Budget at eps=1e-5, with |f| ~ 5.4e6 and |f'| = 105.45 measured:
    #   truncation ~ 1.469e-5 * (1e-5/1e-4)^2 = 1.5e-7   (falls as eps^2)
    #   roundoff   ~ eps_mach*|f| / (2*eps*|f'|) = 5.6e-7 (rises as 1/eps)
    #   total      ~ 7e-7, i.e. ~14x under the 1e-5 rtol.
    # eps=1e-6 would be WORSE (roundoff alone 5.6e-6), so 1e-5 is near
    # the optimum rather than "smaller is safer".
    check_grads(f, (jnp.asarray(q1), jnp.asarray(delp)), order=2,
                modes=("fwd", "rev"), eps=1.0e-5)


def test_profile_jax_check_grads_order2_iv_minus2_bc():
    """The iv=-2 tridiagonal, including the ``qs`` bottom BC -- the only
    operand the default solve does not have.

    Split into the same two arms as the interior gate, for the same
    reason and with the same discriminating power: ``q1`` AND ``qs``
    both enter the tridiagonal RHS linearly, so arm (i) is exactly
    quadratic and needs no step tuning, while ``delp`` enters the
    coefficients rationally and needs the smaller step.  This BC arm is
    a different code path from the interior one
    (``_edge_solve_iv_m2`` vs ``_edge_solve_default``), and it failed
    job 9401521 with the SAME 1.469e-5 relative residual -- which is
    itself evidence for the finite-difference explanation, since a
    defect in one path would not reproduce the other's number.

    ⛔ THE OFF-SWITCH PRECONDITION IS NOT ACHIEVABLE ON THIS PATH, AND
    THAT WAS MEASURED RATHER THAN ASSUMED.  This gate failed at
    ``assert np.float64(0.0) > 1e-09`` -- ``|AL - qbar| == 0`` exactly,
    i.e. a ``flat`` arm HAD fired -- and the docstring above blamed the
    ``qs`` bottom BC for forcing an overshoot at the last layer.  Job
    9417326 (``scripts/validate/fv3_gradient_gate_triage.py``, block P1)
    printed the exact-zero layers per column::

        col 0: [1, 2, 4, 5, 7, 9]
        col 1: [2, 4, 5, 7, 8, 9]
        col 2: [2, 4, 5, 7, 8, 10]

    Six of ten layers in EVERY column, spread through the interior --
    not the bottom layer, and not the BC's doing.  At ``kord = 9`` the
    large-scale constraints flatten wherever the reconstruction would
    overshoot, and a ten-layer column with real structure does that
    almost everywhere.  So the hypothesis in the old docstring is
    RETRACTED: constructing a provably off-switch state for this
    limiter chain is the thing that is hard (STATE lesson 12), and it
    is hard here for a reason that has nothing to do with ``qs``.

    The gate is therefore the TOLERANCE-FREE ADJOINT IDENTITY, which is
    the strategy's primary gradient gate precisely because it needs no
    off-switch fixture -- it runs ON the switching surface, where a
    finite difference is meaningless and ``check_grads`` is certifying a
    jump.

    ⛔ BUT THE IDENTITY ALONE IS NOT ENOUGH, AND THE FIRST VERSION OF
    THIS REPLACEMENT LEFT ``delp`` WITH NOTHING ELSE (codex BLOCKER,
    job 9417397).  ``jvp`` and ``vjp`` are two transformations of the
    SAME program, so ``<Jv,w> == <v,J^T w>`` holds for whatever Jacobian
    that program has -- including one with a missing term, a wrong
    coefficient or a stray ``stop_gradient``.  It certifies AD
    self-consistency, not correctness.  What certifies correctness here
    is (i) the PARITY gates against the NumPy lane, which is the
    authority for the map itself, and (ii) a finite difference, which is
    the only instrument that compares the derivative against the
    function.  So ``order=1`` runs over ALL THREE operands, including
    ``delp``, not just the two the solve is linear in.
    """
    im, km = 3, KMP
    delp = _delp_col(im, km, seed=43)
    q1 = _smooth_column(im=im, km=km, delp=delp)
    qs = _smooth_column_bottom_value(delp=delp)

    def f(a1_, delp_, qs_):
        a4 = jnp.zeros((5, im, km + 1), jnp.float64).at[1].set(a1_)
        out = cs_profile_j(a4, delp_, km, -2, 9, qs=qs_)
        return jnp.sum(out[2] ** 2) + jnp.sum(out[3] ** 2) \
            + jnp.sum(out[4] ** 2)

    # PRIMARY: <J v, w> == <v, J^T w>, both operand groups, on the
    # switching surface.  No FD step, no off-switch fixture.
    check_adjoint("cs_profile iv=-2 q1/qs",
                  lambda a1_, qs_: f(a1_, jnp.asarray(delp), qs_),
                  (jnp.asarray(q1), jnp.asarray(qs)), 1.0e-12)
    check_adjoint("cs_profile iv=-2 q1/delp/qs", f,
                  (jnp.asarray(q1), jnp.asarray(delp), jnp.asarray(qs)),
                  1.0e-12)
    # THE INDEPENDENT CHECK, order 1 only: at order 2 an FD-resolution
    # failure and a wrong Jacobian are indistinguishable (STATE lesson
    # 12), but at order 1 a finite difference is still the only gate
    # here that compares the derivative against the FUNCTION rather than
    # against another transformation of the same program.  It runs over
    # all three operands -- `delp` included, which is the one the first
    # version of this replacement left uncovered.
    check_grads(lambda a1_, qs_: f(a1_, jnp.asarray(delp), qs_),
                (jnp.asarray(q1), jnp.asarray(qs)), order=1,
                modes=("fwd", "rev"))
    # `delp` enters the tridiagonal COEFFICIENTS rationally, so its FD
    # needs the smaller step the interior gate also uses; the default
    # 1e-4 on a 1e2-scale thickness is a 1e-6 relative step and lands in
    # the roundoff regime.
    check_grads(f, (jnp.asarray(q1), jnp.asarray(delp), jnp.asarray(qs)),
                order=1, modes=("fwd", "rev"), eps=1.0e-5)


# ====================================================================== #
# map_scalar / map1_ppm / map1_q2 -- the rezone
# ====================================================================== #

def _dp_1based(pe):
    """1-based thicknesses of a 1-based edge array (a first difference,
    not model numerics)."""
    out = np.zeros((pe.shape[0], pe.shape[1] - 1), dtype=np.float64)
    out[:, 1:] = pe[:, 2:] - pe[:, 1:-1]
    return out


def _midpoint_targets(pe1, km):
    """A target grid whose INTERIOR edges sit at the source CELL MIDPOINTS.

    Maximally far from every source interface, so the interval selection
    is locally constant -- the fixture the coordinate gradients need.
    The two ENDPOINTS are still copied from the source (that is what the
    driver does at :297-300 and what conservation requires).
    """
    mid = 0.5 * (pe1[:, 1:km + 1] + pe1[:, 2:km + 2])       # cells 1..km
    pe2 = np.zeros_like(pe1)
    pe2[:, 1] = pe1[:, 1]
    pe2[:, 2:km + 1] = mid[:, :km - 1]
    pe2[:, km + 1] = pe1[:, km + 1]
    return pe2


def _run_map(kind, pe1, q1, pe2, km, kn, iv, kord, dp2=None, jit=False,
             lane="jax"):
    """One driver for all three map routines and both lanes."""
    if lane == "np":
        if kind == "map_scalar":
            return map_scalar_n(pe1, q1, pe2, km, kn, iv, kord, T_MIN)
        if kind == "map1_ppm":
            return map1_ppm_n(pe1, q1, pe2, km, kn, iv, kord)
        return map1_q2_n(pe1, q1, pe2, dp2, km, kn, iv, kord, 0.0)
    a = (jnp.asarray(pe1), jnp.asarray(q1), jnp.asarray(pe2))
    if kind == "map_scalar":
        fn = map_scalar_jit if jit else map_scalar_j
        return np.asarray(fn(*a, km, kn, iv, kord, T_MIN))
    if kind == "map1_ppm":
        fn = map1_ppm_jit if jit else map1_ppm_j
        return np.asarray(fn(*a, km, kn, iv, kord))
    fn = map1_q2_jit if jit else map1_q2_j
    return np.asarray(fn(a[0], a[1], a[2], jnp.asarray(dp2), km, kn, iv,
                         kord, 0.0))


MAP_KINDS = ("map_scalar", "map1_ppm", "map1_q2")


def _map_args(kind, im=4, km=KM, iv=None):
    """(pe1, q1, pe2, dp2, iv) for one map routine on the shared fixture."""
    pe1, pe2, _, _ = _lagrangian_edges(im=im, km=km)
    q1 = _column(im=im, km=km, amp=8.0)
    dp2 = _dp_1based(pe2)
    if iv is None:
        iv = {"map_scalar": 1, "map1_ppm": -1, "map1_q2": 0}[kind]
    return pe1, q1, pe2, dp2, iv


@pytest.mark.parametrize("kord", ALL_KORD)
@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_matches_numpy_lane_every_kord(kind, kord):
    pe1, q1, pe2, dp2, iv = _map_args(kind, km=KMP)
    ref = _run_map(kind, pe1, q1, pe2, KMP, KMP, iv, kord, dp2, lane="np")
    got = _run_map(kind, pe1, q1, pe2, KMP, KMP, iv, kord, dp2)
    # MEASURED (job 9425294 sweep): worst 4.403e-16; bound = measured x 10 =
    # 4.5e-15.
    gate_scalar(f"map {kind} kord={kord}", _rel(got[:, 1:], ref[:, 1:]),
                4.5e-15)
    # NON-VACUITY: the rezone genuinely moved the field (a lane that
    # copied its input would pass every parity AND conservation test).
    assert np.abs(got[:, 1:] - q1[:, 1:]).max() > 1e-3


@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_jit_matches_eager(kind):
    pe1, q1, pe2, dp2, iv = _map_args(kind, km=KMP)
    eager = _run_map(kind, pe1, q1, pe2, KMP, KMP, iv, 9, dp2)
    jitted = _run_map(kind, pe1, q1, pe2, KMP, KMP, iv, 9, dp2, jit=True)
    # MEASURED (job 9425294 sweep): worst 3.302e-16 (map1_ppm); bound = measured
    # x 10 = 3.4e-15.
    gate_scalar(f"map jit {kind}", _rel(jitted, eager), 3.4e-15)


@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_no_retrace(kind):
    pe1, q1, pe2, dp2, iv = _map_args(kind, km=KMP)
    traces = {"n": 0}
    base = {"map_scalar": map_scalar_j, "map1_ppm": map1_ppm_j,
            "map1_q2": map1_q2_j}[kind]

    def _counted(*a, **kw):
        traces["n"] += 1
        return base(*a, **kw)

    fac = {"map_scalar": make_map_scalar_jit, "map1_ppm": make_map1_ppm_jit,
           "map1_q2": make_map1_q2_jit}[kind]
    fn = fac(_counted)
    for scale in (1.0, 1.001):
        a = (jnp.asarray(pe1), jnp.asarray(q1 * scale), jnp.asarray(pe2))
        if kind == "map_scalar":
            fn(*a, KMP, KMP, iv, 9, T_MIN)
        elif kind == "map1_ppm":
            fn(*a, KMP, KMP, iv, 9)
        else:
            fn(a[0], a[1], a[2], jnp.asarray(dp2), KMP, KMP, iv, 9, 0.0)
    assert traces["n"] == 1, traces["n"]


def test_map1_q2_jax_divides_by_the_callers_dp2():
    """``map1_q2`` must use ``dp2``, not ``pe2(k+1)-pe2(k)``.

    They are equal on a consistent grid, so the only way to see the
    difference is to hand it an inconsistent one.  A port that ignored
    ``dp2`` would return the ``map_scalar`` answer here.
    """
    pe1, q1, pe2, dp2, _ = _map_args("map1_q2", km=KMP)
    ref = _run_map("map1_q2", pe1, q1, pe2, KMP, KMP, 0, 9, dp2)
    skew = dp2.copy()
    skew[:, 1:] *= 2.0
    got = _run_map("map1_q2", pe1, q1, pe2, KMP, KMP, 0, 9, skew)
    assert np.abs(got[:, 1:] - ref[:, 1:]).max() > 1e-12, "dp2 ignored"
    # ...and the skewed answer still tracks the NumPy lane.
    ref_s = _run_map("map1_q2", pe1, q1, pe2, KMP, KMP, 0, 9, skew,
                     lane="np")
    # MEASURED (job 9425294 sweep): 2.201e-16; bound = measured x 10 = 2.3e-15.
    gate_scalar("map1_q2 skewed dp2", _rel(got[:, 1:], ref_s[:, 1:]),
                2.3e-15)


def test_map_scalar_jax_is_not_map1_ppm():
    """``map_scalar`` selects ``scalar_profile``, ``map1_ppm`` selects
    ``cs_profile``.  Both conserve, so no conservation test can tell
    them apart -- this one can."""
    im, km = 3, KMP
    pe1, pe2, _, _ = _lagrangian_edges(im=im, km=km)
    q1 = _cold_column(im=im, km=km)          # below T_MIN: qmin can fire
    a = _run_map("map_scalar", pe1, q1, pe2, km, km, 1, 9)
    b = _run_map("map1_ppm", pe1, q1, pe2, km, km, 1, 9)
    assert np.abs(a[:, 1:] - b[:, 1:]).max() > 1e-9, \
        "map_scalar and map1_ppm returned the same column"


# ------------------------------------------- interface ties (correction 1)

def _tie_edges(im=3, km=KMP, which=4, mode="exact"):
    """Target grid with edge ``which`` ON a source interface, +-1 ULP.

    At ``mode='exact'`` the bracket test is satisfied by BOTH
    ``l = which-1`` and ``l = which`` (``pe1(l) <= x <= pe1(l+1)`` with
    ``x`` equal to the shared interface), so the answer is decided by the
    FIRST-MATCH rule walking up from ``k0``.  One ULP either way leaves
    exactly one candidate.
    """
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    pe2 = _midpoint_targets(pe1, km)
    x = np.array(pe1[:, which], copy=True)
    if mode == "below":
        x = np.nextafter(x, -np.inf)
    elif mode == "above":
        x = np.nextafter(x, np.inf)
    pe2[:, which] = x
    assert (np.diff(pe2[:, 1:], axis=1) > 0).all(), "target not monotone"
    return pe1, pe2


@pytest.mark.parametrize("mode", ["exact", "below", "above"])
@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_matches_numpy_lane_at_an_interface_tie(kind, mode):
    """A one-ULP move at an interface can select a DIFFERENT source layer
    and a DIFFERENT formula, so a spread between the three modes is
    expected and is NOT rounding-scale.  What must hold -- and what is
    asserted -- is that BOTH lanes make the SAME selection."""
    im, km = 3, KMP
    pe1, pe2 = _tie_edges(im=im, km=km, mode=mode)
    q1 = _column(im=im, km=km, amp=8.0)
    dp2 = _dp_1based(pe2)
    iv = {"map_scalar": 1, "map1_ppm": -1, "map1_q2": 0}[kind]
    ref = _run_map(kind, pe1, q1, pe2, km, km, iv, 9, dp2, lane="np")
    got = _run_map(kind, pe1, q1, pe2, km, km, iv, 9, dp2)
    # MEASURED (job 9425294 sweep): worst 3.308e-16 (map1_ppm, all modes); bound
    # = measured x 10 = 3.4e-15.
    gate_scalar(f"map tie {kind} {mode}", _rel(got[:, 1:], ref[:, 1:]),
                3.4e-15)


def test_map_jax_tie_spread_agrees_with_the_numpy_lane():
    """The DECISIVE selection test: the exact-vs-one-ULP spread is a
    property of WHICH layer the search picked, so if the two lanes
    picked differently their spreads would differ grossly.  Comparing
    the spreads is therefore a tie-rule assertion that no tolerance on
    the values alone can make."""
    im, km = 3, KMP
    q1 = _column(im=im, km=km, amp=8.0)
    out = {}
    for lane in ("np", "jax"):
        for mode in ("exact", "below", "above"):
            pe1, pe2 = _tie_edges(im=im, km=km, mode=mode)
            out[lane, mode] = _run_map("map1_ppm", pe1, q1, pe2, km, km,
                                       -1, 9, lane=lane)
    for other in ("below", "above"):
        d_n = np.abs(out["np", "exact"] - out["np", other]).max()
        d_j = np.abs(out["jax", "exact"] - out["jax", other]).max()
        # MEASURED (job 9425294 sweep): exactly 0.0 (both directions); bound = 1e-15
        # eps guard (measured exactly 0.0).
        gate_scalar(f"map tie-spread {other}",
                    abs(d_j - d_n) / max(d_n, 1.0), 1e-15,
                    quantity="tie-spread agreement rel")


# ------------------------------------------- the no-bracket path

def test_map_jax_no_bracket_is_nan_and_flagged_where_numpy_raises():
    """The NumPy lane RAISES ``FloatingPointError`` when a target edge is
    bracketed by no source interval.  A data-dependent raise cannot
    survive jit, so the JAX twin writes NaN AND returns an explicit
    ``ok`` flag -- never a plausible number."""
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    q1 = _column(im=im, km=km, amp=8.0)
    bad = _midpoint_targets(pe1, km)
    bad[:, 1] = pe1[:, 1] - 100.0                # above the model top
    with pytest.raises(FloatingPointError, match="bracketed by no source"):
        map_scalar_n(pe1, q1, bad, km, km, 1, 9, T_MIN)
    got, ok = map_scalar_j(jnp.asarray(pe1), jnp.asarray(q1),
                           jnp.asarray(bad), km, km, 1, 9, T_MIN,
                           return_ok=True)
    assert not bool(ok), "the ok flag missed an unbracketed edge"
    assert np.isnan(np.asarray(got)[:, 1]).all(), np.asarray(got)[:, 1]
    # NON-VACUITY: the well-posed target is flagged ok and has no NaN.
    good = _midpoint_targets(pe1, km)
    got2, ok2 = map_scalar_j(jnp.asarray(pe1), jnp.asarray(q1),
                             jnp.asarray(good), km, km, 1, 9, T_MIN,
                             return_ok=True)
    assert bool(ok2) and np.isfinite(np.asarray(got2)).all()


# ------------------------------------------- CONSERVATION (correction 4)

@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_conserves_the_column_integral_as_well_as_numpy(kind):
    """THE tolerance-independent gate for the rezone.

    INVARIANT: the column integral ``sum_k q(k)*dp(k)`` -- the mass of
    the remapped scalar -- with ``dp1 = pe1(k+1)-pe1(k)`` on the source
    and, on the target, ``pe2(k+1)-pe2(k)`` for map_scalar/map1_ppm and
    the CALLER's ``dp2`` for map1_q2 (each routine's own divisor at
    :1449/:1754).  This is the right conserved quantity because it is
    what the operator is BUILT on: the PPM sub-grid reconstruction has
    cell mean ``q(k)`` over ``dp1(k)``, and each target value is that
    reconstruction's integral over the target cell divided by the target
    thickness.  It holds only because the two grids share their
    endpoints, which the driver enforces by COPYING them (:297-300).

    Its power does not depend on the parity tolerance: a lane that
    reassociated the ``qsum`` accumulation, picked the wrong layer at a
    tie, or dropped a whole-layer term would still look "close" while
    breaking this identity.  The bound is expressed RELATIVE TO THE
    NumPy LANE'S OWN residual, so the gate asks the question the review
    asked -- "preserved to the level the NumPy lane preserves it".
    """
    im, km = 4, KMP
    pe1, pe2, _, _ = _lagrangian_edges(im=im, km=km)
    q1 = _column(im=im, km=km, amp=8.0)
    dp2 = _dp_1based(pe2)
    iv = {"map_scalar": 1, "map1_ppm": -1, "map1_q2": 0}[kind]
    src = (q1[:, 1:] * _dp_1based(pe1)[:, 1:]).sum(axis=1)
    tgt_dp = dp2[:, 1:]
    res = {}
    for lane in ("np", "jax"):
        out = _run_map(kind, pe1, q1, pe2, km, km, iv, 9, dp2, lane=lane)
        res[lane] = np.abs((out[:, 1:] * tgt_dp).sum(axis=1) - src).max() \
            / np.abs(src).max()
    # MEASURED (job 9425294 sweep): res['jax'] = 1.486e-16 for all three kinds;
    # floor = measured x 10 = 1.5e-15 (the 3x-NumPy term unchanged).
    gate_scalar(f"map conservation {kind}", res["jax"],
                max(3.0 * res["np"], 1.5e-15),
                quantity="column-mass conservation defect rel"
                         " (bound floor is the pinned term)")
    # NON-VACUITY: the target grid is genuinely different from the source
    # (on identical grids the rezone is the identity and conserves for
    # free).
    assert np.abs(_dp_1based(pe2)[:, 1:] - _dp_1based(pe1)[:, 1:]).max() \
        > 1e-3 * np.abs(_dp_1based(pe1)[:, 1:]).max()


@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_preserves_a_constant_field(kind):
    """CONSTANT-FIELD PRESERVATION (correction 4): a uniform column must
    remap to the same constant.  Independent of any tolerance on the
    lane comparison, and it fails loudly for a mis-indexed gather, a
    dropped ``A6`` term or an off-by-one in the interval walk."""
    im, km = 4, KMP
    pe1, pe2, _, _ = _lagrangian_edges(im=im, km=km)
    q1 = np.zeros((im, km + 1), dtype=np.float64)
    q1[:, 1:] = 7.25
    iv = {"map_scalar": 1, "map1_ppm": -1, "map1_q2": 0}[kind]
    out = _run_map(kind, pe1, q1, pe2, km, km, iv, 9, _dp_1based(pe2))
    # MEASURED (job 9425294 sweep): 2.450e-16 (all kinds); bound = measured x 10
    # = 2.5e-15.
    gate_scalar(f"map constant-field {kind}",
                np.abs(out[:, 1:] - 7.25).max() / 7.25, 2.5e-15,
                quantity="constant-preservation rel")


def test_map_jax_identity_remap_returns_the_cell_means():
    """pe2 == pe1 must return the input cell means: with pl=0 and pr=1
    the parabola mean collapses to ``AL + (AR-AL)/2 + A6/6``, which is
    ``a4(1)`` by the definition of A6."""
    im, km = 4, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    q1 = _column(im=im, km=km, amp=8.0)
    out = _run_map("map_scalar", pe1, q1, pe1, km, km, 1, 9)
    # MEASURED (job 9425294 sweep): 2.203e-16; bound = measured x 10 = 2.3e-15.
    gate_scalar("map identity remap", _rel(out[:, 1:], q1[:, 1:]),
                2.3e-15)


# ------------------------------------------- guards

@pytest.mark.parametrize("bad", ["pe1", "q1", "pe2"])
def test_map_jax_rejects_float32_every_operand(bad):
    pe1, q1, pe2, _, _ = _map_args("map_scalar")
    args = {"pe1": pe1, "q1": q1, "pe2": pe2}
    args[bad] = args[bad].astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        map_scalar_j(jnp.asarray(args["pe1"]), jnp.asarray(args["q1"]),
                     jnp.asarray(args["pe2"]), KM, KM, 1, 9, T_MIN)
    map_scalar_j(jnp.asarray(pe1), jnp.asarray(q1), jnp.asarray(pe2),
                 KM, KM, 1, 9, T_MIN)          # NON-VACUITY


def test_map1_q2_jax_rejects_float32_dp2():
    pe1, q1, pe2, dp2, _ = _map_args("map1_q2")
    with pytest.raises(TypeError, match="dp2.*float64"):
        map1_q2_j(jnp.asarray(pe1), jnp.asarray(q1), jnp.asarray(pe2),
                  jnp.asarray(dp2, dtype=jnp.float32), KM, KM, 0, 9, 0.0)


@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_low_kord_is_refused(kind):
    pe1, q1, pe2, dp2, iv = _map_args(kind)
    with pytest.raises(ValueError, match="ppm_profile"):
        _run_map(kind, pe1, q1, pe2, KM, KM, iv, 7, dp2)
    _run_map(kind, pe1, q1, pe2, KM, KM, iv, 9, dp2)     # NON-VACUITY


def test_map_jax_rejects_a_wrong_shaped_operand():
    pe1, q1, pe2, _, _ = _map_args("map_scalar")
    with pytest.raises(ValueError, match="pe2 must be 1-based"):
        map_scalar_j(jnp.asarray(pe1), jnp.asarray(q1),
                     jnp.asarray(pe2[:, :-1]), KM, KM, 1, 9, T_MIN)


# ------------------------------------------- gate 4

def test_map_jax_check_grads_order2_wrt_the_field():
    """Gradients w.r.t. the FIELD, with the coordinates held fixed.

    NON-SMOOTH SITES: the profile limiters (measured off-switch below)
    and the interval selection -- which is piecewise constant in the
    COORDINATES and therefore has no bearing on a field gradient at all.
    """
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    pe2 = _midpoint_targets(pe1, km)
    q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])
    dp1 = _dp_1based(pe1)
    _assert_profile_off_switch(_jax_profile("scalar", q1, dp1, km, 1, 9),
                               q1, km)

    def f(q_):
        return jnp.sum(map_scalar_j(jnp.asarray(pe1), q_, jnp.asarray(pe2),
                                    km, km, 1, 9, T_MIN) ** 2)

    check_grads(f, (jnp.asarray(q1),), order=2, modes=("fwd", "rev"))


def test_map_jax_check_grads_order2_wrt_the_coordinates():
    """Gradients w.r.t. ``pe1``/``pe2`` need every target edge STRICTLY
    inside a source cell, because the selected layer is an integer and
    is piecewise constant in the coordinates.

    The margin is measured below.  NAMED non-differentiable site: the
    driver's own target grid COPIES its two endpoints from the source
    (:297-300), which puts them EXACTLY on the switch -- so coordinate
    gradients are sub-differential there and are not claimed.  This
    fixture moves both endpoints strictly inside on purpose.
    """
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    pe2 = _midpoint_targets(pe1, km)
    pe2[:, 1] = 0.5 * (pe1[:, 1] + pe2[:, 2])            # off the endpoint
    pe2[:, km + 1] = 0.5 * (pe2[:, km] + pe1[:, km + 1])
    gap = np.abs(pe2[:, 1:km + 2][:, :, None]
                 - pe1[:, 1:km + 2][:, None, :]).min()
    assert gap > 1.0, f"a target edge is {gap} Pa from a source interface"
    q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])
    _assert_profile_off_switch(
        _jax_profile("scalar", q1, _dp_1based(pe1), km, 1, 9), q1, km)

    def f(pe1_, pe2_):
        return jnp.sum(map_scalar_j(pe1_, jnp.asarray(q1), pe2_, km, km,
                                    1, 9, T_MIN) ** 2)

    check_grads(f, (jnp.asarray(pe1), jnp.asarray(pe2)), order=2,
                modes=("fwd", "rev"))


def test_map1_ppm_jax_check_grads_order2_iv_minus2_bc():
    """The w lane: iv=-2 carries the ``qs`` bottom BC into the rezone."""
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    pe2 = _midpoint_targets(pe1, km)
    q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])
    qs = _smooth_column_bottom_value(edges=pe1[:, 1:])

    def f(q_, qs_):
        return jnp.sum(map1_ppm_j(jnp.asarray(pe1), q_, jnp.asarray(pe2),
                                  km, km, -2, 9, qs=qs_) ** 2)

    check_grads(f, (jnp.asarray(q1), jnp.asarray(qs)), order=2,
                modes=("fwd", "rev"))


# ====================================================================== #
# pad1 / unpad1
# ====================================================================== #

def test_pad1_unpad1_jax_match_the_numpy_lane_and_roundtrip():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(np.asarray(pad1_j(jnp.asarray(a))), pad1_n(a))
    assert np.array_equal(np.asarray(unpad1_j(pad1_j(jnp.asarray(a)))), a)
    assert np.array_equal(np.asarray(unpad1_j(jnp.asarray(pad1_n(a)))),
                          unpad1_n(pad1_n(a)))
    assert np.array_equal(np.asarray(pad1_j(jnp.asarray(a)))[:, 0],
                          np.zeros(3))
    # jit parity, asserted.
    assert np.array_equal(np.asarray(pad1_jit(jnp.asarray(a))),
                          np.asarray(pad1_j(jnp.asarray(a))))
    assert np.array_equal(np.asarray(unpad1_jit(jnp.asarray(a))),
                          np.asarray(unpad1_j(jnp.asarray(a))))


@pytest.mark.parametrize("fn", [pad1_j, unpad1_j])
def test_pad1_unpad1_jax_reject_float32(fn):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    with pytest.raises(TypeError, match="float64"):
        fn(jnp.asarray(a))
    fn(jnp.asarray(a.astype(np.float64)))                # NON-VACUITY


def test_pad1_jax_check_grads_order2():
    a = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    check_grads(lambda x: jnp.sum(pad1_j(x) ** 2), (jnp.asarray(a),),
                order=2, modes=("fwd", "rev"))
    check_grads(lambda x: jnp.sum(unpad1_j(x) ** 2), (jnp.asarray(a),),
                order=2, modes=("fwd", "rev"))


# ====================================================================== #
# lagrangian_to_eulerian -- the driver
# ====================================================================== #

_L2E_FIELDS = ("pe", "peln", "pk", "pkz", "delp", "pt", "u", "v", "ps",
               "omga")


def _tracers(face, scales=(1e-3, 4e-4), phases=(1.4, 0.3), ntracer=None):
    n, ng, km = face["n"], face["ng"], face["km"]
    ia, m_a = ng, n + 2 * ng
    if ntracer is not None and ntracer > len(scales):
        # distinct amplitude and vertical phase per extra tracer, so a
        # slot mix-up in the nq > 5 arm cannot hide behind symmetry
        scales = tuple(scales) + tuple(1e-3 / (2 + i) for i in range(ntracer - len(scales)))
        phases = tuple(phases) + tuple(0.3 + 0.37 * i for i in range(ntracer - len(phases)))
    out = []
    for scale, phase in zip(scales, phases):
        qq = np.zeros((m_a, m_a, km), dtype=np.float64)
        qq[ia:ia + n, ia:ia + n, :] = scale * (
            1.0 + 0.5 * np.cos(np.arange(km) * phase))[None, None, :]
        out.append(qq)
    return out


def _run_driver_both(make_face, ntracer=2, **kw):
    """Run BOTH lanes on byte-identical inputs; return (numpy dict, out).

    ``make_face`` is called twice, so the two lanes never share an array
    -- the NumPy lane mutates its operands and would otherwise poison the
    JAX lane's inputs.
    """
    face_n, face_j = make_face(), make_face()
    for k in _L2E_FIELDS:
        if k in face_n:
            assert np.array_equal(face_n[k], face_j[k]), k
    q_n = _tracers(face_n, ntracer=ntracer)[:ntracer]
    q_j = [np.array(x, copy=True) for x in q_n]
    before = {k: np.array(v, copy=True) for k, v in face_n.items()
              if isinstance(v, np.ndarray)}
    before["q"] = [np.array(x, copy=True) for x in q_n]
    l2e_n(**face_n, q=q_n, **kw)
    out = l2e_j(**{k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                   for k, v in face_j.items()},
                q=[jnp.asarray(x) for x in q_j], **kw)
    face_n["q"] = q_n
    return face_n, out, before


def _cmp_l2e(ref, out, keys, tol, label=""):
    for k in keys:
        got = np.asarray(getattr(out, k))
        want = ref[k]
        assert got.shape == want.shape, (k, got.shape, want.shape)
        gate_scalar(f"l2e {label} {k}", _rel(got, want), tol)


# ------------------------------------------------------------- gate 1
def test_driver_jax_matches_numpy_lane_hydrostatic():
    ref, out, _ = _run_driver_both(lambda: _face()[0])
    # MEASURED (job 9425294 sweep): worst field pkz 8.002e-15 (all others
    # <= 5.5e-16); bound = measured x 10 = 8.1e-14.
    _cmp_l2e(ref, out, _L2E_FIELDS, 8.1e-14, "hydro")
    for iq, qn in enumerate(ref["q"]):
        # MEASURED (job 9425294 sweep): worst q[1] 5.421e-16; bound = measured x 10
        # = 5.5e-15.
        gate_scalar(f"l2e hydro q[{iq}]", _rel(np.asarray(out.q[iq]), qn),
                    5.5e-15)
    assert isinstance(out, LagrangianToEulerianOut)


def test_driver_jax_matches_numpy_lane_not_last_step():
    """``last_step=False`` takes the OTHER closing arm (:996-1001, pt /=
    pkz) and skips the omega block entirely."""
    ref, out, _ = _run_driver_both(lambda: _face()[0], last_step=False)
    # MEASURED (job 9425294 sweep): worst field pkz 8.002e-15 (pt 6.291e-15);
    # bound = measured x 10 = 8.1e-14.
    _cmp_l2e(ref, out, _L2E_FIELDS, 8.1e-14, "not-last")


@pytest.mark.parametrize("w_limiter", [False, True])
def test_driver_jax_matches_numpy_lane_nonhydrostatic(w_limiter):
    """The NH lane: the ideal-gas theta conversion, the w and delz
    remaps, the NH pkz, and (parametrized) the two-pass w limiter."""

    def _mk():
        f = _nh_face(w_const=0.0)
        f["w"][f["ng"] + 2, f["ng"] + 3, 2] = 200.0     # violates w_max=90
        f["ws"][:] = 0.0
        f["w_limiter"] = w_limiter
        return f

    ref, out, _ = _run_driver_both(_mk)
    # MEASURED (job 9425294 sweep): worst field pkz 6.667e-16 (delz 5.890e-16),
    # both limiter arms; bound = measured x 10 = 6.7e-15.
    _cmp_l2e(ref, out, _L2E_FIELDS + ("w", "delz"), 6.7e-15,
             f"nh-limiter={w_limiter}")
    # NON-VACUITY: the limiter arm really is a different answer.
    if w_limiter:
        assert np.asarray(out.w).max() <= W_MAX_MAPZ + 1e-9
    else:
        assert np.asarray(out.w).max() > W_MAX_MAPZ


def test_driver_jax_matches_numpy_lane_moist_closing_conversion():
    """``r_vir != 0`` at last_step divides pt by ``1 + r_vir*q(sphum)``
    using the EXPLICIT sphum index (:975), never tracer 0."""

    def _mk():
        f = _face()[0]
        f["r_vir"] = 1.0
        return f

    ref, out, _ = _run_driver_both(_mk, sphum_index=1)
    # MEASURED (job 9425294 sweep): worst field pkz 8.002e-15; bound = measured
    # x 10 = 8.1e-14.
    _cmp_l2e(ref, out, _L2E_FIELDS, 8.1e-14, "moist")
    # NON-VACUITY: sphum_index=1 is not sphum_index=0 on this fixture.
    f0 = _face()[0]
    f0["r_vir"] = 1.0
    other = l2e_j(**{k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                     for k, v in f0.items()},
                  q=[jnp.asarray(x) for x in _tracers(f0)], sphum_index=0)
    assert np.abs(np.asarray(other.pt) - np.asarray(out.pt)).max() > 1e-9




def test_moist_nonhydrostatic_refuses_the_adiabatic_flag():
    """``fv_mapz.F90:985`` puts the NH closing T_v -> T divide inside
    ``if (.not. adiabatic)``.

    With ``adiabatic=.true.`` the oracle does not convert at all and pt
    stays virtual; this lane always divides, i.e. it computes the
    ``.not. adiabatic`` branch (with ``consv = 0``, ``dtmp`` is
    identically 0, so :987 reduces exactly to it).  Running the ported
    expression under the flag that names the OTHER branch is the silent
    failure this repo ranks worst, so the combination raises.

    The HYDROSTATIC divide at :975 carries no such guard, and the
    second half asserts that -- without it this test would pass on a
    lane that refused moist coupling everywhere.
    """
    def _mk(hydro):
        f = (_face()[0] if hydro else _nh_face(w_const=0.0))
        f["r_vir"] = 1.0
        return f

    f = _mk(False)
    kw = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
          for k, v in f.items()}
    q = [jnp.asarray(x) for x in _tracers(f)]
    with pytest.raises(NotImplementedError, match="985|adiabatic=False"):
        l2e_j(**kw, q=q, sphum_index=1, adiabatic=True)
    # BOTH lanes carry the guard; the NumPy lane is the specification,
    # so a guard present only in the port is a lane divergence.
    fn = _mk(False)
    with pytest.raises(NotImplementedError, match="985|adiabatic=False"):
        l2e_n(**fn, q=_tracers(fn), sphum_index=1, adiabatic=True)
    # the SAME call with the faithful flag runs
    out = l2e_j(**kw, q=q, sphum_index=1, adiabatic=False)
    assert np.all(np.isfinite(np.asarray(out.pt)))

    # CONTROL: hydrostatic + adiabatic=True must NOT raise (:975 is
    # ungated), so the guard is NH-scoped rather than a blanket refusal.
    fh = _mk(True)
    kwh = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
           for k, v in fh.items()}
    l2e_j(**kwh, q=[jnp.asarray(x) for x in _tracers(fh)],
          sphum_index=1, adiabatic=True)


@pytest.mark.parametrize("lane", ["jax", "numpy"])
@pytest.mark.parametrize("bad", [True, 1.5, "1", None])
def test_moist_sphum_index_is_not_coerced(lane, bad):
    """``int()`` accepts ``True``, ``1.5`` and ``"1"`` -- all of which
    would silently select tracer 1 (GLM MINOR, job 9442423).

    BOTH lanes: the NumPy lane is the specification, so a guard present
    only in the port is itself a lane divergence (GLM MINOR, job
    9442483).
    """
    f = _face()[0]
    f["r_vir"] = 1.0
    with pytest.raises(ValueError, match="sphum_index"):
        if lane == "jax":
            kw = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                  for k, v in f.items()}
            l2e_j(**kw, q=[jnp.asarray(x) for x in _tracers(f)],
                  sphum_index=bad)
        else:
            l2e_n(**f, q=_tracers(f), sphum_index=bad)


@pytest.mark.parametrize("lane", ["jax", "numpy"])
def test_moist_sphum_index_accepts_a_numpy_integer(lane):
    """A config-driven caller carries ``np.int64``; rejecting it would
    be a spurious raise, and this pins that the widened guard did not
    also widen to bools (covered above)."""
    f = _face()[0]
    f["r_vir"] = 1.0
    if lane == "jax":
        kw = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
              for k, v in f.items()}
        out = l2e_j(**kw, q=[jnp.asarray(x) for x in _tracers(f)],
                    sphum_index=np.int64(1))
        pt = np.asarray(out.pt)
    else:
        l2e_n(**f, q=_tracers(f), sphum_index=np.int64(1))
        pt = f["pt"]
    assert np.all(np.isfinite(pt))

# ------------------------------------------------------------- gate 2
def test_driver_jax_jit_matches_eager():
    face = _face()[0]
    kw = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
          for k, v in face.items()}
    q = [jnp.asarray(x) for x in _tracers(face)]
    eager = l2e_j(**kw, q=q)
    jitted = lagrangian_to_eulerian_jit(**kw, q=q)
    for k in _L2E_FIELDS:
        # MEASURED (job 9425294 sweep): worst field omga 8.845e-15 (pkz 7.869e-15);
        # bound = measured x 10 = 8.9e-14.
        gate_scalar(f"l2e jit {k}",
                    _rel(np.asarray(getattr(jitted, k)),
                         np.asarray(getattr(eager, k))), 8.9e-14)


def test_driver_jax_no_retrace():
    face = _face()[0]
    traces = {"n": 0}

    def _counted(**kw):
        traces["n"] += 1
        return l2e_j(**kw)

    fn = make_lagrangian_to_eulerian_jit(_counted)
    for scale in (1.0, 1.0001):
        # Only pt is perturbed: scaling the coordinates too would leave
        # pe and peln mutually inconsistent, which is a different test.
        kw = {k: (jnp.asarray(v * scale if k == "pt" else v)
                  if isinstance(v, np.ndarray) else v)
              for k, v in face.items()}
        fn(**kw, q=tuple(jnp.asarray(x) for x in _tracers(face)))
    assert traces["n"] == 1, traces["n"]


# ------------------------------------------------------------- gate 3
@pytest.mark.parametrize("bad", ["pe", "peln", "pk", "pkz", "delp", "pt",
                                 "u", "v", "ps", "omga", "ak", "bk"])
def test_driver_jax_rejects_float32_every_operand(bad):
    face = _face()[0]
    face[bad] = np.asarray(face[bad], dtype=np.float32)
    with pytest.raises(TypeError, match=f"{bad}.*float64"):
        l2e_j(**face, q=[])


def test_driver_jax_rejects_a_float32_tracer():
    face = _face()[0]
    q = _tracers(face)
    q[1] = q[1].astype(np.float32)
    with pytest.raises(TypeError, match=r"q\[1\].*float64"):
        l2e_j(**face, q=q)


@pytest.mark.parametrize("override, needle, ntracer", [
    (dict(consv=1.0), "consv", 0),
    (dict(fill=True), "fillz", 1),
    (dict(kord_tm=9), "kord_tm", 0),
    (dict(do_sat_adj=True), "do_sat_adj", 0),
    (dict(do_inline_mp=True), "do_inline_mp", 0),
    (dict(do_adiabatic_init=True), "do_adiabatic_init", 0),
])
def test_driver_jax_refuses_every_unported_lane(override, needle, ntracer):
    face = _face()[0]
    face.update(override)
    m_a = face["n"] + 2 * face["ng"]
    tr = [np.zeros((m_a, m_a, face["km"]), dtype=np.float64)
          for _ in range(ntracer)]
    with pytest.raises(NotImplementedError, match=needle):
        l2e_j(**face, q=tr)


# ---------------------------------------------------------------------
# mapn_tracer (nq > 5, fv_mapz.F90:327): the nine-slot ice decks
# ---------------------------------------------------------------------

def _mapn_args(nq=7):
    pe1, q1, pe2, dp2, _ = _map_args("map1_q2", km=KMP)
    qs = [q1 * (1.0 + 0.13 * i) + 3e-4 * i for i in range(nq)]
    return pe1, qs, pe2, dp2


def test_mapn_tracer_jax_matches_numpy_lane():
    """The JAX twin against the NumPy authority (the literal Fortran
    transcription), per tracer, at the map1_q2 class.  Mixed kords, so
    a port reading kords[0] for every tracer is caught."""
    from legoesm.core.fv3_mapz import mapn_tracer as mapn_j
    from legoesm.core.fv3_native_mapz import mapn_tracer as mapn_n
    pe1, qs, pe2, dp2 = _mapn_args()
    kords = [9, 10, 9, 11, 9, 12, 9][:len(qs)]
    ref = mapn_n(pe1, [np.array(a, copy=True) for a in qs], pe2, dp2, KMP,
                 kords, 0.0)
    got = mapn_j(jnp.asarray(pe1), [jnp.asarray(a) for a in qs],
                 jnp.asarray(pe2), jnp.asarray(dp2), KMP, kords, 0.0)
    for iq, (a, b) in enumerate(zip(got, ref)):
        # same class as map1_q2 (MEASURED 2.2e-16 there; bound x10)
        gate_scalar(f"mapn_tracer q[{iq}]", _rel(np.asarray(a)[:, 1:],
                                                 b[:, 1:]), 2.3e-15)


def test_mapn_tracer_agrees_with_map1_q2_per_tracer_at_rounding_level():
    """mapn_tracer is the per-tracer remap with the products associated
    as :1808-1824 write them (fac1 = 0.5*(pr+pl) first): equal to
    map1_q2 in exact arithmetic, so a wrong fac1/fac2 in the mapn branch
    fails this bound.  Well-posed only because kord = 9 > 7: both
    routines then take scalar_profile (:1708 dispatches ppm_profile
    below 8, which this lane refuses), so the limiter is shared and only
    the integration's association differs.  MEASURED on this fixture:
    the two associations agree BITWISE (0.0), so no positive lower bound
    is asserted -- that the mapn arm ran is pinned by the dispatch test
    below, not here."""
    from legoesm.core.fv3_native_mapz import mapn_tracer as mapn_n
    pe1, qs, pe2, dp2 = _mapn_args()
    got = mapn_n(pe1, [np.array(a, copy=True) for a in qs], pe2, dp2, KMP,
                 [9] * len(qs), 0.0)
    worst = 0.0
    for a, q in zip(got, qs):
        ref = map1_q2_n(pe1, np.array(q, copy=True), pe2, dp2, KMP, KMP, 0, 9,
                        0.0)
        worst = max(worst, _rel(a[:, 1:], ref[:, 1:]))
    assert worst < 1e-14, worst


@pytest.mark.parametrize("nq, bound, cut", [
    (9, "mapn_tracer", "map1_q2"),
    (2, "map1_q2", "mapn_tracer"),
])
def test_l2e_tracer_arm_binds_on_nq_in_both_lanes(monkeypatch, nq, bound, cut):
    """:327 dispatch: nq > 5 runs mapn_tracer and NOTHING else; nq <= 5
    runs map1_q2 and NOTHING else -- on EACH lane, counted per lane.
    Both arms are wrapped by per-lane spies, so a lane calling the wrong
    arm, both arms, or neither (a pre-bound alias) fails here, and a
    raise in one lane cannot mask the other."""
    import legoesm.core.fv3_mapz as mj
    import legoesm.core.fv3_native_mapz as mn
    calls = {}

    def spy(mod, name):
        real = getattr(mod, name)

        def wrapped(*a, **k):
            calls[(mod.__name__, name)] = calls.get((mod.__name__, name), 0) + 1
            return real(*a, **k)
        monkeypatch.setattr(mod, name, wrapped)
    for mod in (mn, mj):
        spy(mod, "map1_q2")
        spy(mod, "mapn_tracer")
    ref, out, _ = _run_driver_both(
        lambda: {**_face()[0], "kord_tr": [9] * nq}, ntracer=nq)
    assert len(out.q) == nq and len(ref["q"]) == nq
    # the NumPy lane calls per j-row (:300 loop), the JAX lane once per
    # face -- so "ran" is > 0, not an exact count
    for mod in (mn, mj):
        assert calls.get((mod.__name__, bound), 0) > 0, (mod.__name__, calls)
        assert calls.get((mod.__name__, cut), 0) == 0, (mod.__name__, calls)


def test_mapn_tracer_refuses_a_kord_list_of_the_wrong_length():
    from legoesm.core.fv3_mapz import mapn_tracer as mapn_j
    from legoesm.core.fv3_native_mapz import mapn_tracer as mapn_n
    pe1, qs, pe2, dp2 = _mapn_args()
    with pytest.raises(ValueError, match="kord entries"):
        mapn_n(pe1, qs, pe2, dp2, KMP, [9] * (len(qs) - 1), 0.0)
    with pytest.raises(ValueError, match="kord entries"):
        mapn_j(jnp.asarray(pe1), [jnp.asarray(a) for a in qs],
               jnp.asarray(pe2), jnp.asarray(dp2), KMP,
               [9] * (len(qs) - 1), 0.0)


def test_driver_jax_matches_numpy_lane_nine_tracers():
    """nq = 9 (the six water species + three numbers of an ice deck):
    JAX vs NumPy at the hydrostatic gate's class, every tracer.  Which
    arm ran is pinned by test_l2e_tracer_arm_binds_on_nq_in_both_lanes,
    not by this parity (the arms agree at rounding level)."""
    ref, out, before = _run_driver_both(
        lambda: {**_face()[0], "kord_tr": [9] * 9}, ntracer=9)
    _cmp_l2e(ref, out, _L2E_FIELDS, 8.1e-14, "hydro nq=9")
    assert len(out.q) == 9 and len(ref["q"]) == 9
    for iq, qn in enumerate(ref["q"]):
        gate_scalar(f"l2e nq=9 q[{iq}]", _rel(np.asarray(out.q[iq]), qn),
                    5.5e-15)
        assert not np.array_equal(qn, before["q"][iq]), iq   # remapped


def test_driver_jax_guards_do_not_over_refuse_a_non_last_step_call():
    """``consv`` and ``sphum`` are read ONLY inside ``if (last_step)``
    (:628, :964).  A guard that fires where the oracle does nothing is
    as much a divergence as one that fails to fire."""
    l2e_j(**_face()[0], q=[], last_step=False, consv=1.0)
    b = _face()[0]
    b["r_vir"] = 1.0
    l2e_j(**b, q=_tracers(b), last_step=False)
    l2e_j(**_face()[0], q=[], fill=True)          # fillz needs nq > 0


def test_driver_jax_requires_q_and_omga_explicitly():
    """Two DIFFERENT refusals, and job 9400424 showed the first
    expectation was wrong: ``q`` is a keyword-only parameter with no
    default, so OMITTING it raises Python's own
    ``missing 1 required keyword-only argument: 'q'`` before any guard
    in the body can run.  That is the desired behaviour and it is what
    the NumPy lane does too (its own test asserts a bare ``TypeError``);
    only the match string was wrong.  Passing the WRONG TYPE is the case
    the body's guard exists for, and it is asserted separately."""
    with pytest.raises(TypeError, match="required keyword-only argument"):
        l2e_j(**_face()[0])
    with pytest.raises(TypeError, match="q must be a list"):
        l2e_j(**_face()[0], q=np.zeros((3, 3, 3)))
    face = _face()[0]
    face.pop("omga")
    with pytest.raises(ValueError, match="omga"):
        l2e_j(**face, q=[])
    l2e_j(**face, q=[], last_step=False)          # NON-VACUITY


def test_driver_jax_refuses_a_moist_lane_without_a_valid_sphum():
    face = _face()[0]
    face["r_vir"] = 0.6077
    with pytest.raises(ValueError, match="r_vir"):
        l2e_j(**face, q=[])
    f2 = _face()[0]
    f2["r_vir"] = 0.6077
    with pytest.raises(ValueError, match="sphum_index"):
        l2e_j(**f2, q=_tracers(f2))
    f3 = _face()[0]
    f3["r_vir"] = 0.6077
    l2e_j(**f3, q=_tracers(f3), sphum_index=1)    # NON-VACUITY


def test_driver_jax_nh_requires_its_arguments():
    face = _face()[0]
    face["hydrostatic"] = False
    with pytest.raises(ValueError, match="hydrostatic=False needs"):
        l2e_j(**face, q=[])
    nh = _nh_face()
    nh["kord_wz"] = -9
    with pytest.raises(NotImplementedError, match="iv=-3"):
        l2e_j(**nh, q=[])
    l2e_j(**_nh_face(), q=[])                     # NON-VACUITY


def test_driver_jax_low_kord_is_refused_on_every_channel():
    for key in ("kord_mt", "kord_tm", "kord_tr"):
        face = _face()[0]
        face[key] = -7 if key == "kord_tm" else 7
        with pytest.raises(ValueError, match="ppm_profile"):
            l2e_j(**face, q=_tracers(face))


# ---------------------------------------- CONSERVATION (correction 4)
#
# Halo / area-weight statement: see ``_conservation_note`` at the top of
# this file.  Every invariant below is PER COLUMN (fv_mapz is
# column-local, so no cubed-sphere area weight enters), evaluated on the
# COMPUTE WINDOW, with the halo asserted bitwise-unchanged separately.

def test_driver_jax_conservation_invariants():
    """The five invariants, on the JAX lane's own output.

    1. GLOBAL DRY MASS -- the column-integrated ``delp`` is unchanged by
       the remap and equals ``ps - ptop``;
    2. PER-TRACER MASS -- ``sum_k q*delp`` unchanged, EACH tracer
       separately (a loop that remapped only ``q[0]`` is the natural
       off-by-one and a single-tracer test cannot see it);
    3. PRESSURE-THICKNESS CLOSURE -- the returned ``delp`` is exactly the
       difference of the returned ``pe``, and ``peln``/``pk`` are that
       same ``pe`` in log and ``**kappa`` form;
    4. CONSTANT-FIELD PRESERVATION -- a uniform tracer comes back
       uniform;
    5. (the ``w_limiter`` momentum invariant is a separate test below,
       since it needs the NH face).
    """
    face = _face()[0]
    n, ng, km = face["n"], face["ng"], face["km"]
    ia, m_a, ptop, akap = ng, n + 2 * ng, face["ptop"], face["akap"]
    q = _tracers(face)
    qc = np.zeros((m_a, m_a, km), dtype=np.float64)
    qc[ia:ia + n, ia:ia + n, :] = 3.75                 # invariant 4
    q = q + [qc]
    face["kord_tr"] = (9, 9, 9)          # the fixture already carries one
    dp0 = face["delp"][ia:ia + n, ia:ia + n, :].copy()
    m0 = [(x[ia:ia + n, ia:ia + n, :] * dp0).sum(axis=2) for x in q]
    col0 = dp0.sum(axis=2)

    out = l2e_j(**{k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                   for k, v in face.items()},
                q=[jnp.asarray(x) for x in q])

    dp1 = np.asarray(out.delp)[ia:ia + n, ia:ia + n, :]
    ps1 = np.asarray(out.ps)[ia:ia + n, ia:ia + n]
    pe1 = np.asarray(out.pe)[1:n + 1, :, 1:n + 1]      # (i, k, j)
    # 1 -- dry mass, per column, no area weights (column-local operator).
    # MEASURED (job 9425294 sweep): exactly 0.0; bound = 1e-15 eps guard
    # (measured exactly 0.0).
    gate_scalar("l2e dry mass",
                np.abs(dp1.sum(axis=2) - col0).max() / col0.max(), 1e-15,
                quantity="column dry-mass closure rel")
    # MEASURED (job 9425294 sweep): exactly 0.0; bound = 1e-15 eps guard
    # (measured exactly 0.0).
    gate_scalar("l2e ps closure",
                np.abs(dp1.sum(axis=2) - (ps1 - ptop)).max() / col0.max(),
                1e-15, quantity="ps-delp closure rel")
    # 2 -- per-tracer mass, each tracer separately.
    for iq, x in enumerate(q):
        now = np.asarray(out.q[iq])[ia:ia + n, ia:ia + n, :]
        # MEASURED (job 9425294 sweep): worst tracer2 2.929e-16; bound = measured x
        # 10 = 3.0e-15.
        gate_scalar(f"l2e tracer{iq} mass",
                    np.abs((now * dp1).sum(axis=2) - m0[iq]).max()
                    / np.abs(m0[iq]).max(), 3.0e-15,
                    quantity="per-tracer mass closure rel")
        if iq < 2:      # NON-VACUITY: the non-constant tracers DID move
            assert np.abs(now - x[ia:ia + n, ia:ia + n, :]).max() > 1e-9
    # 3 -- pressure-thickness closure of the RETURNED fields.
    # MEASURED (job 9425294 sweep): exactly 0.0; bound = 1e-15 eps guard
    # (measured exactly 0.0).
    gate_scalar("l2e pe-delp closure",
                _rel(np.diff(pe1, axis=1).transpose(0, 2, 1), dp1), 1e-15)
    # MEASURED (job 9425294 sweep): 8.477e-16; bound = measured x 10 = 8.5e-15.
    gate_scalar("l2e peln-pe closure",
                _rel(np.exp(np.asarray(out.peln)), pe1), 8.5e-15)
    # MEASURED (job 9425294 sweep): 5.253e-16; bound = measured x 10 = 5.3e-15.
    gate_scalar("l2e pk-pe closure",
                _rel(np.asarray(out.pk)[ia:ia + n, ia:ia + n, :],
                     (pe1 ** akap).transpose(0, 2, 1)), 5.3e-15)
    # 4 -- constant field in, constant field out.
    qc1 = np.asarray(out.q[2])[ia:ia + n, ia:ia + n, :]
    # MEASURED (job 9425294 sweep): 2.368e-16; bound = measured x 10 = 2.4e-15.
    gate_scalar("l2e constant tracer", np.abs(qc1 - 3.75).max() / 3.75,
                2.4e-15, quantity="constant-tracer preservation rel")


def test_driver_jax_leaves_the_halo_bitwise_unchanged():
    """The routine writes only its compute window; every cell outside it
    must come back BITWISE identical.  That is the correct halo
    statement for a column-local operator, and it is why the
    conservation diagnostics above exclude the halo."""
    face = _face()[0]
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    before = {k: np.array(v, copy=True) for k, v in face.items()
              if isinstance(v, np.ndarray)}
    out = l2e_j(**{k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                   for k, v in face.items()}, q=[])
    for key, win in (("pt", (slice(ia, ia + n), slice(ia, ia + n))),
                     ("delp", (slice(ia, ia + n), slice(ia, ia + n))),
                     ("pk", (slice(ia, ia + n), slice(ia, ia + n))),
                     ("ps", (slice(ia, ia + n), slice(ia, ia + n))),
                     ("u", (slice(ia, ia + n), slice(ia, ia + n + 1))),
                     ("v", (slice(ia, ia + n + 1), slice(ia, ia + n)))):
        got = np.asarray(getattr(out, key))
        mask = np.ones(got.shape[:2], dtype=bool)
        mask[win] = False
        assert mask.any(), key
        assert np.array_equal(got[mask], before[key][mask]), f"{key} halo"
    # pe: only the interior interfaces k=1..km-1 of i,j = 1..n are written.
    pe_got, pe_ref = np.asarray(out.pe), before["pe"]
    assert np.array_equal(pe_got[:, 0, :], pe_ref[:, 0, :])
    assert np.array_equal(pe_got[:, km, :], pe_ref[:, km, :])
    assert np.array_equal(pe_got[0, :, :], pe_ref[0, :, :])
    assert np.array_equal(pe_got[:, :, 0], pe_ref[:, :, 0])
    assert not np.array_equal(pe_got[1:n + 1, 1:km, 1:n + 1],
                              pe_ref[1:n + 1, 1:km, 1:n + 1])


def test_driver_jax_w_limiter_conserves_weighted_momentum():
    """INVARIANT 5: the limiter REDISTRIBUTES ``w`` between levels and
    must not create it -- the thickness-weighted column momentum
    ``sum_k w*dp2`` is unchanged relative to the same run with the
    limiter off.  The top escape valve (:408-416) DISCARDS momentum by
    design, so the fixture stays below it (asserted)."""
    ia = _face()[0]["ng"]

    def _mk(limiter):
        f = _nh_face(w_const=0.0)
        f["w"][ia + 2, ia + 3, 2] = 200.0
        f["ws"][:] = 0.0
        f["w_limiter"] = limiter
        return f

    outs = {}
    for lim in (False, True):
        f = _mk(lim)
        outs[lim] = l2e_j(
            **{k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
               for k, v in f.items()}, q=[])
    n = _face()[0]["n"]
    win = slice(ia, ia + n)
    mom = {}
    for lim, o in outs.items():
        w = np.asarray(o.w)[win, win, :]
        dp = np.asarray(o.delp)[win, win, :]
        mom[lim] = (w * dp).sum(axis=2)
        if lim:
            assert w.max() <= W_MAX_MAPZ + 1e-9
            assert w.min() >= -abs(2.0 * W_MAX_MAPZ)
    # MEASURED (job 9425294 sweep): exactly 0.0; bound = 1e-15 eps guard
    # (measured exactly 0.0).
    gate_scalar("l2e w-limiter momentum",
                np.abs(mom[True] - mom[False]).max()
                / max(np.abs(mom[False]).max(), 1.0), 1e-15,
                quantity="limiter momentum-redistribution closure rel")
    # NON-VACUITY: the limiter really fired (otherwise this is trivial).
    assert np.asarray(outs[False].w).max() > W_MAX_MAPZ
    assert not np.array_equal(np.asarray(outs[True].w),
                              np.asarray(outs[False].w))


def test_driver_jax_nh_delz_column_height_is_conserved():
    """The NH delz remap moves SPECIFIC VOLUME mass-weighted over an
    unchanged column mass, so the column height integral is preserved."""
    f = _nh_face(w_const=5.0)
    before = f["delz"].sum(axis=2).copy()
    out = l2e_j(**{k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                   for k, v in f.items()}, q=[])
    after = np.asarray(out.delz).sum(axis=2)
    # MEASURED (job 9425294 sweep): 3.339e-16; bound = measured x 10 = 3.4e-15.
    gate_scalar("l2e delz height",
                np.abs(after - before).max() / np.abs(before).max(), 3.4e-15,
                quantity="column height closure rel")
    assert np.asarray(out.delz).max() < 0.0
    # A CONSTANT w with a matching ws BC comes back as that constant
    # (the iv=-2 reconstruction of a constant is the constant).
    ia, n = f["ng"], f["n"]
    ww = np.asarray(out.w)[ia:ia + n, ia:ia + n, :]
    # MEASURED (job 9425294 sweep): exactly 0.0; bound = 1e-15 eps guard
    # (measured exactly 0.0).
    gate_scalar("l2e constant w", np.abs(ww - 5.0).max() / 5.0, 1e-15,
                quantity="constant-w preservation rel")


# ------------------------------------------------------------- gate 4
def test_driver_jax_check_grads_order2_off_switch():
    """Order-2 fwd+rev grads w.r.t. ``pt`` and ``delp`` on a small face.

    SCOPE, stated: those two operands reach the output ONLY through the
    ``pt`` lane (``scalar_profile`` on ln p) and the ``pk``/``pkz``/
    ``pe``/``delp`` algebra.  The u/v remaps read the OLD ``pe`` and the
    winds, and the tracer remap reads the tracers, so NONE of their
    limiter switches is in this gradient -- those lanes are covered by
    the ``map1_ppm``/``map1_q2`` gradient gates, which exercise the same
    code.

    NON-SMOOTH SITES in scope: the ``scalar_profile`` limiters.  ``T_v``
    is built PER COLUMN as the exact cell means of a gentle quadratic in
    that column's own ``ln p`` -- a single vector in layer index was the
    fixture defect that failed this gate in job 9400424, because the
    ln p thicknesses are strongly non-uniform and the column was
    therefore not smooth in the coordinate the profile is built on.  The
    margins are MEASURED below on exactly the input the driver hands
    the profile.
    """
    n, ng, km = 3, 3, KM
    face = _face(n=n, ng=ng, km=km)[0]
    ia = ng
    # peln is (i, k, j); the driver's column order is (j, i).
    pln_cols = face["peln"].transpose(2, 0, 1).reshape(-1, km + 1)
    _, tv_cols = _quadratic_on(pln_cols)                 # (n*n, km)
    tv = tv_cols.reshape(n, n, km).transpose(1, 0, 2)    # -> (i, j, k)
    face["pt"][ia:ia + n, ia:ia + n, :] = tv / face["pkz"]
    q1 = np.zeros((n * n, km + 1))
    q1[:, 1:] = tv_cols
    dlnp = np.zeros((n * n, km + 1))
    dlnp[:, 1:] = np.diff(pln_cols, axis=1)
    _assert_profile_off_switch(
        _jax_profile("scalar", q1, dlnp, km, 1, 9), q1, km)

    kw = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
          for k, v in face.items() if k not in ("pt", "delp")}

    def f(pt_, delp_):
        o = l2e_j(**kw, pt=pt_, delp=delp_, q=[], last_step=False)
        return (jnp.sum(o.pt ** 2) + jnp.sum(o.delp ** 2) / 1e6
                + jnp.sum(o.pkz ** 2) + jnp.sum(o.pk ** 2)
                + jnp.sum(o.pe ** 2) / 1e6 + jnp.sum(o.peln ** 2))

    check_grads(f, (jnp.asarray(face["pt"]), jnp.asarray(face["delp"])),
                order=2, modes=("fwd", "rev"))

# ====================================================================== #
# Adjoint consistency -- <J v, w> == <v, J^T w>, NO finite differences
# ====================================================================== #

def _assert_adjoint_consistent(f, args, seed=0, tol=3.9e-13, label=""):
    """The dot-product test: ``<J v, w> == <v, J^T w>``.

    ``J v`` comes from ``jax.jvp`` (forward mode) and ``J^T w`` from
    ``jax.vjp`` (reverse mode), so this compares the two linearisations
    against EACH OTHER with no finite-difference step anywhere.

    WHY IT IS HERE (job 9400424): eight of nine failures were
    ``check_grads(order=2)`` calls, and three causes were live -- (a) a
    wrong gradient, (b) an order-2 finite difference that cannot resolve
    the array's dynamic range, (c) a fixture sitting on a switching
    surface.  This gate has no step size and no dynamic-range
    sensitivity, and -- unlike ``check_grads`` -- it does not require the
    state to be off a switch at all: the identity holds for whatever
    linearisation JAX built, at any point.  So it isolates (a) from (b)
    and (c) instead of leaving them confounded.

    What it does NOT prove: that the linearisation is the derivative of
    the primal.  A jvp and a vjp of the same WRONG Jacobian agree.  That
    is what the ``check_grads`` gates are for; the two together cover
    each other's blind spot.
    """
    rng = np.random.default_rng(seed)
    args = tuple(jnp.asarray(a) for a in args)
    v = tuple(jnp.asarray(rng.standard_normal(tuple(a.shape))) for a in args)
    y, jv = jax.jvp(f, args, v)
    _, vjp_fn = jax.vjp(f, *args)
    w = jnp.asarray(rng.standard_normal(tuple(y.shape)))
    jtw = vjp_fn(w)
    lhs = float(jnp.sum(jv * w))
    rhs = float(sum(jnp.sum(a * b) for a, b in zip(v, jtw)))
    denom = max(abs(lhs), abs(rhs), 1e-30)
    # MEASURED (job 9425294 sweep): worst residual over every caller is 3.825e-14
    # (the profile kord 9/13 group, both kinds; driver/map/tie all
    # <= 3.9e-16); default tol = measured x 10 = 3.9e-13.
    gate_scalar(f"l2e adjoint {label}", abs(lhs - rhs) / denom, tol,
                quantity="adjoint identity residual")
    # NON-VACUITY: a zero Jacobian satisfies the identity trivially.
    assert abs(lhs) > 0.0, (label, "<J v, w> == 0 -- the gate is vacuous")


@pytest.mark.parametrize("kord", [9, 13])
@pytest.mark.parametrize("kind", ["scalar", "cs"])
def test_profile_jax_adjoint_consistency(kind, kord):
    im, km = 3, KMP
    delp = _delp_col(im, km, seed=41)
    q1 = _smooth_column(im=im, km=km, delp=delp)

    def f(a1_, delp_):
        a4 = jnp.zeros((5, im, km + 1), jnp.float64).at[1].set(a1_)
        if kind == "scalar":
            return scalar_profile_j(a4, delp_, km, 1, kord, T_MIN)[2:5]
        return cs_profile_j(a4, delp_, km, 1, kord)[2:5]

    _assert_adjoint_consistent(f, (q1, delp), label=f"{kind}-{kord}")


@pytest.mark.parametrize("kind", MAP_KINDS)
def test_map_jax_adjoint_consistency(kind):
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    pe2 = _midpoint_targets(pe1, km)
    q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])
    dp2 = _dp_1based(pe2)
    iv = {"map_scalar": 1, "map1_ppm": -1, "map1_q2": 0}[kind]

    def f(pe1_, q_, pe2_):
        if kind == "map_scalar":
            return map_scalar_j(pe1_, q_, pe2_, km, km, iv, 9, T_MIN)
        if kind == "map1_ppm":
            return map1_ppm_j(pe1_, q_, pe2_, km, km, iv, 9)
        return map1_q2_j(pe1_, q_, pe2_, jnp.asarray(dp2), km, km, iv, 9,
                         0.0)

    _assert_adjoint_consistent(f, (pe1, q1, pe2), label=kind)


def test_map_jax_adjoint_consistency_at_an_interface_tie():
    """The gate check_grads CANNOT run: exactly ON the switching surface.

    A finite difference here straddles two branches and is meaningless,
    but the jvp/vjp identity is still a hard requirement -- whatever
    one-sided linearisation JAX picked, forward and reverse mode must
    pick the SAME one.
    """
    im, km, which = 3, KMP, 4
    pe1, pe2 = _tie_edges(im=im, km=km, mode="exact", which=which)
    q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])

    def f(pe1_, q_, pe2_):
        return map1_ppm_j(pe1_, q_, pe2_, km, km, -1, 9)

    _assert_adjoint_consistent(f, (pe1, q1, pe2), label="tie")


def test_driver_jax_adjoint_consistency():
    n, ng, km = 3, 3, KM
    face = _face(n=n, ng=ng, km=km)[0]
    ia = ng
    pln_cols = face["peln"].transpose(2, 0, 1).reshape(-1, km + 1)
    _, tv_cols = _quadratic_on(pln_cols)
    face["pt"][ia:ia + n, ia:ia + n, :] = (
        tv_cols.reshape(n, n, km).transpose(1, 0, 2) / face["pkz"])
    kw = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
          for k, v in face.items() if k not in ("pt", "delp")}

    def f(pt_, delp_):
        o = l2e_j(**kw, pt=pt_, delp=delp_, q=[], last_step=False)
        return jnp.concatenate([o.pt.ravel(), o.pkz.ravel(),
                                o.peln.ravel()])

    _assert_adjoint_consistent(f, (face["pt"], face["delp"]),
                               label="driver")


# ====================================================================== #
# One-sided derivatives AT the switching surface (codex #3)
# ====================================================================== #

# AD-vs-finite-difference agreement, RELATIVE TO THE DERIVATIVE'S OWN
# SCALE.  The first version wrote `<= tol * max(|fd|, 1.0)`, and that
# floor is what failed job 9401521: a scale floor exists to stop a
# NEAR-ZERO reference from blowing up a relative bound, but the
# reference here is 1.04e-3 -- nowhere near zero -- so `max(., 1.0)`
# silently converted a relative test into an ABSOLUTE 1e-12 one, three
# orders of magnitude tighter than the finite difference's own floor.
# The two derivatives agreed to nine significant digits and the gate
# called it a failure.  No floor: normalise by the quantity's own scale.
_AD_FD_REL = 1.5e-6
# The one-sided derivative JUMP vs its closed form; see the assertion.
_D_JUMP_REL = 1.9e-5

def _tie_derivative_probe(im, km, which, q1, pe1, pe2b):
    """``f(t)``: the remapped value of the target layer whose TOP edge is
    ``pe1(which) + t``, for column 0.

    Crossing ``t = 0`` changes which source layer the interval search
    selects -- ``which-1`` for t < 0, ``which`` for t > 0 -- so t=0 IS
    the switching surface.
    """

    def f(t):
        pe2 = jnp.asarray(pe2b).at[:, which].set(
            jnp.asarray(pe1[:, which]) + t)
        out = map1_ppm_j(jnp.asarray(pe1), jnp.asarray(q1), pe2, km, km,
                         -1, 9)
        return out[0, which]

    return f


@pytest.mark.parametrize("column", ["smooth", "limited"])
def test_map_jax_one_sided_derivatives_at_an_interface_tie(column):
    """One-sided derivatives approaching the tie from BOTH sides, each
    checked against the documented branch derivative for that side.

    The derivative of a target value w.r.t. its moving top edge ``x`` is
    ``(q - R(x)) / (X - x)`` where ``R`` is the sub-grid reconstruction
    and ``X`` the target's fixed bottom edge.  So the ONE-SIDED
    derivatives differ by exactly

        d+ - d-  =  (AR(l-1) - AL(l)) / (X - x_tie),

    the reconstruction's JUMP at that source interface.  Both arms are
    asserted:

    * ``smooth``  -- an unlimited column has ``AR(l-1) == AL(l)``
      (fv_mapz.F90:1987-1990 writes both from the SAME interface array),
      so the branch changes but the derivative does NOT.  A lane that
      corrupted the adjoint across the branch would break here.
    * ``limited`` -- where a limiter has flattened a neighbouring cell
      the reconstruction IS discontinuous, so the two one-sided
      derivatives genuinely differ, and the difference must equal the
      closed form above.  The interface is CHOSEN by measuring the jump,
      not assumed.

    Each side is also checked against a same-side finite difference
    taken strictly inside its own branch -- the "documented branch
    derivative" with no model knowledge at all.
    """
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    pe2b = _midpoint_targets(pe1, km)
    if column == "smooth":
        q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])
    else:
        q1 = _column(im=im, km=km, amp=8.0)

    # The reconstruction, from the PUBLIC profile, with the same iv/kord
    # the map call below uses.
    a4 = _jax_profile("cs", q1, _dp_1based(pe1), km, -1, 9)
    jumps = np.abs(a4[3, 0, 2:km] - a4[2, 0, 3:km + 1])   # AR(l-1)-AL(l)
    which = int(np.argmax(jumps)) + 3
    jump_ar_al = a4[3, 0, which - 1] - a4[2, 0, which]
    if column == "smooth":
        assert abs(jump_ar_al) < 1e-9 * abs(a4[1, 0, which]), \
            "the smooth fixture's reconstruction is NOT continuous"
    else:
        assert abs(jump_ar_al) > 1e-3 * abs(a4[1, 0, which]), \
            "no limited interface: the 'limited' arm would be vacuous"

    f = _tie_derivative_probe(im, km, which, q1, pe1, pe2b)
    d, h = 1.0e-2, 1.0e-4               # Pa; [d-h, d+h] stays one-sided
    g_minus = float(jax.grad(f)(-d))
    g_plus = float(jax.grad(f)(+d))
    fd_minus = (float(f(-d + h)) - float(f(-d - h))) / (2.0 * h)
    fd_plus = (float(f(+d + h)) - float(f(+d - h))) / (2.0 * h)
    # measured 1.47e-7 relative, bound = measured x ~10 (job 9401521:
    # 1.0363858e-3 AD vs 1.0363860e-3 FD, |delta| = 1.53e-10).  That
    # residual IS the central difference's own roundoff floor --
    # eps_mach*|f| / (2*h*|f'|) = 2.2e-16*250 / (2e-4*1.04e-3) = 2.6e-7
    # relative -- so any bound below ~1e-6 would be measuring the finite
    # difference's noise rather than the gradient.
    assert abs(g_minus - fd_minus) <= _AD_FD_REL * abs(fd_minus), \
        (g_minus, fd_minus)
    assert abs(g_plus - fd_plus) <= _AD_FD_REL * abs(fd_plus), \
        (g_plus, fd_plus)

    # The PRIMAL is continuous across the tie: both branches integrate the
    # same reconstruction over the same interval, and the spanning arm's
    # first term vanishes as the edge approaches the interface.  A
    # discontinuity would be O(1) in the gap, not O(delta).
    gap = abs(float(f(+h)) - float(f(-h)))
    slope = max(abs(g_minus), abs(g_plus), 1e-30)
    assert gap <= 10.0 * h * slope, (gap, h * slope)

    # The DERIVATIVE jump equals the reconstruction's jump / (X - x_tie).
    x_tie = pe1[0, which]
    big_x = pe2b[0, which + 1]
    predicted = jump_ar_al / (big_x - x_tie)
    # Normalised by BOTH scales, whichever is larger: in the `limited`
    # arm `predicted` is the meaningful one, while in the `smooth` arm it
    # is exactly 0 and the residual is the derivative's own drift across
    # the 2d evaluation interval (an O(d) effect, estimated ~1e-5
    # relative), so the derivative scale has to carry the bound there.
    # MEASURED (job 9425294 sweep): worst 1.889e-06 (limited; smooth 1.521e-07);
    # bound _D_JUMP_REL = measured x 10 = 1.9e-05.
    gate_scalar(f"map tie derivative-jump {column}",
                abs((g_plus - g_minus) - predicted)
                / max(abs(predicted), abs(g_plus)), _D_JUMP_REL,
                quantity="one-sided derivative-jump prediction rel")
    if column == "limited":
        # NON-VACUITY: this really is a derivative switching surface.
        assert abs(g_plus - g_minus) > 1e-6 * max(abs(g_plus), 1e-30)


# ====================================================================== #
# The contained zero-thickness target (codex #1 regression)
# ====================================================================== #

def test_map_jax_zero_thickness_contained_target_is_finite_value_and_grad():
    """A ZERO-thickness target cell lying wholly inside one source layer.

    The NumPy authority RETURNS at ``fv3_native_mapz.py:624`` on the
    contained-cell path without ever dividing by the target thickness,
    so this input is legal there and gives a finite answer.  A JAX twin
    that forms ``qsum/denom`` unconditionally computes 0/0, which is NaN
    in the primal select AND -- worse -- in the adjoint of the branch
    that WAS taken.  The divisor is therefore masked exactly where its
    quotient is discarded.
    """
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])
    pe2 = _midpoint_targets(pe1, km)
    pe2[:, 3] = pe2[:, 2]            # target layer 2 has ZERO thickness
    # It is CONTAINED: `_midpoint_targets` puts pe2(2) at the midpoint of
    # SOURCE CELL 1, so both (coincident) edges sit strictly inside that
    # one source layer -- which is what selects the contained branch.
    assert (pe2[:, 2] > pe1[:, 1]).all() and (pe2[:, 2] < pe1[:, 2]).all()

    ref = map_scalar_n(pe1, q1, pe2, km, km, 1, 9, T_MIN)
    assert np.isfinite(ref).all(), "the NumPy authority is finite here"
    got, ok = map_scalar_j(jnp.asarray(pe1), jnp.asarray(q1),
                           jnp.asarray(pe2), km, km, 1, 9, T_MIN,
                           return_ok=True)
    assert bool(ok)
    assert np.isfinite(np.asarray(got)).all(), "PRIMAL NaN at 0/0"
    # MEASURED (job 9425294 sweep): 4.756e-16; bound = measured x 10 = 4.8e-15.
    gate_scalar("map_scalar zero-thickness contained",
                _rel(np.asarray(got)[:, 1:], ref[:, 1:]), 4.8e-15)

    def loss(q_):
        return jnp.sum(map_scalar_j(jnp.asarray(pe1), q_, jnp.asarray(pe2),
                                    km, km, 1, 9, T_MIN) ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(q1)))
    assert np.isfinite(g).all(), "ADJOINT NaN at 0/0 (the discarded arm)"
    assert np.abs(g).max() > 0.0
    # And the same for the source-layer walk's divisor: a spanning
    # target whose walk passes layers it must not divide in.
    def loss_edges(pe1_, pe2_):
        return jnp.sum(map_scalar_j(pe1_, jnp.asarray(q1), pe2_, km, km,
                                    1, 9, T_MIN) ** 2)

    ge = jax.grad(loss_edges, argnums=(0, 1))(jnp.asarray(pe1),
                                              jnp.asarray(pe2))
    assert all(np.isfinite(np.asarray(x)).all() for x in ge)


def test_map_jax_degenerate_source_layer_the_walk_never_visits():
    """M02: a degenerate SOURCE layer outside the walk's range.

    Sibling of the target-thickness regression above, and the one that
    was MISSING: the mutation census (job 9409097) reverted
    ``dp1_safe = where(active & ~whole, dp1_m, 1)`` in the source-layer
    walk and NOTHING in this file went red, while the same census caught
    the target-thickness sibling immediately.

    The fixture is not the M01 one with a different index, because two
    constraints have to hold at once:

    * the NumPy authority must stay FINITE -- its ``do m=l+1,km`` loop
      never visits source layer 1 here, so it never divides by that
      thickness (``fv3_native_mapz.py:640-651``);
    * the PROFILE builder must never divide by it either, and only the
      ``iv=-2`` solve qualifies: ``_edge_solve_iv_m2`` reads ``delp(1)``
      solely as a NUMERATOR (``fv3_native_mapz.py:201``, whose loop
      starts at k=2, so its divisors are ``delp(2..km)``), whereas the
      default solve divides by it outright at ``:230``
      (``grat = delp(2)/delp(1)``).  So this gate runs the w lane,
      ``map1_ppm(iv=-2)``.

    WHERE THE TEETH ARE: without the mask the m=1 step -- INACTIVE, since
    ``m > ell`` fails for every target here -- evaluates ``dp/0``.  The
    PRIMAL is unharmed, because that step's contribution is discarded by
    the ``where(active, ...)``, which is exactly why the census found no
    failing test.  The defect appears only in reverse mode, where the
    discarded arm's ZERO cotangent multiplies an infinite derivative and
    gives NaN.  The gradient assertions are therefore the ones that
    matter; the primal and parity checks only pin that the input is
    legal.
    """
    im, km = 3, KMP
    pe1, _, _, _ = _lagrangian_edges(im=im, km=km)
    pe1 = pe1.copy()
    pe1[:, 1] = pe1[:, 2]                 # SOURCE layer 1: zero thickness
    # Target grid strictly inside source layers 2..km.  The interval
    # search then never SELECTS layer 1 (which would be a 0/0 in `pl`,
    # non-finite in BOTH lanes and a different defect), and the walk,
    # which starts at ell+1 >= 3, never reaches it either.
    lo = pe1[:, 2] + 0.05 * (pe1[:, 3] - pe1[:, 2])
    hi = pe1[:, km + 1]
    frac = np.linspace(0.0, 1.0, km + 1)[None, :]
    pe2 = np.zeros_like(pe1)
    pe2[:, 1:] = lo[:, None] + (hi - lo)[:, None] * frac
    assert (np.diff(pe2[:, 1:], axis=1) > 0).all(), "target not monotone"
    assert (pe2[:, 1] > pe1[:, 2]).all(), \
        "a target edge could bracket in the degenerate layer"
    q1 = _smooth_column(im=im, km=km, edges=pe1[:, 1:])
    qs = _smooth_column_bottom_value(edges=pe1[:, 1:])

    ref = map1_ppm_n(pe1, q1, pe2, km, km, -2, 9, qs=qs)
    assert np.isfinite(ref).all(), "the NumPy authority is finite here"
    got, ok = map1_ppm_j(jnp.asarray(pe1), jnp.asarray(q1),
                         jnp.asarray(pe2), km, km, -2, 9,
                         qs=jnp.asarray(qs), return_ok=True)
    assert bool(ok)
    assert np.isfinite(np.asarray(got)).all(), "PRIMAL non-finite"
    # MEASURED (job 9425294 sweep): 1.582e-16; bound = measured x 10 = 1.6e-15.
    gate_scalar("map1_ppm degenerate-layer walk",
                _rel(np.asarray(got)[:, 1:], ref[:, 1:]), 1.6e-15)

    def loss(q_, pe1_, pe2_):
        return jnp.sum(map1_ppm_j(pe1_, q_, pe2_, km, km, -2, 9,
                                  qs=jnp.asarray(qs)) ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(q1), jnp.asarray(pe1), jnp.asarray(pe2))
    for name, g in zip(("q1", "pe1", "pe2"), grads):
        assert np.isfinite(np.asarray(g)).all(), (
            f"ADJOINT NaN in d/d{name}: the source-layer walk divided by a "
            f"thickness in a layer the NumPy loop never visits")
    # NON-VACUITY: the gradient is real, not an all-zero pass.
    assert np.abs(np.asarray(grads[0])).max() > 0.0
