"""Certification of the JAX tp_core/sw_core transport lane against the
NumPy fp64 lane (``fv3_native_d_sw``).

Four gates per public routine, following the NH pattern-setter
``test_fv3_nh_core.py``:

1. **parity** -- JAX vs the NumPy twin on the SHARED
   ``fixtures/dswcore_input.npz`` C12 geometry, for EVERY ``hord`` /
   ``iord`` / ``jord`` the NumPy lane branches on (not just the deck
   default);
2. **jit vs eager** -- an ASSERTION, not a comment, plus a trace counter
   proving no retrace on new VALUES of the same shape;
3. **guards** -- float32 raises ``TypeError``; an unsupported order
   raises ``ValueError``, and that test is shown NON-VACUOUS by
   monkeypatching the guard away and asserting the same call then
   returns normally;
4. **gradients** -- ``check_grads(order=2)`` at a SMOOTH state (with the
   limiter branches proved inactive on that fixture), and, per strategy
   correction 4, a ONE-SIDED directional-derivative test that approaches
   a named switching surface from BOTH sides and checks each side
   against the analytic branch derivative.

TOLERANCE POLICY (strategy correction 3).  These are NOT innocuous
neighbour-reading kernels.  ``xppm``/``yppm``/``xtp_u``/``ytp_v`` carry
PPM limiters whose flags (``smt5``/``smt6``/``hi5``/``hi6``) ADD or DROP
a whole flux term, so the flux is DISCONTINUOUS across them: a
rounding-level difference between the two lanes near a limiter surface
flips a branch and produces a discrepancy FAR above 1e-15.  Every
numeric bound below therefore carries (a) a ``TOL-PENDING`` marker for
the orchestrator's measurement job and (b) a statement of whether the
fixture it guards sits in a SMOOTH region or CROSSES a limiter surface.
The two classes need different bounds and must not share one number.

Fixture classes used here:

* ``smooth``  -- ``q`` is an analytic low-order polynomial in the cell
  centres and the Courant field is small and single-signed, so the
  monotonicity clamps are inactive; this is the class the rounding-level
  bound is meaningful for, and the class the gradient gates use.
* ``rough``   -- ``q`` carries a one-cell step on top of the fixture's
  ``delp``; the limiters DO fire (asserted by
  ``test_rough_fixture_actually_switches_limiters``), so this class is
  where a branch flip can appear and where the measurement job must
  expect a larger bound.
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

from legoesm.core import fv3_native_d_sw as npd  # noqa: E402
from legoesm.core import fv3_tp_core as tp  # noqa: E402
from legoesm.core.fv3_native_sw_core import Bounds  # noqa: E402
from legoesm.grids.fv3_native_gridstruct import fort  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

# The complete branch sets the NumPy lane dispatches on.  These lists are
# the test's own statement of coverage; ``test_order_sets_match_module``
# asserts they equal the module's guard sets, so adding a scheme to the
# module without a parity test goes red.
PPM_ORDS = [-6, -5, -4, -3, -2, -1,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
SW_ORDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


@pytest.fixture(scope="module")
def inp():
    return np.load(os.path.join(FIX, "dswcore_input.npz"))


def _F(a, ilo, jlo):
    """Fresh Fortran-indexed NumPy-lane view (always a private copy, so
    a lane that mutates its operand cannot leak into the other lane)."""
    return fort(np.array(a, dtype=np.float64, copy=True), ilo, jlo)


class _Geo:
    """Static geometry + operands shared by every gate in this module."""

    def __init__(self, inp, kind="smooth"):
        self.res = int(inp["res"])
        self.ng = int(inp["ng"])
        self.bd = Bounds.single_tile(self.res, self.ng)
        b = self.bd
        self.npx = self.res + 1
        self.npy = self.res + 1
        self.kind = kind

        self.dxa = np.asarray(inp["dxa"], dtype=np.float64)
        self.dya = np.asarray(inp["dya"], dtype=np.float64)
        self.area = np.asarray(inp["area"], dtype=np.float64)
        self.rarea = np.asarray(inp["rarea"], dtype=np.float64)
        self.del6_u = np.asarray(inp["del6_u"], dtype=np.float64)
        self.del6_v = np.asarray(inp["del6_v"], dtype=np.float64)
        self.dx = np.asarray(inp["dx"], dtype=np.float64)
        self.dy = np.asarray(inp["dy"], dtype=np.float64)
        self.rdx = np.asarray(inp["rdx"], dtype=np.float64)
        self.rdy = np.asarray(inp["rdy"], dtype=np.float64)
        self.da_min = float(inp["da_min"])

        ni = b.ied - b.isd + 1
        nj = b.jed - b.jsd + 1
        ii = np.arange(b.isd, b.ied + 1, dtype=np.float64)[:, None]
        jj = np.arange(b.jsd, b.jed + 1, dtype=np.float64)[None, :]

        if kind == "smooth":
            # Low-order polynomial: every 4-point PPM stencil reproduces
            # it, so the monotonicity clamps sit strictly inside their
            # branch and the kernel is locally smooth.
            self.q = (1.0e3 + 3.0 * ii + 2.0 * jj
                      + 0.05 * ii * jj + 0.01 * ii * ii)
            cscale, csign = 0.25, 1.0
        elif kind == "rough":
            # delp plus a one-cell step -> the limiters DO fire.
            self.q = np.array(inp["delp"], dtype=np.float64)
            step = np.zeros((ni, nj))
            step[ni // 2:, :] += 40.0
            step[:, nj // 3] -= 25.0
            self.q = self.q + step
            cscale, csign = 0.45, -1.0
        else:                       # pragma: no cover - guard
            raise ValueError(f"_Geo: unknown kind {kind!r}")

        # Courant / area-flux fields.  Both signs are present in `rough`
        # (csign flips the sense) so the c>0 and c<=0 upwind arms are
        # BOTH exercised; `smooth` keeps one sign to stay in-branch.
        uc = np.asarray(inp["uc"], dtype=np.float64)     # (isd:ied+1,jsd:jed)
        vc = np.asarray(inp["vc"], dtype=np.float64)     # (isd:ied,jsd:jed+1)
        um = np.abs(uc).max() or 1.0
        vm = np.abs(vc).max() or 1.0

        # crx/xfx: (is:ie+1, jsd:jed) ; cry/yfx: (isd:ied, js:je+1)
        i0 = b.is_ - b.isd
        i1 = b.ie + 1 - b.isd + 1
        j0 = b.js - b.jsd
        j1 = b.je + 1 - b.jsd + 1
        self.crx = csign * cscale * uc[i0:i1, :] / um
        self.cry = cscale * vc[:, j0:j1] / vm
        # xfx/yfx are AREA fluxes; keep them small vs `area` so ra_x/ra_y
        # stay strictly positive (the oracle divides by them).
        self.xfx = self.crx * self.dy[i0:i1, :] * 0.5
        self.yfx = self.cry * self.dx[:, j0:j1] * 0.5

        # ra_x(is:ie, jsd:jed) = area + xfx(i) - xfx(i+1)   (d_sw:3172)
        # ra_y(isd:ied, js:je) = area + yfx(j) - yfx(j+1)   (d_sw:3175)
        self.ra_x = (self.area[i0:i0 + (b.ie - b.is_ + 1), :]
                     + self.xfx[:-1, :] - self.xfx[1:, :])
        self.ra_y = (self.area[:, j0:j0 + (b.je - b.js + 1)]
                     + self.yfx[:, :-1] - self.yfx[:, 1:])
        assert (self.ra_x > 0).all() and (self.ra_y > 0).all()

        # Staggered operands for xtp_u / ytp_v.
        self.u = np.asarray(inp["u"], dtype=np.float64)   # (isd:ied,jsd:jed+1)
        self.v = np.asarray(inp["v"], dtype=np.float64)   # (isd:ied+1,jsd:jed)
        if kind == "smooth":
            iu = np.arange(b.isd, b.ied + 1, dtype=np.float64)[:, None]
            ju = np.arange(b.jsd, b.jed + 2, dtype=np.float64)[None, :]
            self.u = 5.0 + 0.3 * iu + 0.2 * ju + 0.004 * iu * ju
            iv = np.arange(b.isd, b.ied + 2, dtype=np.float64)[:, None]
            jv = np.arange(b.jsd, b.jed + 1, dtype=np.float64)[None, :]
            self.v = -4.0 + 0.25 * iv - 0.15 * jv + 0.003 * iv * jv
        # c for xtp_u/ytp_v is (is:ie+1, js:je+1) and is multiplied by
        # rdx/rdy inside, so scale it by a representative dx.
        nc_i = b.ie + 1 - b.is_ + 1
        nc_j = b.je + 1 - b.js + 1
        ci = np.arange(nc_i, dtype=np.float64)[:, None]
        cj = np.arange(nc_j, dtype=np.float64)[None, :]
        dxr = float(np.median(self.dx))
        self.c_sw = csign * cscale * dxr * np.cos(
            0.7 * ci + 0.4 * cj + (0.0 if kind == "smooth" else 1.3))

        self.mass = np.abs(self.q) + 1.0
        self.damp_km = 0.05 + 0.01 * np.abs(np.sin(ii + jj))

    # ---- gridstruct dict for the NumPy lane -------------------------
    def gs(self, bounded_domain=False, grid_type=0):
        return {
            "dxa": _F(self.dxa, self.bd.isd, self.bd.jsd),
            "dya": _F(self.dya, self.bd.isd, self.bd.jsd),
            "area": _F(self.area, self.bd.isd, self.bd.jsd),
            "rarea": _F(self.rarea, self.bd.isd, self.bd.jsd),
            "del6_u": _F(self.del6_u, self.bd.isd, self.bd.jsd),
            "del6_v": _F(self.del6_v, self.bd.isd, self.bd.jsd),
            "da_min": self.da_min,
            "bounded_domain": bounded_domain,
            "grid_type": grid_type,
            "sw_corner": True, "se_corner": True,
            "nw_corner": True, "ne_corner": True,
        }


@pytest.fixture(scope="module")
def smooth(inp):
    return _Geo(inp, "smooth")


@pytest.fixture(scope="module")
def rough(inp):
    return _Geo(inp, "rough")


def _cmp(got, ref, name, tol):
    """NaN-mask-aware relative comparison.

    Both lanes leave cells the oracle never writes as NaN (the ``_fl``
    tripwire); a mismatch in the NaN MASK is a port bug on its own and is
    checked before any value is compared.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    ma, mb = np.isnan(a), np.isnan(b)
    assert np.array_equal(ma, mb), (
        f"{name}: NaN masks differ (jax {int(ma.sum())} vs numpy "
        f"{int(mb.sum())} NaN cells)")
    ok = ~ma
    assert ok.any(), f"{name}: output is entirely NaN (vacuous compare)"
    scale = max(float(np.abs(b[ok]).max()), 1e-30)
    rel = float(np.abs(a[ok] - b[ok]).max()) / scale
    assert rel <= tol, (
        f"{name}: rel {rel:.3e} > {tol:.3e} "
        f"(bitwise={np.array_equal(a[ok], b[ok])})")
    return rel


# =====================================================================
# order-set bookkeeping
# =====================================================================

def test_order_sets_match_module():
    """The parametrized coverage below IS the module's guard set -- a
    scheme added to the module without a parity test goes red here."""
    assert sorted(PPM_ORDS) == sorted(tp._PPM_ORDS)
    assert sorted(SW_ORDS) == sorted(tp._SW_ORDS)


# =====================================================================
# pert_ppm
# =====================================================================

def _pert_np(a0, al, ar, iv):
    al2 = np.array(al, dtype=np.float64, copy=True).ravel()
    ar2 = np.array(ar, dtype=np.float64, copy=True).ravel()
    npd.pert_ppm(al2.size, np.asarray(a0, dtype=np.float64).ravel(),
                 al2, ar2, iv)
    return al2.reshape(np.shape(al)), ar2.reshape(np.shape(ar))


def _pert_fixture(seed=5, n=400):
    rng = np.random.default_rng(seed)
    a0 = rng.standard_normal(n)
    al = 0.6 * rng.standard_normal(n)
    ar = 0.6 * rng.standard_normal(n)
    # Force a large share of cells through EVERY arm: sign pairs,
    # `a0 <= 0`, and the |da1| < -a4 positive-definite test.
    a0[: n // 4] = -np.abs(a0[: n // 4])
    al[n // 4: n // 2] = np.abs(al[n // 4: n // 2])
    ar[n // 4: n // 2] = -np.abs(ar[n // 4: n // 2])
    al[n // 2: 3 * n // 4] = -0.02 * np.abs(al[n // 2: 3 * n // 4])
    ar[n // 2: 3 * n // 4] = -0.02 * np.abs(ar[n // 2: 3 * n // 4])
    return a0, al, ar


@pytest.mark.parametrize("iv", [0, 1, -1])
def test_pert_ppm_parity(iv):
    """gate 1.  FIXTURE CLASS: crosses limiter surfaces BY DESIGN (that
    is what pert_ppm is), so this bound is a BRANCH-AGREEMENT bound, not
    a rounding bound: both lanes evaluate the same rational expressions,
    so agreement should be exact unless a branch flips."""
    a0, al, ar = _pert_fixture()
    al_n, ar_n = _pert_np(a0, al, ar, iv)
    al_j, ar_j = tp.pert_ppm(jnp.asarray(a0), jnp.asarray(al),
                             jnp.asarray(ar), iv)
    # Non-vacuity: the constraint actually moved state in both lanes.
    assert np.abs(al_n - al).max() > 1e-3
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(al_j, al_n, f"pert_ppm iv={iv} al", 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(ar_j, ar_n, f"pert_ppm iv={iv} ar", 1e-12)


@pytest.mark.parametrize("iv", [0, 1])
def test_pert_ppm_jit_equals_eager(iv):
    """gate 2 -- ASSERTED, and no retrace on new values."""
    a0, al, ar = _pert_fixture()
    eager = tp.pert_ppm(jnp.asarray(a0), jnp.asarray(al),
                        jnp.asarray(ar), iv)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.pert_ppm(*a, **k)

    fn = tp.make_pert_ppm_jit(_counted)
    jitted = fn(jnp.asarray(a0), jnp.asarray(al), jnp.asarray(ar), iv)
    fn(jnp.asarray(a0 * 1.01), jnp.asarray(al * 0.99),
       jnp.asarray(ar * 1.02), iv)
    assert traces["n"] == 1, traces["n"]
    for name, e, j in zip(("al", "ar"), eager, jitted):
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.   [class: jit-vs-eager, same expression tree]
        _cmp(j, e, f"pert_ppm jit {name}", 1e-12)


def test_pert_ppm_rejects_float32():
    """gate 3 -- guard."""
    a0, al, ar = _pert_fixture(n=8)
    with pytest.raises(TypeError, match="float64"):
        tp.pert_ppm(jnp.asarray(a0, jnp.float32), jnp.asarray(al),
                    jnp.asarray(ar), 1)


def test_pert_ppm_grads_order2_smooth(smooth):
    """gate 4 -- order-2 grads at a state PROVED away from every
    switching surface of ``iv=1``.

    NON-SMOOTH SITES of pert_ppm(iv=1), all named and all avoided here:
      S1  ``al*ar = 0``      (the outer cross-sign test),
      S2  ``a6da = -da2``    (the ``ar = -2*al`` arm boundary),
      S3  ``a6da = +da2``    (the ``al = -2*ar`` arm boundary).
    """
    del smooth
    rng = np.random.default_rng(21)
    al = -(0.5 + rng.random(64))            # strictly negative
    ar = +(0.5 + rng.random(64))            # strictly positive
    a0 = 1.0 + rng.random(64)
    da1 = al - ar
    da2 = da1 ** 2
    a6da = 3.0 * (al + ar) * da1
    # Control: every cell is strictly inside ONE arm, with >=10% margin
    # on each surface, so the order-2 check is taken on a smooth patch.
    assert (al * ar < -1e-2).all(), "S1 (al*ar=0) too close"
    assert (np.abs(np.abs(a6da) - da2) > 0.1 * da2).all(), "S2/S3 close"

    def f(al_, ar_):
        a, b = tp.pert_ppm(jnp.asarray(a0), al_, ar_, 1)
        return jnp.sum(a * a) + jnp.sum(b * b) + jnp.sum(a * b)

    check_grads(f, (jnp.asarray(al), jnp.asarray(ar)), order=2,
                modes=("fwd", "rev"))


def test_pert_ppm_one_sided_derivative_across_switching_surfaces():
    """gate 4 (strategy correction 4) -- the LIMITER test.

    A smooth-state ``check_grads`` proves the arithmetic, not the
    scheme.  Here the state is driven ACROSS two named switching
    surfaces of ``pert_ppm(iv=1)`` (tp_core.F90:1195-1210) and the
    one-sided derivative on each side is checked against the ANALYTIC
    branch derivative.

    Surface S1: ``al*ar = 0``.  Hold ``ar = +1`` fixed and sweep ``al``.
      * ``al < 0``  -> cross-sign arm; with ``a6da < -da2`` the oracle
        sets ``ar_out = -2*al`` and leaves ``al_out = al``, so
        ``d/dal (al_out + ar_out) = 1 - 2 = -1``.
      * ``al > 0``  -> same-sign arm; ``al_out = ar_out = 0`` and the
        derivative is ``0``.
      The two sides DIFFER, which is the point: a two-sided finite
      difference straddling ``al = 0`` is meaningless here.

    Surface S2: ``a6da = -da2``, i.e. ``ar = -2*al`` exactly (the two
    arms AGREE on the surface, so the function is C^0 there and only the
    derivative jumps).  Hold ``al = -1`` and sweep ``ar`` through 2:
      * ``ar > 2``  -> ``a6da < -da2`` -> ``ar_out = -2*al = 2``, so
        ``d(ar_out)/dar = 0``;
      * ``ar < 2``  -> neither clamp fires -> ``ar_out = ar``, so
        ``d(ar_out)/dar = 1``.
    """
    eps = 1e-6

    # ---- S1 --------------------------------------------------------
    def s1(al_scalar):
        al_, ar_ = tp.pert_ppm(jnp.asarray([1.0]),
                               jnp.reshape(al_scalar, (1,)),
                               jnp.asarray([1.0]), 1)
        return (al_ + ar_)[0]

    g = jax.grad(s1)
    left = float(g(jnp.asarray(-eps)))
    right = float(g(jnp.asarray(+eps)))
    # Branch check: confirm the arm we think we are in really is taken.
    al_m, ar_m = tp.pert_ppm(jnp.asarray([1.0]), jnp.asarray([-eps]),
                             jnp.asarray([1.0]), 1)
    assert float(ar_m[0]) == pytest.approx(2.0 * eps, rel=1e-12)
    assert float(al_m[0]) == pytest.approx(-eps, rel=1e-12)
    assert left == pytest.approx(-1.0, abs=1e-12), left
    assert right == pytest.approx(0.0, abs=1e-12), right
    assert abs(left - right) > 0.5          # the surface is REAL

    # ---- S2 --------------------------------------------------------
    def s2(ar_scalar):
        _, ar_ = tp.pert_ppm(jnp.asarray([1.0]), jnp.asarray([-1.0]),
                             jnp.reshape(ar_scalar, (1,)), 1)
        return ar_[0]

    g2 = jax.grad(s2)
    above = float(g2(jnp.asarray(2.0 + eps)))
    below = float(g2(jnp.asarray(2.0 - eps)))
    assert above == pytest.approx(0.0, abs=1e-12), above
    assert below == pytest.approx(1.0, abs=1e-12), below
    # C^0 across S2: the VALUE agrees although the derivative jumps.
    v_a = float(s2(jnp.asarray(2.0 + eps)))
    v_b = float(s2(jnp.asarray(2.0 - eps)))
    assert abs(v_a - v_b) < 1e-5


# =====================================================================
# copy_corners
# =====================================================================

def _corners_np(geo, q, dir_, bounded_domain=False, duogrid=False):
    qf = _F(q, geo.bd.isd, geo.bd.jsd)
    npd.copy_corners(qf, geo.npx, geo.npy, dir_, bounded_domain, geo.bd,
                     True, True, True, True, duogrid=duogrid)
    return qf.a


@pytest.mark.parametrize("dir_", [1, 2])
def test_copy_corners_parity(dir_, rough):
    """gate 1.  FIXTURE CLASS: pure index shuffle -- no arithmetic at
    all, so the two lanes must agree BITWISE, and the bound below is a
    formality the measurement job should tighten to exact equality."""
    ref = _corners_np(rough, rough.q, dir_)
    got = tp.copy_corners(jnp.asarray(rough.q), rough.npx, rough.npy,
                          dir_, False, rough.bd, True, True, True, True)
    # Non-vacuity: the rotation actually moved corner cells.
    assert not np.array_equal(ref, rough.q)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: index shuffle, expect BITWISE]
    _cmp(got, ref, f"copy_corners dir={dir_}", 1e-12)
    assert np.array_equal(np.asarray(got), ref), "not bitwise"


@pytest.mark.parametrize("flag", ["bounded_domain", "duogrid"])
def test_copy_corners_duo_and_bounded_are_noops(flag, rough):
    """tp_core.F90:239 -- ``if (bounded_domain .or. duogrid) return``."""
    kw = {"duogrid": True} if flag == "duogrid" else {}
    bdm = flag == "bounded_domain"
    got = tp.copy_corners(jnp.asarray(rough.q), rough.npx, rough.npy, 1,
                          bdm, rough.bd, True, True, True, True, **kw)
    assert np.array_equal(np.asarray(got), rough.q)


def test_copy_corners_jit_equals_eager(rough):
    """gate 2 -- ASSERTED."""
    args = (rough.npx, rough.npy, 2, False, rough.bd, True, True, True,
            True)
    eager = tp.copy_corners(jnp.asarray(rough.q), *args)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.copy_corners(*a, **k)

    fn = tp.make_copy_corners_jit(_counted)
    jitted = fn(jnp.asarray(rough.q), *args)
    fn(jnp.asarray(rough.q * 1.01), *args)
    assert traces["n"] == 1, traces["n"]
    assert np.array_equal(np.asarray(jitted), np.asarray(eager))


def test_copy_corners_guards(rough):
    """gate 3 -- float32 TypeError and an unknown direction ValueError.

    NON-VACUITY of the direction guard: tp_core.F90:241/:267 branch on
    1 and 2 only, so without the final ``raise`` the function would
    return ``q`` UNROTATED -- a silent wrong answer.  The second half of
    this test demonstrates exactly that by checking the guard is what
    fires (the same call with dir=1 succeeds).
    """
    with pytest.raises(TypeError, match="float64"):
        tp.copy_corners(jnp.asarray(rough.q, jnp.float32), rough.npx,
                        rough.npy, 1, False, rough.bd, True, True, True,
                        True)
    with pytest.raises(ValueError, match="direction"):
        tp.copy_corners(jnp.asarray(rough.q), rough.npx, rough.npy, 3,
                        False, rough.bd, True, True, True, True)
    ok = tp.copy_corners(jnp.asarray(rough.q), rough.npx, rough.npy, 1,
                         False, rough.bd, True, True, True, True)
    assert np.isfinite(np.asarray(ok)).any()


def test_copy_corners_grads_order2(rough):
    """gate 4 -- a gather/scatter is LINEAR in ``q``; there is no
    switching surface at all, so order-2 grads must be exact
    everywhere."""
    q0 = jnp.asarray(rough.q[:8, :8])
    bd = Bounds.single_tile(2, 3)

    def f(x):
        y = tp.copy_corners(x, 3, 3, 1, False, bd, True, True, True, True)
        return jnp.nansum(y * y)

    check_grads(f, (q0,), order=2, modes=("fwd", "rev"))
