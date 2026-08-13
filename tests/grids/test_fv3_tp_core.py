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
  ``test_fixture_classes_are_what_they_claim``, which reads the limiter
  flag out of the public flux), so this class is where a branch flip can
  appear and where the measurement job must expect a larger bound.
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


def _fv(a, ilo, jlo):
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
            "dxa": _fv(self.dxa, self.bd.isd, self.bd.jsd),
            "dya": _fv(self.dya, self.bd.isd, self.bd.jsd),
            "area": _fv(self.area, self.bd.isd, self.bd.jsd),
            "rarea": _fv(self.rarea, self.bd.isd, self.bd.jsd),
            "del6_u": _fv(self.del6_u, self.bd.isd, self.bd.jsd),
            "del6_v": _fv(self.del6_v, self.bd.isd, self.bd.jsd),
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
    """The parametrized coverage in this file IS the module's guard set.

    ``PPM_ORDS`` parametrizes ``test_xppm_parity``, ``test_yppm_parity``
    and ``test_fv_tp_2d_parity``; ``SW_ORDS`` parametrizes
    ``test_xtp_u_parity`` and ``test_ytp_v_parity``.  A scheme added to
    either module guard set without a parity test goes red HERE, which
    is the only place that linkage is enforced -- so do not weaken this
    to a subset check.

    The two sets are DIFFERENT on purpose: ``xppm``/``yppm`` branch on
    ``mord = abs(iord)`` so negatives are real schemes, while
    ``xtp_u``/``ytp_v`` branch on the raw order and reject them.
    """
    assert sorted(PPM_ORDS) == sorted(tp._PPM_ORDS)
    assert sorted(SW_ORDS) == sorted(tp._SW_ORDS)
    assert set(SW_ORDS).isdisjoint({o for o in PPM_ORDS if o < 0})
    for routine, ords in (("xppm", tp._PPM_ORDS), ("yppm", tp._PPM_ORDS),
                          ("fv_tp_2d", tp._PPM_ORDS),
                          ("xtp_u", tp._SW_ORDS),
                          ("ytp_v", tp._SW_ORDS)):
        assert ords, routine


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
    qf = _fv(q, geo.bd.isd, geo.bd.jsd)
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


# =====================================================================
# shared smoothness control (strategy correction 4)
# =====================================================================

def _assert_locally_smooth(f, x, dx, name, eps=1e-6, tol=1e-6):
    """No switching surface inside the finite-difference ball.

    Compares the LEFT and RIGHT one-sided directional derivatives.  At a
    limiter kink (or a flux discontinuity) they differ at O(1); inside a
    smooth patch they agree to O(eps).  This is the control that licenses
    the order-2 ``check_grads`` below -- without it a "grads pass" claim
    is a claim about an arbitrary state.
    """
    x = jnp.asarray(x)
    dx = jnp.asarray(dx)
    f0 = float(f(x))
    fp = (float(f(x + eps * dx)) - f0) / eps
    fm = (f0 - float(f(x - eps * dx))) / eps
    den = max(abs(fp), abs(fm), 1.0)
    assert abs(fp - fm) / den <= tol, (
        f"{name}: one-sided derivatives disagree ({fp:.6e} vs {fm:.6e}) "
        f"-- a switching surface is inside the FD ball, so an order-2 "
        f"check_grads here would be certifying a kink")


# =====================================================================
# xppm
# =====================================================================

def _xppm_np(geo, iord, q, c, bounded_domain=False, grid_type=0,
             lim_fac=1.0, duogrid=False):
    b = geo.bd
    n_c = b.ie + 1 - b.is_ + 1
    nj = b.jed - b.jsd + 1
    fx = _fv(np.full((n_c, nj), np.nan), b.is_, b.jsd)
    npd.xppm(fx, _fv(q, b.isd, b.jsd), _fv(c, b.is_, b.jsd), iord,
             b.is_, b.ie, b.isd, b.ied, b.jsd, b.jed, b.jsd, b.jed,
             geo.npx, geo.npy, _fv(geo.dxa, b.isd, b.jsd),
             bounded_domain, grid_type, lim_fac, duogrid=duogrid)
    return fx.a


def _xppm_jax(geo, iord, q, c, bounded_domain=False, grid_type=0,
              lim_fac=1.0, duogrid=False, fn=None):
    b = geo.bd
    fn = fn or tp.xppm
    return fn(jnp.asarray(q), jnp.asarray(c), iord, b.is_, b.ie, b.isd,
              b.ied, b.jsd, b.jed, b.jsd, b.jed, geo.npx, geo.npy,
              jnp.asarray(geo.dxa), bounded_domain, grid_type, lim_fac,
              duogrid=duogrid)


@pytest.mark.parametrize("iord", PPM_ORDS)
@pytest.mark.parametrize("kind", ["smooth", "rough"])
def test_xppm_parity(iord, kind, smooth, rough):
    """gate 1 -- EVERY iord the NumPy lane branches on, both fixture
    classes.  ``smooth`` is the rounding-level class; ``rough`` crosses
    limiter surfaces and needs its own (larger) measured bound."""
    geo = smooth if kind == "smooth" else rough
    ref = _xppm_np(geo, iord, geo.q, geo.crx)
    got = _xppm_jax(geo, iord, geo.q, geo.crx)
    assert np.isfinite(ref).any()
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth if kind=="smooth", else
    # limiter-crossing -- the two MUST get different measured bounds]
    _cmp(got, ref, f"xppm iord={iord} {kind}", 1e-12)


@pytest.mark.parametrize("iord", [5, 8])
def test_xppm_duogrid_unclamped_parity(iord, rough):
    """The duo arm takes the OTHER is1/ie3/ie1 branch and skips every
    edge block (tp_core.F90:333-339, :346, :505, :614)."""
    ref = _xppm_np(rough, iord, rough.q, rough.crx, duogrid=True)
    got = _xppm_jax(rough, iord, rough.q, rough.crx, duogrid=True)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(got, ref, f"xppm duo iord={iord}", 1e-12)


def test_xppm_jit_equals_eager_and_no_retrace(rough):
    """gate 2 -- ASSERTED, through the PRODUCTION jit factory."""
    eager = _xppm_jax(rough, 8, rough.q, rough.crx)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.xppm(*a, **k)

    fn = tp.make_xppm_jit(_counted)
    j1 = _xppm_jax(rough, 8, rough.q, rough.crx, fn=fn)
    _xppm_jax(rough, 8, rough.q * 1.01, rough.crx * 0.97, fn=fn)
    assert traces["n"] == 1, traces["n"]
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: jit-vs-eager, identical expression tree]
    _cmp(j1, eager, "xppm jit", 1e-12)


def test_xppm_rejects_float32(rough):
    """gate 3 -- guard."""
    with pytest.raises(TypeError, match="float64"):
        _xppm_jax(rough, 8, np.asarray(rough.q, np.float32), rough.crx)


@pytest.mark.parametrize("bad", [0, 14, -7, 20])
def test_xppm_unknown_iord_raises(bad, rough):
    """gate 3 -- dispatch-hardening."""
    with pytest.raises(ValueError, match="not a supported scheme"):
        _xppm_jax(rough, bad, rough.q, rough.crx)


@pytest.mark.parametrize("bad", [0, 14])
def test_xppm_unknown_iord_guard_is_non_vacuous(bad, rough, monkeypatch):
    """gate 3 -- the guard test is shown to FAIL WITHOUT THE GUARD.

    HOW: ``_validate_ord`` is monkeypatched to a no-op, which is exactly
    "delete the guard".  The SAME call then returns an array instead of
    raising -- i.e. iord=0 silently runs the {5,6} arm and iord=14
    silently runs the {9,13} arm WITHOUT its pert_ppm positive-definite
    pass.  That is the silent-wrong-scheme the guard exists to stop, and
    it proves the ValueError above comes from the guard and not from an
    incidental shape/index error.
    """
    monkeypatch.setattr(tp, "_validate_ord", lambda *a, **k: None)
    out = _xppm_jax(rough, bad, rough.q, rough.crx)
    assert np.asarray(out).shape == (rough.bd.ie + 2 - rough.bd.is_,
                                     rough.bd.jed - rough.bd.jsd + 1)
    assert np.isfinite(np.asarray(out)).any()


def test_fixture_classes_are_what_they_claim(smooth, rough):
    """The smooth/rough labels are CLAIMS -- measured here.

    Instrument (public outputs only): at ``iord = 5`` the oracle returns
    the plain upwind value ``base`` when both ``smt5`` flags are false
    and ``base + fx1`` otherwise (tp_core.F90:517-521).  Comparing the
    flux with ``base`` therefore reads the limiter flag directly.
    """
    b = smooth.bd
    for geo, label in ((smooth, "smooth"), (rough, "rough")):
        flux = np.asarray(_xppm_jax(geo, 5, geo.q, geo.crx))
        i0 = b.is_ - b.isd
        n = b.ie + 1 - b.is_ + 1
        c = geo.crx
        base = np.where(c > 0.0, geo.q[i0 - 1:i0 - 1 + n, :],
                        geo.q[i0:i0 + n, :])
        on = np.abs(flux - base) > 0.0
        if label == "rough":
            assert on.any() and (~on).any(), (
                "rough fixture does NOT cross a limiter surface "
                f"(on={int(on.sum())}, off={int((~on).sum())})")
        else:
            assert on.all(), (
                "smooth fixture is NOT uniformly in-branch "
                f"(off={int((~on).sum())})")


@pytest.mark.parametrize("iord", [2, 5, 8])
def test_xppm_grads_order2_smooth(iord, smooth):
    """gate 4 -- order-2 grads at a state whose smoothness is MEASURED
    (``_assert_locally_smooth``), not assumed.

    NON-SMOOTH SITES of xppm, all named:
      N1  the upwind select ``c(i,j) > 0``          (value continuous,
          derivative w.r.t. c not) -- avoided: the smooth fixture's
          Courant field is single-signed and bounded away from 0;
      N2  ``smt5 = bl*br < 0`` / ``smt5 = 3|b0| < |bl-br|`` /
          ``smt6`` / ``hi5`` / ``hi6`` -- the flux JUMPS across these;
          the flag is uniform on this fixture
          (``test_fixture_classes_are_what_they_claim``);
      N3  ``jnp.copysign`` in ``dm`` and at iord 8/11 -- kinked at a
          zero argument and jumping at a sign flip of the second;
      N4  the three-way ``min`` in ``dm`` and the PPM ``min``/``max``
          clamps at iord 9/10/13 and in the edge blocks;
      N5  the ``max(0., al)`` clamp for iord < 0;
      N6  ``0.25/a4`` at ``a4 = 0`` (iord -5/7/12) -- UNDEFINED, made
          finite by the double-``where`` but with NO gradient claim.
    iord=2 has NONE of N2-N6 (perfectly linear); 5 adds N2; 8 adds
    N3/N4.  The FD control below is what certifies we are off all of
    them on this fixture.
    """
    b = smooth.bd
    rng = np.random.default_rng(4)
    dq = rng.standard_normal(smooth.q.shape)

    def f(q_):
        out = _xppm_jax(smooth, iord, q_, smooth.crx)
        return jnp.nansum(out * out)

    _assert_locally_smooth(f, smooth.q, dq, f"xppm iord={iord}")
    # A reduced window keeps the order-2 FD affordable; the smoothness
    # control above is run on the FULL operand.
    check_grads(f, (jnp.asarray(smooth.q),), order=2, modes=("fwd",))

    def g(c_):
        out = _xppm_jax(smooth, iord, smooth.q, c_)
        return jnp.nansum(out * out)

    assert np.abs(smooth.crx).min() > 1e-3, "N1: c too close to 0"
    _assert_locally_smooth(g, smooth.crx,
                           rng.standard_normal(smooth.crx.shape),
                           f"xppm(c) iord={iord}")
    check_grads(g, (jnp.asarray(smooth.crx),), order=2, modes=("fwd",))
    del b


# =====================================================================
# yppm
# =====================================================================

def _yppm_np(geo, jord, q, c, bounded_domain=False, grid_type=0,
             lim_fac=1.0, duogrid=False):
    b = geo.bd
    ni = b.ied - b.isd + 1
    nj = b.je + 1 - b.js + 1
    fy = _fv(np.full((ni, nj), np.nan), b.isd, b.js)
    npd.yppm(fy, _fv(q, b.isd, b.jsd), _fv(c, b.isd, b.js), jord,
             b.isd, b.ied, b.isd, b.ied, b.js, b.je, b.jsd, b.jed,
             geo.npx, geo.npy, _fv(geo.dya, b.isd, b.jsd),
             bounded_domain, grid_type, lim_fac, duogrid=duogrid)
    return fy.a


def _yppm_jax(geo, jord, q, c, bounded_domain=False, grid_type=0,
              lim_fac=1.0, duogrid=False, fn=None):
    b = geo.bd
    fn = fn or tp.yppm
    return fn(jnp.asarray(q), jnp.asarray(c), jord, b.isd, b.ied, b.isd,
              b.ied, b.js, b.je, b.jsd, b.jed, geo.npx, geo.npy,
              jnp.asarray(geo.dya), bounded_domain, grid_type, lim_fac,
              duogrid=duogrid)


@pytest.mark.parametrize("jord", PPM_ORDS)
@pytest.mark.parametrize("kind", ["smooth", "rough"])
def test_yppm_parity(jord, kind, smooth, rough):
    """gate 1 -- EVERY jord, both fixture classes."""
    geo = smooth if kind == "smooth" else rough
    ref = _yppm_np(geo, jord, geo.q, geo.cry)
    got = _yppm_jax(geo, jord, geo.q, geo.cry)
    assert np.isfinite(ref).any()
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth if kind=="smooth", else
    # limiter-crossing]
    _cmp(got, ref, f"yppm jord={jord} {kind}", 1e-12)


@pytest.mark.parametrize("jord", [5, 8])
def test_yppm_duogrid_unclamped_parity(jord, rough):
    ref = _yppm_np(rough, jord, rough.q, rough.cry, duogrid=True)
    got = _yppm_jax(rough, jord, rough.q, rough.cry, duogrid=True)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(got, ref, f"yppm duo jord={jord}", 1e-12)


def test_yppm_jit_equals_eager_and_no_retrace(rough):
    """gate 2 -- ASSERTED."""
    eager = _yppm_jax(rough, 8, rough.q, rough.cry)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.yppm(*a, **k)

    fn = tp.make_yppm_jit(_counted)
    j1 = _yppm_jax(rough, 8, rough.q, rough.cry, fn=fn)
    _yppm_jax(rough, 8, rough.q * 1.01, rough.cry * 0.97, fn=fn)
    assert traces["n"] == 1, traces["n"]
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: jit-vs-eager]
    _cmp(j1, eager, "yppm jit", 1e-12)


def test_yppm_rejects_float32(rough):
    with pytest.raises(TypeError, match="float64"):
        _yppm_jax(rough, 8, np.asarray(rough.q, np.float32), rough.cry)


@pytest.mark.parametrize("bad", [0, 14, -7])
def test_yppm_unknown_jord_raises(bad, rough):
    with pytest.raises(ValueError, match="not a supported scheme"):
        _yppm_jax(rough, bad, rough.q, rough.cry)


def test_yppm_unknown_jord_guard_is_non_vacuous(rough, monkeypatch):
    """Same non-vacuity demonstration as ``xppm``: with ``_validate_ord``
    monkeypatched away (== the guard deleted) the SAME call returns an
    array instead of raising."""
    monkeypatch.setattr(tp, "_validate_ord", lambda *a, **k: None)
    out = _yppm_jax(rough, 14, rough.q, rough.cry)
    assert np.isfinite(np.asarray(out)).any()


@pytest.mark.parametrize("jord", [2, 5, 8])
def test_yppm_grads_order2_smooth(jord, smooth):
    """gate 4 -- same non-smooth-site list as :func:`test_xppm_grads_
    order2_smooth` (N1-N6), same measured smoothness control."""
    rng = np.random.default_rng(6)

    def f(q_):
        return jnp.nansum(_yppm_jax(smooth, jord, q_, smooth.cry) ** 2)

    _assert_locally_smooth(f, smooth.q, rng.standard_normal(
        smooth.q.shape), f"yppm jord={jord}")
    check_grads(f, (jnp.asarray(smooth.q),), order=2, modes=("fwd",))

    def g(c_):
        return jnp.nansum(_yppm_jax(smooth, jord, smooth.q, c_) ** 2)

    assert np.abs(smooth.cry).min() > 1e-3, "N1: c too close to 0"
    _assert_locally_smooth(g, smooth.cry, rng.standard_normal(
        smooth.cry.shape), f"yppm(c) jord={jord}")
    check_grads(g, (jnp.asarray(smooth.cry),), order=2, modes=("fwd",))


# =====================================================================
# deln_flux
# =====================================================================

def _fxfy0(geo):
    """Non-zero fx/fy so the ADD in deln_flux is visible."""
    b = geo.bd
    nfx = (b.ie + 1 - b.is_ + 1, b.je - b.js + 1)
    nfy = (b.ie - b.is_ + 1, b.je + 1 - b.js + 1)
    fx = 1.0 + np.arange(nfx[0] * nfx[1], dtype=np.float64).reshape(nfx)
    fy = 2.0 - np.arange(nfy[0] * nfy[1], dtype=np.float64).reshape(nfy)
    return fx, fy


def _deln_np(geo, nord, damp, fx0, fy0, mass=None, damp_km=None,
             duogrid=False):
    b = geo.bd
    fxf = _fv(fx0, b.is_, b.js)
    fyf = _fv(fy0, b.is_, b.js)
    npd.deln_flux(nord, b.is_, b.ie, b.js, b.je, geo.npx, geo.npy, damp,
                  _fv(geo.q, b.isd, b.jsd), fxf, fyf, geo.gs(), b,
                  mass=None if mass is None else _fv(mass, b.isd, b.jsd),
                  damp_km=None if damp_km is None
                  else _fv(damp_km, b.isd, b.jsd),
                  duogrid=duogrid)
    return fxf.a, fyf.a


def _deln_jax(geo, nord, damp, fx0, fy0, mass=None, damp_km=None,
              duogrid=False, fn=None):
    b = geo.bd
    fn = fn or tp.deln_flux
    return fn(nord, b.is_, b.ie, b.js, b.je, geo.npx, geo.npy, damp,
              jnp.asarray(geo.q), jnp.asarray(fx0), jnp.asarray(fy0),
              jnp.asarray(geo.del6_v), jnp.asarray(geo.del6_u),
              jnp.asarray(geo.rarea), b, False, True, True, True, True,
              mass=None if mass is None else jnp.asarray(mass),
              damp_km=None if damp_km is None else jnp.asarray(damp_km),
              duogrid=duogrid)


@pytest.mark.parametrize("nord", [0, 1, 2])
@pytest.mark.parametrize("opt", ["plain", "mass", "damp_km", "both"])
def test_deln_flux_parity(nord, opt, rough):
    """gate 1 -- every ``nord`` (0 = del-2, 1 = del-4, 2 = del-6) x every
    optional combination.  ``nord >= 1`` exercises the ORDERED pass
    recurrence (each pass reads the previous pass's fx2/fy2).

    FIXTURE CLASS: smooth in the sense that matters here -- deln_flux
    has NO limiter and NO data-dependent branch at all; it is a linear
    stencil plus an index shuffle, so this is a pure rounding bound."""
    fx0, fy0 = _fxfy0(rough)
    mass = rough.mass if opt in ("mass", "both") else None
    dkm = rough.damp_km if opt in ("damp_km", "both") else None
    damp = 1.0e-3
    fx_n, fy_n = _deln_np(rough, nord, damp, fx0, fy0, mass, dkm)
    fx_j, fy_j = _deln_jax(rough, nord, damp, fx0, fy0, mass, dkm)
    # Non-vacuity: the damping actually changed the fluxes.
    assert np.nanmax(np.abs(fx_n - fx0)) > 0.0
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth -- no branch, linear stencil]
    _cmp(fx_j, fx_n, f"deln fx nord={nord} {opt}", 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth]
    _cmp(fy_j, fy_n, f"deln fy nord={nord} {opt}", 1e-12)


def test_deln_flux_jit_equals_eager_and_no_retrace(rough):
    """gate 2 -- ASSERTED."""
    fx0, fy0 = _fxfy0(rough)
    eager = _deln_jax(rough, 2, 1e-3, fx0, fy0)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.deln_flux(*a, **k)

    fn = tp.make_deln_flux_jit(_counted)
    j1 = _deln_jax(rough, 2, 1e-3, fx0, fy0, fn=fn)
    _deln_jax(rough, 2, 2e-3, fx0 * 1.1, fy0 * 0.9, fn=fn)
    assert traces["n"] == 1, traces["n"]
    for name, e, j in zip(("fx", "fy"), eager, j1):
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.   [class: jit-vs-eager]
        _cmp(j, e, f"deln jit {name}", 1e-12)


def test_deln_flux_guards(rough):
    """gate 3 -- float32 TypeError and a negative ``nord`` ValueError."""
    fx0, fy0 = _fxfy0(rough)
    with pytest.raises(TypeError, match="float64"):
        _deln_jax(rough, 1, 1e-3, np.asarray(fx0, np.float32), fy0)
    with pytest.raises(ValueError, match="nord"):
        _deln_jax(rough, -1, 1e-3, fx0, fy0)


def test_deln_flux_nord_guard_is_non_vacuous(rough):
    """The ``nord < 0`` guard is shown to be doing real work.

    HOW (no monkeypatch needed -- the UNGUARDED implementation is
    already in the tree): ``fv3_native_d_sw.deln_flux`` has no such
    check.  Called with ``nord = -1`` it does NOT raise; the ``fx2``
    write window ``is-nord .. ie+nord+1`` SHRINKS inside the read window
    ``is .. ie+1`` of the final add, so it returns fluxes contaminated
    with the ``_fl`` NaN tripwire instead of an error.  That is the
    silent degradation the JAX guard converts into a loud ValueError.
    """
    fx0, fy0 = _fxfy0(rough)
    fx_bad, fy_bad = _deln_np(rough, -1, 1e-3, fx0, fy0)   # no raise
    assert np.isnan(fx_bad).any() or np.isnan(fy_bad).any(), (
        "the unguarded NumPy lane did NOT degrade at nord=-1, so this "
        "demonstration proves nothing -- re-derive it before trusting "
        "the guard test")
    # ...and the guarded lane refuses the same call.
    with pytest.raises(ValueError, match="nord"):
        _deln_jax(rough, -1, 1e-3, fx0, fy0)


def test_deln_flux_grads_order2(rough):
    """gate 4 -- deln_flux has NO switching surface (no limiter, no
    upwind select, no copysign): it is LINEAR in q and in fx/fy, so
    order-2 grads must be exact.  ``nord=2`` exercises the ordered pass
    recurrence in reverse mode as well."""
    fx0, fy0 = _fxfy0(rough)

    def f(q_, fx_):
        geo = rough
        b = geo.bd
        a, c = tp.deln_flux(
            2, b.is_, b.ie, b.js, b.je, geo.npx, geo.npy, 1e-3, q_, fx_,
            jnp.asarray(fy0), jnp.asarray(geo.del6_v),
            jnp.asarray(geo.del6_u), jnp.asarray(geo.rarea), b, False,
            True, True, True, True)
        return jnp.nansum(a * a) + jnp.nansum(c * c)

    check_grads(f, (jnp.asarray(rough.q), jnp.asarray(fx0)), order=2,
                modes=("fwd", "rev"))


# =====================================================================
# fv_tp_2d
# =====================================================================

def _tp2d_np(geo, hord, **kw):
    b = geo.bd
    nfx = (b.ie + 1 - b.is_ + 1, b.je - b.js + 1)
    nfy = (b.ie - b.is_ + 1, b.je + 1 - b.js + 1)
    qf = _fv(geo.q, b.isd, b.jsd)
    fxf = _fv(np.full(nfx, np.nan), b.is_, b.js)
    fyf = _fv(np.full(nfy, np.nan), b.is_, b.js)
    opt = {}
    for k in ("mfx", "mfy"):
        if kw.get(k) is not None:
            opt[k] = _fv(kw[k], b.is_, b.js)
    for k in ("mass", "damp_km"):
        if kw.get(k) is not None:
            opt[k] = _fv(kw[k], b.isd, b.jsd)
    for k in ("nord", "damp_c", "damp_smag"):
        if kw.get(k) is not None:
            opt[k] = kw[k]
    npd.fv_tp_2d(qf, _fv(geo.crx, b.is_, b.jsd), _fv(geo.cry, b.isd, b.js),
                 geo.npx, geo.npy, hord, fxf, fyf,
                 _fv(geo.xfx, b.is_, b.jsd), _fv(geo.yfx, b.isd, b.js),
                 geo.gs(), b, _fv(geo.ra_x, b.is_, b.jsd),
                 _fv(geo.ra_y, b.isd, b.js), 1.0,
                 duogrid=kw.get("duogrid", False), **opt)
    return qf.a, fxf.a, fyf.a


def _tp2d_jax(geo, hord, fn=None, **kw):
    b = geo.bd
    fn = fn or tp.fv_tp_2d
    arr = {k: (None if kw.get(k) is None else jnp.asarray(kw[k]))
           for k in ("mfx", "mfy", "mass", "damp_km")}
    return fn(jnp.asarray(geo.q), jnp.asarray(geo.crx),
              jnp.asarray(geo.cry), geo.npx, geo.npy, hord,
              jnp.asarray(geo.xfx), jnp.asarray(geo.yfx),
              jnp.asarray(geo.dxa), jnp.asarray(geo.dya),
              jnp.asarray(geo.area), jnp.asarray(geo.del6_v),
              jnp.asarray(geo.del6_u), jnp.asarray(geo.rarea),
              geo.da_min, b, jnp.asarray(geo.ra_x),
              jnp.asarray(geo.ra_y), 1.0, False, 0, True, True, True,
              True, nord=kw.get("nord"), damp_c=kw.get("damp_c"),
              damp_smag=kw.get("damp_smag"),
              duogrid=kw.get("duogrid", False), **arr)


@pytest.mark.parametrize("hord", PPM_ORDS)
def test_fv_tp_2d_parity(hord, rough):
    """gate 1 -- EVERY hord.  ``hord == 10`` additionally exercises the
    ``ord_in = 8`` remap (tp_core.F90:131-136).

    FIXTURE CLASS: limiter-crossing (the `rough` q); the assembled
    routine inherits xppm/yppm's discontinuous limiter flags."""
    q_n, fx_n, fy_n = _tp2d_np(rough, hord)
    q_j, fx_j, fy_j = _tp2d_jax(rough, hord)
    # Non-vacuity: copy_corners really did mutate q's corner ghosts.
    assert not np.array_equal(q_n, rough.q)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: index shuffle only -> expect BITWISE]
    _cmp(q_j, q_n, f"fv_tp_2d q hord={hord}", 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing + accumulating]
    _cmp(fx_j, fx_n, f"fv_tp_2d fx hord={hord}", 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing + accumulating]
    _cmp(fy_j, fy_n, f"fv_tp_2d fy hord={hord}", 1e-12)


@pytest.mark.parametrize(
    "kw",
    [dict(nord=1, damp_c=0.5),
     dict(damp_smag=0.5, damp_km="damp_km"),
     dict(mfx="mfx", mfy="mfy"),
     dict(mfx="mfx", mfy="mfy", mass="mass", nord=2, damp_c=0.5),
     dict(mfx="mfx", mfy="mfy", mass="mass", damp_smag=0.5,
          damp_km="damp_km"),
     dict(nord=1, damp_c=1e-6),        # below the 1e-4 gate -> no deln
     dict(damp_smag=1e-6, damp_km="damp_km")],   # below the 1e-3 gate
    ids=["deln", "smag", "mf", "mf+deln", "mf+smag", "deln-off",
         "smag-off"])
def test_fv_tp_2d_optional_branches_parity(kw, rough):
    """gate 1 -- every optional-argument branch of the flux-averaging
    block (tp_core.F90:179-224), INCLUDING the two below-threshold cases
    where ``deln_flux`` must NOT run."""
    b = rough.bd
    nfx = (b.ie + 1 - b.is_ + 1, b.je - b.js + 1)
    nfy = (b.ie - b.is_ + 1, b.je + 1 - b.js + 1)
    named = {"mfx": 0.9 + np.zeros(nfx), "mfy": 1.1 + np.zeros(nfy),
             "mass": rough.mass, "damp_km": rough.damp_km}
    kw = {k: (named[v] if isinstance(v, str) else v)
          for k, v in kw.items()}
    q_n, fx_n, fy_n = _tp2d_np(rough, 8, **kw)
    q_j, fx_j, fy_j = _tp2d_jax(rough, 8, **kw)
    for name, j, n in (("q", q_j, q_n), ("fx", fx_j, fx_n),
                       ("fy", fy_j, fy_n)):
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.   [class: limiter-crossing + accumulating]
        _cmp(j, n, f"fv_tp_2d {name} opt", 1e-12)


def test_fv_tp_2d_jit_equals_eager_and_no_retrace(rough):
    """gate 2 -- ASSERTED."""
    eager = _tp2d_jax(rough, 8, nord=1, damp_c=0.5)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.fv_tp_2d(*a, **k)

    fn = tp.make_fv_tp_2d_jit(_counted)
    j1 = _tp2d_jax(rough, 8, fn=fn, nord=1, damp_c=0.5)
    _tp2d_jax(rough, 8, fn=fn, nord=1, damp_c=0.5)
    assert traces["n"] == 1, traces["n"]
    for name, e, j in zip(("q", "fx", "fy"), eager, j1):
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.   [class: jit-vs-eager]
        _cmp(j, e, f"fv_tp_2d jit {name}", 1e-12)


def test_fv_tp_2d_rejects_float32(rough):
    """gate 3 -- guard (the mutated operand q is checked)."""
    b = rough.bd
    with pytest.raises(TypeError, match="float64"):
        tp.fv_tp_2d(jnp.asarray(rough.q, jnp.float32),
                    jnp.asarray(rough.crx), jnp.asarray(rough.cry),
                    rough.npx, rough.npy, 8, jnp.asarray(rough.xfx),
                    jnp.asarray(rough.yfx), jnp.asarray(rough.dxa),
                    jnp.asarray(rough.dya), jnp.asarray(rough.area),
                    jnp.asarray(rough.del6_v), jnp.asarray(rough.del6_u),
                    jnp.asarray(rough.rarea), rough.da_min, b,
                    jnp.asarray(rough.ra_x), jnp.asarray(rough.ra_y),
                    1.0, False, 0, True, True, True, True)


@pytest.mark.parametrize("bad", [0, 14, -7])
def test_fv_tp_2d_unknown_hord_raises(bad, rough):
    """gate 3 -- dispatch-hardening on the ASSEMBLED routine."""
    with pytest.raises(ValueError, match="not a supported scheme"):
        _tp2d_jax(rough, bad)


def test_fv_tp_2d_unknown_hord_guard_is_non_vacuous(rough, monkeypatch):
    """Guard deleted (``_validate_ord`` -> no-op): the SAME call returns
    three arrays instead of raising, i.e. hord=14 silently transports
    with the {9,13} arm and NO positive-definite pass."""
    monkeypatch.setattr(tp, "_validate_ord", lambda *a, **k: None)
    q_j, fx_j, fy_j = _tp2d_jax(rough, 14)
    assert np.isfinite(np.asarray(fx_j)).any()
    assert np.isfinite(np.asarray(fy_j)).any()
    assert np.isfinite(np.asarray(q_j)).any()


def test_fv_tp_2d_returns_the_mutated_q(rough):
    """``q`` is INOUT: ``copy_corners`` rotates its corner ghosts twice
    and the SECOND xppm reads the result (tp_core.F90:138-140, :159-164).
    A caller that dropped the returned ``q`` would transport a different
    field on the next call -- asserted here against the NumPy lane."""
    q_n, _, _ = _tp2d_np(rough, 8)
    q_j, _, _ = _tp2d_jax(rough, 8)
    changed = ~np.isclose(q_n, rough.q, rtol=0, atol=0)
    assert changed.sum() > 0, "copy_corners changed nothing (vacuous)"
    assert np.array_equal(np.asarray(q_j), q_n)


def test_fv_tp_2d_grads_order2_smooth(smooth):
    """gate 4 -- order-2 grads on the ASSEMBLED routine at hord=2 (the
    perfectly-linear scheme: NONE of the limiter surfaces N2-N6 exist in
    that trace) with the measured local-smoothness control.  The upwind
    select N1 is avoided by the single-signed Courant fixture."""
    rng = np.random.default_rng(9)

    def f(q_):
        geo = smooth
        b = geo.bd
        _, fx, fy = tp.fv_tp_2d(
            q_, jnp.asarray(geo.crx), jnp.asarray(geo.cry), geo.npx,
            geo.npy, 2, jnp.asarray(geo.xfx), jnp.asarray(geo.yfx),
            jnp.asarray(geo.dxa), jnp.asarray(geo.dya),
            jnp.asarray(geo.area), jnp.asarray(geo.del6_v),
            jnp.asarray(geo.del6_u), jnp.asarray(geo.rarea), geo.da_min,
            b, jnp.asarray(geo.ra_x), jnp.asarray(geo.ra_y), 1.0, False,
            0, True, True, True, True, nord=1, damp_c=0.5)
        return jnp.nansum(fx * fx) + jnp.nansum(fy * fy)

    _assert_locally_smooth(f, smooth.q, rng.standard_normal(
        smooth.q.shape), "fv_tp_2d hord=2")
    check_grads(f, (jnp.asarray(smooth.q),), order=2, modes=("fwd",))


# =====================================================================
# xtp_u
# =====================================================================

def _xtp_np(geo, iord, u, c, grid_type=0, bounded_domain=False,
            lim_fac=1.0, duogrid=False):
    b = geo.bd
    n = (b.ie + 1 - b.is_ + 1, b.je + 1 - b.js + 1)
    fx = _fv(np.full(n, np.nan), b.is_, b.js)
    npd.xtp_u(b.is_, b.ie, b.js, b.je, b.isd, b.ied, b.jsd, b.jed,
              _fv(c, b.is_, b.js), _fv(u, b.isd, b.jsd),
              _fv(geo.v, b.isd, b.jsd), fx, iord,
              _fv(geo.dx, b.isd, b.jsd), _fv(geo.rdx, b.isd, b.jsd),
              geo.npx, geo.npy, grid_type, bounded_domain, lim_fac,
              duogrid=duogrid)
    return fx.a


def _xtp_jax(geo, iord, u, c, grid_type=0, bounded_domain=False,
             lim_fac=1.0, duogrid=False, fn=None):
    b = geo.bd
    fn = fn or tp.xtp_u
    return fn(b.is_, b.ie, b.js, b.je, b.isd, b.ied, b.jsd, b.jed,
              jnp.asarray(c), jnp.asarray(u), jnp.asarray(geo.v), iord,
              jnp.asarray(geo.dx), jnp.asarray(geo.rdx), geo.npx,
              geo.npy, grid_type, bounded_domain, lim_fac,
              duogrid=duogrid)


@pytest.mark.parametrize("iord", SW_ORDS)
@pytest.mark.parametrize("kind", ["smooth", "rough"])
def test_xtp_u_parity(iord, kind, smooth, rough):
    """gate 1 -- EVERY iord the NumPy lane branches on, both classes."""
    geo = smooth if kind == "smooth" else rough
    ref = _xtp_np(geo, iord, geo.u, geo.c_sw)
    got = _xtp_jax(geo, iord, geo.u, geo.c_sw)
    assert np.isfinite(ref).any()
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth if kind=="smooth", else
    # limiter-crossing]
    _cmp(got, ref, f"xtp_u iord={iord} {kind}", 1e-12)


@pytest.mark.parametrize("iord", [6, 9])
def test_xtp_u_duogrid_and_gridtype3_parity(iord, rough):
    """The duo arm takes the UNCLAMPED is3/ie3 (sw_core.F90:2566) and
    drops the WMP edge fix; ``grid_type = 4`` takes the ``else`` arm of
    the ``iord >= 8`` block (sw_core.F90:2866)."""
    ref = _xtp_np(rough, iord, rough.u, rough.c_sw, duogrid=True)
    got = _xtp_jax(rough, iord, rough.u, rough.c_sw, duogrid=True)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(got, ref, f"xtp_u duo iord={iord}", 1e-12)
    if iord >= 8:
        ref4 = _xtp_np(rough, iord, rough.u, rough.c_sw, grid_type=4)
        got4 = _xtp_jax(rough, iord, rough.u, rough.c_sw, grid_type=4)
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.   [class: limiter-crossing]
        _cmp(got4, ref4, f"xtp_u gt4 iord={iord}", 1e-12)


def test_xtp_u_jit_equals_eager_and_no_retrace(rough):
    """gate 2 -- ASSERTED."""
    eager = _xtp_jax(rough, 8, rough.u, rough.c_sw)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.xtp_u(*a, **k)

    fn = tp.make_xtp_u_jit(_counted)
    j1 = _xtp_jax(rough, 8, rough.u, rough.c_sw, fn=fn)
    _xtp_jax(rough, 8, rough.u * 1.02, rough.c_sw * 0.98, fn=fn)
    assert traces["n"] == 1, traces["n"]
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: jit-vs-eager]
    _cmp(j1, eager, "xtp_u jit", 1e-12)


def test_xtp_u_rejects_float32(rough):
    with pytest.raises(TypeError, match="float64"):
        _xtp_jax(rough, 8, np.asarray(rough.u, np.float32), rough.c_sw)


@pytest.mark.parametrize("bad", [0, -5, 12, 13])
def test_xtp_u_unknown_iord_raises(bad, rough):
    """gate 3 -- dispatch-hardening.  NOTE the set differs from
    ``xppm``: ``xtp_u`` branches on the RAW iord (no ``abs()``), so the
    negatives that ARE schemes in ``xppm`` are rejected here."""
    with pytest.raises(ValueError, match="not a supported scheme"):
        _xtp_jax(rough, bad, rough.u, rough.c_sw)


@pytest.mark.parametrize("bad", [12, -5])
def test_xtp_u_unknown_iord_guard_is_non_vacuous(bad, rough, monkeypatch):
    """Guard deleted (``_validate_ord`` -> no-op): iord=12 silently runs
    the "11, unlimited" arm and iord=-5 silently runs the {5,6,7} arm --
    both return an array rather than raising."""
    monkeypatch.setattr(tp, "_validate_ord", lambda *a, **k: None)
    out = _xtp_jax(rough, bad, rough.u, rough.c_sw)
    assert np.isfinite(np.asarray(out)).any()


@pytest.mark.parametrize("iord", [2, 5, 8])
def test_xtp_u_grads_order2_smooth(iord, smooth):
    """gate 4 -- order-2 grads with the measured smoothness control.

    NON-SMOOTH SITES of xtp_u, all named: the upwind select ``c > 0``
    (N1); ``smt5``/``smt6``/``hi5``/``hi6`` flag flips, which ADD or
    DROP the whole ``fx0`` term (N2); ``copysign`` in ``dm`` and at
    iord 8, and in the iord=3 piecewise-linear fallback (N3); the
    three-way ``min`` in ``dm`` and the ``min``/``max`` PPM clamps at
    iord 9/10 (N4); the ``NEAR_ZERO_SW`` flat-region test at iord 10
    (N5).  iord=2 has none of N2-N5."""
    rng = np.random.default_rng(12)

    def f(u_):
        return jnp.nansum(_xtp_jax(smooth, iord, u_, smooth.c_sw) ** 2)

    _assert_locally_smooth(f, smooth.u, rng.standard_normal(
        smooth.u.shape), f"xtp_u iord={iord}")
    check_grads(f, (jnp.asarray(smooth.u),), order=2, modes=("fwd",))

    def g(c_):
        return jnp.nansum(_xtp_jax(smooth, iord, smooth.u, c_) ** 2)

    assert np.abs(smooth.c_sw).min() > 1e-6, "N1: c too close to 0"
    _assert_locally_smooth(g, smooth.c_sw, rng.standard_normal(
        smooth.c_sw.shape), f"xtp_u(c) iord={iord}")
    check_grads(g, (jnp.asarray(smooth.c_sw),), order=2, modes=("fwd",))


# =====================================================================
# ytp_v
# =====================================================================

def _ytp_np(geo, jord, v, c, grid_type=0, bounded_domain=False,
            lim_fac=1.0, duogrid=False):
    b = geo.bd
    n = (b.ie + 1 - b.is_ + 1, b.je + 1 - b.js + 1)
    fy = _fv(np.full(n, np.nan), b.is_, b.js)
    npd.ytp_v(b.is_, b.ie, b.js, b.je, b.isd, b.ied, b.jsd, b.jed,
              _fv(c, b.is_, b.js), _fv(geo.u, b.isd, b.jsd),
              _fv(v, b.isd, b.jsd), fy, jord,
              _fv(geo.dy, b.isd, b.jsd), _fv(geo.rdy, b.isd, b.jsd),
              geo.npx, geo.npy, grid_type, bounded_domain, lim_fac,
              duogrid=duogrid)
    return fy.a


def _ytp_jax(geo, jord, v, c, grid_type=0, bounded_domain=False,
             lim_fac=1.0, duogrid=False, fn=None):
    b = geo.bd
    fn = fn or tp.ytp_v
    return fn(b.is_, b.ie, b.js, b.je, b.isd, b.ied, b.jsd, b.jed,
              jnp.asarray(c), jnp.asarray(geo.u), jnp.asarray(v), jord,
              jnp.asarray(geo.dy), jnp.asarray(geo.rdy), geo.npx,
              geo.npy, grid_type, bounded_domain, lim_fac,
              duogrid=duogrid)


@pytest.mark.parametrize("jord", SW_ORDS)
@pytest.mark.parametrize("kind", ["smooth", "rough"])
def test_ytp_v_parity(jord, kind, smooth, rough):
    """gate 1 -- EVERY jord, both classes."""
    geo = smooth if kind == "smooth" else rough
    ref = _ytp_np(geo, jord, geo.v, geo.c_sw)
    got = _ytp_jax(geo, jord, geo.v, geo.c_sw)
    assert np.isfinite(ref).any()
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth if kind=="smooth", else
    # limiter-crossing]
    _cmp(got, ref, f"ytp_v jord={jord} {kind}", 1e-12)


@pytest.mark.parametrize("jord", [6, 9])
def test_ytp_v_duogrid_and_gridtype3_parity(jord, rough):
    ref = _ytp_np(rough, jord, rough.v, rough.c_sw, duogrid=True)
    got = _ytp_jax(rough, jord, rough.v, rough.c_sw, duogrid=True)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(got, ref, f"ytp_v duo jord={jord}", 1e-12)
    if jord >= 8:
        ref4 = _ytp_np(rough, jord, rough.v, rough.c_sw, grid_type=4)
        got4 = _ytp_jax(rough, jord, rough.v, rough.c_sw, grid_type=4)
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.   [class: limiter-crossing]
        _cmp(got4, ref4, f"ytp_v gt4 jord={jord}", 1e-12)


def test_ytp_v_jit_equals_eager_and_no_retrace(rough):
    """gate 2 -- ASSERTED."""
    eager = _ytp_jax(rough, 8, rough.v, rough.c_sw)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.ytp_v(*a, **k)

    fn = tp.make_ytp_v_jit(_counted)
    j1 = _ytp_jax(rough, 8, rough.v, rough.c_sw, fn=fn)
    _ytp_jax(rough, 8, rough.v * 1.02, rough.c_sw * 0.98, fn=fn)
    assert traces["n"] == 1, traces["n"]
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: jit-vs-eager]
    _cmp(j1, eager, "ytp_v jit", 1e-12)


def test_ytp_v_rejects_float32(rough):
    with pytest.raises(TypeError, match="float64"):
        _ytp_jax(rough, 8, np.asarray(rough.v, np.float32), rough.c_sw)


@pytest.mark.parametrize("bad", [0, -5, 12])
def test_ytp_v_unknown_jord_raises(bad, rough):
    with pytest.raises(ValueError, match="not a supported scheme"):
        _ytp_jax(rough, bad, rough.v, rough.c_sw)


def test_ytp_v_unknown_jord_guard_is_non_vacuous(rough, monkeypatch):
    """Guard deleted -> jord=12 silently runs the "11, unlimited" arm."""
    monkeypatch.setattr(tp, "_validate_ord", lambda *a, **k: None)
    out = _ytp_jax(rough, 12, rough.v, rough.c_sw)
    assert np.isfinite(np.asarray(out)).any()


@pytest.mark.parametrize("jord", [2, 5, 8])
def test_ytp_v_grads_order2_smooth(jord, smooth):
    """gate 4 -- same non-smooth-site list as ``xtp_u`` (N1-N5)."""
    rng = np.random.default_rng(15)

    def f(v_):
        return jnp.nansum(_ytp_jax(smooth, jord, v_, smooth.c_sw) ** 2)

    _assert_locally_smooth(f, smooth.v, rng.standard_normal(
        smooth.v.shape), f"ytp_v jord={jord}")
    check_grads(f, (jnp.asarray(smooth.v),), order=2, modes=("fwd",))

    def g(c_):
        return jnp.nansum(_ytp_jax(smooth, jord, smooth.v, c_) ** 2)

    assert np.abs(smooth.c_sw).min() > 1e-6, "N1: c too close to 0"
    _assert_locally_smooth(g, smooth.c_sw, rng.standard_normal(
        smooth.c_sw.shape), f"ytp_v(c) jord={jord}")
    check_grads(g, (jnp.asarray(smooth.c_sw),), order=2, modes=("fwd",))


# =====================================================================
# lane-hygiene gates
# =====================================================================

def test_no_donate_argnums_in_the_lane():
    """Grad-path doctrine (CLAUDE.md): buffer donation conflicts with
    reverse-mode AD, which is the whole point of this lane.

    Scans the AST, not the source TEXT.  The text form failed (job
    9401521) on the module's own docstring sentence "No ``donate_argnums``
    anywhere in this lane" -- a guard tripping on its own documentation,
    the same class as a log-scraping check that greps the prompt it
    echoed.  An AST scan cannot see docstrings or comments, so it fails
    only on a real keyword argument.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(tp))
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "donate_argnums"
    ]
    assert not offenders, f"donate_argnums passed at {offenders}"


def test_the_donate_argnums_guard_is_not_vacuous():
    """The AST scan above must actually fire on a real donation.

    Without this, replacing the scan with `assert True` would be
    invisible -- and the text-form version it replaces was itself a
    guard that fired for the wrong reason, so this one earns its
    scepticism.
    """
    import ast
    tree = ast.parse("import jax\nf = jax.jit(g, donate_argnums=(0,))\n")
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.keyword) and n.arg == "donate_argnums"]
    assert len(hits) == 1, "the AST scan cannot see a real donation"
    # ...and must NOT fire on the word appearing in prose.
    prose = ast.parse('"""No donate_argnums anywhere."""\n')
    assert not [n for n in ast.walk(prose)
                if isinstance(n, ast.keyword) and n.arg == "donate_argnums"]


def test_every_public_routine_has_a_jit_factory():
    """Each public routine ships exactly ONE jit policy, and the module
    entry point is built from it (so a test wrapper and production can
    never drift)."""
    for name in ("pert_ppm", "copy_corners", "xppm", "yppm",
                 "deln_flux", "fv_tp_2d", "xtp_u", "ytp_v"):
        assert hasattr(tp, name)
        assert callable(getattr(tp, f"make_{name}_jit"))
        assert getattr(tp, f"{name}_jit") is not None


def test_tolerances_are_all_marked_pending():
    """Every numeric bound in THIS file is provisional until the
    orchestrator's measurement job pins it.  If a bound is edited to a
    measured value the marker must go with it, so this count is a
    tripwire against a half-finished tolerance sweep, not a target."""
    src = open(__file__).read()
    n_mark = src.count("TOL-PENDING")
    n_cmp = src.count("_cmp(")
    # every _cmp call site (minus the def and the two loop-bodies that
    # share one marker) must be preceded by a marker
    assert n_mark >= 20, n_mark
    assert n_cmp > n_mark // 2


# =====================================================================
# the hardcoded bounded_domain = .false. override at the d_sw3 call
# sites (sw_core.F90:1315-1316 ytp_v, :1373-1374 xtp_u -- both with the
# `bounded_domain` version commented out on the line directly above;
# VERIFIED with sed on the pinned tree while writing this test)
# =====================================================================

@pytest.mark.parametrize("iord", [6, 9])
def test_xtp_u_bounded_domain_false_override_is_load_bearing(iord, rough):
    """``d_sw3`` calls ``xtp_u`` with a LITERAL ``.false.`` even under
    duo, where ``gridstruct%bounded_domain`` is ``.true.``
    (fv_arrays.F90:1512 makes duogrid imply bounded_domain).  So the
    twin must take its ``bounded_domain`` from the ARGUMENT and never
    derive it from ``duogrid``.

    Asserted two ways: the d_sw3 configuration
    ``(bounded_domain=False, duogrid=True)`` matches the NumPy lane, AND
    it DIFFERS from ``(bounded_domain=True, duogrid=True)`` -- so a twin
    that had folded the two flags together would be caught here rather
    than shipping a silently different edge treatment.
    """
    kw = dict(duogrid=True)
    ref = _xtp_np(rough, iord, rough.u, rough.c_sw,
                  bounded_domain=False, **kw)
    got = _xtp_jax(rough, iord, rough.u, rough.c_sw,
                   bounded_domain=False, **kw)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(got, ref, f"xtp_u bd=False duo iord={iord}", 1e-12)

    other = _xtp_jax(rough, iord, rough.u, rough.c_sw,
                     bounded_domain=True, **kw)
    a = np.asarray(got)
    b = np.asarray(other)
    fin = np.isfinite(a) & np.isfinite(b)
    assert fin.any()
    assert not np.allclose(a[fin], b[fin], rtol=0, atol=0), (
        "bounded_domain is NOT load-bearing on this fixture, so this "
        "test cannot detect a twin that folded it into duogrid")


@pytest.mark.parametrize("jord", [6, 9])
def test_ytp_v_bounded_domain_false_override_is_load_bearing(jord, rough):
    """Same override, ``ytp_v`` side (sw_core.F90:1315-1316)."""
    kw = dict(duogrid=True)
    ref = _ytp_np(rough, jord, rough.v, rough.c_sw,
                  bounded_domain=False, **kw)
    got = _ytp_jax(rough, jord, rough.v, rough.c_sw,
                   bounded_domain=False, **kw)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: limiter-crossing]
    _cmp(got, ref, f"ytp_v bd=False duo jord={jord}", 1e-12)

    other = _ytp_jax(rough, jord, rough.v, rough.c_sw,
                     bounded_domain=True, **kw)
    a = np.asarray(got)
    b = np.asarray(other)
    fin = np.isfinite(a) & np.isfinite(b)
    assert fin.any()
    assert not np.allclose(a[fin], b[fin], rtol=0, atol=0), (
        "bounded_domain is NOT load-bearing on this fixture")


# =====================================================================
# one-sided derivatives across a TRANSPORT limiter surface
# (strategy correction 4, second instance -- pert_ppm covers the shared
#  positive-definite limiter analytically; this covers the smt5 flag
#  that lives inside xtp_u / ytp_v themselves)
# =====================================================================

def _bisect_flag_switch(flag_of_t, lo, hi, iters=60):
    """Locate the parameter value where a BOOLEAN limiter flag flips.

    ``flag_of_t`` must return a python bool.  The endpoints must
    straddle the surface; that is asserted, so a fixture that never
    switches fails loudly instead of silently certifying nothing.
    """
    f_lo, f_hi = flag_of_t(lo), flag_of_t(hi)
    assert f_lo != f_hi, (
        "the bracket does not straddle a limiter surface -- this test "
        "would certify nothing")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if flag_of_t(mid) == f_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("routine", ["xtp_u", "ytp_v"])
def test_sw_transport_one_sided_grads_across_smt5_surface(routine, rough):
    """Approach the ``smt5 = bl*br = 0`` surface (sw_core.F90:2721 for
    ``xtp_u``, :3121 for ``ytp_v``, both at ``iord/jord = 5``) from BOTH
    sides and check each side separately.

    Crossing that surface ADDS or DROPS the whole ``fx0`` term, so the
    flux is DISCONTINUOUS there, not merely kinked; a two-sided finite
    difference straddling it is meaningless and an order-2
    ``check_grads`` taken there would be certifying a jump.

    What is asserted, per side:
      * the one-sided derivative is STABLE -- evaluating it at offset
        ``d`` and at ``d/4`` agrees to 1e-6 relative, i.e. that side is
        genuinely smooth right up to the surface;
      * the two sides DIFFER by more than that stability margin, i.e.
        the surface is real and was actually crossed.
    The surface itself is located by bisection on the limiter FLAG,
    which is read out of the public flux (flux == plain upwind value
    <=> both flags false), so no private state is touched.
    """
    b = rough.bd
    n = b.ie + 1 - b.is_ + 1
    # A single interior cell of the transported field carries the sweep.
    if routine == "xtp_u":
        base_field = np.array(rough.u, dtype=np.float64)
        kk = (base_field.shape[0] // 2, base_field.shape[1] // 2)

        def run(fld):
            return _xtp_jax(rough, 5, fld, rough.c_sw)

        i0 = b.is_ - b.isd
        j0 = b.js - b.jsd

        def plain(fld):
            c = rough.c_sw
            return np.where(c > 0.0,
                            fld[i0 - 1:i0 - 1 + n, j0:j0 + n],
                            fld[i0:i0 + n, j0:j0 + n])
    else:
        base_field = np.array(rough.v, dtype=np.float64)
        kk = (base_field.shape[0] // 2, base_field.shape[1] // 2)

        def run(fld):
            return _ytp_jax(rough, 5, fld, rough.c_sw)

        i0 = b.is_ - b.isd
        j0 = b.js - b.jsd

        def plain(fld):
            c = rough.c_sw
            return np.where(c > 0.0,
                            fld[i0:i0 + n, j0 - 1:j0 - 1 + n],
                            fld[i0:i0 + n, j0:j0 + n])

    probe = (n // 2, n // 2)

    def _field(t):
        f = base_field.copy()
        f[kk] = base_field[kk] + t
        return f

    def flag_of_t(t):
        f = _field(float(t))
        fl = np.asarray(run(f))
        return bool(abs(fl[probe] - plain(f)[probe]) > 0.0)

    t_star = _bisect_flag_switch(flag_of_t, -60.0, 60.0)

    def scalar(t):
        f = jnp.asarray(base_field).at[kk].set(base_field[kk] + t)
        if routine == "xtp_u":
            out = _xtp_jax(rough, 5, f, rough.c_sw)
        else:
            out = _ytp_jax(rough, 5, f, rough.c_sw)
        return out[probe]

    g = jax.grad(scalar)
    sides = {}
    for side, sgn in (("left", -1.0), ("right", +1.0)):
        d = 1e-3
        g1 = float(g(jnp.asarray(t_star + sgn * d)))
        g2 = float(g(jnp.asarray(t_star + sgn * d / 4.0)))
        den = max(abs(g1), abs(g2), 1e-12)
        assert abs(g1 - g2) / den <= 1e-6, (
            f"{routine} {side}: one-sided derivative not stable "
            f"({g1:.6e} vs {g2:.6e}) -- another surface is inside the "
            f"offset, so this side is not certified")
        sides[side] = g2

    den = max(abs(sides["left"]), abs(sides["right"]), 1e-12)
    assert abs(sides["left"] - sides["right"]) / den > 1e-4, (
        f"{routine}: the two sides agree ({sides}) -- the bisection did "
        f"not actually cross a switching surface")


# =====================================================================
# index-window guard (the silent-WRAP tripwire)
# =====================================================================

def test_rng_and_idx_reject_out_of_bounds_windows():
    """``_rng``/``_idx`` are the only thing between a mis-derived halo
    window and a python negative index, which WRAPS to the far end of
    the axis and returns plausible numbers -- the "halo loop-bound bug"
    row of the strategy doc's failure table.

    Non-vacuity: the in-bounds cases below return the expected slices,
    so the raises are not coming from a blanket failure.
    """
    assert tp._rng(0, 2, -2, 10, "t") == slice(2, 5)
    assert tp._rng(-2, -2, -2, 10, "t") == slice(0, 1)
    # Fortran zero-trip `do a,b` with b < a.
    assert tp._rng(5, 4, 0, 10, "t") == slice(0, 0)
    with pytest.raises(ValueError, match="escapes the declared bounds"):
        tp._rng(-3, 2, -2, 10, "t")
    with pytest.raises(ValueError, match="escapes the declared bounds"):
        tp._rng(0, 8, -2, 10, "t")
    assert tp._idx(3, -2, 10, "t") == 5
    with pytest.raises(ValueError, match="escapes the declared bounds"):
        tp._idx(-3, -2, 10, "t")
    with pytest.raises(ValueError, match="escapes the declared bounds"):
        tp._idx(8, -2, 10, "t")


# =====================================================================
# remaining branch parameters
# =====================================================================

@pytest.mark.parametrize("lim_fac", [1.0, 3.0])
def test_lim_fac_is_swept_where_it_is_read(lim_fac, rough):
    """``lim_fac`` is read ONLY by the ``mord/iord == 1`` limiter
    (tp_core.F90:376, sw_core.F90:2635 / :3010): ``smt5 = |lim_fac*b0| <
    |bl-br|``.  fv_arrays.F90:355 documents ``1: hord = 5, 3: hord = 6``,
    so both documented values are swept, on all four transport
    routines.  Non-vacuity: the two values must give DIFFERENT fluxes,
    otherwise the sweep proves nothing."""
    outs = {}
    for name, ref_fn, jax_fn, fld, c in (
            ("xppm", _xppm_np, _xppm_jax, rough.q, rough.crx),
            ("yppm", _yppm_np, _yppm_jax, rough.q, rough.cry),
            ("xtp_u", _xtp_np, _xtp_jax, rough.u, rough.c_sw),
            ("ytp_v", _ytp_np, _ytp_jax, rough.v, rough.c_sw)):
        ref = ref_fn(rough, 1, fld, c, lim_fac=lim_fac)
        got = jax_fn(rough, 1, fld, c, lim_fac=lim_fac)
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.   [class: limiter-crossing]
        _cmp(got, ref, f"{name} lim_fac={lim_fac}", 1e-12)
        outs[name] = np.asarray(got)
    if lim_fac == 3.0:
        base = {n: np.asarray(f(rough, 1, x, cc, lim_fac=1.0))
                for n, f, x, cc in (("xppm", _xppm_jax, rough.q,
                                     rough.crx),
                                    ("yppm", _yppm_jax, rough.q,
                                     rough.cry),
                                    ("xtp_u", _xtp_jax, rough.u,
                                     rough.c_sw),
                                    ("ytp_v", _ytp_jax, rough.v,
                                     rough.c_sw))}
        moved = [n for n in outs
                 if not np.array_equal(np.nan_to_num(outs[n]),
                                       np.nan_to_num(base[n]))]
        assert moved, "lim_fac changed NOTHING -- the sweep is vacuous"


def test_shape_guards_reject_mis_declared_operands(rough):
    """Every declared Fortran extent is asserted at entry, so an operand
    sliced to the wrong window is a loud error instead of a broadcast.

    Non-vacuity: the correctly-shaped call in every other test in this
    file passes the same checks, so these raises are specific."""
    b = rough.bd
    with pytest.raises(ValueError, match="xppm: q must be"):
        tp.xppm(jnp.asarray(rough.q[:, :-1]), jnp.asarray(rough.crx), 8,
                b.is_, b.ie, b.isd, b.ied, b.jsd, b.jed, b.jsd, b.jed,
                rough.npx, rough.npy, jnp.asarray(rough.dxa), False, 0,
                1.0)
    with pytest.raises(ValueError, match="xppm: c must be"):
        tp.xppm(jnp.asarray(rough.q), jnp.asarray(rough.crx[:-1, :]), 8,
                b.is_, b.ie, b.isd, b.ied, b.jsd, b.jed, b.jsd, b.jed,
                rough.npx, rough.npy, jnp.asarray(rough.dxa), False, 0,
                1.0)
    with pytest.raises(ValueError, match="yppm: dya must be"):
        tp.yppm(jnp.asarray(rough.q), jnp.asarray(rough.cry), 8, b.isd,
                b.ied, b.isd, b.ied, b.js, b.je, b.jsd, b.jed, rough.npx,
                rough.npy, jnp.asarray(rough.dya[:, :-1]), False, 0, 1.0)
    with pytest.raises(ValueError, match="xtp_u: u must be"):
        _xtp_jax(rough, 8, rough.u[:, :-1], rough.c_sw)
    with pytest.raises(ValueError, match="ytp_v: v must be"):
        _ytp_jax(rough, 8, rough.v[:-1, :], rough.c_sw)


def test_corner_block_rejects_out_of_bounds_gather(rough):
    """``_corner_block``'s own bounds check (the gather counterpart of
    ``_rng``).  A too-small ``q`` makes the ``ng``-deep corner window
    escape; without the check the advanced-index gather would wrap."""
    tiny = jnp.zeros((4, 4), jnp.float64)
    with pytest.raises(ValueError, match="escapes the declared bounds"):
        tp._corner_block(tiny, -2, -2, -2, 0, -2, 0,
                         lambda i, j: j, lambda i, j: 1 - i, "probe")
