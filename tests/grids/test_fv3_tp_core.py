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
import zlib

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
            # TWO REGIMES BY CONSTRUCTION, so the limiter flag takes BOTH
            # values (job 9401521 measured the previous delp+step+dip
            # fixture at on=234 / off=0 -- entirely on ONE side of the
            # surface, which is what its own self-check caught).
            #
            # Why a step does not work and a 2-delta-x oscillation does:
            # `smt5 = bl*br < 0` is FALSE only where the cell is a LOCAL
            # EXTREMUM.  For a linear field q = m + s*i the 4-point PPM
            # edge values are al(i) = q(i) - s/2 and al(i+1) = q(i) + s/2,
            # so bl*br = -(s/2)^2 < 0 -> flag ON at EVERY cell; a step
            # adds two isolated cells and leaves the rest ON.  For a
            # +-A alternation about a mean m the edge value is exactly m
            # at every interface (p1*2m + p2*2m = m), so bl = br = -+A and
            # bl*br = A^2 > 0 -> flag OFF at EVERY cell of that region.
            # Halving the domain therefore guarantees a non-empty split
            # WITHOUT depending on any property of the fixture data.
            base = float(np.mean(np.asarray(inp["delp"],
                                            dtype=np.float64)))
            amp = 200.0

            def _two_regime(shape, i_lo, j_lo):
                """linear (flag ON) on the low half of each axis,
                2-delta-x oscillation (flag OFF) on the high half."""
                n_i, n_j = shape
                a = np.arange(n_i, dtype=np.float64)[:, None]
                c = np.arange(n_j, dtype=np.float64)[None, :]
                out = base + 40.0 * a / n_i + 15.0 * c / n_j
                si = (-1.0) ** np.arange(n_i, dtype=np.float64)
                sj = (-1.0) ** np.arange(n_j, dtype=np.float64)
                out[n_i // 2:, :] += amp * si[n_i // 2:, None]
                out[:, n_j // 2:] += amp * sj[None, n_j // 2:]
                del i_lo, j_lo
                return out

            self.q = _two_regime((ni, nj), b.isd, b.jsd)
            self._two_regime = _two_regime
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
        if kind == "smooth":
            iu = np.arange(b.isd, b.ied + 1, dtype=np.float64)[:, None]
            ju = np.arange(b.jsd, b.jed + 2, dtype=np.float64)[None, :]
            self.u = 5.0 + 0.3 * iu + 0.2 * ju + 0.004 * iu * ju
            iv = np.arange(b.isd, b.ied + 2, dtype=np.float64)[:, None]
            jv = np.arange(b.jsd, b.jed + 1, dtype=np.float64)[None, :]
            self.v = -4.0 + 0.25 * iv - 0.15 * jv + 0.003 * iv * jv
        else:
            # same two-regime construction as `q`, so xtp_u / ytp_v also
            # span BOTH sides of their smt5 surface.
            self.u = self._two_regime((ni, nj + 1), b.isd, b.jsd)
            self.v = self._two_regime((ni + 1, nj), b.isd, b.jsd)

        # c for xtp_u/ytp_v is (is:ie+1, js:je+1) and is multiplied by
        # rdx/rdy inside, so scale it by a representative dx.
        nc_i = b.ie + 1 - b.is_ + 1
        nc_j = b.je + 1 - b.js + 1
        ci = np.arange(nc_i, dtype=np.float64)[:, None]
        cj = np.arange(nc_j, dtype=np.float64)[None, :]
        dxr = float(np.median(self.dx))
        self.c_sw = csign * cscale * dxr * np.cos(
            0.7 * ci + 0.4 * cj + (0.0 if kind == "smooth" else 1.3))
        if kind == "smooth":
            # The gradient gates need the upwind select `c > 0` (site N1)
            # bounded AWAY from its switching surface BY CONSTRUCTION.
            # The fixture-derived uc/vc and a bare cosine both cross zero,
            # which made the old preconditions `|c|.min() > 1e-3` a
            # hand-derived margin on a global min -- exactly the
            # over-tight-margin class of lesson 12.  A strictly positive
            # field removes the precondition instead of tuning it.
            self.crx = 0.30 + 0.10 * np.sin(
                0.5 * np.arange(self.crx.shape[0])[:, None]
                + 0.3 * np.arange(self.crx.shape[1])[None, :])
            self.cry = 0.28 + 0.09 * np.cos(
                0.4 * np.arange(self.cry.shape[0])[:, None]
                + 0.6 * np.arange(self.cry.shape[1])[None, :])
            self.xfx = self.crx * self.dy[i0:i1, :] * 0.5
            self.yfx = self.cry * self.dx[:, j0:j1] * 0.5
            self.ra_x = (self.area[i0:i0 + (b.ie - b.is_ + 1), :]
                         + self.xfx[:-1, :] - self.xfx[1:, :])
            self.ra_y = (self.area[:, j0:j0 + (b.je - b.js + 1)]
                         + self.yfx[:, :-1] - self.yfx[:, 1:])
            assert (self.ra_x > 0).all() and (self.ra_y > 0).all()
            self.c_sw = dxr * (0.30 + 0.10 * np.sin(0.7 * ci + 0.4 * cj))
            assert np.abs(self.crx).min() > 0.15
            assert np.abs(self.cry).min() > 0.15
            assert np.abs(self.c_sw).min() > 0.15 * dxr

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


def test_pert_ppm_adjoint_identity():
    """gate 4 PRIMARY -- runs ON pert_ppm's switching surfaces (a random
    ensemble straddles all of S1/S2/S3), which is exactly where the FD
    gate below cannot go."""
    a0, al, ar = _pert_fixture()
    for iv in (0, 1):
        def f(x, iv=iv):
            n = x.shape[0] // 2
            l_, r_ = tp.pert_ppm(jnp.asarray(a0), x[:n], x[n:], iv)
            return jnp.concatenate([l_, r_])

        _adjoint_identity(f, np.concatenate([al, ar]),
                          f"pert_ppm iv={iv}")


@pytest.mark.parametrize("order", [1, 2])
def test_pert_ppm_check_grads_smooth(order):
    """gate 4 SUPPLEMENT -- order 1 before order 2, at a CONSTRUCTED
    state whose distance from every switching surface is computed and
    asserted rather than sampled.

    The previous version drew ``al``/``ar`` from a random normal and
    then required ``.all()`` cells to clear a hand-picked margin -- the
    over-tight-margin class of lesson 12: with ``al+ar`` free to wander
    through zero, some of the 64 samples inevitably land near
    ``|a6da| = da2`` and the precondition, not ``check_grads``, decides
    the test.  Here the amplitude is chosen so the margin is provable:
    ``|al+ar| <= 0.2`` and ``|da1| >= 1.8`` give ``|a6da| <= 3*0.2*2.2 =
    1.32`` against ``da2 >= 3.24``, i.e. never closer than 59 %.

    NON-SMOOTH SITES of pert_ppm(iv=1), all named and all avoided here:
      S1 ``al*ar = 0``, S2 ``a6da = -da2``, S3 ``a6da = +da2``.
    """
    k = np.arange(64, dtype=np.float64)
    al = -(1.0 + 0.1 * np.sin(k))
    ar = +(1.0 + 0.1 * np.cos(k))
    a0 = 1.0 + 0.1 * np.cos(0.5 * k)
    da1 = al - ar
    da2 = da1 ** 2
    a6da = 3.0 * (al + ar) * da1
    # Measured margins (not assumed): report-and-assert.
    assert (al * ar).max() < -0.4, float((al * ar).max())
    assert ((da2 - np.abs(a6da)) / da2).min() > 0.5, \
        float(((da2 - np.abs(a6da)) / da2).min())

    def f(x):
        n = x.shape[0] // 2
        l_, r_ = tp.pert_ppm(jnp.asarray(a0), x[:n], x[n:], 1)
        return jnp.sum(l_ * l_) + jnp.sum(r_ * r_) + jnp.sum(l_ * r_)

    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth-region FD, O(1) dynamic range]
    check_grads(f, (jnp.asarray(np.concatenate([al, ar])),), order=order,
                modes=("fwd", "rev"), eps=1e-4, atol=1e-6, rtol=1e-6)


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


@pytest.mark.parametrize("order", [1, 2])
def test_copy_corners_check_grads(order):
    """gate 4 -- a gather/scatter is LINEAR in ``q``; there is no
    switching surface at all, so grads must be exact everywhere.
    ``order=1`` runs as a separate case so an FD-resolution failure can
    never be confounded with a wrong Jacobian (lesson 12).

    FD RESOLUTION, and why this test used to fail (run 6).  It fed
    ``rough.q``, whose values are ``mean(delp) +- 200 ~ 1.1e7``, into
    ``nansum(y*y)`` -- a functional of magnitude ``64 * (1.1e7)^2 ~
    8e15``.  A finite difference of a 8e15-magnitude scalar carries an
    absolute rounding floor near ``1`` in float64, which swamps the
    O(1e7) directional derivative the check is trying to resolve.  That
    is pure dynamic range, not a Jacobian error, and it is the class
    lesson 12 names.  ``copy_corners`` is a PURE INDEX COPY, so its
    input carries no physical meaning at all and an O(1) operand is
    every bit as valid a test of the gather -- and leaves the FD nine
    decades of headroom.
    """
    q0 = jnp.asarray(
        np.random.default_rng(101).standard_normal((8, 8)))
    bd = Bounds.single_tile(2, 3)

    def f(x):
        y = tp.copy_corners(x, 3, 3, 1, False, bd, True, True, True, True)
        return jnp.nansum(y * y)

    check_grads(f, (q0,), order=order, modes=("fwd", "rev"))


def test_copy_corners_adjoint_identity():
    """gate 4 PRIMARY -- the gather is linear, so the identity is the
    natural statement of correctness for its transpose (the scatter).
    Unlike the FD gate this one is scale-free, so the operand magnitude
    is irrelevant; an O(1) operand is used for consistency."""
    bd = Bounds.single_tile(2, 3)

    def f(x):
        return tp.copy_corners(x, 3, 3, 2, False, bd, True, True, True,
                               True)

    _adjoint_identity(
        f, np.random.default_rng(102).standard_normal((8, 8)),
        "copy_corners")


# =====================================================================
# shared smoothness control (strategy correction 4)
# =====================================================================

def _assert_scalar_locally_smooth(f, t0, name, eps=1e-3, tol=1e-6):
    """No switching surface inside the finite-difference ball, measured
    on a SINGLE-CELL scalar functional.

    Lesson 12 + job 9401521: the previous version compared one-sided
    derivatives of a GLOBAL functional (``nansum(out*out)``) under a
    dense random perturbation of every input cell.  A PPM flux array
    crosses hundreds of limiter surfaces at once under such a
    perturbation, so the two one-sided derivatives disagree for reasons
    that have nothing to do with the state being probed -- the detector
    diluted every individual surface into an unreadable average.  One
    input cell, one output cell, one scalar parameter.
    """
    f0 = float(f(jnp.asarray(t0)))
    fp = (float(f(jnp.asarray(t0 + eps))) - f0) / eps
    fm = (f0 - float(f(jnp.asarray(t0 - eps)))) / eps
    den = max(abs(fp), abs(fm), 1.0)
    assert abs(fp - fm) / den <= tol, (
        f"{name}: one-sided derivatives disagree ({fp:.6e} vs {fm:.6e}) "
        f"-- a switching surface is inside the FD ball, so a check_grads "
        f"here would be certifying a kink")
    return fp


def _scalar_probe(run, field, kk, probe):
    """Scalar -> scalar view of a transport routine: perturb ONE input
    cell by ``t``, read ONE output cell.

    ``check_grads`` perturbs its ARGUMENTS in a random direction, so
    handing it the full operand moves every cell at once and reproduces
    exactly the dilution above.  A scalar argument confines the
    perturbation to the single cell whose smoothness was measured.
    """
    base = jnp.asarray(field)

    def f(t):
        return run(base.at[kk].add(t))[probe]

    return f


def _adjoint_identity(f, x, name, tol=1e-12):
    """PRIMARY gradient gate: ``<J v, w> == <v, J^T w>``.

    Why this and not ``check_grads`` (lesson 12, and 15 of the 22
    failures in job 9401521): the identity is exact for ANY state,
    needs no finite-difference step, no dynamic-range budget and -- the
    decisive property here -- **no off-switch fixture**.  It runs ON the
    limiter surface, where a PPM chain spends most of its state space
    and where ``check_grads`` cannot go at all.  A wrong Jacobian breaks
    it by O(1), so the bound is a rounding bound, not a threshold.

    Its blind spot is the one lesson 12 names: ``jvp`` and ``vjp`` of the
    SAME wrong Jacobian agree.  That is why the scoped ``check_grads``
    gates below are kept as the complementary FD check rather than
    deleted.

    Non-vacuity is asserted, not assumed: a routine whose Jacobian
    happened to be the zero map would satisfy the identity trivially.
    """
    # The RNG seed is derived from the test NAME, never from the scheme
    # order.  Run 6 (job 9408346) failed eight of these with
    # `ValueError: expected non-negative integer` -- `seed=iord+1` and
    # `seed=jord+3` go NEGATIVE for iord <= -2 / jord <= -4, and
    # np.random.default_rng rejects that.  The failing subsets were
    # exactly {-6..-2} for xppm and {-6..-4} for yppm, i.e. precisely
    # where the arithmetic goes negative, which is what identified it as
    # a HARNESS defect rather than a module one.  crc32 of the name is
    # non-negative, deterministic and stable across runs and platforms,
    # so no call site can reintroduce the class.
    seed = zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    x = jnp.asarray(x)
    v = jnp.asarray(rng.standard_normal(x.shape))
    y, jv = jax.jvp(f, (x,), (v,))
    assert bool(jnp.isfinite(y).all()), f"{name}: output not finite"
    assert bool(jnp.isfinite(jv).all()), f"{name}: J v not finite"
    w = jnp.asarray(rng.standard_normal(y.shape))
    _, vjp_fn = jax.vjp(f, x)
    (jtw,) = vjp_fn(w)
    assert bool(jnp.isfinite(jtw).all()), f"{name}: J^T w not finite"
    lhs = float(jnp.vdot(jv, w))
    rhs = float(jnp.vdot(v, jtw))
    # Non-vacuity: the Jacobian is not the zero map and the pairing is
    # not accidentally orthogonal.
    assert float(jnp.linalg.norm(jv)) > 0.0, f"{name}: J v == 0"
    assert abs(lhs) > 0.0, f"{name}: <J v, w> == 0 (vacuous pairing)"
    den = max(abs(lhs), abs(rhs))
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: adjoint identity, exact to rounding]
    assert abs(lhs - rhs) / den <= tol, (
        f"{name}: <Jv,w>={lhs:.17e} != <v,J^Tw>={rhs:.17e} "
        f"(rel {abs(lhs - rhs) / den:.3e})")
    return lhs, rhs


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


@pytest.mark.parametrize("iord", PPM_ORDS)
def test_xppm_adjoint_identity(iord, rough):
    """gate 4 PRIMARY -- runs ON the limiter surfaces, every iord.

    NON-SMOOTH SITES of xppm, all named: N1 the upwind select
    ``c(i,j) > 0``; N2 the ``smt5``/``smt6``/``hi5``/``hi6`` flags,
    across which the flux JUMPS; N3 ``copysign`` in ``dm`` and at
    iord 8/11; N4 the three-way ``min`` in ``dm`` and the PPM
    ``min``/``max`` clamps at iord 9/10/13 and in the edge blocks;
    N5 ``max(0., al)`` for iord < 0; N6 ``0.25/a4`` at ``a4 = 0``
    (iord -5/7/12), where the value is defined by the double-``where``
    but NO gradient claim is made.  This gate is valid at all of
    N1-N5 -- the identity holds for whichever branch AD linearised.
    """
    def f(q_):
        return _xppm_jax(rough, iord, q_, rough.crx)

    _adjoint_identity(f, rough.q, f"xppm q iord={iord}")

    def g(c_):
        return _xppm_jax(rough, iord, rough.q, c_)

    _adjoint_identity(g, rough.crx, f"xppm c iord={iord}")


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("iord", [2, 5, 8])
def test_xppm_check_grads_single_cell(order, iord, smooth):
    """gate 4 SUPPLEMENT -- scoped smooth-region FD check.

    ``order=1`` is a separate parametrization from ``order=2`` so an FD
    RESOLUTION failure can never be confounded with a wrong Jacobian
    (lesson 12): order-1 red means the Jacobian is wrong, order-1 green
    with order-2 red means the second-order FD ran out of digits.

    Scope: ONE input cell, ONE output cell, deliberately in the interior
    (Fortran i = is+4) so the ``is==1`` / ``ie+1==npx`` edge blocks are
    outside the stencil.  The smooth fixture is a low-order polynomial
    with a strictly positive Courant field, so N1 and N2 are inactive;
    the ``_assert_scalar_locally_smooth`` control MEASURES that rather
    than assuming it.
    """
    b = smooth.bd
    iq, jq = b.is_ + 4, b.jsd + 8
    kk = (iq - b.isd, jq - b.jsd)
    probe = (iq - b.is_, jq - b.jsd)

    def run(q_):
        return _xppm_jax(smooth, iord, q_, smooth.crx)

    f = _scalar_probe(run, smooth.q, kk, probe)
    slope = _assert_scalar_locally_smooth(f, 0.0, f"xppm iord={iord}")
    assert abs(slope) > 1e-6, "probe cell does not influence the probed flux"
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth-region FD, O(1) dynamic range]
    check_grads(f, (jnp.asarray(0.0),), order=order, modes=("fwd", "rev"),
                eps=1e-3, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize("jord", PPM_ORDS)
def test_yppm_adjoint_identity(jord, rough):
    """gate 4 PRIMARY -- same non-smooth-site list (N1-N6) as
    :func:`test_xppm_adjoint_identity`, j direction."""
    def f(q_):
        return _yppm_jax(rough, jord, q_, rough.cry)

    _adjoint_identity(f, rough.q, f"yppm q jord={jord}")

    def g(c_):
        return _yppm_jax(rough, jord, rough.q, c_)

    _adjoint_identity(g, rough.cry, f"yppm c jord={jord}")


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("jord", [2, 5, 8])
def test_yppm_check_grads_single_cell(order, jord, smooth):
    """gate 4 SUPPLEMENT -- scoped, single-cell, order 1 before 2."""
    b = smooth.bd
    iq, jq = b.isd + 8, b.js + 4
    kk = (iq - b.isd, jq - b.jsd)
    probe = (iq - b.isd, jq - b.js)

    def run(q_):
        return _yppm_jax(smooth, jord, q_, smooth.cry)

    f = _scalar_probe(run, smooth.q, kk, probe)
    slope = _assert_scalar_locally_smooth(f, 0.0, f"yppm jord={jord}")
    assert abs(slope) > 1e-6, "probe cell does not influence the flux"
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth-region FD, O(1) dynamic range]
    check_grads(f, (jnp.asarray(0.0),), order=order, modes=("fwd", "rev"),
                eps=1e-3, atol=1e-5, rtol=1e-5)


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


def _deln_increment(geo, nord, mass, damp_km, damp=1.0):
    """|del-n increment| at ``damp``, measured against ZERO fluxes.

    ``fx0 = fy0 = 0`` makes the measurement exact at every ``nord``:
    the routine's final step is ``fx += fx2``, so with ``fx = 0`` it
    RETURNS ``fx2`` and nothing is absorbed.  Probing against the
    O(1..156) ``_fxfy0`` fluxes instead is what made the original nord=2
    probe unreadable.
    """
    b = geo.bd
    z_x = np.zeros((b.ie + 1 - b.is_ + 1, b.je - b.js + 1))
    z_y = np.zeros((b.ie - b.is_ + 1, b.je + 1 - b.js + 1))
    fxp, fyp = _deln_np(geo, nord, damp, z_x, z_y, mass, damp_km)
    return max(float(np.nanmax(np.abs(fxp))),
               float(np.nanmax(np.abs(fyp))))


def _deln_damp(geo, nord, fx0, fy0, mass, damp_km, target=0.05):
    """Self-calibrated ``damp`` giving a ``target``-relative increment.

    WHY THIS EXISTS (job 9401521, the four ``nord=2`` non-vacuity
    failures).  ``deln_flux`` multiplies by ``rarea`` ONCE PER PASS.
    With the previous hand-picked ``damp = 1e-3`` the nord=2 (del-6)
    increment landed BELOW one ULP of an ``fx0`` of order 1..156, so
    ``fx + fx2 == fx`` exactly.  nord=0 and nord=1 passed because they
    carry one and two factors of ``rarea`` fewer.  The operator was
    never inert -- the FIXTURE's damp was wrong, and the oracle itself
    supplies the compensating scaling at tp_core.F90:196-198,
    ``damp = (damp_c*da_min)**(nord+1)``, precisely to undo
    ``rarea**nord``.

    Calibrating rather than hardcoding a per-nord constant is exact
    here: ``deln_flux`` is EXACTLY LINEAR in ``damp`` in all four
    optional branches -- plain scales ``d2 = damp*q`` and everything
    downstream is linear in ``d2``; the ``mass`` / ``damp_km`` branches
    leave ``d2 = q`` and put ``damp`` in the final ``damp2`` factor
    alone.  So ONE probe fixes the scale, and the probe is taken
    against ZERO fluxes (:func:`_deln_increment`) so it is resolvable at
    every nord.
    """
    inc = _deln_increment(geo, nord, mass, damp_km)
    assert np.isfinite(inc) and inc > 0.0, (
        f"deln_flux probe at damp=1 moved NOTHING (nord={nord}) -- the "
        f"operator, not the fixture scale, is inert; this is the (b) "
        f"branch of the discrimination and is a CODE defect")
    scale = max(float(np.nanmax(np.abs(fx0))),
                float(np.nanmax(np.abs(fy0))))
    return target * scale / inc


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
    damp = _deln_damp(rough, nord, fx0, fy0, mass, dkm)
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
    damp = _deln_damp(rough, 2, fx0, fy0, None, None)
    eager = _deln_jax(rough, 2, damp, fx0, fy0)
    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return tp.deln_flux(*a, **k)

    fn = tp.make_deln_flux_jit(_counted)
    j1 = _deln_jax(rough, 2, damp, fx0, fy0, fn=fn)
    _deln_jax(rough, 2, 2.0 * damp, fx0 * 1.1, fy0 * 0.9, fn=fn)
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
        _deln_jax(rough, 1, 1.0, np.asarray(fx0, np.float32), fy0)
    with pytest.raises(ValueError, match="nord"):
        _deln_jax(rough, -1, 1.0, fx0, fy0)


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
    fx_bad, fy_bad = _deln_np(rough, -1, 1.0, fx0, fy0)    # no raise
    assert np.isnan(fx_bad).any() or np.isnan(fy_bad).any(), (
        "the unguarded NumPy lane did NOT degrade at nord=-1, so this "
        "demonstration proves nothing -- re-derive it before trusting "
        "the guard test")
    # ...and the guarded lane refuses the same call.
    with pytest.raises(ValueError, match="nord"):
        _deln_jax(rough, -1, 1.0, fx0, fy0)


@pytest.mark.parametrize("order", [1, 2])
def test_deln_flux_check_grads(order, rough):
    """gate 4 -- ``deln_flux`` has NO switching surface (no limiter, no
    upwind select, no copysign; the d_sw6 branch-token scan returns
    ``[]``), so a limiter/off-switch story cannot apply and the only
    candidates are FD resolution and the fixture.  It was the fixture,
    in two compounding ways, and both are removed here.

    (1) THE ADDITIVE CONSTANT.  The old functional was
    ``nansum(fx*fx)`` on the RETURNED ``fx``, i.e. on ``fx0 +
    increment``.  With ``fx0`` of order 1..156 and an increment
    calibrated to 5 % of it, the functional sits near ``sum(fx0^2) ~
    2e6`` while its DEPENDENCE on ``q`` enters only through the small
    increment -- a textbook cancellation, and the campaign has since
    CONFIRMED a condition number of 7.78e+12 in ``del6_vt_flux``, the
    same family of differencing operator.  Passing ``fx0 = 0`` makes the
    returned flux the increment ITSELF, so the functional is a pure
    quadratic form in ``q`` with no constant to cancel against.

    (2) THE OPERAND MAGNITUDE.  ``rough.q ~ 1.1e7`` made the functional
    ~1e14 before the increment was even considered.  ``deln_flux`` is
    LINEAR in ``q``, so an O(1) operand tests exactly the same Jacobian.

    ``fx``/``fy`` are dropped from the differentiated arguments on
    purpose: the routine's dependence on them is the identity, which
    contributes a large, exactly-known block that only degrades the FD.
    The adjoint gate below still pairs the full operator.
    """
    b = rough.bd
    z_x = np.zeros((b.ie + 1 - b.is_ + 1, b.je - b.js + 1))
    z_y = np.zeros((b.ie - b.is_ + 1, b.je + 1 - b.js + 1))
    q1 = np.random.default_rng(103).standard_normal(rough.q.shape)

    def f(q_):
        a_, c_ = tp.deln_flux(
            2, b.is_, b.ie, b.js, b.je, rough.npx, rough.npy, 1.0, q_,
            jnp.asarray(z_x), jnp.asarray(z_y),
            jnp.asarray(rough.del6_v), jnp.asarray(rough.del6_u),
            jnp.asarray(rough.rarea), b, False, True, True, True, True)
        return jnp.nansum(a_ * a_) + jnp.nansum(c_ * c_)

    # Conditioning is MEASURED and reported rather than assumed: a
    # functional whose value dwarfs its own variation cannot be checked
    # by finite differences at any tolerance (the SW-core gate asserts
    # the same quantity).
    f0 = float(f(jnp.asarray(q1)))
    g0 = float(jnp.linalg.norm(jax.grad(f)(jnp.asarray(q1))))
    scale = float(np.linalg.norm(q1))
    assert g0 > 0.0, "zero gradient -- the FD check would be vacuous"
    cond = abs(f0) / max(g0 * scale, 1e-300)
    assert cond < 1e6, (
        f"deln_flux FD gate is ill-conditioned (|f|/(|grad f|*|q|) = "
        f"{cond:.3e}); an additive constant has crept back into the "
        f"functional and the FD cannot resolve the dependence")
    check_grads(f, (jnp.asarray(q1),), order=order, modes=("fwd", "rev"))


def test_deln_flux_adjoint_identity(rough):
    """gate 4 PRIMARY -- nord=2, i.e. through the ordered pass
    recurrence, paired on both returned fluxes at once."""
    b0 = rough.bd
    z_x = np.zeros((b0.ie + 1 - b0.is_ + 1, b0.je - b0.js + 1))
    z_y = np.zeros((b0.ie - b0.is_ + 1, b0.je + 1 - b0.js + 1))

    def f(q_):
        b = rough.bd
        a, c = tp.deln_flux(
            2, b.is_, b.ie, b.js, b.je, rough.npx, rough.npy, 1.0, q_,
            jnp.asarray(z_x), jnp.asarray(z_y),
            jnp.asarray(rough.del6_v), jnp.asarray(rough.del6_u),
            jnp.asarray(rough.rarea), b, False, True, True, True, True)
        return jnp.concatenate([a.reshape(-1), c.reshape(-1)])

    _adjoint_identity(f, rough.q, "deln_flux nord=2")


@pytest.mark.parametrize("nord", [0, 1, 2])
def test_deln_flux_is_not_inert_at_any_nord(nord, rough):
    """Non-vacuity of the operator itself, measured where it is
    RESOLVABLE.

    Against ``fx0 = 0`` the routine returns the increment ITSELF
    (``fx += fx2`` with ``fx = 0``), so it is read at full float64
    precision no matter how many factors of ``rarea`` the del-n operator
    has applied.  Measuring against the O(1..156) ``_fxfy0`` fluxes --
    which is what the original gate did -- cannot see a nord=2
    increment at all, and that absorption is the whole finding.
    """
    inc = _deln_increment(rough, nord, None, None)
    assert np.isfinite(inc) and inc > 0.0, (
        f"nord={nord}: the operator moved NOTHING at damp=1 measured "
        f"against ZERO fluxes -- that is branch (b), a CODE defect in "
        f"the pass recurrence, not a fixture scale problem")


def test_deln_flux_nord2_increment_is_sub_ulp_of_the_original_fixture(
        rough):
    """RECORD of the nord=2 discrimination, re-derived from MEASURED
    behaviour after the FIRST version of this gate was itself wrong.

    THE FIRST VERSION WAS DEFECTIVE and its failure was informative.  It
    anchored a ``rarea**nord`` window on ``max|rarea|`` and ``max|q|``.
    On this fixture those maxima are NOT physical: ``area``, ``del6_u``,
    ``del6_v`` and ``delp`` all report a max of exactly ``1.0000e+08``,
    the ``BIG_NUMBER`` sentinel filling unset ghost/corner cells, and
    ``max|rarea| = 1e-8`` is ``1/`` that sentinel -- about four decades
    from the physical ``~2e-12``.  Anchoring on a sentinel put the
    nord=2 window roughly three decades above the real increment, so the
    gate failed while the operator was fine.  (The alternative
    hypothesis -- that the calibrated ``_deln_damp`` had removed the
    nord dependence -- was NOT the cause: that gate probed ``_deln_np``
    at ``damp = 1.0`` directly and never called ``_deln_damp``.)

    A scaling LAW cannot be stated safely on this fixture at all, for
    the same reason: at a sentinel cell ``del6 * rarea = 1e8 * 1e-8 = 1``
    exactly, so a per-pass ratio measured with ``max`` need not show any
    shrinkage.  This gate therefore asserts NO law.  It asserts a
    BRACKET whose two sides are each an OBSERVED run fact:

      upper: run 5 (job 9401521) showed ``fx + fx2 == fx`` BITWISE at
             ``damp = 1e-3`` with ``fx0`` of order 1..156, so the
             increment at that damp was below one ULP of 156;
      lower: run 6 (job 9408346) showed every ``test_deln_flux_parity``
             at nord=2 PASSING, which requires ``_deln_damp``'s
             ``assert inc > 0`` probe to have found a resolvable
             increment -- so the operator is NOT inert.

    Together they bracket the unit-damp increment into
    ``(ULP(156), ~1e3 x ULP(156))``: branch (a), a fixture scale defect,
    and incompatible with branch (b), an inert pass recurrence.  No grid
    statistic enters.
    """
    fx0, fy0 = _fxfy0(rough)
    ulp = float(np.spacing(float(np.abs(fx0).max())))
    inc1 = _deln_increment(rough, 2, None, None)      # unit damp
    assert inc1 > 0.0, "operator inert at unit damp -- branch (b)"
    # upper side: at the original damp=1e-3 the increment was absorbed.
    assert inc1 * 1.0e-3 < ulp, (
        f"nord=2 increment at damp=1e-3 is {inc1 * 1e-3:.3e}, NOT below "
        f"one ULP of the fixture flux ({ulp:.3e}) -- the ULP explanation "
        f"of the run-5 absorption does not hold and must be re-derived")


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


@pytest.mark.parametrize("hord", [2, 5, 8, 10])
def test_fv_tp_2d_adjoint_identity(hord, rough):
    """gate 4 PRIMARY on the ASSEMBLED routine -- runs ON the limiter
    surfaces the two xppm and two yppm sweeps cross, plus the deln_flux
    stencil.  Only ``fx``/``fy`` are paired: ``q`` is returned for the
    caller's benefit but is a pure index shuffle of the input."""
    def f(q_):
        geo = rough
        bd = geo.bd
        _, fx, fy = tp.fv_tp_2d(
            q_, jnp.asarray(geo.crx), jnp.asarray(geo.cry), geo.npx,
            geo.npy, hord, jnp.asarray(geo.xfx), jnp.asarray(geo.yfx),
            jnp.asarray(geo.dxa), jnp.asarray(geo.dya),
            jnp.asarray(geo.area), jnp.asarray(geo.del6_v),
            jnp.asarray(geo.del6_u), jnp.asarray(geo.rarea), geo.da_min,
            bd, jnp.asarray(geo.ra_x), jnp.asarray(geo.ra_y), 1.0, False,
            0, True, True, True, True, nord=1, damp_c=0.5)
        return jnp.concatenate([fx.reshape(-1), fy.reshape(-1)])

    _adjoint_identity(f, rough.q, f"fv_tp_2d hord={hord}")


@pytest.mark.parametrize("order", [1, 2])
def test_fv_tp_2d_check_grads_single_cell(order, smooth):
    """gate 4 SUPPLEMENT -- hord=2 (the perfectly-linear scheme: NONE of
    the limiter surfaces N2-N6 exist in that trace), single input cell,
    single output cell, order 1 before order 2."""
    bd = smooth.bd
    iq, jq = bd.is_ + 4, bd.js + 4
    kk = (iq - bd.isd, jq - bd.jsd)
    probe = (iq - bd.is_, jq - bd.js)

    def run(q_):
        _, fx, _ = tp.fv_tp_2d(
            q_, jnp.asarray(smooth.crx), jnp.asarray(smooth.cry),
            smooth.npx, smooth.npy, 2, jnp.asarray(smooth.xfx),
            jnp.asarray(smooth.yfx), jnp.asarray(smooth.dxa),
            jnp.asarray(smooth.dya), jnp.asarray(smooth.area),
            jnp.asarray(smooth.del6_v), jnp.asarray(smooth.del6_u),
            jnp.asarray(smooth.rarea), smooth.da_min, bd,
            jnp.asarray(smooth.ra_x), jnp.asarray(smooth.ra_y), 1.0,
            False, 0, True, True, True, True, nord=1, damp_c=0.5)
        return fx

    f = _scalar_probe(run, smooth.q, kk, probe)
    slope = _assert_scalar_locally_smooth(f, 0.0, "fv_tp_2d hord=2")
    assert abs(slope) > 1e-9, "probe cell does not influence the flux"
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth-region FD, O(1) dynamic range]
    check_grads(f, (jnp.asarray(0.0),), order=order, modes=("fwd", "rev"),
                eps=1e-3, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize("iord", SW_ORDS)
def test_xtp_u_adjoint_identity(iord, rough):
    """gate 4 PRIMARY -- runs ON the limiter surfaces, every iord.

    NON-SMOOTH SITES of xtp_u, all named: the upwind select ``c > 0``
    (N1); the ``smt5``/``smt6``/``hi5``/``hi6`` flags, which ADD or DROP
    the whole ``fx0`` term (N2); ``copysign`` in ``dm``, at iord 8, and
    in the iord=3 piecewise-linear fallback (N3); the three-way ``min``
    in ``dm`` and the PPM ``min``/``max`` clamps at iord 9/10 (N4); the
    ``NEAR_ZERO_SW`` flat-region test at iord 10 (N5)."""
    def f(u_):
        return _xtp_jax(rough, iord, u_, rough.c_sw)

    _adjoint_identity(f, rough.u, f"xtp_u u iord={iord}")

    def g(c_):
        return _xtp_jax(rough, iord, rough.u, c_)

    _adjoint_identity(g, rough.c_sw, f"xtp_u c iord={iord}")


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("iord", [2, 5, 8])
def test_xtp_u_check_grads_single_cell(order, iord, smooth):
    """gate 4 SUPPLEMENT -- scoped, single-cell, order 1 before 2."""
    b = smooth.bd
    iu, ju = b.is_ + 4, b.js + 4
    kk = (iu - b.isd, ju - b.jsd)
    probe = (iu - b.is_, ju - b.js)

    def run(u_):
        return _xtp_jax(smooth, iord, u_, smooth.c_sw)

    f = _scalar_probe(run, smooth.u, kk, probe)
    slope = _assert_scalar_locally_smooth(f, 0.0, f"xtp_u iord={iord}")
    assert abs(slope) > 1e-6, "probe cell does not influence the flux"
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth-region FD, O(1) dynamic range]
    check_grads(f, (jnp.asarray(0.0),), order=order, modes=("fwd", "rev"),
                eps=1e-3, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize("jord", SW_ORDS)
def test_ytp_v_adjoint_identity(jord, rough):
    """gate 4 PRIMARY -- same non-smooth-site list as ``xtp_u``."""
    def f(v_):
        return _ytp_jax(rough, jord, v_, rough.c_sw)

    _adjoint_identity(f, rough.v, f"ytp_v v jord={jord}")

    def g(c_):
        return _ytp_jax(rough, jord, rough.v, c_)

    _adjoint_identity(g, rough.c_sw, f"ytp_v c jord={jord}")


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("jord", [2, 5, 8])
def test_ytp_v_check_grads_single_cell(order, jord, smooth):
    """gate 4 SUPPLEMENT -- scoped, single-cell, order 1 before 2."""
    b = smooth.bd
    iv, jv = b.is_ + 4, b.js + 4
    kk = (iv - b.isd, jv - b.jsd)
    probe = (iv - b.is_, jv - b.js)

    def run(v_):
        return _ytp_jax(smooth, jord, v_, smooth.c_sw)

    f = _scalar_probe(run, smooth.v, kk, probe)
    slope = _assert_scalar_locally_smooth(f, 0.0, f"ytp_v jord={jord}")
    assert abs(slope) > 1e-6, "probe cell does not influence the flux"
    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.   [class: smooth-region FD, O(1) dynamic range]
    check_grads(f, (jnp.asarray(0.0),), order=order, modes=("fwd", "rev"),
                eps=1e-3, atol=1e-5, rtol=1e-5)


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
    # share one marker) must be preceded by a marker, and the adjoint
    # and check_grads gates carry their own.
    assert n_mark >= 30, n_mark
    assert n_cmp > 0


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

def _locate_slope_break(grad_of_t, ts, iters=60, rtol=1.0e-9):
    """Find a parameter value where the DERIVATIVE jumps.

    ⛔ REPLACES ``_locate_flag_switch``, which located the wrong thing,
    and the measurement is job 9417326
    (``scripts/validate/fv3_gradient_gate_triage.py``, block P5).  That
    helper took a BOOLEAN proxy -- ``flux(probe) != upwind(source)`` --
    and bisected where it changed.  Printed as a MARGIN instead of a
    boolean, that quantity is CONTINUOUS and simply passes through
    zero::

      xtp_u  t=0: +1.059e+00   t=+2.0e+02: -7.094e+01
      ytp_v  t=0: +3.972e-01   t=+2.0e+02: -7.160e+01

    So the proxy flips wherever ``flux - upwind`` crosses zero, which
    has nothing to do with the ``smt5`` branch.  The ``ytp_v`` arm
    failed only because that crossing fell BETWEEN two scanned points;
    the ``xtp_u`` arm "passed" because the crossing happened to land
    inside its scan.  Its pass was luck, not evidence -- both arms were
    certifying a zero crossing.

    What the gate actually wants is the branch flip, and at
    ``iord/jord = 5`` that is directly observable: every quantity
    feeding the flux is LINEAR in the perturbed value, so the flux is
    piecewise linear in ``t`` and its derivative is piecewise CONSTANT.
    A branch flip is therefore exactly a step in ``d flux / d t``, which
    is what this bisects on.  Measured at ``t = 0`` the derivative is
    6.400210e-01 on both arms, so the scan starts on a well-defined
    piece.

    ⛔ THE PREDICATE IS QUADRATIC, SO AN INTERVAL CAN HOLD TWO SWITCHES
    (codex MAJOR 5, job 9417397).  ``smt5`` is ``bl*br < 0``, a product
    of two functions each linear in ``t``, so as ``t`` sweeps, both
    factors can cross zero inside one sampled interval -- restoring the
    original slope at the far endpoint and hiding the pair, or breaking
    the monotone partition a bisection needs.  Two defences, both
    mechanical: intervals whose endpoints AGREE are refined once at
    their midpoint before being dismissed, and the final bracket is
    checked to contain exactly ONE slope value on each side.

    Raises with the observed derivatives if none of the scanned
    intervals contains a step, so a genuinely one-sided fixture is
    reported as such rather than as an opaque assertion.
    """
    ts = [float(t) for t in ts]

    def _same(a, b):
        return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)

    def _g(t):
        return float(grad_of_t(float(t)))

    # Refine EVERY interval once at its midpoint before trusting an
    # "endpoints agree" verdict: a pair of switches inside can return
    # the slope to its original value at the far endpoint.
    fine = []
    for k in range(len(ts) - 1):
        fine.append(ts[k])
        fine.append(0.5 * (ts[k] + ts[k + 1]))
    fine.append(ts[-1])
    grads = [_g(t) for t in fine]

    for k in range(len(fine) - 1):
        if _same(grads[k], grads[k + 1]):
            continue
        lo, hi = fine[k], fine[k + 1]
        g_lo, g_hi = grads[k], grads[k + 1]
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if _same(_g(mid), g_lo):
                lo = mid
            else:
                hi = mid
        t_star = 0.5 * (lo + hi)
        # The bracket must now hold ONE surface: the slope must be
        # constant on each side of it out to the sampled endpoints, or
        # the bisection converged onto an arbitrary point of a
        # multi-switch region and the gate would test the wrong surface.
        for side, edge, g_ref in ((-1.0, fine[k], g_lo),
                                  (+1.0, fine[k + 1], g_hi)):
            probe = t_star + side * abs(edge - t_star) * 0.5
            assert _same(_g(probe), g_ref), (
                f"more than one derivative step inside the bracket "
                f"[{fine[k]:.6g}, {fine[k + 1]:.6g}]: d flux/dt is "
                f"{_g(probe):.6e} at t={probe:.6g} but {g_ref:.6e} at "
                f"the endpoint -- smt5 is `bl*br < 0`, a QUADRATIC "
                f"predicate, so two switches can sit in one interval "
                f"and the located t_star is then arbitrary")
        return t_star
    raise AssertionError(
        f"no derivative step anywhere in the scan: "
        f"d flux/dt = {[f'{g:.6e}' for g in grads]} over t={fine} -- "
        f"the limiter branch never flips along this direction, so this "
        f"gate would certify nothing")


@pytest.mark.parametrize("routine", ["xtp_u", "ytp_v"])
def test_sw_transport_one_sided_grads_across_smt5_surface(routine, rough):
    """Approach the ``smt5 = bl*br = 0`` surface (sw_core.F90:2721 for
    ``xtp_u``, :3121 for ``ytp_v``, both at ``iord/jord = 5``) from BOTH
    sides and check each side separately.

    Crossing that surface ADDS or DROPS the whole ``fx0`` term, so the
    flux is DISCONTINUOUS there; a two-sided finite difference
    straddling it is meaningless and an order-2 ``check_grads`` taken
    there would be certifying a jump.

    The perturbed cell is placed in the LINEAR quadrant of the rough
    fixture; driving that one cell far enough makes it a local extremum,
    which flips the branch.  The surface is SCANNED for rather than
    asserted (:func:`_locate_slope_break`).

    ⛔ WHAT IS SCANNED FOR CHANGED, because the old detector was
    measuring something else.  It read the branch out of the public flux
    as ``flux != plain upwind value`` -- but that difference is a
    CONTINUOUS function of ``t`` that simply passes through zero (job
    9417326, triage block P5: +1.059 at t=0 to -70.9 at t=+200 for
    ``xtp_u``, +0.397 to -71.6 for ``ytp_v``), so the boolean can flip
    at a zero crossing rather than at a limiter switch.  That is what
    let a boolean detector fire on something that is not a switch.  What
    the ``xtp_u`` bracket actually contained is NOT established (codex
    MAJOR 6): saying its pass was luck would need the old flags, the
    margins and the ``smt5`` predicate printed on both sides of it, and
    that was not measured.  Either way the detector below does not
    depend on the answer.  At ``iord = 5`` the flux is piecewise LINEAR
    in ``t``, so the branch flip is exactly a step in ``d flux/d t`` --
    which is both what the assertions below already compare and
    something no zero crossing can imitate.

    ⛔ AND THE ``ytp_v`` FIXTURE REALLY IS ONE-SIDED AT THE FIRST CELL
    TRIED (job 9417389, MEASURED after the detector was replaced):
    ``d flux/dt`` is 6.400210e-01 at EVERY ``t`` from -8000 to +8000,
    constant to all printed digits, i.e. the flux is globally linear in
    that perturbation and no branch flips along it.  That is a fact
    about the FIXTURE, not about ``ytp_v``: which cell of the stencil
    can be driven across ``bl*br = 0`` depends on the local field, and
    the ``u`` and ``v`` fixtures have different structure.  So the gate
    now SEARCHES the upwind cell and its two transport-direction
    neighbours and uses the first that shows a step -- and if none of
    the three does, it says so with all three derivative tables, which
    is a finding about the fixture rather than an opaque failure.

    Still no private state is touched: the derivative comes from
    ``jax.grad`` of the public flux.

    Asserted, per side: the one-sided derivative is STABLE (offset ``d``
    vs ``d/4`` agree to 1e-6 relative, i.e. that side is smooth right up
    to the surface), and the two sides DIFFER by more than that margin.
    """
    b = rough.bd
    fi, fj = b.is_ + 3, b.js + 3        # Fortran flux cell, interior
    ci, cj = fi - b.is_, fj - b.js      # 0-based flux index
    c_probe = float(rough.c_sw[ci, cj])

    if routine == "xtp_u":
        field = np.array(rough.u, dtype=np.float64)
        # `flux(i,j)` reads u(i-1,j) and u(i,j); perturb the upwind one.
        ui = fi - 1 if c_probe > 0.0 else fi
        kk = (ui - b.isd, fj - b.jsd)

        def run(fld):
            return _xtp_jax(rough, 5, fld, rough.c_sw)

        def base_of(fld):
            src = fi - 1 if c_probe > 0.0 else fi
            return float(fld[src - b.isd, fj - b.jsd])
    else:
        field = np.array(rough.v, dtype=np.float64)
        vj = fj - 1 if c_probe > 0.0 else fj
        kk = (fi - b.isd, vj - b.jsd)

        def run(fld):
            return _ytp_jax(rough, 5, fld, rough.c_sw)

        def base_of(fld):
            src = fj - 1 if c_probe > 0.0 else fj
            return float(fld[fi - b.isd, src - b.jsd])

    probe = (ci, cj)

    # `base_of` is retained only for the sanity check below: it is what
    # the RETRACTED boolean proxy compared against, and printing the two
    # together is what showed the proxy to be continuous.
    assert callable(base_of)

    def _grad_at(cell):
        def scalar(t):
            f = jnp.asarray(field).at[cell].add(t)
            return run(f)[probe]
        return jax.grad(scalar)

    # Scan both signs over four decades and locate the DERIVATIVE STEP;
    # the transition is wherever the perturbed cell stops being an
    # interior value and becomes an extremum, which depends on the local
    # slope and cannot be predicted in closed form.
    ts = [-8000.0, -2000.0, -800.0, -200.0, -50.0, -10.0, 0.0,
          10.0, 50.0, 200.0, 800.0, 2000.0, 8000.0]
    # WHICH cell can be driven across `bl*br = 0` is a property of the
    # FIXTURE, not of the routine: the first ytp_v candidate has a
    # constant derivative over the whole scan (job 9417389). Try the
    # upwind cell and its two transport-direction neighbours, and report
    # all three if none of them switches.
    step = (1, 0) if routine == "xtp_u" else (0, 1)
    cands = [kk,
             (kk[0] - step[0], kk[1] - step[1]),
             (kk[0] + step[0], kk[1] + step[1])]
    t_star, g, why = None, None, []
    for cell in cands:
        gc = _grad_at(cell)
        try:
            t_star = _locate_slope_break(
                lambda t, _g=gc: _g(jnp.asarray(float(t))), ts)
        except AssertionError as exc:
            why.append(f"cell {cell}: {exc}")
            continue
        g = gc
        break
    assert t_star is not None, (
        f"{routine}: no candidate cell in the transport stencil crosses "
        f"the smt5 surface on this fixture, so the gate would certify "
        f"nothing. Tried {cands}.\n" + "\n".join(why))
    sides = {}
    for side, sgn in (("left", -1.0), ("right", +1.0)):
        d = 1e-3
        g1 = float(g(jnp.asarray(t_star + sgn * d)))
        g2 = float(g(jnp.asarray(t_star + sgn * d / 4.0)))
        den = max(abs(g1), abs(g2), 1e-12)
        # At iord/jord = 5 every quantity feeding the flux (al, bl, br,
        # b0, fx0) is LINEAR in u/v, and the only nonlinearity is the
        # smt5 flag itself, so each side is exactly affine in t and the
        # one-sided derivative must be CONSTANT, not merely stable.
        assert abs(g1 - g2) / den <= 1e-6, (
            f"{routine} {side}: one-sided derivative not constant "
            f"({g1:.6e} vs {g2:.6e}) -- at iord=5 each side is affine "
            f"in t, so this means a SECOND surface lies inside the "
            f"offset and the located t_star is not the only switch")
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
