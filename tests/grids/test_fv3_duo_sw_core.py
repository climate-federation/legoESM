"""Certification of the JAX ``sw_core`` lane (``c_sw`` + the six
``d_sw`` duo stages) against the NumPy fp64 lane.

Authority: ``fv3_native_sw_core`` / ``fv3_native_duo_sw_core`` are the
SPECIFICATION (hop B of
``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).  Where a stored
Fortran certificate exists (the ``dsw{1..6}_duo_oracle_c12.npz``
family) it is used as a SECONDARY, clearly-labelled check -- it closes
hop A + hop B in one shot but it is not what the per-kernel bound is
calibrated against.

Five gates per public routine, following ``test_fv3_tp_core.py`` and
the first-run lessons in ``fv3_duo_jax_lane_STATE.md``:

1. **parity** -- JAX vs the NumPy twin on the committed C12 fixtures
   (``swcore_input.npz`` for the plain leaves and ``c_sw``'s plain arm,
   ``dswcore_input.npz`` for the duo leaves and the whole d_sw chain),
   with a bound MEASURED in job 9425294 and set to measured x 10;
2. **jit vs eager** -- an ASSERTION, plus a trace counter proving no
   retrace on new VALUES of the same shape (and, for the stages that
   take one, on a new ``dt``).  **Bitwise is asserted ONLY on paths with
   no floating-point sum**: XLA contracts ``x*y + z`` into an FMA in the
   jitted lowering and not in the eager one, so a few-ULP gap on any
   summing path is CORRECT, not a defect (STATE lesson 2).  The bitwise
   paths here are exactly the pure index copies -- ``fill_4corners`` and
   ``fill2_4corners`` -- and they are labelled as such at the site;
3. **guards** -- float32 raises ``TypeError``; an unsupported
   ``hord``/``nord``/``grid_type`` raises ``ValueError``; each guard
   test is shown NON-VACUOUS by monkeypatching the validator to a no-op
   and asserting the same call then returns an array;
4. **gradients** -- the PRIMARY gate is the tolerance-free adjoint
   identity ``<J v, w> == <v, J^T w>`` (``J v`` from ``jax.jvp``,
   ``J^T w`` from ``jax.vjp``).  It uses no finite differences, so
   neither the FD step nor the array's dynamic range can make it fail
   spuriously -- which is exactly what sank ``check_grads(order=2)`` in
   four of five modules on the first run (STATE lesson 11).
   ``check_grads(order=2)`` is kept as a SUPPLEMENT, scoped to a smooth
   region with O(1) dynamic range and guarded by an explicit
   one-sided-derivative smoothness control.  One-sided directional
   derivatives are taken across the named upwind switching surfaces from
   BOTH sides;
5. **conservation** -- ``d_sw1_duo``'s flux/capacitor bookkeeping, see
   ``test_d_sw1_capacitor_accumulation_is_exact`` for which invariant
   was chosen and why.

TOLERANCE POLICY.  These are NOT innocuous neighbour-reading kernels:
every transport call inside ``d_sw1``/``d_sw3``/``d_sw5`` runs the PPM
limiters, whose flags ADD or DROP a whole flux term, so the flux is
DISCONTINUOUS across them and a rounding-level lane difference near a
limiter surface can produce a discrepancy far above 1e-15.  Every
numeric bound below is MEASURED (job 9425294, the LEGOESM_FV3_TOL_MEASURE
sweep) and set to measured x 10, keeping its class label.  Lane parity
measured exactly 0.0 (bitwise) at every gate; the nonzero measurements
are jit-vs-eager, and the two limiter-flip cases (d_sw1 jit yflux
1.216e-10, d_sw3 jit ubbtemp 1.521e-10) carry the loosest bounds in the
file at 1.3e-9/1.6e-9.
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
from legoesm.core import fv3_duo_sw_core as duo  # noqa: E402
from legoesm.core import fv3_native_d_sw as npd  # noqa: E402
from legoesm.core import fv3_native_duo_sw_core as npduo  # noqa: E402
from legoesm.core import fv3_native_sw_core as npsw  # noqa: E402
from legoesm.core import fv3_tp_core as tp  # noqa: E402
from legoesm.core.fv3_native_sw_core import Bounds  # noqa: E402
from legoesm.grids.fv3_native_gridstruct import fort  # noqa: E402

from tests.grids.fv3_gate_helpers import gate_scalar  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

# npz entries that are SCALARS / run parameters, not gridstruct arrays.
_SCALARS = ("res", "ng", "dt", "dt2", "nord", "da_min", "da_min_c",
            "delp", "pt", "w", "u", "v", "uc", "vc", "ua", "va",
            "divg_d_in")


# =====================================================================
# fixtures and shared helpers
# =====================================================================

@pytest.fixture(scope="module")
def sw_inp():
    """c_sw / d2a2c_vect / divergence_corner PLAIN-lane inputs."""
    return np.load(os.path.join(FIX, "swcore_input.npz"))


@pytest.fixture(scope="module")
def dsw_inp():
    """The shared d_sw chain inputs (also used by every duo leaf)."""
    return np.load(os.path.join(FIX, "dswcore_input.npz"))


def _oracle(name):
    return np.load(os.path.join(FIX, name), allow_pickle=True)


def _np_gs(inp, **over):
    """The NumPy lane's mixed ``gridstruct`` dict (raw numpy + flags).

    Defaults match the committed per-stage certificate harnesses
    (``bounded_domain=False``, ``grid_type=0``, all four corner flags
    true), which is the plain-conventions lane those fixtures were
    generated on.
    """
    gs = {k: np.array(inp[k], dtype=np.float64) for k in inp.files
          if k not in _SCALARS}
    gs.update(bounded_domain=False, grid_type=0, sw_corner=True,
              se_corner=True, ne_corner=True, nw_corner=True)
    for k in ("da_min", "da_min_c"):
        if k in inp.files:
            gs[k] = float(inp[k])
    gs.update(over)
    return gs


def _jax_gs(gs_np):
    """The JAX lane's ARRAYS-ONLY gridstruct (the traced half)."""
    return {k: jnp.asarray(v) for k, v in gs_np.items()
            if isinstance(v, np.ndarray)}


class _Geo:
    """Bounds + both gridstruct halves for one fixture."""

    def __init__(self, inp, **over):
        self.res = int(inp["res"])
        self.ng = int(inp["ng"])
        self.bd = Bounds.single_tile(self.res, self.ng)
        self.npx = self.res + 1
        self.npy = self.res + 1
        self.m_a = self.res + 2 * self.ng
        self.gs_np = _np_gs(inp, **over)
        self.gs_j = _jax_gs(self.gs_np)
        self.flags = duo.GridFlags.from_gs(self.gs_np)
        self.f = {k: np.array(inp[k], dtype=np.float64)
                  for k in inp.files
                  if k in _SCALARS and np.asarray(inp[k]).ndim >= 1}
        self.dt = float(inp["dt"]) if "dt" in inp.files else None
        self.dt2 = float(inp["dt2"]) if "dt2" in inp.files else None
        # the deck's own nord, read from the fixture rather than assumed
        self.nord = int(inp["nord"]) if "nord" in inp.files else 1


@pytest.fixture(scope="module")
def geo_sw(sw_inp):
    return _Geo(sw_inp)


@pytest.fixture(scope="module")
def geo_dsw(dsw_inp):
    return _Geo(dsw_inp)


@pytest.fixture(scope="module")
def geo_bounded():
    """The BOUNDED-conventions gridstruct + state the duo runs execute.

    ``duogrid=True`` requires ``bounded_domain=True``
    (fv_arrays.F90:1512), and upstream sets the four corner flags only
    at ``.not.bounded`` (fv_grid_utils.F90:224) -- same fixture the
    NumPy duo c_sw gate uses, so the two lanes are compared on ONE
    configuration.
    """
    from legoesm.grids.fv3_native_gridstruct import (
        analytic_swcore_state,
        build_fv3_native_gridstruct_bounded,
    )
    res, ng = 12, 3
    gs = build_fv3_native_gridstruct_bounded(res, ng)
    gs = {k: v for k, v in gs.items()}
    gs["bounded_domain"] = True
    for k in ("sw_corner", "se_corner", "ne_corner", "nw_corner"):
        gs[k] = False
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(res, ng)
    return gs, st, bd, res + 1, res + 1


# The WORKSPACE FILL CONSTANTS this lane round-trips, by VALUE: 1e30
# (ut/vt, ptc, ub/vb, heat_source, delpc outside the B ring) and 1e25
# (divergence_corner_duo's divg_d init).  A cell is a fill only if it
# holds one of these EXACTLY -- see the retraction in _cmp's docstring
# for why a magnitude threshold was wrong.
_FILL_VALUES = (1.0e30, 1.0e25)


def _cmp(got, ref, name, tol):
    """Mask-aware, PER-ELEMENT relative comparison -- and it must be able
    to fail without also being able to fail spuriously.

    ⛔ RETRACTION (job 9404093).  The first version of this helper
    classified a cell as a "sentinel" by MAGNITUDE (``|x| >= 1e20``) and
    then demanded those cells be BITWISE equal.  That was wrong, and it
    produced three of this module's eight failures -- ``d_sw1``'s
    ``allflux_x``, ``d_sw5``'s ``vortfluxx`` and ``d_sw6``'s ``ut``, at
    values of 1e22 … 1e111.  Those cells are NOT workspace fills: they
    are the documented SENTINEL-PROPAGATED CASCADE.  ``d_sw1``'s
    plain-conventions edge/corner blocks read ut/vt cells the duo
    interior never writes, so ``ut`` picks up ~1e30, ``xfx = dt*ut``
    ~1e31, ``crx`` ~1e26, and ``fv_tp_2d`` on those Courant numbers
    yields 1e54 … 1e111.  Every one of those is ORDINARY ARITHMETIC and
    is therefore subject to XLA's FMA contraction under jit -- demanding
    it bitwise contradicted this file's own tolerance policy.  (The
    JAX-vs-NumPy parity gates passed on the same arrays because both
    lanes evaluate them eagerly.)

    The classification is now by VALUE against the known fill constants,
    and the value comparison is PER ELEMENT:

    * NON-FINITE mask equality -- a cell the oracle never writes must
      stay a tripwire in BOTH lanes;
    * EXACT-FILL mask equality (and, being constants, exact values) --
      ``1e30`` passes every ``isfinite`` guard, so a drifted fill is a
      real defect wearing a finite disguise;
    * every other cell: ``|a-b| / (|b| + scale)`` with ``scale`` the
      MEDIAN of ``|b|`` over those cells.  Per-element relative with a
      robust floor is the error model strategy section 4 asks for, and
      it fixes the other half of the original defect: a single 1e111
      cell can no longer set ``max|ref|`` and divide every physical
      discrepancy to nothing.

    Returns ``(rel, n_over)`` where ``n_over`` counts cells above a
    rounding-scale reference -- the BRANCH-FLIP signature.  A handful of
    cells far out with the rest at 1e-16 is a limiter branch flip; all
    cells uniformly out is something systematic.  The distribution is
    printed on failure so the next run does not have to guess.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)

    na, nb = ~np.isfinite(a), ~np.isfinite(b)
    assert np.array_equal(na, nb), (
        f"{name}: non-finite masks differ (jax {int(na.sum())} vs numpy "
        f"{int(nb.sum())} cells of {a.size})")

    fa = np.zeros(a.shape, bool)
    fb = np.zeros(b.shape, bool)
    for v in _FILL_VALUES:
        fa |= (a == v)
        fb |= (b == v)
    assert np.array_equal(fa, fb), (
        f"{name}: workspace-FILL masks differ (jax {int(fa.sum())} vs "
        f"numpy {int(fb.sum())} cells of {a.size}); the fill constants "
        f"are {list(_FILL_VALUES)}")

    ok = np.isfinite(a) & ~fa
    if not ok.any():
        # every cell is a fill or a tripwire. The mask checks above were
        # exact -- but a comparison with NO physical cell must not read
        # as "0.0 measured" (codex MAJOR). Loud in both modes.
        raise AssertionError(
            f"{name}: no physical cells to compare -- every cell is a "
            f"fill/tripwire; the fixture is vacuous for this gate")
    diff = np.abs(a[ok] - b[ok])
    scale = float(np.median(np.abs(b[ok])))
    if not (scale > 0.0):
        scale = max(float(np.abs(b[ok]).max()), 1e-300)
    den = np.abs(b[ok]) + scale
    per = diff / den
    rel = float(per.max())
    n_over = int((per > 1e-13).sum())
    # MEASUREMENT MODE (LEGOESM_FV3_TOL_MEASURE=1): print and skip ONLY
    # the tolerance assert; the structural checks above still raise.
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE {name!r}: rel {rel:.3e} (bound {tol:.3e}, "
              f"n_over {n_over}, max|diff| {float(diff.max()):.3e})",
              flush=True)
        return rel, n_over
    assert rel <= tol, (
        f"{name}: MEASURED per-element rel {rel:.3e} > {tol:.3e}; "
        f"{n_over} of {int(ok.sum())} compared cells exceed 1e-13 "
        f"(a handful => limiter BRANCH FLIP, all of them => systematic); "
        f"median|ref| {scale:.3e}, max|diff| {float(diff.max()):.3e}, "
        f"bitwise={np.array_equal(a[ok], b[ok])}")
    return rel, n_over


def _jit_gap(got, ref, name, *, max_cells, tail_ratio=None):
    """COUNT-gated jit-vs-eager comparison for the two keys whose gap is
    real but is NOT a tolerance question.

    Rationale (coordinator, run 5).  Two of this module's jit gates show
    a per-element relative gap far above rounding on a SMALL NUMBER of
    cells.  Loosening the bound to cover them would also hide a future
    regression to a systematic failure, which is strictly worse than a
    red gate.  So the ASSERTION is on the cell COUNT -- tolerance-free,
    and it goes red exactly when "a handful" becomes "systematic" -- and
    the magnitude is REPORTED, labelled UNEXPLAINED-BY-TOLERANCE, rather
    than certified.

    ``tail_ratio``: when given, every violating cell must additionally
    have ``|ref|`` at least this many times the MEDIAN ``|ref|`` of the
    NON-violating cells.  That is the sentinel-cascade claim made
    falsifiable without needing an absolute physical ceiling: a cascade
    cell sits in the extreme tail by construction, whereas a PHYSICAL
    cell that started violating would sit near the median and go red.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    fin = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), (
        f"{name}: non-finite masks differ")
    keep = fin & (a != 1.0e30) & (b != 1.0e30)
    assert keep.any(), f"{name}: nothing to compare"
    scale = float(np.median(np.abs(b[keep]))) or 1e-300
    per = np.zeros(a.shape)
    per[keep] = np.abs(a[keep] - b[keep]) / (np.abs(b[keep]) + scale)
    bad = keep & (per > 1e-13)
    n_bad = int(bad.sum())
    n_cmp = int(keep.sum())

    if n_bad:
        k = np.unravel_index(int(np.argmax(per)), per.shape)
        detail = (
            f"{name}: MEASURED max per-element rel {per.max():.3e} at "
            f"cell {tuple(int(x) for x in k)} where |ref| = "
            f"{abs(float(b[k])):.6e}; {n_bad} of {n_cmp} cells exceed "
            f"1e-13; |ref| over violators "
            f"[{np.abs(b[bad]).min():.3e}, {np.abs(b[bad]).max():.3e}], "
            f"median|ref| over the rest "
            f"{float(np.median(np.abs(b[keep & ~bad]))):.3e}")
    else:
        detail = f"{name}: no cell exceeds 1e-13"

    assert n_bad <= max_cells, (
        f"CELL-COUNT GATE RED -- {detail}. A handful of cells is the "
        f"known isolated-amplification signature; {n_bad} > "
        f"{max_cells} means it has become SYSTEMATIC, which is a "
        f"different defect and must not be absorbed by a tolerance.")

    if tail_ratio is not None and n_bad:
        med_rest = float(np.median(np.abs(b[keep & ~bad])))
        worst = float(np.abs(b[bad]).min())
        assert worst >= tail_ratio * med_rest, (
            f"a violating cell is NOT in the extreme tail: min|ref| over "
            f"violators {worst:.3e} < {tail_ratio} x median|ref| over "
            f"the rest {med_rest:.3e} -- so at least one PHYSICAL cell "
            f"is now violating, which is a separate finding. {detail}")
    return per, bad, detail


def _tree_dot(a, b) -> float:
    """Plain inner product -- NO ``nan_to_num``.

    Sanitising here would silently repair a NaN that the adjoint
    identity is supposed to EXPOSE (an R1b dead-branch leak shows up as
    exactly that), so a non-finite leaf must propagate to the assertion
    in :func:`_check_adjoint` and name itself there.
    """
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return float(sum(
        np.dot(np.asarray(x, dtype=np.float64).ravel(),
               np.asarray(y, dtype=np.float64).ravel())
        for x, y in zip(la, lb)))


def _adjoint_residual(f, primals, seed=0):
    """Relative residual of ``<J v, w> == <v, J^T w>``.

    NO finite differences: ``J v`` comes from ``jax.jvp`` and ``J^T w``
    from ``jax.vjp``, so the identity is exact in exact arithmetic and
    the residual is pure floating-point roundoff.  Its power does NOT
    depend on the FD step, on operand scaling, or on the output's
    dynamic range -- which is the whole reason it replaces
    ``check_grads`` as the primary gradient gate here.

    Returns ``(relative residual, <J v, w>, <v, J^T w>)``.
    """
    rng = np.random.default_rng(seed)
    primals = tuple(jnp.asarray(p) for p in primals)
    v = tuple(jnp.asarray(rng.standard_normal(p.shape)) for p in primals)
    _, jv = jax.jvp(f, primals, v)
    for i, leaf in enumerate(jax.tree_util.tree_leaves(jv)):
        assert np.isfinite(np.asarray(leaf)).all(), (
            f"J v leaf {i} is not finite -- the objective window "
            f"includes cells the kernel never writes, or a dead branch "
            f"is leaking a NaN into the gradient (R1b)")
    _, vjp_fn = jax.vjp(f, *primals)
    w = jax.tree_util.tree_map(
        lambda x: jnp.asarray(rng.standard_normal(x.shape)), jv)
    jtw = vjp_fn(w)
    lhs = _tree_dot(jv, w)
    rhs = _tree_dot(v, jtw)
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300), lhs, rhs


def _check_adjoint(name, f, primals, tol, seed=0):
    r, lhs, rhs = _adjoint_residual(f, primals, seed=seed)
    assert abs(lhs) > 0.0, (
        f"{name}: <J v, w> == 0 -- the identity is satisfied trivially, "
        f"so this gate proves nothing (check the window/scale)")
    assert np.isfinite(lhs) and np.isfinite(rhs), (
        f"{name}: the inner products are not finite ({lhs}, {rhs})")
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE {name!r}: rel {r:.3e} (bound {tol:.3e}, "
              f"quantity adjoint identity residual)", flush=True)
        return r
    assert r <= tol, (
        f"{name}: adjoint residual {r:.3e} > {tol:.3e} "
        f"(<J v, w>={lhs:.12e}, <v, J^T w>={rhs:.12e}) -- MEASURED "
        f"value is {r:.3e}")
    return r


def _assert_locally_smooth(f, x, dx, name, eps=1e-7, tol=1e-6):
    """No switching surface inside the finite-difference ball.

    Compares the LEFT and RIGHT one-sided directional derivatives: at a
    limiter kink (or a flux discontinuity) they differ at O(1), inside a
    smooth patch they agree to O(eps).  This is the control that
    licenses the order-2 ``check_grads`` supplement -- without it a
    "grads pass" claim is a claim about an arbitrary state.
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


def _counted(fn):
    """(traced-call counter, wrapper) for the no-retrace assertions."""
    box = {"n": 0}

    def wrapper(*a, **k):
        box["n"] += 1
        return fn(*a, **k)

    return box, wrapper


# =====================================================================
# bookkeeping: the module's own claims about constants and scheme sets
# =====================================================================

def test_scheme_sets_match_tp_core():
    """The RESTATED order sets must equal ``fv3_tp_core``'s.

    They are restated rather than imported because CLAUDE.md's ratchet
    forbids a cross-module PRIVATE import; this test is what makes the
    duplication safe -- a scheme added to tp_core's guard set without
    being added here (or vice versa) goes red.  Attribute access on a
    private symbol from a TEST is not an import and is the established
    pattern in ``test_fv3_tp_core.py``.
    """
    assert tuple(duo._PPM_ORDS) == tuple(tp._PPM_ORDS)
    assert tuple(duo._SW_ORDS) == tuple(tp._SW_ORDS)


def test_constants_are_the_numpy_twins():
    """R2: the interpolation constants are IMPORTED, not restated."""
    assert duo.A1 is npsw.A1 and duo.A2 is npsw.A2
    assert duo.C1 is npsw.C1 and duo.C2 is npsw.C2
    assert duo.C3 is npsw.C3
    assert duo.BIG_NUMBER == npsw.BIG_NUMBER == 1.0e30
    # the four literals that are NOT exported by the twin, each with a
    # pinned sw_core.F90 line in the module
    assert duo._CORNER_DAMP_COEF == 0.0625      # sw_core.F90:740
    assert duo._DDDMP_CAP == 0.20               # sw_core.F90:1816
    assert duo._DDDMP_OFF == 1.0e-5             # sw_core.F90:1790
    assert duo._D_CON_ON == 1.0e-5              # sw_core.F90:1824
    assert duo._DAMP_V_ON == 1.0e-5             # sw_core.F90:1948
    assert duo._DAMP_W_ON == 1.0e-5             # d_sw2 damp_w gate


def test_gridflags_round_trips_the_numpy_dict(geo_dsw):
    """The dict -> NamedTuple split must not drop or default a field."""
    f = duo.GridFlags.from_gs(geo_dsw.gs_np)
    assert f.da_min == geo_dsw.gs_np["da_min"]
    assert f.da_min_c == geo_dsw.gs_np["da_min_c"]
    assert f.bounded_domain is False and f.grid_type == 0
    assert (f.sw_corner, f.se_corner, f.nw_corner, f.ne_corner) == \
        (True, True, True, True)
    # hashable BY VALUE -> one jit cache entry per value
    assert hash(f) == hash(duo.GridFlags.from_gs(geo_dsw.gs_np))


def test_no_donate_argnums_in_this_lane():
    """Strategy R4: buffer donation conflicts with reverse-mode AD, and
    this lane exists to be differentiated.

    The discriminator is the USE (``donate_argnums=``), not the word --
    the module docstring and the jit-factory policy comment both mention
    the name in prose, so a bare substring test would be vacuous.
    """
    import inspect
    src = inspect.getsource(duo)
    assert "donate_argnums=" not in src
    assert "donate_argnums" in src, (
        "the doctrine comment naming the ban has gone missing")


# =====================================================================
# edge_interpolate4
# =====================================================================

def _edge4_fixture(seed=11):
    rng = np.random.default_rng(seed)
    ua = [np.asarray(rng.standard_normal(9)) for _ in range(4)]
    dxa = [np.asarray(1.0 + 0.2 * rng.random(9)) for _ in range(4)]
    return ua, dxa


def test_edge_interpolate4_parity():
    """gate 1.  FIXTURE CLASS: smooth rational, no branch at all."""
    ua, dxa = _edge4_fixture()
    ref = np.empty(9)
    for n in range(9):
        ref[n] = npsw.edge_interpolate4([a[n] for a in ua],
                                        [d[n] for d in dxa])
    got = duo.edge_interpolate4([jnp.asarray(a) for a in ua],
                                [jnp.asarray(d) for d in dxa])
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, ref, "edge_interpolate4", 1e-15)


def test_edge_interpolate4_jit_equals_eager():
    """gate 2 -- ASSERTED.  NOT bitwise: the body contains
    ``(t1 + dxa4[1]) * ua4[1] - dxa4[1] * ua4[0]``, i.e. a mul+add that
    XLA may contract into an FMA when jitted and not when eager."""
    ua, dxa = _edge4_fixture()
    ua = [jnp.asarray(a) for a in ua]
    dxa = [jnp.asarray(d) for d in dxa]
    eager = duo.edge_interpolate4(ua, dxa)
    jitted = duo.make_edge_interpolate4_jit()(ua, dxa)
    # MEASURED (job 9425294 sweep): 1.225e-16; bound = measured x 10 =
    # 1.3e-15.
    _cmp(jitted, eager, "edge_interpolate4 jit", 1.3e-15)


def test_edge_interpolate4_rejects_float32():
    """gate 3 -- guard."""
    ua, dxa = _edge4_fixture()
    ua[0] = np.asarray(ua[0], np.float32)
    with pytest.raises(TypeError, match="float64"):
        duo.edge_interpolate4([jnp.asarray(a) for a in ua],
                              [jnp.asarray(d) for d in dxa])


def test_edge_interpolate4_gradients():
    """gate 4 -- adjoint identity (primary) + order-2 (supplement).

    The routine is a smooth rational function of ``ua4`` at fixed
    ``dxa4`` with no switching surface, so this is the one place where
    order-2 finite differencing is genuinely appropriate.
    """
    ua, dxa = _edge4_fixture()
    dxa_j = [jnp.asarray(d) for d in dxa]

    def f(a0, a1, a2, a3):
        return duo.edge_interpolate4([a0, a1, a2, a3], dxa_j)

    primals = tuple(jnp.asarray(a) for a in ua)
    # MEASURED (job 9425294 sweep): adjoint residual 1.867e-16; bound = measured x 10 =
    # 1.9e-15.
    _check_adjoint("edge_interpolate4", f, primals, 1.9e-15)
    check_grads(lambda a: jnp.sum(f(a, *primals[1:]) ** 2),
                (primals[0],), order=2, modes=("fwd", "rev"))


# =====================================================================
# fill_4corners / fill2_4corners -- the ONLY bitwise-asserted paths
# =====================================================================

def _corner_field(geo, seed=3):
    rng = np.random.default_rng(seed)
    n = geo.m_a
    return 100.0 + rng.standard_normal((n, n))


@pytest.mark.parametrize("direction", [1, 2])
def test_fill_4corners_parity_bitwise(direction, geo_sw):
    """gate 1.  FIXTURE CLASS: PURE INDEX COPY -- no arithmetic at all,
    so bitwise equality is the right assertion and there is no FMA site
    for XLA to contract.  This is one of only two bitwise gates here."""
    q = _corner_field(geo_sw)
    qf = fort(q.copy(), geo_sw.bd.isd, geo_sw.bd.jsd)
    npsw.fill_4corners(qf, direction, geo_sw.npx, geo_sw.npy)
    got = duo.fill_4corners(jnp.asarray(q), direction, geo_sw.npx,
                            geo_sw.npy, geo_sw.bd)
    assert not np.array_equal(qf.a, q), "the fill moved nothing"
    assert np.array_equal(np.asarray(got), qf.a), "not bitwise"


@pytest.mark.parametrize("direction", [1, 2])
def test_fill2_4corners_parity_bitwise(direction, geo_sw):
    """gate 1, same class as above, on the two-operand form."""
    q1 = _corner_field(geo_sw, 3)
    q2 = _corner_field(geo_sw, 4)
    f1 = fort(q1.copy(), geo_sw.bd.isd, geo_sw.bd.jsd)
    f2 = fort(q2.copy(), geo_sw.bd.isd, geo_sw.bd.jsd)
    npsw.fill2_4corners(f1, f2, direction, geo_sw.npx, geo_sw.npy)
    g1, g2 = duo.fill2_4corners(jnp.asarray(q1), jnp.asarray(q2),
                                direction, geo_sw.npx, geo_sw.npy,
                                geo_sw.bd)
    assert np.array_equal(np.asarray(g1), f1.a)
    assert np.array_equal(np.asarray(g2), f2.a)
    assert not np.array_equal(f1.a, q1) and not np.array_equal(f2.a, q2)


def test_fill_4corners_jit_equals_eager_bitwise(geo_sw):
    """gate 2 -- bitwise is legitimate here (index copy only)."""
    q = jnp.asarray(_corner_field(geo_sw))
    args = (2, geo_sw.npx, geo_sw.npy, geo_sw.bd)
    eager = duo.fill_4corners(q, *args)
    box, wrapped = _counted(duo.fill_4corners)
    fn = duo.make_fill_4corners_jit(wrapped)
    jitted = fn(q, *args)
    fn(q * 1.01, *args)
    assert box["n"] == 1, box["n"]
    assert np.array_equal(np.asarray(jitted), np.asarray(eager))


def test_fill_4corners_flags_are_honoured(geo_sw):
    """A disabled corner must leave its cells untouched -- otherwise the
    flag test below could pass on a no-op implementation."""
    q = jnp.asarray(_corner_field(geo_sw))
    all_on = duo.fill_4corners(q, 1, geo_sw.npx, geo_sw.npy, geo_sw.bd)
    none = duo.fill_4corners(q, 1, geo_sw.npx, geo_sw.npy, geo_sw.bd,
                             sw=False, se=False, ne=False, nw=False)
    assert np.array_equal(np.asarray(none), np.asarray(q))
    assert not np.array_equal(np.asarray(all_on), np.asarray(q))


def test_fill_4corners_direction_guard_and_non_vacuity(geo_sw):
    """gate 3 -- an unknown direction RAISES.

    NON-VACUITY: sw_core.F90:3858/:3887 branch on 1 and 2 only, so
    WITHOUT the raise the function would return ``q`` UNFILLED -- a
    silent wrong answer, not an error.  The second half shows the guard
    is what fires: the same call at dir=1 succeeds and CHANGES q.
    """
    q = jnp.asarray(_corner_field(geo_sw))
    with pytest.raises(ValueError, match="direction"):
        duo.fill_4corners(q, 3, geo_sw.npx, geo_sw.npy, geo_sw.bd)
    with pytest.raises(ValueError, match="direction"):
        duo.fill2_4corners(q, q, 0, geo_sw.npx, geo_sw.npy, geo_sw.bd)
    ok = duo.fill_4corners(q, 1, geo_sw.npx, geo_sw.npy, geo_sw.bd)
    assert not np.array_equal(np.asarray(ok), np.asarray(q))


def test_fill_4corners_rejects_float32(geo_sw):
    with pytest.raises(TypeError, match="float64"):
        duo.fill_4corners(jnp.asarray(_corner_field(geo_sw), jnp.float32),
                          1, geo_sw.npx, geo_sw.npy, geo_sw.bd)


def test_fill_4corners_gradients(geo_sw):
    """gate 4 -- a gather/scatter is LINEAR in ``q``; there is no
    switching surface at all, so both gradient gates must be exact."""
    q = jnp.asarray(_corner_field(geo_sw))

    def f(x):
        return duo.fill_4corners(x, 1, geo_sw.npx, geo_sw.npy, geo_sw.bd)

    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise) adjoint residual; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _check_adjoint("fill_4corners", f, (q,), 1e-15)
    check_grads(lambda x: jnp.sum(f(x) ** 2), (q,), order=2,
                modes=("fwd", "rev"))


# =====================================================================
# d2a2c_vect (plain) and d2a2c_vect_duo
# =====================================================================

_D2A2C_OUT = ("ua", "va", "uc", "vc", "ut", "vt")


def _d2a2c_np(geo, u, v, **kw):
    return dict(zip(_D2A2C_OUT,
                    npsw.d2a2c_vect(u, v, geo.gs_np, geo.bd, geo.npx,
                                    geo.npy, **kw)))


def _d2a2c_jax(geo, u, v, fn=None, **kw):
    f = fn or duo.d2a2c_vect
    return dict(zip(_D2A2C_OUT,
                    f(jnp.asarray(u), jnp.asarray(v), geo.gs_j, geo.bd,
                      geo.npx, geo.npy, **kw)))


def test_d2a2c_vect_parity(geo_sw):
    """gate 1 -- all six outputs, FULL array including the halo rings.

    FIXTURE CLASS: no PPM limiter anywhere in ``d2a2c_vect``; the only
    data branch is the ``ut(1,j) > 0`` / ``vt(i,1) > 0`` panel-edge
    ``sin_sg`` select, whose two arms are both plain products.  So this
    gate is in the pointwise/neighbour-reading class and its bound
    should be near rounding.
    """
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    ref = _d2a2c_np(geo_sw, u, v)
    got = _d2a2c_jax(geo_sw, u, v)
    for k in _D2A2C_OUT:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(got[k], ref[k], f"d2a2c_vect.{k}", 1e-15)


def test_d2a2c_vect_duo_parity(geo_dsw):
    """gate 1 -- the DUO branch on the shared d_sw fixture.

    FIXTURE CLASS: the duo arm has NO data branch at all (interior
    formulas everywhere), so this is pure neighbour-reading arithmetic.
    """
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    ref = npduo.d2a2c_vect_duo(u, v, geo_dsw.gs_np, geo_dsw.bd,
                               geo_dsw.npx, geo_dsw.npy)
    got = duo.d2a2c_vect_duo(jnp.asarray(u), jnp.asarray(v),
                             geo_dsw.gs_j, geo_dsw.bd, geo_dsw.npx,
                             geo_dsw.npy)
    for k in _D2A2C_OUT:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(got[k], ref[k], f"d2a2c_vect_duo.{k}", 1e-15)


def test_d2a2c_vect_duo_sentinel_pattern_matches(geo_dsw):
    """The ``big_number`` fill is part of the contract, not decoration:
    the per-stage certificates compare those cells, so the two lanes'
    UNWRITTEN sets must coincide exactly."""
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    ref = npduo.d2a2c_vect_duo(u, v, geo_dsw.gs_np, geo_dsw.bd,
                               geo_dsw.npx, geo_dsw.npy)
    got = duo.d2a2c_vect_duo(jnp.asarray(u), jnp.asarray(v),
                             geo_dsw.gs_j, geo_dsw.bd, geo_dsw.npx,
                             geo_dsw.npy)
    for k in _D2A2C_OUT:
        a = np.asarray(got[k]) == duo.BIG_NUMBER
        b = np.asarray(ref[k]) == duo.BIG_NUMBER
        assert np.array_equal(a, b), (
            f"{k}: sentinel masks differ ({int(a.sum())} vs "
            f"{int(b.sum())} of {a.size})")
    # non-vacuity: at least one output really does keep sentinel cells
    assert any((np.asarray(ref[k]) == duo.BIG_NUMBER).any()
               for k in _D2A2C_OUT)


def test_d2a2c_vect_jit_equals_eager_and_no_retrace(geo_sw):
    """gate 2 -- ASSERTED, through the PRODUCTION jit factory.  NOT
    bitwise: every formula here is ``a2*(x+y) + a1*(z+w)``, a textbook
    FMA contraction site."""
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    eager = _d2a2c_jax(geo_sw, u, v)
    box, wrapped = _counted(duo.d2a2c_vect)
    fn = duo.make_d2a2c_vect_jit(wrapped)
    got = _d2a2c_jax(geo_sw, u, v, fn=fn)
    _d2a2c_jax(geo_sw, u * 1.01, v * 0.99, fn=fn)
    assert box["n"] == 1, box["n"]
    for k in _D2A2C_OUT:
        # MEASURED (job 9425294 sweep): worst vt 1.182e-15; bound = measured x 10 =
        # 1.2e-14.
        _cmp(got[k], eager[k], f"d2a2c_vect jit.{k}", 1.2e-14)


def test_d2a2c_vect_duo_jit_equals_eager(geo_dsw):
    """gate 2 for the duo arm."""
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    eager = duo.d2a2c_vect_duo(jnp.asarray(u), jnp.asarray(v),
                               geo_dsw.gs_j, geo_dsw.bd, geo_dsw.npx,
                               geo_dsw.npy)
    box, wrapped = _counted(duo.d2a2c_vect_duo)
    fn = duo.make_d2a2c_vect_duo_jit(wrapped)
    got = fn(jnp.asarray(u), jnp.asarray(v), geo_dsw.gs_j, geo_dsw.bd,
             geo_dsw.npx, geo_dsw.npy)
    fn(jnp.asarray(u * 1.01), jnp.asarray(v), geo_dsw.gs_j, geo_dsw.bd,
       geo_dsw.npx, geo_dsw.npy)
    assert box["n"] == 1, box["n"]
    for k in _D2A2C_OUT:
        # MEASURED (job 9425294 sweep): worst vt 6.287e-16; bound = measured x 10 =
        # 6.3e-15.
        _cmp(got[k], eager[k], f"d2a2c_vect_duo jit.{k}", 6.3e-15)


def test_d2a2c_vect_guards(geo_sw):
    """gate 3 -- float32, bounded_domain, and a missing metric."""
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    with pytest.raises(TypeError, match="float64"):
        _d2a2c_jax(geo_sw, np.asarray(u, np.float32), v)
    with pytest.raises(NotImplementedError, match="bounded_domain"):
        _d2a2c_jax(geo_sw, u, v, bounded_domain=True)
    thin = {k: val for k, val in geo_sw.gs_j.items() if k != "rsin2"}
    with pytest.raises(KeyError, match="rsin2"):
        duo.d2a2c_vect(jnp.asarray(u), jnp.asarray(v), thin, geo_sw.bd,
                       geo_sw.npx, geo_sw.npy)


@pytest.mark.parametrize("bad", [9, 3.0, -7])
def test_d2a2c_vect_grid_type_guard_raises(bad, geo_sw):
    """gate 3 -- dispatch-hardening on ``grid_type``."""
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    with pytest.raises(ValueError, match="not a supported grid"):
        _d2a2c_jax(geo_sw, u, v, grid_type=bad)


def test_d2a2c_vect_grid_type_guard_is_non_vacuous(geo_sw, monkeypatch):
    """gate 3 -- the guard test FAILS WITHOUT THE GUARD.

    HOW: ``_validate_grid_type`` is monkeypatched to a no-op, which is
    exactly "delete the guard".  The SAME call with ``grid_type = 3.0``
    then returns arrays -- i.e. a float silently takes the
    ``grid_type >= 3`` arm (npt = -2, every panel-edge block skipped,
    the doubly-periodic Ydir formula) on a cubed-sphere gridstruct.
    That is the silent-wrong-numerics the guard exists to stop, and it
    proves the ValueError comes from the guard rather than from an
    incidental shape error.
    """
    monkeypatch.setattr(duo, "_validate_grid_type", lambda *a, **k: None)
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    out = _d2a2c_jax(geo_sw, u, v, grid_type=3.0)
    for k in _D2A2C_OUT:
        assert np.isfinite(np.asarray(out[k])).any(), k
    # and it really is DIFFERENT numerics, not a no-op relabelling
    ref = _d2a2c_jax(geo_sw, u, v, grid_type=0)
    assert not np.allclose(np.nan_to_num(np.asarray(out["vc"])),
                           np.nan_to_num(np.asarray(ref["vc"])))


def test_d2a2c_vect_duo_is_linear_in_the_winds(geo_dsw):
    """gate 4 (structural, tolerance-independent).

    At fixed metrics the duo ``d2a2c_vect`` is a LINEAR operator in
    ``(u, v)``: every stage is a fixed linear combination.  Linearity is
    checkable without any tolerance on the derivative and it is exactly
    what the adjoint identity certifies, so the two gates below are
    reinforcing rather than redundant.
    """
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]

    def run(uu, vv):
        return duo.d2a2c_vect_duo(uu, vv, geo_dsw.gs_j, geo_dsw.bd,
                                  geo_dsw.npx, geo_dsw.npy)["uc"]

    a = np.nan_to_num(np.asarray(run(jnp.asarray(2.0 * u),
                                     jnp.asarray(2.0 * v))))
    b = 2.0 * np.nan_to_num(np.asarray(run(jnp.asarray(u),
                                           jnp.asarray(v))))
    # sentinel cells scale too (1e30 * 2), so compare on the written set
    m = np.abs(b) < 1.0e20
    assert m.any()
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    rel = np.abs(a[m] - b[m]).max() / max(np.abs(b[m]).max(), 1e-30)
    gate_scalar("d2a2c_vect_duo linearity", rel, 1e-15,
                quantity="exact-linearity residual rel")


def test_d2a2c_vect_gradients(geo_sw):
    """gate 4 -- adjoint identity is PRIMARY; order-2 is a supplement
    on a scoped, verified-smooth objective."""
    u = jnp.asarray(geo_sw.f["u"])
    v = jnp.asarray(geo_sw.f["v"])
    bd, npx, npy = geo_sw.bd, geo_sw.npx, geo_sw.npy
    ng = geo_sw.ng
    ring = slice(ng, ng + geo_sw.res)

    def f(uu, vv):
        out = duo.d2a2c_vect(uu, vv, geo_sw.gs_j, bd, npx, npy)
        # score the COMPUTE ring only -- the halo strips carry cells the
        # oracle never writes (0.0 by construction), which would make
        # the inner products insensitive to most of the operator
        return jnp.stack([o[ring, ring] for o in out])

    # MEASURED (job 9425294 sweep): adjoint residual 3.887e-16; bound = measured x 10 =
    # 3.9e-15.
    _check_adjoint("d2a2c_vect", f, (u, v), 3.9e-15)

    rng = np.random.default_rng(0)
    du = jnp.asarray(rng.standard_normal(u.shape))

    def obj(uu):
        return jnp.sum(f(uu, v) ** 2) / 1.0e6

    _assert_locally_smooth(obj, u, du, "d2a2c_vect obj")
    check_grads(obj, (u,), order=2, modes=("fwd",))


def test_d2a2c_vect_edge_select_is_metric_degenerate_on_this_grid(
        geo_sw):
    """gate 4 -- what the ``d2a2c_vect`` upwind selects ACTUALLY are on
    the admitted metrics, measured rather than assumed.

    ⛔ RETRACTED AND REPLACED (job 9404093).  This was written as a
    one-sided derivative gate on ``ut(1,j) > 0``, which selects
    ``uc(1,j) = ut*sin_sg(0,j,3)`` versus ``ut*sin_sg(1,j,1)``.  Its own
    bracket precondition refused the state -- and the precondition was
    right.  MEASURED on ``swcore_input.npz``, over EVERY index, not one
    unlucky row:

        west   max_j |sin_sg(0,j,3) - sin_sg(1,j,1)|          = 2.8e-15
        east   max_j |sin_sg(npx-1,j,3) - sin_sg(npx,j,1)|    = 4.3e-15
        south  max_i |sin_sg(i,0,4) - sin_sg(i,1,2)|          = 2.6e-15

    The two arms are the SAME geometric angle seen from the two sides of
    a panel edge, and on this gnomonic gridstruct with matched halo
    metrics they agree to rounding.  So ``d2a2c_vect`` has NO
    discriminating switching surface on the admitted input: the select
    is value-continuous to ~3e-15 and a one-sided derivative gate there
    can only certify noise.  No choice of ``j`` fixes it, which is why
    this is not a fixture-row problem.

    What this gate now asserts, all falsifiable: (a) the measured
    degeneracy itself -- if a future gridstruct separates the arms this
    goes RED and tells you a real one-sided gate has become possible;
    (b) the select is therefore value-continuous across ``s = 0``; and
    (c) the derivative is finite on both sides.  The DISCRIMINATING
    one-sided gates for this campaign live where the jump is real and
    both are in this file: ``c_sw``'s delp-transport select (arms read
    ``delp(i-1,j)`` vs ``delp(i,j)``) and ``d_sw1``'s ``crx`` select
    (arms read ``rdxa(i-1,j)`` vs ``rdxa(i,j)``).
    """
    u = jnp.asarray(geo_sw.f["u"])
    v = jnp.asarray(geo_sw.f["v"])
    bd, npx, npy = geo_sw.bd, geo_sw.npx, geo_sw.npy
    lo = bd.isd
    sg = np.asarray(geo_sw.gs_np["sin_sg"])

    # (a) the degeneracy, over EVERY index of all three edge families
    seps = {
        "west": np.abs(sg[0 - lo, :, 2] - sg[1 - lo, :, 0]).max(),
        "east": np.abs(sg[npx - 1 - lo, :, 2] - sg[npx - lo, :, 0]).max(),
        "south": np.abs(sg[:, 0 - lo, 3] - sg[:, 1 - lo, 1]).max(),
    }
    for name, s in seps.items():
        assert s < 1e-13, (
            f"{name} arms now SEPARATE by {s:.3e} -- the select has "
            f"become a real switching surface on this gridstruct, so "
            f"replace this measurement with a genuine one-sided "
            f"derivative gate against the two branch metrics")

    jrow = bd.js + 3
    col_uc, jj = 1 - lo, jrow - lo

    def uc1(s):
        out = duo.d2a2c_vect(s * u, s * v, geo_sw.gs_j, bd, npx, npy)
        return out[2][col_uc, jj]

    def ut1(s):
        out = duo.d2a2c_vect(s * u, s * v, geo_sw.gs_j, bd, npx, npy)
        return out[4][col_uc, jj]

    base_ut = float(ut1(jnp.asarray(1.0)))
    assert abs(base_ut) > 1e-8, "the select is not exercised at all"

    # (b) value continuity across s = 0, at the level the metrics allow
    eps = 1e-6
    v_r = float(uc1(jnp.asarray(+eps)))
    v_l = float(uc1(jnp.asarray(-eps)))
    m = float(sg[0 - lo, jj, 2])
    assert abs(v_r + v_l) <= 1e-9 * abs(base_ut * m), (
        f"the select is NOT value-continuous ({v_r:.6e} vs {v_l:.6e}) "
        f"even though its two metrics agree to {seps['west']:.3e}")

    # (c) both one-sided derivatives finite, and equal to the shared
    # metric because the two arms are numerically the same number
    g = jax.grad(uc1)
    right = float(g(jnp.asarray(+eps)))
    left = float(g(jnp.asarray(-eps)))
    assert np.isfinite(right) and np.isfinite(left)
    for side, val in (("right", right), ("left", left)):
        assert val == pytest.approx(base_ut * m, rel=1e-9), (
            f"{side} one-sided derivative {val:.12e} != ut*metric "
            f"{base_ut * m:.12e}")


# =====================================================================
# divergence_corner / divergence_corner_duo
# =====================================================================

def test_divergence_corner_parity(geo_sw):
    """gate 1 -- plain arm, full B-node array (NaN mask included)."""
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    ua, va, *_ = npsw.d2a2c_vect(u, v, geo_sw.gs_np, geo_sw.bd,
                                 geo_sw.npx, geo_sw.npy)
    ref = npsw.divergence_corner(u, v, ua, va, geo_sw.gs_np, geo_sw.bd,
                                 geo_sw.npx, geo_sw.npy)
    got = duo.divergence_corner(jnp.asarray(u), jnp.asarray(v),
                                jnp.asarray(ua), jnp.asarray(va),
                                geo_sw.gs_j, geo_sw.bd, geo_sw.npx,
                                geo_sw.npy)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, ref, "divergence_corner", 1e-15)


def test_divergence_corner_duo_parity(geo_dsw):
    """gate 1 -- duo arm, on the real duo ua/va."""
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    d2a = npduo.d2a2c_vect_duo(u, v, geo_dsw.gs_np, geo_dsw.bd,
                               geo_dsw.npx, geo_dsw.npy)
    ref = npduo.divergence_corner_duo(u, v, d2a["ua"], d2a["va"],
                                      geo_dsw.gs_np, geo_dsw.bd,
                                      geo_dsw.npx, geo_dsw.npy)
    got = duo.divergence_corner_duo(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(d2a["ua"]),
        jnp.asarray(d2a["va"]), geo_dsw.gs_j, geo_dsw.bd, geo_dsw.npx,
        geo_dsw.npy)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, ref, "divergence_corner_duo", 1e-15)


def test_divergence_corner_duo_seam_treatment_is_real(geo_dsw):
    """The duo zeroing/quartering must actually fire -- otherwise the
    parity test above would pass on a plain-formula implementation.

    Instrument: the seam B-nodes must be EXACTLY zero and the
    next-to-seam ones must be exactly a quarter of the unquartered
    value.  Both are structural, so no tolerance is involved.
    """
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    d2a = npduo.d2a2c_vect_duo(u, v, geo_dsw.gs_np, geo_dsw.bd,
                               geo_dsw.npx, geo_dsw.npy)
    got = np.asarray(duo.divergence_corner_duo(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(d2a["ua"]),
        jnp.asarray(d2a["va"]), geo_dsw.gs_j, geo_dsw.bd, geo_dsw.npx,
        geo_dsw.npy))
    bd, npx, npy = geo_dsw.bd, geo_dsw.npx, geo_dsw.npy
    lo = bd.isd
    inner_j = slice(bd.jsd + 1 - lo, bd.jed - lo + 1)
    for i in (bd.is_, bd.ie + 1):
        assert (got[i - lo, inner_j] == 0.0).all(), f"seam i={i}"
    inner_i = slice(bd.isd + 1 - lo, bd.ied - lo + 1)
    for j in (1, bd.je + 1):
        assert (got[inner_i, j - lo] == 0.0).all(), f"seam j={j}"
    # non-vacuity: the interior is NOT zero
    assert np.abs(got[bd.is_ + 3 - lo, bd.js + 3 - lo]) > 0.0
    assert npx == npy   # the verbatim npx-vs-npy quirk is inert here


def test_divergence_corner_jit_equals_eager(geo_sw):
    """gate 2 -- ASSERTED (sums everywhere, so not bitwise)."""
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    ua, va, *_ = npsw.d2a2c_vect(u, v, geo_sw.gs_np, geo_sw.bd,
                                 geo_sw.npx, geo_sw.npy)
    args = (jnp.asarray(u), jnp.asarray(v), jnp.asarray(ua),
            jnp.asarray(va), geo_sw.gs_j, geo_sw.bd, geo_sw.npx,
            geo_sw.npy)
    eager = duo.divergence_corner(*args)
    box, wrapped = _counted(duo.divergence_corner)
    fn = duo.make_divergence_corner_jit(wrapped)
    got = fn(*args)
    fn(args[0] * 1.01, *args[1:])
    assert box["n"] == 1, box["n"]
    # MEASURED (job 9425294 sweep): 6.985e-13 (divg_d cascade, FMA class); bound = measured x 10 =
    # 7.0e-12.
    _cmp(got, eager, "divergence_corner jit", 7.0e-12)


def test_divergence_corner_duo_jit_equals_eager(geo_dsw):
    """gate 2 for the duo arm."""
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    ua, va = geo_dsw.f["ua"], geo_dsw.f["va"]
    args = (jnp.asarray(u), jnp.asarray(v), jnp.asarray(ua),
            jnp.asarray(va), geo_dsw.gs_j, geo_dsw.bd, geo_dsw.npx,
            geo_dsw.npy)
    eager = duo.divergence_corner_duo(*args)
    box, wrapped = _counted(duo.divergence_corner_duo)
    fn = duo.make_divergence_corner_duo_jit(wrapped)
    got = fn(*args)
    fn(args[0] * 1.01, *args[1:])
    assert box["n"] == 1, box["n"]
    # MEASURED (job 9425294 sweep): 9.767e-13 (FMA class); bound = measured x 10 =
    # 9.8e-12.
    _cmp(got, eager, "divergence_corner_duo jit", 9.8e-12)


@pytest.mark.parametrize("fn_name", ["divergence_corner",
                                     "divergence_corner_duo"])
def test_divergence_corner_guards(fn_name, geo_dsw):
    """gate 3 -- float32 and an unsupported grid_type."""
    fn = getattr(duo, fn_name)
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    ua, va = geo_dsw.f["ua"], geo_dsw.f["va"]
    base = (jnp.asarray(v), jnp.asarray(ua), jnp.asarray(va),
            geo_dsw.gs_j, geo_dsw.bd, geo_dsw.npx, geo_dsw.npy)
    with pytest.raises(TypeError, match="float64"):
        fn(jnp.asarray(u, jnp.float32), *base)
    with pytest.raises(ValueError, match="not a supported grid"):
        fn(jnp.asarray(u), *base, grid_type=9)
    with pytest.raises(NotImplementedError, match="grid_type"):
        fn(jnp.asarray(u), *base, grid_type=4)


def test_divergence_corner_grid_type_guard_is_non_vacuous(
        geo_dsw, monkeypatch):
    """gate 3 -- WITHOUT the guard, ``grid_type = 3.0`` silently runs the
    cubed-sphere arm (``3.0 > 3`` is False, so the NotImplementedError
    never fires) and returns a plausible field."""
    monkeypatch.setattr(duo, "_validate_grid_type", lambda *a, **k: None)
    u, v = geo_dsw.f["u"], geo_dsw.f["v"]
    out = duo.divergence_corner_duo(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(geo_dsw.f["ua"]),
        jnp.asarray(geo_dsw.f["va"]), geo_dsw.gs_j, geo_dsw.bd,
        geo_dsw.npx, geo_dsw.npy, grid_type=3.0)
    assert np.isfinite(np.asarray(out)).any()


def test_divergence_corner_gradients(geo_dsw):
    """gate 4 -- adjoint identity on the duo arm (the production one).

    ``divergence_corner_duo`` is linear in ``(u, v, ua, va)`` at fixed
    metrics apart from the seam zero/quarter masks, which are STATIC, so
    the identity should hold at roundoff.
    """
    bd = geo_dsw.bd
    lo = bd.isd
    ring = slice(bd.is_ - lo, bd.ie + 2 - lo)

    def f(u, v, ua, va):
        out = duo.divergence_corner_duo(u, v, ua, va, geo_dsw.gs_j, bd,
                                        geo_dsw.npx, geo_dsw.npy)
        return out[ring, ring]

    primals = tuple(jnp.asarray(geo_dsw.f[k])
                    for k in ("u", "v", "ua", "va"))
    # MEASURED (job 9425294 sweep): adjoint residual 3.617e-16; bound = measured x 10 =
    # 3.7e-15.
    _check_adjoint("divergence_corner_duo", f, primals, 3.7e-15)
    check_grads(lambda x: jnp.sum(f(x, *primals[1:]) ** 2) / 1e6,
                (primals[0],), order=2, modes=("fwd",))


# =====================================================================
# del6_vt_flux
# =====================================================================

def _del6_np(geo, nord, damp, q, damp_km=None, duogrid=False,
             fx2_seed=None, fy2_seed=None):
    """Run the NumPy twin, which writes into CALLER work arrays."""
    bd = geo.bd
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1
    qf = fort(np.array(q, dtype=np.float64, copy=True), isd, jsd)
    d2 = fort(np.full((nid, njd), np.nan), isd, jsd)
    fx2 = fort(np.full((nid + 1, njd), np.nan)
               if fx2_seed is None
               else np.array(fx2_seed, dtype=np.float64, copy=True),
               isd, jsd)
    fy2 = fort(np.full((nid, njd + 1), np.nan)
               if fy2_seed is None
               else np.array(fy2_seed, dtype=np.float64, copy=True),
               isd, jsd)
    gsf = {
        "del6_v": fort(geo.gs_np["del6_v"], isd, jsd),
        "del6_u": fort(geo.gs_np["del6_u"], isd, jsd),
        "rarea": fort(geo.gs_np["rarea"], isd, jsd),
        "bounded_domain": geo.gs_np["bounded_domain"],
        "sw_corner": geo.gs_np["sw_corner"],
        "se_corner": geo.gs_np["se_corner"],
        "nw_corner": geo.gs_np["nw_corner"],
        "ne_corner": geo.gs_np["ne_corner"],
    }
    dk = None if damp_km is None else fort(
        np.array(damp_km, dtype=np.float64), isd, jsd)
    npd.del6_vt_flux(nord, geo.npx, geo.npy, damp, qf, d2, fx2, fy2,
                     gsf, bd, damp_km=dk, duogrid=duogrid)
    return fx2.a, fy2.a


def _del6_jax(geo, nord, damp, q, damp_km=None, duogrid=False, fn=None,
              fx2_seed=None, fy2_seed=None):
    f = fn or duo.del6_vt_flux
    fl = geo.flags
    return f(nord, geo.npx, geo.npy, damp, jnp.asarray(q), geo.bd,
             geo.gs_j["del6_u"], geo.gs_j["del6_v"], geo.gs_j["rarea"],
             fl.bounded_domain, fl.sw_corner, fl.se_corner,
             fl.nw_corner, fl.ne_corner, duogrid,
             None if damp_km is None else jnp.asarray(damp_km),
             None if fx2_seed is None else jnp.asarray(fx2_seed),
             None if fy2_seed is None else jnp.asarray(fy2_seed))


def _del6_q(geo, seed=7):
    rng = np.random.default_rng(seed)
    n = geo.m_a
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (1.0e-4 * np.cos(0.4 * ii) * np.sin(0.3 * jj)
            + 1.0e-5 * rng.standard_normal((n, n)))


@pytest.mark.parametrize("nord", [0, 1, 2])
@pytest.mark.parametrize("duogrid", [False, True])
def test_del6_vt_flux_parity(nord, duogrid, geo_dsw):
    """gate 1 -- every ``nord`` the operator is defined for, on BOTH the
    plain (``copy_corners`` active) and duo (``copy_corners`` an
    early-return) lanes.

    FIXTURE CLASS: repeated stencil passes over ``nord``, so the error
    is locally ACCUMULATED and window-sensitive; the bound belongs to
    the accumulating class, not the pointwise one.
    """
    q = _del6_q(geo_dsw)
    damp = 3.5e7
    ref_x, ref_y = _del6_np(geo_dsw, nord, damp, q, duogrid=duogrid)
    got_x, got_y = _del6_jax(geo_dsw, nord, damp, q, duogrid=duogrid)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), fx2 and fy2, every nord/duo; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got_x, ref_x, f"del6 fx2 nord={nord} duo={duogrid}", 1e-15)
    _cmp(got_y, ref_y, f"del6 fy2 nord={nord} duo={duogrid}", 1e-15)


def test_del6_vt_flux_damp_km_parity(geo_dsw):
    """gate 1 -- the ``damp_km`` rescale.  NOT in the pinned oracle
    (``sw_core.F90:2008`` has no such argument); it comes from the
    extraction the NumPy lane was transcribed from, and the NumPy lane
    is the authority for this hop, so it is mirrored and tested."""
    q = _del6_q(geo_dsw)
    dk = 0.05 + 0.01 * np.abs(np.sin(np.arange(geo_dsw.m_a))[:, None]
                              + np.arange(geo_dsw.m_a)[None, :])
    ref_x, ref_y = _del6_np(geo_dsw, 1, 3.5e7, q, damp_km=dk)
    got_x, got_y = _del6_jax(geo_dsw, 1, 3.5e7, q, damp_km=dk)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), both components; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got_x, ref_x, "del6 fx2 damp_km", 1e-15)
    _cmp(got_y, ref_y, "del6 fy2 damp_km", 1e-15)


def test_del6_vt_flux_seeded_work_arrays_carry_forward(geo_dsw):
    """The ``fx2``/``fy2`` seeds are what make ``d_sw6``'s returned
    ut/vt match the twin: the NumPy lane writes INTO caller work arrays,
    so cells OUTSIDE the oracle's windows keep their incoming values.
    This test pins that behaviour directly."""
    q = _del6_q(geo_dsw)
    bd = geo_dsw.bd
    nid = bd.ied - bd.isd + 1
    njd = bd.jed - bd.jsd + 1
    rng = np.random.default_rng(2)
    sx = rng.standard_normal((nid + 1, njd))
    sy = rng.standard_normal((nid, njd + 1))
    ref_x, ref_y = _del6_np(geo_dsw, 1, 3.5e7, q, fx2_seed=sx,
                            fy2_seed=sy)
    got_x, got_y = _del6_jax(geo_dsw, 1, 3.5e7, q, fx2_seed=sx,
                             fy2_seed=sy)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), both components; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got_x, ref_x, "del6 fx2 seeded", 1e-15)
    _cmp(got_y, ref_y, "del6 fy2 seeded", 1e-15)
    # non-vacuity: some seeded cells really do survive untouched
    keep = np.asarray(got_x) == sx
    assert keep.any(), "no seeded cell survived -- the test proves nothing"
    assert not keep.all(), "nothing was written -- the test proves nothing"


def test_del6_vt_flux_jit_equals_eager(geo_dsw):
    """gate 2 -- ASSERTED (flux differences everywhere: not bitwise)."""
    q = _del6_q(geo_dsw)
    eager = _del6_jax(geo_dsw, 1, 3.5e7, q)
    box, wrapped = _counted(duo.del6_vt_flux)
    fn = duo.make_del6_vt_flux_jit(wrapped)
    got = _del6_jax(geo_dsw, 1, 3.5e7, q, fn=fn)
    _del6_jax(geo_dsw, 1, 4.0e7, q * 1.01, fn=fn)   # new VALUES only
    assert box["n"] == 1, box["n"]
    # MEASURED (job 9425294 sweep): worst fy2 2.104e-16; bound = measured x 10 =
    # 2.2e-15.
    _cmp(got[0], eager[0], "del6 fx2 jit", 2.2e-15)
    _cmp(got[1], eager[1], "del6 fy2 jit", 2.2e-15)


def test_del6_vt_flux_guards(geo_dsw):
    """gate 3 -- float32, a negative nord, and a non-bool flag."""
    q = _del6_q(geo_dsw)
    with pytest.raises(TypeError, match="float64"):
        _del6_jax(geo_dsw, 1, 3.5e7, np.asarray(q, np.float32))
    with pytest.raises(ValueError, match="non-negative python int"):
        _del6_jax(geo_dsw, -1, 3.5e7, q)
    with pytest.raises(ValueError, match="not a python bool"):
        duo.del6_vt_flux(1, geo_dsw.npx, geo_dsw.npy, 1.0,
                         jnp.asarray(q), geo_dsw.bd,
                         geo_dsw.gs_j["del6_u"], geo_dsw.gs_j["del6_v"],
                         geo_dsw.gs_j["rarea"], 0, True, True, True,
                         True, False)


def test_del6_vt_flux_nord_guard_is_non_vacuous(geo_dsw, monkeypatch):
    """gate 3 -- WITHOUT the guard, ``nord = -1`` returns a plausible
    field computed on SHIFTED windows (``i1 = is-1-nord`` becomes
    ``is``, and the ``nord > 0`` pass loop never runs), i.e. a silently
    different operator rather than an error."""
    monkeypatch.setattr(duo, "_validate_nord", lambda *a, **k: None)
    q = _del6_q(geo_dsw)
    got_x, got_y = _del6_jax(geo_dsw, -1, 3.5e7, q)
    assert np.isfinite(np.asarray(got_x)).any()
    assert np.isfinite(np.asarray(got_y)).any()
    ref_x, _ = _del6_jax(geo_dsw, 1, 3.5e7, q)
    assert not np.array_equal(np.nan_to_num(np.asarray(got_x)),
                              np.nan_to_num(np.asarray(ref_x)))


def test_del6_vt_flux_gradients(geo_dsw):
    """gate 4 -- ``del6_vt_flux`` is LINEAR in ``q`` at fixed ``damp``
    (every stage is a fixed stencil), so the adjoint identity must hold
    at roundoff and order-2 finite differencing is exact."""
    bd = geo_dsw.bd
    lo = bd.isd
    ring = slice(bd.is_ - lo, bd.ie + 1 - lo)

    def f(q):
        fx2, fy2 = duo.del6_vt_flux(
            1, geo_dsw.npx, geo_dsw.npy, 3.5e7, q, bd,
            geo_dsw.gs_j["del6_u"], geo_dsw.gs_j["del6_v"],
            geo_dsw.gs_j["rarea"], False, True, True, True, True, True)
        return jnp.stack([fx2[ring, ring], fy2[ring, ring]])

    q = jnp.asarray(_del6_q(geo_dsw))
    # MEASURED (job 9425294 sweep): adjoint residual 2.719e-16; bound = measured x 10 =
    # 2.8e-15.
    _check_adjoint("del6_vt_flux", f, (q,), 2.8e-15)
    check_grads(lambda x: jnp.sum(f(x) ** 2), (q,), order=2,
                modes=("fwd", "rev"))


# =====================================================================
# c_sw
# =====================================================================

_CSW_OUT = ("delpc", "ptc", "wc", "uc", "vc", "ua", "va", "ut", "vt",
            "divg_d")


@pytest.mark.parametrize("hydrostatic", [True, False])
def test_c_sw_plain_parity(hydrostatic, geo_sw):
    """gate 1 -- the PLAIN arm (duogrid=False, bounded=False) on the
    committed one-step c_sw fixture, all ten outputs, full arrays.

    FIXTURE CLASS: no PPM limiter in ``c_sw`` at all; the data branches
    are upwind SELECTS whose arms are gathers, so this sits in the
    accumulating (flux-divergence sum) class.  Both the hydrostatic and
    the non-hydrostatic arms run, the latter exercising the two
    ``fill_4corners(W)`` calls and the ``wc`` update.
    """
    delp, pt = geo_sw.f["delp"], geo_sw.f["pt"]
    u, v = geo_sw.f["u"], geo_sw.f["v"]
    w = (np.zeros_like(delp) if hydrostatic
         else 0.1 * np.asarray(pt, np.float64))
    kw = dict(nord=geo_sw.nord, hydrostatic=hydrostatic, dord4=True,
              grid_type=0)
    ref = npsw.c_sw(delp=delp, pt=pt, w=w, u=u, v=v, gs=geo_sw.gs_np,
                    bd=geo_sw.bd, npx=geo_sw.npx, npy=geo_sw.npy,
                    dt2=geo_sw.dt2, duogrid=False, **kw)
    got = duo.c_sw(jnp.asarray(delp), jnp.asarray(pt), jnp.asarray(w),
                   jnp.asarray(u), jnp.asarray(v), geo_sw.gs_j,
                   geo_sw.bd, geo_sw.npx, geo_sw.npy, geo_sw.dt2,
                   duogrid=False, bounded_domain=False, **kw)
    keys = _CSW_OUT if not hydrostatic else \
        tuple(k for k in _CSW_OUT if k != "wc")
    for k in keys:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key, both arms; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(got[k], ref[k], f"c_sw plain hydro={hydrostatic}.{k}",
             1e-15)


@pytest.mark.parametrize("hydrostatic", [True, False])
def test_c_sw_duo_parity(hydrostatic, geo_bounded):
    """gate 1 -- the DUO arm on the BOUNDED gridstruct, which is the
    only configuration upstream can reach (fv_arrays.F90:1512)."""
    gs, st, bd, npx, npy = geo_bounded
    gs_j = {k: jnp.asarray(v) for k, v in gs.items()
            if isinstance(v, np.ndarray)}
    delp, pt = st["delp"], st["pt"]
    w = (np.zeros_like(np.asarray(delp))
         if hydrostatic else 0.1 * np.asarray(pt, np.float64))
    kw = dict(nord=1, hydrostatic=hydrostatic, dord4=True, grid_type=0)
    ref = npsw.c_sw(delp=delp, pt=pt, w=w, u=st["u"], v=st["v"], gs=gs,
                    bd=bd, npx=npx, npy=npy, dt2=112.5, duogrid=True,
                    **kw)
    got = duo.c_sw(jnp.asarray(delp), jnp.asarray(pt), jnp.asarray(w),
                   jnp.asarray(st["u"]), jnp.asarray(st["v"]), gs_j, bd,
                   npx, npy, 112.5, duogrid=True, bounded_domain=True,
                   **kw)
    keys = _CSW_OUT if not hydrostatic else \
        tuple(k for k in _CSW_OUT if k != "wc")
    for k in keys:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key, both arms; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(got[k], ref[k], f"c_sw duo hydro={hydrostatic}.{k}", 1e-15)


def test_c_sw_duo_arm_is_the_only_one_that_runs_bounded(geo_bounded):
    """The duo/plain split is a CLAIM -- measured here.

    Both arms are asked for on the SAME bounded gridstruct and the SAME
    state, so the only variable is the ``duogrid`` flag.  The duo arm
    produces a finite compute domain; the plain arm is REFUSED, and the
    refusal comes from ``d2a2c_vect`` (reached first), exactly as in the
    NumPy authority -- stating that precisely matters, because the
    obvious guess is the KE guard, which the call never reaches.
    """
    gs, st, bd, npx, npy = geo_bounded
    gs_j = {k: jnp.asarray(v) for k, v in gs.items()
            if isinstance(v, np.ndarray)}
    w = jnp.zeros_like(jnp.asarray(st["delp"]))
    common = (jnp.asarray(st["delp"]), jnp.asarray(st["pt"]), w,
              jnp.asarray(st["u"]), jnp.asarray(st["v"]), gs_j, bd,
              npx, npy, 112.5)
    out = duo.c_sw(*common, duogrid=True, bounded_domain=True)
    lo = bd.isd
    ring = slice(bd.is_ - lo, bd.ie + 1 - lo)
    for k in ("delpc", "ptc", "uc", "vc", "divg_d"):
        assert np.isfinite(np.asarray(out[k])[ring, ring]).all(), k
    with pytest.raises(NotImplementedError, match="bounded_domain"):
        duo.c_sw(*common, duogrid=False, bounded_domain=True)
    # and the NumPy authority refuses the SAME combination the SAME way
    with pytest.raises(NotImplementedError):
        npsw.c_sw(delp=st["delp"], pt=st["pt"], w=np.zeros_like(
            np.asarray(st["delp"])), u=st["u"], v=st["v"], gs=gs, bd=bd,
            npx=npx, npy=npy, dt2=112.5, duogrid=False)


def test_c_sw_ke_branch_guard(geo_sw):
    """gate 3 -- the KE guard specifically, reached on the ONE path that
    gets there: ``grid_type = 3`` with ``duogrid = False`` and unbounded
    metrics.  ``d2a2c_vect`` has both grid_type arms and
    ``divergence_corner`` only refuses ``> 3``, so the call runs all the
    way to ``(bounded .or. grid_type>=3) .and. .not.duogrid``."""
    delp, pt = jnp.asarray(geo_sw.f["delp"]), jnp.asarray(geo_sw.f["pt"])
    u, v = jnp.asarray(geo_sw.f["u"]), jnp.asarray(geo_sw.f["v"])
    w = jnp.zeros_like(delp)
    with pytest.raises(NotImplementedError, match="KE branch"):
        duo.c_sw(delp, pt, w, u, v, geo_sw.gs_j, geo_sw.bd, geo_sw.npx,
                 geo_sw.npy, geo_sw.dt2, grid_type=3, duogrid=False)


def test_c_sw_jit_equals_eager_and_no_retrace_on_dt(geo_sw):
    """gate 2 -- ASSERTED through the production factory, and ``dt2`` is
    proved DYNAMIC: a second call at a different time step must NOT
    retrace (strategy section 7 mechanics)."""
    delp, pt = jnp.asarray(geo_sw.f["delp"]), jnp.asarray(geo_sw.f["pt"])
    u, v = jnp.asarray(geo_sw.f["u"]), jnp.asarray(geo_sw.f["v"])
    w = jnp.zeros_like(delp)
    args = (delp, pt, w, u, v, geo_sw.gs_j, geo_sw.bd, geo_sw.npx,
            geo_sw.npy)
    eager = duo.c_sw(*args, geo_sw.dt2)
    box, wrapped = _counted(duo.c_sw)
    fn = duo.make_c_sw_jit(wrapped)
    got = fn(*args, geo_sw.dt2)
    fn(*args, 2.0 * geo_sw.dt2)
    assert box["n"] == 1, f"retraced on a new dt2: {box['n']}"
    for k in _CSW_OUT:
        # MEASURED (job 9425294 sweep): worst divg_d 6.985e-13 (FMA class); bound = measured x 10 =
        # 7.0e-12.
        _cmp(got[k], eager[k], f"c_sw jit.{k}", 7.0e-12)


def test_c_sw_guards(geo_sw, geo_bounded):
    """gate 3 -- every entry guard, including the one-way duo/bounded
    implication of fv_arrays.F90:1512."""
    delp, pt = jnp.asarray(geo_sw.f["delp"]), jnp.asarray(geo_sw.f["pt"])
    u, v = jnp.asarray(geo_sw.f["u"]), jnp.asarray(geo_sw.f["v"])
    w = jnp.zeros_like(delp)
    args = (delp, pt, w, u, v, geo_sw.gs_j, geo_sw.bd, geo_sw.npx,
            geo_sw.npy, geo_sw.dt2)
    with pytest.raises(ValueError, match=r"fv_arrays\.F90:1512"):
        duo.c_sw(*args, duogrid=True, bounded_domain=False)
    with pytest.raises(TypeError, match="float64"):
        duo.c_sw(jnp.asarray(delp, jnp.float32), *args[1:])
    with pytest.raises(ValueError, match="not a supported grid"):
        duo.c_sw(*args, grid_type=9)
    with pytest.raises(ValueError, match="non-negative python int"):
        duo.c_sw(*args, nord=-1)
    with pytest.raises(ValueError, match="not a python bool"):
        duo.c_sw(*args, hydrostatic=1)
    # the plain arm still runs -- so the raises above are the guards
    ok = duo.c_sw(*args)
    assert np.isfinite(np.asarray(ok["delpc"])).any()


def test_c_sw_duo_bounded_guard_is_non_vacuous(geo_sw):
    """gate 3 -- WITHOUT the guard, ``duogrid=True`` on unbounded
    metrics would RUN (the duo leaves do not test ``bounded_domain``),
    silently producing a field for a combination no upstream
    configuration can reach.  Shown by monkeypatching the whole
    ``c_sw`` guard away via ``_require_bool`` is not possible -- the
    check is an explicit ``if`` -- so it is demonstrated instead on the
    LEAVES the guard protects, which do run unbounded."""
    u, v = jnp.asarray(geo_sw.f["u"]), jnp.asarray(geo_sw.f["v"])
    out = duo.d2a2c_vect_duo(u, v, geo_sw.gs_j, geo_sw.bd, geo_sw.npx,
                             geo_sw.npy)
    assert np.isfinite(np.asarray(out["uc"])).any(), (
        "the duo leaf refuses unbounded metrics, so c_sw's guard would "
        "be redundant -- re-derive this test")


def test_c_sw_gradients(geo_sw):
    """gate 4 -- adjoint identity on the compute ring of every output.

    The objective is restricted to the compute ring because the halo
    strips carry cells ``c_sw`` never writes (0.0 or NaN by
    construction), and including them would either poison the inner
    products with NaN or make them insensitive to most of the operator.
    """
    delp = jnp.asarray(geo_sw.f["delp"])
    pt = jnp.asarray(geo_sw.f["pt"])
    u, v = jnp.asarray(geo_sw.f["u"]), jnp.asarray(geo_sw.f["v"])
    w = jnp.zeros_like(delp)
    bd = geo_sw.bd
    lo = bd.isd
    ring = slice(bd.is_ - lo, bd.ie + 1 - lo)

    def f(dp, ptx, uu, vv):
        out = duo.c_sw(dp, ptx, w, uu, vv, geo_sw.gs_j, bd, geo_sw.npx,
                       geo_sw.npy, geo_sw.dt2)
        return jnp.stack([out[k][ring, ring]
                          for k in ("delpc", "ptc", "uc", "vc")])

    # MEASURED (job 9425294 sweep): adjoint residual 4.736e-16; bound = measured x 10 =
    # 4.8e-15.
    _check_adjoint("c_sw", f, (delp, pt, u, v), 4.8e-15)


def test_c_sw_one_sided_across_the_transport_upwind_surface(geo_sw):
    """gate 4 (strategy correction 4) -- the SWITCHING-SURFACE test for
    the ``c_sw`` transport.

    Surface: ``ut(i,j) > 0`` in the delp flux
    (``fx1 = delp(i-1,j)`` versus ``delp(i,j)``).  The two arms read
    DIFFERENT cells, so the flux -- and hence ``delpc`` -- is
    DISCONTINUOUS across the surface, not merely kinked; the only
    correct statement is the pair of one-sided derivatives.

    Construction: scaling ``(u, v)`` by ``s`` scales ``ut`` by ``s``
    (the whole d2a2c -> transport-wind chain is linear in the winds
    before the ``sin_sg`` select, and the select's two arms are both
    ``dt2 * ut * dy * metric``).  So ``d(delpc)/ds`` from ``s > 0`` and
    from ``s < 0`` pick DIFFERENT upwind cells, and the two one-sided
    derivatives must differ by a term proportional to the local ``delp``
    jump.
    """
    delp = jnp.asarray(geo_sw.f["delp"])
    pt = jnp.asarray(geo_sw.f["pt"])
    u, v = jnp.asarray(geo_sw.f["u"]), jnp.asarray(geo_sw.f["v"])
    w = jnp.zeros_like(delp)
    bd = geo_sw.bd
    lo = bd.isd
    icell, jcell = bd.is_ + 4, bd.js + 4

    def dpc(s):
        out = duo.c_sw(delp, pt, w, s * u, s * v, geo_sw.gs_j, bd,
                       geo_sw.npx, geo_sw.npy, geo_sw.dt2)
        return out["delpc"][icell - lo, jcell - lo]

    g = jax.grad(dpc)
    eps = 1e-7
    right = float(g(jnp.asarray(+eps)))
    left = float(g(jnp.asarray(-eps)))
    assert np.isfinite(right) and np.isfinite(left)
    # The surface is REAL: the two sides use different upwind cells, so
    # the one-sided derivatives differ unless delp happens to be locally
    # constant -- assert that the fixture is not in that degenerate case.
    d = np.asarray(geo_sw.f["delp"])
    local = d[icell - 1 - lo:icell + 2 - lo, jcell - lo]
    # np.ptp(), not ndarray.ptp() -- the method was REMOVED in numpy 2.0
    # and this line was the whole failure (job 9404093); the gradient
    # itself never ran.
    assert np.ptp(local) > 0.0, "delp is locally constant -- no surface"
    assert abs(right - left) > 0.0, (
        f"one-sided derivatives coincide ({right:.6e}) -- the upwind "
        f"surface was not crossed")
    # VALUE continuity is NOT claimed here (the arms read different
    # cells), which is exactly why a two-sided FD across s=0 would be
    # meaningless and check_grads is not used at this state.


# =====================================================================
# the d_sw chain -- d_sw1 ... d_sw6
# =====================================================================

_DSW_KW = dict(hord_tr=8, hord_vt=6, hord_tm=6, hord_dp=6, nord_v=1,
               nord_t=0, damp_v=0.2, damp_t=0.0)


def _zero_caps(geo):
    """The four capacitors at their Fortran declared bounds."""
    res, m_a = geo.res, geo.m_a
    return (np.zeros((res + 1, res)), np.zeros((res, res + 1)),
            np.zeros((res + 1, m_a)), np.zeros((m_a, res + 1)))


def _d_sw1_np(geo, caps=None, w_override=None, **kw):
    """``w_override`` exists because the committed fixture's ``w`` is
    identically zero, which makes every NH gate vacuous; BOTH lanes take
    the same array so the comparison stays one-variable."""
    xf, yf, cx, cy = caps if caps is not None else _zero_caps(geo)
    f = geo.f
    w = f["w"] if w_override is None else np.asarray(w_override,
                                                     np.float64)
    return npduo.d_sw1_duo(f["delp"], f["pt"], w, f["uc"], f["vc"],
                           xf, yf, cx, cy, geo.gs_np, geo.bd, geo.npx,
                           geo.npy, dt=geo.dt, **{**_DSW_KW, **kw})


def _d_sw1_jax(geo, caps=None, fn=None, w_override=None, **kw):
    xf, yf, cx, cy = caps if caps is not None else _zero_caps(geo)
    f = geo.f
    w = f["w"] if w_override is None else np.asarray(w_override,
                                                     np.float64)
    g = fn or duo.d_sw1_duo
    return g(jnp.asarray(f["delp"]), jnp.asarray(f["pt"]),
             jnp.asarray(w), jnp.asarray(f["uc"]),
             jnp.asarray(f["vc"]), jnp.asarray(xf), jnp.asarray(yf),
             jnp.asarray(cx), jnp.asarray(cy), geo.gs_j, geo.flags,
             geo.bd, geo.npx, geo.npy, dt=geo.dt, **{**_DSW_KW, **kw})


@pytest.fixture(scope="module")
def chain(geo_dsw):
    """Both lanes' full d_sw1 -> d_sw6 chain on the committed inputs.

    The stage wiring (which output feeds which input, and the raw
    ``kee = 0.5*(ubbtemp*vbbtemp + ubb*vbb)`` assembly dyn_core does
    inline between d_sw3 and d_sw4) is copied from the certified NumPy
    per-stage harnesses so BOTH lanes see byte-identical inputs at every
    stage -- the controlled-comparison requirement.
    """
    geo = geo_dsw
    f, bd, npx, npy, dt = geo.f, geo.bd, geo.npx, geo.npy, geo.dt
    ng, res, m_a = geo.ng, geo.res, geo.m_a
    ring = slice(ng, ng + res + 1)

    n1 = _d_sw1_np(geo)
    j1 = _d_sw1_jax(geo)

    n2 = npduo.d_sw2_duo(n1["delp"], n1["pt"], n1["allflux_x"],
                         n1["allflux_y"], geo.gs_np, bd)
    j2 = duo.d_sw2_duo(j1["delp"], j1["pt"], j1["allflux_x"],
                       j1["allflux_y"], geo.gs_j, geo.flags, bd)

    n3 = npduo.d_sw3_duo(f["u"], f["v"], f["uc"], f["vc"], geo.gs_np,
                         bd, npx, npy, dt=dt, hord_mt=6)
    j3 = duo.d_sw3_duo(jnp.asarray(f["u"]), jnp.asarray(f["v"]),
                       jnp.asarray(f["uc"]), jnp.asarray(f["vc"]),
                       geo.gs_j, geo.flags, bd, npx, npy, dt=dt,
                       hord_mt=6)

    def _kee(s3, xp):
        ke = xp.full((m_a + 1, m_a + 1), 1.0e30)
        blk = 0.5 * (s3["ubbtemp"] * s3["vbbtemp"]
                     + s3["ubb"] * s3["vbb"])
        if xp is np:
            ke[ring, ring] = blk
            return ke
        return jnp.asarray(ke).at[ring, ring].set(blk)

    n4 = npduo.d_sw4_duo(f["u"], f["v"], n1["ut"], n1["vt"],
                         _kee(n3, np), geo.gs_np, bd, npx, npy, dt=dt)
    j4 = duo.d_sw4_duo(jnp.asarray(f["u"]), jnp.asarray(f["v"]),
                       j1["ut"], j1["vt"], _kee(j3, jnp), geo.flags, bd,
                       npx, npy, dt=dt)

    d5kw = dict(hord_vt=6, nord=1, dddmp=0.2, d2_bg=0.0, d4_bg=0.12,
                d_con=0.0)
    n5 = npduo.d_sw5_duo(f["delp"], f["u"], f["v"], f["uc"], f["vc"],
                         f["ua"], f["va"], f["divg_d_in"],
                         n1["crx_adv"], n1["cry_adv"], n1["xfx_adv"],
                         n1["yfx_adv"], n1["ra_x"], n1["ra_y"],
                         n4["ke"], geo.gs_np, bd, npx, npy, dt=dt,
                         **d5kw)
    j5 = duo.d_sw5_duo(jnp.asarray(f["delp"]), jnp.asarray(f["u"]),
                       jnp.asarray(f["v"]), jnp.asarray(f["uc"]),
                       jnp.asarray(f["vc"]), jnp.asarray(f["ua"]),
                       jnp.asarray(f["va"]),
                       jnp.asarray(f["divg_d_in"]), j1["crx_adv"],
                       j1["cry_adv"], j1["xfx_adv"], j1["yfx_adv"],
                       j1["ra_x"], j1["ra_y"], j4["ke"], geo.gs_j,
                       geo.flags, bd, npx, npy, dt=dt, **d5kw)

    d6kw = dict(nord_v=1, damp_v=0.2, d_con=0.0)
    n6 = npduo.d_sw6_duo(f["u"], f["v"], n5["ut"], n5["vt"], n5["ke"],
                         n5["wk"], n5["vortfluxx"], n5["vortfluxy"],
                         geo.gs_np, bd, npx, npy, **d6kw)
    j6 = duo.d_sw6_duo(jnp.asarray(f["u"]), jnp.asarray(f["v"]),
                       j5["ut"], j5["vt"], j5["ke"], j5["wk"],
                       j5["vortfluxx"], j5["vortfluxy"], geo.gs_j,
                       geo.flags, bd, npx, npy, **d6kw)

    return {"np": (n1, n2, n3, n4, n5, n6),
            "jx": (j1, j2, j3, j4, j5, j6)}


_D_SW1_KEYS = ("crx_adv", "cry_adv", "xfx_adv", "yfx_adv", "ra_x",
               "ra_y", "ut", "vt", "allflux_x", "allflux_y", "delp",
               "pt", "w", "cx", "cy", "xflux", "yflux")


def test_d_sw1_duo_parity(chain):
    """gate 1 -- every ``d_sw1`` output, full arrays including the ut/vt
    workspace-sentinel cells.

    FIXTURE CLASS: LIMITER-CROSSING.  ``d_sw1`` runs ``fv_tp_2d`` twice
    (delp at hord 6, pt at hord 6), so the PPM monotonicity flags fire
    and a rounding-level lane difference near one of them can flip a
    branch.  This bound must NOT be shared with the pointwise gates.
    """
    n1, _, _, _, _, _ = chain["np"]
    j1 = chain["jx"][0]
    for k in _D_SW1_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j1[k], n1[k], f"d_sw1_duo.{k}", 1e-15)


def test_d_sw1_duo_matches_the_stored_fortran_certificate(chain):
    """SECONDARY (hop A + hop B in one shot).

    The stored ``dsw1_duo_oracle_c12.npz`` holds the VERBATIM Fortran
    stage outputs on these exact inputs.  Hop B's authority is the NumPy
    lane, so this is not what the bound is calibrated against -- but a
    JAX-vs-Fortran agreement here means neither translation step
    introduced the defect, which is the only cheap check of that kind
    available at stage level.
    """
    j1 = chain["jx"][0]
    orc = _oracle("dsw1_duo_oracle_c12.npz")
    pairs = {"crx": "crx_adv", "xfx": "xfx_adv", "rax": "ra_x",
             "cry": "cry_adv", "yfx": "yfx_adv", "ray": "ra_y",
             "ut": "ut", "vt": "vt", "xflux_out": "xflux",
             "yflux_out": "yflux", "cx_out": "cx", "cy_out": "cy"}
    for okey, jkey in pairs.items():
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j1[jkey], np.asarray(orc[okey], np.float64),
             f"d_sw1 vs Fortran.{okey}", 1e-15)
    for slot, okey in ((0, "afx1"), (3, "afx4")):
        _cmp(np.asarray(j1["allflux_x"])[:, :, slot],
             np.asarray(orc[okey], np.float64),
             f"d_sw1 vs Fortran.{okey}", 1e-15)
    for slot, okey in ((0, "afy1"), (3, "afy4")):
        _cmp(np.asarray(j1["allflux_y"])[:, :, slot],
             np.asarray(orc[okey], np.float64),
             f"d_sw1 vs Fortran.{okey}", 1e-15)


def test_d_sw1_workspace_sentinel_cells_coincide(chain, geo_dsw):
    """The ut/vt ``workspace_sentinel`` fill is a CONTRACT, not cosmetics
    (the plain-conventions edge/corner blocks read cells the duo interior
    never writes).  Both lanes must leave the SAME set at 1e30, and the
    set must be non-empty -- otherwise the contract is untested."""
    n1 = chain["np"][0]
    j1 = chain["jx"][0]
    for k in ("ut", "vt"):
        a = np.asarray(j1[k]) == 1.0e30
        b = np.asarray(n1[k]) == 1.0e30
        assert np.array_equal(a, b), (
            f"{k}: sentinel sets differ ({int(a.sum())} vs "
            f"{int(b.sum())} of {a.size})")
    assert (np.asarray(n1["vt"]) == 1.0e30).any(), (
        "no sentinel cell survived -- the workspace contract is untested")


def test_d_sw1_capacitor_accumulation_is_exact(geo_dsw):
    """gate 5a -- the TOLERANCE-INDEPENDENT conservation gate.

    INVARIANT: the four capacitors are the running SUM of exactly the
    fluxes the stage exports, cell for cell::

        cx_out    - cx_in    == crx_adv
        cy_out    - cy_in    == cry_adv
        xflux_out - xflux_in == allflux_x[..., slot 1]
        yflux_out - yflux_in == allflux_y[..., slot 1]

    WHY THIS IS THE RIGHT QUANTITY FOR THIS STAGE.  ``d_sw1`` computes
    FLUXES only; the thing dyn_core's first barrier
    (``dyn_core.F90:872``, ``CGRID_NE``) averages across a panel seam is
    exactly the interface flux, and the capacitors are its time
    accumulation.  If the accumulation window, the slot mapping, or the
    array origin drifted, the barrier would average one flux while the
    tracer transport advected another -- i.e. mass would be created at
    the seam, which is the single defect a flux-form core must not have.
    The identity is checked BITWISE because ``cx_in + crx`` is one
    float64 add: recomputing it in numpy gives the identical double, so
    the gate's power does not depend on any tolerance and cannot be
    weakened by tuning one.  It is run with a NONZERO ``cx_in`` so that
    ``0 + x == x`` is not what carries it.
    """
    rng = np.random.default_rng(17)
    res, m_a = geo_dsw.res, geo_dsw.m_a
    caps = (rng.standard_normal((res + 1, res)) * 1.0e3,
            rng.standard_normal((res, res + 1)) * 1.0e3,
            rng.standard_normal((res + 1, m_a)) * 1.0e-2,
            rng.standard_normal((m_a, res + 1)) * 1.0e-2)
    out = _d_sw1_jax(geo_dsw, caps=caps)
    xf_in, yf_in, cx_in, cy_in = caps
    assert np.array_equal(np.asarray(out["cx"]),
                          cx_in + np.asarray(out["crx_adv"]))
    assert np.array_equal(np.asarray(out["cy"]),
                          cy_in + np.asarray(out["cry_adv"]))
    assert np.array_equal(np.asarray(out["xflux"]),
                          xf_in + np.asarray(out["allflux_x"])[:, :, 0])
    assert np.array_equal(np.asarray(out["yflux"]),
                          yf_in + np.asarray(out["allflux_y"])[:, :, 0])
    # non-vacuity: the increments are not zero
    assert np.abs(np.asarray(out["crx_adv"])).max() > 0.0
    assert np.abs(np.asarray(out["allflux_x"])[:, :, 0]).max() > 0.0


def test_d_sw1_mass_flux_divergence_telescopes(chain):
    """gate 5b -- the FLUX-FORM mass invariant, structural.

    The delp update ``d_sw2`` applies is
    ``(fx(i)-fx(i+1)+fy(j)-fy(j+1)) * rarea``.  Summing the bracket over
    the compute domain must TELESCOPE to the boundary flux alone -- an
    identity of the flux-form discretisation, with no interior source.
    Its residual is pure summation roundoff, so the check is structural;
    the bound below scales the residual by the total flux magnitude and
    is therefore dimensionless and grid-independent.

    This is the tier-2c "global dry mass" invariant evaluated at stage
    level.  Note explicitly that the sum runs over the OWNED compute
    domain (is..ie, js..je) and excludes halo cells.
    """
    j1 = chain["jx"][0]
    fx = np.asarray(j1["allflux_x"])[:, :, 0]
    fy = np.asarray(j1["allflux_y"])[:, :, 0]
    div = fx[:-1, :] - fx[1:, :] + fy[:, :-1] - fy[:, 1:]
    interior = float(div.sum())
    boundary = float((fx[0, :] - fx[-1, :]).sum()
                     + (fy[:, 0] - fy[:, -1]).sum())
    scale = float(np.abs(fx).sum() + np.abs(fy).sum())
    assert scale > 0.0
    rel = abs(interior - boundary) / scale
    # MEASURED (job 9425294 sweep): 2.187e-16; bound = measured x 10 =
    # 2.2e-15.
    gate_scalar("d_sw1 flux-divergence telescoping", rel, 2.2e-15,
                quantity="telescoping-identity residual rel")


def test_d_sw1_no_retrace_on_a_new_dt(geo_dsw):
    """gate 2a -- ``dt`` is DYNAMIC: a new time step must NOT retrace.

    SPLIT OUT (job 9404093) from the jit-vs-eager gate below, so a
    retrace regression cannot hide behind a tolerance change to the
    numeric half.  This half is tolerance-free.
    """
    box, wrapped = _counted(duo.d_sw1_duo)
    fn = duo.make_d_sw1_duo_jit(wrapped)
    _d_sw1_jax(geo_dsw, fn=fn)
    xf, yf, cx, cy = _zero_caps(geo_dsw)
    f = geo_dsw.f
    fn(jnp.asarray(f["delp"]), jnp.asarray(f["pt"]),
       jnp.asarray(f["w"]), jnp.asarray(f["uc"]), jnp.asarray(f["vc"]),
       jnp.asarray(xf), jnp.asarray(yf), jnp.asarray(cx),
       jnp.asarray(cy), geo_dsw.gs_j, geo_dsw.flags, geo_dsw.bd,
       geo_dsw.npx, geo_dsw.npy, dt=2.0 * geo_dsw.dt, **_DSW_KW)
    assert box["n"] == 1, f"retraced on a new dt: {box['n']}"


def test_d_sw1_jit_equals_eager(geo_dsw):
    """gate 2b -- the NUMERIC half, asserted with a measured bound.

    NOT bitwise.  ``d_sw1``'s outputs carry the documented
    sentinel-propagated cascade (values to ~1e111, see ``_cmp``'s
    retraction), and every one of those cells is ordinary arithmetic
    containing ``x*y + z`` -- which XLA contracts into an FMA when
    jitted and not when eager.  A few-ULP gap there is correct.
    """
    eager = _d_sw1_jax(geo_dsw)
    got = _d_sw1_jax(geo_dsw, fn=duo.make_d_sw1_duo_jit())
    # The two allflux stacks carry the sentinel-propagated CASCADE, and
    # FMA contraction on cascade arithmetic shows up there far above
    # rounding: MEASURED at job 9408115, `allflux_x` = 7.276e-12
    # per-element rel on 28 of 312 cells, median|ref| 1.526e+16 (four
    # decades above any physical flux on this C12 fixture) and max|diff|
    # 1.142e+99.  Those magnitudes are the cascade's, not the model's.
    # The COUNT is gated and every violator must sit in the extreme
    # tail; the magnitude is reported, not certified.
    for k in ("allflux_x", "allflux_y"):
        _jit_gap(got[k], eager[k], f"d_sw1 jit.{k}",
                 max_cells=60, tail_ratio=1.0e4)
    for k in _D_SW1_KEYS:
        if k in ("allflux_x", "allflux_y"):
            continue
        # MEASURED (job 9425294 sweep): worst yflux 1.216e-10 with a HANDFUL of cells over
        # 1e-13 (PPM limiter branch flip under FMA contraction, the documented
        # fourth kernel class; xflux 7.276e-12, all Courant/area keys
        # <= 3.4e-16); bound = measured x 10 = 1.3e-09.
        _cmp(got[k], eager[k], f"d_sw1 jit.{k}", 1.3e-9)


def test_d_sw1_guard_matrix(geo_dsw):
    """gate 3 -- explicit, one assertion per guard."""
    f = geo_dsw.f
    xf, yf, cx, cy = _zero_caps(geo_dsw)
    base = (jnp.asarray(f["pt"]), jnp.asarray(f["w"]),
            jnp.asarray(f["uc"]), jnp.asarray(f["vc"]), jnp.asarray(xf),
            jnp.asarray(yf), jnp.asarray(cx), jnp.asarray(cy),
            geo_dsw.gs_j, geo_dsw.flags, geo_dsw.bd, geo_dsw.npx,
            geo_dsw.npy)
    delp = jnp.asarray(f["delp"])
    with pytest.raises(TypeError, match="float64"):
        duo.d_sw1_duo(jnp.asarray(f["delp"], jnp.float32), *base,
                      dt=geo_dsw.dt, **_DSW_KW)
    for bad_key in ("hord_dp", "hord_tm", "hord_vt", "hord_tr"):
        kw = {**_DSW_KW, bad_key: 14}
        with pytest.raises(ValueError, match="not a supported scheme"):
            duo.d_sw1_duo(delp, *base, dt=geo_dsw.dt, **kw)
    with pytest.raises(ValueError, match="non-negative python int"):
        duo.d_sw1_duo(delp, *base, dt=geo_dsw.dt,
                      **{**_DSW_KW, "nord_v": -1})
    with pytest.raises(NotImplementedError, match="DUO-stage port"):
        duo.d_sw1_duo(delp, *base, dt=geo_dsw.dt, duogrid=False,
                      **_DSW_KW)
    with pytest.raises(NotImplementedError, match="inline_q"):
        duo.d_sw1_duo(delp, *base, dt=geo_dsw.dt, inline_q=True,
                      **_DSW_KW)


def test_d_sw1_hord_guard_is_non_vacuous(geo_dsw, monkeypatch):
    """gate 3 -- WITHOUT the guard, ``hord_dp = 14`` silently runs the
    ``{9, 13}`` PPM arm and SKIPS its ``pert_ppm`` positive-definite
    pass, i.e. different numerics with no error.  ``_validate_ord`` is
    patched away in BOTH modules because ``fv_tp_2d`` re-validates."""
    monkeypatch.setattr(duo, "_validate_ord", lambda *a, **k: None)
    monkeypatch.setattr(tp, "_validate_ord", lambda *a, **k: None)
    out = _d_sw1_jax(geo_dsw, hord_dp=14)
    assert np.isfinite(np.asarray(out["allflux_x"])[:, :, 0]).any()
    ref = _d_sw1_jax(geo_dsw)
    assert not np.array_equal(
        np.asarray(out["allflux_x"])[:, :, 0],
        np.asarray(ref["allflux_x"])[:, :, 0])


def test_d_sw1_gradients(geo_dsw):
    """gate 4 -- adjoint identity through the whole transport stage.

    Scored on the exported fluxes and the Courant numbers, which is what
    barrier 1 and every downstream stage actually consume.
    """
    f = geo_dsw.f
    xf, yf, cx, cy = _zero_caps(geo_dsw)

    def run(delp, pt, uc, vc):
        out = duo.d_sw1_duo(
            delp, pt, jnp.asarray(f["w"]), uc, vc, jnp.asarray(xf),
            jnp.asarray(yf), jnp.asarray(cx), jnp.asarray(cy),
            geo_dsw.gs_j, geo_dsw.flags, geo_dsw.bd, geo_dsw.npx,
            geo_dsw.npy, dt=geo_dsw.dt, **_DSW_KW)
        return (out["allflux_x"][:, :, 0], out["allflux_y"][:, :, 0],
                out["crx_adv"], out["cry_adv"])

    primals = (jnp.asarray(f["delp"]), jnp.asarray(f["pt"]),
               jnp.asarray(f["uc"]), jnp.asarray(f["vc"]))
    # MEASURED (job 9425294 sweep): adjoint residual 1.505e-16; bound = measured x 10 =
    # 1.6e-15.
    _check_adjoint("d_sw1_duo", run, primals, 1.6e-15)


def test_d_sw1_one_sided_across_the_crx_upwind_surface(geo_dsw):
    """gate 4 -- the ``xfx_adv > 0`` surface in ``crx``/``xfx``.

    Surface: ``crx = xfx*rdxa(i-1,j)`` versus ``xfx*rdxa(i,j)`` (and the
    matching ``sin_sg`` pair for ``xfx``).  Scaling ``uc``/``vc`` by
    ``s`` scales ``ut``, hence ``xfx_adv``, linearly, so ``s`` crossing
    zero flips the branch at every cell at once.  The two one-sided
    derivatives of ``crx`` must therefore equal ``ut*dt*rdxa`` evaluated
    at the two DIFFERENT metric cells; they coincide only where
    ``rdxa(i-1,j) == rdxa(i,j)``, so the test asserts the surface is
    real at a cell where they differ.
    """
    f = geo_dsw.f
    xf, yf, cx, cy = _zero_caps(geo_dsw)
    bd = geo_dsw.bd
    lo_i = bd.is_
    icell, jcell = bd.is_ + 5, bd.js + 5

    def crx_at(s):
        out = duo.d_sw1_duo(
            jnp.asarray(f["delp"]), jnp.asarray(f["pt"]),
            jnp.asarray(f["w"]), s * jnp.asarray(f["uc"]),
            s * jnp.asarray(f["vc"]), jnp.asarray(xf), jnp.asarray(yf),
            jnp.asarray(cx), jnp.asarray(cy), geo_dsw.gs_j,
            geo_dsw.flags, bd, geo_dsw.npx, geo_dsw.npy, dt=geo_dsw.dt,
            **_DSW_KW)
        return out["crx_adv"][icell - lo_i, jcell - bd.jsd]

    rdxa = np.asarray(geo_dsw.gs_np["rdxa"])
    m_pos = float(rdxa[icell - 1 - bd.isd, jcell - bd.jsd])
    m_neg = float(rdxa[icell - bd.isd, jcell - bd.jsd])
    assert abs(m_pos - m_neg) > 0.0, "the two arms are indistinguishable"

    # Which arm s>0 selects depends on the SIGN of ut*dt at this cell,
    # which is a property of the fixture -- read it rather than assume.
    base = float(crx_at(jnp.asarray(1.0)))
    assert abs(base) > 0.0, "crx is zero here -- no surface to cross"
    g = jax.grad(crx_at)
    eps = 1e-7
    right = float(g(jnp.asarray(+eps)))
    left = float(g(jnp.asarray(-eps)))
    assert np.isfinite(right) and np.isfinite(left)
    assert abs(right - left) > 0.0, (
        "the one-sided derivatives coincide -- the upwind surface was "
        "not crossed")
    # Each side is the SAME base quantity times its OWN metric, so the
    # ratio of the one-sided derivatives is the ratio of the two rdxa
    # cells (or its reciprocal, set by the sign of ut*dt).
    want = m_pos / m_neg if base > 0.0 else m_neg / m_pos
    assert right / left == pytest.approx(want, rel=1e-9), (
        f"one-sided ratio {right / left:.12e} != rdxa ratio "
        f"{want:.12e} -- the branch derivative is not the documented "
        f"one")


# ---------------------------------------------------------------------
# d_sw2
# ---------------------------------------------------------------------

_D_SW2_KEYS = ("delp", "pt", "heat_source", "ptc", "dw")


def test_d_sw2_duo_parity(chain):
    """gate 1 -- the hydrostatic UPDATE stage, full data domain (the
    compute update AND the untouched halo strips) plus the two
    sentinel round-trips.

    FIXTURE CLASS: pure cell update given the fluxes -- no limiter, no
    branch at all -- so this is the accumulating class and its bound
    should be near the flux-divergence rounding level.
    """
    n2 = chain["np"][1]
    j2 = chain["jx"][1]
    for k in _D_SW2_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j2[k], n2[k], f"d_sw2_duo.{k}", 1e-15)
    assert j2["w"] is None and n2["w"] is None


def test_d_sw2_duo_matches_the_stored_fortran_certificate(chain):
    """SECONDARY -- hop A + hop B against the verbatim Fortran chain."""
    j2 = chain["jx"][1]
    orc = _oracle("dsw2_duo_oracle_c12.npz")
    for okey, jkey in (("delp2", "delp"), ("pt2", "pt"),
                       ("ptc2", "ptc"), ("heat", "heat_source"),
                       ("dw", "dw")):
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j2[jkey], np.asarray(orc[okey], np.float64),
             f"d_sw2 vs Fortran.{okey}", 1e-15)


def test_d_sw2_duo_nh_arm_parity(geo_dsw):
    """gate 1 -- the NON-hydrostatic arm: the ``w`` mass-weighted update
    from allflux slot 2, plus the ``damp_w`` del-6 block that writes
    ``dw`` and ``heat_source``.  Both are dead on the hydrostatic lane,
    so without this test the whole NH half of the stage is unexercised.

    ⛔ FIXED (job 9404093).  The committed ``dswcore_input.npz`` carries
    ``w`` IDENTICALLY ZERO (measured: ``max|w| = 0.0``, 0 nonzero cells
    of 324), so ``del6_vt_flux(nord_w, damp4, w=0, …)`` gives
    ``d2 = damp*0 = 0``, ``fx2 = fy2 = 0`` and ``dw ≡ 0``.  The five
    parity comparisons all PASSED -- the two lanes agree on the NH arm,
    including ``w`` and ``dw`` -- and what failed was this test's own
    non-vacuity assertion, correctly: *a control that perturbs a zero is
    not a control*.  The arm is now driven with a NON-ZERO ``w``,
    byte-identical on both lanes, and the fixture's zero is asserted so
    the reason for the override is recorded rather than lost.
    """
    geo = geo_dsw
    assert np.abs(np.asarray(geo.f["w"])).max() == 0.0, (
        "the fixture's w is no longer zero -- re-derive whether the "
        "override below is still needed")
    ii = np.arange(geo.m_a, dtype=np.float64)[:, None]
    jj = np.arange(geo.m_a, dtype=np.float64)[None, :]
    w0 = 0.5 * np.cos(0.35 * ii) * np.sin(0.27 * jj) + 0.05 * ii / geo.m_a
    n1 = _d_sw1_np(geo, hydrostatic=False, w_override=w0)
    j1 = _d_sw1_jax(geo, hydrostatic=False, w_override=w0)
    kw = dict(w=None, npx=geo.npx, npy=geo.npy, dt=geo.dt, kgb=1.0e-3,
              nord_w=2, damp_w=0.15, hydrostatic=False)
    n2 = npduo.d_sw2_duo(n1["delp"], n1["pt"], n1["allflux_x"],
                         n1["allflux_y"], geo.gs_np, geo.bd,
                         **{**kw, "w": n1["w"]})
    j2 = duo.d_sw2_duo(j1["delp"], j1["pt"], j1["allflux_x"],
                       j1["allflux_y"], geo.gs_j, geo.flags, geo.bd,
                       **{**kw, "w": j1["w"]})
    for k in _D_SW2_KEYS + ("w",):
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j2[k], n2[k], f"d_sw2_duo NH.{k}", 1e-15)
    # non-vacuity, in two independent parts: the block EXECUTED (dw left
    # the workspace fill) and the arithmetic was NOT trivial (dw != 0).
    assert (np.asarray(j2["dw"]) != 1.0e30).all(), "damp_w block skipped"
    assert np.abs(np.asarray(j2["dw"])).max() > 0.0, (
        "dw is identically zero -- the del-6 damping is being fed a "
        "constant/zero w and this gate proves nothing")


def test_d_sw2_duo_jit_equals_eager(chain, geo_dsw):
    """gate 2 -- ASSERTED."""
    j1 = chain["jx"][0]
    args = (j1["delp"], j1["pt"], j1["allflux_x"], j1["allflux_y"],
            geo_dsw.gs_j, geo_dsw.flags, geo_dsw.bd)
    eager = duo.d_sw2_duo(*args)
    box, wrapped = _counted(duo.d_sw2_duo)
    fn = duo.make_d_sw2_duo_jit(wrapped)
    got = fn(*args)
    fn(args[0] * 1.001, *args[1:])
    assert box["n"] == 1, box["n"]
    for k in _D_SW2_KEYS:
        # MEASURED (job 9425294 sweep): worst pt 1.183e-16; bound = measured x 10 =
        # 1.2e-15.
        _cmp(got[k], eager[k], f"d_sw2 jit.{k}", 1.2e-15)


def test_d_sw2_duo_guards(chain, geo_dsw):
    """gate 3."""
    j1 = chain["jx"][0]
    args = (j1["delp"], j1["pt"], j1["allflux_x"], j1["allflux_y"],
            geo_dsw.gs_j, geo_dsw.flags, geo_dsw.bd)
    with pytest.raises(NotImplementedError, match="inline_q"):
        duo.d_sw2_duo(*args, inline_q=True)
    with pytest.raises(ValueError, match="NH arm needs"):
        duo.d_sw2_duo(*args, hydrostatic=False)
    with pytest.raises(TypeError, match="float64"):
        duo.d_sw2_duo(jnp.asarray(j1["delp"], jnp.float32), *args[1:])


def test_d_sw2_duo_gradients(chain, geo_dsw):
    """gate 4 -- adjoint identity on the updated ``delp``/``pt``."""
    j1 = chain["jx"][0]

    def f(delp, pt, afx, afy):
        out = duo.d_sw2_duo(delp, pt, afx, afy, geo_dsw.gs_j,
                            geo_dsw.flags, geo_dsw.bd)
        lo = geo_dsw.bd.isd
        ring = slice(geo_dsw.bd.is_ - lo, geo_dsw.bd.ie + 1 - lo)
        return jnp.stack([out["delp"][ring, ring],
                          out["pt"][ring, ring]])

    primals = (j1["delp"], j1["pt"], jnp.nan_to_num(j1["allflux_x"]),
               jnp.nan_to_num(j1["allflux_y"]))
    # MEASURED (job 9425294 sweep): adjoint residual 4.500e-16; bound = measured x 10 =
    # 4.6e-15.
    _check_adjoint("d_sw2_duo", f, primals, 4.6e-15)


# ---------------------------------------------------------------------
# d_sw3
# ---------------------------------------------------------------------

_D_SW3_KEYS = ("ubbtemp", "vbbtemp", "ubb", "vbb")


def test_d_sw3_duo_parity(chain):
    """gate 1 -- the four B-grid fields barrier 2 acts on.

    FIXTURE CLASS: LIMITER-CROSSING (``ytp_v``/``xtp_u`` at hord 6 run
    the PPM monotonicity flags).
    """
    n3, j3 = chain["np"][2], chain["jx"][2]
    for k in _D_SW3_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j3[k], n3[k], f"d_sw3_duo.{k}", 1e-15)


def test_d_sw3_duo_matches_the_stored_fortran_certificate(chain):
    """SECONDARY -- hop A + hop B."""
    j3 = chain["jx"][2]
    orc = _oracle("dsw3_duo_oracle_c12.npz")
    for k in _D_SW3_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j3[k], np.asarray(orc[k], np.float64),
             f"d_sw3 vs Fortran.{k}", 1e-15)


def test_d_sw3_duo_snapshot_precedes_the_recompute(chain):
    """The ``ubbtemp``/``vbbtemp`` snapshot is taken BEFORE ``ub`` is
    recomputed and BEFORE ``xtp_u`` overwrites ``vb`` -- a claim about
    ORDER, so it is measured rather than asserted from the source: the
    snapshot must DIFFER from the final pair."""
    j3 = chain["jx"][2]
    assert not np.allclose(np.asarray(j3["ubbtemp"]),
                           np.asarray(j3["ubb"]))
    assert not np.allclose(np.asarray(j3["vbbtemp"]),
                           np.asarray(j3["vbb"]))


def test_d_sw3_duo_jit_equals_eager(geo_dsw):
    """gate 2 -- ASSERTED, and ``dt`` proved DYNAMIC."""
    f = geo_dsw.f
    args = (jnp.asarray(f["u"]), jnp.asarray(f["v"]),
            jnp.asarray(f["uc"]), jnp.asarray(f["vc"]), geo_dsw.gs_j,
            geo_dsw.flags, geo_dsw.bd, geo_dsw.npx, geo_dsw.npy)
    eager = duo.d_sw3_duo(*args, dt=geo_dsw.dt, hord_mt=6)
    box, wrapped = _counted(duo.d_sw3_duo)
    fn = duo.make_d_sw3_duo_jit(wrapped)
    got = fn(*args, dt=geo_dsw.dt, hord_mt=6)
    fn(*args, dt=2.0 * geo_dsw.dt, hord_mt=6)
    assert box["n"] == 1, f"retraced on a new dt: {box['n']}"
    for k in _D_SW3_KEYS:
        # MEASURED (job 9425294 sweep): worst ubbtemp 1.521e-10 with n_over 1 -- ONE cell,
        # a PPM limiter branch flip under FMA contraction, same class and
        # scale as job 9404093's 2.403e-10 (vbb 1.428e-10 / n_over 2;
        # ubb and vbbtemp <= 2.2e-16); bound = measured x 10 = 1.6e-09.
        # NOT the plain FMA class -- see the retraction banner in _cmp.
        _cmp(got[k], eager[k], f"d_sw3 jit.{k}", 1.6e-9)


def test_d_sw3_duo_guards(geo_dsw):
    """gate 3 -- hord_mt is validated against the STAGGERED set (no
    ``abs()``, so negatives are not a scheme there)."""
    f = geo_dsw.f
    args = (jnp.asarray(f["u"]), jnp.asarray(f["v"]),
            jnp.asarray(f["uc"]), jnp.asarray(f["vc"]), geo_dsw.gs_j,
            geo_dsw.flags, geo_dsw.bd, geo_dsw.npx, geo_dsw.npy)
    for bad in (0, -6, 12):
        with pytest.raises(ValueError, match="not a supported scheme"):
            duo.d_sw3_duo(*args, dt=geo_dsw.dt, hord_mt=bad)
    with pytest.raises(NotImplementedError, match="DUO-stage port"):
        duo.d_sw3_duo(*args, dt=geo_dsw.dt, duogrid=False)
    with pytest.raises(TypeError, match="float64"):
        duo.d_sw3_duo(jnp.asarray(f["u"], jnp.float32), *args[1:],
                      dt=geo_dsw.dt)


def test_d_sw3_hord_guard_is_non_vacuous(geo_dsw, monkeypatch):
    """gate 3 -- WITHOUT the guard, ``hord_mt = 12`` silently takes
    ``xtp_u``/``ytp_v``'s ``{11}`` unlimited else-arm."""
    monkeypatch.setattr(duo, "_validate_ord", lambda *a, **k: None)
    monkeypatch.setattr(tp, "_validate_ord", lambda *a, **k: None)
    f = geo_dsw.f
    out = duo.d_sw3_duo(jnp.asarray(f["u"]), jnp.asarray(f["v"]),
                        jnp.asarray(f["uc"]), jnp.asarray(f["vc"]),
                        geo_dsw.gs_j, geo_dsw.flags, geo_dsw.bd,
                        geo_dsw.npx, geo_dsw.npy, dt=geo_dsw.dt,
                        hord_mt=12)
    assert np.isfinite(np.asarray(out["ubb"])).any()


def test_d_sw3_duo_gradients(geo_dsw):
    """gate 4 -- adjoint identity through both staggered transports."""
    f = geo_dsw.f

    def run(u, v, uc, vc):
        out = duo.d_sw3_duo(u, v, uc, vc, geo_dsw.gs_j, geo_dsw.flags,
                            geo_dsw.bd, geo_dsw.npx, geo_dsw.npy,
                            dt=geo_dsw.dt, hord_mt=6)
        return jnp.stack([out[k] for k in _D_SW3_KEYS])

    primals = tuple(jnp.asarray(f[k]) for k in ("u", "v", "uc", "vc"))
    # MEASURED (job 9425294 sweep): adjoint residual 1.529e-16; bound = measured x 10 =
    # 1.6e-15.
    _check_adjoint("d_sw3_duo", run, primals, 1.6e-15)


# ---------------------------------------------------------------------
# d_sw4
# ---------------------------------------------------------------------

def test_d_sw4_duo_parity(chain):
    """gate 1 -- the four-corner KE fix, full B array."""
    n4, j4 = chain["np"][3], chain["jx"][3]
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(j4["ke"], n4["ke"], "d_sw4_duo.ke", 1e-15)


def test_d_sw4_duo_matches_the_stored_fortran_certificate(chain,
                                                          geo_dsw):
    """SECONDARY -- hop A + hop B, on the certificate's OWN input.

    ⛔ FIXED (job 9404093).  This gate previously fed ``d_sw4`` the
    ``chain`` fixture's ``kee`` assembly and compared the result against
    a certificate the Fortran driver produced from a DIFFERENT input.
    The fixture says so itself -- ``dsw4_duo_oracle_c12.npz``'s
    ``input_lineage`` reads *"ke=1e30 pre-call on both sides, only the 4
    corner B-nodes written"* -- and the committed NumPy harness
    (``test_fv3_native_dsw4_duo._run_chain``) feeds
    ``ke0 = np.full((m_a+1, m_a+1), SENTINEL)``.  The failure was
    exactly that mismatch: 192 fill cells (361 − the 169-cell kee ring)
    against the certificate's 357 (361 − 4).  The OPERATOR was never in
    question -- ``test_d_sw4_duo_parity`` (JAX vs NumPy on the kee
    input) and ``test_d_sw4_duo_writes_exactly_four_cells`` both passed.

    ``ke`` is INTENT(INOUT), so the input IS part of the contract: the
    certificate is only meaningful against the input it was generated
    from.  The chain keeps feeding the kee assembly because that is what
    ``d_sw5`` consumes; this gate runs its own call.
    """
    geo = geo_dsw
    j1 = chain["jx"][0]
    orc = _oracle("dsw4_duo_oracle_c12.npz")
    lineage = str(orc["input_lineage"])
    assert "ke=1e30 pre-call" in lineage, (
        f"the certificate's input convention changed: {lineage!r} -- "
        f"re-derive ke0 before trusting this gate")
    ke0 = jnp.full((geo.m_a + 1, geo.m_a + 1), 1.0e30, jnp.float64)
    j4 = duo.d_sw4_duo(jnp.asarray(geo.f["u"]), jnp.asarray(geo.f["v"]),
                       j1["ut"], j1["vt"], ke0, geo.flags, geo.bd,
                       geo.npx, geo.npy, dt=geo.dt)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(j4["ke"], np.asarray(orc["ke"], np.float64),
         "d_sw4 vs Fortran.ke", 1e-15)
    # the gate is non-vacuous only if the four corners are NOT fills
    got = np.asarray(j4["ke"])
    assert int((got != 1.0e30).sum()) == 4, (
        f"expected exactly 4 written corners, got "
        f"{int((got != 1.0e30).sum())}")


def test_d_sw4_duo_writes_exactly_four_cells(chain, geo_dsw):
    """STRUCTURAL, tolerance-independent: ``d_sw4`` is INTENT(INOUT) on
    ``ke`` and the oracle writes ONLY the four corner B-nodes.  Anything
    else touched is a window bug, and a NUMERIC gate would not see it if
    the extra write happened to be small."""
    j3, j4 = chain["jx"][2], chain["jx"][3]
    bd, ng, res = geo_dsw.bd, geo_dsw.ng, geo_dsw.res
    m_a = geo_dsw.m_a
    ring = slice(ng, ng + res + 1)
    # rebuild the INPUT with the identical jnp expression the chain
    # fixture used, so "unchanged" means bitwise unchanged and the cell
    # census is not confounded by a numpy-vs-jnp rounding difference
    blk = 0.5 * (j3["ubbtemp"] * j3["vbbtemp"] + j3["ubb"] * j3["vbb"])
    ke_in = np.asarray(
        jnp.full((m_a + 1, m_a + 1), 1.0e30).at[ring, ring].set(blk))
    ke_out = np.asarray(j4["ke"])
    changed = np.argwhere(ke_in != ke_out)
    lo = bd.isd
    want = {(1 - lo, 1 - lo), (geo_dsw.npx - lo, 1 - lo),
            (geo_dsw.npx - lo, geo_dsw.npy - lo),
            (1 - lo, geo_dsw.npy - lo)}
    assert {tuple(c) for c in changed} == want, (
        f"d_sw4 touched {len(changed)} cells, expected the four "
        f"corners {sorted(want)}")


def test_d_sw4_duo_jit_and_gradients(chain, geo_dsw):
    """gates 2 and 4 in one -- the stage is four scalar writes."""
    f, j1, j3 = geo_dsw.f, chain["jx"][0], chain["jx"][2]
    m_a, ng, res = geo_dsw.m_a, geo_dsw.ng, geo_dsw.res
    ring = slice(ng, ng + res + 1)
    ke = jnp.full((m_a + 1, m_a + 1), 1.0e30).at[ring, ring].set(
        0.5 * (j3["ubbtemp"] * j3["vbbtemp"] + j3["ubb"] * j3["vbb"]))
    args = (jnp.asarray(f["u"]), jnp.asarray(f["v"]), j1["ut"],
            j1["vt"], ke, geo_dsw.flags, geo_dsw.bd, geo_dsw.npx,
            geo_dsw.npy)
    eager = duo.d_sw4_duo(*args, dt=geo_dsw.dt)
    box, wrapped = _counted(duo.d_sw4_duo)
    fn = duo.make_d_sw4_duo_jit(wrapped)
    got = fn(*args, dt=geo_dsw.dt)
    fn(*args, dt=2.0 * geo_dsw.dt)
    assert box["n"] == 1, f"retraced on a new dt: {box['n']}"
    # MEASURED (job 9425294 sweep): 7.277e-17; bound = measured x 10 =
    # 7.3e-16.
    _cmp(got["ke"], eager["ke"], "d_sw4 jit.ke", 7.3e-16)

    lo = geo_dsw.bd.isd

    def run(u, v):
        out = duo.d_sw4_duo(u, v, jnp.nan_to_num(j1["ut"]),
                            jnp.nan_to_num(j1["vt"]), ke, geo_dsw.flags,
                            geo_dsw.bd, geo_dsw.npx, geo_dsw.npy,
                            dt=geo_dsw.dt)
        idx = [(1 - lo, 1 - lo), (geo_dsw.npx - lo, 1 - lo),
               (geo_dsw.npx - lo, geo_dsw.npy - lo),
               (1 - lo, geo_dsw.npy - lo)]
        return jnp.stack([out["ke"][i, j] for i, j in idx])

    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise) adjoint residual; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _check_adjoint("d_sw4_duo", run,
                   (jnp.asarray(f["u"]), jnp.asarray(f["v"])), 1e-15)


# ---------------------------------------------------------------------
# d_sw5
# ---------------------------------------------------------------------

_D_SW5_KEYS = ("delpc", "divg_d", "wk", "ke", "vortfluxx", "vortfluxy",
               "uc", "vc", "ut", "vt", "ptc", "ub", "vb")


def test_d_sw5_duo_parity(chain):
    """gate 1 -- every ``d_sw5`` output including the CLOBBERED uc/vc
    gradient workspaces and the three sentinel round-trips.

    FIXTURE CLASS: LIMITER-CROSSING (the vorticity transport is a full
    ``fv_tp_2d`` at hord 6) AND non-smooth in its own right (the
    ``sqrt`` and the ``max/min`` clamp).
    """
    n5, j5 = chain["np"][4], chain["jx"][4]
    for k in _D_SW5_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j5[k], n5[k], f"d_sw5_duo.{k}", 1e-15)
    assert j5["w"] is None and n5["w"] is None


def test_d_sw5_duo_matches_the_stored_fortran_certificate(chain):
    """SECONDARY -- hop A + hop B."""
    j5 = chain["jx"][4]
    orc = _oracle("dsw5_duo_oracle_c12.npz")
    for k in ("delpc", "ptc", "wk", "divg_d", "ke", "uc", "ut", "vc",
              "vt", "vortfluxx", "vortfluxy", "ub", "vb"):
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j5[k], np.asarray(orc[k], np.float64),
             f"d_sw5 vs Fortran.{k}", 1e-15)


@pytest.mark.parametrize("nord,dddmp", [(2, 0.2), (1, 0.0)])
def test_d_sw5_duo_branch_matrix_parity(nord, dddmp, geo_dsw, chain):
    """gate 1 -- the two branches the default arm does NOT reach: a
    SECOND damping pass (``nord = 2``, i.e. the recurrence really
    recurring) and the ``dddmp < 1e-5`` arm, which skips ``a2b_ord4``
    entirely and zeroes ``vort``.  Without this the pass loop is only
    ever exercised at trip count 1."""
    geo = geo_dsw
    f = geo.f
    n1, j1 = chain["np"][0], chain["jx"][0]
    n4, j4 = chain["np"][3], chain["jx"][3]
    kw = dict(hord_vt=6, nord=nord, dddmp=dddmp, d2_bg=0.0, d4_bg=0.12,
              d_con=0.0)
    n5 = npduo.d_sw5_duo(f["delp"], f["u"], f["v"], f["uc"], f["vc"],
                         f["ua"], f["va"], f["divg_d_in"],
                         n1["crx_adv"], n1["cry_adv"], n1["xfx_adv"],
                         n1["yfx_adv"], n1["ra_x"], n1["ra_y"],
                         n4["ke"], geo.gs_np, geo.bd, geo.npx, geo.npy,
                         dt=geo.dt, **kw)
    j5 = duo.d_sw5_duo(jnp.asarray(f["delp"]), jnp.asarray(f["u"]),
                       jnp.asarray(f["v"]), jnp.asarray(f["uc"]),
                       jnp.asarray(f["vc"]), jnp.asarray(f["ua"]),
                       jnp.asarray(f["va"]),
                       jnp.asarray(f["divg_d_in"]), j1["crx_adv"],
                       j1["cry_adv"], j1["xfx_adv"], j1["yfx_adv"],
                       j1["ra_x"], j1["ra_y"], j4["ke"], geo.gs_j,
                       geo.flags, geo.bd, geo.npx, geo.npy, dt=geo.dt,
                       **kw)
    for k in _D_SW5_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key/nord/dddmp; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j5[k], n5[k], f"d_sw5 nord={nord} dddmp={dddmp}.{k}", 1e-15)


def test_d_sw5_duo_nh_arm_parity(geo_dsw):
    """gate 1 -- the NH ``w`` finalisation (mass-weighted ``w`` divided
    by the UPDATED ``delp``, plus the ``d_sw2`` ``dw`` increment).

    The inputs are produced by the real NH pipeline
    (d_sw1(NH) -> d_sw2(NH)), so the two lanes see byte-identical
    operands and the only variable is the lane.
    """
    geo = geo_dsw
    f = geo.f
    n1 = _d_sw1_np(geo, hydrostatic=False)
    j1 = _d_sw1_jax(geo, hydrostatic=False)
    d2kw = dict(npx=geo.npx, npy=geo.npy, dt=geo.dt, kgb=1.0e-3,
                nord_w=2, damp_w=0.15, hydrostatic=False)
    n2 = npduo.d_sw2_duo(n1["delp"], n1["pt"], n1["allflux_x"],
                         n1["allflux_y"], geo.gs_np, geo.bd,
                         w=n1["w"], **d2kw)
    j2 = duo.d_sw2_duo(j1["delp"], j1["pt"], j1["allflux_x"],
                       j1["allflux_y"], geo.gs_j, geo.flags, geo.bd,
                       w=j1["w"], **d2kw)
    n3 = npduo.d_sw3_duo(f["u"], f["v"], f["uc"], f["vc"], geo.gs_np,
                         geo.bd, geo.npx, geo.npy, dt=geo.dt,
                         hord_mt=6)
    ring = slice(geo.ng, geo.ng + geo.res + 1)
    ke0 = np.full((geo.m_a + 1, geo.m_a + 1), 1.0e30)
    ke0[ring, ring] = 0.5 * (n3["ubbtemp"] * n3["vbbtemp"]
                             + n3["ubb"] * n3["vbb"])
    d5kw = dict(hord_vt=6, nord=1, dddmp=0.2, d2_bg=0.0, d4_bg=0.12,
                d_con=0.0, hydrostatic=False, damp_w=0.15)
    n5 = npduo.d_sw5_duo(n2["delp"], f["u"], f["v"], f["uc"], f["vc"],
                         f["ua"], f["va"], f["divg_d_in"],
                         n1["crx_adv"], n1["cry_adv"], n1["xfx_adv"],
                         n1["yfx_adv"], n1["ra_x"], n1["ra_y"], ke0,
                         geo.gs_np, geo.bd, geo.npx, geo.npy,
                         dt=geo.dt, w=n2["w"], dw=n2["dw"], **d5kw)
    j5 = duo.d_sw5_duo(j2["delp"], jnp.asarray(f["u"]),
                       jnp.asarray(f["v"]), jnp.asarray(f["uc"]),
                       jnp.asarray(f["vc"]), jnp.asarray(f["ua"]),
                       jnp.asarray(f["va"]),
                       jnp.asarray(f["divg_d_in"]), j1["crx_adv"],
                       j1["cry_adv"], j1["xfx_adv"], j1["yfx_adv"],
                       j1["ra_x"], j1["ra_y"], jnp.asarray(ke0),
                       geo.gs_j, geo.flags, geo.bd, geo.npx, geo.npy,
                       dt=geo.dt, w=j2["w"], dw=j2["dw"], **d5kw)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), w and every key; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(j5["w"], n5["w"], "d_sw5_duo NH.w", 1e-15)
    for k in ("wk", "ke", "divg_d"):
        _cmp(j5[k], n5[k], f"d_sw5_duo NH.{k}", 1e-15)


def test_d_sw5_duo_jit_equals_eager(chain, geo_dsw):
    """gate 2 -- ASSERTED, and ``dt`` proved DYNAMIC."""
    geo, f = geo_dsw, geo_dsw.f
    j1, j4 = chain["jx"][0], chain["jx"][3]
    args = (jnp.asarray(f["delp"]), jnp.asarray(f["u"]),
            jnp.asarray(f["v"]), jnp.asarray(f["uc"]),
            jnp.asarray(f["vc"]), jnp.asarray(f["ua"]),
            jnp.asarray(f["va"]), jnp.asarray(f["divg_d_in"]),
            j1["crx_adv"], j1["cry_adv"], j1["xfx_adv"], j1["yfx_adv"],
            j1["ra_x"], j1["ra_y"], j4["ke"], geo.gs_j, geo.flags,
            geo.bd, geo.npx, geo.npy)
    kw = dict(hord_vt=6, nord=1, dddmp=0.2, d2_bg=0.0, d4_bg=0.12,
              d_con=0.0)
    eager = duo.d_sw5_duo(*args, dt=geo.dt, **kw)
    box, wrapped = _counted(duo.d_sw5_duo)
    fn = duo.make_d_sw5_duo_jit(wrapped)
    got = fn(*args, dt=geo.dt, **kw)
    fn(*args, dt=2.0 * geo.dt, **kw)
    assert box["n"] == 1, f"retraced on a new dt: {box['n']}"
    for k in _D_SW5_KEYS:
        # MEASURED (job 9425294 sweep): worst vortfluxy 1.134e-13 (sentinel-cascade FMA class); bound = measured x 10 =
        # 1.2e-12.
        _cmp(got[k], eager[k], f"d_sw5 jit.{k}", 1.2e-12)


def test_d_sw5_duo_guards(chain, geo_dsw):
    """gate 3 -- the oracle-lane refusals and the scheme guards."""
    geo, f = geo_dsw, geo_dsw.f
    j1, j4 = chain["jx"][0], chain["jx"][3]
    args = (jnp.asarray(f["delp"]), jnp.asarray(f["u"]),
            jnp.asarray(f["v"]), jnp.asarray(f["uc"]),
            jnp.asarray(f["vc"]), jnp.asarray(f["ua"]),
            jnp.asarray(f["va"]), jnp.asarray(f["divg_d_in"]),
            j1["crx_adv"], j1["cry_adv"], j1["xfx_adv"], j1["yfx_adv"],
            j1["ra_x"], j1["ra_y"], j4["ke"], geo.gs_j, geo.flags,
            geo.bd, geo.npx, geo.npy)
    with pytest.raises(ValueError, match="not a supported scheme"):
        duo.d_sw5_duo(*args, dt=geo.dt, hord_vt=14)
    with pytest.raises(NotImplementedError, match="oracle lane only"):
        duo.d_sw5_duo(*args, dt=geo.dt, nord=3)
    with pytest.raises(NotImplementedError, match="oracle lane only"):
        duo.d_sw5_duo(*args, dt=geo.dt, d_con=0.5)
    with pytest.raises(ValueError, match="NH arm needs w"):
        duo.d_sw5_duo(*args, dt=geo.dt, hydrostatic=False)
    with pytest.raises(TypeError, match="float64"):
        duo.d_sw5_duo(jnp.asarray(f["delp"], jnp.float32), *args[1:],
                      dt=geo.dt)


def test_d_sw5_duo_gradients(chain, geo_dsw):
    """gate 4 -- adjoint identity through the damping recurrence, the
    ``a2b_ord4`` corner interpolation and the vorticity transport."""
    geo, f = geo_dsw, geo_dsw.f
    j1, j4 = chain["jx"][0], chain["jx"][3]
    lo = geo.bd.isd
    ring = slice(geo.bd.is_ - lo, geo.bd.ie + 1 - lo)

    def run(u, v, divg_in, ke):
        out = duo.d_sw5_duo(
            jnp.asarray(f["delp"]), u, v, jnp.asarray(f["uc"]),
            jnp.asarray(f["vc"]), jnp.asarray(f["ua"]),
            jnp.asarray(f["va"]), divg_in, j1["crx_adv"], j1["cry_adv"],
            j1["xfx_adv"], j1["yfx_adv"], j1["ra_x"], j1["ra_y"], ke,
            geo.gs_j, geo.flags, geo.bd, geo.npx, geo.npy, dt=geo.dt,
            hord_vt=6, nord=1, dddmp=0.2, d2_bg=0.0, d4_bg=0.12,
            d_con=0.0)
        return jnp.stack([out["ke"][ring, ring],
                          out["divg_d"][ring, ring],
                          out["wk"][ring, ring]])

    primals = (jnp.asarray(f["u"]), jnp.asarray(f["v"]),
               jnp.asarray(f["divg_d_in"]),
               jnp.nan_to_num(j4["ke"], posinf=0.0, neginf=0.0))
    # MEASURED (job 9425294 sweep): adjoint residual 1.515e-16; bound = measured x 10 =
    # 1.6e-15.
    _check_adjoint("d_sw5_duo", run, primals, 1.6e-15)


def test_d_sw5_damping_recurrence_is_ordered(geo_dsw, chain):
    """The ``do n=1,nord`` loop is a RECURRENCE, not a repeated
    independent pass -- a claim about ORDER, so it is measured: running
    at ``nord = 2`` must NOT equal running twice at ``nord = 1`` (the
    windows shrink with ``nt = nord - n``), and it must differ from
    ``nord = 1``."""
    geo, f = geo_dsw, geo_dsw.f
    j1, j4 = chain["jx"][0], chain["jx"][3]
    base = (jnp.asarray(f["delp"]), jnp.asarray(f["u"]),
            jnp.asarray(f["v"]), jnp.asarray(f["uc"]),
            jnp.asarray(f["vc"]), jnp.asarray(f["ua"]),
            jnp.asarray(f["va"]), jnp.asarray(f["divg_d_in"]),
            j1["crx_adv"], j1["cry_adv"], j1["xfx_adv"], j1["yfx_adv"],
            j1["ra_x"], j1["ra_y"], j4["ke"], geo.gs_j, geo.flags,
            geo.bd, geo.npx, geo.npy)
    kw = dict(dt=geo.dt, hord_vt=6, dddmp=0.2, d2_bg=0.0, d4_bg=0.12,
              d_con=0.0)
    a = np.asarray(duo.d_sw5_duo(*base, nord=1, **kw)["divg_d"])
    b = np.asarray(duo.d_sw5_duo(*base, nord=2, **kw)["divg_d"])
    m = np.isfinite(a) & np.isfinite(b)
    assert m.any()
    assert not np.allclose(a[m], b[m]), (
        "nord=1 and nord=2 give the same divg_d -- the pass loop is not "
        "running")


# ---------------------------------------------------------------------
# d_sw6
# ---------------------------------------------------------------------

_D_SW6_KEYS = ("u", "v", "ut", "vt", "ub", "vb", "heat_source")


def test_d_sw6_duo_parity(chain):
    """gate 1 -- the final circulation-form wind update plus the del-6
    vorticity damping that CLOBBERS ut/vt."""
    n6, j6 = chain["np"][5], chain["jx"][5]
    for k in _D_SW6_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j6[k], n6[k], f"d_sw6_duo.{k}", 1e-15)


def test_d_sw6_duo_matches_the_stored_fortran_certificate(chain):
    """SECONDARY -- hop A + hop B, and the check that pins the
    ``fx2``/``fy2`` SEED behaviour end to end: ``ut``/``vt`` come back
    with the del-6 fluxes written into the incoming arrays, so a fresh
    NaN allocation would fail here even though the wind update would
    still be right."""
    j6 = chain["jx"][5]
    orc = _oracle("dsw6_duo_oracle_c12.npz")
    for k in ("u", "v", "ut", "vt", "ub", "vb"):
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key incl heat; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j6[k], np.asarray(orc[k], np.float64),
             f"d_sw6 vs Fortran.{k}", 1e-15)
    _cmp(j6["heat_source"], np.asarray(orc["heat"], np.float64),
         "d_sw6 vs Fortran.heat", 1e-15)


def test_d_sw6_duo_jit_equals_eager(chain, geo_dsw):
    """gate 2 -- ASSERTED."""
    geo, f = geo_dsw, geo_dsw.f
    j5 = chain["jx"][4]
    args = (jnp.asarray(f["u"]), jnp.asarray(f["v"]), j5["ut"],
            j5["vt"], j5["ke"], j5["wk"], j5["vortfluxx"],
            j5["vortfluxy"], geo.gs_j, geo.flags, geo.bd, geo.npx,
            geo.npy)
    eager = duo.d_sw6_duo(*args)
    box, wrapped = _counted(duo.d_sw6_duo)
    fn = duo.make_d_sw6_duo_jit(wrapped)
    got = fn(*args)
    fn(args[0] * 1.001, *args[1:])
    assert box["n"] == 1, box["n"]
    # ``ut``/``vt`` are ``del6_vt_flux``'s outputs -- a del-6 operator is
    # a CHAIN OF DIFFERENCES OF LARGE NEARLY-EQUAL NUMBERS, so its
    # condition number is enormous wherever the field is locally smooth
    # and a 1-ULP FMA perturbation is amplified at isolated cells.
    # MEASURED at job 9408115: `ut` = 1.257e-06 per-element rel on
    # exactly 2 of 342 cells.  See
    # test_d_sw6_ut_gap_is_not_a_limiter_branch_flip for why the
    # branch-flip explanation is REFUTED here.  Count-gated, magnitude
    # reported.
    for k in ("ut", "vt"):
        _jit_gap(got[k], eager[k], f"d_sw6 jit.{k}", max_cells=8)
    for k in _D_SW6_KEYS:
        if k in ("ut", "vt"):
            continue
        # MEASURED (job 9425294 sweep): worst v 9.664e-16; bound = measured x 10 =
        # 9.7e-15.
        _cmp(got[k], eager[k], f"d_sw6 jit.{k}", 9.7e-15)


def test_d_sw6_duo_guards(chain, geo_dsw):
    """gate 3."""
    geo, f = geo_dsw, geo_dsw.f
    j5 = chain["jx"][4]
    args = (jnp.asarray(f["u"]), jnp.asarray(f["v"]), j5["ut"],
            j5["vt"], j5["ke"], j5["wk"], j5["vortfluxx"],
            j5["vortfluxy"], geo.gs_j, geo.flags, geo.bd, geo.npx,
            geo.npy)
    with pytest.raises(NotImplementedError, match="d_con=0"):
        duo.d_sw6_duo(*args, d_con=0.5)
    with pytest.raises(NotImplementedError, match="DUO-stage port"):
        duo.d_sw6_duo(*args, duogrid=False)
    with pytest.raises(ValueError, match="non-negative python int"):
        duo.d_sw6_duo(*args, nord_v=-1)
    with pytest.raises(TypeError, match="float64"):
        duo.d_sw6_duo(jnp.asarray(f["u"], jnp.float32), *args[1:])


def test_d_sw6_duo_gradients(chain, geo_dsw):
    """gate 4 -- adjoint identity on the updated winds."""
    geo, f = geo_dsw, geo_dsw.f
    j5 = chain["jx"][4]
    lo = geo.bd.isd
    ring = slice(geo.bd.is_ - lo, geo.bd.ie + 1 - lo)

    def run(u, v, ke, wk):
        out = duo.d_sw6_duo(u, v, jnp.nan_to_num(j5["ut"]),
                            jnp.nan_to_num(j5["vt"]), ke, wk,
                            jnp.nan_to_num(j5["vortfluxx"]),
                            jnp.nan_to_num(j5["vortfluxy"]), geo.gs_j,
                            geo.flags, geo.bd, geo.npx, geo.npy)
        return jnp.stack([out["u"][ring, ring], out["v"][ring, ring]])

    primals = (jnp.asarray(f["u"]), jnp.asarray(f["v"]),
               jnp.nan_to_num(j5["ke"], posinf=0.0, neginf=0.0),
               jnp.nan_to_num(j5["wk"], posinf=0.0, neginf=0.0))
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise) adjoint residual; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _check_adjoint("d_sw6_duo", run, primals, 1e-15)
def test_d_sw6_ut_gap_is_not_a_limiter_branch_flip(chain, geo_dsw):
    """⛔ REFUTES the proposed mechanism for ``d_sw6``'s jit gap, and
    names + measures the surviving candidate.

    PROPOSED (coordinator, run 5): FMA contraction shifts a sum by a few
    ULP under jit, that flips a ``>`` comparison in a PPM limiter, the
    limiter adds or drops a WHOLE term, and an O(1) change on two cells
    propagates into ``u``.

    REFUTED, structurally.  ``d_sw6``'s ``ut`` is written by exactly one
    operator -- ``del6_vt_flux``, called at sw_core.F90:1948-1951 -- and
    that routine contains NO data-dependent branch of any kind: no
    ``jnp.where``, no ``minimum``/``maximum``, no ``copysign``, no
    ``lax.cond``, no comparison on an array.  It is ``d2 = damp*q``,
    ``fx2 = del6_v*(d2[i-1]-d2[i])`` and an ordered pass loop, all
    linear with fixed coefficients; its only ``if``s are on the STATIC
    ``nord``/``bounded_domain``/``damp_km is None``.  There is no
    predicate for an FMA shift to flip.  This test asserts that
    branch-freedom mechanically, on the symbol that actually RUNS.

    Two corrections while we are here, both checkable above:
      * in ``d_sw6`` ``ut`` feeds ``v`` (``v -= ut``) and ``vt`` feeds
        ``u`` (``u += vt``) -- so "``d_sw6`` writes ``ut``, therefore
        ``u``" is off by one.  ``vt`` comes from the SAME
        ``del6_vt_flux`` call, so a localisation to the stage chain can
        still hold through ``vt``;
      * ``vt`` PASSED at 1e-12 in the same call in which ``ut`` failed
        at 1.26e-6.  Same operator, same inputs, one output clean --
        which is itself evidence against anything systematic.

    SURVIVING CANDIDATE (PLAUSIBLE, measured below): CATASTROPHIC
    CANCELLATION.  A del-6 operator IS a chain of differences of large
    nearly-equal numbers, so at a cell where the field is locally smooth
    the condition number of the final difference can reach ~1e10 and a
    1-ULP upstream perturbation lands at 1e-6.  The discriminator is the
    cancellation ratio at the worst cell: if the operator's terms are
    many decades larger than their difference, cancellation is the
    mechanism; if the ratio is O(1), this is refuted too and something
    else is producing the gap.
    """
    import inspect

    # ---- (1) branch-freedom of the ONLY writer, mechanically --------
    tokens = ("jnp.where", "jnp.minimum", "jnp.maximum", "copysign",
              "lax.cond", "lax.select")
    src = inspect.getsource(duo.del6_vt_flux)
    body = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    found = [t for t in tokens if t in body]
    assert not found, (
        f"del6_vt_flux is no longer branch-free ({found}) -- the "
        f"branch-flip mechanism may now apply and this refutation must "
        f"be re-derived")
    # NON-VACUITY: the same scan MUST find branches in a routine that
    # has them, otherwise it proves nothing about del6_vt_flux.
    ref_src = inspect.getsource(duo.d_sw1_duo)
    assert [t for t in tokens if t in ref_src], (
        "the branch scanner found nothing in d_sw1_duo either -- the "
        "scanner is broken, so the assertion above is vacuous")

    # ---- (2) the cancellation ratio at the worst cell ---------------
    geo, f = geo_dsw, geo_dsw.f
    j5 = chain["jx"][4]
    args = (jnp.asarray(f["u"]), jnp.asarray(f["v"]), j5["ut"],
            j5["vt"], j5["ke"], j5["wk"], j5["vortfluxx"],
            j5["vortfluxy"], geo.gs_j, geo.flags, geo.bd, geo.npx,
            geo.npy)
    eager = duo.d_sw6_duo(*args)
    jitted = duo.make_d_sw6_duo_jit()(*args)
    per, bad, detail = _jit_gap(jitted["ut"], eager["ut"],
                                "d_sw6 ut mechanism", max_cells=8)
    if not bad.any():
        pytest.skip(f"no cell exceeds 1e-13 on this build -- {detail}")

    k = np.unravel_index(int(np.argmax(per)), per.shape)
    # The operator's own terms at that cell: damp * q * del6_v, versus
    # the output it produced.  del6_vt_flux is linear, so this ratio IS
    # the cancellation the difference performed.
    damp4 = (0.2 * geo.flags.da_min_c) ** (1 + 1)
    wk = np.abs(np.asarray(j5["wk"]))
    d6v = np.abs(np.asarray(geo.gs_j["del6_v"]))
    term = damp4 * float(np.median(wk[np.isfinite(wk)])) * \
        float(np.median(d6v[np.isfinite(d6v)]))
    out = abs(float(np.asarray(eager["ut"])[k]))
    cond = term / max(out, 1e-300)
    eps = float(np.finfo(np.float64).eps)
    predicted = cond * eps
    measured = float(per[k])
    assert cond >= 1.0e4, (
        f"NO significant cancellation at the worst cell {k}: "
        f"terms {term:.3e} vs output {out:.3e} gives condition "
        f"{cond:.3e}. The cancellation mechanism is REFUTED too -- the "
        f"gap has another cause. MEASURED gap {measured:.3e}. {detail}")
    # Reported, NOT asserted as an equality: the condition number is a
    # median-based estimate, so it fixes the DECADE, not the digit.
    print(f"\nd_sw6 ut cancellation check at cell {k}: "
          f"condition ~{cond:.3e}, cond*eps ~{predicted:.3e}, "
          f"MEASURED gap {measured:.3e} "
          f"(UNEXPLAINED-BY-TOLERANCE; count-gated, not bounded)")


# ---------------------------------------------------------------------
# R1b regression for _sel_div -- the four panel-edge divides
# ---------------------------------------------------------------------

def _edge_site(geo, site):
    """``(pred, pos_slot, neg_slot)`` for one ``d_sw1_duo`` panel edge.

    ``pred`` is the site's predicate evaluated in NUMPY on the fixture,
    and each slot is the ``(i_index, j_index, k)`` addressing of one
    arm's denominator, given as a callable of the free index so the
    caller can zero it only where that arm is DEAD.
    """
    lo = geo.bd.isd
    dt, npx, npy = geo.dt, geo.npx, geo.npy
    uc = np.asarray(geo.f["uc"])
    vc = np.asarray(geo.f["vc"])
    if site == "west":                       # sw_core.F90:658-663
        return (uc[1 - lo, :] * dt > 0.0,
                (0 - lo, slice(None), 2), (1 - lo, slice(None), 0))
    if site == "east":
        return (uc[npx - lo, :] * dt > 0.0,
                (npx - 1 - lo, slice(None), 2), (npx - lo, slice(None), 0))
    if site == "south":
        return (vc[:, 1 - lo] * dt > 0.0,
                (slice(None), 0 - lo, 3), (slice(None), 1 - lo, 1))
    if site == "north":
        return (vc[:, npy - lo] * dt > 0.0,
                (slice(None), npy - 1 - lo, 3), (slice(None), npy - lo, 1))
    raise ValueError(site)                   # pragma: no cover


def _run_both_lanes(geo, sg_bad):
    """Both lanes of ``d_sw1_duo`` on ONE (possibly malformed)
    gridstruct, so the comparison stays one-variable."""
    gs_np_bad = dict(geo.gs_np)
    gs_np_bad["sin_sg"] = sg_bad
    gs_j_bad = dict(geo.gs_j)
    gs_j_bad["sin_sg"] = jnp.asarray(sg_bad)
    xf, yf, cx, cy = _zero_caps(geo)
    f = geo.f
    n1 = npduo.d_sw1_duo(f["delp"], f["pt"], f["w"], f["uc"], f["vc"],
                         xf, yf, cx, cy, gs_np_bad, geo.bd, geo.npx,
                         geo.npy, dt=geo.dt, **_DSW_KW)
    j1 = duo.d_sw1_duo(
        jnp.asarray(f["delp"]), jnp.asarray(f["pt"]),
        jnp.asarray(f["w"]), jnp.asarray(f["uc"]), jnp.asarray(f["vc"]),
        jnp.asarray(xf), jnp.asarray(yf), jnp.asarray(cx),
        jnp.asarray(cy), gs_j_bad, geo.flags, geo.bd, geo.npx, geo.npy,
        dt=geo.dt, **_DSW_KW)
    return n1, j1, gs_j_bad


@pytest.mark.parametrize("site", ["west", "east", "south", "north"])
def test_sel_div_survives_a_zero_in_the_dead_denominator(site, geo_dsw):
    """R1b regression for ``_sel_div``, ONE SITE PER TEST so a failure
    names the site (coordinator, run 5).

    ⛔ The previous version of this test was MIS-CONSTRUCTED and its
    failure was mine, not ``_sel_div``'s.  It zeroed ``sin_sg(1,j,1)``
    across the WHOLE west column.  MEASURED on the fixture:
    ``uc(1,j)*dt > 0`` holds for 15 of 18 j and FAILS for 3 -- so at
    those 3 j the zeroed entry is the SELECTED denominator, both lanes
    legitimately divide by zero, and a non-finite gradient is the
    CORRECT answer.  The reported "3 non-finite entries" is exactly
    those 3 cells.  ``_sel_div`` never leaked.

    This version zeroes each arm's denominator ONLY at the indices where
    that arm is DEAD, which is precisely the R1b hazard and nothing
    else: NumPy (a python ``if``) never touches it, and an unsanitized
    twin would form Inf/NaN there and carry it into the selected
    branch's gradient as ``NaN * 0``.

    Two assertions, and the second is the one the unsanitized version
    would fail: value parity vs NumPy, and a FINITE reverse-mode
    gradient.
    """
    geo = geo_dsw
    pred, pos_slot, neg_slot = _edge_site(geo, site)
    sg_bad = np.array(geo.gs_np["sin_sg"], dtype=np.float64)
    # the POSITIVE arm's denominator is dead where pred is False;
    # the NEGATIVE arm's is dead where pred is True.
    n_zeroed = 0
    for slot, dead in ((pos_slot, ~pred), (neg_slot, pred)):
        if not dead.any():
            continue
        idx = list(slot)
        free = 0 if isinstance(slot[0], slice) else 1
        sel = np.where(dead)[0]
        if free == 0:
            sg_bad[sel, idx[1], idx[2]] = 0.0
        else:
            sg_bad[idx[0], sel, idx[2]] = 0.0
        n_zeroed += int(dead.sum())
    assert n_zeroed > 0, (
        f"{site}: no index has a dead arm on this fixture, so nothing "
        f"was zeroed and this gate proves nothing")

    n1, j1, gs_j_bad = _run_both_lanes(geo, sg_bad)

    # (1) value parity with a zero in every DEAD denominator
    for k in ("ut", "vt", "crx_adv", "cry_adv"):
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every site/key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(j1[k], n1[k], f"d_sw1 {site} dead-zero.{k}", 1e-15)

    # (2) the gradient stays FINITE -- not implied by (1), and this is
    # what an unsanitized `where`-over-divide fails
    f = geo.f
    xf, yf, cx, cy = _zero_caps(geo)

    def obj(uc_in):
        out = duo.d_sw1_duo(
            jnp.asarray(f["delp"]), jnp.asarray(f["pt"]),
            jnp.asarray(f["w"]), uc_in, jnp.asarray(f["vc"]),
            jnp.asarray(xf), jnp.asarray(yf), jnp.asarray(cx),
            jnp.asarray(cy), gs_j_bad, geo.flags, geo.bd, geo.npx,
            geo.npy, dt=geo.dt, **_DSW_KW)
        return jnp.sum(jnp.nan_to_num(out["ut"]) ** 2) \
            + jnp.sum(jnp.nan_to_num(out["vt"]) ** 2)

    g = np.asarray(jax.grad(obj)(jnp.asarray(f["uc"])))
    n_bad = int((~np.isfinite(g)).sum())
    assert n_bad == 0, (
        f"{site}: reverse-mode gradient has {n_bad} non-finite entries "
        f"with {n_zeroed} DEAD denominators zeroed -- the dead arm's "
        f"division is leaking through the where VJP, i.e. _sel_div is "
        f"not sanitizing this site")


@pytest.mark.parametrize("site", ["west", "east", "south", "north"])
def test_sel_div_still_propagates_a_zero_in_the_live_denominator(
        site, geo_dsw):
    """The CONTROL for the test above, and the reason its verdict is
    trustworthy.

    ``_sel_div`` must sanitize the DEAD arm and leave the LIVE one
    alone.  If it silently repaired the live denominator too, the
    dead-arm test would still pass while the operator quietly returned a
    wrong finite number instead of the Inf both lanes owe.  So: put the
    zero in the SELECTED denominator and assert BOTH lanes go
    non-finite, on the SAME cells.  This is also the direct measurement
    behind the retraction above -- it is what the previous test was
    accidentally doing.
    """
    geo = geo_dsw
    pred, pos_slot, neg_slot = _edge_site(geo, site)
    sg_bad = np.array(geo.gs_np["sin_sg"], dtype=np.float64)
    # zero the POSITIVE arm's denominator where pred is TRUE == LIVE
    live = pred
    if not live.any():
        pytest.skip(f"{site}: the positive arm is never selected here")
    idx = list(pos_slot)
    sel = np.where(live)[0]
    if isinstance(pos_slot[0], slice):
        sg_bad[sel, idx[1], idx[2]] = 0.0
    else:
        sg_bad[idx[0], sel, idx[2]] = 0.0

    n1, j1, _ = _run_both_lanes(geo, sg_bad)
    key = "ut" if site in ("west", "east") else "vt"
    a = np.asarray(j1[key])
    b = np.asarray(n1[key])
    assert not np.isfinite(a).all(), (
        f"{site}: JAX stayed finite with the LIVE denominator zeroed -- "
        f"_sel_div is sanitizing the SELECTED arm, which silently "
        f"changes the answer instead of propagating the division by "
        f"zero both lanes owe")
    assert np.array_equal(~np.isfinite(a), ~np.isfinite(b)), (
        f"{site}: the two lanes disagree about WHICH cells go "
        f"non-finite (jax {int((~np.isfinite(a)).sum())} vs numpy "
        f"{int((~np.isfinite(b)).sum())}) -- that is a real lane "
        f"divergence, not a shared division by zero")
