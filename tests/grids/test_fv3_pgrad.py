"""Certification of the JAX pressure-gradient chain against the NumPy
fp64 lane (``legoesm.core.fv3_pgrad`` vs ``fv3_native_pgrad``).

Five gates per public routine -- the four from the NH JAX-mirror
pattern-setter (``test_fv3_nh_core.py``) plus one this chain earns:

1. **equivalence** vs the NumPy twin on the SAME fixtures the existing
   NumPy gate uses (``_synthetic``/``_gs``/``_nh_pgrad_fields`` are
   imported from ``test_fv3_native_pgrad``, not re-invented), plus the
   committed C12 oracle fixture for the geopk -> p_grad_c -> one_grad_p
   chain;
2. **jit vs eager** -- an ASSERTION on the arrays, plus a trace counter
   proving the production jit policy does not retrace on new VALUES;
3. **guards** -- float32 raises TypeError; every unported/unknown
   dispatch value raises;
4. ``jax.test_util.check_grads(order=2)`` away from the singular sites,
   with the site named and a control proving it is inactive on the
   fixture;
5. **NaN poison** -- the operand region each routine must never read is
   filled with NaN, and BOTH the primal and the reverse-mode gradient
   must stay finite.  This is the local form of the "a select evaluates
   both operands" hazard: this module contains no ``jnp.where`` at all
   (every branch is a static Python ``if``), but it DOES carry the NumPy
   twin's NaN-filled scratch, and a one-index window slip would pull a
   NaN into a live expression -- in the primal or, via ``0 * NaN``, in
   the VJP alone.

TOLERANCES: every numeric bound below is marked ``TOL-PENDING`` and set
to a provisional 1e-12.  They are NOT measured -- the measurement job
replaces each with ``measured X, bound = measured x N``.
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

# The NumPy twin is imported as a MODULE (`npg.geopk`, `npg.p_grad_c`,
# ...) rather than as eight aliased names: it names the lane at every
# call site in a two-lane parity file, and it keeps ruff's isort from
# splitting an aliased member list into eight separate from-imports.
from legoesm.core import fv3_native_pgrad as npg  # noqa: E402
from legoesm.core.fv3_native_sw_core import BIG_NUMBER, Bounds  # noqa: E402
from legoesm.core.fv3_pgrad import (  # noqa: E402
    a2b_ord4,
    a2b_gridstruct_view,
    geopk,
    geopk_jit,
    make_geopk_jit,
    make_nh_p_grad_jit,
    make_one_grad_p_jit,
    make_p_grad_c_jit,
    make_pe_halo_jit,
    make_pk3_halo_jit,
    make_pln_halo_jit,
    nh_p_grad,
    nh_p_grad_jit,
    one_grad_p,
    one_grad_p_jit,
    p_grad_c,
    p_grad_c_jit,
    pe_halo,
    pe_halo_jit,
    pk3_halo,
    pk3_halo_jit,
    pln_halo,
    pln_halo_jit,
)

from tests.grids.test_fv3_native_pgrad import (  # noqa: E402
    M_A,
    M_B,
    NG,
    N,
    _gs,
    _nh_pgrad_fields,
    _synthetic,
)

# Same single-tile bounds the NumPy gate's `bd` fixture builds.  (That
# fixture is a pytest fixture function and cannot be called directly, so
# the ONE line is repeated rather than the numerics.)
BD = Bounds.single_tile(N, NG)
NPX = NPY = N + 1
PTOP, AKAP, CP_AIR = 100.0, 2.0 / 7.0, 1004.0

_GS_CACHE: dict = {}


# The keys this chain READS, and nothing else.  ``_gs()``'s full
# gridstruct also carries Python ints (`n`, `ng`, `npx`) and bool masks;
# handing that whole dict to a jitted routine would turn every one of
# them into a traced leaf for no reason.  BOTH lanes get this reduced
# dict, so the comparison stays one-variable: the NumPy twin's
# ``a2b_gridstruct_view`` reads the corner/domain flags through
# ``gs.get(..., default)`` and the defaults (bounded_domain=False,
# grid_type=0, all four corners True) are exactly what ``_gs()`` sets and
# what the JAX side is told explicitly.
_GS_KEYS = ("rdxc", "rdyc", "rdx", "rdy", "grid_lon", "grid_lat",
            "agrid_lon", "agrid_lat", "dxa", "dya", "edge_w", "edge_e",
            "edge_s", "edge_n")


def _gs_cached() -> dict:
    """The read keys of ``_gs()``, memoised (it builds real C12 geometry)."""
    if "gs" not in _GS_CACHE:
        full = _gs()
        _GS_CACHE["gs"] = {k: np.asarray(full[k], dtype=np.float64)
                           for k in _GS_KEYS}
    return _GS_CACHE["gs"]


# check_grads, READ FROM THE INSTALLED SOURCE (jax
# `_src/public_test_util.py`), not from memory:
#   :33   EPS = 1e-4                        -- an ABSOLUTE central-difference step
#   :88   default_gradient_tolerance[float64] = 1e-5
#   :159-164 _assert_numpy_close scales BOTH bounds by the leaf size:
#            `_assert_numpy_allclose(a, b, atol=atol * a.size,
#                                    rtol=rtol * b.size)`
# so the effective per-leaf bound is 1e-5 * size -- which is exactly the
# `rtol=0.00312, atol=0.00312` printed by job 9400424 on a 312-element
# leaf (12 x 13 x 2).  Two consequences drive every gradient gate below:
#   * a DIRECTIONAL DERIVATIVE below ~1e-5 * leaf_size (3e-3 for the
#     u/v leaves here, 5e-3 for the 507-element pk/gz leaves) is
#     compared against atol and the check is VACUOUS.  Each call
#     therefore normalises its inputs AND its outputs to O(1), which
#     turns every derivative into a RELATIVE sensitivity, and operand
#     groups whose sensitivity differs by orders of magnitude get their
#     own call instead of being swamped in the shared VJP inner product.
#   * with O(1) inputs the 1e-4 step is a 1e-4 RELATIVE perturbation,
#     the usual f64 central-difference choice.
# An earlier version of this comment said the floor was 1e-5 flat; that
# was wrong by the size factor (~2.5 orders here) and is corrected here.
_GRAD_TOL_PER_ELEMENT = 1e-5


def _grad_vacuity_floor(n_elements: int) -> float:
    """The bound a directional derivative must clear to be tested."""
    return _GRAD_TOL_PER_ELEMENT * n_elements


def _rel(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _cmp(name, got, want, tol):
    r = _rel(got, want)
    assert r <= tol, (
        f"{name}: rel {r:.3e} > {tol:.3e} "
        f"(bitwise={np.array_equal(np.asarray(got), np.asarray(want))})")


def _finite(name, x):
    assert np.all(np.isfinite(np.asarray(x))), (
        f"{name}: non-finite entries "
        f"({int(np.sum(~np.isfinite(np.asarray(x))))} of "
        f"{np.asarray(x).size})")


def test_grad_vacuity_floor_matches_the_installed_jax_rule():
    """Pin the rule every gradient gate's scaling is built on.

    ``0.00312`` in job 9400424's log is ``1e-5 * 312`` -- the per-element
    gradient tolerance times the leaf size.  If a jax upgrade changes
    either factor, every scaling choice below silently moves toward
    vacuous, so the rule is asserted rather than commented.  A moved
    module path fails LOUDLY here on purpose; do not turn that into a
    skip."""
    from jax._src import public_test_util as ptu

    assert ptu.default_gradient_tolerance[np.dtype(np.float64)] == \
        _GRAD_TOL_PER_ELEMENT
    assert _grad_vacuity_floor(312) == pytest.approx(0.00312, rel=1e-12)
    # and the step is ABSOLUTE, which is why the fixtures are normalised
    assert ptu.EPS == 1e-4


def _tree_dot(a, b) -> float:
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return float(sum(
        np.dot(np.asarray(x, dtype=np.float64).ravel(),
               np.asarray(y, dtype=np.float64).ravel())
        for x, y in zip(la, lb)))


def _adjoint_residual(f, primals, seed=0):
    """Relative residual of the adjoint identity ``<J v, w> == <v, J^T w>``.

    NO finite differences: ``J v`` comes from ``jax.jvp`` and ``J^T w``
    from ``jax.vjp``, so the identity is exact in exact arithmetic and
    the residual is pure floating-point roundoff -- its power does NOT
    depend on the FD step, on the operand scaling, or on the dynamic
    range of the output, which is precisely what an FD check loses on a
    badly spread array (job 9400424).

    It is also strictly stronger than an FD check for a THREADING
    defect: if the forward pass and the adjoint disagree about which
    array a value came from -- the `replace=True` read-after-write
    through ``a2b_ord4`` being the candidate here -- the two inner
    products differ at O(1), not at the noise floor.

    Returns ``(relative residual, <J v, w>, <v, J^T w>)``.  The caller
    asserts on the residual AND on ``<J v, w> != 0`` (a zero Jacobian
    would satisfy the identity trivially).
    """
    rng = np.random.default_rng(seed)
    primals = tuple(jnp.asarray(p) for p in primals)
    v = tuple(jnp.asarray(rng.standard_normal(p.shape)) for p in primals)
    _, jv = jax.jvp(f, primals, v)
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
    assert r <= tol, (
        f"{name}: adjoint residual {r:.3e} > {tol:.3e} "
        f"(<J v, w>={lhs:.12e}, <v, J^T w>={rhs:.12e}) -- MEASURED "
        f"value is {r:.3e}")


# =====================================================================
# a2b_ord4 -- the private mirror one_grad_p / nh_p_grad are built on
# =====================================================================

def _a2b_field(seed=3, shape=(M_A, M_A)):
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                         indexing="ij")
    return (100.0 + 3.0 * np.cos(2.0 * np.pi * ii / shape[0])
            * np.sin(2.0 * np.pi * (jj + 0.5) / shape[1])
            + 0.05 * rng.standard_normal(shape))


def _run_a2b_np(qin, gs, *, replace, duogrid, grid_type=0,
                bounded_domain=False):
    from legoesm.core.fv3_native_d_sw import a2b_ord4
    from legoesm.grids.fv3_native_gridstruct import fort

    gsv = dict(gs)
    gsv.update(grid_type=grid_type, bounded_domain=bounded_domain)
    gsf = npg.a2b_gridstruct_view(gsv, BD)
    qi = np.array(qin, dtype=np.float64, copy=True)
    qo = np.full((M_A, M_A), np.nan)
    a2b_ord4(fort(qi, BD.isd, BD.jsd), fort(qo, BD.isd, BD.jsd), gsf,
             NPX, NPY, BD.is_, BD.ie, BD.js, BD.je, NG, replace=replace,
             duogrid=duogrid)
    return qi, qo


def _run_a2b_jax(qin, gs, *, replace, duogrid, grid_type=0,
                 bounded_domain=False):
    geom = a2b_gridstruct_view(gs, BD)
    qi, qo = a2b_ord4(
        jnp.asarray(qin), jnp.full((M_A, M_A), jnp.nan, jnp.float64),
        geom, NPX, NPY, BD.is_, BD.ie, BD.js, BD.je, NG, replace=replace,
        duogrid=duogrid, grid_type=grid_type,
        bounded_domain=bounded_domain, sw_corner=True, se_corner=True,
        ne_corner=True, nw_corner=True)
    return np.asarray(qi), np.asarray(qo)


# ---------------------------------------------------------------- gate 1
# ALL THREE arms, because the JAX mirror ports all three: duo/bounded
# (what the pgrad lane runs), the plain corner+edge arm, and the
# doubly-periodic grid_type>=3 arm.  On this single tile is==1, ie+1==npx,
# js==1 and je+1==npy all hold, so the plain arm exercises every corner
# and every one-sided edge block.
@pytest.mark.parametrize(
    "arm", ["duo", "plain", "bounded", "periodic"])
@pytest.mark.parametrize("replace", [None, True, False])
def test_a2b_ord4_jax_matches_numpy_lane(arm, replace):
    gs = _gs_cached()
    q = _a2b_field()
    kw = dict(duogrid=(arm == "duo"),
              bounded_domain=(arm == "bounded"),
              grid_type=(3 if arm == "periodic" else 0))
    qi_n, qo_n = _run_a2b_np(q, gs, replace=replace, **kw)
    qi_j, qo_j = _run_a2b_jax(q, gs, replace=replace, **kw)

    box = (slice(NG, NG + N + 1), slice(NG, NG + N + 1))
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"a2b[{arm}] qout box", qo_j[box], qo_n[box], 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"a2b[{arm}] qin", qi_j, qi_n, 1e-12)
    # replace semantics are tolerance-independent: False and None are
    # BOTH no-ops, True rewrites exactly the B box.
    if replace:
        assert not np.array_equal(qi_j[box], np.asarray(q)[box])
    else:
        assert np.array_equal(qi_j, np.asarray(q, dtype=np.float64))
    # The unwritten qout slots must be NaN on BOTH lanes -- that is the
    # contract the poison gates below rely on.
    assert np.array_equal(np.isnan(qo_j), np.isnan(qo_n))


def test_a2b_ord4_jax_rejects_bad_shape_and_small_ng():
    gs = _gs_cached()
    geom = a2b_gridstruct_view(gs, BD)
    q3 = jnp.zeros((M_A, M_A, 2), jnp.float64)
    with pytest.raises(ValueError, match="2-D"):
        a2b_ord4(q3, q3, geom, NPX, NPY, BD.is_, BD.ie, BD.js, BD.je, NG)
    q2 = jnp.zeros((M_A, M_A), jnp.float64)
    with pytest.raises(ValueError, match="ng=1"):
        a2b_ord4(q2, q2, geom, NPX, NPY, BD.is_, BD.ie, BD.js, BD.je, 1)


@pytest.mark.parametrize("arm", ["duo", "plain"])
def test_a2b_ord4_jax_adjoint_consistency(arm):
    """FD-free adjoint identity on the operator the whole pgrad lane is
    built from.  This is where a ``replace=True`` threading defect would
    live, and it also proves the NaN-filled ``qout`` scratch stays out of
    the ADJOINT (a constant NaN whose cotangent is discarded), not just
    out of the primal."""
    gs = _gs_cached()
    q = _a2b_field()
    geom = a2b_gridstruct_view(gs, BD)
    box = (slice(NG, NG + N + 1), slice(NG, NG + N + 1))

    def f(qin):
        qi, qo = a2b_ord4(
            qin, jnp.full((M_A, M_A), jnp.nan, jnp.float64), geom, NPX,
            NPY, BD.is_, BD.ie, BD.js, BD.je, NG, replace=True,
            duogrid=(arm == "duo"), bounded_domain=False, grid_type=0,
            sw_corner=True, se_corner=True, ne_corner=True,
            nw_corner=True)
        return qi[box], qo[box]

    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint(f"a2b_ord4[{arm}]", f, (q,), 1e-12)


def test_a2b_gridstruct_view_rejects_float32_and_missing_keys():
    gs = dict(_gs_cached())
    with pytest.raises(TypeError, match="float64"):
        a2b_gridstruct_view({**gs, "dxa": np.asarray(gs["dxa"],
                                                    dtype=np.float32)}, BD)
    del gs["edge_w"]
    with pytest.raises(KeyError, match="edge_w"):
        a2b_gridstruct_view(gs, BD)


# =====================================================================
# geopk
# =====================================================================

_GEOPK_SITES = [
    # (cg, computehalo) -- the two dyn_core call sites: :534 (C grid) and
    # :1401 (D grid, computehalo=.true.)
    (True, False),
    (False, True),
]


def _geopk_kw(km, *, cg, computehalo, fill=BIG_NUMBER, npx=NPX,
              npy=NPY):
    return dict(km=km, ptop=PTOP, akap=AKAP, cp_air=CP_AIR, cg=cg,
                duogrid=True, computehalo=computehalo, npx=npx, npy=npy,
                a2b_ord=4, bounded_domain=False, sw_dynamics=False,
                q_con=None, use_cond=False, unwritten_fill=fill)


# ---------------------------------------------------------------- gate 1
@pytest.mark.parametrize("km", [1, 2, 3])
@pytest.mark.parametrize("cg,computehalo", _GEOPK_SITES)
def test_geopk_jax_matches_numpy_lane(km, cg, computehalo):
    st = _synthetic(km)
    kw = _geopk_kw(km, cg=cg, computehalo=computehalo, fill=0.0)
    want = npg.geopk(st["delp"], st["pt"], st["hs"], BD, **kw)
    got = geopk(st["delp"], st["pt"], st["hs"], BD, **kw)
    for name in ("pk", "gz", "pe", "peln", "pkz"):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"geopk[{km},cg={cg}].{name}", got[name], want[name], 1e-12)
    # Non-vacuity: the chain actually produced structure.
    assert float(np.abs(np.asarray(got["gz"])).max()) > 1.0


@pytest.mark.parametrize("km", [1, 2, 3])
@pytest.mark.parametrize("cg,computehalo", _GEOPK_SITES)
def test_geopk_jax_write_footprint_is_identical(km, cg, computehalo):
    """Tolerance-INDEPENDENT footprint check (strategy doc S4: prefer a
    check whose power does not depend on how tight the bound is).  The
    1e30 sentinel marks every slot the Fortran never writes, so equality
    of the sentinel MASK certifies the write window itself."""
    st = _synthetic(km)
    kw = _geopk_kw(km, cg=cg, computehalo=computehalo)
    want = npg.geopk(st["delp"], st["pt"], st["hs"], BD, **kw)
    got = geopk(st["delp"], st["pt"], st["hs"], BD, **kw)
    for name in ("pk", "gz", "pe", "peln", "pkz"):
        mw = np.asarray(want[name]) == BIG_NUMBER
        mg = np.asarray(got[name]) == BIG_NUMBER
        assert np.array_equal(mg, mw), (
            f"geopk.{name}: write footprint differs "
            f"({int(mg.sum())} vs {int(mw.sum())} sentinel slots)")
    # Non-vacuity: at least one field HAS unwritten slots at the C site.
    if cg:
        assert (np.asarray(want["pkz"]) == BIG_NUMBER).all()


# ---------------------------------------------------------------- gate 2
def test_geopk_jax_jit_eager_parity_and_no_retrace():
    km = 3
    st = _synthetic(km)
    kw = _geopk_kw(km, cg=False, computehalo=True, fill=0.0)
    eager = geopk(st["delp"], st["pt"], st["hs"], BD, **kw)
    jitted = geopk_jit(st["delp"], st["pt"], st["hs"], BD, **kw)
    for name in ("pk", "gz", "pe", "peln", "pkz"):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"geopk jit/eager {name}", jitted[name], eager[name], 1e-12)

    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return geopk(*a, **k)

    fn = make_geopk_jit(_counted)          # the PRODUCTION policy
    fn(st["delp"], st["pt"], st["hs"], BD, **kw)
    # Different VALUES, same shapes/dtypes/statics -> must NOT retrace.
    fn(1.01 * st["delp"], st["pt"] + 1.0, st["hs"], BD, **kw)
    assert traces["n"] == 1, traces["n"]


# ---------------------------------------------------------------- gate 3
def test_geopk_jax_guards_raise():
    st = _synthetic(2)
    kw = _geopk_kw(2, cg=True, computehalo=False)
    with pytest.raises(ValueError, match="USE_COND"):
        geopk(st["delp"], st["pt"], st["hs"], BD,
              **{**kw, "use_cond": True})
    with pytest.raises(ValueError, match="a2b_ord"):
        geopk(st["delp"], st["pt"], st["hs"], BD, **{**kw, "a2b_ord": 3})
    with pytest.raises(ValueError, match="km"):
        geopk(st["delp"], st["pt"], st["hs"], BD, **{**kw, "km": 0})
    # dispatch hardening: a non-bool branch selector is not a branch
    with pytest.raises(ValueError, match="not a Python bool"):
        geopk(st["delp"], st["pt"], st["hs"], BD, **{**kw, "cg": 1})
    with pytest.raises(ValueError, match="not a Python bool"):
        geopk(st["delp"], st["pt"], st["hs"], BD,
              **{**kw, "sw_dynamics": np.bool_(True)})


@pytest.mark.parametrize("bad", ["delp", "pt", "hs"])
def test_geopk_jax_rejects_float32_every_operand(bad):
    st = _synthetic(2)
    args = {k: np.asarray(st[k], dtype=np.float64)
            for k in ("delp", "pt", "hs")}
    args[bad] = np.asarray(args[bad], dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        geopk(args["delp"], args["pt"], args["hs"], BD,
              **_geopk_kw(2, cg=True, computehalo=False))


# ---------------------------------------------------------------- gate 4
def _small_geopk_fixture(km=3, n=4, ng=3, seed=5):
    """A small tile for the gradient checks (geopk needs no gridstruct,
    so the grad fixture does not have to be the C12 one)."""
    bd = Bounds.single_tile(n, ng)
    m = n + 2 * ng
    rng = np.random.default_rng(seed)
    shape = 1.0 + 0.05 * rng.standard_normal((m, m))
    frac = np.array([0.2, 0.45, 0.35])[:km]
    frac = frac / frac.sum()
    delp = np.stack([f * (1.0e5 - PTOP) * shape for f in frac], axis=-1)
    pt = np.stack([(300.0 - 10.0 * (k + 1)) * shape for k in range(km)],
                  axis=-1)
    hs = 1.5e4 * shape
    return bd, delp, pt, hs


def test_geopk_jax_check_grads_order2():
    """Order-2 fwd+rev grads.  geopk is smooth algebra (a running sum,
    ``log``, ``exp``, one division) with NO limiter and NO select.  Its
    only singular sites are ``log(p1d)``/``exp`` at ``p1d <= 0`` and the
    pkz division by an ``akap * peln`` difference at a zero-thickness
    layer; the control below shows both are far from the fixture."""
    km = 3
    bd, delp, pt, hs = _small_geopk_fixture(km=km)
    npx = npy = bd.ie + 1
    kw = _geopk_kw(km, cg=False, computehalo=True, fill=0.0, npx=npx,
                   npy=npy)

    base = geopk(delp, pt, hs, bd, **kw)
    # SINGULARITY CONTROL (the analogue of the p_fac floor-inactivity
    # control in fv3_nh_core): pressures strictly positive, and the pkz
    # denominator bounded away from zero with an FD-safe margin.
    assert float(np.asarray(base["pe"]).min()) >= PTOP
    dpeln = np.diff(np.asarray(base["peln"]), axis=1)
    assert float(np.abs(dpeln).min()) > 1e-3, float(np.abs(dpeln).min())

    box = (slice(bd.is_ - bd.isd - 2, bd.ie - bd.isd + 3),) * 2

    # Inputs AND outputs normalised to O(1) (see _GRAD_MIN_SENSITIVITY):
    # every derivative below is then a relative sensitivity in the
    # 1e-2..1 band, comfortably above the 1e-5 vacuity floor, and the
    # default 1e-4 absolute step becomes a 1e-4 relative one.
    s_dp, s_pt, s_hs = 1.0e4, 300.0, 1.0e4
    s_pk, s_gz, s_pe, s_pl, s_kz = 30.0, 1.0e6, 1.0e5, 12.0, 30.0

    def f(delp_h, pt_h, hs_h):
        out = geopk(delp_h * s_dp, pt_h * s_pt, hs_h * s_hs, bd, **kw)
        return (out["pk"][box] / s_pk, out["gz"][box] / s_gz,
                out["pe"] / s_pe, out["peln"] / s_pl,
                out["pkz"] / s_kz)

    check_grads(f, (jnp.asarray(delp / s_dp), jnp.asarray(pt / s_pt),
                    jnp.asarray(hs / s_hs)),
                order=2, modes=("fwd", "rev"))


def test_geopk_jax_adjoint_consistency():
    """FD-free adjoint identity for the whole geopk chain, including both
    ``lax.scan`` recurrences (the top-down p1d accumulator and the
    bottom-up gz integral) and the pkz division."""
    km = 3
    bd, delp, pt, hs = _small_geopk_fixture(km=km)
    npx = npy = bd.ie + 1
    kw = _geopk_kw(km, cg=False, computehalo=True, fill=0.0, npx=npx,
                   npy=npy)
    box = (slice(bd.is_ - bd.isd - 2, bd.ie - bd.isd + 3),) * 2

    def f(delp_, pt_, hs_):
        o = geopk(delp_, pt_, hs_, bd, **kw)
        return (o["pk"][box], o["gz"][box], o["pe"], o["peln"],
                o["pkz"])

    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint("geopk", f, (delp, pt, hs), 1e-12)


# ---------------------------------------------------------------- gate 5
def _poison_ring(a, width=2):
    """NaN in the OUTERMOST ``width`` rows/cols of the (i, j) axes."""
    out = np.array(a, dtype=np.float64, copy=True)
    out[:width, ...] = np.nan
    out[-width:, ...] = np.nan
    out[:, :width, ...] = np.nan
    out[:, -width:, ...] = np.nan
    return out


def test_geopk_jax_nan_poison_stays_out_of_primal_and_vjp():
    """The CG site reads delp/pt/hs on i,j = is-1..ie+1 only, so the
    outer TWO storage rings are dead.  Poisoning them must leave both
    the written window and the reverse-mode gradient finite; a one-index
    window slip pulls the NaN in (primal) or multiplies a zero cotangent
    by it (VJP only).  This is the local form of the "a select evaluates
    both operands" hazard -- this module has no ``jnp.where``, but it
    does carry the NumPy twin's NaN scratch."""
    km = 2
    st = _synthetic(km)
    kw = _geopk_kw(km, cg=True, computehalo=False, fill=0.0)
    dp = _poison_ring(st["delp"])
    ptp = _poison_ring(st["pt"])
    hsp = _poison_ring(st["hs"])

    out = geopk(dp, ptp, hsp, BD, **kw)
    box = (slice(NG - 1, NG + N + 1),) * 2
    for name in ("pk", "gz"):
        _finite(f"geopk poison {name}", np.asarray(out[name])[box])
    for name in ("pe", "peln"):
        _finite(f"geopk poison {name}", out[name])
    # Non-vacuity: the poison really is present in the operands.
    assert np.isnan(dp).any()

    def loss(d):
        o = geopk(d, jnp.nan_to_num(ptp), jnp.nan_to_num(hsp), BD, **kw)
        return jnp.sum(o["pk"][box] ** 2) + jnp.sum(o["pe"] ** 2)

    g = jax.grad(loss)(jnp.asarray(dp))
    _finite("geopk poison d(loss)/d(delp) inside the read box",
            np.asarray(g)[box])


# =====================================================================
# p_grad_c
# =====================================================================

def _pgc_fixture(km=2, seed=17):
    st = _synthetic(km)
    rng = np.random.default_rng(seed)
    pkc = np.cumsum(
        np.abs(rng.standard_normal((M_A, M_A, km + 1))) + 1.0, axis=2)
    gz = np.cumsum(
        np.abs(rng.standard_normal((M_A, M_A, km + 1))) + 5.0,
        axis=2)[:, :, ::-1].copy()
    return st, pkc, gz


# ---------------------------------------------------------------- gate 1
@pytest.mark.parametrize("hydrostatic", [True, False])
def test_p_grad_c_jax_matches_numpy_lane(hydrostatic):
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()
    delpc = pkc[:, :, 1:] - pkc[:, :, :-1] if not hydrostatic else st["delp"]

    uc_n, vc_n = np.array(st["uc"]), np.array(st["vc"])
    npg.p_grad_c(1.0, delpc, pkc, gz, uc_n, vc_n, gs, BD, npz=km,
                hydrostatic=hydrostatic)
    uc_j, vc_j = p_grad_c(1.0, delpc, pkc, gz, st["uc"], st["vc"], gs, BD,
                          npz=km, hydrostatic=hydrostatic)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp("p_grad_c uc", uc_j, uc_n, 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp("p_grad_c vc", vc_j, vc_n, 1e-12)
    # Non-vacuity: the update moved the winds.
    assert float(np.abs(np.asarray(uc_j) - st["uc"]).max()) > 0.0
    # Halo slots outside the write window are carried through EXACTLY.
    assert np.array_equal(np.asarray(uc_j)[0, :, :], st["uc"][0, :, :])


# ---------------------------------------------------------------- gate 2
def test_p_grad_c_jax_jit_eager_parity_and_no_retrace():
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()
    args = (1.0, st["delp"], pkc, gz, st["uc"], st["vc"], gs, BD)
    eager = p_grad_c(*args, npz=km, hydrostatic=True)
    jitted = p_grad_c_jit(*args, npz=km, hydrostatic=True)
    for name, e, j in zip(("uc", "vc"), eager, jitted):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"p_grad_c jit/eager {name}", j, e, 1e-12)

    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return p_grad_c(*a, **k)

    fn = make_p_grad_c_jit(_counted)
    fn(*args, npz=km, hydrostatic=True)
    fn(1.0, st["delp"], 1.01 * pkc, gz, st["uc"], st["vc"], gs, BD,
       npz=km, hydrostatic=True)
    assert traces["n"] == 1, traces["n"]


# ---------------------------------------------------------------- gate 3
def test_p_grad_c_jax_guards_raise():
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()
    # hydrostatic/non-hydrostatic dispatch on a NON-bool value raises
    # instead of silently taking the truthy arm.
    for bad in (1, "true", np.bool_(True), None):
        with pytest.raises(ValueError, match="not a Python bool"):
            p_grad_c(1.0, st["delp"], pkc, gz, st["uc"], st["vc"], gs, BD,
                     npz=km, hydrostatic=bad)


@pytest.mark.parametrize("bad", ["pkc", "gz", "uc", "vc"])
def test_p_grad_c_jax_rejects_float32_every_operand(bad):
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()
    ops = {"pkc": pkc, "gz": gz, "uc": st["uc"], "vc": st["vc"]}
    ops[bad] = np.asarray(ops[bad], dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        p_grad_c(1.0, st["delp"], ops["pkc"], ops["gz"], ops["uc"],
                 ops["vc"], gs, BD, npz=km, hydrostatic=True)


def test_p_grad_c_jax_rejects_float32_delpc_on_the_nh_branch():
    """The NH branch is the ONLY one that reads delpc, so the gate must
    fire there and must NOT fire on the hydrostatic branch (where the
    NumPy twin deletes the operand)."""
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()
    dp32 = np.asarray(pkc[:, :, 1:] - pkc[:, :, :-1], dtype=np.float32)
    with pytest.raises(TypeError, match="delpc"):
        p_grad_c(1.0, dp32, pkc, gz, st["uc"], st["vc"], gs, BD, npz=km,
                 hydrostatic=False)
    p_grad_c(1.0, None, pkc, gz, st["uc"], st["vc"], gs, BD, npz=km,
             hydrostatic=True)


# ---------------------------------------------------------------- gate 4
def test_p_grad_c_jax_check_grads_order2():
    """p_grad_c is a rational function of its operands -- no select, no
    limiter.  Its ONLY singular site is the denominator
    ``wk(i-1,j) + wk(i,j)`` (:2118), which blows up where two adjacent
    interface-pressure differences sum to zero.  The control below pins
    that sum strictly positive on the fixture."""
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()

    wk = pkc[:, :, 1:] - pkc[:, :, :-1]
    w = wk[NG - 1:NG + N + 2, NG - 1:NG + N + 2, :]
    denom = np.abs(w[:-1, :, :] + w[1:, :, :])
    assert denom.min() > 1e-2, denom.min()   # FD-safe margin

    uw = (slice(NG, NG + N + 1), slice(NG, NG + N))
    vw = (slice(NG, NG + N), slice(NG, NG + N + 1))
    z_uc = np.zeros_like(st["uc"])
    z_vc = np.zeros_like(st["vc"])

    # The p_grad_c increment is ~1e-6 of the wind it is added to (rdxc ~
    # 1.2e-6 at C12), so differentiating the TOTAL wind would drown the
    # pkc/gz sensitivity in the exact d(uc)/d(uc) = 1 pass-through and
    # the check would pass on a broken bracket.  Run from ZERO winds so
    # the output IS the increment (the same instrument argument the
    # NumPy gate's linearity control makes), and scale it to O(1).
    inc = np.abs(np.asarray(p_grad_c(1.0, None, pkc, gz, z_uc, z_vc, gs,
                                     BD, npz=km,
                                     hydrostatic=True)[0])[uw]).max()
    assert inc > 0.0
    s_pk, s_gz = 3.0, 10.0

    def f_bracket(pkc_h, gz_h):
        uo, vo = p_grad_c(1.0, None, pkc_h * s_pk, gz_h * s_gz, z_uc,
                          z_vc, gs, BD, npz=km, hydrostatic=True)
        return uo[uw] / inc, vo[vw] / inc

    check_grads(f_bracket, (jnp.asarray(pkc / s_pk),
                            jnp.asarray(gz / s_gz)),
                order=2, modes=("fwd", "rev"))

    # The uc/vc pass-through is the other direction: exactly linear, and
    # non-vacuous only when the output is NOT scaled by 1/inc.
    def f_pass(uc_, vc_):
        uo, vo = p_grad_c(1.0, None, pkc, gz, uc_, vc_, gs, BD, npz=km,
                          hydrostatic=True)
        return uo[uw] / 10.0, vo[vw] / 10.0

    check_grads(f_pass, (jnp.asarray(st["uc"]), jnp.asarray(st["vc"])),
                order=2, modes=("fwd", "rev"))


def test_p_grad_c_jax_adjoint_consistency():
    """FD-free adjoint identity, split the same way as the FD gate: the
    increment is ~1e-6 of the wind, so a single call would let a defect
    in the pkc/gz block hide under the uc/vc pass-through's contribution
    to the inner product."""
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()
    uw = (slice(NG, NG + N + 1), slice(NG, NG + N))
    vw = (slice(NG, NG + N), slice(NG, NG + N + 1))
    z_uc = np.zeros_like(st["uc"])
    z_vc = np.zeros_like(st["vc"])

    def f_bracket(pkc_, gz_):
        uo, vo = p_grad_c(1.0, None, pkc_, gz_, z_uc, z_vc, gs, BD,
                          npz=km, hydrostatic=True)
        return uo[uw], vo[vw]

    def f_pass(uc_, vc_):
        uo, vo = p_grad_c(1.0, None, pkc, gz, uc_, vc_, gs, BD, npz=km,
                          hydrostatic=True)
        return uo[uw], vo[vw]

    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint("p_grad_c bracket", f_bracket, (pkc, gz), 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint("p_grad_c passthrough", f_pass,
                   (st["uc"], st["vc"]), 1e-12)


# ---------------------------------------------------------------- gate 5
def test_p_grad_c_jax_nan_poison_stays_out_of_primal_and_vjp():
    """p_grad_c reads pkc/gz on i,j = is-1..ie+1 only -> the outer TWO
    storage rings are dead."""
    km = 2
    st, pkc, gz = _pgc_fixture(km)
    gs = _gs_cached()
    pkp, gzp = _poison_ring(pkc), _poison_ring(gz)
    assert np.isnan(pkp).any()

    uc_j, vc_j = p_grad_c(1.0, None, pkp, gzp, st["uc"], st["vc"], gs, BD,
                          npz=km, hydrostatic=True)
    _finite("p_grad_c poison uc", np.asarray(uc_j)[NG:NG + N + 1,
                                                   NG:NG + N, :])
    _finite("p_grad_c poison vc", np.asarray(vc_j)[NG:NG + N,
                                                   NG:NG + N + 1, :])

    def loss(pk_):
        uo, vo = p_grad_c(1.0, None, pk_, gzp, st["uc"], st["vc"], gs, BD,
                          npz=km, hydrostatic=True)
        return (jnp.sum(uo[NG:NG + N + 1, NG:NG + N, :] ** 2)
                + jnp.sum(vo[NG:NG + N, NG:NG + N + 1, :] ** 2))

    g = np.asarray(jax.grad(loss)(jnp.asarray(pkp)))
    _finite("p_grad_c poison d(loss)/d(pkc) inside the read box",
            g[NG - 1:NG + N + 2, NG - 1:NG + N + 2, :])


# =====================================================================
# one_grad_p
# =====================================================================

def _ogp_kw(km, *, d_ext=0.0):
    return dict(npx=NPX, npy=NPY, npz=km, dt=30.0, ptop=PTOP, akap=AKAP,
                hydrostatic=True, a2b_ord=4, d_ext=d_ext, ng=NG,
                duogrid=True)


def _ogp_flags():
    return dict(bounded_domain=False, grid_type=0, sw_corner=True,
                se_corner=True, nw_corner=True, ne_corner=True)


def _ogp_fixture(km=2, seed=51):
    delp, pk, gz, _pp, u, v = _nh_pgrad_fields(km, seed=seed)
    rng = np.random.default_rng(seed + 1)
    divg2 = rng.standard_normal((N + 1, N + 1))
    return delp, pk, gz, u, v, divg2


def _smooth2d(shape, amp, phase=0.0):
    """1 + amp * (a full-domain cosine x sine) -- smooth at the grid
    scale (period = the whole side, 18 or 19 cells), so a2b_ord4's
    4-point stencils reproduce it to O(h^4 * curvature)."""
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                         indexing="ij")
    return 1.0 + amp * (np.cos(2.0 * np.pi * (ii + phase) / shape[0])
                        * np.sin(2.0 * np.pi * (jj + 0.5) / shape[1]))


def _wellcond_fields(km):
    """A SMOOTH, physically-shaped column for the order-2 FD gates.

    Every 3-D field is ``(level profile) * (1 +- 2% smooth in i, j)``, so
    a2b's B-grid interface difference stays within 2% of the level
    increment -- see the rationale in
    :func:`test_one_grad_p_jax_check_grads_order2_well_conditioned`.
    The level increments (2.5, 3.5, 4.5) VARY with k on purpose: a
    constant increment would make the momentum weight ``wk`` independent
    of k and a k-shift in it invisible.
    """
    sa = _smooth2d((M_A, M_A), 0.02)
    p_lev = np.array([2.0 + 2.0 * k + 0.5 * k * k for k in range(km + 1)])
    g_lev = np.array([2.0e4 * (km - k) + 1.0e3 for k in range(km + 1)])
    su = _smooth2d((M_A, M_B), 0.3)
    sv = _smooth2d((M_B, M_A), 0.3, phase=0.5)
    return {
        "pk": np.stack([p * sa for p in p_lev], axis=-1),
        "gz": np.stack([g * sa for g in g_lev], axis=-1),
        "pp": np.stack([30.0 * (k + 1) * sa for k in range(km + 1)],
                       axis=-1),
        "delp": np.stack([1.0e4 * (1.0 + 0.1 * k) * sa
                          for k in range(km)], axis=-1),
        "u": np.stack([10.0 * (1.0 + 0.1 * k) * su for k in range(km)],
                      axis=-1),
        "v": np.stack([8.0 * (1.0 - 0.05 * k) * sv for k in range(km)],
                      axis=-1),
        "divg2": _smooth2d((N + 1, N + 1), 0.5),
    }


# ---------------------------------------------------------------- gate 1
@pytest.mark.parametrize("km", [1, 2, 3])
@pytest.mark.parametrize("d_ext", [0.0, 0.02])
def test_one_grad_p_jax_matches_numpy_lane(km, d_ext):
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = _ogp_kw(km, d_ext=d_ext)

    u_n, v_n = np.array(u), np.array(v)
    pk_n, gz_n = np.array(pk), np.array(gz)
    npg.one_grad_p(u_n, v_n, pk_n, gz_n, divg2, None, gs, BD, **kw)

    u_j, v_j, pk_j, gz_j = one_grad_p(u, v, pk, gz, divg2, None, gs, BD,
                                      **kw, **_ogp_flags())
    for name, j, n in (("u", u_j, u_n), ("v", v_j, v_n),
                       ("pk", pk_j, pk_n), ("gz", gz_j, gz_n)):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"one_grad_p[km={km},d_ext={d_ext}].{name}", j, n, 1e-12)
    # Non-vacuity: a2b replace really rewrote the B box, and the winds moved.
    assert not np.array_equal(np.asarray(pk_j), np.asarray(pk))
    assert float(np.abs(np.asarray(u_j) - u).max()) > 0.0


# ---------------------------------------------------------------- gate 2
def test_one_grad_p_jax_jit_eager_parity_and_no_retrace():
    km = 2
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = {**_ogp_kw(km, d_ext=0.02), **_ogp_flags()}
    eager = one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **kw)
    jitted = one_grad_p_jit(u, v, pk, gz, divg2, None, gs, BD, **kw)
    for name, e, j in zip(("u", "v", "pk", "gz"), eager, jitted):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"one_grad_p jit/eager {name}", j, e, 1e-12)

    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return one_grad_p(*a, **k)

    fn = make_one_grad_p_jit(_counted)
    fn(u, v, pk, gz, divg2, None, gs, BD, **kw)
    fn(1.01 * u, v, pk, gz, divg2, None, gs, BD, **kw)
    assert traces["n"] == 1, traces["n"]


# ---------------------------------------------------------------- gate 3
def test_one_grad_p_jax_guards_raise():
    km = 2
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = {**_ogp_kw(km), **_ogp_flags()}
    with pytest.raises(NotImplementedError, match="a2b_ord"):
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD,
                   **{**kw, "a2b_ord": 2})
    with pytest.raises(NotImplementedError, match="non-hydrostatic"):
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD,
                   **{**kw, "hydrostatic": False})
    with pytest.raises(ValueError, match="not a Python bool"):
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD,
                   **{**kw, "hydrostatic": 1})
    # a2b's dummy REBASES qin to (is-ng, js-ng): a bounds/ng mismatch is
    # a silent shifted read on both lanes, so it must raise here.
    with pytest.raises(ValueError, match="isd must equal"):
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **{**kw, "ng": 2})


@pytest.mark.parametrize("bad", ["u", "v", "pk", "gz", "divg2"])
def test_one_grad_p_jax_rejects_float32_every_operand(bad):
    km = 2
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    ops = {"u": u, "v": v, "pk": pk, "gz": gz, "divg2": divg2}
    ops[bad] = np.asarray(ops[bad], dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        one_grad_p(ops["u"], ops["v"], ops["pk"], ops["gz"], ops["divg2"],
                   None, gs, BD, **_ogp_kw(km), **_ogp_flags())


UW = (slice(NG, NG + N), slice(NG, NG + N + 1))       # u write window
VW = (slice(NG, NG + N + 1), slice(NG, NG + N))       # v write window
BW = (slice(NG, NG + N + 1), slice(NG, NG + N + 1))   # the B box


def _ogp_groups(fields, gs, kw, scales):
    """The two operand groups of one_grad_p, as closures.

    Group A = pk/gz, which ARE the momentum bracket.  Group B = u, v and
    divg2, which enter it LINEARLY and are then multiplied by
    rdx ~ 1.2e-6, so their sensitivity is ~1e-5 of group A's; a shared
    call would compare them against atol and pass vacuously, so group B
    gets its own call with the wind outputs scaled by 1e6 instead.
    """
    pk, gz, u, v, divg2 = fields
    s_pk, s_gz, s_u = scales

    def f_pg(pk_h, gz_h):
        uo, vo, pko, gzo = one_grad_p(u, v, pk_h * s_pk, gz_h * s_gz,
                                      divg2, None, gs, BD, **kw)
        return (uo[UW] / s_u, vo[VW] / s_u, pko[BW] / s_pk,
                gzo[BW] / s_gz)

    def f_lin(u_, v_, divg2_):
        uo, vo, _, _ = one_grad_p(u_, v_, pk, gz, divg2_, None, gs, BD,
                                  **kw)
        return uo[UW] * 1.0e6, vo[VW] * 1.0e6

    return f_pg, f_lin


# ---------------------------------------------------------------- gate 4
def test_one_grad_p_jax_check_grads_order1():
    """FIRST-ORDER FD on the harsh (random) fixture -- and the
    DISCRIMINATOR for job 9400424's order-2 failure.

    That run raised on ``JVP of JVP tangent``.  ``_check_grads``
    (jax `_src/public_test_util.py`) runs the order-1 forward check
    FIRST and only recurses into ``_check_grads(jvp(f), ..., order-1)``
    if it passes, and the recursion is what prefixes the message with
    ``JVP of``.  So order-1 forward was already green there and the
    defect was in the second tangent, NOT in the Jacobian -- but the
    order-1 REVERSE check is never reached once the forward recursion
    raises, so it was untested.  This test makes both mechanical.

    The remaining reverse-mode question -- whether the ``replace=True``
    read-after-write is threaded consistently between the forward pass
    and the adjoint -- is settled tolerance-free by
    :func:`test_one_grad_p_jax_adjoint_consistency`.
    """
    km = 2
    _delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = {**_ogp_kw(km, d_ext=0.02), **_ogp_flags()}
    s_pk, s_gz = 3.0, 2.0e3
    s_u = np.abs(np.asarray(
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **kw)[0])[UW]).max()
    assert s_u > 0.0

    f_pg, f_lin = _ogp_groups((pk, gz, u, v, divg2), gs, kw,
                              (s_pk, s_gz, s_u))
    check_grads(f_pg, (jnp.asarray(pk / s_pk), jnp.asarray(gz / s_gz)),
                order=1, modes=("fwd", "rev"))
    check_grads(f_lin, (jnp.asarray(u), jnp.asarray(v),
                        jnp.asarray(divg2)), order=1,
                modes=("fwd", "rev"))


def test_one_grad_p_jax_check_grads_order2_well_conditioned():
    """SECOND-ORDER FD, on a fixture whose momentum denominator is
    conditioned by CONSTRUCTION.

    Why not the random fixture: job 9400424 measured a second tangent of
    3.472e5 at element [8, 1, 0] of a 312-element (12 x 13 x 2) wind
    leaf whose neighbours print at ~1e2 and ~1e-2 -- a ~3.5e3 spread on
    an output normalised to O(1).  The only nonlinearity here is
    ``1 / (wk(i,j) + wk(i+1,j))`` (:2466), whose second derivative goes
    as that denominator^-3, so a 3.5e3 spread implies (PLAUSIBLE, not
    measured) a denominator ~15x below the typical one at that cell.
    The old control -- ``den.min() > 1e-3`` -- could not see that: it
    admitted a denominator three orders below the typical value.

    The fix is the fixture, not the bound.  ``a2b_ord4``'s stencils are
    symmetric 4-point with weights summing to 1 (B1+B2 = 1/2 twice,
    A1+A2 = 1/2 twice), so on a field that is smooth at the grid scale
    the B-grid value is the corner value to O(h^4 * curvature).  Building
    ``pk`` as ``P_k * (1 + 0.02 * smooth(i, j))`` with strictly
    increasing ``P_k`` therefore pins every B-grid interface difference
    near ``P_(k+1) - P_k`` -- while still VARYING in (i, j), so a window
    slip in ``wk`` stays visible.  The random column does not have that
    property: ``cumsum(|N(0,1)| + 1)`` has O(1) curvature at the grid
    scale, and a2b's negative outer weights (-1/12, -1/16) can then
    drive a B-grid difference far below the A-grid minimum of 1.
    """
    km = 3
    f = _wellcond_fields(km)
    gs = _gs_cached()
    kw = {**_ogp_kw(km, d_ext=0.02), **_ogp_flags()}
    pk, gz, u, v, divg2 = (f["pk"], f["gz"], f["u"], f["v"], f["divg2"])

    _, _, pk_b, _ = one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **kw)
    b = np.asarray(pk_b)[BW]
    wk = b[:, :, 1:] - b[:, :, :-1]
    den_u = np.abs(wk[:-1, :, :] + wk[1:, :, :])
    den_v = np.abs(wk[:, :-1, :] + wk[:, 1:, :])
    # Conditioning control, stated as a RATIO (the absolute-floor form is
    # what failed to catch the 9400424 outlier).  By construction the
    # level increments are 2.5 / 3.5 / 4.5 and the (i, j) modulation is
    # +-2%, so den = wk(i) + wk(i+1) spans ~4.9 to ~9.2.
    for nm, d in (("u", den_u), ("v", den_v)):
        assert d.min() > 1.0, (nm, d.min())
        assert d.min() / d.max() > 0.3, (nm, d.min(), d.max())
        # Non-vacuity of the control itself: wk must actually vary, or a
        # window slip in it would be invisible.
        assert d.max() - d.min() > 1e-6, (nm, d.min(), d.max())

    s_pk, s_gz = 10.0, 2.0e4
    s_u = np.abs(np.asarray(
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **kw)[0])[UW]).max()
    assert s_u > 0.0
    f_pg, f_lin = _ogp_groups((pk, gz, u, v, divg2), gs, kw,
                              (s_pk, s_gz, s_u))
    check_grads(f_pg, (jnp.asarray(pk / s_pk), jnp.asarray(gz / s_gz)),
                order=2, modes=("fwd", "rev"))
    check_grads(f_lin, (jnp.asarray(u), jnp.asarray(v),
                        jnp.asarray(divg2)), order=2,
                modes=("fwd", "rev"))


def test_one_grad_p_jax_adjoint_consistency():
    """``<J v, w> == <v, J^T w>`` on the HARSH (random) fixture -- the
    gate the campaign strategy requires (S7 part 3).

    No finite differences, so this is the check that survives the
    dynamic range that defeated order-2 FD above, and it is the one that
    would catch a mis-threaded ``replace=True`` (a forward pass and an
    adjoint that disagree about which array a value came from differ at
    O(1) here, not at the noise floor).

    Run PER GROUP, with the same split as the FD gates: the identity is
    one SCALAR comparison, so folding a direction whose sensitivity is
    1e-5 of another's into the same call lets a defect in the weak one
    hide under the strong one's contribution to the inner product."""
    km = 2
    _delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = {**_ogp_kw(km, d_ext=0.02), **_ogp_flags()}
    s_pk, s_gz = 3.0, 2.0e3
    s_u = np.abs(np.asarray(
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **kw)[0])[UW]).max()
    f_pg, f_lin = _ogp_groups((pk, gz, u, v, divg2), gs, kw,
                              (s_pk, s_gz, s_u))
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint("one_grad_p pk/gz", f_pg, (pk / s_pk, gz / s_gz), 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint("one_grad_p u/v/divg2", f_lin, (u, v, divg2), 1e-12)


# ---------------------------------------------------------------- gate 5
def test_one_grad_p_jax_nan_poison_stays_out_of_primal_and_vjp():
    """a2b's duo arm reads qin on i,j = is-2..ie+2, so exactly the
    OUTERMOST storage ring of pk/gz is dead.  It also exercises the
    internal NaN scratch (a2b's qout, and the wk weight): a one-index
    slip in ANY of the ~20 windows pulls a NaN into u/v."""
    km = 2
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = {**_ogp_kw(km, d_ext=0.02), **_ogp_flags()}
    pkp, gzp = _poison_ring(pk, width=1), _poison_ring(gz, width=1)
    assert np.isnan(pkp).any()

    u_j, v_j, pk_j, gz_j = one_grad_p(u, v, pkp, gzp, divg2, None, gs, BD,
                                      **kw)
    _finite("one_grad_p poison u",
            np.asarray(u_j)[NG:NG + N, NG:NG + N + 1, :])
    _finite("one_grad_p poison v",
            np.asarray(v_j)[NG:NG + N + 1, NG:NG + N, :])
    _finite("one_grad_p poison pk B box",
            np.asarray(pk_j)[NG:NG + N + 1, NG:NG + N + 1, :])

    def loss(pk_):
        uo, vo, _, _ = one_grad_p(u, v, pk_, gzp, divg2, None, gs, BD,
                                  **kw)
        return (jnp.sum(uo[NG:NG + N, NG:NG + N + 1, :] ** 2)
                + jnp.sum(vo[NG:NG + N + 1, NG:NG + N, :] ** 2))

    g = np.asarray(jax.grad(loss)(jnp.asarray(pkp)))
    _finite("one_grad_p poison d(loss)/d(pk) inside the read box",
            g[1:M_A - 1, 1:M_A - 1, :])


# =====================================================================
# nh_p_grad
# =====================================================================

def _nhpg_kw(km, *, use_logp=False):
    return dict(npx=NPX, npy=NPY, npz=km, dt=30.0, ptop=PTOP, akap=AKAP,
                use_logp=use_logp, ng=NG, duogrid=True)


# ---------------------------------------------------------------- gate 1
@pytest.mark.parametrize("km", [1, 2, 3])
@pytest.mark.parametrize("use_logp", [False, True])
def test_nh_p_grad_jax_matches_numpy_lane(km, use_logp):
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    kw = _nhpg_kw(km, use_logp=use_logp)

    u_n, v_n = np.array(u), np.array(v)
    pp_n, pk_n, gz_n = np.array(pp), np.array(pk3), np.array(gz)
    npg.nh_p_grad(u_n, v_n, pp_n, gz_n, np.array(delp), pk_n, gs, BD, **kw)

    u_j, v_j, pp_j, pk_j, gz_j = nh_p_grad(u, v, pp, gz, delp, pk3, gs,
                                           BD, **kw, **_ogp_flags())
    for name, j, n in (("u", u_j, u_n), ("v", v_j, v_n),
                       ("pp", pp_j, pp_n), ("pk3", pk_j, pk_n),
                       ("gz", gz_j, gz_n)):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"nh_p_grad[km={km},logp={use_logp}].{name}", j, n, 1e-12)
    assert float(np.abs(np.asarray(u_j) - u).max()) > 0.0


def test_nh_p_grad_jax_leaves_delp_unmodified():
    """:2191 calls a2b WITHOUT `replace`, so delp must come back
    untouched -- a tolerance-independent contract."""
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    dp_in = np.array(delp, copy=True)
    nh_p_grad(u, v, pp, gz, dp_in, pk3, gs, BD, **_nhpg_kw(km),
              **_ogp_flags())
    assert np.array_equal(dp_in, delp)


# ---------------------------------------------------------------- gate 2
def test_nh_p_grad_jax_jit_eager_parity_and_no_retrace():
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    kw = {**_nhpg_kw(km), **_ogp_flags()}
    eager = nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD, **kw)
    jitted = nh_p_grad_jit(u, v, pp, gz, delp, pk3, gs, BD, **kw)
    for name, e, j in zip(("u", "v", "pp", "pk3", "gz"), eager, jitted):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"nh_p_grad jit/eager {name}", j, e, 1e-12)

    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return nh_p_grad(*a, **k)

    fn = make_nh_p_grad_jit(_counted)
    fn(u, v, pp, gz, delp, pk3, gs, BD, **kw)
    fn(u, v, 1.01 * pp, gz, delp, pk3, gs, BD, **kw)
    assert traces["n"] == 1, traces["n"]


# ---------------------------------------------------------------- gate 3
def test_nh_p_grad_jax_guards_raise():
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    kw = {**_nhpg_kw(km), **_ogp_flags()}
    with pytest.raises(ValueError, match="not a Python bool"):
        nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD,
                  **{**kw, "use_logp": 1})
    with pytest.raises(ValueError, match="isd must equal"):
        nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD, **{**kw, "ng": 2})
    with pytest.raises(ValueError, match="not a Python bool"):
        nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD,
                  **{**kw, "duogrid": np.bool_(True)})


@pytest.mark.parametrize("bad", ["u", "v", "pp", "gz", "delp", "pk3"])
def test_nh_p_grad_jax_rejects_float32_every_operand(bad):
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    ops = {"u": u, "v": v, "pp": pp, "gz": gz, "delp": delp, "pk3": pk3}
    ops[bad] = np.asarray(ops[bad], dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        nh_p_grad(ops["u"], ops["v"], ops["pp"], ops["gz"], ops["delp"],
                  ops["pk3"], gs, BD, **_nhpg_kw(km), **_ogp_flags())


def _nhpg_groups(fields, gs, kw, scales):
    """nh_p_grad's two operand groups (same split rationale as
    :func:`_ogp_groups`).

    Group A = pk3/gz, the dominant hydrostatic ``du1`` direction.
    Group B = pp (the NH term, ~0.3% of the bracket), the linear u/v
    pass-through, and delp -- which enters ONLY through the NH weight
    ``wk1 = a2b(delp)``, i.e. as ``1/delp``.  All three are ~1e-5 of
    group A's sensitivity, so their call scales the wind outputs by 1e6;
    ``ppo`` carries pp's own O(1) direction.
    """
    pk3, gz, pp, delp, u, v = fields
    s_pk, s_gz, s_pp, s_dp, s_u = scales

    def f_pg(pk3_h, gz_h):
        uo, vo, _, pko, gzo = nh_p_grad(u, v, pp, gz_h * s_gz, delp,
                                        pk3_h * s_pk, gs, BD, **kw)
        return (uo[UW] / s_u, vo[VW] / s_u, pko[BW] / s_pk,
                gzo[BW] / s_gz)

    def f_nh(u_, v_, pp_h, delp_h):
        uo, vo, ppo, _, _ = nh_p_grad(u_, v_, pp_h * s_pp, gz,
                                      delp_h * s_dp, pk3, gs, BD, **kw)
        return uo[UW] * 1.0e6, vo[VW] * 1.0e6, ppo[BW] / s_pp

    return f_pg, f_nh


# ---------------------------------------------------------------- gate 4
def test_nh_p_grad_jax_check_grads_order1():
    """FIRST-ORDER FD on the harsh (random) fixture.  Same discriminator
    argument as :func:`test_one_grad_p_jax_check_grads_order1`: job
    9400424 raised on ``JVP of JVP tangent`` here too, which is only
    reachable after the order-1 forward check passes."""
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    kw = {**_nhpg_kw(km), **_ogp_flags()}
    s = (3.0, 2.0e3, 30.0, 1.0e4,
         np.abs(np.asarray(nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD,
                                     **kw)[0])[UW]).max())
    assert s[4] > 0.0
    f_pg, f_nh = _nhpg_groups((pk3, gz, pp, delp, u, v), gs, kw, s)
    check_grads(f_pg, (jnp.asarray(pk3 / s[0]), jnp.asarray(gz / s[1])),
                order=1, modes=("fwd", "rev"))
    check_grads(f_nh, (jnp.asarray(u), jnp.asarray(v),
                       jnp.asarray(pp / s[2]), jnp.asarray(delp / s[3])),
                order=1, modes=("fwd", "rev"))


def test_nh_p_grad_jax_check_grads_order2_well_conditioned():
    """SECOND-ORDER FD on the smooth fixture -- see the full rationale in
    :func:`test_one_grad_p_jax_check_grads_order2_well_conditioned`.

    Two singular sites here, both denominators (:2201 / :2218): the
    hydrostatic weight ``wk`` (B-grid pk3 differences) and the NH weight
    ``wk1`` (B-grid delp).  Both get a ratio control below; the old
    absolute floor (``> 1e-3`` on a quantity whose typical value is
    ~7) is what let the 9400424 outlier through."""
    km = 3
    f = _wellcond_fields(km)
    gs = _gs_cached()
    kw = {**_nhpg_kw(km), **_ogp_flags()}
    pk3, gz, pp, delp, u, v = (f["pk"], f["gz"], f["pp"], f["delp"],
                               f["u"], f["v"])

    _, _, _, pk_b, _ = nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD, **kw)
    b = np.asarray(pk_b)[BW]
    wk = b[:, :, 1:] - b[:, :, :-1]
    for nm, d in (("u", np.abs(wk[:-1, :, :] + wk[1:, :, :])),
                  ("v", np.abs(wk[:, :-1, :] + wk[:, 1:, :]))):
        assert d.min() > 1.0, (nm, d.min())
        assert d.min() / d.max() > 0.3, (nm, d.min(), d.max())
        assert d.max() - d.min() > 1e-6, (nm, d.min(), d.max())
    # wk1 = a2b(delp): delp is ~1e4 +- 2% and strictly positive, so its
    # B-grid interpolant stays within 2% of 1e4.
    dp = np.asarray(delp)
    assert dp.min() > 1.0e3 and dp.min() / dp.max() > 0.3

    s = (10.0, 2.0e4, 30.0, 1.0e4,
         np.abs(np.asarray(nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD,
                                     **kw)[0])[UW]).max())
    assert s[4] > 0.0
    f_pg, f_nh = _nhpg_groups((pk3, gz, pp, delp, u, v), gs, kw, s)
    check_grads(f_pg, (jnp.asarray(pk3 / s[0]), jnp.asarray(gz / s[1])),
                order=2, modes=("fwd", "rev"))
    check_grads(f_nh, (jnp.asarray(u), jnp.asarray(v),
                       jnp.asarray(pp / s[2]), jnp.asarray(delp / s[3])),
                order=2, modes=("fwd", "rev"))


def test_nh_p_grad_jax_adjoint_consistency():
    """``<J v, w> == <v, J^T w>`` on the HARSH fixture -- FD-free, so it
    survives the dynamic range that defeated order-2 FD, and it is the
    gate that would catch a mis-threaded ``replace=True``.  Per group,
    for the same reason as :func:`test_one_grad_p_jax_adjoint_consistency`."""
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    kw = {**_nhpg_kw(km), **_ogp_flags()}
    s = (3.0, 2.0e3, 30.0, 1.0e4,
         np.abs(np.asarray(nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD,
                                     **kw)[0])[UW]).max())
    f_pg, f_nh = _nhpg_groups((pk3, gz, pp, delp, u, v), gs, kw, s)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint("nh_p_grad pk3/gz", f_pg, (pk3 / s[0], gz / s[1]),
                   1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _check_adjoint("nh_p_grad pp/u/v/delp", f_nh,
                   (u, v, pp / s[2], delp / s[3]), 1e-12)


# ---------------------------------------------------------------- gate 5
def test_nh_p_grad_jax_nan_poison_stays_out_of_primal_and_vjp():
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    kw = {**_nhpg_kw(km), **_ogp_flags()}
    ppp = _poison_ring(pp, width=1)
    gzp = _poison_ring(gz, width=1)
    pkp = _poison_ring(pk3, width=1)
    dpp = _poison_ring(delp, width=1)
    assert np.isnan(ppp).any()

    u_j, v_j, _, _, _ = nh_p_grad(u, v, ppp, gzp, dpp, pkp, gs, BD, **kw)
    _finite("nh_p_grad poison u",
            np.asarray(u_j)[NG:NG + N, NG:NG + N + 1, :])
    _finite("nh_p_grad poison v",
            np.asarray(v_j)[NG:NG + N + 1, NG:NG + N, :])

    def loss(pp_):
        uo, vo, _, _, _ = nh_p_grad(u, v, pp_, gzp, dpp, pkp, gs, BD,
                                    **kw)
        return (jnp.sum(uo[NG:NG + N, NG:NG + N + 1, :] ** 2)
                + jnp.sum(vo[NG:NG + N + 1, NG:NG + N, :] ** 2))

    g = np.asarray(jax.grad(loss)(jnp.asarray(ppp)))
    _finite("nh_p_grad poison d(loss)/d(pp) inside the read box",
            g[1:M_A - 1, 1:M_A - 1, :])


# =====================================================================
# pk3_halo / pln_halo / pe_halo
# =====================================================================

def _halo_fixture(km=3, seed=53):
    rng = np.random.default_rng(seed)
    delp = np.abs(1.0e4 + 200.0 * rng.standard_normal((M_A, M_A, km)))
    return delp


SENT = 4.4e30


# ---------------------------------------------------------------- gate 1
@pytest.mark.parametrize("km", [1, 3])
def test_pk3_and_pln_halo_jax_match_numpy_lane(km):
    delp = _halo_fixture(km)
    pk3_n = np.full((M_A, M_A, km + 1), SENT)
    npg.pk3_halo(pk3_n, delp, BD, npz=km, ptop=PTOP, akap=AKAP)
    pk3_j = pk3_halo(np.full((M_A, M_A, km + 1), SENT), delp, BD, npz=km,
                     ptop=PTOP, akap=AKAP)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"pk3_halo[km={km}]", pk3_j, pk3_n, 1e-12)

    pln_n = np.full((M_A, M_A, km + 1), SENT)
    npg.pln_halo(pln_n, delp, BD, npz=km, ptop=PTOP)
    pln_j = pln_halo(np.full((M_A, M_A, km + 1), SENT), delp, BD, npz=km,
                     ptop=PTOP)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"pln_halo[km={km}]", pln_j, pln_n, 1e-12)

    # Tolerance-independent footprint: the same slots stay sentinel on
    # both lanes, and level 1 is NEVER written.
    assert np.array_equal(np.asarray(pk3_j) == SENT, pk3_n == SENT)
    assert np.array_equal(np.asarray(pln_j) == SENT, pln_n == SENT)
    assert (np.asarray(pk3_j)[:, :, 0] == SENT).all()


@pytest.mark.parametrize("km", [1, 3])
def test_pe_halo_jax_matches_numpy_lane(km):
    delp = _halo_fixture(km)
    pe_n = np.full((N + 2, km + 1, N + 2), SENT)
    npg.pe_halo(pe_n, delp, BD, npz=km, ptop=PTOP)
    pe_j = pe_halo(np.full((N + 2, km + 1, N + 2), SENT), delp, BD,
                   npz=km, ptop=PTOP)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"pe_halo[km={km}]", pe_j, pe_n, 1e-12)
    assert np.array_equal(np.asarray(pe_j) == SENT, pe_n == SENT)
    # the whole interior keeps its sentinel (only the border ring runs)
    assert (np.asarray(pe_j)[1:N + 1, :, 1:N + 1] == SENT).all()


# ---------------------------------------------------------------- gate 2
@pytest.mark.parametrize(
    "name,fn,jfn,mk,extra",
    [("pk3_halo", pk3_halo, pk3_halo_jit, make_pk3_halo_jit,
      {"akap": AKAP}),
     ("pln_halo", pln_halo, pln_halo_jit, make_pln_halo_jit, {}),
     ("pe_halo", pe_halo, pe_halo_jit, make_pe_halo_jit, {})])
def test_halo_jax_jit_eager_parity_and_no_retrace(name, fn, jfn, mk,
                                                  extra):
    km = 3
    delp = _halo_fixture(km)
    base = (np.full((N + 2, km + 1, N + 2), SENT) if name == "pe_halo"
            else np.full((M_A, M_A, km + 1), SENT))
    kw = dict(npz=km, ptop=PTOP, **extra)
    eager = fn(base, delp, BD, **kw)
    jitted = jfn(base, delp, BD, **kw)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"{name} jit/eager", jitted, eager, 1e-12)

    traces = {"n": 0}

    def _counted(*a, **k):
        traces["n"] += 1
        return fn(*a, **k)

    j2 = mk(_counted)
    j2(base, delp, BD, **kw)
    j2(base, 1.01 * delp, BD, **kw)
    assert traces["n"] == 1, traces["n"]


# ---------------------------------------------------------------- gate 3
@pytest.mark.parametrize("bad", ["state", "delp"])
def test_halo_jax_rejects_float32_every_operand(bad):
    km = 2
    delp = _halo_fixture(km)
    pk3 = np.full((M_A, M_A, km + 1), SENT)
    pe = np.full((N + 2, km + 1, N + 2), SENT)
    if bad == "delp":
        delp = np.asarray(delp, dtype=np.float32)
    else:
        pk3 = np.asarray(pk3, dtype=np.float32)
        pe = np.asarray(pe, dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        pk3_halo(pk3, delp, BD, npz=km, ptop=PTOP, akap=AKAP)
    with pytest.raises(TypeError, match="float64"):
        pln_halo(pk3, delp, BD, npz=km, ptop=PTOP)
    with pytest.raises(TypeError, match="float64"):
        pe_halo(pe, delp, BD, npz=km, ptop=PTOP)


def test_halo_jax_reject_bad_npz():
    """``npz`` deeper than ``delp`` must RAISE, not silently integrate a
    clipped column: ``delp[..., 0:npz]`` clips without complaint, so the
    guard is the only thing between a short column and a plausible
    number."""
    km = 2
    delp = _halo_fixture(km)
    pk3 = np.full((M_A, M_A, km + 1), SENT)
    pe = np.full((N + 2, km + 1, N + 2), SENT)
    for fn, kw in ((pk3_halo, dict(npz=km + 2, ptop=PTOP, akap=AKAP)),
                   (pln_halo, dict(npz=km + 2, ptop=PTOP))):
        with pytest.raises(ValueError, match="silently clipped"):
            fn(pk3, delp, BD, **kw)
    with pytest.raises(ValueError, match="silently clipped"):
        pe_halo(pe, delp, BD, npz=km + 2, ptop=PTOP)


# ---------------------------------------------------------------- gate 4
def test_halo_jax_check_grads_order2():
    """The halo rebuilds are a running sum followed by ``log`` (pln) or
    ``exp(akap*log(.))`` (pk3); the ONLY singular site is ``p <= 0``, and
    delp here is ~1e4 and strictly positive, so ``pe >= ptop`` at every
    level (control below)."""
    km = 3
    delp = _halo_fixture(km)
    assert float(delp.min()) > 0.0
    pk3 = np.zeros((M_A, M_A, km + 1))
    pe = np.zeros((N + 2, km + 1, N + 2))

    # Normalised delp (see _GRAD_MIN_SENSITIVITY): d(pk3)/d(delp) is
    # ~7e-5 in raw units, which scaled by 1/30 would sit BELOW the 1e-5
    # vacuity floor.  With delp_hat = delp/1e4 the sensitivities land at
    # 2.5e-2 (pk3), 8e-3 (pln) and 1e-1 (pe).
    s_dp = 1.0e4

    def f_pk3(d_h):
        return pk3_halo(pk3, d_h * s_dp, BD, npz=km, ptop=PTOP,
                        akap=AKAP) / 30.0

    def f_pln(d_h):
        return pln_halo(pk3, d_h * s_dp, BD, npz=km, ptop=PTOP) / 12.0

    def f_pe(d_h):
        return pe_halo(pe, d_h * s_dp, BD, npz=km, ptop=PTOP) / 1.0e5

    for f in (f_pk3, f_pln, f_pe):
        check_grads(f, (jnp.asarray(delp / s_dp),), order=2,
                    modes=("fwd", "rev"))


def test_halo_jax_adjoint_consistency():
    """FD-free adjoint identity for the three ring rebuilds, including
    their ``lax.scan`` column integral."""
    km = 3
    delp = _halo_fixture(km)
    pk3 = np.zeros((M_A, M_A, km + 1))
    pe = np.zeros((N + 2, km + 1, N + 2))

    for name, f in (
            ("pk3_halo",
             lambda d: pk3_halo(pk3, d, BD, npz=km, ptop=PTOP, akap=AKAP)),
            ("pln_halo",
             lambda d: pln_halo(pk3, d, BD, npz=km, ptop=PTOP)),
            ("pe_halo",
             lambda d: pe_halo(pe, d, BD, npz=km, ptop=PTOP))):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _check_adjoint(name, f, (delp,), 1e-12)


# ---------------------------------------------------------------- gate 5
def test_halo_jax_nan_poison_stays_out_of_primal_and_vjp():
    """pk3_halo/pln_halo read i,j = is-2..ie+2 and pe_halo reads
    i,j = is-1..ie+1, so the outermost storage ring is dead for the first
    two and the outer two rings for pe_halo."""
    km = 3
    delp = _halo_fixture(km)
    dp1 = _poison_ring(delp, width=1)
    dp2 = _poison_ring(delp, width=2)
    assert np.isnan(dp1).any()
    ring = (slice(1, M_A - 1), slice(1, M_A - 1))

    got = pk3_halo(np.zeros((M_A, M_A, km + 1)), dp1, BD, npz=km,
                   ptop=PTOP, akap=AKAP)
    _finite("pk3_halo poison", np.asarray(got)[ring])
    got = pln_halo(np.zeros((M_A, M_A, km + 1)), dp1, BD, npz=km,
                   ptop=PTOP)
    _finite("pln_halo poison", np.asarray(got)[ring])
    got = pe_halo(np.zeros((N + 2, km + 1, N + 2)), dp2, BD, npz=km,
                  ptop=PTOP)
    _finite("pe_halo poison", got)

    def loss(d):
        return jnp.sum(pk3_halo(np.zeros((M_A, M_A, km + 1)), d, BD,
                                npz=km, ptop=PTOP, akap=AKAP)[ring] ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(dp1)))
    _finite("pk3_halo poison d(loss)/d(delp) inside the read box",
            g[ring])


# =====================================================================
# Fixture-backed chain gate: the committed C12 oracle inputs
# =====================================================================

def _oracle_fixture(km):
    """The COMMITTED geopk/pgrad oracle inputs (real cube geometry, the
    real duo settings, km = 2 and 3).  Reused from the NumPy gate rather
    than re-invented; skipped only if the npz is absent, because the
    synthetic gates above already cover the routines."""
    import os as _os

    path = _os.path.join(_os.path.dirname(__file__), "fixtures",
                         f"geopk_pgrad_oracle_c12_km{km}.npz")
    if not _os.path.exists(path):
        pytest.skip(f"{_os.path.basename(path)} not present")
    npz = np.load(path, allow_pickle=True)
    return {k[3:]: np.array(npz[k]) for k in npz.files
            if k.startswith("in_")}


@pytest.mark.parametrize("km", [2, 3])
def test_jax_chain_matches_numpy_lane_on_the_oracle_fixture(km):
    """geopk(C) -> p_grad_c -> geopk(D) -> one_grad_p, JAX vs NumPy, on
    the fixture's own serialized inputs.  Same call sequence as
    ``gen_geopk_pgrad_oracle.run_port`` (dyn_core.F90:533 / :629 / :1401
    / :1531); only the lane differs."""
    f = _oracle_fixture(km)
    gs = {k: f[k] for k in ("rdxc", "rdyc", "rdx", "rdy", "dxa", "dya",
                            "grid_lon", "grid_lat", "agrid_lon",
                            "agrid_lat", "edge_w", "edge_e", "edge_s",
                            "edge_n")}
    gs_np = {**gs, "bounded_domain": False, "grid_type": 0,
             "sw_corner": True, "se_corner": True, "nw_corner": True,
             "ne_corner": True}
    d_ext = 0.02
    kwc = _geopk_kw(km, cg=True, computehalo=False, fill=0.0)
    kwd = _geopk_kw(km, cg=False, computehalo=True, fill=0.0)

    # ---- C site
    gc_n = npg.geopk(f["delpc"], f["ptc"], f["hs"], BD, **kwc)
    gc_j = geopk(f["delpc"], f["ptc"], f["hs"], BD, **kwc)
    for name in ("pk", "gz", "pe", "peln", "pkz"):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"oracle km={km} geopk_c.{name}", gc_j[name], gc_n[name],
             1e-12)

    # ---- p_grad_c
    uc_n, vc_n = np.array(f["uc"]), np.array(f["vc"])
    npg.p_grad_c(1.0, f["delpc"], np.array(gc_n["pk"]),
                np.array(gc_n["gz"]), uc_n, vc_n, gs_np, BD, npz=km,
                hydrostatic=True)
    uc_j, vc_j = p_grad_c(1.0, None, gc_j["pk"], gc_j["gz"], f["uc"],
                          f["vc"], gs, BD, npz=km, hydrostatic=True)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"oracle km={km} uc_pgc", uc_j, uc_n, 1e-12)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"oracle km={km} vc_pgc", vc_j, vc_n, 1e-12)

    # ---- D site + one_grad_p
    gd_n = npg.geopk(f["delp"], f["pt"], f["hs"], BD, **kwd)
    gd_j = geopk(f["delp"], f["pt"], f["hs"], BD, **kwd)
    u_n, v_n = np.array(f["u"]), np.array(f["v"])
    npg.one_grad_p(u_n, v_n, np.array(gd_n["pk"]), np.array(gd_n["gz"]),
                  f["divg2"], None, gs_np, BD, npx=NPX, npy=NPY, npz=km,
                  dt=30.0, ptop=PTOP, akap=AKAP, hydrostatic=True,
                  a2b_ord=4, d_ext=d_ext, ng=NG, duogrid=True)
    u_j, v_j, pk_j, gz_j = one_grad_p(
        f["u"], f["v"], gd_j["pk"], gd_j["gz"], f["divg2"], None, gs, BD,
        **_ogp_kw(km, d_ext=d_ext), **_ogp_flags())
    for name, j, n in (("u_ogp", u_j, u_n), ("v_ogp", v_j, v_n)):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"oracle km={km} {name}", j, n, 1e-12)
    # Non-vacuity: the chain moved the D-grid winds.
    assert float(np.abs(np.asarray(u_j) - f["u"]).max()) > 0.0
