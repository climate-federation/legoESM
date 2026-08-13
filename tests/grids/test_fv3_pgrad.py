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

from legoesm.core.fv3_native_pgrad import (  # noqa: E402
    a2b_gridstruct_view as a2b_view_np,
    geopk as geopk_np,
    nh_p_grad as nh_p_grad_np,
    one_grad_p as one_grad_p_np,
    p_grad_c as p_grad_c_np,
    pe_halo as pe_halo_np,
    pk3_halo as pk3_halo_np,
    pln_halo as pln_halo_np,
)
from legoesm.core.fv3_native_sw_core import BIG_NUMBER, Bounds  # noqa: E402
from legoesm.core.fv3_pgrad import (  # noqa: E402
    _a2b_ord4,
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
    N,
    NG,
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


# check_grads' f64 gradient tolerance is atol = rtol = 1e-5 and its step
# `eps` is ABSOLUTE (jax `_src/public_test_util.py`: EPS = 1e-4,
# `default_gradient_tolerance[float64] = 1e-5`, central difference).  Two
# consequences drive every gradient gate below:
#   * a DIRECTIONAL DERIVATIVE smaller than ~1e-5 is compared against
#     atol and the check becomes VACUOUS -- so each call normalises its
#     inputs AND its outputs to O(1), which makes every derivative a
#     RELATIVE sensitivity, and operand groups whose sensitivity differs
#     by orders of magnitude get their own call rather than being
#     swamped in the shared VJP inner product;
#   * with O(1) inputs the default 1e-4 step is a 1e-4 RELATIVE
#     perturbation, the usual f64 central-difference choice.
_GRAD_MIN_SENSITIVITY = 1e-5


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
    gsf = a2b_view_np(gsv, BD)
    qi = np.array(qin, dtype=np.float64, copy=True)
    qo = np.full((M_A, M_A), np.nan)
    a2b_ord4(fort(qi, BD.isd, BD.jsd), fort(qo, BD.isd, BD.jsd), gsf,
             NPX, NPY, BD.is_, BD.ie, BD.js, BD.je, NG, replace=replace,
             duogrid=duogrid)
    return qi, qo


def _run_a2b_jax(qin, gs, *, replace, duogrid, grid_type=0,
                 bounded_domain=False):
    geom = a2b_gridstruct_view(gs, BD)
    qi, qo = _a2b_ord4(
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
        _a2b_ord4(q3, q3, geom, NPX, NPY, BD.is_, BD.ie, BD.js, BD.je, NG)
    q2 = jnp.zeros((M_A, M_A), jnp.float64)
    with pytest.raises(ValueError, match="ng=1"):
        _a2b_ord4(q2, q2, geom, NPX, NPY, BD.is_, BD.ie, BD.js, BD.je, 1)


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
    want = geopk_np(st["delp"], st["pt"], st["hs"], BD, **kw)
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
    want = geopk_np(st["delp"], st["pt"], st["hs"], BD, **kw)
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
    p_grad_c_np(1.0, delpc, pkc, gz, uc_n, vc_n, gs, BD, npz=km,
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


# ---------------------------------------------------------------- gate 1
@pytest.mark.parametrize("km", [1, 2, 3])
@pytest.mark.parametrize("d_ext", [0.0, 0.02])
def test_one_grad_p_jax_matches_numpy_lane(km, d_ext):
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = _ogp_kw(km, d_ext=d_ext)

    u_n, v_n = np.array(u), np.array(v)
    pk_n, gz_n = np.array(pk), np.array(gz)
    one_grad_p_np(u_n, v_n, pk_n, gz_n, divg2, None, gs, BD, **kw)

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


# ---------------------------------------------------------------- gate 4
def test_one_grad_p_jax_check_grads_order2():
    """one_grad_p is rational in its operands: a2b_ord4's duo arm is a
    fixed linear stencil and the momentum bracket is a product of
    differences over ``wk(i,j) + wk(i+1,j)`` (:2466).  That denominator
    is the ONLY singular site on this arm (the plain arm adds
    ``arcsin(sqrt(.))`` in ``_great_circle_dist``, non-smooth at
    coincident/antipodal points -- not exercised here, and named in the
    module docstring).  The control below pins the denominator away from
    zero with an FD-safe margin."""
    km = 2
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    gs = _gs_cached()
    kw = {**_ogp_kw(km, d_ext=0.02), **_ogp_flags()}

    _, _, pk_b, _ = one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **kw)
    b = np.asarray(pk_b)[NG:NG + N + 1, NG:NG + N + 1, :]
    wk = b[:, :, 1:] - b[:, :, :-1]
    den_u = np.abs(wk[:-1, :, :] + wk[1:, :, :])
    den_v = np.abs(wk[:, :-1, :] + wk[:, 1:, :])
    assert den_u.min() > 1e-3, den_u.min()
    assert den_v.min() > 1e-3, den_v.min()

    uw = (slice(NG, NG + N), slice(NG, NG + N + 1))
    vw = (slice(NG, NG + N + 1), slice(NG, NG + N))
    bw = (slice(NG, NG + N + 1), slice(NG, NG + N + 1))
    s_pk, s_gz = 3.0, 2.0e3
    u_out = np.abs(np.asarray(
        one_grad_p(u, v, pk, gz, divg2, None, gs, BD, **kw)[0])[uw]).max()
    assert u_out > 0.0

    # Group A: the DOMINANT directions (the whole momentum bracket is
    # the pk/gz term -- rdx ~ 1.2e-6 times ~3e4).
    def f_pg(pk_h, gz_h):
        uo, vo, pko, gzo = one_grad_p(u, v, pk_h * s_pk, gz_h * s_gz,
                                      divg2, None, gs, BD, **kw)
        return (uo[uw] / u_out, vo[vw] / u_out, pko[bw] / s_pk,
                gzo[bw] / s_gz)

    check_grads(f_pg, (jnp.asarray(pk / s_pk), jnp.asarray(gz / s_gz)),
                order=2, modes=("fwd", "rev"))

    # Group B: u/v/divg2 enter the bracket LINEARLY and are multiplied by
    # rdx ~ 1.2e-6, so their sensitivity is ~1e-5 of group A's -- a
    # shared call would compare them against atol and pass vacuously.
    # Their own call scales the output by 1e6 instead, putting the
    # derivative at ~1.2.
    def f_lin(u_, v_, divg2_):
        uo, vo, _, _ = one_grad_p(u_, v_, pk, gz, divg2_, None, gs, BD,
                                  **kw)
        return uo[uw] * 1.0e6, vo[vw] * 1.0e6

    check_grads(f_lin, (jnp.asarray(u), jnp.asarray(v),
                        jnp.asarray(divg2)), order=2,
                modes=("fwd", "rev"))


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
    nh_p_grad_np(u_n, v_n, pp_n, gz_n, np.array(delp), pk_n, gs, BD, **kw)

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


# ---------------------------------------------------------------- gate 4
def test_nh_p_grad_jax_check_grads_order2():
    """Two singular sites, both denominators (:2201 / :2218): the
    hydrostatic weight ``wk`` (B-grid pk3 differences) and the NH weight
    ``wk1`` (B-grid delp).  Both controls below are FD-safe margins."""
    km = 2
    delp, pk3, gz, pp, u, v = _nh_pgrad_fields(km)
    gs = _gs_cached()
    kw = {**_nhpg_kw(km), **_ogp_flags()}

    _, _, _, pk_b, _ = nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD, **kw)
    b = np.asarray(pk_b)[NG:NG + N + 1, NG:NG + N + 1, :]
    wk = b[:, :, 1:] - b[:, :, :-1]
    assert np.abs(wk[:-1, :, :] + wk[1:, :, :]).min() > 1e-3
    assert np.abs(wk[:, :-1, :] + wk[:, 1:, :]).min() > 1e-3
    # wk1 is a2b(delp): delp is ~1e4 and strictly positive, so its
    # B-grid interpolant cannot approach zero.
    assert float(np.asarray(delp).min()) > 1.0e3

    uw = (slice(NG, NG + N), slice(NG, NG + N + 1))
    vw = (slice(NG, NG + N + 1), slice(NG, NG + N))
    bw = (slice(NG, NG + N + 1), slice(NG, NG + N + 1))
    s_pk, s_gz, s_pp, s_dp = 3.0, 2.0e3, 30.0, 1.0e4
    u_out = np.abs(np.asarray(
        nh_p_grad(u, v, pp, gz, delp, pk3, gs, BD, **kw)[0])[uw]).max()
    assert u_out > 0.0

    # Group A: the dominant hydrostatic du1 direction.
    def f_pg(pk3_h, gz_h):
        uo, vo, _, pko, gzo = nh_p_grad(u, v, pp, gz_h * s_gz, delp,
                                        pk3_h * s_pk, gs, BD, **kw)
        return (uo[uw] / u_out, vo[vw] / u_out, pko[bw] / s_pk,
                gzo[bw] / s_gz)

    check_grads(f_pg, (jnp.asarray(pk3 / s_pk), jnp.asarray(gz / s_gz)),
                order=2, modes=("fwd", "rev"))

    # Group B: pp (the NH term, ~0.3% of the bracket), the linear u/v
    # pass-through, and delp (which enters ONLY through the NH weight
    # wk1 = a2b(delp), i.e. as 1/delp).  All three are ~1e-5 of group
    # A's sensitivity, so they get their own call with the wind outputs
    # scaled by 1e6; ppo carries pp's own O(1) direction.
    def f_nh(u_, v_, pp_h, delp_h):
        uo, vo, ppo, _, _ = nh_p_grad(u_, v_, pp_h * s_pp, gz,
                                      delp_h * s_dp, pk3, gs, BD, **kw)
        return uo[uw] * 1.0e6, vo[vw] * 1.0e6, ppo[bw] / s_pp

    check_grads(f_nh, (jnp.asarray(u), jnp.asarray(v),
                       jnp.asarray(pp / s_pp), jnp.asarray(delp / s_dp)),
                order=2, modes=("fwd", "rev"))


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
    pk3_halo_np(pk3_n, delp, BD, npz=km, ptop=PTOP, akap=AKAP)
    pk3_j = pk3_halo(np.full((M_A, M_A, km + 1), SENT), delp, BD, npz=km,
                     ptop=PTOP, akap=AKAP)
    # TOL-PENDING: provisional bound; the orchestrator's measurement job will
    # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
    _cmp(f"pk3_halo[km={km}]", pk3_j, pk3_n, 1e-12)

    pln_n = np.full((M_A, M_A, km + 1), SENT)
    pln_halo_np(pln_n, delp, BD, npz=km, ptop=PTOP)
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
    pe_halo_np(pe_n, delp, BD, npz=km, ptop=PTOP)
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
    gc_n = geopk_np(f["delpc"], f["ptc"], f["hs"], BD, **kwc)
    gc_j = geopk(f["delpc"], f["ptc"], f["hs"], BD, **kwc)
    for name in ("pk", "gz", "pe", "peln", "pkz"):
        # TOL-PENDING: provisional bound; the orchestrator's measurement job will
        # replace this with `measured X, bound = measured x N`.  DO NOT SHIP.
        _cmp(f"oracle km={km} geopk_c.{name}", gc_j[name], gc_n[name],
             1e-12)

    # ---- p_grad_c
    uc_n, vc_n = np.array(f["uc"]), np.array(f["vc"])
    p_grad_c_np(1.0, f["delpc"], np.array(gc_n["pk"]),
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
    gd_n = geopk_np(f["delp"], f["pt"], f["hs"], BD, **kwd)
    gd_j = geopk(f["delp"], f["pt"], f["hs"], BD, **kwd)
    u_n, v_n = np.array(f["u"]), np.array(f["v"])
    one_grad_p_np(u_n, v_n, np.array(gd_n["pk"]), np.array(gd_n["gz"]),
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
