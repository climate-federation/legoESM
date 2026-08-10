"""Direct unit test for ``legoesm.core.fv3_native_pgrad`` (the ONE
km-general geopk / p_grad_c / one_grad_p implementation) and for the
``gen_geopk_pgrad_oracle`` serializer.

Deliberately FIXTURE-FREE: the bit-exact Fortran certificate lives in
``test_fv3_native_geopk_pgrad.py`` and needs the sbatch-generated npz
files; everything here runs from synthetic state so the module and the
gen script are covered even before the oracle has been regenerated.

What it pins:

* dispatch hardening — every unported branch RAISES instead of silently
  running the wrong formulas (``use_cond``, non-hydrostatic,
  ``a2b_ord != 4``, bad ``a2b_ord``/``km``);
* the geopk write WINDOWS (CG 1-halo vs D 2-halo vs the ``computehalo``
  full-data-domain extension) and the column recursion invariants;
* the SW-adapter REFACTOR EQUIVALENCE (UNCERTAIN U10): the four km=1
  entry points in ``fv3_native_duo_stepper`` lost their bodies to the
  shared kernel, so this file carries the pre-refactor km=1 formulas as
  a frozen LEGACY REFERENCE and asserts bit-for-bit equality.  That is
  the only reason numerics are repeated in this file.
"""

import hashlib
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
# N=12 (not 8): build_fv3_native_gridstruct needs the EXTENDED halo
# ngw = ng+1 = 4 for its kinked-corner lattice, and
# build_kinked_corner_lonlat raises "ng=4 unsupported for n=8".  12 is
# the size the whole fv3_native oracle family is certified at.
N, NG = 12, 3
M_A = N + 2 * NG
M_B = M_A + 1
SENTINEL = 1.0e30


def _load_gen():
    path = os.path.join(REPO, "scripts", "validate", "fv3_native",
                        "gen_geopk_pgrad_oracle.py")
    spec = importlib.util.spec_from_file_location(
        "gen_geopk_pgrad_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bd():
    from legoesm.core.fv3_native_sw_core import Bounds

    return Bounds.single_tile(N, NG)


def _synthetic(km, seed=7):
    """Smooth, strictly positive, level- and (i,j)-varying column."""
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(M_A), np.arange(M_A), indexing="ij")
    shape = 1.0 + 0.05 * np.cos(2.0 * np.pi * ii / M_A) \
        * np.sin(2.0 * np.pi * (jj + 0.5) / M_A)
    frac = np.array([0.2, 0.45, 0.35])[:km]
    frac = frac / frac.sum()
    delp = np.stack([f * (1.0e5 - 100.0) * shape for f in frac], axis=-1)
    pt = np.stack([(300.0 - 10.0 * (k + 1)) * shape for k in range(km)],
                  axis=-1)
    hs = 1.5e4 * shape
    return {
        "delp": delp, "pt": pt, "hs": hs,
        "uc": rng.standard_normal((M_B, M_A, km)) * 10.0,
        "vc": rng.standard_normal((M_A, M_B, km)) * 10.0,
        "u": rng.standard_normal((M_A, M_B, km)) * 10.0,
        "v": rng.standard_normal((M_B, M_A, km)) * 10.0,
    }


def _gs():
    """Gridstruct dict with real cube geometry at this resolution.

    ``rdyc`` is truncated to the (m_a, m_a) p_grad_c dummy sub-block
    exactly as the oracle serializes it (UNCERTAIN U4).
    """
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        build_fv3_native_gridstruct,
    )

    gs = dict(build_fv3_native_gridstruct(N, NG, tile=1,
                                          radius=FV3_RADIUS_M,
                                          omega=FV3_OMEGA))
    gs["rdyc"] = np.asarray(gs["rdyc"])[:, :M_A]
    gs.update(bounded_domain=False, grid_type=0, sw_corner=True,
              se_corner=True, nw_corner=True, ne_corner=True)
    return gs


# ---------------------------------------------------------------- dispatch

def test_geopk_use_cond_raises(bd):
    from legoesm.core.fv3_native_pgrad import geopk

    st = _synthetic(2)
    with pytest.raises(ValueError, match="USE_COND"):
        geopk(st["delp"], st["pt"], st["hs"], bd, km=2, ptop=100.0,
              akap=2.0 / 7.0, cp_air=1004.0, cg=True, duogrid=True,
              computehalo=False, npx=N + 1, npy=N + 1, a2b_ord=4,
              use_cond=True)


def test_geopk_unknown_a2b_ord_and_km_raise(bd):
    from legoesm.core.fv3_native_pgrad import geopk

    st = _synthetic(2)
    kw = dict(km=2, ptop=100.0, akap=2.0 / 7.0, cp_air=1004.0, cg=True,
              duogrid=True, computehalo=False, npx=N + 1, npy=N + 1)
    with pytest.raises(ValueError, match="a2b_ord"):
        geopk(st["delp"], st["pt"], st["hs"], bd, a2b_ord=3, **kw)
    with pytest.raises(ValueError, match="km"):
        geopk(st["delp"], st["pt"], st["hs"], bd,
              **{**kw, "km": 0, "a2b_ord": 4})


def test_p_grad_c_nonhydrostatic_branch(bd):
    """The NH branch differs from hydrostatic ONLY in the denominator
    weight (dyn_core.F90:2103-2113: wk = delpc vs wk = pkc-diff).

    Certified by an identity control plus a linearity control:
    (a) feeding delpc == the pkc interface difference makes the two
        branches BITWISE identical -- the momentum expressions are
        shared, so any divergence is a branch bug;
    (b) doubling delpc exactly halves the NH increment (the weight is
        the only place delpc enters).
    An earlier test asserted this branch raises NotImplementedError;
    it is implemented now (NH port unit 4).
    """
    from legoesm.core.fv3_native_pgrad import p_grad_c

    km = 2
    st = _synthetic(km)
    gs = _gs()
    rng = np.random.default_rng(17)
    pkc = np.cumsum(
        np.abs(rng.standard_normal((M_A, M_A, km + 1))) + 1.0, axis=2)
    gz = np.cumsum(
        np.abs(rng.standard_normal((M_A, M_A, km + 1))) + 5.0, axis=2)[:, :, ::-1].copy()

    # (a) identity control
    dpk = pkc[:, :, 1:] - pkc[:, :, :-1]
    uc_h = np.array(st["uc"]); vc_h = np.array(st["vc"])
    uc_n = np.array(st["uc"]); vc_n = np.array(st["vc"])
    p_grad_c(1.0, st["delp"], pkc, gz, uc_h, vc_h, gs, bd,
             npz=km, hydrostatic=True)
    p_grad_c(1.0, dpk, pkc, gz, uc_n, vc_n, gs, bd,
             npz=km, hydrostatic=False)
    assert np.array_equal(uc_h, uc_n)
    assert np.array_equal(vc_h, vc_n)
    # non-vacuity: the update moved the winds
    assert np.abs(uc_h - st["uc"]).max() > 0.0

    # (b) linearity control (denominator ~ 1/delpc).  Run it from
    # uc = vc = 0 so the increment IS the output -- extracting it by
    # subtracting a large uc loses ~eps*|uc| to cancellation, which at
    # small increments swamps a 1e-12 ratio tolerance (that was this
    # test's own first bug).  Doubling delpc doubles the fp-exact
    # denominator, and x/(2y) == (x/y)/2 exactly in binary fp.
    uc_a = np.zeros_like(st["uc"]); vc_a = np.zeros_like(st["vc"])
    uc_b = np.zeros_like(st["uc"]); vc_b = np.zeros_like(st["vc"])
    p_grad_c(1.0, dpk, pkc, gz, uc_a, vc_a, gs, bd,
             npz=km, hydrostatic=False)
    p_grad_c(1.0, 2.0 * dpk, pkc, gz, uc_b, vc_b, gs, bd,
             npz=km, hydrostatic=False)
    nz = np.abs(uc_a) > 0.0
    assert nz.any()
    ratio = uc_b[nz] / uc_a[nz]
    assert np.abs(ratio - 0.5).max() < 1e-14
    # codex NH r2 #4: a mutant using the HYDROSTATIC weight only for vc
    # passes the identity control and the uc halving -- vc must respond
    # to delpc too.  dpk is level- and (i,j)-random, so a k-shift in the
    # vc weight also fails here.
    nzv = np.abs(vc_a) > 0.0
    assert nzv.any()
    ratio_v = vc_b[nzv] / vc_a[nzv]
    assert np.abs(ratio_v - 0.5).max() < 1e-14


def test_one_grad_p_guards_raise(bd):
    from legoesm.core.fv3_native_pgrad import one_grad_p

    st = _synthetic(2)
    z = np.zeros((M_A, M_A, 3))
    div = np.zeros((N + 1, N + 1))
    kw = dict(npx=N + 1, npy=N + 1, npz=2, dt=1.0, ptop=100.0,
              akap=2.0 / 7.0)
    with pytest.raises(NotImplementedError, match="a2b_ord"):
        one_grad_p(st["u"], st["v"], z, z, div, None, {}, bd,
                   a2b_ord=2, **kw)
    with pytest.raises(NotImplementedError, match="non-hydrostatic"):
        one_grad_p(st["u"], st["v"], z, z, div, None, {}, bd,
                   hydrostatic=False, **kw)


# ------------------------------------------------------------ geopk shape

@pytest.mark.parametrize("km", (1, 2, 3))
def test_geopk_write_windows(bd, km):
    """CG=T takes the 1-halo box; CG=F/a2b_ord=4 the 2-halo box; the
    duo ``computehalo`` extension reaches the FULL data domain."""
    from legoesm.core.fv3_native_pgrad import geopk

    st = _synthetic(km)
    kw = dict(km=km, ptop=100.0, akap=2.0 / 7.0, cp_air=1004.0,
              duogrid=True, npx=N + 1, npy=N + 1, a2b_ord=4,
              unwritten_fill=SENTINEL)
    lo = 1 - NG
    gc = geopk(st["delp"], st["pt"], st["hs"], bd, cg=True,
               computehalo=False, **kw)
    inner = slice(1 - 1 - lo, N + 1 - lo + 1)
    assert np.isfinite(gc["pk"][inner, inner]).all()
    assert (gc["pk"][inner, inner] != SENTINEL).all()
    mask = np.ones((M_A, M_A), dtype=bool)
    mask[inner, inner] = False
    assert (gc["pk"][mask] == SENTINEL).all()
    assert (gc["gz"][mask] == SENTINEL).all()
    # pkz is NEVER written on the CG pass (dyn_core.F90:2781)
    assert (gc["pkz"] == SENTINEL).all()

    gd2 = geopk(st["delp"], st["pt"], st["hs"], bd, cg=False,
                computehalo=False, **kw)
    wide = slice(1 - 2 - lo, N + 2 - lo + 1)
    mask2 = np.ones((M_A, M_A), dtype=bool)
    mask2[wide, wide] = False
    assert (gd2["pk"][mask2] == SENTINEL).all()

    gdh = geopk(st["delp"], st["pt"], st["hs"], bd, cg=False,
                computehalo=True, **kw)
    assert (gdh["pk"] != SENTINEL).all()
    assert (gdh["gz"] != SENTINEL).all()
    assert np.isfinite(gdh["pkz"]).all()


@pytest.mark.parametrize("km", (2, 3))
def test_geopk_column_invariants(bd, km):
    from legoesm.core.fv3_native_pgrad import geopk

    st = _synthetic(km)
    akap = 2.0 / 7.0
    ptop = 100.0
    out = geopk(st["delp"], st["pt"], st["hs"], bd, km=km, ptop=ptop,
                akap=akap, cp_air=1004.0, cg=False, duogrid=True,
                computehalo=True, npx=N + 1, npy=N + 1, a2b_ord=4,
                unwritten_fill=SENTINEL)
    pk, gz = out["pk"], out["gz"]
    assert pk.shape == (M_A, M_A, km + 1)
    # k=1 interface is exactly ptop**akap, the `**` OPERATOR form
    assert (pk[:, :, 0] == ptop ** akap).all()
    # pressure integrates DOWNWARD: pk strictly increasing in k
    assert (np.diff(pk, axis=-1) > 0.0).all()
    # gz seeded at the SURFACE from hs and integrated UPWARD
    assert np.array_equal(gz[:, :, km], st["hs"])
    assert (np.diff(gz, axis=-1) < 0.0).all()
    # pe/peln are the same column in different units, on the peln window
    ci = slice(NG, NG + N)
    assert np.allclose(np.exp(out["peln"]), out["pe"][1:-1, :, 1:-1],
                       rtol=1e-12, atol=0.0)
    # pkz strictly between its bracketing interfaces
    pkc = pk[ci, ci, :]
    assert (out["pkz"] > pkc[:, :, :-1]).all()
    assert (out["pkz"] < pkc[:, :, 1:]).all()


def test_geopk_accumulation_is_top_down(bd):
    """The running ``p1d`` accumulator is order-sensitive — graded on
    ``pe``, the PRESSURE, not on ``pk``.

    INSTRUMENT FIX (job 9320294 self-reported a "fixture defect" here):
    ``pk = exp(akap*log(p))`` compresses a last-bit pressure difference
    by the factor ``akap`` and destroys ~95-98% of the sum-order signal
    — measured on this very column, 96 discriminating cells on ``p``
    collapse to 2 on ``pk`` at m=14, and 42 -> 13 at m=18.  The original
    check graded on ``pk``, found 0, and correctly refused to pass; the
    defect was the INSTRUMENT, not the column.  ``pe`` is also
    libm-free, so this stays exact on any host.

    Two-sided:
      POSITIVE — a top-down reconstruction reproduces ``pe`` BITWISE;
      NEGATIVE — the deliberately REVERSED accumulation is DETECTED.
    """
    from legoesm.core.fv3_native_pgrad import geopk

    km = 3
    st = _synthetic(km)
    ptop = 100.0
    out = geopk(st["delp"], st["pt"], st["hs"], bd, km=km, ptop=ptop,
                akap=2.0 / 7.0, cp_air=1004.0, cg=False, duogrid=True,
                computehalo=True, npx=N + 1, npy=N + 1, a2b_ord=4,
                unwritten_fill=SENTINEL)
    # pe origin is (is-1, 1, js-1); delp origin is (isd, jsd) = (1-ng, 1-ng)
    sl = slice(NG - 1, NG + N + 1)
    d = st["delp"][sl, sl, :]
    pe_sfc = np.ascontiguousarray(out["pe"][:, km, :])

    top = np.full(d.shape[:2], ptop)
    for k in range(km):
        top = top + d[:, :, k]
    assert _bitsame(top, pe_sfc), (
        "the top-down reconstruction does not reproduce the port's own pe "
        "bitwise — the running p1d accumulator or its window is wrong")

    bot = np.zeros(d.shape[:2])
    for k in range(km - 1, -1, -1):
        bot = bot + d[:, :, k]
    bot = bot + ptop
    assert not _bitsame(bot, pe_sfc), (
        "the REVERSED accumulation is indistinguishable from the correct "
        "one at every cell — this column cannot discriminate sum order, "
        "so the gate is vacuous (a fixture defect, not a pass)")


# ------------------------------------------------- SW adapter equivalence
# The four km=1 entry points lost their bodies to the shared kernel
# (spec 3.2 / UNCERTAIN U10).  The functions below are the FROZEN
# pre-refactor implementations, kept ONLY as the equivalence reference.

def _legacy_geopk_sw(delp2d, hs, bd, pt, halo):
    is_, ie = bd.is_, bd.ie
    m = delp2d.shape[0]
    pkc = np.zeros((m, m, 2))
    gz = np.zeros((m, m, 2))
    lo = 1 - bd.ng
    sl = slice(is_ - halo - lo, ie + halo - lo + 1)
    pkc[sl, sl, 0] = 0.0
    pkc[sl, sl, 1] = np.exp(1.0 * np.log(delp2d[sl, sl]))
    gz[sl, sl, 1] = hs[sl, sl]
    ptv = 1.0 if pt is None else pt[sl, sl]
    gz[sl, sl, 0] = gz[sl, sl, 1] + ptv * (pkc[sl, sl, 1] - pkc[sl, sl, 0])
    return pkc, gz


def _legacy_p_grad_c(dt2, pkc, gz, uc, vc, gs, bd):
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    lo = 1 - bd.ng
    rdxc, rdyc = gs["rdxc"], gs["rdyc"]
    wk = pkc[:, :, 1] - pkc[:, :, 0]

    def wk_at(i, j):
        return wk[i - lo, j - lo]

    def gz_at(i, j, k):
        return gz[i - lo, j - lo, k - 1]

    def pk_at(i, j, k):
        return pkc[i - lo, j - lo, k - 1]

    for j in range(js, je + 1):
        for i in range(is_, ie + 1 + 1):
            uc[i - lo, j - lo] += dt2 * rdxc[i - lo, j - lo] / (
                wk_at(i - 1, j) + wk_at(i, j)) * (
                (gz_at(i - 1, j, 2) - gz_at(i, j, 1))
                * (pk_at(i, j, 2) - pk_at(i - 1, j, 1))
                + (gz_at(i - 1, j, 1) - gz_at(i, j, 2))
                * (pk_at(i - 1, j, 2) - pk_at(i, j, 1)))
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1):
            vc[i - lo, j - lo] += dt2 * rdyc[i - lo, j - lo] / (
                wk_at(i, j - 1) + wk_at(i, j)) * (
                (gz_at(i, j - 1, 2) - gz_at(i, j, 1))
                * (pk_at(i, j, 2) - pk_at(i, j - 1, 1))
                + (gz_at(i, j - 1, 1) - gz_at(i, j, 2))
                * (pk_at(i, j - 1, 2) - pk_at(i, j, 1)))


def _legacy_one_grad_p(u, v, pkc, gz, divg2, gs, bd, npx, npy, dt, d_ext):
    from legoesm.core.fv3_native_d_sw import a2b_ord4
    from legoesm.grids.fv3_native_gridstruct import fort

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd
    ng = bd.ng
    lo = 1 - ng
    gsf = {
        "grid_lon": fort(gs["grid_lon"], isd, jsd),
        "grid_lat": fort(gs["grid_lat"], isd, jsd),
        "agrid_lon": fort(gs["agrid_lon"], isd, jsd),
        "agrid_lat": fort(gs["agrid_lat"], isd, jsd),
        "dxa": fort(gs["dxa"], isd, jsd), "dya": fort(gs["dya"], isd, jsd),
        "edge_w": gs["edge_w"], "edge_e": gs["edge_e"],
        "edge_s": gs["edge_s"], "edge_n": gs["edge_n"],
        "bounded_domain": False, "grid_type": 0,
        "sw_corner": True, "se_corner": True,
        "nw_corner": True, "ne_corner": True,
    }
    pk1 = np.array(pkc[:, :, 0], copy=True)
    pk2 = np.array(pkc[:, :, 1], copy=True)
    gz1 = np.array(gz[:, :, 0], copy=True)
    gz2 = np.array(gz[:, :, 1], copy=True)
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            pk1[i - lo, j - lo] = 0.0
    wkb = np.zeros_like(pk2)
    for arr in (pk2, gz1, gz2):
        a2b_ord4(fort(arr, isd, jsd), fort(wkb, isd, jsd), gsf, npx, npy,
                 is_, ie, js, je, ng, replace=True, duogrid=True)
    wk2 = np.zeros((ie - is_ + 1, je + 1 - js + 1))
    wk1 = np.zeros((ie + 1 - is_ + 1, je - js + 1))
    if d_ext > 0.0:
        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1):
                wk2[i - 1, j - 1] = divg2[i - 1, j - 1] - divg2[i, j - 1]
        for j in range(js, je + 1):
            for i in range(is_, ie + 1 + 1):
                wk1[i - 1, j - 1] = divg2[i - 1, j - 1] - divg2[i - 1, j]

    def at(a, i, j):
        return a[i - lo, j - lo]

    wk = pk2 - pk1
    rdx, rdy = gs["rdx"], gs["rdy"]
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1):
            u[i - lo, j - lo] = rdx[i - lo, j - lo] * (
                wk2[i - 1, j - 1] + u[i - lo, j - lo]
                + dt / (at(wk, i, j) + at(wk, i + 1, j)) * (
                    (at(gz2, i, j) - at(gz1, i + 1, j))
                    * (at(pk2, i + 1, j) - at(pk1, i, j))
                    + (at(gz1, i, j) - at(gz2, i + 1, j))
                    * (at(pk2, i, j) - at(pk1, i + 1, j))))
    for j in range(js, je + 1):
        for i in range(is_, ie + 1 + 1):
            v[i - lo, j - lo] = rdy[i - lo, j - lo] * (
                wk1[i - 1, j - 1] + v[i - lo, j - lo]
                + dt / (at(wk, i, j) + at(wk, i, j + 1)) * (
                    (at(gz2, i, j) - at(gz1, i, j + 1))
                    * (at(pk2, i, j + 1) - at(pk1, i, j))
                    + (at(gz1, i, j) - at(gz2, i, j + 1))
                    * (at(pk2, i, j) - at(pk1, i, j + 1))))


def _bitsame(a, b):
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and int(
        (a.view(np.uint64) != b.view(np.uint64)).sum()) == 0


@pytest.mark.parametrize("with_pt", (True, False))
def test_sw_geopk_adapters_match_legacy(bd, with_pt):
    from legoesm.core.fv3_native_duo_stepper import (
        geopk_sw_1lev,
        geopk_sw_1lev_d,
    )

    rng = np.random.default_rng(11)
    delp = np.abs(rng.standard_normal((M_A, M_A))) + 2.0
    hs = rng.standard_normal((M_A, M_A)) * 1.0e3
    pt = (np.full((M_A, M_A), 0.7) + 0.01 * rng.standard_normal((M_A, M_A))
          if with_pt else None)
    for fn, halo in ((geopk_sw_1lev, 1), (geopk_sw_1lev_d, 2)):
        got_pk, got_gz = fn(delp, hs, bd, pt=pt)
        want_pk, want_gz = _legacy_geopk_sw(delp, hs, bd, pt, halo)
        assert _bitsame(got_pk, want_pk), fn.__name__
        assert _bitsame(got_gz, want_gz), fn.__name__


def test_sw_p_grad_c_adapter_matches_legacy(bd):
    from legoesm.core.fv3_native_duo_stepper import (
        geopk_sw_1lev,
        p_grad_c_1lev,
    )

    gs = _gs()
    rng = np.random.default_rng(13)
    delpc = np.abs(rng.standard_normal((M_A, M_A))) + 2.0
    hs = rng.standard_normal((M_A, M_A)) * 1.0e3
    pkc, gz = geopk_sw_1lev(delpc, hs, bd, pt=np.full((M_A, M_A), 0.9))
    uc0 = rng.standard_normal((M_B, M_A))
    vc0 = rng.standard_normal((M_A, M_B))
    uc_a, vc_a = uc0.copy(), vc0.copy()
    uc_b, vc_b = uc0.copy(), vc0.copy()
    p_grad_c_1lev(0.5 * 225.0, delpc, pkc, gz, uc_a, vc_a, gs, bd)
    _legacy_p_grad_c(0.5 * 225.0, pkc, gz, uc_b, vc_b, gs, bd)
    assert _bitsame(uc_a, uc_b)
    assert _bitsame(vc_a, vc_b)
    assert not _bitsame(uc_a, uc0), "the PG update did not move uc"


def test_sw_one_grad_p_adapter_matches_legacy(bd):
    from legoesm.core.fv3_native_duo_stepper import (
        geopk_sw_1lev_d,
        one_grad_p_1lev,
    )

    gs = _gs()
    rng = np.random.default_rng(17)
    delp = np.abs(rng.standard_normal((M_A, M_A))) + 2.0
    hs = rng.standard_normal((M_A, M_A)) * 1.0e3
    pt = np.full((M_A, M_A), 0.9)
    divg2 = rng.standard_normal((N + 1, N + 1)) * 1.0e3
    u0 = rng.standard_normal((M_A, M_B))
    v0 = rng.standard_normal((M_B, M_A))

    pk_a, gz_a = geopk_sw_1lev_d(delp, hs, bd, pt=pt)
    ua, va = u0.copy(), v0.copy()
    one_grad_p_1lev(ua, va, pk_a, gz_a, divg2, gs, bd, N + 1, N + 1,
                    dt=225.0, d_ext=0.02)

    pk_b, gz_b = geopk_sw_1lev_d(delp, hs, bd, pt=pt)
    ub, vb = u0.copy(), v0.copy()
    _legacy_one_grad_p(ub, vb, pk_b, gz_b, divg2, gs, bd, N + 1, N + 1,
                       225.0, 0.02)
    assert _bitsame(ua, ub)
    assert _bitsame(va, vb)
    assert not _bitsame(ua, u0), "the PG tail did not move u"


# --------------------------------------------------------- gen serializer

def _mutation_footprint(call, arrays):
    """``({name: n_changed_words}, {name: post_call_bytes}, return_value)``
    for one call over the CALLER-OWNED arrays it was handed."""
    before = {k: np.array(v, dtype=np.float64, copy=True)
              for k, v in arrays.items()}
    ret = call()
    fp, after = {}, {}
    for k, v in arrays.items():
        a = np.ascontiguousarray(v, dtype=np.float64)
        b = np.ascontiguousarray(before[k], dtype=np.float64)
        fp[k] = int((a.view(np.uint64) != b.view(np.uint64)).sum())
        after[k] = a.copy()
    return fp, after, ret


def test_sw_adapter_mutation_footprints(bd):
    """The FULL aliasing contract of all four km=1 adapters (codex r21
    blocker A).

    Comparing only the returned/updated winds is NOT bit-identity: the
    first version of ``one_grad_p_1lev`` passed the caller's pkc/gz
    straight into the shared kernel, whose ``a2b_ord4(replace=True)``
    mutates them in place — 169 pkc words and 338 gz words moved where
    the legacy body moved ZERO, and the u/v-only test happily passed.
    This asserts the EXACT mutation footprint (which caller-owned arrays
    change, and by how many words) against the frozen legacy reference
    for every adapter, plus bitwise equality of every array afterwards.
    """
    from legoesm.core.fv3_native_duo_stepper import (
        geopk_sw_1lev,
        geopk_sw_1lev_d,
        one_grad_p_1lev,
        p_grad_c_1lev,
    )

    gs = _gs()
    rng = np.random.default_rng(29)
    delp = np.abs(rng.standard_normal((M_A, M_A))) + 2.0
    hs = rng.standard_normal((M_A, M_A)) * 1.0e3
    pt = np.full((M_A, M_A), 0.9) + 0.01 * rng.standard_normal((M_A, M_A))
    divg2 = rng.standard_normal((N + 1, N + 1)) * 1.0e3
    u0 = rng.standard_normal((M_A, M_B))
    v0 = rng.standard_normal((M_B, M_A))
    uc0 = rng.standard_normal((M_B, M_A))
    vc0 = rng.standard_normal((M_A, M_B))

    # ---- adapters 1 & 2: the geopk pair must mutate NOTHING
    for fn, halo in ((geopk_sw_1lev, 1), (geopk_sw_1lev_d, 2)):
        aa = {"delp": delp.copy(), "hs": hs.copy(), "pt": pt.copy()}
        fp_a, _af, ret_a = _mutation_footprint(
            lambda a=aa, f=fn: f(a["delp"], a["hs"], bd, pt=a["pt"]), aa)
        ll = {"delp": delp.copy(), "hs": hs.copy(), "pt": pt.copy()}
        fp_l, _lf, ret_l = _mutation_footprint(
            lambda a=ll, h=halo: _legacy_geopk_sw(a["delp"], a["hs"], bd,
                                                  a["pt"], h), ll)
        assert fp_a == fp_l, (fn.__name__, fp_a, fp_l)
        assert fp_a == {"delp": 0, "hs": 0, "pt": 0}, (fn.__name__, fp_a)
        assert _bitsame(ret_a[0], ret_l[0]) and _bitsame(ret_a[1], ret_l[1])

    # ---- adapter 3: p_grad_c updates uc/vc ONLY
    pkc, gz = geopk_sw_1lev(delp, hs, bd, pt=pt)
    aa = {"uc": uc0.copy(), "vc": vc0.copy(), "delpc": delp.copy(),
          "pkc": pkc.copy(), "gz": gz.copy(),
          "rdxc": np.array(gs["rdxc"]), "rdyc": np.array(gs["rdyc"])}
    gsa = dict(gs, rdxc=aa["rdxc"], rdyc=aa["rdyc"])
    fp_a, af_a, _r = _mutation_footprint(
        lambda: p_grad_c_1lev(112.5, aa["delpc"], aa["pkc"], aa["gz"],
                              aa["uc"], aa["vc"], gsa, bd), aa)
    ll = {"uc": uc0.copy(), "vc": vc0.copy(), "delpc": delp.copy(),
          "pkc": pkc.copy(), "gz": gz.copy(),
          "rdxc": np.array(gs["rdxc"]), "rdyc": np.array(gs["rdyc"])}
    gsl = dict(gs, rdxc=ll["rdxc"], rdyc=ll["rdyc"])
    fp_l, af_l, _r = _mutation_footprint(
        lambda: _legacy_p_grad_c(112.5, ll["pkc"], ll["gz"], ll["uc"],
                                 ll["vc"], gsl, bd), ll)
    assert fp_a == fp_l, ("p_grad_c_1lev", fp_a, fp_l)
    assert fp_a["uc"] > 0 and fp_a["vc"] > 0
    for key in ("delpc", "pkc", "gz", "rdxc", "rdyc"):
        assert fp_a[key] == 0, (key, fp_a[key])
    for key in aa:
        assert _bitsame(af_a[key], af_l[key]), key

    # ---- adapter 4: one_grad_p updates u/v ONLY (the regression)
    pkd, gzd = geopk_sw_1lev_d(delp, hs, bd, pt=pt)
    aa = {"u": u0.copy(), "v": v0.copy(), "pkc": pkd.copy(),
          "gz": gzd.copy(), "divg2": divg2.copy(),
          "rdx": np.array(gs["rdx"]), "rdy": np.array(gs["rdy"])}
    gsa = dict(gs, rdx=aa["rdx"], rdy=aa["rdy"])
    fp_a, af_a, _r = _mutation_footprint(
        lambda: one_grad_p_1lev(aa["u"], aa["v"], aa["pkc"], aa["gz"],
                                aa["divg2"], gsa, bd, N + 1, N + 1,
                                dt=225.0, d_ext=0.02), aa)
    ll = {"u": u0.copy(), "v": v0.copy(), "pkc": pkd.copy(),
          "gz": gzd.copy(), "divg2": divg2.copy(),
          "rdx": np.array(gs["rdx"]), "rdy": np.array(gs["rdy"])}
    gsl = dict(gs, rdx=ll["rdx"], rdy=ll["rdy"])
    fp_l, af_l, _r = _mutation_footprint(
        lambda: _legacy_one_grad_p(ll["u"], ll["v"], ll["pkc"], ll["gz"],
                                   ll["divg2"], gsl, bd, N + 1, N + 1,
                                   225.0, 0.02), ll)
    assert fp_a == fp_l, ("one_grad_p_1lev", fp_a, fp_l)
    assert fp_a["u"] > 0 and fp_a["v"] > 0
    for key in ("pkc", "gz", "divg2", "rdx", "rdy"):
        assert fp_a[key] == 0, (
            f"one_grad_p_1lev mutated caller-owned {key} in {fp_a[key]} "
            "words; the legacy body mutated ZERO — copy-on-entry lost")
    for key in aa:
        assert _bitsame(af_a[key], af_l[key]), key


def test_gen_serializer_layout_and_tamper():
    gen = _load_gen()
    fields = {}
    lo = 1 - gen.NG
    m_a = gen.RES + 2 * gen.NG
    km = 2
    rng = np.random.default_rng(23)
    for _tok, key, origin in gen.INPUT_FIELDS:
        if origin == "s":
            fields[key] = 1.25
        elif key in ("edge_w", "edge_e", "edge_s", "edge_n"):
            fields[key] = rng.standard_normal(gen.RES + 1)
        elif key == "logexp_probe":
            fields[key] = np.linspace(100.0, 1.0e5, gen.NPROBE)
        elif key == "divg2":
            fields[key] = rng.standard_normal((gen.RES + 1, gen.RES + 1))
        elif key in ("delpc", "ptc", "delp", "pt", "q_con"):
            fields[key] = rng.standard_normal((m_a, m_a, km))
        elif key in ("uc", "v"):
            fields[key] = rng.standard_normal((m_a + 1, m_a, km))
        elif key in ("vc", "u"):
            fields[key] = rng.standard_normal((m_a, m_a + 1, km))
        elif key == "rdxc":
            fields[key] = rng.standard_normal((m_a + 1, m_a))
        elif key in ("rdx",):
            fields[key] = rng.standard_normal((m_a, m_a + 1))
        elif key in ("rdy",):
            fields[key] = rng.standard_normal((m_a + 1, m_a))
        elif key in ("grid_lon", "grid_lat"):
            fields[key] = rng.standard_normal((m_a + 1, m_a + 1))
        else:
            fields[key] = rng.standard_normal((m_a, m_a))

    blob = gen.serialize_geopk_pgrad_inputs(fields, gen.RES, gen.NG, km)
    text = blob.decode()
    head = text.split("\n")
    assert head[0] == f"# res {gen.RES}"
    assert head[1] == f"# ng {gen.NG}"
    assert head[2] == f"# km {km}"
    for tag in ("# dt ", "# dt2 ", "# ptop ", "# akap ", "# cpair ",
                "# dext "):
        assert any(ln.startswith(tag) for ln in head[:12]), tag
    # every declared token appears, in the declared ORDER
    pos = -1
    for tok, _key, _o in gen.INPUT_FIELDS:
        idx = text.find(f"\n{tok} ")
        assert idx > 0, tok
        assert idx > pos, f"{tok} out of canonical order"
        pos = idx
    # 3-D records carry FOUR fields after the token, 2-D three, 1-D two
    by_tok = {}
    for ln in head:
        if not ln or ln.startswith("#"):
            continue
        pp = ln.split()
        by_tok.setdefault(pp[0], set()).add(len(pp))
    assert by_tok["DELP"] == {5}
    assert by_tok["HS"] == {4}
    assert by_tok["EDGE_W"] == {3}
    assert by_tok["DA_MIN_C"] == {2}
    # halo-origin fields start at 1-ng, base-1 fields at 1
    assert f"\nHS {lo} {lo} " in text
    assert "\nDIVG2 1 1 " in text
    # determinism + tamper
    assert gen.serialize_geopk_pgrad_inputs(fields, gen.RES, gen.NG,
                                            km) == blob
    tampered = dict(fields)
    tampered["delp"] = fields["delp"].copy()
    tampered["delp"].flat[0] = np.nextafter(tampered["delp"].flat[0], np.inf)
    bad = gen.serialize_geopk_pgrad_inputs(tampered, gen.RES, gen.NG, km)
    assert hashlib.sha256(bad).hexdigest() != \
        hashlib.sha256(blob).hexdigest()


def test_verify_manifest_rejects_tampering(tmp_path):
    """The run-manifest gate must be NON-VACUOUS (codex r21 blocker B).

    Round 20's fix checked only the six ``source.*`` hashes, so a
    hand-edited manifest carrying an arbitrary ``executable_sha256`` and
    ``repo_sha`` was still accepted — the binding did not exist.  This
    builds a manifest that PASSES, then tampers with each load-bearing
    field in turn and requires a refusal every time.
    """
    import hashlib as _h

    gen = _load_gen()
    head = gen.git_head()
    if head is None:
        pytest.skip("no git checkout to verify repo_sha against")
    work = str(tmp_path)
    km = 2
    # ORDER IS LOAD-BEARING. verify_manifest refuses an output that
    # predates the executable or its input, so the fixture has to write
    # them in that order -- writing `drv` last made the check a race that
    # only passed when all three landed in the same clock tick, and it
    # lost the race here (exe 1786117722.5693974 vs output
    # 1786117722.5683975, a 1 ms gap). The test is about TAMPERING, so a
    # baseline that fails on filesystem timestamp resolution is noise.
    (tmp_path / "drv").write_bytes(b"\x7fELF-not-really")
    (tmp_path / "geopk_pgrad_input.txt").write_bytes(b"# res 12\n")
    (tmp_path / f"geopk_pgrad_output_km{km}.txt").write_bytes(b"X 1 1 1 0\n")

    def _sha_path(p):
        return _h.sha256(open(p, "rb").read()).hexdigest()

    good = {
        "schema": gen.MANIFEST_SCHEMA, "km": str(km), "repo_sha": head,
        "compiler": "gfortran", "compiler_version": "GNU Fortran x",
        "compiler_target": "x86_64-pc-linux-gnu",
        "flags": "-O2 -fdefault-real-8", "defines": "NONE",
        "executable_sha256": _sha_path(tmp_path / "drv"),
        "input_sha256": _sha_path(tmp_path / "geopk_pgrad_input.txt"),
        "output_sha256": _sha_path(
            tmp_path / f"geopk_pgrad_output_km{km}.txt"),
    }
    for name in gen.MANIFEST_SOURCES:
        good[f"source.{name}"] = _sha_path(
            os.path.join(REPO, "scripts", "validate", "fv3_native", name))

    def _write(man):
        with open(gen.manifest_path(work, km), "w") as fh:
            for k, v in man.items():
                fh.write(f"{k}={v}\n")

    _write(good)
    gen.verify_manifest(work, km, good["input_sha256"])   # baseline PASSES

    bogus = "0" * 64
    for field, value in (("executable_sha256", bogus),
                         ("repo_sha", "f" * 40),
                         ("output_sha256", bogus),
                         (f"source.{gen.MANIFEST_SOURCES[0]}", bogus),
                         ("schema", "wrong_schema"),
                         ("km", "3")):
        bad = dict(good)
        bad[field] = value
        _write(bad)
        with pytest.raises(SystemExit):
            gen.verify_manifest(work, km, good["input_sha256"])
    # a manifest that simply omits a required field is also refused
    for field in ("executable_sha256", "repo_sha"):
        bad = {k: v for k, v in good.items() if k != field}
        _write(bad)
        with pytest.raises(SystemExit):
            gen.verify_manifest(work, km, good["input_sha256"])
    # and a mismatched input hash (staging vs run) is refused
    _write(good)
    with pytest.raises(SystemExit):
        gen.verify_manifest(work, km, bogus)


def test_gen_output_layout_covers_every_token():
    gen = _load_gen()
    shapes, origins = gen.output_layout(gen.RES, gen.NG, 3)
    for _tok, key in gen.TOKEN2KEY.items():
        assert key in shapes, key
        assert key in origins, key
        assert len(origins[key]) == len(shapes[key])
    assert shapes["pe_c"] == (gen.RES + 2, 4, gen.RES + 2)
    assert origins["pe_c"] == (0, 1, 0)
    assert shapes["logexp_out"] == (gen.NPROBE,)


def test_gen_dead_input_claims_are_declared_fields():
    gen = _load_gen()
    keys = {k for _t, k, _o in gen.INPUT_FIELDS}
    for key, reason in gen.DEAD_INPUTS.items():
        assert key in keys, key
        assert len(reason) > 30, key
    assert "q_con" in gen.DEAD_INPUTS
    assert gen.KM_PROFILES[2] != gen.KM_PROFILES[3][:2]


def test_gen_km_profiles_are_not_a_refinement():
    """km=3 must NOT reproduce the km=2 interfaces, else
    ``test_km2_km3_differ`` in the oracle gate would be a tautology."""
    gen = _load_gen()
    for km, frac in gen.KM_PROFILES.items():
        assert abs(sum(frac) - 1.0) < 1e-12, km
        assert len(set(frac)) == len(frac), f"km={km} has equal layers"
    assert abs(gen.KM_PROFILES[3][0] - gen.KM_PROFILES[2][0]) > 1e-3
