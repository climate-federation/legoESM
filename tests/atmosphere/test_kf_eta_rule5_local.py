"""RULE 5 tests of the KF-Eta port's local and source-geometry switches (#2280, PRs 2-3/7).

Smoothed here, each by one ``KFSmoothing`` width (0 = the oracle):

* ``lut_knot_width``: F1 (ALU log table) and F2 (TTAB/QSTAB and THE0K):
  knots and edge clamps of the piecewise-linear lookups;
* ``tpmix_width_kgkg``: D7a (TPMIX2 condense / evaporate / deficit split);
* ``condload_sqrt_width_m2s2``: D7b and the CONDLOAD square roots (E11 there);
* ``wtw_floor_width_m2s2``: D7d (the |WTW| < 1e-4 floor);
* ``prof5_width``: D7e (PROF5's two polynomial branches);
* PR 3/7, source geometry: ``lcl_gather_width_m`` (C-KLCLg, TENV/QENV/WKL/QSLCL/
  PLCL at ZLCL), ``l5_gather_width_pa`` (C-L5g, the L5 wind speed),
  ``q0_clip_width_kgkg`` / ``q0_floor_width_kgkg`` (E1) and
  ``tlcl_cap_width_k`` (E2).  The level indices themselves stay hard, and so
  do the mixed-layer averages (their source-layer extraction profile is hard;
  smoothing one without the other breaks the water budget, so both go with
  PR 7).
* PR 4/7, updraft level-local switches: E3, D1, D3, E4, E5, D4, E7, E8, D7g,
  E6, E9 (``KFSmoothing`` docstring).  Cloud top LTOP, LET, the A4 test and
  the D7f PROF5 endpoint overrides stay hard.
* PR 5a/7, cloud top by survival weights: ``top_stop_wtw_width_m2s2`` and
  ``top_stop_umf_width_kgs`` (C-LTOPs, C-LETs; the D4 weights are quintic).

The three tests per switch: the hard limit is the oracle (the oracle suite in
test_kain_fritsch_oracle.py runs ``KF_HARD``); the per-case gap at the default,
each width alone and all together, stored per fixture arm, column and output in
``oracle_fixtures/kf_eta_rule5_gaps.json`` (1e-6 relative, 1e-12 absolute
floor); and ``check_grads`` through each smoothed switch at the default.

Regenerate the stored gaps (fp64), one case per process, with
``JAX_ENABLE_X64=1 python tests/atmosphere/test_kf_eta_rule5_local.py --write CASE``
for CASE in each ``KFSmoothing`` field and ``all``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402
from legoesm.atmosphere.physics.convection import _kf_eta as kfe  # noqa: E402
from legoesm.atmosphere.physics.convection.config import KainFritschConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.kain_fritsch import kain_fritsch_convection  # noqa: E402
from legoesm.atmosphere.physics.convection.kain_fritsch_wrf import _lego_constants  # noqa: E402

from legoesm import constants  # noqa: E402
from tests.atmosphere.test_kain_fritsch_oracle import (  # noqa: E402
    _FIX,
    _OUT,
    _arms,
    _WRF_DEFAULT,
    _consts,
    _run_core,
    _wrapper_inputs,
)

_GAPS = Path(__file__).parent / "oracle_fixtures" / "kf_eta_rule5_gaps.json"
_DEFAULT = kfe.kf_smoothing(KainFritschConfig())
_FIELDS = kfe.KFSmoothing._fields
# each width alone at its default (the others hard), then all together
_CASES = {f: kfe.KF_HARD._replace(**{f: getattr(_DEFAULT, f)}) for f in _FIELDS}
_CASES["all"] = _DEFAULT
_CASES["hard"] = kfe.KF_HARD  # the port's own round-off gap, the baseline of the others
_OUTS = _OUT + ("pratec",)


def _sweep(fix, sm):
    """Per sweep column, the outputs of the port with widths ``sm``."""
    out = {k: np.zeros_like(fix[f"sweep_{k}"]) for k in _OUTS}
    for ci in np.unique(fix["sweep_combo"]):
        cs, trig, warm, fqi, fqs = fix["sweep_combos"][ci]
        sel = np.where(fix["sweep_combo"] == ci)[0]
        kc = _consts(fix, cs)

        def one(u, v, t, q, p, dz, w, rh, dt, dx):
            o = kfe.kf_eta_para(u[None], v[None], t[None], q[None], p[None], dz[None], w[None],
                              rh[None], dt=dt, dx=dx, constants=kc, trigger=int(trig),
                              warm_rain=warm == "1", f_qi=fqi == "1", f_qs=fqs == "1",
                              cudt=float(fix["cudt"]), smoothing=sm)
            return {k: o[k][0] for k in _OUTS}

        args = [jnp.asarray(fix[f"sweep_{k}"][sel]) for k in
                ("u", "v", "T", "q", "p", "dz", "w", "rhoe", "dt", "dx")]
        o = jax.jit(jax.vmap(one))(*args)
        for k in _OUTS:
            out[k][sel] = np.asarray(o[k])
    return out


def _colmax(ours, ref):
    d = np.abs(np.asarray(ours, np.float64) - np.asarray(ref, np.float64))
    return d.reshape(d.shape[0], -1).max(axis=1)


def measure_case(fix, case):
    """{arm or "sweep": {output: [max over levels |port - oracle| per column]}}."""
    sm = _CASES[case]
    g = {}
    for arm in _arms():
        o = _run_core(fix, arm, smoothing=sm)
        g[arm] = {k: _colmax(o[k], fix[f"{arm}_{k}"]).tolist() for k in _OUTS}
    o = _sweep(fix, sm)
    g["sweep"] = {k: _colmax(o[k], fix[f"sweep_{k}"]).tolist() for k in _OUTS}
    return g


def _provenance():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parent).stdout.strip()
    return dict(fixture_sha256=hashlib.sha256(_FIX.read_bytes()).hexdigest(),
                widths=_DEFAULT._asdict(), x64=bool(jax.config.jax_enable_x64),
                generated_at_commit=sha, metric="max over levels of |port - WRF oracle|")


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Drop compiled code after each test.  Every case compiles fresh KF graphs
    whose code stays mapped; one process climbed ~16k memory maps per gap case
    to the kernel's vm.max_map_count (65530) and LLVM then failed ("Cannot
    allocate memory"), failing and aborting the rest of the file.  Same
    mitigation as tests/land/conftest.py."""
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


@pytest.fixture(scope="module")
def stored():
    return json.loads(_GAPS.read_text())


def test_stored_gaps_are_for_this_fixture_and_these_widths(stored):
    p = stored["provenance"]
    assert p["fixture_sha256"] == hashlib.sha256(_FIX.read_bytes()).hexdigest()
    assert p["widths"] == _DEFAULT._asdict()
    assert p["x64"]
    assert set(stored["gaps"]) == set(_CASES)


@pytest.mark.parametrize("case", list(_CASES))
def test_default_gap_per_case_matches_record(fix, stored, case):
    """KNOWN DIFFERENCE, per fixture arm/column/output, at the default widths."""
    got = measure_case(fix, case)
    assert set(got) == set(stored["gaps"][case])
    for arm, per_out in stored["gaps"][case].items():
        for k, want in per_out.items():
            np.testing.assert_allclose(got[arm][k], want, rtol=1e-6, atol=1e-12,
                                       err_msg=f"{case} {arm} {k}")


# Widths whose switch no fixture column reaches (their gap equals the hard gap
# everywhere): the CONDLOAD root bridge needs 0 < G1 < 1e-4 m2/s2 and the floor
# ramp |WTW| near 1e-4, both only at an updraft's last level.  Covered instead by
# the synthetic unit tests below (both sides of each switch).  Likewise the Q0
# clip / floor, L5 and (PR 4) WKLCL cap, DTLCL onset, entrainment tiny weight,
# eqf clip and CHMIN clamp widths leave every fixture gap at the hard gap
# (measured); their unit tests below cover both sides.
_UNEXERCISED = {"condload_sqrt_width_m2s2", "wtw_floor_width_m2s2", "q0_clip_width_kgkg",
                "q0_floor_width_kgkg", "l5_gather_width_pa", "wklcl_cap_width_m",
                "dtlcl_onset_width_ms", "entrain_tiny_width_k", "eqf_clip_width",
                "chmin_clip_width_k", "top_stop_umf_width_kgs",
                # the band column is mid-band at the DEFAULT widths only (the
                # "all" case); stop1 alone is exercised on it shifted in-test
                "top_stop_wtw_width_m2s2"}


def _moved(stored, case):
    h = stored["gaps"]["hard"]
    return any(abs(a - b) > 1e-6 * abs(b) + 1e-14
               for arm, po in stored["gaps"][case].items() for k, v in po.items()
               for a, b in zip(v, h[arm][k]))


def test_fixture_coverage_of_each_width_is_as_recorded(stored):
    """A width counts as exercised by the fixture when its gap differs from the
    hard limit's in at least one arm/column/output."""
    assert {f for f in _FIELDS if not _moved(stored, f)} == _UNEXERCISED


# --- unit level: the smooth forms, their joins and gradients --------------------

def _tables(sm):
    return kfe._kf_tables(_lego_constants(), jnp.float64, sm)


_LUT_S, _LUT_H = _tables(_DEFAULT), _tables(kfe.KF_HARD)
_LUT_TP = _tables(kfe.KF_HARD._replace(tpmix_width_kgkg=_DEFAULT.tpmix_width_kgkg))


def _knot_a1(i, frac):
    return kfe._ALU_ASTRT + (i + frac) * kfe._ALU_AINC


def test_lut_smooth_is_exact_away_from_knots_and_c1_across_them():
    w = _DEFAULT.lut_knot_width
    for frac in (w + 0.05, 0.5, 1.0 - w - 0.05):
        a1 = _knot_a1(40, frac)
        np.testing.assert_allclose(kfe._tlog_lookup(_LUT_S, a1), kfe._tlog_lookup(_LUT_H, a1),
                                   rtol=1e-13)
    g = jax.grad(lambda a: kfe._tlog_lookup(_LUT_S, a))
    gh = jax.grad(lambda a: kfe._tlog_lookup(_LUT_H, a))
    k = _knot_a1(40, 0.0)
    eps = 1e-9 * kfe._ALU_AINC
    assert abs(float(g(k - eps) - g(k + eps))) < 1e-5 * abs(float(g(k)))
    assert abs(float(gh(k - eps) - gh(k + eps))) > 1e-3 * abs(float(gh(k)))
    np.testing.assert_allclose(g(_knot_a1(40, 0.5)), gh(_knot_a1(40, 0.5)), rtol=1e-12)


def test_lut_edges_extrapolate_then_hold_like_the_oracle():
    for i in (-0.5, -3.0, kfe._ALU_N + 5):
        a1 = _knot_a1(i, 0.0)
        np.testing.assert_allclose(kfe._tlog_lookup(_LUT_S, a1), kfe._tlog_lookup(_LUT_H, a1),
                                   rtol=1e-13)


def test_tab_interp_smooth_matches_bilinear_away_from_knots():
    p = float(_LUT_S.plutop) + 100.5 / float(_LUT_S.rdpr)
    bth = float(kfe._interp_c1(_LUT_S.the0k_pad, jnp.asarray(100.5), _DEFAULT.lut_knot_width))
    thes = bth + 30.5 / float(_LUT_S.rdthk)
    np.testing.assert_allclose(kfe._tpmix2dd(_LUT_S, p, thes), kfe._tpmix2dd(_LUT_H, p, thes),
                               rtol=1e-13)


def test_lut_check_grads_at_knots():
    check_grads(lambda a: kfe._tlog_lookup(_LUT_S, a), (_knot_a1(40, 0.1),), order=1,
                modes=("fwd", "rev"), eps=1e-7)
    p0 = float(_LUT_S.plutop) + 100.1 / float(_LUT_S.rdpr)
    for i in (0, 1):
        def f(p, t, i=i):
            return kfe._tpmix2dd(_LUT_S, p, t)[i]
        check_grads(f, (p0, 330.0), order=1, modes=("fwd", "rev"), eps=1e-3)
        assert float(jnp.abs(jax.grad(f, argnums=1)(p0, 330.0))) > 0.0


_P, _THES = 7.0e4, 335.0


def _tp2(lut, qu, qliq, qice):
    return kfe._tpmix2(lut, _P, _THES, qu, qliq, qice, constants.L_v, 0.0)


def _qs():
    return float(kfe._tpmix2dd(_LUT_H, _P, _THES)[1])


def test_tpmix2_smooth_conserves_water_and_is_exact_outside_band():
    qs, w = _qs(), _DEFAULT.tpmix_width_kgkg
    for dq, qtot in ((-5 * w, 1e-4), (5 * w, 1e-3), (1e-3, 5 * w), (0.3 * w, 0.1 * w)):
        qu, ql, qi = qs - dq, 0.7 * qtot, 0.3 * qtot
        _, qu2, ql2, qi2, qnl, _ = _tp2(_LUT_TP, qu, ql, qi)
        # closed budget: vapour + condensate + new condensate (qnewlq) in = out
        np.testing.assert_allclose(qu2 + ql2 + qi2 + qnl, qu + ql + qi, rtol=1e-14, atol=0)
        if abs(dq) > w and abs(dq - qtot) > w:
            # bound: everything the smooth form does differently sits at the
            # oracle's 1e-10 kg/kg guard (KNOWN DIFFERENCE in _tpmix2)
            np.testing.assert_allclose(_tp2(_LUT_TP, qu, ql, qi)[:5],
                                       _tp2(_LUT_H, qu, ql, qi)[:5], rtol=1e-9,
                                       atol=kfe._TPMIX_QEPS)


@pytest.mark.parametrize("dq,qtot,out", [(-0.1, 100.0, 4), (100.0, 99.9, 0)])
def test_tpmix2_check_grads_inside_bands(dq, qtot, out):
    """In the condensation band (output qnewlq) and the deficit band (output tu)."""
    w = _DEFAULT.tpmix_width_kgkg
    dq, qtot = dq * w, qtot * w

    def f(qu):
        return _tp2(_LUT_TP, qu, 0.7 * qtot, 0.3 * qtot)[out]

    check_grads(f, (_qs() - dq,), order=1, modes=("fwd", "rev"), eps=1e-9)
    assert float(jax.grad(f)(_qs() - dq)) != 0.0


def _cl(sm, wtw, g1):
    """CONDLOAD with no condensate: g1 = wtw + boterm; returns the 5 outputs."""
    return kfe._condload(0.0, 0.0, wtw, 100.0, g1 - wtw, 0.0, 0.0, 0.0, constants.g, sm)


def test_condload_smooth_is_exact_outside_bands():
    ws = _DEFAULT.condload_sqrt_width_m2s2
    for wtw, g1 in ((1.0, 2.0), (4.0, 3 * ws)):
        np.testing.assert_allclose(_cl(_DEFAULT, wtw, g1), _cl(kfe.KF_HARD, wtw, g1),
                                   rtol=1e-13)


def test_condload_check_grads_in_sqrt_bridge_and_floor_ramp():
    ws, wf = _DEFAULT.condload_sqrt_width_m2s2, _DEFAULT.wtw_floor_width_m2s2

    load = 2.0 * constants.g * 100.0 * 0.5e-3 / kfe._KE_DIV  # g1's loading term

    def fall(g1):  # the root bridge of g1, seen through the fallout qlqout
        return kfe._condload(1e-3, 0.0, 1.0, 100.0, g1 - 1.0 + load, 0.0, 0.0, 0.0,
                             constants.g, _DEFAULT)[3]

    check_grads(fall, (0.3 * ws,), order=1, modes=("fwd", "rev"), eps=1e-8)
    assert float(jax.grad(fall)(0.3 * ws)) != 0.0
    # the WTW floor: the jump at -1e-4 is a ramp of width wf below it
    def wtw_out(x):  # boterm sets the new WTW to x
        return kfe._condload(0.0, 0.0, 1.0, 100.0, x - 1.0, 0.0, 0.0, 0.0, constants.g,
                           _DEFAULT)[2]
    x = -kfe._WTW_FLOOR - 0.5 * wf
    check_grads(wtw_out, (x,), order=1, modes=("fwd", "rev"), eps=1e-7)
    assert float(jax.grad(wtw_out)(x)) != 0.0


def test_prof5_smooth_blends_branches():
    w = _DEFAULT.prof5_width
    eq0 = kfe._P5_Y_OFF / kfe._P5_Y_SCALE  # y = 0
    for eq in (eq0 + 2 * w / kfe._P5_Y_SCALE, eq0 - 2 * w / kfe._P5_Y_SCALE, 0.9, 0.1):
        np.testing.assert_allclose(kfe.prof5(jnp.asarray(eq), w), kfe.prof5(jnp.asarray(eq)),
                                   rtol=1e-12)
    for i in (0, 1):
        def f(e, i=i):
            return kfe.prof5(e, w)[i]
        check_grads(f, (jnp.asarray(eq0 + 1e-3),), order=1, modes=("fwd", "rev"), eps=1e-6)
        g = jax.grad(f)
        assert abs(float(g(eq0 - 1e-9)) - float(g(eq0 + 1e-9))) < 1e-6 * abs(float(g(eq0)))


def test_prof5_fractions_stay_nonnegative_up_to_the_width_bound():
    """Inside the blend band the fractions stay >= 0 (outside it they are the
    oracle's, whose polynomial is -2e-11 at eq = 0)."""
    eq = jnp.linspace(0.0, 1.0, 20001)
    y = kfe._P5_Y_SCALE * eq - kfe._P5_Y_OFF
    for w in (_DEFAULT.prof5_width, kfe._PROF5_WIDTH_MAX):
        ee, ud = kfe.prof5(eq, w)
        band = jnp.abs(y) < w
        assert int(band.sum()) > 10
        assert float(jnp.min(ee[band])) >= 0.0 and float(jnp.min(ud[band])) >= 0.0, w


def test_tpmix2_smooth_keeps_vapour_and_condensate_nonnegative():
    """Cold saturated air with no condensate: nothing condenses at qu = qs."""
    qs, w = _qs(), _DEFAULT.tpmix_width_kgkg
    for qu, qtot in ((qs, 0.0), (qs + 0.3 * w, 0.0), (qs - 0.3 * w, 0.2 * w),
                     (qs - 0.7 * w, 0.1 * w), (qs - 0.5 * w, 0.4 * w),
                     (qs - 0.02 * w, 0.75 * w)):
        tu, qu2, ql2, qi2, qnl, _ = _tp2(_LUT_TP, qu, 0.7 * qtot, 0.3 * qtot)
        assert min(float(qu2), float(ql2), float(qi2), float(qnl)) >= 0.0
        assert float(qnl) <= max(qu - qs, 0.0) + 1e-18
        assert float(ql2) <= 0.7 * qtot + 1e-21  # never more condensate than given
    # supersaturated with a little condensate: nothing evaporates
    _, _, ql2, qi2, _, _ = _tp2(_LUT_TP, qs + 5 * w, 0.35 * w, 0.15 * w)
    assert float(ql2) == 0.35 * w and float(qi2) == 0.15 * w


# --- source geometry (PR 3) ------------------------------------------------------

def _col(n=12):
    """1-based padded column: heights rising, pressures falling, one smooth field."""
    rng = np.random.default_rng(3)
    dz = np.concatenate([[200.0], 300.0 + 1200.0 * rng.random(n - 1)])
    z = np.concatenate([[0.0], np.cumsum(dz) - 0.5 * dz, [0.0]])
    z[-1] = z[-2]
    p = 1.0e5 * np.exp(-z / 8000.0)
    kk = jnp.arange(n + 2)
    valid = (kk >= 1) & (kk <= n)
    f = 290.0 - 6.5e-3 * z + 2.0 * np.sin(z / 700.0)
    return jnp.asarray(z), jnp.asarray(p), kk, valid, jnp.asarray(f)


def _hard_gather(f, z0, z, lc):
    """The oracle's KLCL search and interpolation (concrete z)."""
    hit = (z <= np.asarray(z0[1:-1])) & (np.arange(1, len(z0) - 1) >= lc)
    k = int(np.argmax(hit)) + 1 if hit.any() else len(z0) - 2
    return f[k - 1] + (f[k] - f[k - 1]) * (z - z0[k - 1]) / (z0[k] - z0[k - 1])


def test_cap_c1_is_one_sided_exact_outside_band_and_differentiable():
    w = 1.0e-6
    x = jnp.linspace(-5 * w, 5 * w, 1001)
    c = kfe._cap_c1(x, 0.0, w)
    assert float(jnp.max(c)) <= 0.0
    np.testing.assert_array_equal(c[x <= -w], x[x <= -w])
    np.testing.assert_array_equal(kfe._cap_c1(x, 0.0, 0.0), jnp.minimum(x, 0.0))
    check_grads(lambda v: kfe._cap_c1(v, 0.0, w), (-0.3 * w,), order=1, modes=("fwd", "rev"),
                eps=1e-9)
    g = jax.grad(lambda v: kfe._cap_c1(v, 0.0, w))
    assert 0.0 < float(g(-0.3 * w)) < 1.0 and float(g(0.5 * w)) == 0.0


def test_gather_weights_sum_to_one_reproduce_lines_and_match_oracle_off_levels():
    z0, _, kk, valid, f = _col()
    w, lc = _DEFAULT.lcl_gather_width_m, 3
    g = lambda fld, z: kfe._gather_at_height(fld, z0, z, kk, valid, lc, w)  # noqa: E731
    for z in np.linspace(-500.0, float(z0[-2]) + 500.0, 157):
        np.testing.assert_allclose(g(jnp.ones_like(f), z), 1.0, rtol=1e-14)
        np.testing.assert_allclose(g(3.0 + 0.01 * z0, z), 3.0 + 0.01 * z, rtol=1e-13)
        if np.min(np.abs(z - np.asarray(z0[lc:-1]))) > w / 2:
            np.testing.assert_allclose(g(f, z), _hard_gather(f, z0, z, lc), rtol=1e-13)


def test_gather_from_the_first_level_extrapolates_like_the_oracle():
    z0, _, kk, valid, f = _col()
    w = _DEFAULT.lcl_gather_width_m
    for z in (0.0, 0.5 * float(z0[1]), float(z0[1]) + 0.6 * w, float(z0[3]) - 0.6 * w):
        np.testing.assert_allclose(kfe._gather_at_height(f, z0, z, kk, valid, 1, w),
                                   _hard_gather(f, z0, z, 1), rtol=1e-13)


def test_gather_is_c1_across_a_level_where_the_oracle_kinks():
    z0, _, kk, valid, f = _col()
    w, lc, zk = _DEFAULT.lcl_gather_width_m, 3, float(z0[6])
    g = jax.grad(lambda z: kfe._gather_at_height(f, z0, z, kk, valid, lc, w))
    gh = jax.grad(lambda z: jnp.interp(z, z0[lc - 1:-1], f[lc - 1:-1]))  # oracle, inside
    eps = 1e-6
    assert abs(float(g(zk - eps) - g(zk + eps))) < 1e-6 * abs(float(g(zk)))
    assert abs(float(gh(zk - eps) - gh(zk + eps))) > 1e-3 * abs(float(gh(zk)))
    check_grads(lambda z, fld: kfe._gather_at_height(fld, z0, z, kk, valid, lc, w),
                (zk + 0.2 * w, f), order=1, modes=("fwd", "rev"), eps=1e-4)


def test_l5_weights_one_hot_off_band_sum_to_one_and_differentiable():
    _, p, kk, valid, _ = _col()
    w = _DEFAULT.l5_gather_width_pa
    wt = kfe._l5_weights(p, kk, valid, w)
    pc = kfe._L5_FRAC * float(p[1])
    l5 = int(np.max(np.where(np.asarray(valid) & (np.asarray(p) >= pc), np.asarray(kk), 0)))
    np.testing.assert_allclose(float(wt.sum()), 1.0, rtol=1e-14)
    if np.min(np.abs(np.asarray(p[1:-1]) - pc)) > w / 2:
        np.testing.assert_array_equal(wt, (np.asarray(kk) == l5).astype(float))
    # move the level just above L5 into the band: two levels share the weight
    p2 = p.at[l5 + 1].set(pc - 0.1 * w)
    wt2 = kfe._l5_weights(p2, kk, valid, w)
    assert 0.0 < float(wt2[l5 + 1]) < 1.0
    np.testing.assert_allclose(float(wt2.sum()), 1.0, rtol=1e-14)
    check_grads(lambda q: kfe._l5_weights(q, kk, valid, w)[l5 + 1], (p2,), order=1,
                modes=("fwd", "rev"), eps=1e-2)


# --- updraft (PR 4) -----------------------------------------------------------------

def _sm(**kw):
    return kfe.KF_HARD._replace(**{k: getattr(_DEFAULT, k) for k in kw})


def _grad_nonzero(f, x, eps):
    check_grads(f, (x,), order=1, modes=("fwd", "rev"), eps=eps)
    assert float(jax.grad(f)(x)) != 0.0


def test_clip_c1_exact_outside_bands_and_c1_inside():
    w = 0.05
    x = jnp.linspace(-0.5, 1.5, 2001)
    far = (jnp.abs(x) >= w / 2) & (jnp.abs(x - 1.0) >= w / 2)
    np.testing.assert_array_equal(kfe._clip_c1(x, 0.0, 1.0, w)[far], jnp.clip(x, 0.0, 1.0)[far])
    np.testing.assert_array_equal(kfe._clip_c1(x, 0.0, 1.0, 0.0), jnp.clip(x, 0.0, 1.0))
    for x0 in (0.3 * w, 1.0 - 0.3 * w):
        _grad_nonzero(lambda v: kfe._clip_c1(v, 0.0, 1.0, w), x0, 1e-6)


def test_abe_increment_positive_part_shared_by_accumulation_and_rollback():
    w = _DEFAULT.abe_pos_width_m
    d = jnp.linspace(-3 * w, 3 * w, 601)
    pos_h, inc_h = kfe._abe_increment(d, constants.g, 0.0)
    pos_s, inc_s = kfe._abe_increment(d, constants.g, w)
    hard = jnp.where(pos_h, inc_h, 0.0)
    soft = jnp.where(pos_s, inc_s, 0.0)
    assert float(jnp.min(soft)) >= 0.0 and bool(jnp.all(soft <= hard + 1e-15))
    np.testing.assert_array_equal(soft[d >= w], hard[d >= w])
    _grad_nonzero(lambda v: kfe._abe_increment(v, constants.g, w)[1], 0.3 * w, 1e-6 * w)


def test_dtlcl_ramps_in_above_the_oracle_jump():
    w, c, e = _DEFAULT.dtlcl_onset_width_ms, 4.64, 0.33
    hard = lambda v: kfe._dtlcl(v, c, e, 0.0)  # noqa: E731
    soft = lambda v: kfe._dtlcl(v, c, e, w)  # noqa: E731
    lo, hi = kfe._WKL_MIN, kfe._WKL_MIN + w
    assert float(hard(lo)) > 0.2 and float(hard(lo * (1 - 1e-9))) == 0.0  # the oracle jump
    for v in (-1.0, 0.5 * lo, lo * (1 - 1e-9), hi * (1 + 1e-9), 0.05):
        np.testing.assert_allclose(soft(v), hard(v), rtol=1e-14)
    _grad_nonzero(soft, lo + 0.3 * w, 1e-3 * w)
    assert abs(float(soft(lo + 1e-12 * w))) < 1e-10  # continuous at the threshold


def test_wlcl_ramps_in_above_the_oracle_jump():
    w = _DEFAULT.wlcl_onset_width_k
    k = 2.0 * constants.g * kfe._GDT_DEPTH_M / 300.0
    f = lambda dt, ww: kfe._wlcl(dt, k * dt, k * kfe._DTTOT_MIN_K, ww)  # noqa: E731
    lo = kfe._DTTOT_MIN_K
    for v in (0.0, 0.5 * lo, lo + 1.01 * w, 2.0):
        np.testing.assert_allclose(f(v, w), f(v, 0.0), rtol=1e-14)
    assert float(f(lo * (1 + 1e-9), 0.0)) - float(f(lo, 0.0)) > 0.02  # the oracle jump
    _grad_nonzero(lambda v: f(v, w), lo + 0.3 * w, 1e-4 * w)
    # E4 cap on top: one-sided, never above 3 m/s
    cap = kfe._cap_c1(jnp.linspace(2.0, 4.0, 401), kfe._WLCL_MAX, _DEFAULT.wlcl_cap_width_ms)
    assert float(jnp.max(cap)) <= kfe._WLCL_MAX


def _sel(dcold, dvb, tvdiff, ws, wt, ee_s=0.3, ud_s=0.6, eqf=0.4):
    a = [jnp.asarray(x) for x in (dcold, dvb, tvdiff, ee_s, ud_s, eqf)]
    return kfe._entrain_select(a[0] >= 0.0, a[1] > 0.0, a[2] < kfe._TVDIFF_MIN, a[0], a[1],
                               a[2], a[3], a[4], a[5], ws, wt)


def test_entrain_select_matches_oracle_outside_bands_and_is_c1():
    ws, wt = _DEFAULT.entrain_select_width_k, _DEFAULT.entrain_tiny_width_k
    rng = np.random.default_rng(5)
    for _ in range(300):
        dc, dv = rng.choice([-1, 1], 2) * (0.5 * ws + rng.random(2))
        td = rng.choice([0.5e-3, 2.5e-3 + rng.random()])
        np.testing.assert_allclose(_sel(dc, dv, td, ws, wt), _sel(dc, dv, td, 0.0, 0.0),
                                   rtol=1e-14, atol=0)
    for i in range(3):
        _grad_nonzero(lambda v, i=i: _sel(v, -1.0, 0.1, ws, wt)[i], 0.2 * ws, 1e-4 * ws)
        _grad_nonzero(lambda v, i=i: _sel(-1.0, v, 0.1, ws, wt)[i], 0.2 * ws, 1e-4 * ws)
        _grad_nonzero(lambda v, i=i: _sel(-1.0, -1.0, v, ws, wt)[i], 1.4e-3, 1e-3 * wt)
    # every value stays a convex blend of the three branches
    ee, ud, eqfrc = _sel(0.01, 0.02, 1.5e-3, ws, wt)
    assert 0.3 <= float(ee) <= 1.0 and 0.0 <= float(ud) <= 1.0 and 0.0 <= float(eqfrc) <= 1.0


def _glac(tu, sm, ttemp=kfe._TTFRZ_K):
    kc = _lego_constants()
    ab = (kc.svp1 * kfe._KPA_TO_PA, kc.svp2, kc.svp2 * kc.svpt0, kc.svp3)
    qs = float(kfe._tpmix2dd(_LUT_H, 5.0e4, 330.0)[1])
    return kfe._glaciate(ab, 5.0e4, tu, 330.0, qs, 1e-3, 2e-4, 3e-4, 1e-4, ttemp, sm)


def test_glaciate_conserves_water_and_matches_oracle_outside_bands():
    smf = _sm(freeze_onset_width_k=1, frc_clip_width=1)
    wf = _DEFAULT.freeze_onset_width_k
    water = lambda o: o[2] + o[3] + o[4] + o[5] + o[6]  # noqa: E731
    for tu in (kfe._TTFRZ_K + 1.0, kfe._TTFRZ_K - 0.3 * wf, kfe._TTFRZ_K - wf - 0.5,
               kfe._TBFRZ_K - 3.0):
        o = _glac(tu, smf)
        np.testing.assert_allclose(water(o), water(_glac(tu, kfe.KF_HARD)), rtol=1e-14)
        if abs(tu - kfe._TTFRZ_K) > wf and abs(tu - kfe._TBFRZ_K) > 1.0:
            np.testing.assert_allclose(o[:7], _glac(tu, kfe.KF_HARD)[:7], rtol=1e-13)
    for i in (0, 4):  # tu and qice through the freezing-onset blend
        _grad_nonzero(lambda t, i=i: _glac(t, smf)[i], kfe._TTFRZ_K - 0.3 * wf, 1e-4)
    # the frozen-fraction cap near TBFRZ (ttemp0 = 260 K, frc reaches 1 at TBFRZ)
    _grad_nonzero(lambda t: _glac(t, smf, 260.0)[4], kfe._TBFRZ_K + 0.002, 1e-5)


def test_wklcl_cap_is_c1_at_zref_and_exact_outside():
    w, ref, zref = _DEFAULT.wklcl_cap_width_m, 0.02, 2.0e3
    for z in (500.0, zref - 0.6 * w, zref + 0.6 * w, 5.0e3):
        np.testing.assert_allclose(kfe._wklcl(z, ref, zref, w), kfe._wklcl(z, ref, zref, 0.0),
                                   rtol=1e-14)
    assert float(kfe._wklcl(zref, ref, zref, w)) < float(kfe._wklcl(zref, ref, zref, 0.0))
    _grad_nonzero(lambda z: kfe._wklcl(z, ref, zref, w), zref + 0.2 * w, 1e-3)
    g = jax.grad(lambda z: kfe._wklcl(z, ref, zref, w))
    gh = jax.grad(lambda z: kfe._wklcl(z, ref, zref, 0.0))
    assert abs(float(g(zref - 1e-6) - g(zref + 1e-6))) < 1e-3 * ref / zref  # C1 at zref
    assert abs(float(gh(zref - 1e-6) - gh(zref + 1e-6))) > 0.5 * ref / zref  # oracle kink


def test_chmin_clamp_is_c1_at_both_kinks():
    w, lo, hi = _DEFAULT.chmin_clip_width_k, kfe._CHMIN_TCOLD_K, kfe._CHMIN_TWARM_K
    for t in (lo, hi):
        assert float(kfe._clip_c1(t, lo, hi, w)) != float(kfe._clip_c1(t, lo, hi, 0.0))
        _grad_nonzero(lambda v: kfe._clip_c1(v, lo, hi, w), t + 0.2 * w, 1e-4)
        g = jax.grad(lambda v: kfe._clip_c1(v, lo, hi, w))
        assert abs(float(g(t - 1e-7) - g(t + 1e-7))) < 1e-5


def test_chmin_and_rad_clamps_hit_the_oracle_ends():
    w = _DEFAULT.rad_clip_width
    for x in (-0.5, -0.6 * w, 0.5, 1.0 + 0.6 * w, 3.0):
        np.testing.assert_array_equal(kfe._clip_c1(x, 0.0, 1.0, w),
                                      kfe._clip_c1(x, 0.0, 1.0, 0.0))
    assert float(jnp.min(kfe._clip_c1(jnp.linspace(-1.0, 2.0, 3001), 0.0, 1.0, w))) >= 0.0
    assert kfe._CHMIN_DEEP_M == kfe._CHMIN_SHALLOW_M + kfe._CHMIN_SLOPE_M_PER_K * (
        kfe._CHMIN_TWARM_K - kfe._CHMIN_TCOLD_K)
    w = _DEFAULT.chmin_clip_width_k
    for t in (kfe._CHMIN_TCOLD_K - w, kfe._CHMIN_TWARM_K + w):
        np.testing.assert_array_equal(
            kfe._clip_c1(t, kfe._CHMIN_TCOLD_K, kfe._CHMIN_TWARM_K, w),
            kfe._clip_c1(t, kfe._CHMIN_TCOLD_K, kfe._CHMIN_TWARM_K, 0.0))


# --- soft cloud top, survival weights (PR 5a) -------------------------------------

def _oracle_top_block(j, t, f):
    """NumPy transcription of the oracle's cloud-top block (l.1421-1475) for
    LET = j, LTOP = t on plume arrays ``f`` (dict of 1-based padded arrays)."""
    umf, uer, udr = f["UMF"].copy(), f["UER"].copy(), f["UDR"].copy()
    detlq, detic = f["DETLQ"].copy(), f["DETIC"].copy()
    pl, pi_ = f["PPTLIQ"].copy(), f["PPTICE"].copy()
    if j == t:
        udr_t = umf[t] + udr[t] - uer[t]
        rat = f["UPN"][t - 1] / f["UPO"][t - 1]
        udr[t], detlq[t], detic[t] = udr_t, f["QLIQ"][t] * udr_t * rat, f["QICE"][t] * udr_t * rat
        uer[t], umf[t] = 0.0, 0.0
    elif j < t:
        dumfdp = umf[j] / f["DP"][j + 1:t + 1].sum()
        for k in range(j + 1, t + 1):
            un = umf[k - 1] - f["DP"][k] * dumfdp
            ue = un * (1.0 - 1.0 / f["DILFRC"][k])
            ud = umf[k - 1] - un + ue
            if k == t:
                ud, ue, un = umf[k - 1], 0.0, umf[k]
            if k >= j + 2:
                pl[k], pi_[k] = umf[k - 1] * f["QLQOUT"][k], umf[k - 1] * f["QICOUT"][k]
            umf[k], uer[k], udr[k] = un, ue, ud
            detlq[k] = ud * f["QLIQ"][k] * f["DILFRC"][k]
            detic[k] = ud * f["QICE"][k] * f["DILFRC"][k]
    out = [umf, uer, udr, detlq, detic, pl, pi_]
    for x in out:
        x[t + 1:] = 0.0
    return out


def _plume(n=10, seed=7):
    rng = np.random.default_rng(seed)
    f = {k: 0.5 + rng.random(n + 2) for k in
         ("UMF", "UER", "UDR", "DETLQ", "DETIC", "PPTLIQ", "PPTICE", "QLIQ", "QICE",
          "QLQOUT", "QICOUT", "UPO", "UPN")}
    f["DILFRC"] = 1.0 + 0.3 * rng.random(n + 2)
    f["DP"] = 2000.0 + 3000.0 * rng.random(n + 2)
    return f


def _expect(W, f, kk):
    return kfe._expected_top_profile(
        jnp.asarray(W), *(jnp.asarray(f[k]) for k in (
            "UMF", "UER", "UDR", "DETLQ", "DETIC", "PPTLIQ", "PPTICE", "QLIQ", "QICE",
            "DILFRC", "QLQOUT", "QICOUT", "UPO", "UPN", "DP")), kk)


def test_expected_top_profile_one_hot_is_the_oracle_block():
    f, n = _plume(), 10
    kk = jnp.arange(n + 2)
    for j, t in ((4, 4), (3, 7), (6, 7), (2, 9), (5, 10), (8, 5)):
        W = np.zeros((n + 2, n + 2))
        W[j, t] = 1.0
        got = _expect(W, f, kk)
        want = _oracle_top_block(min(j, t), t, f)
        for g, w_ in zip(got, want):
            np.testing.assert_allclose(np.asarray(g), w_, rtol=1e-13, atol=1e-14, err_msg=f"{j},{t}")


def test_expected_top_profile_is_linear_in_weights_and_c2():
    f, n = _plume(), 10
    kk = jnp.arange(n + 2)
    rng = np.random.default_rng(1)
    W = rng.random((n + 2, n + 2))
    W[[0, n + 1], :] = 0.0
    W[:, [0, n + 1]] = 0.0
    W /= W.sum()
    got = _expect(W, f, kk)
    want = [sum(W[j, t] * np.asarray(_oracle_top_block(min(j, t), t, f)[i])
                for j in range(n + 2) for t in range(1, n + 1)) for i in range(7)]
    for g, w_ in zip(got, want):
        np.testing.assert_allclose(np.asarray(g)[1:n + 1], np.asarray(w_)[1:n + 1], rtol=1e-12)
    check_grads(lambda w, u: _expect_w(w, u, f, kk), (jnp.asarray(W), jnp.asarray(f["UMF"])),
                order=2, modes=("fwd", "rev"), eps=1e-4)


def _expect_w(W, umf, f, kk):
    g = dict(f, UMF=umf)
    return jnp.sum(jnp.stack(_expect(W, {k: v for k, v in g.items()}, kk)) ** 2)


def _weights(seed=3, n=12, K=3, wc=None):
    rng = np.random.default_rng(seed)
    kk = jnp.arange(n + 2)
    h1, h2 = 0.2 * rng.random(n + 2), 0.1 * rng.random(n + 2)
    s, E1, E2 = 1.0, np.zeros(n + 2), np.zeros(n + 2)
    for k in range(K, n):
        E1[k], E2[k] = s * h1[k], s * (1 - h1[k]) * h2[k]
        s *= (1 - h1[k]) * (1 - h2[k])
    CW = rng.random(n + 2) if wc is None else wc
    return kk, jnp.asarray(E1), jnp.asarray(E2), s, jnp.asarray(CW), K, n


def test_top_weights_sum_to_p_convecting_and_reduce_to_the_oracle_let():
    kk, E1, E2, s, CW, K, n = _weights()
    W = kfe._top_weights(E1, E2, s, CW, kk, K, K + 1, n, K)
    # tops at or below LCL = K + 1 are no-convection members: zero weight
    p_conv = 1.0 - float(jnp.sum((E1 + E2)[K:K + 2]))
    np.testing.assert_allclose(float(jnp.sum(W)), p_conv, rtol=1e-13)
    assert float(jnp.min(W)) >= 0.0 and p_conv < 1.0
    # a mixed layer above LCL: tops at or below KPBL and LETs below KPBL - 1
    # are rejected as well (the oracle's no-convection test, per member)
    Wp = np.asarray(kfe._top_weights(E1, E2, s, CW, kk, K, K + 1, n, K + 3))
    assert not Wp[:, :K + 4].any() and not Wp[:K + 3, :].any()
    np.testing.assert_array_equal(Wp[K + 3:, K + 4:], np.asarray(W)[K + 3:, K + 4:])
    # the shallow LET override keeps the rejected members out
    e1 = np.zeros(n + 2)
    e1[9], e1[12] = 0.25, 0.75
    cw = np.ones(n + 2)
    cw[10] = 0.0
    args = (jnp.asarray(e1), E1 * 0, 0.0, jnp.asarray(cw), kk, K, K + 1, n, K + 3)
    Wn, Ws = kfe._top_weights(*args), kfe._top_weights(*args, let_fixed=K + 3)
    np.testing.assert_allclose([float(jnp.sum(Wn)), float(jnp.sum(Ws))], [0.75, 0.75], rtol=1e-15)
    # hard cold flags: LET given a stop1 top t is the last non-cold level <= t
    cold = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0], float)
    E1h = np.zeros(n + 2)
    E1h[8] = 1.0
    W = np.asarray(kfe._top_weights(jnp.asarray(E1h), E1 * 0, 0.0, jnp.asarray(cold), kk,
                                    K, K + 1, n, K))
    assert W[5, 8] == 1.0 and W.sum() == 1.0  # levels 6-8 cold: LET = 5
    W = np.asarray(kfe._top_weights(jnp.asarray(E1h), E1 * 0, 0.0, jnp.asarray(cold * 0 + 1),
                                    kk, K, K + 1, n, K))
    assert W[K + 1, 8] == 1.0  # all cold: the initial LET = LCL
    E2h = np.zeros(n + 2)
    E2h[7] = 1.0
    W = np.asarray(kfe._top_weights(E1 * 0, jnp.asarray(E2h), 0.0, jnp.asarray(cold), kk,
                                    K, K + 1, n, K))
    assert W[7, 7] == 1.0  # stop2 puts LET at the top
    # exhaustion: the residual survival tops out at KX
    W = np.asarray(kfe._top_weights(E1 * 0, E1 * 0, 1.0, jnp.asarray(cold), kk, K, K + 1, n, K))
    assert W[:, n].sum() == 1.0
    check_grads(lambda e, c: kfe._top_weights(e, E2, s, c, kk, K, K + 1, n, K), (E1, CW),
                order=2, modes=("fwd", "rev"), eps=1e-4)


def test_deep_test_abe_counts_only_accepted_members():
    """75% of the plume exits at a rejected top after increments 2 and 8; the
    accepted quarter tops out at 12: its ABE is 10, not the 16 that
    survival weighting over all members would give after normalising."""
    dabe = jnp.zeros(14).at[4].set(2.0).at[5].set(8.0)
    ptop = jnp.zeros(14).at[12].set(0.25)
    np.testing.assert_allclose(float(kfe._conv_abe(dabe, ptop)) / 0.25, 10.0, rtol=1e-15)
    ptop = ptop.at[5].set(0.5)  # a member topping at 5 keeps only the level-4 increment
    np.testing.assert_allclose(float(kfe._conv_abe(dabe, ptop)), 0.25 * 10.0 + 0.5 * 2.0, rtol=1e-15)


def test_cont_floor_is_identity_above_5_and_c2():
    x = jnp.linspace(5.0, 40.0, 50)
    np.testing.assert_array_equal(kfe._cont_floor(x), x)
    assert float(jnp.min(kfe._cont_floor(jnp.linspace(-10.0, 5.0, 301)))) >= 1.0
    check_grads(kfe._cont_floor, (3.0,), order=2, modes=("fwd", "rev"), eps=1e-4)


def test_updraft_mass_closes_with_a_rejected_top(monkeypatch):
    """Three quarters of the exit mass on a cloud top at the source level K
    (no fixture column reaches it; the oracle's no-convection test rejects
    it) carry no flux and draw no source air, while the accepted quarter
    still convects: updraft entrainment and detrainment integrate to the same
    mass, and the column's water closes.
    (No fixture or sweep column has KPBL above LCL, so the KPBL rejection is
    covered by the weights unit test only.)"""
    top_weights = kfe._top_weights

    def with_top_at_k(E1, E2, s_ex, CW, kk, K, LCL, KX, KPBL, let_fixed=None):
        E1 = 0.25 * E1 + 0.75 * (kk == K)
        return top_weights(E1, 0.25 * E2, 0.25 * s_ex, CW, kk, K, LCL, KX, KPBL, let_fixed)

    monkeypatch.setattr(kfe, "_top_weights", with_top_at_k)
    fx = dict(np.load(_FIX))
    i = 90
    cs, trig, warm, fqi, fqs = fx["sweep_combos"][fx["sweep_combo"][i]]
    a = [jnp.asarray(fx[f"sweep_{k}"][i])[None] for k in ("u", "v", "T", "q", "p", "dz", "w", "rhoe")]
    o = kfe.kf_eta_para(*a, dt=fx["sweep_dt"][i], dx=fx["sweep_dx"][i], constants=_consts(fx, cs),
                        trigger=int(trig), warm_rain=warm == "1", f_qi=fqi == "1", f_qs=fqs == "1",
                        cudt=float(fx["cudt"]),
                        smoothing=_DEFAULT._replace(top_stop_umf_width_kgs=2.000001))
    assert bool(o["fired"][0])
    uer, udr = np.asarray(o["uer"][0]), np.asarray(o["udr"][0])
    assert abs(uer.sum() - udr.sum()) <= 1e-12 * uer.sum()
    der, ddr = np.asarray(o["der"][0]), np.asarray(o["ddr"][0])
    assert np.abs(der).sum() > 0.0
    assert abs(der.sum() + ddr.sum()) <= 1e-12 * np.abs(der).sum()
    _, x = _wrapper_inputs(fx, "L_nat_w0")
    out, _ = kain_fritsch_convection(
        **x, config=_WRF_DEFAULT._replace(wrf_top_stop_umf_width_kgs=2.000001))
    dp = fx["p_half"][:, 1:] - fx["p_half"][:, :-1]
    tot = (np.asarray(out.dq_v_dt) + np.asarray(out.dq_c_conv_dt)
           + np.asarray(out.dq_r_conv_dt)) * dp / constants.g
    scale = np.sum(np.abs(np.asarray(out.dq_v_dt)) * dp / constants.g, axis=1)
    f = fx["L_nat_w0_fired"] == 1
    assert np.all(np.asarray(out.convective_mask)[f] == 1.0)
    assert np.all(np.abs(tot.sum(axis=1))[f] <= 1.0e-9 * scale[f])


def _sweep_column(fx, i, T=None, smoothing=_DEFAULT, dtype=jnp.float64):
    cs, trig, warm, fqi, fqs = fx["sweep_combos"][fx["sweep_combo"][i]]
    a = [jnp.asarray(fx[f"sweep_{k}"][i], dtype)[None]
         for k in ("u", "v", "T", "q", "p", "dz", "w", "rhoe")]
    if T is not None:
        a[2] = T
    return kfe.kf_eta_para(*a, dt=fx["sweep_dt"][i], dx=fx["sweep_dx"][i], constants=_consts(fx, cs),
                           trigger=int(trig), warm_rain=warm == "1", f_qi=fqi == "1",
                           f_qs=fqs == "1", cudt=float(fx["cudt"]), smoothing=smoothing), a[2]


def _split_weights(monkeypatch, share):
    top_weights = kfe._top_weights

    def split(E1, E2, s_ex, CW, kk, K, LCL, KX, KPBL, let_fixed=None):
        W = top_weights(E1, E2, s_ex, CW, kk, K, LCL, KX, KPBL, let_fixed)
        m = jnp.maximum(LCL, KPBL)
        low = (kk == m + 1)[:, None] * jnp.sum(W, axis=0)[None, :]
        return (1.0 - share) * W + share * jnp.where(kk[None, :] > m, low, 0.0)

    monkeypatch.setattr(kfe, "_top_weights", split)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_downdraft_mixture_gradient_is_finite_for_a_tiny_share(monkeypatch, dtype):
    """A LET member carrying 1e-20 of the weight (its own downdraft source
    level) keeps the temperature gradient finite."""
    fx = dict(np.load(_FIX))
    _split_weights(monkeypatch, 1e-20)
    sm = _DEFAULT._replace(top_stop_umf_width_kgs=2.0000011)
    _, T = _sweep_column(fx, 90, smoothing=sm, dtype=dtype)
    g = jax.grad(lambda T: jnp.sum(_sweep_column(fx, 90, T, sm, dtype)[0]["dtdt"]))(T)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_survival_inflow_limit_is_the_capped_ratio_with_finite_gradients(dtype):
    """min(a / wl, cap) with cap where wl = 0; a 1e-20 share keeps the reverse
    gradient finite (in fp32 the unguarded a / wl gives a non-finite one)."""
    a = jnp.asarray([2.0, 2.0, 2.0, 2.0, 0.0], dtype)
    wl = jnp.asarray([1.0, 0.5, 1e-20, 0.0, 0.3], dtype)
    cap = kfe._AINCMX0
    np.testing.assert_allclose(np.asarray(kfe._survival_inflow_limit(a, wl)),
                               [2.0, 4.0, cap, cap, 0.0], rtol=1e-6)
    g = jax.grad(lambda a, w: jnp.sum(kfe._survival_inflow_limit(a, w)), argnums=(0, 1))(a, wl)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)
    old = jax.grad(lambda w: jnp.sum(jnp.minimum(cap, jnp.where(w > 0, a / jnp.where(
        w > 0, w, 1.0), cap))))(wl)
    assert bool(jnp.all(jnp.isfinite(old))) == (dtype == jnp.float64)


def test_downdraft_mixes_over_the_source_levels_of_the_let_members(monkeypatch):
    """Half of each top's weight moved to the lowest accepted LET puts those
    members' downdraft source one level below it, under the column's own LFS
    (a downdraft deep enough to run): the downdraft is the mixture of both
    runs (its normalised profile changes shape, not only amplitude), its mass
    closes, the column's water closes, and the temperature gradient stays
    finite.  The base run's width differs by 2e-7 kg/s only to get its own
    trace; the top widths move no fixture case (1.6e-15)."""
    fx = dict(np.load(_FIX))
    base, _ = _sweep_column(fx, 90, smoothing=_DEFAULT._replace(top_stop_umf_width_kgs=2.0000007))
    _split_weights(monkeypatch, 0.5)
    sm = _DEFAULT._replace(top_stop_umf_width_kgs=2.0000009)
    o, T = _sweep_column(fx, 90, smoothing=sm)
    assert bool(o["fired"][0])
    der, ddr = np.asarray(o["der"][0]), np.asarray(o["ddr"][0])
    assert abs(der.sum() + ddr.sum()) <= 1e-12 * np.abs(der).sum()
    d0 = np.asarray(base["der"][0])
    assert float(np.abs(der / der.sum() - d0 / d0.sum()).max()) > 1e-6
    g = jax.grad(lambda T: jnp.sum(_sweep_column(fx, 90, T, sm)[0]["dtdt"]))(T)
    assert bool(jnp.all(jnp.isfinite(g)))
    _, x = _wrapper_inputs(fx, "L_nat_w0")
    out, _ = kain_fritsch_convection(
        **x, config=_WRF_DEFAULT._replace(wrf_top_stop_umf_width_kgs=2.0000009))
    dp = fx["p_half"][:, 1:] - fx["p_half"][:, :-1]
    tot = (np.asarray(out.dq_v_dt) + np.asarray(out.dq_c_conv_dt)
           + np.asarray(out.dq_r_conv_dt)) * dp / constants.g
    scale = np.sum(np.abs(np.asarray(out.dq_v_dt)) * dp / constants.g, axis=1)
    f = fx["L_nat_w0_fired"] == 1
    assert np.all(np.asarray(out.convective_mask)[f] == 1.0)
    assert np.all(np.abs(tot.sum(axis=1))[f] <= 1.0e-9 * scale[f])



# Band column (make_fixture.BAND, appended last): the default top weights are
# 14:0.343 / 13:0.657.  Shifting its upper troposphere by _HARD_FLIP_DT puts it
# at the hard stop1 flip + 2e-7 K instead, where the stop1 width alone splits it.
_HARD_FLIP_DT = 0.01304718491


def _band(fx):
    i = len(fx["sweep_combo"]) - 1
    up = jnp.asarray(fx["sweep_p"][i] < 5.0e4, jnp.float64)
    return i, up, jnp.asarray(fx["sweep_T"][i])[None]


def _frac(fx, i, T, up, lo_sm, mid_sm):
    """Where ``mid_sm`` at T sits between ``lo_sm`` at T -/+ 1e-4 K (upper)."""
    lo, _ = _sweep_column(fx, i, T - 1e-4 * up, lo_sm)
    hi, _ = _sweep_column(fx, i, T + 1e-4 * up, lo_sm)
    mid, _ = _sweep_column(fx, i, T, mid_sm)
    jump = float(jnp.max(jnp.abs(hi["dtdt"] - lo["dtdt"])))
    return float(jnp.max(jnp.abs(mid["dtdt"] - lo["dtdt"]))) / jump, lo, hi


def test_band_column_top_sits_inside_the_smooth_band():
    """At the default widths the band column's tendencies lie strictly between
    the two saturated tops' (the column the default-gap test records); shifted
    to the hard flip, the stop1 width alone splits it between the hard tops."""
    fx = dict(np.load(_FIX))
    i, up, T = _band(fx)
    frac, _, _ = _frac(fx, i, T, up, _DEFAULT, _DEFAULT)
    assert 0.2 < frac < 0.8
    wonly = kfe.KF_HARD._replace(top_stop_wtw_width_m2s2=_DEFAULT.top_stop_wtw_width_m2s2)
    frac, lo, hi = _frac(fx, i, T + _HARD_FLIP_DT * up, up, kfe.KF_HARD, wonly)
    assert int(lo["ltop"][0]) != int(hi["ltop"][0]) and 0.2 < frac < 0.8


def _band_grad_fn():
    """Jitted scalar f(s) = sum(dtdt(T0 + s d) r) on the band column at the
    default widths (d, r random); jitted because eager order-2 tracing of the
    whole scheme exhausts LLVM."""
    fx = dict(np.load(_FIX))
    i, up, T0 = _band(fx)
    rng = np.random.default_rng(0)
    r, d = (jnp.asarray(rng.normal(size=32)) for _ in range(2))
    f = jax.jit(lambda s: jnp.sum(_sweep_column(fx, i, T0 + s * d, _DEFAULT)[0]["dtdt"][0] * r))
    return f, fx, i, up, T0, r


def test_default_gradient_through_the_band_column_is_c2():
    """Order-2 check_grads (fp64) of the band column's heating at the default
    widths.  f carries a 3e-7 relative noise floor and f''/f' ~ -2.5e5 /K inside
    the band, so no 2-point step meets 1e-5 (best ~2e-5): check_grads runs at
    1e-3 (vote 17, 3/3); the accuracy test below holds 1e-5."""
    f, fx, i, up, T0, r = _band_grad_fn()
    check_grads(f, (0.0,), order=2, modes=("fwd", "rev"), eps=3e-9, atol=1e-3, rtol=1e-3)


def test_default_gradient_through_the_band_column_is_accurate_and_carries_the_top():
    """First and second AD derivatives on the band column match 4-point
    Richardson differences to 1e-5 (measured 2.4e-6 / 3.0e-6), and the slope
    along the uniform upper shift carries the top flip."""
    f, fx, i, up, T0, r = _band_grad_fn()
    g1 = jax.jit(jax.grad(f))
    g2 = jax.jit(jax.grad(jax.grad(f)))
    e = 3e-8
    for g, ad in ((f, g1), (g1, g2)):
        fd = (8.0 * (g(e) - g(-e)) - (g(2 * e) - g(-2 * e))) / (12.0 * e)
        np.testing.assert_allclose(float(ad(0.0)), float(fd), rtol=1e-5)
    # Non-vacuity: along the uniform upper shift the default slope carries the
    # top flip (its jump over the ~2e-6 K band), not just the smooth background.
    g = jax.jit(lambda s: jnp.sum(_sweep_column(fx, i, T0 + s * up, _DEFAULT)[0]["dtdt"][0] * r))
    jump = abs(float(g(1e-4) - g(-1e-4)))
    assert jump > 0.0 and abs(float(jax.grad(g)(0.0))) > 0.1 * jump / 1e-5


def test_band_column_converges_to_the_hard_top_as_the_width_shrinks():
    """The stop1 width alone (others hard) at w, w/10, w/100, w*1e-6, on the band
    column shifted to the hard flip: its distance from the hard path shrinks to
    round-off as the band leaves it."""
    fx = dict(np.load(_FIX))
    i, up, T = _band(fx)
    T = T + _HARD_FLIP_DT * up
    hard, _ = _sweep_column(fx, i, T, kfe.KF_HARD)
    w = _DEFAULT.top_stop_wtw_width_m2s2
    d = [float(jnp.max(jnp.abs(_sweep_column(
        fx, i, T, kfe.KF_HARD._replace(top_stop_wtw_width_m2s2=w * f))[0]["dtdt"]
        - hard["dtdt"]))) for f in (1.0, 0.1, 0.01, 1e-6)]
    assert d[0] > 1e-6 * float(jnp.max(jnp.abs(hard["dtdt"])))
    assert d[3] <= 1e-12 and d[2] <= d[0] and d[3] <= d[1]


def test_top_hazards_are_the_oracle_steps_at_zero_width_and_c2():
    """Each cloud-top exit hazard is the oracle's strict step at width 0, one
    half at the threshold, saturated outside its band, and C2 inside it."""
    sm = _DEFAULT
    for i, (thr, w) in enumerate(((1.0e-3, sm.top_stop_wtw_width_m2s2),
                                  (10.0, sm.top_stop_umf_width_kgs))):
        def h(x, s=sm, i=i):
            args = (x, 50.0) if i == 0 else (1.0, x)
            return kfe._top_hazards(*args, s)[i]
        xs = jnp.array([thr - w, thr - 0.5 * w, thr, thr + 0.5 * w, thr + w])
        np.testing.assert_allclose(np.asarray(h(xs)), [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-15)
        np.testing.assert_array_equal(np.asarray(h(xs, kfe.KF_HARD)), [1.0, 1.0, 0.0, 0.0, 0.0])
        hz = lambda z, h=h, thr=thr, w=w: h(thr + w * z)  # noqa: E731  (band units)
        check_grads(hz, (0.2,), order=2, modes=("fwd", "rev"), eps=1e-5)
        assert float(jax.grad(hz)(0.2)) < 0.0


def test_widths_are_validated():
    with pytest.raises(ValueError, match="lut_knot_width"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(lut_knot_width=1.5))
    with pytest.raises(ValueError, match="top_stop_umf_width_kgs"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(top_stop_umf_width_kgs=10.5))
    with pytest.raises(ValueError, match="top_stop_wtw_width_m2s2"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(top_stop_wtw_width_m2s2=2.5e-3))
    with pytest.raises(ValueError, match="prof5_width"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(prof5_width=3.0))
    with pytest.raises(ValueError, match="entrain_tiny_width_k"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(entrain_tiny_width_k=-1.0))
    with pytest.raises(ValueError, match="tpmix_width_kgkg"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(tpmix_width_kgkg=-1.0))
    with pytest.raises(ValueError, match="prof5_width"):
        kfe.validate_kf_smoothing(_DEFAULT._replace(prof5_width=True))
    kfe.validate_kf_smoothing(kfe.KF_HARD)


def test_surrogate_refuses_wrf_widths():
    from legoesm.atmosphere.physics.convection.kain_fritsch import kain_fritsch_convection
    t = jnp.full((1, 4), 280.0)
    with pytest.raises(ValueError, match="wrf_prof5_width"):
        kain_fritsch_convection(t, t * 0, t * 0 + 5e4, jnp.full((1, 5), 5e4), t * 0, t * 0,
                                300.0, KainFritschConfig(wrf_prof5_width=0.1))


if __name__ == "__main__":
    # one case per process: all cases in one process exhaust LLVM code memory
    if len(sys.argv) != 3 or sys.argv[1] != "--write" or sys.argv[2] not in _CASES:
        sys.exit(f"usage: JAX_ENABLE_X64=1 python {Path(__file__).name} --write CASE; "
                 f"CASE in {list(_CASES)}")
    data = json.loads(_GAPS.read_text()) if _GAPS.exists() else {"gaps": {}}
    data["provenance"] = _provenance()
    data["gaps"][sys.argv[2]] = measure_case(dict(np.load(_FIX)), sys.argv[2])
    _GAPS.write_text(json.dumps(data, indent=0) + "\n")
    print("wrote", sys.argv[2], "to", _GAPS)
